"""weather_stems: rain, thunder, wind, ocean and ambience beds rebuilt from stems of CC0 recordings
(numbers only, no audio). See docs/weather.md.

Stems per recording:
* noise — the residual's own band envelopes (raised-cosine 1/8-octave, every 1 ms) imposed on fresh noise
  carriers, then the residual's 1/6-octave colour
* events (optional) — loud transients (a thunder crack, a drip, a wave crash) stored as their own band
  envelopes and rebuilt from noise
* a static treble ceiling per take (above 10 kHz the rebuild is only ever cut to where the recording sits)

Beds loop or run to any length: the stems are cut at quiet points into forward-only pieces and replayed in a
new order, joined with 100 ms power-complementary crossfades. Nothing is averaged, no band is pitch-shifted
(size is tape speed of the whole render), nothing is held or time-squeezed.

Intensity over time (rain, wind): when a family has recordings at more than one intensity, the renderer
crossfades those takes with equal-power weights that follow a curve (constant, linear, or swell). The
recordings themselves are not stretched.
"""
from __future__ import annotations

import functools
import json
import lzma
import math
import struct
from pathlib import Path

import numpy as np
from scipy import signal

from ..core import Ctx, Param, RecipeError, model
from .air_stems import SR, _eq_to, _ltas6
from .machine_stems import _blend, _lp, _q, _read, _shelf, _tape, _vocoder

DATA = Path(__file__).with_name("weather_stems.bin")
CR = 1000
XF = 0.1
CHUNK = (1.2, 2.4)
LAST = {}
SOURCES = ()  # filled after the bin loads; the Param uses a loader


@functools.lru_cache(maxsize=1)
def _data():
    raw = DATA.read_bytes()
    if raw[:8] != b"SLWTH001":
        raise RecipeError(f"{DATA.name}: not a soundlab weather stem file")
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    hdr = json.loads(p[4:4 + n].decode())
    blob = p[4 + n:]

    def arr(d):
        return np.frombuffer(blob, np.dtype(d["dt"]), int(np.prod(d["sh"])), d["o"]).reshape(d["sh"])

    out = {}
    for k, s in hdr["sources"].items():
        st = dict(n=s["n"], env=_q(arr(s["env"]), s["env_lo"]), env_sr=s["env_sr"],
                  ltas6=arr(s["ltas6"]).astype(np.float64), kind=s["kind"],
                  onsets=arr(s["onsets"]).astype(np.float64) / 1000.0,
                  hf_ceil_db=arr(s["hf_ceil_db"]).astype(np.float64),
                  chunk=tuple(s.get("chunk", CHUNK)),
                  group=s.get("group"), intensity=float(s.get("intensity", 0.5)))
        if "ev_env" in s:
            E = _q(arr(s["ev_env"]), s["ev_lo"])
            off, t, ln, pad = (arr(s[q]) for q in ("ev_off", "ev_t", "ev_len", "ev_pad"))
            st["events"] = [(int(t[j]), int(ln[j]), int(pad[j]), E[:, off[j]:off[j + 1]]) for j in range(len(ln))]
            st.update(ev_rate=s["ev_rate"], ev_energy=arr(s["ev_energy"]).astype(np.float64),
                      ev_eq6=arr(s["ev_eq6"]).astype(np.float64))
        out[k] = (st, {kk: v for kk, v in s.items() if kk in ("file", "t0", "t1", "settings")})
    groups = hdr.get("groups") or {}
    return dict(sources=out, files=hdr["files"], groups=groups)


def sources() -> list:
    if not DATA.exists():
        return []
    return sorted(_data()["sources"])


def _act(st):
    t = st["onsets"]
    return lambda a, b: float(np.sum((t >= a) & (t < b)) / max(b - a, 1e-9))


def _distance(d):
    ld = math.log2(max(float(d), 1e-3))
    if d > 1:
        return -5 * ld, 0.0, min(14000 / math.sqrt(d), 20000)
    if d < 1:
        return 0.0, -3 * ld, 0.0
    return 0.0, 0.0, 0.0


def random_timeline(T_src, T_out, rng, clen=CHUNK, onsets=None, act=None, bias=0.0):
    """Forward-only pieces of random length, cut at quiet points; 100 ms power-complementary joins."""
    on = np.asarray(onsets) if onsets is not None else np.zeros(0)

    def ok(t):
        return not np.any(np.abs(on - t) < 0.06)

    b = [0.0]
    while True:
        t = b[-1] + rng.uniform(*clen)
        if t + XF + clen[0] > T_src:
            break
        c = t + np.arange(-0.15, 0.1501, 0.01)
        g = [u for u in c if ok(u) and ok(u + XF) and u + XF < T_src and u > b[-1] + 0.3]
        if g:
            t = g[int(np.argmin(np.abs(np.array(g) - t)))]
        b.append(t)
    if T_src - XF - 0.15 - b[-1] >= 0.3:
        b.append(T_src - XF - 0.15)
    segs = [(b[i], b[i + 1]) for i in range(len(b) - 1)]
    if not segs:
        raise RecipeError("weather_stems: piece length longer than the recording")
    out, tot, last = [], 0.0, []
    while tot < T_out + XF:
        if bias == 0:
            for i in rng.permutation(len(segs)):
                out.append(segs[i])
                tot += segs[i][1] - segs[i][0]
                if tot >= T_out + XF:
                    break
        else:
            a = np.array([act(u, v) for u, v in segs])
            z = (a - a.mean()) / (a.std() + 1e-9)
            p = np.exp(bias * z)
            for r in last[-3:]:
                p[r] = 0
            if p.sum() <= 0:
                p = np.ones(len(segs))
            i = int(rng.choice(len(segs), p=p / p.sum()))
            last.append(i)
            out.append(segs[i])
            tot += segs[i][1] - segs[i][0]
    ch, o = [], 0.0
    for (u, v) in out:
        ch.append((u, o, v - u + XF))
        o += v - u
    return ch


def _map(ch, T_out):
    m_n = int(math.ceil(T_out * CR)) + 2
    a = np.zeros(m_n)
    b = np.zeros(m_n)
    w = np.zeros(m_n)
    x = int(XF * CR)
    for i, (s, o, L) in enumerate(ch):
        i0 = int(round(o * CR))
        m = int(round(L * CR))
        j = np.arange(i0, min(i0 + m, m_n))
        if not len(j):
            continue
        r = s + (j - i0) / CR
        if i > 0:
            k = j < i0 + x
            b[j[k]] = r[k]
            w[j[k]] = np.sin(np.linspace(0, np.pi / 2, x))[:k.sum()] ** 2
            a[j[~k]] = r[~k]
        else:
            a[j] = r
    b = np.where(w > 0, b, a)
    one = np.ones(m_n)
    return dict(a=a, b=b, w=w, sa=one, sb=one, n=int(round(T_out * SR)))


def _event_wave(st, j, seed):
    t, ln, pad, E = st["events"][j]
    w = _vocoder(E, st["ev_rate"], ln, None, 7919 + j + seed, iters=8)[:ln]
    if "ev_eq6" in st:
        w = _eq_to(w, _ltas6(w) + st["ev_eq6"])
    return t, pad, w


def _events(st, ev, n, seed):
    y = np.zeros(n)
    for j, so in ev:
        t, pad, w = _event_wave(st, j, seed)
        s0 = so - pad
        if s0 < 0:
            w, s0 = w[-s0:], 0
        m = min(len(w), n - s0)
        if m > 0 and s0 < n:
            y[s0:s0 + m] += w[:m]
    return y


def place_events(st, ch, T_out):
    ev = []
    for s, o, L in ch:
        for j, (t, ln, pad, E) in enumerate(st["events"]):
            tt = t / SR
            if s <= tt < s + L - 1e-9:
                ev.append((j, int(round((o + (tt - s)) * SR))))
    return ev


def _timelines(st, p, rng, cents):
    T = st["n"] / SR
    r = 2 ** (cents / 1200)
    kind = st["kind"]
    var = int(round(p["variation"]))
    ch = None
    if kind == "loop" and (var or p["bias"] or (p["length_s"] and abs(p["length_s"] - T) > 1e-6)):
        T_out = (float(p["length_s"]) * r if p["length_s"] else T) + 0.05
        lo, hi = float(p["chunk_min_s"]) or st["chunk"][0], float(p["chunk_max_s"]) or st["chunk"][1]
        if not 0.3 <= lo <= hi <= max(T / 2, lo):
            raise RecipeError(f"weather_stems: piece length {lo:g} .. {hi:g} s outside 0.3 .. {T / 2:g}")
        ch = random_timeline(T, T_out, rng, (lo, hi), st["onsets"], _act(st), float(p["bias"]))
    else:
        T_out = float(p["length_s"]) * r if (p["length_s"] and kind == "loop") else T
        if kind == "loop" and p["length_s"] and abs(p["length_s"] - T) > 1e-6:
            T_out = float(p["length_s"]) * r + 0.05
    return ch, T_out


def _render_one(src, p, seed, rng, T_force=None):
    st = _data()["sources"][src][0]
    cents = float(p["speed_cents"])
    p2 = dict(p)
    if T_force is not None and st["kind"] == "loop":
        p2["length_s"] = T_force / (2 ** (cents / 1200))
    ch, T_out = _timelines(st, p2, rng, cents)
    n = int(round(T_out * SR))
    if ch is None:
        m = dict(a=None, n=n, identity=True)
        y = _vocoder(st["env"], st["env_sr"], n, st["ltas6"], seed)[:n]
    else:
        m = _map(ch, T_out)
        env = _blend(_read(st["env"], st["env_sr"], m["a"]), _read(st["env"], st["env_sr"], m["b"]), m, power=True)
        y = _vocoder(env, CR, n, st["ltas6"], seed)[:n]
    if "events" in st:
        ev = place_events(st, ch or [(0.0, 0.0, T_out)], T_out)
        y = y + _events(st, ev, n, seed)
    c = st["hf_ceil_db"]
    f = np.fft.rfftfreq(4096, 1 / SR)[:len(c)]
    y = np.fft.irfft(np.fft.rfft(y) * 10 ** (np.interp(np.fft.rfftfreq(n, 1 / SR), f, c) / 20), n)
    if ch is not None and st["kind"] == "loop":
        y = y[:int(round((float(p2["length_s"]) or st["n"] / SR / 2 ** (cents / 1200)) * SR))]
    return y, ch, st


def _curve(p, n, sr):
    typ = str(p.get("curve") or "constant")
    if typ == "constant":
        return np.full(n, float(p["intensity"]))
    t = np.arange(n) / sr
    T = max(n / sr, 1e-9)
    if typ == "linear":
        return np.linspace(float(p["curve_start"]), float(p["curve_end"]), n)
    if typ == "swell":
        s0, pk, s1 = float(p["curve_start"]), float(p["curve_peak"]), float(p["curve_end"])
        mid = float(p["curve_peak_at"]) * T
        mid = min(max(mid, 1e-6), T - 1e-6)
        I = np.empty(n)
        for i, ti in enumerate(t):
            if ti <= mid:
                u = 0.5 - 0.5 * math.cos(math.pi * ti / mid)
                I[i] = s0 + (pk - s0) * u
            else:
                u = 0.5 - 0.5 * math.cos(math.pi * (ti - mid) / (T - mid))
                I[i] = pk + (s1 - pk) * u
        return I
    raise RecipeError(f"weather_stems: unknown curve {typ!r}")


def _group_members(src):
    data = _data()
    st0 = data["sources"][src][0]
    g = st0.get("group")
    if not g:
        return [(src, st0.get("intensity", 0.5))]
    members = []
    for name, (st, _meta) in data["sources"].items():
        if st.get("group") == g:
            members.append((name, float(st.get("intensity", 0.5))))
    members.sort(key=lambda kv: kv[1])
    return members or [(src, st0.get("intensity", 0.5))]


def _weights_at(members, I):
    """Equal-power weights for each member at scalar intensity I (0..1)."""
    I = float(np.clip(I, members[0][1], members[-1][1]))
    w = np.zeros(len(members))
    if len(members) == 1:
        w[0] = 1.0
        return w
    for i in range(len(members) - 1):
        a, b = members[i][1], members[i + 1][1]
        if a <= I <= b or (i == len(members) - 2 and I >= b):
            u = 0.0 if b <= a else (I - a) / (b - a)
            w[i] = math.cos(0.5 * math.pi * u)
            w[i + 1] = math.sin(0.5 * math.pi * u)
            return w
    w[-1] = 1.0
    return w


def render_weather(p: dict, rng=None) -> np.ndarray:
    src = p["source"]
    if src not in dict(_data()["sources"]):
        raise RecipeError(f"weather_stems: unknown source {src!r}; stored: {sources()}")
    var = int(round(p["variation"]))
    seed = int(round(p["seed"]))
    if var:
        seed = seed * 1000 + abs(var) % 100000
    rng = rng if rng is not None else np.random.default_rng(seed + 100)
    members = _group_members(src)
    st0 = _data()["sources"][src][0]
    cents = float(p["speed_cents"])
    T_nat = st0["n"] / SR
    T = float(p["length_s"]) if p["length_s"] else T_nat
    I = _curve(p, int(round(T * SR * 2 ** (cents / 1200))) or 1, SR)
    # which members have any weight
    need = []
    for j, (name, _) in enumerate(members):
        # max weight over a coarse sample of the curve
        step = max(1, len(I) // 64)
        if any(_weights_at(members, I[i])[j] > 1e-4 for i in range(0, len(I), step)):
            need.append(j)
    if p.get("curve", "constant") == "constant" and len(members) >= 1:
        # exact match to one source: identity path
        w0 = _weights_at(members, float(p["intensity"]))
        if (w0 > 1 - 1e-6).sum() == 1 and abs(float(p["intensity"]) - members[int(np.argmax(w0))][1]) < 1e-6:
            need = [int(np.argmax(w0))]
    LAST.clear()
    LAST.update(members=[members[j][0] for j in need], timeline={})
    waves = []
    T_target = T
    for j in need:
        name = members[j][0]
        # A lone identity member must use seed 0 so the default take matches the stored rebuild.
        # Several members mixed together get offset seeds so their noise is uncorrelated.
        use_seed = seed if len(need) == 1 else seed + 17 * (j + 1)
        y, ch, st = _render_one(name, p, use_seed, rng, T_force=T_target if st0["kind"] == "loop" else None)
        LAST["timeline"][name] = ch
        waves.append((j, y))
    n = min(int(round(T * SR)), min(len(y) for _, y in waves))
    mix = np.zeros(n)
    I = _curve(p, n, SR)
    # weights at ~1 kHz, interpolated (a per-sample Python loop is too slow for long beds)
    cr = 1000
    m = max(2, int(math.ceil(n / SR * cr)) + 1)
    idx = np.linspace(0, n - 1, m).astype(int)
    Wc = np.stack([_weights_at(members, I[i]) for i in idx], 0)
    tt = np.linspace(0, m - 1, n)
    W = np.column_stack([np.interp(tt, np.arange(m), Wc[:, j]) for j in range(len(members))])
    for j, y in waves:
        mix += y[:n] * W[:n, j]
    hs, ls, dlp = _distance(p["distance"])
    y = mix
    if p["ls_db"] + ls:
        y = signal.sosfiltfilt(_shelf(p["ls_hz"], p["ls_db"] + ls, "low"), y)
    if p["hs_db"] + hs:
        y = signal.sosfiltfilt(_shelf(p["hs_hz"], p["hs_db"] + hs, "high"), y)
    y = _tape(y, cents)
    y = _lp(y, dlp or p["lp_hz"])
    k = min(len(y), int(0.03 * SR))
    y[-k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    return y / (np.abs(y).max() + 1e-12) * 0.89


def _source_choices():
    try:
        return tuple(sources()) or ("rain_light",)
    except Exception:
        return ("rain_light",)


@model("weather_stems", "sound",
       "Rain, thunder, wind, ocean and ambience beds rebuilt from stems of CC0 recordings (numbers only, no "
       "audio). Band-envelope noise beds, optional real events, looping pieces for any length. Rain and wind "
       "can follow an intensity curve that crossfades recordings at several intensities (no stretching).",
       [
           Param("source", "rain_light", "", choices=_source_choices(), doc="stored source"),
           Param("length_s", 0.0, "s", 0, 600, "looping sources: length of the take (0 = as recorded)"),
           Param("intensity", 0.5, "", 0.0, 1.0, "static intensity (0 = light, 1 = heavy) when curve is constant; "
                 "also the default level the curve is built around"),
           Param("curve", "constant", "", choices=("constant", "linear", "swell"),
                 doc="intensity over time: constant, linear start..end, or a swell (start, peak, end)"),
           Param("curve_start", 0.15, "", 0.0, 1.0, "linear/swell: intensity at the start"),
           Param("curve_end", 0.3, "", 0.0, 1.0, "linear/swell: intensity at the end"),
           Param("curve_peak", 1.0, "", 0.0, 1.0, "swell: intensity at the peak"),
           Param("curve_peak_at", 0.55, "", 0.05, 0.95, "swell: when the peak occurs, as a fraction of the length"),
           Param("bias", 0.0, "", -2.0, 2.0, "fresh take drawn towards busier (> 0) or quieter (< 0) stretches"),
           Param("chunk_min_s", 0.0, "s", 0, 5, "fresh take: shortest piece (0 = the source's own)"),
           Param("chunk_max_s", 0.0, "s", 0, 8, "fresh take: longest piece (0 = the source's own)"),
           Param("speed_cents", 0.0, "cents", -1200, 1200, "tape speed (pitch and time together)"),
           Param("distance", 1.0, "", 0.25, 4.0, "1 = as recorded; further = high shelf down and a softer low-pass; "
                 "nearer = more low end (level: the recipe's master gain)"),
           Param("hs_hz", 2500.0, "Hz", 200, 16000),
           Param("hs_db", 0.0, "dB", -24, 12),
           Param("ls_hz", 250.0, "Hz", 30, 2000),
           Param("ls_db", 0.0, "dB", -12, 12),
           Param("lp_hz", 0.0, "Hz", 0, 20000, "treble roll-off (0 or 20000 = off)"),
           Param("seed", 0, "", 0, 2**31 - 1, "noise seed (the recipe default reproduces its reference)"),
           Param("variation", 0, "", -1e6, 1e6, "0 = the reference; others reseed (looping sources: a fresh take)"),
           Param("at", 0.0, "s", 0, None),
       ],
       ["McDermott & Simoncelli 2011 (sound texture statistics)",
        "Freesound CC0 recordings (numbers only; THIRD_PARTY.md)"],
       min_s=lambda p: 0.1)
def weather_stems(ctx: Ctx, x, **p):
    p = dict(p)
    at = float(p.pop("at"))
    y = render_weather(p)
    if ctx.sr != SR:
        g = math.gcd(int(ctx.sr), SR)
        y = signal.resample_poly(y, int(ctx.sr) // g, SR // g)
    out = np.zeros(ctx.n)
    i0 = int(at * ctx.sr)
    if i0 < ctx.n:
        m = min(len(y), ctx.n - i0)
        out[i0:i0 + m] = y[:m]
    return out
