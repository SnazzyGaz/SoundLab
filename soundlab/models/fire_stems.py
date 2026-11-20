"""fire_stems (v0.33): fire, crackle and electricity rebuilt from stems of CC0 recordings (numbers only, no audio):
a campfire, a large blaze, embers, a match strike, a gas stove igniting, a sparkler, a fuse, a transformer hum, an
electric arc and a flickering fluorescent tube. See docs/fire.md.

Stems per recording:
* noise - the residual's own band envelopes (76 raised-cosine bands at 1/8 octave, every 1 ms, 0.5 dB steps) imposed
  on fresh noise carriers, then the residual's 1/6-octave colour
* events - every crackle, pop, click or strike found in the recording, stored as its own 1/8-octave band envelopes
  (0.5 ms; 0.125 ms for the gas stove's clicks) and rebuilt from noise, with one static 1/6-octave EQ per take
* harmonics / lines (transformer, arc, tube) - the tracked fundamental and each harmonic's real amplitude and phase
  wobble every 4 ms; persistent lines off the series (tube)
* a static treble ceiling per take (above 10 kHz the rebuild is only ever cut to where the recording sits; the tube
  matches the recording above 3 kHz)

Fresh takes: the stems are cut at quiet points into forward-only pieces of random length and replayed in a new
order, joined with 100 ms power-complementary crossfades; each real event plays with the piece it belongs to. Nothing
is averaged, no band is pitch-shifted (speed = tape speed of the whole render), nothing is held or time-squeezed and
no smoothing is added.
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
from .liquid_stems import _depth
from .machine_stems import _blend, _harm_phi, _harmonics, _lines, _lp, _q, _read, _shelf, _tape, _vocoder

DATA = Path(__file__).with_name("fire_stems.bin")
CR = 1000                      # control rate of the stems and the take maps (Hz)
XF = 0.1                       # crossfade between pieces of a fresh take (s)
CHUNK = (0.6, 1.5)             # default piece length range (s)
LAST = {}                      # the last render's timeline and placed events (for inspection)
SOURCES = ("campfire", "large_blaze", "embers", "match_strike", "gas_stove_ignite", "sparkler", "fuse",
           "transformer_hum", "electric_arc", "fluorescent_flicker")


@functools.lru_cache(maxsize=1)
def _data():
    raw = DATA.read_bytes()
    if raw[:8] != b"SLFIR001":
        raise RecipeError(f"{DATA.name}: not a soundlab fire stem file")
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    hdr = json.loads(p[4:4 + n].decode())
    blob = p[4 + n:]

    def arr(d):
        return np.frombuffer(blob, np.dtype(d["dt"]), int(np.prod(d["sh"])), d["o"]).reshape(d["sh"])

    out = {}
    for k, s in hdr["sources"].items():
        st = dict(n=s["n"], env=_q(arr(s["env"]), s["env_lo"]), env_sr=s["env_sr"], ltas6=arr(s["ltas6"]).astype(np.float64),
                  kind=s["kind"], onsets=arr(s["onsets"]).astype(np.float64) / 1000.0,
                  hf_ceil_db=arr(s["hf_ceil_db"]).astype(np.float64), chunk=tuple(s.get("chunk", CHUNK)),
                  clicks=s.get("clicks"), ignite_s=s.get("ignite_s"))
        if "f0" in s:
            f0 = arr(s["f0"]).astype(np.float64)
            st.update(f0_t=s["f0_t0"] + s["f0_dt"] * np.arange(len(f0)), f0=f0, h_k=arr(s["h_k"]).astype(int),
                      h_sr=s["h_sr"], h_amp=_q(arr(s["h_amp"]), s["h_lo"]),
                      h_ph=np.unwrap(arr(s["h_ph"]).astype(np.float64) * (2 * np.pi / 256), axis=1))
        if "l_f" in s:
            lf = arr(s["l_f"]).astype(np.float64)
            st.update(l_t=s["l_t0"] + s["l_dt"] * np.arange(lf.shape[1]), l_f=lf, l_amp=_q(arr(s["l_amp"]), s["l_lo"]),
                      l_ph=np.unwrap(arr(s["l_ph"]).astype(np.float64) * (2 * np.pi / 256), axis=1))
        if "ev_env" in s:
            E = _q(arr(s["ev_env"]), s["ev_lo"])
            off, t, ln, pad = (arr(s[q]) for q in ("ev_off", "ev_t", "ev_len", "ev_pad"))
            st["events"] = [(int(t[j]), int(ln[j]), int(pad[j]), E[:, off[j]:off[j + 1]]) for j in range(len(ln))]
            st.update(ev_rate=s["ev_rate"], ev_energy=arr(s["ev_energy"]).astype(np.float64),
                      ev_eq6=arr(s["ev_eq6"]).astype(np.float64), ev_causal=bool(s["ev_causal"]))
        out[k] = (st, {kk: v for kk, v in s.items() if kk in ("file", "t0", "t1", "settings")})
    return dict(sources=out, files=hdr["files"])


def sources() -> list:
    return sorted(_data()["sources"])


# ---------------------------------------------------------------------------------------------- take timelines
# A timeline is a list of pieces (source start s, output start o, length L), all read forwards; consecutive pieces
# overlap by XF (the crossfade), so o[i+1] = o[i] + L[i] - XF.

def random_timeline(T_src, T_out, rng, clen=CHUNK, onsets=None, act=None, bias=0.0):
    """Forward-only pieces of random length cut at quiet points (no recorded transient within 60 ms of either end of
    a crossfade). bias == 0: the source is cut into pieces and replayed in a random order without repeats (a fresh
    take with the recording's own content); bias != 0: pieces drawn with replacement, weighted by their transient
    activity (exp(bias x z-score)), no piece back within three picks."""
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
        b.append(T_src - XF - 0.15)               # nearly to the end (the last analysis frames are unreliable)
    segs = [(b[i], b[i + 1]) for i in range(len(b) - 1)]
    if not segs:
        raise RecipeError("fire_stems: piece length longer than the recording")
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


def seq_timeline(intervals):
    """Consecutive source intervals (a, b) played in order; each piece runs XF past b into its own source."""
    ch, o = [], 0.0
    for i, (a, b) in enumerate(intervals):
        ch.append((a, o, (b - a) + (XF if i < len(intervals) - 1 else 0.0)))
        o += b - a
    return ch


def click_intervals(st, clicks, rng):
    """Gas stove: the knob and first click, then 0-3 of the three recorded click cycles before the burner catches (fewer
    clicks), then repeats of the recorded cycles (more clicks: never one cycle twice in a row), then the last three
    clicks as the burner catches and burns."""
    C = st["clicks"]
    T = st["n"] / SR
    k = int(round(clicks))
    if not 4 <= k <= 13:
        raise RecipeError(f"fire_stems: clicks {k} outside 4 .. 13")
    cyc = [(C[i] - 0.1, C[i + 1] - 0.1) for i in range(4)]
    keep = cyc[1:1 + min(3, k - 4)]
    ex = []
    while len(ex) < k - 7:
        p = [int(v) for v in rng.permutation(3)]
        if ex and p[0] == ex[-1]:
            p = p[::-1]
        ex += p
    return [(0.0, C[1] - 0.1)] + keep + [cyc[j] for j in ex[:max(0, k - 7)]] + [(C[4] - 0.1, T)]


def _map(ch, T_out):
    """Timeline -> control map for the stem readers: reading a (and b with weight w inside a crossfade; power
    weights sin^2 / cos^2, as the pieces' envelopes are combined in power)."""
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


def _weights(ch, R, T_out):
    """Per piece: (output frame, source frame, frames, power weight) at control rate R."""
    n = int(round(T_out * R))
    out = []
    x = int(XF * R)
    for i, (s, o, L) in enumerate(ch):
        m = int(round(L * R))
        w = np.ones(m)
        if i > 0:
            w[:x] = np.sin(np.linspace(0, np.pi / 2, x)) ** 2
        if i < len(ch) - 1:
            w[-x:] = np.cos(np.linspace(0, np.pi / 2, x)) ** 2
        out.append((int(round(o * R)), int(round(s * R)), m, w))
    return n, out


def _remap_pow(A, ch, R, T_out):
    """Amplitude tracks through the timeline, combined in power across a crossfade."""
    n, W = _weights(ch, R, T_out)
    out = np.zeros(A.shape[:-1] + (n,))
    for i0, s0, m, w in W:
        m2 = min(m, n - i0, A.shape[-1] - s0)
        if m2 > 0:
            out[..., i0:i0 + m2] += A[..., s0:s0 + m2] ** 2 * w[:m2]
    return np.sqrt(out)


def _remap_lin(a, ch, R, T_out, phase=False):
    """Frequency or phase tracks through the timeline; a phase track is shifted by whole cycles to meet what is
    already there, so the crossfade blends two nearly equal phases."""
    n, W = _weights(ch, R, T_out)
    out = np.zeros(n)
    ws = np.zeros(n)
    for i0, s0, m, w in W:
        m2 = min(m, n - i0, len(a) - s0)
        if m2 <= 0:
            continue
        p = a[s0:s0 + m2].copy()
        if phase and ws[i0] > 0:
            p += 2 * np.pi * np.round((out[i0] / ws[i0] - p[0]) / (2 * np.pi))
        out[i0:i0 + m2] += p * w[:m2]
        ws[i0:i0 + m2] += w[:m2]
    return out / np.maximum(ws, 1e-9)


def _tonal_take(st, ch, T_out, n, harm_db=0.0, tilt=0.0):
    """Harmonics and lines through a fresh-take timeline: amplitudes in power, f0 / line frequency linearly, phase
    tracks aligned by whole cycles."""
    ta = np.arange(n) / SR
    y = np.zeros(n)
    hr = st["h_sr"]
    if "f0" in st:
        t1 = np.arange(int(st["n"] / SR * CR)) / CR
        f0o = _remap_lin(np.interp(t1, st["f0_t"], st["f0"]), ch, CR, T_out)
        phi = 2 * np.pi * np.cumsum(np.interp(ta, np.arange(len(f0o)) / CR, f0o)) / SR
        A = _remap_pow(st["h_amp"], ch, hr, T_out)
        te = np.arange(A.shape[1]) / hr
        g = 10 ** (harm_db / 20) * (st["h_k"] / 8.0) ** tilt
        for j, k in enumerate(st["h_k"]):
            if A[j].max() <= 0:
                continue
            ph = _remap_lin(st["h_ph"][j], ch, hr, T_out, phase=True)
            y += g[j] * np.interp(ta, te, A[j]) * np.cos(k * phi + np.interp(ta, te, ph))
    if "l_f" in st:
        A = _remap_pow(st["l_amp"], ch, hr, T_out)
        te = np.arange(A.shape[1]) / hr
        t1 = np.arange(int(st["n"] / SR * CR)) / CR
        for j in range(len(st["l_f"])):
            fo = _remap_lin(np.interp(t1, st["l_t"], st["l_f"][j]), ch, CR, T_out)
            ph = _remap_lin(st["l_ph"][j], ch, hr, T_out, phase=True)
            y += np.interp(ta, te, A[j]) * np.cos(2 * np.pi * np.cumsum(np.interp(ta, np.arange(len(fo)) / CR, fo)) / SR
                                                  + np.interp(ta, te, ph))
    return y


def place_events(st, ch, T_out):
    """(event index, output sample of the event's start) for every event whose recorded time falls in a piece's core
    (crossfades split at their middle). A piece used twice brings its events both times."""
    ev = []
    for i, (cs, co, L) in enumerate(ch):
        lo = co + (XF / 2 if i > 0 else -1e9)
        hi = co + L - (XF / 2 if i < len(ch) - 1 else -1e9)
        for j, (t, _, _, _) in enumerate(st["events"]):
            ts = t / SR
            to = co + ts - cs
            if cs <= ts < cs + L and lo <= to < hi and to < T_out:
                ev.append((j, int(round(to * SR))))
    ev.sort(key=lambda e: e[1])
    return ev


@functools.lru_cache(maxsize=4096)
def _event_wave(src, j, seed):
    st = _data()["sources"][src][0]
    t, ln, pad, E = st["events"][j]
    w = _vocoder(E, st["ev_rate"], ln, None, seed, iters=8)
    if st["ev_causal"]:          # nothing before the click's own start (fresh-noise low bands would pre-echo)
        mk = np.ones(len(w))
        mk[:pad] = 0
        fd = int(0.0005 * SR)
        mk[pad:pad + fd] = np.sin(np.linspace(0, np.pi / 2, fd)) ** 2
        w = w * mk
    return w


def _events(src, st, ev, n, seed):
    y = np.zeros(n)
    for k, (j, so) in enumerate(ev):
        w = _event_wave(src, j, 7919 + 1000 * seed + k)
        s0 = so - st["events"][j][2]
        if s0 < 0:
            w, s0 = w[-s0:], 0
        mm = min(len(w), n - s0)
        if mm > 0:
            y[s0:s0 + mm] += w[:mm]
    return _eq_to(y, _ltas6(y) + st["ev_eq6"]) if ev and np.any(y) else y


# ---------------------------------------------------------------------------------------------- render

def _act(st):
    t = st["onsets"]
    return lambda a, b: float(np.sum((t >= a) & (t < b)) / (b - a))


def _distance(d):
    """Further (d > 1): high shelf -5 dB at 2.5 kHz and a softer low-pass per doubling (level: the recipe's master
    gain, -6 dB per doubling). Nearer (d < 1): low shelf +3 dB per halving. Returns (hs dB, ls dB, low-pass Hz)."""
    ld = math.log2(max(float(d), 1e-3))
    if d > 1:
        return -5 * ld, 0.0, min(14000 / math.sqrt(d), 20000)
    if d < 1:
        return 0.0, -3 * ld, 0.0
    return 0.0, 0.0, 0.0


def _timelines(src, st, p, seed, rng, cents):
    """The take's timeline (None = as recorded), the second timeline that adds events (density > 1) and the length."""
    T = st["n"] / SR
    r = 2 ** (cents / 1200)
    kind = st["kind"]
    var = int(round(p["variation"]))
    ch = None
    if kind == "clicks":
        if int(round(p["clicks"])) != len(st["clicks"]):
            seq = click_intervals(st, p["clicks"], rng)
            ch = seq_timeline(seq)
            T = sum(b - a for a, b in seq)
        T_out = T
    elif kind == "loop" and (var or p["bias"] or (p["length_s"] and abs(p["length_s"] - T) > 1e-6)):
        T_out = (float(p["length_s"]) * r if p["length_s"] else T) + 0.05
        lo, hi = float(p["chunk_min_s"]) or st["chunk"][0], float(p["chunk_max_s"]) or st["chunk"][1]
        if not 0.3 <= lo <= hi <= T / 2:
            raise RecipeError(f"fire_stems: piece length {lo:g} .. {hi:g} s outside 0.3 .. {T / 2:g}")
        ch = random_timeline(T, T_out, rng, (lo, hi), None if "events" in st else st["onsets"], _act(st), float(p["bias"]))
    else:
        T_out = T
    ch2 = None
    if p["density"] > 1 and "events" in st:
        ch2 = random_timeline(st["n"] / SR, T_out, rng) if kind == "loop" else (ch or [(0.0, 0.0, T_out)])
    return ch, ch2, T_out


def _render(src, p, seed, rng):
    st = _data()["sources"][src][0]
    cents = float(p["speed_cents"])
    ch, ch2, T_out = _timelines(src, st, p, seed, rng, cents)
    LAST.clear()
    LAST.update(timeline=ch, events=[])
    n = int(round(T_out * SR))
    s = dict(st)
    if p["flame_db"] and st.get("ignite_s"):        # the burner, from when it catches, louder or softer
        tt = np.arange(st["env"].shape[1]) / st["env_sr"]
        s["env"] = st["env"] * 10 ** (p["flame_db"] * np.clip((tt - st["ignite_s"]) / 0.2, 0, 1) / 20)[None, :]
    if p["env_depth"] != 1.0:
        E = s["env"]
        eps = 1e-4 * E.max() + 1e-12
        s["env"] = np.maximum(np.exp(_depth(np.log(E + eps), float(p["env_depth"]), int(0.151 * st["env_sr"]) | 1)) - eps, 0)
    if ch is None:
        m = dict(a=None, n=n, identity=True)
        y = _vocoder(s["env"], s["env_sr"], n, s["ltas6"], seed)[:n]
    else:
        m = _map(ch, T_out)
        env = _blend(_read(s["env"], s["env_sr"], m["a"]), _read(s["env"], s["env_sr"], m["b"]), m, power=True)
        y = _vocoder(env, CR, n, s["ltas6"], seed)[:n]
    y = y * 10 ** (p["bed_db"] / 20)
    if ch is None:
        if "f0" in st:
            phi, _ = _harm_phi(st, m)
            y = y + _harmonics(st, m, phi, p["harm_db"], p["tilt"])
        if "l_f" in st:
            y = y + _lines(st, m)
    elif "f0" in st or "l_f" in st:
        y = y + _tonal_take(st, ch, T_out, n, p["harm_db"], p["tilt"])
    if "events" in st:
        drng = np.random.default_rng(seed + 99)
        ev = place_events(st, ch or [(0.0, 0.0, T_out)], T_out)
        d = float(p["density"])
        if d < 1:           # fewer: the small ticks thin out first, the big pops mostly stay
            wj = (st["ev_energy"] / np.median(st["ev_energy"])) ** 0.3
            wm = np.mean([wj[j] for j, _ in ev]) if ev else 1.0
            ev = [e for e in ev if drng.uniform() < min(1.0, d * wj[e[0]] / wm)]
        if d > 1 and ch2 is not None:   # more: events of a second, independent take added with probability d - 1
            ev += [e for e in place_events(st, ch2, T_out) if drng.uniform() < d - 1]
        y = y + _events(src, st, ev, n, seed) * 10 ** (p["crackle_db"] / 20)
        LAST["events"] = [so / SR for _, so in ev]
    c = st["hf_ceil_db"]
    f = np.fft.rfftfreq(4096, 1 / SR)[:len(c)]
    y = np.fft.irfft(np.fft.rfft(y) * 10 ** (np.interp(np.fft.rfftfreq(n, 1 / SR), f, c) / 20), n)
    hs, ls, dlp = _distance(p["distance"])
    if p["ls_db"] + ls:
        y = signal.sosfiltfilt(_shelf(p["ls_hz"], p["ls_db"] + ls, "low"), y)
    if p["hs_db"] + hs:
        y = signal.sosfiltfilt(_shelf(p["hs_hz"], p["hs_db"] + hs, "high"), y)
    y = _tape(y, cents)
    if cents < 0 and p["slow_mode"] == "tape_lp":
        y = _lp(y, 16000 * 2 ** (cents / 1200) * 0.8)      # the recording's ~16 kHz edge moves down with the tape
    y = _lp(y, dlp or p["lp_hz"])
    if ch is not None and st["kind"] == "loop":     # the take's length (as recorded: follows the tape speed)
        y = y[:int(round((float(p["length_s"]) or st["n"] / SR / 2 ** (cents / 1200)) * SR))]
    return y


def render_fire(p: dict, rng=None) -> np.ndarray:
    """Render a fire_stems parameter set at 44.1 kHz (peak 0.89)."""
    src = p["source"]
    if src not in sources():
        raise RecipeError(f"fire_stems: unknown source {src!r}; stored: {sources()}")
    var = int(round(p["variation"]))
    seed = int(round(p["seed"]))
    if var:
        seed = seed * 1000 + abs(var) % 100000
    y = _render(src, p, seed, rng if rng is not None else np.random.default_rng(seed + 100))
    k = min(len(y), int(0.03 * SR))          # the take ends where the clip ends: a 30 ms fade, then silence
    y[-k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    return y / (np.abs(y).max() + 1e-12) * 0.89


@model("fire_stems", "sound",
       "Fire, crackle and electricity rebuilt from stems of CC0 recordings (numbers only, no audio): a campfire, a "
       "large blaze, embers, a match strike, a gas stove igniting, a sparkler, a fuse, a transformer hum, an electric "
       "arc and a flickering fluorescent tube. Band-envelope noise beds, every real crackle / pop / click as its own "
       "band envelopes, tracked harmonics and lines. Fresh takes replay forward-only pieces of the stems in a new "
       "order (the events ride with their pieces); speed is tape speed.",
       [
           Param("source", "campfire", "", choices=SOURCES, doc="stored source"),
           Param("length_s", 0.0, "s", 0, 600, "looping sources: length of the take (0 = as recorded); another length "
                 "plays a fresh take"),
           Param("bed_db", 0.0, "dB", -24, 12, "noise bed level"),
           Param("crackle_db", 0.0, "dB", -24, 12, "event layer level (crackles, pops, clicks, the strike)"),
           Param("density", 1.0, "", 0.3, 3.0, "events: < 1 thins them (small ticks first, big pops mostly stay); > 1 "
                 "adds the events of a second, independent take with probability density - 1"),
           Param("bias", 0.0, "", -2.0, 2.0, "looping sources: fresh take drawn towards the busier (> 0) or quieter "
                 "(< 0) stretches of the recording, by their transient activity (0 = every piece once, new order)"),
           Param("chunk_min_s", 0.0, "s", 0, 5, "fresh take: shortest piece (0 = the source's own, 0.6 s; blaze 1.2 s)"),
           Param("chunk_max_s", 0.0, "s", 0, 6, "fresh take: longest piece (0 = the source's own, 1.5 s; blaze 2.8 s)"),
           Param("harm_db", 0.0, "dB", -24, 12, "harmonic layer level (transformer, arc, tube)"),
           Param("tilt", 0.0, "", -1, 1, "harmonic tilt: amp x (k/8)^tilt"),
           Param("env_depth", 1.0, "", 0.5, 1.5, "noise envelope contrast"),
           Param("clicks", 7, "", 4, 13, "gas stove: clicks before the burner catches and burns (7 = as recorded); "
                 "fewer drop recorded click cycles, more repeat them (never one twice in a row)"),
           Param("flame_db", 0.0, "dB", -12, 12, "gas stove: burner level from when it catches"),
           Param("speed_cents", 0.0, "cents", -1200, 1200, "tape speed (pitch and time together)"),
           Param("slow_mode", "tape", "", choices=("tape", "tape_lp"), doc="slower: plain tape speed, or tape_lp: a "
                 "treble roll-off that follows the speed (the recording's ~16 kHz edge moves down)"),
           Param("lp_hz", 0.0, "Hz", 0, 20000, "treble roll-off (2nd order low-pass; 0 or 20000 = off)"),
           Param("distance", 1.0, "", 0.25, 4.0, "1 = as recorded; further = high shelf down and a softer low-pass; "
                 "nearer = more low end (level: the recipe's master gain)"),
           Param("hs_hz", 2500.0, "Hz", 200, 16000, "high shelf corner"),
           Param("hs_db", 0.0, "dB", -24, 12, "high shelf gain"),
           Param("ls_hz", 250.0, "Hz", 30, 2000, "low shelf corner"),
           Param("ls_db", 0.0, "dB", -12, 12, "low shelf gain (size, proximity)"),
           Param("seed", 0, "", 0, 2**31 - 1, "noise seed (the recipe default reproduces its reference)"),
           Param("variation", 0, "", -1e6, 1e6, "0 = the reference; others reseed (looping sources: a fresh take)"),
           Param("at", 0.0, "s", 0, None),
       ],
       ["McDermott & Simoncelli 2011 (sound texture statistics)", "McAulay & Quatieri 1986 (sinusoidal modelling)",
        "Hermes 1988 (subharmonic summation)", "Freesound CC0 recordings (numbers only; THIRD_PARTY.md)"],
       min_s=lambda p: 0.1)
def fire_stems(ctx: Ctx, x, **p):
    p = dict(p)
    at = float(p.pop("at"))
    y = render_fire(p)
    if ctx.sr != SR:
        g = math.gcd(int(ctx.sr), SR)
        y = signal.resample_poly(y, int(ctx.sr) // g, SR // g)
    out = np.zeros(ctx.n)
    i0 = int(at * ctx.sr)
    if i0 < ctx.n:
        m = min(len(y), ctx.n - i0)
        out[i0:i0 + m] = y[:m]
    return out
