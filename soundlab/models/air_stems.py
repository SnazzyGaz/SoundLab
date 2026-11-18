"""air_stems (v0.30): wind, whooshes, fly-bys and fans rebuilt from stems of CC0 recordings (numbers only).

Noise-dominant sounds get their own stems (see docs/air.md):
1. noise  - band-noise envelope: 49 log bands (40 Hz - 20 kHz) x 2.9 ms frames, tones taken out, smoothed ACROSS
            FREQUENCY (7 bands) with only 10 ms in time (fans: one blade period), so real flutter stays and the
            per-band estimation noise goes
2. tonal  - whistle / blade / buzz partial tracks (>= 60 ms, >= 10 dB over the local noise)
3. trans  - a 4-band 1 ms envelope, imposed only +-15 ms around onsets (snaps, cracks)
4. colour - a 1/6-octave long-term spectrum used as a static EQ (23 ms frames smear +-86 Hz at the bottom)
5. rotor  - fans: blade rate and the per-band shape of one blade cycle (folded over ~60 cycles)
6. passes - fly-bys: each pass of a recording separately, with its peak time and the Doppler geometry (speed,
            closest distance) of the recording; a variation re-flies a real pass faster/nearer: time scales with
            distance / speed, the Doppler curve is re-drawn per pass, level 1/r, air absorption brighter when near.
Resynthesis: fresh complex noise shaped by the envelope (one calibration pass), plus the tracks, the onset
envelopes and the colour EQ.
"""
from __future__ import annotations

import copy
import functools
import json
import lzma
import math
import struct
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from ..core import Ctx, Param, RecipeError, model

SR, N, H = 44100, 1024, 128
FR = SR / H
MS = int(SR / 1000)
FB = [(20, 500), (500, 2000), (2000, 6000), (6000, 20000)]
GRID6 = 1000 * 2 ** (np.arange(-34, 26) / 6)       # 1/6 octave, 20 Hz - 18 kHz
C_AIR = 343.0
DATA = Path(__file__).with_name("air_stems.bin")


# ---------------------------------------------------------------------------------------------- data
@functools.lru_cache(maxsize=1)
def _data():
    raw = DATA.read_bytes()
    if raw[:8] != b"SLAIR001":
        raise RecipeError(f"{DATA.name}: not a soundlab air stem file")
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    hdr = json.loads(p[4:4 + n].decode())
    blob = p[4 + n:]

    def arr(d):
        return np.frombuffer(blob, np.dtype(d["dt"]), int(np.prod(d["sh"])), d["o"]).reshape(d["sh"])

    segs = {}
    for k, s in hdr["segments"].items():
        st = dict(n=s["n"], B=[tuple(b) for b in hdr["B"]], cen_b=arr(hdr["cen_b"]).astype(np.float64),
                  E=10 ** ((arr(s["E"]).astype(np.float64) * 0.5 + s["E_lo"]) / 10),
                  fine=10 ** ((arr(s["fine"]).astype(np.float64) * 0.5 + s["fine_lo"]) / 20),
                  w=arr(s["w"]).astype(np.float64) / 255 * s["w_max"], ltas6=arr(s["ltas6"]).astype(np.float64))
        st0, ln = arr(s["t_start"]), arr(s["t_len"])
        f, db, ph = arr(s["t_f"]).astype(np.float64), arr(s["t_db"]).astype(np.float64), arr(s["t_ph"]).astype(np.float64)
        cut = np.concatenate([[0], np.cumsum(ln)])
        st["tonal"] = [(int(st0[j]), f[cut[j]:cut[j + 1]], 10 ** (db[cut[j]:cut[j + 1]] / 20), ph[cut[j]:cut[j + 1]])
                       for j in range(len(ln))]
        st["rotor"] = None
        if "rotor" in s:
            r = s["rotor"]
            st["rotor"] = dict(rate=r["rate"], period_fr=r["period_fr"], depth_db=r["depth_db"],
                               shape=arr(r["shape"]).astype(np.float64))
        meta = {kk: v for kk, v in s.items() if kk in ("family", "src", "t0", "pass_peak", "tc", "v", "d")}
        segs[k] = (st, meta)
    return dict(segments=segs, families=hdr["families"], files=hdr["files"])


def segment_keys() -> dict:
    return {k: list(v) for k, v in _data()["families"].items()}


# ---------------------------------------------------------------------------------------------- dsp
def _stft(x):
    f, _, Z = signal.stft(x, SR, nperseg=N, noverlap=N - H, boundary="even", padded=True)
    return f, Z


def _istft(Z, n):
    _, y = signal.istft(Z, SR, nperseg=N, noverlap=N - H, boundary=True)
    y = y[:n]
    return np.pad(y, (0, n - len(y)))


def _band_env(P, B):
    return np.stack([P[a:b].mean(0) for a, b in B])


def _gain(E, B, nf):
    G = np.zeros((nf, E.shape[1]))
    for (a, b), e in zip(B, E):
        G[a:b] = e[None, :]
    return G


def _ltas6(x):
    f, P = signal.welch(x, SR, nperseg=min(8192, len(x)))
    out = []
    for g in GRID6:
        m = (f >= g / 2 ** (1 / 12)) & (f < g * 2 ** (1 / 12))
        out.append(10 * np.log10((P[m].mean() if m.any() else P[np.argmin(np.abs(f - g))]) + 1e-20))
    return np.array(out)


def _eq_to(y, target_db):
    d = np.clip(target_db - _ltas6(y), -40, 20)
    f, Z = _stft(y)
    g = 10 ** (np.interp(np.log(np.maximum(f, 10)), np.log(GRID6), d) / 20)
    return _istft(Z * g[:, None], len(y))


def _synth_track(out, s, f, A, P):
    two_pi = 2 * np.pi
    n = len(f)
    i = np.arange(H)
    for m in range(-1, n):
        if m == -1:
            a0, a1 = 0.0, A[0]; w0 = w1 = two_pi * f[0] / SR; p1 = P[0]; p0 = p1 - w1 * H
        elif m == n - 1:
            a0, a1 = A[m], 0.0; w0 = w1 = two_pi * f[m] / SR; p0 = P[m]; p1 = p0 + w0 * H
        else:
            a0, a1 = A[m], A[m + 1]; w0 = two_pi * f[m] / SR; w1 = two_pi * f[m + 1] / SR; p0 = P[m]; p1 = P[m + 1]
        M = np.floor(((p0 + w0 * H - p1) + (w1 - w0) * H / 2) / two_pi + 0.5)       # MQ cubic phase
        e = p1 + two_pi * M - p0 - w0 * H
        al = 3.0 / H ** 2 * e - (w1 - w0) / H
        be = -2.0 / H ** 3 * e + (w1 - w0) / H ** 2
        base = (s + m) * H
        lo, hi = max(0, -base), min(H, len(out) - base)
        if lo >= hi:
            continue
        ii = i[lo:hi]
        out[base + lo:base + hi] += (a0 + (a1 - a0) * ii / H) * np.cos(p0 + w0 * ii + al * ii * ii + be * ii ** 3)


def resynth(st, seed=0):
    rng = np.random.default_rng(seed)
    n, B, E = st["n"], st["B"], st["E"]
    nf = N // 2 + 1
    nt = E.shape[1]
    W = (rng.standard_normal((nf, nt)) + 1j * rng.standard_normal((nf, nt))) / np.sqrt(2)
    if st.get("rotor"):
        r_ = st["rotor"]
        nph = r_["shape"].shape[1]
        ph = (np.arange(nt) * r_["rate"] / FR) % 1.0
        xs = np.arange(nph + 1) / nph
        E = E * 10 ** (np.stack([np.interp(ph, xs, np.append(sh, sh[0])) for sh in r_["shape"]]) / 10)
    G = _gain(E, B, nf)
    y = _istft(W * np.sqrt(G), n)
    _, Zy = _stft(y)
    if Zy.shape[1] < nt:
        Zy = np.pad(Zy, ((0, 0), (0, nt - Zy.shape[1])), mode="edge")
    Ey = _band_env(np.abs(Zy[:, :nt]) ** 2, B)
    k = int(0.025 * FR) | 1
    r = uniform_filter1d(E, k, axis=1) / (uniform_filter1d(Ey, k, axis=1) + 1e-20)
    y = _istft(W * np.sqrt(G * _gain(r, B, nf)), n)
    if st["tonal"]:
        tone = np.zeros(n)
        for s0, f, A, P in st["tonal"]:
            _synth_track(tone, s0, f, A, P)
        y = y + tone
    if st["w"].max() > 0:
        fe, w = st["fine"], st["w"]
        out = y.copy()
        for i, (lo, hi) in enumerate(FB):
            sos = signal.butter(4, [lo / (SR / 2), min(hi, SR / 2 - 100) / (SR / 2)], "band", output="sos")
            yb = signal.sosfiltfilt(sos, y)
            eb = np.sqrt(uniform_filter1d(yb ** 2, MS)[::MS] + 1e-20)
            m = min(len(eb), fe.shape[1], len(w))
            g = 1 + w[:m] * (np.clip(fe[i, :m] / eb[:m], 0.1, 10) - 1)
            out += yb * (np.interp(np.arange(n) / MS, np.arange(m), g) - 1)
        y = out
    return _eq_to(y, st["ltas6"])


# ---------------------------------------------------------------------------------------------- transforms
def _tstretch(a, m):
    n = a.shape[-1]
    xs = np.linspace(0, n - 1, m)
    return np.apply_along_axis(lambda v: np.interp(xs, np.arange(n), v), -1, a)


def transform(st, stretch=1.0, cents=0.0, gain_db=0.0, shelf=None, rotor_scale=1.0, cents_curve=None, depth=1.0):
    """stretch (>1 slower), cents (whole spectrum), shelf=(hz, db) on the colour, rotor_scale (blade rate),
    cents_curve (per output frame: Doppler), depth (<1: band envelopes pulled towards their mean, shallower gusts)."""
    s = copy.copy(st)
    nt = st["E"].shape[1]
    mt = max(4, int(round(nt * stretch)))
    s["n"] = int(round(st["n"] * stretch))
    E = _tstretch(st["E"], mt)
    if depth != 1.0:
        L = np.log(E + 1e-30)
        mu = L.mean(axis=1, keepdims=True)
        E = np.exp(mu + depth * (L - mu))
    mm = max(4, int(round(st["fine"].shape[1] * stretch)))
    s["fine"] = _tstretch(st["fine"], mm)
    s["w"] = _tstretch(st["w"], mm)
    cc = np.full(mt, float(cents))
    if cents_curve is not None:
        cc = cc + np.interp(np.arange(mt), np.linspace(0, mt - 1, len(cents_curve)), cents_curve)
    lc = np.log2(st["cen_b"])
    if np.any(cc != 0):
        LE = np.log(E + 1e-30)
        for t in np.nonzero(cc)[0]:
            LE[:, t] = np.interp(lc - cc[t] / 1200, lc, LE[:, t])
        E = np.exp(LE)
    E = E * 10 ** (gain_db / 10)
    s["E"] = E
    s["fine"] = s["fine"] * 10 ** (gain_db / 20)
    tr = []
    for s0, fq, A, _ in st["tonal"]:
        L_ = max(2, int(round(len(fq) * stretch)))
        t0 = int(round(s0 * stretch))
        idx = np.clip(np.arange(t0, t0 + L_), 0, mt - 1)
        f2 = _tstretch(fq, L_) * 2 ** (cc[idx] / 1200)
        tr.append((t0, f2, _tstretch(A, L_) * 10 ** (gain_db / 20), np.cumsum(2 * np.pi * f2 * H / SR)))
    s["tonal"] = tr
    l6 = np.interp(np.log2(GRID6) - np.mean(cc) / 1200, np.log2(GRID6), st["ltas6"]) + gain_db
    if shelf:
        hz, db = shelf
        l6 = l6 + db / (1 + (hz / GRID6) ** 4)
    s["ltas6"] = l6
    if st.get("rotor"):
        s["rotor"] = dict(st["rotor"], rate=st["rotor"]["rate"] * rotor_scale / stretch)
    return s


def _doppler_cents(t, tc, v, d):
    vr = v * v * (tc - t) / np.sqrt(v * v * (t - tc) ** 2 + d * d)      # radial speed towards the listener
    return 1200 * np.log2(C_AIR / (C_AIR - np.clip(vr, -0.9 * C_AIR, 0.9 * C_AIR)))


def repass(st, meta, speed=1.0, closeness=1.0):
    v, d, tc = meta["v"], meta["d"], meta["tc"]
    v2, d2 = v * speed, d / closeness
    sf = (d2 / v2) / (d / v)
    mt = int(round(st["E"].shape[1] * sf))
    t_new = np.arange(mt) / FR
    tc2 = tc * sf
    dc = _doppler_cents(t_new, tc2, v2, d2) - _doppler_cents(tc + (t_new - tc2) / sf, tc, v, d)
    return transform(st, stretch=sf, cents_curve=dc, gain_db=20 * np.log10(closeness),
                     shelf=(5000, 3 * np.log2(closeness)))


# ---------------------------------------------------------------------------------------------- render
def _segment(key, mute):
    segs = _data()["segments"]
    if key not in segs:
        raise RecipeError(f"air_stems: unknown segment {key!r}; stored: {sorted(segs)}")
    st, meta = segs[key]
    if mute:
        st = dict(st, tonal=[t for t in st["tonal"] if np.median(t[1]) < mute])
    return st, meta


def render_air(p: dict) -> np.ndarray:
    """Render an air_stems parameter set at 44.1 kHz (peak 0.89)."""
    var = int(round(p["variation"]))
    seed = int(round(p["seed"]))
    if p["mapping"] == "flyby":
        return _flyby(p, seed + abs(var) % 1000003)
    key = p["segment"]
    if var:
        fam = _data()["families"][_data()["segments"][key][1]["family"]]
        key = fam[var % len(fam)]
        seed = seed * 1000 + abs(var) % 100000
    st, _ = _segment(key, p["mute_above_hz"])
    s_ = float(p["strength"])
    speed, size = float(p["speed"]), float(p["size"])
    kw = {}
    if p["mapping"] == "fan":
        kw = dict(cents=p["pitch"] + 1200 * math.log2(speed) - 960 * math.log2(size), rotor_scale=speed / size,
                  gain_db=50 * math.log10(speed), stretch=p["stretch"])
    else:
        r = 0.25 + 0.75 * s_                               # wind speed re the recording (strength 1)
        kw = dict(stretch=p["stretch"] * size ** 0.5 / speed * (1 + 0.8 * (1 - s_)),
                  cents=p["pitch"] + 600 * math.log2(speed) - 840 * math.log2(size) + 600 * math.log2(r),
                  depth=0.35 + 0.65 * s_)
    shelf_db = p["air_db"] - 8.0 * (1 - s_)
    if shelf_db:
        kw["shelf"] = (p["air_hz"], shelf_db)
    ident = (kw.get("stretch", 1) == 1 and kw.get("cents", 0) == 0 and not kw.get("shelf") and kw.get("depth", 1) == 1
             and kw.get("rotor_scale", 1) == 1 and kw.get("gain_db", 0) == 0)
    y = resynth(st if ident else transform(st, **kw), seed=seed)
    return y / (np.abs(y).max() + 1e-12) * 0.89


def _flyby(p, seed):
    segs = _data()["families"]["flyby"]
    P = [_segment(k, 0) for k in segs]
    rng = np.random.default_rng(seed)
    order, out = [], np.zeros(int(0.3 * SR))
    count, spacing, speed, close = int(round(p["count"])), p["spacing"], p["speed"], p["closeness"]
    for i in range(count):
        j = int(rng.integers(len(P)))
        while order and j == order[-1]:
            j = int(rng.integers(len(P)))
        order.append(j)
        st, meta = P[j]
        y = resynth(repass(st, meta, speed=speed * rng.uniform(0.92, 1.08), closeness=close * rng.uniform(0.85, 1.15)),
                    seed=seed * 10 + i)
        y = y / (np.abs(y).max() + 1e-12) * meta["pass_peak"] * close     # the real pass's own level, 1/r
        start = len(out) - int(0.15 * SR) if i else len(out)
        gap = int(spacing * rng.uniform(0.7, 1.3) * SR)
        need = start + gap + len(y)
        if need > len(out):
            out = np.pad(out, (0, need - len(out)))
        out[start + gap:start + gap + len(y)] += y
    y = np.pad(out, (0, int(0.3 * SR)))
    return y / (np.abs(y).max() + 1e-12) * 0.89


@model("air_stems", "sound",
       "Wind gusts, wind in trees, whooshes, arrow fly-bys and fans rebuilt from stems of CC0 recordings (numbers "
       "only, no audio): a band-noise envelope smoothed across frequency, tonal tracks, onset envelopes, colour, "
       "a fan's blade cycle and a fly-by's passes with their Doppler geometry. Real segments, new noise, mild changes.",
       [
           Param("segment", "wind_352421@10", "", doc="stored segment key (family@start); variations pick others of "
                 "the same family"),
           Param("mapping", "wind", "", choices=("wind", "whoosh", "fan", "flyby"),
                 doc="how speed/size act: wind/whoosh (time and spectrum), fan (blade rate, tip-speed brightness), "
                 "flyby (passes)"),
           Param("stretch", 1.0, "", 0.5, 3.0, "time stretch (>1 slower)"),
           Param("pitch", 0.0, "cents", -1200, 1200, "spectral shift of noise bands, tones and colour"),
           Param("speed", 1.0, "", 0.3, 3.0, "motion speed: faster = shorter and a little higher; fan: blade rate"),
           Param("size", 1.0, "", 0.3, 3.0, "bigger = lower and longer; fan: lower, slower blades"),
           Param("strength", 1.0, "", 0.0, 1.0, "wind strength re the recording (1): lower = darker, slower, "
                 "shallower gusts (level via the recipe's master gain)"),
           Param("air_db", 0.0, "dB", -24, 12, "high shelf on the colour (harshness / air)"),
           Param("air_hz", 3500.0, "Hz", 1000, 12000),
           Param("mute_above_hz", 0.0, "Hz", 0, 20000, "mute tonal tracks above this (0 = keep all)"),
           Param("count", 3, "", 1, 12, "fly-by: number of passes"),
           Param("spacing", 0.9, "s", 0.1, 4.0, "fly-by: mean gap between passes"),
           Param("closeness", 1.0, "", 0.3, 3.0, "fly-by: >1 nearer (louder, shorter, steeper Doppler, brighter)"),
           Param("seed", 0, "", 0, 2**31 - 1, "noise seed (the recipe default reproduces its reference)"),
           Param("variation", 0, "", -1e6, 1e6, "0 = the reference; other values pick another real segment and reseed"),
           Param("at", 0.0, "s", 0, None),
       ],
       ["Serra & Smith 1990 (sinusoids + residual)", "McAulay & Quatieri 1986 (sinusoidal tracking)",
        "Doppler for a straight pass: f' = f c / (c - v_r)", "fan law: sound power ~ tip speed^5",
        "Freesound CC0 recordings 352421, 760990, 507466, 675821, 761929 (numbers only; THIRD_PARTY.md)"],
       min_s=lambda p: 0.1)
def air_stems(ctx: Ctx, x, **p):
    p = dict(p)
    at = float(p.pop("at"))
    y = render_air(p)
    if ctx.sr != SR:
        g = math.gcd(int(ctx.sr), SR)
        y = signal.resample_poly(y, int(ctx.sr) // g, SR // g)
    out = np.zeros(ctx.n)
    i0 = int(at * ctx.sr)
    if i0 < ctx.n:
        m = min(len(y), ctx.n - i0)
        out[i0:i0 + m] = y[:m]
    return out
