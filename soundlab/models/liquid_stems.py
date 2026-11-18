"""liquid_stems (v0.31): splashes, drips, a pour, streams, rivers, a waterfall, boiling and sloshing rebuilt from
stems of public-domain / CC0 recordings (numbers only, no audio). See docs/liquids.md.

Two stem engines:
* band envelopes (splashes, drip, pour, river, waterfall): 78 raised-cosine bands at 1/8 octave (30 Hz - 20 kHz),
  each band's own amplitude envelope stored every 2 ms (0.5 dB steps), imposed on fresh noise carriers with a few
  band-limiting passes; prominent bubbles are taken out by subtraction first and stored as chirps (start, log-linear
  pitch, amplitude per 1.45 ms frame, start phase), then added back.
* STFT stems (stream, boiling, slosh): 49 log bands x 2.9 ms frames, bubble tracks, 1 ms onset envelopes in four
  bands (the air_stems engine, without its rotor/passes).
Both end with a 1/6-octave colour EQ. Variations remix the real stems (block reorder, re-sequenced drops, a
time-offset copy of the same bubbles); nothing is averaged.
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
from .air_stems import FB, FR, GRID6, H, MS, N, SR, _band_env, _eq_to, _gain, _istft, _stft

BH, BN = 64, 512                                   # bubble tracks: 1.45 ms frames
DATA = Path(__file__).with_name("liquid_stems.bin")
FAMILY = {"splash_small": "splash", "splash_medium": "splash", "splash_large": "splash", "drip": "drip",
          "pour_glass": "pour", "stream": "flow", "river": "flow", "waterfall": "flow", "boiling": "boiling",
          "slosh": "slosh"}


# ---------------------------------------------------------------------------------------------- data
@functools.lru_cache(maxsize=1)
def _data():
    raw = DATA.read_bytes()
    if raw[:8] != b"SLLIQ001":
        raise RecipeError(f"{DATA.name}: not a soundlab liquid stem file")
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    hdr = json.loads(p[4:4 + n].decode())
    blob = p[4 + n:]

    def arr(d):
        return np.frombuffer(blob, np.dtype(d["dt"]), int(np.prod(d["sh"])), d["o"]).reshape(d["sh"])

    out = {}
    for k, s in hdr["sources"].items():
        st = dict(n=s["n"], ltas6=arr(s["ltas6"]).astype(np.float64), engine=s["engine"])
        if s["engine"] == "env":
            q = arr(s["env"]).astype(np.float64)
            e = 10 ** ((q * 0.5 + s["env_lo"]) / 20)
            e[q == 0] = 0.0
            st.update(per_oct=s["per_oct"], env_sr=s["env_sr"], env=e)
            t0, ln = arr(s["c_start"]), arr(s["c_len"])
            a, b, ph0 = arr(s["c_a"]).astype(np.float64), arr(s["c_b"]).astype(np.float64), arr(s["c_ph"]).astype(np.float64)
            db = arr(s["c_db"]).astype(np.float64) * 0.5 + s["c_db_lo"]
            cut = np.concatenate([[0], np.cumsum(ln)])
            tr = []
            for j in range(len(ln)):
                tt = np.arange(ln[j])
                f = 2 ** (a[j] + b[j] * tt)
                tr.append((int(t0[j]), f, 10 ** (db[cut[j]:cut[j + 1]] / 20), np.cumsum(2 * np.pi * f * BH / SR) + ph0[j]))
            st["tonal"] = tr
        else:
            st.update(B=[tuple(b) for b in s["B"]],
                      E=10 ** ((arr(s["E"]).astype(np.float64) * 0.5 + s["E_lo"]) / 10),
                      fine=10 ** ((arr(s["fine"]).astype(np.float64) * 0.5 + s["fine_lo"]) / 20),
                      w=arr(s["w"]).astype(np.float64) / 255 * s["w_max"])
            st0, ln = arr(s["t_start"]), arr(s["t_len"])
            f, db, ph = arr(s["t_f"]).astype(np.float64), arr(s["t_db"]).astype(np.float64), arr(s["t_ph"]).astype(np.float64)
            cut = np.concatenate([[0], np.cumsum(ln)])
            st["tonal"] = [(int(st0[j]), f[cut[j]:cut[j + 1]], 10 ** (db[cut[j]:cut[j + 1]] / 20), ph[cut[j]:cut[j + 1]])
                           for j in range(len(ln))]
        out[k] = (st, {kk: v for kk, v in s.items() if kk in ("file", "t0", "t1", "cleaning", "settings")})
    return dict(sources=out, files=hdr["files"])


# ---------------------------------------------------------------------------------------------- dsp
def _synth_tracks(tracks, n):
    out = np.zeros(n)
    two_pi = 2 * np.pi
    i = np.arange(BH)
    for s, f, A, P in tracks:
        for m in range(-1, len(f)):
            if m == -1:
                a0, a1 = 0.0, A[0]; w0 = w1 = two_pi * f[0] / SR; p1 = P[0]; p0 = p1 - w1 * BH
            elif m == len(f) - 1:
                a0, a1 = A[m], 0.0; w0 = w1 = two_pi * f[m] / SR; p0 = P[m]; p1 = p0 + w0 * BH
            else:
                a0, a1 = A[m], A[m + 1]; w0 = two_pi * f[m] / SR; w1 = two_pi * f[m + 1] / SR; p0 = P[m]; p1 = P[m + 1]
            M = np.floor(((p0 + w0 * BH - p1) + (w1 - w0) * BH / 2) / two_pi + 0.5)      # MQ cubic phase
            e = p1 + two_pi * M - p0 - w0 * BH
            al = 3.0 / BH ** 2 * e - (w1 - w0) / BH
            be = -2.0 / BH ** 3 * e + (w1 - w0) / BH ** 2
            base = (s + m) * BH
            lo, hi = max(0, -base), min(BH, n - base)
            if lo >= hi:
                continue
            ii = i[lo:hi]
            out[base + lo:base + hi] += (a0 + (a1 - a0) * ii / BH) * np.cos(p0 + w0 * ii + al * ii * ii + be * ii ** 3)
    return out


def _bank(n, per_oct, lo=30.0, hi=20000.0):
    f = np.fft.rfftfreq(n, 1 / SR)
    lf = np.log2(np.maximum(f, 1.0))
    c = np.arange(np.log2(lo), np.log2(hi) + 1e-9, 1 / per_oct)
    d = 1 / per_oct
    Hs = []
    for i, ck in enumerate(c):
        u = (lf - ck) / d
        h = np.where(np.abs(u) < 1, np.cos(np.pi / 2 * u), 0.0)
        if i == 0:
            h = np.where(lf <= ck, 1.0, h)
        if i == len(c) - 1:
            h = np.where(lf >= ck, 1.0, h)
        Hs.append(h)
    return np.array(Hs), c


def _analytic(X, h, n):
    Y = np.zeros(n, complex)
    m = len(X)
    Y[:m] = X * h * 2
    Y[0] *= 0.5
    if n % 2 == 0:
        Y[m - 1] *= 0.5
    return np.fft.ifft(Y)


def _resynth_env(st, seed, iters=4):
    rng = np.random.default_rng(seed)
    n = st["n"]
    Hs, _ = _bank(n, st["per_oct"])
    t = np.arange(n) / SR
    te = np.arange(st["env"].shape[1]) / st["env_sr"]
    y = rng.standard_normal(n)
    E = [np.interp(t, te, e) for e in st["env"]]
    for _ in range(iters):
        Y = np.fft.rfft(y)
        out = np.zeros(len(Y), complex)
        for h, e in zip(Hs, E):
            a = _analytic(Y, h, n)
            c = a / (np.abs(a) + 1e-9 * (np.abs(a).max() + 1e-30))
            out += np.fft.rfft((e * c).real) * h
        y = np.fft.irfft(out, n)
    if st["tonal"]:
        y = y + _synth_tracks(st["tonal"], n + BN)[:n]
    return _eq_to(y, st["ltas6"])


def _resynth_stft(st, seed):
    rng = np.random.default_rng(seed)
    n, B, E = st["n"], st["B"], st["E"]
    nf = N // 2 + 1
    nt = E.shape[1]
    W = (rng.standard_normal((nf, nt)) + 1j * rng.standard_normal((nf, nt))) / np.sqrt(2)
    G = _gain(E, B, nf)
    y = _istft(W * np.sqrt(G), n)
    _, Zy = _stft(y)
    if Zy.shape[1] < nt:
        Zy = np.pad(Zy, ((0, 0), (0, nt - Zy.shape[1])), mode="edge")
    Ey = _band_env(np.abs(Zy[:, :nt]) ** 2, B)
    k = int(0.01 * FR) | 1
    r = np.maximum(uniform_filter1d(E, k, axis=1), 0) / (np.maximum(uniform_filter1d(Ey, k, axis=1), 0) + 1e-20)
    y = _istft(W * np.sqrt(G * _gain(r, B, nf)), n)
    if st["tonal"]:
        y = y + _synth_tracks(st["tonal"], n + BN)[:n]
    if st["w"].max() > 0:
        fe, w = st["fine"], st["w"]
        out = y.copy()
        for i, (lo, hi) in enumerate(FB):
            sos = signal.butter(4, [lo / (SR / 2), min(hi, SR / 2 - 100) / (SR / 2)], "band", output="sos")
            yb = signal.sosfiltfilt(sos, y)
            eb = np.sqrt(np.maximum(uniform_filter1d(yb ** 2, MS)[::MS], 0) + 1e-20)
            m = min(len(eb), fe.shape[1], len(w))
            g = 1 + w[:m] * (np.clip(fe[i, :m] / eb[:m], 0.1, 10) - 1)
            out += yb * (np.interp(np.arange(n) / MS, np.arange(m), g) - 1)
        y = out
    return _eq_to(y, st["ltas6"])


def resynth(st, seed=0):
    y = _resynth_env(st, seed) if st["engine"] == "env" else _resynth_stft(st, seed)
    if st.get("tape"):                               # tape speed: pitch and time together (no band interpolation)
        y = signal.resample(y, int(round(len(y) / 2 ** (st["tape"] / 1200))))
    if st.get("lp"):                                 # soft roll-off (lower flow strength)
        y = signal.sosfiltfilt(signal.butter(2, st["lp"], fs=SR, output="sos"), y)
    return y


# ---------------------------------------------------------------------------------------------- transforms
def _warp_cols(A, rate, n_out_s, n_in_s, tin):
    """Resample the columns of A (rate columns/s) to n_out_s seconds; column j holds input time tin(j / rate). Past
    the end the real stems are read backwards (mirror), never held: a held last frame turns into a tone."""
    m = max(4, int(round(A.shape[-1] * n_out_s / n_in_s)))
    ti = tin(np.arange(m) / rate) * rate
    L = A.shape[-1] - 1
    p = np.mod(np.maximum(ti, 0), 2 * L)
    ti = np.where(p <= L, p, 2 * L - p)
    i0 = np.floor(ti).astype(int)
    i1 = np.minimum(i0 + 1, L)
    fr = ti - i0
    return A[..., i0] * (1 - fr) + A[..., i1] * fr


def _shift_rows(LE, lc, cents):
    if not cents:
        return LE
    k = np.interp(lc - cents / 1200.0, lc, np.arange(len(lc)))
    i0 = np.floor(k).astype(int)
    i1 = np.minimum(i0 + 1, len(lc) - 1)
    fr = (k - i0)[:, None]
    return LE[i0] * (1 - fr) + LE[i1] * fr


def _depth(LE, depth, win):
    if depth == 1.0:
        return LE
    m = uniform_filter1d(LE, win, axis=1, mode="nearest")
    return m + depth * (LE - m)


def _stretch(s):
    return (lambda to: to / s), (lambda ti: ti * s)


def _tail_warp(tp, k):
    return (lambda to: np.where(to < tp, to, tp + (to - tp) / k)), (lambda ti: np.where(ti < tp, ti, tp + (ti - tp) * k))


def _band_centres(st):
    if st["engine"] == "env":
        return _bank(256, st["per_oct"])[1]
    f = np.fft.rfftfreq(N, 1 / SR)
    return np.log2(np.array([np.sqrt(f[a] * f[b - 1] + 1) for a, b in st["B"]]))


def transform(st, warp=None, dur_s=None, cents=0.0, gain_db=0.0, shelf=None, depth=1.0, tr_dur=1.0, tr_gain_db=0.0,
              tr_keep=None):
    """warp=(tin(tout), tout(tin)) in seconds; cents shifts bands, bubbles and colour; depth scales each band's own
    modulation around its 150 ms mean (log); tr_dur scales bubble ring length."""
    s = copy.copy(st)
    tin, tout = warp or _stretch(1.0)
    n_s = st["n"] / SR
    out_s = dur_s if dur_s is not None else float(tout(n_s))
    s["n"] = int(round(out_s * SR))
    lc = _band_centres(st)
    if st["engine"] == "env":
        E = st["env"]
        eps = 1e-4 * E.max() + 1e-12
        LE = np.log(_warp_cols(E, st["env_sr"], out_s, n_s, tin) + eps)
        LE = _depth(_shift_rows(LE, lc, cents), depth, int(0.15 * st["env_sr"]) | 1)
        s["env"] = np.maximum(np.exp(LE) - eps, 0)
    else:
        E = st["E"]
        eps = 1e-6 * E.max() + 1e-30
        LE = np.log(_warp_cols(E, SR / H, out_s, n_s, tin) + eps)
        LE = _depth(_shift_rows(LE, lc, cents), depth, max(3, int(0.15 * FR)) | 1)
        s["E"] = np.maximum(np.exp(LE) - eps, 0)
        s["fine"] = _warp_cols(st["fine"], SR / MS, out_s, n_s, tin)
        s["w"] = _warp_cols(st["w"], SR / MS, out_s, n_s, tin)
        if depth != 1.0:
            s["fine"] = np.exp(_depth(np.log(s["fine"] + 1e-12), depth, 151))
    tr = []
    for i, (s0, fq, A, ph) in enumerate(st["tonal"]):
        if tr_keep is not None and not tr_keep[i]:
            continue
        t0 = float(tout(s0 * BH / SR))
        L = max(3, int(round(len(fq) * tr_dur)))
        xs = np.linspace(0, len(fq) - 1, L)
        f2 = np.interp(xs, np.arange(len(fq)), fq) * 2 ** (cents / 1200)
        A2 = np.interp(xs, np.arange(len(fq)), A) * 10 ** (tr_gain_db / 20)
        s1 = int(round(t0 * SR / BH))
        if s1 * BH < s["n"] - BN:
            tr.append((s1, f2, A2, np.cumsum(2 * np.pi * f2 * BH / SR) + ph[0]))
    s["tonal"] = tr
    l6 = np.interp(np.log2(GRID6) - cents / 1200, np.log2(GRID6), st["ltas6"]) + gain_db
    if shelf:
        hz, db = shelf
        l6 = l6 + db / (1 + (hz / GRID6) ** 4)
    s["ltas6"] = l6
    return s


def shuffle_blocks(st, rng, block_s=1.6, xf_s=0.08):
    """Remix a steady texture: reorder real blocks of its stems (and the bubbles inside them), power crossfades."""
    s = copy.copy(st)
    n_s = st["n"] / SR
    nb = max(2, int(n_s // block_s))
    edges = np.linspace(0, n_s, nb + 1)
    order = rng.permutation(nb)
    while nb > 2 and np.all(order == np.arange(nb)):
        order = rng.permutation(nb)

    def remap(A, rate):
        out = np.zeros_like(A)
        pos = 0.0
        xf = int(xf_s * rate)
        L = A.shape[-1] - 1
        for j, b in enumerate(order):
            a0, a1 = int(edges[b] * rate), int(edges[b + 1] * rate)
            p0 = int(pos * rate)
            if j == len(order) - 1:                  # last block runs on to the end (mirror-read), no silent gap
                a1 = a0 + A.shape[-1] - p0
            ii = np.mod(np.arange(a0, a1), 2 * L)
            seg = A[..., np.where(ii <= L, ii, 2 * L - ii)] ** 2
            m = min(seg.shape[-1], A.shape[-1] - p0)
            w = np.ones(m)
            k = min(xf, m // 2)
            if p0 > 0:
                w[:k] = np.linspace(0, 1, k)
            if p0 + m < A.shape[-1]:
                w[m - k:] = np.minimum(w[m - k:], np.linspace(1, 0, k))
            out[..., p0:p0 + m] += seg[..., :m] * w
            pos += (edges[b + 1] - edges[b]) - (xf_s if p0 + m < A.shape[-1] else 0)
        return np.sqrt(out)

    starts, pos = {}, 0.0
    for b in order:
        starts[b] = pos
        pos += edges[b + 1] - edges[b] - xf_s
    if st["engine"] == "env":
        s["env"] = remap(st["env"], st["env_sr"])
    else:
        s["E"] = remap(st["E"], FR)
        s["fine"] = remap(st["fine"], SR / MS)
        s["w"] = remap(st["w"], SR / MS)
    tr = []
    for s0, fq, A, ph in st["tonal"]:
        t = s0 * BH / SR
        b = min(nb - 1, int(np.searchsorted(edges, t, "right") - 1))
        t2 = starts[b] + t - edges[b]
        if t2 * SR + len(fq) * BH < st["n"]:
            tr.append((int(round(t2 * SR / BH)), fq, A, ph))
    s["tonal"] = tr
    return s


def drop_events(st, rel_db=30.0, min_gap=0.35):
    E = st["env"]
    es = st["env_sr"]
    tot = 20 * np.log10(np.sqrt((E ** 2).sum(0)) + 1e-12)
    d = np.diff(uniform_filter1d(tot, max(1, int(round(0.005 * es)))))
    on = []
    for i in np.nonzero((d > 1.5 * 1000 / es) & (tot[1:] > tot.max() - rel_db))[0]:
        if not on or i - on[-1] > min_gap * es:
            on.append(i)
    return [max(0, o - int(0.01 * es)) for o in on]


def drip_remix(st, rng, rate=1.0, cents=0.0, jitter=0.15):
    """Re-sequence the real drops: each drop's envelope and its ring bubbles move as one event; rate scales the drop
    rate, cents shifts each drop (smaller drop = higher, shorter ring)."""
    es = st["env_sr"]
    on = drop_events(st)
    E = st["env"]
    T = E.shape[1]
    segs = []
    for i, o in enumerate(on):
        e = on[i + 1] if i + 1 < len(on) else T
        trk = [(s0 - int(round(o / es * SR / BH)), fq, A, ph) for s0, fq, A, ph in st["tonal"] if o <= s0 * BH / SR * es < e]
        segs.append((E[:, o:e], trk))
    gaps = np.diff(on)
    g0 = float(np.median(gaps)) if len(gaps) else 0.5 * es
    out = np.zeros_like(E)
    tr = []
    t = on[0] if on else 0
    last = -1
    while t < T and segs:
        j = int(rng.choice([i for i in range(len(segs)) if i != last] or [0]))
        last = j
        seg, trk = segs[j]
        sub = dict(st, n=int(seg.shape[1] / es * SR), env=seg, tonal=trk)
        if cents:
            sub = transform(sub, cents=cents, tr_dur=2 ** (-cents / 1200))
        m = min(sub["env"].shape[1], T - t)
        out[:, t:t + m] = np.sqrt(out[:, t:t + m] ** 2 + sub["env"][:, :m] ** 2)
        for s0, fq, A, ph in sub["tonal"]:
            s1 = s0 + int(round(t / es * SR / BH))
            if (s1 + len(fq)) * BH < st["n"]:
                tr.append((s1, fq, A, ph))
        t += max(1, int(g0 / rate * (1 + jitter * rng.uniform(-1, 1))))
    s = copy.copy(st)
    s["env"] = out
    s["tonal"] = tr
    s["ltas6"] = np.interp(np.log2(GRID6) - cents / 1200, np.log2(GRID6), st["ltas6"])
    return s


def shorten_tail(st, tail):
    """Shorter splash tail: a faster decay after the peak (time-compressing the tail sped its own sloshing into a
    wobble)."""
    s = copy.copy(st)
    E = st["env"]
    es = st["env_sr"]
    tot = 10 * np.log10((E ** 2).sum(0) + 1e-30)
    ip = int(np.argmax(tot))
    tp = ip + int(0.04 * es)
    seg = tot[tp:]
    end = tp + int(np.argmax(seg < tot[ip] - 40)) if np.any(seg < tot[ip] - 40) else len(tot)
    d = -np.polyfit(np.arange(tp, end) / es, tot[tp:end], 1)[0] if end - tp > 0.02 * es else 60.0
    k = d * (1 / tail - 1)
    t = np.maximum(np.arange(E.shape[1]) - tp, 0) / es
    s["env"] = E * (10 ** (-k * t / 20))[None, :]
    s["tonal"] = [(s0, fq, A * 10 ** (-k * np.maximum((s0 + np.arange(len(fq))) * BH / SR - tp / es, 0) / 20), ph)
                  for s0, fq, A, ph in st["tonal"]]
    return s


# ---------------------------------------------------------------------------------------------- macros
def splash(st, size=1.0, distance=1.0, tail=1.0, tape=False):
    """size: bigger = lower, longer (band shift, or tape speed with tape=True); distance: quieter, duller, softer
    onset; tail: >1 slows the tail, <1 fades it faster."""
    c = -700 * math.log2(size)
    if tape:
        s = transform(st, _stretch(size ** 0.5 * 2 ** (c / 1200)), gain_db=9 * math.log2(size))
        if c:
            s["tape"] = c
    elif size != 1.0:
        s = transform(st, _stretch(size ** (0.5 if size >= 1 else 0.3)), cents=c, gain_db=9 * math.log2(size), tr_dur=size)
    else:
        s = st
    if tail > 1.0:
        tp = float(np.argmax((s["env"] ** 2).sum(0))) / s["env_sr"] + 0.04
        s = transform(s, _tail_warp(tp, tail))
    elif tail < 1.0:
        s = shorten_tail(s, tail)
    if distance != 1.0:
        ld = math.log2(distance)
        s = transform(s, gain_db=-6 * ld, shelf=(2500, -5 * ld), depth=1 / (1 + 0.15 * ld))
    return s


def flow(st, strength=1.0):
    """stream / river / waterfall: lower = quieter, darker (shelf + soft low-pass), slower (envelope stretch),
    gentler. No pitch shift."""
    if strength == 1.0:
        return st
    ls = math.log2(strength)
    s = transform(st, _stretch(strength ** -0.3), dur_s=st["n"] / SR, gain_db=8 * ls, shelf=(3000, 6 * ls),
                  depth=strength ** 0.15, tr_gain_db=4 * ls)
    if strength < 1:
        s["lp"] = 16000.0 * strength ** 0.7
    return s


def pour(st, fill=1.0, vessel=1.0):
    """fill: fill speed (the cavity resonance rises fill x as fast); vessel: bigger = lower resonance, longer fill."""
    if fill == 1.0 and vessel == 1.0:
        return st
    return transform(st, _stretch(vessel ** 1.5 / fill), cents=-700 * math.log2(vessel), tr_dur=vessel)


def boiling(st, intensity, rng):
    """Lower drops real bubbles and softens the rest; higher adds a time-offset copy of the same real stems."""
    if intensity == 1.0:
        return st
    li = math.log2(intensity)
    keep = rng.random(len(st["tonal"])) < min(1.0, intensity)
    s = transform(st, gain_db=6 * li, shelf=(2000, 4 * li), depth=intensity ** 0.3, tr_keep=keep)
    if intensity > 1.0:
        k = intensity - 1.0
        s["E"] = s["E"] + k * np.roll(s["E"], int(0.37 * s["n"] / H), axis=1)
        off = int(0.37 * s["n"] / BH)
        s["tonal"] = s["tonal"] + [((s0 + off) % int(s["n"] / BH - len(fq) - 8), fq, A * np.sqrt(k), ph)
                                   for s0, fq, A, ph in s["tonal"]]
    return s


def slosh(st, amount=1.0):
    if amount == 1.0:
        return st
    la = math.log2(amount)
    return transform(st, _stretch(amount ** -0.4), cents=120 * la, gain_db=7 * la, shelf=(2500, 3 * la), depth=amount ** 0.1)


# ---------------------------------------------------------------------------------------------- render
def sources() -> list:
    return sorted(_data()["sources"])


def render_liquid(p: dict) -> np.ndarray:
    """Render a liquid_stems parameter set at 44.1 kHz (peak 0.89)."""
    src = p["source"]
    if src not in _data()["sources"]:
        raise RecipeError(f"liquid_stems: unknown source {src!r}; stored: {sources()}")
    st = _data()["sources"][src][0]
    fam = FAMILY[src]
    var = int(round(p["variation"]))
    seed = int(round(p["seed"]))
    if var:
        seed = seed * 1000 + abs(var) % 100000
    rng = np.random.default_rng(seed + 100)
    if fam == "splash":
        s = splash(st, p["size"], p["distance"], p["tail"], tape=p["size_mode"] == "tape")
    elif fam == "drip":
        s = drip_remix(st, rng, p["rate"], p["pitch"]) if (var or p["rate"] != 1.0 or p["pitch"] != 0.0) else st
    elif fam == "pour":
        s = pour(st, p["fill"], p["vessel"])
    elif fam == "flow":
        s = flow(shuffle_blocks(st, np.random.default_rng(seed)) if var else st, p["strength"])
    elif fam == "boiling":
        s = boiling(st, p["intensity"], rng)
    else:
        s = slosh(st, p["amount"])
    y = resynth(s, seed=seed)
    return y / (np.abs(y).max() + 1e-12) * 0.89


@model("liquid_stems", "sound",
       "Splashes, drips, a pour, a stream, a river, a waterfall, boiling and sloshing rebuilt from stems of "
       "public-domain / CC0 recordings (numbers only, no audio): per-band noise envelopes or STFT band envelopes, "
       "bubbles as chirps, onset envelopes and colour. Variations remix the real stems.",
       [
           Param("source", "splash_large", "", choices=tuple(FAMILY), doc="stored source"),
           Param("size", 1.0, "", 0.5, 2.0, "splash: bigger = lower, longer"),
           Param("size_mode", "shift", "", choices=("shift", "tape"),
                 doc="splash size by band shift, or by tape speed (pitch and time together)"),
           Param("distance", 1.0, "", 0.5, 4.0, "splash: further = quieter, duller, softer onset"),
           Param("tail", 1.0, "", 0.5, 2.0, "splash: >1 slower tail, <1 faster fade"),
           Param("rate", 1.0, "", 0.3, 3.0, "drip: drop rate re the recording"),
           Param("pitch", 0.0, "cents", -600, 600, "drip: smaller drop = higher, shorter ring"),
           Param("fill", 1.0, "", 0.5, 2.0, "pour: fill speed (resonance rises faster)"),
           Param("vessel", 1.0, "", 0.6, 1.6, "pour: bigger vessel = lower resonance, longer fill"),
           Param("strength", 1.0, "", 0.2, 1.5, "stream/river/waterfall: lower = quieter, darker, slower"),
           Param("intensity", 1.0, "", 0.3, 2.0, "boiling: fewer/softer .. denser bubbles"),
           Param("amount", 1.0, "", 0.5, 1.6, "slosh: gentler/slower .. harder/faster"),
           Param("seed", 0, "", 0, 2**31 - 1, "noise seed (the recipe default reproduces its reference)"),
           Param("variation", 0, "", -1e6, 1e6, "0 = the reference; others reseed and remix the real stems"),
           Param("at", 0.0, "s", 0, None),
       ],
       ["McDermott & Simoncelli 2011 (sound texture statistics)", "Minnaert 1933 (bubble resonance)",
        "McAulay & Quatieri 1986 (sinusoidal tracking)",
        "Wikimedia Commons / Internet Archive public-domain and CC0 recordings (numbers only; THIRD_PARTY.md)"],
       min_s=lambda p: 0.1)
def liquid_stems(ctx: Ctx, x, **p):
    p = dict(p)
    at = float(p.pop("at"))
    y = render_liquid(p)
    if ctx.sr != SR:
        g = math.gcd(int(ctx.sr), SR)
        y = signal.resample_poly(y, int(ctx.sr) // g, SR // g)
    out = np.zeros(ctx.n)
    i0 = int(at * ctx.sr)
    if i0 < ctx.n:
        m = min(len(y), ctx.n - i0)
        out[i0:i0 + m] = y[:m]
    return out
