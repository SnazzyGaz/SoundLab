"""machine_stems (v0.32): engines, pass-bys, motors, a drill, an AC unit, a clock, a tram, a mower and a sewing machine
rebuilt from stems of CC0 / public-domain recordings (numbers only, no audio). See docs/machines.md.

Stems per recording:
* harmonics - the tracked fundamental (subharmonic summation + Viterbi) and each prominent harmonic's real amplitude
  and phase wobble (heterodyne demodulation), stored every 4 ms; each harmonic plays amp x cos(k*phi + phase track)
* lines - persistent whines off the harmonic series, demodulated the same way (petrol engine, drills)
* noise - the residual's own band envelopes (76 raised-cosine bands at 1/8 octave, every 1 ms, 0.5 dB steps) imposed
  on fresh noise carriers, then the residual's 1/6-octave colour
* cycle pulses - per octave band, each firing cycle's own pulse shape folded on the cycle (diesel, mower); the noise
  is reshaped pitch-synchronously so each cycle carries it
* events - ticks and stitches as their own band envelopes at 0.5 ms, rebuilt from noise
No band pitch shifts (speed = tape speed, darker = treble roll-off), no holds or frozen frames (stems are read
backwards past an end), no time squeezing, no added smoothing.

Gestures: a timeline of {action: start | hold | change | stop | pause} plays a drill, servo, sewing machine or mower
under control: the real spin-up, the real steady run looped by reading it forwards and backwards (or, for a short
steady stretch, forwards-only chunks from random points, crossfaded), speed by tape speed
(with a matching treble roll-off when slower), the real spin-down. An empty timeline plays the recording as taken.
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
from scipy.ndimage import median_filter, uniform_filter1d

from ..core import Ctx, Param, RecipeError, model
from .air_stems import GRID6, SR, _eq_to
from ..fastpath import memo_arrays
from .liquid_stems import _bank, _depth, _interp_rows, _vocoder_passes

DATA = Path(__file__).with_name("machine_stems.bin")
GESTURE_TAKE = {"power_drill": "power_drill_gesture", "lawn_mower": "lawn_mower_gesture"}
NPB = 24                       # cycle pulse phase bins
CR = 1000                      # control rate of the gesture maps (Hz)
XF = 0.06                      # crossfade (s) where a gesture jumps between two places in the stems
SYN_UP, SYN_DOWN = 0.6, 0.9    # start / stop ramps for takes without a recorded spin-up / spin-down (s)


def _q(a, lo):
    q = a.astype(np.float64)
    v = 10 ** ((q * 0.5 + lo) / 20)
    v[q == 0] = 0.0
    return v


@functools.lru_cache(maxsize=1)
def _data():
    raw = DATA.read_bytes()
    if raw[:8] != b"SLMAC001":
        raise RecipeError(f"{DATA.name}: not a soundlab machine stem file")
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    hdr = json.loads(p[4:4 + n].decode())
    blob = p[4 + n:]

    def arr(d):
        return np.frombuffer(blob, np.dtype(d["dt"]), int(np.prod(d["sh"])), d["o"]).reshape(d["sh"])

    out = {}
    for k, s in hdr["sources"].items():
        st = dict(n=s["n"], env=_q(arr(s["env"]), s["env_lo"]), env_sr=s["env_sr"], per_oct=s["per_oct"],
                  ltas6=arr(s["ltas6"]).astype(np.float64), gesture=s.get("gesture"))
        if "f0" in s:
            f0 = arr(s["f0"]).astype(np.float64)
            st.update(f0_t=s["f0_t0"] + s["f0_dt"] * np.arange(len(f0)), f0=f0, h_k=arr(s["h_k"]).astype(int),
                      h_sr=s["h_sr"], h_amp=_q(arr(s["h_amp"]), s["h_lo"]),
                      h_ph=np.unwrap(arr(s["h_ph"]).astype(np.float64) * (2 * np.pi / 256), axis=1))
        if "l_f" in s:
            lf = arr(s["l_f"]).astype(np.float64)
            st.update(l_t=s["l_t0"] + s["l_dt"] * np.arange(lf.shape[1]), l_f=lf, l_amp=_q(arr(s["l_amp"]), s["l_lo"]),
                      l_ph=np.unwrap(arr(s["l_ph"]).astype(np.float64) * (2 * np.pi / 256), axis=1))
        if "pulses" in s:
            st["pulses"] = arr(s["pulses"]).astype(np.float64)
        if "ev_env" in s:
            E = _q(arr(s["ev_env"]), s["ev_lo"])
            off, st0, ln = arr(s["ev_off"]), arr(s["ev_start"]), arr(s["ev_len"])
            st["events"] = [(int(st0[j]), int(ln[j]), E[:, off[j]:off[j + 1]]) for j in range(len(ln))]
            st["ev_rate"] = s["ev_rate"]
        out[k] = (st, {kk: v for kk, v in s.items() if kk in ("file", "t0", "t1", "settings", "gesture")})
    return dict(sources=out, files=hdr["files"])


def sources() -> list:
    return sorted(k for k in _data()["sources"] if not k.endswith("_gesture"))


# ---------------------------------------------------------------------------------------------- maps
# A map says which input time (seconds into the stems) each output control frame reads. Two readings (a, b) with a
# weight w cover crossfades; |da|, |db| are the reading speeds (1 for forwards or backwards play).

def _identity_map(n):
    m = int(np.ceil(n / SR * CR)) + 2
    u = np.arange(m) / CR
    one = np.ones(m)
    return dict(a=u, b=u, w=np.zeros(m), sa=one, sb=one, n=n)


def _mirror(t, lo, hi):
    L = hi - lo
    if L <= 0:
        return np.full_like(t, lo)
    p = np.mod(t - lo, 2 * L)
    return lo + np.where(p <= L, p, 2 * L - p)


def _read(A, sr, t):
    """Columns of A (sr columns/s) at input times t, linear; outside the take the stems are read backwards."""
    L = A.shape[-1] - 1
    x = _mirror(np.asarray(t) * sr, 0.0, float(L))
    i0 = np.minimum(np.floor(x).astype(int), L - 1)
    fr = x - i0
    return A[..., i0] * (1 - fr) + A[..., i0 + 1] * fr


def _read_prev(A, sr, t):
    L = A.shape[-1] - 1
    x = _mirror(np.asarray(t) * sr, 0.0, float(L))
    return A[..., np.minimum(np.floor(x).astype(int), L - 1)]


def _blend(fa, fb, m, power=False):
    if not m["w"].any():
        return fa
    w = m["w"]
    return np.sqrt((1 - w) * fa ** 2 + w * fb ** 2) if power else (1 - w) * fa + w * fb


def _ctrl_to_audio(v, n):
    return np.interp(np.arange(n) / SR, np.arange(v.shape[-1]) / CR, v)


# ---------------------------------------------------------------------------------------------- layers

@memo_arrays
def _vocoder(env, env_sr, n, ltas6, seed, iters=4, per_oct=8):
    rng = np.random.default_rng(seed)
    Hs, _ = _bank(n, per_oct)
    t = np.arange(n) / SR
    te = np.arange(env.shape[1]) / env_sr
    y = rng.standard_normal(n)
    E = _interp_rows(t, te, env)
    y = _vocoder_passes(y, Hs, E, n, iters)
    return _eq_to(y, ltas6) if ltas6 is not None else y


def _noise(st, m, seed):
    n = m["n"]
    if m.get("identity"):
        env, sr = st["env"], st["env_sr"]
    else:
        env = _blend(_read(st["env"], st["env_sr"], m["a"]), _read(st["env"], st["env_sr"], m["b"]), m, power=True)
        sr = CR
        if not m.get("gesture"):
            return _vocoder(env, sr, n, st["ltas6"], seed)
        # the take's spectrum is a long-term average over the whole clip (silences and spin-downs included): a
        # gesture that plays mostly the running machine keeps the running machine's noise spectrum, so the target
        # moves band by band with the envelope power the gesture actually reads
        gw = m["g"][:env.shape[1]] ** 2
        pg = (env ** 2 * gw).sum(1) / max(gw.sum(), 1e-12)
        pt = np.mean(st["env"] ** 2, 1)
        d = 10 * np.log10((pg + 1e-20) / (pt + 1e-20))
        c = _bank(16, 8)[1]
        return _vocoder(env, sr, n, st["ltas6"] + np.interp(np.log2(GRID6), c, np.clip(d, -30, 30)), seed)
    return _vocoder(env, sr, n, st["ltas6"], seed)


def _f0_ctrl(st, m):
    fa = np.interp(_mirror(m["a"], st["f0_t"][0], st["f0_t"][-1]), st["f0_t"], st["f0"])
    fb = np.interp(_mirror(m["b"], st["f0_t"][0], st["f0_t"][-1]), st["f0_t"], st["f0"])
    return _blend(fa, fb, m)


def _phase_ctrl(P, sr, m):
    """Phase tracks read through the map: their rate (not their value) is read, so a backwards read keeps each
    partial's own frequency offset and a jump needs no phase splice."""
    dP = np.diff(P, axis=-1, append=P[..., -1:]) * sr
    ra = _read_prev(dP, sr, m["a"]) * m["sa"]
    rb = _read_prev(dP, sr, m["b"]) * m["sb"] if m["w"].any() else ra
    r = _blend(ra, rb, m)
    p0 = _read(P, sr, m["a"][:1])
    return p0 + np.concatenate([np.zeros(r.shape[:-1] + (1,)), np.cumsum(r[..., :-1], axis=-1) / CR], axis=-1)


def _harm_phi(st, m):
    """Fundamental phase (cycles) at audio rate and the f0 control track."""
    n = m["n"]
    if m.get("identity"):
        fa = np.interp(np.arange(n) / SR, st["f0_t"], st["f0"])
        return np.cumsum(fa) / SR, None
    f0c = _f0_ctrl(st, m)
    return np.cumsum(_ctrl_to_audio(f0c, n)) / SR, f0c


def _harmonics(st, m, phi, harm_db=0.0, tilt=0.0):
    n = m["n"]
    y = np.zeros(n)
    k = st["h_k"]
    g = 10 ** (harm_db / 20) * (k / 8.0) ** tilt
    if m.get("identity"):
        te = np.arange(st["h_amp"].shape[1]) / st["h_sr"]
        ta = np.arange(n) / SR
        for j in range(len(k)):
            y += g[j] * np.interp(ta, te, st["h_amp"][j]) * np.cos(2 * np.pi * k[j] * phi + np.interp(ta, te, st["h_ph"][j]))
        return y
    A = _blend(_read(st["h_amp"], st["h_sr"], m["a"]), _read(st["h_amp"], st["h_sr"], m["b"]), m)
    P = _phase_ctrl(st["h_ph"], st["h_sr"], m)
    for j in range(len(k)):
        if A[j].max() <= 0:
            continue
        y += g[j] * _ctrl_to_audio(A[j], n) * np.cos(2 * np.pi * k[j] * phi + _ctrl_to_audio(P[j], n))
    return y


def _lines(st, m):
    n = m["n"]
    y = np.zeros(n)
    ta = np.arange(n) / SR
    if m.get("identity"):
        te = np.arange(st["l_amp"].shape[1]) / st["h_sr"]
        for j in range(len(st["l_f"])):
            phi = 2 * np.pi * np.cumsum(np.interp(ta, st["l_t"], st["l_f"][j])) / SR
            y += np.interp(ta, te, st["l_amp"][j]) * np.cos(phi + np.interp(ta, te, st["l_ph"][j]))
        return y
    A = _blend(_read(st["l_amp"], st["h_sr"], m["a"]), _read(st["l_amp"], st["h_sr"], m["b"]), m)
    P = _phase_ctrl(st["l_ph"], st["h_sr"], m)
    lt = st["l_t"]
    for j in range(len(st["l_f"])):
        fa = np.interp(_mirror(m["a"], lt[0], lt[-1]), lt, st["l_f"][j])
        fb = np.interp(_mirror(m["b"], lt[0], lt[-1]), lt, st["l_f"][j])
        phi = 2 * np.pi * np.cumsum(_ctrl_to_audio(_blend(fa, fb, m), n)) / SR
        y += _ctrl_to_audio(A[j], n) * np.cos(phi + _ctrl_to_audio(P[j], n))
    return y


def _octbank(n):
    Hs, _ = _bank(n, 1, 125.0, 16000.0)
    return Hs ** 2


def _profiles(v, ph, sig_cyc=0.7):
    """Per octave band, per cycle: the band envelope folded on cycle phase (NPB bins), mean 1."""
    from scipy.ndimage import gaussian_filter1d
    n = len(v)
    V = np.fft.rfft(v)
    cyc = np.floor(ph).astype(int)
    nb = ((ph % 1) * NPB).astype(int)
    nc = cyc.max() + 1
    idx = cyc * NPB + nb
    P = []
    for h in _octbank(n):
        e = np.abs(signal.hilbert(np.fft.irfft(V * h, n)))
        s = np.bincount(idx, e, nc * NPB)
        c = np.bincount(idx, None, nc * NPB)
        mm = np.where(c > 0, s / np.maximum(c, 1), np.nan).reshape(nc, NPB)
        for r in range(nc):
            row = mm[r]
            ok = ~np.isnan(row)
            mm[r] = 1.0 if not ok.any() else np.interp(np.arange(NPB), np.nonzero(ok)[0], row[ok], period=NPB)
        mm = gaussian_filter1d(mm, sig_cyc, axis=0, mode="nearest")
        P.append(mm / (mm.mean(1, keepdims=True) + 1e-12))
    return np.array(P)


def _pulse_apply(noise, ph_out, cyc_in, target):
    """Reshape the noise so each cycle carries the residual's own pulse shape per octave band (gain = target profile /
    the noise's own profile). ph_out: output cycle phase (cycles); cyc_in: which recorded cycle each sample plays."""
    n = len(noise)
    own = _profiles(noise, ph_out)
    N = np.fft.rfft(noise)
    cyc_o = np.floor(ph_out).astype(int)
    cyc_i = np.clip(cyc_in, 0, target.shape[1] - 1)
    fb = (ph_out % 1) * NPB
    i0 = np.floor(fb).astype(int) % NPB
    i1 = (i0 + 1) % NPB
    w = fb - np.floor(fb)
    out = np.zeros(n)
    for b, h in enumerate(_octbank(n)):
        G0 = target[b][cyc_i, i0] / (own[b][cyc_o, i0] + 1e-6)
        G1 = target[b][cyc_i, i1] / (own[b][cyc_o, i1] + 1e-6)
        out += np.fft.irfft(N * h, n) * ((1 - w) * np.clip(G0, 0.1, 10) + w * np.clip(G1, 0.1, 10))
    return out


@functools.lru_cache(maxsize=8)
def _event_wave(src, j, seed):
    st = _data()["sources"][src][0]
    a, n, E = st["events"][j]
    return _vocoder(E, st["ev_rate"], n, None, seed * 7919 + j, iters=8)


def _events(src, st, m, seed, positions=None, order=None):
    """Ticks / stitches, each rebuilt from its own band envelopes. In a gesture an event plays (forwards) when the
    reading passes its recorded time, in either direction."""
    n = m["n"]
    y = np.zeros(n)
    ev = st["events"]
    if positions is None:
        if m.get("identity"):
            positions = [(a, j) for j, (a, _, _) in enumerate(ev)]
        else:
            tin = np.where(m["w"] < 0.5, m["a"], m["b"]) * SR
            positions = []
            for j, (a, ln, _) in enumerate(ev):
                c = a + int(0.01 * SR)
                hit = np.nonzero(np.diff(np.sign(tin - c)) != 0)[0]
                positions += [(int(round(i * SR / CR)) - int(0.01 * SR), j) for i in hit]
    for a, j in positions:
        g = _event_wave(src, j, seed)
        s0 = a
        if s0 < 0:
            g, s0 = g[-s0:], 0
        mm = min(len(g), n - s0)
        if mm > 0:
            y[s0:s0 + mm] += g[:mm]
    return y


# ---------------------------------------------------------------------------------------------- shaping / effects

def _shelf(fc, db, kind):
    A = 10 ** (db / 40)
    w = 2 * np.pi * fc / SR
    a = np.sin(w) / 2 * np.sqrt(2)
    c = np.cos(w)
    s = 2 * np.sqrt(A) * a
    if kind == "high":
        b = [A * ((A + 1) + (A - 1) * c + s), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - s)]
        aa = [(A + 1) - (A - 1) * c + s, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - s]
    else:
        b = [A * ((A + 1) - (A - 1) * c + s), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - s)]
        aa = [(A + 1) + (A - 1) * c + s, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - s]
    return signal.tf2sos(np.array(b) / aa[0], np.array(aa) / aa[0])


def _tape(y, cents):
    return signal.resample(y, int(round(len(y) / 2 ** (cents / 1200)))) if cents else y


def _room_tail(ev, rt, wet_db, lp, pre=0.012, seed=3):
    """Each real tick excites a smooth exponentially decaying noise tail, with the tick's narrow peaks flattened so the
    room does not sustain a ring into a tone (no comb or feedback delays)."""
    rng = np.random.default_rng(seed)
    L = int(rt * 1.3 * SR)
    t = np.arange(L) / SR
    ir = rng.standard_normal(L) * np.exp(-6.91 * t / rt)
    ir[:int(pre * SR)] = 0
    ir = signal.sosfiltfilt(signal.butter(2, lp, fs=SR, output="sos"), ir)
    ir /= np.sqrt(np.sum(ir ** 2))
    EV = np.fft.rfft(ev)
    df = SR / len(ev)
    P = uniform_filter1d(np.abs(EV) ** 2, max(1, int(5 / df)))
    pk = np.maximum(P / (median_filter(P, max(3, int(200 / df)) | 1) + 1e-30), 1.0)
    return signal.fftconvolve(np.fft.irfft(EV / np.sqrt(pk), len(ev)), ir)[:len(ev)] * 10 ** (wet_db / 20)


def _resequence(st, period_s, rng):
    """Re-time the ticks: each tick moves with its own following stretch of the noise stem (from 15 ms before it);
    a stretch shorter than the new period is continued by reading it backwards, never held."""
    E = st["env"]
    T = E.shape[1]
    er = st["env_sr"]
    pos = [a + int(0.01 * SR) for a, _, _ in st["events"]]
    pr = int(0.015 * er)
    cols = [max(0, int(a / SR * er) - pr) for a in pos]
    segs = [E[:, cols[i]:(cols[i + 1] if i + 1 < len(cols) else T)] for i in range(len(cols))]
    P = int(period_s * er)
    xf = max(2, int(0.01 * er))
    out = np.zeros_like(E)
    placed = []
    t, last = cols[0], -1
    while t < T:
        j = int(rng.choice([i for i in range(len(segs)) if i != last]))
        last = j
        seg = segs[j]
        L = seg.shape[1] - 1
        idx = np.arange(P + xf)
        p = np.mod(idx, 2 * L)
        idx = np.where(p <= L, p, 2 * L - p)
        piece = seg[:, idx] ** 2
        w = np.ones(P + xf)
        if t > cols[0]:
            w[:xf] = np.linspace(0, 1, xf)
        w[-xf:] = np.linspace(1, 0, xf)
        mm = min(P + xf, T - t)
        out[:, t:t + mm] += piece[:, :mm] * w[:mm]
        a = int((t + pr) / er * SR) + (pos[j] - int(pos[j] / SR * er) * SR // er) - int(0.01 * SR)
        if a + st["events"][j][1] < st["n"]:
            placed.append((a, j))
        t += P
    out[:, :cols[0]] = E[:, :cols[0]] ** 2
    s = dict(st)
    s["env"] = np.sqrt(out)
    return s, placed


def _reshape(st, pulse_gamma=1.0, env_depth=1.0):
    s = dict(st)
    if st.get("pulses") is not None and pulse_gamma != 1.0:
        P = np.maximum(st["pulses"], 1e-3) ** pulse_gamma
        s["pulses"] = P / P.mean(2, keepdims=True)
    if env_depth != 1.0:
        E = st["env"]
        eps = 1e-4 * E.max() + 1e-12
        s["env"] = np.maximum(np.exp(_depth(np.log(E + eps), env_depth, int(0.151 * st["env_sr"]) | 1)) - eps, 0)
    return s


def _cycles_in(st, tin):
    """Recorded cycle number at input times tin (cumulative f0 of the take)."""
    t = np.arange(int(st["n"] / SR * CR) + 2) / CR
    c = np.cumsum(np.interp(t, st["f0_t"], st["f0"])) / CR
    return np.floor(np.interp(_mirror(tin, 0, t[-1]), t, c)).astype(int)


def _layers(src, st, m, seed, p, positions=None):
    """All stems through map m: returns noise and tonal (harmonics + lines + events) and events alone."""
    noise = _noise(st, m, seed)
    tonal = np.zeros(m["n"])
    ev = np.zeros(m["n"])
    if "f0" in st:
        phi, f0c = _harm_phi(st, m)
        tonal += _harmonics(st, m, phi, p["harm_db"], p["tilt"])
        if st.get("pulses") is not None:
            if m.get("identity"):
                cyc = np.floor(phi).astype(int)
            else:
                tin = np.where(m["w"] < 0.5, m["a"], m["b"])
                cyc = np.round(_ctrl_to_audio(_cycles_in(st, tin).astype(float), m["n"])).astype(int)
            noise = _pulse_apply(noise, phi, cyc, st["pulses"])
    if "l_f" in st:
        tonal += _lines(st, m)
    if "events" in st:
        ev = _events(src, st, m, seed, positions)
    return noise, tonal, ev


# ---------------------------------------------------------------------------------------------- as recorded

def _render_take(src, st, p, seed, rng):
    n = st["n"]
    cents = float(p["speed_cents"])
    positions = None
    if "events" in st and (p["tick_rate"] != 1.0 or (p["variation"] and src == "clock_tick")):
        st, positions = _resequence(st, _tick_period(st) / float(p["tick_rate"]), rng)
    st = _reshape(st, p["pulse_gamma"], p["env_depth"] * _distance(p)[3])
    split = cents < 0 and p["slow_mode"] == "split"
    if split:
        # slower without moving the preview's ~16 kHz edge into hearing: tonal stems by tape speed, the noise
        # envelopes only stretched in time (read backwards past the end), the pulses following the slowed f0
        r = 2 ** (cents / 1200)
        n2 = int(round(n / r))
        m2 = dict(_identity_map(n2), identity=False)
        m2["a"] = m2["b"] = np.arange(len(m2["a"])) / CR * r
        m2["sa"] = m2["sb"] = np.full(len(m2["a"]), r)
        noise = _noise(st, m2, seed)
        if st.get("pulses") is not None:
            f0 = np.interp(np.arange(n2) / SR, st["f0_t"] / r, st["f0"] * r)
            phi = np.cumsum(f0) / SR
            noise = _pulse_apply(noise, phi, np.floor(phi).astype(int), st["pulses"])
        _, tonal, ev = _layers(src, dict(st, pulses=None), dict(_identity_map(n), identity=True), seed, p, positions)
        tn = _tape(tonal + ev, cents)[:n2]
        y = noise[:len(tn)] * 10 ** (p["noise_db"] / 20) + tn
    else:
        noise, tonal, ev = _layers(src, st, dict(_identity_map(n), identity=True), seed, p, positions)
        y = noise * 10 ** (p["noise_db"] / 20) + tonal + ev
        if p["room_rt"] > 0 and ev.any():
            y = y + _room_tail(ev, p["room_rt"], p["room_wet_db"], p["room_lp_hz"])
    y = _colour(y, p)
    if not split:
        y = _tape(y, cents)
        if cents < 0 and p["slow_mode"] == "tape_lp":
            y = _lp(y, 16000 * 2 ** (cents / 1200) * 0.8)    # the recording's ~16 kHz edge moves down with the tape
    return _lp(y, _distance(p)[2] or p["lp_hz"])


def _lp(y, fc):
    """2nd-order treble roll-off; off at 0 or from 20 kHz up."""
    if not fc or fc >= 20000:
        return y
    return signal.sosfiltfilt(signal.butter(2, fc, fs=SR, output="sos"), y)


def _distance(p):
    """distance d (1 = as recorded): further = high shelf -5 dB and a softer low-pass per doubling (level is the
    recipe's master gain, -6 dB per doubling); nearer = low shelf up 3 dB per halving and a deeper envelope.
    Returns (high-shelf dB, low-shelf dB, low-pass Hz or 0, envelope depth factor)."""
    d = float(p["distance"])
    ld = math.log2(max(d, 1e-3))
    if d > 1:
        dep = 1 / (1 + 0.15 * ld) if p["distance_depth"] else 1.0
        return -5 * ld, 0.0, min(14000 / math.sqrt(d), 20000), dep
    if d < 1:
        return 0.0, -3 * ld, 0.0, 1 - 0.15 * ld
    return 0.0, 0.0, 0.0, 1.0


def _colour(y, p):
    hs, ls, _, _ = _distance(p)
    if p["ls_db"] + ls:
        y = signal.sosfiltfilt(_shelf(p["ls_hz"], p["ls_db"] + ls, "low"), y)
    if p["hs_db"] + hs:
        y = signal.sosfiltfilt(_shelf(p["hs_hz"], p["hs_db"] + hs, "high"), y)
    return y


def _tick_period(st):
    pos = np.array([a for a, _, _ in st["events"]]) / SR
    d = np.diff(pos)
    return float(np.median(d[d > 0.3])) if (d > 0.3).any() else 1.0


# ---------------------------------------------------------------------------------------------- gestures

def _gesture_plan(events, seg, speed0):
    """Turn a timeline into output-time control tracks: rate r (tape speed), gain g, and which piece plays (pieces:
    ('lin', t_start) for a real spin-up / spin-down, ('loop', lo, hi, start, dir) for the steady run). Piece
    boundaries are in output frames; the input reading is filled in afterwards in virtual (tape) time."""
    up, (s0, s1), down = seg.get("up"), seg["steady"], seg.get("down")
    r, g, piece = [], [], []
    pieces = []
    cur = speed0
    playing = False
    loop_id = None

    def emit(nf, rr, gg, pid):
        r.extend(rr if hasattr(rr, "__len__") else [rr] * nf)
        g.extend(gg if hasattr(gg, "__len__") else [gg] * nf)
        piece.extend([pid] * nf)

    def ramp(a, b, nf):
        return list(np.exp(np.linspace(np.log(a), np.log(b), nf))) if nf else []

    for e in events:
        act = e.get("action")
        sp = float(e.get("speed", cur / speed0)) * speed0
        if not 0.2 <= sp / speed0 <= 3.0:
            raise RecipeError(f"machine_stems gesture: speed {sp / speed0:g} outside 0.2 .. 3")
        dur = float(e.get("duration", 0.0))
        if act == "start":
            if playing:
                raise RecipeError("machine_stems gesture: start while running")
            cur = sp
            pieces.append(("loop", s0, s1, s0, 1))
            loop_id = len(pieces) - 1
            if up:
                pieces.insert(loop_id, ("lin", up[0]))
                loop_id += 1
                nf = int(round((up[1] - up[0]) / cur * CR))
                gg = np.ones(nf)
                gg[:10] = np.linspace(0, 1, 10)
                emit(nf, cur, list(gg), loop_id - 1)
            else:
                nf = int(SYN_UP * CR)
                x = np.linspace(0, 1, nf)
                emit(nf, ramp(0.3 * cur, cur, nf), list(np.sin(np.pi / 2 * x) ** 2), loop_id)
            playing = True
        elif act in ("hold", "change"):
            if not playing:
                raise RecipeError(f"machine_stems gesture: {act} before start")
            nf = int(round(dur * CR))
            if act == "change" or sp == cur:
                emit(nf, ramp(cur, sp, nf) if act == "change" else cur, 1.0, loop_id)
            else:
                k = min(nf, int(0.2 * CR))
                emit(k, ramp(cur, sp, k), 1.0, loop_id)
                emit(nf - k, sp, 1.0, loop_id)
            cur = sp
        elif act == "stop":
            if not playing:
                raise RecipeError("machine_stems gesture: stop before start")
            if down:
                pieces.append(("lin", down[0]))
                nf = int(round((down[1] - down[0]) / cur * CR))
                gg = np.ones(nf)
                gg[-30:] = np.linspace(1, 0, 30)
                emit(nf, cur, list(gg), len(pieces) - 1)
            else:
                nf = int(SYN_DOWN * CR)
                x = np.linspace(0, 1, nf)
                emit(nf, ramp(cur, 0.3 * cur, nf), list(np.cos(np.pi / 2 * x) ** 2), loop_id)
            playing = False
        elif act == "pause":
            if playing:
                raise RecipeError("machine_stems gesture: pause while running (use hold)")
            emit(int(round(dur * CR)), 1.0, 0.0, -1)
        else:
            raise RecipeError(f"machine_stems gesture: unknown action {act!r} (start, hold, change, stop, pause)")
    if playing:
        raise RecipeError("machine_stems gesture: the timeline must end with stop")
    emit(int(0.1 * CR), 1.0, 0.0, -1)
    return np.array(r), np.array(g), np.array(piece), pieces


def _forward_chunks(pid, pieces, seg, seed):
    """Steady run read forwards only (seg loop='forward'): the hold is cut into chunks of random length (seg chunk,
    s), each read forwards from a random point of the steady stretch; the chunks crossfade over seg xf. The first
    chunk carries straight on from the spin-up."""
    rng = np.random.default_rng(seed + 7)
    s0, s1 = seg["steady"]
    c0, c1 = seg.get("chunk", (0.15, 0.25))
    xf = seg.get("xf", XF)
    pid, pieces = pid.copy(), list(pieces)
    for q, k in enumerate(list(pieces)):
        if k[0] != "loop":
            continue
        idx = np.nonzero(pid == q)[0]
        for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1) if len(idx) else []:
            i, first = run[0], True
            while i <= run[-1]:
                ln = rng.uniform(c0, c1)
                st = s0 if first else rng.uniform(s0 + xf / 2, max(s0 + xf / 2, s1 - ln - xf / 2))
                j = min(i + int(round(ln * CR)), run[-1] + 1)
                pieces.append(("lin", st))
                pid[i:j] = len(pieces) - 1
                i, first = j, False
    return pid, pieces


def _gesture_map(r, g, piece, pieces, dur, seg=None, seed=0):
    """Virtual-time map: u = integral of r over output time. Each piece reads its own stretch (spin-ups and -downs
    forwards, the steady run forwards and backwards between its ends, or forwards-only chunks for loop='forward');
    where consecutive pieces do not join, the two readings crossfade over XF (or the take's xf)."""
    seg = seg or {}
    t_out = np.arange(len(r) + 1) / CR
    u_out = np.concatenate([[0.0], np.cumsum(r) / CR])
    U = u_out[-1]
    m_n = int(np.ceil(U * CR)) + 2
    u = np.arange(m_n) / CR
    fi = np.minimum(np.searchsorted(u_out, u, side="right") - 1, len(piece) - 1)
    pid = piece[fi]
    gu = g[fi]
    # silent stretches follow the previous piece (and play nothing)
    for i in range(1, len(pid)):
        if pid[i] < 0:
            pid[i] = pid[i - 1]
    if pid[0] < 0:
        pid[pid < 0] = 0 if pieces else -1
    if seg.get("loop") == "forward":
        pid, pieces = _forward_chunks(pid, pieces, seg, seed)
    starts = {}
    for i, q in enumerate(pid):
        starts.setdefault(int(q), u[i])

    def f(q, uu):
        k = pieces[q]
        if k[0] == "lin":
            return _mirror(k[1] + (uu - starts[q]), 0.0, dur)
        _, lo, hi, s, d = k
        return _mirror(s + d * (uu - starts[q]), lo, hi)

    a = f(int(pid[0]), u)
    for q in np.unique(pid):
        sel = pid == q
        a[sel] = f(int(q), u[sel])
    b = a.copy()
    w = np.zeros(m_n)
    h = int(seg.get("xf", XF) * CR / 2)
    for i in np.nonzero(np.diff(pid) != 0)[0]:
        q0, q1 = int(pid[i]), int(pid[i + 1])
        if gu[i] == 0 or gu[i + 1] == 0:
            continue                                  # from silence: nothing to crossfade
        if abs(f(q0, u[i + 1:i + 2])[0] - f(q1, u[i + 1:i + 2])[0]) < 2.0 / CR:
            continue                                  # joins (spin-up into the steady run)
        lo, hi = max(0, i + 1 - h), min(m_n, i + 1 + h)
        a[lo:hi] = f(q0, u[lo:hi])
        b[lo:hi] = f(q1, u[lo:hi])
        w[lo:hi] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(lo, hi) - lo + 0.5) / (hi - lo))
    return dict(a=a, b=b, w=w, sa=np.ones(m_n), sb=np.ones(m_n), n=int(np.ceil(U * SR)), g=gu, u_out=u_out, t_out=t_out,
                gesture=True)


def _sinc_read(y, pos, half=16, beta=8.0):
    """Band-limited read of y at fractional sample positions (Kaiser-windowed sinc, 2 x half taps)."""
    i0 = np.floor(pos).astype(int)
    fr = pos - i0
    yp = np.concatenate([np.zeros(half), y, np.zeros(half + 1)])
    out = np.zeros(len(pos))
    for j in range(-half + 1, half + 1):
        x = j - fr
        w = np.i0(beta * np.sqrt(np.clip(1 - (x / half) ** 2, 0, 1))) / np.i0(beta)
        out += yp[np.clip(i0 + j + half, 0, len(yp) - 1)] * np.sinc(x) * w
    return out


def _vary_tape(y, u_out, t_out, r):
    """Variable tape speed: output time t reads the virtual-time render at u(t). Slower stretches get a treble
    roll-off that follows the speed (the recording's ~16 kHz edge moves down with tape speed)."""
    if r.max() > 1.25:
        y = signal.sosfiltfilt(signal.butter(4, (SR / 2 - 1500) / r.max(), fs=SR, output="sos"), y)
    n_out = int(round(t_out[-1] * SR))
    out = _sinc_read(y, np.interp(np.arange(n_out) / SR, t_out, u_out) * SR)
    rmin = r.min()
    if rmin < 0.97:
        fc = min(16000 * rmin * 0.8, 16000)
        lo = signal.sosfiltfilt(signal.butter(2, fc, fs=SR, output="sos"), out)
        rk = np.interp(np.arange(n_out) / SR, t_out[:-1], r)
        wl = np.clip((1 - rk) / (1 - rmin), 0, 1)
        out = (1 - wl) * out + wl * lo
    return out


def _render_gesture(src, p, seed):
    take = GESTURE_TAKE.get(src, src)
    st, meta = _data()["sources"][take]
    seg = meta.get("gesture")
    if not seg:
        raise RecipeError(f"machine_stems: source {src!r} has no gesture take (gestures: "
                          f"{sorted(k for k, v in _data()['sources'].items() if v[1].get('gesture') and not k.endswith('_gesture') or k in GESTURE_TAKE)})")
    st = _reshape(st, p["pulse_gamma"], p["env_depth"] * _distance(p)[3])
    speed0 = 2 ** (float(p["speed_cents"]) / 1200)
    r, g, piece, pieces = _gesture_plan(list(p["gesture"]), seg, speed0)
    m = _gesture_map(r, g, piece, pieces, st["n"] / SR, seg, seed)
    noise, tonal, ev = _layers(take, st, m, seed, p)
    y = (noise * 10 ** (p["noise_db"] / 20) + tonal + ev) * _ctrl_to_audio(m["g"], m["n"])
    y = _colour(y, p)
    y = _vary_tape(y, m["u_out"], m["t_out"], r)
    return _lp(y, _distance(p)[2] or p["lp_hz"])


# ---------------------------------------------------------------------------------------------- entry

def render_machine(p: dict) -> np.ndarray:
    """Render a machine_stems parameter set at 44.1 kHz (peak 0.89)."""
    src = p["source"]
    if src not in sources():
        raise RecipeError(f"machine_stems: unknown source {src!r}; stored: {sources()}")
    var = int(round(p["variation"]))
    seed = int(round(p["seed"]))
    if var:
        seed = seed * 1000 + abs(var) % 100000
    rng = np.random.default_rng(seed + 100)
    if p["gesture"]:
        y = _render_gesture(src, p, seed)
    else:
        y = _render_take(src, _data()["sources"][src][0], p, seed, rng)
        k = min(len(y), int(0.03 * SR))          # the take ends where the clip ends: a 30 ms fade, then silence
        y[-k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    return y / (np.abs(y).max() + 1e-12) * 0.89


@model("machine_stems", "sound",
       "Engines (petrol rev, diesel idle), pass-bys (motorbike, car, tram), a servo motor, a drill, an AC unit, a "
       "clock, a lawn mower and a sewing machine rebuilt from stems of CC0 / public-domain recordings (numbers only, no "
       "audio): tracked harmonics with their real amplitude and phase wobble, persistent lines, band-envelope noise, "
       "per-cycle firing pulses and band-envelope events. Speed is tape speed; gestures play a controlled start, "
       "steady run at chosen speeds and stop from the real spin-up, steady and spin-down stems.",
       [
           Param("source", "petrol_rev", "", choices=("petrol_rev", "diesel_idle", "motorbike_passby", "car_drive_by",
                                                      "servo_motor", "power_drill", "ac_unit", "clock_tick", "tram",
                                                      "lawn_mower", "sewing_machine"), doc="stored source"),
           Param("speed_cents", 0.0, "cents", -1200, 1200, "tape speed (pitch and time together); in a gesture it "
                 "scales every speed"),
           Param("slow_mode", "tape", "", choices=("tape", "tape_lp", "split"), doc="slower: plain tape speed; "
                 "tape_lp adds a treble roll-off that follows the speed (the recording's ~16 kHz edge moves down); "
                 "split: tonal stems by tape speed, the noise envelopes stretched in time (no edge moves)"),
           Param("harm_db", 0.0, "dB", -24, 12, "harmonic layer level"),
           Param("tilt", 0.0, "", -1, 1, "harmonic tilt: amp x (k/8)^tilt (sharper / softer firing)"),
           Param("pulse_gamma", 1.0, "", 0.3, 2.0, "deepen (>1) or soften (<1) each cycle's own pulse shape"),
           Param("env_depth", 1.0, "", 0.5, 1.5, "noise envelope contrast"),
           Param("noise_db", 0.0, "dB", -24, 12, "noise layer level"),
           Param("lp_hz", 0.0, "Hz", 0, 20000, "treble roll-off (2nd order low-pass; 0 or 20000 = off)"),
           Param("distance", 1.0, "", 0.25, 4.0, "1 = as recorded; further = high shelf down, softer low-pass, "
                 "flatter envelope; nearer = more low end, deeper envelope (level: the recipe's master gain)"),
           Param("distance_depth", 1, "", 0, 1, "1 = distance also flattens the noise envelope when further"),
           Param("hs_hz", 2500.0, "Hz", 200, 16000, "high shelf corner"),
           Param("hs_db", 0.0, "dB", -24, 12, "high shelf gain (distance: negative)"),
           Param("ls_hz", 200.0, "Hz", 30, 2000, "low shelf corner"),
           Param("ls_db", 0.0, "dB", -12, 12, "low shelf gain (load, proximity)"),
           Param("tick_rate", 1.0, "", 0.25, 4.0, "clock: tick rate re the recording; the real ticks are re-timed, "
                 "each with its own following stretch (read backwards when short, never held)"),
           Param("room_rt", 0.0, "s", 0, 2, "room tail decay excited by the events (0 = dry)"),
           Param("room_wet_db", -10.0, "dB", -40, 0, "room tail level"),
           Param("room_lp_hz", 4000.0, "Hz", 500, 16000, "room tail low-pass"),
           Param("gesture", [], "", doc="gesture timeline (power_drill, servo_motor, sewing_machine, lawn_mower): list "
                 "of {action: start|hold|change|stop|pause, speed (1 = as recorded), duration (s)}; [] = the recording "
                 "as taken", is_list=True),
           Param("seed", 0, "", 0, 2**31 - 1, "noise seed (the recipe default reproduces its reference)"),
           Param("variation", 0, "", -1e6, 1e6, "0 = the reference; others reseed (the clock also re-sequences)"),
           Param("at", 0.0, "s", 0, None),
       ],
       ["Hermes 1988 (subharmonic summation)", "McAulay & Quatieri 1986 (sinusoidal modelling)",
        "McDermott & Simoncelli 2011 (sound texture statistics)",
        "Freesound CC0 and Wikimedia Commons public-domain recordings (numbers only; THIRD_PARTY.md)"],
       min_s=lambda p: 0.1)
def machine_stems(ctx: Ctx, x, **p):
    p = dict(p)
    at = float(p.pop("at"))
    y = render_machine(p)
    if ctx.sr != SR:
        g = math.gcd(int(ctx.sr), SR)
        y = signal.resample_poly(y, int(ctx.sr) // g, SR // g)
    out = np.zeros(ctx.n)
    i0 = int(at * ctx.sr)
    if i0 < ctx.n:
        m = min(len(y), ctx.n - i0)
        out[i0:i0 + m] = y[:m]
    return out
