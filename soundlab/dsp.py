"""DSP primitives. All original code; filter formulas follow R. Bristow-Johnson's
Audio EQ Cookbook and the Chamberlin/Zavalishin state-variable filter."""
from __future__ import annotations

import math

import numpy as np
from scipy import signal


# ---------------------------------------------------------------- filters
def biquad_coeffs(kind: str, f: float, q: float, sr: int, gain_db: float = 0.0):
    f = float(np.clip(f, 1.0, sr * 0.49))
    w0 = 2 * np.pi * f / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    A = 10 ** (gain_db / 40)
    if kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":  # constant 0 dB peak gain
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf" or kind == "highshelf":
        sq = 2 * np.sqrt(A) * alpha
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * cw + sq), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sq)]
            a = [(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sq]
        else:
            b = [A * ((A + 1) + (A - 1) * cw + sq), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sq)]
            a = [(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sq]
    else:
        raise ValueError(kind)
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return b, a


def biquad(x: np.ndarray, kind: str, f: float, q: float, sr: int, gain_db: float = 0.0) -> np.ndarray:
    b, a = biquad_coeffs(kind, f, q, sr, gain_db)
    return signal.lfilter(b, a, x)


def lowpass(x, f, sr, q=0.707):
    return biquad(x, "lowpass", f, q, sr)


def highpass(x, f, sr, q=0.707):
    return biquad(x, "highpass", f, q, sr)


def bandpass(x, f, sr, q=1.0):
    return biquad(x, "bandpass", f, q, sr)


def onepole_lp(x, f, sr):
    a = np.exp(-2 * np.pi * f / sr)
    return signal.lfilter([1 - a], [1, -a], x)


def svf_loop(x: np.ndarray, fc: np.ndarray, q: float, sr: int, mode: str = "bp") -> np.ndarray:
    """Reference per-sample TPT state-variable filter (slow Python loop, kept for tests). mode: lp|bp|hp|bpn
    (bpn = unity-gain band-pass, added in 0.9 for moving cutoffs)."""
    fc = np.broadcast_to(np.asarray(fc, float), x.shape)
    g = np.tan(np.pi * np.clip(fc, 5.0, sr * 0.45) / sr)
    k = 1.0 / q
    a1 = 1.0 / (1.0 + g * (g + k))
    ic1 = ic2 = 0.0
    out = np.empty_like(x, dtype=float)
    m = {"lp": 0, "bp": 1, "hp": 2, "bpn": 1}[mode]
    scale = k if mode == "bpn" else 1.0
    xs = x.tolist()
    gs = g.tolist()
    a1s = a1.tolist()
    for i in range(len(xs)):
        gi = gs[i]
        v3 = xs[i] - ic2
        v1 = a1s[i] * (ic1 + gi * v3)
        v2 = ic2 + gi * v1
        ic1 = 2 * v1 - ic1
        ic2 = 2 * v2 - ic2
        if m == 0:
            out[i] = v2
        elif m == 1:
            out[i] = scale * v1
        else:
            out[i] = xs[i] - k * v1 - v2
    return out


def comb_waveguide(x: np.ndarray, delay_samples: float, feedback: float, sr: int,
                   damp_hz: float = 6000.0, sign: float = 1.0) -> np.ndarray:
    """Delay line with a one-pole lowpass in the loop (digital waveguide / Karplus-Strong family).

    y[n] = x[n] + sign * feedback * LP(y[n - D]). Built as a rational filter so lfilter runs it."""
    D = max(1, int(round(delay_samples)))
    p = np.exp(-2 * np.pi * damp_hz / sr)  # LP: (1-p) / (1 - p z^-1)
    # Y (1 - p z^-1) - s g (1-p) z^-D Y = X (1 - p z^-1)
    a = np.zeros(D + 1)
    a[0] = 1.0
    a[1] += -p
    a[D] += -sign * feedback * (1 - p)
    b = np.array([1.0, -p])
    return signal.lfilter(b, a, x)


def dispersive_waveguide(x: np.ndarray, delay_samples: float, feedback: float, sr: int, damp_hz: float,
                         coef: float, stages: int) -> np.ndarray:
    """Waveguide loop with an allpass chain inside it (stiff string / long wire / spring).

    y[n] = x[n] + g * (LP . AP^k)(y)[n - D], AP(z) = (c + z^-1)/(1 + c z^-1). With c < 0 low frequencies are
    delayed more, so each round trip arrives as a falling chirp. Processed in blocks of D samples so each
    first-order section runs as its own stable filter (no high-order polynomial)."""
    p = np.exp(-2 * np.pi * damp_hz / sr)
    ap_dc_delay = (1 - coef) / (1 + coef)
    D = max(1, int(round(delay_samples - stages * ap_dc_delay)))
    n = len(x)
    y = np.zeros(n)
    fy = np.zeros(n)
    zi = [np.zeros(1) for _ in range(stages + 1)]
    for s0 in range(0, n, D):
        e0 = min(n, s0 + D)
        fb = np.zeros(e0 - s0)
        lo = s0 - D
        if lo + (e0 - s0) > 0:
            a0 = max(lo, 0)
            fb[a0 - lo:] = fy[a0:e0 - D]
        y[s0:e0] = x[s0:e0] + feedback * fb
        blk = y[s0:e0]
        for k in range(stages):
            blk, zi[k] = signal.lfilter([coef, 1.0], [1.0, coef], blk, zi=zi[k])
        blk, zi[stages] = signal.lfilter([1 - p], [1.0, -p], blk, zi=zi[stages])
        fy[s0:e0] = blk
    return y


def allpass_dispersion(x: np.ndarray, coef: float, stages: int) -> np.ndarray:
    """Chain of first-order allpasses: frequency-dependent delay (stiff-wire 'pew' dispersion)."""
    y = x
    for _ in range(stages):
        y = signal.lfilter([coef, 1.0], [1.0, coef], y)
    return y


# ---------------------------------------------------------------- sources
def white(rng, n):
    return rng.standard_normal(n)


def pink(rng, n):
    """Pink noise by spectral shaping (1/f power)."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X))
    f[0] = 1
    X /= np.sqrt(f)
    y = np.fft.irfft(X, n)
    return y / (np.std(y) + 1e-12)


def brown(rng, n):
    y = np.cumsum(rng.standard_normal(n))
    y = signal.lfilter([1, -1], [1, -0.995], y)
    return y / (np.std(y) + 1e-12)


def noise(rng, n, color="white"):
    return {"white": white, "pink": pink, "brown": brown}[color](rng, n)


def poisson_times(rng, rate: np.ndarray, sr: int) -> np.ndarray:
    """Sample indices of events for a (per-sample) rate in events/second ('Dust')."""
    rate = np.asarray(rate, float)
    p = np.clip(rate / sr, 0, 1)
    return np.nonzero(rng.random(len(p)) < p)[0]


def place(out: np.ndarray, grain: np.ndarray, at: int, gain: float = 1.0) -> None:
    if at >= len(out) or at < 0:
        return
    m = min(len(grain), len(out) - at)
    out[at:at + m] += gain * grain[:m]


# ---------------------------------------------------------------- envelopes & shaping
def exp_decay(n: int, sr: int, t60: float) -> np.ndarray:
    return np.exp(-6.91 * np.arange(n) / sr / max(t60, 1e-4))


def ar_env(n: int, sr: int, attack: float, release_t60: float) -> np.ndarray:
    t = np.arange(n) / sr
    a = np.clip(t / max(attack, 1e-5), 0, 1)
    d = np.exp(-6.91 * np.maximum(t - attack, 0) / max(release_t60, 1e-4))
    return a * d


def softclip(x, drive=1.0):
    return np.tanh(drive * x) / np.tanh(drive) if drive > 0 else x


def fade(x: np.ndarray, sr: int, fin: float = 0.002, fout: float = 0.01) -> np.ndarray:
    y = x.copy()
    ni, no = int(fin * sr), int(fout * sr)
    if ni > 0:
        y[:ni] *= np.linspace(0, 1, ni)
    if no > 0:
        y[-no:] *= np.linspace(1, 0, no)
    return y


def normalize_peak(x: np.ndarray, db: float = -1.0) -> np.ndarray:
    pk = np.max(np.abs(x)) + 1e-12
    return x * (10 ** (db / 20) / pk)


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


def fft_filter(x: np.ndarray, sr: int, gain_fn) -> np.ndarray:
    """Zero-phase static filter: multiply the spectrum by gain_fn(freqs)."""
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(max(n, 2))))
    X = np.fft.rfft(x, nfft)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    return np.fft.irfft(X * gain_fn(f), nfft)[:n]


def variable_delay(x: np.ndarray, delay_samples: np.ndarray) -> np.ndarray:
    """Read x at (n - delay[n]) with linear interpolation (Doppler, moving reflections)."""
    n = np.arange(len(x))
    pos = n - delay_samples
    return np.interp(pos, n, x, left=0.0, right=0.0)


# ---------------------------------------------------------------- fast time-varying SVF (v0.7)
_SVF_MODES = ("lp", "bp", "hp", "notch", "peak", "bpn")


def _svf_out(mode, a1, g, k):
    """Output row C (on state [ic1, ic2]) and feed-through D of the TPT SVF for one mode."""
    c_v1, d_v1 = np.array([a1, -a1 * g]), a1 * g
    c_v2, d_v2 = np.array([g * a1, 1 - a1 * g * g]), a1 * g * g
    if mode == "lp":
        return c_v2, d_v2
    if mode == "bp":  # raw band-pass (peak gain 1/k), as dsp.svf always had
        return c_v1, d_v1
    if mode == "bpn":  # unity-gain band-pass
        return k * c_v1, k * d_v1
    if mode == "hp":
        return -k * c_v1 - c_v2, 1 - k * d_v1 - d_v2
    if mode == "notch":
        return -k * c_v1, 1 - k * d_v1
    return 2 * c_v2 + k * c_v1, 2 * d_v2 - 1 + k * d_v1  # peak = lp - hp


def _tf2(A, C, D, B):
    """Transfer function (b, a) of a 2-state system s' = A s + B x, y = C s + D x."""
    tr, det = A[0, 0] + A[1, 1], A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    a = np.array([1.0, -tr, det])
    cb = C @ B
    k0 = -C[0] * B[0] * A[1, 1] + C[0] * A[0, 1] * B[1] + C[1] * A[1, 0] * B[0] - C[1] * B[1] * A[0, 0]
    b = np.array([D, cb - D * tr, k0 + D * det])
    return b, a


def svf_fast(x, fc, q, sr, mode="bp", block=32):
    """TPT state-variable filter (Zavalishin 2012) with time-varying cutoff and Q, vectorised: the
    coefficients are held for `block` samples and each block runs through scipy lfilter, with the filter's
    physical state (the two integrators) carried exactly from block to block, so there are no switching
    transients. Constant cutoff/Q: one block, identical to the per-sample loop. ~50-100x faster than the loop."""
    x = np.asarray(x, float)
    n = len(x)
    fc = np.broadcast_to(np.asarray(fc, float), (n,))
    qa = np.broadcast_to(np.asarray(q, float), (n,))
    const = (np.ptp(fc) == 0) and (np.ptp(qa) == 0)
    edges = [0, n] if const else list(range(0, n, block)) + [n]
    s = np.zeros(2)
    y = np.empty(n)
    for i0, i1 in zip(edges[:-1], edges[1:]):
        if i1 <= i0:
            continue
        f = float(np.mean(fc[i0:i1])) if not const else float(fc[0])
        g = math.tan(math.pi * min(max(f, 5.0), sr * 0.45) / sr)
        k = 1.0 / min(max(float(np.mean(qa[i0:i1])), 0.05), 100.0)
        a1 = 1.0 / (1.0 + g * (g + k))
        A = np.array([[2 * a1 - 1, -2 * a1 * g], [2 * g * a1, 1 - 2 * a1 * g * g]])
        B = np.array([2 * a1 * g, 2 * a1 * g * g])
        xs = x[i0:i1]
        outs = []
        for C, D in (_svf_out(mode, a1, g, k), (A[0], B[0]), (A[1], B[1])):
            b, a = _tf2(A, C, D, B)
            y0 = C @ s
            zi = np.array([y0, C @ (A @ s) + a[1] * y0])
            outs.append(signal.lfilter(b, a, xs, zi=zi)[0])
        y[i0:i1] = outs[0]
        s = np.array([outs[1][-1], outs[2][-1]])
    return y


def svf(x: np.ndarray, fc: np.ndarray, q: float, sr: int, mode: str = "bp") -> np.ndarray:
    """TPT state-variable filter (per-sample cutoff). mode: lp|bp|hp (svf_fast also: notch|peak|bpn).
    v0.7: constant cutoff/Q runs as one lfilter (exact, ~10x faster); a moving cutoff uses the per-sample
    loop, which is exact and already ~12 ms per second of audio (measured: faster than block-wise lfilter
    unless blocks are long enough to cause small coefficient steps)."""
    fca = np.asarray(fc, float)
    if fca.ndim == 0 or np.ptp(fca) == 0:
        return svf_fast(x, fca, q, sr, mode)
    return svf_loop(x, fca, q, sr, mode)
