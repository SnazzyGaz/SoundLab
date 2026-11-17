"""Stereo beds (v0.22): ambience loops and one-shots rendered as two related channels.

A game positions one-shots itself (mono in, panned by the engine), but ambience beds (rain, wind, fire, water)
are played as stereo files. Two ways to get the second channel, chosen per recipe in `"stereo"`:

- `"seeds"`: a second take with another seed. Every drop, crackle and bubble lands in a different place on
  each side, as a spaced microphone pair hears a field of independent sources. Right for rain, fire, water.
- `"decorrelate"`: one take, its copy passed through a short (~20 ms) flat-magnitude random-phase filter. Gusts,
  swells and engine rhythm stay common to both sides; only the fine structure spreads. Right for wind and
  engines, where two seeds would give each ear its own weather.

`width` (0 = mono, 1 = fully uncorrelated) sets the inter-channel correlation of the band above
`mono_below_hz`: the two signals are mixed as L = a X1 + b X2, R = b X1 + a X2 with a^2 + b^2 = 1, so the
correlation is 2ab = 1 - width and both sides keep the same level. Below `mono_below_hz` (default 150 Hz) both
channels carry X1: low end in phase, as game mixes and mono fold-down need.

    "stereo": {"mode": "seeds", "width": 0.8, "mono_below_hz": 150}

Every operation is circular for loops, so a stereo loop is as seamless as the mono one.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT = {"mode": "decorrelate", "width": 0.6, "mono_below_hz": 150.0}
SECOND_SEED = 1_000_003     # offset of the second take's seed ('seeds' mode)


def config(recipe: dict | None) -> dict:
    c = dict(DEFAULT)
    c.update((recipe or {}).get("stereo") or {})
    if c["mode"] not in ("seeds", "decorrelate"):
        raise ValueError(f"stereo mode must be 'seeds' or 'decorrelate', not {c['mode']!r}")
    c["width"] = float(np.clip(float(c["width"]), 0.0, 1.0))
    return c


def _conv(x: np.ndarray, h: np.ndarray, circular: bool) -> np.ndarray:
    n = len(x)
    if circular:
        if len(h) > n:   # fold a filter longer than the loop onto it
            h = np.bincount(np.arange(len(h)) % n, weights=h, minlength=n)
        return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(h, n), n)
    m = 1 << int(np.ceil(np.log2(n + len(h))))
    return np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(h, m), m)[:n]


def decorrelator(sr: int, seed: int = 7, max_delay_s: float = 0.02) -> np.ndarray:
    """Allpass FIR with a smooth random group delay between 1 ms and `max_delay_s` (Kendall 1995 style): every
    frequency keeps its level exactly (flat magnitude, so the fold-down L+R is not coloured), but the copy's
    waveform differs from the original, which is what decorrelates the two sides. A smooth group delay keeps
    the impulse response compact (no long pre/post ringing), so transients smear by at most ~20 ms."""
    nf = 1 << int(np.ceil(np.log2(8 * max_delay_s * sr)))
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(nf, 1 / sr)
    # random group delay, smoothed over ~150 Hz so the phase has no jumps
    k = max(1, int(150.0 / (sr / nf)))
    tau = np.convolve(rng.uniform(-1, 1, len(f) + 2 * k), np.hanning(2 * k + 1), "same")[k:-k]
    tau = 0.001 + (max_delay_s - 0.001) * (tau - tau.min()) / (np.ptp(tau) + 1e-12)
    phase = -2 * math.pi * np.concatenate([[0.0], np.cumsum(tau[1:] * np.diff(f))])
    phase[-1] = np.round(phase[-1] / math.pi) * math.pi            # real-valued Nyquist bin
    H = np.exp(1j * phase)
    return np.fft.irfft(H, nf)


def _split(x: np.ndarray, sr: int, fc: float, circular: bool):
    """Exact complementary low/high split (zero phase, smooth 1-octave crossover): low + high == x."""
    n = len(x)
    m = n if circular else 1 << int(np.ceil(np.log2(max(n, 2))))
    X = np.fft.rfft(x, m)
    f = np.fft.rfftfreq(m, 1 / sr)
    u = np.clip(np.log2(np.maximum(f, 1e-9) / fc) + 0.5, 0.0, 1.0)     # 0 below fc/sqrt2, 1 above fc*sqrt2
    g = 0.5 * (1 + np.cos(math.pi * u))                                   # 1 -> 0 raised cosine
    low = np.fft.irfft(X * g, m)[:n]
    return low, x - low


def combine(x1: np.ndarray, x2: np.ndarray, sr: int, width: float, mono_below_hz: float = 150.0,
            circular: bool = False) -> np.ndarray:
    """(n, 2) from two takes of the same sound (see the module doc)."""
    rho = 1.0 - width
    theta = 0.5 * math.asin(rho)              # 2 sin(theta) cos(theta) = rho
    a, b = math.cos(theta), math.sin(theta)
    lo1, hi1 = _split(x1, sr, mono_below_hz, circular) if mono_below_hz > 0 else (0.0 * x1, x1)
    _, hi2 = _split(x2, sr, mono_below_hz, circular) if mono_below_hz > 0 else (None, x2)
    return np.stack([lo1 + a * hi1 + b * hi2, lo1 + b * hi1 + a * hi2], axis=1)


def from_mono(x: np.ndarray, sr: int, cfg: dict, circular: bool = False) -> np.ndarray:
    """'decorrelate' mode: stereo from one take."""
    x2 = _conv(x, decorrelator(sr), circular)
    x2 *= np.std(x) / (np.std(x2) + 1e-12)
    return combine(x, x2, sr, cfg["width"], float(cfg["mono_below_hz"]), circular)


def correlation(y: np.ndarray, sr: int, f_lo: float = 0.0, f_hi: float | None = None) -> float:
    """Inter-channel correlation of a stereo signal, optionally in a band."""
    L, R = y[:, 0], y[:, 1]
    if f_lo > 0 or f_hi:
        n = len(L)
        f = np.fft.rfftfreq(n, 1 / sr)
        g = (f >= f_lo) & (f <= (f_hi or sr / 2))
        L, R = np.fft.irfft(np.fft.rfft(L) * g, n), np.fft.irfft(np.fft.rfft(R) * g, n)
    return float(np.sum(L * R) / (np.sqrt(np.sum(L * L) * np.sum(R * R)) + 1e-20))
