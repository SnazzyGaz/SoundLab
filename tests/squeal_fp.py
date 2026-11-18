"""Numeric fingerprints of sounds (v0.29): 1/3-octave band levels (125 Hz - 16 kHz, 22 bands) in 50 ms frames,
dB re the loudest cell, in 0.5 dB steps. Small enough to keep as numbers in tests/data; no audio is stored."""
import numpy as np
from scipy import signal

SR = 48000
HOP = 2400
CEN = 1000 * 2 ** (np.arange(-9, 13) / 3)


def fingerprint(y, sr=SR):
    y = np.asarray(y, float)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        y = signal.resample_poly(y, SR // g, int(sr) // g)
    f, _, Z = signal.stft(y, SR, nperseg=4096, noverlap=4096 - HOP, boundary=None, padded=True)
    P = np.abs(Z) ** 2
    B = np.stack([P[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].sum(0) for c in CEN])
    L = 10 * np.log10(B + 1e-20)
    L = np.maximum(L - L.max(), -60.0)
    return np.round(L * 2).astype(int)          # half-dB steps


def distance(a, b, floor_db=-40.0):
    """Mean absolute level difference (dB) over the cells where either sound is within floor_db of its loudest."""
    a = np.asarray(a, float) / 2
    b = np.asarray(b, float) / 2
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n], b[:, :n]
    m = np.maximum(a, b) > floor_db
    return float(np.mean(np.abs(a[m] - b[m])))


def active_frames(fp, floor_db=-25.0):
    """Frames whose total level is within floor_db of the loudest frame."""
    a = np.asarray(fp, float) / 2
    t = 10 * np.log10(np.sum(10 ** (a / 10), axis=0))
    return t > t.max() + floor_db
