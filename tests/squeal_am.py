"""Modulation checks for squeals (v0.29): bubbling and purring show up as fast level modulation of band envelopes
(8-30 Hz) on the steady parts of a sound; a stray low drone as a lasting peak in the 150-330 Hz band."""
import numpy as np
from scipy import signal
from scipy.ndimage import binary_erosion, maximum_filter1d

HOP = 64


def _env(x, sr):
    f, _, Z = signal.stft(np.asarray(x, float), sr, nperseg=512, noverlap=512 - HOP)
    return f, np.abs(Z) ** 2, sr / HOP


def _steady(P, fe):
    tot = 10 * np.log10(P.sum(0) + 1e-20)
    act = (tot > tot.max() - 30) & (tot > maximum_filter1d(tot, int(0.3 * fe)) - 12)
    act = binary_erosion(act, np.ones(int(0.04 * fe) * 2 + 1))
    if act.sum() < 20:
        act = tot > tot.max() - 20
    return act


def am_depth_db(x, sr, lo=250.0, hi=10000.0, band=(8.0, 30.0)):
    """Energy-weighted 8-30 Hz level modulation (dB rms) of the 1/3-octave band envelopes between lo and hi."""
    f, P, fe = _env(x, sr)
    act = _steady(P, fe)
    b, a = signal.butter(2, [band[0] / (fe / 2), band[1] / (fe / 2)], "band")
    edges = lo * 2 ** (np.arange(0, 60) / 3)
    edges = edges[edges <= hi]
    dep, w = [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (f >= e0) & (f < e1)
        if not m.any():
            continue
        e = 10 * np.log10(P[m].sum(0) + 1e-20)
        e = np.where(act, e, np.median(e[act]))
        dep.append(np.std(signal.filtfilt(b, a, e)[act]))
        w.append(P[m][:, act].sum())
    return float(np.average(dep, weights=w))


def low_band_share_db(x, sr, lo=150.0, hi=330.0):
    """Level of the lo-hi band against the whole sound over its active parts (dB): a drone keeps it up."""
    f, P, fe = _env(x, sr)
    act = _steady(P, fe)
    m = (f >= lo) & (f < hi)
    return float(10 * np.log10(P[m][:, act].sum() / P[:, act].sum()))
