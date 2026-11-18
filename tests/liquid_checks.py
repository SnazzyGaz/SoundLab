"""Listening proxies for liquid_stems renders (v0.31): persistent narrow tones, frozen stretches, shimmer
(isolated time-frequency peaks), high-frequency hiss and buzz. Each compares a render with the default render."""
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter, median_filter


def persistent_lines(x, sr, prom=10.0, frac=0.15, nper=4096):
    """Narrow spectral lines (10 dB over a 31-bin median) present in at least `frac` of the active frames."""
    f, _, Z = signal.stft(x, sr, nperseg=nper, noverlap=nper * 3 // 4)
    L = 20 * np.log10(np.abs(Z) + 1e-12)
    act = L.max(0) > L.max() - 30
    pk = (L - median_filter(L, size=(31, 1)) > prom) & (L > L.max(0, keepdims=True) - 45)
    fr = pk[:, act].mean(1)
    k, _ = signal.find_peaks(fr, height=frac)
    return [float(f[i]) for i in k if f[i] > 100]


def new_lines(y, dflt, sr, cents=0.0):
    """Lines in y that the default (moved by `cents`) does not have."""
    k = 2 ** (cents / 1200)
    base = [u * k for u in persistent_lines(dflt, sr, frac=0.1)]
    return [round(u) for u in persistent_lines(y, sr) if all(abs(u - b) > 0.03 * b + 10 for b in base)]


def frozen(x, sr, win=1.0):
    """Worst 1 s window of frame-to-frame band change (1/3-octave bands, 46 ms frames) vs the clip median, dB.
    A held envelope reads strongly negative."""
    f, _, Z = signal.stft(x, sr, nperseg=2048, noverlap=0)
    P = np.abs(Z) ** 2
    ed = 250 * 2 ** (np.arange(0, 19) / 3)
    B = np.array([10 * np.log10(P[(f >= a) & (f < b)].sum(0) + 1e-20) for a, b in zip(ed[:-1], ed[1:]) if b < sr / 2])
    k = max(4, int(win * sr / 2048))
    v = [np.median(np.diff(B[:, i:i + k], axis=1).std(1)) for i in range(0, B.shape[1] - k + 1, max(1, k // 2))]
    return float(20 * np.log10(min(v) / np.median(v) + 1e-9))


def shimmer(x, sr):
    """Isolated time-frequency peaks per second (musical noise): 3x3 maxima > 12 dB over the 7x7 median."""
    f, _, Z = signal.stft(x, sr, nperseg=512, noverlap=384)
    L = 20 * np.log10(np.abs(Z) + 1e-12)
    m = (L == maximum_filter(L, 3)) & (L - median_filter(L, 7) > 12) & (L > L.max() - 50) & (f[:, None] > 200)
    return float(m.sum() / (len(x) / sr))


def hf_levels(x, sr):
    f, P = signal.welch(x, sr, nperseg=4096)
    tot = P[(f > 200) & (f < 4000)].sum()
    return np.array([10 * np.log10(P[(f >= c / 1.12) & (f < c * 1.12)].sum() / tot + 1e-20)
                     for c in (4000, 5040, 6350, 8000, 10080, 12700, 16000)])


def hf_buzz(x, sr):
    """30-300 Hz envelope modulation (dB re the envelope mean) in 1/3-octave bands from 4 to 16 kHz."""
    out = []
    for c in (4000, 6350, 10080, 16000):
        sos = signal.butter(4, [c / 1.12, min(c * 1.12, sr / 2 - 500)], "band", fs=sr, output="sos")
        e = np.abs(signal.hilbert(signal.sosfiltfilt(sos, x)))
        fr, P = signal.welch(e - e.mean(), sr, nperseg=8192)
        m = (fr > 30) & (fr < 300)
        out.append(10 * np.log10(P[m].sum() * (fr[1] - fr[0]) / e.mean() ** 2 + 1e-12))
    return np.array(out)


def artefacts(y, dflt, sr):
    """Fails on new shimmer, hiss (+1.5 dB in any 4-16 kHz band re the 0.2-4 kHz level) or buzz (+1.5 dB)."""
    my, md = shimmer(y, sr), shimmer(dflt, sr)
    dl = float((hf_levels(y, sr) - hf_levels(dflt, sr)).max())
    dz = float((hf_buzz(y, sr) - hf_buzz(dflt, sr)).max())
    fails = [k for k, bad in (("shimmer", my > max(1.25 * md, md + 10)), ("hiss", dl > 1.5), ("buzz", dz > 1.5)) if bad]
    return dict(shimmer=(round(my), round(md)), hiss_db=round(dl, 1), buzz_db=round(dz, 1)), fails
