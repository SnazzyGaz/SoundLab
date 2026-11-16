"""First analysis tool: split a sound into stems and report features.

Pipeline (all original code, following the published methods):
  1. STFT.
  2. Harmonic/percussive/residual split by median filtering with a separation margin
     (FitzGerald, DAFx 2010; Driedger, Mueller & Disch, ISMIR 2014).
  3. Sinusoidal peak tracking on the harmonic part (Serra & Smith 1990, SMS) -> partial list with
     frequency, level and decay (T60), plus a resynthesised 'sines' stem.
  4. Features: f0 (YIN), centroid, flatness, octave-band energy, onsets, attack time, decay (Schroeder),
     envelope modulation rate (e.g. engine firing frequency), per-stem energy shares.
Parameter fitting to engine recipes comes in a later build step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

from . import dsp

N_FFT = 2048
HOP = 256


def stft(x, sr, n_fft=N_FFT, hop=HOP):
    f, t, Z = signal.stft(x, sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann", boundary="even", padded=True)
    return f, t, Z


def istft(Z, sr, n, n_fft=N_FFT, hop=HOP):
    _, y = signal.istft(Z, sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann", boundary=True)
    y = y[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y


def hpr_split(x, sr, kernel_ms=200.0, kernel_hz=500.0, margin=2.0):
    """Harmonic / percussive / residual masks (Driedger et al. 2014 extension of FitzGerald 2010)."""
    f, t, Z = stft(x, sr)
    S = np.abs(Z)
    kt = max(3, int(kernel_ms / 1000 * sr / HOP) | 1)
    kf = max(3, int(kernel_hz / (sr / N_FFT)) | 1)
    H = ndimage.median_filter(S, size=(1, kt), mode="nearest")
    P = ndimage.median_filter(S, size=(kf, 1), mode="nearest")
    Mh = H > margin * P
    Mp = P >= margin * H
    Mr = ~(Mh | Mp)
    n = len(x)
    return {
        "harmonic": istft(Z * Mh, sr, n),
        "transient": istft(Z * Mp, sr, n),
        "noise": istft(Z * Mr, sr, n),
    }, (f, t, S)


def track_partials(x, sr, max_peaks=40, floor_db=-70.0, max_jump_hz=None):
    """Simple SMS-style peak picking + nearest-neighbour tracking."""
    f, t, Z = stft(x, sr)
    S = np.abs(Z) + 1e-12
    D = 20 * np.log10(S)
    ref = D.max()
    df = f[1] - f[0]
    max_jump = max_jump_hz or 3 * df
    tracks: list[dict] = []
    active: list[dict] = []
    for j in range(D.shape[1]):
        col = D[:, j]
        pk, _ = signal.find_peaks(col, height=ref + floor_db, distance=2)
        if len(pk) > max_peaks:
            pk = pk[np.argsort(col[pk])[-max_peaks:]]
        peaks = []
        for k in pk:
            if 0 < k < len(col) - 1:
                a, b, c = col[k - 1], col[k], col[k + 1]
                p = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
                peaks.append((f[k] + p * df, b - 0.25 * (a - c) * p))
        used = set()
        new_active = []
        for tr in active:
            best, bd = None, max_jump
            for i, (pf, pa) in enumerate(peaks):
                if i in used:
                    continue
                d = abs(pf - tr["f"][-1])
                if d < bd:
                    best, bd = i, d
            if best is not None:
                used.add(best)
                tr["f"].append(peaks[best][0]); tr["a"].append(peaks[best][1]); tr["t"].append(t[j])
                new_active.append(tr)
            else:
                tracks.append(tr)
        for i, (pf, pa) in enumerate(peaks):
            if i not in used:
                new_active.append({"f": [pf], "a": [pa], "t": [t[j]]})
        active = new_active
    tracks.extend(active)
    out = []
    for tr in tracks:
        if len(tr["t"]) < 4:
            continue
        tt, aa = np.array(tr["t"]), np.array(tr["a"])
        i0 = int(np.argmax(aa))
        t60 = None
        if len(tt) - i0 >= 4:
            slope = np.polyfit(tt[i0:], aa[i0:], 1)[0]  # dB per second
            if slope < -0.5:
                t60 = float(-60.0 / slope)
        out.append({
            "freq_hz": float(np.median(tr["f"])), "peak_db": float(aa.max() - ref),
            "start_s": float(tt[0]), "end_s": float(tt[-1]), "t60_s": t60,
            "_t": tt, "_f": np.array(tr["f"]), "_a": aa,
        })
    out.sort(key=lambda d: -d["peak_db"])
    return out, ref


def resynth_sines(tracks, sr, n, ref_db):
    y = np.zeros(n)
    tt = np.arange(n) / sr
    for tr in tracks:
        i0, i1 = int(tr["_t"][0] * sr), min(int(tr["_t"][-1] * sr), n - 1)
        if i1 <= i0:
            continue
        seg = tt[i0:i1]
        fr = np.interp(seg, tr["_t"], tr["_f"])
        am = 10 ** ((np.interp(seg, tr["_t"], tr["_a"]) - ref_db) / 20)
        ph = 2 * np.pi * np.cumsum(fr) / sr
        y[i0:i1] += am * np.sin(ph)
    return y


def envelope(x, sr, win_ms=10.0):
    w = max(1, int(win_ms / 1000 * sr))
    e = np.sqrt(np.convolve(x ** 2, np.ones(w) / w, mode="same"))
    return e


def f0_estimate(x, sr, fmin=40.0, fmax=1500.0, threshold=0.15):
    """YIN pitch (de Cheveigne & Kawahara 2002): cumulative-mean-normalised difference function.
    Returns (median f0 over voiced frames, number of voiced frames)."""
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = max(2, int(sr / fmax)), min(int(sr / fmin), frame // 2)
    vals = []
    gate = 10 ** (-50 / 20) * (np.max(np.abs(x)) + 1e-12)
    for i in range(0, len(x) - frame, hop):
        fr = x[i:i + frame]
        if np.sqrt(np.mean(fr ** 2)) < gate:
            continue
        w = frame // 2
        # difference function via FFT autocorrelation
        a = fr[:w]
        d = np.empty(hi + 1)
        d[0] = 0.0
        e0 = np.sum(a ** 2)
        cs = np.concatenate([[0.0], np.cumsum(fr ** 2)])
        ac = np.fft.irfft(np.fft.rfft(fr, 2 * frame) * np.conj(np.fft.rfft(a, 2 * frame)))[: hi + 1]
        for tau in range(1, hi + 1):
            d[tau] = e0 + (cs[tau + w] - cs[tau]) - 2 * ac[tau]
        cm = d[1:] * np.arange(1, hi + 1) / (np.cumsum(d[1:]) + 1e-12)
        cm = np.concatenate([[1.0], cm])
        seg = cm[lo - 1:hi + 1]
        mins = np.nonzero((seg[1:-1] < seg[:-2]) & (seg[1:-1] <= seg[2:]) & (seg[1:-1] < threshold))[0]
        if len(mins) == 0:
            continue
        k = lo + mins[0]
        if 0 < k < hi:
            a_, b_, c_ = cm[k - 1], cm[k], cm[k + 1]
            p = 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_ + 1e-12)
        else:
            p = 0.0
        vals.append(sr / (k + p))
    return (float(np.median(vals)), len(vals)) if vals else (None, 0)


def schroeder_t60(x, sr):
    e = x ** 2
    if e.sum() <= 0:
        return None
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-20)
    t = np.arange(len(x)) / sr
    i5 = np.argmax(edc_db < -5)
    i25 = np.argmax(edc_db < -25)
    if i25 <= i5:
        return None
    slope = np.polyfit(t[i5:i25], edc_db[i5:i25], 1)[0]
    return float(-60 / slope) if slope < 0 else None


def modulation_rate(x, sr, fmin=5.0, fmax=400.0):
    """Dominant periodicity of the amplitude envelope (e.g. engine firing frequency)."""
    env = envelope(x, sr, 2.0)
    env = env - env.mean()
    d = 8
    env = signal.decimate(env, d, zero_phase=True)
    esr = sr / d
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    if ac[0] <= 0:
        return None
    ac /= ac[0]
    lo, hi = int(esr / fmax), min(int(esr / fmin), len(ac) - 1)
    if hi <= lo + 1:
        return None
    pk, props = signal.find_peaks(ac[lo:hi], height=0.2)
    if len(pk) == 0:
        return None
    k = lo + pk[0]
    return float(esr / k)


def onsets(x, sr, thresh=1.5):
    f, t, Z = stft(x, sr)
    S = np.log1p(np.abs(Z) * 100)
    flux = np.maximum(np.diff(S, axis=1), 0).sum(axis=0)
    flux = np.concatenate([[0], flux])
    med = ndimage.median_filter(flux, size=15) + 1e-9
    pk, _ = signal.find_peaks(flux, height=med * thresh + flux.mean() * 0.5, distance=max(1, int(0.05 * sr / HOP)))
    return [float(t[i]) for i in pk]


def spectral_stats(S, f):
    p = S ** 2 + 1e-20
    tot = p.sum(axis=0)
    centroid = (f[:, None] * p).sum(axis=0) / tot
    cum = np.cumsum(p, axis=0) / tot
    rolloff = f[np.argmax(cum > 0.85, axis=0)]
    flat = np.exp(np.mean(np.log(p), axis=0)) / np.mean(p, axis=0)
    w = tot / tot.sum()
    return float((centroid * w).sum()), float((rolloff * w).sum()), float((flat * w).sum())


def octave_bands(x, sr):
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    out = {}
    tot = X.sum() + 1e-20
    for fc in [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]:
        m = (f >= fc / np.sqrt(2)) & (f < fc * np.sqrt(2))
        out[str(fc)] = round(float(10 * np.log10(X[m].sum() / tot + 1e-20)), 1)
    return out


def analyze(x: np.ndarray, sr: int, out_dir: str | Path | None = None, plot: bool = True) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    stems, (f, t, S) = hpr_split(x, sr)
    tracks, ref = track_partials(stems["harmonic"], sr)
    top = tracks[:20]
    sines = resynth_sines(top, sr, n, ref)
    env = envelope(x, sr)
    pk = env.max() + 1e-12
    ipk = int(np.argmax(env))
    i10 = int(np.argmax(env > 0.1 * pk))
    i90 = int(np.argmax(env > 0.9 * pk))
    energy = {k: float(np.sum(v ** 2)) for k, v in stems.items()}
    etot = sum(energy.values()) + 1e-20
    f0, voiced = f0_estimate(stems["harmonic"], sr)
    cen, roll, flat = spectral_stats(S, f)
    feats = {
        "duration_s": round(n / sr, 3),
        "sample_rate": sr,
        "peak_dbfs": round(float(dsp.db(np.max(np.abs(x)))), 2),
        "rms_dbfs": round(float(dsp.db(np.sqrt(np.mean(x ** 2)))), 2),
        "stem_energy_share": {k: round(v / etot, 3) for k, v in energy.items()},
        "f0_hz": round(f0, 1) if f0 else None,
        "voiced_frames": voiced,
        "spectral_centroid_hz": round(cen, 1),
        "rolloff85_hz": round(roll, 1),
        "spectral_flatness": round(flat, 4),
        "octave_band_db": octave_bands(x, sr),
        "attack_s": round(max(i90 - i10, 0) / sr, 4),
        "peak_time_s": round(ipk / sr, 3),
        "decay_t60_s": (lambda v: round(v, 3) if v else None)(schroeder_t60(x[ipk:], sr)),
        "envelope_modulation_hz": (lambda v: round(v, 2) if v else None)(modulation_rate(x, sr)),
        "onsets_s": [round(o, 3) for o in onsets(x, sr)][:200],
        "partials": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in tr.items() if not k.startswith("_")}
                     for tr in top],
        "envelope_db_10ms": [round(float(v), 1) for v in dsp.db(env[:: int(0.01 * sr)] / pk)],
    }
    if out_dir:
        from .audio_io import write_wav
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        for k, v in stems.items():
            write_wav(od / f"{k}.wav", v, sr)
        write_wav(od / "sines.wav", sines / (np.max(np.abs(sines)) + 1e-12) * 0.9, sr)
        (od / "features.json").write_text(json.dumps(feats, indent=1))
        if plot:
            plot_analysis(x, sr, f, t, S, env, feats, od / "analysis.png")
    return feats


def plot_analysis(x, sr, f, t, S, env, feats, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    D = 20 * np.log10(S / S.max() + 1e-9)
    ax[0].pcolormesh(t, f, D, vmin=-90, vmax=0, shading="auto", cmap="magma")
    ax[0].set_yscale("symlog", linthresh=500)
    ax[0].set_ylim(30, sr / 2)
    ax[0].set_ylabel("Hz")
    ax[0].set_title("Spectrogram (dB)")
    tt = np.arange(len(env)) / sr
    ax[1].plot(tt, dsp.db(env / env.max()))
    ax[1].set_ylim(-80, 3)
    ax[1].set_ylabel("dB")
    ax[1].set_title("Envelope (10 ms RMS)")
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fx = np.fft.rfftfreq(len(x), 1 / sr)
    sm = ndimage.uniform_filter1d(dsp.db(X / X.max()), 21)
    ax[2].semilogx(fx[1:], sm[1:])
    ax[2].set_xlim(20, sr / 2)
    ax[2].set_ylim(-100, 3)
    ax[2].set_title(f"Long-term spectrum  f0={feats['f0_hz']}  centroid={feats['spectral_centroid_hz']} Hz")
    ax[2].set_xlabel("Hz")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
