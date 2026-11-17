"""Generic blocks for analysis-and-resynthesis (v0.17).

bandnoise: noise shaped by an editable band x time level grid (a channel-vocoder-style noise source). It is the
"body" layer the resynthesis builder fills from a recording's band energies; every number in it is a level in dB
you can edit, smooth or stretch.

harmonic: a voiced source (additive harmonics on an f0 curve) through a formant envelope, with breath noise,
vibrato, slow pitch wander and detuned unison voices: cats, voices, choirs, sung or bowed tones.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..core import Ctx, Param, ctrl, model

THIRD_OCTAVES = [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                 4000, 5000, 6300, 8000, 10000, 12500, 16000]
BN_NFFT = 1024


def _smooth_noise(rng, n_frames, frame_hz, rate_hz):
    """Unit-variance random wobble at about rate_hz, sampled at frame_hz."""
    w = rng.standard_normal(n_frames + 64)
    b, a = signal.butter(2, min(0.99, max(1e-3, rate_hz / (frame_hz / 2))))
    y = signal.lfilter(b, a, w)[64:]
    return y / (np.std(y) + 1e-12)


@model("bandnoise", "synth",
       "Noise shaped by a band x time level grid: `bands` (Hz centres) and `frames` [[t, [dB per band]], ...]. "
       "Levels are interpolated between frames (in time) and between band centres (in log frequency). `tilt_db` "
       "brightens or darkens around 1 kHz, `stretch` scales the timeline, `grain` adds an independent random "
       "level wobble per band (burbling, sloshing, gusting texture) at about `grain_hz`. The resynthesis builder "
       "fills the grid from a recording's band energies; edit it like an EQ that changes over time.",
       [
           Param("bands", THIRD_OCTAVES, "Hz", doc="band centre frequencies", is_list=True),
           Param("frames", [[0.0, [0.0] * len(THIRD_OCTAVES)]], "[[s, [dB, ...]], ...]",
                 doc="level of each band at each time (held after the last frame)", is_list=True),
           Param("tilt_db", 0.0, "dB/oct", -12, 12, "spectral tilt around 1 kHz (+ = brighter)"),
           Param("stretch", 1.0, "", 0.25, 4.0, "time scale of the frame timeline (2 = twice as slow)"),
           Param("grain", 0.0, "", 0, 1, "random per-band level wobble (0 = smooth, 1 = about 12 dB swings)"),
           Param("grain_hz", 8.0, "Hz", 0.5, 80, "wobble rate"),
           Param("level", 1.0, "", 0, 4, "output level (automatable)", automatable=True),
       ], ["Dudley 1939 channel vocoder; Serra & Smith 1990 (stochastic residual as filtered noise); "
           "McDermott & Simoncelli 2011 (sound texture statistics)"])
def bandnoise(ctx: Ctx, x, bands, frames, tilt_db, stretch, grain, grain_hz, level):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    nfft, hop = BN_NFFT, BN_NFFT // 4
    w = rng.standard_normal(n + nfft)
    f, t, Z = signal.stft(w, sr, nperseg=nfft, noverlap=nfft - hop)
    bands = np.asarray(bands, float)
    fr = sorted(frames, key=lambda p: float(p[0]))
    ft = np.array([float(p[0]) for p in fr]) * float(stretch)
    G = np.array([[float(v) for v in p[1]] for p in fr])  # frames x bands
    if G.shape[1] != len(bands):
        raise ValueError(f"bandnoise: each frame needs {len(bands)} levels, got {G.shape[1]}")
    # time interpolation per band, then log-frequency interpolation per STFT frame
    Gt = np.stack([np.interp(t, ft, G[:, j]) for j in range(len(bands))], 1)  # T x bands
    if grain > 0:
        frame_hz = sr / hop
        Gt = Gt + 12.0 * grain * np.stack([_smooth_noise(rng, len(t), frame_hz, grain_hz)
                                           for _ in bands], 1)
    lf = np.log2(np.maximum(f, 1.0))
    lb = np.log2(bands)
    D = np.stack([np.interp(lf, lb, row) for row in Gt], 1)  # F x T
    D += float(tilt_db) * np.log2(np.maximum(f, 20.0) / 1000.0)[:, None]
    D[f < 20.0] = -120.0
    _, y = signal.istft(Z * 10 ** (D / 20), sr, nperseg=nfft, noverlap=nfft - hop)
    y = y[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y * ctrl(ctx, level)


def formant_gain(f, formants, shift=1.0, floor_db=-40.0):
    """Linear magnitude of a formant envelope at frequencies f: sum of resonance peaks [[Hz, bandwidth Hz, dB]]."""
    f = np.asarray(f, float)
    g = np.full(f.shape, 10 ** (floor_db / 20))
    for fc, bw, db in formants:
        fc = float(fc) * shift
        half = max(float(bw), 10.0) / 2
        g = g + 10 ** (float(db) / 20) / np.sqrt(1 + ((f - fc) / half) ** 2)
    return g


def eq_gain(f, eq):
    """Linear gain of an [[Hz, dB], ...] correction curve (flat outside its points; 1 if empty)."""
    f = np.asarray(f, float)
    if not eq:
        return np.ones(f.shape)
    pts = sorted((float(a), float(b)) for a, b in eq)
    return 10 ** (np.interp(np.log2(np.maximum(f, 1.0)), np.log2([p[0] for p in pts]), [p[1] for p in pts]) / 20)


def _grid_db(times, frames, n_bands):
    """Interpolate frames [[t, [dB per band]], ...] to `times` -> array [len(times), n_bands]."""
    fr = sorted(frames, key=lambda p: float(p[0]))
    ft = np.array([float(p[0]) for p in fr])
    G = np.array([[float(v) for v in p[1]] for p in fr])
    if G.shape[1] != n_bands:
        raise ValueError(f"each frame needs {n_bands} levels, got {G.shape[1]}")
    return np.stack([np.interp(times, ft, G[:, j]) for j in range(n_bands)], 1)


def _lookup_db(E, lb, f):
    """E [T, B] dB grid over log2 band centres lb; f [T] Hz -> dB [T] (linear in log frequency, edges held)."""
    pos = np.interp(np.log2(np.maximum(f, 1.0)), lb, np.arange(len(lb)))
    lo = np.clip(np.floor(pos).astype(int), 0, len(lb) - 2)
    w = np.clip(pos - lo, 0, 1)
    r = np.arange(len(f))
    return E[r, lo] * (1 - w) + E[r, lo + 1] * w


@model("harmonic", "excitation",
       "Voiced source for resynthesis (harmonic + noise model): harmonics of an f0 curve shaped by a spectral "
       "envelope, plus aperiodic (breath) noise. The envelope is either fixed formants [[Hz, bandwidth Hz, dB], ...] "
       "or, for moving vowels, `env_frames` [[t, [dB per band]], ...] on `bands` (a time-varying envelope, as in "
       "the WORLD vocoder). `ap_frames` sets how noisy each band is over time (aperiodicity). The noise is "
       "modulated by the glottal cycle (`pulse_noise`) so it fuses with the voice instead of hissing beside it, "
       "as turbulence follows glottal flow in articulatory models (Klatt 1990; Pink Trombone; CHIMERA). Also "
       "source slope `tilt_db`, vibrato, slow pitch wander, `voices` detuned unison copies (a choir), a fine `eq`.",
       [
           Param("f0", 220.0, "Hz", 30, 2000, "fundamental (automatable; interpolated on a log scale)",
                 automatable=True),
           Param("pitch_st", 0.0, "semitones", -24, 24, "transpose"),
           Param("formants", [[700, 130, 0], [1200, 150, -6], [2600, 200, -12]], "[[Hz, bw Hz, dB], ...]",
                 doc="fixed formant envelope (ignored when env_frames is given)", is_list=True),
           Param("bands", THIRD_OCTAVES, "Hz", doc="band centres for env_frames / ap_frames", is_list=True),
           Param("env_frames", [], "[[s, [dB, ...]], ...]", doc="time-varying spectral envelope (moving vowels)",
                 is_list=True),
           Param("ap_frames", [], "[[s, [dB, ...]], ...]", doc="aperiodicity: noise level per band over time "
                 "(0 dB = the noisiest band; empty = shaped like the envelope)", is_list=True),
           Param("formant_shift", 1.0, "", 0.5, 2.0, "scale all formant / envelope frequencies (vocal tract size)"),
           Param("tilt_db", -3.0, "dB/oct", -24, 6, "source slope (- = darker, softer voice)"),
           Param("breath", 0.1, "", 0, 1, "aperiodic (breath) noise level, relative to the harmonics"),
           Param("pulse_noise", 0.7, "", 0, 1, "how much the noise follows the glottal cycle (0 = steady hiss)"),
           Param("vibrato_hz", 5.0, "Hz", 0, 12, "vibrato rate"),
           Param("vibrato_cents", 0.0, "cents", 0, 200, "vibrato depth"),
           Param("wander_cents", 0.0, "cents", 0, 200, "slow random pitch drift"),
           Param("voices", 1, "", 1, 8, "unison voices"),
           Param("detune_cents", 10.0, "cents", 0, 60, "spread of the unison voices"),
           Param("eq", [], "[[Hz, dB], ...]", doc="fine spectral correction (interpolated in log frequency)",
                 is_list=True),
           Param("max_hz", 8000.0, "Hz", 1000, 20000, "highest harmonic"),
           Param("level", 1.0, "", 0, 4, "amplitude (automatable: the loudness envelope)", automatable=True),
       ], ["Fant 1960 source-filter theory; Serra & Smith 1990 (sinusoidal + noise modelling); Morise et al. 2016 "
           "(WORLD: f0, spectral envelope, aperiodicity); Klatt & Klatt 1990 (noise modulated by the glottal "
           "cycle); Ternstrom 1991 (choir unison spread)"])
def harmonic(ctx: Ctx, x, f0, pitch_st, formants, bands, env_frames, ap_frames, formant_shift, tilt_db, breath,
             pulse_noise, vibrato_hz, vibrato_cents, wander_cents, voices, detune_cents, eq, max_hz, level):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    base = ctrl(ctx, f0, "exp") * 2 ** (float(pitch_st) / 12)
    tt = ctx.t
    hop = 64
    idx = np.arange(0, n, hop)
    nyq = min(float(max_hz), 0.45 * sr)
    out = np.zeros(n)
    voices = max(1, int(voices))
    lb = np.log2(np.asarray(bands, float) * float(formant_shift))
    E = _grid_db(idx / sr, env_frames, len(lb)) if env_frames else None

    def env_k(fk):  # linear gain of the envelope at per-control-frame frequencies fk
        g = eq_gain(fk, eq)
        if E is not None:
            return g * 10 ** (_lookup_db(E, lb, fk) / 20)
        return g * formant_gain(fk, formants, formant_shift)

    glot = None
    for v in range(voices):
        det = 0.0 if voices == 1 else (v / (voices - 1) - 0.5) * float(detune_cents)
        cents = det + float(vibrato_cents) * np.sin(2 * np.pi * float(vibrato_hz) * tt + rng.uniform(0, 2 * np.pi))
        if wander_cents > 0:
            wn = _smooth_noise(rng, len(idx), sr / hop, 1.5)
            cents = cents + float(wander_cents) * np.interp(np.arange(n), idx, wn)
        fv = base * 2 ** (cents / 1200)
        phi = 2 * np.pi * np.cumsum(fv) / sr + rng.uniform(0, 2 * np.pi)
        if glot is None:
            glot = ((1 + np.cos(phi)) / 2) ** 3
            glot /= glot.mean() + 1e-12
        fmin = max(20.0, float(np.min(fv)))
        kmax = int(nyq // fmin)
        fc = fv[idx]
        for k in range(1, kmax + 1):
            fk = k * fc
            a = env_k(fk) * k ** (float(tilt_db) / 6.02) * (fk < nyq)
            if not np.any(a):
                continue
            out += np.interp(np.arange(n), idx, a) * np.sin(k * phi)
    out /= voices ** 0.5
    if breath > 0:
        nfft, nh = BN_NFFT, BN_NFFT // 4
        f, t, Z = signal.stft(rng.standard_normal(n + nfft), sr, nperseg=nfft, noverlap=nfft - nh)
        if E is not None:
            Et = _grid_db(t, env_frames, len(lb))
            D = np.stack([np.interp(np.log2(np.maximum(f, 1.0)), lb, row) for row in Et], 1)
        else:
            D = 20 * np.log10(formant_gain(f, formants, formant_shift))[:, None] + np.zeros((1, len(t)))
        if ap_frames:
            At = _grid_db(t, ap_frames, len(lb))
            D = D + np.stack([np.interp(np.log2(np.maximum(f, 1.0)), lb, row) for row in At], 1)
        D += 20 * np.log10(eq_gain(f, eq))[:, None]
        _, nz = signal.istft(Z * 10 ** (D / 20), sr, nperseg=nfft, noverlap=nfft - nh)
        nz = np.pad(nz[:n], (0, max(0, n - len(nz[:n]))))
        pn = float(pulse_noise)
        nz = nz * ((1 - pn) + pn * glot)
        nz *= (np.std(out) + 1e-9) / (np.std(nz) + 1e-12)
        out = out + float(breath) * 2.0 * nz
    return out * ctrl(ctx, level)


@model("bubble_cloud", "excitation",
       "Many bubbles at once, fast: the sound of flowing, splashing and pouring water. Each bubble rings at its "
       "Minnaert frequency f = 3.26/r with van den Doel's damping and pitch rise; radii follow a power law between "
       "r_min and r_max; amplitude grows as r^amp_exponent. Bubbles are grouped into radius bins and each bin is one "
       "Poisson impulse train convolved with that radius's ringing, so thousands of bubbles per second render "
       "quickly. `rate` and `level` can follow an envelope (a splash's bubble burst).",
       [
           Param("rate", 2000.0, "bubbles/s", 0, 50000, "bubble formation rate", automatable=True),
           Param("r_min_mm", 0.5, "mm", 0.1, 20, "smallest radius (6.5 kHz at 0.5 mm)"),
           Param("r_max_mm", 6.0, "mm", 0.2, 60, "largest radius (540 Hz at 6 mm)"),
           Param("size_exponent", 2.5, "", 0.5, 5, "power-law exponent of the radius distribution (more small ones)"),
           Param("amp_exponent", 0.5, "", 0, 2, "bubble amplitude ~ r^amp_exponent"),
           Param("xi", 0.1, "", 0, 20, "pitch-rise constant (bubbles rising, shrinking pressure)"),
           Param("damping", 1.0, "", 0.3, 4, "scale van den Doel's damping (more = shorter, drier blips)"),
           Param("bins", 40, "", 8, 120, "radius bins (quality vs speed)"),
           Param("level", 1.0, "", 0, 4, "level", automatable=True),
       ], ["Minnaert 1933; van den Doel 2005, ACM TAP 2(4) (liquid sounds from bubble populations)",
           "Moss et al. 2010 (sounding liquids: bubble-based fluid sound)", "Zheng & James 2009 (harmonic fluids)"])
def bubble_cloud(ctx: Ctx, x, rate, r_min_mm, r_max_mm, size_exponent, amp_exponent, xi, damping, bins, level):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    rmin, rmax = sorted((r_min_mm * 1e-3, max(r_max_mm, r_min_mm * 1.01) * 1e-3))
    rmin = max(rmin, 3.26 / (0.45 * sr))
    if rmax <= rmin:
        return np.zeros(n)
    edges = np.geomspace(rmin, rmax, int(bins) + 1)
    a = 1 - size_exponent
    cdf = (edges ** a - rmin ** a) / (rmax ** a - rmin ** a) if abs(a) > 1e-9 else np.log(edges / rmin) / np.log(rmax / rmin)
    share = np.diff(cdf)
    R = ctrl(ctx, rate)
    lev = ctrl(ctx, level)
    out = np.zeros(n)
    for b in range(int(bins)):
        if share[b] <= 0:
            continue
        r = float(np.sqrt(edges[b] * edges[b + 1]))
        p = np.clip(R * share[b] / sr, 0, 1)
        hits = rng.random(n) < p
        if not hits.any():
            continue
        imp = hits * lev * rng.uniform(0.5, 1.0, n) * (r / 1e-3) ** amp_exponent
        f0 = 3.26 / r
        d = (0.13 / r + 0.0072 * r ** -1.5) * damping
        L = min(int(sr * 7.0 / d), n, sr)
        tg = np.arange(L) / sr
        grain = np.sin(2 * np.pi * np.cumsum(f0 * (1 + xi * d * tg)) / sr) * np.exp(-d * tg)
        out += signal.fftconvolve(imp, grain)[:n]
    return out * 0.3


@model("eq", "fx",
       "Graphic EQ curve [[Hz, dB], ...], interpolated in log frequency (zero phase). The resynthesis builder "
       "puts the remaining long-term spectral difference here.",
       [Param("curve", [], "[[Hz, dB], ...]", doc="gain points", is_list=True),
        Param("gain_db", 0.0, "dB", -40, 40, "overall gain")], ["Graphic equaliser (standard)"])
def eq_fx(ctx: Ctx, x, curve, gain_db):
    from .. import dsp
    if not curve:
        return x * 10 ** (gain_db / 20)
    return dsp.fft_filter(x, ctx.sr, lambda f: eq_gain(f, curve)) * 10 ** (gain_db / 20)
