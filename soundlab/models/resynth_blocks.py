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


@model("harmonic", "excitation",
       "Voiced source for resynthesis: harmonics of an f0 curve shaped by a fixed formant envelope (so the vowel "
       "stays put while the pitch moves, as in a voice), a source slope `tilt_db`, breath noise through the same "
       "formants, vibrato, slow pitch wander, and `voices` detuned unison copies (a choir). Formants are "
       "[[Hz, bandwidth Hz, dB], ...].",
       [
           Param("f0", 220.0, "Hz", 30, 2000, "fundamental (automatable; interpolated on a log scale)",
                 automatable=True),
           Param("pitch_st", 0.0, "semitones", -24, 24, "transpose"),
           Param("formants", [[700, 130, 0], [1200, 150, -6], [2600, 200, -12]], "[[Hz, bw Hz, dB], ...]",
                 doc="formant envelope", is_list=True),
           Param("formant_shift", 1.0, "", 0.5, 2.0, "scale all formant frequencies (size of the vocal tract)"),
           Param("tilt_db", -3.0, "dB/oct", -24, 6, "source slope (- = darker, softer voice)"),
           Param("breath", 0.1, "", 0, 1, "breath noise through the formants"),
           Param("vibrato_hz", 5.0, "Hz", 0, 12, "vibrato rate"),
           Param("vibrato_cents", 0.0, "cents", 0, 200, "vibrato depth"),
           Param("wander_cents", 0.0, "cents", 0, 200, "slow random pitch drift"),
           Param("voices", 1, "", 1, 8, "unison voices"),
           Param("detune_cents", 10.0, "cents", 0, 60, "spread of the unison voices"),
           Param("eq", [], "[[Hz, dB], ...]", doc="fine spectral correction (interpolated in log frequency; "
                 "the resynthesis builder fills it from the remaining 1/3-octave difference)", is_list=True),
           Param("max_hz", 8000.0, "Hz", 1000, 20000, "highest harmonic"),
           Param("level", 1.0, "", 0, 4, "amplitude (automatable: the loudness envelope)", automatable=True),
       ], ["Fant 1960 source-filter theory; Serra & Smith 1990 (sinusoidal modelling); "
           "Ternstrom 1991 (choir unison spread)"])
def harmonic(ctx: Ctx, x, f0, pitch_st, formants, formant_shift, tilt_db, breath, vibrato_hz, vibrato_cents,
             wander_cents, voices, detune_cents, eq, max_hz, level):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    base = ctrl(ctx, f0, "exp") * 2 ** (float(pitch_st) / 12)
    tt = ctx.t
    hop = 64
    idx = np.arange(0, n, hop)
    nyq = min(float(max_hz), 0.45 * sr)
    out = np.zeros(n)
    voices = max(1, int(voices))
    env_fn = lambda ff: formant_gain(ff, formants, formant_shift) * eq_gain(ff, eq)  # noqa: E731
    for v in range(voices):
        det = 0.0 if voices == 1 else (v / (voices - 1) - 0.5) * float(detune_cents)
        cents = det + float(vibrato_cents) * np.sin(2 * np.pi * float(vibrato_hz) * tt + rng.uniform(0, 2 * np.pi))
        if wander_cents > 0:
            wn = _smooth_noise(rng, len(idx), sr / hop, 1.5)
            cents = cents + float(wander_cents) * np.interp(np.arange(n), idx, wn)
        fv = base * 2 ** (cents / 1200)
        phi = 2 * np.pi * np.cumsum(fv) / sr + rng.uniform(0, 2 * np.pi)
        fmin = max(20.0, float(np.min(fv)))
        kmax = int(nyq // fmin)
        fc = fv[idx]
        for k in range(1, kmax + 1):
            fk = k * fc
            a = env_fn(fk) * k ** (float(tilt_db) / 6.02) * (fk < nyq)
            if not np.any(a):
                continue
            out += np.interp(np.arange(n), idx, a) * np.sin(k * phi)
    out /= voices ** 0.5
    if breath > 0:
        nz = dsp_noise = rng.standard_normal(n)
        nz = np.fft.irfft(np.fft.rfft(dsp_noise) * env_fn(np.fft.rfftfreq(n, 1 / sr)),
                          n)
        nz *= (np.std(out) + 1e-9) / (np.std(nz) + 1e-12)
        out = out + float(breath) * 2.0 * nz
    return out * ctrl(ctx, level)
