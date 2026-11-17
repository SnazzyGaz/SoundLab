"""v0.21 creature voices: a rough laryngeal source with period doubling and chaos, a ventricular-fold layer at f0/2,
and a long, low vocal tract (lossy tube) with a throat resonance.

The glottal source is ported from CHIMERA by Jameson Hodge (https://github.com/jamesondh/chimera-vocal-synth,
MIT licence, crates/chimera/src/sources/glottis.rs, commit 24bd2e4): the raised-cosine area pulse with skew and
closure sharpness, and the per-cycle perturbations (Gaussian jitter and shimmer, alternating diplophonia, the
logistic-map "bifurcation" control that walks through period doubling and period four into chaos, and the
double-pulsing of vocal fry). CHIMERA feeds that area into a full airway network; here the radiated source is the
time derivative of the area pulse and the tract is a lossy tube (feedback comb) (see `creature_voice`). The
explicit f0/2 ventricular-fold layer, the rasp noise and the tract are SoundLab's (after Bailly, Henrich & Pelorson
2010, JASA 127: ventricular folds vibrating at half the vocal-fold rate in growls and kargyraa; Fitch 1997 and
Riede & Fitch 1999 for vocal tract length and formant spacing in animals). CHIMERA's creature presets (alligator,
seal, elephant) informed the recipes' pitch, chaos and tract settings.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..core import Ctx, Param, ctrl, model

C_AIR = 350.0  # m/s, warm humid air


def chimera_shape(ph, oq, skew, sharp):
    """Normalised glottal area pulse (0..1) at cycle phase ph (array). Port of CHIMERA `shape()` (MIT)."""
    out = np.zeros_like(ph)
    m = ph < oq
    t = ph[m] / oq
    s = float(np.clip(skew, 0.3, 0.9))
    rise = t < s
    v = np.empty_like(t)
    v[rise] = 0.5 - 0.5 * np.cos(np.pi * t[rise] / s)
    u = (t[~rise] - s) / (1 - s)
    soft = 0.5 + 0.5 * np.cos(np.pi * u)
    hard = np.cos(0.5 * np.pi * u)
    v[~rise] = soft + (hard - soft) * sharp
    out[m] = v
    return out


def glottal_cycles(f0, sr, rng, chaos, jitter, shimmer, diplophonia, x0=0.37):
    """Cycle start samples, periods (samples) and amplitudes. Per-cycle perturbations as in CHIMERA's
    ParametricGlottis (MIT): logistic map x <- r x (1 - x), r = 2.9 + 1.1 chaos, deviation (x - 0.5) scales the
    period (0.45 chaos) and amplitude (1.1 chaos); diplophonia alternates period (+-8 %) and amplitude (-35 %)."""
    n = len(f0)
    starts, periods, amps, odd = [], [], [], []
    t, k, x = 0.0, 0, x0
    while t < n:
        i = int(t)
        c = float(chaos[i])
        alt = 1.0 if k & 1 else -1.0
        cp = ca = 0.0
        if c > 0:
            r = 2.9 + 1.1 * min(1.0, c)
            x = r * x * (1 - x)
            if not 0 < x < 1:
                x = 0.37
            cp, ca = 0.45 * c * (x - 0.5), 1.1 * c * (x - 0.5)
        sc = float(np.clip(1 + jitter * rng.standard_normal() + diplophonia * 0.08 * alt + cp, 0.55, 1.8))
        a = max(0.12, 1 + shimmer * rng.standard_normal() - diplophonia * 0.35 * (alt > 0) + ca)
        P = sr / max(float(f0[i]), 5.0) * sc
        starts.append(t); periods.append(P); amps.append(a); odd.append(k & 1)
        t += P
        k += 1
    return np.array(starts), np.array(periods), np.array(amps), np.array(odd)


def tract_formants(length_cm, mouth, n, shift=1.0):
    """Resonances of a uniform tube of length_cm: closed at the larynx and open at the mouth they fall at
    (2k-1) c / 4L; with the mouth shut (sound leaves through the nose and the body) they move towards k c / 2L.
    `mouth` 0..1 blends between the two."""
    L = max(length_cm, 1.0) / 100.0
    k = np.arange(1, n + 1)
    open_ = (2 * k - 1) * C_AIR / (4 * L)
    closed = k * C_AIR / (2 * L)
    return shift * (mouth * open_ + (1 - mouth) * closed)


def _resonator(x, f, bw, sr):
    """Klatt-style two-pole resonator with unity gain at DC."""
    if f >= 0.48 * sr:
        return x
    r = np.exp(-np.pi * bw / sr)
    c = -r * r
    b = 2 * r * np.cos(2 * np.pi * f / sr)
    a0 = 1 - b - c
    return signal.lfilter([a0], [1, -b, -c], x)


@model("creature_voice", "excitation",
       "Creature and monster voices: a rough laryngeal pulse source (CHIMERA glottis port: jitter, shimmer, "
       "diplophonia, period doubling to chaos via `chaos`, vocal fry), an optional ventricular-fold layer pulsing at "
       "f0/2 (`ventricular`, the growl and kargyraa sound), rasp noise from the narrowed false folds, and a long "
       "vocal tract (`tract_cm`; `mouth` 0 = shut, sound through the nose and body) with a throat resonance. "
       "Pitch, chaos, mouth and level are automatable.",
       [
           Param("f0", 80.0, "Hz", 8, 1500, "vocal-fold rate (automatable, log)", automatable=True),
           Param("chaos", 0.3, "", 0, 1, "bifurcation: ~0.3 period doubling, ~0.55 period four, >0.62 chaos",
                 automatable=True),
           Param("diplophonia", 0.0, "", 0, 1, "alternating strong/weak cycles (another route to f0/2)"),
           Param("jitter", 0.01, "", 0, 0.2, "cycle-to-cycle period perturbation (fraction)"),
           Param("shimmer", 0.08, "", 0, 0.8, "cycle-to-cycle amplitude perturbation (fraction)"),
           Param("fry", 0.0, "", 0, 1, "vocal fry: double pulses with a short open phase"),
           Param("open_quotient", 0.5, "", 0.25, 0.95, "open part of each cycle (lower = pressed, brighter)"),
           Param("skew", 0.62, "", 0.3, 0.9, "opening share of the open phase (higher = faster closing)"),
           Param("sharpness", 0.8, "", 0, 1, "abrupt closure (more high harmonics)"),
           Param("ventricular", 0.0, "", 0, 2, "ventricular-fold layer at f0/2, relative level"),
           Param("ventricular_oq", 0.5, "", 0.2, 0.9, "open quotient of the ventricular pulse (over 2 cycles)"),
           Param("rasp", 0.0, "", 0, 2, "turbulent noise from the narrowed false folds, pulsed with the cycles"),
           Param("breath", 0.05, "", 0, 2, "aspiration noise through the glottis"),
           Param("tract_cm", 17.0, "cm", 4, 400, "vocal tract length (human 17, big dog 25, lion 35, elephant ~80)"),
           Param("mouth", 0.7, "", 0, 1, "mouth opening (0 = closed: nasal / body radiation)", automatable=True),
           Param("formant_shift", 1.0, "", 0.5, 2.0, "scale all tract resonances"),
           Param("bandwidth", 1.0, "", 0.3, 4.0, "tract losses: resonance width and high-frequency damping (more = softer, wetter walls)"),
           Param("throat_hz", 0.0, "Hz", 0, 2000, "extra throat / chest resonance (0 = off)"),
           Param("throat_q", 3.0, "", 0.5, 20, "sharpness of the throat resonance"),
           Param("muffle_hz", 0.0, "Hz", 0, 20000, "low-pass for sound radiated through skin and nose (0 = off)"),
           Param("drift_cents", 8.0, "cents", 0, 200, "slow random pitch drift"),
           Param("vibrato_hz", 0.0, "Hz", 0, 30, "vibrato / tremor rate"),
           Param("vibrato_cents", 0.0, "cents", 0, 600, "vibrato depth"),
           Param("level", 1.0, "", 0, 4, "level (automatable envelope)", automatable=True),
       ], ["CHIMERA vocal synth by Jameson Hodge (MIT): glottal source ported (see THIRD_PARTY.md)",
           "Bailly, Henrich & Pelorson 2010, JASA 127 (ventricular folds at f0/2 in growl)",
           "Fitch 1997, JASA 102; Riede & Fitch 1999 (vocal tract length and formant dispersion)",
           "Kelly & Lochbaum 1962; J. O. Smith, Physical Audio Signal Processing (lossy tube as a feedback comb)"])
def creature_voice(ctx: Ctx, x, f0, chaos, diplophonia, jitter, shimmer, fry, open_quotient, skew, sharpness,
                   ventricular, ventricular_oq, rasp, breath, tract_cm, mouth, formant_shift, bandwidth,
                   throat_hz, throat_q, muffle_hz, drift_cents, vibrato_hz, vibrato_cents, level):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    tt = ctx.t
    fz = ctrl(ctx, f0, "exp")
    cents = np.zeros(n)
    if drift_cents > 0:
        k = max(4, int(n / sr * 3) + 4)
        cents += drift_cents * np.interp(tt, np.linspace(0, tt[-1] + 1e-9, k), rng.standard_normal(k))
    if vibrato_cents > 0 and vibrato_hz > 0:
        cents += vibrato_cents * np.sin(2 * np.pi * vibrato_hz * tt)
    fz = fz * 2 ** (cents / 1200)
    if fry > 0:
        fz = fz * (1 - fry) + np.minimum(fz, 48.0) * fry
    ch = ctrl(ctx, chaos)
    st, per, amp, odd = glottal_cycles(fz, sr, rng, ch, float(jitter), float(shimmer), float(diplophonia))
    idx = np.clip(np.searchsorted(st, np.arange(n), side="right") - 1, 0, len(st) - 1)
    ph = (np.arange(n) - st[idx]) / per[idx]
    oq = float(open_quotient) * (1 - fry) + 0.28 * fry
    area = chimera_shape(ph, oq, skew, sharpness)
    if fry > 0.3:
        ph2 = ph - 0.38
        area = area + 0.55 * np.clip((fry - 0.3) / 0.5, 0, 1) * np.where(ph2 > 0, chimera_shape(np.maximum(ph2, 0), oq * 0.8, skew, sharpness), 0)
    area = area * amp[idx]
    src = np.diff(area, prepend=0.0) * sr / (np.mean(fz) + 1e-9)
    ref = np.std(src) + 1e-12
    vf = None
    if ventricular > 0 or rasp > 0:
        # ventricular folds close once every two glottal cycles: phase over a cycle pair
        pair = np.where(odd[idx] == 1, ph + 1.0, ph) / 2.0
        vsh = 1.0 + 0.15 * rng.standard_normal(len(st))
        vf = chimera_shape(pair, float(ventricular_oq), 0.55, 0.5) * vsh[idx]
    if ventricular > 0:
        vsrc = np.diff(vf, prepend=0.0) * sr / (np.mean(fz) + 1e-9)
        src = src + float(ventricular) * vsrc * ref / (np.std(vsrc) + 1e-12)
    noise_mod = area / (np.max(area) + 1e-12)
    if breath > 0:
        nz = rng.standard_normal(n)
        nz = signal.lfilter([1, -0.9], [1], nz)
        src = src + float(breath) * ref * nz * (0.3 + 0.7 * noise_mod) / (np.std(nz) + 1e-12)
    if rasp > 0:
        nz = signal.sosfilt(signal.butter(2, [300, min(6000, 0.45 * sr)], "bandpass", fs=sr, output="sos"),
                            rng.standard_normal(n))
        g = 0.2 + 0.8 * vf / (np.max(vf) + 1e-12)
        src = src + float(rasp) * ref * nz * g / (np.std(nz) + 1e-12)
    # tract: a lossy uniform tube as a feedback comb (round trip 2L/c, a one-pole wall/radiation low-pass in the
    # loop). Open mouth: the wave comes back inverted (quarter-wave resonances (2k-1)c/4L); closed mouth: it comes
    # back upright (half-wave resonances kc/2L). `mouth` blends the two tubes. Resonance spacing stays right for
    # any length, and the long-term spectrum stays flat (no build-up of overlapping peaks).
    m = float(np.clip(np.mean(ctrl(ctx, mouth)), 0, 1))
    L = float(tract_cm) / 100.0 / float(formant_shift)
    D = max(2, int(round(2 * L / C_AIR * sr)))
    T = D / sr
    g = float(np.exp(-np.pi * 40.0 * float(bandwidth) * T))  # ~40 Hz x bandwidth resonance width at low frequency
    fc = min(0.45 * sr, 5000.0 / float(bandwidth))
    a = float(np.exp(-2 * np.pi * fc / sr))
    y = np.zeros(n)
    for sign, w in ((-1.0, m), (1.0, 1.0 - m)):
        if w <= 0:
            continue
        den = np.zeros(D + 1)
        den[0], den[1] = 1.0, -a
        den[D] += -sign * g * (1 - a)
        y = y + w * signal.lfilter([1.0, -a], den, src)
    y = y * (ref / (np.std(y) + 1e-12))
    if throat_hz > 0:
        y = y + 0.8 * _resonator(src, float(throat_hz), float(throat_hz) / float(throat_q), sr) * (np.std(y) / (ref + 1e-12))
    if muffle_hz > 0 and muffle_hz < 0.45 * sr:
        y = signal.sosfilt(signal.butter(2, muffle_hz, fs=sr, output="sos"), y)
    y = signal.sosfilt(signal.butter(2, 15.0, "highpass", fs=sr, output="sos"), y)
    return y * ctrl(ctx, level)
