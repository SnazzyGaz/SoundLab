"""Generic synthesis toolkit (v0.6, roadmap step 1).

Free-form blocks for sounds with no physical source (UI blips, magic, sci-fi, retro, power-ups) and for
layering on top of the physical models. They use the same JSON recipe format: a `synth` block starts a chain,
`fx` blocks process it, and any automatable control takes a breakpoint list [[t, v], ...] (times relative to
the layer start), which is how pitch and filter sweeps are written.

Design notes
- Oscillators are band-limited with PolyBLEP corrections (Valimaki & Huovilainen 2007, "Antialiasing
  oscillators in subtractive synthesis", IEEE Signal Processing Magazine), rendered at 2x and decimated with a
  steep FIR; the triangle is a leaky-integrated PolyBLEP square. Plain digital saws/squares alias audibly
  above ~1 kHz.
- Waveshapers run at 4x oversampling (except the deliberately aliasing bit/sample-rate crusher).
- The filter is a TPT state-variable filter (Zavalishin, "The Art of VA Filter Design", 2012) with per-sample
  cutoff and resonance, so sweeps do not zipper.
- The reverb is a mono Freeverb (Jezar at Dreampoint, public domain): 8 damped combs + 4 allpasses.
All written from those descriptions; no code copied.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .. import dsp
from ..core import Ctx, Param, ctrl, is_automation, model

WAVES = ("sine", "saw", "square", "pulse", "triangle")


# ----------------------------------------------------------------------------- helpers
def _polyblep(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Polynomial band-limited step residual for phase t in [0,1) and increment dt."""
    out = np.zeros_like(t)
    a = t < dt
    x = t[a] / dt[a]
    out[a] = x + x - x * x - 1.0
    b = t > 1.0 - dt
    x = (t[b] - 1.0) / dt[b]
    out[b] = x * x + x + x + 1.0
    return out


def _phase(freq: np.ndarray, sr: int, phase0: float = 0.0):
    dt = np.clip(freq / sr, 1e-7, 0.49)
    ph = (phase0 + np.concatenate([[0.0], np.cumsum(dt)[:-1]])) % 1.0
    return ph, dt


def osc_wave(wave: str, freq: np.ndarray, sr: int, pw: np.ndarray | float = 0.5, phase0: float = 0.0) -> np.ndarray:
    """One band-limited oscillator voice. freq and pw are per-sample arrays (or scalars)."""
    t, dt = _phase(np.atleast_1d(np.asarray(freq, float)), sr, phase0)
    if wave == "sine":
        return np.sin(2 * np.pi * t)
    if wave == "saw":
        return 2.0 * t - 1.0 - _polyblep(t, dt)
    pw = np.clip(np.broadcast_to(np.asarray(pw, float), t.shape), 0.02, 0.98) if wave == "pulse" else np.full(t.shape, 0.5)
    sq = np.where(t < pw, 1.0, -1.0) + _polyblep(t, dt) - _polyblep((t - pw) % 1.0, dt)
    if wave in ("square", "pulse"):
        return sq - (2 * pw - 1)  # remove the DC of an asymmetric pulse
    # triangle: integrate the band-limited square (leaky so it never drifts), scaled to +-1
    leak = 0.9995
    p0 = phase0 % 1.0
    v0 = -1.0 + 4.0 * p0 if p0 < 0.5 else 3.0 - 4.0 * p0  # start the integrator on the triangle (no DC step)
    return signal.lfilter([1.0], [1.0, -leak], 4.0 * dt * sq, zi=[leak * v0])[0]


_DECIM = signal.firwin(129, 0.46, window=("kaiser", 10.0))  # steep 2x decimator: passband to ~0.9 x Nyquist


def decimate2(y: np.ndarray, n: int) -> np.ndarray:
    """2x -> 1x with a steep linear-phase FIR (scipy's default resample filter leaves ~-33 dB aliases)."""
    return signal.resample_poly(y, 1, 2, window=_DECIM)[:n]


def osc_bl(wave: str, freq: np.ndarray, sr: int, pw=0.5, phase0: float = 0.0) -> np.ndarray:
    """PolyBLEP oscillator rendered at 2x and decimated: residual aliasing about -43 dB for a 1.2 kHz saw
    (plain PolyBLEP at 1x: -26 dB; naive: -11 dB)."""
    if wave == "sine":
        return osc_wave(wave, freq, sr, pw, phase0)
    n = len(freq)
    pw2 = np.repeat(np.broadcast_to(np.asarray(pw, float), (n,)), 2)
    return decimate2(osc_wave(wave, np.repeat(freq, 2), 2 * sr, pw2, phase0), n)


# ----------------------------------------------------------------------------- sources
@model("osc", "synth",
       "Band-limited oscillator (PolyBLEP): sine, saw, square, pulse (variable width) or triangle, with optional "
       "unison voices (detuned, for a fat 'supersaw') and vibrato. Automate `freq` for pitch sweeps, glides and "
       "arpeggios (it is interpolated in log-frequency, so sweeps sound even).",
       [
           Param("wave", "sine", "", choices=WAVES),
           Param("freq", 440.0, "Hz", 10, 20000, "pitch (automatable: [[t, Hz], ...] for sweeps/steps)", automatable=True),
           Param("pulse_width", 0.5, "", 0.02, 0.98, "duty cycle of `pulse` (automatable: PWM)", automatable=True),
           Param("voices", 1, "count", 1, 9, "unison voices"),
           Param("detune_cents", 12.0, "cents", 0, 100, "spread of the unison voices"),
           Param("vibrato_hz", 0.0, "Hz", 0, 30, "vibrato rate"),
           Param("vibrato_cents", 0.0, "cents", 0, 1200, "vibrato depth (automatable)", automatable=True),
           Param("level", 1.0, "", 0, 4, "output level (automatable)", automatable=True),
       ], ["Valimaki & Huovilainen 2007, Antialiasing oscillators in subtractive synthesis, IEEE SPM 24(2)"])
def osc(ctx: Ctx, x, wave, freq, pulse_width, voices, detune_cents, vibrato_hz, vibrato_cents, level):
    sr, n = ctx.sr, ctx.n
    f = ctrl(ctx, freq, "exp")
    if vibrato_hz > 0:
        vc = ctrl(ctx, vibrato_cents)
        f = f * 2 ** (vc / 1200 * np.sin(2 * np.pi * vibrato_hz * ctx.t))
    pw = ctrl(ctx, pulse_width)
    v = int(max(1, voices))
    out = np.zeros(n)
    for k in range(v):
        c = 0.0 if v == 1 else detune_cents * (2 * k / (v - 1) - 1)
        ph0 = 0.0 if v == 1 else float(ctx.rng.random())
        out += osc_bl(wave, f * 2 ** (c / 1200), sr, pw, ph0)
    return out / math.sqrt(v) * ctrl(ctx, level)


@model("fm", "synth",
       "Two-operator FM (phase modulation, Chowning 1973): a sine carrier whose phase is pushed by a sine "
       "modulator at `ratio` x the carrier. `index` sets brightness (automate it for bell/zap decays); "
       "non-integer ratios give clangorous, metallic tones. Optional modulator self-feedback for noisy grit. "
       "Rendered at 2x and decimated to keep high indices from folding back.",
       [
           Param("freq", 440.0, "Hz", 10, 12000, "carrier pitch (automatable)", automatable=True),
           Param("ratio", 2.0, "", 0.01, 32, "modulator / carrier frequency ratio"),
           Param("index", 2.0, "rad", 0, 50, "modulation index (automatable)", automatable=True),
           Param("feedback", 0.0, "", 0, 1.5, "modulator self-feedback (grit)"),
           Param("level", 1.0, "", 0, 4, "output level (automatable)", automatable=True),
       ], ["Chowning 1973, The synthesis of complex audio spectra by means of frequency modulation, JAES 21(7)"])
def fm(ctx: Ctx, x, freq, ratio, index, feedback, level):
    sr2 = ctx.sr * 2
    sub = Ctx(sr2, ctx.n * 2, ctx.rng)
    fc = ctrl(sub, freq, "exp")
    I = ctrl(sub, index)
    pc = 2 * np.pi * np.cumsum(fc) / sr2
    pm = 2 * np.pi * np.cumsum(fc * ratio) / sr2
    if feedback > 0:
        # one-sample feedback needs a loop; keep it cheap by iterating on whole arrays (3 fixed-point passes)
        mod = np.sin(pm)
        for _ in range(3):
            mod = np.sin(pm + feedback * np.concatenate([[0.0], mod[:-1]]))
    else:
        mod = np.sin(pm)
    y = np.sin(pc + I * mod)
    return decimate2(y, ctx.n) * ctrl(ctx, level)


NOISE_COLOURS = ("white", "pink", "brown", "blue", "violet")


@model("noisegen", "synth",
       "Noise source in five colours (white flat, pink -3 dB/oct, brown -6, blue +3, violet +6) with an optional "
       "sample-and-hold rate for retro, pitched 'chip' noise (like an NES noise channel).",
       [
           Param("color", "white", "", choices=NOISE_COLOURS),
           Param("hold_hz", 0.0, "Hz", 0, 48000, "sample-and-hold rate; 0 = off (automatable)", automatable=True),
           Param("level", 1.0, "", 0, 4, "output level (automatable)", automatable=True),
       ], ["Wikipedia: Colors of noise"])
def noisegen(ctx: Ctx, x, color, hold_hz, level):
    n, rng = ctx.n, ctx.rng
    if color in ("white", "pink", "brown"):
        y = dsp.noise(rng, n, color)
    elif color == "blue":
        y = np.diff(dsp.pink(rng, n + 1))
    else:
        y = np.diff(rng.standard_normal(n + 1))
    y = y / (np.std(y) + 1e-12)
    h = ctrl(ctx, hold_hz)
    if np.any(h > 0):
        # sample-and-hold: advance a counter by hold/sr per sample and keep the value until it wraps
        idx = np.floor(np.cumsum(np.clip(h, 0, ctx.sr) / ctx.sr)).astype(int)
        keep = np.r_[0, np.nonzero(np.diff(idx))[0] + 1]
        held = y[keep][np.searchsorted(keep, np.arange(n), side="right") - 1]
        y = np.where(h > 0, held, y)
    return y * ctrl(ctx, level)


@model("grains", "synth",
       "Random grains (granular cloud, 'Dust'-style timing): short windowed sine, noise or saw grains at a "
       "random rate around `rate`, pitched randomly within `spread_oct` around `freq`. Sparkles, shimmers, "
       "bubbling magic, sci-fi computer chatter, crackles.",
       [
           Param("rate", 30.0, "grains/s", 0, 5000, "average grain rate (automatable)", automatable=True),
           Param("grain_ms", 30.0, "ms", 0.5, 500, "grain length"),
           Param("freq", 2000.0, "Hz", 20, 18000, "centre pitch (automatable)", automatable=True),
           Param("spread_oct", 1.0, "oct", 0, 6, "random pitch spread (total width, octaves)"),
           Param("source", "sine", "", choices=("sine", "noise", "saw")),
           Param("quantize", 0, "steps/oct", 0, 24, "snap grain pitches to a scale grid (12 = semitones, 0 = off)"),
           Param("amp_spread_db", 12.0, "dB", 0, 40, "random grain level spread"),
           Param("decay", 0.5, "", 0, 1, "grain window: 0 = symmetric (Hann), 1 = percussive (fast attack, decay)"),
           Param("level", 1.0, "", 0, 4, "output level (automatable)", automatable=True),
       ], ["Roads 2001, Microsound (MIT Press)", "Farnell 2010, Designing Sound (Dust)"])
def grains(ctx: Ctx, x, rate, grain_ms, freq, spread_oct, source, quantize, amp_spread_db, decay, level):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    out = np.zeros(n)
    m = max(8, int(grain_ms * 1e-3 * sr))
    tt = np.arange(m) / sr
    u = np.arange(m) / m
    hann = np.sin(np.pi * u) ** 2
    perc = (1 - np.exp(-u / 0.02)) * np.exp(-u / 0.25)
    win = (1 - decay) * hann + decay * perc / (perc.max() + 1e-12)
    fc = ctrl(ctx, freq, "exp")
    for i in dsp.poisson_times(rng, ctrl(ctx, rate), sr):
        o = (rng.random() - 0.5) * spread_oct
        if quantize > 0:
            o = round(o * quantize) / quantize
        f = fc[i] * 2 ** o
        a = 10 ** (-amp_spread_db * rng.random() / 20)
        if source == "sine":
            g = np.sin(2 * np.pi * f * tt + rng.random() * 2 * np.pi)
        elif source == "saw":
            g = osc_bl("saw", np.full(m, f), sr, phase0=float(rng.random()))
        else:
            g = dsp.bandpass(rng.standard_normal(m), min(f, sr * 0.45), sr, 4.0)
            g /= np.std(g) + 1e-12
        dsp.place(out, g * win, int(i), a)
    return out * ctrl(ctx, level)


# ----------------------------------------------------------------------------- processors
def svf_tv(x: np.ndarray, fc: np.ndarray, q: np.ndarray, sr: int, mode: str) -> np.ndarray:
    """TPT state-variable filter with per-sample cutoff and Q. mode: lp|hp|bp|notch|peak.
    Constant cutoff and Q: one exact lfilter pass (dsp.svf_fast); moving: the per-sample loop below."""
    if np.ptp(fc) == 0 and np.ptp(q) == 0:
        return dsp.svf_fast(x, fc, q, sr, "bpn" if mode == "bp" else mode)
    g = np.tan(np.pi * np.clip(fc, 5.0, sr * 0.45) / sr).tolist()
    k = (1.0 / np.clip(q, 0.05, 100.0)).tolist()
    xs = np.asarray(x, float).tolist()
    out = [0.0] * len(xs)
    ic1 = ic2 = 0.0
    md = ("lp", "bp", "hp", "notch", "peak").index(mode)
    for i in range(len(xs)):
        gi, ki = g[i], k[i]
        a1 = 1.0 / (1.0 + gi * (gi + ki))
        v3 = xs[i] - ic2
        v1 = a1 * (ic1 + gi * v3)
        v2 = ic2 + gi * v1
        ic1 = 2 * v1 - ic1
        ic2 = 2 * v2 - ic2
        if md == 0:
            out[i] = v2
        elif md == 1:
            out[i] = ki * v1  # unity gain at the centre
        elif md == 2:
            out[i] = xs[i] - ki * v1 - v2
        elif md == 3:
            out[i] = xs[i] - ki * v1
        else:
            out[i] = v2 - (xs[i] - ki * v1 - v2)
    return np.array(out)


@model("svf", "fx",
       "Resonant state-variable filter: low-pass, high-pass, band-pass, notch or peak. `cutoff` and `resonance` "
       "are automatable, which is how filter sweeps (wah, whoosh, laser) are written. Resonance is the Q: 0.5 "
       "smooth, 0.707 flat, 5-20 a singing peak.",
       [
           Param("mode", "lp", "", choices=("lp", "hp", "bp", "notch", "peak")),
           Param("cutoff", 1000.0, "Hz", 20, 20000, "cutoff/centre (automatable, log-interpolated)", automatable=True),
           Param("resonance", 0.707, "Q", 0.1, 30, "resonance (automatable)", automatable=True),
       ], ["Zavalishin 2012, The Art of VA Filter Design (TPT SVF)"])
def svf(ctx: Ctx, x, mode, cutoff, resonance):
    return svf_tv(x, ctrl(ctx, cutoff, "exp"), ctrl(ctx, resonance), ctx.sr, mode)


@model("env", "fx",
       "Amplitude envelope. ADSR (attack, decay, sustain level, release after `gate_s`) or, if `points` is given, "
       "a breakpoint list [[t, level], ...]. `curve` exp makes decays and releases exponential (natural), "
       "linear makes them straight lines (synthy).",
       [
           Param("attack", 0.005, "s", 0, 10, "attack time"),
           Param("decay", 0.1, "s", 0, 10, "decay time to the sustain level"),
           Param("sustain", 0.6, "", 0, 1, "sustain level"),
           Param("release", 0.2, "s", 0, 20, "release time (exp: time to -60 dB)"),
           Param("gate_s", 0.3, "s", 0, None, "note length: release starts here (after `at`)"),
           Param("at", 0.0, "s", 0, None, "envelope start (silence before it)"),
           Param("points", [], "", doc="breakpoints [[t, level], ...] overriding ADSR", is_list=True),
           Param("curve", "exp", "", choices=("exp", "linear")),
       ], ["Wikipedia: Envelope (music)"])
def env(ctx: Ctx, x, attack, decay, sustain, release, gate_s, at, points, curve):
    t = ctx.t - at
    if points:
        pts = sorted((float(a), float(b)) for a, b in points)
        ts = np.array([p[0] for p in pts])
        vs = np.array([p[1] for p in pts])
        if curve == "exp":
            e = 10 ** (np.interp(t, ts, 20 * np.log10(np.maximum(vs, 1e-4))) / 20)
            e[e <= 1.01e-4] = 0.0
        else:
            e = np.interp(t, ts, vs)
        if ts[0] > 0:
            e[t < ts[0]] = 0.0
        return x * e
    a = np.clip(t / max(attack, 1e-5), 0, 1)
    td = np.maximum(t - attack, 0)
    if curve == "exp":
        d = sustain + (1 - sustain) * np.exp(-6.91 * td / max(decay, 1e-5))
    else:
        d = 1 - (1 - sustain) * np.clip(td / max(decay, 1e-5), 0, 1)
    e = np.where(t < attack, a, d)
    # level reached at the gate, then release from there
    g = np.interp(gate_s, t, e) if gate_s < t[-1] else None
    if g is not None:
        tr = np.maximum(t - gate_s, 0)
        rel = g * (np.exp(-6.91 * tr / max(release, 1e-5)) if curve == "exp"
                   else np.clip(1 - tr / max(release, 1e-5), 0, 1))
        e = np.where(t < gate_s, e, rel)
    e[t < 0] = 0.0
    return x * e


SHAPES = ("tanh", "hardclip", "fold", "asym", "bitcrush", "downsample")


@model("shaper", "fx",
       "Distortion and lo-fi: tanh saturation, hard clip, wavefolder, asymmetric (tube-like, adds even "
       "harmonics), bit-depth crusher, or sample-rate reducer. The smooth shapers run at 4x oversampling; the "
       "crushers alias on purpose (retro).",
       [
           Param("mode", "tanh", "", choices=SHAPES),
           Param("drive", 2.0, "", 0.1, 50, "input gain into the shaper (automatable)", automatable=True),
           Param("bits", 8, "bits", 1, 16, "bitcrush depth"),
           Param("rate_hz", 8000.0, "Hz", 100, 48000, "downsample target rate"),
           Param("mix", 1.0, "", 0, 1, "wet/dry"),
       ], ["Wikipedia: Distortion (music); Bitcrusher"])
def shaper(ctx: Ctx, x, mode, drive, bits, rate_hz, mix):
    pk = np.max(np.abs(x)) + 1e-12
    xn = x / pk
    if mode == "bitcrush":
        q = 2 ** (int(bits) - 1)
        y = np.round(xn * ctrl(ctx, drive) * q) / q
        y = np.clip(y, -1, 1)
    elif mode == "downsample":
        step = max(1.0, ctx.sr / rate_hz)
        idx = (np.floor(np.arange(ctx.n) / step) * step).astype(int)
        y = xn[idx]
    else:
        f = 4
        up = signal.resample_poly(xn, f, 1)
        d = np.repeat(ctrl(ctx, drive), f)[: len(up)]
        u = up * d
        if mode == "tanh":
            w = np.tanh(u)
        elif mode == "hardclip":
            w = np.clip(u, -1, 1)
        elif mode == "fold":
            w = 4 * np.abs(((u - 1) / 4) % 1 - 0.5) - 1  # triangle fold
        else:
            w = np.tanh(u + 0.3) - np.tanh(0.3)
        y = signal.resample_poly(w, 1, f)[: ctx.n]
    return pk * (mix * y + (1 - mix) * xn)


def _feedback_delay(x: np.ndarray, D: int, fb: float, damp: float) -> np.ndarray:
    """y[n] = x[n] + fb * lowpass(y[n-D]); processed in chunks of D samples (exact, vectorised)."""
    n = len(x)
    y = np.zeros(n)
    zi = np.zeros(1)
    b, a = [1 - damp], [1, -damp]
    for s in range(0, n, D):
        e = min(s + D, n)
        if s >= D:
            prev, zi = signal.lfilter(b, a, y[s - D:e - D], zi=zi)
            y[s:e] = x[s:e] + fb * prev
        else:
            y[s:e] = x[s:e]
    return y


@model("delay", "fx",
       "Echo: feedback delay with a darkening filter in the loop (each repeat duller, like tape/analogue).",
       [
           Param("time_ms", 250.0, "ms", 1, 4000, "delay time"),
           Param("feedback", 0.35, "", 0, 0.97, "repeat level"),
           Param("damp", 0.3, "", 0, 0.99, "darkening per repeat"),
           Param("mix", 0.3, "", 0, 1, "wet level"),
       ], ["Wikipedia: Delay (audio effect)"])
def delay(ctx: Ctx, x, time_ms, feedback, damp, mix):
    D = max(1, int(time_ms * 1e-3 * ctx.sr))
    wet = np.zeros(ctx.n)
    wet[D:] = _feedback_delay(x, D, feedback, damp)[:-D] if D < ctx.n else 0
    return x + mix * wet


@model("chorus", "fx",
       "Chorus/flanger: copies of the signal through short delays swept by slow LFOs. 1-5 ms + feedback-free = "
       "flanger-ish; 10-25 ms with several voices = chorus (thick, shimmering).",
       [
           Param("rate_hz", 0.8, "Hz", 0.01, 20, "LFO rate"),
           Param("depth_ms", 3.0, "ms", 0, 20, "LFO depth"),
           Param("base_ms", 15.0, "ms", 0.5, 50, "centre delay"),
           Param("voices", 3, "count", 1, 8, "number of delayed copies"),
           Param("mix", 0.5, "", 0, 1, "wet level"),
       ], ["Wikipedia: Chorus effect"])
def chorus(ctx: Ctx, x, rate_hz, depth_ms, base_ms, voices, mix):
    wet = np.zeros(ctx.n)
    v = int(max(1, voices))
    for k in range(v):
        ph = 2 * np.pi * k / v
        d = (base_ms + depth_ms * np.sin(2 * np.pi * rate_hz * ctx.t + ph)) * 1e-3 * ctx.sr
        wet += dsp.variable_delay(x, d)
    return (1 - 0.5 * mix) * x + mix * wet / v


_COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)  # Freeverb tunings at 44.1 kHz
_APS = (556, 441, 341, 225)


@model("reverb", "fx",
       "Algorithmic reverb (mono Freeverb: 8 damped parallel combs + 4 series allpasses). For stylised sounds "
       "and quick space; for physically scaled rooms use `room` / `outdoor`.",
       [
           Param("size", 0.6, "", 0, 1, "room size / decay length"),
           Param("damp", 0.4, "", 0, 1, "high-frequency damping in the tail"),
           Param("predelay_ms", 10.0, "ms", 0, 200, "gap before the tail"),
           Param("mix", 0.3, "", 0, 1, "wet level"),
       ], ["Freeverb by Jezar at Dreampoint (public domain), described in J.O. Smith, Physical Audio Signal "
           "Processing (CCRMA): Freeverb"])
def reverb(ctx: Ctx, x, size, damp, predelay_ms, mix):
    sr = ctx.sr
    s = sr / 44100.0
    fb = 0.7 + 0.28 * size
    d = 0.4 * damp
    pre = int(predelay_ms * 1e-3 * sr)
    xin = np.concatenate([np.zeros(pre), x])[: ctx.n] * 0.015
    acc = np.zeros(ctx.n)
    for L in _COMBS:
        D = int(L * s)
        y = _feedback_delay(xin, D, fb, d)
        acc[D:] += y[:-D]  # comb output = delayed line
    out = acc
    for L in _APS:
        D = int(L * s)
        buf = _feedback_delay(out, D, 0.5, 0.0)  # buf[n] = in[n] + 0.5 buf[n-D]
        bd = np.zeros(ctx.n)
        bd[D:] = buf[:-D]
        out = bd - out
    out = out * (np.std(x) / (np.std(out) + 1e-12)) * 0.6
    return (1 - 0.3 * mix) * x + mix * out


@model("gain", "fx",
       "Level change with optional tremolo (automatable gain in dB). Use it to duck or shape a layer in a chain.",
       [
           Param("gain_db", 0.0, "dB", -80, 24, "gain (automatable)", automatable=True),
           Param("tremolo_hz", 0.0, "Hz", 0, 40, "tremolo rate (0 = off)"),
           Param("tremolo_depth", 0.0, "", 0, 1, "tremolo depth"),
       ])
def gain(ctx: Ctx, x, gain_db, tremolo_hz, tremolo_depth):
    g = 10 ** (ctrl(ctx, gain_db) / 20)
    if tremolo_hz > 0 and tremolo_depth > 0:
        g = g * (1 - tremolo_depth * 0.5 * (1 - np.cos(2 * np.pi * tremolo_hz * ctx.t)))
    return x * g
