"""Stage 2: resonators (what vibrates and colours the excitation)."""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .. import dsp, materials
from ..core import Ctx, Param, RecipeError, ctrl, model, scalar

# Free-free beam: beta_n * L roots (Euphonics gives ratios 1, 2.76, 5.40, 8.93, 13.34)
BAR_BETA_L = [4.7300, 7.8532, 10.9956, 14.1372, 17.2788]
# Circular membrane Bessel zeros j_mn in ascending order (Rochester PHY103 lists the first four)
MEMBRANE_J = [2.405, 3.832, 5.136, 5.520, 6.380, 7.016, 7.588, 8.417, 8.654, 8.771, 9.761, 9.936, 10.173, 11.065,
              11.086]  # v0.21.2: j_61 = 9.936 was missing


def bar_modes(n: int) -> list[float]:
    # roots 1-5 from the table, then the asymptote (2k + 1) pi / 2 from k = 6 (until v0.21.1 this started at
    # k = 5, which is mode 5 again: a doubled mode 5 and every higher mode one place too low)
    bl = BAR_BETA_L + [(2 * k + 1) * math.pi / 2 for k in range(len(BAR_BETA_L) + 1, n + 2)]
    return [(b / bl[0]) ** 2 for b in bl[:n]]


def mode_set(shape: str, mat: dict, length: float, width: float, thickness: float, f0: float, n: int,
             ratios: list[float] | None, strike_pos: float, rng: np.random.Generator):
    """Return (freqs, relative amplitudes) for a resonator shape."""
    E, rho, nu = mat["E"], mat["rho"], mat["nu"]
    if shape == "custom":
        if not ratios:
            raise RecipeError("modal: shape 'custom' needs 'ratios'")
        r = np.array(ratios[:n], float)
        base = f0 or 440.0
        f = base * r
        amp = 1 / np.sqrt(np.arange(1, len(r) + 1))
    elif shape == "bar":
        r = np.array(bar_modes(n))
        base = f0 or (BAR_BETA_L[0] ** 2 / (2 * math.pi * length ** 2)) * thickness * math.sqrt(E / (12 * rho))
        f = base * r
        k = np.arange(1, n + 1)
        amp = np.abs(np.sin((k + 0.5) * math.pi * strike_pos)) + 0.05  # approximate mode shape at strike point
    elif shape == "plate":  # simply-supported rectangular plate (textbook result)
        D = E * thickness ** 3 / (12 * (1 - nu ** 2))
        c0 = (math.pi / 2) * math.sqrt(D / (rho * thickness))
        mm = [(a, b) for a in range(1, 12) for b in range(1, 12)]
        fr = np.array([c0 * ((a / length) ** 2 + (b / width) ** 2) for a, b in mm])
        order = np.argsort(fr)[:n]
        f = fr[order]
        if f0:
            f = f * (f0 / f[0])
        sp = (strike_pos, 0.37 + 0.2 * strike_pos)
        amp = np.array([abs(math.sin(mm[i][0] * math.pi * sp[0]) * math.sin(mm[i][1] * math.pi * sp[1])) + 0.05
                        for i in order])
    elif shape == "membrane":
        r = np.array(MEMBRANE_J[:n]) / MEMBRANE_J[0]
        f = (f0 or 150.0) * r
        amp = np.exp(-0.5 * strike_pos * np.arange(len(r))) * rng.uniform(0.6, 1.0, len(r))
        amp[0] *= 1.0 - 0.8 * strike_pos  # striking off-centre weakens the fundamental
    elif shape == "string":
        k = np.arange(1, n + 1)
        f = (f0 or 220.0) * k
        amp = np.abs(np.sin(k * math.pi * strike_pos)) / k + 1e-3
    elif shape in ("tube_open", "tube_closed"):
        L = length
        if shape == "tube_open":
            k = np.arange(1, n + 1)
            base = f0 or 343.0 / (2 * L)
        else:
            k = np.arange(1, 2 * n, 2)[:n]
            base = f0 or 343.0 / (4 * L)
        f = base * k
        amp = 1 / k
    else:
        raise RecipeError(f"modal: unknown shape {shape!r}")
    return np.asarray(f, float), np.asarray(amp, float)


@model("modal", "resonator",
       "Modal resonator: sum of exponentially decaying sinusoids. Frequencies come from geometry and "
       "material (or f0 + mode ratios); each mode decays at rate pi*f*eta (eta = material loss factor), "
       "plus a frequency-independent mounting/radiation loss.",
       [
           Param("material", "steel", "", choices=tuple(materials.MATERIALS)),
           Param("shape", "bar", "", choices=("bar", "plate", "membrane", "string", "tube_open", "tube_closed", "custom")),
           Param("length", 0.3, "m", 0.005, 20, "length (bar/plate/tube)"),
           Param("width", 0.2, "m", 0.005, 20, "width (plate)"),
           Param("thickness", 0.005, "m", 0.0002, 0.5, "thickness (bar/plate)"),
           Param("f0", 0.0, "Hz", 0, 20000, "override the lowest mode (0 = compute from geometry)"),
           Param("ratios", None, "", None, None, "custom mode ratios (shape 'custom')", is_list=True),
           Param("n_modes", 24, "count", 1, 200, "number of modes"),
           Param("strike_pos", 0.2, "", 0, 1, "strike position along the object (0 = edge/end)"),
           Param("eta", 0.0, "", 0, 1, "loss factor override (0 = use material table)"),
           Param("klatzky", "none", "", choices=("none", "rubber", "wood", "glass", "steel"),
                 doc="use Klatzky et al. 2000 perceptual material decay instead of the material table"),
           Param("extra_damping", 3.0, "1/s", 0, 2000, "frequency-independent loss (mounting, radiation, hand)"),
           Param("detune", 0.01, "", 0, 0.2, "random mode detuning (real objects are imperfect)"),
           Param("max_t60", 8.0, "s", 0.01, 60, "cap on ring time (keeps renders bounded)"),
           Param("gain", 1.0, "", 0, 100, "output gain"),
       ],
       ["van den Doel, Kry & Pai 2001", "Klatzky, Pai & Krotkov 2000", "Cremer & Heckl via vibrationdata",
        "Euphonics: free-free beam modes", "Rochester PHY103: membrane modes"])
def modal(ctx: Ctx, x, material, shape, length, width, thickness, f0, ratios, n_modes, strike_pos, eta, klatzky,
          extra_damping, detune, max_t60, gain):
    mat = materials.get(material)
    f, amp = mode_set(shape, mat, length, width, thickness, f0, int(n_modes), ratios, strike_pos, ctx.rng)
    if detune > 0:
        f = f * (1 + detune * ctx.rng.uniform(-1, 1, len(f)) * (np.arange(len(f)) > 0))
    keep = f < ctx.sr * 0.45
    f, amp = f[keep], amp[keep]
    if len(f) == 0:
        return np.zeros_like(x)
    if klatzky != "none":
        e = materials.eta_from_klatzky(klatzky)
    else:
        e = eta or mat["eta_default"]
    alpha = math.pi * f * e + extra_damping  # amplitude decay rate, 1/s
    t60 = np.minimum(6.91 / alpha, max_t60)
    alpha = 6.91 / t60
    L = int(min(ctx.n, max(t60.max(), 0.01) * ctx.sr))
    t = np.arange(L) / ctx.sr
    ph = ctx.rng.uniform(0, 0.3, len(f))
    ir = np.zeros(L)
    for fi, ai, al, p in zip(f, amp, alpha, ph):
        ir += ai * np.exp(-al * t) * np.sin(2 * math.pi * fi * t + p)
    ir *= 2 * math.pi * f[0] / ctx.sr * 4  # roughly level-independent of pitch
    y = signal.fftconvolve(x, ir)[: len(x)]
    return y * gain


@model("string", "resonator",
       "Digital waveguide string (Karplus-Strong family): delay line of one period with a lowpass loop filter.",
       [
           Param("f0", 220.0, "Hz", 20, 5000, "fundamental"),
           Param("t60", 2.0, "s", 0.01, 30, "ring time of the fundamental (a low `brightness` caps how long high notes "
                 "can ring: the loop lowpass alone loses energy every period)"),
           Param("brightness", 5000.0, "Hz", 200, 20000, "loop lowpass corner (lower = duller, faster high decay)"),
           Param("dispersion", 0.0, "", 0, 0.95, "stiffness: allpass dispersion inside the loop (wires, springs; "
                 "high values give the falling 'pew' chirp)"),
           Param("dispersion_stages", 16, "count", 1, 64, "number of allpass sections (more = stronger chirp)"),
           Param("gain", 1.0, "", 0, 10, "output gain"),
       ], ["J. O. Smith, Physical Audio Signal Processing", "Wikipedia: Digital waveguide synthesis"])
def string(ctx: Ctx, x, f0, t60, brightness, dispersion, dispersion_stages, gain):
    D = ctx.sr / f0
    g = 10 ** (-3 / (f0 * t60))
    if dispersion > 0:
        y = dsp.dispersive_waveguide(x, D, g, ctx.sr, brightness, -dispersion, int(dispersion_stages))
    else:
        # v0.21.2: tuned loop (exact pitch, see dsp.comb_waveguide) and t60 measured on the fundamental: the loop
        # lowpass also loses |H(f0)| per period, which shortened high notes' ring far below the given t60
        p = math.exp(-2 * math.pi * brightness / ctx.sr)
        w0 = 2 * math.pi * f0 / ctx.sr
        h0 = (1 - p) / abs(1 - p * np.exp(-1j * w0))
        y = dsp.comb_waveguide(x, D, min(g / h0, 0.99999), ctx.sr, brightness, 1.0, tune=True)
    return y * gain * 0.2


@model("tube", "resonator",
       "Acoustic tube as a digital waveguide: round-trip delay 2L/c, reflection sign set by the ends. "
       "Closed-open tubes resonate at odd multiples of c/4L, open-open (and cones) at all multiples of c/2L.",
       [
           Param("length", 0.5, "m", 0.02, 30, "tube length"),
           Param("ends", "closed_open", "", choices=("closed_open", "open_open", "cone")),
           Param("loss", 0.85, "", 0, 0.995, "round-trip reflection magnitude (lower = more damped, wider bore loss)"),
           Param("damp_hz", 4000.0, "Hz", 100, 20000, "high-frequency loss in the loop"),
           Param("mix", 1.0, "", 0, 1, "wet/dry"),
       ], ["J. O. Smith, PASP (cylindrical and conical bores)"])
def tube(ctx: Ctx, x, length, ends, loss, damp_hz, mix):
    D = 2 * length / ctx.c * ctx.sr
    sign = -1.0 if ends == "closed_open" else 1.0
    y = dsp.comb_waveguide(x, D, loss, ctx.sr, damp_hz, sign, tune=True) * (1 - loss) * 2   # v0.21.2: in tune
    return mix * y + (1 - mix) * x


@model("formant", "resonator",
       "Parallel band-pass resonances (source-filter model): vocal tracts, tracheae, cabin and body resonances.",
       [
           Param("formants", [[700, 5, 0], [1800, 4, 6], [3000, 6, -3]], "[[Hz, Q, dB], ...]", None, None,
                 "list of resonances", is_list=True),
           Param("shift", 1.0, "", 0.25, 4, "scale all formant frequencies (size of the tract)", automatable=True),
           Param("dry", 0.05, "", 0, 1, "unfiltered amount"),
       ], ["Source-filter model (speech science)", "Farnell 2010"])
def formant(ctx: Ctx, x, formants, shift, dry):
    s = ctrl(ctx, shift)
    y = dry * x
    const = not isinstance(shift, (list, tuple))
    for fq, q, g in formants:
        if const:
            y = y + 10 ** (g / 20) * dsp.bandpass(x, fq * s[0], ctx.sr, q)
        else:
            y = y + 10 ** (g / 20) * dsp.svf(x, fq * s, q, ctx.sr, "bp")
    return y


@model("helmholtz", "resonator",
       "Helmholtz cavity (bottle, muffler chamber): f = c/(2 pi) sqrt(A / (V L)).",
       [
           Param("volume", 0.001, "m^3", 1e-6, 10, "cavity volume"),
           Param("neck_area", 3e-4, "m^2", 1e-6, 1, "neck cross-section"),
           Param("neck_length", 0.05, "m", 0.001, 2, "effective neck length"),
           Param("q", 8.0, "", 0.5, 200, "sharpness"),
           Param("mix", 0.7, "", 0, 1, "wet/dry"),
       ], ["Wikipedia: Helmholtz resonance"])
def helmholtz(ctx: Ctx, x, volume, neck_area, neck_length, q, mix):
    f = ctx.c / (2 * math.pi) * math.sqrt(neck_area / (volume * neck_length))
    return mix * dsp.bandpass(x, f, ctx.sr, q) * math.sqrt(q) + (1 - mix) * x


@model("filter", "resonator", "Generic biquad filter (craft control).",
       [
           Param("kind", "lowpass", "", choices=("lowpass", "highpass", "bandpass", "peak", "lowshelf", "highshelf")),
           Param("freq", 1000.0, "Hz", 10, 22000, "corner/centre frequency", automatable=True),
           Param("q", 0.707, "", 0.1, 50, "resonance"),
           Param("gain_db", 0.0, "dB", -48, 48, "gain for peak/shelf"),
       ])
def filt(ctx: Ctx, x, kind, freq, q, gain_db):
    if isinstance(freq, (list, tuple)) and kind in ("lowpass", "highpass", "bandpass"):
        mode = {"lowpass": "lp", "highpass": "hp", "bandpass": "bp"}[kind]
        return dsp.svf(x, ctrl(ctx, freq, "exp"), q, ctx.sr, mode)
    return dsp.biquad(x, kind, scalar(freq), q, ctx.sr, gain_db)
