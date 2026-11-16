"""Stage 3: radiation (how the vibration leaves the object)."""
from __future__ import annotations

import math

from .. import dsp
from ..core import Ctx, Param, model


@model("radiation", "radiation",
       "Small-source radiation: a source of radius a radiates poorly below about c/(2 pi a), so the low end "
       "is rolled off (monopole/dipole high-pass). Optional extra tilt for orientation.",
       [
           Param("radius", 0.05, "m", 0.001, 10, "effective radiating radius (opening, plate size)"),
           Param("order", 1, "", 1, 2, "1 = monopole-like (6 dB/oct), 2 = dipole-like (12 dB/oct)"),
           Param("tilt_db", 0.0, "dB", -24, 24, "high-shelf above 4 kHz (facing towards/away)"),
       ], ["J. O. Smith, PASP (radiation)", "Euphonics"])
def radiation(ctx: Ctx, x, radius, order, tilt_db):
    fc = ctx.c / (2 * math.pi * radius)
    y = x
    for _ in range(int(order)):
        y = dsp.biquad(y, "highpass", fc, 0.6, ctx.sr)
    if tilt_db:
        y = dsp.biquad(y, "highshelf", 4000.0, 0.7, ctx.sr, tilt_db)
    return y


@model("horn", "radiation",
       "Horn/trumpet: the bell passes frequencies above about c/(2 pi a_mouth) and the horn's air column adds "
       "weak resonances (open-open spacing c/2L).",
       [
           Param("length", 2.0, "m", 0.05, 20, "horn length"),
           Param("mouth_radius", 0.2, "m", 0.005, 3, "bell radius"),
           Param("resonance", 0.45, "", 0, 0.95, "strength of the horn's own resonances"),
           Param("damp_hz", 3000.0, "Hz", 200, 20000, "high-frequency loss in the horn"),
       ], ["J. O. Smith, PASP (conical bores, bells)", "Terry Pepper: diaphone trumpets"])
def horn(ctx: Ctx, x, length, mouth_radius, resonance, damp_hz):
    D = 2 * length / ctx.c * ctx.sr
    y = dsp.comb_waveguide(x, D, resonance, ctx.sr, damp_hz, 1.0) * (1 - resonance)
    fc = ctx.c / (2 * math.pi * mouth_radius)
    return dsp.biquad(y, "highpass", fc, 0.7, ctx.sr)
