"""Stage 5: environment (rooms and outdoor spaces)."""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .. import dsp
from ..core import Ctx, Param, model

BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]


def late_ir(rng, sr: int, rt60: float, hf_ratio: float, length: float) -> np.ndarray:
    """Statistical late reverb: band-limited noise decaying at a per-band T60 (Sabine/Eyring diffuse field)."""
    n = int(length * sr)
    t = np.arange(n) / sr
    ir = np.zeros(n)
    for i, fc in enumerate(BANDS):
        k = i / (len(BANDS) - 1)
        t60 = rt60 * (1.0 + (hf_ratio - 1.0) * k)  # highs decay faster when hf_ratio < 1
        band = dsp.bandpass(rng.standard_normal(n), fc, sr, 1.4)
        ir += band * np.exp(-6.91 * t / max(t60, 0.02))
    return ir / (np.sqrt(np.sum(ir ** 2)) + 1e-12)


def shoebox_early(sr: int, dims, src, lst, absorption: float, c: float) -> np.ndarray:
    """First- and second-order image sources of a shoebox (Allen & Berkley image method)."""
    Lx, Ly, Lz = dims
    taps = []
    refl = math.sqrt(1 - absorption)
    for nx in range(-2, 3):
        for ny in range(-2, 3):
            for nz in range(-2, 3):
                order = abs(nx) + abs(ny) + abs(nz)
                if order == 0 or order > 2:
                    continue
                def img(n, s, L):
                    return n * L + (s if n % 2 == 0 else L - s)
                p = (img(nx, src[0], Lx), img(ny, src[1], Ly), img(nz, src[2], Lz))
                d = math.dist(p, lst)
                taps.append((d, refl ** order / d))
    d0 = math.dist(src, lst)
    n = int((max(t[0] for t in taps) - d0) / c * sr) + 2
    ir = np.zeros(n)
    for d, g in taps:
        ir[int((d - d0) / c * sr)] += g * d0
    return ir


@model("room", "room",
       "Room: early reflections from a shoebox image-source model plus a statistical late tail. "
       "RT60 from Sabine's formula T60 = 0.161 V / (S a) unless given.",
       [
           Param("volume", 200.0, "m^3", 5, 100000, "room volume"),
           Param("absorption", 0.25, "", 0.01, 0.99, "average absorption coefficient of the surfaces"),
           Param("rt60", 0.0, "s", 0, 20, "override reverb time (0 = Sabine from volume/absorption)"),
           Param("hf_ratio", 0.5, "", 0.1, 1.5, "T60 at 8 kHz relative to 125 Hz"),
           Param("source_distance", 3.0, "m", 0.2, 100, "source-listener distance (sets direct/reverb ratio)"),
           Param("wet", 1.0, "", 0, 2, "scale on the physically derived reverb level"),
       ], ["Wikipedia: Reverberation (Sabine)", "Murch, Transom 2005 (direct-to-reverb = depth)",
           "pyroomacoustics docs (image-source method)"])
def room(ctx: Ctx, x, volume, absorption, rt60, hf_ratio, source_distance, wet):
    Lz = min(3.0 + volume ** (1 / 3) * 0.3, volume ** (1 / 3))
    Lx = math.sqrt(volume / Lz * 1.5)
    Ly = volume / Lz / Lx
    S = 2 * (Lx * Ly + Lx * Lz + Ly * Lz)
    T = rt60 or 0.161 * volume / (S * absorption)
    src = (Lx * 0.3, Ly * 0.4, min(1.5, Lz * 0.5))
    ang = 0.6
    lst = (min(src[0] + source_distance * math.cos(ang), Lx * 0.95), min(src[1] + source_distance * math.sin(ang), Ly * 0.95), 1.6 if Lz > 1.8 else Lz * 0.5)
    early = shoebox_early(ctx.sr, (Lx, Ly, Lz), src, lst, absorption, ctx.c)
    # critical distance (Sabine diffuse field): r_c ~ 0.057 sqrt(V / T60); reverb energy ~ (r / r_c)^2 x direct
    rc = 0.057 * math.sqrt(volume / T)
    rev_gain = (source_distance / rc) * wet
    tail = late_ir(ctx.rng, ctx.sr, T, hf_ratio, min(T * 1.2, 12.0))
    ir = np.zeros(max(len(early), 1) + len(tail))
    ir[: len(early)] += early
    pre = int(len(early) * 0.5)
    ir[pre:pre + len(tail)] += tail * rev_gain
    ir[0] = 1.0
    y = signal.fftconvolve(x, ir)[: len(x)]
    return y / (1 + 0.3 * rev_gain)


@model("outdoor", "room",
       "Outdoor space: discrete echoes from distant reflectors (cliffs, buildings, tree lines) and an optional "
       "long diffuse low-passed tail (valley/forest rumble).",
       [
           Param("reflectors", [[120, 0.5], [260, 0.35], [600, 0.25]], "[[m, reflection], ...]", None, None,
                 "distance and reflection strength of each reflector", is_list=True),
           Param("tail_s", 1.5, "s", 0, 20, "diffuse tail T60 (0 = none)"),
           Param("tail_level", 0.15, "", 0, 2, "diffuse tail level"),
           Param("tail_lowpass", 1500.0, "Hz", 100, 16000, "tail brightness"),
       ], ["Farnell 2010 practical 31", "Wikipedia: Thunder (rolling echoes)"])
def outdoor(ctx: Ctx, x, reflectors, tail_s, tail_level, tail_lowpass):
    from .propagation import air_absorb
    y = x.copy()
    for d, g in reflectors:
        path = 2.0 * d
        n = int(path / ctx.c * ctx.sr)
        if n >= len(x):
            continue
        e = air_absorb(x[: len(x) - n], ctx.sr, path) * g / max(path / 20.0, 1.0)
        y[n:] += dsp.lowpass(e, 5000.0, ctx.sr)
    if tail_s > 0 and tail_level > 0:
        L = int(min(tail_s * 1.2, ctx.duration) * ctx.sr)
        t = np.arange(L) / ctx.sr
        ir = dsp.lowpass(ctx.rng.standard_normal(L), tail_lowpass, ctx.sr) * np.exp(-6.91 * t / tail_s)
        ir *= np.clip(t / 0.05, 0, 1)
        ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
        y += signal.fftconvolve(x, ir)[: len(x)] * tail_level
    return y
