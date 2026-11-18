"""friction_family (v0.27): creaks, squeaks, scrapes, rubbing and bowing through the material_impact bodies.

Method (all original code):
- Stick-slip: the elasto-plastic bristle model (`ep_friction`; Dupont et al. 2002; Avanzini, Serafin & Rocchesso
  2005, the SDT friction method) is solved together with the struck body's lowest modes, taken from the same
  shape x material tables and damping laws as `material_impact`. A light contact on a springy drive (`drive_hz` >
  0) sticks, winds up and slips quickly, so the stick time sets a creak rate that rises with speed and falls
  with pressure; a rigid drive squeals or bows. Wood modes keep wood damping, so a wooden creak is a
  dense, short-ringing knock train, not a metallic tone.
- The rest of the body: the slip pulses also excite the higher modes and the cavity (convolution with the
  body's modal response), plus a fibre-noise residual shaped by the body's spectrum and following the slip
  activity (Ren, Yeh & Lin 2013 residual idea).
- Scrape: a fractal surface profile read at the contact speed (`surface`; van den Doel, Kry & Pai 2001; Rath &
  Rocchesso 2004) drives the full body. Rub: the same force with no modes (cloth, sandpaper on soft backing),
  band-limited by the material.
- Squeal (v0.27.1): the sound of a dry pin turning in its knuckle, a rubber sole or a loaded rope comes from the
  stick-slip excitation itself, not from a ringing body. A slip-by-slip relaxation oscillator (slow stick ramp,
  quick slip) whose rate follows speed and pressure, with per-cycle period and amplitude jitter, missed slips
  and slip noise for roughness; the body only colours it through a short (<= 40 ms) modal response, and the
  level follows the speed smoothly, so the stroke ends without a knock.
- Gesture: speed and pressure over `length_s` with a start/end ramp, a speed change (slowing to a stop ..
  speeding up), and irregularity (slow random wobble of speed and pressure, as a hand or a settling door gives).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .. import dsp
from ..core import REGISTRY, Ctx, Param, model
from .impacts import IMPACT_MATERIALS, SHAPES, cavity_modes, decay_rates, modal_ir, mode_table

MATERIALS = tuple(k for k, v in IMPACT_MATERIALS.items() if not v.get("ground")) + ("cloth",)
# band of the rubbing noise for modeless materials: (high-pass, low-pass) Hz
SQUEAL_V_REF, SQUEAL_N_REF = 0.05, 5.0   # m/s, N: `pitch_hz` is the slip rate at this speed and pressure
RUB_BAND = {"cloth": (700, 7000), "rubber": (300, 3000), "flesh": (200, 2500)}


def _smooth_noise(rng, n, sr, rate_hz):
    k = max(4, int(n / sr * rate_hz) + 4)
    pts = rng.standard_normal(k)
    return np.interp(np.linspace(0, k - 3, n), np.arange(k), pts)


def _gesture(ctx, at, length_s, speed, speed_change, pressure, irregularity):
    """Per-sample speed and pressure for one stroke."""
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    t = np.arange(n) / sr
    u = np.clip((t - at) / max(length_s, 1e-3), 0, 1)
    on = (t >= at) & (t <= at + length_s)
    ramp = np.minimum(1, (t - at) / (0.08 * length_s + 1e-3)) * np.minimum(1, (at + length_s - t) / (0.1 * length_s + 1e-3))
    v = speed * np.exp(speed_change * math.log(4.0) * (u - 0.5)) * np.clip(ramp, 0, 1) * on
    # v0.27.1: the contact stays pressed after the motion stops (a door coming to rest still bears on its hinge);
    # releasing it at the stroke end let the wound-up contact slip once more and knock the body
    N = pressure * np.clip((t - at) / (0.05 * length_s + 1e-3), 0, 1)
    if irregularity > 0:
        v = v * np.clip(1 + 0.7 * irregularity * _smooth_noise(rng, n, sr, 4.0), 0.05, None)
        N = N * np.clip(1 + 0.5 * irregularity * _smooth_noise(rng, n, sr, 6.0), 0.1, None)
    return v, N


def _bp(arr, sr, rate=400):
    """Per-sample array -> breakpoint list for blocks that take automation."""
    step = max(1, sr // rate)
    idx = np.arange(0, len(arr), step)
    return [[float(i / sr), float(arr[i])] for i in idx] + [[float((len(arr) - 1) / sr), float(arr[-1])]]


def _squeal(ctx, V, N, speed, pressure, pitch_hz, follow, slip_shape, irregularity, roughness, crackle):
    """Slip-by-slip stick-slip excitation: (force waveform, slip activity)."""
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    vs = np.maximum(V, 0.0) / max(speed, 1e-6)                  # stroke shape (level)
    va = np.maximum(V, 0.0) / SQUEAL_V_REF                       # absolute speed and pressure (rate), so the
    ns = np.maximum(N, 1e-6) / SQUEAL_N_REF                      # speed and pressure sliders move the pitch
    # level: smooth in speed (the squeal fades as the motion dies, no step), one-pole 4 ms up / 40 ms down
    lvl = np.clip((vs - 0.08) / 0.5, 0, 1) ** 1.5 * np.sqrt(np.clip(N / max(pressure, 1e-6), 0, 3))
    lvl = signal.lfilter([1 - math.exp(-1 / (0.004 * sr))], [1, -math.exp(-1 / (0.004 * sr))], lvl)
    f = pitch_hz * np.maximum(va, 0.02) ** follow * ns ** -0.2
    f = f * np.exp(0.05 * irregularity * _smooth_noise(rng, n, sr, 7.0))
    f = np.clip(f, 20, 0.4 * sr)
    jit = 0.004 + 0.03 * roughness + 0.02 * irregularity
    shim = 0.03 + 0.25 * roughness
    miss = 0.12 * irregularity * roughness
    out = np.zeros(n)
    act = np.zeros(n)
    on = np.nonzero(lvl > 1e-3)[0]
    i = int(on[0]) if len(on) else n
    end = int(on[-1]) + 1 if len(on) else 0
    while i < end:
        L = sr / f[i] * (1 + jit * rng.standard_normal())
        if rng.random() < miss:                       # a slip that does not come: the next one is bigger and later
            L *= 2
        L = int(max(4, round(L)))
        a = max(0.0, 1 + shim * rng.standard_normal())
        d = max(2, int(round(slip_shape * L)))
        seg = np.concatenate([np.linspace(-1, 1, L - d, endpoint=False), np.linspace(1, -1, d, endpoint=False)])
        j = min(n, i + L)
        out[i:j] += a * seg[:j - i] * lvl[i:j]                 # level per sample: a slow last cycle fades out
        act[i:min(n, i + d)] += a * lvl[i:min(n, i + d)]
        i += L
    # slip noise (roughness): a short burst at every slip
    nz = rng.standard_normal(n) * np.convolve(act, np.ones(int(0.0006 * sr)), "same")
    out = dsp.highpass(out, 40.0, sr) + (0.05 + 0.45 * roughness) * dsp.highpass(nz, 1500.0, sr) * 0.3
    # the pin, sole or fibre is soft at the slip: the upper harmonics roll off (two poles above ~5 f0)
    f_on = f[lvl > 0.1 * lvl.max()] if lvl.max() > 0 else f
    fc = min(0.45 * sr, 2500.0 + 5.0 * float(np.median(f_on)))
    out = dsp.lowpass(dsp.lowpass(out, fc, sr), fc, sr)
    if crackle > 0:                                   # fibres giving way: sparse clicks where the rope works
        p = np.clip(lvl, 0, 1) * 60.0 * crackle / sr
        clicks = (rng.random(n) < p) * rng.standard_normal(n)
        cr = dsp.lowpass(dsp.highpass(np.convolve(clicks, np.hanning(int(0.0015 * sr)), "same"), 300, sr), 4000, sr)
        out = out + crackle * 0.6 * cr / (np.abs(cr).max() + 1e-12) * np.abs(out).max()
    return out, lvl


@model("friction_family", "sound",
       "Creaks, squeaks, scrapes, rubbing and bowing through the same material bodies as `material_impact`. "
       "`mode` stick_slip: elasto-plastic friction solved with the body's lowest modes (springy `drive_hz` > 0 "
       "creaks and ticks; rigid 0 squeals or bows), the slips also ringing the rest of the body and a fibre-noise "
       "residual. `mode` scrape: a fractal surface read at the contact speed drives the whole body. `mode` rub: the "
       "same surface force, no modes (cloth, padded sandpaper). One stroke of `length_s` with `speed`, "
       "`speed_change` (-1 slowing to a stop .. +1 speeding up), `pressure` and `irregularity`.",
       [
           Param("mode", "stick_slip", "", choices=("stick_slip", "scrape", "rub", "squeal")),
           Param("material", "wood_hard", "", choices=MATERIALS),
           Param("shape", "plate", "", choices=SHAPES),
           Param("size", 0.5, "m", 0.01, 5.0, "largest dimension of the rubbed body (pitch ~ 1/size)"),
           Param("hollowness", 0.3, "", 0, 1, "0 = solid, 1 = thin-walled and hollow"),
           Param("contact_point", 0.3, "", 0, 1, "0 = edge/end .. 1 = middle"),
           Param("at", 0.05, "s", 0, None),
           Param("length_s", 1.5, "s", 0.05, 30, "stroke length"),
           Param("speed", 0.08, "m/s", 0.001, 5.0, "rubbing speed"),
           Param("speed_change", 0.0, "", -1, 1, "-1 slows to a stop .. +1 speeds up (x0.5 .. x2 over the stroke)"),
           Param("pressure", 4.0, "N", 0.05, 40, "pressing force"),
           Param("irregularity", 0.4, "", 0, 1, "wobble of speed and pressure (uneven slips)"),
           Param("drive_hz", 500.0, "Hz", 0, 3000, "stick_slip: natural frequency of the springy rubbing contact; with a light contact the creak pulse rate rises with speed and falls with pressure (0 = rigid squeal)"),
           Param("mu_static", 0.6, "", 0.05, 1.5, "static friction"),
           Param("mu_dynamic", 0.3, "", 0.02, 1.2, "dynamic friction"),
           Param("stiffness", 0.0, "N/m", 0, 1e6, "bristle stiffness (0 = auto: 1e5 springy, 3000 rigid)"),
           Param("roughness", 0.5, "", 0, 1, "surface roughness (scrape/rub, and grit in stick_slip)"),
           Param("grain_mm", 0.5, "mm", 0.01, 20, "scrape/rub: bump spacing"),
           Param("loop_modes", 0, "count", 0, 12, "stick_slip: body modes solved with the friction (0 = auto: 2 springy so the creak rate follows speed and pressure, 6 rigid)"),
           Param("body", 0.6, "", 0, 2, "level of the rest of the body (higher modes, cavity)"),
           Param("residual", 1.0, "", 0, 3, "fibre / grit noise level"),
           Param("muffle", 0.0, "", 0, 1, "damping (hand on the door, carpet, grease)"),
           Param("max_t60", 3.0, "s", 0.02, 20, "cap on ring time"),
           Param("pitch_hz", 600.0, "Hz", 20, 6000, "squeal: slip rate at 0.05 m/s and 5 N (rate ~ speed^follow x pressure^-0.2)"),
           Param("pitch_follow", 0.6, "", 0, 1.5, "squeal: how strongly the slip rate follows speed (rate ~ speed^follow)"),
           Param("slip_shape", 0.15, "", 0.02, 0.5, "squeal: fraction of each cycle spent slipping (small = buzzier)"),
           Param("crackle", 0.0, "", 0, 2, "squeal: sparse fibre crackle following the slip activity (ropes)"),
       ],
       ["Dupont, Hayward, Armstrong & Altpeter 2002 (elasto-plastic friction)",
        "Avanzini, Serafin & Rocchesso 2005, IEEE TSAP (friction-induced sound); SDT: method only",
        "van den Doel, Kry & Pai 2001 (scraping); Rath & Rocchesso 2004 (rolling/scraping control)",
        "Aramaki et al. 2011 (material damping laws); Ren, Yeh & Lin 2013 (residual)",
        "Akay 2002, JASA (acoustics of friction: stick-slip, squeal, creak)"])
def friction_family(ctx: Ctx, x, mode, material, shape, size, hollowness, contact_point, at, length_s, speed,
                    speed_change, pressure, irregularity, drive_hz, mu_static, mu_dynamic, stiffness, roughness,
                    grain_mm, loop_modes, body, residual, muffle, max_t60, pitch_hz, pitch_follow, slip_shape,
                    crackle):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    V, N = _gesture(ctx, at, length_s, speed, speed_change, pressure, irregularity)
    if mode == "squeal":
        y, _ = _squeal(ctx, V, N, speed, pressure, pitch_hz, pitch_follow, slip_shape, irregularity, roughness, crackle)
        y = y / (np.abs(y).max() + 1e-12)
        if body > 0 and material != "cloth":
            m = IMPACT_MATERIALS[material]
            fm, am, _ = mode_table(shape, m, size, hollowness, contact_point, 30, rng)
            al = np.maximum(decay_rates(fm, m, muffle), 6.91 / 0.04)       # colour only: nothing rings past 40 ms
            keep = (fm > 30) & (fm < 0.4 * sr)
            if keep.any():
                ir = modal_ir(fm[keep], am[keep], al[keep], sr, 0.04, m["cap"], rng, 0.0)
                b = signal.fftconvolve(y, ir)[:n]
                y = y + body * b / (np.abs(b).max() + 1e-12)
        if muffle > 0:
            y = dsp.lowpass(y, 12000 * (1 - muffle) + 1500 * muffle, sr)
        return dsp.highpass(y, 40.0, sr)
    modeless = material == "cloth" or mode == "rub"
    if not modeless:
        m = IMPACT_MATERIALS[material]
        f, amp, _ = mode_table(shape, m, size, hollowness, contact_point, 40, rng)
        alpha = decay_rates(f, m, muffle)
        keep = (f > 30) & (f < 0.4 * sr)
        f, amp, alpha = f[keep], amp[keep], alpha[keep]
        if len(f) == 0:
            f, amp, alpha = np.array([0.3 * sr]), np.array([1.0]), np.array([200.0])
        tc = 1.0 / alpha                                           # amplitude decay time per mode
        if m["cap"]:
            tc = np.minimum(tc, m["cap"] / 6.91)
        tc = np.minimum(tc, max_t60 / 6.91)

    if mode == "stick_slip" and not modeless:
        springy = drive_hz > 0
        K = int(min(loop_modes or (2 if springy else 6), len(f)))
        modes = [[float(f[i]), float(tc[i]), float(amp[i] / amp[:K].max())] for i in range(K)]
        if springy:   # light contact on a spring: slips are quick, so the stick time (speed, pressure) sets the rate
            mass = 0.003 + 0.006 * size * (1.4 - 0.8 * hollowness)
        else:         # rigid: the rubbed body itself carries the contact
            mass = 0.02 + 0.3 * size * (1.4 - 0.8 * hollowness)
        stiffness = stiffness or (1e5 if springy else 3000.0)
        sub = ctx.child()
        y = REGISTRY["ep_friction"].fn(sub, None, velocity=_bp(V, sr), normal_force=_bp(N, sr),
                                       mu_static=mu_static, mu_dynamic=mu_dynamic, stribeck_v=0.05,
                                       stiffness=stiffness, bristle_damping=0.3, viscosity=0.2,
                                       noise=0.1 + 0.5 * roughness, break_away=0.7, mass_g=mass * 1000,
                                       drive_hz=drive_hz, modes=modes)
        y = np.nan_to_num(y)
        y = y / (np.abs(y).max() + 1e-12)
        drive = np.diff(y, prepend=0.0)                          # slip events excite the rest of the body
        drive = drive / (np.abs(drive).max() + 1e-12)
        out = y.copy()
        rest_f, rest_a, rest_al = f[K:], amp[K:], alpha[K:]
        if hollowness > 0 and shape in ("box", "shell", "ball", "plate"):
            fc = cavity_modes(size if shape not in ("shell", "ball") else size * 0.6)
            rest_f = np.concatenate([rest_f, fc])
            rest_a = np.concatenate([rest_a, hollowness / np.arange(1, len(fc) + 1)])
            rest_al = np.concatenate([rest_al, np.full(len(fc), 30.0 + 60 * muffle)])
        if body > 0 and len(rest_f):
            ir = modal_ir(rest_f, rest_a, rest_al, sr, max_t60, m["cap"], rng, 0.0 if m.get("metal") else 0.02)
            b = signal.fftconvolve(drive, ir)[:n]
            out += body * b / (np.abs(b).max() + 1e-12)
        if residual > 0 and m["resid"] > 0:
            act = np.convolve(np.abs(drive), np.ones(int(0.004 * sr)) / (0.004 * sr), "same")
            ir_all = modal_ir(f, amp, alpha, sr, 0.05, m["cap"], rng, 0.02)
            ir_all *= np.exp(-np.arange(len(ir_all)) / (0.003 * sr))
            nz = signal.fftconvolve(rng.standard_normal(n) * act, ir_all)[:n]
            out += residual * m["resid"] * nz / (np.abs(nz).max() + 1e-12)
    else:
        sub = ctx.child()
        F = REGISTRY["surface"].fn(sub, None, mode="scrape", velocity=_bp(V, sr), force=_bp(np.minimum(N / 4, 4), sr),
                                   roughness=roughness, grain_mm=grain_mm, size=min(size, 1.0) * 0.5, bounce=0.5)
        F = np.nan_to_num(F) * (N > 0)
        if modeless:
            lo, hi = RUB_BAND.get(material, (500, 6000))
            out = dsp.lowpass(dsp.highpass(F, lo, sr), hi * (1 - 0.6 * muffle), sr)
            out = out / (np.abs(out).max() + 1e-12)
        else:
            ir = modal_ir(f, amp, alpha, sr, max_t60, m["cap"], rng, 0.0 if m.get("metal") else 0.02)
            out = signal.fftconvolve(F, ir)[:n]
            out = out / (np.abs(out).max() + 1e-12)
            if residual > 0:
                g = dsp.highpass(F, 1500.0, sr)
                out += residual * m["resid"] * g / (np.abs(g).max() + 1e-12)
    if muffle > 0:
        out = dsp.lowpass(out, 16000 * (1 - muffle) + 1500 * muffle, sr)
    return dsp.highpass(out, 40.0, sr)
