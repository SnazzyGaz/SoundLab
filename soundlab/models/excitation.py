"""Stage 1: excitations (what puts energy in)."""
from __future__ import annotations

import math

import numpy as np

from .. import dsp
from ..core import Ctx, Param, ctrl, model, scalar


def friedlander(n: int, sr: int, t_star: float, peak: float = 1.0) -> np.ndarray:
    """Friedlander blast waveform P(t) = Ps e^(-t/t*) (1 - t/t*) (Wikipedia, Blast wave)."""
    t = np.arange(n) / sr
    p = peak * np.exp(-t / t_star) * (1 - t / t_star)
    p[t > 8 * t_star] = 0.0
    # sub-sample rise: one sample of ramp avoids an aliasing step
    if n > 1:
        p[0] *= 0.5
    return p


@model("impact", "excitation",
       "Contact force of a hit (half-sine pulse, Hertz-like). Shorter contact = harder hammer = brighter. "
       "Optional bounces with a coefficient of restitution.",
       [
           Param("at", 0.0, "s", 0, None, "time of first contact"),
           Param("velocity", 1.0, "m/s", 0.05, 20, "impact speed; force scales with it"),
           Param("contact_ms", 1.0, "ms", 0.05, 20, "contact duration; metal hammer ~0.1-0.5, felt mallet ~2-10 (estimates)"),
           Param("bounces", 1, "count", 1, 40, "number of contacts (1 = single hit)"),
           Param("restitution", 0.6, "", 0.0, 0.98, "speed ratio after each bounce (gaps and levels shrink by this)"),
           Param("first_gap", 0.25, "s", 0.01, 2.0, "time between first and second contact"),
       ],
       ["van den Doel, Kry & Pai 2001 (FoleyAutomatic)", "Cook 2002, Real Sound Synthesis"])
def impact(ctx: Ctx, x, at, velocity, contact_ms, bounces, restitution, first_gap):
    out = np.zeros(ctx.n)
    t0, gap, v = float(at), float(first_gap), float(velocity)
    for _ in range(int(bounces)):
        tc = max(contact_ms * 1e-3 * (v / velocity) ** -0.2, 2.0 / ctx.sr)  # faster hit -> slightly shorter contact
        m = max(int(tc * ctx.sr), 2)
        pulse = np.sin(np.pi * np.arange(m) / m) ** 1.5
        dsp.place(out, pulse * v, int(t0 * ctx.sr))
        t0 += gap
        gap *= restitution
        v *= restitution
        if v < 0.01 * velocity or gap < 1e-3:
            break
    return out


@model("friction", "excitation",
       "Stick-slip friction: a mass-spring held by static friction against a moving surface, released "
       "when the spring force beats static friction (relaxation oscillation). Output is the friction force.",
       [
           Param("velocity", 0.3, "m/s", 0.0, 3.0, "sliding (belt) speed", automatable=True),
           Param("normal_force", 2.0, "N", 0.0, 30.0, "pressing force", automatable=True),
           Param("mu_static", 0.6, "", 0.05, 1.5, "static friction coefficient"),
           Param("mu_kinetic", 0.35, "", 0.02, 1.2, "kinetic friction coefficient (lower than static gives stick-slip)"),
           Param("stribeck_v", 0.05, "m/s", 0.001, 1.0, "speed over which friction falls from static to kinetic"),
           Param("f_osc", 900.0, "Hz", 50, 5000, "natural frequency of the contact mass-spring"),
           Param("zeta", 0.05, "", 0.001, 0.5, "damping ratio of the contact oscillator"),
           Param("roughness", 0.4, "", 0.0, 1.0, "surface roughness: random modulation of normal force"),
           Param("roughness_hz", 2500.0, "Hz", 50, 12000, "roughness bandwidth (grain of the surface at this speed)"),
       ],
       ["Wikipedia: Stick-slip phenomenon", "Avanzini, Serafin & Rocchesso, DAFx 2002 (friction models)"])
def friction(ctx: Ctx, x, velocity, normal_force, mu_static, mu_kinetic, stribeck_v, f_osc, zeta, roughness,
             roughness_hz):
    sr = ctx.sr
    vb = ctrl(ctx, velocity)
    N = ctrl(ctx, normal_force)
    if roughness > 0:
        r = dsp.lowpass(ctx.rng.standard_normal(ctx.n), roughness_hz, sr)
        r /= np.std(r) + 1e-12
        N = np.maximum(N * (1 + roughness * 0.5 * r), 0)
    m = 1e-3
    w = 2 * np.pi * f_osc
    k = m * w * w
    c = 2 * zeta * m * w
    dt = 1.0 / sr
    xpos = 0.0
    v = 0.0
    stuck = True
    out = np.zeros(ctx.n)
    vbl, Nl = vb.tolist(), N.tolist()
    mus, muk, vs = float(mu_static), float(mu_kinetic), float(stribeck_v)
    for i in range(ctx.n):
        vbi, Ni = vbl[i], Nl[i]
        if stuck:
            v = vbi
            xpos += v * dt
            fs = -k * xpos - c * v
            ff = -fs
            if abs(fs) > mus * Ni:
                stuck = False
        else:
            fs = -k * xpos - c * v
            vr = v - vbi
            sgn = 1.0 if vr > 0 else -1.0
            ff = -sgn * Ni * (muk + (mus - muk) * math.exp(-abs(vr) / vs))
            v_new = v + (fs + ff) / m * dt
            if (v_new - vbi) * vr <= 0 and abs(-k * xpos - c * vbi) <= mus * Ni:
                stuck = True
                v_new = vbi
            v = v_new
            xpos += v * dt
        out[i] = ff
    out = dsp.highpass(out, 20.0, sr)
    return out


@model("noise", "excitation", "Coloured noise burst with a band and an envelope.",
       [
           Param("color", "white", "", choices=("white", "pink", "brown")),
           Param("f_lo", 20.0, "Hz", 10, 20000, "high-pass corner"),
           Param("f_hi", 16000.0, "Hz", 20, 22000, "low-pass corner"),
           Param("at", 0.0, "s", 0, None, "start time"),
           Param("attack", 0.005, "s", 0.0001, 10, "attack time"),
           Param("hold", 1.0, "s", 0, 60, "time at full level"),
           Param("release", 0.2, "s", 0.001, 30, "T60 of the release"),
           Param("level", 1.0, "", 0, 4, "linear level", automatable=True),
       ])
def noise(ctx: Ctx, x, color, f_lo, f_hi, at, attack, hold, release, level):
    y = dsp.noise(ctx.rng, ctx.n, color)
    if f_lo > 15:
        y = dsp.highpass(y, f_lo, ctx.sr)
    if f_hi < ctx.sr * 0.45:
        y = dsp.lowpass(y, f_hi, ctx.sr)
    t = ctx.t - at
    env = np.clip(t / attack, 0, 1) * np.where(t < attack + hold, 1.0, np.exp(-6.91 * (t - attack - hold) / release))
    env[t < 0] = 0
    return y * env * ctrl(ctx, level) * 0.3


@model("turbulence", "excitation",
       "Airflow noise: broadband jet hiss plus an Aeolian (vortex-shedding) tone f = St U / D with St ~ 0.2.",
       [
           Param("flow_speed", 10.0, "m/s", 0, 80, "air speed past the obstacle", automatable=True),
           Param("obstacle_d", 0.005, "m", 0.0005, 0.5, "diameter of wire/edge (sets the Aeolian tone)"),
           Param("strouhal", 0.2, "", 0.1, 0.3, "Strouhal number (about 0.2 over a wide range)",
                 source="Wikipedia: Aeolian sound"),
           Param("tone", 0.3, "", 0, 1, "Aeolian tone amount vs broadband hiss"),
           Param("tone_q", 12.0, "", 1, 60, "sharpness of the Aeolian tone"),
           Param("speed_exponent", 3.0, "", 1, 4, "amplitude ~ U^n (n=3 matches dipole power ~U^6; standard aeroacoustics, "
                 "not from our fetched sources)"),
       ], ["Wikipedia: Aeolian sound", "Farnell 2010 (wind)"])
def turbulence(ctx: Ctx, x, flow_speed, obstacle_d, strouhal, tone, tone_q, speed_exponent):
    U = ctrl(ctx, flow_speed)
    amp = (np.maximum(U, 0) / 10.0) ** speed_exponent
    w = ctx.rng.standard_normal(ctx.n)
    hiss = dsp.svf(w, np.clip(U * 300, 200, 16000), 0.5, ctx.sr, "lp")
    f_tone = strouhal * U / obstacle_d
    whistle = dsp.svf(ctx.rng.standard_normal(ctx.n), f_tone, tone_q, ctx.sr, "bp") * np.sqrt(tone_q)
    return 0.2 * amp * ((1 - tone) * hiss + tone * whistle)


def _phase(ctx: Ctx, f: np.ndarray, jitter: float) -> np.ndarray:
    if jitter > 0:
        j = dsp.lowpass(ctx.rng.standard_normal(ctx.n), 40.0, ctx.sr)
        j /= np.std(j) + 1e-12
        f = f * (1 + jitter * j)
    return np.cumsum(f) / ctx.sr


@model("valve_pulse", "excitation",
       "Pressure-controlled valve (vocal folds, bird syrinx, reed): Rosenberg-style flow pulse per cycle, "
       "output as its derivative (the radiated excitation). Jitter, shimmer and period-doubling give rasp.",
       [
           Param("f0", 300.0, "Hz", 30, 2000, "fundamental (valve rate)", automatable=True),
           Param("level", 1.0, "", 0, 4, "amplitude (follows lung/air pressure)", automatable=True),
           Param("open_quotient", 0.6, "", 0.2, 0.95, "fraction of each cycle the valve is open"),
           Param("speed_quotient", 2.5, "", 1.0, 5.0, "opening time / closing time (higher = sharper closure, brighter)"),
           Param("jitter", 0.01, "", 0, 0.1, "random cycle-to-cycle pitch variation (fraction)"),
           Param("shimmer", 0.1, "", 0, 0.8, "random cycle-to-cycle amplitude variation"),
           Param("rasp", 0.0, "", 0, 1, "period doubling + turbulent noise in the pulses"),
           Param("aspiration", 0.05, "", 0, 1, "breath noise gated by the open valve"),
       ], ["Rosenberg glottal pulse (speech science)", "Ducks Unlimited duck-calling study (rasp/entropy)"])
def valve_pulse(ctx: Ctx, x, f0, level, open_quotient, speed_quotient, jitter, shimmer, rasp, aspiration):
    f = ctrl(ctx, f0, "exp")
    ph = _phase(ctx, f, jitter)
    cyc = np.floor(ph).astype(int)
    u = ph - cyc
    tp = open_quotient * speed_quotient / (1 + speed_quotient)
    tn = open_quotient - tp
    flow = np.where(u < tp, 0.5 * (1 - np.cos(np.pi * u / tp)),
                    np.where(u < tp + tn, np.cos(0.5 * np.pi * (u - tp) / max(tn, 1e-4)), 0.0))
    ncyc = cyc.max() + 2
    camp = 1 + shimmer * ctx.rng.standard_normal(ncyc) * 0.5
    if rasp > 0:
        camp *= np.where(np.arange(ncyc) % 2 == 0, 1.0, 1.0 - 0.6 * rasp)  # period doubling (subharmonic)
        camp *= 1 + rasp * 0.5 * ctx.rng.standard_normal(ncyc)
    camp = np.maximum(camp, 0.05)
    flow = flow * camp[cyc]
    dflow = np.diff(flow, prepend=0.0) * ctx.sr / np.maximum(f, 1.0) * 0.15
    asp = ctx.rng.standard_normal(ctx.n) * flow * (aspiration + 0.6 * rasp) * 0.5
    asp = dsp.highpass(asp, 800.0, ctx.sr)
    return (dflow + asp) * ctrl(ctx, level)


@model("chopper", "excitation",
       "Air chopper (diaphone piston or siren disc): compressed air gated open/closed at a rate, "
       "giving a strong, harmonic-rich pressure wave. Output is the derivative of the gated flow.",
       [
           Param("freq", 250.0, "Hz", 20, 2000, "chopping rate (piston frequency)", automatable=True),
           Param("pressure", 1.0, "", 0, 2, "supply pressure, scaled (1 = nominal, e.g. 35-40 psi for a Type F diaphone)",
                 automatable=True),
           Param("open_fraction", 0.45, "", 0.1, 0.9, "fraction of each cycle the ports are open"),
           Param("edge", 0.15, "", 0.01, 0.5, "softness of port opening (smaller = brighter)"),
           Param("jitter", 0.003, "", 0, 0.05, "rate wobble"),
           Param("flow_noise", 0.08, "", 0, 1, "turbulent hiss through the ports"),
       ], ["Terry Pepper, Seeing The Light: the diaphone", "Wikipedia: Foghorn"])
def chopper(ctx: Ctx, x, freq, pressure, open_fraction, edge, jitter, flow_noise):
    f = ctrl(ctx, freq, "exp")
    P = ctrl(ctx, pressure)
    ph = _phase(ctx, f, jitter) % 1.0
    # smooth trapezoid gate
    e = edge * open_fraction
    g = np.clip(ph / e, 0, 1) * np.clip((open_fraction - ph) / e, 0, 1)
    g = 0.5 - 0.5 * np.cos(np.pi * g)
    flow = g * np.sqrt(np.maximum(P, 0))  # orifice flow ~ sqrt(pressure)
    turb = ctx.rng.standard_normal(ctx.n) * g * flow_noise
    d = np.diff(flow, prepend=0.0) * ctx.sr / np.maximum(f, 1.0) * 0.1
    return d + dsp.highpass(turb, 500.0, ctx.sr) * P


@model("combustion", "excitation",
       "Pressure release: Friedlander blast pulse (sharp rise, exponential decay, negative phase) plus a "
       "turbulent noise burst.",
       [
           Param("at", 0.0, "s", 0, None, "time of the event"),
           Param("t_star_ms", 5.0, "ms", 0.2, 200, "positive-phase duration t* (bigger charge or further away = longer)"),
           Param("peak", 1.0, "", 0, 4, "peak overpressure, scaled"),
           Param("noise_mix", 0.4, "", 0, 1, "turbulent burst amount"),
           Param("noise_t60", 0.15, "s", 0.005, 5, "turbulent burst decay"),
       ], ["Wikipedia: Blast wave (Friedlander waveform)", "Farnell 2010 practical 31"])
def combustion(ctx: Ctx, x, at, t_star_ms, peak, noise_mix, noise_t60):
    out = np.zeros(ctx.n)
    n0 = int(at * ctx.sr)
    m = ctx.n - n0
    if m <= 0:
        return out
    p = friedlander(m, ctx.sr, t_star_ms * 1e-3, peak)
    nz = ctx.rng.standard_normal(m) * dsp.ar_env(m, ctx.sr, 0.001, noise_t60) * peak * 0.5
    out[n0:] = (1 - noise_mix) * p + noise_mix * nz
    return out


@model("crackle", "excitation",
       "Random short grains (Poisson 'Dust' events) with power-law sizes: fire crackle, debris, sputter, "
       "crumpling, granular compaction.",
       [
           Param("rate", 20.0, "events/s", 0, 5000, "mean event rate", automatable=True),
           Param("grain_ms", 3.0, "ms", 0.1, 200, "typical grain length"),
           Param("f_lo", 800.0, "Hz", 20, 20000, "lowest grain centre frequency"),
           Param("f_hi", 6000.0, "Hz", 50, 22000, "highest grain centre frequency"),
           Param("q", 2.0, "", 0.3, 50, "grain resonance (higher = more tonal clicks)"),
           Param("size_exponent", 2.0, "", 1.1, 4, "power-law exponent of grain sizes (lower = more big events)"),
           Param("level", 1.0, "", 0, 4, "level", automatable=True),
       ], ["Cook 1997 PhISEM", "Farnell 2010 practical 11 (fire)"])
def crackle(ctx: Ctx, x, rate, grain_ms, f_lo, f_hi, q, size_exponent, level):
    out = np.zeros(ctx.n)
    idx = dsp.poisson_times(ctx.rng, ctrl(ctx, rate), ctx.sr)
    lev = ctrl(ctx, level)
    for i in idx:
        L = max(int(ctx.sr * grain_ms * 1e-3 * ctx.rng.uniform(0.4, 2.0)), 8)
        fc = np.exp(ctx.rng.uniform(np.log(f_lo), np.log(f_hi)))
        size = min((1 - ctx.rng.random()) ** (-1 / (size_exponent - 1)), 30.0) / 3.0
        tt = np.arange(L) / ctx.sr
        dec = np.exp(-6.91 * tt / (L / ctx.sr))
        if q >= 4:
            g = np.sin(2 * np.pi * fc * tt + ctx.rng.uniform(0, 6.28)) * dec
        else:
            g = dsp.bandpass(ctx.rng.standard_normal(L), fc, ctx.sr, q) * dec * 2
        dsp.place(out, g, i, size * lev[i])
    return out * 0.3


@model("bubbles", "excitation",
       "Bubble population: each bubble is a damped sine at the Minnaert frequency f = 3.26/r (r in m), "
       "damping d = 0.13/r + 0.0072 r^-1.5, pitch rising f(t) = f0 (1 + xi d t) (van den Doel 2005).",
       [
           Param("rate", 30.0, "bubbles/s", 0, 20000, "bubble formation rate", automatable=True),
           Param("r_min_mm", 0.5, "mm", 0.05, 20, "smallest radius"),
           Param("r_max_mm", 4.0, "mm", 0.1, 50, "largest radius"),
           Param("size_exponent", 2.5, "", 1.1, 5, "power-law size distribution exponent (more small bubbles)"),
           Param("xi", 0.1, "", 0, 20, "pitch-rise constant (0.1 drops; ~10 blowing through a straw)"),
           Param("level", 1.0, "", 0, 4, "level", automatable=True),
       ], ["van den Doel 2005, ACM TAP 2(4)", "Cirio et al. 2013 (IEEE ToH)", "dougjam Bubble Sound Bank",
           "Wikipedia: Minnaert resonance", "Phillips et al. 2018, Sci Rep (dripping tap)"])
def bubbles(ctx: Ctx, x, rate, r_min_mm, r_max_mm, size_exponent, xi, level):
    out = np.zeros(ctx.n)
    idx = dsp.poisson_times(ctx.rng, ctrl(ctx, rate), ctx.sr)
    lev = ctrl(ctx, level)
    rmin, rmax = r_min_mm * 1e-3, r_max_mm * 1e-3
    a = 1 - size_exponent
    for i in idx:
        u = ctx.rng.random()
        r = ((rmax ** a - rmin ** a) * u + rmin ** a) ** (1 / a)
        bubble = bubble_grain(ctx.sr, r, xi)
        dsp.place(out, bubble, i, lev[i] * (r / 1e-3) ** 0.5)
    return out * 0.3


def bubble_grain(sr: int, r: float, xi: float = 0.1) -> np.ndarray:
    f0 = 3.26 / r
    d = 0.13 / r + 0.0072 * r ** -1.5
    L = min(int(sr * 7.0 / d), sr)
    t = np.arange(L) / sr
    f = f0 * (1 + xi * d * t)
    ph = 2 * np.pi * np.cumsum(f) / sr
    if f0 > sr * 0.45:
        return np.zeros(1)
    return np.sin(ph) * np.exp(-d * t)


@model("rough_pulses", "excitation",
       "Irregular pulse train from a vibrating valve that closes sharply (bird syrinx labia, vocal fry, rough "
       "voice). Each pulse is placed one at a time, so period jitter, period doubling (alternating long/short, "
       "strong/weak) and shimmer are true cycle-to-cycle effects, not a smooth vibrato. Rough voices sound "
       "'raspy' rather than sung: use this instead of valve_pulse when the sound should not have a clear pitch.",
       [
           Param("f0", 200.0, "Hz", 20, 2000, "mean pulse rate", automatable=True),
           Param("level", 1.0, "", 0, 4, "amplitude", automatable=True),
           Param("jitter", 0.12, "", 0, 0.5, "period variation (std, as a fraction of the period); speech ~0.01, "
                 "rough/raspy voices 0.05-0.2", source="Titze 1994 (jitter in pathological voice)"),
           Param("period_doubling", 0.4, "", 0, 1, "alternate long/short and strong/weak cycles (subharmonic)"),
           Param("shimmer", 0.4, "", 0, 1, "cycle-to-cycle amplitude variation"),
           Param("pulse_ms", 0.4, "ms", 0.05, 5, "pulse width; shorter = brighter, more 'clicky' closures"),
           Param("aspiration", 0.2, "", 0, 1, "turbulent airflow noise gated by the pulses"),
       ],
       ["Titze 1994, Principles of Voice Production", "Fletcher 1988 (bird song syrinx model)",
        "Herzel et al. 1995 (subharmonics and chaos in voice)", "Our reference analysis: mallard quacks are "
        "irregular ~150-250 Hz pulse trains through a ~2 kHz resonance (refs/SOURCES.md)"])
def rough_pulses(ctx: Ctx, x, f0, level, jitter, period_doubling, shimmer, pulse_ms, aspiration):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    F = np.maximum(ctrl(ctx, f0, "exp"), 1.0)
    A = ctrl(ctx, level)
    w = max(3, int(pulse_ms * 1e-3 * sr))
    # asymmetric flow pulse (slow opening 70%, fast closing 30%), differentiated -> strong closure spike
    up = int(w * 0.7)
    g = np.concatenate([0.5 - 0.5 * np.cos(np.pi * np.arange(up) / up),
                        0.5 + 0.5 * np.cos(np.pi * np.arange(w - up) / max(w - up, 1))])
    pulse = np.diff(g, append=0.0)
    pulse /= np.max(np.abs(pulse)) + 1e-12
    out = np.zeros(n)
    gate = np.zeros(n)
    t = float(rng.uniform(0, sr / F[0]))
    k = 0
    while t < n:
        i = int(t)
        P = sr / F[min(i, n - 1)]
        side = 1 if k % 2 else -1
        period = P * (1 + 0.3 * period_doubling * side) * (1 + jitter * rng.standard_normal())
        amp = A[min(i, n - 1)] * max(0.05, 1 + 0.5 * shimmer * rng.standard_normal()) \
            * (1 - 0.5 * period_doubling * (k % 2))
        dsp.place(out, pulse, i, amp)
        dsp.place(gate, g, i, abs(amp))
        t += max(period, 0.3 * P)
        k += 1
    asp = dsp.highpass(rng.standard_normal(n), 1000.0, sr) * dsp.lowpass(gate, 300.0, sr) * aspiration * 0.5
    return out + asp
