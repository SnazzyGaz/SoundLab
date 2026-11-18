"""Physically informed interaction and texture models, v0.19.

Methods follow the published models collected in the Sound Design Toolkit (SDT, SkAT-VG project; GPL-3.0) and
the papers behind it. No SDT code is used: each model is re-implemented here from the method description.
- hc_contact: Hunt-Crossley non-linear contact (Hunt & Crossley 1975; Avanzini & Rocchesso 2001; Papetti et al.
  2011) with a bouncing control layer (Rath 2003).
- ep_friction: elasto-plastic friction (Dupont et al. 2002), coupled to a modal resonator, for creaks, squeaks
  and bowing (Avanzini, Serafin & Rocchesso 2005).
- crumple: crumpling and breaking control (Sethna et al. 2001 crackling noise; Houben et al.; Bresin et al. 2010
  SDT crumpling): events with power-law energy drive small modal fragments.
- surface: rolling and scraping control (van den Doel, Kry & Pai 2001; Rath & Rocchesso 2004): a fractal surface
  profile read at the contact speed, smoothed by the object's size, gives the contact force.
- wind_cavity: air flow over a pipe, bottle or doorway: modes of the cavity selected by flow speed
  (edge-tone/Strouhal locking; Farnell 2010; SDT WindCavity).
- dc_motor: electric motor (rotor partials, load unevenness, gear whine, brush and air noise, chassis comb;
  SDT DCMotor method). mains_hum: transformer / mains hum and buzz.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .. import dsp
from ..core import Ctx, Param, ctrl, model

G = 9.81


# ----------------------------------------------------------------------------- Hunt-Crossley contact
def hc_pulse(v_in, mass, k, alpha, mu, sr, sub=16, max_s=0.05):
    """Contact force of a mass hitting a stiff surface at v_in (m/s): F = k x^alpha (1 + mu xdot).
    Returns (force sampled at sr, rebound speed). Harder (larger k) or faster contacts are shorter and
    brighter; alpha > 1 makes duration fall with speed (the non-linearity a half-sine pulse lacks)."""
    dt = 1.0 / (sr * sub)
    x, v = 0.0, float(v_in)
    out = []
    acc = 0.0
    i = 0
    nmax = int(max_s * sr * sub)
    while i < nmax:
        f = k * x ** alpha * (1 + mu * v) if x > 0 else 0.0
        f = max(f, 0.0)
        v -= f / mass * dt
        x += v * dt
        acc += f
        i += 1
        if i % sub == 0:
            out.append(acc / sub)
            acc = 0.0
        if x <= 0 and v < 0 and i > sub:
            break
    if acc:
        out.append(acc / sub)
    return np.array(out), max(0.0, -v)


@model("hc_contact", "excitation",
       "Hunt-Crossley contact force of an object (`mass_g`) hitting a surface at `velocity`: F = k x^alpha "
       "(1 + damping * xdot). Faster or harder hits get shorter, brighter contacts by themselves. Optional "
       "bouncing: each rebound speed comes from the contact, times `restitution`, with `irregularity`; flight "
       "time from gravity. Feed a resonator (modal) with it.",
       [
           Param("at", 0.0, "s", 0, None),
           Param("velocity", 1.0, "m/s", 0.05, 20, "impact speed"),
           Param("mass_g", 50.0, "g", 0.5, 20000, "mass of the striking object"),
           Param("stiffness", 1e8, "N/m^alpha", 1e4, 1e12, "contact stiffness (rubber ~1e5, wood ~1e8, steel ~1e11)"),
           Param("alpha", 1.5, "", 1.0, 3.0, "contact exponent (1.5 = Hertz, spheres)"),
           Param("damping", 0.3, "s/m", 0.0, 5.0, "Hunt-Crossley dissipation (more = deader, less rebound)"),
           Param("bounces", 1, "count", 1, 60, "maximum number of contacts"),
           Param("restitution", 0.7, "", 0.0, 0.99, "extra speed kept per bounce (losses in the bodies)"),
           Param("irregularity", 0.2, "", 0.0, 1.0, "random variation of each bounce"),
       ], ["Hunt & Crossley 1975, J. Appl. Mech.", "Avanzini & Rocchesso 2001 (modelling collision sounds)",
           "Papetti, Avanzini & Rocchesso 2011, IEEE TASLP (numerical impact models)",
           "Rath 2003 (bouncing control); Sound Design Toolkit (SDT), SkAT-VG: method only"])
def hc_contact(ctx: Ctx, x, at, velocity, mass_g, stiffness, alpha, damping, bounces, restitution, irregularity):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    out = np.zeros(n)
    t = float(at)
    v = float(velocity)
    m = mass_g / 1000.0
    for _ in range(int(bounces)):
        if v < 0.02 or t >= ctx.duration:
            break
        pulse, vout = hc_pulse(v, m, stiffness, alpha, damping, sr)
        dsp.place(out, pulse, int(t * sr), 1.0)
        v = vout * restitution * (1 - irregularity * rng.random())
        t += len(pulse) / sr + 2 * v / G
    peak = np.max(np.abs(out))
    return out / (peak + 1e-12) * min(1.0, velocity)


# ----------------------------------------------------------------------------- elasto-plastic friction
@model("ep_friction", "excitation",
       "Elasto-plastic friction (bristle model) between a rubbing object and a small modal resonator, solved "
       "together so stick-slip, pre-sliding and squeal emerge: creaking hinges, squeaky shoes and floors, "
       "bowed glass and metal, tyre squeal. `velocity` (rubbing speed) and `normal_force` can be automated. "
       "`modes` [[Hz, decay s, gain], ...] are the resonator's modes. Output: the resonator's surface velocity.",
       [
           Param("velocity", 0.1, "m/s", 0.0, 3.0, "rubbing speed (automatable)", automatable=True),
           Param("normal_force", 1.0, "N", 0.0, 30.0, "pressing force (automatable)", automatable=True),
           Param("mu_static", 0.6, "", 0.05, 1.5, "static friction coefficient"),
           Param("mu_dynamic", 0.3, "", 0.02, 1.2, "dynamic (Coulomb) friction coefficient"),
           Param("stribeck_v", 0.05, "m/s", 0.001, 1.0, "Stribeck velocity"),
           Param("stiffness", 2000.0, "N/m", 100, 1e6, "bristle stiffness sigma0"),
           Param("bristle_damping", 0.3, "", 0.0, 2.0, "bristle damping sigma1 (re critical)"),
           Param("viscosity", 0.2, "N s/m", 0.0, 5.0, "viscous friction sigma2"),
           Param("noise", 0.2, "", 0.0, 1.0, "surface roughness noise (sigma3, re normal force)"),
           Param("break_away", 0.7, "", 0.05, 0.99, "pre-sliding zone as a fraction of the steady bristle bend"),
           Param("mass_g", 30.0, "g", 1, 5000, "effective mass of the resonator at the contact"),
           Param("drive_hz", 0.0, "Hz", 0.0, 200.0, "compliance of the rubbing object: 0 = rigid (squeal, bowing); "
                 "10-60 Hz = a springy drive that sticks, winds up and slips (creaking, ticking)"),
           Param("modes", [[260.0, 0.25, 1.0], [610.0, 0.18, 0.6], [1180.0, 0.12, 0.4]],
                 "[[Hz, decay s, gain], ...]", doc="resonator modes", is_list=True),
       ], ["Dupont, Hayward, Armstrong & Altpeter 2002, IEEE TAC (elasto-plastic friction)",
           "Avanzini, Serafin & Rocchesso 2005, IEEE TSAP (interactive simulation of rigid body interaction with "
           "friction-induced sound generation)", "Sound Design Toolkit (SDT), SkAT-VG: method only"])
def ep_friction(ctx: Ctx, x, velocity, normal_force, mu_static, mu_dynamic, stribeck_v, stiffness, bristle_damping,
                viscosity, noise, break_away, mass_g, drive_hz, modes):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    sub = 2
    dt = 1.0 / (sr * sub)
    VB = ctrl(ctx, velocity)
    NF = ctrl(ctx, normal_force)
    W = rng.standard_normal(n)
    m = mass_g / 1000.0
    # v0.25: modes at or above ~Nyquist make the explicit integrator blow up (a creak fitted at 48 kHz rendered
    # at 16 kHz gave NaNs); they are inaudible at that rate anyway, so drop them (keep at least the lowest)
    modes = [m for m in modes if float(m[0]) < 0.45 * ctx.sr] or sorted(modes, key=lambda m: float(m[0]))[:1]
    md = [(2 * math.pi * float(f), 1.0 / max(1e-3, float(d)), float(gn)) for f, d, gn in modes]
    K = len(md)
    w2 = [a * a for a, _, _ in md]
    dmp = [2 * b for _, b, _ in md]  # x'' + 2/T x' ... decay rate 1/T on amplitude
    gk = [c for _, _, c in md]
    xs = [0.0] * K
    vs = [0.0] * K
    z = 0.0
    s0 = float(stiffness)
    s1 = float(bristle_damping) * 2 * math.sqrt(s0 * m)
    s2 = float(viscosity)
    out = np.empty(n)
    vs_ = float(stribeck_v)
    springy = drive_hz > 0
    kd = m * (2 * math.pi * drive_hz) ** 2
    cd = 2 * 0.05 * math.sqrt(kd * m)
    xdrv = xrub = vrub = 0.0
    for i in range(n):
        vb = VB[i]
        N = max(NF[i], 0.0)
        fc, fs = mu_dynamic * N, mu_static * N
        nz = noise * N * W[i] * 0.1 * min(1.0, abs(vb) / 0.002)   # v0.27.1: no grit once the motion stops
        for _ in range(sub):
            vr = 0.0
            for k in range(K):
                vr += gk[k] * vs[k]
            if springy:
                xdrv += vb * dt
                v = vrub - vr
            else:
                v = vb - vr
            if N <= 0:
                F, z = 0.0, 0.0
            else:
                zss = (fc + (fs - fc) * math.exp(-(v / vs_) ** 2)) / s0
                zss = zss if v >= 0 else -zss
                zba = break_away * abs(zss)
                az = abs(z)
                if v * z <= 0 or az <= zba:
                    al = 0.0
                elif az >= abs(zss):
                    al = 1.0
                else:
                    al = 0.5 * (1 + math.sin(math.pi * ((az - 0.5 * (abs(zss) + zba)) / (abs(zss) - zba + 1e-15))))
                # bristle update, implicit in the relaxation term: z' = v (1 - al z / zss). With stiff
                # bristles |v| / zss gets large (sigma0 = 1e6: ~3e5 /s) and explicit Euler blew up; here the
                # denominator is >= 1 (zss has the sign of v), so it is stable for any stiffness and equal to
                # the explicit step when al = 0.
                zn = (z + v * dt) / (1.0 + dt * al * v / (zss if zss != 0 else 1e-15))
                zd = (zn - z) / dt
                z = zn
                F = s0 * z + s1 * zd + s2 * v + nz
            for k in range(K):
                a = -w2[k] * xs[k] - dmp[k] * vs[k] + F * gk[k] / m
                vs[k] += a * dt
                xs[k] += vs[k] * dt
            if springy:
                vrub += (kd * (xdrv - xrub) - cd * vrub - F) / m * dt
                xrub += vrub * dt
        vr = 0.0
        for k in range(K):
            vr += gk[k] * vs[k]
        out[i] = vr
    return out


# ----------------------------------------------------------------------------- crumpling / breaking
@model("crumple", "sound",
       "Crumpling and breaking as many tiny impacts (crackling noise). `mode` crumple: events at `rate` while "
       "the gesture lasts (paper, plastic, foil, cloth, leaves, snow, gravel underfoot). `mode` break: a stored "
       "energy is spent in bursts that come fast at first and thin out as the object falls apart (glass, ice, "
       "pottery, wood splintering). Each event's energy follows a power law; its fragment size sets the pitch "
       "(small pieces ring higher and shorter) through a modal grain with `ratios` and `decay_ms`.",
       [
           Param("mode", "crumple", "", choices=("crumple", "break")),
           Param("at", 0.0, "s", 0, None),
           Param("length_s", 1.0, "s", 0.05, 30, "gesture length (crumple)"),
           Param("rate", 300.0, "events/s", 1, 20000, "event rate (crumple) or initial rate (break)"),
           Param("energy", 0.5, "", 0.0, 1.0, "crushing energy per event"),
           Param("fragmentation", 0.5, "", 0.0, 1.0, "spread of fragment sizes"),
           Param("freq_hz", 1500.0, "Hz", 50, 15000, "ring of a unit-size fragment"),
           Param("ratios", [1.0, 2.32, 4.25, 6.63], "", doc="mode ratios of a fragment", is_list=True),
           Param("decay_ms", 8.0, "ms", 0.2, 2000, "decay of a unit-size fragment"),
           Param("noise", 0.5, "", 0.0, 1.0, "share of a broadband click in each event (paper is noisy, glass not)"),
           Param("gesture", 0.5, "", 0.0, 1.0, "crumple: how bursty the hand's squeezes are"),
       ], ["Sethna, Dahmen & Myers 2001, Nature (crackling noise)", "Houben, Kohlrausch & Hermes 2004 (auditory "
           "cues for size and speed of rolling/crumpling)", "Bresin et al. 2010 (SDT crumpling, walking on "
           "surfaces)", "Sound Design Toolkit (SDT), SkAT-VG: method only"])
def crumple(ctx: Ctx, x, mode, at, length_s, rate, energy, fragmentation, freq_hz, ratios, decay_ms, noise, gesture):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    out = np.zeros(n)
    t0 = int(at * sr)
    if mode == "crumple":
        L = min(n - t0, int(length_s * sr))
        if L <= 0:
            return out
        tt = np.arange(L) / sr
        g = np.ones(L)
        if gesture > 0:  # squeezes: smooth random envelope at ~3 Hz, raised to sharpen bursts
            kt = np.arange(0, length_s + 0.4, 0.33)
            sq = np.abs(np.interp(tt, kt, rng.standard_normal(len(kt))))
            g = (1 - gesture) + gesture * 2.5 * sq ** 2
        g *= np.minimum(1, tt / 0.03) * np.minimum(1, (length_s - tt) / 0.05).clip(0)
        idx = t0 + np.nonzero(rng.random(L) < np.clip(rate * g / sr, 0, 1))[0]
        sizes = (1 - fragmentation + fragmentation * rng.random(len(idx))) * (0.5 + 0.5 * rng.random(len(idx)))
        rem = np.ones(len(idx))
    else:
        idx, sizes, rem = [], [], []
        remaining, t = 1.0, t0
        while remaining > 0.02 and t < n:
            t += int(rng.exponential(sr / max(rate * remaining, 1e-3))) + 1
            frag = 1 - fragmentation + fragmentation * remaining
            idx.append(t)
            sizes.append(max(0.05, frag * (0.5 + 0.5 * rng.random())))
            rem.append(remaining)
            remaining -= energy * 0.02 * min(4.0, (1 - rng.random()) ** (-1 / 1.45))
        idx, sizes, rem = np.array(idx, int), np.array(sizes), np.array(rem)
    E = energy * np.clip((1 - rng.random(len(idx))) ** (-1 / 1.45), 0.2, 10) / 3
    for i, s, e, r in zip(idx, sizes, E, rem):
        if i >= n:
            break
        s = max(float(s), 0.05)
        f = freq_hz / s
        T = decay_ms * 1e-3 * s
        Lg = min(int(T * 7 * sr) + 32, sr)
        tg = np.arange(Lg) / sr
        gr = np.zeros(Lg)
        for k, ra in enumerate(ratios):
            fk = f * ra * (1 + 0.03 * rng.standard_normal())
            if fk < 0.45 * sr:
                gr += np.sin(2 * np.pi * fk * tg + rng.uniform(0, 6.28)) * np.exp(-tg / (T / (1 + 0.5 * k))) / (k + 1)
        if noise > 0:
            cl = rng.standard_normal(min(Lg, int(0.0015 * sr) + 4)) * np.exp(-np.arange(min(Lg, int(0.0015 * sr) + 4)) / (0.0003 * sr))
            gr[:len(cl)] += noise * 3 * cl
        dsp.place(out, gr, int(i), math.sqrt(e) * s * (0.5 + 0.5 * r))
    return out


# ----------------------------------------------------------------------------- rolling / scraping
def fractal_profile(rng, m, beta):
    """1/f^beta random surface profile of m samples, unit variance."""
    X = np.fft.rfft(rng.standard_normal(m))
    f = np.arange(len(X), dtype=float)
    f[0] = 1
    y = np.fft.irfft(X / f ** (beta / 2), m)
    return (y - y.mean()) / (y.std() + 1e-12)


@model("surface", "excitation",
       "Contact force of an object rolling or scraping over a rough surface. The surface is a fractal profile "
       "(`roughness`: 0 smooth, 1 coarse) read at the contact `velocity` (automatable); the object's `size` stops "
       "it following narrow dips (the ground trace). Scrape: force from every rise, growing with speed. Roll: the "
       "object also flies off bumps under gravity and lands again, which gives rolling its rumble and rhythm. "
       "Feed a modal resonator (the scraped plate, the rolling ball or the floor).",
       [
           Param("mode", "scrape", "", choices=("scrape", "roll")),
           Param("velocity", 0.3, "m/s", 0.0, 10.0, "contact speed (automatable)", automatable=True),
           Param("force", 1.0, "", 0.0, 4.0, "pressing force / weight (automatable)", automatable=True),
           Param("roughness", 0.5, "", 0.0, 1.0, "surface coarseness (0 = polished, 1 = gravel)"),
           Param("grain_mm", 0.5, "mm", 0.01, 20, "typical bump spacing"),
           Param("size", 0.3, "", 0.0, 1.0, "object size: bigger objects bridge more dips (duller)"),
           Param("bounce", 0.5, "", 0.0, 1.0, "roll: how easily the object leaves the surface"),
       ], ["van den Doel, Kry & Pai 2001, SIGGRAPH (FoleyAutomatic: scraping and rolling)",
           "Rath & Rocchesso 2004 (continuous sonic feedback from a rolling ball)",
           "Sound Design Toolkit (SDT), SkAT-VG: method only"])
def surface(ctx: Ctx, x, mode, velocity, force, roughness, grain_mm, size, bounce):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    V = np.maximum(ctrl(ctx, velocity), 0)
    Fn = ctrl(ctx, force)
    pos = np.cumsum(V) / sr  # metres travelled
    dx = grain_mm * 1e-3 / 4
    m = int(pos[-1] / dx) + 8
    beta = 2.5 - 2.0 * roughness  # rough = flatter spectrum
    prof = fractal_profile(rng, max(m, 64), beta)
    h = np.interp(pos / dx, np.arange(len(prof)), prof)
    dec = (0.02 + 0.5 * size ** 2) * np.maximum(V, 1e-4) / max(dx * sr, 1e-12) * 0.05
    out = np.zeros(n)
    trace = h[0]
    flight = 0.0
    grav = 0.002 * (1 - 0.8 * bounce)
    roll = mode == "roll"
    for i in range(n):
        cur = max(trace - dec[i], h[i])
        if cur > trace and (not roll or flight <= 0):
            bump = (cur - trace)
            if roll:
                flight = bump * 4.0 * (0.2 + bounce)
                out[i] = bump * Fn[i] - grav
            else:
                out[i] = bump * Fn[i] * V[i] * V[i]
        elif roll:
            out[i] = -grav * 0.0
        trace = cur
        if roll:
            flight = max(0.0, flight - grav)
    out = out - np.mean(out)
    return out / (np.max(np.abs(out)) + 1e-12)


# ----------------------------------------------------------------------------- wind through a cavity
@model("wind_cavity", "excitation",
       "Wind blowing across a pipe, bottle neck, doorway gap or tunnel: turbulence excites the cavity's modes "
       "(open pipe f_n = n c / 2L), and the flow speed picks which mode sings (edge tone f = St U / D locks to the "
       "nearest mode, so the howl jumps up a harmonic as the wind rises). Plus a broadband turbulence floor.",
       [
           Param("length", 0.5, "m", 0.02, 50, "cavity length"),
           Param("diameter", 0.03, "m", 0.002, 5, "opening size (sets which speed excites which mode)"),
           Param("wind_speed", 8.0, "m/s", 0.0, 60, "flow speed (automatable)", automatable=True),
           Param("q", 40.0, "", 3, 400, "mode sharpness"),
           Param("n_modes", 10, "count", 1, 40, "modes simulated"),
           Param("strouhal", 0.3, "", 0.05, 1.0, "edge-tone Strouhal number"),
           Param("floor", 0.1, "", 0.0, 1.0, "broadband turbulence level"),
           Param("closed", False, "", doc="closed at one end (bottle): odd modes f = (2n-1) c / 4L"),
       ], ["Farnell 2010, Designing Sound (wind, pipes)", "Howe 1998, Acoustics of fluid-structure interactions "
           "(edge tones)", "Sound Design Toolkit (SDT) WindCavity, SkAT-VG: method only"])
def wind_cavity(ctx: Ctx, x, length, diameter, wind_speed, q, n_modes, strouhal, floor, closed):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    U = np.maximum(ctrl(ctx, wind_speed), 0)
    w = rng.standard_normal(n)
    out = np.zeros(n)
    fe = strouhal * U / max(diameter, 1e-4)
    for k in range(1, int(n_modes) + 1):
        fk = ((2 * k - 1) * ctx.c / (4 * length)) if closed else (k * ctx.c / (2 * length))
        if fk > 0.45 * sr:
            break
        weight = np.exp(-((np.log2(np.maximum(fe, 1.0) / fk)) / 0.35) ** 2)
        if weight.max() < 1e-3:
            continue
        b, a = signal.iirpeak(fk, q, fs=sr)
        out += signal.lfilter(b, a, w) * weight
    loud = (U / 10.0) ** 2
    turb = dsp.bandpass(w, 800.0, sr, 0.5) * floor
    return (out * math.sqrt(q / 10.0) + turb) * loud


# ----------------------------------------------------------------------------- electric motor and hum
@model("dc_motor", "sound",
       "Electric motor: rotor partials at the rotation rate times the number of coils (whine), unevenness from "
       "the load (FM), gear whine at rpm x teeth, brush crackle at the commutation rate, fan/air noise rising "
       "with speed, and a chassis resonance (comb). Drills, fans, servos, RC cars, electric scooters, toys.",
       [
           Param("rpm", 6000.0, "rpm", 0, 60000, "rotor speed (automatable)", automatable=True),
           Param("load", 0.2, "", 0.0, 1.0, "load unevenness (wobble)"),
           Param("coils", 3, "count", 1, 24, "commutator segments / poles"),
           Param("gear_teeth", 12, "count", 0, 200, "gear mesh teeth per rotor turn (0 = no gearbox)"),
           Param("harshness", 0.5, "", 0.0, 1.0, "brighter, buzzier partials"),
           Param("rotor", 1.0, "", 0.0, 2.0, "rotor whine level"),
           Param("gear", 0.5, "", 0.0, 2.0, "gear whine level"),
           Param("brush", 0.4, "", 0.0, 2.0, "brush crackle level"),
           Param("air", 0.3, "", 0.0, 2.0, "fan / air noise level"),
           Param("chassis_hz", 900.0, "Hz", 50, 8000, "chassis resonance"),
       ], ["Sound Design Toolkit (SDT) DCMotor, SkAT-VG: method only", "Farnell 2010, Designing Sound (motors)"])
def dc_motor(ctx: Ctx, x, rpm, load, coils, gear_teeth, harshness, rotor, gear, brush, air, chassis_hz):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    fr = np.maximum(ctrl(ctx, rpm), 0) / 60.0
    wob = 1 + 0.02 * load * np.sin(2 * np.pi * np.cumsum(fr) / sr)
    ph = 2 * np.pi * np.cumsum(fr * wob) / sr
    y = np.zeros(n)
    base = coils
    for k in range(1, 25):
        fk = fr * base * k
        a = (k ** (-2.0 + 1.5 * harshness)) * (fk < 0.45 * sr)
        y += rotor * a * np.sin(base * k * ph)
    y += rotor * 0.3 * np.sin(ph)  # imbalance at the rotation rate
    if gear_teeth > 0:
        for k in (1, 2, 3):
            y += gear * 0.5 / k * np.sin(gear_teeth * k * ph + 0.3 * load * np.sin(ph)) * (fr * gear_teeth * k < 0.45 * sr)
    comm = (np.cos(base * ph) > 0.8).astype(float)
    y += brush * 0.5 * rng.standard_normal(n) * comm * np.clip(fr / 50, 0, 1)
    nz = dsp.bandpass(rng.standard_normal(n), 1500.0, sr, 0.4)
    y += air * nz * np.clip(fr / 100, 0, 3)
    d = max(2, int(sr / chassis_hz))
    yc = y.copy()
    yc[d:] += 0.6 * y[:-d]
    return yc


@model("mains_hum", "sound",
       "Electrical hum: mains frequency and harmonics, the 2x-mains magnetostriction buzz of transformers and "
       "lamp ballasts, and an optional rattle (loose laminations, a buzzing fluorescent tube).",
       [
           Param("mains_hz", 50.0, "Hz", 40, 70, "50 (UK/EU) or 60 (US)"),
           Param("buzz", 0.5, "", 0.0, 1.0, "harmonic richness (clipped, rectified hum)"),
           Param("rattle", 0.0, "", 0.0, 1.0, "loose-part rattle"),
           Param("level", 1.0, "", 0.0, 4.0, "level", automatable=True),
       ], ["Mains hum (power engineering): transformer magnetostriction at twice the mains frequency"])
def mains_hum(ctx: Ctx, x, mains_hz, buzz, rattle, level):
    tt = ctx.t
    f2 = 2 * mains_hz
    y = 0.4 * np.sin(2 * np.pi * mains_hz * tt) + np.sin(2 * np.pi * f2 * tt)
    hb = np.abs(np.sin(2 * np.pi * mains_hz * tt))
    y += buzz * 2 * np.tanh(4 * buzz * (hb - hb.mean()))
    if rattle > 0:
        gate = (np.sin(2 * np.pi * f2 * tt) > 0.7).astype(float)
        y += 0.4 * rattle * gate * dsp.lowpass(dsp.highpass(ctx.rng.standard_normal(ctx.n), 800, ctx.sr), 4000, ctx.sr)
    return y * ctrl(ctx, level)
