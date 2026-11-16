"""Recipe-library models added in v0.9 (roadmap step 4): wind, rain, glass break, sword swing, blade clash /
metal clang, door creak and slam, body hit, bullet impact, gunshot, creature growl, rustle, splash.

Physics where the mechanism decides the sound (stick-slip hinge, shard ring, Minnaert bubbles, aeolian tones,
Friedlander blast), physically informed noise otherwise. Numbers marked (est.) are our estimates. listening 
notes guide the defaults: no digital artefacts, no machine-regular repetition, no small-cavity ringing, no
tinny tops, no heavy low end on light objects.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import interpolate

from .. import dsp
from ..core import Ctx, Param, ctrl, model
from .excitation import bubble_grain, friedlander
from .foley import _conv, _hann, _mode_ir, _norm, _push
from .resonator import mode_set
from .. import materials

G = 9.81
C = 343.0


def call(name, ctx, x=None, **params):
    from ..core import REGISTRY
    m = REGISTRY[name]
    return m.fn(ctx, x, **m.resolve(params))


def smooth_random(rng, n, sr, hz, order=3):
    """Band-limited random control signal (zero mean, unit std) with features ~1/hz s long: random control
    points at 2*hz, cubic-interpolated. Robust at very low rates (a biquad at 0.1 Hz / 48 kHz is not)."""
    dur = n / sr
    k = max(4, int(dur * hz * 2) + 4)
    tp = np.linspace(-1 / hz, dur + 1 / hz, k)
    v = rng.standard_normal(k)
    y = interpolate.CubicSpline(tp, v)(np.arange(n) / sr)
    return (y - y.mean()) / (y.std() + 1e-12)


def gesture(rng, n, sr, moves, dur, at=0.0, irregular=0.3):
    """A hand/body movement envelope: `moves` bell-shaped pushes spread irregularly over `dur` seconds."""
    env = np.zeros(n)
    if moves <= 0:
        return env
    times = np.sort(at + rng.uniform(0, max(dur, 1e-3), int(moves))) if irregular else \
        at + np.linspace(0, dur, int(moves), endpoint=False)
    for t in times:
        L = int(rng.uniform(0.08, 0.35) * sr)
        dsp.place(env, _hann(L) * rng.uniform(0.4, 1.0), int(t * sr))
    return env


def damped_grain(sr, freqs, t60s, amps, L=None):
    return _mode_ir(sr, freqs, t60s, amps, max_len=L)


# =================================================================== wind
@model("wind", "sound",
       "Wind: turbulence noise whose level and brightness follow a gusting wind speed (aeroacoustic dipole power "
       "~ v^6, so level ~ v^3), slow 'howl' resonances whose pitch rides the speed (vortex shedding from wires "
       "and edges, f = St v / d, St ~ 0.2), optional whistle through a gap or keyhole (edge tone, high Q), and "
       "leaf rustle from trees (dense tiny contacts, rate ~ v^2).",
       [
           Param("speed", 8.0, "m/s", 1, 35, "mean wind speed (Beaufort 4 ~ 6-8 m/s, gale ~ 17-20 m/s)"),
           Param("gustiness", 0.5, "", 0, 1, "how much the speed varies (std as a fraction of the mean)"),
           Param("gust_hz", 0.15, "Hz", 0.02, 2, "how quickly gusts come and go"),
           Param("howl", 0.3, "", 0, 1, "level of the moaning resonances"),
           Param("whistle", 0.0, "", 0, 1, "level of a whistle through a gap/keyhole"),
           Param("gap_mm", 6.0, "mm", 1, 40, "gap size (whistle pitch = 0.2 v / gap)"),
           Param("trees", 0.4, "", 0, 1, "level of leaf rustle (trees, bushes)"),
           Param("rumble", 0.15, "", 0, 1, "low buffeting (wind on a mic or a building); keep low"),
       ], ["Farnell 2010, Designing Sound (wind)", "Wikipedia: Aeolian sound; Strouhal number",
           "Lighthill / Curle (dipole sound power ~ U^6)"])
def wind(ctx: Ctx, x, speed, gustiness, gust_hz, howl, whistle, gap_mm, trees, rumble):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    g = smooth_random(rng, n, sr, gust_hz) + 0.3 * smooth_random(rng, n, sr, gust_hz * 4)
    v = np.maximum(speed * (1 + gustiness * 0.6 * g), 0.15 * speed)
    rel = v / speed
    amp = rel ** 3 / np.max(rel ** 3)
    # body: pink noise, low-passed by a cutoff that opens with speed
    fc = np.clip(250.0 + 45.0 * v, 150, 4000)
    body = dsp.svf(dsp.pink(rng, n), fc, 0.6, sr, "lp") * amp
    out = _norm(body)
    if howl > 0:
        h = np.zeros(n)
        for k, (base, q) in enumerate(((22.0, 8.0), (37.0, 10.0), (61.0, 7.0))):
            drift = 1 + 0.04 * smooth_random(rng, n, sr, 0.3)
            h += dsp.svf(dsp.white(rng, n), np.clip(base * v * drift, 60, 3000), q, sr, "bpn") * (0.6 ** k)
        out += howl * 0.8 * _norm(h * amp ** 0.8)
    if whistle > 0:
        f_w = np.clip(0.2 * v / (gap_mm * 1e-3), 150, 3500)
        on = np.clip((rel - 0.8) / 0.4, 0, 1) ** 2  # only sounds above a threshold speed (est.)
        w = dsp.svf(dsp.white(rng, n), f_w, 40.0, sr, "bpn") + 0.3 * dsp.svf(dsp.white(rng, n), f_w * 2.02, 30.0, sr, "bpn")
        out += whistle * 0.7 * _norm(w * on * amp ** 0.5)
    if trees > 0:
        # 0.10 ('very grainy'): leaves were thousands of separate clicks per second, which read as grain.
        # A canopy is so many leaves that the sum is continuous: filtered noise whose level flutters smoothly
        # (leaf groups moving, ~5-15 Hz) and swells with the gusts.
        flut = np.exp(0.35 * smooth_random(rng, n, sr, 9.0)) * np.exp(0.2 * smooth_random(rng, n, sr, 2.5))
        leaves = dsp.lowpass(dsp.bandpass(dsp.pink(rng, n), 1800.0 + 600.0 * min(speed / 10.0, 2.0), sr, 0.45),
                             5000.0, sr) * flut * amp ** 0.7
        out += trees * 0.45 * _norm(leaves)
    if rumble > 0:
        r = dsp.lowpass(dsp.brown(rng, n), 90.0, sr, 0.5) * amp
        out += rumble * 0.5 * _norm(r)
    return dsp.highpass(out, 35.0, sr)


# =================================================================== rain
RAIN_SURFACES = {
    # (modes [(Hz, t60)], click_hz, bubble_chance, level of click vs ring)
    "ground": dict(modes=[(900, .006), (1700, .005), (2800, .004)], click_hz=2200.0, bubble=0.0, ring=0.4),
    "leaves": dict(modes=[(700, .015), (1300, .012), (2300, .01), (3600, .008)], click_hz=2600.0, bubble=0.0, ring=0.8),
    "roof": dict(modes=[(420, .08), (650, .07), (980, .06), (1450, .05), (2100, .04), (3000, .03)], click_hz=1800.0,
                 bubble=0.0, ring=1.0),
    "window": dict(modes=[(1150, .03), (1900, .025), (2900, .02), (4100, .015)], click_hz=3500.0, bubble=0.0, ring=0.6),
    "puddle": dict(modes=[(1200, .004)], click_hz=2500.0, bubble=0.35, ring=0.2),
}


@model("rain", "sound",
       "Rain: individual near drops (impact click through the surface's response, and on water a Minnaert "
       "bubble when the drop entrains air) over the hiss of thousands of distant drops. Drop sizes are "
       "exponential (Marshall-Palmer); rate and hiss level follow the rain rate. Drop arrival is Poisson (no "
       "repeating pattern).",
       [
           Param("rate_mm_h", 5.0, "mm/h", 0.2, 100, "rain rate: drizzle ~0.5, moderate ~4, heavy ~16, downpour 50+"),
           Param("surface", "ground", "", choices=tuple(RAIN_SURFACES)),
           Param("near", 0.6, "", 0, 1, "level of individually heard near drops"),
           Param("hiss", 0.6, "", 0, 1, "level of the distant hiss"),
           Param("drop_mm", 1.8, "mm", 0.5, 5, "median drop diameter (grows with rain rate)"),
           Param("brightness", 0.5, "", 0, 1, "top end of the hiss (0 = far/indoors, 1 = close, outdoors)"),
       ], ["Marshall & Palmer 1948 (drop size distribution)", "Prosperetti & Oguz 1993 (rain bubbles)",
           "Wikipedia: Minnaert resonance", "van den Doel 2005 (liquid sounds)"])
def rain(ctx: Ctx, x, rate_mm_h, surface, near, hiss, drop_mm, brightness):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    su = RAIN_SURFACES[surface]
    R = rate_mm_h
    swell = np.exp(0.15 * smooth_random(rng, n, sr, 0.2))
    out = np.zeros(n)
    if near > 0:
        rate = 25.0 * R ** 0.8 * swell                    # near drops heard individually (est.)
        idx = dsp.poisson_times(rng, rate, sr)
        D = np.clip(rng.exponential(drop_mm, len(idx)), 0.3, 6.0)
        amp = (D / drop_mm) ** 1.5
        imp = np.zeros(n)
        np.add.at(imp, idx, amp * rng.uniform(0.5, 1.0, len(idx)))
        # 0.10 ('too much crackle on top'): a drop is not a one-sample click. Its impact lasts ~0.5-2 ms
        # (a 2 mm drop at ~6 m/s spreading), so each drop is a smooth pulse; three widths so they are not cloned.
        soft = np.zeros(n)
        for k, w_ms in enumerate((1.0, 1.7, 2.6)):
            part = imp * (rng.integers(0, 3, n) == k)
            w = max(3, int(w_ms * 1e-3 * sr))
            soft += np.convolve(part, _hann(w) / w * 4, "same")
        fm = np.array([f for f, _ in su["modes"]])
        ring = np.zeros(n)
        for k in range(3):                                # three spots on the surface: not one cloned response
            part = soft * (rng.integers(0, 3, n) == k)
            ring += _conv(part, _mode_ir(sr, fm * rng.uniform(0.85, 1.15, len(fm)), [d for _, d in su["modes"]],
                                         rng.uniform(0.4, 1.0, len(fm))))
        click = dsp.lowpass(dsp.bandpass(soft, su["click_hz"] * 0.7, sr, 0.6), 3500.0, sr)
        drops = dsp.lowpass(su["ring"] * _norm(ring) + (1 - 0.5 * su["ring"]) * 0.6 * _norm(click), 5000.0, sr)
        if su["bubble"] > 0:
            b = np.zeros(n)
            for i, d in zip(idx, D):
                if rng.random() < su["bubble"]:
                    r = rng.uniform(0.25, 0.6) * d * 1e-3       # entrained bubble radius (est.)
                    dsp.place(b, bubble_grain(sr, r, 0.15), i + int(rng.uniform(0.005, 0.02) * sr), (d / drop_mm) ** 0.5)
            drops = drops + 0.8 * _norm(b)
        out += near * _norm(drops)
    if hiss > 0:
        # the diffuse wash: drops too many and too far to hear one by one. Darker top than before (0.10).
        top = 2200.0 + 4500.0 * brightness
        patter = np.exp(0.3 * smooth_random(rng, n, sr, 30.0))     # dense, smooth patter, no clicks
        h = dsp.lowpass(dsp.highpass(dsp.pink(rng, n), 400.0, sr), top, sr) * swell * patter
        out += hiss * (0.45 + 0.6 * min(R / 20.0, 1.0)) * _norm(h)
        # the low wash of thousands of distant splashes and run-off
        wash = dsp.lowpass(dsp.highpass(dsp.pink(rng, n), 70.0, sr), 450.0, sr) * swell
        out += hiss * (0.12 + 0.45 * min(R / 16.0, 1.0)) * _norm(wash)
    return dsp.highpass(dsp.lowpass(out, 12000.0, sr), 60.0, sr)


# =================================================================== glass break
@model("glass_break", "sound",
       "A pane or bottle breaking: the fracture is a sharp broadband crack plus a brief, choked ring of the whole "
       "piece; then the shards fall (free fall, then 1-3 bounces) and each impact rings with its own small "
       "plate modes (f ~ h / size^2: big shards ~1-3 kHz, small ones only click), with a contact click, "
       "and a few late tinkles as pieces slide and settle. Shard sizes follow a power law (many small, few big).",
       [
           Param("size", 0.4, "m", 0.05, 2, "size of the pane/bottle"),
           Param("thickness_mm", 4.0, "mm", 1, 12, "glass thickness"),
           Param("shards", 60, "count", 5, 400, "number of pieces"),
           Param("energy", 0.7, "", 0, 1, "how hard it breaks (crack level, scatter)"),
           Param("height", 0.9, "m", 0.1, 3, "fall height of the shards"),
           Param("floor", "hard", "", choices=("hard", "wood", "soft"), doc="what the shards land on"),
           Param("at", 0.02, "s", 0, None, "fracture time"),
       ], ["Leissa 1969 (plate modes)", "Farnell 2010 (glass breaking)", "Cook 2002 (PhISEM)"])
def glass_break(ctx: Ctx, x, size, thickness_mm, shards, energy, height, floor, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    h = thickness_mm * 1e-3
    i0 = int(at * sr)
    out = np.zeros(n)
    # fracture: crack noise (5-15 ms) + choked ring of the whole pane
    L = int(0.03 * sr)
    tt = np.arange(L) / sr
    crack = dsp.highpass(rng.standard_normal(L), 1500.0, sr) * np.exp(-tt / (0.004 + 0.006 * energy))
    f, a = mode_set("plate", materials.get("glass"), size, size * 0.75, h, 0, 30, None, 0.3, rng)
    pane = _mode_ir(sr, f, np.full(len(f), 0.05), a, max_len=int(0.1 * sr))
    frac = 0.5 * _norm(pane)
    frac[:L] += 0.9 * _norm(crack)
    dsp.place(out, frac, i0, 0.5 + 0.5 * energy)
    # shards
    E, rho, nu = 60e9, 2500, 0.24
    k = math.sqrt(E * h * h / (12 * rho * (1 - nu * nu)))
    # shard sizes: power law, mostly 1-3 cm, a few up to half the pane (est.)
    s = np.clip(size * 0.06 * rng.random(int(shards)) ** (-1 / 2.0), 0.008, size * 0.5)
    shard_out = np.zeros(n)
    floor_t60 = {"hard": 1.0, "wood": 0.6, "soft": 0.25}[floor]
    floor_lp = {"hard": 12000.0, "wood": 7000.0, "soft": 3000.0}[floor]
    for si in s:
        t = at + rng.uniform(0, 0.03) + math.sqrt(2 * height * rng.uniform(0.6, 1.0) / G)
        v = math.sqrt(2 * G * height) * rng.uniform(0.6, 1.0)
        # free plate of diameter si: f1 = lambda^2 / (2 pi a^2) sqrt(D / rho h), lambda^2 ~ 5.25, a = si / 2
        f1 = 5.25 / (2 * math.pi * (si / 2) ** 2) * k * rng.uniform(0.8, 1.25)
        for b in range(rng.integers(1, 4)):
            i = int(t * sr)
            if i >= n or v < 0.15:
                break
            A = v * min(si / 0.03, 2.0)       # big pieces hit harder, but break up / damp on the floor
            fr = f1 * np.array([1.0, rng.uniform(1.6, 1.8), rng.uniform(2.2, 2.5), rng.uniform(3.8, 4.2)])
            # big pieces land flat and are choked by the floor (est.)
            t60 = np.clip(0.25 * (2000.0 / np.maximum(fr, 200)) ** 0.5, 0.03, 0.35) * floor_t60 * min(1.0, (0.04 / si) ** 0.7)
            g = _mode_ir(sr, fr, t60, [1, .6, .4, .25], max_len=int(0.35 * sr))   # empty if all above Nyquist
            m = max(3, int(rng.uniform(0.03, 0.1) * 1e-3 * sr))
            clk = np.zeros(m + 96)
            clk[:m] = _hann(m)
            clk = _norm(dsp.bandpass(clk, rng.uniform(3500, 7500), sr, 0.8))
            gk = np.zeros(max(len(g), len(clk)))
            if np.any(g):
                gk[:len(g)] += 0.6 * _norm(g)
            gk[:len(clk)] += 0.5 * clk
            dsp.place(shard_out, gk, i, A)
            v *= rng.uniform(0.2, 0.4)
            t += 2 * v / G + rng.uniform(0.002, 0.01)
    # late tinkles (pieces sliding and settling), rate decaying
    tail = np.zeros(n)
    t_end = at + math.sqrt(2 * height / G)
    for _ in range(int(shards * 0.3)):
        t = t_end + rng.exponential(0.25)
        fr = rng.uniform(2500, 7000) * np.array([1.0, 1.62])
        dsp.place(tail, _mode_ir(sr, fr, [0.04, 0.03], [1, .5]), int(t * sr), rng.uniform(0.05, 0.3))
    debris = dsp.lowpass(_norm(shard_out) + 0.3 * _norm(tail), floor_lp, sr)
    out += 0.5 * _norm(debris) * (0.7 + 0.3 * energy)
    return dsp.highpass(dsp.lowpass(out, 13000.0, sr), 150.0, sr)


# =================================================================== sword swing / whoosh
@model("swing", "sound",
       "Something swung fast past the ear (sword, stick, staff). Aeroacoustics: turbulent wake noise centred on "
       "the vortex-shedding frequency f = St v / d (St ~ 0.2, d = cross-section), level ~ v^3 (dipole); a thin "
       "edge also gives a narrow aeolian tone ('zing'). Speed follows a fast swing profile; the pitch is "
       "Doppler-shifted (up while approaching, down after the closest point) and the level peaks as it passes.",
       [
           Param("speed", 25.0, "m/s", 3, 60, "peak tip speed (sword swing ~20-40 m/s, est.)"),
           Param("swing_s", 0.35, "s", 0.08, 2, "duration of the swing"),
           Param("width_mm", 30.0, "mm", 3, 200, "cross-section facing the air (blade width / stick diameter)"),
           Param("edge_mm", 4.0, "mm", 0.5, 30, "thin edge thickness (aeolian tone pitch)"),
           Param("edge", 0.35, "", 0, 1, "level of the edge tone ('zing')"),
           Param("doppler", 1.0, "", 0, 2, "Doppler amount (1 = physical)"),
           Param("closest_m", 0.6, "m", 0.1, 5, "closest distance to the listener"),
           Param("at", 0.03, "s", 0, None, "start of the swing"),
       ], ["Wikipedia: Aeolian sound; Strouhal number", "Farnell 2010, Designing Sound (swishes)",
           "Curle 1955 (dipole sound from a body in flow)"])
def swing(ctx: Ctx, x, speed, swing_s, width_mm, edge_mm, edge, doppler, closest_m, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    t = np.arange(n) / sr
    u = np.clip((t - at) / swing_s, 0, 1)
    prof = np.sin(np.pi * u ** 0.8) ** 2                     # accelerates fast, slows less abruptly
    v = speed * prof * (1 + 0.04 * smooth_random(rng, n, sr, 12))
    # path: passes the listener at the closest point at u ~ 0.5 (an arc of radius ~1 m, est.)
    s_path = np.cumsum(v) / sr
    s_mid = np.interp(0.5, u, s_path) if np.any(u > 0.5) else s_path[-1] / 2
    dx = s_path - s_mid
    r = np.sqrt(dx ** 2 + closest_m ** 2)
    v_rad = -v * dx / r                                      # approaching = positive
    dop = 1.0 / np.maximum(1 - doppler * v_rad / C, 0.5)
    loud = (v / speed) ** 3 * (closest_m / r)
    fc = np.clip(0.2 * v / (width_mm * 1e-3) * dop, 80, 9000)
    body = dsp.svf(dsp.pink(rng, n) * 3, fc * 2.5, 0.9, sr, "bpn") + 0.6 * dsp.svf(dsp.white(rng, n), fc * 6, 0.8, sr, "bpn")
    out = _norm(body * loud)
    if edge > 0:
        fe = np.clip(0.2 * v / (edge_mm * 1e-3) * dop, 200, 9000)
        zing = dsp.svf(dsp.white(rng, n), fe, 18.0, sr, "bpn") * loud * prof
        out += edge * 0.7 * _norm(zing)
    return dsp.highpass(dsp.lowpass(out, 11000.0, sr), 80.0, sr)


# =================================================================== blade clash / metal clang
CLASH_OBJECTS = {
    # shape, length, width, thickness (m), extra damping (grip), contact ms
    # hp: the hand/grip and mounting soak up the lowest bending modes (est.; CC0 clash refs put most energy
    # at 3-8 kHz with ~0.1-0.3 s decays)
    "sword": dict(shape="bar", length=0.75, width=0.04, thickness=0.0045, damp=12.0, contact=0.05, hp=350.0),
    "pipe": dict(shape="bar", length=0.9, width=0.04, thickness=0.012, damp=2.0, contact=0.12, hp=150.0),
    "plate": dict(shape="plate", length=0.5, width=0.4, thickness=0.002, damp=3.0, contact=0.1, hp=200.0),
    "pan": dict(shape="plate", length=0.28, width=0.28, thickness=0.003, damp=12.0, contact=0.15, hp=150.0),
}


@model("clash", "sound",
       "Steel struck by steel: two blades (or a pipe/plate) ring with their own bending modes (each a bar with "
       "dense inharmonic overtones, strike near the edge excites many), slightly different so they beat; a "
       "very short metal-on-metal contact; then optional scrape as the edges slide (friction noise through the "
       "same modes: the 'shing').",
       [
           Param("object", "sword", "", choices=tuple(CLASH_OBJECTS)),
           Param("velocity", 3.0, "m/s", 0.2, 15, "closing speed"),
           Param("scrape_s", 0.25, "s", 0, 2, "how long the edges slide after the hit (0 = clean clang)"),
           Param("scrape", 0.4, "", 0, 1, "scrape level"),
           Param("damping", 1.0, "", 0.2, 10, "grip/hand damping multiplier (higher = shorter ring)"),
           Param("size", 1.0, "", 0.4, 2.5, "scales the object's length (bigger = lower)"),
           Param("second", 0.7, "", 0, 1, "level of the second object's ring (0 = struck by something dead)"),
           Param("at", 0.02, "s", 0, None),
       ], ["van den Doel, Kry & Pai 2001", "Euphonics: free-free beam modes", "Avanzini et al. 2002 (friction)"])
def clash(ctx: Ctx, x, object, velocity, scrape_s, scrape, damping, size, second, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    ob = CLASH_OBJECTS[object]
    i0 = int(at * sr)
    m = max(3, int(ob["contact"] * 1e-3 * sr * (velocity / 3.0) ** -0.2))
    exc = np.zeros(n)
    dsp.place(exc, _hann(m), i0, velocity)
    scr = np.zeros(n)
    if scrape_s > 0 and scrape > 0:
        L = int(scrape_s * sr)
        tt = np.arange(L) / sr
        env = np.exp(-tt / (0.45 * scrape_s)) * (1 - np.exp(-tt / 0.004))
        rough = dsp.highpass(rng.standard_normal(L), 1500.0, sr) * (1 + 0.6 * np.abs(smooth_random(rng, L, sr, 60)))
        dsp.place(scr, rough * env, i0 + m, scrape * velocity * 0.05)
    out = np.zeros(n)
    for j, lev in ((0, 1.0), (1, second)):
        if lev <= 0:
            continue
        y = call("modal", ctx.child(), exc + scr, material="steel", shape=ob["shape"],
                 length=ob["length"] * size * (1 + 0.07 * j * rng.uniform(0.5, 1.5)), width=ob["width"],
                 thickness=ob["thickness"], n_modes=64, strike_pos=rng.uniform(0.02, 0.08),
                 extra_damping=ob["damp"] * damping, detune=0.01, max_t60=3.0)
        out += lev * _norm(y)
    # keep the top civil (no tinny tops): gentle shelf down above ~7 kHz
    out = 0.75 * out + 0.25 * dsp.lowpass(out, 7000.0, sr)
    return dsp.highpass(dsp.highpass(out, ob["hp"], sr), ob["hp"], sr)


# =================================================================== door creak / slam
@model("door_creak", "sound",
       "A door creaking open: stick-slip at the hinge. The pin sticks, the door's push builds up, it slips; "
       "each slip is a small impulse into the hinge (small, well-damped metal modes) and the door leaf (a big, "
       "damped wooden panel). The slip rate rises with angular speed and pressure, and wanders cycle to cycle "
       "(no machine-regular buzz). Slow movement = separate ticks; faster = a tonal groan.",
       [
           Param("open_s", 1.8, "s", 0.3, 8, "how long the opening takes"),
           Param("pitch_hz", 260.0, "Hz", 40, 1500, "slip rate at full speed (the creak's pitch)"),
           Param("pressure", 0.6, "", 0, 1, "how hard the hinge binds (louder, rougher, a bit higher)"),
           Param("wander", 0.35, "", 0, 1, "slip-rate wandering (hand speed changes, stick-slip chaos)"),
           Param("hinge", 0.5, "", 0, 1, "metal hinge resonance vs wooden door body"),
           Param("door_size", 1.0, "", 0.3, 2, "door size (1 = 2.0 x 0.8 m)"),
           Param("at", 0.05, "s", 0, None),
       ], ["Wikipedia: Stick-slip phenomenon", "Avanzini, Serafin & Rocchesso 2002 (friction models)",
           "Farnell 2010, Designing Sound (creaking door)"])
def door_creak(ctx: Ctx, x, open_s, pitch_hz, pressure, wander, hinge, door_size, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    t = np.arange(n) / sr
    u = np.clip((t - at) / open_s, 0, 1)
    speed = np.sin(np.pi * u) ** 0.8 * (1 + 0.25 * wander * smooth_random(rng, n, sr, 2.5))
    speed = np.clip(speed, 0, None) * (u < 1) * (t >= at)
    rate = pitch_hz * (0.35 + 0.65 * speed) * (0.8 + 0.4 * pressure) * (1 + 0.08 * wander * smooth_random(rng, n, sr, 6))
    pulses = np.zeros(n)
    tt = at
    while tt < min(at + open_s, n / sr):
        i = int(tt * sr)
        sp = speed[i]
        if sp > 0.06:
            dsp.place(pulses, _hann(4), i, (sp ** 0.6) * (0.6 + 0.8 * pressure) * rng.uniform(0.7, 1.3))
            tt += (1.0 / rate[i]) * (1 + 0.12 * wander * rng.standard_normal())
        else:
            tt += rng.uniform(0.005, 0.03)       # nearly stopped: sparse ticks
            if rng.random() < 0.4:
                dsp.place(pulses, _hann(4), int(tt * sr), 0.3 * rng.uniform(0.5, 1.0))
    hf = np.array([1150.0, 2350.0, 3900.0]) * rng.uniform(0.9, 1.1, 3)
    hinge_y = _conv(pulses, _mode_ir(sr, hf, [0.02, 0.015, 0.01], [1, .6, .3]))
    f, a = mode_set("plate", materials.get("oak"), 2.0 * door_size, 0.8 * door_size, 0.04, 0, 60, None, 0.1, rng)
    door_y = _conv(pulses, _mode_ir(sr, f * rng.uniform(0.97, 1.03, len(f)), 6.91 / (np.pi * f * 0.01 + 40.0), a))
    out = hinge * _norm(hinge_y) + (1 - 0.6 * hinge) * _norm(door_y)
    return dsp.highpass(dsp.lowpass(out, 9000.0, sr), 70.0, sr)


@model("door_slam", "sound",
       "A heavy door slammed shut. 0.10 (after 'two wood planks hitting'): the mass is what you hear. "
       "A 25-40 kg leaf stops against the frame through a soft seal (4-8 ms contact), so most of the energy goes "
       "into low, damped motion: the leaf, frame and wall swing together (~60-150 Hz thump), the frame shudders "
       "(a few small, low rebounds), the latch bolt clunks home (a damped metal clunk, ~0.6-1.5 kHz, not a "
       "click), the leaf's panel modes ring only weakly and darkly, and the room answers. The bright "
       "wood-on-wood clack is kept small.",
       [
           Param("force", 0.7, "", 0, 1, "how hard (closing speed)"),
           Param("door_size", 1.0, "", 0.3, 2, "door size/mass (1 = 2.0 x 0.8 x 0.04 m solid door, ~30 kg)"),
           Param("latch", 0.6, "", 0, 1, "latch clunk level"),
           Param("rattle", 0.4, "", 0, 1, "frame shudder / rebound rattle"),
           Param("thump", 0.7, "", 0, 1, "low mass thump of leaf + frame + wall (the weight of the door)"),
           Param("clack", 0.15, "", 0, 1, "bright wood-on-wood edge clack (keep low for heavy doors)"),
           Param("at", 0.03, "s", 0, None),
       ], ["van den Doel et al. 2001 (modal impacts)", "Farnell 2010 (doors)"])
def door_slam(ctx: Ctx, x, force, door_size, latch, rattle, thump, clack, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    i0 = int(at * sr)
    v = 0.5 + 3.0 * force
    mass = door_size ** 1.5                           # relative mass
    # soft seal + heavy leaf: long contact (est. 4-8 ms), shorter when slammed harder
    m = int((8.0 - 4.0 * force) * 1e-3 * sr * mass ** 0.25)
    exc = np.zeros(n)
    dsp.place(exc, _push(m), i0, v)
    t, a = at, v
    for _ in range(int(1 + 3 * rattle)):              # frame shudder: small low rebounds
        t += rng.uniform(0.025, 0.06)
        a *= rng.uniform(0.2, 0.35)
        dsp.place(exc, _push(int(m * 0.7)), int(t * sr), a * (0.3 + rattle))
    # mass thump: leaf + frame + wall moving together, heavily damped low modes (est.)
    fm = np.array([72.0, 105.0, 145.0, 195.0]) / mass ** 0.3 * rng.uniform(0.92, 1.08, 4)
    body = _conv(exc, _mode_ir(sr, fm, [0.22, 0.16, 0.12, 0.09], [1.0, 0.8, 0.55, 0.35]))
    # leaf panel modes: present but dark (damped, low-passed)
    f, amp = mode_set("plate", materials.get("oak"), 2.0 * door_size, 0.8 * door_size, 0.04, 0, 60, None, 0.45, rng)
    f = f * (1 + 0.04 * rng.uniform(-1, 1, len(f)))
    # the leaf's own panel is struck by its hard edge on the stop (shorter, ~1.5-2.5 ms) - that is what makes
    # it read as a door rather than a padded thud
    exc_p = np.zeros(n)
    dsp.place(exc_p, _push(int((2.6 - 1.2 * force) * 1e-3 * sr)), i0, v)
    panel = dsp.lowpass(_conv(exc_p, _mode_ir(sr, f, 6.91 / (np.pi * f * 0.02 + 30.0), amp)), 2500.0, sr)
    out = thump * 0.7 * _norm(body) + 1.0 * _norm(panel)
    # low air/wall boom: the pressure push of the leaf closing on the room (a broad lowpassed burst)
    bl = int(0.12 * sr)
    boom = dsp.bandpass(rng.standard_normal(bl), 110.0, sr, 0.8) * np.exp(-np.arange(bl) / (0.035 * sr))
    bo = np.zeros(n)
    dsp.place(bo, boom, i0, 1.0)
    out += thump * 0.2 * _norm(bo)
    if clack > 0:                                     # edge on stop: short, mid, kept small
        ce = np.zeros(n)
        Lc = int(0.006 * sr)
        dsp.place(ce, np.exp(-np.arange(Lc) / (0.0012 * sr)), i0, 1.0)
        out += clack * 0.4 * _norm(dsp.lowpass(dsp.bandpass(dsp.white(rng, n), 1100.0, sr, 0.7) * ce, 3500.0, sr))
    if latch > 0:                                     # bolt clunk: damped metal, mid, a few ms after impact
        lk = np.zeros(n)
        il = i0 + int(rng.uniform(0.004, 0.012) * sr)
        dsp.place(lk, _push(int(0.0008 * sr)), il, 1.0)
        hf = np.array([720.0, 1150.0, 1700.0, 2600.0]) * rng.uniform(0.9, 1.1, 4)
        clunk = _conv(lk, _mode_ir(sr, hf, [0.045, 0.035, 0.025, 0.015], [1, .7, .45, .2]))
        out += latch * 0.3 * _norm(dsp.lowpass(clunk, 4000.0, sr))
    return dsp.highpass(dsp.highpass(dsp.lowpass(out, 6000.0, sr), 50.0, sr), 50.0, sr)


# =================================================================== body hit (punch)
@model("body_hit", "sound",
       "A punch or body blow: a soft, heavy contact (fist into flesh, 6-15 ms) gives a low, damped thump from the "
       "torso (a lossy cavity, ~100-250 Hz); the skin/glove slaps (short broadband noise); clothing rustles. "
       "A face hit is bonier (shorter contact, higher, optional crack). Real punches are duller than film ones; "
       "`punch_up` adds the film sweetening.",
       [
           Param("target", "body", "", choices=("body", "face", "belly")),
           Param("force", 0.7, "", 0, 1, "how hard"),
           Param("glove", "bare", "", choices=("bare", "glove"), doc="bare knuckle (sharp slap) or boxing glove (pop)"),
           Param("cloth", 0.4, "", 0, 1, "clothing rustle level"),
           Param("crack", 0.0, "", 0, 1, "bone crack (face)"),
           Param("punch_up", 0.15, "", 0, 1, "film sweetening: extra low thump"),
           Param("at", 0.02, "s", 0, None),
       ], ["Farnell 2010 (impacts)", "Cook 2002 (contact models)"])
def body_hit(ctx: Ctx, x, target, force, glove, cloth, crack, punch_up, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    i0 = int(at * sr)
    tg = {"body": dict(ms=10.0, modes=[(130, .04), (210, .03), (340, .02)]),
          "belly": dict(ms=14.0, modes=[(95, .05), (160, .035)]),
          "face": dict(ms=5.0, modes=[(260, .025), (520, .02), (900, .015)])}[target]
    m = int(tg["ms"] * (1.2 - 0.4 * force) * 1e-3 * sr * rng.uniform(0.9, 1.1))
    F = np.zeros(n)
    dsp.place(F, _push(m), i0, 0.4 + force)
    fm = np.array([f for f, _ in tg["modes"]]) * rng.uniform(0.9, 1.1, len(tg["modes"]))
    torso = _conv(F, _mode_ir(sr, fm, [d for _, d in tg["modes"]], [1, .6, .4][:len(fm)]))
    thump = dsp.lowpass(dsp.lowpass(np.diff(F, prepend=0.0), 140.0, sr), 140.0, sr)
    env = np.zeros(n)
    L = int(0.006 * sr)
    dsp.place(env, np.exp(-np.arange(L) / (0.0015 * sr)), i0, 1.0)
    slap_hz = 3000.0 if glove == "bare" else 1100.0
    slap = dsp.bandpass(dsp.white(rng, n), slap_hz, sr, 0.7) * env
    out = _norm(torso) + 0.7 * _norm(slap) * (0.6 + 0.6 * force) + punch_up * 0.7 * _norm(thump)
    if cloth > 0:
        g = np.zeros(n)
        Lc = int(rng.uniform(0.06, 0.12) * sr)
        dsp.place(g, _hann(Lc), max(0, i0 - int(0.03 * sr)), 1.0)
        out += cloth * 0.25 * _norm(dsp.bandpass(dsp.pink(rng, n), 1800.0, sr, 0.6) * g)
    if crack > 0:
        c = np.zeros(n)
        for _ in range(rng.integers(2, 5)):
            dsp.place(c, _hann(3), i0 + int(rng.uniform(0.0, 0.012) * sr), rng.uniform(0.5, 1.0))
        out += crack * 0.5 * _norm(dsp.bandpass(c, 2500.0, sr, 1.2))
    return dsp.highpass(out, 60.0, sr)


# =================================================================== bullet impact
@model("bullet_impact", "sound",
       "A bullet hitting a surface (not the gunshot): a near-instant hard contact (tens of microseconds) gives "
       "a sharp crack. Wood: splintering noise and a damped board knock, splinters falling. Metal: a bright "
       "ping/clang of the plate, optional ricochet whine (a spinning, tumbling bullet: a noisy tone gliding "
       "down by Doppler as it leaves). Concrete: a hard crack, chips and grit raining down, a little dust hiss.",
       [
           Param("material", "concrete", "", choices=("wood", "metal", "concrete")),
           Param("energy", 0.7, "", 0, 1, "impact energy (calibre/speed)"),
           Param("debris", 0.6, "", 0, 1, "chips/splinters/grit level"),
           Param("ricochet", 0.0, "", 0, 1, "ricochet whine level (metal and concrete)"),
           Param("at", 0.02, "s", 0, None),
       ], ["Farnell 2010 (bullets, ricochets)", "van den Doel et al. 2001"])
def bullet_impact(ctx: Ctx, x, material, energy, debris, ricochet, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    i0 = int(at * sr)
    imp = np.zeros(n)
    dsp.place(imp, _hann(3), i0, 1.0)
    L = int(0.02 * sr)
    tt = np.arange(L) / sr
    crk = np.zeros(n)
    dsp.place(crk, dsp.highpass(rng.standard_normal(L), 1200.0, sr) * np.exp(-tt / 0.0025), i0, 1.0)
    out = (0.6 + 0.4 * energy) * _norm(crk)
    if material == "wood":
        f, a = mode_set("plate", materials.get("oak"), 0.8, 0.25, 0.025, 0, 60, None, 0.3, rng)
        out += 0.9 * _norm(_conv(imp, _mode_ir(sr, f, 6.91 / (np.pi * f * 0.01 + 70.0), a)))
        spl = dsp.bandpass(dsp.white(rng, n), 2500.0, sr, 0.6) * np.exp(-np.maximum(np.arange(n) - i0, 0) / (0.012 * sr)) * (np.arange(n) >= i0)
        out += 0.5 * _norm(spl)
        grain = (500, 3500, 0.010)
    elif material == "metal":
        f, a = mode_set("plate", materials.get("steel"), 0.6, 0.4, 0.004, 0, 50, None, 0.2, rng)
        out += 1.0 * _norm(_conv(imp, _mode_ir(sr, f, np.minimum(6.91 / (np.pi * f * 3e-4 + 6.0), 1.2), a)))
        grain = (2000, 6000, 0.004)
    else:
        f, a = mode_set("plate", materials.get("concrete"), 0.8, 0.8, 0.12, 0, 30, None, 0.3, rng)
        out += 0.5 * _norm(_conv(imp, _mode_ir(sr, f, 6.91 / (np.pi * f * 0.006 + 60.0), a)))
        grain = (1500, 7000, 0.003)
    if debris > 0:
        rate = [[0, 0], [at, 0], [at + 0.01, 400 * debris], [at + 0.15, 120 * debris], [at + 0.5, 20 * debris],
                [at + 1.0, 0]]
        d = call("crackle", ctx.child(), rate=rate, grain_ms=grain[2] * 1000, f_lo=grain[0], f_hi=grain[1], q=1.0,
                 size_exponent=2.3, level=1.0)
        out += debris * 0.45 * _norm(d)
    if ricochet > 0 and material != "wood":
        t = np.arange(n) / sr - at
        on = (t > 0.005) & (t < 0.5)
        f_r = 3200.0 * np.exp(-np.maximum(t, 0) / 0.6) * (1 + 0.03 * smooth_random(rng, n, sr, 25))
        whine = dsp.svf(dsp.white(rng, n), np.clip(f_r, 300, 8000), 25.0, sr, "bpn") * on * np.exp(-np.maximum(t, 0) / 0.18)
        out += ricochet * 0.6 * _norm(whine)
    return dsp.highpass(dsp.lowpass(out, 12000.0, sr), 80.0, sr)


# =================================================================== gunshot
GUNS = {
    # Friedlander positive phase t* (s), gas-body low-pass (Hz), body decay (s), boom decay (s), boom level,
    # crack level, N-wave length (s). Estimates from Maher 2006 / Beck et al. 2011 muzzle-blast durations.
    "pistol": dict(ts=0.0011, body_fc=6000.0, body_tau=0.022, boom_tau=0.045, boom=0.55, crack=0.35, nw=0.00025),
    "rifle": dict(ts=0.0016, body_fc=8000.0, body_tau=0.028, boom_tau=0.060, boom=0.6, crack=1.0, nw=0.0004),
    "shotgun": dict(ts=0.0026, body_fc=4500.0, body_tau=0.040, boom_tau=0.090, boom=0.8, crack=0.0, nw=0.0004),
}


@model("gunshot", "sound",
       "A gunshot (0.10, after 'a stone hitting a mildly metallic wall'). Layers, each physical: (1) the "
       "muzzle blast, a Friedlander pressure pulse (positive phase ~1-3 ms by calibre) that starts the sound "
       "with a very fast, loud broadband crack; (2) the expanding, turbulent propellant gas: a dense broadband "
       "burst decaying in tens of ms (the body); (3) the low-mid boom of the gas cloud pushing air (a "
       "low-passed burst, ~40-90 ms); all three drive the recorder or ear into overload (oversampled soft clip: "
       "loud, compressed, no aliasing); (4) the supersonic bullet's N-wave crack (rifles; none for shotguns); "
       "(5) the environment: discrete slap echoes off nearby walls/ground and a dark diffuse tail. No ringing "
       "parts: the action clack is a dry noise burst.",
       [
           Param("calibre", "pistol", "", choices=tuple(GUNS)),
           Param("crack", 0.5, "", 0, 1, "supersonic crack level (scaled by calibre; shotguns have none)"),
           Param("drive", 2.0, "", 0, 4, "overload of the recorder/ear (soft clip): more = louder, denser"),
           Param("body", 0.7, "", 0, 1, "gas body + low boom level"),
           Param("echo", 0.6, "", 0, 1, "slap echoes and tail level"),
           Param("space", "street", "", choices=("street", "room", "open"), doc="where: street (walls near, "
                 "strong slaps), room (dense, short), open field (one late ground/treeline echo, long dark tail)"),
           Param("mech", 0.25, "", 0, 1, "action/slide clack level (dry)"),
           Param("at", 0.02, "s", 0, None),
       ], ["Wikipedia: Blast wave (Friedlander)", "Maher 2006, Modeling and signal processing of acoustic gunshot "
           "recordings", "Beck, Nakasone & Marr 2011 (muzzle blast variation)", "Farnell 2010 (guns)"])
def gunshot(ctx: Ctx, x, calibre, crack, drive, body, echo, space, mech, at):
    from scipy import signal
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    g = GUNS[calibre]
    i0 = int(at * sr)
    m = n - i0
    os = 4
    so = sr * os
    L = min(m, int(0.4 * sr)) * os
    tt = np.arange(L) / so
    blast = friedlander(L, so, g["ts"] * rng.uniform(0.9, 1.1))
    gas = dsp.lowpass(rng.standard_normal(L), g["body_fc"], so) * np.exp(-tt / g["body_tau"]) \
        * (1 - np.exp(-tt / 0.0004))
    boom = dsp.lowpass(dsp.lowpass(rng.standard_normal(L), 180.0, so), 180.0, so) * np.exp(-tt / g["boom_tau"]) \
        * (1 - np.exp(-tt / 0.002))
    src = _norm(blast) + body * (0.8 * _norm(gas) + g["boom"] * 0.7 * _norm(boom))
    if drive > 0:
        src = np.tanh(drive * 1.5 * src / np.max(np.abs(src))) / np.tanh(drive * 1.5)
    dry = signal.resample_poly(src, 1, os)[: L // os]
    out = np.zeros(n)
    dsp.place(out, dry, i0, 1.0)
    if crack > 0 and g["crack"] > 0:                  # N-wave arrives just before the blast (bullet outruns it)
        Ln = int(g["nw"] * sr) + 2
        cr = np.zeros(n)
        dsp.place(cr, np.linspace(1, -1, Ln), max(0, i0 - int(0.0015 * sr)), 1.0)
        out += crack * g["crack"] * 0.8 * dsp.lowpass(cr, 15000.0, sr)
    if mech > 0:                                      # dry action clack: filtered noise burst, no modes
        Lm = int(0.012 * sr)
        ck = dsp.bandpass(rng.standard_normal(Lm), 2200.0, sr, 0.6) * np.exp(-np.arange(Lm) / (0.002 * sr))
        dsp.place(out, ck / (np.max(np.abs(ck)) + 1e-12), i0 + int(rng.uniform(0.05, 0.09) * sr), 0.12 * mech)
    if echo > 0:
        sp = {"street": dict(slaps=(0.018, 0.045, 0.07, 0.11, 0.16), lvl=0.55, tail=0.9, tail_lvl=0.18, lp=3500.0),
              "room": dict(slaps=(0.006, 0.011, 0.017, 0.024), lvl=0.5, tail=0.5, tail_lvl=0.25, lp=5000.0),
              "open": dict(slaps=(0.09, 0.42), lvl=0.35, tail=1.8, tail_lvl=0.12, lp=2200.0)}[space]
        slap_src = dsp.lowpass(out, sp["lp"], sr)
        wet = np.zeros(n)
        for k, d in enumerate(sp["slaps"]):
            dd = int(d * rng.uniform(0.85, 1.15) * sr)
            if dd < n:
                wet[dd:] += slap_src[:n - dd] * sp["lvl"] * (0.7 ** k) * rng.uniform(0.7, 1.0)
        # diffuse tail: noise with the shot's energy, dark, exponential (est. reverberant decay)
        Lt = n - i0
        tl = np.arange(Lt) / sr
        tail = dsp.lowpass(rng.standard_normal(Lt), sp["lp"] * 0.6, sr) * np.exp(-6.91 * tl / sp["tail"]) \
            * (1 - np.exp(-tl / 0.015))
        dsp.place(wet, tail * np.max(np.abs(out)) * sp["tail_lvl"], i0, 1.0)
        out += echo * wet
    return dsp.highpass(out, 40.0, sr)


# =================================================================== creature growl
def glottal_source(rng, n, sr, f0_curve, amp_curve, jitter, shimmer, subharm, oq=0.6):
    """Voiced source, one cycle at a time: Rosenberg glottal flow pulses (open quotient `oq`, smooth opening,
    faster closing) whose derivative is the radiated excitation. Period jitter, shimmer and period doubling
    (alternate long/strong and short/weak cycles = subharmonics) are true cycle-to-cycle effects. Returns
    (excitation, flow) - the flow gates breath noise."""
    out = np.zeros(n)
    flow = np.zeros(n)
    t = float(rng.uniform(0, sr / max(f0_curve[0], 1.0)))
    k = 0
    while t < n - 2:
        i = int(t)
        P = sr / max(f0_curve[i], 1.0)
        side = 1 if k % 2 else -1
        period = P * (1 + 0.25 * subharm * side) * (1 + jitter * rng.standard_normal())
        period = max(period, 0.4 * P)
        A = amp_curve[i] * max(0.1, 1 + shimmer * rng.standard_normal()) * (1 - 0.45 * subharm * (k % 2))
        if A > 1e-4:
            To = max(4, int(oq * period))
            Tp = max(2, int(0.66 * To))
            u = np.arange(To, dtype=float)
            g = np.where(u < Tp, 0.5 * (1 - np.cos(np.pi * u / Tp)), np.cos(0.5 * np.pi * (u - Tp) / max(To - Tp, 1)))
            dsp.place(flow, g, i, A)
            dsp.place(out, np.diff(g, append=0.0), i, A)
        t += period
        k += 1
    return out, flow


@model("growl", "sound",
       "A creature growl, voiced (0.10, after 'something pushed along a floor'). The old source was "
       "narrow clicks at ~45 Hz: a click train at that rate is what stick-slip friction sounds like. Now: "
       "vocal-fold pulses with a real open phase (Rosenberg flow, open quotient ~0.6), pitch ~60-120 Hz with "
       "chaotic wander, jitter, shimmer and period doubling (the subharmonic an octave down is the growl's "
       "menace), through a long open vocal tract (formants of a tube closed at the glottis, F_n = (2n-1) c / 4L; "
       "bigger creature = longer tract = lower, denser formants), breath noise gated by the airflow and shaped by "
       "the same tract, and slow amplitude swells.",
       [
           Param("f0", 85.0, "Hz", 25, 400, "voice pitch (big cat roar/growl ~ 60-120 Hz with subharmonics; dog ~ 150-300)"),
           Param("size", 1.0, "", 0.3, 3, "creature size (1 = ~45 cm vocal tract; formants ~ 1/size)"),
           Param("roughness", 0.6, "", 0, 1, "jitter, shimmer, subharmonics and pitch chaos"),
           Param("flutter_hz", 18.0, "Hz", 5, 60, "rate of the gentle rattle in the airflow"),
           Param("snarl", 0.3, "", 0, 1, "open, bright snarl (raised formants, more breath)"),
           Param("length_s", 1.6, "s", 0.2, 6, "growl length"),
           Param("rise", 0.4, "", 0, 1, "swells: pitch/level rise toward the middle"),
           Param("at", 0.05, "s", 0, None),
       ], ["Titze 1994 (voice source)", "Rosenberg 1971 (glottal pulse)", "Fitch 1997 (formant dispersion and "
           "body size)", "Herzel et al. 1995 (subharmonics)", "Peters et al. 2009 (big cat vocal anatomy)"])
def growl(ctx: Ctx, x, f0, size, roughness, flutter_hz, snarl, length_s, rise, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    t = np.arange(n) / sr
    u = np.clip((t - at) / length_s, 0, 1)
    on = (t >= at) & (u < 1)
    attack = np.clip((t - at) / 0.12, 0, 1)
    env = np.where(on, np.sin(np.pi * u) ** 0.6 * attack, 0.0)
    env *= (1 + rise * 0.6 * np.sin(np.pi * u)) * np.exp(0.25 * smooth_random(rng, n, sr, 1.5))   # swells
    chaos = 0.03 + 0.07 * roughness
    fcurve = f0 * (1 + rise * 0.2 * np.sin(np.pi * u)) * np.exp(chaos * smooth_random(rng, n, sr, 6)
                                                                + 0.02 * smooth_random(rng, n, sr, 1))
    flutter = 1 + 0.18 * roughness * np.sin(2 * np.pi * np.cumsum(flutter_hz * (1 + 0.15 * smooth_random(rng, n, sr, 3))) / sr)
    src, flow = glottal_source(rng, n, sr, fcurve, env * flutter, jitter=0.01 + 0.05 * roughness,
                               shimmer=0.05 + 0.15 * roughness, subharm=0.15 + 0.5 * roughness)
    # breath: turbulent noise at the glottis, modulated by the flow (pulsed, not steady hiss)
    breath = dsp.highpass(rng.standard_normal(n), 300.0, sr) * (0.3 + dsp.lowpass(flow, 400.0, sr)) * env
    Lt = 0.45 * size                                     # tract length (m)
    s = 1.0 + 0.25 * snarl
    formants = []
    for k in range(6):
        fk = (2 * k + 1) * C / (4 * Lt) * s
        if fk > 6000:
            break
        formants.append([fk, 5.0 + k, -1.5 * k + 1.5 * snarl * k])
    exc = src / (np.max(np.abs(src)) + 1e-12) + (0.12 + 0.25 * snarl) * breath / (np.max(np.abs(breath)) + 1e-12)
    y = call("formant", ctx, exc, formants=formants, shift=1.0, dry=0.03)
    y = _norm(y) + 0.12 * _norm(dsp.lowpass(src, 250.0, sr))        # chest: the fundamental + subharmonic body
    return dsp.highpass(dsp.lowpass(y, 6500.0, sr), 35.0, sr)


# =================================================================== rustle
@model("rustle", "sound",
       "Paper, cloth, leaves or plastic handled: hand movements (irregular gestures) drive the material. Paper "
       "and plastic crumple: sudden buckling of creases = crisp clicks with a power-law size spread "
       "(crackling noise, Houle & Sethna 1996), dense during fast movement. Cloth: soft friction noise between "
       "fibres, band-limited, with occasional snaps. Leaves: crisp, brittle grains.",
       [
           Param("material", "paper", "", choices=("paper", "cloth", "leaves", "plastic")),
           Param("moves", 5, "count", 1, 40, "number of hand movements"),
           Param("length_s", 1.5, "s", 0.2, 10, "how long the handling lasts"),
           Param("intensity", 0.6, "", 0, 1, "how vigorous"),
           Param("at", 0.03, "s", 0, None),
       ], ["Houle & Sethna 1996, Phys Rev E 54 (acoustic emission from crumpling paper)", "Cook 1997 (PhISEM)",
           "Farnell 2010"])
def rustle(ctx: Ctx, x, material, moves, length_s, intensity, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    g = gesture(rng, n, sr, moves, length_s, at)
    g = g / (g.max() + 1e-12)
    out = np.zeros(n)
    if material in ("paper", "plastic", "leaves"):
        spec = {"paper": (2500, 0.7, 1500), "plastic": (4000, 0.9, 2500), "leaves": (3200, 0.8, 1800)}[material]
        rate = (300 + 2500 * intensity) * g ** 1.5
        idx = dsp.poisson_times(rng, rate, sr)
        imp = np.zeros(n)
        imp[idx] = np.minimum(rng.pareto(1.6, len(idx)) + 0.1, 20.0) * rng.choice([-1, 1], len(idx))
        cr = dsp.bandpass(imp, spec[0], sr, spec[1]) + 0.4 * dsp.bandpass(imp, spec[2] * 0.5, sr, 0.7)
        bed = dsp.bandpass(dsp.pink(rng, n), spec[2], sr, 0.5) * g
        air = dsp.bandpass(imp, spec[0] * 2.4, sr, 0.8)       # the crisp top of each crinkle (CC0 refs: strong 8-16 kHz)
        out = _norm(cr) + 0.25 * _norm(bed) + 0.3 * _norm(air)
    else:
        fr = dsp.bandpass(dsp.pink(rng, n), 1600.0, sr, 0.45) * g * (1 + 0.4 * np.abs(smooth_random(rng, n, sr, 30)))
        snaps = np.zeros(n)
        idx = dsp.poisson_times(rng, 6 * intensity * g, sr)
        snaps[idx] = rng.uniform(0.5, 1.0, len(idx))
        out = _norm(fr) + 0.3 * _norm(dsp.bandpass(snaps, 1200.0, sr, 1.0))
    return dsp.highpass(dsp.lowpass(out, 13000.0, sr), 150.0, sr)


# =================================================================== splash
@model("splash", "sound",
       "An object dropping into water (0.10, after 'gamey and lasery, crunchy on top'). The splash is "
       "mostly a broadband, turbulent wash (the crown and the water thrown up, a few hundred ms), started by a "
       "soft impact slap. The pinch-off bubble ('bloop', Minnaert f = 3.26 / r) is real but in a splash it is "
       "strongly damped by the churning surface and barely rises in pitch, so it sits low and short under the "
       "wash; small bubbles overlap into a fizz rather than separate chirps. Droplets falling back are soft "
       "patters (low-passed, a few ms), only a few with a tiny damped bubble.",
       [
           Param("size", 0.08, "m", 0.01, 0.6, "object size"),
           Param("speed", 3.0, "m/s", 0.5, 12, "entry speed"),
           Param("bloop", 0.35, "", 0, 1, "level of the pinch-off bubble"),
           Param("droplets", 0.4, "", 0, 1, "level of the falling-back droplets"),
           Param("at", 0.03, "s", 0, None),
       ], ["Minnaert 1933", "Prosperetti & Oguz 1993", "van den Doel 2005", "Phillips et al. 2018"])
def splash(ctx: Ctx, x, size, speed, bloop, droplets, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    i0 = int(at * sr)
    out = np.zeros(n)
    E = (speed / 3.0) * (size / 0.08) ** 0.7               # relative splash energy
    # slap: short soft impact, darker for bigger objects
    L = int((0.006 + 0.03 * size) * sr)
    tt = np.arange(L) / sr
    slap = dsp.lowpass(rng.standard_normal(L), 6000.0 / (1 + 4 * size), sr) * np.exp(-tt / (0.003 + 0.012 * size)) \
        * (1 - np.exp(-tt / 0.0005))
    dsp.place(out, _norm(slap), i0, 0.6)
    # wash: the crown and thrown water, turbulent broadband noise with fast random fluctuation
    dur = 0.15 + 0.5 * min(E, 3.0) ** 0.5 * (size / 0.08) ** 0.3
    Lw = min(n - i0, int((dur * 2.5) * sr))
    tw = np.arange(Lw) / sr
    wenv = (1 - np.exp(-tw / 0.012)) * np.exp(-tw / (0.35 * dur))
    turb = np.exp(0.5 * smooth_random(rng, Lw, sr, 35))
    wash = dsp.bandpass(dsp.pink(rng, Lw), 1200.0 / (1 + 2 * size), sr, 0.35) * wenv * turb
    wash = dsp.lowpass(wash, 7000.0, sr)
    dsp.place(out, _norm(wash), i0 + int(0.004 * sr), 1.0)
    # bloop: pinch-off bubble, damped x4 by the churn, tiny pitch rise, low level
    r = np.clip(0.15 * size, 0.002, 0.06) * rng.uniform(0.85, 1.15)
    tb = at + 0.03 + 0.3 * size / max(speed, 0.5) ** 0.5
    f0 = 3.26 / r
    Lb = int(min(0.25, 6.0 / (4 * (0.13 / r + 0.0072 * r ** -1.5))) * sr)
    tbb = np.arange(Lb) / sr
    d = 4 * (0.13 / r + 0.0072 * r ** -1.5)
    bb = np.sin(2 * np.pi * np.cumsum(f0 * (1 + 0.6 * tbb)) / sr) * np.exp(-d * tbb) * (1 - np.exp(-tbb / 0.002))
    dsp.place(out, bb, int(tb * sr), bloop * 0.35)
    # fizz: many small bubbles overlapping, near-constant pitch, low level, softened top
    rate = [[0, 0], [at, 0], [at + 0.03, 400 * min(E, 3.0)], [at + 0.3, 60], [at + 0.9, 0]]
    cloud = call("bubbles", ctx.child(), rate=rate, r_min_mm=0.8, r_max_mm=max(1.5, 20 * size), size_exponent=2.6, xi=0.02)
    out += 0.18 * _norm(dsp.lowpass(cloud, 3500.0, sr))
    if droplets > 0:
        dr = np.zeros(n)
        k = int(8 + 40 * size * speed / 3)
        for _ in range(k):
            t = at + 0.1 + rng.gamma(2.0, 0.07 + 0.25 * size)
            i = int(t * sr)
            if i >= n:
                continue
            Ld = int(rng.uniform(0.002, 0.005) * sr)
            pat = dsp.lowpass(rng.standard_normal(Ld) * _hann(Ld), rng.uniform(1500, 3500), sr)
            dsp.place(dr, pat / (np.max(np.abs(pat)) + 1e-12), i, rng.uniform(0.2, 1.0))
            if rng.random() < 0.2:                       # an occasional tiny, damped drip bubble
                rr = rng.uniform(0.0015, 0.003)
                gb = bubble_grain(sr, rr, 0.02)
                gb = gb[: int(0.02 * sr)] * np.exp(-np.arange(min(len(gb), int(0.02 * sr))) / (0.004 * sr))
                dsp.place(dr, gb, i + int(0.003 * sr), 0.25 * rng.uniform(0.3, 1.0))
        out += droplets * 0.35 * _norm(dr)
    return dsp.highpass(dsp.lowpass(out, 9000.0, sr), 60.0, sr)
