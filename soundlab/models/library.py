"""Recipe-library models added in v0.9 (roadmap step 4): wind, rain, glass break, sword swing, blade clash /
metal clang, door creak and slam, body hit, bullet impact, gunshot, creature growl, rustle, splash.

Physics where the mechanism decides the sound (stick-slip hinge, shard ring, Minnaert bubbles, aeolian tones,
Friedlander blast), physically informed noise otherwise. Numbers marked (est.) are our estimates. Design
goals for the defaults: no digital artefacts, no machine-regular repetition, no small-cavity ringing, no
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
           Param("indoors", 0.0, "", 0, 1, "heard from inside through a window or wall: the wind noise is "
                 "low-passed (mass law), the whistle made at the gap itself is not"),
       ], ["Farnell 2010, Designing Sound (wind)", "Wikipedia: Aeolian sound; Strouhal number",
           "Lighthill / Curle (dipole sound power ~ U^6)"])
def wind(ctx: Ctx, x, speed, gustiness, gust_hz, howl, whistle, gap_mm, trees, rumble, indoors):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    # v0.14: gusts carry faster flurries as well (0.5-2 Hz), so the level and brightness keep moving on a
    # scale of seconds; CC0 wind refs have ~4x more envelope energy below 4 Hz than the old model
    g = smooth_random(rng, n, sr, gust_hz) + 0.5 * smooth_random(rng, n, sr, gust_hz * 5)
    g = g / (g.std() + 1e-12)
    v = np.maximum(speed * (1 + gustiness * 0.7 * g), 0.12 * speed)
    rel = v / speed
    amp = rel ** 3 / np.max(rel ** 3)
    # body: pink noise through a low-pass that opens with speed. v0.14: the corner sits at ~1.5 kHz at
    # 8 m/s and moves strongly with the gusts (f ~ v^1.5): real wind is broadband up to 2-4 kHz and gets
    # brighter as it gets louder; the old ~600 Hz corner made a dark rumble whose grain sat at 60-250 Hz
    fc = np.clip(3500.0 * rel ** 1.5 * (speed / 8.0) ** 0.5, 250, 10000)
    body = dsp.svf(dsp.pink(rng, n), fc, 0.5, sr, "lp") + 0.3 * dsp.svf(dsp.pink(rng, n), np.minimum(4 * fc, 16000), 0.5, sr, "lp")
    # turbulent eddies make the level waver irregularly at a few Hz (two unrelated random rates, shallow: a
    # deep regular flutter is the plastic-bag sound)
    eddy = np.exp(0.18 * smooth_random(rng, n, sr, 4.1) + 0.12 * smooth_random(rng, n, sr, 10.7))
    out = _norm(body * amp * eddy)
    if howl > 0:
        # broad moaning resonances (Q 2.5-4; narrow ones beat into a fast grainy flutter)
        h = np.zeros(n)
        for k, (base, q) in enumerate(((22.0, 3.0), (37.0, 4.0), (61.0, 2.5))):
            drift = 1 + 0.04 * smooth_random(rng, n, sr, 0.3)
            h += dsp.svf(dsp.white(rng, n), np.clip(base * v * drift, 60, 3000), q, sr, "bpn") * (0.6 ** k)
        out += howl * 0.5 * _norm(h * amp ** 0.8)
    if indoors > 0:
        # a wall or closed window passes little above a few hundred Hz (mass law: ~6 dB per octave more loss)
        wall = dsp.lowpass(dsp.lowpass(out, 220.0, sr, 0.6), 600.0, sr)
        out = (1 - indoors) * out + indoors * 1.5 * wall
    if whistle > 0:
        # v0.14: the edge tone locks onto the gap's own resonance, so its pitch is nearly steady (CC0 refs: a
        # thin, stable line) while its level comes and goes with the flow; only a slight pull with speed
        f_res = np.clip(0.2 * speed / (gap_mm * 1e-3), 150, 3500) * rng.uniform(0.97, 1.03)
        f_w = f_res * (1 + 0.02 * (rel - 1)) * (1 + 0.004 * smooth_random(rng, n, sr, 2.0))
        on = np.clip((rel - 0.75) / 0.4, 0, 1) ** 2  # only sounds above a threshold speed (est.)
        flick = np.exp(0.35 * smooth_random(rng, n, sr, 3.0))   # the lock is unsteady: the tone flickers
        w = dsp.svf(dsp.white(rng, n), f_w, 120.0, sr, "bpn") + 0.15 * dsp.svf(dsp.white(rng, n), f_w * 2.0, 90.0, sr, "bpn")
        out += whistle * 0.5 * _norm(w * on * flick * amp ** 0.5)
    if trees > 0:
        # continuous leaf noise, not clicks (clicks read as grain). A deep, regular 9 Hz flutter sounds like a
        # plastic bag waving, so only a shallow, irregular wobble (two unrelated rates) at a low level; off by default in wind.json (the `trees` macro brings it in).
        flut = np.exp(0.10 * smooth_random(rng, n, sr, 3.3) + 0.06 * smooth_random(rng, n, sr, 11.7))
        leaves = dsp.lowpass(dsp.bandpass(dsp.pink(rng, n), 1800.0 + 600.0 * min(speed / 10.0, 2.0), sr, 0.4),
                             5000.0, sr) * flut * amp ** 0.7
        out += trees * 0.3 * _norm(leaves)
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
    # only a gentle, slow variation (faster AM sounds like paper bags shuddering); the wash itself is stationary
    swell = np.exp(0.05 * smooth_random(rng, n, sr, 0.1))
    out = np.zeros(n)
    if near > 0:
        rate = 25.0 * R ** 0.8 * swell                    # near drops heard individually (est.)
        idx = dsp.poisson_times(rng, rate, sr)
        D = np.clip(rng.exponential(drop_mm, len(idx)), 0.3, 6.0)
        amp = (D / drop_mm) ** 1.5
        imp = np.zeros(n)
        np.add.at(imp, idx, amp * rng.uniform(0.5, 1.0, len(idx)))
        # a drop is not a one-sample click (clicks crackle on top). Its impact lasts ~0.5-2 ms
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
        # the diffuse wash (0.11): tens of thousands of distant micro-drops per second, each a soft 0.1-0.25 ms tap with a
        # random size, summed. Many independent taps average into a smooth, statistically stationary hiss with a
        # fine rain texture; no amplitude modulation is applied (the 0.10 30 Hz 'patter' AM was the shudder).
        top = 2200.0 + 4500.0 * brightness
        mrate = (25000.0 + 4000.0 * R) * swell                  # far drops per second (est.; enough to overlap)
        idx = dsp.poisson_times(rng, mrate, sr)
        imp = np.zeros(n)
        np.add.at(imp, idx, rng.exponential(1.0, len(idx)) * rng.choice([-1.0, 1.0], len(idx)))
        micro = np.zeros(n)
        for k, w_ms in enumerate((0.08, 0.15, 0.25)):
            part = imp * (rng.integers(0, 3, n) == k)
            w = max(3, int(w_ms * 1e-3 * sr))
            micro += np.convolve(part, _hann(w), "same")
        h = dsp.lowpass(dsp.highpass(micro, 400.0, sr), top, sr)
        out += hiss * (0.45 + 0.6 * min(R / 20.0, 1.0)) * _norm(h)
        # the low wash of distant splashes and run-off: stationary
        wash = dsp.lowpass(dsp.highpass(dsp.pink(rng, n), 70.0, sr), 450.0, sr)
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
    # fracture (v0.14): not one click but a cascade. The crack branches across the pane at ~1500 m/s, so a
    # 40 cm pane fails in well under a millisecond per branch, but the pieces keep cracking off and knocking
    # each other for 30-120 ms: a dense burst of bright clicks whose rate decays. CC0 refs: the break is the
    # loudest moment, with energy up to 16-20 kHz.
    L = int(0.03 * sr)
    tt = np.arange(L) / sr
    crack = dsp.highpass(rng.standard_normal(L), 1500.0, sr) * np.exp(-tt / (0.004 + 0.006 * energy))
    f, a = mode_set("plate", materials.get("glass"), size, size * 0.75, h, 0, 30, None, 0.3, rng)
    pane = _mode_ir(sr, f, np.full(len(f), 0.03), a, max_len=int(0.1 * sr))
    frac = 0.2 * _norm(pane)
    frac[:L] += 0.9 * _norm(crack)
    Lc = int((0.03 + 0.09 * energy) * sr)
    tc = np.arange(Lc) / sr
    rate = (2500.0 + 4000.0 * energy) * np.exp(-tc / (0.012 + 0.03 * energy))
    idx = dsp.poisson_times(rng, rate, sr)
    clicks = np.zeros(Lc)
    clicks[idx] = np.minimum(rng.pareto(1.5, len(idx)) + 0.2, 8.0) * rng.choice([-1.0, 1.0], len(idx))
    clicks = dsp.highpass(clicks, 2500.0, sr) + 0.5 * dsp.bandpass(clicks, rng.uniform(4000, 7000), sr, 1.2)
    frac = np.pad(frac, (0, max(0, Lc - len(frac))))
    frac[:Lc] += 0.6 * _norm(clicks)
    dsp.place(out, frac, i0, 0.5 + 0.5 * energy)
    # shards
    E, rho, nu = 60e9, 2500, 0.24
    k = math.sqrt(E * h * h / (12 * rho * (1 - nu * nu)))
    # shard sizes: power law, mostly 1-3 cm, a few up to half the pane (est.)
    s = np.clip(size * 0.06 * rng.random(int(shards)) ** (-1 / 2.0), 0.008, size * 0.5)
    shard_out = np.zeros(n)
    floor_t60 = {"hard": 1.0, "wood": 0.6, "soft": 0.25}[floor]
    floor_lp = {"hard": 16000.0, "wood": 8000.0, "soft": 3000.0}[floor]
    for si in s:
        # pieces break off all over the pane (and off the frame) and knock into each other on the way down, so
        # the first impacts come right after the break and keep coming: each piece falls a random part of the
        # height (v0.14; one common fall time left a silent gap that does not happen in real breaks)
        hf = height * rng.uniform(0.02, 1.0) ** 1.5
        t = at + rng.uniform(0.002, 0.04) + math.sqrt(2 * hf / G)
        v = math.sqrt(2 * G * max(hf, 0.05)) * rng.uniform(0.6, 1.0)
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
            m = max(3, int(rng.uniform(0.02, 0.06) * 1e-3 * sr))
            clk = np.zeros(m + 96)
            clk[:m] = _hann(m)
            clk = _norm(dsp.bandpass(clk, rng.uniform(5000, 12000), sr, 0.8))
            gk = np.zeros(max(len(g), len(clk)))
            if np.any(g):
                gk[:len(g)] += 0.6 * _norm(g)
            gk[:len(clk)] += 0.5 * clk
            dsp.place(shard_out, gk, i, A)
            v *= rng.uniform(0.2, 0.4)
            t += 2 * v / G + rng.uniform(0.002, 0.01)
    # tinkles of pieces sliding and settling: they start while the shower is still landing and thin out
    # gradually (no gap before them)
    tail = np.zeros(n)
    t_end = at + math.sqrt(2 * height / G)
    for _ in range(int(shards * 0.4)):
        t = at + 0.05 + rng.exponential(0.6 * (t_end - at) + 0.1)
        fr = rng.uniform(3000, 11000) * np.array([1.0, 1.62])
        dsp.place(tail, _mode_ir(sr, fr, [0.04, 0.03], [1, .5]), int(t * sr),
                  rng.uniform(0.05, 0.3) * math.exp(-max(0.0, t - t_end) / 0.4))
    # big pieces land flat and are choked by the floor: their low modes hardly ring (high-pass, est.)
    debris = dsp.highpass(dsp.lowpass(_norm(shard_out) + 0.3 * _norm(tail), floor_lp, sr), 1000.0, sr)
    out += 0.2 * _norm(debris) * (0.7 + 0.3 * energy)
    return dsp.highpass(dsp.lowpass(out, 18000.0, sr), 150.0, sr)


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
    # sword (v0.14): a thin blade radiates poorly below its coincidence frequency, the grip and the other blade
    # choke the ring quickly, and the edge-on-edge contact itself is a bright burst (CC0 clash refs: most energy
    # 4-12 kHz, -20 dB after ~0.12 s). radiate/crash/shelf are only set for the sword, so the others are unchanged.
    "sword": dict(shape="bar", length=0.75, width=0.04, thickness=0.0045, damp=30.0, contact=0.05, hp=120.0,
                  radiate=True, crash=0.8, shelf=False),
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
    if ob.get("radiate"):
        # radiation efficiency of a thin plate/blade falls off below the coincidence frequency
        # f_c = c^2 / (1.81 c_L h) (~2.8 kHz for 4.5 mm steel): the low bending modes barely reach the air
        mat = materials.get("steel")
        f_c = C ** 2 / (1.81 * math.sqrt(mat["E"] / mat["rho"]) * ob["thickness"])
        out = dsp.highpass(out, 1.2 * f_c, sr, 0.6)
    if ob.get("crash", 0) > 0:
        # edge-on-edge contact: a bright, very short grinding burst (a few ms, rough), louder for harder blows
        L = int(0.04 * sr)
        tt = np.arange(L) / sr
        burst = dsp.highpass(rng.standard_normal(L), 2500.0, sr) * np.exp(-tt / (0.004 + 0.002 * min(velocity / 3, 2))) \
            * (1 - np.exp(-tt / 0.0003)) * (1 + 0.5 * smooth_random(rng, L, sr, 400))
        # plus the dull knock of the two blades' bodies and hilts (low, damped within ~15 ms)
        knock = dsp.lowpass(rng.standard_normal(L), 500.0, sr) * np.exp(-tt / 0.006)
        cr = np.zeros(n)
        dsp.place(cr, _norm(burst) + 0.05 * _norm(knock), i0, 1.0)
        out = _norm(out) + ob["crash"] * min(velocity / 3, 1.5) * cr
    if ob.get("shelf", True):
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
       "A heavy door slammed shut. The mass is what you hear (without it, it sounds like two wood planks hitting). "
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
    # shock: positive-phase length of the muzzle shock (s); roar: propellant combustion decay (s) and its
    # brightness (Hz); boom: low gas push level/decay; sonic: supersonic N-wave level; tail: reflective decay
    # time (s) at 'street'. Estimates: Maher 2006, Beck et al. 2011; tail ratios from CC0 gunshot/shotgun
    # recordings (level 0.3-0.6 s after the shot ~-12 dB, 0.6-1 s ~-15..-20 dB re the first 5 ms).
    "pistol": dict(shock=0.00025, roar_tau=0.018, roar_hz=3500.0, boom=0.1, boom_tau=0.025, sonic=0.35, tail=2.6,
                   slide=True),
    "rifle": dict(shock=0.00035, roar_tau=0.024, roar_hz=4200.0, boom=0.12, boom_tau=0.03, sonic=1.0, tail=3.0,
                  slide=False),
    "shotgun": dict(shock=0.0005, roar_tau=0.034, roar_hz=2600.0, boom=0.22, boom_tau=0.045, sonic=0.0, tail=3.2,
                    slide=False),
}


@model("gunshot", "sound",
       "A firearm shot (not an air-gun punch). What makes it a firearm, in order: "
       "(1) the muzzle shock: a near-instant pressure jump (microseconds) lasting a fraction of a millisecond, "
       "so the first millisecond is very loud and full of high frequencies; (2) the propellant combustion "
       "roar: bright, broadband, turbulent flame/gas, tens of ms; (3) only a modest low gas push (too much "
       "low-mid is the 'poof' of an air cannon); (4) all of it saturates the microphone/ear: hard, almost "
       "painful clipping and compression (oversampled, so no aliasing); (5) the supersonic N-wave for rifles; "
       "(6) the dry clack of the action; (7) the reflections: loud slapback off walls/ground and a long "
       "reverberant tail, which is what gives a gunshot its size (CC0 recordings: still ~-12 dB 0.3-0.6 s "
       "after the shot, ~-15..-20 dB up to 1 s).",
       [
           Param("calibre", "pistol", "", choices=tuple(GUNS)),
           Param("crack", 0.5, "", 0, 1, "supersonic crack level (scaled by calibre; shotguns have none)"),
           Param("drive", 3.0, "", 0, 6, "saturation of the mic/ear: more = louder, denser, harsher"),
           Param("body", 0.5, "", 0, 1, "combustion roar + low push level relative to the shock"),
           Param("echo", 0.8, "", 0, 1, "slapback and tail level"),
           Param("space", "street", "", choices=("street", "room", "open"), doc="street (walls near: strong "
                 "slaps, ~2 s tail), room (dense, short, boomy), open field (late treeline/hill echoes, long tail)"),
           Param("mech", 0.4, "", 0, 1, "action clack level (slide/bolt, dry)"),
           Param("at", 0.02, "s", 0, None),
       ], ["Wikipedia: Blast wave (Friedlander)", "Maher 2006, Modeling and signal processing of acoustic gunshot "
           "recordings", "Beck, Nakasone & Marr 2011 (muzzle blast variation)", "Farnell 2010 (guns)"])
def gunshot(ctx: Ctx, x, calibre, crack, drive, body, echo, space, mech, at):
    from scipy import signal
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    g = GUNS[calibre]
    i0 = int(at * sr)
    os = 4
    so = sr * os
    L = min(n - i0, int(0.25 * sr)) * os
    tt = np.arange(L) / so
    # (1) muzzle shock: instant rise, Friedlander decay over a fraction of a ms, plus the shock front's own
    # broadband turbulence (first ~1 ms); the low part of the pulse is trimmed (that is the 'poof')
    shock = friedlander(L, so, g["shock"] * rng.uniform(0.9, 1.1))
    shock = dsp.highpass(shock, 250.0, so)
    front = dsp.highpass(rng.standard_normal(L), 1500.0, so) * np.exp(-tt / 0.0006)
    # (2) combustion roar: bright broadband, fast attack, tens of ms
    roar = dsp.highpass(rng.standard_normal(L), 300.0, so)
    roar = roar + 0.8 * dsp.bandpass(roar, g["roar_hz"], so, 0.6)
    roar *= np.exp(-tt / g["roar_tau"]) * (1 - np.exp(-tt / 0.0002))
    # (3) modest low gas push
    boom = dsp.lowpass(dsp.lowpass(rng.standard_normal(L), 200.0, so), 200.0, so) * np.exp(-tt / g["boom_tau"]) \
        * (1 - np.exp(-tt / 0.001))
    src = 1.0 * _norm(shock) + 0.6 * _norm(front) + body * (1.0 * _norm(roar) + g["boom"] * _norm(boom))
    # (4) saturation: hard soft-clip, then a second stage (mic preamp) - the first 20-40 ms sit at the ceiling
    if drive > 0:
        k = 1.0 + 2.5 * drive
        src = np.tanh(k * src / np.max(np.abs(src))) / np.tanh(k)
        src = np.clip(1.2 * src, -1.0, 1.0)
    dry = signal.resample_poly(src, 1, os)[: L // os]
    out = np.zeros(n)
    dsp.place(out, dry, i0, 1.0)
    if crack > 0 and g["sonic"] > 0:                  # (5) N-wave just ahead of the blast
        Ln = int(0.0004 * sr) + 2
        cr = np.zeros(n)
        dsp.place(cr, np.linspace(1, -1, Ln), max(0, i0 - int(0.0015 * sr)), 1.0)
        out += crack * g["sonic"] * 0.9 * dsp.lowpass(cr, 16000.0, sr)
    if mech > 0:                                      # (6) dry action: slide back + forward (pistol), bolt tick
        times = (0.006, 0.03) if g["slide"] else (0.05,)
        for tm in times:
            Lm = int(0.006 * sr)
            ck = dsp.highpass(rng.standard_normal(Lm), 1200.0, sr) * np.exp(-np.arange(Lm) / (0.0012 * sr))
            dsp.place(out, ck / (np.max(np.abs(ck)) + 1e-12), i0 + int(tm * rng.uniform(0.85, 1.15) * sr), 0.18 * mech)
    if echo > 0:                                      # (7) reflections: they give the size
        sp = {"street": dict(slaps=(0.014, 0.032, 0.055, 0.085, 0.13, 0.2, 0.29), lvl=0.85, decay=0.8,
                             tail=1.25 * g["tail"], tail_db=-3.0, lp=4500.0),
              "room": dict(slaps=(0.004, 0.007, 0.011, 0.016, 0.022, 0.03), lvl=0.8, decay=0.8,
                           tail=0.35 * g["tail"] + 0.2, tail_db=-4.0, lp=5000.0),
              "open": dict(slaps=(0.012, 0.35, 0.62, 0.95), lvl=0.6, decay=0.75,
                           tail=g["tail"] * 1.4, tail_db=-10.0, lp=3500.0)}[space]
        slap_src = dsp.lowpass(out, sp["lp"], sr)
        wet = np.zeros(n)
        for k, d in enumerate(sp["slaps"]):
            dd = int(d * rng.uniform(0.9, 1.1) * sr)
            if dd < n:
                wet[dd:] += slap_src[:n - dd] * sp["lvl"] * (sp["decay"] ** k) * rng.uniform(0.75, 1.0)
        # diffuse reverberant tail: stationary noise under an exponential (RT60 = tail), darkening as it decays
        Lt = n - i0
        tl = np.arange(Lt) / sr
        nz = rng.standard_normal(Lt)
        bright = dsp.lowpass(nz, sp["lp"] * 0.7, sr)
        dark = dsp.lowpass(nz, sp["lp"] * 0.25, sr)
        mix = np.exp(-tl / (0.25 * sp["tail"]))
        tail = (mix * bright / np.std(bright) + (1 - mix) * dark / np.std(dark)) * np.exp(-6.91 * tl / sp["tail"]) \
            * (1 - np.exp(-tl / 0.008))
        tail = dsp.highpass(tail, 150.0, sr)
        dsp.place(wet, tail * 10 ** (sp["tail_db"] / 20) / (np.max(np.abs(tail[: int(0.05 * sr)])) + 1e-12), i0 + int(0.004 * sr), 1.0)
        out += echo * wet
    # the mic hears shot and reflections together: one last soft limit so the shock stays the loudest moment
    out = dsp.highpass(out, 45.0, sr)
    return np.tanh(1.6 * out / (np.max(np.abs(out[i0:i0 + int(0.003 * sr)])) + 1e-12)) / np.tanh(1.6)


# =================================================================== creature growl
def glottal_source(rng, n, sr, f0_curve, amp_curve, jitter, shimmer, subharm, oq=0.6, ta=0.04):
    """Voiced source, one cycle at a time, LF-style (Liljencrants-Fant shape, simplified): the glottal flow
    opens smoothly (raised cosine to its peak at ~0.66 of the open phase), falls to the excitation instant Te,
    then closes through an exponential return phase of length `ta` x period instead of slamming shut. The
    return phase is what keeps a low voice from sounding like a train of clicks (a hard closure at 40-80 Hz is
    a ratchet / stick-slip judder). The excitation is the flow derivative. Period jitter, shimmer and period
    doubling (alternating long/strong and short/weak cycles = subharmonics) are cycle-to-cycle.
    Returns (excitation, flow)."""
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
        A = amp_curve[i] * max(0.1, 1 + shimmer * rng.standard_normal()) * (1 - 0.5 * subharm * (k % 2))
        if A > 1e-4:
            Te = max(4, int(oq * period * rng.uniform(0.92, 1.08)))
            Tp = max(2, int(0.7 * Te))
            Ta = max(2.0, ta * period)
            Lc = int(Te + 5 * Ta)
            u = np.arange(Lc, dtype=float)
            g = np.where(u < Tp, 0.5 * (1 - np.cos(np.pi * u / Tp)),
                         np.where(u < Te, 0.5 + 0.5 * np.cos(np.pi * (u - Tp) / max(Te - Tp, 1)), 0.0))
            ge = 0.5 + 0.5 * np.cos(np.pi * (Te - 1 - Tp) / max(Te - Tp, 1))     # flow left at Te
            g = np.where(u >= Te, ge * np.exp(-(u - Te) / Ta), g)
            dsp.place(flow, g, i, A)
            dsp.place(out, np.diff(g, append=0.0), i, A)
        t += period
        k += 1
    return out, flow


def trill_gate(rng, n, sr, rate_hz, depth, irregular=0.2):
    """Uvular / ventricular-fold trill: the soft tissue flaps in the airflow at ~18-35 Hz, briefly choking it
    each cycle. Cycle by cycle (random period and depth), smooth closures, so it is a fleshy flutter, not a
    periodic tick. Returns a gain curve (1 = open)."""
    gate = np.ones(n)
    t = float(rng.uniform(0, sr / rate_hz))
    while t < n:
        P = sr / rate_hz * (1 + irregular * rng.standard_normal())
        P = max(P, 0.4 * sr / rate_hz)
        Lc = int(P * rng.uniform(0.35, 0.6))             # closed part of the cycle
        d = depth * rng.uniform(0.55, 1.0)
        i = int(t)
        seg = gate[i:i + Lc]
        seg *= 1 - d * _hann(Lc)[: len(seg)]
        t += P
    return gate


@model("growl", "sound",
       "A creature growl, guttural and voiced. "
       "Source-filter. Source: LF-style vocal-fold pulses (smooth opening, soft exponential return phase, no hard "
       "clicks) with chaotic pitch, jitter, shimmer and strong period doubling (subharmonics); a uvular / "
       "ventricular-fold trill at ~18-35 Hz chokes the airflow irregularly each cycle (the 'uvulation' - CC0 "
       "tiger/growl recordings flutter at 18-22 Hz); wet throat turbulence (noise gated by the airflow and the "
       "trill). Filter: a long open tract (tube formants (2n-1) c / 4L) with BROAD resonances: narrow ones ring "
       "after every pulse like a wooden board, which was the 'chair on a floor'. A chest band carries the bass.",
       [
           Param("f0", 70.0, "Hz", 25, 400, "voice pitch (big cat ~ 50-120 Hz plus subharmonics; dog ~ 150-300)"),
           Param("size", 1.0, "", 0.3, 3, "creature size (1 = ~50 cm vocal tract; formants ~ 1/size)"),
           Param("roughness", 0.6, "", 0, 1, "jitter, shimmer, subharmonics and pitch chaos"),
           Param("flutter_hz", 24.0, "Hz", 8, 45, "uvular/ventricular trill rate"),
           Param("trill", 0.6, "", 0, 1, "trill depth (0 = plain voice)"),
           Param("wet", 0.5, "", 0, 1, "wet throat turbulence level"),
           Param("snarl", 0.3, "", 0, 1, "open, bright snarl (raised formants, more breath)"),
           Param("length_s", 1.6, "s", 0.2, 6, "growl length"),
           Param("rise", 0.4, "", 0, 1, "swells: pitch/level rise toward the middle"),
           Param("at", 0.05, "s", 0, None),
       ], ["Titze 1994 (voice source)", "Fant, Liljencrants & Lin 1985 (LF glottal model)", "Fitch 1997 (formant "
           "dispersion and body size)", "Herzel et al. 1995 (subharmonics)", "Peters et al. 2009 (big cat vocal "
           "anatomy)", "Sakakibara et al. 2004 (ventricular-fold vibration, growl voice)"])
def growl(ctx: Ctx, x, f0, size, roughness, flutter_hz, trill, wet, snarl, length_s, rise, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    t = np.arange(n) / sr
    u = np.clip((t - at) / length_s, 0, 1)
    on = (t >= at) & (u < 1)
    # v0.14 envelope: a growl is one exhalation, so it builds in ~0.1-0.3 s and then sags as the breath runs
    # out (CC0 growl refs), rather than a symmetric swell; the release is a soft 15% of the length
    attack = np.clip((t - at) / rng.uniform(0.1, 0.3), 0, 1) ** 1.5
    sag = np.exp(-u * (0.9 + 0.8 * (1 - rise)))
    release = np.clip((1 - u) / 0.15, 0, 1) ** 1.5
    env = np.where(on, attack * sag * release, 0.0)
    env *= (1 + rise * 0.25 * np.sin(np.pi * u)) * np.exp(0.2 * smooth_random(rng, n, sr, 1.5))   # swells
    chaos = 0.03 + 0.08 * roughness
    fcurve = f0 * (1 + rise * 0.2 * np.sin(np.pi * u)) * np.exp(chaos * smooth_random(rng, n, sr, 5)
                                                                + 0.02 * smooth_random(rng, n, sr, 1))
    gate = trill_gate(rng, n, sr, flutter_hz, trill) if trill > 0 else np.ones(n)
    # v0.14 source: a longer open phase and a slow return phase. The closure is soft tissue meeting soft
    # tissue, so each cycle carries little energy above ~1 kHz; sharp closures plus the tract made a
    # stick-slip judder: a pushed object, not a throat
    # Cycle-to-cycle jitter kept moderate (~1-4%): the CC0 growls show clear harmonics up to 2-5 kHz, their
    # roughness comes from subharmonics and the trill; 6-9% jitter smeared the harmonics into scraping noise.
    src, flow = glottal_source(rng, n, sr, fcurve, env * gate, jitter=0.01 + 0.035 * roughness,
                               shimmer=0.05 + 0.15 * roughness, subharm=0.3 + 0.6 * roughness, oq=0.75, ta=0.12)
    # wet throat: turbulent noise riding on the airflow and choked by the trill (it gurgles with it)
    turb = dsp.bandpass(rng.standard_normal(n), 700.0, sr, 0.5) * (0.25 + dsp.lowpass(flow, 300.0, sr)) * env * gate
    Lt = 0.5 * size
    s = 1.0 + 0.25 * snarl
    # v0.14 tract: the jaw and tongue move during the growl, so the formants move (F1 most, ~+-20% with the
    # mouth opening as the growl gets louder, plus a slow wander); fixed resonances ring like a board.
    # Broad resonances, and the upper ones weaker: big growls have little energy above 1-2 kHz.
    jaw = np.clip(0.5 * env / (env.max() + 1e-12) + 0.15 * smooth_random(rng, n, sr, 2.0), 0, 1)
    srcn = src / (np.max(np.abs(src)) + 1e-12)
    turbn = turb / (np.max(np.abs(turb)) + 1e-12)
    exc = srcn + (0.15 + 0.5 * wet + 0.2 * snarl) * turbn
    y = 0.05 * exc
    for k in range(6):
        fk = (2 * k + 1) * C / (4 * Lt) * s
        if fk > 5500:
            break
        move = (0.8 + 0.4 * jaw) if k == 0 else (0.92 + 0.16 * jaw) * (1 + 0.04 * smooth_random(rng, n, sr, 0.7))
        g = 10 ** ((-4.0 * k + 1.5 * snarl * k) / 20)
        y = y + g * dsp.svf(exc, np.clip(fk * move, 40, 0.45 * sr), 1.8 + 0.3 * k, sr, "bp")
    chest = dsp.lowpass(dsp.lowpass(srcn, 220.0, sr), 220.0, sr)       # guttural bass: fundamental + subharmonic
    y = _norm(y) + 0.8 * _norm(chest)
    return dsp.highpass(dsp.lowpass(dsp.lowpass(y, 3500.0, sr, 0.6), 8000.0, sr), 30.0, sr)


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
        # cloth (v0.14). Fabric rubbing on fabric is a smooth friction hiss whose level follows the hand's speed:
        # each stroke swells and fades over ~0.25-0.6 s, strokes overlap (the hand reverses, it does not stop),
        # and the spectrum is broad: rising to 2-8 kHz and still strong at 10-16 kHz (CC0 cloth refs). The
        # old 1.6 kHz band-pass with short, separated pushes sounded muffled and chopped into bursts.
        g = np.zeros(n)
        m = max(1, int(moves))
        span = max(length_s, 0.3)
        starts = at + np.sort(rng.uniform(0, 1, m)) * span * 0.85
        starts = at + (np.arange(m) + 0.5 * rng.uniform(-0.6, 0.6, m)) * span / m if m > 1 else starts
        gap = span / m
        for t0 in starts:
            L = int(np.clip(gap * rng.uniform(0.65, 0.9), 0.15, 0.6) * sr)
            up = rng.uniform(0.35, 0.65)                  # rise and fall need not be symmetric
            k = np.arange(L) / L
            shape = np.where(k < up, np.sin(0.5 * np.pi * k / up) ** 2, np.cos(0.5 * np.pi * (k - up) / (1 - up)) ** 2)
            dsp.place(g, shape * rng.uniform(0.5, 1.0), int(max(t0, 0) * sr))
        tt = np.arange(n) / sr
        held = ((tt >= at) & (tt < at + span)).astype(float)
        held = np.convolve(held, _hann(int(0.1 * sr)) / (0.05 * sr), "same")
        g = g / (g.max() + 1e-12)
        g = np.maximum(g, 0.06 * held)                     # fibres keep brushing between strokes
        hiss = dsp.lowpass(dsp.lowpass(dsp.white(rng, n), 9000.0, sr, 0.6), 14000.0, sr)
        hiss = hiss + 0.5 * dsp.bandpass(dsp.pink(rng, n), 900.0, sr, 0.5)
        fr = hiss * g * (1 + 0.25 * smooth_random(rng, n, sr, 25))
        # fibre stick-slip: dense micro-crackle riding on the speed, which gives the rub its crisp grain
        idx = dsp.poisson_times(rng, (400 + 1600 * intensity) * g ** 1.5, sr)
        cr = np.zeros(n)
        cr[idx] = rng.pareto(2.5, len(idx)) * rng.choice([-1, 1], len(idx))
        cr = dsp.highpass(cr, 2500.0, sr)
        snaps = np.zeros(n)
        idx = dsp.poisson_times(rng, 6 * intensity * g, sr)
        snaps[idx] = rng.uniform(0.5, 1.0, len(idx))
        # a soft low bump from the arm/body moving the garment
        body = dsp.lowpass(dsp.brown(rng, n), 200.0, sr) * g
        out = _norm(fr) + 0.25 * _norm(cr) + 0.15 * _norm(dsp.bandpass(snaps, 1200.0, sr, 1.0)) + 0.12 * _norm(body)
        return dsp.highpass(dsp.lowpass(out, 16000.0, sr), 60.0, sr)
    return dsp.highpass(dsp.lowpass(out, 13000.0, sr), 150.0, sr)


# =================================================================== splash
@model("splash", "sound",
       "An object dropping into water. In order: "
       "(1) the impact: the object slaps the surface - a sharp, short broadband crack (a few ms), with a dull "
       "thud for big objects; (2) the crown and sheet of water thrown up: a short, smooth noise burst (no "
       "amplitude modulation - a fluttering noise envelope is what read as wing flaps); (3) the air cavity "
       "collapsing into the pinch-off bubble: a damped low 'bloop' (Minnaert f = 3.26 / r) with the small "
       "natural pitch rise; (4) the spray falling back: many small droplets (tens to hundreds) over ~0.3-1 s, "
       "each a soft tick with, often, its own tiny entrained bubble (a 'plip' at 1-6 kHz that dies in a few ms). "
       "Overlapping, they make the dense patter that says 'water'.",
       [
           Param("size", 0.08, "m", 0.01, 0.6, "object size"),
           Param("speed", 3.0, "m/s", 0.5, 12, "entry speed"),
           Param("bloop", 0.5, "", 0, 1, "level of the cavity-collapse bubble"),
           Param("droplets", 0.8, "", 0, 1, "level of the falling-back spray"),
           Param("at", 0.03, "s", 0, None),
       ], ["Minnaert 1933", "Prosperetti & Oguz 1993", "van den Doel 2005", "Phillips et al. 2018",
           "Franz 1959 (splashes as sources of sound in liquids)"])
def splash(ctx: Ctx, x, size, speed, bloop, droplets, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    i0 = int(at * sr)
    out = np.zeros(n)
    E = (speed / 3.0) * (size / 0.08) ** 0.7               # relative splash energy
    # (1) impact slap: sharp broadband crack + a low 'plunk' (v0.14: the water mass pushed aside and the
    # opening air cavity give a strong 60-250 Hz thump; CC0 splash refs carry as much energy at 63-125 Hz as
    # at 1-2 kHz, the old thud was ~15 dB too weak)
    L = int(0.06 * sr)
    tt = np.arange(L) / sr
    crack = dsp.highpass(rng.standard_normal(L), 400.0, sr) * np.exp(-tt / (0.0012 + 0.004 * size)) \
        * (1 - np.exp(-tt / 0.0002))
    f_pl = np.clip(25.0 / max(size, 0.02) ** 0.5, 50.0, 250.0) * rng.uniform(0.9, 1.1)   # cavity/mass thump (est.)
    plunk = np.sin(2 * np.pi * f_pl * tt * (1 + 2.0 * tt)) * np.exp(-tt / (0.012 + 0.04 * size)) \
        * (1 - np.exp(-tt / 0.002))
    thud = dsp.lowpass(rng.standard_normal(L), 300.0, sr) * np.exp(-tt / (0.004 + 0.02 * size))
    dsp.place(out, _norm(crack) + 0.6 * _norm(plunk) + 0.4 * _norm(thud), i0, 1.0)
    # (2) crown / sheet: smooth broadband burst, single envelope (no AM)
    dur = 0.04 + 0.12 * min(E, 3.0) ** 0.5
    Lw = int(dur * 4 * sr)
    tw = np.arange(Lw) / sr
    sheet = dsp.lowpass(dsp.highpass(rng.standard_normal(Lw), 300.0, sr), 7000.0 / (1 + 2 * size), sr) \
        * (1 - np.exp(-tw / 0.004)) * np.exp(-tw / dur)
    dsp.place(out, _norm(sheet), i0 + int(0.002 * sr), 0.45)
    # (3) cavity collapse: pinch-off bubble, damped x2, natural small pitch rise (xi ~ 0.1)
    r = np.clip(0.15 * size, 0.002, 0.06) * rng.uniform(0.85, 1.15)
    tb = at + 0.03 + 0.3 * size / max(speed, 0.5) ** 0.5
    d = 2 * (0.13 / r + 0.0072 * r ** -1.5)
    Lb = int(min(0.3, 6.0 / d) * sr)
    tbb = np.arange(Lb) / sr
    f0 = 3.26 / r
    bb = np.sin(2 * np.pi * np.cumsum(f0 * (1 + 0.1 * d * tbb)) / sr) * np.exp(-d * tbb) * (1 - np.exp(-tbb / 0.002))
    dsp.place(out, bb, int(tb * sr), bloop * 0.45)
    # (4) spray falling back. v0.14: the water thrown up flies for 2 v_up / g (v_up ~ 0.6 x entry speed, est.):
    # ~0.3-0.8 s, spread over a wide range, so the fall-back lasts about a second, not a few hundred ms
    # ("too fast"). Under the separate drops, the churned water and bubble cloud give a continuous, decaying
    # wash; without it the tail fell apart into isolated ticks with silence between them.
    if droplets > 0:
        dr = np.zeros(n)
        k = int(40 + 220 * min(E, 4.0) * (size / 0.08) ** 0.3)
        v_up = 0.6 * speed * (size / 0.08) ** 0.15
        t_peak = 0.08 + 2 * v_up / G * 0.6
        for _ in range(k):
            t = at + 0.03 + rng.gamma(2.0, t_peak / 1.0)
            i = int(t * sr)
            if i >= n - 10:
                continue
            a = rng.pareto(2.0) + 0.2
            a = min(a, 3.0) * np.exp(-max(0.0, t - at - t_peak) / 0.3)
            Ld = max(4, int(rng.uniform(0.0005, 0.0015) * sr))       # soft tick
            tick = dsp.lowpass(rng.standard_normal(Ld) * _hann(Ld), rng.uniform(2500, 6000), sr)
            dsp.place(dr, tick / (np.max(np.abs(tick)) + 1e-12), i, 0.35 * a)
            if rng.random() < 0.6:                                   # entrained bubble 'plip'
                rr = rng.uniform(0.0006, 0.0025)
                gb = bubble_grain(sr, rr, 0.1)
                Lg = min(len(gb), int(0.012 * sr))
                gb = gb[:Lg] * (1 - np.exp(-np.arange(Lg) / (0.0003 * sr)))
                dsp.place(dr, gb, i + int(rng.uniform(0.0005, 0.003) * sr), a * rng.uniform(0.4, 1.0) * (rr / 0.0015) ** 0.5)
        # the wash: dense small bubbles and churn, rate and level decaying over ~1 s
        t = np.arange(n) / sr
        # it swells as the thrown-up water lands (peak near t_peak) and dies away after it
        tr = np.maximum(t - at, 0)
        decay = (0.4 * np.exp(-tr / 0.12) + 0.6 * (tr / t_peak) ** 1.5 * np.exp(1.5 * (1 - tr / t_peak))
                 * np.where(tr > t_peak, np.exp(-(tr - t_peak) / 0.5), 1.0)) * np.clip(tr / 0.03, 0, 1) * (t >= at)
        churn = dsp.lowpass(dsp.highpass(rng.standard_normal(n), 250.0, sr), 4500.0, sr) * decay
        cloud = np.zeros(n)
        for j in dsp.poisson_times(rng, 900.0 * decay, sr):
            rr = rng.uniform(0.001, 0.004)
            gb = bubble_grain(sr, rr, 0.12)
            dsp.place(cloud, gb[: int(0.01 * sr)], j, rng.uniform(0.2, 1.0))
        wash = 0.6 * _norm(churn) + 0.5 * _norm(cloud)
        out += droplets * (0.3 * _norm(dr) + 0.35 * _norm(wash))
    return dsp.highpass(dsp.lowpass(out, 14000.0, sr), 40.0, sr)
