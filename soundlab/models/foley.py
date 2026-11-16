"""Foley models added in v0.8: a body-weight footstep and a real metal coin (drop / jingle).

Written after the reference v0.7 notes: the v0.7 footstep (two hard hits on an oak plate) read as tapping on wood,
and the coin recipe was the retro arcade blip. Numbers marked (est.) are our estimates; the rest cite sources.
Shape of both models is checked against CC0 Freesound recordings kept OUTSIDE the repo (reference only).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .. import dsp
from ..core import Ctx, Param, model

G = 9.81


def _hann(m: int) -> np.ndarray:
    m = max(int(m), 2)
    return np.sin(np.pi * np.arange(m) / m) ** 2


def _push(m: int) -> np.ndarray:
    """Force pulse of a sole: half-sine (the force jumps in slope at first contact, so it carries more
    mid-frequency 'pat' than a Hann pulse of the same length)."""
    m = max(int(m), 2)
    return np.sin(np.pi * np.arange(m) / m)


def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)


def _mode_ir(sr, freqs, t60s, amps, phases=None, max_len=None):
    """Sum of exponentially decaying sines (a modal impulse response)."""
    L = int(max(t60s) * sr) + 1
    if max_len:
        L = min(L, max_len)
    t = np.arange(L) / sr
    ir = np.zeros(L)
    for i, (f, t60, a) in enumerate(zip(freqs, t60s, amps)):
        if f >= sr * 0.45:
            continue
        ph = 0.0 if phases is None else phases[i]
        ir += a * np.exp(-6.91 * t / t60) * np.sin(2 * np.pi * f * t + ph)
    return ir


def _conv(x, ir):
    if not np.any(x):
        return np.zeros_like(x)
    return signal.fftconvolve(x, ir)[: len(x)]


def _hann_spectrum(f, T):
    """|spectrum| of a raised-cosine (Hann) contact pulse of length T, normalised to 1 at DC."""
    u = np.asarray(f) * T
    s = np.sinc(u) / np.maximum(np.abs(1 - u ** 2), 1e-3)
    return np.minimum(np.abs(s), 1.0)


# =================================================================== footstep
# Sole/shoe table. heel_ms is the duration of the heel-strike force transient: heel mass (~5-8 kg effective,
# est.) on the sole's compliance. Gait-lab force plates show the heel-strike transient rising in ~10-50 ms
# (shod walking; Whittle 1999, Gait Analysis; Lieberman et al. 2010, Nature 463). A Hann pulse of length T
# has almost no energy above ~2/T Hz, so a foam trainer (T ~ 25 ms) can only thump (< ~100 Hz), a hard
# heel (~5 ms) reaches ~400 Hz and its tip adds a click. Values are our estimates within that range.
SHOES = {
    "trainer": dict(heel_ms=22.0, toe_ms=16.0, slap=0.35, slap_hz=2800.0, click=0.0, click_hz=2600.0,
                    scuff_hz=1300.0, toe_gain=0.5),
    "boot": dict(heel_ms=12.0, toe_ms=10.0, slap=0.55, slap_hz=5000.0, click=0.18, click_hz=3000.0,
                 scuff_hz=2000.0, toe_gain=0.55),
    "heel": dict(heel_ms=6.0, toe_ms=7.0, slap=0.4, slap_hz=6000.0, click=2.6, click_hz=4200.0,
                 scuff_hz=2800.0, toe_gain=0.4),
}
# Floor table: (Hz, t60 s) of the floor's own modes. A joisted timber floor rings in the ~70-250 Hz range
# (CC0 wood-floor refs: strongest lines at 80-130 and ~220 Hz, energy below 150 Hz within 1-2 dB of the
# total) and is heavily damped (refs: -20 dB in 5-20 ms, -40 dB in ~100-240 ms with the room). Concrete/stone
# is a massive slab: almost no floor ring, the sound is the shoe, the click and grit. Gravel: no floor ring,
# dense crunch grains (refs: energy peak 1-2.5 kHz, centroid ~2 kHz, crunch lasting ~0.1-0.2 s per step).
SURFACES = {
    "wood": dict(modes=[(82, .05), (105, .05), (131, .045), (165, .04), (220, .04), (290, .035), (380, .03),
                        (500, .03), (650, .025), (850, .02), (1100, .02), (1450, .015), (1900, .012)],
                 floor=1.0, board=[(900, .03), (1400, .025), (2100, .02), (3100, .015), (4600, .01)], board_gain=1.2,
                 grit=0.0, crunch=0.0, soft=1.0, click=0.8, scuff=0.7, thump=0.5, slap=1.6),
    "stone": dict(modes=[(55, .03), (95, .025), (180, .02)], floor=0.35,
                  board=[(2800, .006), (4300, .005), (6200, .004)], board_gain=0.4,
                  grit=0.35, crunch=0.0, soft=0.85, click=1.3, scuff=1.0, thump=0.45),
    "gravel": dict(modes=[(60, .03), (110, .025)], floor=0.15, board=[(1500, .004)], board_gain=0.1,
                   grit=0.0, crunch=1.0, soft=1.8, click=0.25, scuff=0.0, thump=0.25),
}


@model("footstep", "sound",
       "A walking footstep: body weight, not a tap. Heel strike = a long, soft force pulse (the sole is a "
       "spring, so the floor gets a 6-25 ms push, not a click) driving the floor's heavily damped low modes and "
       "a broadband thump; the sole slaps (contact noise); then the foot rolls to the toe (a smaller, later "
       "push) with a little scuff; a hard heel adds a tip click. Surfaces: wood (low boom + optional board "
       "creak from stick-slip in the nails/joints), stone (hard click + grit grains), gravel (dense crunch "
       "grains from stones knocking under the load). Several steps can be rendered at a walking interval.",
       [
           Param("shoe", "trainer", "", choices=tuple(SHOES), doc="trainer (soft foam), boot (rubber), heel (hard heel tip)"),
           Param("surface", "wood", "", choices=tuple(SURFACES), doc="wood boards, stone/concrete, gravel"),
           Param("weight_kg", 75.0, "kg", 30, 150, "walker's weight: more weight = longer, lower, louder push"),
           Param("intensity", 0.5, "", 0, 1, "sneaking .. stomping (heel speed ~0.2-0.7 m/s; harder = shorter, "
                 "brighter contact, more slap and scuff)", source="heel vertical speed ~0.4 m/s at strike (est. "
                 "from gait literature)"),
           Param("roll_s", 0.11, "s", 0.03, 0.4, "heel-to-toe time (slow walk longer)"),
           Param("creak", 0.3, "", 0, 1, "wood only: chance and level of a board creak during loading"),
           Param("scuff", 0.4, "", 0, 1, "friction noise as the foot rolls (stone/wood)"),
           Param("floor_scale", 1.0, "", 0.4, 2.5, "scales the floor's mode frequencies (stiffer/smaller floor > 1)"),
           Param("damping", 1.0, "", 0.3, 4.0, "floor damping multiplier (carpet-like > 1, hollow floor < 1)"),
           Param("steps", 1, "count", 1, 64, "number of steps"),
           Param("interval_s", 0.55, "s", 0.25, 2.0, "time between steps (normal walk ~0.5-0.6 s)"),
           Param("at", 0.02, "s", 0, None, "first heel strike time"),
       ],
       ["Whittle 1999, Gait Analysis (heel-strike transient)", "Lieberman et al. 2010, Nature 463 (impact transient)",
        "Cook 2002, Real Sound Synthesis (footsteps / PhISM crunch)", "Farnell 2010, Designing Sound (footsteps)",
        "Turchet et al. 2010, physically based footstep sounds (solid vs aggregate surfaces)",
        "CC0 Freesound footstep recordings (refs/cc0, analysis only)"])
def footstep(ctx: Ctx, x, shoe, surface, weight_kg, intensity, roll_s, creak, scuff, floor_scale, damping, steps,
             interval_s, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    sh, su = SHOES[shoe], SURFACES[surface]
    w = weight_kg / 75.0
    I = float(np.clip(intensity, 0, 1))
    hard = 0.6 + 0.8 * I                           # harder strike -> shorter contact
    force = np.zeros(n)                            # floor drive (heel + toe pushes)
    slap_env = np.zeros(n)
    clicks = np.zeros(n)
    scuff_env = np.zeros(n)
    load_env = np.zeros(n)                         # loading curve for grit / crunch grain rates
    creak_sig = np.zeros(n)
    t = float(at)
    for s in range(int(steps)):
        if s:
            t += interval_s * (1 + 0.03 * rng.standard_normal())
        i0 = int(t * sr)
        if i0 >= n:
            break
        foot = 1.0 if s % 2 == 0 else 0.88 + 0.08 * rng.random()
        th = sh["heel_ms"] * 1e-3 * math.sqrt(w) * su["soft"] / hard * rng.uniform(0.9, 1.1)
        A = w * (0.3 + 0.7 * I) * foot * rng.uniform(0.9, 1.1)
        mh = int(th * sr)
        heel = _push(mh)
        dsp.place(force, heel, i0, A)
        d = np.abs(np.diff(heel, prepend=0.0))
        dsp.place(slap_env, d / (d.max() + 1e-12), i0, A)
        if sh["click"] > 0:
            dsp.place(clicks, _hann(max(3, int(0.25e-3 * sr))), i0, A * sh["click"] * (0.6 + 0.6 * I))
        # roll to the toe
        rt = roll_s * w ** 0.15 * rng.uniform(0.88, 1.12)
        it = i0 + int(rt * sr)
        tt = sh["toe_ms"] * 1e-3 * su["soft"] / hard * rng.uniform(0.9, 1.1)
        mt = int(tt * sr)
        toe = _push(mt)
        At = A * sh["toe_gain"] * rng.uniform(0.75, 1.15)
        dsp.place(force, toe, it, At)
        d = np.abs(np.diff(toe, prepend=0.0))
        dsp.place(slap_env, d / (d.max() + 1e-12), it, At * 0.9)
        # scuff while the foot rolls (heel mid-contact to just after the toe lands)
        a, b = i0 + mh // 2, it + int(0.04 * sr)
        if b > a:
            dsp.place(scuff_env, _hann(b - a) ** 0.5, a, A * (0.4 + 0.8 * I))
        # loading: fast rise at the heel, second bump at the toe, gone by toe-off
        Lh = int((rt + 0.06) * sr)
        tl = np.arange(Lh) / sr
        dsp.place(load_env, (tl / 0.02) * np.exp(1 - tl / 0.02), i0, A)
        tl2 = np.arange(int(0.12 * sr)) / sr
        dsp.place(load_env, 0.6 * (tl2 / 0.02) * np.exp(1 - tl2 / 0.02), it, At / max(A, 1e-9))
        # wood: board creak (stick-slip of nails/joints under the moving load), not on every step
        if surface == "wood" and creak > 0 and rng.random() < min(1.0, 0.25 + creak):
            c0 = i0 + int(rng.uniform(0.02, 0.07) * sr)
            dur = rng.uniform(0.1, 0.25) * (0.6 + creak)
            r0 = rng.uniform(25, 60)
            r1 = r0 * rng.uniform(1.4, 3.0)
            tau, tc = 0.0, []
            while tau < dur:
                tc.append(tau)
                r = r0 + (r1 - r0) * tau / dur
                tau += (1.0 / r) * (1 + 0.12 * rng.standard_normal())
            pulses = np.zeros(int(dur * sr) + 8)
            for k, tk in enumerate(tc):
                pulses[int(tk * sr)] += rng.uniform(0.6, 1.0)
            fc = rng.uniform(280, 600)
            cr = (dsp.bandpass(pulses, fc, sr, 12) + 0.5 * dsp.bandpass(pulses, fc * 2.3, sr, 10)
                  + 0.25 * dsp.bandpass(pulses, fc * 4.1, sr, 8)) * _hann(len(pulses))
            dsp.place(creak_sig, cr, c0, creak * w)
    if not np.any(force):
        return np.zeros(n)
    # floor: heavily damped low modes, position-dependent (random) tuning
    fm = np.array([f for f, _ in su["modes"]]) * floor_scale * rng.uniform(0.94, 1.06, len(su["modes"]))
    t60 = np.array([d for _, d in su["modes"]]) / damping
    amps = rng.uniform(0.5, 1.0, len(fm)) * np.sqrt(fm[0] / fm)
    floor = _conv(force, _mode_ir(sr, fm, t60, amps))
    # thump: the push itself as heard in the air (derivative of the force), the body-weight 'weight'
    thump = dsp.lowpass(np.diff(force, prepend=0.0), 250.0, sr)
    # sole slap: broadband contact noise shaped by the sole (soft foam = dull)
    nz = dsp.white(rng, n)
    sl_env = _conv(slap_env, dsp.exp_decay(int(0.004 * sr), sr, 0.004))
    slap = dsp.lowpass(dsp.highpass(nz, 180.0, sr) * sl_env, sh["slap_hz"], sr)
    # the slap also knocks the boards/slab surface (small-scale modes: the 'wood' in a wooden floor)
    bf0 = np.array([f for f, _ in su["board"]]) * rng.uniform(0.95, 1.05, len(su["board"]))
    board_ir = _mode_ir(sr, bf0, [d / damping for _, d in su["board"]], rng.uniform(0.5, 1.0, len(bf0)))
    slap = _norm(slap) + su["board_gain"] * _norm(_conv(dsp.lowpass(slap, sh["slap_hz"], sr), board_ir))
    out = su["floor"] * _norm(floor) + su["thump"] * _norm(thump) + sh["slap"] * su.get("slap", 1.0) * (0.7 + 0.6 * I) * _norm(slap)
    # hard heel tip / hard sole edge click, coloured by the floor's small-scale (board / slab) modes
    if np.any(clicks):
        tip = _conv(clicks, _mode_ir(sr, [sh["click_hz"], sh["click_hz"] * 1.62], [0.012, 0.008], [1.0, 0.5]))
        bf = [f for f, _ in su["board"]]
        bd = _conv(clicks, _mode_ir(sr, np.array(bf) * rng.uniform(0.95, 1.05, len(bf)),
                                    [d / damping for _, d in su["board"]], rng.uniform(0.5, 1.0, len(bf))))
        ck = _norm(tip) + su["board_gain"] * _norm(bd)
        out += sh["click"] * su["click"] * 0.8 * _norm(ck) * min(1.0, np.max(np.abs(clicks)) / max(w, 1e-9) + 0.4)
    if scuff > 0 and su["scuff"] > 0:
        sc = dsp.bandpass(dsp.pink(rng, n), sh["scuff_hz"], sr, 0.8) * scuff_env
        out += 0.25 * scuff * su["scuff"] * _norm(sc)
    if su["grit"] > 0:                               # sand/grit crushed between sole and slab
        rate = 900.0 * su["grit"] * load_env / (load_env.max() + 1e-12) + 300.0 * su["grit"] * scuff_env / (scuff_env.max() + 1e-12)
        g = np.zeros(n)
        idx = dsp.poisson_times(rng, rate, sr)
        g[idx] = np.minimum(rng.pareto(2.5, len(idx)) + 0.2, 4.0) * rng.choice([-1, 1], len(idx))
        grit = dsp.highpass(dsp.bandpass(g, 5500.0, sr, 0.8), 2500.0, sr)
        out += 0.3 * su["grit"] / 0.35 * _norm(grit)
    if su["crunch"] > 0:                             # gravel: stones knocking/sliding under the load
        rate = 2600.0 * su["crunch"] * w * (0.5 + I) * load_env / (load_env.max() + 1e-12)
        idx = dsp.poisson_times(rng, rate, sr)
        amp = np.minimum(rng.pareto(2.2, len(idx)) + 0.15, 5.0) * rng.choice([-1, 1], len(idx))
        band = rng.integers(0, 6, len(idx))
        centres = [650, 1000, 1450, 2100, 3100, 5000]
        cr = np.zeros(n)
        for bi, fc in enumerate(centres):
            g = np.zeros(n)
            sel = band == bi
            np.add.at(g, idx[sel], amp[sel] * (1.0 if 900 < fc < 2500 else 0.6))
            cr += dsp.bandpass(g, fc * rng.uniform(0.9, 1.1), sr, 2.5)
        out += 1.1 * su["crunch"] * _norm(cr)
    if np.any(creak_sig):
        out += 0.35 * creak_sig / (np.max(np.abs(creak_sig)) + 1e-12) * min(1.0, np.max(np.abs(creak_sig)))
    return dsp.highpass(out, 30.0, sr)


# =================================================================== coin
# Coin metals: density, Young's modulus, Poisson ratio, loss factor (est.; struck coins ring ~0.3-1 s).
COIN_METALS = {
    "cupronickel": (8900, 135e9, 0.34, 4e-4),
    "bronze": (8800, 110e9, 0.34, 6e-4),
    "steel": (7800, 210e9, 0.30, 2.5e-4),        # plated steel (UK 1p/2p/5p/10p since 1992/2012)
    "brass": (8500, 100e9, 0.35, 5e-4),
    "aluminium": (2700, 70e9, 0.34, 1.5e-3),
}
# Free circular plate, lambda^2 for the lowest modes (n nodal diameters, s nodal circles), nu ~ 0.33.
# Leissa 1969, Vibration of Plates (NASA SP-160), table 2.3: 5.253, 9.084, 12.23, 20.52, 21.6, 33.1, 35.2, 38.5.
PLATE_MODES = [(2, 5.253), (0, 9.084), (3, 12.23), (1, 20.52), (4, 21.6), (5, 33.1), (2, 35.2), (0, 38.5)]
# Floor tables (Hz, t60). 'table' from the strongest lines of a CC0 coin-on-table recording (292, 468, 726,
# 820, 1359, 2660, 3773 Hz); the others are estimates (ceramic tile: small stiff panel, bright and short).
FLOORS = {
    "tile": dict(modes=[(1250, .03), (2100, .025), (3300, .02), (4700, .015), (6900, .01)], click_hz=6000.0, level=0.6),
    "wood": dict(modes=[(180, .05), (320, .045), (720, .04), (1400, .03), (2500, .02)], click_hz=3500.0, level=0.6),
    "table": dict(modes=[(292, .08), (468, .07), (726, .06), (820, .06), (1359, .04), (2660, .03), (3773, .025)],
                  click_hz=4000.0, level=0.7),
    "stone": dict(modes=[(1800, .01), (3500, .008), (5200, .006)], click_hz=7000.0, level=0.5),
}


def coin_mode_freqs(diameter_mm, thickness_mm, metal, ring_hz=0.0):
    """(freqs, n) of the free-plate modes: f = lambda^2 / (2 pi a^2) * sqrt(D / (rho h))."""
    rho, E, nu, _ = COIN_METALS[metal]
    a, h = diameter_mm * 5e-4, thickness_mm * 1e-3
    D = E * h ** 3 / (12 * (1 - nu ** 2))
    k = math.sqrt(D / (rho * h)) / (2 * math.pi * a * a)
    f = np.array([lam * k for _, lam in PLATE_MODES])
    if ring_hz > 0:
        f = f * (ring_hz / f[0])
    return f, [nn for nn, _ in PLATE_MODES]


def euler_wobble_rate(T, r0, p, sr, r_max=300.0):
    """Precession rate of a coin rolling down like Euler's disk: r0 * (T / (T - t))^p toward a finite end time T
    (Moffatt 2000), capped at r_max. Returns (rate per sample, phase in revolutions)."""
    t = np.arange(int(T * sr)) / sr
    r = np.minimum(r0 * (T / np.maximum(T - t, 1e-5)) ** p, r_max)
    return r, np.cumsum(r) / sr


def euler_wobble_times(T, r0, p, sr=4000, r_max=300.0):
    """Times at which the precession completes a turn (one 'wobble' each): the gaps shrink toward T."""
    r, ph = euler_wobble_rate(T, r0, p, sr, r_max)
    k = np.nonzero(np.diff(np.floor(ph)) > 0)[0] + 1
    return k / sr


class _CoinVoice:
    """One coin's modes, split into degenerate pairs; collects excitation impulses per mode and state."""

    def __init__(self, ctx, freqs, ns, eta, split, contact_t60):
        self.ctx, self.sr = ctx, ctx.sr
        rng = ctx.rng
        self.members = []  # (freq, n, which)
        for f, nn in zip(freqs, ns):
            if nn == 0:
                self.members.append((f, nn, 0))
            else:
                d = split * rng.uniform(0.3, 1.0)
                self.members.append((f * (1 - d / 2), nn, 0))
                self.members.append((f * (1 + d / 2), nn, 1))
        self.members = [m for m in self.members if m[0] < self.sr * 0.45]
        self.eta = eta
        self.contact_t60 = contact_t60
        self.ex = {st: np.zeros((len(self.members), ctx.n)) for st in ("free", "contact", "settle")}

    def strike(self, i, amp, theta, radial, contact_s, state="free"):
        if i < 0 or i >= self.ctx.n:
            return
        for k, (f, nn, which) in enumerate(self.members):
            shape = (math.cos(nn * theta) if which == 0 else math.sin(nn * theta)) if nn else radial
            self.ex[state][k, i] += amp * shape * _hann_spectrum(f, contact_s) / math.sqrt(1 + 0.15 * k)

    def render(self):
        out = {}
        for st, ex in self.ex.items():
            y = np.zeros(self.ctx.n)
            for k, (f, nn, which) in enumerate(self.members):
                if not np.any(ex[k]):
                    continue
                t60 = (min(6.91 / (math.pi * f * self.eta + 1.0), 2.5) if st == "free"
                       else self.contact_t60 if st == "contact" else 0.03)   # settle: lying flat, ring choked
                y += _conv(ex[k], _mode_ir(self.sr, [f], [t60], [1.0], [self.ctx.rng.uniform(0, 0.3)]))
            out[st] = y
        return out


@model("coin", "sound",
       "A real metal coin, not the arcade blip. The coin is a thin free disc: its ring is the free-plate modes "
       "(lowest = 2 nodal diameters, ~9-12 kHz for a 24-28 mm coin; ratios 1, 1.73, 2.33, ... (Leissa)), each "
       "non-axisymmetric mode a slightly split pair (coins are not perfect discs: the 'shimmer' beat). "
       "drop: falls from a height (v = sqrt(2gh)), bounces with random restitution, landing on edge (bright "
       "ring) or flat (slap + chatter), then may roll/wobble like Euler's disk: contacts speed up toward a "
       "finite end time (rate ~ (t0 - t)^-p) and it settles flat with a clack that chokes the ring. "
       "jingle: a few coins knocking together in a hand or pocket.",
       [
           Param("mode", "drop", "", choices=("drop", "jingle")),
           Param("diameter_mm", 26.0, "mm", 12, 45, "coin diameter (UK 10p 24.5, 50p 27.3, US quarter 24.3)"),
           Param("thickness_mm", 1.85, "mm", 0.8, 4, "coin thickness"),
           Param("metal", "cupronickel", "", choices=tuple(COIN_METALS)),
           Param("ring_hz", 0.0, "Hz", 0, 20000, "override the lowest ring mode (0 = from size/metal); the "
                 "pitch-first value measured on a reference"),
           Param("split", 0.012, "", 0, 0.05, "relative split of each degenerate mode pair (ref: 9022/9216 Hz = 2%)"),
           Param("floor", "tile", "", choices=tuple(FLOORS)),
           Param("drop_height_m", 0.8, "m", 0.05, 3, "drop height (impact speed sqrt(2 g h): 0.8 m -> 4 m/s)"),
           Param("restitution", 0.45, "", 0.1, 0.8, "mean bounce speed ratio (coin on hard floor ~0.35-0.6, est.)"),
           Param("edge_chance", 0.6, "", 0, 1, "chance a bounce lands on the rim (bright ring) rather than flat"),
           Param("spin_s", 0.9, "s", 0, 4, "drop: roll/wobble time before it settles (0 = settles after the bounces)"),
           Param("wobble_exp", 0.5, "", 0.1, 0.8, "wobble speed-up exponent. Moffatt 2000 gives ~1/6 for an ideal disk's long "
                 "final phase; a coin's ~1 s wobble needs ~0.5 to go from ~10 to ~100 contacts/s as heard (our choice)"),
           Param("wobble_level", 0.4, "", 0, 1, "strength of each rolling micro-contact relative to the first impact"),
           Param("micro_contacts", 6.0, "count", 1, 30, "micro-contacts per precession turn (rolling roughness, est.)"),
           Param("ring_t60", 0.3, "s", 0.02, 1.0, "ring time while the coin rolls on its rim (touching the floor)"),
           Param("floor_level", 1.0, "", 0, 3, "level of the floor's knock vs the coin's ring"),
           Param("count", 5, "count", 2, 30, "jingle: number of coins"),
           Param("shake_s", 1.0, "s", 0.1, 10, "jingle: how long the coins are shaken"),
           Param("rate_hz", 4.0, "Hz", 0.5, 12, "jingle: shakes per second"),
           Param("at", 0.03, "s", 0, None, "first impact time"),
       ],
       ["Leissa 1969, Vibration of Plates, NASA SP-160 (free circular plate)", "Moffatt 2000, Nature 404 (Euler's disk)",
        "van den Doel, Kry & Pai 2001 (modal impacts)", "Cook 2002, Real Sound Synthesis (PhISM coins)",
        "CC0 Freesound coin recordings (refs/cc0, analysis only)"])
def coin(ctx: Ctx, x, mode, diameter_mm, thickness_mm, metal, ring_hz, split, floor, drop_height_m, restitution,
         edge_chance, spin_s, wobble_exp, wobble_level, micro_contacts, ring_t60, floor_level, count, shake_s, rate_hz, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    eta = COIN_METALS[metal][3]
    if mode == "jingle":
        return _jingle(ctx, diameter_mm, thickness_mm, metal, ring_hz, split, eta, int(count), shake_s, rate_hz, at)
    freqs, ns = coin_mode_freqs(diameter_mm, thickness_mm, metal, ring_hz)
    voice = _CoinVoice(ctx, freqs, ns, eta, split, contact_t60=ring_t60)
    fl = FLOORS[floor]
    fimp = np.zeros(n)            # floor-knock impulses (velocity-weighted contact pulses)
    v = math.sqrt(2 * G * drop_height_m)
    v0 = v
    t = float(at)
    last = t
    while True:
        i = int(t * sr)
        if i >= n:
            break
        last = t
        edge = rng.random() < edge_chance
        theta = rng.uniform(0, 2 * math.pi)
        if edge:   # rim strike: short metal-on-hard contact, strong ring
            tc = rng.uniform(0.03, 0.07) * 1e-3
            voice.strike(i, v * rng.uniform(0.7, 1.0), theta, 0.8, tc)
            dsp.place(fimp, _hann(max(2, int(tc * sr * 3))), i, v * 0.6)
        else:      # flat landing: a slap, weaker ring, then fast chatter as it rocks down
            tc = rng.uniform(0.1, 0.25) * 1e-3
            voice.strike(i, v * 0.35, theta, 0.3, tc)
            dsp.place(fimp, _hann(max(2, int(tc * sr * 3))), i, v)
            gap = rng.uniform(2e-3, 6e-3)
            a = 0.5
            tt = t
            for _ in range(rng.integers(2, 6)):
                tt += gap
                gap *= rng.uniform(0.5, 0.8)
                th2 = theta + rng.uniform(-0.6, 0.6)
                voice.strike(int(tt * sr), v * 0.3 * a, th2, 0.4, tc)
                dsp.place(fimp, _hann(max(2, int(tc * sr * 3))), int(tt * sr), v * 0.4 * a)
                a *= 0.6
        e = float(np.clip(restitution * rng.uniform(0.75, 1.25), 0.05, 0.9))
        v *= e
        dt = 2 * v / G
        if v < 0.06 * v0 or dt < 0.004:
            break
        t += dt
    end = last + 0.01
    if spin_s > 0:
        # Euler's-disk style roll: the coin stays in rolling contact; the contact point runs round the rim at the
        # precession rate, which rises toward a finite end time. Heard as a rough whirr: many micro-contacts
        # (rate = precession x micro_contacts) whose strength swells once per precession turn.
        Tw = spin_s * rng.uniform(0.85, 1.15)
        t0 = last + rng.uniform(0.01, 0.04)
        rate, ph = euler_wobble_rate(Tw, rng.uniform(9, 15), wobble_exp, sr)
        swell = (0.3 + 0.7 * (0.5 + 0.5 * np.cos(2 * np.pi * ph))) ** 2
        ev = dsp.poisson_times(rng, rate * micro_contacts * swell, sr)
        i0 = int(t0 * sr)
        for j in ev:
            prog = j / len(rate)
            # refs: the wobble is ring-dominated and holds or even builds toward the end, then stops abruptly
            amp = wobble_level * v0 * (0.6 + 0.6 * prog) * swell[j] * rng.uniform(0.6, 1.2)
            theta = 2 * np.pi * ph[j] + rng.uniform(-0.3, 0.3)          # contact point walks round the rim
            voice.strike(i0 + j, amp, theta, 0.6, 0.08e-3, state="contact")
            dsp.place(fimp, _hann(3), i0 + j, amp * 0.35)
        end = t0 + Tw
    # settle: lands flat with a clack, ring choked
    ie = int(end * sr)
    voice.strike(ie, 0.12 * v0, rng.uniform(0, 2 * math.pi), 0.3, 0.2e-3, state="settle")
    dsp.place(fimp, _hann(max(2, int(0.6e-3 * sr))), ie, 0.06 * v0)
    gap = rng.uniform(3e-3, 7e-3)
    tt, a = end, 0.5
    for _ in range(4):
        tt += gap
        gap *= 0.6
        voice.strike(int(tt * sr), 0.1 * v0 * a, rng.uniform(0, 2 * math.pi), 0.3, 0.2e-3, state="settle")
        dsp.place(fimp, _hann(3), int(tt * sr), 0.05 * v0 * a)
        a *= 0.55
    rings = voice.render()
    choke = np.ones(n)
    ic = int((end + 0.025) * sr)
    if ic < n:
        tail = np.exp(-np.arange(n - ic) / (0.006 * sr))
        choke[ic:] = 0.03 + 0.97 * tail
    ring = (rings["free"] + rings["contact"]) * choke + rings["settle"]
    fm = np.array([f for f, _ in fl["modes"]]) * rng.uniform(0.95, 1.05, len(fl["modes"]))
    fknock = _conv(fimp, _mode_ir(sr, fm, [d for _, d in fl["modes"]], rng.uniform(0.5, 1.0, len(fm))))
    click = dsp.bandpass(fimp, fl["click_hz"], sr, 0.7)   # the contact itself (metal on a hard floor)
    out = _norm(ring) + floor_level * fl["level"] * (_norm(fknock) * 0.7 + 0.5 * _norm(click))
    return dsp.highpass(out, 60.0, sr)


def _jingle(ctx, diameter_mm, thickness_mm, metal, ring_hz, split, eta, count, shake_s, rate_hz, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    voices = []
    for c in range(count):
        d = diameter_mm * rng.uniform(0.72, 1.12)        # mixed change
        th = thickness_mm * rng.uniform(0.8, 1.15)
        f, ns = coin_mode_freqs(d, th, metal, ring_hz * d / diameter_mm if ring_hz else 0.0)
        f = f if not ring_hz else f * (diameter_mm / d) ** 2
        voices.append(_CoinVoice(ctx, f, ns, eta * 3, split, contact_t60=0.05))  # hand/pocket damping
    hand = np.zeros(n)
    nrev = max(1, int(round(2 * rate_hz * shake_s)))
    t = float(at)
    for r in range(nrev):
        if r:
            t += 1 / (2 * rate_hz) * (1 + 0.1 * rng.standard_normal())
        i0 = int(t * sr)
        if i0 >= n:
            break
        k = rng.poisson(count * 1.6)
        for _ in range(k):
            dt = abs(rng.gamma(2.0, 0.008))
            i = i0 + int(dt * sr)
            a = min(rng.pareto(2.5) + 0.3, 3.0)
            c1, c2 = rng.choice(count, 2, replace=False)
            tc = rng.uniform(0.04, 0.1) * 1e-3
            for c in (c1, c2):
                st = "free" if rng.random() < 0.5 else "contact"
                voices[c].strike(i, a, rng.uniform(0, 2 * math.pi), rng.uniform(0.3, 0.9), tc, st)
        L = int(0.12 * sr)
        dsp.place(hand, _hann(L), i0, 1.0)
    out = np.zeros(n)
    for vc in voices:
        rs = vc.render()
        out += rs["free"] + rs["contact"] + rs["settle"]
    rustle = dsp.bandpass(dsp.pink(rng, n), 1800.0, sr, 0.5) * dsp.lowpass(hand, 20.0, sr)
    return dsp.highpass(_norm(out) + 0.08 * _norm(rustle), 80.0, sr)
