"""Complete sound models (multi-layer recipes built from the stage blocks).

Each model documents which numbers come from sources and which are our estimates."""
from __future__ import annotations

import math

import numpy as np

from .. import dsp
from ..materials import MATERIALS
from ..core import Ctx, Param, REGISTRY, ctrl, model, scalar
from .excitation import bubble_grain, friedlander


def call(name: str, ctx: Ctx, x=None, **params):
    m = REGISTRY[name]
    return m.fn(ctx, x, **m.resolve(params))


def norm(x):
    return x / (np.max(np.abs(x)) + 1e-12)


# =================================================================== duck
@model("duck", "sound",
       "Mallard call. Syrinx = pressure-controlled valve (valve_pulse) with rasp; trachea + beak as a "
       "formant filter peaking around 1.5-2.5 kHz; female call = 2-10 quacks in a decrescendo, male = "
       "quieter 1-2 note rasping 'raeb' with the bulla widening the band (~500-2000 Hz).",
       [
           Param("voice", "female", "", choices=("female", "male")),
           Param("notes", 5, "count", 1, 10, "quacks per call (female typical 5-6, range 2-10)",
                 source="Cornell All About Birds; Ducks Unlimited"),
           Param("f0", 420.0, "Hz", 150, 650, "syrinx pulse rate (sources report ~180 to ~585 Hz)",
                 source="Wildfowl journal (Greenewalt 1958: 179 Hz; example 585 Hz)"),
           Param("formant_hz", 1900.0, "Hz", 1200, 3000, "main resonance (max amplification 1.5-2.5 kHz)",
                 source="Wildfowl journal"),
           Param("rasp", 0.55, "", 0, 1, "aperiodicity: the main realism control (Ducks Unlimited 'entropy')"),
           Param("note_s", 0.17, "s", 0.05, 0.5, "length of one quack (our estimate; male raeb ~0.3 s)"),
           Param("gap_s", 0.09, "s", 0.02, 0.5, "silence between quacks (our estimate)"),
           Param("decrescendo_db", 2.5, "dB", 0, 8, "level drop per quack"),
           Param("pitch_fall", 0.03, "", 0, 0.2, "relative pitch drop per quack"),
           Param("start", 0.05, "s", 0, None, "start time"),
       ],
       ["Cornell All About Birds: Mallard sounds", "Wildfowl: Observations on sound production in the Anatidae",
        "Abs 1970, Vocalizations of the Mallard", "Ducks Unlimited duck-calling study", "Wikipedia: Syrinx"])
def duck(ctx: Ctx, x, voice, notes, f0, formant_hz, rasp, note_s, gap_s, decrescendo_db, pitch_fall, start):
    sr = ctx.sr
    out = np.zeros(ctx.n)
    male = voice == "male"
    if male:
        notes, note_s, f0, rasp = min(int(notes), 2), max(note_s, 0.28), f0 * 0.7, max(rasp, 0.8)
    t0 = start
    for k in range(int(notes)):
        L = int(note_s * (1 + 0.1 * ctx.rng.uniform(-1, 1)) * sr)
        sub = ctx.child(L)
        tt = np.arange(L) / L
        # pitch contour inside a quack: quick rise then fall (estimate), whole call falls note by note
        fk = f0 * (1 - pitch_fall * k)
        contour = fk * (0.85 + 0.25 * np.sin(np.pi * np.minimum(tt * 1.4, 1.0)) - 0.1 * tt)
        amp = np.clip(tt / 0.12, 0, 1) * np.clip((1 - tt) / 0.35, 0, 1) ** 1.3
        src = call("valve_pulse", sub, f0=[[i / sr, v] for i, v in zip(range(0, L, 64), contour[::64])],
                   level=[[i / sr, v] for i, v in zip(range(0, L, 64), amp[::64])],
                   open_quotient=0.55, speed_quotient=3.5, jitter=0.008 + 0.015 * rasp,
                   shimmer=0.15 + 0.3 * rasp, rasp=rasp, aspiration=0.1)
        if male:
            fm = [[700, 2.5, 0], [1300, 2.0, 3], [formant_hz, 2.0, 2], [3200, 4, -8]]
        else:
            fm = [[0.5 * formant_hz, 6, -4], [formant_hz, 3.5, 6], [1.55 * formant_hz, 5, 0], [2.3 * formant_hz, 6, -9]]
        y = call("formant", sub, src, formants=fm, dry=0.03)
        # trachea: short closed-open tube adds the nasal 'honk'
        y = call("tube", sub, y, length=0.14, ends="closed_open", loss=0.5, damp_hz=5000, mix=0.5)
        y = dsp.highpass(y, 350.0, sr)
        gain = 10 ** (-decrescendo_db * k / 20) * (0.35 if male else 1.0)
        dsp.place(out, norm(y) * gain, int(t0 * sr))
        t0 += note_s + gap_s * (1 + 0.25 * ctx.rng.uniform(-1, 1))
    return out


# =================================================================== engine
FIRING = {
    "even": None,  # 720/cylinders spacing
    "v_twin_45": [0.0, 315 / 720],  # 45-degree V-twin fires at 0 and 315 crank degrees
    "crossplane_bank": [0.0, 180 / 720, 450 / 720, 540 / 720],  # one bank of a cross-plane V8: 180/270/90/180
}


@model("engine", "sound",
       "Combustion engine. Per-cylinder exhaust pressure pulses at the firing frequency RPM x cylinders / 120 "
       "(4-stroke), offsets from the firing order, cycle-to-cycle jitter; exhaust pipe as a closed-open "
       "waveguide; muffler; intake roar; valve-train ticks; transmission whine; off-throttle pops.",
       [
           Param("rpm", 1800.0, "rev/min", 300, 12000, "engine speed", automatable=True),
           Param("load", 0.4, "", 0, 1, "throttle/load: pulse strength, brightness, intake roar", automatable=True),
           Param("cylinders", 4, "count", 1, 12, "number of cylinders"),
           Param("firing", "even", "", choices=tuple(FIRING), doc="firing pattern (uneven = burble)"),
           Param("exhaust_length", 2.2, "m", 0.3, 6, "exhaust pipe length (closed-open waveguide)"),
           Param("muffler", 0.5, "", 0, 1, "muffler strength (0 = straight pipe)"),
           Param("jitter", 0.03, "", 0, 0.2, "cycle-to-cycle timing/amplitude variation"),
           Param("intake", 0.3, "", 0, 1, "intake roar level"),
           Param("mechanical", 0.15, "", 0, 1, "valve-train tick level"),
           Param("whine", 0.05, "", 0, 1, "transmission whine level"),
           Param("whine_order", 9.0, "x rev", 1, 60, "whine frequency as a multiple of crank rate (gear teeth; estimate)"),
           Param("pops", 0.3, "", 0, 1, "chance of exhaust pops when the throttle closes at speed"),
       ],
       ["Baldan et al., SIVE 2015", "Farnell 2010 practical 22", "Borla: engine sound and drone physics",
        "Wikipedia: Firing order"])
def engine(ctx: Ctx, x, rpm, load, cylinders, firing, exhaust_length, muffler, jitter, intake, mechanical, whine,
           whine_order, pops):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    R = ctrl(ctx, rpm, "exp")
    Ld = np.clip(ctrl(ctx, load), 0, 1)
    cyc_rate = R / 120.0  # 4-stroke cycles per second
    tj = dsp.lowpass(rng.standard_normal(n), 30.0, sr)
    tj /= np.std(tj) + 1e-12
    phase = np.cumsum(cyc_rate * (1 + 0.15 * jitter * tj)) / sr
    offs = FIRING[firing]
    if offs is None:
        offs = [i / cylinders for i in range(int(cylinders))]
    elif len(offs) != cylinders:
        reps = int(math.ceil(cylinders / len(offs)))
        offs = [(o + r / reps) % 1 for r in range(reps) for o in offs][: int(cylinders)]
    pulses = np.zeros(n)
    noise_env = np.zeros(n)
    width = 0.10 - 0.04 * Ld  # pulse width as a fraction of the cycle (higher load = sharper; estimate)
    for ci, o in enumerate(offs):
        p = phase + o
        cyc = np.floor(p).astype(int)
        u = (p - cyc) / width
        ncyc = cyc.max() + 2
        a_cyc = 1 + jitter * 3 * rng.standard_normal(ncyc)  # combustion varies cycle to cycle
        a_cyc *= 1 + 0.06 * rng.standard_normal()  # each cylinder slightly different
        shape = np.where(u < 1, (1 - np.cos(2 * np.pi * np.clip(u, 0, 1))) * 0.5 * np.exp(-2.5 * u), 0.0)
        pulses += shape * a_cyc[cyc]
        noise_env += shape
    strength = 0.25 + 0.75 * Ld
    pulses *= strength
    # combustion/blowdown turbulence inside each pulse
    turb = dsp.highpass(rng.standard_normal(n), 300.0, sr) * noise_env * (0.08 + 0.25 * Ld)
    src = np.diff(pulses, prepend=0.0) * sr / np.maximum(cyc_rate * 40, 1.0) + turb
    src = dsp.highpass(src, 25.0, sr)
    # exhaust pipe: closed (valve end) - open (tailpipe) waveguide
    D = 2 * exhaust_length / ctx.c * sr
    pipe = dsp.comb_waveguide(src, D, 0.55, sr, 2500.0, -1.0)
    # muffler: chamber lowpass + mild resonance (Helmholtz-like)
    cutoff = 3500.0 * (1 - muffler) + 600.0 * muffler
    y = dsp.lowpass(pipe, cutoff, sr, 0.9)
    y = dsp.biquad(y, "peak", 120.0 + 60 * (1 - muffler), 1.5, sr, 4.0 * muffler)
    y = (1 - 0.5 * muffler) * pipe * (1 - muffler) + y
    # intake roar: band noise modulated by the pulses, grows with load
    intake_sig = dsp.bandpass(rng.standard_normal(n), 900.0, sr, 0.7) * (0.3 + noise_env) * Ld * intake * 0.4
    # valve-train ticks: small steel clicks twice per cycle per cylinder (estimate)
    ticks = np.zeros(n)
    if mechanical > 0:
        cam = np.cumsum(cyc_rate * 2 * cylinders) / sr
        idx = np.nonzero(np.diff(np.floor(cam)) > 0)[0]
        click = np.sin(2 * np.pi * 4500 * np.arange(120) / sr) * np.exp(-np.arange(120) / 25.0)
        for i in idx:
            dsp.place(ticks, click, i, rng.uniform(0.5, 1.0))
        ticks = dsp.bandpass(ticks, 4500.0, sr, 2.0) * mechanical * 0.15
    # transmission whine locked to crank rate
    wph = 2 * np.pi * np.cumsum(R / 60.0 * whine_order) / sr
    whine_sig = (np.sin(wph) + 0.3 * np.sin(2 * wph)) * whine * 0.05 * Ld * (R / 3000.0)
    # off-throttle pops: load falling fast at high rpm
    pop_sig = np.zeros(n)
    if pops > 0:
        dL = np.diff(Ld, prepend=Ld[0]) * sr
        lift = (dL < -0.5) | ((Ld < 0.08) & (R > 2500))
        rate = lift * pops * 6.0
        for i in dsp.poisson_times(rng, rate, sr):
            L = int(0.03 * sr)
            pp = friedlander(L, sr, 0.002 + 0.004 * rng.random()) + 0.3 * rng.standard_normal(L) * np.exp(-np.arange(L) / (0.006 * sr))
            dsp.place(pop_sig, pp, i, rng.uniform(0.3, 1.0) * 1.5)
        pop_sig = dsp.comb_waveguide(pop_sig, D, 0.4, sr, 3000.0, -1.0)
    mix = norm(y) + intake_sig + ticks + whine_sig + pop_sig * 0.6
    return mix


# =================================================================== foghorn
@model("foghorn", "sound",
       "Diaphone foghorn. Compressed air chopped by a reciprocating slotted piston (chopper) into a long conical "
       "trumpet (horn). Type F high tone ~250 Hz at 35-40 psi; on shut-off the piston slows and the pitch drops "
       "to the 93-150 Hz 'grunt'; the two-tone F2T sustains the low tone.",
       [
           Param("tone_hz", 250.0, "Hz", 60, 600, "main tone (Type F ~250 Hz)", source="Terry Pepper: the diaphone"),
           Param("grunt_hz", 110.0, "Hz", 60, 200, "end pitch of the grunt (93-150 Hz)", source="Terry Pepper"),
           Param("mode", "grunt", "", choices=("single", "grunt", "two_tone")),
           Param("blast_s", 2.5, "s", 0.3, 20, "length of the main tone"),
           Param("attack_s", 0.35, "s", 0.02, 3, "pressure build-up / piston spin-up (estimate)"),
           Param("grunt_s", 1.1, "s", 0.1, 6, "length of the falling grunt or the low tone (estimate)"),
           Param("pressure", 1.0, "", 0.3, 1.5, "air pressure re nominal 35-40 psi"),
           Param("horn_length", 3.0, "m", 0.5, 8, "trumpet length (estimate)"),
           Param("mouth_radius", 0.3, "m", 0.05, 1.5, "trumpet bell radius (estimate)"),
           Param("start", 0.1, "s", 0, None, "start time"),
       ], ["Terry Pepper, Seeing The Light: the diaphone", "Wikipedia: Foghorn"])
def foghorn(ctx: Ctx, x, tone_hz, grunt_hz, mode, blast_s, attack_s, grunt_s, pressure, horn_length, mouth_radius,
            start):
    s0, a, b = start, start + attack_s, start + attack_s + blast_s
    if mode == "single":
        f = [[0, tone_hz * 0.8], [s0, tone_hz * 0.8], [a, tone_hz], [b, tone_hz], [b + 0.15, tone_hz * 0.97]]
        P = [[0, 0], [s0, 0], [a, pressure], [b, pressure], [b + 0.2, 0]]
    elif mode == "grunt":
        f = [[0, tone_hz * 0.8], [s0, tone_hz * 0.8], [a, tone_hz], [b, tone_hz], [b + 0.12, tone_hz * 0.9],
             [b + grunt_s, grunt_hz]]
        P = [[0, 0], [s0, 0], [a, pressure], [b, pressure], [b + 0.1, pressure * 0.85], [b + grunt_s * 0.8, pressure * 0.35],
             [b + grunt_s, 0]]
    else:  # two-tone: high then sustained low
        f = [[0, tone_hz * 0.8], [s0, tone_hz * 0.8], [a, tone_hz], [b, tone_hz], [b + 0.12, grunt_hz * 1.05],
             [b + grunt_s, grunt_hz]]
        P = [[0, 0], [s0, 0], [a, pressure], [b, pressure], [b + 0.12, pressure * 0.9], [b + grunt_s - 0.15, pressure * 0.8],
             [b + grunt_s, 0]]
    Pc = ctrl(ctx, P)
    # higher pressure opens the ports more abruptly -> brighter (estimate)
    src = call("chopper", ctx, freq=f, pressure=P, open_fraction=0.42, edge=0.25, jitter=0.002, flow_noise=0.06)
    y = call("horn", ctx, src, length=horn_length, mouth_radius=mouth_radius, resonance=0.5, damp_hz=2500.0)
    y = dsp.lowpass(y, 2200.0, ctx.sr, 0.6)
    y = dsp.biquad(y, "peak", 2 * tone_hz, 1.0, ctx.sr, 3.0)
    hiss = dsp.bandpass(ctx.rng.standard_normal(ctx.n), 1500.0, ctx.sr, 0.8) * np.sqrt(np.maximum(Pc, 0)) * 0.01
    return norm(y) + hiss


# =================================================================== explosion
@model("explosion", "sound",
       "Explosion after Farnell's decomposition: Friedlander shock, a blast of noise through a resonant band "
       "swept down from ~7 kHz with clipping and random sub-blasts, a low fireball (~70 Hz) and debris. Put "
       "distance/ground/outdoor blocks after it for the far field.",
       [
           Param("size", 0.6, "", 0, 1, "overall scale: shock duration, blast length, low-end weight (uncalibrated)"),
           Param("sub_blasts", 4, "count", 0, 20, "secondary bursts in the first ~0.6 s"),
           Param("fireball_s", 2.5, "s", 0.2, 10, "fireball/rumble T60"),
           Param("debris", 0.5, "", 0, 1, "falling debris amount"),
           Param("drive", 3.0, "", 0, 10, "clipping of the blast (a close blast overloads any mic)"),
           Param("at", 0.05, "s", 0, None, "time of detonation"),
       ], ["Farnell 2010 practical 31", "Wikipedia: Blast wave", "Rajaram & Lieuwen 2006 (combustion noise)",
           "Murch, Transom 2005 ('bad recorder' layer)"])
def explosion(ctx: Ctx, x, size, sub_blasts, fireball_s, debris, drive, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    n0 = int(at * sr)
    m = n - n0
    t = np.arange(m) / sr
    t_star = 0.004 + 0.03 * size
    shock = friedlander(m, sr, t_star)
    # blast: resonant band swept down from ~7 kHz
    sweep_s = 0.25 + 0.6 * size
    fc = 150 + 6850 * np.exp(-t / (sweep_s / 3))
    w = rng.standard_normal(m)
    blast = dsp.svf(w, fc, 1.2, sr, "bp") * dsp.ar_env(m, sr, 0.002, 0.4 + 1.0 * size) * 2.5
    for _ in range(int(sub_blasts)):
        ts = rng.uniform(0.02, 0.6) * (0.5 + size)
        i = int(ts * sr)
        if i < m:
            L = m - i
            sb = friedlander(L, sr, t_star * rng.uniform(0.2, 0.6)) * 0.5 + rng.standard_normal(L) * dsp.ar_env(L, sr, 0.001, 0.15) * 0.4
            blast[i:] += sb * rng.uniform(0.3, 0.8)
    blast = dsp.softclip(blast, drive) if drive > 0 else blast
    # fireball: low turbulent roar ~70 Hz, slow puffs
    puffs = 1 + 0.5 * dsp.lowpass(rng.standard_normal(m), 4.0, sr) / 0.05
    puffs = np.clip(puffs, 0.2, 3)
    fire = dsp.lowpass(dsp.brown(rng, m), 70.0 + 50 * (1 - size), sr, 0.9) * dsp.ar_env(m, sr, 0.04, fireball_s) * puffs
    rumble = dsp.lowpass(rng.standard_normal(m), 40.0, sr) * dsp.ar_env(m, sr, 0.1, fireball_s * 1.4) * 4
    # debris: decaying rate of small hard grains
    deb = np.zeros(m)
    if debris > 0:
        sub = ctx.child(m)
        rate = [[0, 0], [0.15, 120 * debris], [1.0, 60 * debris], [3.0, 5 * debris], [6, 0]]
        deb = call("crackle", sub, rate=rate, grain_ms=6, f_lo=600, f_hi=7000, q=3, size_exponent=2.2, level=0.6)
    out = np.zeros(n)
    out[n0:] = (1.0 * norm(shock) + 0.65 * norm(blast) + (0.25 + 0.3 * size) * norm(fire)
                + (0.15 + 0.25 * size) * norm(rumble) + 0.25 * deb)
    return out


# =================================================================== match
@model("match", "sound",
       "Match strike and ignition (our design from sourced parts): stick-slip scrape of the head over the "
       "striker exciting the matchbox, an ignition pop, a bright flare with dense crackle, then a small steady "
       "flame (soft low 'woof' + hiss + sparse crackle).",
       [
           Param("strike_s", 0.22, "s", 0.05, 1, "length of the scrape (estimate)"),
           Param("strike_speed", 0.6, "m/s", 0.05, 3, "peak scrape speed (estimate)"),
           Param("strike_force", 3.0, "N", 0.2, 15, "pressing force (estimate)"),
           Param("ignite", True, "", None, None, "whether it lights (failed strikes are realistic too)"),
           Param("flare_s", 0.45, "s", 0.05, 2, "flare decay T60 (estimate)"),
           Param("flame_s", 2.5, "s", 0, 30, "how long the steady flame lasts"),
           Param("flame_size", 0.4, "", 0, 1, "flame size: match ~0.3, lighter ~0.5, torch ~1"),
           Param("start", 0.1, "s", 0, None, "start time"),
       ], ["Avanzini, Serafin & Rocchesso DAFx 2002", "Tsugi GameSynth physical modules (match friction)",
           "Farnell 2010 practical 11; Wikibooks Designing Sound in SuperCollider: Fire",
           "Wikipedia: Match; Stick-slip phenomenon", "Chadwick & James 2011 (fire sound)"])
def match(ctx: Ctx, x, strike_s, strike_speed, strike_force, ignite, flare_s, flame_s, flame_size, start):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    s0, s1 = start, start + strike_s
    vel = [[0, 0], [s0, 0], [s0 + strike_s * 0.3, strike_speed], [s0 + strike_s * 0.8, strike_speed * 0.8], [s1, 0]]
    force = [[0, 0], [s0, strike_force * 0.6], [s0 + strike_s * 0.5, strike_force], [s1, strike_force * 0.4],
             [s1 + 0.005, 0]]
    sub = ctx.child()
    fr = call("friction", sub, velocity=vel, normal_force=force, mu_static=0.7, mu_kinetic=0.4, stribeck_v=0.02,
              f_osc=1400, zeta=0.08, roughness=0.9, roughness_hz=6000)
    box = call("modal", sub, fr, material="cardboard", shape="plate", length=0.05, width=0.035, thickness=0.0015,
               n_modes=20, extra_damping=40, strike_pos=0.3)
    # gritty striker surface: band noise following speed
    V = ctrl(ctx, vel)
    grit = dsp.bandpass(rng.standard_normal(n), 3500.0, sr, 0.8) * (V / max(strike_speed, 1e-3)) ** 1.5
    grit *= 1 + 0.8 * dsp.lowpass(rng.standard_normal(n), 300.0, sr) / 0.1
    scrape = 0.6 * norm(fr) + 0.5 * norm(box) + 0.5 * norm(grit) * 0.6
    out = scrape * 0.8
    if ignite:
        ti = s0 + strike_s * 0.75
        ni = int(ti * sr)
        m = n - ni
        tt = np.arange(m) / sr
        pop = friedlander(m, sr, 0.0012) * 0.8
        flare_env = dsp.ar_env(m, sr, 0.012, flare_s)
        flare = dsp.bandpass(rng.standard_normal(m), 2500.0, sr, 0.6) * flare_env
        fsub = ctx.child(m)
        crk = call("crackle", fsub, rate=[[0, 400], [flare_s * 0.5, 120], [flare_s, 25], [flare_s + 0.6, 4], [flare_s + 1.0, 0]],
                   grain_ms=1.5, f_lo=1500, f_hi=9000, q=1.5, size_exponent=2.4, level=1.0)
        # steady flame: low woof (lapping) + soft hiss, slowly modulated (Farnell fire layers)
        flame_env = np.clip((tt - 0.08) / 0.25, 0, 1) * np.clip((flame_s + 0.3 - tt) / 0.4, 0, 1)
        lap = np.clip(1 + 0.7 * dsp.lowpass(rng.standard_normal(m), 3.0, sr) / 0.04, 0.1, 3)
        # small flames peak higher (St = f L / U with small L); centre frequency is our estimate
        fc_flame = 180.0 + 700.0 * (1 - flame_size)
        woof = dsp.lowpass(dsp.bandpass(rng.standard_normal(m), fc_flame, sr, 0.7), 1200.0, sr) * lap * flame_env
        hiss = dsp.lowpass(dsp.highpass(rng.standard_normal(m) ** 2, 2500.0, sr), 7000.0, sr) * flame_env
        sparse = call("crackle", ctx.child(m), rate=3.0 * flame_size + 1, grain_ms=1.0, f_lo=2000, f_hi=8000, q=1.5,
                      level=1.0) * flame_env
        # a match flame is quiet: about -24 dB re the flare, bigger flames louder (estimate)
        flame_gain = 10 ** ((-26 + 16 * flame_size) / 20)
        flame = flame_gain * (norm(woof) + 0.12 * norm(hiss)) + 0.12 * sparse
        burn = 0.5 * norm(pop) + 0.9 * norm(flare) + 0.5 * crk + flame
        out[ni:] += burn
    return out


# =================================================================== fire
@model("fire", "sound",
       "Fire after Farnell: hiss (squared noise, high-passed), crackles (Dust-triggered grains), low lapping "
       "'woof' and occasional whooshes.",
       [
           Param("intensity", 0.5, "", 0, 1, "size/fuel of the fire", automatable=True),
           Param("crackle_rate", 8.0, "events/s", 0, 200, "crackles per second at intensity 1"),
           Param("woof", 0.6, "", 0, 1, "low lapping flame level"),
           Param("hiss", 0.3, "", 0, 1, "hiss level"),
       ], ["Farnell 2010 practical 11", "Wikibooks: Designing Sound in SuperCollider/Fire",
           "Chadwick & James 2011"])
def fire(ctx: Ctx, x, intensity, crackle_rate, woof, hiss):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    I = ctrl(ctx, intensity)
    lap = np.clip(1 + 0.8 * dsp.lowpass(rng.standard_normal(n), 2.5, sr) / 0.035, 0.1, 3)
    w = dsp.lowpass(rng.standard_normal(n), 180.0, sr) * lap * I
    h = dsp.highpass(rng.standard_normal(n) ** 2 * np.sign(rng.standard_normal(n)), 1500.0, sr) * I
    c = call("crackle", ctx.child(), rate=[[i / sr, float(v)] for i, v in zip(range(0, n, 480), (crackle_rate * I)[::480])],
             grain_ms=2.0, f_lo=1000, f_hi=8000, q=1.5, size_exponent=2.0, level=1.0)
    return woof * norm(w) + hiss * 0.5 * norm(h) + 0.6 * c


# =================================================================== water
@model("water", "sound",
       "Liquid sounds from bubbles (the splash itself is nearly silent; Phillips et al. 2018): drip, stream, rain.",
       [
           Param("mode", "drip", "", choices=("drip", "stream", "rain")),
           Param("rate", 2.0, "events/s", 0.1, 20000, "drips/s, or bubble rate for stream/rain", automatable=True),
           Param("radius_mm", 3.0, "mm", 0.1, 20, "typical bubble radius (drip: ~3 mm -> ~1 kHz)"),
           Param("xi", 0.1, "", 0, 20, "pitch rise"),
       ], ["van den Doel 2005", "Phillips, Agarwal & Jordan 2018, Sci Rep", "Wikipedia: Minnaert resonance"])
def water(ctx: Ctx, x, mode, rate, radius_mm, xi):
    if mode == "drip":
        out = np.zeros(ctx.n)
        for i in dsp.poisson_times(ctx.rng, ctrl(ctx, rate), ctx.sr):
            r = radius_mm * 1e-3 * ctx.rng.uniform(0.8, 1.25)
            dsp.place(out, bubble_grain(ctx.sr, r, xi), i, 1.0)
        return out
    if mode == "stream":
        return call("bubbles", ctx, rate=rate, r_min_mm=radius_mm * 0.2, r_max_mm=radius_mm * 2, size_exponent=2.2, xi=xi)
    return call("bubbles", ctx, rate=rate, r_min_mm=0.15, r_max_mm=max(radius_mm, 0.3), size_exponent=3.0, xi=xi)


# =================================================================== hit
@model("hit", "sound", "Object struck by a hammer: impact -> modal resonator -> radiation.",
       [
           Param("material", "steel", "", choices=tuple(MATERIALS)),
           Param("shape", "bar", "", choices=("bar", "plate", "membrane", "string", "tube_open", "tube_closed")),
           Param("length", 0.3, "m", 0.01, 10), Param("width", 0.2, "m", 0.01, 10),
           Param("thickness", 0.006, "m", 0.0005, 0.3), Param("velocity", 1.5, "m/s", 0.05, 20),
           Param("contact_ms", 0.5, "ms", 0.05, 20), Param("bounces", 1, "count", 1, 40),
           Param("strike_pos", 0.22, "", 0, 1), Param("extra_damping", 2.0, "1/s", 0, 2000),
           Param("at", 0.02, "s", 0, None),
       ], ["van den Doel et al. 2001", "Klatzky et al. 2000"])
def hit(ctx: Ctx, x, material, shape, length, width, thickness, velocity, contact_ms, bounces, strike_pos, extra_damping, at):
    f = call("impact", ctx, at=at, velocity=velocity, contact_ms=contact_ms, bounces=bounces)
    y = call("modal", ctx, f, material=material, shape=shape, length=length, width=width, thickness=thickness,
             strike_pos=strike_pos, extra_damping=extra_damping)
    return call("radiation", ctx, y, radius=max(length, width) / 2, order=1)
