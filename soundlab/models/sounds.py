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
       "Mallard call. The syrinx is a pulse train (rough_pulses) rather than a smooth sung tone, then a nasal vocal "
       "tract: broad "
       "resonances around 1.2-3.3 kHz (strongest ~2 kHz) with an anti-resonance notch. Each quack swells then "
       "stops abruptly; pitch does not arc (an arc reads as a laugh). Female call = 2-10 quacks in a decrescendo; "
       "male = quieter, longer, breathier 1-2 note 'raeb'. v3 (after 'slight grain'): pulses are placed at exact "
       "sub-sample times, much less jitter and breath noise (the reference mallard ref shows clean harmonics every "
       "~168 Hz up to 3 kHz; v2's heavy jitter smeared them into grain), a low resonance that lifts the second "
       "harmonic (~330 Hz line in every ref quack), and a gentle top roll-off instead of a steep cut.",
       [
           Param("voice", "female", "", choices=("female", "male")),
           Param("notes", 6, "count", 1, 10, "quacks per call (female typical 5-6, range 2-10)",
                 source="Cornell All About Birds; Ducks Unlimited"),
           Param("f0", 175.0, "Hz", 100, 500, "mean syrinx pulse rate. Our refs show harmonic spacing ~168-210 Hz; "
                 "literature reports ~180-585 Hz", source="refs/SOURCES.md; Wildfowl journal (Greenewalt 1958)"),
           Param("formant_hz", 2100.0, "Hz", 1200, 3200, "main nasal resonance (max amplification 1.5-2.5 kHz)",
                 source="Wildfowl journal; our reference analysis"),
           Param("rasp", 0.7, "", 0, 1, "roughness: jitter, period doubling, shimmer and breath noise together. "
                 "Low = sung/'cartoon', high = real raspy quack"),
           Param("nasal", 0.6, "", 0, 1, "depth of the nasal anti-resonance and the high resonance cluster"),
           Param("note_s", 0.22, "s", 0.05, 0.5, "length of one quack (refs: 0.13-0.27 s)"),
           Param("gap_s", 0.1, "s", 0.02, 0.8, "silence between quacks (a hurried hen call ~0.1 s; a calm series "
                 "~0.4 s)"),
           Param("decrescendo_db", 2.0, "dB", 0, 8, "level drop per quack"),
           Param("pitch_fall", 0.02, "", 0, 0.2, "relative pitch drop per quack"),
           Param("low_hz", 330.0, "Hz", 150, 800, "low (tracheal) resonance; refs show a strong line at ~330 Hz"),
           Param("low_level", 0.25, "", 0, 1.5, "level of the low resonance"),
           Param("start", 0.05, "s", 0, None, "start time"),
       ],
       ["Cornell All About Birds: Mallard sounds", "Wildfowl: Observations on sound production in the Anatidae",
        "Abs 1970, Vocalizations of the Mallard", "Ducks Unlimited duck-calling study", "Wikipedia: Syrinx",
        "Herzel et al. 1995 (subharmonics in voice)", "refs/SOURCES.md (Commons/xeno-canto mallard recordings)"])
def duck(ctx: Ctx, x, voice, notes, f0, formant_hz, rasp, nasal, note_s, gap_s, decrescendo_db, pitch_fall,
         low_hz, low_level, start):
    sr = ctx.sr
    out = np.zeros(ctx.n)
    male = voice == "male"
    if male:
        notes, note_s, f0, rasp = min(int(notes), 2), max(note_s, 0.28), f0 * 0.75, max(rasp, 0.85)
    t0 = start
    fc = formant_hz
    for k in range(int(notes)):
        L = int(note_s * (1 + 0.08 * ctx.rng.uniform(-1, 1)) * sr)
        if L < 64:
            break
        sub = ctx.child(L)
        tt = np.arange(L) / L
        # level: quick onset, swell, abrupt stop (seen in every reference quack)
        amp = np.clip(tt / 0.1, 0, 1) * (0.45 + 0.55 * tt ** 0.8) * np.clip((1 - tt) / 0.07, 0, 1)
        fk = f0 * (1 - pitch_fall * k)
        contour = fk * (1.04 - 0.08 * tt)  # gentle fall only
        step = 32
        src = call("rough_pulses", sub, f0=[[i / sr, float(v)] for i, v in zip(range(0, L, step), contour[::step])],
                   level=[[i / sr, float(v)] for i, v in zip(range(0, L, step), amp[::step])],
                   jitter=0.004 + 0.03 * rasp, period_doubling=0.35 * rasp, shimmer=0.06 + 0.2 * rasp,
                   pulse_ms=0.35, aspiration=0.02 + 0.05 * rasp)
        if male:
            fm = [[650, 2.5, 0], [1300, 2.0, 3], [fc * 0.9, 2.0, 2], [3000, 3, -6]]
        else:
            fm = [[0.55 * fc, 4.0, -3], [fc, 3.0, 8], [1.45 * fc, 5.0, 2 * nasal], [2.1 * fc, 7.0, -12 + 4 * nasal]]
        # the resonance slides down a little through the quack (beak closing; seen in the refs)
        shift = [[0, 1.06], [L / sr, 0.92]]
        y = call("formant", sub, src, formants=fm, shift=shift, dry=0.02)
        y = dsp.biquad(y, "peak", 0.5 * fc, 1.6, sr, -10 * nasal)  # nasal anti-resonance
        # v3: a weak low (tracheal/syrinx) resonance near 300-330 Hz, present in every mallard recording
        y = y + low_level * norm(dsp.bandpass(src, low_hz, sr, 4.0)) * np.max(np.abs(y))
        y = dsp.highpass(y, 200.0, sr, 0.7)
        # v3: gentle top roll-off (one 2-pole at 2.6 x the resonance) instead of v2's steep double lowpass;
        # real quacks have click energy up to ~10 kHz, 25-35 dB down. The steep cut made it dull/boxy.
        y = dsp.lowpass(y, 2.6 * fc, sr, 0.6)
        gain = 10 ** (-decrescendo_db * k / 20) * (0.35 if male else 1.0)
        dsp.place(out, norm(y) * gain, int(t0 * sr))
        t0 += L / sr + gap_s * (1 + 0.2 * ctx.rng.uniform(-1, 1))
    return out


# =================================================================== engine
FIRING = {
    "even": None,  # 720/cylinders spacing
    "v_twin_45": [0.0, 315 / 720],  # 45-degree V-twin fires at 0 and 315 crank degrees
    "crossplane_bank": [0.0, 180 / 720, 450 / 720, 540 / 720],  # one bank of a cross-plane V8: 180/270/90/180
    # cross-plane V8 heard from both tailpipes: each bank fires unevenly into its own pipe (the 'burble')
    "crossplane_v8": [[0.0, 180 / 720, 450 / 720, 540 / 720], [90 / 720, 270 / 720, 360 / 720, 630 / 720]],
}


@model("engine", "sound",
       "Combustion engine. v2: per-cylinder exhaust blowdown pressure pulses (not differentiated, so high RPM "
       "stays full instead of tinny) at the firing frequency RPM x cylinders / 120, firing-order offsets, "
       "cycle-to-cycle combustion variation (gives the half-order 'lumpy' body); broadband blowdown noise gated "
       "by the pulses (rasp); a lightly damped exhaust pipe (less ringing); muffler with a low Helmholtz chamber "
       "and body shelf; intake roar that grows with load; block/valve-train noise; transmission whine; pops. "
       "Loudness grows with RPM and load like a real rev.",
       [
           Param("rpm", 1800.0, "rev/min", 300, 12000, "engine speed", automatable=True),
           Param("load", 0.4, "", 0, 1, "throttle/load: pulse strength, noise, intake roar", automatable=True),
           Param("cylinders", 4, "count", 1, 12, "number of cylinders"),
           Param("firing", "even", "", choices=tuple(FIRING), doc="firing pattern (uneven = burble)"),
           Param("exhaust_length", 2.2, "m", 0.3, 6, "exhaust pipe length (closed-open waveguide)"),
           Param("muffler", 0.5, "", 0, 1, "muffler strength (0 = straight pipe)"),
           Param("jitter", 0.03, "", 0, 0.2, "cycle-to-cycle timing/amplitude variation"),
           Param("intake", 0.3, "", 0, 1, "intake roar level"),
           Param("mechanical", 0.15, "", 0, 1, "valve-train / block noise level"),
           Param("whine", 0.05, "", 0, 1, "transmission whine level"),
           Param("whine_order", 9.0, "x rev", 1, 60, "whine frequency as a multiple of crank rate (gear teeth; estimate)"),
           Param("pops", 0.3, "", 0, 1, "chance of exhaust pops when the throttle closes at speed"),
           Param("body", 0.6, "", 0, 1, "low-end weight: muffler chamber resonance + low shelf (our estimate)"),
           Param("exhaust_noise", 0.5, "", 0, 1, "broadband blowdown noise (rasp) relative to the tonal pulses"),
           Param("pipe_ring", 0.3, "", 0, 0.9, "exhaust pipe resonance feedback; high values ring/whistle"),
           Param("edge", 0.8, "", 0, 1, "v5: steepness of the blowdown pressure rise. 0 = slow smooth bump (dark, "
                 "v2-v4), 1 = sharp valve-opening front (bright, clear exhaust note)"),
           Param("rasp", 0.3, "", 0, 1.5, "v5: smooth broadband exhaust/flow turbulence (0.7-9 kHz), softly following the "
                 "firing. the reference V8 has a flat, smooth top above 400 Hz (frame kurtosis ~0); without it the engine sounds "
                 "muffled, and with spiky cracks instead it sounds crackly"),
           Param("exhaust_crack", 0.0, "", 0, 1, "v3: short broadband crack at the start of each blowdown (the "
                 "exhaust valve opening). Off by default since v5: it read as crackling on top; use `edge` for "
                 "brightness instead"),
       ],
       ["Baldan et al., SIVE 2015", "Farnell 2010 practical 22", "Borla: engine sound and drone physics",
        "Wikipedia: Firing order", "Heywood 1988, Internal Combustion Engine Fundamentals (blowdown)",
        "refs/SOURCES.md (Rover V8, start-idle-rev recordings)"])
def engine(ctx: Ctx, x, rpm, load, cylinders, firing, exhaust_length, muffler, jitter, intake, mechanical, whine,
           whine_order, pops, body, exhaust_noise, pipe_ring, edge, rasp, exhaust_crack):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    R = ctrl(ctx, rpm, "exp")
    Ld = np.clip(ctrl(ctx, load), 0, 1)
    cyc_rate = R / 120.0  # 4-stroke cycles per second
    tj = dsp.lowpass(rng.standard_normal(n), 30.0, sr)
    tj /= np.std(tj) + 1e-12
    phase = np.cumsum(cyc_rate * (1 + 0.15 * jitter * tj)) / sr
    spec = FIRING[firing]
    if spec is None:
        banks = [[i / cylinders for i in range(int(cylinders))]]
    elif isinstance(spec[0], list):
        banks = spec
    else:
        offs = list(spec)
        if len(offs) != cylinders:
            reps = int(math.ceil(cylinders / len(offs)))
            offs = [(o + r / reps) % 1 for r in range(reps) for o in offs][: int(cylinders)]
        banks = [offs]
    width = 0.16 - 0.04 * Ld  # blowdown lasts ~100-120 crank degrees of the 720 cycle (estimate after Heywood)
    # v5: at the tailpipe the pulse has steepened and shortened (nonlinear wave steepening in the pipe;
    # estimate): a sharper `edge` also means a shorter pulse
    width = width * (1 - 0.6 * float(np.clip(edge, 0, 1)))
    # rise time of the blowdown front: ~6 ms (edge 0, a slow bump) down to ~0.25 ms (edge 1, valve snapping
    # open); as a fraction of the pulse it depends on rpm (estimate after Heywood blowdown traces)
    rise_s = 0.25e-3 + 6e-3 * (1 - float(np.clip(edge, 0, 1))) ** 2
    r_edge = np.clip(rise_s * cyc_rate / width, 0.01, 0.5)
    flow = (R / 3000.0) ** 0.5
    pink = dsp.lowpass(dsp.pink(rng, n), 1800.0, sr)
    pink /= np.std(pink) + 1e-12
    y = np.zeros(n)
    gate_all = np.zeros(n)
    crack_gate = np.zeros(n)
    for bi, offs in enumerate(banks):
        pulses = np.zeros(n)
        gate = np.zeros(n)
        for o in offs:
            p = phase + o
            cyc = np.floor(p).astype(int)
            u = (p - cyc) / width
            ncyc = cyc.max() + 2
            a_cyc = 1 + jitter * 3 * rng.standard_normal(ncyc)
            a_cyc *= 1 + 0.06 * rng.standard_normal()
            # v5: blowdown rise time set by `edge`. A real blowdown starts with a steep pressure rise when the
            # valve cracks open, then decays; that steep front is what gives a clear, bright exhaust note
            # (harmonics into the kHz range) without any noise layer. v2-v4 used a slow symmetric bump
            # (edge 0), which needed noise/cracks on top to not sound muffled.
            uc_ = np.clip(u, 0, 1)
            rise = np.sin(0.5 * np.pi * np.clip(uc_ / r_edge, 0, 1)) ** 2
            fall = np.cos(0.5 * np.pi * np.clip((uc_ - r_edge) / (1 - r_edge), 0, 1)) ** 2 \
                * np.exp(-1.5 * np.clip(uc_ - r_edge, 0, 1))
            shape = np.where(u < 1, np.where(uc_ < r_edge, rise, fall), 0.0)
            pulses += shape * a_cyc[cyc]
            # v3: the blowdown noise follows a much sharper burst (shape^3: the first, fastest ~30 % of the
            # blowdown). With 8 cylinders x 0.16 cycle the old gate overlapped into a constant: stationary hiss.
            gate += shape ** 3 * a_cyc[cyc]
            # valve-opening crack: the first ~3 % of the cycle
            uc = (p - cyc) / 0.03
            # v4: each firing's crack is different (random strength, about a third almost silent). Identical
            # cracks on every firing are a pulse train at the firing rate (60-300 Hz): a buzz on top.
            c_amp = np.exp(0.7 * rng.standard_normal(ncyc)) * (rng.random(ncyc) > 0.35)
            crack_gate += np.where(uc < 1, np.exp(-3 * np.clip(uc, 0, 1)) * (1 - np.clip(uc, 0, 1)), 0.0) \
                * c_amp[cyc] * (1.0 - 0.3 * bi)
        src = pulses * (0.3 + 0.7 * Ld)
        src = src + pink * gate * exhaust_noise * (0.8 + 1.6 * Ld) * flow
        src = dsp.highpass(src, 20.0, sr)
        D = 2 * exhaust_length * (1 + 0.07 * bi) / ctx.c * sr
        # the listener is nearer one tailpipe (estimate): bank imbalance is what makes a cross-plane V8 burble
        y += dsp.comb_waveguide(src, D, pipe_ring, sr, 1200.0, -1.0) * (1.0 - 0.3 * bi)
        gate_all += gate
    # muffler: chamber lowpass, low Helmholtz chamber peak, body shelf
    cutoff = 2800.0 * (1 - muffler) + 700.0 * muffler
    # v5: a muffler's attenuation grows only gradually with frequency (expansion chambers + packing), so use
    # one gentle 2-pole at Q 0.5 plus an absorptive tilt, not a resonant-ish cut that made v4 sound muffled
    y = dsp.lowpass(y, cutoff * (1 + 2 * (1 - muffler)), sr, 0.5)
    y = dsp.biquad(y, "peak", 70.0 + 40 * (1 - muffler), 1.2, sr, 3.0 + 3.0 * body)
    y = dsp.biquad(y, "lowshelf", 180.0, 0.7, sr, 7.0 * body)
    y = dsp.biquad(y, "highshelf", 1500.0, 0.7, sr, -4.0 * body)
    loud = (0.5 + 0.5 * Ld) * (R / 3000.0) ** 0.35  # refs: rev peaks ~10-15 dB above idle
    y = norm(y * loud)
    # intake roar: low-mid band noise, grows with airflow (load x rpm)
    roar = dsp.lowpass(dsp.highpass(dsp.pink(rng, n), 120.0, sr), 700.0, sr)
    gl = dsp.lowpass(gate_all, 150.0, sr)
    roar = roar / (np.std(roar) + 1e-12) * (0.25 + 0.75 * gl / (gl.mean() + 1e-9))  # pulses with the intake strokes
    roar *= Ld ** 1.5 * flow * intake * 0.1
    # block / valve-train: band noise 1.5-4 kHz rising with rpm + soft ticks
    # v3: valve-train noise comes in bursts (valve seating), not as a steady band of hiss
    blk = dsp.bandpass(rng.standard_normal(n), 2200.0, sr, 0.8) * (R / 3000.0) ** 1.5 * mechanical * 0.03
    blk *= 0.2 + 0.8 * gl / (gl.max() + 1e-9)
    # plus a low floor that follows the firing activity (the refs never drop to silence between cracks)
    crack_gate = crack_gate + 1.2 * dsp.lowpass(crack_gate, 15.0, sr)
    cn = dsp.lowpass(dsp.highpass(dsp.pink(rng, n), 900.0, sr, 0.5), 5000.0, sr, 0.5)
    crk = cn / (np.std(cn) + 1e-12) * crack_gate
    crk *= exhaust_crack * 0.12 * (0.4 + 0.6 * Ld) * np.sqrt(loud)
    # v5 rasp: turbulent jet noise from the tailpipe plus flow noise, continuous but swelling with each firing
    # (envelope smoothed to ~40 Hz so it never becomes a crackle). Measured target: the V8 ref's 0.8-12.8 kHz
    # bands sit 20-28 dB under the total and are smooth (frame kurtosis 0.1)
    re_ = dsp.lowpass(gate_all, 40.0, sr)
    re_ = 0.45 + 0.55 * re_ / (np.percentile(re_, 99) + 1e-9)
    rn = dsp.lowpass(dsp.highpass(dsp.pink(rng, n), 700.0, sr, 0.5), 9000.0, sr, 0.5)
    rasp_sig = rn / (np.std(rn) + 1e-12) * re_ * rasp * 0.06 * (0.5 + Ld) * np.sqrt(loud)
    ticks = np.zeros(n)
    if mechanical > 0:
        cam = np.cumsum(cyc_rate * 2 * cylinders) / sr
        idx = np.nonzero(np.diff(np.floor(cam)) > 0)[0]
        # v3: broadband clicks (was a 3.2 kHz sine burst + Q 2 bandpass: a tonal 'tink' line in the spectrum)
        # v4: valves do not tick like a metronome: random timing (+-0.6 ms), strength, and some missing
        for i in idx:
            if rng.random() < 0.4:
                continue
            m = int(rng.uniform(0.3, 0.8) * 1e-3 * sr) + 2
            j = int(i + rng.normal(0, 0.6e-3 * sr))
            dsp.place(ticks, rng.standard_normal(m) * np.hanning(m), max(j, 0), np.exp(0.5 * rng.standard_normal()) * 0.6)
        ticks = dsp.bandpass(ticks, 2200.0, sr, 0.6) * mechanical * 0.012  # v5: quieter (top crackle)
    wph = 2 * np.pi * np.cumsum(R / 60.0 * whine_order) / sr
    whine_sig = (np.sin(wph) + 0.3 * np.sin(2 * wph)) * whine * 0.03 * Ld * (R / 3000.0)
    pop_sig = np.zeros(n)
    if pops > 0:
        dL = np.diff(Ld, prepend=Ld[0]) * sr
        lift = (dL < -0.5) | ((Ld < 0.08) & (R > 2500))
        rate = lift * pops * 6.0
        for i in dsp.poisson_times(rng, rate, sr):
            L = int(0.03 * sr)
            pp = friedlander(L, sr, 0.003 + 0.005 * rng.random()) + 0.3 * rng.standard_normal(L) * np.exp(-np.arange(L) / (0.006 * sr))
            dsp.place(pop_sig, pp, i, rng.uniform(0.3, 1.0) * 1.5)
        pop_sig = dsp.lowpass(dsp.comb_waveguide(pop_sig, 2 * exhaust_length / ctx.c * sr, 0.3, sr, 2000.0, -1.0), 3000.0, sr)
    return y + roar + blk + crk + rasp_sig + ticks + whine_sig + pop_sig * 0.5


# =================================================================== foghorn
@model("foghorn", "sound",
       "Air-driven foghorn (diaphone or large air horn). v3, rebuilt after 'sci-fi / digital' feedback: the tone "
       "is steady like a real horn (a few per cent scoop at most; real horns do not glide), with slow random "
       "pressure and pitch wobble from the air supply and the bell, turbulence noise gated by the air puffs, "
       "an asymmetric puff shape (not a square wave, whose missing even harmonics sound synthetic), the horn "
       "throat formants that make harmonics 3 and 6 stand out in real recordings, and a natural pressure "
       "release tail. The source is rendered 4x oversampled. The v2 grunt (pitch fall with choppy pulsing) and "
       "the octave-down growl are still there as options but off by default.",
       [
           Param("tone_hz", 110.0, "Hz", 50, 600, "main tone. Refs: barge horn ~109 Hz; diaphones 87-150 Hz; "
                 "historic Type F ~180-250 Hz", source="refs/SOURCES.md; Terry Pepper: the diaphone"),
           Param("grunt_hz", 85.0, "Hz", 40, 200, "end pitch of the grunt / the low tone", source="Terry Pepper"),
           Param("mode", "single", "", choices=("single", "grunt", "two_tone")),
           Param("blast_s", 2.2, "s", 0.3, 20, "length of the main tone"),
           Param("attack_s", 0.12, "s", 0.02, 3, "pressure build-up (refs: almost immediate)"),
           Param("grunt_s", 0.9, "s", 0.1, 6, "length of the falling grunt or the low tone (estimate)"),
           Param("pressure", 1.0, "", 0.3, 1.5, "air pressure re nominal 35-40 psi"),
           Param("roughness", 0.0, "", 0, 1, "growl: octave-down puff (period doubling) + fast flutter. 0 = clean"),
           Param("chop", 0.0, "", 0, 1, "depth of the pulsing as the piston slows in the grunt"),
           Param("blasts", 1, "count", 1, 6, "number of blasts"),
           Param("interval_s", 3.5, "s", 0.5, 60, "start-to-start time between blasts"),
           Param("brightness", 0.4, "", 0, 1, "how much upper harmonic buzz gets out (lowpass 500 Hz .. 3 kHz)"),
           Param("horn_length", 4.0, "m", 0.5, 8, "trumpet length (estimate)"),
           Param("mouth_radius", 0.6, "m", 0.05, 1.5, "trumpet bell radius; sets the low cutoff c/(2 pi a) "
                 "(0.6 m -> 91 Hz)"),
           Param("throat_hz", 330.0, "Hz", 100, 2000, "horn throat formant: harmonics near it and near 2x it "
                 "stand out (barge ref: harmonics 3 and 6)"),
           Param("wobble", 0.5, "", 0, 1, "slow random pressure/pitch wobble (air supply, bell; refs ~0.5-3 Hz)"),
           Param("release_s", 0.35, "s", 0.05, 3, "pressure release after the valve closes"),
           Param("scoop", 0.03, "", 0, 0.4, "pitch rise at the start (fraction); v2 used 0.4 (sounded synthetic)"),
           Param("start", 0.1, "s", 0, None, "start time"),
       ], ["Terry Pepper, Seeing The Light: the diaphone", "Wikipedia: Foghorn; Wikipedia: Air horn",
           "Fletcher & Rossing 1998, The Physics of Musical Instruments ch. 8 (horns)", "Herzel et al. 1995",
           "refs/SOURCES.md (East Brother Light, echoing foghorn, barge horn reference)"])
def foghorn(ctx: Ctx, x, tone_hz, grunt_hz, mode, blast_s, attack_s, grunt_s, pressure, roughness, chop, blasts,
            interval_s, brightness, horn_length, mouth_radius, throat_hz, wobble, release_s, scoop, start):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    lo = tone_hz * (1 - scoop)
    f, P, C = [[0, lo]], [[0, 0]], [[0, 0]]
    for k in range(int(blasts)):
        s0 = start + k * interval_s
        a, b = s0 + attack_s, s0 + attack_s + blast_s
        f += [[s0, lo], [a, tone_hz], [b, tone_hz]]
        P += [[s0, 0], [s0 + attack_s * 0.5, pressure * 0.8], [a, pressure * 1.03], [b, pressure]]
        C += [[b, 0]]
        if mode == "single":
            f += [[b + release_s, tone_hz * (1 - 0.6 * scoop - 0.01)]]
            P += [[b + release_s * 0.35, pressure * 0.35], [b + release_s, 0]]
            end = b + release_s
        elif mode == "grunt":
            f += [[b + 0.1, tone_hz * 0.88], [b + grunt_s, grunt_hz]]
            P += [[b + 0.1, pressure * 0.9], [b + grunt_s * 0.8, pressure * 0.4], [b + grunt_s, 0]]
            C += [[b + 0.1, chop], [b + grunt_s, chop]]
            end = b + grunt_s
        else:
            f += [[b + 0.1, grunt_hz * 1.05], [b + grunt_s, grunt_hz]]
            P += [[b + 0.1, pressure * 0.95], [b + grunt_s - 0.15, pressure * 0.85], [b + grunt_s, 0]]
            C += [[b + 0.1, chop * 0.4], [b + grunt_s, chop * 0.4]]
            end = b + grunt_s
        C += [[end + 0.01, 0]]
    # slow organic wobble of pressure (+-1.5 dB) and pitch (+-0.4 %), breakpoints every 20 ms
    if wobble > 0:
        dur = n / sr
        tw = np.arange(0, dur + 0.02, 0.02)
        w1 = dsp.lowpass(rng.standard_normal(len(tw)), 2.0, 50)
        w2 = dsp.lowpass(rng.standard_normal(len(tw)), 2.5, 50)
        w1 /= np.std(w1) + 1e-12
        w2 /= np.std(w2) + 1e-12
        Fb = ctrl_at(f, tw, log=True)
        Pb = ctrl_at(P, tw)
        f = [[float(t), float(v * (1 + 0.004 * wobble * a1))] for t, v, a1 in zip(tw, Fb, w1)]
        P = [[float(t), float(max(v * (1 + 0.17 * wobble * a2), 0.0))] for t, v, a2 in zip(tw, Pb, w2)]
    F = ctrl(ctx, f, "exp")
    # asymmetric air puffs (open 38 %): all harmonics present, unlike a 50 % square-ish puff
    src = call("chopper", ctx, freq=f, pressure=P, open_fraction=0.38, edge=0.25, jitter=0.002, flow_noise=0.12)
    if roughness > 0:
        sub = call("chopper", ctx.child(), freq=[[t, v / 2] for t, v in f], pressure=P, open_fraction=0.3,
                   edge=0.2, jitter=0.006, flow_noise=0.0)
        src = src + 0.35 * roughness * sub
        flutter = dsp.lowpass(rng.standard_normal(n), 18.0, sr)
        src *= 1 + 0.5 * roughness * flutter / (np.std(flutter) + 1e-12) * 0.35
    Cc = np.clip(ctrl(ctx, C), 0, 1)
    if np.any(Cc > 0):
        ph = np.cumsum(np.clip(F / 8.0, 4.0, 25.0)) / sr
        am = 0.5 + 0.5 * np.cos(2 * np.pi * ph)
        src *= 1 - Cc * (1 - am ** 1.5)
    y = call("horn", ctx, src, length=horn_length, mouth_radius=mouth_radius, resonance=0.15, damp_hz=1200.0)
    # throat formants (broad, so the harmonics near them are lifted without a ringing whistle)
    y = dsp.biquad(y, "peak", throat_hz, 2.0, sr, 7.0)
    y = dsp.biquad(y, "peak", 2.0 * throat_hz, 2.5, sr, 5.0)
    lp = 500.0 * 6.0 ** brightness
    y = dsp.lowpass(y, lp, sr, 0.5)  # gentle single 2-pole: real horns keep weak harmonics up to several kHz
    y = dsp.biquad(y, "lowshelf", 160.0, 0.7, sr, 4.0)
    Pc = np.maximum(ctrl(ctx, P), 0)
    # air rush at the mouth: pink, broad, follows the pressure
    hiss = dsp.bandpass(dsp.pink(rng, n), 900.0, sr, 0.5)
    hiss = hiss / (np.std(hiss) + 1e-12) * np.sqrt(Pc) * 0.012
    return norm(y) + hiss


def ctrl_at(value, times, log=False):
    """Evaluate a constant or breakpoint list at arbitrary times (helper for building automation)."""
    times = np.asarray(times, float)
    if not isinstance(value, (list, tuple)):
        return np.full(len(times), float(value))
    pts = sorted((float(a), float(b)) for a, b in value)
    ts = np.array([q[0] for q in pts])
    vs = np.array([q[1] for q in pts])
    if log and np.all(vs > 0):
        return np.exp(np.interp(times, ts, np.log(vs)))
    return np.interp(times, ts, vs)


# =================================================================== explosion
@model("explosion", "sound",
       "Detonation. v3, rebuilt after 'explosion in a plastic bucket' feedback. The v2 'boom' was a tuned 2-pole "
       "resonator ringing at ~120 Hz for ~0.6 s: a narrow, pitched resonance is exactly what a small cavity "
       "sounds like. v3 has no tonal parts. It is: a Friedlander N-wave (length grows with charge size) with a "
       "broadband crack that overloads the mic; a critically damped (non-ringing) low thump; and the main body, "
       "a long turbulent fireball roar of broadband noise with slow random surges, whose high end dies first "
       "(air absorption and a cooling fireball) while the low end lasts longest. Big refs stay within 10 dB of "
       "the peak for several seconds and fade over 5-7 s. Optional secondary blasts and debris. Put "
       "distance/ground/outdoor blocks after it for the far field.",
       [
           Param("size", 0.6, "", 0, 1, "overall scale: N-wave length, thump weight, roar length and darkness"),
           Param("sub_blasts", 0, "count", 0, 20, "secondary detonations in the first ~0.5 s"),
           Param("fireball_s", 4.0, "s", 0.2, 12, "how long the roar lasts (T60 of the low part)"),
           Param("debris", 0.3, "", 0, 1, "falling debris amount"),
           Param("drive", 2.5, "", 0, 10, "mic overload on the shock (a close blast overloads any mic)"),
           Param("crack", 0.5, "", 0, 1, "level of the short high-frequency crack at the front"),
           Param("boom_hz", 0.0, "Hz", 0, 300, "corner of the low thump (no resonance); 0 = from size"),
           Param("roar", 0.8, "", 0, 2, "level of the fireball roar re the front"),
           Param("surge", 0.5, "", 0, 1, "depth of the slow random surges in the roar"),
           Param("at", 0.05, "s", 0, None, "time of detonation"),
       ], ["Wikipedia: Blast wave (Friedlander waveform)", "Kinney & Graham 1985, Explosive Shocks in Air "
           "(positive-phase duration grows with cube root of yield)", "Farnell 2010 practical 31",
           "ISO 9613-1 (air absorption: high frequencies die first with distance)",
           "refs/SOURCES.md (Commons LS100155, oxyhydrogen detonation, huge explosion, explosion reference recordings)"])
def explosion(ctx: Ctx, x, size, sub_blasts, fireball_s, debris, drive, crack, boom_hz, roar, surge, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    n0 = int(at * sr)
    m = n - n0
    t = np.arange(m) / sr
    t_pos = 0.0015 + 0.035 * size ** 1.5
    shock = friedlander(m, sr, t_pos)
    crk = dsp.highpass(rng.standard_normal(m), 900.0, sr) * np.exp(-t / (0.004 + 0.01 * size)) * crack
    front = norm(shock) + 0.8 * crk
    for _ in range(int(sub_blasts)):
        i = int(rng.uniform(0.04, 0.5) * sr)
        if i < m:
            L = m - i
            tp = t_pos * rng.uniform(0.3, 0.7)
            sb = norm(friedlander(L, sr, tp)) + 0.6 * crack * dsp.highpass(rng.standard_normal(L), 900.0, sr) \
                * np.exp(-np.arange(L) / sr / 0.004)
            front[i:] += sb * rng.uniform(0.25, 0.6)
    front = dsp.softclip(front, drive) / max(dsp.softclip(np.array([1.0]), drive)[0], 1e-9) if drive > 0 else front
    # low thump: broadband pressure pulse, two cascaded 1-pole-like (Q 0.5) lowpasses: no ringing
    fb = boom_hz if boom_hz > 0 else 140.0 - 80.0 * size
    burst = shock + 0.5 * dsp.brown(rng, m) / 30.0 * np.exp(-t / (0.08 + 0.25 * size))
    thump = dsp.lowpass(dsp.lowpass(burst, fb, sr, 0.5), fb * 1.5, sr, 0.5)
    # fireball roar: pink noise, broadband; high band closes from ~8 kHz to ~250 Hz as the fireball cools.
    T = max(fireball_s, 0.2)
    fc = 300.0 + 9700.0 * np.exp(-t / (0.2 * T + 0.5 * size))
    body = dsp.svf(dsp.pink(rng, m), fc, 0.5, sr, "lp")
    # slow surges (billowing): smoothed noise, 0.3-3 Hz
    sg = dsp.lowpass(rng.standard_normal(m), 1.5, sr)
    sg /= np.std(sg) + 1e-12
    surge_env = np.exp(0.5 * surge * sg)  # log-normal, ~+-4 dB at surge 1
    # envelope: fast rise, a slowly sagging plateau (the fireball keeps burning for 0.4 x fireball_s), then a
    # decay with T60 = fireball_s
    rise = 1 - np.exp(-t / 0.03)
    hold = 0.4 * T
    env = rise * np.where(t < hold, 10 ** (-6 * t / hold / 20), 10 ** (-6 / 20) * np.exp(-6.91 * (t - hold) / T)) \
        * (1 + 0.6 * np.exp(-t / (0.2 + 0.3 * size)))
    body = norm(body * env * surge_env)
    low = dsp.lowpass(dsp.highpass(dsp.brown(rng, m), 30.0, sr, 0.7), 160.0 - 60.0 * size, sr, 0.5)
    low = norm(low * rise * np.exp(-6.91 * np.maximum(t - hold, 0) / (1.3 * T)) * surge_env)
    deb = np.zeros(m)
    if debris > 0:
        rate = [[0, 0], [0.3, 0], [0.5, 50 * debris], [1.5, 20 * debris], [3.0, 3 * debris], [5, 0]]
        deb = call("crackle", ctx.child(m), rate=rate, grain_ms=5, f_lo=400, f_hi=4000, q=0.8, size_exponent=2.2,
                   level=0.5)
    out = np.zeros(n)
    out[n0:] = (1.0 * front + (0.7 + 0.5 * size) * norm(thump) + roar * body + (0.3 + 0.4 * size) * low
                + 0.12 * deb)
    # v3: remove infrasound. The N-wave and brown noise put most of their energy below 20 Hz (v2 renders were
    # ~50 % infrasound, early v3 ~78 %): inaudible, it only eats headroom, so after normalising the audible
    # body is quieter and thinner on real speakers. A measurement mic + preamp also rolls off there.
    return dsp.highpass(dsp.highpass(out, 22.0, sr, 0.54), 22.0, sr, 1.31)


# =================================================================== match
@model("match", "sound",
       "Match strike and ignition. v2, retimed to our reference recordings: a short stick-slip scrape of the "
       "head over the striker exciting the matchbox, an ignition burst that is the loudest and brightest part "
       "(broadband 1-14 kHz with a dense crackle), then a steady flame that is mostly a bright hiss about 15-25 dB "
       "below the burst with sparse crackle and only a little low 'woof'.",
       [
           Param("strike_s", 0.08, "s", 0.02, 1, "length of the scrape (refs: 0.03-0.1 s)"),
           Param("strike_speed", 0.9, "m/s", 0.05, 3, "peak scrape speed (estimate)"),
           Param("strike_force", 3.0, "N", 0.2, 15, "pressing force (estimate)"),
           Param("ignite", True, "", None, None, "whether it lights (failed strikes are realistic too)"),
           Param("flare_s", 0.25, "s", 0.03, 2, "ignition burst decay T60"),
           Param("flame_s", 1.0, "s", 0, 30, "how long the steady flame lasts"),
           Param("flame_size", 0.4, "", 0, 1, "flame size: match ~0.3, lighter ~0.5, torch ~1"),
           Param("flame_db", -20.0, "dB", -45, -5, "flame hiss level re the ignition burst (refs: about -20 dB)"),
           Param("brightness", 0.6, "", 0, 1, "spectral tilt of burst and flame (close mics hear more top)"),
           Param("start", 0.05, "s", 0, None, "start time"),
       ], ["Avanzini, Serafin & Rocchesso DAFx 2002", "Tsugi GameSynth physical modules (match friction)",
           "Farnell 2010 practical 11; Wikibooks Designing Sound in SuperCollider: Fire",
           "Wikipedia: Match; Stick-slip phenomenon", "Chadwick & James 2011 (fire sound)",
           "refs/SOURCES.md (Freesound CC0 match strikes)"])
def match(ctx: Ctx, x, strike_s, strike_speed, strike_force, ignite, flare_s, flame_s, flame_size, flame_db,
          brightness, start):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    s0, s1 = start, start + strike_s
    vel = [[0, 0], [s0, 0], [s0 + strike_s * 0.3, strike_speed], [s0 + strike_s * 0.8, strike_speed * 0.8], [s1, 0]]
    force = [[0, 0], [s0, strike_force * 0.6], [s0 + strike_s * 0.5, strike_force], [s1, strike_force * 0.4],
             [s1 + 0.005, 0]]
    sub = ctx.child()
    fr = call("friction", sub, velocity=vel, normal_force=force, mu_static=0.7, mu_kinetic=0.4, stribeck_v=0.02,
              f_osc=1400, zeta=0.08, roughness=0.9, roughness_hz=6000)
    box = call("modal", sub, fr, material="cardboard", shape="plate", length=0.05, width=0.035, thickness=0.0015,
               n_modes=20, extra_damping=180, strike_pos=0.3)  # v3: cardboard is very lossy (was 40: tinny)
    V = ctrl(ctx, vel)
    grit = dsp.bandpass(rng.standard_normal(n), 3500.0, sr, 0.7) * (V / max(strike_speed, 1e-3)) ** 1.5
    grit *= 1 + 0.8 * dsp.lowpass(rng.standard_normal(n), 300.0, sr) / 0.1
    out = 0.5 * (0.6 * norm(fr) + 0.5 * norm(box) + 0.4 * norm(grit))
    if ignite:
        ti = s0 + strike_s * 0.6
        ni = int(ti * sr)
        m = n - ni
        tt = np.arange(m) / sr
        pop = friedlander(m, sr, 0.0008)
        burst_env = dsp.ar_env(m, sr, 0.004, flare_s)
        burst = dsp.highpass(rng.standard_normal(m), 700.0, sr)
        burst = dsp.biquad(burst, "highshelf", 5000.0, 0.7, sr, -9 + 12 * brightness) * burst_env
        burst *= 1 + 0.6 * np.abs(dsp.lowpass(rng.standard_normal(m), 150.0, sr)) / 0.05
        # v4: the head's sputter is a short spray of sparks (refs: ~0.1 s), not 500 grains/s for half a second,
        # which read as "bacon frying on high heat"
        crk = call("crackle", ctx.child(m), rate=[[0, 160], [flare_s * 0.4, 50], [flare_s, 8], [flare_s + 0.3, 2],
                                                   [flare_s + 0.5, 0]],
                   grain_ms=1.2, f_lo=1200, f_hi=8000, q=0.7, size_exponent=2.6, level=1.0)
        flame_env = np.clip((tt - 0.02) / 0.08, 0, 1) * np.clip((flame_s + 0.05 - tt) / 0.05, 0, 1)
        # v3: flame flutter has slow (~5 Hz, flicker) and fast (~30 Hz, turbulence) parts; the hiss is pink-ish
        # with a soft top instead of flat white noise (white hiss above 5 kHz read as 'tinny')
        f1 = dsp.lowpass(rng.standard_normal(m), 6.0, sr)
        f2 = dsp.lowpass(rng.standard_normal(m), 30.0, sr)
        flut = np.exp(0.45 * f1 / (np.std(f1) + 1e-12) + 0.25 * f2 / (np.std(f2) + 1e-12))
        # a small flame's hiss is a soft band around 1-4 kHz (the reference match ref: the flame is a low rumble with
        # sparse crackle; a bright 2-20 kHz hiss band was the remaining 'tinny' part)
        hiss = dsp.highpass(dsp.pink(rng, m), 900.0 + 300 * brightness, sr)
        hiss = dsp.lowpass(hiss, 2500.0 + 1500.0 * brightness, sr, 0.5)
        hiss = norm(hiss) * flut * flame_env
        # v3: the 'woof' (buoyant flame flow) sits low, 90-250 Hz; v2's ~530 Hz band was a honk
        fc_flame = 90.0 + 160.0 * (1 - flame_size)
        woof = norm(dsp.lowpass(dsp.pink(rng, m), fc_flame, sr, 0.5)) * flut * flame_env * (0.3 + 0.7 * flame_size)
        sparse = call("crackle", ctx.child(m), rate=4.0 * flame_size + 2, grain_ms=0.8, f_lo=2000, f_hi=10000,
                      q=0.7, level=1.0) * flame_env
        g = 10 ** (flame_db / 20)
        # hand/box thump as the head catches and the hand jerks back (refs: energy below 100 Hz)
        thump = dsp.lowpass(friedlander(m, sr, 0.006), 120.0, sr, 0.5)
        burn = 0.4 * norm(pop) + norm(burst) + 0.35 * crk + g * (hiss + woof) * 2.5 + 0.2 * sparse \
            + 0.1 * norm(thump)
        out[ni:] += burn
    return dsp.highpass(out, 25.0, sr, 0.7)  # v3: no infrasound from the pop/thump pulses


# =================================================================== shaker / matchbox
@model("shaker", "sound",
       "Things rattling in a shaken box (matchbox by default). Physics: the hand moves the box back and forth; "
       "at each reversal the loose contents keep going and slam into the far wall and into each other. So "
       "every stroke gives a cluster of many tiny impacts (wood on wood: bright, very short) a few ms to tens of "
       "ms long, and the bigger hits excite the box itself (a small, very lossy cardboard plate, plus a dull "
       "body knock). Between strokes the contents slide (soft noise). After Cook's PhISM shakers.",
       [
           Param("rate_hz", 3.0, "Hz", 0.5, 15, "shake strokes per second (each back-and-forth = 2 reversals; "
                 "the reference clip ~3)"),
           Param("shake_s", 0.8, "s", 0.05, 30, "how long the shaking lasts"),
           Param("strokes", 0, "count", 0, 200, "explicit number of reversals (0 = from rate_hz x shake_s)"),
           Param("items", 40, "count", 1, 300, "number of loose items (matches in a full box: ~40)"),
           Param("intensity", 0.7, "", 0, 1, "how hard it is shaken (more and louder impacts, more box knock)"),
           Param("spread_ms", 50.0, "ms", 1, 150, "how long a reversal's cluster of collisions lasts (the reference clip: "
                 "bursts of ~50-100 ms)"),
           Param("item_hz", 4500.0, "Hz", 500, 15000, "centre of the item click spectrum (thin wood sticks)"),
           Param("box_size", 0.05, "m", 0.01, 0.5, "box length (sets the body knock and plate modes)"),
           Param("fill", 0.8, "", 0, 1, "how full the box is (fuller = more, duller hits, less free flight)"),
           Param("knock", 0.5, "", 0, 2, "level of the box body knock"),
           Param("handling", 0.05, "", 0, 2, "low rumble of the hand moving the box near the mic (the reference clip has a "
                 "little below ~100 Hz). v5: default 0.05; at 0.3-0.9 it made the box sound heavy, like a dog pawing at "
                 "a door"),
           Param("card_mm", 0.5, "mm", 0.2, 3, "v5: box wall thickness. A matchbox is thin card (~0.5 mm): its "
                 "modes sit high and die in a few ms"),
           Param("slide", 0.15, "", 0, 1, "level of the soft sliding between strokes"),
           Param("start", 0.05, "s", 0, None, "start time"),
       ], ["Cook 1997, Physically Informed Sonic Modeling (PhISM shakers), Computer Music Journal 21(3)",
           "Farnell 2010 (rattles)", "matchbox shake reference (analysis only)"])
def shaker(ctx: Ctx, x, rate_hz, shake_s, strokes, items, intensity, spread_ms, item_hz, box_size, fill, knock,
           handling, card_mm, slide, start):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    nrev = int(strokes) if strokes > 0 else max(1, int(round(2 * rate_hz * shake_s)))
    half = 1.0 / (2 * rate_hz)
    clicks = np.zeros(n)
    knocks = np.zeros(n)
    slide_env = np.zeros(n)
    t = start
    for r in range(nrev):
        if r:
            t += half * (1 + 0.12 * rng.standard_normal())  # human timing: strokes are not metronomic
        i0 = int(t * sr)
        if i0 >= n:
            break
        force = intensity * (0.6 + 0.4 * rng.random()) * (1.0 if r % 2 == 0 else 0.8)
        # the bulk of the contents slams into the wall: one knock per reversal
        # v4: a soft ~2-4 ms push of the whole bundle (v3 used a 1-sample impulse: a broadband click)
        # v5: the bundle of matches weighs ~4 g: a light, short push (0.6-1.5 ms), not v4's 2-4 ms thud
        mk = max(4, int(rng.uniform(0.6e-3, 1.5e-3) * sr))
        dsp.place(knocks, np.sin(np.pi * np.arange(mk) / mk) ** 2 / (mk / 2), i0 + int(rng.uniform(0, 3e-3) * sr), force)
        # the cluster of item collisions
        # v4: only part of the contents hit per reversal and they arrive as a smeared group: most of the box
        # moves as one mass (a dense, packed box), so a few clear knocks plus a soft cluster, not a spray
        k = rng.poisson(max(1.0, items * (0.08 + 0.2 * intensity) * (1.2 - 0.5 * fill)))
        dt = np.abs(rng.gamma(2.0, spread_ms * 1e-3 / 5.0, k))
        amps = force * np.minimum(rng.pareto(3.0, k) + 0.3, 3.0)
        for d, a in zip(dt, amps):
            j = i0 + int(d * sr)
            # wood on wood / wood on card: smooth (sin^2) contacts of 0.15-0.5 ms (v3: 0.05-0.25 ms half-sines:
            # needle-sharp clicks with a hard spectral edge)
            # v5: a matchstick weighs ~0.1 g, so its contact time is tiny (Hertz contact, ~0.05-0.15 ms);
            # v4's 0.1-0.35 ms contacts cut everything above ~3 kHz and made the sticks sound heavy
            m = max(3, int(rng.uniform(0.05, 0.15) * 1e-3 * sr))
            dsp.place(clicks, np.sin(np.pi * np.arange(m) / m) ** 2 * rng.choice([-1, 1]), j, a)
        L = int(half * 0.7 * sr)
        if L > 8:
            dsp.place(slide_env, np.sin(np.pi * np.arange(L) / L) ** 2 * force * 0.3, i0 + int(0.15 * half * sr))
    # item clicks: two damped stick/box resonances (refs: peaks ~1.5 and ~3.5 kHz), no bare high-pass part
    # v5: a fir stick 45 x 2 mm has its first free-free bending mode near f1 = 1.03 c t / L^2 ~ 4.8 kHz
    # (c ~ 4700 m/s along the grain); the 2nd sits at 2.76 f1 (very damped). The card wall adds 1-3 kHz.
    # the reference is roughly flat from 800 Hz to 6.4 kHz (octave bands within ~2 dB)
    it = dsp.bandpass(clicks, item_hz, sr, 1.4) + 0.3 * dsp.bandpass(clicks, item_hz * 2.76, sr, 1.5) \
        + 0.45 * dsp.bandpass(clicks, item_hz * 0.4, sr, 0.9)
    it = dsp.lowpass(it, 11000.0, sr, 0.5)
    # box: lossy cardboard plate + a damped body knock
    box = call("modal", ctx.child(), clicks * 0.3 + 0.3 * knocks, material="cardboard", shape="plate",
               length=box_size, width=box_size * 0.7, thickness=card_mm * 1e-3, n_modes=16, extra_damping=400,
               strike_pos=0.3)
    body_hz = 9.0 / max(box_size, 0.01)  # ~180 Hz for a 5 cm box (estimate)
    kn = dsp.bandpass(knocks, body_hz * 2, sr, 0.7)
    mot = dsp.lowpass(slide_env, 30.0, sr)
    sl = dsp.bandpass(dsp.pink(rng, n), 2500.0, sr, 0.4) * mot
    hand = dsp.lowpass(dsp.lowpass(dsp.brown(rng, n), 100.0, sr, 0.5), 150.0, sr, 0.5)
    hand = dsp.highpass(hand, 25.0, sr, 0.7) * dsp.lowpass(slide_env, 8.0, sr)
    # a matchbox is small: its body adds little below ~500 Hz (the refs dip at 300-600 Hz)
    # v5: thin card cannot hold low-mid energy: body high-passed at 800 Hz (v4: 500 Hz, which read as wood)
    body = dsp.highpass(0.35 * norm(box) * (0.5 + 0.5 * intensity) + knock * norm(kn) * intensity, 800.0, sr, 0.6)
    return 0.9 * norm(it) + body + slide * norm(sl) + handling * norm(hand)


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
@model("hit", "sound",
       "Object struck by a hammer, stick or knuckle: impact -> modal resonator -> radiation. v0.8 adds what a "
       "real knock on a board or door has besides the ring: many dense, detuned modes (a plank or panel is not "
       "a tuned bar), the contact noise of the striker (knuckle skin/bone, stick tip) and a low 'thump' (the "
       "panel moving as a whole). With n_modes 24, detune 0.01, contact_noise 0 and thump 0 it is exactly the "
       "v0.7 model (the quick resonant 'dowel' sound).",
       [
           Param("material", "steel", "", choices=tuple(MATERIALS)),
           Param("shape", "bar", "", choices=("bar", "plate", "membrane", "string", "tube_open", "tube_closed")),
           Param("length", 0.3, "m", 0.01, 10), Param("width", 0.2, "m", 0.01, 10),
           Param("thickness", 0.006, "m", 0.0005, 0.3), Param("velocity", 1.5, "m/s", 0.05, 20),
           Param("contact_ms", 0.5, "ms", 0.05, 20), Param("bounces", 1, "count", 1, 40),
           Param("strike_pos", 0.22, "", 0, 1), Param("extra_damping", 2.0, "1/s", 0, 2000),
           Param("n_modes", 24, "count", 1, 200, "number of modes (a panel: many; a bar: few)"),
           Param("detune", 0.01, "", 0, 0.2, "random mode detuning (grain, joints, glue: less pure ring)"),
           Param("contact_noise", 0.0, "", 0, 3, "level of the striker's contact noise (knuckle/stick slap)"),
           Param("noise_hz", 1500.0, "Hz", 200, 12000, "centre of the contact noise"),
           Param("thump", 0.0, "", 0, 3, "level of the low whole-panel thump"),
           Param("at", 0.02, "s", 0, None),
       ], ["van den Doel et al. 2001", "Klatzky et al. 2000", "Cook 2002, Real Sound Synthesis (contact noise)"])
def hit(ctx: Ctx, x, material, shape, length, width, thickness, velocity, contact_ms, bounces, strike_pos, extra_damping,
        n_modes, detune, contact_noise, noise_hz, thump, at):
    f = call("impact", ctx, at=at, velocity=velocity, contact_ms=contact_ms, bounces=bounces)
    y = call("modal", ctx, f, material=material, shape=shape, length=length, width=width, thickness=thickness,
             strike_pos=strike_pos, extra_damping=extra_damping, n_modes=n_modes, detune=detune)
    y = call("radiation", ctx, y, radius=max(length, width) / 2, order=1)
    if contact_noise <= 0 and thump <= 0:
        return y
    pk = np.max(np.abs(y)) + 1e-12
    sr = ctx.sr
    if contact_noise > 0:
        # contact noise follows the force (plus a few ms of surface/skin rattle)
        signal_env = np.convolve(np.abs(f), np.exp(-np.arange(int(0.004 * sr)) / (0.0015 * sr)))[: len(f)]
        nz = dsp.bandpass(ctx.child().rng.standard_normal(len(f)), noise_hz, sr, 0.6) * signal_env
        y = y + contact_noise * 0.6 * pk * norm(nz)
    if thump > 0:
        th = dsp.lowpass(dsp.lowpass(np.diff(f, prepend=0.0), 160.0, sr), 160.0, sr)
        y = y + thump * 0.6 * pk * norm(th)
    return y
