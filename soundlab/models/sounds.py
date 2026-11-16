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
       "Mallard call. v2: the syrinx is modelled as a rough, irregular pulse train (rough_pulses: low ~150-250 Hz "
       "rate, strong jitter and period doubling) rather than a smooth sung tone, then a nasal vocal tract: broad "
       "resonances around 1.2-3.3 kHz (strongest ~2 kHz) with an anti-resonance notch. Each quack swells then "
       "stops abruptly; pitch does not arc (an arc reads as a laugh). Female call = 2-10 quacks in a decrescendo; "
       "male = quieter, longer, breathier 1-2 note 'raeb'.",
       [
           Param("voice", "female", "", choices=("female", "male")),
           Param("notes", 6, "count", 1, 10, "quacks per call (female typical 5-6, range 2-10)",
                 source="Cornell All About Birds; Ducks Unlimited"),
           Param("f0", 200.0, "Hz", 100, 500, "mean syrinx pulse rate. Our refs show harmonic spacing ~170-210 Hz; "
                 "literature reports ~180-585 Hz", source="refs/SOURCES.md; Wildfowl journal (Greenewalt 1958)"),
           Param("formant_hz", 2100.0, "Hz", 1200, 3200, "main nasal resonance (max amplification 1.5-2.5 kHz)",
                 source="Wildfowl journal; our reference analysis"),
           Param("rasp", 0.7, "", 0, 1, "roughness: jitter, period doubling, shimmer and breath noise together. "
                 "Low = sung/'cartoon', high = real raspy quack"),
           Param("nasal", 0.6, "", 0, 1, "depth of the nasal anti-resonance and the high resonance cluster"),
           Param("note_s", 0.15, "s", 0.05, 0.5, "length of one quack (refs: 0.13-0.2 s)"),
           Param("gap_s", 0.1, "s", 0.02, 0.8, "silence between quacks (a hurried hen call ~0.1 s; a calm series "
                 "~0.4 s)"),
           Param("decrescendo_db", 2.0, "dB", 0, 8, "level drop per quack"),
           Param("pitch_fall", 0.02, "", 0, 0.2, "relative pitch drop per quack"),
           Param("start", 0.05, "s", 0, None, "start time"),
       ],
       ["Cornell All About Birds: Mallard sounds", "Wildfowl: Observations on sound production in the Anatidae",
        "Abs 1970, Vocalizations of the Mallard", "Ducks Unlimited duck-calling study", "Wikipedia: Syrinx",
        "Herzel et al. 1995 (subharmonics in voice)", "refs/SOURCES.md (Commons/xeno-canto mallard recordings)"])
def duck(ctx: Ctx, x, voice, notes, f0, formant_hz, rasp, nasal, note_s, gap_s, decrescendo_db, pitch_fall, start):
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
                   jitter=0.02 + 0.16 * rasp, period_doubling=0.6 * rasp, shimmer=0.15 + 0.45 * rasp,
                   pulse_ms=0.35, aspiration=0.1 + 0.3 * rasp)
        if male:
            fm = [[650, 2.5, 0], [1300, 2.0, 3], [fc * 0.9, 2.0, 2], [3000, 3, -6]]
        else:
            fm = [[0.55 * fc, 4.0, -3], [fc, 3.0, 8], [1.45 * fc, 5.0, 2 * nasal], [2.1 * fc, 7.0, -12 + 4 * nasal]]
        # the resonance slides down a little through the quack (beak closing; seen in the refs)
        shift = [[0, 1.06], [L / sr, 0.92]]
        y = call("formant", sub, src, formants=fm, shift=shift, dry=0.02)
        y = dsp.biquad(y, "peak", 0.33 * fc, 1.6, sr, -14 * nasal)  # nasal anti-resonance
        y = dsp.highpass(y, 380.0, sr, 0.9)
        # the energy stops fairly sharply above ~1.6 x the main resonance (refs: ~3.3 kHz)
        y = dsp.lowpass(dsp.lowpass(y, 1.65 * fc, sr, 0.9), 1.9 * fc, sr, 0.7)
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
       ],
       ["Baldan et al., SIVE 2015", "Farnell 2010 practical 22", "Borla: engine sound and drone physics",
        "Wikipedia: Firing order", "Heywood 1988, Internal Combustion Engine Fundamentals (blowdown)",
        "refs/SOURCES.md (Rover V8, start-idle-rev recordings)"])
def engine(ctx: Ctx, x, rpm, load, cylinders, firing, exhaust_length, muffler, jitter, intake, mechanical, whine,
           whine_order, pops, body, exhaust_noise, pipe_ring):
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
    flow = (R / 3000.0) ** 0.5
    pink = dsp.lowpass(dsp.pink(rng, n), 2500.0, sr)
    pink /= np.std(pink) + 1e-12
    y = np.zeros(n)
    gate_all = np.zeros(n)
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
            shape = np.where(u < 1, np.sin(np.pi * np.clip(u, 0, 1)) ** 2 * np.exp(-1.5 * u), 0.0)
            pulses += shape * a_cyc[cyc]
            gate += shape
        src = pulses * (0.3 + 0.7 * Ld)
        src = src + pink * gate * exhaust_noise * (0.6 + 1.2 * Ld) * flow
        src = dsp.highpass(src, 20.0, sr)
        D = 2 * exhaust_length * (1 + 0.07 * bi) / ctx.c * sr
        # the listener is nearer one tailpipe (estimate): bank imbalance is what makes a cross-plane V8 burble
        y += dsp.comb_waveguide(src, D, pipe_ring, sr, 1200.0, -1.0) * (1.0 - 0.3 * bi)
        gate_all += gate
    # muffler: chamber lowpass, low Helmholtz chamber peak, body shelf
    cutoff = 2800.0 * (1 - muffler) + 700.0 * muffler
    y = dsp.lowpass(y, cutoff, sr, 0.6)
    y = dsp.biquad(y, "peak", 70.0 + 40 * (1 - muffler), 1.2, sr, 3.0 + 3.0 * body)
    y = dsp.biquad(y, "lowshelf", 180.0, 0.7, sr, 7.0 * body)
    y = dsp.biquad(y, "highshelf", 1500.0, 0.7, sr, -4.0 * body)
    loud = (0.5 + 0.5 * Ld) * (R / 3000.0) ** 0.35  # refs: rev peaks ~10-15 dB above idle
    y = norm(y * loud)
    # intake roar: low-mid band noise, grows with airflow (load x rpm)
    roar = dsp.lowpass(dsp.highpass(dsp.pink(rng, n), 120.0, sr), 900.0, sr)
    roar = roar / (np.std(roar) + 1e-12) * (0.6 + 0.4 * dsp.lowpass(gate_all, 200.0, sr) / (gate_all.mean() + 1e-9))
    roar *= Ld * flow * intake * 0.12
    # block / valve-train: band noise 1.5-4 kHz rising with rpm + soft ticks
    blk = dsp.bandpass(rng.standard_normal(n), 2500.0, sr, 0.8) * (R / 3000.0) ** 1.5 * mechanical * 0.08
    ticks = np.zeros(n)
    if mechanical > 0:
        cam = np.cumsum(cyc_rate * 2 * cylinders) / sr
        idx = np.nonzero(np.diff(np.floor(cam)) > 0)[0]
        click = np.sin(2 * np.pi * 3200 * np.arange(120) / sr) * np.exp(-np.arange(120) / 20.0)
        for i in idx:
            dsp.place(ticks, click, i, rng.uniform(0.5, 1.0))
        ticks = dsp.bandpass(ticks, 3200.0, sr, 2.0) * mechanical * 0.05
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
    return y + roar + blk + ticks + whine_sig + pop_sig * 0.5


# =================================================================== foghorn
@model("foghorn", "sound",
       "Diaphone foghorn. Compressed air chopped by a reciprocating slotted piston (chopper) into a long conical "
       "trumpet (horn). v2: tuned to our reference recordings (fundamentals 87-150 Hz, energy concentrated in "
       "the 125-250 Hz octaves, very buzzy harmonics): lower default tone, sharper port edges, a bark at the "
       "start (pressure overshoot + pitch scoop), growl (flutter + period doubling) and a choppy grunt where the "
       "slowing piston audibly pulses. Several blasts can be chained.",
       [
           Param("tone_hz", 140.0, "Hz", 50, 600, "main tone. Our refs: 87-150 Hz; historic Type F ~180-250 Hz",
                 source="refs/SOURCES.md; Terry Pepper: the diaphone"),
           Param("grunt_hz", 85.0, "Hz", 40, 200, "end pitch of the grunt / the low tone", source="Terry Pepper"),
           Param("mode", "grunt", "", choices=("single", "grunt", "two_tone")),
           Param("blast_s", 2.2, "s", 0.3, 20, "length of the main tone"),
           Param("attack_s", 0.2, "s", 0.02, 3, "pressure build-up / piston spin-up (estimate)"),
           Param("grunt_s", 0.9, "s", 0.1, 6, "length of the falling grunt or the low tone (estimate)"),
           Param("pressure", 1.0, "", 0.3, 1.5, "air pressure re nominal 35-40 psi"),
           Param("roughness", 0.5, "", 0, 1, "growl: piston flutter + period doubling (0 = clean alarm-like tone)"),
           Param("chop", 0.6, "", 0, 1, "depth of the pulsing as the piston slows in the grunt"),
           Param("blasts", 1, "count", 1, 6, "number of blasts"),
           Param("interval_s", 3.5, "s", 0.5, 60, "start-to-start time between blasts"),
           Param("brightness", 0.4, "", 0, 1, "how much upper harmonic buzz gets out (lowpass 500 Hz .. 3 kHz); refs "
                 "have harmonics 2-5 strongest"),
           Param("horn_length", 4.0, "m", 0.5, 8, "trumpet length (estimate)"),
           Param("mouth_radius", 0.6, "m", 0.05, 1.5, "trumpet bell radius; sets the low cutoff c/(2 pi a) "
                 "(0.6 m -> 91 Hz)"),
           Param("start", 0.1, "s", 0, None, "start time"),
       ], ["Terry Pepper, Seeing The Light: the diaphone", "Wikipedia: Foghorn", "Herzel et al. 1995",
           "refs/SOURCES.md (East Brother Light, echoing foghorn recordings)"])
def foghorn(ctx: Ctx, x, tone_hz, grunt_hz, mode, blast_s, attack_s, grunt_s, pressure, roughness, chop, blasts,
            interval_s, brightness, horn_length, mouth_radius, start):
    sr, n = ctx.sr, ctx.n
    f, P, C = [[0, tone_hz * 0.6]], [[0, 0]], [[0, 0]]
    for k in range(int(blasts)):
        s0 = start + k * interval_s
        a, b = s0 + attack_s, s0 + attack_s + blast_s
        f += [[s0, tone_hz * 0.6], [a, tone_hz], [b, tone_hz]]
        P += [[s0, 0], [s0 + attack_s * 0.6, pressure * 1.15], [a, pressure], [b, pressure]]
        C += [[b, 0]]
        if mode == "single":
            f += [[b + 0.15, tone_hz * 0.95]]
            P += [[b + 0.2, 0]]
            end = b + 0.2
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
    F = ctrl(ctx, f, "exp")
    src = call("chopper", ctx, freq=f, pressure=P, open_fraction=0.5, edge=0.2, jitter=0.003, flow_noise=0.05)
    if roughness > 0:
        # period doubling: a weaker chopper an octave down (piston/air-column interaction; estimate)
        sub = call("chopper", ctx.child(), freq=[[t, v / 2] for t, v in f], pressure=P, open_fraction=0.3,
                   edge=0.2, jitter=0.006, flow_noise=0.0)
        src = src + 0.35 * roughness * sub
        flutter = dsp.lowpass(ctx.rng.standard_normal(n), 18.0, sr)
        src *= 1 + 0.5 * roughness * flutter / (np.std(flutter) + 1e-12) * 0.35
    # choppy grunt: amplitude pulses at a rate that slows with the piston (pulse rate ~ f / 8, estimate)
    Cc = np.clip(ctrl(ctx, C), 0, 1)
    if np.any(Cc > 0):
        ph = np.cumsum(np.clip(F / 8.0, 4.0, 25.0)) / sr
        am = 0.5 + 0.5 * np.cos(2 * np.pi * ph)
        src *= 1 - Cc * (1 - am ** 1.5)
    y = call("horn", ctx, src, length=horn_length, mouth_radius=mouth_radius, resonance=0.35, damp_hz=1500.0)
    lp = 500.0 * 6.0 ** brightness
    y = dsp.lowpass(dsp.lowpass(y, lp, sr, 0.6), lp * 1.5, sr, 0.6)
    y = dsp.biquad(y, "lowshelf", 160.0, 0.7, sr, 6.0)
    Pc = ctrl(ctx, P)
    hiss = dsp.bandpass(ctx.rng.standard_normal(n), 1200.0, sr, 0.8) * np.sqrt(np.maximum(Pc, 0)) * 0.008
    return norm(y) + hiss


# =================================================================== explosion
@model("explosion", "sound",
       "Detonation. v2 replaces v1's long swept noise band (which read as gas venting) with the parts a "
       "microphone actually records: a Friedlander N-wave (sharp rise, positive phase, negative phase) whose "
       "length scales with charge size; a short broadband crack that overloads the mic; a heavy low 'boom' "
       "(the pressure pulse ringing a low resonance of ground/air coupling); a rolling broadband roar whose "
       "band closes down as it decays; very low rumble; optional secondary blasts and debris. Put "
       "distance/ground/outdoor blocks after it for the far field.",
       [
           Param("size", 0.6, "", 0, 1, "overall scale: N-wave length, boom pitch and length, rumble weight"),
           Param("sub_blasts", 0, "count", 0, 20, "secondary detonations in the first ~0.5 s"),
           Param("fireball_s", 1.6, "s", 0.2, 10, "rumble T60"),
           Param("debris", 0.3, "", 0, 1, "falling debris amount"),
           Param("drive", 2.5, "", 0, 10, "mic overload on the shock (a close blast overloads any mic)"),
           Param("crack", 0.5, "", 0, 1, "level of the short high-frequency crack at the front"),
           Param("boom_hz", 0.0, "Hz", 0, 200, "low boom frequency; 0 = from size (75 Hz small .. 35 Hz large)"),
           Param("at", 0.05, "s", 0, None, "time of detonation"),
       ], ["Wikipedia: Blast wave (Friedlander waveform)", "Kinney & Graham 1985, Explosive Shocks in Air "
           "(positive-phase duration grows with cube root of yield)", "Farnell 2010 practical 31",
           "refs/SOURCES.md (Commons LS100155, oxyhydrogen detonation, huge explosion)"])
def explosion(ctx: Ctx, x, size, sub_blasts, fireball_s, debris, drive, crack, boom_hz, at):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    n0 = int(at * sr)
    m = n - n0
    t = np.arange(m) / sr
    # positive-phase duration: ms for small charges, tens of ms for large (cube-root scaling; our mapping)
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
    # boom: the pressure pulse rings a low, heavily damped resonance (estimate: ground/air coupling and mic
    # low-frequency response); bigger charges = lower and longer
    fb = boom_hz if boom_hz > 0 else 75.0 - 40.0 * size
    t60_b = 0.25 + 0.6 * size
    r = 10 ** (-3 / (t60_b * sr))
    w0 = 2 * np.pi * fb / sr
    from scipy.signal import lfilter
    boom = lfilter([1.0], [1.0, -2 * r * np.cos(w0), r * r], dsp.lowpass(shock, 400.0, sr))
    # rolling roar: broadband noise through a gentle (non-resonant, so no whistling sweep) lowpass that closes
    # from ~4 kHz to ~150 Hz as it decays; every reference shows broadband energy that dies top-down
    fc = 150.0 + 3850.0 * np.exp(-t / (0.12 + 0.3 * size))
    roll = dsp.svf(rng.standard_normal(m), fc, 0.5, sr, "lp")
    rollmod = np.clip(1 + 0.6 * dsp.lowpass(rng.standard_normal(m), 3.0, sr) / 0.03, 0.2, 2.5)
    roll *= dsp.ar_env(m, sr, 0.015, fireball_s) * rollmod
    low = dsp.lowpass(dsp.brown(rng, m), 60.0, sr, 0.8) * dsp.ar_env(m, sr, 0.05, fireball_s * 1.3)
    deb = np.zeros(m)
    if debris > 0:
        rate = [[0, 0], [0.3, 0], [0.5, 50 * debris], [1.5, 20 * debris], [3.0, 3 * debris], [5, 0]]
        deb = call("crackle", ctx.child(m), rate=rate, grain_ms=5, f_lo=500, f_hi=5000, q=3, size_exponent=2.2,
                   level=0.5)
    out = np.zeros(n)
    out[n0:] = (1.0 * front + (0.9 + 0.5 * size) * norm(boom) + 0.45 * norm(roll) + (0.3 + 0.4 * size) * norm(low)
                + 0.12 * deb)
    return out


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
               n_modes=20, extra_damping=40, strike_pos=0.3)
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
        crk = call("crackle", ctx.child(m), rate=[[0, 500], [flare_s * 0.5, 150], [flare_s, 20], [flare_s + 0.3, 6], [flare_s + 0.6, 0]],
                   grain_ms=1.0, f_lo=1500, f_hi=12000, q=1.5, size_exponent=2.4, level=1.0)
        flame_env = np.clip((tt - 0.02) / 0.08, 0, 1) * np.clip((flame_s + 0.05 - tt) / 0.05, 0, 1)
        flut = np.clip(1 + 0.5 * dsp.lowpass(rng.standard_normal(m), 8.0, sr) / 0.06, 0.3, 2.0)
        hiss = dsp.lowpass(dsp.highpass(rng.standard_normal(m), 900.0 + 1200 * brightness, sr), 12000.0, sr)
        hiss = norm(hiss) * flut * flame_env
        fc_flame = 180.0 + 700.0 * (1 - flame_size)
        woof = norm(dsp.bandpass(rng.standard_normal(m), fc_flame, sr, 0.7)) * flut * flame_env * (0.1 + 0.4 * flame_size)
        sparse = call("crackle", ctx.child(m), rate=4.0 * flame_size + 2, grain_ms=0.8, f_lo=2000, f_hi=10000,
                      q=1.5, level=1.0) * flame_env
        g = 10 ** (flame_db / 20)
        burn = 0.4 * norm(pop) + norm(burst) + 0.5 * crk + g * (hiss + woof) * 2.5 + 0.25 * sparse
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
