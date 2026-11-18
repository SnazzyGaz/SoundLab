"""Numbers fitted from CC0 recordings for the `creak` model (v0.28). Only measured parameters live here; no audio
is stored in the repository. Sources (all CC0 1.0, Freesound; see THIRD_PARTY.md):

  455324  "Heavy Door Creaking_05" by rambler52     https://freesound.org/people/rambler52/sounds/455324/
  766155  "creaking door" by dianegray26            https://freesound.org/people/dianegray26/sounds/766155/
  397687  "Creaking Gate" by piermic                https://freesound.org/people/piermic/sounds/397687/
  88502   "sneaker squeak rubber" by shakaharu      https://freesound.org/people/shakaharu/sounds/88502/
  124289  "squeaky-floor-loud" by alienistcog       https://freesound.org/people/alienistcog/sounds/124289/
  506665  "Wood Creak Single V10" by Rudmer_Rotteveel https://freesound.org/people/Rudmer_Rotteveel/sounds/506665/
  788701  "Soft_sqeuak_door_hinge" by T.I.B           https://freesound.org/people/T.I.B/sounds/788701/

How they were measured: LPC (order 24-30) on groan passages for the fixed resonances; STFT peak tracking for squeal
pitch, harmonic levels relative to the fundamental and tone-to-floor; envelope jumps for the big slips; 5 ms
envelope thresholding for burst lengths and gaps.
"""

# Fixed body resonances heard through the slip clicks: (centre Hz, bandwidth Hz, gain dB).
BODIES = {
    # heavy door, 455324: LPC 931/948, 2117/1250, 4397/1206, 5681/704 (steady); 1127/1004, 2034/987, 4466/892,
    # 5599/599 (during slips); energy mostly under 2.5 kHz
    "door_heavy": [(1000.0, 1000.0, 0.0), (2100.0, 1100.0, -3.0), (4400.0, 1050.0, -13.0), (5640.0, 650.0, -16.0)],
    # lighter door, 766155 groan passages: LPC 1418/822, 3246/655, 4350/425 (and harmonic peaks near 680, 1350 Hz)
    "door_light": [(680.0, 300.0, -2.0), (1400.0, 820.0, 0.0), (3250.0, 650.0, -8.0), (4350.0, 430.0, -12.0)],
    # floorboards, 124289: 566/246, 1675/1032, 3505/702, 6305/879; 506665: 826/608, 2757/747, 5522/1124
    "floor": [(570.0, 250.0, 0.0), (820.0, 600.0, -3.0), (1700.0, 1000.0, -5.0), (3500.0, 700.0, -12.0), (6300.0, 880.0, -18.0)],
    # metal gate, 397687: 1197/403, 3121 (narrow), 4772/448, 6252 (narrow), 7804/434
    "gate": [(1200.0, 400.0, 0.0), (3120.0, 250.0, -4.0), (4770.0, 450.0, -8.0), (6250.0, 250.0, -10.0), (7800.0, 430.0, -14.0)],
}

# Spectral tilt of the groan (two-pole low-pass, Hz): 455324 is 10-20 dB down above 3 kHz; the light-door and
# floor recordings keep energy to ~5 kHz; the gate is bright.
BODY_TILT = {"door_heavy": 2800.0, "door_light": 5000.0, "floor": 4500.0, "gate": 9000.0}

# Squeal voices: lock pitches (Hz), harmonic levels in dB re the fundamental, an extra inharmonic partial
# (ratio, dB, extra decay dB/s), onset hook (cents below the lock pitch at the start), and slow pitch wobble (cents).
VOICES = {
    # 766155: a near-pure 3.7-3.9 kHz squeal (harmonics -86 dB) and a 2.57 kHz -> 313 Hz glide whose 2nd harmonic
    # is as loud as the fundamental (-0.9 dB) with nothing above (-61 dB); wobble ~1 cent per 5 ms
    "door": dict(pitches=[2570.0, 3790.0, 1900.0], harm=[0.0, -1.0, -60.0], extra=None, rise=60.0, wobble=1.0),
    # 397687: 1.05-1.10 kHz bursts (and one at 514 Hz) with a rich, slowly falling series; onset hooks of 90-330 cents
    "gate": dict(pitches=[1100.0, 1050.0, 514.0], harm=[0.0, -10.0, -10.0, -12.5, -11.0, -12.0, -10.0, -12.0], extra=None,
                 rise=200.0, wobble=1.5),
    # hinges: 397687 burst 4 (0, -12, -22, -22, -25, -29, -28, -37) and the 788701 glides (few harmonics)
    "hinge": dict(pitches=[1100.0, 1450.0, 860.0], harm=[0.0, -12.0, -22.0, -24.0, -28.0, -32.0], extra=None,
                  rise=120.0, wobble=1.5),
    # 88502: 2.40-2.43 kHz, 2nd -11/-19 dB, 3rd -27/-31 dB, 4th -32/-41 dB, an inharmonic partial at 1.34-1.38 f0;
    # the 1.34-1.38 f0 partial starts up to +5 dB over f0 and fades ~10-14 dB faster; drifts down 35-115 cents,
    # decays 91-115 dB/s, contact tick 1-10 ms, 5.5-8 kHz scuff noise 30-40 dB under the tone
    "shoe": dict(pitches=[2420.0, 2380.0, 2480.0], harm=[0.0, -15.0, -29.0, -36.0], extra=(1.355, -3.0, 140.0), rise=0.0,
                 wobble=2.0),
}

# Groan statistics. 455324: click periodicity 144 Hz (p10-p90 103-343, autocorrelation only 0.25: irregular),
# big slips every 0.115 s (0.10-0.20), +8.5 dB jumps decaying ~120 dB/s. 766155 late groan: slips every 0.116 s.
# Floor 124289: bursts 25-175 ms (median 60 ms) with ~0.27 s gaps, clicks ~250 Hz, slips +12 dB.
GROAN = dict(pulse_hz=145.0, pulse_jitter=0.3, slip_interval_s=0.115, slip_jump_db=8.5, slip_decay_db_s=120.0)
FLOOR = dict(burst_ms=60.0, gap_ms=270.0, pulse_hz=240.0)
SHOE = dict(f0=2420.0, drift_cents=-70.0, decay_db_s=100.0, tick_ms=3.0)
