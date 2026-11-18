# Friction family (v0.27, v0.27.1) and the creak model (v0.28)

`friction_family` makes creaks, squeaks, scrapes, rubbing and bowing out of the same material bodies as
`material_impact` ([impacts.md](impacts.md)): the material table, shapes, damping laws and cavity modes are shared,
so a wooden door creaks through wood modes that die in a few tens of milliseconds instead of ringing like metal.

## Three ways of rubbing

| `mode` | What happens | Used for |
|---|---|---|
| `stick_slip` | Elasto-plastic friction (bristle model) solved sample by sample with the body's lowest modes. A springy contact (`drive_hz` > 0) sticks, loads up and slips: a creak whose pulse rate rises with speed and falls with pressure. A rigid contact (`drive_hz` = 0) locks onto a body mode: a squeal or bowed tone. The slips also ring the rest of the body (FFT convolution) and a short fibre-noise residual. | floorboard and chair creaks; glass rub |
| `scrape` | A fractal surface profile read at the contact speed (the `surface` block) pushes the whole modal body. | metal and stone scrapes, sandpaper on wood |
| `rub` | The same surface force band-limited, no modes. | cloth, padded rubbing |
| `squeal` (v0.27.1) | The stick-slip excitation itself: a slip-by-slip relaxation oscillator (slow stick ramp, quick slip) whose rate follows speed and pressure (`pitch_hz` at 0.05 m/s and 5 N, `pitch_follow`), with per-cycle period and amplitude jitter, missed slips, slip noise (`roughness`) and optional fibre `crackle`. The body only colours it through a modal response cut to 40 ms, and the level follows the speed, so the stroke ends without a knock. | model-level squeals (since v0.28 the creak and squeak recipes use `creak`, below) |

## Controls

`material`, `shape`, `size`, `hollowness`, `contact_point` (body); `length_s`, `speed` (m/s), `speed_change`
(-1 slowing to a stop .. +1 speeding up, x0.5..x2 over the stroke), `pressure` (N), `irregularity` (wobble of
speed and pressure, so slips are uneven); `drive_hz`, `mu_static`, `mu_dynamic`, `stiffness` (friction);
`roughness`, `grain_mm` (surface); `body`, `residual`, `muffle`, `max_t60`.

Every friction recipe has the same macros: material, pressure, speed, speed_change, irregularity, size,
hollowness, length (stroke and render length together) and distance (`cloth_rub` drops the body ones).

## Recipes

`friction_family`: `chair_creak`, `glass_rub`, `metal_scrape`, `stone_scrape`, `sandpaper`, `cloth_rub`.
`creak` (v0.28): `door_creak`, `hinge_squeak`, `shoe_squeak`, `floorboard_creak`. `rope_creak` and
`door_creak_wood` were removed in v0.28. `creak_friction`, `squeak_friction` and `scrape_sdt` stay as they were.

## v0.28: the creak model

The v0.27.1 squeals were a continuous 300-500 Hz relaxation oscillator with many harmonics and a hiss bed from
per-slip noise; heard on its own it read as a voice, not a hinge. v0.28 replaces it for creaks and squeaks with
`creak`, an event model whose numbers were measured from 32 CC0 recordings (credits in
[THIRD_PARTY.md](../THIRD_PARTY.md)), with the heavy door, the slow door going into a squeal, the gate and the
sneaker weighted most. Only the fitted numbers are stored (`soundlab/models/creak_data.py`).

Two regimes:

- **groan**: bursts of slip clicks (a sample impulse with a 0.1 ms tail) through fixed body resonances
  (parallel band-pass bank plus a low-pass tilt). The click rate follows speed (`pulse_hz` at 0.5, x speed^0.8,
  with jitter) and big slips every ~0.115 s jump +8.5 dB and decay at ~120 dB/s. The resonances do not move
  with speed. The level follows speed.
- **squeal**: a near-pure tone with a few harmonics, locked to one of the voice's resonance pitches. It
  glides with speed inside a burst (`glide_follow`), may hook up into pitch at the start (`rise_cents`), has
  a slow wobble of about 1 cent, and hops to another lock pitch between bursts (`hop`). No noise bed.

In `auto` a burst squeals once the speed passes `squeal_above` (a soft switch), so a slow swing groans and a
fast one squeals. `tick` and `scuff` add a sole's contact tick and a scuff that decays with the tone;
`tone_decay_db_s` lets a squeal ring down past its burst (the shoe).

Fitted numbers (from the recordings):

| | measured | model |
|---|---|---|
| heavy door groan (455324) | click rate 144 Hz (103-343); big slips every 0.115 s, +8.5 dB, 123 dB/s; resonances 930/950, 2120/1250, 4400/1200, 5680/700 Hz (centre/bandwidth); energy mostly under 2.5 kHz | `door_heavy` body, `pulse_hz` 145, low-pass tilt 2.8 kHz |
| slow door into squeal (766155) | squeals at 3.79 kHz (harmonics -86 dB) and 2.57 kHz with h2 at -1 dB gliding down to ~310 Hz over 1.9 s; late groan slips every 0.116 s | `door` voice 2570/3790/1900 Hz, harmonics 0/-1 dB |
| gate (397687) | bursts 0.3-0.9 s with ~40 ms gaps; locks 1100, 1050 and 514 Hz; rise of up to 330 cents at the start; up to 8 harmonics within 13 dB | `gate` body and voice; `hinge` voice 1100/1450/860 Hz |
| hinge (788701) | soft glides around 1.9-2.3 kHz with few harmonics | `hinge` voice |
| sneaker (88502) | contact tick 1-10 ms; f0 2400-2430 Hz, drifting down 35-115 cents; decays 91-115 dB/s; h2 -11 to -19 dB; a partial at 1.34-1.38 f0 that starts level with f0 and fades 10-15 dB faster; scuff noise 30-40 dB under the tone | `shoe` voice, `tone_decay_db_s` 100, `drift_cents` -70 |
| floor (124289, 506665) | click rate ~250 Hz; bursts 25-175 ms (median 60 ms) with ~0.27 s gaps, one longer main creak; slips +12 dB, 260 dB/s; resonances 570/250, 1680/1030, 3500/700, 6300/880 Hz | `floor` body, `pulse_hz` 240 |

Recipes: `door_creak` (groan when slow, squeals on a faster swing; macros speed, length, pitch, irregularity,
body door_heavy/door_light/gate, squeal, distance), `hinge_squeak` (squeal bursts that hop between hinge
pitches), `shoe_squeak` (tick, scuff and a decaying 2.4 kHz tone), `floorboard_creak` (irregular groan bursts).

Checks (`tests/test_creak.py`):

- No voice buzz: under 5 % of loud frames may show a strong 250-550 Hz fundamental with 8+ harmonics and HNR
  over 3 dB (v0.27.1 door creak 0.21, hinge 0.24, door_creak_wood 0.66; the recordings 0.00-0.13).
- No hiss bed: the 6-12 kHz band sits more than 38 dB under the strongest 0.2-6 kHz bin (v0.27.1 shoe -34 dB,
  the sneaker recording -41 dB).
- Groan: click rate 100-220 Hz at speed 0.5, up 1.3x or more when speed doubles; the smoothed spectral peak
  moves less than 10 % when speed doubles, for every body; the heavy-door peak sits at 750-1300 Hz with under
  15 % of the energy above 3 kHz; big slips give 5 dB or more of envelope range.
- Squeal: tone-to-floor over 40 dB (recordings 40-86 dB); at most 8 harmonics within 30 dB for the hinge, at
  most 4 for door and shoe; pitch rises 1.5x or more when speed rises through the stroke; the hinge squeaks in
  2 or more bursts of 0.08-1.2 s and hops between pitches across variations.
- Shoe: a broadband tick at onset, a 2.2-2.7 kHz tone 25 dB over the floor with its 1.3-1.4 f0 partial within
  15 dB, decaying 15-40 dB over 0.25 s.
- Nothing rings on after the stroke (40 dB down; the shoe gets 0.6 s to decay).
- The slow door creak keeps over 90 % of its energy under 2.5 kHz in most variations.

The first targets in the research notes (tonal bursts under 0.2 s, 2-4 harmonics, a hooked shoe chirp of
30-80 ms) were replaced by the detailed measurements above: the gate's bursts run 0.3-0.9 s, its first burst
has 8 harmonics within 13 dB, and the sneaker squeak is a ~0.3 s tone that decays rather than a hooked chirp.

## v0.27.1 changes

- A door creak is a dry hinge pin, not the wood: `door_creak_wood`, `hinge_squeak`, `shoe_squeak` and
  `rope_creak` moved to `squeal`, where the sound comes from the excitation rather than a ringing body.
- The contact now stays pressed after the motion stops (a door at rest still bears on its hinge). Releasing it
  at the stroke end let the wound-up contact slip once more and knock the body, which was audible at the end of
  every creak. `ep_friction`'s grit noise also stops with the motion.
- Tests: no knock after the stroke (the end is never the loudest part, squeals 30 dB down 50 ms later), squeals
  harmonic-rich and rough (5+ harmonics within 30 dB, HNR 0.5-12 dB, rope -3-12 dB; a pure sine fails both),
  squeal pitch follows speed, and the shoe squeak starts tonal (spectral flatness of the first 15 ms under 0.2).
- (v0.28: the recipe squeals moved to `creak`; `rope_creak` was removed.)

## Checks

- (v0.27, stick_slip creaks) Wood creaks keep their strongest sustained partial under 26 dB above the local spectrum (recorded door creaks:
  7-34 dB, median about 14); the steel hinge and the glass rub sit 15 dB or more above every wood creak.
- After the stroke a wood body is 45 dB down within 0.3 s; steel rings at least 15 dB longer.
- The creak pulse rate of a springy contact is 150-600 Hz at walking-door speeds (recordings: about 190-590 Hz),
  rises with speed and falls with pressure.
- Against CC0 door-creak recordings (kept outside the repo) the new door creak matches pulse rate, brightness
  and tonality, but on plain spectral distance it is no closer than the older `door_creak` recipe. Automatic
  fits only scored better by thinning the creak to a few clicks plus noise, so they were not used.
  `cloth_rub` was fitted to a CC0 cloth recording; the older `cloth_rustle` still scores closer.

## Cost

The friction loop is plain Python per sample: a springy creak keeps 2 modes in the loop (so the pulse rate
follows speed and pressure rather than locking to a body mode), a rigid squeal 6. About 0.5-1.3 s per second of
audio.

## Methods

Elasto-plastic friction after P. Dupont, V. Hayward, B. Armstrong, F. Altpeter (2002) and the Sound Design
Toolkit's friction model (F. Avanzini, S. Serafin, D. Rocchesso, 2005; GPL-3.0, method only); friction sound
categories after A. Akay, "Acoustics of friction", JASA 111(4), 2002; scraping and rolling through modal bodies
after K. van den Doel, P. G. Kry, D. K. Pai, "FoleyAutomatic" (2001).
