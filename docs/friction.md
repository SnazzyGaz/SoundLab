# Friction family (v0.27, v0.27.1), the creak model (v0.28) and stem-based squeals (v0.29)

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
`squeal_stems` (v0.29): `door_squeal`, `door_creak`, `hinge_squeak`, `hinge_squeak_2`, `gate_squeal`,
`gate_squeal_long`, `shoe_squeak`, `shoe_squeak_2`, `floorboard_creak` (below).
`creak` (v0.28) is kept as the physical fallback: its four recipes moved to `recipes/creak_model/` (any length,
an exact swing time, no stored data; `tools/godot_export.py` uses `creak_model/door_creak` for door open/close).
`rope_creak` and `door_creak_wood` were removed in v0.28. `creak_friction`, `squeak_friction` and `scrape_sdt`
stay as they were.

## v0.29: stem-based squeals (`squeal_stems`)

Physical models of a squeal (v0.27.1 relaxation oscillator, v0.28 `creak`) get the regime right but not the
detail that makes it believable: the exact way a squeal's pitch hooks, wobbles and breaks, how the noise
around it breathes, the cracks at the start of a motion. v0.29 keeps that detail from recordings without
storing audio: each burst of 15 CC0 recordings ([THIRD_PARTY.md](../THIRD_PARTY.md)) is analysed into stems,
and recipes recombine real stems.

### Method

Analysis (sinusoids + residual, Serra & Smith 1990): 44.1 kHz, 2.9 ms frames (hop 128). Each burst becomes

| stem | what it is |
|---|---|
| tonal | partial tracks: start frame, pitch (cents) and level (dB) per frame |
| micro | the dominant partial's wobble: pitch and level deviation from a 60/40 ms smoothed contour, replayed at its own rate (mirror-looped), never stretched |
| noise | the residual as 96 log-spaced band envelopes, with clicks bridged out, and its average shape |
| transients | the onset (first ~30 ms) and the clicks (frames more than 6 dB over a 30 ms median) |
| body | 1/3-octave long-term spectrum minus its tilt (includes tone energy, so it is a colour, not a clean body) |

Rendering (`render_burst`): partials as sines (pitch shift, time stretch, optional extra jitter), the noise
envelopes re-imposed on fresh noise, onset and clicks re-inserted, body colour applied as a smooth EQ. Any stem
can come from another burst (a donor): wobble, onset, noise colour or body. The stored file
(`soundlab/models/squeal_stems.bin`, about 2 MB, lzma) holds numbers only: tracks as cents and dB offsets
(float16) from an exact per-track start pitch and peak, noise envelopes in 0.6 dB steps, donors as their
derived arrays. The 35 stored bursts are what the recipes play plus a small pool for the `motions` macro.

### Gesture timeline

A recipe is a list of **motions** (`{"burst", "action", "speed", "pressure", "duration", "gap_after"}`). Each
motion plays one burst: `speed` stretches it (faster = shorter) with a small pitch rise (1.2 st per doubling),
`pressure` scales its level and tilts it brighter, `duration` fits it to a time. Between motions the hinge is
at rest: the pause is silence plus the room bed, drawn from the source recording's own pauses (x `stretch`,
x `gap_scale`) or set with `gap_after`. A door creak is open, pause, close; it is never a single squeal cut in
two. `motions_n` repeats the pattern with other bursts of the same kind.

### Recipes and their stems

| recipe | motions (bursts) | other stems | settings |
|---|---|---|---|
| `door_squeal` | door_766155 #5 | onset and body: door_563136 #1 | stretch 0.9, -1.5 st, jitter 2 c / 0.3 dB, noise above 1.5 kHz -12 dB, bed -14 dB, cut at 2.05 s (0.3 s fade) |
| `door_creak` | open door_575206 #2, close #1 | wobble door_455324 #0 at 40 %; body door_268574 #2 | stretch 1.1, +1 st, jitter 5 c / 0.6 dB; stray steady tones and the 150-330 Hz drone muted, that band's noise clamped |
| `hinge_squeak` | hinge_535530 #3+#4 joined | onset hinge_788701 #1, noise hinge_784964 #1, body hinge_204759 #2 | smoothed stems, half wobble, stretch 0.85, +2 st |
| `hinge_squeak_2` | open hinge_535530 #9, close #7 | own | stretch 1.08, -1 st |
| `gate_squeal` | as `hinge_squeak_2` | latch: gate_207140 #0, #1 | brightness 1; latch 0.35 s after, 6 dB under (macro on/off) |
| `gate_squeal_long` | gate_397687 #0 | wobble gate_483166 #0; onset and body gate_207140 #2 | stretch 1.2, -1 st, jitter 4 c / 0.5 dB |
| `shoe_squeak`, `shoe_squeak_2` | shoe_88502 #5 #6 #7 / #8 #9 #11 | own | +0.5 / -0.7 st |
| `floorboard_creak` | floor_124289 #8 #17 #27 | own | stretch 1.1, -2 st, -2 dB/oct |

Macros on every recipe: `brightness` (as recorded .. gate: +2 st, +2 dB/oct, +3 dB on partials above
2.5 kHz, no room bed, noise tails pulled down 10 dB: a hinge outdoors on metal), `stretch`, `pitch`,
`motions` (1-6), `gaps` (pause length) and `seed` (another take). `gate_squeal` adds `latch` on/off.
Variations reseed the model and jitter stretch and pitch.

### Lessons

- **Never average.** Averaging bursts or spectra smears exactly the detail that makes a squeal real (one
  hook, one break, one wobble rate) into a generic tone. Pick real stems and recombine them.
- **Each burst keeps its own wobble.** A borrowed wobble track that is stronger than the burst's own, plus
  added jitter, turned squeals into trills and bubbling. Borrow only when it measures like the own one
  (within 25 %, `borrow: if_similar`) or at reduced depth (`micro_depth`).
- **Smooth the noise, not the clicks.** Band envelopes of noise read every 2.9 ms are themselves random; playing
  them back on new noise doubles the flutter and bubbles. The anti-bubble pass (`smoothing`) power-smooths the
  noise envelopes (25-70 ms), low-passes the partials' own pitch/level (zero phase, 10/6 Hz) and returns
  short or very weak tracks ("birdies") to the noise. Clicks and onsets are separate stems, so they stay sharp.
  It is not always better: a hinge needed it, the door and floor creaks sound truer raw.
- **Remove stray tracks by rule, keep the gain.** A tracker also follows things that are not the squeal: a
  steady 3-6 kHz beep, a 190-250 Hz drone purring at 8-9 Hz. They are muted by rule (frequency band, length,
  pitch range) and kept in the list, so every random draw stays the same; the burst keeps the level it had
  with them (rendered both ways from the same random state), so removing a tone does not raise the rest.
- **Model the motion, not the clip.** An open and a close are two motions with a stop between them; the
  pause comes from the gesture timeline, and its length from the recording's own pauses.
- **A gate is a brighter, drier hinge.** Gate recordings bring their own problems (sheet-metal rattle, a
  single beep); the hinge with the brightness macro and a real latch reads better.

### Checks (`tests/test_squeal_stems.py`)

- Reproduction: every default recipe against a numeric fingerprint of its reference render
  (`tests/data/squeal_fingerprints.json`: 1/3-octave levels, 125 Hz-16 kHz, 50 ms frames, 0.5 dB steps).
  Mean level distance under 0.5 dB (measured 0.01-0.12 dB; another take of the same recipe is over 1.5 dB away).
  Not bit-exact: tracks are stored as float16 offsets, noise in 0.6 dB steps, and the render is resampled to 48 kHz.
- Gesture: open and close are separate motions at least 0.12 s apart, the pause 20 dB or more under both;
  `gaps` scales the pause, `motions` changes the count, long gestures fit their duration.
- No bubbling: the 8-30 Hz level modulation of the band envelopes on steady parts stays under 3.3 dB
  (bubbling renders: 3.9-4.5 dB). No low drone: the 150-330 Hz share stays under -15 dB for `door_creak`
  (the version with the ~200 Hz drone: -10 dB) and under -20 dB for hinges, gates and shoes.
- Data: numbers only, frame-rate arrays, every recording credited (`tests/test_no_reference_audio.py` also
  rejects any file that starts like audio, whatever its name).


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

Recipes (since v0.29 in `recipes/creak_model/`): `door_creak` (groan when slow, squeals on a faster swing; macros speed, length, pitch, irregularity,
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
