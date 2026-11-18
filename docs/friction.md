# Friction family (v0.27, v0.27.1)

`friction_family` makes creaks, squeaks, scrapes, rubbing and bowing out of the same material bodies as
`material_impact` ([impacts.md](impacts.md)): the material table, shapes, damping laws and cavity modes are shared,
so a wooden door creaks through wood modes that die in a few tens of milliseconds instead of ringing like metal.

## Three ways of rubbing

| `mode` | What happens | Used for |
|---|---|---|
| `stick_slip` | Elasto-plastic friction (bristle model) solved sample by sample with the body's lowest modes. A springy contact (`drive_hz` > 0) sticks, loads up and slips: a creak whose pulse rate rises with speed and falls with pressure. A rigid contact (`drive_hz` = 0) locks onto a body mode: a squeal or bowed tone. The slips also ring the rest of the body (FFT convolution) and a short fibre-noise residual. | floorboard and chair creaks; glass rub |
| `scrape` | A fractal surface profile read at the contact speed (the `surface` block) pushes the whole modal body. | metal and stone scrapes, sandpaper on wood |
| `rub` | The same surface force band-limited, no modes. | cloth, padded rubbing |
| `squeal` (v0.27.1) | The stick-slip excitation itself: a slip-by-slip relaxation oscillator (slow stick ramp, quick slip) whose rate follows speed and pressure (`pitch_hz` at 0.05 m/s and 5 N, `pitch_follow`), with per-cycle period and amplitude jitter, missed slips, slip noise (`roughness`) and optional fibre `crackle`. The body only colours it through a modal response cut to 40 ms, and the level follows the speed, so the stroke ends without a knock. | door creak (dry hinge pin), hinge squeak, shoe squeak, rope creak |

## Controls

`material`, `shape`, `size`, `hollowness`, `contact_point` (body); `length_s`, `speed` (m/s), `speed_change`
(-1 slowing to a stop .. +1 speeding up, x0.5..x2 over the stroke), `pressure` (N), `irregularity` (wobble of
speed and pressure, so slips are uneven); `drive_hz`, `mu_static`, `mu_dynamic`, `stiffness` (friction);
`roughness`, `grain_mm` (surface); `body`, `residual`, `muffle`, `max_t60`.

Every friction recipe has the same macros: material, pressure, speed, speed_change, irregularity, size,
hollowness, length (stroke and render length together) and distance (`cloth_rub` drops the body ones).

## Recipes

`door_creak_wood`, `floorboard_creak`, `chair_creak`, `rope_creak`, `hinge_squeak`, `shoe_squeak`, `glass_rub`,
`metal_scrape`, `stone_scrape`, `sandpaper`, `cloth_rub`. The older `door_creak`, `creak_friction`,
`squeak_friction` and `scrape_sdt` recipes stay as they were.

## v0.27.1 changes

- A door creak is a dry hinge pin, not the wood: `door_creak_wood`, `hinge_squeak`, `shoe_squeak` and
  `rope_creak` moved to `squeal`, where the sound comes from the excitation rather than a ringing body.
- The contact now stays pressed after the motion stops (a door at rest still bears on its hinge). Releasing it
  at the stroke end let the wound-up contact slip once more and knock the body, which was audible at the end of
  every creak. `ep_friction`'s grit noise also stops with the motion.
- Tests: no knock after the stroke (the end is never the loudest part, squeals 30 dB down 50 ms later), squeals
  harmonic-rich and rough (5+ harmonics within 30 dB, HNR 0.5-12 dB, rope -3-12 dB; a pure sine fails both),
  squeal pitch follows speed, and the shoe squeak starts tonal (spectral flatness of the first 15 ms under 0.2).
- The rope creak has no recorded reference to check against.

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
