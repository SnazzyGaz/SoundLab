# Friction family (v0.27)

`friction_family` makes creaks, squeaks, scrapes, rubbing and bowing out of the same material bodies as
`material_impact` ([impacts.md](impacts.md)): the material table, shapes, damping laws and cavity modes are shared,
so a wooden door creaks through wood modes that die in a few tens of milliseconds instead of ringing like metal.

## Three ways of rubbing

| `mode` | What happens | Used for |
|---|---|---|
| `stick_slip` | Elasto-plastic friction (bristle model) solved sample by sample with the body's lowest modes. A springy contact (`drive_hz` > 0) sticks, loads up and slips: a creak whose pulse rate rises with speed and falls with pressure. A rigid contact (`drive_hz` = 0) locks onto a body mode: a squeal or bowed tone. The slips also ring the rest of the body (FFT convolution) and a short fibre-noise residual. | door, floorboard, chair, rope creaks; hinge, shoe squeaks; glass rub |
| `scrape` | A fractal surface profile read at the contact speed (the `surface` block) pushes the whole modal body. | metal and stone scrapes, sandpaper on wood |
| `rub` | The same surface force band-limited, no modes. | cloth, padded rubbing |

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

## Checks

- Wood creaks keep their strongest sustained partial under 26 dB above the local spectrum (recorded door creaks:
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
