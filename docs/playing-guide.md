# Playing guide

How to get convincing results from each model, what every important control does physically, and which
perceptual cues decide whether a sound reads as real. Drawn from the SoundLab knowledge base (research Page,
16 Nov 2026). Numbers marked *estimate* are ours and should be refined with recordings and the analysis tool.

## General rules (perception research)

1. **Get the decay right first.** Listeners judge material mainly from decay rate, not pitch (Klatzky, Pai &
   Krotkov 2000). In `modal`, the material's loss factor sets decay: a mode at f Hz rings for
   T60 = 6.91 / (pi f eta). Use `klatzky: "wood" | "glass" | ...` to jump to perceptual categories.
2. **Real sources are irregular.** Add jitter and variation (`jitter`, `shimmer`, `rasp`, `detune`). Duck-call
   research found rasp (spectral entropy) separated real birds from fake calls (Ducks Unlimited). Use a new
   `--seed` for every repeat of a sound, as game audio does (Wwise SoundSeed varies modes per playback).
3. **Layer, but not too much.** Murch: about five layers per moment if they sit in different frequency
   ranges, about 2.5 if they are alike. Give each layer its own band (masking in critical bands).
4. **Bind layers with a shared onset.** Layers that start and move together fuse into one object (auditory
   scene analysis). Events within a few ms fuse (precedence effect: 1-5 ms for clicks, up to about 40 ms for
   complex sounds).
5. **Space sets distance.** Direct-to-reverb ratio and air absorption tell the ear how far away a source is.
   Use `distance` (ISO 9613-2 air absorption) and `room`/`outdoor` instead of just turning things down.
6. **A little dirt helps.** Murch adds a "bad recorder" layer like "a dash of bitters": `drive` on the
   explosion is that.

## Duck (`duck`)

Physics: the syrinx is a pressure-driven valve (`valve_pulse`); the trachea and beak filter it (`formant` +
short `tube`). The classic quack is the hen's: 2-10 quacks in a decrescendo, typically 5-6. Drakes give a
quieter 1-2 note rasping "raeb" (about 0.3 s, peak near 2 kHz).

| control | what it does | realistic values |
|---|---|---|
| `voice` | hen quack or drake raeb | `female`, `male` |
| `notes` | quacks per call | 2-10 (5-6 typical) |
| `f0` | syrinx pulse rate | sources report about 180 to 585 Hz |
| `formant_hz` | main resonance | 1.5-2.5 kHz |
| `rasp` | period doubling + noise: the realism knob | 0.4-0.8 for a hen |
| `decrescendo_db` | level drop per quack | 2-4 dB |
| `note_s`, `gap_s` | timing | *estimate* 0.12-0.2 s, 0.06-0.12 s |

Tips: high `rasp` lowers the perceived pitch (subharmonic). For a duck call (the hunter's reed instrument)
use lower rasp and steadier pitch.

## Car engine (`engine`)

Physics: each cylinder's exhaust pulse at the firing frequency **f = RPM x cylinders / 120** (four-stroke);
pipe and muffler are acoustic waveguides; uneven firing intervals give the burble.

| control | what it does | realistic values |
|---|---|---|
| `rpm` | engine speed (automate it) | idle about 700-900, redline 6000-8000 for cars (*typical, not sourced*) |
| `load` | throttle: pulse strength, brightness, intake roar | 0-1 (automate with rpm) |
| `cylinders`, `firing` | engine layout | `even`, `v_twin_45`, `crossplane_bank` |
| `exhaust_length` | pipe length; resonances at odd multiples of c/4L | 1.5-3.5 m |
| `muffler` | 0 = straight pipe, 1 = quiet | 0.3-0.7 road car |
| `jitter` | combustion variation cycle to cycle | 0.02-0.06 |
| `pops` | crackle on lift-off at speed | 0-1 |

Tips: always move `load` with `rpm`; a revving engine under no load sounds synthetic. Snap `load` to 0 when
the revs peak to get lift-off pops. Put `passby` after it for drive-bys; `ground` adds the comb of the road
reflection.

## Foghorn (`foghorn`)

Physics: diaphone. Compressed air (Type F 35-40 psi) chopped by a reciprocating slotted piston into a long
conical trumpet. High tone about 250 Hz; when the motor air shuts off the piston slows and the pitch falls to
the 93-150 Hz "grunt". The F2T two-tone sustains the low note (Terry Pepper, Seeing The Light).

| control | what it does | realistic values |
|---|---|---|
| `tone_hz` | main tone | about 250 Hz (Type F) |
| `grunt_hz` | where the grunt ends | 93-150 Hz |
| `mode` | `single`, `grunt`, `two_tone` | the grunt is what makes it read as a diaphone |
| `blast_s`, `grunt_s` | timing | 2-4 s blast, about 1 s grunt (*estimate*) |
| `horn_length`, `mouth_radius` | trumpet colour | *estimate* 2-4 m, 0.2-0.5 m |

Tips: foghorns are heard far away. Follow with `distance` (hundreds of metres, `spreading: false` if you
normalise) and `outdoor` with reflectors at a few hundred metres for cliff echoes. Low tones carry further.

## Explosion (`explosion`)

Physics (Farnell's decomposition): Friedlander shock P(t) = Ps e^(-t/t*) (1 - t/t*), a blast of noise through a
resonant band swept down from about 7 kHz with clipping and sub-blasts, a low fireball (about 70 Hz), debris.

| control | what it does |
|---|---|
| `size` | shock length t*, blast length and low-end weight (not yet calibrated to charge mass) |
| `sub_blasts` | secondary bursts in the first ~0.6 s |
| `fireball_s` | rumble length (T60) |
| `debris` | falling fragments |
| `drive` | overload of the blast (a close blast clips every mic) |

Tips: distance is most of the character. Close: crack + blast + debris. Far: the shock softens to a boom,
highs vanish (`distance`), and terrain echoes roll for seconds (`outdoor` with reflectors 300-2000 m and a long
low-passed tail), like thunder rumbles.

## Match (`match`) — our design

No paper describes a match recipe. This model is built from sourced parts: stick-slip friction for the scrape
(`friction` exciting a cardboard `modal` plate), a combustion pop at ignition, a bright flare with dense
crackle, then a small steady flame (Farnell's fire layers at low level).

| control | what it does | values (*estimates*) |
|---|---|---|
| `strike_s`, `strike_speed`, `strike_force` | the gesture | 0.15-0.3 s, 0.3-1 m/s, 2-5 N |
| `ignite` | lights or fails | failed strikes add realism |
| `flare_s` | flare decay | 0.3-0.6 s |
| `flame_s`, `flame_size` | steady flame | match ~0.3, lighter ~0.5 |

Tips: the gap between scrape and flare sells it; strike twice (two layers, the first with `ignite: false`).

## Building blocks

- `impact` -> `modal` -> `radiation` = anything struck. Pick `material`, `shape` and size; frequencies follow
  geometry (bar: f1 = 4.730^2 / (2 pi L^2) x h sqrt(E / 12 rho)). Shorter `contact_ms` = harder hammer = brighter.
- `friction` -> `modal`: squeaks, creaks, scrapes, bowed glass. Static friction above kinetic gives stick-slip.
- `turbulence`: wind, whooshes, whistling wires (Aeolian tone f = 0.2 U / D).
- `bubbles` / `water`: liquids are bubbles. Minnaert f = 3.26 / r; drips are single bubbles of about 3 mm.
- `crackle`: grains for fire, gravel, crumpling, debris. Foley equivalents: cornstarch for snow, magnetic tape
  for leaves, beads for ice.
- `string` with `dispersion`: stiff wires and springs; the struck guy-wire "pew" (Burtt's blaster prop; the
  dispersion explanation is our physics reading).
- `tube`, `horn`, `helmholtz`, `formant`: air columns, cavities and vocal tracts.
