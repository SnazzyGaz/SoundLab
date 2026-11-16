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

Physics: the syrinx is a vibrating valve whose labia slap shut irregularly. v2 models it as a **rough pulse
train** (`rough_pulses`): about 150-300 pulses a second with strong cycle-to-cycle jitter, period doubling and
shimmer. That train goes through a nasal vocal tract: broad resonances around 1.2-3.3 kHz (strongest near
2 kHz) and an anti-resonance notch. The classic quack is the hen's: 2-10 quacks in a decrescendo, typically
5-6. Drakes give a quieter 1-2 note rasping "raeb".

What our reference recordings show (refs/SOURCES.md): each quack is 0.13-0.2 s long, swells and then **stops
abruptly**, has no clear pitch (YIN periodicity 0.36-0.50, where 1 is a pure tone), and puts its energy in
1-3.3 kHz with a fairly sharp top edge.

Why v1 sounded like a laughing gnome: it used a smooth sung source at 420 Hz with a rise-then-fall pitch arc in
every note (periodicity 0.61). A pitched note with an arc is a laugh. A rough, unpitched buzz through a nasal
filter is a quack.

| control | what it does | realistic values |
|---|---|---|
| `voice` | hen quack or drake raeb | `female`, `male` |
| `notes` | quacks per call | 2-10 (5-6 typical) |
| `f0` | mean pulse rate (not a sung pitch) | 200-300 Hz (fit to the refs: about 280) |
| `formant_hz` | main nasal resonance | 1.5-2.2 kHz (fit: about 1650) |
| `rasp` | jitter + period doubling + shimmer + breath: the realism knob | 0.6-0.9; below 0.3 sounds "cartoon" |
| `nasal` | depth of the anti-resonance and upper cluster | 0.3-0.7 |
| `decrescendo_db` | level drop per quack | 2-4 dB |
| `note_s`, `gap_s` | timing | 0.13-0.2 s quacks; gaps of 0.1 s (hurried) to 0.4 s (calm) |

Tips: keep `pitch_fall` small; any pitch movement inside a quack makes it read as a voice. For a hunter's duck
call (a reed instrument), lower `rasp` to 0.3-0.4.

## Car engine (`engine`)

Physics: each cylinder's exhaust blowdown is a pressure pulse at the firing frequency **f = RPM x cylinders /
120** (four-stroke). The pipe and muffler are acoustic waveguides, and uneven firing intervals give the burble.

v2 changes, after "fake/tinny when it revs high":
- The pulses are no longer differentiated. Differentiating tilts the spectrum up by 6 dB/octave, and the tilt
  follows the rpm, so high revs went thin.
- Blowdown noise is gated by the pulses: the rasp a real exhaust has.
- The pipe feedback is lower (`pipe_ring`), so there is less whistle and ring.
- A muffler chamber resonance and a low shelf (`body`) add weight.
- Intake roar grows with airflow (load x rpm), and block/valve noise rises with rpm.
- Loudness rises with rpm and load. Rev peaks sit about 10-15 dB over idle, as in the refs.
- New firing mode `crossplane_v8` has two banks with their own pipes. The listener is nearer one of them, which
  produces the V8 burble.

| control | what it does | realistic values |
|---|---|---|
| `rpm` | engine speed (automate it) | idle about 700-900, redline 6000-8000 for cars (*typical, not sourced*) |
| `load` | throttle: pulse strength, noise, intake roar | 0-1 (automate with rpm) |
| `cylinders`, `firing` | engine layout | `even`, `v_twin_45`, `crossplane_bank`, `crossplane_v8` |
| `exhaust_length` | pipe length; resonances at odd multiples of c/4L | 1.5-3.5 m |
| `muffler` | 0 = straight pipe, 1 = quiet | 0.3-0.7 road car |
| `body` | low-end weight (chamber + shelf) | 0.5-0.8 |
| `exhaust_noise` | blowdown rasp vs tone | 0.5-0.9 (fit pushed it up) |
| `pipe_ring` | pipe resonance feedback | 0.1-0.3; above 0.5 whistles |
| `intake`, `mechanical` | intake roar, block/valve noise | 0.3-0.8; close-up engine-bay mics hear more `mechanical` |
| `jitter` | combustion variation cycle to cycle | 0.03-0.08 |
| `pops` | crackle on lift-off at speed | 0-1 |

Tips: always move `load` with `rpm`, because a revving engine under no load sounds synthetic. Snap `load` to 0
when the revs peak to get lift-off pops. Put `passby` after it for drive-bys; `ground` adds the comb of the road
reflection, a static notch pattern that can sound phasey up close.

## Foghorn (`foghorn`)

Physics: a diaphone. Compressed air is chopped by a reciprocating slotted piston into a long conical trumpet.
When the air shuts off the piston slows and the pitch falls (the "grunt"). Our reference recordings have
fundamentals of **87-150 Hz**, energy concentrated in the 125-250 Hz octaves, and very buzzy harmonics. The
often-quoted "about 250 Hz" (Type F) is too high for what people expect a foghorn to sound like.

v2 changes, after "too high and alarm-like; real ones are bassy and choppier":
- Lower default tone (120 Hz, grunt 75 Hz).
- A wide bell, so the horn's low cutoff c/(2 pi a) is about 90 Hz instead of 180 Hz, plus a low shelf.
- A bark at the start: pressure overshoot plus a pitch scoop from 60 %.
- `roughness`: flutter plus a period-doubled octave below, for growl.
- `chop`: the slowing piston audibly pulses during the grunt.
- `blasts`/`interval_s` for signal patterns.

| control | what it does | realistic values |
|---|---|---|
| `tone_hz` | main tone | 85-150 Hz (refs); historic Type F up to ~250 |
| `grunt_hz` | where the grunt ends | 70-110 Hz |
| `mode` | `single`, `grunt`, `two_tone` | the grunt is what makes it read as a diaphone |
| `roughness` | growl (flutter + subharmonic) | 0.2-0.5; 0 = clean alarm tone |
| `chop` | pulsing in the grunt | 0.5-0.9 |
| `brightness` | how much upper buzz gets out | 0.6-0.9 |
| `blasts`, `interval_s` | signal pattern | e.g. 3 blasts every 3 s (`recipes/foghorn_pattern.json`) |
| `blast_s`, `grunt_s` | timing | 1.5-3 s blast, about 1 s grunt |
| `horn_length`, `mouth_radius` | trumpet colour, low cutoff | 3-5 m, 0.5-0.8 m |

Tips: foghorns are heard far away. Follow with `distance` (hundreds of metres, `spreading: false` if you
normalise) and `outdoor` with reflectors at a few hundred metres for cliff echoes. Low tones carry further.

## Explosion (`explosion`)

Physics, v2: a detonation is a **pressure pulse**, not a flow of gas.
- A Friedlander N-wave P(t) = Ps e^(-t/t*) (1 - t/t*): instant rise, positive phase, then negative phase. The
  positive-phase length grows with charge size (cube-root scaling, Kinney & Graham).
- A very short broadband crack that overloads the microphone (`drive`).
- A heavy low **boom**: the pulse ringing a low, damped resonance.
- A broadband rolling roar whose band closes from about 4 kHz down to 150 Hz as it dies. It uses a
  non-resonant filter, so there is no whistling sweep.
- Very low rumble, optional secondary blasts and debris.

Why v1 sounded like "gas escaping through a vent": its main layer was noise through a *resonant* band swept
down from 7 kHz over ~0.7 s, so it took about 0.2 s to reach its peak and had a pitched hiss. v2 peaks in about
4 ms and its hiss above 2 kHz lasts about 0.4 s, in line with the refs (0.4-0.6 s).

| control | what it does |
|---|---|
| `size` | N-wave length, boom pitch/length, rumble weight (not calibrated to charge mass) |
| `boom_hz` | low boom frequency (0 = from size); refs suggest 80-160 Hz for the felt "thump" |
| `crack` | level of the front-edge crack |
| `drive` | mic overload on the shock |
| `fireball_s` | rumble length (T60) |
| `sub_blasts` | secondary detonations in the first 0.5 s |
| `debris` | falling fragments |

Tips: distance is most of the character. Close up you hear crack + boom + roar. Far away, `distance` takes out
the highs and terrain echoes roll for seconds (`outdoor` with reflectors 300-2000 m).

## Match (`match`): our design, retimed to recordings

No paper describes a match recipe. This model is built from sourced parts: stick-slip friction for the scrape
(`friction` exciting a cardboard `modal` plate), an ignition burst, and a flame.

The refs show:
- A short scrape (0.03-0.1 s).
- An ignition burst that is the loudest, brightest part (broadband up to ~14 kHz).
- A steady flame that is mostly bright **hiss** 24-31 dB below the burst, with sparse crackle.

v1 had a long scrape and a low "woof" flame.

| control | what it does | values |
|---|---|---|
| `strike_s`, `strike_speed`, `strike_force` | the gesture | 0.04-0.1 s (refs), 0.5-1 m/s, 2-5 N |
| `ignite` | lights or fails | failed strikes add realism |
| `flare_s` | ignition burst decay | 0.15-0.3 s |
| `flame_db` | flame level re burst | -24 to -31 dB (refs) |
| `brightness` | spectral tilt | 0.1-0.6 (close mic = brighter) |
| `flame_s`, `flame_size` | steady flame | match ~0.3-0.5, lighter ~0.5 |

Tips: strike twice (two layers, the first with `ignite: false`).

## Fitting a sound to a recording

`soundlab match ref.wav --recipe base.json -p 0.f0:120:350 -p 0.rasp ...` renders the recipe many times, nudging
the listed controls to bring it closer to the recording. It reports the before/after distance. See
[fitting.md](fitting.md). Fit with a *dry* recipe that matches the recording's conditions (close mic, no room).
Then copy the fitted sound controls into your scene recipe, which has its own distance and room.

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
