# Synthesis artefacts: what made v2 sound "grainy, hissy, tinny, digital, plastic" (v3, 0.3.0)

After listening to v2: the duck has a slight **grain**, the engine a slight **hiss**, the foghorn
sounds like a **sci-fi horn, very digital**, the explosion is **like an explosion in a plastic bucket**, and the
match has some **tinniness**. Those words usually point at synthesis artefacts rather than at the physics of
the sources, so we measured each one. This page lists what we found, how we measured it, and what v3 changed.
Tests for every fix are in `tests/test_artefacts.py`.

## How we looked

- Spectrograms and long-term spectra of each v2 render next to the reference recordings
  (`refs/`, see refs/SOURCES.md; reference only).
- **Inharmonic energy test** (aliasing): render a source at f0 = 123.4 Hz (chosen so harmonics do not land on
  round numbers), take the energy above 3 kHz that is *not* within 4 Hz of a harmonic, relative to all
  energy above 3 kHz. A clean periodic source gives a tiny number; aliasing folds partials onto inharmonic
  frequencies and raises it.
- The same render at 48 kHz and 192 kHz, compared band by band.
- Band envelopes over time (is the noise steady or does it follow the events?).
- Share of energy below 20 Hz, 20-200 Hz and above 200 Hz.

## Findings and fixes

| complaint | cause found | measurement (v2 -> v3) | fix |
|---|---|---|---|
| duck "grain" | `rough_pulses` placed each glottal pulse on the nearest sample. That turns every period into a whole number of samples, a hidden +-0.5 sample jitter, which is broadband inharmonic grain | inharmonic energy above 3 kHz: **-18.7 dB -> -43.4 dB** | pulses are evaluated analytically at their exact sub-sample time (`_flow_pulse`) |
| duck "grain" | far too much jitter, shimmer and breath noise. the reference mallard shows clean harmonics every ~168 Hz up to 3 kHz; v2's jitter smeared them | visible in the spectrogram; spectral distance 0.69 -> 0.28 | jitter 0.02-0.18 -> 0.004-0.034, aspiration roughly halved; low resonance that lifts the 2nd harmonic; gentle top roll-off |
| general | `valve_pulse` builds its waveform from a phase ramp with hard corners and no band-limiting: partials above 24 kHz fold back | inharmonic energy: **-33.5 dB -> -46.0 dB** | `valve_pulse` and `chopper` now run **4x oversampled** (192 kHz) and decimate through a linear-phase anti-alias filter (`excitation.oversampled`) |
| general | white noise of unit variance spreads its power over 0..sr/2, so its level per Hz depends on the sample rate (renders at 192 kHz were 6 dB quieter in noise) | - | `noise_scale(sr)` keeps noise layers at the 48 kHz level inside oversampled blocks |
| engine "hiss" | the blowdown noise gate was the pulse shape itself: 8 cylinders x 0.16 of a cycle = more than 100 % of the time, so the "gated" noise was in fact **stationary** hiss. The valve-train band noise was stationary too | high-rpm noise share (character metric): **0.37 -> 0.021** (the reference V8: 0.023) | noise gated by `shape^3` (the fast first part of the blowdown), valve noise in bursts, roar pulsed with the intake strokes; the high frequencies now come as short broadband **cracks locked to each firing** (`exhaust_crack`), which is what the reference shows |
| engine "hiss" | the `ground` block added a perfectly coherent, static copy: at 4 m it dug fixed comb notches at ~0.8, 2.3 and 3.9 kHz, which sounds phasey | deepest notch below the running median: **7.0 dB -> 2.7 dB** | the reflection now loses high frequencies (`ground_hz`, porous ground) and its path wobbles randomly (`turbulence`), which smears the notches as real air does |
| engine (minor) | valve ticks were 3.2 kHz sine bursts through a Q 2 bandpass: a tonal "tink" line | visible line in the spectrum | broadband noise clicks |
| foghorn "sci-fi / digital" | v2 added a **40 % pitch scoop** at every onset, an **octave-down** sub-tone, and a **tremolo-like AM chop**. Real horns hold their pitch (the reference barge horn: spread 0.03 semitones). A glide plus tremolo is the classic synth-horn sound | f0 108.7 Hz now matches the reference exactly; pitch std inside the blast < 0.3 semitones, but not frozen | scoop 3 %, sub-octave and chop off by default (still available); slow random pressure (+-1.5 dB) and pitch (+-0.4 %) **wobble**; turbulence noise gated by the air puffs |
| foghorn "digital" | the air puffs were 50 % open, close to a square wave, which lacks even harmonics; there were no horn formants, so the harmonic ladder was flat and static | real ref: harmonics 2, 3 strong | puffs 38 % open (all harmonics present); throat formants at `throat_hz` and 2x; gentle single-filter top roll-off instead of two cascaded lowpasses + a shelf |
| foghorn (aliasing) | the chopper was already fairly clean (-58 dB) | **-58 dB -> -67 dB** with oversampling | 4x oversampling |
| explosion "plastic bucket" | the "boom" was a **tuned 2-pole resonator ringing at 120 Hz for ~0.6 s**. A narrow, pitched resonance is what a small cavity sounds like | no narrow low peak (test: < 8 dB above the local trend) | no tonal part at all: a critically damped low thump |
| explosion "small" | the body decayed in about 1.5 s; the reference explosions stay within 10 dB for several seconds and fade over 5-7 s | time the >2 kHz band stays within 20 dB: 0.4 s -> 4.0 s (ref 4.5 s) | long fireball roar (plateau then decay, `fireball_s`), slow random surges, high band closing slowly as it cools; low end lasts longest |
| explosion "thin on speakers" | most of the energy was **below 20 Hz** (the N-wave and brown noise): 52 % in v2, 78 % in an early v3 draft. Inaudible, it only eats headroom, so after normalising the audible body is quieter | energy 20-200 Hz: **0.46 -> 0.73** (ref 0.73); below 20 Hz **0.52 -> 0.006** | infrasound highpass (22 Hz, 4th order) at the model output |
| match "tinny" | the flame "woof" was a band of noise at ~530 Hz (a mid honk); the cardboard box ringing was barely damped (`extra_damping` 40); crackle grains were resonant (Q 1.5, little pings); the flame hiss was flat white noise up to 12 kHz | 400-700 Hz flame band no longer above 150-300 Hz | woof 90-250 Hz lowpassed pink; box damping 180; crackle Q 0.7; pink hiss with a soft top and two-speed flicker; a low hand thump; infrasound highpass |
| all MP3s | 192 kbps | - | default 320 kbps |

## What was not the cause

- Quantisation: renders are float32 WAV; 16-bit would be 96 dB down.
- The sample rate itself: 48 kHz is fine once the sources are band-limited.
- `chopper` (foghorn source) aliasing was already small; the "digital" sound was the pitch scoop, sub-octave,
  tremolo, square-ish puff and static harmonic ladder.

## Still imperfect

- The inharmonic test cannot judge sources with deliberate noise (the foghorn model as a whole reads +11 dB
  because of its turbulence layers).
- `rough_pulses` still has a C1 pulse corner, so -43 dB rather than -60 dB. Oversampling it would cost 4x
  for little audible gain.
- Plain `standard_normal` noise outside the oversampled blocks still depends on the sample rate. Renders are
  48 kHz by default, so this only matters if you change `sample_rate`.

## v4 (0.4.0): after the v3 notes

| note | cause found | measurement (v3 -> v4) | fix |
|---|---|---|---|
| matchbox "crackle like bacon frying on high heat, at first" | (1) the shake: needle-sharp 0.05-0.25 ms clicks, a 1-sample "knock" impulse per stroke (a broadband click), and 7.8 strokes/s, so the top was a continuous spiky spray; (2) the ignition: crackle at up to 500 grains/s for about half a second | shake: median 50 ms kurtosis of the >2 kHz band 20.5 -> 8.0 (the reference clip 7.4); ignition: 5.6 -> 1.4 (clip 0.8) | soft sin^2 contacts, 2-4 ms knocks, grouped collisions per stroke, two damped stick resonances, box body highpassed at 500 Hz, hand-handling rumble, ~3.4 strokes/s; ignition crackle 160 grains/s peak, gone within ~0.5 s |
| foghorn "weird sustained bit at the end, like ocean waves or long distance" | the scene recipe had a constant pink-noise "sea" layer (-30 dB for 7 s) plus far reflectors (900, 1600 m) smeared by `diffusion` 0.7 into noise washes | level 3 s after the blast stopped: -36 dB -> below -45 dB within 1 s | no sea layer; one 350 m reflector with low diffusion; 1.5 s tail at 0.12 |
| engine "buzzing on top" | every firing got an identical exhaust crack and the valve ticks were metronomic, so the top was a pulse train at the firing rate (60 Hz at idle) | high-band envelope periodicity at idle 0.75 -> 0.38; the high-band level matches the reference V8 at idle (-25.5 vs -25.8 dB) | random strength per crack (a third nearly silent), pink 0.9-5 kHz noise instead of white to 7 kHz, ticks with random timing/strength/dropouts at half the level |
