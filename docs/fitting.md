# Fitting a recipe to a recording (`soundlab match`)

Analysis-by-synthesis: render the recipe, measure how far it is from a reference recording, change the
controls, render again, and keep the best. It is roadmap step 8, first version.

```bash
soundlab match refs/duck.wav --recipe recipes/fit/duck_fit.json --start 0.3 --end 2.9 \
    -p 0.f0:120:350 -p 0.formant_hz -p 0.rasp -p 0.nasal -p 0.note_s:0.1:0.3 -p 0.gap_s:0.2:0.7 \
    -p 0.start:0.05:0.4 -p 1.gain_db:-45:-20 --budget 140 -o fits/duck
```

Writes `fits/duck/`:
- `fitted.json`: the recipe with fitted values.
- `report.md` / `report.json`: before/after distance per loss part, each control's range and before/after
  value, and a feature table (reference vs before vs after).
- `comparison.png`: long-term spectrum and onset-aligned envelope, reference vs before vs after.
- `fitted_preview.wav`: the fitted render at 24 kHz.

To build a recipe from a recording rather than fit an existing one, see [resynthesis.md](resynthesis.md)
(`soundlab resynth`, v0.17).

## Rules

1. **Pitch first** (default). The reference's fundamental is measured; every pitch control in the fit set
   (`f0`, `tone_hz`, `freq`) is set to it, checked on a render (within 5 cents) and **locked**, then the other
   controls are fitted. Spectral losses alone tend to trade pitch for brightness. If the reference has no
   confident pitch, pitch controls are fitted with the rest and the report says so. `--no-pitch-first`
   restores the old behaviour (pitch initialised from the reference, then optimised).
2. **Hold-out check.** Matching one clip is not the same as sounding right. Give a second recording of the
   same sound with `--holdout AUDIO[:START[:END]]` (repeatable): the report scores the recipe before and after
   on it and flags a fit that got worse there as likely overfitted. Without one, the report says
   **Missing** in bold and the fit should not be adopted. Every fit also writes listening renders in `ear/`: the fitted recipe, then
   each numeric macro at 0.15 and 0.85 (others at default) and every non-default choice of a choice macro
   Recipes without macros fall back to all fitted controls moved 25 % of their range down / up.
3. **Reference audio stays private.** Reference clips and anything derived from them (stems, crops,
   previews, fit output folders) live outside the repo. `tests/test_no_reference_audio.py` fails if any audio
   file is in the repo tree or tracked by git. Point `-o` outside the repo.

```bash
soundlab match refs/foghorn.wav --recipe recipes/fit/foghorn_fitted.json -p 0.tone_hz -p 0.brightness \
    --holdout refs/foghorn_other.wav --budget 120 -o ../fits/foghorn
```

## Choosing controls

- `-p L.name` fits control `name` of layer `L` (first block of a chain). `-p L.B.name` picks block `B` of a
  chain. `:lo:hi` narrows the range, which you should almost always do. `L.gain_db` fits a layer's level.
- With no `-p`, every scalar numeric control with a range is fitted, except times and counts. That is usually
  too many.
- Ranges spanning 10x or more are searched on a log scale.
- Fit **sound** controls with a **dry** recipe matched to the recording: a close mic, no room, and a noise
  layer if the recording has background noise (fit its `gain_db`). Then copy the values into the scene recipe.
  The shipped `recipes/fit/*_fit.json` do this for our five references.
- Automated controls (breakpoint lists) are not fitted. Shape them by hand to follow the recording; for example
  the rpm curve in `engine_fit.json` follows the Rover V8 revs.

## The distance

Five parts, each scaled so about 1.0 means "clearly different". `total` is their weighted mean (weights
spectral 1, stft 1, envelope 0.7, texture 0.7, sparsity 0.5, pitch 0.7; change them with `--weights`). Both signals are
resampled to 24 kHz and peak-normalised. Level differences more than 45 dB below the peak are ignored, so a
field recording's background hiss does not dominate.

| part | measures | needs alignment? |
|---|---|---|
| spectral | long-term 1/3-octave spectrum difference, dB / 10: bassy vs tinny | no |
| stft | multi-resolution (FFT 256/1024/4096) log spectrogram in 1/3-octave bands, dB / 10 (cf. DDSP multi-scale spectral loss, Engel et al. 2020) | yes (onset-aligned) |
| envelope | 10 ms level envelope, dB / 10: half onset-aligned difference, half difference of level quantiles (timing-free duty cycle) | half |
| texture | McDermott & Simoncelli (2011) style: per-band envelope level, variation, skewness and modulation power 0.5-256 Hz: rough vs smooth, steady vs choppy | no |
| sparsity | v0.19: STFT power divided by its local average (15 bins x 9 frames, 300 Hz-10 kHz); the 50 / 99 / 99.9 % log quantiles separate sparse ringing grains (bubbles, drops, crackle) from shaped noise, which always has the same light tail | no |
| pitch | octaves between candidate and reference f0, only if the reference is pitched | no |

Why a pitch term: spectral losses are poor at pitch (Turian & Henry 2020). In our first foghorn fit, the
optimiser raised the pitch to match the reference's brightness. Pitch-like controls (`f0`, `tone_hz`) are also
initialised from the reference's analysed f0.

## Optimisers

- `cmaes` (default): a compact (mu/mu_w, lambda)-CMA-ES after Hansen's tutorial, on controls scaled to [0, 1].
  Samples outside the box are clipped and penalised.
- `nelder-mead`: scipy's simplex. Faster on 2-3 controls, more easily stuck.

All renders use the recipe's seed (common random numbers), so the loss is smooth enough for a gradient-free
search. 100-150 renders take a minute or two for a few-second sound. The fit never returns a worse recipe: if
nothing beats the start, the original values are kept.

## Reading a report honestly

- A lower distance means closer **by these measures**. It is a guide for the ear, not a replacement.
- Many sounds are not identifiable: two controls can trade off (f_lo/f_hi swapped, size vs boom frequency). Fix
  one by hand.
- Fitting cannot fix a wrong model. v1's duck and explosion had the wrong physics, and no setting of their
  controls sounded right. v2 changed the models first and then fitted them.
- Fitting to one recording copies its mic and distance. Fit to several references and look at the spread.

## Comparing versions

`tools/compare_to_refs.py TARGETS.json REF_DIR OUT.md v1=DIR v2=DIR` computes the distance of each version's
renders to every reference listed in `TARGETS.json` (file, crop start, crop end) and averages them per sound.
`tools/character_metrics.py` (same arguments) prints a few targeted numbers per listening note, such as the
duck's periodicity, the explosion's rise time and how long its hiss lasts, and the foghorn's f0 and share of
energy below 250 Hz.


## Fits to reference recordings

`recipes/fit/*_fitted.json` are dry recipes already holding the fitted values for crops of reference
recordings kept outside the repo (duck: one quack; engine: idle + three blips; foghorn: one blast; explosion: one big
blast; match: strike + flame; shaker: the box shake). Fit reports stay outside the repo. Before -> after (distance total):

| sound | before | after |
|---|---|---|
| duck | 0.562 | 0.388 |
| engine | 0.566 | 0.450 (two passes; `exhaust_crack` added between them) |
| foghorn | 0.406 | 0.364 |
| explosion | 0.758 | 0.381 |
| match | 0.548 | 0.444 (the flame model changed between the two passes) |
| shaker | 0.670 | 0.603 |

Refits after the shaker and match models were rebuilt: shaker 0.761 -> 0.656 (dry, one fit pass), match 0.437 -> 0.423. The
`matchbox` scene deliberately uses fewer items (60) and less slide than the fit asked for: the distance
prefers a dense high-frequency fizz, which sounds like frying.

## Fits to CC0 recordings

References: Freesound previews licensed CC0, kept OUTSIDE the repo (research folder `refs/cc0/`, with a
SOURCES file listing URL, author and licence). Reference only; no audio or stems ship. Distance totals (lower is
better) from `fit.compare_signals` at 24 kHz; "hold-out" = a second recording or a different step not used for
fitting.

| recipe | fit clip | before | after | hold-outs: before -> after |
|---|---|---|---|---|
| `footstep` (boot, wood) | wood-floor walk, one step | 0.499 | 0.383 | other wood recording 1.220 -> 0.902; another step 0.573 -> 0.472 |
| `hit` (board) | door knock | 1.046 | 0.692 | second door knock 1.122 -> 0.906; single wood knock 1.205 -> 0.584 |
| `coin` (drop, tile) | single coin drop | 1.119 (retro `coin_pickup`) | 0.695 | coin on ground 0.994 -> 0.504; coin on table 1.005 -> 0.798 |

How each was done, honestly:
- **Pitch first.** Only the coin has a pitch, and its ring (~9 kHz) is far above the YIN range that
  `lock_pitch` uses (40-1500 Hz), so it was measured by hand: strongest 5-19 kHz peak after the first impact,
  9054 Hz (pair 9022/9216 Hz). The default coin size (28.5 x 1.85 mm cupronickel, 9.1 kHz by free-plate
  theory) was set from that before anything else was tuned. The footstep and knock have no pitch control;
  their floor/panel modes were left to the fit.
- **footstep:** CMA-ES (90 renders) on intensity, roll time, floor scale, damping, scuff and weight. The fit
  pushed the walker's weight to the lower bound (45 kg); we kept 75 kg (an adult's body weight; the `size`
  macro goes lighter) and took the rest. That costs some of the gain (fit clip 0.339 -> 0.383) but both
  hold-outs still improve. Stone was fitted the same way (leather shoes on concrete): small gain, kept as
  offsets in the `surface: stone` choice; the second concrete recording got very slightly worse (0.930 ->
  0.938). A gravel fit made both gravel hold-outs worse, so it was not kept.
- **hit (board):** CMA-ES (90 renders) on thickness, damping, contact time, contact noise, thump, length and
  noise centre. One hold-out improved a lot, the other door knock got worse than our first hand-set board
  (0.847 -> 0.906) though still far better than before. Recorded knocks are knuckle knocks, darker than a stick.
- **coin:** a 7-control CMA-ES fit pushed two controls to their bounds and made the coin-on-ground hold-out
  worse (0.521 -> 0.641), so it was rejected. Kept instead: one control (floor level 1 -> 2) plus the wobble
  rework (rolling micro-contacts), which improves the fit clip and both hold-outs.

### Heel-to-toe gap: listening over the metric

With the fitted gap of 0.2 s, heel and toe sound like two separate hits. The gap
is 35 ms by default (~20-60 ms with shoe, weight and the `pace` macro), so heel and toe flam into one step.
The CC0 recordings preferred the long gap: with 35 ms the wood distances go back to roughly pre-fit levels (fit
clip 0.383 -> 0.496; hold-outs 0.902 -> 1.228 and 0.472 -> 0.579), stone 0.566 -> 0.577 (hold-outs 0.666 ->
0.750, 0.938 -> 0.745), gravel 0.751 -> 0.794 (hold-outs 0.549 -> 0.610, 0.562 -> 0.709). A refit of the other
footstep controls with the gap locked gained little (0.496 -> 0.456) and pushed three controls to their
bounds, so it was not kept. Here listening overrides the metric: those recordings seem to be slow walks with a
distinct toe-down, and a single-number spectral distance rewards copying that.

## Game library checks against CC0 recordings

Refs: Freesound CC0 previews picked automatically from searches (2 per sound, rain roof/window found none),
kept in `soundlab-research/refs/cc0` with `SOURCES.md`; never in the repo. They are loose matches: a
"paper" search returned a 48 s handling recording dominated by rumble, a "bullet metal" one is 0.1 s. So most
sounds were checked on band balance (energy in 20-100/100-300/300-1k/1-3k/3-8k/8-16k Hz) and decay, and only
three were fitted with `soundlab match` plus a hold-out:

| recipe | fitted to | controls | fit before -> after | hold-out (second recording) | adopted |
|---|---|---|---|---|---|
| rain | rain_heavy_1 (10-16 s) | near, hiss, brightness, drop_mm | 0.566 -> 0.402 | rain_heavy_2: 0.637 -> 0.529 | yes (hiss 0.2, near 0.55, brightness 0.43, drop 1.3 mm) |
| door_slam | door_slam_1 | force, rattle, thump, latch | 0.264 -> 0.255 | door_slam_2: 1.25 -> 1.248 (that take has a long room tail) | yes (force 0.85, rattle 0.25, latch 0.45) |
| glass_break | glass_smash_1 | shards, energy, height | 0.538 -> 0.335 | glass_smash_2: 0.475 -> 0.433 | partly: shards 100; height hit its lower bound (0.1 m), set to 0.4 m (hold-out 0.463, fit 0.475) |

Changes driven by the band checks (before -> after, distance in dB RMS over the six bands to the closest ref):
- **Rain** lacked everything under 300 Hz (-38 dB vs the refs' -13): added the low wash; 10.9 -> 3.4 dB.
- **Door slam** was a boom (100-300 Hz on top, contact 3.9 ms): the leaf's lowest modes are pinned at impact
  and the edge contact is 1-2.6 ms; 10.4 -> 2.6 dB.
- **Glass**: the first version's shards rang above Nyquist (wrong plate formula); fixed, now 4.4 dB.
- **Sword clash**: the refs put most energy at 3-8 kHz with 0.1-0.3 s decays; ours was 300 Hz-3 kHz and 0.8 s.
  Grip damping and a high-pass moved it up (decay 0.37 s). Still ~15 dB from the refs on bands, partly on
  purpose (no tinny tops).
- **Creature growl**: pitch first. YIN finds almost no voicing in rough growls (1-21 % of frames), so the pulse
  rate was taken from the envelope periodicity: 32-63 Hz across four refs -> default f0 45 Hz, size 1.5.
  The refs disagree on balance (two low-heavy, one mid, one bright), so the band distance stays 6-22 dB.
- **Door creak**: refs range from 26 Hz ticking to a 1.1 kHz squeal; ours (260 Hz) sits between; 5.5 dB to one.

Not fitted: wind (3.6 dB to the trees ref as is), swing, clash, bullets, gunshot, punch, rustles, splash.

### Listening over metrics

Rain (fitted hiss 0.2 overruled: louder diffuse wash, softer near drops) and the door slam (the CC0 match was
a light door) were then set by listening. Band distances: rain 5.2/3.5 dB to the two heavy-rain refs, door
slam 13.6/15.8 dB (heavier than both refs, on purpose), growl 4-5 dB to the big-creature ref.

The gunshot and growl are not fitted. Five CC0 firearm recordings (gunshot_1/2, gun_shotgun_2/3) gave the
gunshot's attack, first-5 ms brightness and tail ratios; four growls and one tiger growl gave the trill rate
(18-22 Hz). Listening decides over the metrics.

## Seven sounds against their references (v0.14)

Cloth rustle, creature growl, glass break, splash, sword clash, wind and wind through a gap were compared with
their CC0 references, with both sides prepared exactly like blind-pack clips (onset trim, 4 s cap, tail
trim, loudness match): spectrograms, envelopes, octave bands, attack and decay times, onset density,
envelope modulation and how much the brightness moves. Each change fixes a cause in the model; one reference
per sound was held out and not looked at while tuning. "Fit" is `fit.compare_signals` (lower is better,
mean over three variations); "bands" is the mean octave-band difference 125 Hz-16 kHz in dB.

| sound | fit (tuning refs) | bands dB | hold-out: fit | hold-out: bands dB |
|---|---|---|---|---|
| cloth_rustle | 1.279 -> 0.868 | 5.3 -> 2.7 | cloth_2: 1.382 -> 0.901 | 5.6 -> 1.9 |
| creature_growl | 0.809 -> 0.678 | 15.3 -> 13.0 | growl_2: 0.920 -> 0.720 | 11.2 -> 16.0 |
| glass_break | 0.676 -> 0.489 | 4.2 -> 4.0 | glass_smash_2: 0.542 -> 0.535 | 4.0 -> 3.7 |
| splash | 0.716 -> 0.631 | 6.5 -> 5.3 | splash_l1: 0.968 -> 0.845 | 12.8 -> 10.5 |
| sword_clash | 1.069 -> 0.618 | 11.2 -> 6.0 | clash_2: 0.689 -> 0.338 | 10.5 -> 2.2 |
| wind | 0.749 -> 0.784 | 9.4 -> 11.9 | wind_trees_1: 0.566 -> 0.423 | 8.6 -> 3.1 |
| wind_gap | 0.650 -> 0.557 | 9.3 -> 8.1 | wind_whistle_2: 0.543 -> 0.401 | 7.0 -> 3.4 |

- **Sword clash**: energy sat at 300 Hz-3 kHz (refs: 4-12 kHz) and the ring held for 0.3 s to -20 dB (refs
  0.12 s). Cause: every bending mode of the blade radiated equally. A 4.5 mm blade radiates poorly below its
  coincidence frequency (~2.8 kHz), the grip and the other blade choke the ring, and the edge-on-edge contact
  is itself a bright burst. Added the coincidence high-pass, more grip damping, the contact burst and a
  faint body knock, and a lighter scrape. Centroid 2.7 -> 7.1 kHz. Pipe, plate and pan are unchanged.
- **Glass break**: the shards all fell from one height, so after the crack there was a silent gap, then the
  loudest moment came when they landed (attack 75 ms vs 20 ms), and late tinkles stood alone after more
  silence; the top stopped at 13 kHz. Now the fracture is a cascade of crack clicks (30-120 ms), pieces fall
  from all over the pane so impacts follow the break at once, tinkles thin out without a gap, big pieces
  land choked and the band reaches 18 kHz. Attack 28 ms.
- **Splash**: over too fast (-40 dB after 0.5 s vs 1.4 s), the late tail fell apart into lone ticks with
  silence between, and the impact had no low end (63-125 Hz ~15 dB under the refs). Added the low plunk
  (water mass and opening cavity), a fall-back timed from the throw-up speed (2 v / g), and a continuous wash
  of churn and small bubbles that swells as the water lands. -40 dB now after ~0.9 s, peak at the impact.
- **Cloth rustle**: a 1.6 kHz band-pass (centroid 1.9 kHz vs 3.5 kHz, 8-16 kHz 10-12 dB short) in short,
  separate pushes that dropped to silence. Now a broadband friction hiss with fibre micro-crackle, strokes
  that swell and fade over 0.25-0.6 s and overlap, a floor while the hand keeps moving, and a closer, drier
  room (the small room's reflections drew comb lines through the hiss).
- **Wind**: a dark rumble (corner ~600 Hz; the refs are flat to 1 kHz and fall slowly) with little movement in
  level and brightness. The corner now sits at ~3.5 kHz and moves with the gust speed (v^1.5), gusts carry
  faster flurries, eddies make the level waver irregularly, and the howl resonances are broad (narrow ones
  beat into grain). Better against the trees and gusty-midrange refs; further from a recording that is
  mostly wind buffeting a microphone.
- **Wind gap**: the whistle's pitch followed every gust (refs: a thin, steady line) and the wind behind it was
  bright. The edge tone now locks to the gap's resonance (pitch steady within ~2 %, level flickering with
  the flow), and a new `indoors` control low-passes the wind noise as heard through a wall or window.
- **Creature growl**: steady, narrow tract resonances, hard glottal closures and 6-9 % cycle jitter gave a
  buzzing, scraping texture with energy up to 2-4 kHz (refs fall off from ~500 Hz) and a symmetric swell.
  Now softer closures (longer open and return phases), moderate jitter (harmonics show again), formants
  that move with the jaw, a stronger chest band, a growl that builds and sags like one breath, a lower
  pitch (55 Hz) and a longer default (2.6 s). The bright `growl_2` reference is further away on bands: the
  recipe sits at the deep end (see the roadmap).

## Texture statistics against hold-out recordings (v0.15)

Seven sounds were compared with real recordings using texture statistics in the style of McDermott &
Simoncelli (2011): 20 ERB-spaced bands from 80 Hz to 14 kHz, compressed Hilbert envelopes, and per band the
level, coefficient of variation, kurtosis, modulation power in octave bands from 0.5 to 128 Hz, and the
correlation of each band's envelope with its neighbour and with the band four away. Hold-out recordings (later
windows of longer files and further public-domain / CC recordings, none used for tuning before) were weighted
over the references already used. "Texture" is the mean |z| of the engine's statistics against the
recordings (lower is better); "fit" is `compare_signals`. Both averaged over three variations.

| sound | texture, hold-out | fit, hold-out | texture, earlier refs | fit, earlier refs |
|---|---|---|---|---|
| cloth_rustle | 1.31 -> 0.65 | 0.858 -> 0.554 | 1.28 -> 0.61 | 0.884 -> 0.534 |
| creature_growl | 0.74 -> 0.55 | 0.776 -> 0.757 | 0.67 -> 0.48 | 0.692 -> 0.678 |
| glass_break | 1.09 -> 0.73 | 0.939 -> 0.870 | 0.66 -> 0.55 | 0.504 -> 0.504 |
| splash | 1.07 -> 0.82 | 0.683 -> 0.730 | 1.07 -> 0.84 | 0.702 -> 0.763 |
| wind_gap | 1.28 -> 0.68 | 0.786 -> 0.733 | 1.00 -> 0.64 | 0.479 -> 0.437 |
| door_creak | 0.85 -> 0.58 | 0.822 -> 0.766 | 0.93 -> 0.67 | 0.786 -> 0.711 |
| rain | 0.76 -> 0.61 | 0.682 -> 0.608 | 0.42 -> 0.63 | 0.391 -> 0.374 |

A new helper, `band_modulate`, splits a noise texture into half-octave bands and gives each band its own slow
log-normal level fluctuation, partly shared (`corr`). The other 30 recipes render bit-identically.

- **Cloth rustle**: one noise source under one hand-speed envelope made every band rise and fall together
  (adjacent-band envelope correlation 0.84 vs 0.69-0.76), the dips between strokes were deeper than in
  recordings, and the room's early reflections drew comb lines. Now the friction hiss goes through
  `band_modulate` (bands come and go on their own, brightness follows the hand speed), a higher floor between
  strokes, heavier-tailed fibre snags, sparse soft knocks of seams and buttons, a 4 s gesture, and the
  `distance` block instead of the room (no comb lines).
- **Creature growl**: no energy above 7 kHz and what was there only drifted slowly (recordings: breath noise
  chopped at 16-128 Hz), voiced energy at 2-7 kHz too strong against the hold-out growls, and the pitch held
  still. Now breath noise above 3 kHz is chopped by each glottal pulse and by the trill, the uvula is
  audible as a rattle each time the flap lets go, the trill chokes the voice more deeply (`trill` 0.8), the
  voiced mids are darker, the pitch bends slowly by a few semitones, and the room is closer and drier.
- **Glass break**: one crack followed by a smooth, dense, high tinkle wash (mid-band envelope kurtosis
  negative; recordings: a sawtooth of separate hits). Now a handful of distinct crashes (broadband smash,
  short glassy ring, dull knock), spaced irregularly and getting weaker, a shorter fracture cascade, fewer
  shards (30), shorter late tinkles and a drier room.
- **Splash**: the low end was a lone tonal plunk (low-band envelope kurtosis 5 vs ~1, 63-125 Hz too strong)
  and the level decayed smoothly in dB from the first hit. Now the low end is moving water (noisy churn), with
  one to three sub-splashes (the crown collapsing, the jet falling back), sloshing swells in the wash, a fizz
  of bursting foam above 6 kHz, late bubbles that get smaller, and no stray late drops.
- **Wind gap**: the gust envelope was too deep: one common envelope on all bands made them far too correlated
  (band four apart 0.54 vs 0.29) and the slow modulation dominated where recordings are mostly fast, steady
  turbulence. Gustiness 0.45 -> 0.1, slower gusts, and a louder whistle (it sounds above a threshold speed).
  The `wind` model is unchanged.
- **Door creak**: a click per slip with a smooth speed arc: too regular and spiky at 2-4 kHz (envelope
  kurtosis 1.7 vs -0.3). Now stick-slip is a continuous relaxation oscillation (force builds, drops in
  10-25 % of the cycle), with slip-rate regime jumps (0.6-1.6x, quick glides), cycle jitter and shimmer, grip
  chatter at ~9 Hz, two or three pushes of different strength, a friction-noise floor, a lighter low end, and
  a slower opening (2.6 s).
- **Rain**: no air above 7 kHz (12 kHz low-pass, dull hiss), five times too little slow modulation, and a
  uniform fine texture. A new `texture` control (default 0, so other uses are unchanged) adds slow band-by-band
  swells, air up to 14-16 kHz, a few drops right by the microphone and a tiny spray crackle on each near drop.

## Rain, splash and paper against hold-out recordings (v0.16)

In the third blind pack all engine clips of rain, splash and paper rustle were identified (per-sound balanced
accuracy at or near 100 %). Each was compared again with the v0.15 texture statistics, now with more
public-domain / CC hold-out recordings (four more rain recordings, three more paper recordings), weighted over
the references used before. Same measures as above, three variations each (lower is better).

| sound | texture, hold-out | fit, hold-out | texture, earlier refs | fit, earlier refs |
|---|---|---|---|---|
| paper_rustle | 1.91 -> 0.53 | 1.212 -> 0.816 | 1.45 -> 0.69 | 1.161 -> 1.032 |
| rain | 0.64 -> 0.49 | 0.542 -> 0.480 | 0.66 -> 0.50 | 0.374 -> 0.386 |
| splash | 0.82 -> 0.66 | 0.730 -> 0.724 | 0.84 -> 0.68 | 0.763 -> 0.765 |

The other 34 recipes render bit-identically. Glass break is unchanged (texture 0.73, fit 0.870 on hold-outs).

- **Paper rustle**: a few short, smooth hand gestures with silence between them over a band-passed bed: slow
  modulation dominated (0.88 vs 0.39 in the lowest bands), fast modulation was missing, the bands moved
  together (band four apart 0.92 vs 0.66) and the high-band envelopes were smooth where recordings are spiky
  (kurtosis +2 to +3). Paper now has its own branch (plastic and leaves are unchanged): crumpling is crackling
  noise, so crease snaps have power-law sizes and come in avalanches (the event rate bursts on a 50-300 ms
  scale); each snap is a broadband click 0.15-2 ms long from one of three families of different brightness;
  the biggest also flex the sheet (a soft 250-450 Hz crunch); the sliding hiss goes through `band_modulate`;
  and the hands keep working the sheet over a floor, never to silence. The recipe is a 4 s crumple (3.7 s of
  handling) in a drier, closer room.
- **Rain**: the top bands only swelled slowly (slow modulation 0.38 vs 0.15) and lacked the 4-30 Hz
  modulation and spikiness (kurtosis 2-5 above 2 kHz) that separately heard drops give. A new `patter`
  control (default 0) adds the drops a metre or a few away: hundreds per second, clustered, each a short
  bright tap (0.6-3 ms) below 7 kHz, with gentler band swells and a lighter low wash. The recipe uses
  `patter` 1.
- **Splash**: the churn, bubble cloud and foam all followed one envelope, so the bands rose and fell together
  (band four apart 0.83 vs 0.65) with too little fast flutter. A new `slosh` control (default 0) gives each
  its own irregular 9 and 25 Hz envelope and a slightly longer tail; the recipe uses 0.5. The onset-aligned
  fit is about unchanged; the plateau of continuous splashing is still not modelled.

