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

## Rules (0.6.0)

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
   (0.7.0). Recipes without macros fall back to all fitted controls moved 25 % of their range down / up.
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
spectral 1, stft 1, envelope 0.7, texture 0.7, pitch 0.7; change them with `--weights`). Both signals are
resampled to 24 kHz and peak-normalised. Level differences more than 45 dB below the peak are ignored, so a
field recording's background hiss does not dominate.

| part | measures | needs alignment? |
|---|---|---|
| spectral | long-term 1/3-octave spectrum difference, dB / 10: bassy vs tinny | no |
| stft | multi-resolution (FFT 256/1024/4096) log spectrogram in 1/3-octave bands, dB / 10 (cf. DDSP multi-scale spectral loss, Engel et al. 2020) | yes (onset-aligned) |
| envelope | 10 ms level envelope, dB / 10: half onset-aligned difference, half difference of level quantiles (timing-free duty cycle) | half |
| texture | McDermott & Simoncelli (2011) style: per-band envelope level, variation, skewness and modulation power 0.5-256 Hz: rough vs smooth, steady vs choppy | no |
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


## Fits to the reference recordings (0.3.0)

`recipes/fit/*_fitted.json` are dry recipes already holding the fitted values for the crops in
`refs/targets.json` (duck: one quack; engine: idle + three blips; foghorn: one blast; explosion: one big
blast; match: strike + flame; shaker: the box shake). Fit reports are in the research folder
(`fits/<sound>/`). Before -> after (distance total):

| sound | before | after |
|---|---|---|
| duck | 0.562 | 0.388 |
| engine | 0.566 | 0.450 (two passes; `exhaust_crack` added between them) |
| foghorn | 0.406 | 0.364 |
| explosion | 0.758 | 0.381 |
| match | 0.548 | 0.444 (the flame model changed between the two passes) |
| shaker | 0.670 | 0.603 |

v4 refits (new shaker and match models): shaker 0.761 -> 0.656 (dry, one fit pass), match 0.437 -> 0.423. The
`matchbox` scene deliberately uses fewer items (60) and less slide than the fit asked for: the distance
prefers a dense high-frequency fizz, which is the "frying" sound.

## Fits to CC0 recordings (0.8.0)

References: Freesound previews licensed CC0, kept OUTSIDE the repo (research folder `refs/cc0/`, with a
SOURCES file listing URL, author and licence). Reference only; no audio or stems ship. Distance totals (lower is
better) from `fit.compare_signals` at 24 kHz; "hold-out" = a second recording or a different step not used for
fitting.

| recipe | fit clip | v0.7 | v0.8 | hold-outs: v0.7 -> v0.8 |
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
  pushed the walker's weight to the lower bound (45 kg); we kept 75 kg (an adult body weight; the `size`
  macro goes lighter) and took the rest. That costs some of the gain (fit clip 0.339 -> 0.383) but both
  hold-outs still improve. Stone was fitted the same way (leather shoes on concrete): small gain, kept as
  offsets in the `surface: stone` choice; the second concrete recording got very slightly worse (0.930 ->
  0.938). A gravel fit made both gravel hold-outs worse, so it was not kept.
- **hit (board):** CMA-ES (90 renders) on thickness, damping, contact time, contact noise, thump, length and
  noise centre. One hold-out improved a lot, the other door knock got worse than our first hand-set board
  (0.847 -> 0.906) though still far better than v0.7. Recorded knocks are knuckle knocks, darker than a stick.
- **coin:** a 7-control CMA-ES fit pushed two controls to their bounds and made the coin-on-ground hold-out
  worse (0.521 -> 0.641), so it was rejected. Kept instead: one control (floor level 1 -> 2) plus the wobble
  rework (rolling micro-contacts), which improves the fit clip and both hold-outs.

### 0.8.1: heel-to-toe gap set by ear

The 0.8.0 footstep's heel and toe sounded like two separate hits (gap 0.2 s, which came from the fit). The gap
is now 35 ms by default (~20-60 ms with shoe, weight and the `pace` macro), so heel and toe flam into one step.
The CC0 recordings preferred the long gap: with 35 ms the wood distances go back to roughly v0.7 levels (fit
clip 0.383 -> 0.496; hold-outs 0.902 -> 1.228 and 0.472 -> 0.579), stone 0.566 -> 0.577 (hold-outs 0.666 ->
0.750, 0.938 -> 0.745), gravel 0.751 -> 0.794 (hold-outs 0.549 -> 0.610, 0.562 -> 0.709). A refit of the other
footstep controls with the gap locked gained little (0.496 -> 0.456) and pushed three controls to their
bounds, so it was not kept. Here the ear overrides the metric: those recordings seem to be slow walks with a
distinct toe-down, and a single-number spectral distance rewards copying that.
