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
