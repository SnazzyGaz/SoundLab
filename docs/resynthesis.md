# Analysis and resynthesis (`soundlab resynth`)

Turn a recording into an editable, layered recipe: analyse it, build a recipe from generic blocks, fit the
free controls, then check the result on material the fit did not see. Roadmap: the analysis-and-resynthesis
phase, first version (v0.17).

```bash
soundlab resynth ../refs/splash.wav --kind impact --fit-end 1.2 --holdout-section 1.2:2.0 -o ../fits/splash
soundlab resynth ../refs/stream.wav --kind texture --clean-tonal --fit-end 6 --holdout-section 6:12 -o ../fits/stream
soundlab resynth ../refs/meow1.wav --kind voice --holdout ../refs/meow2.wav -o ../fits/meow
```

Writes `recipe.json` (the editable recipe), `resynth.wav` and `report.json` (features, distances, fitted
controls). Reference audio and these outputs stay outside the repo (see the rules in [fitting.md](fitting.md)).

## 1. Analysis (`resynth.analyse_clip`)

Level envelope, spectral centroid over time, 1/3-octave band levels over time, onsets, f0 (YIN, with voicing
gates), and the tonal / transient / noise share from the harmonic-percussive-residual split. For dense or
reverberant voices, where YIN finds no single clear pitch, `harmonic_sum_f0` tracks pitch by subharmonic
summation (Hermes 1988): a wide search for the overall pitch, then a narrow one around it, which stops octave
jumps.

`--clean-tonal` (`resynth.suppress_tonal`) removes narrow tonal peaks such as birdsong or whistles from a noisy
texture before analysis: STFT bins in 1.5-10 kHz more than 12 dB above the local median spectrum are set to
that median, keeping their phase.

## 2. Recipe kinds (`resynth.build_recipe`)

| kind | layers | macros |
|---|---|---|
| `impact` | `bandnoise` body whose band x time grid follows the recording (breakpoints kept only where linear interpolation would be off by more than 2.5 dB), plus a `splash` model at each main onset (up to two) | level, brightness, length |
| `texture` | stationary `bandnoise` (mean spectrum of the fit section plus a random per-band wobble) and a `bubbles` layer. It does not copy the timeline, so it loops and varies | level, brightness, churn |
| `voice` | `harmonic` on the measured f0 curve (breakpoints within 20 cents), with formants read from the harmonic peaks and the loudness envelope as automation, plus a `bandnoise` breath/noise layer from the residual | level, pitch, size |
| `choir` | `harmonic` with 4 detuned unison voices on the harmonic-sum pitch track, a breath layer and a `reverb` post block | level, pitch, size |

`--kind auto` guesses: pitched and voiced is `voice`; tonal with no clear pitch and no onsets is `choir`; long
and steady is `texture`; anything else is `impact`.

### New blocks

- **`bandnoise`** (synth): noise shaped by `bands` (Hz) x `frames` `[[t, [dB per band]], ...]`, interpolated
  in time and in log frequency, with `tilt_db`, `stretch`, `grain`/`grain_hz` (random level wobble per band) and
  `level`. A channel-vocoder-style noise source (Dudley 1939; the stochastic part of Serra & Smith 1990).
- **`harmonic`** (excitation): a formant/vocal source. Harmonics of an f0 curve (log interpolation) pass through
  a fixed formant envelope `[[Hz, bandwidth, dB], ...]`, so the vowel stays put while the pitch moves. It adds
  source slope `tilt_db`, `breath` noise through the same formants, vibrato, `wander_cents`, unison `voices`
  with `detune_cents`, a fine `eq` curve `[[Hz, dB], ...]`, and `level` automation.

## 3. Fit and checks (`resynth.fit_and_check`)

1. **Pitch first.** Pitch comes from the analysis (the f0 breakpoints) and is not left to the optimiser. After
   fitting, the median cents error between the reference's and the render's pitch tracks is reported.
2. The free controls of each kind (tilt, wobble, splash size/speed/bloop/droplets, breath, formant shift,
   vibrato, detune, reverb, layer levels) are fitted with `fit.fit` (CMA-ES) on the **fit section**.
3. **Spectral fine-tune** (`eq_correct`): the remaining long-term 1/3-octave difference is folded into the
   harmonic layer's `eq`, or into every frame of the `bandnoise` grid. It is kept only if the distance improves.
4. **Checks.** The report gives distances on the whole clip, on the whole clip with another random seed (the
   recipe should not depend on one lucky noise draw) and on each **hold-out**:
   - `--holdout-section A:B`: for impacts and other fixed timelines, the full render is cropped at the same
     times. For textures, the recipe is rendered fresh, because it does not copy the timeline.
   - `--holdout AUDIO`: another recording of the same sound.

Distances are `fit.compare_signals` (the same measure as `soundlab compare`; about 1.0 means clearly
different).

## First results (private reference recordings, kept outside the repo)

| clip | kind | stock v0.16 recipe | fit section: before -> after | whole clip | other seed | hold-out |
|---|---|---|---|---|---|---|
| medium splash (2.0 s) | impact | 0.684 (`splash.json`) | 0.336 -> 0.234 | 0.197 | 0.200 | tail 1.2-2.0 s: 0.235 |
| big heavy splash (2.6 s) | impact | 0.634 | 0.182 -> 0.165 | 0.126 | 0.140 | tail 1.56-2.6 s: 0.277 |
| double splash (2.65 s) | impact | 0.553 | 0.333 -> 0.256 | 0.229 | 0.215 | tail 1.59-2.65 s: 0.278 |
| stream with birdsong (12 s, birds suppressed, 7.9 % of energy removed) | texture | 0.486 (`water.json`) | 0.171 -> 0.161 | 0.170 | 0.176 | second half 6-12 s: 0.177 |
| cat meow (1.1 s) | voice | - | 0.343 -> 0.268 (0.204 after fine-tune) | 0.204 | 0.238 | second meow recording: 0.291; pitch error 18.5 cents |
| reverberant chant (15 s) | choir | - | 0.474 -> 0.393 (first 6 s) | 0.263 | 0.266 | 6-15 s: 0.230; pitch error 25 cents |

Read these honestly:
- The splash and voice recipes follow their clip's timeline: the band grid and the f0 and level curves come
  from the recording. Low distances there partly measure how closely the recipe copies the clip, not
  understanding of the physics. The macros and the editable grid are what make it a recipe, not a sample.
- The texture recipe copies only the mean spectrum. Its hold-out on unseen audio (0.177) is about as good as its
  fit (0.161), so it generalises within the clip. Its weakest part is `stft` (0.36): the fine time-frequency
  detail of real water (individual bubbles and gurgles) is smoother in the recipe.
- The cat's weakest parts are `stft` and `spectral`: real meows have formant motion (the mouth opening and
  closing) that a fixed formant envelope does not have. Moving formants (an automatable `formant_shift` or
  per-formant curves) are the next step.
- The chant is a rough first attempt: one melodic line as 4 unison voices plus Freeverb. Real choirs have
  vowel changes, consonants, and a long, natural hall.
