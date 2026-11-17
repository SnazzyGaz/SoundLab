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
| `impact` | (v0.18) `bubble_cloud` following the recording's level, quiet `bandnoise` bed, gated `slap`, `splash` at each main onset, post `eq` | level, bubble_size, churn |
| `texture` | (v0.18) `bubble_cloud` plus a spray `bandnoise` bed and post `eq`. It does not copy the timeline, so it loops and varies | level, bubble_size, churn |
| `voice` | `harmonic` on the measured f0 curve (breakpoints within 20 cents), with a time-varying envelope and aperiodicity (v0.18) and the loudness envelope as automation | level, pitch, size |
| `choir` | `harmonic` with 4 detuned unison voices on the harmonic-sum pitch track, through `reverb`, plus a `hall_air` layer | level, pitch, size |

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

## v0.18: liquids as bubbles, voices as harmonic + envelope + aperiodicity

The v0.17 recipes scored well, but they told us what the distance misses. Water built from band-shaped noise
sounds like air, and a splash body made of noise sounds rushy or muddy. Real liquid sound is bubbles (van den
Doel 2005; the liquids models of the Sound Design Toolkit; Moss et al. 2010). v0.18 rebuilds the water kinds on
them, following methods from open-source projects and papers (credits in [THIRD_PARTY.md](../THIRD_PARTY.md)):

- **`bubble_cloud`** (excitation): thousands of Minnaert bubbles per second (f = 3.26/r, van den Doel damping
  and pitch rise, power-law radii, amplitude ~ r^`amp_exponent`), vectorised by radius bins. `rate` and `level`
  can follow an envelope.
- **`impact`** recipes: a `bubble_cloud` whose level follows the recording, a quiet `bandnoise` bed (fit range
  capped at 12 dB under the bubbles), a `slap` noise layer gated to about 70 ms after each onset, the `splash`
  model at each onset, and a post `eq`.
- **`texture`** recipes: `bubble_cloud` (at least 1,000 bubbles/s) plus a spray bed capped at -12 dB, and a
  post `eq`.
- **`harmonic`** now takes `env_frames` (a time-varying spectral envelope: moving vowels) and `ap_frames` (band
  aperiodicity), as in the WORLD vocoder. Its breath noise is modulated by the glottal cycle (`pulse_noise`;
  Klatt 1990, Pink Trombone, CHIMERA). Voice recipes no longer have a separate, always-on breath layer.
- **`eq`** (fx): a graphic EQ curve. The spectral fine-tune now writes to the recipe's post `eq`, for every kind.
- `choir` adds a `hall_air` layer (the diffuse residual of the hall) and uses a steady vowel colour.

Distances, v0.17 -> v0.18 (whole clip; hold-out in brackets):

| clip | v0.17 | v0.18 | notes |
|---|---|---|---|
| medium splash | 0.197 (tail 0.235) | 0.190 (tail 0.383) | bubbles instead of a noise body; tail section worse |
| big heavy splash | 0.126 (0.277) | 0.225 (0.360) | the noise copy scored better; v0.18 trades distance for liquid character |
| double splash | 0.229 (0.278) | 0.208 (0.732) | whole clip closer, sparse tail droplets do not match (hold-out much worse) |
| stream | 0.170 (second half 0.177) | 0.203 (0.197) | now bubble-dominated by design; the distance prefers noise |
| cat meow | 0.204 (other meow 0.291) | 0.176 (0.309) | moving envelope, no separate breath layer; pitch error 15 cents |
| chant | 0.263 (6-15 s 0.230) | 0.244 (0.240) | |

Lesson: our distance rewards a noise copy of the spectrum over a liquid made of bubbles. For water, constrain
the fit to the right physics and judge by ear. A bubble-aware term in the distance (spectral peakiness or
event statistics) is on the roadmap.

## v0.19: a sparsity term in the fit loss

The v0.18 notes said it: the distance prefers noise for water. v0.19 adds a `sparsity` part to the loss (see
[fitting.md](fitting.md)), which compares the heavy tail of locally normalised STFT power. Shaped noise has a
light tail; bubbles and drops a heavy one. It is weighted 0.5 for `texture` and `impact` recipes and switched off
for `voice` and `choir` (there it pulled the timbre away and did not help). Re-run with budget 120, scored with
the v0.19 loss (both versions scored the same way; "old" = the v0.18 loss without sparsity):

| clip | v0.18 old / new loss | v0.19 old / new loss | sparsity v0.18 -> v0.19 |
|---|---|---|---|
| medium splash | 0.166 / 0.157 | 0.159 / 0.155 | 0.08 -> 0.12 |
| big heavy splash | 0.222 / 0.323 | 0.301 / 0.337 | 1.16 -> 0.63 (grainier, but the spectrum is further off) |
| double splash | 0.215 / 0.409 | 0.184 / 0.382 | 2.0 -> 2.0 (still far: the reference is much sparser) |
| stream | 0.156 / 0.197 | 0.166 / 0.161 | 0.53 -> 0.12 (bubble grain now matches) |
| cat meow, chant | unchanged | unchanged | not used for pitched kinds |

## First results, v0.17 (private reference recordings, kept outside the repo)

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
