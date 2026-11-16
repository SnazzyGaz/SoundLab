# Roadmap (build order from the knowledge base)

Status as of 0.5.0 (16 Nov 2026). 0.5.0 opened up the muffled engine (sharp valve front + smooth `rasp`, no top cracks) and made the matchbox light (thin card, light sticks). 0.4.0 fixed the matchbox fry, the foghorn noise tail and the engine buzz from the v3 listening notes. 0.3.0 fixed the synthesis artefacts behind the v2 listening notes (see [artefacts.md](artefacts.md)), fitted all five sounds to the reference recordings and added the `shaker` model.

| step | content | status |
|---|---|---|
| 1 | Repo, recipe schema, render CLI, tests | done |
| 2 | Modal resonators + exciters (impact, friction, scrape), material table, Klatzky decay | done (scrape = friction + roughness; no separate rolling model yet) |
| 3 | Noise, stochastic events, source-filter (crackle, turbulence, valve pulse, formants, bubbles) | done |
| 4 | Waveguides and tubes (string, tube, cone, dispersion, Helmholtz, horn) | done (1D only) |
| 5 | Propagation and room (distance + ISO air absorption, Doppler pass-by, ground, image-source room, outdoor) | done (mono) |
| 6 | Five reference recipes + docs | done; v2 reworked all five after listening notes and reference recordings |
| 7 | Analysis and stem split | done: HPR stems, sinusoidal tracks, features; robust pitch/decay/modulation (0.2) |
| 8 | Parameter matching (analysis-by-synthesis) | first version done: `soundlab match`, gradient-free CMA-ES / Nelder-Mead, spectral + MR-STFT + envelope + texture + pitch losses. A differentiable twin is still future work |
| 9 | AIDAW plugin, MCP server, compiled real-time core, recipe library | later |

## Next concrete tasks

- Measure the simple Python versions of the five sounds (the files never reached the build machine)
  against ours and his refs, and check the v5 lessons hold.
- Engine: the 200-400 Hz octave is still ~6 dB hot against the V8 clip, and the long `engine.json` rev to 4500 rpm
  has no reference at that speed.

- Oversample `rough_pulses` (or use a smoother pulse) to take it from -43 dB to below -60 dB inharmonic energy.
- Make every noise source sample-rate invariant (only the oversampled blocks are now).
- Explosion: one recipe cannot match both a 2.5 s blast and a 7 s cinematic one; calibrate `size` and
  `fireball_s` against several refs at once. Against the older field-recording refs v3 is worse than v2
  (the recipe is now cinematic scale).
- Duck and match moved away from the older refs while getting much closer to the references. Fit against both sets at
  once (mean loss) to find values that suit both.

- Fit `modal` parameters (frequencies, decays, amplitudes) from `analysis.track_partials` output, closing the loop
  for impacts first.
- Stereo/binaural output and per-layer propagation.
- Rolling, crumpling and fracture models (SDT papers as reference; reimplement).
- Calibrate explosion `size` to charge/distance.
- Engine: the two engine refs disagree wildly. One is a close, bright engine-bay recording; the other is a
  bass-heavy recording that is nearly all below 100 Hz. The v2 distance moved only a little (0.935 -> 0.886),
  and a fit to the bright one asks for less `body`, the opposite of the listening note. Get a matched set of
  recordings (same car, idle + rev, known mic distance) and fit against that.
- Fit against several references at once (mean loss), and add a perceptual embedding distance (e.g. DAC/CLAP)
  as an optional external hook.
- Own-recorded, full-quality references. Several current refs are 128 kbps Freesound previews.
- Optional external hooks: Demucs (MIT) for isolation; DAC (MIT) embeddings for FAD-style scoring.
