# Roadmap (build order from the knowledge base)

Status as of 0.1.0 (16 Nov 2026).

| step | content | status |
|---|---|---|
| 1 | Repo, recipe schema, render CLI, tests | done |
| 2 | Modal resonators + exciters (impact, friction, scrape), material table, Klatzky decay | done (scrape = friction + roughness; no separate rolling model yet) |
| 3 | Noise, stochastic events, source-filter (crackle, turbulence, valve pulse, formants, bubbles) | done |
| 4 | Waveguides and tubes (string, tube, cone, dispersion, Helmholtz, horn) | done (1D only) |
| 5 | Propagation and room (distance + ISO air absorption, Doppler pass-by, ground, image-source room, outdoor) | done (mono) |
| 6 | Five reference recipes + docs | done; quality notes below |
| 7 | Analysis and stem split | first version: HPR stems, sinusoidal tracks, features |
| 8 | Differentiable matching (PyTorch twin, analytic init, MR-STFT + texture-statistic losses) | next |
| 9 | AIDAW plugin, MCP server, compiled real-time core, recipe library | later |

## Next concrete tasks

- Fit `modal` parameters (frequencies, decays, amplitudes) from `analysis.track_partials` output, closing the loop
  for impacts first.
- Stereo/binaural output and per-layer propagation.
- Rolling, crumpling and fracture models (SDT papers as reference; reimplement).
- Calibrate explosion `size` to charge/distance; measure match timings from recordings.
- Collect a small CC0 / own-recorded reference set (duck, engines, foghorn, match) for objective checks.
- Optional external hooks: Demucs (MIT) for isolation; DAC (MIT) embeddings for FAD-style scoring.
