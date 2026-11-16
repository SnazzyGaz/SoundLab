# Architecture

SoundLab is a standalone repository. AIDAW (GPL-3.0-or-later) depends on it as a library; SoundLab never imports
AIDAW. MIT code can be used inside a GPL project, not the other way round.

```
soundlab/
  core.py        Ctx (sample rate, length, seeded RNG), Param specs, automation (ctrl), model registry
  dsp.py         filters (RBJ biquads, TPT state-variable), waveguide loops, noise, envelopes, FFT filtering
  materials.py   material table (Cremer & Heckl) and Klatzky perceptual decay
  models/
    excitation.py   impact, friction, noise, turbulence, valve_pulse, chopper, combustion, crackle, bubbles
    resonator.py    modal, string, tube, formant, helmholtz, filter
    radiation.py    radiation, horn
    propagation.py  distance (ISO 9613-2), passby (Doppler), ground
    room.py         room (image source + statistical tail), outdoor (echoes + tail)
    sounds.py       duck, engine, foghorn, explosion, match, fire, water, hit
  recipe.py      JSON recipe validation and rendering
  analysis.py    stem split + partial tracking + features
  audio_io.py    WAV (scipy) and MP3 (ffmpeg)
  cli.py         soundlab command
recipes/         reference and example recipes
tests/           physics and recipe tests
tools/           doc generator
```

## Block contract

A block is a function `fn(ctx, x, **params) -> np.ndarray` registered with `@model(name, stage, doc, params,
sources)`. Sources (`excitation`, `sound`) get `x=None`; processors get the running signal. Every parameter is a
`Param` with default, unit, realistic range, meaning and (where possible) a source. This metadata drives
validation, `soundlab describe`, the generated docs, and later the AIDAW UI and the fitting code.

## Determinism

Each layer gets its own RNG seeded from the recipe seed, so adding a layer does not change the others.

## Using it from AIDAW

```python
import soundlab
audio, sr = soundlab.render(recipe_dict)        # mono float64
```

Planned AIDAW integration (not built yet): a `soundlab` instrument type whose JSON is a SoundLab recipe, ops for
render/analyse, and the MCP tools exposed through AIDAW's server.

## Performance (0.1)

Pure Python + NumPy/SciPy, offline. Most blocks are vectorised or run as IIR filters through `scipy.signal.lfilter`.
Per-sample loops remain in `friction` (stick-slip) and the time-varying state-variable filter; the shipped recipes
took 0.02-0.5 s on the box for 1.5-9 s of audio (faster than real time, but offline). A real-time streaming engine comes with the compiled core (roadmap step 9).
