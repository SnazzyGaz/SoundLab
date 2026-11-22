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

## 0.6.0: generic synth blocks

`soundlab/models/synth.py` holds the generic toolkit (stages `synth` and `fx`). Written from published descriptions: PolyBLEP (Valimaki & Huovilainen 2007), TPT SVF (Zavalishin 2012), FM (Chowning 1973), Freeverb structure (Jezar, public domain, as described in J.O. Smith, *Physical Audio Signal Processing*). No code copied.

## Slider app and blind test (0.12)

- `soundlab/gui/server.py`: standard-library HTTP server (`ThreadingHTTPServer`, bound to 127.0.0.1, port 8766)
  with a JSON API: `GET /api/recipes`, `GET /api/recipe?id=` (macros + every model control), `POST /api/render`
  (returns a WAV URL, waveform and log-frequency spectrogram), `/api/export`, `/api/batch`, `/api/save`.
  Requests carry `{id, macros, overrides, variation, seed}`; overrides are absolute values set after the macros
  (`"L.name"`, `"L.B.name"` for chain block B, `"post.J.name"`).
- Speed (`soundlab/fastpath.py`; none of it changes a sample, tests/test_fastpath.py checks bit identity):
  the slider app renders live while a slider is dragged (60 ms debounce). The page aborts the request a newer
  move replaces and sends `"supersede": true`; the server runs one render at a time and a render that a newer one has
  replaced stops at its next check (`{"superseded": true}`). Finished renders are kept per request (Play again, undo
  a slider), recipe blocks whose params, length, random state and input are unchanged are reused, and the
  band-envelope vocoders of the recording-stem families are memoised (their noise depends on the take, not on most
  sliders) and run their bands on every core (`SOUNDLAB_THREADS`, default 1 outside the app; `SOUNDLAB_MEMO_MB`,
  default 512; `SOUNDLAB_MEMO=0` turns the memos off). After a stem recipe is shown, the idle server renders each
  slider's ends and the next variation ahead, so the slider moves after that only run the cheap stages.
- `soundlab/gui/static/index.html`, `blindtest.html`: plain HTML + JavaScript, no build step, no CDN.
- `soundlab/gui/blindtest.py`: `/api/bt/*` (pack info, clips, start/resume a listener, save an answer, results).
- `soundlab/checkpoint.py`: listening packs and scoring (single-clip `build_pack` / `score_pack`, A/B `build` /
  `score`); `tools/checkpoint.py` is a thin wrapper for the A/B command line.
