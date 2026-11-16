# Recipe format (version 1)

```json
{
  "soundlab": 1,                  // format version
  "name": "free text",
  "sample_rate": 48000,           // Hz
  "duration": 3.0,                // seconds
  "seed": 1,                      // integer; same seed = identical render
  "layers": [ LAYER, ... ],
  "post":   [ BLOCK, ... ],       // processing blocks applied to the mix (propagation, room, filters)
  "master": {"normalize_db": -1.0, "fade_out": 0.02, "highpass_hz": 0}
}
```

`LAYER` is either a single block or a chain:

```json
{"name": "engine", "model": "engine", "params": {...}, "start": 0.0, "gain_db": 0, "normalize": true}
{"name": "tap", "chain": [BLOCK, BLOCK, ...], "start": 0.5, "gain_db": -6}
```

`BLOCK` is `{"model": "<name>", "params": {...}}`.

Rules:

- A chain starts with a source (an `excitation` or `sound` model). Later blocks process the signal.
  A source placed later in a chain is added to the running signal.
- Each layer is peak-normalised before `gain_db` unless `"normalize": false`, so `gain_db` sets the balance
  between layers.
- Unknown models, unknown parameters, bad choices and automation on a non-automatable control are errors
  (`soundlab validate`).
- Automation: any control marked `automatable` accepts `[[t_seconds, value], ...]`. Frequencies and rpm
  interpolate in log space, other values linearly. Times are relative to the layer start.
- List-valued controls (`formants`, `ratios`, `reflectors`) take plain lists.
- Output is mono float, peak-normalised by `master.normalize_db` (set it to `null` to keep raw level).

The CLI can override a single-model layer's parameter: `--set 0.rpm=3000` (layer index, parameter).

## Generic synth blocks (0.6.0)

Besides the five physical stages there are two generic stages: `synth` blocks (osc, fm, noisegen, grains) are sources and start a chain; `fx` blocks (svf, env, shaper, delay, chorus, reverb, gain) process a chain or run in `post`. Extra top-level keys such as `category` are ignored by the renderer (reserved for batch naming). See [synth-blocks.md](synth-blocks.md).

## Macros and variations (0.7.0)

Optional top-level keys:
- `"category"`: used in batch file names (`impact`, `ui`, `vehicle`, ...).
- `"macros"`: `{name: {"default": 0.5, "doc": "...", "jitter": 0.04, "map": [{"to": "0.length", "op": "mul",
  "points": [[0, 0.35], [0.5, 1], [1, 2.5]]}, ...]}}`, or a named choice
  `{name: {"default": "wood", "choices": {"wood": [{"to": "0.material", "op": "set", "value": "oak"}], ...}}}`.
  Ops: `mul`, `add`, `stretch` (breakpoint times), `set`; `range` or piecewise `points`; optional `curve`,
  `clamp`. Targets: `L.param`, `L.B.param` (L = layer index or name), `L.gain_db`, `L.start`,
  `post.J.param`, `duration`, `master.gain_db`. At their defaults macros leave the recipe unchanged.
- `"vary"`: `{target: {"cents"|"rel"|"db"|"abs": amount}}`, the per-control jitter for variations.
- `"master": {"gain_db": ...}`: level after normalising (used by `distance` macros).
See [macros-and-variations.md](macros-and-variations.md).
