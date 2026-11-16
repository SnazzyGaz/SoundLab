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
