"""Recipe format and renderer.

A recipe is plain JSON:

{
  "soundlab": 1,
  "name": "steel bar hit in a hall",
  "sample_rate": 48000, "duration": 3.0, "seed": 1,
  "layers": [
    {"name": "hit", "start": 0.0, "gain_db": 0,
     "chain": [
        {"model": "impact", "params": {"velocity": 2}},
        {"model": "modal", "params": {"material": "steel", "shape": "bar", "length": 0.4}},
        {"model": "radiation", "params": {"radius": 0.02}}
     ]},
    {"model": "fire", "params": {"intensity": 0.4}, "gain_db": -12}
  ],
  "post": [ {"model": "distance", "params": {"distance": 30}}, {"model": "room", "params": {"volume": 800}} ],
  "master": {"normalize_db": -1.0, "fade_out": 0.05}
}

A layer is either a single model ({"model": ..., "params": ...}) or a chain. The first block of a chain must be a
source (excitation or sound); later blocks process. "post" blocks (propagation/room) run on the mixed layers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import dsp
from .core import REGISTRY, Ctx, RecipeError
from . import fastpath
from .fastpath import check

FORMAT_VERSION = 1


def load(path_or_obj: Any) -> dict:
    if isinstance(path_or_obj, dict):
        return path_or_obj
    p = Path(path_or_obj)
    return json.loads(p.read_text())


def _block(ctx: Ctx, spec: dict, x: np.ndarray | None) -> np.ndarray:
    check()     # the slider app stops a render that a newer one has replaced (no-op elsewhere)
    if not fastpath.block_memo_on():
        return _block_run(ctx, spec, x)
    # slider app: a block whose params, length, random state and input are unchanged gives the same samples and
    # leaves the random stream where it did, so reuse both (a distance or room move skips the source layers)
    import json
    key = ("block", json.dumps(spec, sort_keys=True, default=repr), ctx.sr, ctx.n, ctx.c,
           json.dumps(ctx.rng.bit_generator.state, sort_keys=True, default=repr),
           None if x is None else fastpath.digest(x))
    hit = fastpath.MEMO.get(key)
    if hit is not None:
        y, state = hit[0]
        ctx.rng.bit_generator.state = state
        return y.copy()
    y = _block_run(ctx, spec, x)
    fastpath.MEMO.put(key, (y.copy(), ctx.rng.bit_generator.state), y.nbytes)
    return y


def _block_run(ctx: Ctx, spec: dict, x: np.ndarray | None) -> np.ndarray:
    if "model" not in spec:
        raise RecipeError(f"block without 'model': {spec}")
    name = spec["model"]
    if name not in REGISTRY:
        raise RecipeError(f"unknown model {name!r}; run 'soundlab models' for the list")
    m = REGISTRY[name]
    params = m.resolve(spec.get("params"))
    if m.needs_input and x is None:
        raise RecipeError(f"{name} ({m.stage}) needs an input; put a source block before it")
    if m.needs_input:
        y = m.fn(ctx, x, **params)
    else:
        need = m.min_samples(params, ctx.sr)
        if need > ctx.n:    # too short for the model's own windows (or the event is past the end): render, crop
            y = np.asarray(m.fn(Ctx(ctx.sr, need, ctx.rng, ctx.c), None, **params), float)[:ctx.n]
        else:
            y = m.fn(ctx, None, **params)
        if x is not None:   # a source in the middle of a chain is added to the running signal
            y = x + np.asarray(y, float)
    y = np.asarray(y, float)
    if y.shape != (ctx.n,):
        raise RecipeError(f"{name} returned shape {y.shape}, expected ({ctx.n},)")
    if not np.all(np.isfinite(y)):
        raise RecipeError(f"{name} produced non-finite samples")
    return y


def validate(recipe: dict) -> list[str]:
    """Return a list of problems (empty if the recipe is valid)."""
    errs = []
    if recipe.get("macros") or recipe.get("vary"):
        from .macros import validate_macros
        errs += validate_macros(recipe)
    if not isinstance(recipe.get("layers"), list) or not recipe["layers"]:
        errs.append("recipe needs a non-empty 'layers' list")
        return errs
    for i, layer in enumerate(recipe["layers"]):
        chain = layer.get("chain") or ([layer] if "model" in layer else None)
        if not chain:
            errs.append(f"layer {i}: needs 'model' or 'chain'")
            continue
        for j, b in enumerate(chain):
            name = b.get("model")
            if name not in REGISTRY:
                errs.append(f"layer {i} block {j}: unknown model {name!r}")
                continue
            m = REGISTRY[name]
            if j == 0 and m.needs_input:
                errs.append(f"layer {i}: first block {name!r} is a {m.stage}; chains start with a source")
            try:
                m.resolve(b.get("params"))
            except RecipeError as e:
                errs.append(f"layer {i} block {j}: {e}")
    for j, b in enumerate(recipe.get("post", [])):
        name = b.get("model")
        if name not in REGISTRY or not REGISTRY[name].needs_input:
            errs.append(f"post {j}: {name!r} must be a processing block (resonator/radiation/propagation/room/fx)")
        else:
            try:
                REGISTRY[name].resolve(b.get("params"))
            except RecipeError as e:
                errs.append(f"post {j}: {e}")
    return errs


def resolve(recipe: Any, macros: dict | None = None, variation: int = 0) -> tuple[dict, int]:
    """Apply macro values (defaults where not given) and, for variation k >= 1, the seeded jitter.
    Returns (concrete recipe without 'macros'/'vary', seed to render with)."""
    from . import models  # noqa: F401
    from .macros import variation as _variation
    r = load(recipe)
    rr, _, seed = _variation(r, int(variation or 0), macros)
    return rr, seed


def ring_tail_s(recipe: dict) -> float:
    """How long the recipe keeps sounding after its sources stop: per layer the tails of its processing blocks
    in series, the longest layer, then the post blocks in series (see soundlab/models/tails.py)."""
    from . import models  # noqa: F401
    def chain_tail(blocks):
        t = 0.0
        for b in blocks:
            m = REGISTRY.get(b.get("model"))
            if m is not None and m.needs_input:
                t += m.tail_seconds(m.resolve(b.get("params")))
        return t
    layers = [chain_tail(layer.get("chain") or [layer]) for layer in recipe.get("layers", [])]
    return max(layers + [0.0]) + chain_tail(recipe.get("post", []))


def render(recipe: Any, sample_rate: int | None = None, seed: int | None = None, return_layers: bool = False,
           macros: dict | None = None, variation: int = 0, ring_out: bool | float = False):
    """Render a recipe (dict or path) to a mono float array. Deterministic for a given seed.

    macros: {name: value} for the recipe's macro sliders (defaults otherwise). variation: 0 = the recipe as
    written; k >= 1 = seeded variation k (jittered macros/controls and its own seed).

    ring_out: the sources still stop at `duration` (with the master fade-out), but rooms, reverbs, ringing
    resonators and echoes keep running for their own tail (ring_tail_s; True caps it at 20 s, a number caps it
    at that many seconds), so the result is longer than `duration` and nothing is cut mid-ring. Exports use it
    and then trim the silent end. False (default): exactly `duration`, as always."""
    from . import models  # noqa: F401  (registers blocks)

    r, vseed = resolve(recipe, macros, variation)
    if seed is None and variation:
        seed = vseed
    errs = validate(r)
    if errs:
        raise RecipeError("; ".join(errs))
    sr = int(sample_rate or r.get("sample_rate", 48000))
    dur = float(r.get("duration", 3.0))
    n = int(round(dur * sr))
    master = r.get("master", {})
    ring = 0
    if ring_out:
        cap = 20.0 if ring_out is True else float(ring_out)
        ring = int(round(min(ring_tail_s(r), cap) * sr))
    N = n + ring
    fade_out = float(master.get("fade_out", 0.02))

    def stop(x):    # sources end at `duration`: fade them there, then silence while the tails ring
        return np.concatenate([dsp.fade(x, sr, 0.0, fade_out), np.zeros(ring)])

    root = np.random.default_rng(int(r.get("seed", 0) if seed is None else seed))
    mix = np.zeros(N)
    stems = {}
    for i, layer in enumerate(r["layers"]):
        chain = layer.get("chain") or [layer]
        start = int(float(layer.get("start", 0.0)) * sr)
        ln = n - start      # a negative start is pre-roll: the layer began before t = 0 and its head is cut
        layer_seed = int(root.integers(0, 2**31 - 1))   # drawn even for skipped layers: later layers keep their seed
        if ln <= 0:
            continue
        ctx = Ctx(sr, ln, np.random.default_rng(layer_seed))
        x = None
        for b in chain:
            m = REGISTRY.get(b.get("model"))
            if ring and m is not None and x is not None and len(x) == ln:
                if m.needs_input:               # first processing block: from here on the layer rings out
                    x = stop(x)
                    ctx = Ctx(sr, ln + ring, ctx.rng, ctx.c)
                    x = _block(ctx, b, x)
                else:                           # a source added mid-chain still stops at `duration`
                    x = _block(ctx, b, x)
                continue
            if ring and m is not None and not m.needs_input and x is not None:
                x = x + stop(_block(Ctx(sr, ln, ctx.rng, ctx.c), b, None))
                continue
            x = _block(ctx, b, x)
        if ring and len(x) == ln:
            x = stop(x)
        g = 10 ** (float(layer.get("gain_db", 0.0)) / 20)
        if start < 0:
            x = x[-start:]
        if layer.get("normalize", True):
            x = x / (np.max(np.abs(x)) + 1e-12)
        y = np.zeros(N)
        y[max(start, 0):] = x * g
        stems[layer.get("name", f"layer{i}")] = y
        mix += y
    pctx = Ctx(sr, N, np.random.default_rng(int(root.integers(0, 2**31 - 1))))
    for b in r.get("post", []):
        mix = _block(pctx, b, mix)
    if master.get("highpass_hz"):
        mix = dsp.highpass(mix, float(master["highpass_hz"]), sr)
    # DC blocker on every recipe: 2nd-order high pass at 7 Hz (inaudible, removes offsets that cost headroom
    # and click at edits). "dc_block_hz": 0 in "master" turns it off.
    dc_hz = float(master.get("dc_block_hz", 7.0))
    if dc_hz > 0:
        mix = dsp.highpass(mix - mix.mean(), dc_hz, sr)
    mix = dsp.fade(mix, sr, 0.0005, fade_out)
    if master.get("normalize_db", -1.0) is not None:
        mix = dsp.normalize_peak(mix, float(master.get("normalize_db", -1.0)))
    if master.get("gain_db"):
        mix = mix * 10 ** (float(master["gain_db"]) / 20)  # v0.7: level after normalising (distance macros)
    if return_layers:
        return mix, sr, stems
    return mix, sr
