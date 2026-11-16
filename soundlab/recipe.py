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

FORMAT_VERSION = 1


def load(path_or_obj: Any) -> dict:
    if isinstance(path_or_obj, dict):
        return path_or_obj
    p = Path(path_or_obj)
    return json.loads(p.read_text())


def _block(ctx: Ctx, spec: dict, x: np.ndarray | None) -> np.ndarray:
    if "model" not in spec:
        raise RecipeError(f"block without 'model': {spec}")
    name = spec["model"]
    if name not in REGISTRY:
        raise RecipeError(f"unknown model {name!r}; run 'soundlab models' for the list")
    m = REGISTRY[name]
    params = m.resolve(spec.get("params"))
    if m.needs_input and x is None:
        raise RecipeError(f"{name} ({m.stage}) needs an input; put a source block before it")
    if not m.needs_input and x is not None:
        # a source in the middle of a chain is added to the running signal
        return x + m.fn(ctx, None, **params)
    y = m.fn(ctx, x, **params)
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


def render(recipe: Any, sample_rate: int | None = None, seed: int | None = None, return_layers: bool = False,
           macros: dict | None = None, variation: int = 0):
    """Render a recipe (dict or path) to a mono float array. Deterministic for a given seed.

    macros: {name: value} for the recipe's macro sliders (defaults otherwise). variation: 0 = the recipe as
    written; k >= 1 = seeded variation k (jittered macros/controls and its own seed)."""
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
    root = np.random.default_rng(int(r.get("seed", 0) if seed is None else seed))
    mix = np.zeros(n)
    stems = {}
    for i, layer in enumerate(r["layers"]):
        chain = layer.get("chain") or [layer]
        start = int(float(layer.get("start", 0.0)) * sr)
        ln = n - start
        if ln <= 0:
            continue
        ctx = Ctx(sr, ln, np.random.default_rng(int(root.integers(0, 2**31 - 1))))
        x = None
        for b in chain:
            x = _block(ctx, b, x)
        g = 10 ** (float(layer.get("gain_db", 0.0)) / 20)
        if layer.get("normalize", True):
            x = x / (np.max(np.abs(x)) + 1e-12)
        y = np.zeros(n)
        y[start:] = x * g
        stems[layer.get("name", f"layer{i}")] = y
        mix += y
    pctx = Ctx(sr, n, np.random.default_rng(int(root.integers(0, 2**31 - 1))))
    for b in r.get("post", []):
        mix = _block(pctx, b, mix)
    master = r.get("master", {})
    if master.get("highpass_hz"):
        mix = dsp.highpass(mix, float(master["highpass_hz"]), sr)
    mix = dsp.fade(mix, sr, 0.0005, float(master.get("fade_out", 0.02)))
    if master.get("normalize_db", -1.0) is not None:
        mix = dsp.normalize_peak(mix, float(master.get("normalize_db", -1.0)))
    if master.get("gain_db"):
        mix = mix * 10 ** (float(master["gain_db"]) / 20)  # v0.7: level after normalising (distance macros)
    if return_layers:
        return mix, sr, stems
    return mix, sr
