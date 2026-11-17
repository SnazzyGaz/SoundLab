"""SoundLab: a physical-modelling sound engine (excitation -> resonator -> radiation -> propagation -> room).

Library API (what AIDAW and other hosts use):

    import soundlab
    audio, sr = soundlab.render("recipes/duck.json")          # or a dict
    audio, sr = soundlab.render("recipes/hit.json", macros={"material": "metal", "size": 0.8}, variation=3)
    soundlab.batch(["recipes/blocks"], "out/", variations=5)
    soundlab.render_to_file(recipe, "duck.wav", mp3=True)
    soundlab.list_models(); soundlab.describe("engine")
    feats = soundlab.analyze_file("sample.wav", out_dir="stems/")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import models as _models  # noqa: F401  (registers all blocks)
from . import materials
from .core import REGISTRY, RecipeError
from .recipe import load as load_recipe
from .recipe import render, validate

__version__ = "0.25.3"
__all__ = ["render", "render_stereo", "render_loop", "render_to_file", "list_models", "describe", "macros", "variation", "batch", "validate",
           "load_recipe", "analyze",
           "analyze_file", "materials", "RecipeError", "__version__", "match", "compare"]


def list_models(stage: str | None = None) -> list[dict]:
    return [{"name": m.name, "stage": m.stage, "doc": m.doc} for m in REGISTRY.values()
            if stage is None or m.stage == stage]


def describe(name: str) -> dict:
    """A model's controls, or (for a recipe path/dict) the recipe's macros first, then its layers."""
    if isinstance(name, dict) or (isinstance(name, (str, Path)) and str(name).endswith(".json")):
        from .macros import describe_recipe
        return describe_recipe(load_recipe(name))
    if name not in REGISTRY:
        raise RecipeError(f"unknown model {name!r}")
    return REGISTRY[name].describe()


def macros(recipe: Any) -> dict:
    """The recipe's macro sliders: {name: spec}."""
    from .macros import macro_specs
    return macro_specs(load_recipe(recipe))


def variation(recipe: Any, k: int, macros: dict | None = None) -> dict:
    """Concrete recipe for variation k (macros applied, seeded jitter); render it with its 'seed'."""
    from .macros import variation as _v
    r, vals, seed = _v(load_recipe(recipe), k, macros)
    r["seed"] = seed
    r["macro_values"] = vals
    return r


def batch(*args, **kw):
    """Batch export (see soundlab.batch.batch_export)."""
    from .batch import batch_export
    return batch_export(*args, **kw)


def _takes(raw: dict, seed: int | None, variation: int, macros: dict | None, one_take) -> tuple:
    """Stereo from `one_take(seed) -> (mono, sr)`, per the recipe's "stereo" config (see soundlab.stereo)."""
    from . import stereo as st
    from .recipe import resolve
    cfg = st.config(raw)
    if seed is None:
        _, vseed = resolve(raw, macros, variation)
        seed = vseed if variation else int(raw.get("seed", 0))
    x1, sr = one_take(seed)
    if cfg["mode"] == "seeds":
        x2, _ = one_take(int(seed) + st.SECOND_SEED)
        y = st.combine(x1, x2, sr, cfg["width"], float(cfg["mono_below_hz"]), circular=one_take.circular)
        pk = max(np.max(np.abs(x1)), np.max(np.abs(x2)))
    else:
        y = st.from_mono(x1, sr, cfg, circular=one_take.circular)
        pk = np.max(np.abs(x1))
    return y * (pk / (np.max(np.abs(y)) + 1e-12)), sr       # same peak as a mono take (keeps master.gain_db)


def render_stereo(recipe: Any, macros: dict | None = None, variation: int = 0, sample_rate: int | None = None,
                  seed: int | None = None, ring_out: bool | float = False):
    """Stereo render (n, 2) following the recipe's "stereo" config: two seeds or a decorrelated copy, mono
    below ~150 Hz (see soundlab.stereo). Returns (audio, sr)."""
    raw = load_recipe(recipe)

    def take(s):
        return render(raw, sample_rate=sample_rate, seed=s, macros=macros, variation=variation, ring_out=ring_out)
    take.circular = False
    return _takes(raw, seed, variation, macros, take)


def render_loop(recipe: Any, seconds: float, macros: dict | None = None, variation: int = 0,
                sample_rate: int | None = None, seed: int | None = None, xfade_s: float | None = None,
                stereo: bool = False):
    """Render a seamless loop of `seconds`: render lead-in + loop + crossfade, drop the lead-in (onsets,
    build-ups), fold the extra part back over the start with an equal-power crossfade. Returns (audio, sr).
    Recipes tagged with "loop" ({"lead_s", "xfade_s"}) are the stationary ones this is meant for.
    stereo=True: an (n, 2) loop, each step circular so the seam stays seamless on both sides."""
    import copy
    from .finish import fold_loop
    from .recipe import resolve
    raw = load_recipe(recipe)
    cfg = raw.get("loop") or {}
    lead = float(cfg.get("lead_s", 1.0))
    xf = float(xfade_s if xfade_s is not None else cfg.get("xfade_s", 0.5))
    xf = min(xf, seconds / 2)
    r, vseed = resolve(raw, macros, variation)
    r = copy.deepcopy(r)
    r["duration"] = lead + seconds + xf + 0.2
    r.setdefault("master", {})["fade_out"] = 0.0
    if seed is None and variation:
        seed = vseed

    def take(s):
        y, sr = render(r, sample_rate=sample_rate, seed=s)
        return fold_loop(y[int(round(lead * sr)):], sr, seconds, xf), sr
    take.circular = True
    if not stereo:
        return take(seed)
    return _takes(dict(raw, seed=r.get("seed", raw.get("seed", 0))), seed, 0, None, take)


def is_loopable(recipe: Any) -> bool:
    return bool(load_recipe(recipe).get("loop"))


def wants_stereo(recipe: Any, loop: bool, stereo: bool | None = None) -> bool:
    """stereo=None (auto): stereo for loops of recipes that declare a "stereo" bed config (rain, wind, fire,
    water, ...); mono otherwise, since a game positions one-shots itself. True/False force it."""
    if stereo is not None:
        return bool(stereo)
    return bool(loop) and bool(load_recipe(recipe).get("stereo"))


def render_to_file(recipe: Any, path: str | Path, mp3: bool = False, sample_rate: int | None = None,
                   seed: int | None = None, bits: int = 24, macros: dict | None = None, variation: int = 0,
                   level: str = "lufs", trim: bool = True, loop: float | None = None,
                   stereo: bool | None = None) -> dict:
    """Render and write a game-ready file: tail trimmed, loudness-normalised per category with a true-peak
    ceiling (see soundlab.finish). level='peak' keeps the raw peak-normalised render; trim=False keeps the
    full length; loop=SECONDS writes a seamless loop of that length instead. With trim, reverb and ringing
    tails run past the recipe's duration until they decay (render(ring_out=True)) before the trim.
    stereo: None = auto (stereo loops for recipes with a "stereo" config), True/False to force."""
    from .audio_io import write_mp3, write_wav
    from .finish import finish
    raw = load_recipe(recipe)
    st = wants_stereo(raw, bool(loop), stereo)
    if loop:
        audio, sr = render_loop(raw, float(loop), macros=macros, variation=variation, sample_rate=sample_rate,
                                seed=seed, stereo=st)
    elif st:
        audio, sr = render_stereo(raw, macros=macros, variation=variation, sample_rate=sample_rate, seed=seed,
                                  ring_out=trim)
    else:
        # v0.22: let rooms, reverbs and ringing parts finish (the trim then cuts the silence); --no-trim keeps
        # exactly the recipe length
        audio, sr = render(raw, sample_rate=sample_rate, seed=seed, macros=macros, variation=variation,
                           ring_out=trim)
    audio, fin = finish(audio, sr, raw, level=level, trim=trim, loop=bool(loop))
    path = Path(path)
    out = {"wav": str(write_wav(path, audio, sr, bits)), "sample_rate": sr, "seconds": len(audio) / sr,
           "channels": 1 if audio.ndim == 1 else audio.shape[1], "finish": fin}
    if mp3:
        out["mp3"] = str(write_mp3(path.with_suffix(".mp3"), audio, sr))
    return out


def analyze(audio, sr: int, out_dir: str | Path | None = None, plot: bool = True) -> dict:
    from .analysis import analyze as _an
    return _an(audio, sr, out_dir, plot)


def analyze_file(path: str | Path, out_dir: str | Path | None = None, plot: bool = True) -> dict:
    from .audio_io import read_audio
    x, sr = read_audio(path)
    return analyze(x, sr, out_dir, plot)


def match(recipe: Any, ref_path: str | Path, params=None, out_dir: str | Path | None = None, **kw) -> dict:
    """Fit a recipe's controls to a reference recording (see soundlab.fit). Returns the fit result."""
    from . import fit as _fit
    from .audio_io import read_audio
    x, sr = read_audio(ref_path)
    res = _fit.fit(load_recipe(recipe), x, sr, params, **kw)
    if out_dir:
        _fit.report(res, x, out_dir, Path(ref_path).name)
    return res


def compare(ref_path: str | Path, cand_path: str | Path, ref_start=None, ref_end=None) -> dict:
    """Distance between a reference recording and a render (spectral/stft/envelope/texture/total)."""
    from . import fit as _fit
    from .audio_io import read_audio
    r, rsr = read_audio(ref_path)
    c, csr = read_audio(cand_path)
    return _fit.compare_signals(r, rsr, c, csr, ref_start, ref_end)
