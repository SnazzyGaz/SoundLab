"""SoundLab: a physical-modelling sound engine (excitation -> resonator -> radiation -> propagation -> room).

Library API (what AIDAW and other hosts use):

    import soundlab
    audio, sr = soundlab.render("recipes/duck.json")          # or a dict
    soundlab.render_to_file(recipe, "duck.wav", mp3=True)
    soundlab.list_models(); soundlab.describe("engine")
    feats = soundlab.analyze_file("sample.wav", out_dir="stems/")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import models as _models  # noqa: F401  (registers all blocks)
from . import materials
from .core import REGISTRY, RecipeError
from .recipe import load as load_recipe
from .recipe import render, validate

__version__ = "0.1.0"
__all__ = ["render", "render_to_file", "list_models", "describe", "validate", "load_recipe", "analyze",
           "analyze_file", "materials", "RecipeError", "__version__"]


def list_models(stage: str | None = None) -> list[dict]:
    return [{"name": m.name, "stage": m.stage, "doc": m.doc} for m in REGISTRY.values()
            if stage is None or m.stage == stage]


def describe(name: str) -> dict:
    if name not in REGISTRY:
        raise RecipeError(f"unknown model {name!r}")
    return REGISTRY[name].describe()


def render_to_file(recipe: Any, path: str | Path, mp3: bool = False, sample_rate: int | None = None,
                   seed: int | None = None, bits: int = 24) -> dict:
    from .audio_io import write_mp3, write_wav
    audio, sr = render(recipe, sample_rate=sample_rate, seed=seed)
    path = Path(path)
    out = {"wav": str(write_wav(path, audio, sr, bits)), "sample_rate": sr, "seconds": len(audio) / sr}
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
