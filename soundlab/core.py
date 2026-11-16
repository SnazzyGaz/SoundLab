"""Core types: render context, parameter specs, controls (constants or automation)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s at 20 C (Wikipedia, Speed of sound)


class RecipeError(ValueError):
    """Raised for invalid recipes or parameters."""


@dataclass
class Ctx:
    """Render context shared by every block in a chain."""

    sr: int
    n: int
    rng: np.random.Generator
    c: float = SPEED_OF_SOUND

    @property
    def duration(self) -> float:
        return self.n / self.sr

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n) / self.sr

    def child(self, n: int | None = None) -> "Ctx":
        """Same settings, independent random stream (keeps renders deterministic per layer)."""
        seed = int(self.rng.integers(0, 2**31 - 1))
        return Ctx(self.sr, self.n if n is None else n, np.random.default_rng(seed), self.c)


@dataclass
class Param:
    """One physical control: name, default, unit, realistic range, meaning, source."""

    name: str
    default: Any
    unit: str = ""
    lo: float | None = None
    hi: float | None = None
    doc: str = ""
    automatable: bool = False
    choices: Sequence[str] | None = None
    source: str = ""
    is_list: bool = False  # value is a list (e.g. formants), never automation

    def as_dict(self) -> dict:
        d = {"name": self.name, "default": self.default, "unit": self.unit, "doc": self.doc}
        if self.lo is not None or self.hi is not None:
            d["range"] = [self.lo, self.hi]
        if self.automatable:
            d["automatable"] = True
        if self.choices:
            d["choices"] = list(self.choices)
        if self.source:
            d["source"] = self.source
        return d


def is_automation(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (list, tuple))


def ctrl(ctx: Ctx, value: Any, curve: str = "linear") -> np.ndarray:
    """Turn a constant or a breakpoint list [[t, v], ...] into a per-sample array.

    curve="exp" interpolates in log space (good for frequency and RPM)."""
    if is_automation(value):
        pts = sorted((float(a), float(b)) for a, b in value)
        ts = np.array([p[0] for p in pts])
        vs = np.array([p[1] for p in pts])
        if curve == "exp" and np.all(vs > 0):
            return np.exp(np.interp(ctx.t, ts, np.log(vs)))
        return np.interp(ctx.t, ts, vs)
    return np.full(ctx.n, float(value))


def scalar(value: Any) -> float:
    """Single value of a control (first breakpoint value for automation)."""
    if is_automation(value):
        return float(value[0][1])
    return float(value)


@dataclass
class Model:
    name: str
    stage: str  # excitation | resonator | radiation | propagation | room | sound | synth | fx
    fn: Callable[..., np.ndarray]
    doc: str
    params: list[Param] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    needs_input: bool = False

    def resolve(self, given: dict | None) -> dict:
        given = dict(given or {})
        known = {p.name: p for p in self.params}
        unknown = set(given) - set(known)
        if unknown:
            raise RecipeError(f"{self.name}: unknown parameter(s) {sorted(unknown)}; known: {sorted(known)}")
        out = {}
        for name, p in known.items():
            v = given.get(name, p.default)
            if p.choices and not is_automation(v) and v not in p.choices:
                raise RecipeError(f"{self.name}.{name}: {v!r} not in {list(p.choices)}")
            if is_automation(v) and not p.automatable and not p.is_list:
                raise RecipeError(f"{self.name}.{name} cannot be automated")
            out[name] = v
        return out

    def describe(self) -> dict:
        return {
            "name": self.name,
            "stage": self.stage,
            "doc": self.doc,
            "needs_input": self.needs_input,
            "params": [p.as_dict() for p in self.params],
            "sources": self.sources,
        }


REGISTRY: dict[str, Model] = {}
STAGES = ("excitation", "resonator", "radiation", "propagation", "room", "sound", "synth", "fx")


def model(name: str, stage: str, doc: str, params: list[Param], sources: list[str] | None = None,
          needs_input: bool | None = None):
    """Decorator registering a block. Signature: fn(ctx, x, **params) -> np.ndarray."""
    if stage not in STAGES:
        raise ValueError(stage)

    def deco(fn):
        ni = needs_input if needs_input is not None else stage in ("resonator", "radiation", "propagation", "room", "fx")
        REGISTRY[name] = Model(name, stage, fn, doc, params, sources or [], ni)
        return fn

    return deco
