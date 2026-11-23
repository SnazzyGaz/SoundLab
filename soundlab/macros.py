"""Macro sliders and seeded variations (v0.7, roadmap steps 2-3).

Macros are the primary interface: a recipe declares a few named sliders (0-1, or a named choice such as a
material) and how each one moves the underlying controls. Mappings are RELATIVE to the recipe's own values
and anchored at the macro default, so at the default every recipe renders exactly as written (fits stay
valid: `soundlab match` tunes the underlying values, macros move around them).

    "macros": {
      "size": {"default": 0.5, "doc": "small charge .. huge blast", "jitter": 0.04,
               "map": [{"to": "0.size", "op": "add", "range": [-0.35, 0.05]},
                       {"to": "0.fireball_s", "op": "mul", "range": [0.3, 1.3]},
                       {"to": "duration", "op": "mul", "range": [0.5, 1.0]}]},
      "material": {"default": "wood", "choices": {
               "wood":  [{"to": "0.material", "op": "set", "value": "wood"}],
               "steel": [{"to": "0.material", "op": "set", "value": "steel"},
                         {"to": "0.extra_damping", "op": "mul", "value": 0.3}]}}
    }

Ops (m = macro value, m0 = its default; optional "curve": m -> m**curve first):
  mul      value * F(m) / F(m0), F interpolating the factors in `range` exponentially (log-even)
  add      value + L(m) - L(m0), L interpolating `range` linearly (dB, cents, seconds, plain units)
  stretch  multiplies breakpoint TIMES (or a plain time) by F(m) / F(m0): longer/shorter gestures
  set      absolute: lo + (hi - lo) * m (or exponential with "interp": "exp"); with "points", a piecewise
           numeric value or a stepwise choice (last point whose macro value is <= m); for choices, "value"
Instead of "range", "points": [[0, 0.5], [0.5, 1], [1, 3]] gives a piecewise mapping (factors for
mul/stretch, interpolated in log; offsets for add), again anchored at the default. "clamp": [lo, hi] limits
the result (e.g. master.gain_db never above 0).
Choice macros apply their list of {to, op, value} (mul/add/set with a fixed value); the default choice must
leave the recipe as written (a test checks every shipped recipe).

Targets ("to"): "L.param" (first block of layer L; L = index or layer name), "L.B.param" (block B of a chain),
"L.gain_db", "L.start", "post.J.param", "duration", "master.gain_db" (level after peak normalising, so a
"distance" macro can make a sound quieter). Automation (breakpoint lists) is mapped point by point (values,
not times). Results are clipped to the control's declared range.

Display (the app only; nothing here changes the sound): "label" is the plain name shown for the macro;
"ui" is "front" (default), "advanced" (under Advanced) or "hidden" (kept so older calls and saved recipes
still work, not shown). A numeric macro may name positions with "options": {"wood": 0.0, ...} (a slider
value) or {"creaky wood": [{to, op, value}, ...]} (its own targets, like a choice); the app shows these as a
dropdown and sends the name, and "default_label" names the option the default stands for. Numeric values
still work as before.

Variations: "vary": {"0.freq": {"cents": 30}, "0.velocity": {"rel": 0.15}, "1.gain_db": {"db": 2},
"1.start": {"abs": 0.01}} gives uniform random offsets per control; every numeric macro also jitters by its
"jitter" (default 0.04 of its range); and each variation renders with its own seed so the stochastic parts
differ. Variation k of a recipe with seed s is fully determined by (s, k); k = 0 is the recipe itself.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .core import REGISTRY, RecipeError, is_automation

DEFAULT_JITTER = 0.04


# ----------------------------------------------------------------------------- target access
def _layer_index(recipe, key):
    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
        return int(key)
    for i, layer in enumerate(recipe["layers"]):
        if layer.get("name") == key:
            return i
    raise RecipeError(f"macro target: no layer named {key!r}")


def _resolve(recipe, to: str):
    """Return (container dict, key, Param or None) for a target path."""
    parts = to.split(".")
    if to == "duration":
        return recipe, "duration", None
    if parts[0] == "master":
        return recipe.setdefault("master", {}), parts[1], None
    if parts[0] == "post":
        b = recipe.get("post", [])[int(parts[1])]
        m = REGISTRY[b["model"]]
        return b.setdefault("params", {}), parts[2], next((p for p in m.params if p.name == parts[2]), None)
    li = _layer_index(recipe, parts[0])
    layer = recipe["layers"][li]
    if len(parts) == 2 and parts[1] in ("gain_db", "start"):
        return layer, parts[1], None
    bi = int(parts[1]) if len(parts) == 3 else 0
    name = parts[-1]
    b = (layer.get("chain") or [layer])[bi]
    m = REGISTRY.get(b["model"])
    if m is None:
        raise RecipeError(f"macro target {to}: unknown model {b['model']!r}")
    p = next((q for q in m.params if q.name == name), None)
    if p is None:
        raise RecipeError(f"macro target {to}: {b['model']} has no control {name!r}")
    return b.setdefault("params", {}), name, p


def _current(cont, key, p):
    if key in cont:
        return cont[key]
    if p is not None:
        return p.default
    return {"duration": 3.0, "gain_db": 0.0, "start": 0.0}.get(key, 0.0)


def _clip(v, p):
    if p is None or isinstance(v, (str, bool)):
        return v
    if p.lo is not None:
        v = max(v, p.lo)
    if p.hi is not None:
        v = min(v, p.hi)
    if isinstance(p.default, int) and not isinstance(p.default, bool):
        v = int(round(v))
    return v


def _map_value(v, fn, p):
    if is_automation(v):
        return [[t, _clip(fn(float(x)), p)] for t, x in v]
    if isinstance(v, (str, bool)) or v is None:
        return v
    return _clip(fn(float(v)), p)


def _set_target(recipe, to, fn=None, value=None, clamp=None, stretch=None):
    cont, key, p = _resolve(recipe, to)
    if stretch is not None:  # scale breakpoint TIMES (or a plain time value)
        v = _current(cont, key, p)
        cont[key] = [[float(t) * stretch, x] for t, x in v] if is_automation(v) else _clip(float(v) * stretch, p)
        return
    if fn is None:
        cont[key] = value
        return
    if clamp is not None:
        lo, hi = clamp
        f0 = fn
        fn = lambda v, f0=f0: min(max(f0(v), lo), hi)  # noqa: E731
    cont[key] = _map_value(_current(cont, key, p), fn, p)


# ----------------------------------------------------------------------------- macros
def _interp(rng, m, kind):
    lo, hi = float(rng[0]), float(rng[1])
    if kind == "exp":
        if lo <= 0 or hi <= 0:
            raise RecipeError(f"exponential mapping needs positive range, got {rng}")
        return lo * (hi / lo) ** m
    return lo + (hi - lo) * m


def macro_specs(recipe) -> dict:
    return recipe.get("macros") or {}


def macro_defaults(recipe) -> dict:
    return {k: v.get("default", 0.5) for k, v in macro_specs(recipe).items()}


def _apply_targets(r, targets):
    for t in targets:
        op, x = t.get("op", "set"), t.get("value")
        if op == "set":
            _set_target(r, t["to"], value=x)
        elif op == "mul":
            _set_target(r, t["to"], fn=lambda v, x=x: v * float(x))
        elif op == "add":
            _set_target(r, t["to"], fn=lambda v, x=x: v + float(x))


def _apply_one(r, name, spec, val):
    if isinstance(val, str) and "options" in spec and "choices" not in spec:
        # a slider with named positions (a dropdown in the app): a name is either a slider value or its own targets
        if val not in spec["options"]:
            raise RecipeError(f"macro {name}: {val!r} not in {list(spec['options'])}")
        o = spec["options"][val]
        if isinstance(o, list):
            _apply_targets(r, o)
            return
        val = o
    if "choices" in spec:
        if val not in spec["choices"]:
            raise RecipeError(f"macro {name}: {val!r} not in {list(spec['choices'])}")
        _apply_targets(r, spec["choices"][val])
        return
    m = float(val)
    if not 0.0 <= m <= 1.0:
        raise RecipeError(f"macro {name}={m}: macros run 0..1")
    m0 = float(spec.get("default", 0.5))
    for t in spec.get("map", []):
        c = float(t.get("curve", 1.0))
        mm, mm0 = m ** c, m0 ** c
        op = t.get("op", "mul")
        clamp = t.get("clamp")
        if "points" in t:  # piecewise mapping [[macro, factor_or_offset], ...], anchored at the default
            xs = [float(a) for a, _ in t["points"]]
            raw_ys = [b for _, b in t["points"]]
            numeric = all(isinstance(b, (int, float)) and not isinstance(b, bool) for b in raw_ys)
            if op in ("mul", "stretch") and numeric:
                ys = [float(b) for b in raw_ys]
                ly = np.log(ys)
                f = float(np.exp(np.interp(m, xs, ly) - np.interp(m0, xs, ly)))
                if op == "mul":
                    _set_target(r, t["to"], fn=lambda v, f=f: v * f, clamp=clamp)
                else:
                    _set_target(r, t["to"], stretch=f)
            elif op == "add" and numeric:
                ys = [float(b) for b in raw_ys]
                d = float(np.interp(m, xs, ys) - np.interp(m0, xs, ys))
                _set_target(r, t["to"], fn=lambda v, d=d: v + d, clamp=clamp)
            elif op == "set":
                if numeric:
                    x = float(np.interp(m, xs, [float(b) for b in raw_ys]))
                    _set_target(r, t["to"], fn=lambda v, x=x: x, clamp=clamp)
                else:
                    # stepwise (source names, materials, ...): last point whose x <= m
                    pick = raw_ys[0]
                    for a, b in zip(xs, raw_ys):
                        if m + 1e-12 >= a:
                            pick = b
                    _set_target(r, t["to"], value=pick)
            else:
                raise RecipeError(f"macro {name}: 'points' works with mul/add/stretch/set, not {op!r}")
            continue
        if op == "stretch":
            f = _interp(t["range"], mm, "exp") / _interp(t["range"], mm0, "exp")
            _set_target(r, t["to"], stretch=f)
        elif op == "mul":
            f = _interp(t["range"], mm, "exp") / _interp(t["range"], mm0, "exp")
            _set_target(r, t["to"], fn=lambda v, f=f: v * f, clamp=clamp)
        elif op == "add":
            d = _interp(t["range"], mm, "lin") - _interp(t["range"], mm0, "lin")
            _set_target(r, t["to"], fn=lambda v, d=d: v + d, clamp=clamp)
        elif op == "set":
            x = _interp(t["range"], mm, t.get("interp", "lin"))
            _set_target(r, t["to"], fn=lambda v, x=x: x)
        else:
            raise RecipeError(f"macro {name}: unknown op {op!r}")


def apply_macros(recipe: dict, values: dict | None = None) -> dict:
    """Return a copy of `recipe` with macro values applied (defaults for the ones not given).
    The result has no 'macros' key left, so applying twice is impossible."""
    r = copy.deepcopy(recipe)
    specs = r.pop("macros", None) or {}
    values = dict(values or {})
    unknown = set(values) - set(specs)
    if unknown:
        raise RecipeError(f"unknown macro(s) {sorted(unknown)}; this recipe has {sorted(specs)}")
    for name, spec in specs.items():
        val = values.get(name, spec.get("default", 0.5))
        if val == spec.get("default", 0.5):
            # anchored mappings are identity at the default, and the default choice must leave the recipe as
            # written (tested), so skip it: then `soundlab match` can move the values a default choice names
            continue
        _apply_one(r, name, spec, val)
    return r


def parse_macro_args(items) -> dict:
    """['size=0.8', 'material=steel'] -> {'size': 0.8, 'material': 'steel'}."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            out[k.strip()] = v.strip()
    return out


# ----------------------------------------------------------------------------- variations
def variation(recipe: dict, k: int, macros: dict | None = None) -> tuple[dict, dict, int]:
    """Variation k (k >= 1) of a recipe: jittered macro values, jittered 'vary' controls, its own seed.
    Returns (recipe with macros applied and vary applied, the macro values used, seed). k = 0: unchanged."""
    base_seed = int(recipe.get("seed", 0))
    specs = macro_specs(recipe)
    vals = dict(macro_defaults(recipe), **(macros or {}))
    if k <= 0:
        r = apply_macros(recipe, vals)
        r.pop("vary", None)
        return r, vals, base_seed
    rng = np.random.default_rng([base_seed, int(k), 7919])
    for name, spec in specs.items():
        if "choices" in spec:
            continue
        j = float(spec.get("jitter", DEFAULT_JITTER))
        u = rng.uniform(-j, j)            # drawn for named positions too, so later draws stay where they were
        if isinstance(vals[name], str):
            continue
        vals[name] = float(np.clip(float(vals[name]) + u, 0.0, 1.0))
    r = apply_macros(recipe, vals)
    for to, spec in (r.pop("vary", None) or {}).items():
        u = rng.uniform(-1.0, 1.0)
        if "cents" in spec:
            f = 2 ** (u * float(spec["cents"]) / 1200)
            _set_target(r, to, fn=lambda v, f=f: v * f)
        elif "rel" in spec:
            f = 1 + u * float(spec["rel"])
            _set_target(r, to, fn=lambda v, f=f: v * f)
        elif "db" in spec:
            d = u * float(spec["db"])
            _set_target(r, to, fn=lambda v, d=d: v + d)
        elif "abs" in spec:
            d = u * float(spec["abs"])
            _set_target(r, to, fn=lambda v, d=d: v + d)
        else:
            raise RecipeError(f"vary {to}: give one of cents/rel/db/abs")
    seed = base_seed + 1000 * int(k)
    return r, vals, seed


def validate_macros(recipe: dict) -> list[str]:
    errs = []
    specs = macro_specs(recipe)
    for name, spec in specs.items():
        try:
            if "choices" in spec:
                if spec.get("default") not in spec["choices"]:
                    errs.append(f"macro {name}: default {spec.get('default')!r} is not one of its choices")
                for c in spec["choices"]:
                    apply_macros(recipe, {name: c})
            else:
                for m in (0.0, 1.0):
                    apply_macros(recipe, {name: m})
                for o in spec.get("options") or {}:
                    apply_macros(recipe, {name: o})
            if spec.get("ui") not in (None, "front", "advanced", "hidden"):
                errs.append(f"macro {name}: ui must be front, advanced or hidden")
        except (RecipeError, KeyError, IndexError, ValueError, TypeError) as e:
            errs.append(f"macro {name}: {e}")
    for to, spec in (recipe.get("vary") or {}).items():
        try:
            _resolve(copy.deepcopy(recipe), to)
        except (RecipeError, KeyError, IndexError, ValueError) as e:
            errs.append(f"vary {to}: {e}")
    return errs


def describe_recipe(recipe: dict) -> dict:
    out = {"name": recipe.get("name", ""), "category": recipe.get("category", "sfx"),
           "duration": recipe.get("duration"), "macros": [], "layers": [], "vary": recipe.get("vary", {})}
    for name, spec in macro_specs(recipe).items():
        d = {"name": name, "default": spec.get("default", 0.5), "doc": spec.get("doc", ""),
             "label": spec.get("label", name), "ui": spec.get("ui", "front")}
        if "choices" in spec:
            d["choices"] = list(spec["choices"])
        else:
            if "options" in spec:
                d["options"] = list(spec["options"])
            d["jitter"] = spec.get("jitter", DEFAULT_JITTER)
            d["moves"] = [f"{t['to']} ({t.get('op', 'mul')} " + (", ".join(f"{a:g}:{b:g}" for a, b in t["points"])
                          if "points" in t else str(t.get("range"))) + ")" for t in spec.get("map", [])]
        out["macros"].append(d)
    for i, layer in enumerate(recipe["layers"]):
        out["layers"].append({"index": i, "name": layer.get("name", f"layer{i}"),
                              "blocks": [b["model"] for b in (layer.get("chain") or [layer])]})
    out["post"] = [b["model"] for b in recipe.get("post", [])]
    return out
