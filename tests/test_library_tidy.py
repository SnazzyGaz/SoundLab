"""The library: plain names in a Domain/Group tree, hidden aliases for merged or replaced ids (audio unchanged),
hubs whose choices equal the old separate recipes, and the display keys on macros."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import soundlab
from soundlab.core import RecipeError
from soundlab.macros import apply_macros, variation

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "recipes"
TOPS = {"Foley", "Impacts & breaks", "Combat", "Fire", "Water", "Weather", "Machines", "Vehicles", "Animals",
        "Creatures", "Game UI & sci-fi"}


def all_recipes():
    out = {}
    for p in sorted(R.rglob("*.json")):
        rel = p.relative_to(R).with_suffix("").as_posix()
        if not rel.startswith("fit/"):
            out[rel] = json.loads(p.read_text())
    return out


def raw(name, **kw):
    y, sr = soundlab.render(str(R / f"{name}.json"), **kw)
    return np.asarray(y)


def test_every_recipe_is_listed_or_a_hidden_alias():
    rs = all_recipes()
    visible = {k for k, r in rs.items() if not r.get("hidden")}
    for k, r in rs.items():
        if r.get("hidden"):
            tgt = r.get("replaced_by")
            assert tgt is None or tgt in visible, (k, tgt)
        else:
            assert r.get("title") and r.get("library"), k
            assert r["library"].split("/")[0] in TOPS, (k, r["library"])
            assert "_" not in r["title"], k
    titles = [(r["library"], r["title"]) for k, r in rs.items() if not r.get("hidden")]
    assert len(titles) == len(set(titles))


def test_macro_display_keys_are_valid():
    for k, r in all_recipes().items():
        for name, m in (r.get("macros") or {}).items():
            assert m.get("ui", "front") in ("front", "advanced", "hidden"), (k, name)
            if name == "seed":
                assert m.get("ui") == "hidden", k


@pytest.mark.parametrize("name", ["coin_drop", "wood_knock", "door_creak", "splash_small", "power_drill_on_off"])
def test_hidden_aliases_still_render(name):
    y = raw(name)
    assert len(y) > 1000 and np.max(np.abs(y)) > 1e-4


@pytest.mark.parametrize("hub,mac,old,kw", [
    ("splash_large", {"splash": "small"}, "splash_small", {}),
    ("splash_large", {"splash": "medium", "size": 0.8}, "splash_medium", {"macros": {"size": 0.8}}),
    ("foley/coin_drop", {"shape": "spin"}, "coin_drop", {}),
    ("footstep", {"surface": "carpet"}, "carpet_walk", {}),
    ("footstep", {"shoe": "trainer"}, "trainers_walk", {}),
])
def test_hub_choice_equals_the_old_recipe(hub, mac, old, kw):
    a = raw(hub, macros=mac)
    b = raw(old, **kw)
    assert a.shape == b.shape and np.array_equal(a, b)


def test_hub_variation_equals_the_old_recipe_variation():
    a = raw("splash_large", macros={"splash": "small"}, variation=3)
    b = raw("splash_small", variation=3)
    assert np.array_equal(a, b)


def test_machine_shape_on_off_equals_the_old_recipe():
    a = raw("servo_motor", macros={"shape": "on-off"})
    b = raw("servo_motor_on_off")
    assert np.array_equal(a, b)


def test_named_options_and_old_numbers_both_work():
    r = json.loads((R / "footstep.json").read_text())
    rr = apply_macros(r, {"surface": "gravel"})
    assert rr["layers"][0]["params"]["surface"] == "gravel"
    rr = apply_macros(r, {"surface": 0.32})            # the old slider value for gravel
    assert rr["layers"][0]["params"]["surface"] == "gravel"
    with pytest.raises(RecipeError):
        apply_macros(r, {"surface": "lava"})
    v, vals, _ = variation(r, 2, {"surface": "snow"})
    assert vals["surface"] == "snow" and v["layers"][0]["params"]["surface"] == "snow"


def test_pace_run_uses_the_run_recording():
    from soundlab.recipe import render
    a = raw("footstep", macros={"surface": "concrete", "pace": "run"})
    r = json.loads((R / "concrete_run.json").read_text())
    r["duration"] = 1.99                    # the hub keeps the surface's buffer length; the step itself is the same
    b, _ = render(r)
    assert np.array_equal(a, np.asarray(b))


def test_explosion_has_no_long_tail():
    y, sr = soundlab.render(str(R / "explosion.json"))
    y = np.asarray(y, float)
    assert len(y) < 9 * sr
    fr = int(0.05 * sr)
    k = len(y) // fr
    e = 10 * np.log10(np.mean(y[:k * fr].reshape(k, fr) ** 2, 1) + 1e-20)
    assert e[-1] - e.max() < -45
