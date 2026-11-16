"""v0.7: macro sliders, seeded variations, batch export and naming."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import soundlab
from soundlab.batch import batch_export, file_name, parse_sweep
from soundlab.macros import apply_macros, macro_defaults, variation

ROOT = Path(__file__).resolve().parents[1]
RECIPES = sorted((ROOT / "recipes").glob("*.json")) + sorted((ROOT / "recipes" / "blocks").glob("*.json"))


def _strip(r):
    r = copy.deepcopy(r)
    r.pop("macros", None)
    r.pop("vary", None)
    return r


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: p.name)
def test_every_recipe_has_macros_and_is_unchanged_at_defaults(path):
    rec = soundlab.load_recipe(str(path))
    assert rec.get("macros"), "every shipped recipe declares macro sliders"
    assert rec.get("category"), "category is used for file names"
    assert soundlab.validate(rec) == []
    at_default = apply_macros(rec, macro_defaults(rec))
    at_default.pop("vary", None)
    assert at_default == _strip(rec), "macros at their defaults must leave the recipe exactly as written"


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: p.name)
def test_macro_extremes_render(path):
    rec = soundlab.load_recipe(str(path))
    for v in (0.0, 1.0):
        mv = {k: v for k, s in rec["macros"].items() if "choices" not in s}
        y, sr = soundlab.render(rec, sample_rate=16000, macros=mv)
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0.1


def test_macro_mapping_mul_add_clamp_choice():
    rec = soundlab.load_recipe(str(ROOT / "recipes" / "hit.json"))
    big = apply_macros(rec, {"size": 1.0})
    assert big["layers"][0]["params"]["length"] == pytest.approx(0.45 * 2.5)
    near = apply_macros(rec, {"distance": 0.0})
    assert near.get("master", {}).get("gain_db", 0.0) == 0.0  # clamped: never louder than normalised
    far = apply_macros(rec, {"distance": 1.0})
    assert far["master"]["gain_db"] == pytest.approx(-12.0)
    metal = apply_macros(rec, {"material": "metal"})
    assert metal["layers"][0]["params"]["material"] == "steel"
    with pytest.raises(soundlab.RecipeError):
        apply_macros(rec, {"material": "jelly"})
    with pytest.raises(soundlab.RecipeError):
        apply_macros(rec, {"nonsense": 0.3})


def test_automation_is_scaled_and_stretched():
    eng = soundlab.load_recipe(str(ROOT / "recipes" / "engine.json"))
    hi = apply_macros(eng, {"rpm": 1.0})
    a = [v for _, v in eng["layers"][0]["params"]["rpm"]]
    b = [v for _, v in hi["layers"][0]["params"]["rpm"]]
    assert b[0] == pytest.approx(a[0] * 1.6)
    coin = soundlab.load_recipe(str(ROOT / "recipes" / "blocks" / "coin_pickup.json"))
    long_ = apply_macros(coin, {"length": 1.0})
    t0 = [t for t, _ in coin["layers"][0]["chain"][2]["params"]["points"]]
    t1 = [t for t, _ in long_["layers"][0]["chain"][2]["params"]["points"]]
    assert t1[-1] == pytest.approx(t0[-1] * 2)


def test_variations_are_deterministic_and_different():
    rec = soundlab.load_recipe(str(ROOT / "recipes" / "hit.json"))
    y0, _ = soundlab.render(rec, 16000)
    y0b, _ = soundlab.render(rec, 16000, variation=0)
    assert np.array_equal(y0, y0b)
    y3, _ = soundlab.render(rec, 16000, variation=3)
    y3b, _ = soundlab.render(rec, 16000, variation=3)
    y4, _ = soundlab.render(rec, 16000, variation=4)
    assert np.array_equal(y3, y3b)
    assert not np.allclose(y3, y4) and not np.allclose(y0, y3)
    r3, vals, seed = variation(rec, 3)
    assert seed != rec.get("seed", 0)
    assert all(abs(vals[k] - 0.5) <= 0.04 + 1e-9 for k in ("size", "intensity"))
    v = r3["layers"][0]["params"]["velocity"]
    assert 2.0 * 0.8 * 0.85 <= v <= 2.0 * 1.2 * 1.15  # vary rel 0.2 on top of the intensity jitter


def test_batch_naming_sweep_and_manifest(tmp_path):
    out = batch_export([ROOT / "recipes" / "hit.json"], tmp_path, variations=2,
                       sweep=parse_sweep(["material=wood,metal"]), sample_rate=16000, log=lambda m: None)
    names = sorted(Path(m["wav"]).name for m in out)
    assert names == ["impact_hit_materialmetal_v01.wav", "impact_hit_materialmetal_v02.wav",
                     "impact_hit_materialwood_v01.wav", "impact_hit_materialwood_v02.wav"]
    assert json.loads((tmp_path / "manifest.json").read_text())[0]["recipe"].endswith("hit.json")
    assert file_name({"category": "UI"}, "Click Big", 9, {"size": 0.8}) == "ui_click-big_size0.8_v10.wav"


def test_cli_describe_shows_macros_first():
    r = subprocess.run([sys.executable, "-m", "soundlab", "describe", str(ROOT / "recipes" / "engine.json")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    assert r.stdout.index("MACROS") < r.stdout.index("LAYERS") and "rpm" in r.stdout


def test_fit_ear_renders_use_macros(tmp_path):
    from soundlab import fit as F
    rec = soundlab.load_recipe(str(ROOT / "recipes" / "blocks" / "ui_click.json"))
    ref, sr = soundlab.render(rec, 24000, macros={"pitch": 0.7})
    res = F.fit(rec, ref, sr, ["0.1.decay:0.005:0.1"], budget=8, log=lambda m: None)
    files = [Path(f).name for f in F.ear_renders(res, tmp_path, sr=16000)]
    assert "fitted.wav" in files and "pitch0.15.wav" in files and "pitch0.85.wav" in files
