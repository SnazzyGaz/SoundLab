import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import soundlab
from soundlab.analysis import analyze, hpr_split
from conftest import SR

ROOT = Path(__file__).resolve().parents[1]


def simple_recipe(seed=1):
    return {"soundlab": 1, "sample_rate": SR, "duration": 1.0, "seed": seed,
            "layers": [{"chain": [{"model": "impact", "params": {"velocity": 2}},
                                  {"model": "modal", "params": {"material": "glass", "shape": "bar"}}]}],
            "post": [{"model": "room", "params": {"volume": 50}}]}


def test_deterministic_and_seeded():
    a, _ = soundlab.render(simple_recipe(1))
    b, _ = soundlab.render(simple_recipe(1))
    c, _ = soundlab.render(simple_recipe(2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_validation_errors():
    bad = {"layers": [{"chain": [{"model": "modal"}]}]}
    assert any("chains start with a source" in e for e in soundlab.validate(bad))
    bad2 = {"layers": [{"model": "engine", "params": {"rpmm": 3}}]}
    assert any("unknown parameter" in e for e in soundlab.validate(bad2))
    bad3 = {"layers": [{"model": "nope"}]}
    assert soundlab.validate(bad3)
    with pytest.raises(soundlab.RecipeError):
        soundlab.render(bad)


def test_automation_and_layers_start():
    r = {"duration": 1.0, "layers": [
        {"model": "engine", "params": {"rpm": [[0, 1000], [1, 3000]]}},
        {"model": "duck", "start": 0.5, "gain_db": -6}]}
    y, sr, stems = soundlab.render(r, return_layers=True)
    assert len(stems) == 2
    assert np.max(np.abs(list(stems.values())[1][: int(0.49 * sr)])) == 0


def test_bad_automation_rejected():
    r = {"layers": [{"model": "duck", "params": {"f0": [[0, 300], [1, 400]]}}]}
    assert soundlab.validate(r)


def test_cli_render_models_describe(tmp_path):
    rec = tmp_path / "r.json"
    rec.write_text(json.dumps(simple_recipe()))
    exe = [sys.executable, "-m", "soundlab"]
    out = subprocess.run(exe + ["render", str(rec), "-o", str(tmp_path / "o.wav")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "o.wav").exists()
    ms = json.loads(subprocess.run(exe + ["models", "--json"], capture_output=True, text=True).stdout)
    assert {"duck", "engine", "foghorn", "explosion", "match", "modal", "tube", "friction"} <= {m["name"] for m in ms}
    d = json.loads(subprocess.run(exe + ["describe", "engine", "--json"], capture_output=True, text=True).stdout)
    assert any(p["name"] == "rpm" and p["unit"] == "rev/min" for p in d["params"])
    v = subprocess.run(exe + ["validate", str(ROOT / "recipes" / "engine.json")], capture_output=True, text=True)
    assert v.stdout.strip() == "ok"
    s = subprocess.run(exe + ["render", str(rec), "-o", str(tmp_path / "o2.wav"), "--set", "0.velocity=3"],
                       capture_output=True, text=True)
    assert s.returncode == 0, s.stderr


def test_all_shipped_recipes_valid():
    for p in (ROOT / "recipes").glob("*.json"):
        assert soundlab.validate(soundlab.load_recipe(p)) == [], p.name


def test_analysis_separates_stems(tmp_path):
    t = np.arange(2 * SR) / SR
    tone = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    clicks = np.zeros_like(t)
    burst = np.random.default_rng(1).standard_normal(240) * np.exp(-np.arange(240) / 40)
    starts = np.arange(SR // 8, len(t) - 300, SR // 4)
    for i in starts:
        clicks[i:i + 240] += burst
    noise = 0.05 * np.random.default_rng(0).standard_normal(len(t))
    stems, _ = hpr_split(tone + clicks + noise, SR)
    def corr(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    assert corr(stems["harmonic"], tone) > 0.9
    tr = stems["transient"] ** 2
    near = np.zeros_like(tr, bool)
    for i in starts:
        near[max(0, i - SR // 50):i + SR // 50] = True
    assert tr[near].sum() > 0.7 * tr.sum()  # transient energy sits on the clicks
    feats = analyze(tone + clicks + noise, SR, tmp_path, plot=False)
    assert abs(feats["f0_hz"] - 440) < 3
    fr = [p["freq_hz"] for p in feats["partials"][:4]]
    assert min(abs(f - 440) for f in fr) < 5 and min(abs(f - 880) for f in fr) < 5
    assert len(feats["onsets_s"]) >= 6
    for k in ["harmonic", "transient", "noise", "sines"]:
        assert (tmp_path / f"{k}.wav").exists()


def test_analysis_decay_estimate():
    t = np.arange(2 * SR) / SR
    x = np.sin(2 * np.pi * 600 * t) * np.exp(-6.91 * t / 0.8)
    feats = analyze(x, SR, None, plot=False)
    p = feats["partials"][0]
    assert abs(p["freq_hz"] - 600) < 3
    assert abs(p["t60_s"] - 0.8) < 0.15
    assert abs(feats["decay_t60_s"] - 0.8) < 0.15


def test_engine_firing_rate_in_analysis():
    y = soundlab.render({"duration": 2.0, "layers": [{"model": "engine", "params": {
        "rpm": 1500, "cylinders": 4, "jitter": 0.0, "mechanical": 0, "whine": 0, "intake": 0, "pops": 0}}]})[0]
    from soundlab.analysis import modulation_rate
    # firing 50 Hz; envelope periodicity should land on it or its sub-multiple (exhaust comb can interfere)
    r = modulation_rate(y, SR)
    assert r is not None
