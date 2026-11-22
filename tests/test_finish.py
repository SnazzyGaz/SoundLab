"""Export finishing: BS.1770 loudness, true peak, per-category targets, tail trim, DC blocker, seamless loops."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import soundlab
from soundlab import finish as F
from soundlab.batch import batch_export, file_name

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "recipes"

# v0.34: the shipped {names} recipes use recording stems; the previous procedural versions live in recipes/impact_model/ (the same arrangement as recipes/creak_model/).
MOVED = {
    "footstep",
    "footstep_wood",
    "footstep_dirt",
    "footstep_grass",
    "footstep_gravel",
    "footstep_metal",
    "footstep_stone",
    "ceramic_break",
    "coin_jingle",
    "dowel",
    "drumstick",
    "glass_break",
    "glass_clink",
    "metal_clang",
    "pickaxe_earth",
    "pickaxe_rock",
    "sword_clash",
}


def recipe_path(name):
    name = name[:-5] if str(name).endswith(".json") else str(name)
    if name in MOVED:
        return R / "impact_model" / f"{name}.json"
    return R / f"{name}.json"

SR = 48000


def sine(f=997.0, amp=1.0, s=5.0, sr=SR):
    t = np.arange(int(s * sr)) / sr
    return amp * np.sin(2 * np.pi * f * t)


# ------------------------------------------------------------------------------------------- loudness
def test_bs1770_reference_tone():
    # BS.1770: a full-scale 997 Hz sine measures -3.01 LUFS; -20 dB amplitude -> -23.01
    assert F.integrated_lufs(sine(), SR) == pytest.approx(-3.01, abs=0.05)
    assert F.integrated_lufs(sine(amp=0.1), SR) == pytest.approx(-23.01, abs=0.05)
    assert F.momentary_max_lufs(sine(amp=0.1), SR) == pytest.approx(-23.01, abs=0.05)
    # other sample rates use the same weighting
    assert F.integrated_lufs(sine(sr=44100), 44100) == pytest.approx(-3.01, abs=0.1)


def test_k_weighting_shapes_spectrum():
    lo, hi = F.integrated_lufs(sine(f=40, amp=0.1), SR), F.integrated_lufs(sine(f=4000, amp=0.1), SR)
    assert hi - lo > 3          # +4 dB shelf above ~2 kHz, roll-off below 60 Hz


def test_gating_ignores_silence():
    x = np.concatenate([sine(amp=0.1, s=2), np.zeros(SR * 8)])
    assert F.integrated_lufs(x, SR) == pytest.approx(-23.01, abs=0.5)   # edge blocks count a little


def test_true_peak_finds_intersample_overs():
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * SR / 4 * t + np.pi / 4)   # samples at +-0.707, true peak 1.0
    assert 20 * np.log10(np.max(np.abs(x))) == pytest.approx(-3.0, abs=0.1)
    assert F.true_peak_db(x) == pytest.approx(0.0, abs=0.3)


def test_normalize_hits_target_and_respects_ceiling():
    y, info = F.normalize_loudness(sine(amp=0.01, s=1), SR, -20.0)
    assert F.momentary_max_lufs(y, SR) == pytest.approx(-20.0, abs=0.1) and not info["limited_by_peak"]
    # a spiky sound cannot reach -10 LUFS under -1 dBTP with gain alone: limiter (<= 6 dB), never over
    x = np.zeros(SR)
    x[1000:1040] = np.hanning(40)
    x += 0.01 * np.random.default_rng(0).standard_normal(SR)
    y, info = F.normalize_loudness(x, SR, -10.0)
    assert info["limited_by_peak"] and 0 < info["limiter_db"] <= 6.0
    assert F.true_peak_db(y) <= -1.0 + 0.05


def test_category_targets():
    assert F.target_for({"category": "ui"}) < F.target_for({"category": "impact"}) < F.target_for({"category": "weapon"})
    assert F.target_for({"category": "ambience"}, loop=True) == F.TARGETS["ambience"]
    assert F.target_for({"category": "weapon", "loudness_lufs": -30}) == -30
    assert F.target_for({"category": "nothing_known"}) == F.DEFAULT_TARGET


@pytest.mark.parametrize("name", ["coin", "door_slam", "glass_break", "creature_growl", "splash"])
def test_exports_land_on_their_category_target(name):
    raw = soundlab.load_recipe(recipe_path(name))
    y, sr = soundlab.render(raw)
    z, info = F.finish(y, sr, raw)
    assert F.true_peak_db(z) <= -0.95
    assert F.momentary_max_lufs(z, sr) == pytest.approx(F.target_for(raw), abs=1.0)


def test_spread_across_library_is_small(tmp_path):
    """Peak-normalised renders differ by ~20 dB in loudness; finished exports sit near their targets."""
    devs = []
    for name in ("coin", "match", "duck", "rain", "gunshot", "whoosh"):
        p = R / f"{name}.json" if (R / f"{name}.json").exists() else R / "blocks" / f"{name}.json"
        raw = soundlab.load_recipe(p)
        y, sr = soundlab.render(raw)
        z, info = F.finish(y, sr, raw)
        devs.append(info["result"] - info["target"])
    assert max(devs) - min(devs) < 3.5 and min(devs) > -3.0


# --------------------------------------------------------------------------------------------- trim
def test_trim_cuts_silent_tail_only():
    x = np.zeros(3 * SR)
    n = int(0.3 * SR)
    x[:n] = np.random.default_rng(1).standard_normal(n) * np.exp(-np.arange(n) / (0.03 * SR))
    y = F.trim_tail(x, SR)
    assert 0.1 < len(y) / SR < 0.6
    # what was cut is more than 60 dB under the loudest 10 ms
    assert np.sqrt(np.mean(x[len(y):] ** 2) + 1e-30) < np.sqrt(np.mean(x[:480] ** 2)) * 1e-3
    assert abs(y[-1]) < 1e-6            # faded to zero
    assert len(F.trim_tail(np.zeros(SR), SR)) == int(0.1 * SR)       # minimum length kept


def test_trim_keeps_stationary_sound():
    x = 0.3 * np.random.default_rng(2).standard_normal(2 * SR)
    assert len(F.trim_tail(x, SR)) == len(x)


@pytest.mark.parametrize("name,max_s", [("glass_hit_room", 1.2), ("sword_clash", 1.0), ("hit", 0.6),
                                         ("dowel", 0.7), ("door_slam", 0.8)])
def test_recipe_tails_are_trimmed(name, max_s):
    raw = soundlab.load_recipe(recipe_path(name))
    y, sr = soundlab.render(raw)
    z, info = F.finish(y, sr, raw)
    assert info["seconds"] < max_s < info["seconds_in"]


def test_trim_keeps_reverb_tail_down_to_minus_60():
    raw = soundlab.load_recipe(R / "glass_hit_room.json")
    y, sr = soundlab.render(raw)
    z = F.trim_tail(y, sr)
    w = int(0.01 * sr)
    env = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, mode="valid"))
    cut = len(z)
    assert env[cut:].max() < env.max() * 10 ** (-58 / 20)


# ------------------------------------------------------------------------------------------------ DC
def test_dc_blocker_on_master():
    y, sr = soundlab.render(R / "water.json")
    assert abs(np.mean(y)) < 1e-3          # was ~1 % before the blocker
    r = soundlab.load_recipe(R / "water.json")
    r["master"]["dc_block_hz"] = 0
    y0, _ = soundlab.render(r)
    assert abs(np.mean(y0)) > abs(np.mean(y))


# --------------------------------------------------------------------------------------------- loops
def test_fold_loop_is_seamless():
    rng = np.random.default_rng(3)
    from soundlab import dsp
    y = dsp.lowpass(rng.standard_normal(SR * 4), 2000.0, SR)
    naive = y[: SR * 3]
    lp = F.fold_loop(y, SR, 3.0, 0.5)
    assert len(lp) == 3 * SR
    assert F.loop_seam_jump(lp) < 3.0
    assert F.loop_seam_jump(naive) > F.loop_seam_jump(lp)
    # crossfade region keeps the level (equal power)
    assert np.std(lp[: SR // 2]) == pytest.approx(np.std(y[SR:2 * SR]), rel=0.3)


@pytest.mark.parametrize("name", ["rain", "wind", "fire", "engine_idle", "water"])
def test_loopable_recipes_loop(name):
    assert soundlab.is_loopable(R / f"{name}.json")
    y, sr = soundlab.render_loop(R / f"{name}.json", 4.0)
    assert len(y) == 4 * sr
    jump = abs(y[0] - y[-1])
    step = np.percentile(np.abs(np.diff(y)), 99)
    assert jump <= step                      # the wrap-around is no bigger than an ordinary sample step
    if name != "wind":                       # gusty wind swings ~10x in level on its own
        mid = np.std(y[sr:3 * sr])           # no fade-in/out at the ends
        assert np.std(y[: sr // 10]) > 0.3 * mid and np.std(y[-sr // 10:]) > 0.3 * mid


def test_one_shots_are_not_tagged_loopable():
    for name in ("gunshot", "hit", "coin", "door_slam"):
        assert not soundlab.is_loopable(R / f"{name}.json")


# ---------------------------------------------------------------------------------------- files, CLI
def test_file_name_no_doubled_category():
    assert file_name({"category": "footstep"}, "footstep", 0) == "footstep_v01.wav"
    assert file_name({"category": "impact"}, "hit", 2) == "impact_hit_v03.wav"


def test_render_to_file_defaults_and_peak_mode(tmp_path):
    a = soundlab.render_to_file(R / "hit.json", tmp_path / "a.wav")
    b = soundlab.render_to_file(R / "hit.json", tmp_path / "b.wav", level="peak", trim=False)
    assert a["seconds"] < b["seconds"] == pytest.approx(1.5)
    assert a["finish"]["level"] == "lufs" and "result" in a["finish"] and "result" not in b["finish"]


def test_batch_loop_and_names(tmp_path):
    man = batch_export([R / "rain.json", R / "hit.json"], tmp_path, loop=3, log=lambda *_: None)
    names = sorted(Path(m["wav"]).name for m in man)
    assert names == ["ambience_rain_loop3s_stereo_v01.wav", "impact_hit_v01.wav"]   # v0.22: beds are stereo
    rain = [m for m in man if "rain" in m["wav"]][0]
    assert rain["seconds"] == pytest.approx(3.0)


def test_cli_level_peak_and_loop(tmp_path):
    out = tmp_path / "r.wav"
    r = subprocess.run([sys.executable, "-m", "soundlab", "render", str(R / "wind.json"), "-o", str(out),
                        "--loop", "2"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout)
    assert info["seconds"] == pytest.approx(2.0) and info["finish"]["measured_by"] == "integrated"
    r = subprocess.run([sys.executable, "-m", "soundlab", "render", str(R / "hit.json"), "-o", str(out),
                        "--level", "peak", "--no-trim"], capture_output=True, text=True, cwd=ROOT)
    info = json.loads(r.stdout)
    assert info["seconds"] == pytest.approx(1.5) and info["finish"]["level"] == "peak"
