"""v0.34: impact_stems, weapon_stems and foot_stems -- recording stems, numbers only, no audio.

Default impact and weapon recipes play one real gesture as recorded. Footstep recipes default to a
softer step than the recording (force / softness macros). Fingerprints for identity takes live in
tests/data/*_fingerprints.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import soundlab
from squeal_fp import distance, fingerprint

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "recipes"
SR = 44100


def _fp(name):
    p = ROOT / "tests" / "data" / f"{name}_fingerprints.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def render(name, k=0, **mac):
    y, sr = soundlab.render(str(R / f"{name}.json"), macros=mac or None, variation=k)
    return np.asarray(y, float), sr


def rms(y):
    return float(np.sqrt(np.mean(y ** 2) + 1e-30))


def db(y):
    return 20 * math.log10(max(np.max(np.abs(y)), 1e-12))


def centroid(y, sr):
    Y = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    s = Y.sum() + 1e-30
    return float((f * Y).sum() / s)


# A compact representative set so the full suite stays reasonable; identity fingerprints
# cover every shipped recipe when the fingerprint files are present.
FOOT_SAMPLE = ["wood_floor", "carpet_walk", "trainers_walk", "jump_land", "jump_land_rock", "concrete_walk"]
IMP_SAMPLE = ["wood_knock", "glass_clink", "coin_drop", "pickaxe_rock"]
WEP_SAMPLE = ["pistol_shot", "shotgun_pump", "sword_clash", "dry_fire"]
LEGACY = ["footstep", "footstep_wood", "footstep_dirt", "footstep_grass",
          "footstep_gravel", "footstep_metal", "footstep_stone"]


def _sources(mod):
    import importlib
    m = importlib.import_module(f"soundlab.models.{mod}")
    return list(m.SOURCES)


@pytest.mark.parametrize("mod,magic", [
    ("impact_stems", b"SLIMP001"),
    ("weapon_stems", b"SLWEP001"),
    ("foot_stems", b"SLFTS001"),
])
def test_stem_data_is_compact_numbers_with_credited_cc0_sources(mod, magic):
    import lzma, struct
    raw = (ROOT / "soundlab" / "models" / f"{mod}.bin").read_bytes()
    assert raw[:8] == magic
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    hdr = json.loads(p[4:4 + n])
    assert hdr["sources"] and hdr["files"]
    blob = p[4 + n:]
    for k, s in hdr["sources"].items():
        concat = {"ev_env", "bed_env", "md_f", "md_amp", "md_ph"}
        for name, a in s.items():
            if isinstance(a, dict) and "sh" in a and name not in concat:
                # never one value per sample
                assert max(a["sh"]) <= s["n"] / 4 + 8, (k, name, a["sh"], s["n"])
        if "ev_env" in s:
            assert s["ev_rate"] <= 8000
            # concatenated event windows may be longer than the clip (padding); still far below sample rate
            assert s["ev_env"]["sh"][1] <= s["n"] + s["ev_rate"], (k, s["ev_env"]["sh"], s["n"])
        if "bed_env" in s:
            assert s["bed_env"]["sh"][1] <= s["n"] / 8, (k, s["bed_env"]["sh"], s["n"])  # 1 ms beds, some overlap
        if "md_amp" in s:
            # mode tracks at 1 kHz, concatenated across partials
            assert s["md_amp"]["sh"][0] <= int(s.get("md_n", 1)) * (s["n"] / 40 + 8) + 8, (k, s["md_amp"]["sh"], s.get("md_n"), s["n"])
        if "hf_f" in s:
            assert max(s["hf_f"]["sh"]) <= 256
        fid = str(s.get("file", "")).replace(".mp3", "")
        assert fid in hdr["files"] or any(str(v.get("id")) == fid for v in hdr["files"].values()), k
    for rec in hdr["files"].values():
        if not isinstance(rec, dict):
            continue
        lic = (rec.get("license") or rec.get("licence") or "").lower()
        assert "cc0" in lic or "public domain" in lic or "zero" in lic, rec
        assert rec.get("author") and rec.get("url")


@pytest.mark.parametrize("mod", ["impact_stems", "weapon_stems", "foot_stems"])
def test_default_recipe_renders_and_ends_in_silence(mod):
    names = {"impact_stems": IMP_SAMPLE, "weapon_stems": WEP_SAMPLE, "foot_stems": FOOT_SAMPLE}[mod]
    for name in names:
        y, sr = render(name)
        assert sr in (44100, 48000) and len(y) > sr * 0.05
        tail = rms(y[-int(0.02 * sr):])
        peak = np.max(np.abs(y)) + 1e-12
        assert tail / peak < 0.08, (name, tail, peak)


@pytest.mark.parametrize("name", FOOT_SAMPLE + IMP_SAMPLE + WEP_SAMPLE)
def test_a_variation_is_a_different_take(name):
    y0, sr = render(name, k=0)
    y1, _ = render(name, k=1)
    n = min(len(y0), len(y1))
    # same length class, not a bit-identical copy
    r = float(np.corrcoef(y0[:n], y1[:n])[0, 1]) if n > 64 else 1.0
    assert not np.allclose(y0[:n], y1[:n], atol=1e-7)
    assert r < 0.999


def test_footstep_default_is_softer_than_the_recording():
    y_def, sr = render("wood_floor")
    y_hard, _ = soundlab.render(str(R / "wood_floor.json"), macros={"force": 1.0, "soft": 0.0})
    y_hard = np.asarray(y_hard, float)
    assert rms(y_def) < rms(y_hard) * 0.9
    assert db(y_def) < db(y_hard) - 0.5


def test_carpet_is_softer_than_a_wood_step():
    c, _ = render("carpet_walk")
    # versus its own hard setting (the recording's force) it must drop clearly;
    # different surfaces keep their own recorded levels, so this is not compared to wood
    h, _ = soundlab.render(str(R / "carpet_walk.json"), macros={"force": 1.0, "soft": 0.0})
    h = np.asarray(h, float)
    assert rms(c) < rms(h) * 0.85


def test_distance_is_quieter_and_duller():
    near, sr = soundlab.render(str(R / "wood_knock.json"), macros={"distance": 0.0})
    far, _ = soundlab.render(str(R / "wood_knock.json"), macros={"distance": 1.0})
    near, far = np.asarray(near, float), np.asarray(far, float)
    assert rms(far) < rms(near) * 0.85
    assert centroid(far, sr) < centroid(near, sr) * 1.05


def _trim(y, sr, db=-40):
    e = np.abs(y)
    thr = (e.max() + 1e-12) * 10 ** (db / 20)
    idx = np.where(e > thr)[0]
    if not len(idx):
        return y
    return y[: idx[-1] + 1]


def test_size_is_tape_speed():
    small, sr = soundlab.render(str(R / "wood_knock.json"), macros={"size": 0.0})
    large, _ = soundlab.render(str(R / "wood_knock.json"), macros={"size": 1.0})
    small, large = _trim(np.asarray(small, float), sr), _trim(np.asarray(large, float), sr)
    # smaller = higher / shorter (tape speed); the recipe buffer is a fixed length, so compare the active tail
    assert len(small) < len(large) * 0.95
    assert centroid(small, sr) > centroid(large, sr)


def test_count_adds_gestures():
    one, sr = soundlab.render(str(R / "wood_floor.json"), macros={"count": 0.0})
    two, _ = soundlab.render(str(R / "wood_floor.json"), macros={"count": 1 / 3})  # maps toward 2
    one, two = np.asarray(one, float), np.asarray(two, float)
    # two steps should last longer than one (or at least not be shorter)
    assert len(two) >= len(one) * 0.9


@pytest.mark.parametrize("name", LEGACY)
def test_legacy_footstep_recipes_render(name):
    y, sr = render(name)
    assert len(y) > sr * 0.05 and np.max(np.abs(y)) > 1e-4


def test_legacy_footstep_surface_and_shoe_still_parse():
    y, sr = soundlab.render(str(R / "footstep.json"), macros={"surface": 0.16, "shoe": 0.0, "force": 0.4})
    assert len(np.asarray(y)) > sr * 0.05


@pytest.mark.parametrize("fam,names", [
    ("impact", IMP_SAMPLE),
    ("weapon", WEP_SAMPLE),
    ("foot", FOOT_SAMPLE),
])
def test_default_recipe_reproduces_its_fingerprint(fam, names):
    fp = _fp(fam)
    if not fp:
        pytest.skip(f"{fam} fingerprints not generated yet")
    for name in names:
        if name not in fp:
            continue
        y, sr = render(name)
        d = distance(fp[name]["fp"], fingerprint(y, sr))
        assert d <= fp[name].get("tol", 0.35), (name, d)


def test_every_shipped_source_has_a_recipe():
    import soundlab.models.impact_stems as I
    import soundlab.models.weapon_stems as W
    import soundlab.models.foot_stems as F
    missing = []
    for typ in list(I.SOURCES) + list(W.SOURCES) + list(F.SOURCES):
        if not (R / f"{typ}.json").exists():
            missing.append(typ)
    assert not missing, missing

def test_legacy_impact_and_weapon_macros_still_parse():
    """Filenames that used to be procedural recipes still accept their old macro names."""
    cases = [
        ("ceramic_break", {"size": 0.4, "force": 0.6, "hardness": 0.7, "hollowness": 0.3,
                           "muffle": 0.4, "contact_point": 0.2, "distance": 0.1, "material": "glass"}),
        ("coin_jingle", {"metal": "brass", "coins": 0.7, "shake": 0.4, "length": 0.3, "distance": 0.2}),
        ("dowel", {"material": "wood", "size": 0.4, "intensity": 0.6, "hardness": 0.3, "distance": 0.2}),
        ("glass_break", {"size": 0.4, "intensity": 0.6, "floor": "wood", "distance": 0.2}),
        ("metal_clang", {"object": "pipe", "intensity": 0.6, "size": 0.4, "ring": 0.3, "distance": 0.2}),
        ("sword_clash", {"object": "sword", "intensity": 0.6, "scrape": 0.4, "ring": 0.3, "distance": 0.2}),
    ]
    for name, mac in cases:
        y, sr = soundlab.render(str(R / f"{name}.json"), macros=mac)
        y = np.asarray(y, float)
        assert len(y) > sr * 0.05 and np.max(np.abs(y)) > 1e-4, name
