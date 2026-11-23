"""interact_stems: everyday interaction sounds from recording stems (numbers only, no audio).

One real gesture by default; Part choices pick between the different actions a recording holds; the shared
recstems engine's optional parts (grain level cap, segment bed masks, recording-relative mode phases, silent
stretches) are used as in foley_stems and door_stems. Fingerprints of a sample of default takes are in tests/data.
"""
from __future__ import annotations

import json
import lzma
import struct
from pathlib import Path

import numpy as np
import pytest

import soundlab
from squeal_fp import distance, fingerprint

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "recipes"
SAMPLE = ["interact/camera_shutter", "interact/can_open", "interact/velcro", "interact/keyboard",
          "interact/mouse_click", "interact/cork_pop"]


def render(name, k=0, **mac):
    y, sr = soundlab.render(str(R / f"{name}.json"), macros=mac or None, variation=k)
    return np.asarray(y, float), sr


def rms(y):
    return float(np.sqrt(np.mean(np.asarray(y, float) ** 2) + 1e-30))


def _hdr():
    raw = (ROOT / "soundlab" / "models" / "interact_stems.bin").read_bytes()
    assert raw[:8] == b"SLINT001"
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    return json.loads(p[4:4 + n])


def test_numbers_only_with_credited_cc0_sources():
    hdr = _hdr()
    concat = {"ev_env", "bed_env", "md_f", "md_amp", "md_ph", "ev_cap"}
    for k, s in hdr["sources"].items():
        for name, a in s.items():
            if isinstance(a, dict) and "sh" in a and name not in concat:
                assert max(a["sh"]) <= s["n"] / 4 + 8, (k, name, a["sh"], s["n"])
        assert s["ev_cap_hop"] >= 8
        assert s.get("bed_exact") and s.get("md_abs")
        fid = str(s.get("file", "")).replace(".mp3", "")
        assert any(str(v.get("id")) == fid for v in hdr["files"].values() if isinstance(v, dict)), k
    for rec in hdr["files"].values():
        if isinstance(rec, dict):
            assert "cc0" in (rec.get("license") or "").lower() and rec.get("author") and rec.get("url")
    text = (ROOT / "THIRD_PARTY.md").read_text()
    assert "`interact_stems`" in text
    for rec in hdr["files"].values():
        if isinstance(rec, dict):
            assert rec["url"] in text


def test_every_source_has_a_recipe():
    from soundlab.models import interact_stems as M
    missing = [t for t in M.SOURCES if not (R / "interact" / f"{t}.json").exists()]
    assert not missing


@pytest.mark.parametrize("name", SAMPLE)
def test_default_is_one_gesture_ending_in_silence(name):
    y, sr = render(name)
    assert len(y) > 0.1 * sr
    assert rms(y[-int(0.02 * sr):]) / (np.max(np.abs(y)) + 1e-12) < 0.05


@pytest.mark.parametrize("name", ["interact/camera_shutter", "interact/keyboard"])
def test_a_variation_is_a_different_take(name):
    y0, _ = render(name)
    y1, _ = render(name, k=1)
    n = min(len(y0), len(y1))
    assert not np.allclose(y0[:n], y1[:n], atol=1e-7)


def test_part_choice_plays_the_other_action():
    a, sr = render("interact/velcro")
    b, _ = render("interact/velcro", part="close")
    n = min(len(a), len(b))
    assert abs(len(a) - len(b)) > 0.05 * sr or float(np.corrcoef(a[:n], b[:n])[0, 1]) < 0.5


@pytest.mark.parametrize("name", SAMPLE)
def test_default_recipes_reproduce_their_fingerprints(name):
    fp = json.loads((ROOT / "tests" / "data" / "interact_fingerprints.json").read_text())
    y, sr = render(name)
    d = distance(fp[name]["fp"], fingerprint(y, sr))
    assert d <= fp[name].get("tol", 0.35), (name, d)
