"""foley_stems and door_stems: everyday handling sounds and doors from recording stems (numbers only, no audio).

One real gesture by default; Part choices pick between the different actions a recording holds; the shared
recstems engine's optional parts (grain level cap, segment bed masks, recording-relative mode phases, silent
stretches) are used by these two families only. Fingerprints of the default takes are in tests/data.
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
FAMS = {"foley_stems": ("foley", b"SLFOL001"), "door_stems": ("doors", b"SLDOR001")}
SAMPLE = ["foley/cloth_handling", "foley/cup_down", "foley/coin_drop", "foley/page_turn", "doors/door_wood_close",
          "doors/knock_wood", "doors/door_creak_small", "doors/gate_metal"]


def render(name, k=0, **mac):
    y, sr = soundlab.render(str(R / f"{name}.json"), macros=mac or None, variation=k)
    return np.asarray(y, float), sr


def rms(y):
    return float(np.sqrt(np.mean(np.asarray(y, float) ** 2) + 1e-30))


def _hdr(mod):
    raw = (ROOT / "soundlab" / "models" / f"{mod}.bin").read_bytes()
    assert raw[:8] == FAMS[mod][1]
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    return json.loads(p[4:4 + n])


@pytest.mark.parametrize("mod", list(FAMS))
def test_numbers_only_with_credited_cc0_sources(mod):
    hdr = _hdr(mod)
    concat = {"ev_env", "bed_env", "md_f", "md_amp", "md_ph", "ev_cap"}
    for k, s in hdr["sources"].items():
        for name, a in s.items():
            if isinstance(a, dict) and "sh" in a and name not in concat:
                assert max(a["sh"]) <= s["n"] / 4 + 8, (k, name, a["sh"], s["n"])
        assert s["ev_cap_hop"] >= 8                   # the level cap is a coarse envelope, not samples
        assert s.get("bed_exact") and s.get("md_abs")
        fid = str(s.get("file", "")).replace(".mp3", "")
        assert any(str(v.get("id")) == fid for v in hdr["files"].values() if isinstance(v, dict)), k
    for rec in hdr["files"].values():
        if isinstance(rec, dict):
            assert "cc0" in (rec.get("license") or "").lower() and rec.get("author") and rec.get("url")
    text = (ROOT / "THIRD_PARTY.md").read_text()
    assert f"`{mod}`" in text
    for rec in hdr["files"].values():
        if isinstance(rec, dict):
            assert rec["url"] in text


@pytest.mark.parametrize("mod", list(FAMS))
def test_every_source_has_a_recipe(mod):
    import importlib
    m = importlib.import_module(f"soundlab.models.{mod}")
    folder = FAMS[mod][0]
    missing = [t for t in m.SOURCES if not (R / folder / f"{t}.json").exists()]
    assert not missing


@pytest.mark.parametrize("name", SAMPLE)
def test_default_is_one_gesture_ending_in_silence(name):
    y, sr = render(name)
    assert len(y) > 0.1 * sr
    assert rms(y[-int(0.02 * sr):]) / (np.max(np.abs(y)) + 1e-12) < 0.05


@pytest.mark.parametrize("name", ["foley/cloth_handling", "doors/knock_wood"])
def test_a_variation_is_a_different_take(name):
    y0, _ = render(name)
    y1, _ = render(name, k=1)
    n = min(len(y0), len(y1))
    assert not np.allclose(y0[:n], y1[:n], atol=1e-7)


def test_part_choice_plays_the_other_action():
    a, sr = render("doors/deadbolt")
    b, _ = render("doors/deadbolt", part="unlock")
    n = min(len(a), len(b))
    assert abs(len(a) - len(b)) > 0.1 * sr or float(np.corrcoef(a[:n], b[:n])[0, 1]) < 0.5


def test_silent_stretches_of_the_recording_stay_silent():
    from soundlab.models import door_stems as D, foley_stems as F, recstems as RS
    found = None
    for mod in (F, D):
        for name in mod.SOURCES:
            st = RS.get_source(mod._data(), name)
            for a, b in st.get("sil_fr") or []:
                if b - a >= 40:
                    found = (st, a, b)
                    break
            if found:
                break
        if found:
            break
    assert found, "no source with a long silent stretch"
    st, a, b = found
    hop = int(st.get("sil_hop", 44))
    y = RS.render_window(st, (a + 15) * hop / RS.SR, (b - 15) * hop / RS.SR, 0)
    assert np.max(np.abs(y)) < 1e-9


def test_hinge_voice_is_stored_as_harmonic_mode_tracks():
    from soundlab.models import door_stems as D, recstems as RS
    st = RS.get_source(D._data(), "door_creak_small")
    nom = np.sort(np.asarray(st["md_nom"], float))
    assert len(nom) >= 3
    f1 = nom[0]
    assert np.allclose(nom[:3] / f1, [1, 2, 3], atol=0.02)


def test_engine_options_do_not_touch_the_older_stem_families():
    from soundlab.models import impact_stems as I, recstems as RS
    st = RS.get_source(I._data(), "wood_drop")
    assert not any(k in st for k in ("ev_cap", "bed_exact", "md_abs", "sil_fr"))


@pytest.mark.parametrize("fam", ["foley", "door"])
def test_default_recipes_reproduce_their_fingerprints(fam):
    p = ROOT / "tests" / "data" / f"{fam}_fingerprints.json"
    fp = json.loads(p.read_text())
    names = [n for n in SAMPLE if n.startswith("foley/" if fam == "foley" else "doors/")]
    for name in names:
        y, sr = render(name)
        d = distance(fp[name]["fp"], fingerprint(y, sr))
        assert d <= fp[name].get("tol", 0.35), (name, d)
