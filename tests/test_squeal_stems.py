"""v0.29: squeal_stems -- squeals, squeaks and creaks rebuilt from stems of CC0 recordings (docs/friction.md).

The stored data is numbers only (partial tracks, band envelopes, wobble tracks, onset/body spectra); every default
recipe must reproduce its reference render (fingerprints in tests/data, a 1/3-octave level map, no audio)."""
import json
import lzma
import struct
from pathlib import Path

import numpy as np
import pytest

import soundlab
from soundlab.core import REGISTRY, RecipeError
from soundlab.models import squeal_stems as S

from squeal_am import am_depth_db, low_band_share_db
from squeal_fp import distance, fingerprint

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "recipes"
FP = json.loads((Path(__file__).parent / "data" / "squeal_fingerprints.json").read_text())
RECIPES = ["door_squeal", "door_creak", "hinge_squeak", "hinge_squeak_2", "gate_squeal", "gate_squeal_long",
           "shoe_squeak", "shoe_squeak_2", "floorboard_creak"]
GESTURES = {"door_creak": None, "gate_squeal": {"latch": "off"}, "hinge_squeak_2": None}   # open, pause, close
_cache = {}


def render(name, k=0, **mac):
    key = (name, k, json.dumps(mac, sort_keys=True))
    if key not in _cache:
        _cache[key] = soundlab.render(R / f"{name}.json", variation=k, macros=mac or None)
    return _cache[key]


def params(name, **upd):
    p = dict(json.loads((R / f"{name}.json").read_text())["layers"][0]["params"])
    p.pop("at")
    p.update(upd)
    return REGISTRY["squeal_stems"].resolve(p)


def timeline(name, **upd):
    tl = []
    S.render_stems(params(name, **upd), tl)
    return tl


def db(y):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(y) ** 2)) + 1e-12)


# ---- the stored data ------------------------------------------------------------------------------------------

def test_stem_data_is_compact_numbers_with_credited_cc0_sources():
    raw = S.DATA.read_bytes()
    assert raw[:8] == b"SLSTEMS1" and len(raw) < 2_500_000
    p = lzma.decompress(raw[8:])
    hdr = json.loads(p[4:4 + struct.unpack("<I", p[:4])[0]])
    assert hdr["sr"] == 44100 and hdr["hop"] == 128
    for k, b in hdr["bursts"].items():           # frame-rate envelopes and tracks only, never samples
        assert b["E_q"]["sh"] == [96, b["nfr"]] and b["E_q"]["dt"] == "uint8", k
        assert sum(b["tracks"]["length"]["sh"]) < 4 * b["nfr"] * 96, k
    third = (ROOT / "THIRD_PARTY.md").read_text()
    for f in hdr["files"].values():
        assert f["license"].startswith("CC0") and f["url"].startswith("https://freesound.org/")
        assert f["url"] in third and f["author"] in third, f["url"]


def test_unknown_burst_is_a_recipe_error():
    with pytest.raises(RecipeError):
        S.render_stems(params("hinge_squeak_2", motions=[{"burst": "hinge_535530/99"}]))


# ---- reproduction ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FP))
def test_default_render_reproduces_its_reference(name):
    rec, mac = ("gate_squeal", {"latch": "off"}) if name == "gate_squeal_no_latch" else (name, {})
    y, sr = render(rec, **mac)
    d = distance(FP[name]["fp"], fingerprint(y, sr))
    assert d < 0.5, (name, d)                    # measured: 0.01-0.12 dB


def test_reproduction_distance_tells_takes_apart():
    for name in ("door_creak", "hinge_squeak_2"):
        ref = FP[name]["fp"]
        assert distance(ref, fingerprint(*render(name, 1))) > 1.5
    assert distance(FP["shoe_squeak"]["fp"], FP["shoe_squeak_2"]["fp"]) > 3


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_renders_and_varies(name):
    a, sr = render(name)
    b, _ = render(name, 1)
    assert sr == 48000 and np.all(np.isfinite(a)) and np.abs(a).max() > 0.5
    n = min(len(a), len(b))
    assert np.abs(a[:n] - b[:n]).max() > 0.01


# ---- the gesture ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(GESTURES))
def test_open_and_close_are_two_motions_with_a_quiet_gap(name):
    (a0, a1, act_a), (b0, b1, act_b) = timeline(name, latch=False)
    assert (act_a, act_b) == ("open", "close")
    assert b0 - a1 >= 0.12, (a1, b0)             # the door stops: no squeal between the motions
    y, sr = render(name, **(GESTURES[name] or {}))
    seg = lambda s, e: y[int(s * sr):int(e * sr)]
    gap = db(seg(a1 + 0.01, b0 - 0.01))
    assert gap < db(seg(a0, a1)) - 20 and gap < db(seg(b0, b1)) - 20, name


def test_gaps_macro_scales_the_pause_and_motions_macro_adds_motions():
    (_, a1, _), (b0, _, _) = timeline("door_creak")
    (_, c1, _), (d0, _, _) = timeline("door_creak", gap_scale=2.0)
    assert abs((d0 - c1) / (b0 - a1) - 2.0) < 0.05
    assert len(timeline("door_creak", motions_n=4)) == 4
    assert len(timeline("door_creak", motions_n=1)) == 1


def test_long_gestures_fit_their_duration():
    for name, mac in (("door_creak", {"motions": 1.0, "gaps": 1.0}), ("shoe_squeak", {"motions": 1.0, "stretch": 1.0})):
        y, sr = render(name, **mac)
        assert db(y[-int(0.1 * sr):]) < db(y) - 30, name     # the end is the tail pad, nothing cut


# ---- no stray drone, no bubbling ------------------------------------------------------------------------------

LOW_MAX = {"door_creak": -15, "door_squeal": -13, "floorboard_creak": None}   # 150-330 Hz share, dB


@pytest.mark.parametrize("name", RECIPES)
def test_no_bubbling_and_no_low_drone(name):
    y, sr = render(name)
    assert am_depth_db(y, sr) < 3.3, name                     # bubbling/purring renders measured 3.9-4.5 dB
    lim = LOW_MAX.get(name, -20)
    if lim is not None:
        assert low_band_share_db(y, sr) < lim, name           # the ~200 Hz drone version of door_creak: -10 dB


@pytest.mark.parametrize("name", ["door_creak", "hinge_squeak", "hinge_squeak_2", "gate_squeal", "gate_squeal_long"])
def test_variations_stay_clean(name):
    for k in (1, 2):
        y, sr = render(name, k)
        assert am_depth_db(y, sr) < 3.5, (name, k)
        if name == "door_creak":
            assert low_band_share_db(y, sr) < -15


# ---- brightness and latch -------------------------------------------------------------------------------------

def centroid(y, sr):
    P = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return float((P * f).sum() / P.sum())


def test_brightness_turns_the_hinge_into_a_gate():
    hinge, sr = render("hinge_squeak_2")
    gate, _ = render("hinge_squeak_2", brightness=1.0)
    assert centroid(gate, sr) > 1.15 * centroid(hinge, sr)
    dull, _ = render("gate_squeal", brightness=0.0, latch="off")
    bright, _ = render("gate_squeal", latch="off")
    assert centroid(bright, sr) > 1.15 * centroid(dull, sr)


def test_latch_clanks_after_the_squeal():
    with_l, sr = render("gate_squeal")
    without, _ = render("gate_squeal", latch="off")
    (_, _, _), (_, b1, _) = timeline("gate_squeal", latch=False)
    tail = with_l[int((b1 + 0.2) * sr):]
    assert db(tail) > db(without[int((b1 + 0.2) * sr):]) + 15
    assert 0.4 < np.abs(tail).max() / np.abs(with_l).max() < 0.65     # its peak 6 dB under the squeal peak


# ---- the v0.28 creak recipes ----------------------------------------------------------------------------------

def test_old_creak_recipes_are_the_documented_fallback():
    for n in ("door_creak", "hinge_squeak", "shoe_squeak", "floorboard_creak"):
        r = json.loads((R / "creak_model" / f"{n}.json").read_text())
        assert r["layers"][0]["model"] == "creak"
        assert json.loads((R / f"{n}.json").read_text())["layers"][0]["model"] == "squeal_stems"
