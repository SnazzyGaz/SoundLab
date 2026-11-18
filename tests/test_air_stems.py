"""v0.30: air_stems -- wind, whooshes, arrow fly-bys and fans rebuilt from stems of CC0 recordings (docs/air.md).

The stored data is numbers only (band-noise envelopes, tonal tracks, onset envelopes, colour, blade cycle, pass
geometry); every default recipe must reproduce its reference render (fingerprints in tests/data, a 1/3-octave level
map, no audio), and so must the stored variation parameter sets."""
import json
import lzma
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy import signal
from scipy.ndimage import median_filter

import soundlab
from soundlab.core import REGISTRY, RecipeError
from soundlab.models import air_stems as A

from squeal_am import am_depth_db
from squeal_fp import distance, fingerprint

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "recipes"
FP = json.loads((Path(__file__).parent / "data" / "air_fingerprints.json").read_text())
RECIPES = ["wind_gust", "wind_trees", "whoosh", "arrow_flyby", "fan"]
_cache = {}


def render(name, k=0, **mac):
    key = (name, k, json.dumps(mac, sort_keys=True))
    if key not in _cache:
        _cache[key] = soundlab.render(R / f"{name}.json", variation=k, macros=mac or None)
    return _cache[key]


def db(y):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(y) ** 2)) + 1e-12)


def stray_tones(y, sr):
    """Narrow peaks > 15 dB over the local spectrum (and within 50 dB of the loudest bin)."""
    f, P = signal.welch(y, sr, nperseg=16384)
    L = 10 * np.log10(P + 1e-20)
    pk, _ = signal.find_peaks(L - median_filter(L, 101), height=15)
    return [round(float(f[i])) for i in pk if L[i] > L.max() - 50 and f[i] > 40]


def centroid(y, sr):
    f, P = signal.welch(y, sr, nperseg=4096)
    return float((f * P).sum() / P.sum())


# ---- the stored data ------------------------------------------------------------------------------------------

def test_air_data_is_compact_numbers_with_credited_cc0_sources():
    raw = A.DATA.read_bytes()
    assert raw[:8] == b"SLAIR001" and len(raw) < 1_000_000
    p = lzma.decompress(raw[8:])
    hdr = json.loads(p[4:4 + struct.unpack("<I", p[:4])[0]])
    assert set(hdr["families"]) == {"wind_gust", "wind_trees", "whoosh", "fan", "flyby"}
    third = (ROOT / "THIRD_PARTY.md").read_text()
    for f in hdr["files"].values():
        assert f["license"].startswith("CC0") and f["url"].startswith("https://freesound.org/")
        assert f["url"] in third and f["author"] in third, f["url"]


def test_unknown_segment_is_a_recipe_error():
    with pytest.raises(RecipeError):
        A.render_air(REGISTRY["air_stems"].resolve({"segment": "wind_352421@99"}))


# ---- reproduction ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", RECIPES)
def test_default_recipe_reproduces_its_reference(name):
    y, sr = render(name)
    d = distance(FP[name]["fp"], fingerprint(y, sr))
    assert d < 0.5, (name, d)                    # measured: 0.05-0.15 dB


@pytest.mark.parametrize("name", sorted(k for k in FP if "params" in FP[k]))
def test_variation_parameter_sets_reproduce_their_references(name):
    e = FP[name]
    y = A.render_air(REGISTRY["air_stems"].resolve(e["params"]))
    d = distance(e["fp"], fingerprint(y, e["sr"]))
    assert d < 0.5, (name, d)                    # measured: 0.04-0.09 dB


def test_reproduction_distance_tells_takes_apart():
    assert distance(FP["wind_trees"]["fp"], fingerprint(*render("wind_trees", 1))) > 1.0
    assert distance(FP["fan"]["fp"], FP["fan_v2"]["fp"]) > 1.0


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_renders_and_varies(name):
    a, sr = render(name)
    b, _ = render(name, 1)
    assert sr == 48000 and np.all(np.isfinite(a)) and np.abs(a).max() > 0.5
    n = min(len(a), len(b))
    assert np.abs(a[:n] - b[:n]).max() > 0.01


# ---- no bubbling, no stray tones ------------------------------------------------------------------------------

AM_MAX = {"whoosh": 4.2, "arrow_flyby": 4.2}     # single gestures: the swell itself reads as 8-30 Hz AM


@pytest.mark.parametrize("name", RECIPES)
def test_no_bubbling_and_no_stray_tones(name):
    for k in (0, 1):
        y, sr = render(name, k)
        assert am_depth_db(y, sr) < AM_MAX.get(name, 3.0), (name, k)    # measured: wind/fan 1.7-2.5, gestures 2.8-3.9
        assert not stray_tones(y, sr), (name, k, stray_tones(y, sr))    # e.g. the bird whistle in the gust recording


# ---- macros ---------------------------------------------------------------------------------------------------

def test_strength_lowers_level_and_brightness_monotonically():
    lv, ce = [], []
    for s in (1.0, 0.7, 0.4, 0.1):
        y, sr = render("wind_trees", strength=s)
        lv.append(db(y))
        ce.append(centroid(y, sr))
    assert all(a > b for a, b in zip(lv, lv[1:])), lv
    assert all(a > b for a, b in zip(ce, ce[1:])), ce
    assert lv[0] - lv[-1] > 20 and ce[-1] < 0.6 * ce[0], (lv, ce)


def test_strength_slows_the_gusts_and_fits_the_duration():
    y, sr = render("wind_trees", strength=0.3)
    assert len(y) / sr > 1.5 * 8.24
    rec = json.loads((R / "wind_trees.json").read_text())["layers"][0]["params"]
    rec = {k: v for k, v in rec.items() if k != "at"}
    for s in (0.0, 0.3, 0.55, 0.8):                      # the duration macro holds the whole stretched segment
        n = len(A.render_air(REGISTRY["air_stems"].resolve(dict(rec, strength=s)))) / A.SR
        assert n <= len(render("wind_trees", strength=s)[0]) / sr + 0.01, s
    a, _ = render("wind_trees")
    assert am_depth_db(y, sr) < am_depth_db(a, sr)       # shallower


def test_whoosh_speed_and_flyby_count():
    fast, sr = render("whoosh", speed=1.0)
    slow, _ = render("whoosh", speed=0.0)
    assert len(slow) > 1.5 * len(fast) and centroid(fast, sr) > centroid(slow, sr)
    one = A._flyby(REGISTRY["air_stems"].resolve({"mapping": "flyby", "count": 1, "seed": 41}), 41)
    five = A._flyby(REGISTRY["air_stems"].resolve({"mapping": "flyby", "count": 5, "seed": 41}), 41)
    assert len(five) > 2.5 * len(one)


def test_fan_speed_raises_blade_rate_and_brightness():
    lo, sr = render("fan", speed=0.0)
    hi, _ = render("fan", speed=1.0)
    assert centroid(hi, sr) > centroid(lo, sr)
