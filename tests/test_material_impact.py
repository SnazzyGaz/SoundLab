"""v0.26 material_impact: contact + modal body + residual, material table, shapes, macros, derived recipes."""
import json
from pathlib import Path

import numpy as np
import pytest

import soundlab
from soundlab.models.impacts import IMPACT_MATERIALS, SHAPES

from conftest import SR, run

RECIPES = Path(__file__).resolve().parents[1] / "recipes"
NEW = ["impact_any", "knock", "drop", "bounce", "footstep_wood", "footstep_stone", "footstep_metal", "footstep_dirt",
       "footstep_gravel", "footstep_grass", "pickaxe_earth", "pickaxe_rock", "blade_clash", "glass_clink",
       "ceramic_break", "body_hit"]


def t60(y, sr=SR):
    """Time from the peak to 40 dB below it (x 1.5 -> T60 estimate) on a 5 ms RMS envelope."""
    h = int(0.005 * sr)
    e = np.sqrt(np.convolve(y ** 2, np.ones(h) / h, "same")) + 1e-12
    i = int(np.argmax(e))
    below = np.nonzero(20 * np.log10(e[i:] / e[i]) < -40)[0]
    return 1.5 * (below[0] if len(below) else len(e) - i) / sr


def centroid(y, sr=SR):
    Y = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    return float((Y * f).sum() / (Y.sum() + 1e-20))


@pytest.mark.parametrize("material", sorted(IMPACT_MATERIALS))
@pytest.mark.parametrize("shape", SHAPES)
def test_every_material_and_shape_renders(material, shape):
    y = run("material_impact", dur=0.8, material=material, shape=shape, size=0.3)
    assert np.all(np.isfinite(y)) and np.abs(y).max() > 1e-3


def test_non_metals_do_not_ring_like_metal():
    ring = {m: t60(run("material_impact", dur=4.0, material=m, shape="plate", size=0.4, striker="wood"))
            for m in ("wood_soft", "wood_hard", "stone", "plastic", "rubber", "flesh", "steel", "aluminium")}
    for m in ("wood_soft", "wood_hard", "stone", "plastic", "rubber", "flesh"):
        assert ring[m] < 0.6, (m, ring[m])
    assert ring["steel"] > 1.5 and ring["aluminium"] > 1.5
    assert ring["wood_hard"] < ring["steel"] / 4


def test_size_scales_pitch_inversely():
    from conftest import peak_freq
    small = peak_freq(run("material_impact", dur=2.0, material="steel", shape="bar", size=0.3, striker="none"))
    large = peak_freq(run("material_impact", dur=2.0, material="steel", shape="bar", size=0.6, striker="none"))
    assert 0.35 < large / small < 0.65


def test_harder_and_faster_contacts_are_brighter():
    soft = centroid(run("material_impact", material="wood_hard", hardness=0.0, velocity=0.5))
    hard = centroid(run("material_impact", material="wood_hard", hardness=1.0, velocity=6.0))
    assert hard > 1.3 * soft


def test_hollow_box_is_lower_and_longer_than_solid():
    solid = run("material_impact", dur=2.0, material="wood_hard", shape="box", size=0.6, hollowness=0.0)
    hollow = run("material_impact", dur=2.0, material="wood_hard", shape="box", size=0.6, hollowness=1.0)
    assert centroid(hollow) < centroid(solid) and t60(hollow) > t60(solid)


def test_muffle_shortens_and_darkens():
    a = run("material_impact", dur=3.0, material="steel", shape="plate", size=0.5, muffle=0.0)
    b = run("material_impact", dur=3.0, material="steel", shape="plate", size=0.5, muffle=0.8)
    assert t60(b) < t60(a) / 2 and centroid(b) < centroid(a)


def test_footstep_gesture_has_heel_and_ball():
    y = run("material_impact", dur=0.5, material="wood_hard", gesture="footstep", striker="shoe", velocity=1.5)
    h = int(0.004 * SR)
    e = np.sqrt(np.convolve(y ** 2, np.ones(h) / h, "same"))
    from scipy.signal import find_peaks
    pk, _ = find_peaks(e, height=0.15 * e.max(), distance=int(0.04 * SR))
    assert len(pk) >= 2 and 0.04 < (pk[1] - pk[0]) / SR < 0.16


def test_ground_is_not_tonal_and_fracture_adds_debris():
    def peakiness(y):
        Y = 20 * np.log10(np.abs(np.fft.rfft(y, 1 << 15)) + 1e-9)
        return float(np.max(Y) - np.median(Y[100:8000]))
    assert peakiness(run("material_impact", material="dirt")) < peakiness(run("material_impact", material="steel",
                                                                                  shape="bar", striker="steel"))
    whole = run("material_impact", dur=1.0, material="ceramic", shape="shell", size=0.12, fracture=0.0, seed=4)
    broke = run("material_impact", dur=1.0, material="ceramic", shape="shell", size=0.12, fracture=0.8, seed=4)
    late = slice(int(0.08 * SR), int(0.6 * SR))
    assert np.sum(broke[late] ** 2) / np.sum(broke ** 2) > np.sum(whole[late] ** 2) / np.sum(whole ** 2)


@pytest.mark.parametrize("name", NEW)
def test_impact_recipes_render_and_vary(name):
    p = RECIPES / f"{name}.json"
    r = json.loads(p.read_text())
    assert r["layers"][0]["model"] == "material_impact"
    y0, sr = soundlab.render(p)
    y1, _ = soundlab.render(p, variation=1)
    assert np.all(np.isfinite(y0)) and np.abs(y0).max() > 0.1
    n = min(len(y0), len(y1))
    assert np.sqrt(np.mean((y0[:n] - y1[:n]) ** 2)) > 1e-3       # game takes differ


def test_material_macro_switches_material():
    a, _ = soundlab.render(RECIPES / "impact_any.json")
    b, _ = soundlab.render(RECIPES / "impact_any.json", macros={"material": "steel"})
    assert t60(b) > 3 * t60(a)


def test_pickaxe_earth_keeps_the_fitted_cues():
    """Fitted to a recorded pick strike into soil: short (40 dB down within ~0.1 s), bright crumble (centroid
    roughly 2.5-6 kHz in the real take)."""
    y, sr = soundlab.render(RECIPES / "pickaxe_earth.json")
    assert t60(y, sr) < 0.35
    assert 1500 < centroid(y, sr) < 7000
