"""v0.27: friction_family -- stick-slip / scrape / rub driving the material_impact modes, and dull gravel steps."""
from pathlib import Path

import numpy as np
import pytest

import soundlab
from soundlab.models.friction import MATERIALS

from conftest import SR, run
from test_v0261_impact_fixes import sustained_partial_db

R = Path(__file__).resolve().parents[1] / "recipes"
WOOD_CREAKS = ["door_creak_wood", "floorboard_creak", "chair_creak", "rope_creak"]
RECIPES = WOOD_CREAKS + ["hinge_squeak", "shoe_squeak", "glass_rub", "metal_scrape", "stone_scrape", "sandpaper", "cloth_rub"]


def render(name, k=0, **mac):
    return soundlab.render(R / f"{name}.json", variation=k, macros=mac or None)[0]


def rms(y):
    return float(np.sqrt(np.mean(y ** 2)))


@pytest.mark.parametrize("mode", ["stick_slip", "scrape", "rub"])
def test_every_material_renders(mode):
    for mat in MATERIALS:
        y = run("friction_family", dur=0.4, mode=mode, material=mat, length_s=0.3, size=0.4)
        assert np.all(np.isfinite(y)) and np.abs(y).max() > 1e-4, (mode, mat)


def test_wood_stops_ringing_after_the_stroke_but_steel_rings_on():
    tail = {}
    for mat in ("wood_hard", "steel"):
        y = run("friction_family", dur=1.6, material=mat, length_s=0.5, at=0.05, speed=0.03, pressure=8.0, drive_hz=600.0)
        a, b = int(0.55 * SR), int(0.85 * SR)
        tail[mat] = 20 * np.log10(rms(y[b:]) / rms(y[int(0.1 * SR):a]) + 1e-9)
    assert tail["wood_hard"] < -45
    assert tail["steel"] > tail["wood_hard"] + 15


@pytest.mark.parametrize("name", WOOD_CREAKS)
def test_wood_creaks_have_no_long_metallic_partials(name):
    # recorded door creaks sit at 7-34 dB (median ~14) on this measure; a ringing metal hinge or glass is 45+
    assert sustained_partial_db(render(name), 0.1) < 26


def test_metal_and_glass_stay_tonal():
    wood = max(sustained_partial_db(render(n), 0.1) for n in WOOD_CREAKS)
    for n in ("hinge_squeak", "glass_rub"):
        assert sustained_partial_db(render(n), 0.1) > wood + 15, n


def test_rigid_drive_squeals_more_than_springy_creak():
    kw = dict(dur=0.8, material="steel", length_s=0.6, speed=0.05, pressure=10.0, size=0.3)
    squeal = sustained_partial_db(run("friction_family", drive_hz=0.0, **kw), 0.1)
    creak = sustained_partial_db(run("friction_family", drive_hz=500.0, irregularity=0.8, **kw), 0.1)
    assert squeal > creak


def creak_rate(y):
    """Pulse rate of a creak (Hz) from the autocorrelation of its 0.5 ms envelope."""
    h = int(SR * 0.0005)
    e = np.sqrt(np.convolve(y ** 2, np.ones(h) / h, "same"))[int(0.15 * SR):int(0.85 * SR)]
    e = e - e.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    lo, hi = SR // 1000, SR // 40
    return SR / (lo + np.argmax(ac[lo:hi]))


def test_creak_rate_rises_with_speed_and_falls_with_pressure():
    kw = dict(dur=1.0, material="wood_hard", size=0.35, hollowness=0.5, length_s=0.8, at=0.05, irregularity=0.0,
              drive_hz=600.0, residual=0)
    base = creak_rate(run("friction_family", speed=0.03, pressure=8.0, **kw))
    assert 150 < base < 600                                               # recorded door creaks: ~190-590 Hz
    assert creak_rate(run("friction_family", speed=0.06, pressure=8.0, **kw)) > 1.3 * base
    assert creak_rate(run("friction_family", speed=0.03, pressure=16.0, **kw)) < base / 1.3


def test_pressure_and_speed_change_the_sound():
    base = render("door_creak_wood")
    for mac in ({"pressure": 0.0}, {"pressure": 1.0}, {"speed": 0.0}, {"speed": 1.0}):
        y = render("door_creak_wood", **mac)
        n = min(len(y), len(base))
        assert np.abs(y[:n] - base[:n]).max() > 0.05, mac


def test_speed_change_accelerates_a_scrape():
    def halves(sc):
        y = run("friction_family", dur=1.2, mode="scrape", material="stone", length_s=1.0, at=0.05, speed=0.3, speed_change=sc)
        return rms(y[int(0.6 * SR):int(1.05 * SR)]) / rms(y[int(0.1 * SR):int(0.55 * SR)])
    assert halves(1.0) > 1.3 * halves(-1.0)


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_renders_and_varies(name):
    a, b = render(name, 0), render(name, 1)
    assert np.all(np.isfinite(a)) and np.abs(a).max() > 0.1
    n = min(len(a), len(b))
    assert np.abs(a[:n] - b[:n]).max() > 0.01


def test_cloth_is_noisy_not_tonal():
    assert sustained_partial_db(render("cloth_rub"), 0.1) < 10


def test_gravel_steps_are_dull_crunch_not_tinkling():
    for k in range(3):
        y = render("footstep_gravel", k)
        assert sustained_partial_db(y, 0.03) < 10, k            # v0.26.1 was 14-16: ringing pebbles
        X = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / SR)
        tot = X[(f > 40) & (f < 16000)].sum()
        crunch = X[(f >= 250) & (f < 4000)].sum() / tot
        sparkle = X[(f >= 4000) & (f < 16000)].sum() / tot
        assert crunch > 0.5 and sparkle < 0.25, (k, crunch, sparkle)
