"""v0.27: friction_family -- stick-slip / scrape / rub driving the material_impact modes, and dull gravel steps.
v0.27.1: squeal mode (stick-slip from the excitation, no body ring), no end-of-stroke knock.
v0.28: door_creak, hinge_squeak, shoe_squeak and floorboard_creak moved to the fitted creak model (tests/test_creak.py);
rope_creak and door_creak_wood are gone. The friction model's own squeal mode is still tested here."""
from pathlib import Path

import numpy as np
import pytest

import soundlab
from soundlab.models.friction import MATERIALS

from conftest import SR, run
from test_v0261_impact_fixes import sustained_partial_db

R = Path(__file__).resolve().parents[1] / "recipes"

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

WOOD_CREAKS = ["chair_creak"]
RECIPES = WOOD_CREAKS + ["glass_rub", "metal_scrape", "stone_scrape", "sandpaper", "cloth_rub"]


def render(name, k=0, **mac):
    return soundlab.render(recipe_path(name), variation=k, macros=mac or None)[0]


def rms(y):
    return float(np.sqrt(np.mean(y ** 2)))


@pytest.mark.parametrize("mode", ["stick_slip", "scrape", "rub", "squeal"])
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
    assert sustained_partial_db(render("glass_rub"), 0.1) > wood + 15


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


# ---- v0.27.1 squeals -------------------------------------------------------------------------------------------

def params(name):
    import json
    return json.loads((R / f"{name}.json").read_text())["layers"][0]["params"]


def env10(y):
    w = int(0.01 * SR)
    return np.sqrt(np.convolve(y ** 2, np.ones(w) / w, "same"))


_w = np.hanning(int(0.03 * SR))
WAC = np.correlate(_w, _w, "full")[len(_w) - 1:] / np.sum(_w ** 2)


def voice_stats(y, f_lo, f_hi, t0=0.0, t1=None):
    """Median normalised-autocorrelation peak in the f0 range -> (f0, HNR dB), over loud 30 ms frames."""
    y = y[int(t0 * SR):int(t1 * SR) if t1 else None]
    W, H = int(0.03 * SR), int(0.01 * SR)
    e0 = rms(y)
    f0s, rs = [], []
    for i in range(0, len(y) - W, H):
        f = y[i:i + W] * np.hanning(W)
        if rms(f) < 0.5 * e0:
            continue
        ac = np.correlate(f, f, "full")[W - 1:] / (WAC[:W] + 1e-9)   # Boersma: divide out the window's own lag
        ac /= ac[0] + 1e-12
        a, b = int(SR / f_hi), int(SR / f_lo)
        k = a + int(np.argmax(ac[a:b]))
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]                  # parabolic peak (non-integer periods)
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        f0s.append(SR / (k + d))
        rs.append(min(0.999, y1 - 0.25 * (y0 - y2) * d))
    r = float(np.clip(np.median(rs), 1e-3, 0.999))
    return float(np.median(f0s)), 10 * np.log10(r / (1 - r))


def harmonic_count(y, f0, floor_db=30):
    X = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1 / SR)
    hs = []
    for k in range(1, 40):
        if k * f0 > 16000:
            break
        m = (fr > k * f0 * 0.94) & (fr < k * f0 * 1.06)
        hs.append(X[m].max())
    hs = 20 * np.log10(np.array(hs) + 1e-12)
    return int(np.sum(hs > hs.max() - floor_db))


@pytest.mark.parametrize("name", WOOD_CREAKS)
def test_no_knock_after_the_stroke_ends(name):
    p = params(name)
    end = int((p.get("at", 0.05) + 1.15 * p["length_s"]) * SR)     # variations stretch the stroke by up to 10 %
    for k in range(3):
        e = env10(render(name, k))
        before = e[:max(1, end - int(0.05 * SR))].max()
        assert e[end - int(0.05 * SR):].max() < before, (name, k)  # the stroke's end is never its loudest part


def test_squeal_pitch_follows_speed():
    kw = dict(dur=1.0, mode="squeal", material="steel", length_s=0.8, at=0.05, pitch_hz=800.0, irregularity=0.0)
    lo = voice_stats(run("friction_family", speed=0.05, speed_change=-1.0, **kw), 550, 1600, 0.45, 0.7)[0]
    hi = voice_stats(run("friction_family", speed=0.05, speed_change=1.0, **kw), 550, 1600, 0.45, 0.7)[0]
    assert hi > 1.2 * lo

