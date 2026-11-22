"""v0.8 foley: body-weight footsteps, board knock vs dowel, physical coin (drop + jingle)."""
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

import soundlab
from soundlab.models.foley import coin_mode_freqs, euler_wobble_times

RECIPES = Path(__file__).resolve().parents[1] / "recipes"

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
        return RECIPES / "impact_model" / f"{name}.json"
    return RECIPES / f"{name}.json"

SR = 48000


def render(name, **macros):
    y, sr = soundlab.render(recipe_path(name), macros=macros or None)
    assert sr == SR and np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0.5
    return y


def bands(y, edges=(30, 150, 400, 1000, 2500, 6000, 16000)):
    P = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    tot = P[f > 30].sum()
    return np.array([10 * np.log10(P[(f >= a) & (f < b)].sum() / tot + 1e-12) for a, b in zip(edges[:-1], edges[1:])])


def tonal_peak_db(y, lo=150, hi=4000):
    """Strongest narrow spectral peak above its local median (a 'pure ring' scores high)."""
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 16)) ** 2
    f = np.fft.rfftfreq(1 << 16, 1 / SR)
    m = (f > lo) & (f < hi)
    d = 10 * np.log10(S[m] + 1e-20)
    return float(np.max(d - signal.medfilt(d, 301)))


def decay_ms(y, drop_db=40):
    env = np.convolve(np.abs(signal.hilbert(y)), np.ones(48) / 48, "same")
    i = int(np.argmax(env))
    below = np.nonzero(env[i:] < env[i] * 10 ** (-drop_db / 20))[0]
    return below[0] / SR * 1000 if len(below) else np.inf


# ------------------------------------------------------------------ footstep
def test_footstep_has_body_weight_and_no_pure_ring():
    y = render("footstep")[: int(0.4 * SR)]
    b = bands(y)
    assert b[0] > -4, "a body-weight step on wood has most of its energy below 150 Hz (CC0 refs: -1 to -3 dB)"
    assert tonal_peak_db(y) < 25, "no tonal ring above 150 Hz (the v0.7 'tapping' step rang)"
    old = soundlab.render(recipe_path("dowel"))[0][: int(0.4 * SR)]
    assert tonal_peak_db(old) > tonal_peak_db(y) + 10


def test_footstep_shoes_and_surfaces_differ():
    top = {}
    for shoe in ("trainer", "boot", "heel"):
        y = render("footstep", shoe=shoe)[: int(0.4 * SR)]
        top[shoe] = bands(y, (30, 2500, 20000))[1]
    assert top["trainer"] < top["boot"] < top["heel"], f"soft foam is dullest, a hard heel brightest: {top}"
    wood = bands(render("footstep")[: int(0.4 * SR)])
    gravel = bands(render("footstep", surface="gravel")[: int(0.4 * SR)])
    stone = bands(render("footstep", surface="stone")[: int(0.4 * SR)])
    assert gravel[3] > wood[3] + 6 and gravel[0] < wood[0] - 6, "gravel = crunch (1-2.5 kHz), not boom"
    assert stone[4] > wood[4], "stone is harder/grittier than wood"


def test_footstep_steps_and_determinism():
    from soundlab.recipe import render as rr
    rec = soundlab.load_recipe(str(recipe_path("footstep")))
    rec["layers"][0]["params"].update(steps=3, interval_s=0.5)
    rec["duration"] = 1.6
    a, _ = rr(rec)
    b, _ = rr(rec)
    assert np.array_equal(a, b)
    env = np.convolve(np.abs(a), np.ones(480) / 480, "same")
    for t in (0.02, 0.52, 1.02):  # a heel strike near each step time
        i = int(t * SR)
        assert env[i:i + int(0.06 * SR)].max() > 0.2 * env.max()


# ------------------------------------------------------------------ hit: board knock vs dowel
def test_board_knock_is_damped_broadband_and_dowel_is_v07():
    board = render("hit")[: int(0.6 * SR)]
    dowel = render("dowel")[: int(0.6 * SR)]
    assert decay_ms(board) < 0.6 * decay_ms(dowel)
    assert tonal_peak_db(board) < tonal_peak_db(dowel) - 10
    assert bands(board)[0] > bands(dowel)[0] + 15, "the board knock has low thump; the dowel has none"
    via_macro = render("hit", object="dowel")
    assert np.array_equal(via_macro, soundlab.render(recipe_path("dowel"))[0])


# ------------------------------------------------------------------ coin
def test_coin_modes_follow_free_plate_theory():
    f, ns = coin_mode_freqs(28.5, 1.85, "cupronickel")
    assert 8500 < f[0] < 9700, "28.5 x 1.85 mm cupronickel: (2,0) mode ~9.1 kHz (CC0 ref measured 9054 Hz)"
    assert abs(f[1] / f[0] - 1.729) < 0.01 and abs(f[2] / f[0] - 2.328) < 0.01  # Leissa ratios
    f2, _ = coin_mode_freqs(28.5 * 0.7, 1.85, "cupronickel")
    assert f2[0] == pytest.approx(f[0] / 0.49, rel=1e-6)  # f ~ h / a^2


def test_coin_drop_rings_at_its_mode_and_settles():
    y = render("coin")
    f, P = signal.welch(y, SR, nperseg=8192)
    m = (f > 5000) & (f < 20000)
    peak = f[m][np.argmax(P[m])]
    f0 = coin_mode_freqs(28.5, 1.85, "cupronickel")[0][0]
    assert abs(peak / f0 - 1) < 0.03, (peak, f0)
    env = np.convolve(np.abs(y), np.ones(480) / 480, "same")
    assert env[-int(0.2 * SR):].max() < 0.01 * env.max(), "it settles and the ring is choked"
    assert render("coin", size=0.0) is not None


def test_euler_wobble_accelerates():
    t = euler_wobble_times(1.0, 10.0, 0.5)
    d = np.diff(t)
    assert len(t) > 12 and np.all(np.diff(d) <= 1e-9), "wobbles come faster and faster"
    assert d[0] > 5 * d[-1] and t[-1] < 1.0


def test_coin_jingle_and_retro_coin_kept():
    y = render("coin_jingle")
    assert y.size
    assert (RECIPES / "blocks" / "coin_pickup.json").exists(), "the retro coin stays"


def test_heel_and_toe_flam_into_one_step():
    """Heel and toe must come almost as one. Gap ~20-60 ms, scaled a little by shoe/pace."""
    from soundlab.models.foley import SHOES
    from soundlab.recipe import render as rr
    rec = soundlab.load_recipe(str(recipe_path("footstep")))
    for pace in (0.0, 0.5, 1.0):
        roll = apply_pace(rec, pace)
        for shoe in SHOES.values():
            assert 0.015 <= roll * shoe["roll"] <= 0.075, (pace, roll, shoe["roll"])
    dry = soundlab.load_recipe(str(recipe_path("footstep")))
    dry["post"] = []
    dry["layers"][0]["params"].update(scuff=0.0, creak=0.0)
    y, _ = rr(dry)
    env = np.convolve(np.abs(y), np.ones(96) / 96, "same")
    i = int(np.argmax(env))
    assert i < int(0.1 * SR) and env[i + int(0.08 * SR):].max() < 0.25 * env[i], "no separate late toe hit"


def apply_pace(rec, pace):
    from soundlab.macros import apply_macros
    return apply_macros(rec, {"pace": pace})["layers"][0]["params"]["roll_s"]
