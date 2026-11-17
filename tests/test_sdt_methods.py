"""v0.19 blocks built from Sound Design Toolkit methods (method only) and related papers."""
import numpy as np
import pytest

import soundlab.recipe as R
from soundlab import fit as F

SR = 24000


def render(layer, dur=1.5, seed=3):
    y, sr = R.render({"duration": dur, "seed": seed, "sample_rate": SR, "layers": [layer],
                      "master": {"normalize_db": None}}, SR)
    y = np.asarray(y, float)
    assert np.isfinite(y).all()
    return y if y.ndim == 1 else y.mean(1)


def centroid(y):
    S = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float((S * f).sum() / (S.sum() + 1e-12))


def crest(y):
    return float(np.abs(y).max() / (np.sqrt(np.mean(y ** 2)) + 1e-12))


def knock(v):
    return {"chain": [{"model": "hc_contact", "params": {"at": 0.01, "velocity": v, "bounces": 1}},
                      {"model": "modal", "params": {"material": "oak", "shape": "plate", "length": 0.5,
                                                    "width": 0.3, "thickness": 0.02}}]}


def test_hunt_crossley_harder_hits_are_brighter():
    assert centroid(render(knock(5.0))) > 1.3 * centroid(render(knock(0.5)))


def test_hunt_crossley_bounces_make_several_contacts():
    y = render({"model": "hc_contact", "params": {"at": 0.01, "velocity": 2.0, "bounces": 8, "restitution": 0.7}}, 2.0)
    env = np.abs(y) > 0.05 * np.abs(y).max()
    onsets = np.sum(np.diff(env.astype(int)) == 1)
    assert onsets >= 4


MODES = [[180, 0.08, 1.0], [410, 0.05, 0.6]]


def test_ep_friction_springy_drive_creaks_and_rigid_squeals():
    creak = render({"model": "ep_friction", "params": {"velocity": 0.02, "drive_hz": 20, "modes": MODES}}, 1.0)
    squeal = render({"model": "ep_friction", "params": {"velocity": 0.05, "drive_hz": 0, "modes": MODES}}, 1.0)
    assert crest(creak) > crest(squeal) + 1.5


def test_crumple_and_break():
    c = render({"model": "crumple", "params": {"mode": "crumple", "length_s": 1.0, "rate": 200}})
    b = render({"model": "crumple", "params": {"mode": "break", "rate": 80, "energy": 0.9}}, 2.0)
    assert np.abs(c).max() > 0 and np.abs(b).max() > 0
    # breaking runs out of energy: the second half is quieter than the first
    h = len(b) // 2
    assert np.sum(b[:h] ** 2) > np.sum(b[h:] ** 2)
    # smaller fragments ring higher
    lo = render({"model": "crumple", "params": {"length_s": 1.0, "freq_hz": 800}})
    hi = render({"model": "crumple", "params": {"length_s": 1.0, "freq_hz": 4000}})
    assert centroid(hi) > centroid(lo)


@pytest.mark.parametrize("mode", ["scrape", "roll"])
def test_surface_faster_is_louder(mode):
    slow = render({"model": "surface", "params": {"mode": mode, "velocity": 0.1}})
    fast = render({"model": "surface", "params": {"mode": mode, "velocity": 0.8}})
    assert np.sqrt(np.mean(fast ** 2)) > np.sqrt(np.mean(slow ** 2))


def test_wind_cavity_locks_to_higher_mode_at_higher_speed():
    lo = render({"model": "wind_cavity", "params": {"wind_speed": 4, "floor": 0.0}}, 2.0)
    hi = render({"model": "wind_cavity", "params": {"wind_speed": 20, "floor": 0.0}}, 2.0)
    assert centroid(hi) > centroid(lo)


def test_dc_motor_pitch_follows_rpm():
    f = np.fft.rfftfreq(int(1.5 * SR), 1 / SR)
    peaks = []
    for rpm in (3000, 15000):
        y = render({"model": "dc_motor", "params": {"rpm": rpm, "air": 0, "brush": 0, "gear": 0}})
        peaks.append(f[np.argmax(np.abs(np.fft.rfft(y)))])
    assert peaks[1] == pytest.approx(5 * peaks[0], rel=0.05)


def test_mains_hum_peaks_at_twice_mains():
    y = render({"model": "mains_hum", "params": {"mains_hz": 50, "buzz": 0.0}}, 2.0)
    S = np.abs(np.fft.rfft(y))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    assert abs(f[np.argmax(S)] - 100) < 2


def test_sparsity_term_tells_bubbles_from_noise():
    bub = render({"model": "bubble_cloud", "params": {"rate": 2000}}, 3.0)
    bub2 = render({"model": "bubble_cloud", "params": {"rate": 2000}}, 3.0, seed=9)
    noise = render({"model": "noisegen", "params": {"color": "pink"}}, 3.0)
    d_noise = F.compare_signals(bub, SR, noise, SR)["sparsity"]
    d_bub = F.compare_signals(bub, SR, bub2, SR)["sparsity"]
    assert d_noise > 3 * d_bub
    assert "sparsity" in F.DEFAULT_WEIGHTS


def test_resynth_fit_weights_skip_sparsity_for_pitched_kinds():
    from soundlab import resynth as RS
    assert RS.FIT_WEIGHTS["voice"]["sparsity"] == 0 and RS.FIT_WEIGHTS["choir"]["sparsity"] == 0
    assert RS.FIT_WEIGHTS["texture"]["sparsity"] > 0
