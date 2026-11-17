"""v0.23: water and wind refitted to real recordings (stream, wind in trees). The references stay outside the
repo, so these tests pin what the fit established: the bands that were missing, the texture, and that the new
controls leave older recipes untouched at their defaults."""
from pathlib import Path

import numpy as np
from scipy import signal

import soundlab
from soundlab.core import REGISTRY

from conftest import SR, run

R = Path(__file__).resolve().parents[1] / "recipes"


def band_share(y, lo, hi):
    f, P = signal.welch(y, SR, nperseg=8192)
    return P[(f >= lo) & (f < hi)].sum() / P.sum()


def mod_depth(y, lo, hi):
    b = signal.sosfiltfilt(signal.butter(4, [lo, hi], "band", fs=SR, output="sos"), y)
    w = int(0.01 * SR)
    e = np.sqrt(np.convolve(b * b, np.ones(w) / w, "valid"))[::w // 2]
    return np.std(e) / e.mean()


def test_water_reaches_the_top_octaves():
    """The bubble-only stream stopped near 5 kHz (bubbles >= 0.6 mm); real streams carry energy to ~16 kHz.
    The refs put ~4 % of the power above 6 kHz; v0.22 had ~0.05 %."""
    y, _ = soundlab.render(R / "water.json", variation=1)
    assert band_share(y, 6000, 16000) > 0.02
    assert band_share(y, 250, 500) > 0.03          # the big bubbles (8-10 mm) that were missing


def test_water_has_plinks_not_only_a_wash():
    y, _ = soundlab.render(R / "water.json", variation=2)
    b = signal.sosfiltfilt(signal.butter(4, [800, 4000], "band", fs=SR, output="sos"), y)
    e = np.sqrt(np.convolve(b * b, np.ones(480) / 480, "valid"))[::240]
    d = e / e.mean() - 1
    assert np.mean(d ** 4) / np.mean(d ** 2) ** 2 > 4.5        # spiky envelope (refs ~8, v0.22 ~3.5)


def test_wind_is_a_steady_breeze_in_trees_with_leaves():
    y, _ = soundlab.render(R / "wind.json", variation=1)
    assert band_share(y, 4000, 16000) > 0.06                     # leaf rustle (v0.22: ~0.01)
    assert band_share(y, 30, 120) < 0.05                          # no mic-buffet rumble (v0.22: ~0.3)
    assert mod_depth(y, 800, 4000) < 0.45                         # no huge slow swells (refs ~0.2, v0.22 ~0.8)


def test_wind_gusts_slider_still_makes_it_gusty():
    calm = np.mean([mod_depth(soundlab.render(R / "wind.json", variation=k)[0], 800, 4000) for k in (1, 2)])
    gusty = np.mean([mod_depth(soundlab.render(R / "wind.json", macros={"gusts": 1.0}, variation=k)[0], 800, 4000)
                     for k in (1, 2)])
    assert gusty > 1.5 * calm


def test_new_controls_default_to_the_old_sound():
    p = {pp.name: pp.default for pp in REGISTRY["wind"].params}
    assert (p["rustle"], p["eddy"], p["low_cut_hz"]) == (0.0, 1.0, 35.0)
    assert REGISTRY["bubble_cloud"].resolve({})["burst"] == 0.0
    a = run("bubble_cloud", dur=1.0, seed=3, rate=3000.0)
    b = run("bubble_cloud", dur=1.0, seed=3, rate=3000.0, burst=0.0)
    assert np.array_equal(a, b)


def test_bubble_burst_clumps_the_bubbles_but_keeps_the_mean_rate():
    steady = run("bubble_cloud", dur=4.0, seed=4, rate=400.0, r_min_mm=2.0, r_max_mm=3.0, bins=8)
    clumped = run("bubble_cloud", dur=4.0, seed=4, rate=400.0, r_min_mm=2.0, r_max_mm=3.0, bins=8, burst=1.5,
                  burst_hz=6.0)
    ratio = np.sqrt(np.mean(clumped ** 2) / np.mean(steady ** 2))
    assert 0.6 < ratio < 1.6                                       # same mean bubble rate (log-normal, mean 1)
    assert mod_depth(clumped, 800, 3000) > 1.3 * mod_depth(steady, 800, 3000)
