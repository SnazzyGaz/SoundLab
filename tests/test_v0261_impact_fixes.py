"""v0.26.1: normal-walking footsteps, a ceramic crack instead of a ding, an untuned body thump."""
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

import soundlab

from conftest import SR
from test_material_impact import t60

R = Path(__file__).resolve().parents[1] / "recipes"
FOOT = ["footstep_wood", "footstep_stone", "footstep_metal", "footstep_dirt", "footstep_gravel", "footstep_grass"]


def env(y, ms=1.0):
    h = max(1, int(ms * 1e-3 * SR))
    return np.sqrt(np.convolve(y ** 2, np.ones(h) / h, "same"))


def peakiness(y, lo=60, hi=4000):
    """Strongest narrow spectral peak above its +-140 Hz median (dB, Welch): pitched modes give 15-45 dB."""
    f, P = signal.welch(y, SR, nperseg=4096, noverlap=3072)
    A = 10 * np.log10(P + 1e-20)
    return max(A[i] - np.median(A[max(0, i - 12):i + 13]) for i in range(len(f)) if lo < f[i] < hi)


def sustained_partial_db(y, t0=0.05):
    """After t0: the strongest partial of the time-averaged spectrum above its +-1 kHz median (dB)."""
    f, t, S = signal.stft(y, SR, nperseg=2048, noverlap=1536)
    P = 20 * np.log10(np.abs(S) + 1e-9)
    sel = (t > t0) & (P.max(0) > P.max() - 50)
    A = 10 * np.log10(np.mean(10 ** (P[:, sel] / 10), 1) + 1e-12)
    w = int(1000 / f[1])
    return max(A[i] - np.median(A[max(0, i - w):i + w]) for i in range(5, len(f) - w))


def render(name, k=0, **mac):
    return soundlab.render(R / f"{name}.json", variation=k, macros=mac or None)[0]


@pytest.mark.parametrize("name", FOOT)
def test_footsteps_are_quiet_beside_a_knock(name):
    knock = render("knock")
    for k in range(3):
        y = render(name, k)
        assert 20 * np.log10(np.abs(y).max()) < 20 * np.log10(np.abs(knock).max()) - 9


@pytest.mark.parametrize("name", ["footstep_wood", "footstep_stone", "footstep_metal"])
def test_hard_floor_footsteps_thud_without_ringing(name):
    for k in range(3):
        y = render(name, k)
        assert t60(y) < 0.3
        e = env(y, 2.0)
        i = int(np.argmax(e))
        late = e[i + int(0.15 * SR):]
        assert late.size == 0 or late.max() < 0.1 * e[i]            # nothing rings on 150 ms after the heel


def test_ball_of_foot_and_scuff_are_quieter_than_the_heel():
    from soundlab.models.impacts import material_impact  # noqa: F401
    from conftest import run
    y = run("material_impact", dur=0.6, material="wood_hard", gesture="footstep", striker="shoe", velocity=1.0, seed=3)
    e = env(y, 4.0)
    heel = e[: int(0.06 * SR)].max()
    later = e[int(0.08 * SR):].max()
    assert later < 0.6 * heel


def test_ceramic_break_has_no_sustained_ding():
    for k in range(3):
        y = render("ceramic_break", k)
        assert sustained_partial_db(y) < 20
    ding = render("glass_clink")
    assert sustained_partial_db(ding) > 40                         # the test does catch a ringing body


def test_ceramic_break_starts_with_a_broadband_snap_and_scatters_shards():
    y = render("ceramic_break")
    on = int(np.argmax(env(y) > 0.1 * env(y).max()))
    snap = y[on:on + int(0.004 * SR)]
    Y = np.abs(np.fft.rfft(snap, 4096)) ** 2
    f = np.fft.rfftfreq(4096, 1 / SR)
    assert Y[f > 2000].sum() / Y.sum() > 0.3                       # a crack, not a low knock
    e = env(y, 5.0)
    assert e[on + int(0.15 * SR):on + int(0.6 * SR)].max() > 0.03 * e.max()   # shards landing later


def test_body_hit_is_an_instant_untuned_thump():
    for k in range(3):
        y = render("body_hit", k)
        e = env(y)
        on = int(np.argmax(e > 0.1 * e.max()))
        pk = int(np.argmax(e))
        assert (pk - on) / SR < 0.005                               # fast attack, peak in the first 5 ms
        assert env(y, 5.0)[pk + int(0.015 * SR):].max() < 0.5 * env(y, 5.0)[pk]   # nothing swells afterwards
        assert peakiness(y) < 11                                    # no pitched mode
    assert peakiness(render("blade_clash")) > 25


def test_flesh_on_any_shape_has_no_pitch():
    from conftest import run
    for shape in ("block", "plate", "bar"):
        assert peakiness(run("material_impact", dur=0.5, material="flesh", shape=shape, striker="flesh")) < 11
