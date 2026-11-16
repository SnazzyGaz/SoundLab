"""v0.15: texture statistics (after McDermott & Simoncelli 2011) for the sounds whose sub-band envelopes
gave them away. Each test checks a measured property, not a particular waveform."""
from pathlib import Path

import numpy as np
from scipy import signal, stats

import soundlab
from soundlab.models.library import band_modulate

RECIPES = Path(__file__).resolve().parents[1] / "recipes"
SR = 48000


def rec(name, k=1):
    y, sr = soundlab.render(RECIPES / f"{name}.json", variation=k)
    assert sr == SR and np.all(np.isfinite(y))
    return np.asarray(y, float)


def band_env(y, lo, hi, dec=60):
    b = signal.sosfilt(signal.butter(2, [lo, hi], btype="band", fs=SR, output="sos"), y)
    return np.maximum(signal.resample_poly(np.abs(signal.hilbert(b)), 1, dec), 0) ** 0.3


def band_share(y, lo, hi):
    f, P = signal.welch(y, SR, nperseg=4096)
    return P[(f >= lo) & (f < hi)].sum() / P.sum()


def active(y, e, dec=60, floor_db=-30):
    lvl = 20 * np.log10(np.convolve(np.abs(y), np.ones(dec) / dec, "same")[::dec] + 1e-12)
    m = lvl > lvl.max() + floor_db
    return e[: len(m)][m[: len(e)]]


def test_band_modulate_decorrelates_bands_and_keeps_level():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4 * SR)
    lo_c = band_modulate(x, SR, np.random.default_rng(1), depth=0.6, rates=(1.0, 4.0), corr=0.0)
    hi_c = band_modulate(x, SR, np.random.default_rng(1), depth=0.6, rates=(1.0, 4.0), corr=1.0)
    for y in (lo_c, hi_c):
        assert np.all(np.isfinite(y)) and 0.5 < np.std(y) / np.std(x) < 2.0

    def corr(y):
        a, b = band_env(y, 500, 700, 600), band_env(y, 3000, 4000, 600)
        return np.corrcoef(a, b)[0, 1]
    assert corr(lo_c) < corr(hi_c) - 0.15


def test_rain_has_air_on_top_and_slow_swells():
    y = rec("rain")
    assert band_share(y, 7000, 20000) > 0.008                    # v0.14: 0.001 (12 kHz lowpass, dull hiss)
    e = band_env(y, 1000, 1400)
    f, P = signal.welch(e - e.mean(), SR / 60, nperseg=2048)
    assert P[(f >= 0.25) & (f < 4)].sum() / P[(f >= 0.25) & (f < 128)].sum() > 0.1   # v0.14: 0.035


def test_door_creak_is_a_continuous_squeal_not_clicks():
    ks = []
    for k in (1, 2, 3):
        y = rec("door_creak", k)
        ks.append(stats.kurtosis(active(y, band_env(y, 2000, 4000))))
    assert np.mean(ks) < 0.1                                     # v0.14 click train: ~0.35


def test_creature_has_an_audible_trill_and_breath():
    for k in (1, 2):
        y = rec("creature_growl", k)
        e = active(y, band_env(y, 100, 400))
        f, P = signal.welch(e - e.mean(), SR / 60, nperseg=1024)
        assert P[(f >= 15) & (f < 40)].sum() / P[(f >= 2) & (f < 128)].sum() > 0.12   # the trill flutters the voice
        assert band_share(y, 30, 300) > 0.5                      # guttural bass still carries it
        top = band_env(y, 7000, 12000)
        f, P = signal.welch(top - top.mean(), SR / 60, nperseg=1024)
        # breath chopped by the pulses and the trill (v0.14: ~0.1, the top only drifted slowly)
        assert P[(f >= 16) & (f < 128)].sum() / P[(f >= 0.5) & (f < 128)].sum() > 0.18


def test_glass_break_is_a_series_of_separate_hits():
    ks = [stats.kurtosis(active(y, band_env(y, 2000, 7000))) for y in (rec("glass_break", k) for k in (1, 2, 3))]
    assert np.mean(ks) > 0.0                                     # v0.14 smooth tinkle wash: ~ -0.2


def test_cloth_bands_come_and_go_on_their_own():
    y = rec("cloth_rustle")
    a, b = active(y, band_env(y, 600, 800)), active(y, band_env(y, 6000, 8000))
    assert np.corrcoef(a, b)[0, 1] < 0.72                        # v0.14: 0.82; recordings 0.59-0.75
