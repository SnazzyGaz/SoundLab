"""v0.24: fixes for measured faults (water comb buzz, wind flicker, fire swell, splash with no body,
match metallic and boomy). The references stay outside the repo; these tests pin the measured cause of each."""
from pathlib import Path

import numpy as np
from scipy import signal

import soundlab

from conftest import SR
from test_v023_water_wind import band_share, mod_depth

R = Path(__file__).resolve().parents[1] / "recipes"


def peakiness(y):
    """dB rms deviation of the fine spectrum from its 1/6-octave smooth: a comb shows up here."""
    f, P = signal.welch(y, SR, nperseg=16384)
    m = (f > 200) & (f < 12000)
    L = 10 * np.log10(P[m] + 1e-20)
    lf = np.log2(f[m])
    sm = np.array([L[np.abs(lf - v) < 1 / 12].mean() for v in lf])
    return float(np.sqrt(np.mean((L - sm) ** 2)))


def roughness(y, lo, hi):
    """Share of the band envelope's modulation power at 20-70 Hz (flicker that smears into a wash or buzz)."""
    b = signal.sosfiltfilt(signal.butter(4, [lo, hi], "band", fs=SR, output="sos"), y)
    e = signal.resample_poly(np.abs(signal.hilbert(b)), 1, 100)
    f, P = signal.welch(e - e.mean(), SR / 100, nperseg=1024)
    return float(P[(f >= 20) & (f < 70)].sum() / P[(f >= 0.5) & (f < 70)].sum())


def render(name, v):
    return soundlab.render(R / f"{name}.json", variation=v)[0]


def test_water_has_no_room_comb():
    """The room's early reflections combed the stream (peakiness ~5.2 dB; real streams ~0.65): a saw-like buzz."""
    for v in (1, 2):
        assert peakiness(render("water", v)) < 2.5


def test_wind_leaves_tick_instead_of_mush():
    for v in (1, 2):
        assert roughness(render("wind", v), 4000, 12000) < 0.45     # v0.23: ~0.53


def test_fire_is_pop_led_not_gusty():
    """Real wood fire: mid-band envelope depth ~2.5 (sparse pops). v0.23 was ~0.8-1.0 (a swelling wash)."""
    for v in (1, 2):
        assert mod_depth(render("fire", v)[int(2 * SR):], 1000, 4000) > 1.5


def test_splash_has_body_and_less_hiss():
    for v in (1, 2):
        y = render("splash", v)
        assert band_share(y, 250, 1000) > 0.4                        # v0.23: ~0.17 (refs 0.33-0.82)
        assert band_share(y, 4000, 16000) < 0.1                      # v0.23: ~0.33 (refs 0.01-0.05)


def test_match_low_end_is_not_a_boom():
    for v in (1, 2):
        assert band_share(render("match", v), 40, 250) < 0.03


def test_room_scattering_defaults_to_the_old_room():
    base = soundlab.render({"duration": 0.5, "layers": [{"model": "noise", "params": {}}],
                            "post": [{"model": "room", "params": {"volume": 40}}]}, seed=3)[0]
    same = soundlab.render({"duration": 0.5, "layers": [{"model": "noise", "params": {}}],
                            "post": [{"model": "room", "params": {"volume": 40, "scattering": 0.0}}]}, seed=3)[0]
    diff = soundlab.render({"duration": 0.5, "layers": [{"model": "noise", "params": {}}],
                            "post": [{"model": "room", "params": {"volume": 40, "scattering": 0.7}}]}, seed=3)[0]
    assert np.array_equal(base, same)
    assert np.isfinite(diff).all() and not np.allclose(base, diff)
