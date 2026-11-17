"""v0.25: ten recipes fitted to a second, new-domain reference batch (BigSoundBank and BBC Sound Effects, outside
the repo; fitted on some takes, others held out for checking). These pin the cues each fit established; the
ranges span the real takes (tune and hold-out) with some margin, and each notes the v0.24 value it replaced."""
from pathlib import Path

import numpy as np
from scipy import signal

import soundlab
from soundlab.checkpoint import prepare

from conftest import SR

R = Path(__file__).resolve().parents[1] / "recipes"


def clip(name, v):
    return prepare(soundlab.render(R / f"{name}.json", variation=v)[0], 6.0)


def centroid(x):
    f, P = signal.welch(x, SR, nperseg=4096)
    return float((f * P).sum() / P.sum())


def decay_s(x):
    w = 480
    e = 20 * np.log10(np.sqrt(np.convolve(x * x, np.ones(w) / w, "same")) + 1e-9)
    p = int(np.argmax(e))
    b = np.nonzero(e[p:] < e[p] - 30)[0]
    return (b[0] if len(b) else len(e) - p) / SR


def test_ball_bounce_is_bright_ticks_not_a_ringing_plate():
    for v in (1, 2):
        x = clip("hc_bounce", v)
        assert centroid(x) > 6000          # real 8.5-9.5 kHz (the ball's own modes); v0.24: 1.3 kHz plate
        assert decay_s(x) < 0.1            # each tick dies in ms; v0.24 rang 1.2 s
        assert len(x) / SR > 1.5           # a long run of bounces (real ~3 s, restitution ~0.97)


def test_drumstick_click_is_short_and_bright():
    for v in (1, 2):
        x = clip("drumstick", v)
        assert centroid(x) > 3500          # real ~5.1 kHz; the v0.7 dowel it was fitted from: 0.7 kHz
        assert len(x) / SR < 0.35          # real 0.15-0.16 s


def test_creak_is_a_bright_stick_slip_not_a_low_groan():
    for v in (1, 2):
        x = clip("creak_friction", v)
        assert centroid(x) > 3000          # real parquet 5.0-5.4 kHz; v0.24: 150 Hz
        assert len(x) / SR < 1.5           # real 0.7-1.0 s; v0.24: 2.5 s


def test_scrape_is_short_and_bright():
    for v in (1, 2):
        x = clip("scrape_sdt", v)
        assert centroid(x) > 3500          # real 5.1-7.3 kHz; v0.24: 1.1 kHz
        assert len(x) / SR < 1.0           # real 0.4-0.6 s; v0.24: 1.9 s


def test_dog_growl_and_wind_howl_sit_in_the_real_range():
    for v in (1, 2):
        assert 350 < centroid(clip("dog_growl", v)) < 1300     # real 390-1100 Hz; v0.24: 290 Hz hum
        assert 250 < centroid(clip("wind_howl", v)) < 1700     # real 300-1600 Hz; v0.24: 2.1 kHz


def test_squeak_knock_clang_foghorn_in_the_real_range():
    for v in (1, 2):
        assert 1300 < centroid(clip("squeak_friction", v)) < 4500   # real 1.7-4.1 kHz
        assert 0.08 < len(clip("hc_knock", v)) / SR < 0.4           # real 0.14-0.28 s; v0.24: 0.37 s
        assert 2500 < centroid(clip("metal_clang", v)) < 7000       # real 3.1-4.5 kHz; v0.24: 2.1 kHz
        assert 150 < centroid(clip("foghorn", v)) < 500             # real 180-370 Hz


def test_friction_drops_modes_above_nyquist():
    """The creak's top modes (13-20 kHz) are above Nyquist at 16 kHz: they are dropped, not integrated to NaN."""
    y, sr = soundlab.render(R / "creak_friction.json", sample_rate=16000)
    assert sr == 16000 and np.isfinite(y).all() and np.abs(y).max() > 0.01
