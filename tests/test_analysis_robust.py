"""Robust analysis: pitch only for pitched sounds, decay only for decaying sounds, windowed modulation."""
import numpy as np

from soundlab import analysis as A
from conftest import SR, run

rng = np.random.default_rng(0)


def saw(f, dur):
    t = np.arange(int(dur * SR)) / SR
    return 2 * ((f * t) % 1.0) - 1


def test_pitch_none_for_noise_and_blast():
    assert A.f0_estimate(rng.standard_normal(SR), SR)[0] is None
    y = run("explosion", 2.0, at=0.05)
    assert A.f0_estimate(y, SR)[0] is None  # v1 analysis reported a spurious ~1 kHz


def test_pitch_for_tone_and_periodicity():
    info = A.f0_estimate(saw(220, 1.0), SR, return_info=True)
    assert abs(info["f0_hz"] - 220) < 2
    assert info["periodicity"] > 0.9
    noise = A.f0_estimate(rng.standard_normal(SR), SR, return_info=True)
    assert noise["periodicity"] < 0.7


def test_envelope_class_decaying_with_t60():
    t = np.arange(2 * SR) / SR
    x = rng.standard_normal(len(t)) * 10 ** (-3 * t / 0.8)  # T60 = 0.8 s
    r = A.envelope_class(x, SR)
    assert r["kind"] == "decaying"
    assert abs(r["decay_t60_s"] - 0.8) < 0.16


def test_envelope_class_sustained_has_no_decay():
    t = np.arange(3 * SR) / SR
    x = np.sin(2 * np.pi * 150 * t) * np.clip((2.8 - t) / 0.1, 0, 1)
    r = A.envelope_class(x, SR)
    assert r["kind"] == "sustained" and r["decay_t60_s"] is None


def test_envelope_class_repeated_events():
    x = np.zeros(2 * SR)
    for k in range(5):
        i = int((0.1 + 0.35 * k) * SR)
        x[i:i + int(0.12 * SR)] = rng.standard_normal(int(0.12 * SR))
    r = A.envelope_class(x, SR)
    assert r["kind"] == "repeated" and r["events"] == 5
    assert abs(r["event_rate_hz"] - 1 / 0.35) < 0.3


def test_modulation_rate_am_noise_and_none_for_steady():
    t = np.arange(2 * SR) / SR
    x = rng.standard_normal(len(t)) * (1 + 0.9 * np.sin(2 * np.pi * 50 * t))
    assert abs(A.modulation_rate(x, SR) - 50) < 3
    assert A.modulation_rate(rng.standard_normal(2 * SR), SR) is None
