"""Checks of the five reference sounds against their recipe expectations."""
import json
from pathlib import Path

import numpy as np
import pytest

import soundlab
from soundlab.analysis import f0_estimate, onsets, analyze
from conftest import SR, run

RECIPES = Path(__file__).resolve().parents[1] / "recipes"


def band_db(y, lo, hi):
    Y = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return 10 * np.log10(Y[(f >= lo) & (f < hi)].sum() / Y.sum())


@pytest.mark.parametrize("name", ["duck", "engine", "foghorn", "explosion", "match"])
def test_recipes_render(name):
    audio, sr = soundlab.render(RECIPES / f"{name}.json")
    assert np.all(np.isfinite(audio)) and np.max(np.abs(audio)) > 0.5


def test_duck_formant_and_notes():
    y = run("duck", 2.0, notes=5, f0=420, formant_hz=1900)
    from soundlab.analysis import spectral_stats, stft
    f, t, Z = stft(y, SR)
    cen, _, _ = spectral_stats(np.abs(Z), f)
    assert 1200 < cen < 2800  # main resonance 1.5-2.5 kHz
    f0, voiced = f0_estimate(y, SR)
    assert f0 is not None and 180 < f0 < 520  # rasp (period doubling) pulls perceived pitch down
    assert 4 <= len(onsets(y, SR)) <= 7


def test_duck_decrescendo():
    y = run("duck", 2.0, notes=5, decrescendo_db=3.0, rasp=0.2)
    from soundlab.analysis import envelope
    env = envelope(y, SR, 30)
    # first-quack peak louder than last-quack peak
    assert env[: SR // 3].max() > 1.5 * env[int(0.9 * SR):].max()


def test_engine_firing_frequency():
    # 4 cylinders at 3000 rpm -> 100 Hz (RPM x cylinders / 120)
    y = run("engine", 2.0, rpm=3000, load=0.5, cylinders=4, jitter=0.0, mechanical=0, whine=0, intake=0, pops=0)
    Y = np.abs(np.fft.rfft(y[SR // 2:]))
    f = np.fft.rfftfreq(len(y) - SR // 2, 1 / SR)
    on = Y[(f > 98) & (f < 102)].max()
    off = Y[(f > 70) & (f < 80)].max()
    assert on > 3 * off


def test_engine_rpm_tracks():
    lo = run("engine", 2.0, rpm=1200, cylinders=4, jitter=0.0, mechanical=0, whine=0, pops=0, intake=0)
    hi = run("engine", 2.0, rpm=4800, cylinders=4, jitter=0.0, mechanical=0, whine=0, pops=0, intake=0)
    from soundlab.analysis import modulation_rate
    def firing_peak(y, lo_f, hi_f):
        Y = np.abs(np.fft.rfft(y[SR // 2:]))
        f = np.fft.rfftfreq(len(y) - SR // 2, 1 / SR)
        m = (f > lo_f) & (f < hi_f)
        return f[m][np.argmax(Y[m])]
    assert abs(firing_peak(lo, 30, 50) - 40) < 2
    assert abs(firing_peak(hi, 140, 180) - 160) < 3


def test_engine_uneven_firing_has_half_orders():
    even = run("engine", 2.0, rpm=3000, cylinders=2, firing="even", jitter=0.0, mechanical=0, whine=0, pops=0, intake=0)
    vtw = run("engine", 2.0, rpm=3000, cylinders=2, firing="v_twin_45", jitter=0.0, mechanical=0, whine=0, pops=0, intake=0)
    # cycle rate 25 Hz; even 2-cyl fires at 50 Hz so 25 Hz is weak; uneven V-twin has strong 25 Hz
    def ratio(y):
        Y = np.abs(np.fft.rfft(y[SR // 2:]))
        f = np.fft.rfftfreq(len(y) - SR // 2, 1 / SR)
        return Y[(f > 24) & (f < 26)].max() / Y[(f > 49) & (f < 51)].max()
    assert ratio(vtw) > 3 * ratio(even)


def test_foghorn_tone_and_grunt():
    y = run("foghorn", 6.0, tone_hz=250, grunt_hz=110, mode="grunt", blast_s=2.5, attack_s=0.35, grunt_s=1.1, start=0.1)
    mid = y[int(1.0 * SR):int(2.5 * SR)]
    assert abs(f0_estimate(mid, SR)[0] - 250) < 5
    end = y[int(3.75 * SR):int(3.95 * SR)]
    f_end = f0_estimate(end, SR, fmin=60)[0]
    assert f_end is not None and f_end < 170


def test_explosion_starts_sharp_and_is_bass_heavy():
    y = run("explosion", 4.0, size=0.7, at=0.05)
    i = int(0.05 * SR)
    assert np.max(np.abs(y[i:i + int(0.02 * SR)])) > 0.5 * np.max(np.abs(y))  # shock arrives at once
    far = run("distance", 4.0, y, distance=1000.0, spreading=False)
    assert band_db(far, 20, 200) > band_db(far, 2000, 20000)  # distance leaves the low rumble


def test_match_phases():
    y = run("match", 3.5, strike_s=0.22, flare_s=0.45, flame_s=2.0, start=0.1)
    from soundlab.analysis import envelope
    env = envelope(y, SR, 20)
    strike = env[int(0.12 * SR):int(0.3 * SR)].max()
    flame = env[int(1.2 * SR):int(1.8 * SR)].mean()
    assert flame < 0.25 * strike  # steady flame much quieter than strike/flare
    assert flame > 0.005 * strike  # but present
    assert env[int(3.2 * SR):].max() < 0.02 * strike  # goes out
    no = run("match", 3.5, ignite=False)
    assert np.max(np.abs(no[int(1.0 * SR):])) < 1e-3


def test_match_failed_strike_has_no_flame():
    y = run("match", 2.0, ignite=False)
    assert np.max(np.abs(y[: int(0.4 * SR)])) > 0
