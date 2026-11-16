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
    y = run("duck", 2.0, notes=5, gap_s=0.12)
    from soundlab.analysis import spectral_stats, stft, envelope_class
    f, t, Z = stft(y, SR)
    cen, _, _ = spectral_stats(np.abs(Z), f)
    assert 1200 < cen < 2800  # main resonance 1.5-2.5 kHz
    ev = envelope_class(y, SR)
    assert ev["kind"] == "repeated" and ev["events"] == 5


def test_duck_is_rough_not_sung():
    """v2: a real quack is aperiodic (refs: YIN periodicity 0.36-0.50); v1 was 0.61 and read as a laugh."""
    rough = f0_estimate(run("duck", 2.0, notes=5), SR, return_info=True)
    clean = f0_estimate(run("duck", 2.0, notes=5, rasp=0.0), SR, return_info=True)
    assert rough["periodicity"] < 0.6
    assert clean["periodicity"] > rough["periodicity"] + 0.1


def test_duck_no_pitch_arc_inside_a_quack():
    from soundlab.analysis import f0_track
    y = run("duck", 0.6, notes=1, rasp=0.0, f0=250, note_s=0.3, start=0.05)
    T, F, _, _ = f0_track(y, SR, 100, 600, threshold=0.3)
    v = F[np.isfinite(F)]
    assert len(v) >= 5
    st = 12 * np.log2(v / np.median(v))
    assert np.percentile(st, 90) - np.percentile(st, 10) < 2.5  # v1 rose and fell ~4 semitones


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
    kw = dict(rpm=3000, cylinders=2, jitter=0.0, mechanical=0, whine=0, pops=0, intake=0, exhaust_noise=0)
    even = run("engine", 2.0, firing="even", **kw)
    vtw = run("engine", 2.0, firing="v_twin_45", **kw)
    # cycle rate 25 Hz; even 2-cyl fires at 50 Hz so 25 Hz is weak; uneven V-twin has strong 25 Hz
    def ratio(y):
        Y = np.abs(np.fft.rfft(y[SR // 2:]))
        f = np.fft.rfftfreq(len(y) - SR // 2, 1 / SR)
        return Y[(f > 24) & (f < 26)].max() / Y[(f > 49) & (f < 51)].max()
    assert ratio(vtw) > 3 * ratio(even)


def test_engine_high_rpm_not_tinny():
    """v2: brightness should not scale with rpm (v1 differentiated the pulses, so 4x rpm = much brighter)."""
    from soundlab.analysis import spectral_stats, stft
    def cen(rpm):
        y = run("engine", 1.5, rpm=rpm, load=0.8, cylinders=8, firing="crossplane_v8", mechanical=0, whine=0, pops=0)
        f, t, Z = stft(y[SR // 2:], SR)
        return spectral_stats(np.abs(Z), f)[0]
    # v3: idle is now much darker (no stationary hiss), so the old ratio test no longer means anything;
    # the absolute ceiling is what guards against tinny revs
    assert cen(6000) < 900


def test_engine_crossplane_v8_burble():
    # two banks firing unevenly: strong half-order (cycle rate) content at 3000 rpm -> 25 Hz
    y = run("engine", 2.0, rpm=3000, cylinders=8, firing="crossplane_v8", jitter=0.0, mechanical=0, whine=0,
            pops=0, intake=0, exhaust_noise=0)
    e = run("engine", 2.0, rpm=3000, cylinders=8, firing="even", jitter=0.0, mechanical=0, whine=0,
            pops=0, intake=0, exhaust_noise=0)
    def sub(y):
        Y = np.abs(np.fft.rfft(y[SR // 2:]))
        f = np.fft.rfftfreq(len(y) - SR // 2, 1 / SR)
        return sum(Y[(f > k * 25 - 1) & (f < k * 25 + 1)].max() for k in (1, 2, 3)) / Y[(f > 199) & (f < 201)].max()
    assert sub(y) > 2 * sub(e)


def test_foghorn_defaults_are_bassy():
    y = run("foghorn", 5.0)
    f0 = f0_estimate(y, SR)[0]
    assert f0 is not None and 60 < f0 < 160  # refs 87-150 Hz; v1 was 250
    assert band_db(y, 20, 300) > -4.5  # about half the energy below 300 Hz (barge horn reference: -2.6 dB)


def test_foghorn_grunt_is_choppy():
    from soundlab.analysis import envelope
    def depth(chop):
        y = run("foghorn", 5.0, mode="grunt", chop=chop, roughness=0.0, blast_s=1.5, grunt_s=1.5, start=0.1)
        e = envelope(y, SR, 5.0)[int(2.0 * SR):int(2.9 * SR)]
        return np.std(20 * np.log10(e / e.max() + 1e-6))
    assert depth(0.9) > depth(0.0) + 2.0


def test_foghorn_tone_and_grunt():
    y = run("foghorn", 6.0, tone_hz=250, grunt_hz=110, mode="grunt", blast_s=2.5, attack_s=0.35, grunt_s=1.1, start=0.1,
            roughness=0.0, chop=0.0)
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


def test_explosion_is_a_blast_not_a_vent():
    """Fast rise, heavy lows early; v3: the high band dies before the low band (refs: big explosions keep
    broadband energy for ~2 s, so v2's 'HF gone in 0.7 s' rule was dropped)."""
    from soundlab.analysis import envelope
    y = run("explosion", 4.0, size=0.65, at=0.05, debris=0.0)
    e = envelope(y, SR, 2.0)
    i10, i90 = int(np.argmax(e > 0.1 * e.max())), int(np.argmax(e > 0.9 * e.max()))
    assert (i90 - i10) / SR < 0.01  # v1 took ~0.2 s to peak
    import scipy.signal as ss
    hi = ss.sosfilt(ss.butter(4, 2000, "high", fs=SR, output="sos"), y)
    h = envelope(hi, SR, 20.0)
    lo = envelope(ss.sosfilt(ss.butter(4, 200, "low", fs=SR, output="sos"), y), SR, 20.0)
    assert (h > 0.1 * h.max()).sum() < (lo > 0.1 * lo.max()).sum()
    first = y[int(0.05 * SR):int(0.55 * SR)]
    assert band_db(first, 20, 200) > band_db(first, 2000, 20000) + 6


def test_match_phases():
    y = run("match", 3.5, strike_s=0.08, flare_s=0.25, flame_s=2.0, flame_db=-26, start=0.1)
    from soundlab.analysis import envelope
    env = envelope(y, SR, 20)
    strike = env[int(0.1 * SR):int(0.4 * SR)].max()
    flame = env[int(1.2 * SR):int(1.8 * SR)].mean()
    assert flame < 0.25 * strike  # steady flame much quieter than strike/flare
    assert flame > 0.005 * strike  # but present
    assert env[int(3.2 * SR):].max() < 0.02 * strike  # goes out
    no = run("match", 3.5, ignite=False)
    assert np.max(np.abs(no[int(1.0 * SR):])) < 1e-3


def test_match_failed_strike_has_no_flame():
    y = run("match", 2.0, ignite=False)
    assert np.max(np.abs(y[: int(0.4 * SR)])) > 0
