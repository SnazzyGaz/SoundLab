"""v0.14: the cues that gave seven engine sounds away next to real recordings are fixed at their cause.
Each test checks the measured property, not a particular waveform."""
import json
from pathlib import Path

import numpy as np
from scipy import signal

import soundlab

RECIPES = Path(__file__).resolve().parents[1] / "recipes"
SR = 48000


def rec(name, k=1):
    y, sr = soundlab.render(RECIPES / f"{name}.json", variation=k)
    assert sr == SR and np.all(np.isfinite(y))
    return np.asarray(y, float)


def band_share(y, lo, hi):
    f, P = signal.welch(y, SR, nperseg=4096)
    return P[(f >= lo) & (f < hi)].sum() / P.sum()


def env_db(y, win_s=0.02):
    w = int(win_s * SR)
    e = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, "same"))
    return 20 * np.log10(e / e.max() + 1e-9)


def test_sword_blade_rings_high_and_short():
    for k in (1, 2):
        y = rec("sword_clash", k)
        assert band_share(y, 3000, 20000) > 0.6          # a thin blade radiates above its coincidence frequency
        e = env_db(y)
        pk = int(np.argmax(e))
        assert np.nonzero(e[pk:] < -20)[0][0] / SR < 0.3  # grip + other blade choke the ring


def test_metal_clang_is_bright_with_a_little_body():
    """v0.25: fitted to real anvil strikes and a dropped iron bar (BigSoundBank, outside the repo): 19-87 % of
    their power is above 3 kHz and 1-2 % below 1 kHz. The v0.13-0.24 pipe kept a low ring (< 60 % above 3 kHz)."""
    for k in (1, 2):
        y = rec("metal_clang", k)
        assert 0.3 < band_share(y, 3000, 20000) < 0.95
        assert band_share(y, 0, 1000) < 0.3


def test_glass_break_is_loudest_at_the_break_and_has_no_gap():
    r = json.loads((RECIPES / "glass_break.json").read_text())
    at = r["layers"][0]["params"]["at"]
    for k in (1, 2):
        y = rec("glass_break", k)
        e = env_db(y, 0.01)
        assert abs(np.argmax(e) / SR - at) < 0.06
        seg = e[int((at + 0.02) * SR):int((at + 0.35) * SR)]
        assert seg.min() > -40                           # shards land continuously after the break


def test_splash_has_a_low_body_and_a_continuous_tail():
    for k in (1, 2):
        y = rec("splash", k)
        # low weight present but not dominant (CC0/PD splash recordings: 40-250 Hz holds ~1-6% of the power)
        assert 0.015 < band_share(y, 40, 250) < 0.25
        e = env_db(y, 0.05)
        pk = int(np.argmax(e))
        assert pk / SR < 0.15                            # loudest at the impact
        tail = e[pk:pk + int(0.8 * SR)]
        assert tail.min() > -45                          # the wash carries the fall-back, no silent gaps
        assert np.nonzero(e[pk:] < -40)[0][0] / SR > 0.6  # not over in a few hundred ms


def test_cloth_is_broadband_with_overlapping_strokes():
    y = rec("cloth_rustle")
    assert band_share(y, 2000, 16000) > 0.5
    assert band_share(y, 8000, 20000) > 0.08
    e = env_db(y, 0.05)
    on = np.nonzero(e > -20)[0]
    inside = e[on[0]:on[-1]]
    assert inside.min() > -40                            # fibres keep brushing between strokes


def test_paper_rustle_keeps_its_crinkle():
    y = rec("paper_rustle")
    assert band_share(y, 2000, 16000) > 0.3


def test_wind_gets_brighter_as_it_gets_louder():
    # v0.23: the default is now a steady breeze in trees (fitted to two real recordings, whose level and
    # brightness are not linked: -0.35..+0.54 per 8 s window); the cue belongs to gusts, so it is tested with
    # the gusts slider up, averaged over variations (one 8 s take is noisy)
    cs = []
    for k in range(1, 5):
        y, _ = soundlab.render(RECIPES / "wind.json", macros={"gusts": 1.0}, variation=k)
        f, t, Z = signal.stft(y, SR, nperseg=4096, noverlap=2048)
        P = np.abs(Z) ** 2
        lvl = 10 * np.log10(P.sum(0) + 1e-20)
        cen = (f[:, None] * P).sum(0) / (P.sum(0) + 1e-20)
        keep = lvl > lvl.max() - 30
        cs.append(np.corrcoef(lvl[keep], np.log(cen[keep]))[0, 1])
    assert np.mean(cs) > 0.2 and min(cs) > 0
    assert band_share(y, 1000, 8000) > 0.15


def test_wind_gap_whistle_pitch_is_steady_through_gusts():
    y = rec("wind_gap")
    f, t, Z = signal.stft(y, SR, nperseg=8192, noverlap=6144)
    P = np.abs(Z) ** 2
    m = (f > 200) & (f < 2000)
    pk = f[m][np.argmax(P[m], 0)]
    loud = P[m].max(0) > np.percentile(P[m].max(0), 50)
    mid = np.median(pk[loud])
    assert np.mean(np.abs(pk[loud] / mid - 1) < 0.05) > 0.8
    assert band_share(y, 1000, 20000) < 0.1             # heard indoors: the wind noise behind it is dark


def test_growl_is_bass_heavy_and_its_formants_move():
    for k in (1, 2):
        y = rec("creature_growl", k)
        assert band_share(y, 30, 400) > 0.5
        assert band_share(y, 2000, 20000) < 0.03
        f, t, Z = signal.stft(y, SR, nperseg=4096, noverlap=3072)
        P = np.abs(Z[(f > 200) & (f < 2000)]) ** 2
        ff = f[(f > 200) & (f < 2000)]
        loud = P.sum(0) > 0.05 * P.sum(0).max()
        cen = (ff[:, None] * P).sum(0)[loud] / P.sum(0)[loud]
        assert np.std(cen) / np.mean(cen) > 0.05
