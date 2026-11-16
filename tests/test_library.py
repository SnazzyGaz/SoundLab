"""v0.9 game library: wind, rain, glass, swing, clash, doors, punch, bullets, gunshot, growl, rustle, splash."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

import soundlab
from soundlab.core import Ctx
from soundlab.models.library import call

RECIPES = Path(__file__).resolve().parents[1] / "recipes"
SR = 48000
NEW = ["wind", "wind_gap", "rain", "glass_break", "sword_swing", "sword_clash", "metal_clang", "door_creak",
       "door_slam", "punch", "bullet_impact", "gunshot", "creature_growl", "paper_rustle", "cloth_rustle", "splash"]


def run(name, dur=1.0, seed=1, **p):
    y = call(name, Ctx(SR, int(dur * SR), np.random.default_rng(seed)), None, **p)
    assert np.all(np.isfinite(y))
    return y


def rms_db(y):
    return 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)


def peak_hz(y, lo, hi, nfft=1 << 17):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y)), nfft))
    f = np.fft.rfftfreq(nfft, 1 / SR)
    m = (f > lo) & (f < hi)
    return float(f[m][np.argmax(S[m])])


def test_library_size_and_metadata():
    names = sorted(p.stem for p in RECIPES.glob("*.json"))
    assert len(names) >= 30, names
    for n in NEW:
        r = json.loads((RECIPES / f"{n}.json").read_text())
        assert r.get("category") and r.get("macros") and r.get("vary"), n


@pytest.mark.parametrize("name", NEW)
def test_new_recipes_render_and_are_deterministic(name):
    y1, sr = soundlab.render(RECIPES / f"{name}.json")
    y2, _ = soundlab.render(RECIPES / f"{name}.json")
    assert sr == SR and np.all(np.isfinite(y1)) and np.max(np.abs(y1)) > 0.5
    assert np.array_equal(y1, y2)
    yv, _ = soundlab.render(RECIPES / f"{name}.json", variation=3)
    assert not np.array_equal(y1, yv)          # jitter actually changes something


def centroid(y):
    P = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float((P * f).sum() / P.sum())


def test_wind_brightens_with_speed():
    # the model is level-normalised (loudness is the mix's job); speed shows up as a brighter, more open sound
    slow = run("wind", 3.0, speed=4.0, gustiness=0.0)
    fast = run("wind", 3.0, speed=16.0, gustiness=0.0)
    assert centroid(fast) > 1.3 * centroid(slow)


def test_wind_gusts_follow_v_cubed():
    calm = run("wind", 6.0, seed=4, speed=8.0, gustiness=0.0, howl=0, trees=0, rumble=0)
    gusty = run("wind", 6.0, seed=4, speed=8.0, gustiness=1.0, howl=0, trees=0, rumble=0)
    def spread(y):
        e = 20 * np.log10(np.sqrt(np.convolve(y ** 2, np.ones(SR // 5) / (SR // 5), "valid")) + 1e-9)
        return np.percentile(e, 95) - np.percentile(e, 5)
    assert spread(gusty) > spread(calm) + 6


def test_wind_gap_whistle_pitch_follows_strouhal():
    v, gap = 10.0, 5.0
    y = run("wind", 3.0, speed=v, gustiness=0.0, whistle=1.0, gap_mm=gap, howl=0, trees=0, rumble=0)
    want = 0.2 * v / (gap * 1e-3)
    assert abs(peak_hz(y, 150, 3000) / want - 1) < 0.2


def test_rain_rate_raises_level_and_density():
    light = run("rain", 3.0, rate_mm_h=0.5, hiss=0.0)
    heavy = run("rain", 3.0, rate_mm_h=30.0, hiss=0.0)
    def density(y):     # onsets per second above the running level
        e = np.abs(y)
        return np.mean(e > 4 * np.median(e) + 1e-9)
    assert density(heavy) != density(light)
    # heavy rain: many overlapping drops, so the level is far steadier (lower crest factor)
    crest = lambda y: np.max(np.abs(y)) / np.sqrt(np.mean(y ** 2))
    assert crest(heavy) < 0.7 * crest(light)


def test_glass_shards_ring_high():
    y = run("glass_break", 2.0, at=0.02)
    tail = y[int(0.3 * SR):]
    P = np.abs(np.fft.rfft(tail)) ** 2
    f = np.fft.rfftfreq(len(tail), 1 / SR)
    assert P[f > 1500].sum() > 2 * P[(f > 20) & (f < 1500)].sum()


def test_swing_doppler_falls_through_the_pass():
    # closest point is at u = 0.5 (t = 0.25 s): with Doppler the approach is brighter than the retreat
    def drop(d):
        r = []
        for seed in range(3):
            y = run("swing", 0.6, seed=seed, doppler=d, edge=1.0, at=0.05, swing_s=0.4, closest_m=0.3, speed=40.0)
            r.append(centroid(y[int(0.15 * SR):int(0.25 * SR)]) / centroid(y[int(0.25 * SR):int(0.35 * SR)]))
        return np.mean(r)
    assert drop(1.0) > drop(0.0) * 1.05


def test_door_creak_pitch_follows_slip_rate():
    def rate(p):
        y = run("door_creak", 2.0, pitch_hz=p, wander=0.0, open_s=1.8, at=0.05)
        seg = np.abs(signal.hilbert(y[int(0.8 * SR):int(1.1 * SR)]))
        return peak_hz(seg - seg.mean(), 60, 1200)
    r1, r2 = rate(200.0), rate(400.0)
    assert 1.6 < r2 / r1 < 2.5


def test_growl_is_voiced_with_its_pitch():
    # 0.10: the growl must be a voice (periodic glottal pulses), not a friction-like click train
    from soundlab.analysis import f0_track
    y = run("growl", 1.8, f0=80.0, roughness=0.0, rise=0.0, snarl=0.0, length_s=1.6)
    T, F, A, R = f0_track(y, SR, 30, 1500, 0.25)
    F = np.array(F)
    v = F[np.isfinite(F)]
    assert len(v) / len(F) > 0.8 and abs(np.median(v) / 80.0 - 1) < 0.05


def test_growl_roughness_adds_subharmonic():
    def sub(rough):
        y = run("growl", 1.8, seed=2, f0=80.0, roughness=rough, rise=0.0, snarl=0.0, length_s=1.6)
        seg = y[int(0.4 * SR):int(1.4 * SR)]
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        return S[(f > 34) & (f < 46)].sum() / S[(f > 74) & (f < 86)].sum()
    assert sub(1.0) > 3 * sub(0.0)


def _grain(y, lo=2000, hi=8000):
    """99.9th percentile of the band envelope over its local (50 ms) mean: ~4-5 for smooth noise, >>5 for crackle."""
    b, a = signal.butter(4, [lo / (SR / 2), hi / (SR / 2)], "band")
    e = np.abs(signal.lfilter(b, a, y))
    return float(np.percentile(e / (np.convolve(e, np.ones(2400) / 2400, "same") + 1e-9), 99.9))


def test_wind_and_rain_are_not_grainy():
    # 0.10 (wind 'very grainy', rain 'too much crackle on top')
    assert _grain(run("wind", 4.0, trees=1.0)) < 6.0
    assert _grain(run("rain", 4.0, rate_mm_h=5.0, near=0.3, hiss=0.7, brightness=0.3)) < 6.0


def test_door_slam_has_mass():
    # 0.10 ('two wood planks'): the 60-150 Hz thump outweighs the 1-4 kHz clack
    y = run("door_slam", 1.0)
    P = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    assert P[(f > 60) & (f < 150)].sum() > 10 * P[(f > 1000) & (f < 4000)].sum()


def test_gunshot_has_no_metallic_ring_and_calibres_differ():
    def tonal(y):
        S = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 16)) ** 2
        d = 10 * np.log10(S[300:8000] + 1e-20)
        return float(np.max(d - signal.medfilt(d, 301)))
    ys = {c: run("gunshot", 1.0, calibre=c, echo=0.0) for c in ("pistol", "rifle", "shotgun")}
    for y in ys.values():
        assert tonal(y) < 25
    def t30(y):
        e = np.convolve(y ** 2, np.ones(240) / 240, "same")
        i = int(np.argmax(e))
        return int(np.argmax(e[i:] < e[i] * 1e-3))
    assert t30(ys["shotgun"]) > t30(ys["pistol"])


def test_splash_bloop_at_minnaert_frequency():
    with_b = run("splash", 1.5, seed=2, bloop=1.0, droplets=0.0, size=0.08)
    no_b = run("splash", 1.5, seed=2, bloop=0.0, droplets=0.0, size=0.08)
    r = 0.15 * 0.08
    # starts at Minnaert 3.26/r and chirps up as the bubble rises (xi), so the average peak sits a bit above
    ratio = peak_hz(with_b - no_b, 80, 3000) / (3.26 / r)
    assert 0.9 < ratio < 1.3


def test_bullet_materials_differ():
    def tonal(y):
        S = np.abs(np.fft.rfft(y * np.hanning(len(y)), 1 << 16)) ** 2
        d = 10 * np.log10(S[200:6000] + 1e-20)
        return float(np.max(d - signal.medfilt(d, 301)))
    metal = run("bullet_impact", 1.0, material="metal")
    conc = run("bullet_impact", 1.0, material="concrete")
    assert tonal(metal) > tonal(conc) + 6
