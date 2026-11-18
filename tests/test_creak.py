"""v0.28: the creak model -- two regimes fitted to CC0 recordings (see docs/friction.md).

groan  = bursts of slip clicks through fixed body resonances (clicks speed up, the resonances stay put);
squeal = a near-pure tone locked to a resonance that glides with speed and hops between bursts.
Targets are numbers measured from the recordings; no audio is stored."""
import json
from pathlib import Path

import numpy as np
import pytest

import soundlab
from soundlab.models.creak_data import BODIES, VOICES

from conftest import SR, run

R = Path(__file__).resolve().parents[1] / "recipes"
CREAKS = ["door_creak", "hinge_squeak", "shoe_squeak", "floorboard_creak"]


def render(name, k=0, **mac):
    return soundlab.render(R / f"{name}.json", variation=k, macros=mac or None)[0]


def params(name):
    return json.loads((R / f"{name}.json").read_text())["layers"][0]["params"]


def rms(y):
    return float(np.sqrt(np.mean(y ** 2)) + 1e-15)


def spec_db(f, nfft=8192):
    X = np.abs(np.fft.rfft(f * np.hanning(len(f)), nfft))
    return np.fft.rfftfreq(nfft, 1 / SR), 20 * np.log10(X + 1e-12)


def tone_to_floor(seg):
    fr, X = spec_db(seg)
    m = (fr > 200) & (fr < 8000)
    return float(X[m].max() - np.median(X[m])), float(fr[m][np.argmax(X[m])])


def harmonics_within(seg, f0, floor_db=30, top=16000):
    fr, X = spec_db(seg, 16384)
    hs = [X[(fr > k * f0 * 0.97) & (fr < k * f0 * 1.03)].max() for k in range(1, int(top / f0) + 1)]
    hs = np.array(hs)
    return int(np.sum(hs > hs.max() - floor_db))


def bursts(y, thr_db=-20, min_gap=0.015):
    """(start, end) seconds of the runs where the 5 ms envelope is within thr_db of its peak."""
    w = int(0.005 * SR)
    e = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, "same"))[::w]
    on = 20 * np.log10(e / e.max() + 1e-12) > thr_db
    runs, s = [], None
    for i, a in enumerate(on):
        if a and s is None:
            s = i
        if not a and s is not None:
            runs.append([s, i])
            s = None
    if s is not None:
        runs.append([s, len(on)])
    merged = []
    for r in runs:
        if merged and (r[0] - merged[-1][1]) * 0.005 < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a * 0.005, b * 0.005) for a, b in merged]


def envelope_peak_hz(y, lo=300, hi=6000):
    """Peak of the 1/3-octave-smoothed long-term spectrum (where the body resonance sits)."""
    W = 4096
    acc = 0
    for i in range(0, len(y) - W, W // 2):
        acc = acc + np.abs(np.fft.rfft(y[i:i + W] * np.hanning(W))) ** 2
    fr = np.fft.rfftfreq(W, 1 / SR)
    c = np.geomspace(lo, hi, 120)
    env = [acc[(fr > f / 2 ** (1 / 6)) & (fr < f * 2 ** (1 / 6))].mean() for f in c]
    return float(c[int(np.argmax(env))])


def buzz_share(y, f_lo=250, f_hi=550):
    """Share of loud 30 ms frames that look like a voiced buzz: a strong fundamental in f_lo..f_hi (within 20 dB
    of the frame peak), 8+ harmonics within 30 dB and an autocorrelation HNR above 3 dB.
    The v0.27.1 renders score 0.21 (door_creak), 0.24 (hinge_squeak) and 0.66 (door_creak_wood); the CC0
    recordings 0.00-0.13."""
    W = int(0.03 * SR)
    w = np.hanning(W)
    wac = np.correlate(w, w, "full")[W - 1:] / np.sum(w ** 2)
    e0 = rms(y)
    n = flag = 0
    for i in range(0, len(y) - W, int(0.01 * SR)):
        f = y[i:i + W]
        if rms(f) < 0.5 * e0:
            continue
        n += 1
        g = f * w
        ac = np.correlate(g, g, "full")[W - 1:] / (wac + 1e-9)
        ac /= ac[0] + 1e-12
        a, b = int(SR / f_hi), int(SR / f_lo)
        k = a + int(np.argmax(ac[a:b]))
        r = float(np.clip(ac[k], 1e-3, 0.999))
        f0 = SR / k
        fr, X = spec_db(f)
        fund = X[(fr > f0 * 0.9) & (fr < f0 * 1.1)].max()
        if 10 * np.log10(r / (1 - r)) > 3 and fund > X.max() - 20 and harmonics_within(f, f0) >= 8:
            flag += 1
    return flag / max(n, 1)


def hiss_db(y):
    """Median over loud frames of the 6-12 kHz level against the strongest 0.2-6 kHz bin (a hiss bed sits high).
    v0.27.1 shoe_squeak: -34 dB; the CC0 sneaker recording -41 dB."""
    e0 = rms(y)
    out = []
    for i in range(0, len(y) - int(0.05 * SR), int(0.025 * SR)):
        f = y[i:i + int(0.05 * SR)]
        if rms(f) < 0.5 * e0:
            continue
        fr, X = spec_db(f)
        P = 10 ** (X / 10)
        out.append(10 * np.log10(P[(fr > 6000) & (fr < 12000)].mean() / P[(fr > 200) & (fr < 6000)].max()))
    return float(np.median(out))


def click_rate(y, t0, t1):
    """Click rate (Hz) from the autocorrelation of the 0.5 ms envelope."""
    h = int(SR * 0.0005)
    e = np.sqrt(np.convolve(y ** 2, np.ones(h) / h, "same"))[int(t0 * SR):int(t1 * SR)]
    e = e - e.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    lo, hi = SR // 1000, SR // 40
    return SR / (lo + np.argmax(ac[lo:hi]))


def groan(**kw):
    p = dict(dur=1.6, at=0.05, length_s=1.4, regime="groan", body="door_heavy", irregularity=0.0, speed_change=0.0,
             burst_ms=2000.0, gap_ms=0.0, burst_spread=0.0)
    p.update(kw)
    return run("creak", **p)


def squeal(**kw):
    p = dict(dur=1.0, at=0.05, length_s=0.8, regime="squeal", body="gate", voice="hinge", irregularity=0.0,
             speed_change=0.0, burst_ms=2000.0, gap_ms=0.0, burst_spread=0.0, hop=0.0)
    p.update(kw)
    return run("creak", **p)


# ---- every recipe --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", CREAKS)
def test_recipe_renders_and_varies(name):
    a, b = render(name, 0), render(name, 1)
    assert np.all(np.isfinite(a)) and np.abs(a).max() > 0.1
    n = min(len(a), len(b))
    assert np.abs(a[:n] - b[:n]).max() > 0.01


@pytest.mark.parametrize("name", CREAKS)
def test_no_voice_buzz_and_no_hiss_bed(name):
    for k in range(3):
        y = render(name, k)
        assert buzz_share(y) < 0.05, (name, k)
        assert hiss_db(y) < -38, (name, k)


@pytest.mark.parametrize("name", CREAKS)
def test_nothing_rings_on_after_the_stroke(name):
    p = params(name)
    ring = 0.6 if p.get("tone_decay_db_s", 0) > 0 else 0.05     # the shoe tone decays away over ~0.55 s
    for k in range(3):
        y = render(name, k)
        end = min(int((p["at"] + 1.15 * p["length_s"] + ring) * SR), len(y) - int(0.05 * SR))
        assert rms(y[end:]) < rms(y) * 10 ** (-40 / 20), (name, k)


def test_rope_creak_is_gone():
    assert not (R / "rope_creak.json").exists() and not (R / "door_creak_wood.json").exists()


# ---- groan ---------------------------------------------------------------------------------------------------

def test_groan_click_rate_matches_the_heavy_door_and_rises_with_speed():
    base = click_rate(groan(speed=0.5), 0.2, 1.2)
    assert 100 < base < 220                                  # the CC0 heavy door: ~144 Hz
    assert click_rate(groan(speed=1.0), 0.2, 1.2) > 1.3 * base


def test_groan_resonances_stay_put_when_speed_doubles():
    for body in BODIES:
        a = envelope_peak_hz(groan(speed=0.3, body=body))
        b = envelope_peak_hz(groan(speed=0.6, body=body))
        assert abs(b / a - 1) < 0.1, (body, a, b)


def test_heavy_door_groan_peaks_near_its_fitted_formant_and_is_dark():
    y = groan(speed=0.5)
    assert 750 < envelope_peak_hz(y) < 1300                   # the CC0 heavy door: 930 Hz
    fr, X = spec_db(y[int(0.2 * SR):int(1.2 * SR)], 65536)
    P = 10 ** (X / 10)
    assert P[(fr > 3000) & (fr < 8000)].sum() / P[(fr > 100) & (fr < 8000)].sum() < 0.15


def test_groan_has_big_slips():
    """Big slips every ~0.115 s (+8.5 dB) give the groan its pulsing grain."""
    y = groan(speed=0.5)
    w = int(0.01 * SR)
    e = 20 * np.log10(np.sqrt(np.convolve(y ** 2, np.ones(w) / w, "same"))[int(0.2 * SR):int(1.2 * SR):w] + 1e-9)
    assert np.percentile(e, 95) - np.percentile(e, 20) > 5


def test_default_door_creak_is_a_groan_and_fast_swing_squeals():
    slow = render("door_creak", 0)
    fast = render("door_creak", 0, speed=0.9)
    assert buzz_share(slow) < 0.05
    # the slow swing's level sits around the heavy-door formant, the fast swing's squeal at 2-4 kHz
    assert envelope_peak_hz(slow) < 1400
    assert envelope_peak_hz(fast) > 1800


# ---- squeal --------------------------------------------------------------------------------------------------

def test_squeal_is_a_near_pure_tone_with_few_harmonics():
    y = squeal(speed=0.5)
    seg = y[int(0.3 * SR):int(0.3 * SR) + 4096]
    tf, pk = tone_to_floor(seg)
    assert tf > 40, tf                                          # the CC0 squeals: 40-86 dB
    assert harmonics_within(seg, pk) <= 8                       # the gate's first burst: 8 within 13 dB


@pytest.mark.parametrize("voice", ["door", "shoe"])
def test_door_and_shoe_squeals_have_two_to_four_harmonics(voice):
    y = squeal(voice=voice, speed=0.5, squeal_body=0.0, tick=0.0, scuff=0.0)
    seg = y[int(0.3 * SR):int(0.3 * SR) + 4096]
    _, pk = tone_to_floor(seg)
    assert abs(pk / VOICES[voice]["pitches"][0] - 1) < 0.08
    assert harmonics_within(seg, pk) <= 4


def test_squeal_pitch_glides_with_speed():
    lo = tone_to_floor(squeal(speed=0.5, speed_change=-1.0, glide_follow=1.0)[int(0.7 * SR):int(0.7 * SR) + 4096])[1]
    hi = tone_to_floor(squeal(speed=0.5, speed_change=1.0, glide_follow=1.0)[int(0.7 * SR):int(0.7 * SR) + 4096])[1]
    assert hi > 1.5 * lo


def test_hinge_squeaks_in_bursts_and_hops_pitch():
    for k in range(3):
        y = render("hinge_squeak", k)
        b = [s for s in bursts(y) if s[1] - s[0] > 0.05]
        assert len(b) >= 2, k                                   # the gate: 4 bursts of 0.3-0.9 s
        assert all(0.08 < e - s < 1.2 for s, e in b), b
    pks = set()
    for k in range(6):
        y = render("hinge_squeak", k)
        for s, e in bursts(y):
            if e - s > 0.1:
                m = int((s + 0.05) * SR)
                pks.add(round(tone_to_floor(y[m:m + 4096])[1], -2))
    assert len(pks) >= 2                                        # hops between hinge resonances


# ---- shoe ----------------------------------------------------------------------------------------------------

def test_shoe_squeak_ticks_then_a_decaying_2_4_khz_tone_with_its_odd_partial():
    for k in range(3):
        y = render("shoe_squeak", k)
        on = bursts(y, -30)[0][0]
        i = int(on * SR)
        tick = y[max(0, i - int(0.005 * SR)):i + int(0.004 * SR)]
        fr, X = spec_db(tick, 4096)
        P = 10 ** (X / 10)
        assert P[(fr > 5000) & (fr < 12000)].sum() / P[(fr > 200) & (fr < 12000)].sum() > 0.1, k   # broadband tick
        a = y[i + int(0.03 * SR):i + int(0.03 * SR) + 2048]
        tf, f0 = tone_to_floor(a)
        assert 2200 < f0 < 2700 and tf > 25, (k, f0, tf)
        fr, X = spec_db(a)
        odd = X[(fr > 1.3 * f0) & (fr < 1.42 * f0)].max()
        assert odd > X.max() - 15, k                            # the 1.34-1.38 f0 partial, strong at onset


def test_shoe_tone_decays_like_the_sneaker():
    p = {k: v for k, v in params("shoe_squeak").items() if k not in ("at", "length_s")}
    p.update(burst_ms=400.0, burst_spread=0.0, irregularity=0.0)
    y = run("creak", dur=1.0, at=0.05, length_s=0.3, **p)
    a = y[int(0.1 * SR):int(0.1 * SR) + 2048]
    b = y[int(0.35 * SR):int(0.35 * SR) + 2048]
    drop = 20 * np.log10(rms(a) / rms(b))
    assert 15 < drop < 40, drop                                 # the CC0 sneaker: 91-115 dB/s -> ~23-29 dB here
