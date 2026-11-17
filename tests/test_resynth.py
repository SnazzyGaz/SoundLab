"""v0.17 analysis-and-resynthesis: new blocks, analysis helpers, recipe builder, fit checks, CLI.
All signals are synthesised here; no reference audio is used or stored."""
import json

import numpy as np
import pytest

from soundlab import analysis as A
from soundlab import fit as F
from soundlab import resynth as RS
from soundlab.cli import main
from soundlab.models.resynth_blocks import THIRD_OCTAVES, eq_gain, formant_gain
from soundlab.recipe import render, validate

SR = 24000


def _r(layers, dur=1.0, post=None):
    return render({"layers": layers, "duration": dur, "seed": 3, "post": post or []}, SR)[0]


def _centroid(y):
    S = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return float((S * f).sum() / S.sum())


def test_bandnoise_follows_band_grid():
    lv = [-60.0] * len(THIRD_OCTAVES)
    lv[THIRD_OCTAVES.index(1000)] = 0.0
    y = _r([{"model": "bandnoise", "params": {"frames": [[0, lv]]}}])
    assert 800 < _centroid(y) < 1300
    # level falls 40 dB between two frames
    y = _r([{"model": "bandnoise", "params": {"frames": [[0, [0.0] * 26], [0.5, [-40.0] * 26]]}}])
    a, b = np.std(y[int(0.02 * SR):int(0.1 * SR)]), np.std(y[int(0.6 * SR):int(0.9 * SR)])
    assert 30 < 20 * np.log10(a / b) < 50


def test_bandnoise_tilt_and_stretch():
    flat = [[0, [0.0] * 26]]
    dark = _r([{"model": "bandnoise", "params": {"frames": flat, "tilt_db": -6}}])
    bright = _r([{"model": "bandnoise", "params": {"frames": flat, "tilt_db": 6}}])
    assert _centroid(bright) > 2 * _centroid(dark)
    fr = [[0, [0.0] * 26], [0.25, [-60.0] * 26]]
    def seg(y):
        return np.std(y[int(0.3 * SR):int(0.4 * SR)]) / np.std(y[int(0.02 * SR):int(0.05 * SR)])
    y1 = _r([{"model": "bandnoise", "params": {"frames": fr}}])
    y2 = _r([{"model": "bandnoise", "params": {"frames": fr, "stretch": 2}}])
    assert seg(y2) > 5 * seg(y1)


def test_harmonic_pitch_formants_and_voices():
    y = _r([{"model": "harmonic", "params": {"f0": 300.0, "breath": 0.0}}])
    f0 = A.f0_estimate(y, SR, return_info=True)['f0_hz']
    assert abs(1200 * np.log2(f0 / 300.0)) < 5
    y = _r([{"model": "harmonic", "params": {"f0": 200.0, "voices": 4, "detune_cents": 10, "breath": 0.2}}])
    assert abs(1200 * np.log2(A.f0_estimate(y, SR, return_info=True)['f0_hz'] / 200.0)) < 15
    lo = _r([{"model": "harmonic", "params": {"f0": 150.0, "formants": [[500, 100, 0]], "breath": 0}}])
    hi = _r([{"model": "harmonic", "params": {"f0": 150.0, "formants": [[2500, 200, 0]], "breath": 0}}])
    assert _centroid(hi) > 2 * _centroid(lo)


def test_formant_and_eq_gain():
    g = formant_gain(np.array([500.0, 1000.0, 3000.0]), [[1000, 100, 0]])
    assert g[1] > 5 * g[0] and g[1] > 5 * g[2]
    assert np.allclose(eq_gain([100.0, 1000.0], []), 1)
    assert np.isclose(eq_gain([1000.0], [[500, 0], [1000, 6], [2000, 0]])[0], 10 ** 0.3)


def test_harmonic_follows_pitch_curve():
    y = _r([{"model": "harmonic", "params": {"f0": [[0, 300], [1, 600]], "breath": 0}}])
    t, f, *_ = A.f0_track(y, SR)
    mid = np.nanmedian(f[(t > 0.45) & (t < 0.55)])
    assert abs(1200 * np.log2(mid / (300 * 2 ** 0.5))) < 60  # log interpolation: geometric midpoint


def test_simplify_frames_tolerance():
    t = np.linspace(0, 1, 101)
    G = np.stack([-60 * t, -30 * t ** 2], 1)
    fr = RS.simplify_frames(t, G, tol_db=1.0)
    ks = [p[0] for p in fr]
    approx = np.stack([np.interp(t, ks, [p[1][j] for p in fr]) for j in range(2)], 1)
    assert np.abs(approx - G).max() <= 1.1
    assert len(fr) < 20


def test_suppress_tonal_removes_chirp_keeps_noise():
    rng = np.random.default_rng(0)
    n = 2 * SR
    noise = rng.standard_normal(n) * 0.1
    t = np.arange(n) / SR
    chirp = 0.3 * np.sin(2 * np.pi * (3000 * t + 400 * t ** 2)) * (t < 1.5)
    y, removed = RS.suppress_tonal(noise + chirp, SR)
    resid = y - noise
    assert np.sum(resid ** 2) < 0.1 * np.sum(chirp ** 2)
    y2, removed2 = RS.suppress_tonal(noise, SR)
    assert removed2 < 0.03


def test_harmonic_sum_f0_on_rich_tone():
    t = np.arange(2 * SR) / SR
    x = sum(np.sin(2 * np.pi * 147 * k * t) / k for k in range(1, 10))
    _, f0, _ = RS.harmonic_sum_f0(x, SR)
    assert abs(1200 * np.log2(np.median(f0) / 147)) < 30


def _impact_ref():
    rng = np.random.default_rng(1)
    n = int(1.2 * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n) * np.exp(-t / 0.15) * (t > 0.05)
    from soundlab import dsp
    return dsp.lowpass(x, 3000, SR)


def test_build_impact_recipe_is_editable_and_close():
    x = _impact_ref()
    rec = RS.build_recipe(x, SR, "impact", "burst")
    assert validate(rec) == []
    assert [l["model"] for l in rec["layers"]][:2] == ["bandnoise", "splash"]
    assert set(rec["macros"]) >= {"level", "brightness", "length"}
    y = RS.render(rec)
    assert F.compare_signals(x, SR, y, F.FIT_SR)["total"] < 0.35


def test_fit_and_check_reports_holdout_and_seed():
    x = _impact_ref()
    rec = RS.build_recipe(x, SR, "impact", "burst")
    res = RS.fit_and_check(rec, x, SR, "impact", fit_section=(0, 0.7),
                           holdout=[("tail", x, SR, 0.7, 1.2, "timeline")], budget=8)
    assert res["fit_after"]["total"] <= res["fit_before"]["total"] + 1e-9
    assert res["holdout"][0]["name"] == "tail" and np.isfinite(res["holdout"][0]["total"])
    assert np.isfinite(res["full_other_seed"]["total"])
    assert res["recipe"]["duration"] == rec["duration"]


def test_voice_resynthesis_gets_pitch_first():
    ref = _r([{"model": "harmonic", "params": {"f0": [[0, 500], [0.6, 620], [1.0, 540]],
                                               "formants": [[1200, 300, 0], [3000, 400, -6]], "breath": 0.05,
                                               "level": [[0, 0], [0.1, 1], [0.9, 1], [1.0, 0]]}}], dur=1.0)
    rec = RS.build_recipe(ref, SR, "voice", "tone")
    assert validate(rec) == []
    res = RS.fit_and_check(rec, ref, SR, "voice", budget=6)
    assert res["pitch_error_cents"] is not None and res["pitch_error_cents"] < 30
    assert res["recipe"]["layers"][0]["params"].get("eq") is not None or res["full"]["total"] < 0.4


def test_auto_kind():
    base = {"f0": {"f0_hz": None, "voiced_fraction": 0}, "tonal_share": 0.1, "duration_s": 1.0, "onsets_s": [0.1]}
    assert RS.auto_kind(base) == "impact"
    assert RS.auto_kind({**base, "duration_s": 10, "onsets_s": []}) == "texture"
    assert RS.auto_kind({**base, "f0": {"f0_hz": 500, "voiced_fraction": 0.8}}) == "voice"
    assert RS.auto_kind({**base, "tonal_share": 0.7, "duration_s": 10, "onsets_s": []}) == "choir"


def test_cli_resynth(tmp_path):
    from soundlab.audio_io import write_wav
    ref = tmp_path / "burst.wav"
    write_wav(ref, _impact_ref() * 0.5, SR)
    out = tmp_path / "rs"
    assert main(["resynth", str(ref), "--kind", "impact", "--budget", "4", "--holdout-section", "0.7:1.2",
                 "-o", str(out)]) in (0, None)
    rep = json.loads((out / "report.json").read_text())
    assert rep["kind"] == "impact" and rep["holdout"]
    assert validate(json.loads((out / "recipe.json").read_text())) == []
    assert (out / "resynth.wav").exists()
