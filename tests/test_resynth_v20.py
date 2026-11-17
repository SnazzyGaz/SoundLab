"""v0.20: hybrid splash tail, human choir voices (Rosenberg pulse, jitter/shimmer, vibrato onset), conv hall, drone."""
import copy

import numpy as np

import soundlab.recipe as R
from soundlab import resynth as RS
from soundlab.models.resynth_blocks import rosenberg_ripple

SR = 24000


def _r(layers, dur=2.0, seed=2):
    y, _ = R.render({"duration": dur, "seed": seed, "sample_rate": SR, "layers": layers,
                     "master": {"normalize_db": None}}, SR)
    return np.asarray(y, float)


def test_rosenberg_ripple_is_unit_rms_and_changes_with_open_quotient():
    a, b = rosenberg_ripple(30, 0.4), rosenberg_ripple(30, 0.8)
    assert np.sqrt(np.mean(a ** 2)) == np.sqrt(np.mean(a ** 2))  # finite
    assert abs(np.sqrt(np.mean(a ** 2)) - 1) < 0.2
    assert np.max(np.abs(np.log(a) - np.log(b))) > 0.1


def test_harmonic_defaults_unchanged_and_new_controls_render():
    base = {"model": "harmonic", "params": {"f0": 200, "voices": 3, "vibrato_cents": 10}}
    y0 = _r([base])
    y1 = _r([{"model": "harmonic", "params": dict(base["params"], glottal="none", jitter_cents=0.0)}])
    assert np.array_equal(y0, y1)
    yh = _r([{"model": "harmonic", "params": dict(base["params"], glottal="rosenberg", jitter_cents=8,
                                                   shimmer_db=1.0, vibrato_onset=0.5, voice_spread=0.05)}])
    assert np.isfinite(yh).all() and not np.allclose(yh, y0)


def test_vibrato_onset_starts_without_vibrato():
    p = {"f0": 220, "vibrato_cents": 80, "vibrato_hz": 5, "max_hz": 3000}
    late = _r([{"model": "harmonic", "params": dict(p, vibrato_onset=1.5)}], 3.0)
    from soundlab import analysis as A
    t, f0 = A.f0_track(late, SR)[:2]
    t, f0 = np.asarray(t), np.asarray(f0)
    ok = np.isfinite(f0) & (f0 > 0)
    early = f0[ok & (t > 0.1) & (t < 0.4)]
    after = f0[ok & (t > 2.0) & (t < 2.9)]
    assert np.std(1200 * np.log2(early / 220)) < np.std(1200 * np.log2(after / 220))


def test_conv_reverb_adds_no_noise_and_decays():
    sil = _r([{"chain": [{"model": "harmonic", "params": {"f0": 200, "level": 0.0}},
                         {"model": "conv_reverb", "params": {"rt60": 2.0}}]}])
    assert np.max(np.abs(sil)) < 1e-9
    y = _r([{"chain": [{"model": "harmonic", "params": {"f0": 200, "level": [[0, 1], [0.5, 1], [0.52, 0]]}},
                       {"model": "conv_reverb", "params": {"rt60": 1.0, "mix": 0.8}}]}], 3.0)
    e1 = np.sqrt(np.mean(y[int(0.7 * SR):int(0.9 * SR)] ** 2))
    e2 = np.sqrt(np.mean(y[int(1.7 * SR):int(1.9 * SR)] ** 2))
    assert 20 * np.log10(e1 / e2) > 30  # ~60 dB per rt60 (1 s)


def test_drone_pitch_finds_a_held_low_note():
    t = np.arange(4 * SR) / SR
    mel = np.sin(2 * np.pi * (220 + 40 * np.sign(np.sin(2 * np.pi * 0.5 * t))) * t)
    x = mel + 0.7 * np.sin(2 * np.pi * 110 * t)
    fd, lv = RS.drone_pitch(x, SR, 230.0)
    assert abs(fd - 110) < 3
    assert RS.drone_pitch(mel, SR, 230.0) is None


def test_bubble_tail_keeps_the_onset():
    sr = SR
    rng = np.random.default_rng(0)
    n = int(2.0 * sr)
    x = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.15 * sr))
    x[:int(0.05 * sr)] = 0
    feats = RS.analyse_clip(x, sr)
    rec = RS.build_recipe(x, sr, "impact", "t", feats=feats)
    ts = RS.tail_start(x, sr, rec["strong_onsets_s"])
    assert ts is not None
    out = RS.add_bubble_tail(rec, x, sr, ts, budget=6)
    assert out["layers"][-1]["name"] == "tail_bubbles" and out["layers"][-1]["start"] == ts
    a, b = RS.render(rec), RS.render(out)
    k = int(ts * RS.F.FIT_SR)
    ca = a[:k] / np.max(np.abs(a)); cb = b[:k] / np.max(np.abs(b))
    assert np.corrcoef(ca, cb)[0, 1] > 0.99


def test_impact_fit_leaves_sparsity_to_the_tail():
    assert RS.FIT_WEIGHTS["impact"]["sparsity"] == 0
