"""v0.6: generic synthesis toolkit (osc, fm, noisegen, grains, svf, env, shaper, delay, chorus, reverb, gain)."""
import numpy as np

import soundlab
from conftest import SR, peak_freq, run


def _inharmonic_db(y, f0, fmin=2000.0, tol=3.0):
    y = y * np.hanning(len(y))
    Y = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    sel = f > fmin
    h = np.abs(f / f0 - np.round(f / f0)) * f0 < tol
    return 10 * np.log10(Y[sel & ~h].sum() / (Y[sel].sum() + 1e-30) + 1e-30)


def test_osc_pitch_and_bandlimiting():
    for wave in ("saw", "square", "triangle", "pulse"):
        y = run("osc", 1.0, wave=wave, freq=1234.5)
        assert abs(peak_freq(y, lo=200) - 1234.5) < 3, wave
        # naive digital saw at 1.2 kHz: -11 dB; plain 1x PolyBLEP -26 dB; ours (2x + steep FIR) about -43 dB
        assert _inharmonic_db(y, 1234.5) < -38, wave
    assert abs(peak_freq(run("osc", 1.0, wave="sine", freq=440)) - 440) < 2


def test_naive_saw_would_alias():
    # sanity check of the measure itself
    f0 = 1234.5
    t = np.arange(SR) / SR
    naive = 2 * ((f0 * t) % 1) - 1
    assert _inharmonic_db(naive, f0) > -25


def test_osc_sweep_follows_automation():
    y = run("osc", 1.0, wave="sine", freq=[[0, 200], [1, 3200]])
    assert peak_freq(y[:4800], lo=50) < 300
    assert peak_freq(y[-4800:], lo=50) > 2500


def test_fm_sidebands():
    y = run("fm", 1.0, freq=500, ratio=1.0, index=0.0)
    assert abs(peak_freq(y) - 500) < 2
    y2 = run("fm", 1.0, freq=500, ratio=1.0, index=3.0)
    Y = np.abs(np.fft.rfft(y2 * np.hanning(SR)))
    assert Y[1500] > 0.1 * Y.max()  # 3rd harmonic sideband present


def test_noise_colours_slope():
    def slope(c):
        y = run("noisegen", 2.0, color=c)
        Y = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / SR)
        lo = Y[(f > 200) & (f < 400)].mean()
        hi = Y[(f > 6400) & (f < 12800)].mean()
        return 10 * np.log10(hi / lo) / 5  # dB per octave over 5 octaves
    assert abs(slope("white")) < 1
    assert -4.5 < slope("pink") < -1.5
    assert slope("brown") < -4.5
    assert 1.5 < slope("blue") < 4.5
    assert slope("violet") > 4.5


def test_sample_hold_noise_is_stepped():
    y = run("noisegen", 0.5, color="white", hold_hz=1000)
    assert np.mean(np.diff(y) == 0) > 0.95


def test_svf_modes():
    x = run("noisegen", 1.0, color="white")
    def band(y, lo, hi):
        Y = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / SR)
        return 10 * np.log10(Y[(f > lo) & (f < hi)].mean())
    lp = run("svf", 1.0, x, mode="lp", cutoff=500)
    hp = run("svf", 1.0, x, mode="hp", cutoff=5000)
    bp = run("svf", 1.0, x, mode="bp", cutoff=2000, resonance=5)
    assert band(lp, 100, 300) - band(lp, 5000, 10000) > 30
    assert band(hp, 8000, 15000) - band(hp, 300, 800) > 30
    assert band(bp, 1900, 2100) - band(bp, 200, 400) > 20


def test_env_adsr_shape():
    x = np.ones(SR)
    e = run("env", 1.0, x, attack=0.01, decay=0.1, sustain=0.5, release=0.2, gate_s=0.5, curve="exp")
    assert e[0] == 0 and abs(e[int(0.01 * SR)] - 1) < 0.05
    assert abs(e[int(0.4 * SR)] - 0.5) < 0.02
    assert e[int(0.75 * SR)] < 0.5 * 10 ** (-40 / 20) * 10  # well into the release
    p = run("env", 1.0, x, points=[[0, 0], [0.1, 1], [0.5, 0.0]], curve="linear")
    assert abs(p[int(0.3 * SR)] - 0.5) < 0.01


def test_shaper_adds_harmonics_and_crushes():
    x = run("osc", 1.0, wave="sine", freq=300)
    y = run("shaper", 1.0, x, mode="tanh", drive=8)
    Y = np.abs(np.fft.rfft(y * np.hanning(SR)))
    assert Y[900] > 0.05 * Y[300]
    c = run("shaper", 1.0, x, mode="bitcrush", bits=3, drive=1)
    assert len(np.unique(np.round(c, 6))) <= 9


def test_delay_and_reverb_tails():
    x = np.zeros(SR)
    x[:100] = 1.0
    d = run("delay", 1.0, x, time_ms=200, feedback=0.5, mix=1.0)
    assert abs(d[int(0.2 * SR) + 50]) > 0.3 and abs(d[int(0.4 * SR) + 50]) > 0.1
    r = run("reverb", 1.0, x, size=0.8, mix=1.0)
    assert np.sqrt(np.mean(r[int(0.3 * SR):int(0.5 * SR)] ** 2)) > 1e-4


def test_grains_density_follows_rate():
    lo = run("grains", 1.0, rate=10, grain_ms=5, freq=3000)
    hi = run("grains", 1.0, rate=400, grain_ms=5, freq=3000)
    assert np.mean(np.abs(hi) > 1e-3) > 3 * np.mean(np.abs(lo) > 1e-3)


def test_block_recipes_render(tmp_path):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "recipes" / "blocks"
    files = sorted(root.glob("*.json"))
    assert len(files) >= 6
    for f in files:
        rec = soundlab.load_recipe(str(f))
        assert soundlab.validate(rec) == [], f.name
        # generic-blocks-only demos
        from soundlab.core import REGISTRY
        for layer in rec["layers"]:
            for b in layer.get("chain") or [layer]:
                assert REGISTRY[b["model"]].stage in ("synth", "fx"), (f.name, b["model"])
        y, sr = soundlab.render(rec, sample_rate=24000)
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) > 0.5


def test_fast_svf_matches_loop_for_constant_cutoff():
    from soundlab import dsp
    x = np.random.default_rng(2).standard_normal(SR)
    for mode in ("lp", "bp", "hp"):
        a = dsp.svf_loop(x, np.full(SR, 1234.0), 2.0, SR, mode)
        b = dsp.svf(x, 1234.0, 2.0, SR, mode)
        assert np.max(np.abs(a - b)) < 1e-9


def test_audition_renders_preview(tmp_path):
    from pathlib import Path
    from soundlab.batch import audition
    root = Path(__file__).resolve().parents[1]
    p = audition(root / "recipes" / "blocks" / "ui_click.json", {"pitch": 0.8}, out_dir=tmp_path, do_play=False,
                 log=lambda m: None)
    assert p.exists()
