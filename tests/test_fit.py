"""Parameter fitting (analysis-by-synthesis) and the match/compare CLI."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import soundlab
from soundlab import fit as F
from soundlab.audio_io import write_wav

ROOT = Path(__file__).resolve().parents[1]


def noise_recipe(f_lo=300.0, f_hi=6000.0, release=0.3):
    return {"soundlab": 1, "sample_rate": 24000, "duration": 1.0, "seed": 3,
            "layers": [{"model": "noise", "params": {"color": "white", "f_lo": f_lo, "f_hi": f_hi, "at": 0.05,
                                                      "attack": 0.005, "hold": 0.1, "release": release}}],
            "master": {"normalize_db": -1.0}}


def test_distance_zero_for_identical_and_grows():
    y, sr = soundlab.render(noise_recipe(), 24000)
    t = F.Target.from_signal(F.prepare(y, sr), F.FIT_SR)
    same = F.distance(t, F.prepare(y, sr))
    other = F.distance(t, F.prepare(soundlab.render(noise_recipe(2000, 9000, 1.5), 24000)[0], sr))
    assert same["total"] < 0.02
    assert other["total"] > same["total"] + 0.2
    assert other["spectral"] > 0.1 and other["envelope"] > 0.1


def test_pitch_term_only_with_pitched_reference():
    t = np.arange(24000) / 24000
    tone = np.sin(2 * np.pi * 200 * t)
    tgt = F.Target.from_signal(tone, F.FIT_SR)
    assert tgt.f0 and abs(tgt.f0 - 200) < 3
    assert F.distance(tgt, np.sin(2 * np.pi * 400 * t))["pitch"] > 0.9  # one octave off
    nt = F.Target.from_signal(np.random.default_rng(0).standard_normal(24000), F.FIT_SR)
    assert nt.f0 is None and F.distance(nt, tone)["pitch"] == 0.0


def test_param_parsing_and_units():
    r = noise_recipe()
    p = F.parse_param("0.f_lo", r)
    assert p.log and p.lo == 10 and p.hi == 20000
    q = F.parse_param("0.release:0.05:2", r)
    assert (q.lo, q.hi) == (0.05, 2.0)
    for v in (0.07, 0.3, 1.9):
        assert abs(q.from_unit(q.to_unit(v)) - v) < 1e-9
    g = F.parse_param("0.gain_db", r)
    assert g.lo == -30
    names = {fp.name for fp in F.default_params(r)}
    assert "f_lo" in names and "at" not in names and "color" not in names


def test_cmaes_minimises_quadratic():
    x, fx, n = F.cmaes(lambda u: float(np.sum((u - np.array([0.2, 0.7, 0.5])) ** 2)), np.array([0.5, 0.5, 0.5]),
                       budget=300, seed=1)
    assert fx < 1e-3 and n <= 300


def test_fit_recovers_band_and_decay():
    ref, sr = soundlab.render(noise_recipe(2000, 8000, 0.8), 24000)
    # (f_lo and f_hi together are not identifiable: the noise block accepts them swapped)
    res = F.fit(noise_recipe(300, 8000, 0.2), ref, sr, ["0.f_lo:100:5000", "0.release:0.05:2"],
                budget=60, log=lambda m: None)
    assert res["after"]["total"] < 0.5 * res["before"]["total"]
    got = {p["key"]: p["after"] for p in res["params"]}
    assert 1000 < got["0.f_lo"] < 3500
    assert 0.4 < got["0.release"] < 1.5


def test_match_and_compare_cli(tmp_path):
    ref, sr = soundlab.render(noise_recipe(1500, 7000, 0.6), 24000)
    write_wav(tmp_path / "ref.wav", ref, sr)
    rp = tmp_path / "base.json"
    rp.write_text(json.dumps(noise_recipe(300, 3000, 0.2)))
    out = tmp_path / "fit"
    cmd = [sys.executable, "-m", "soundlab", "match", str(tmp_path / "ref.wav"), "--recipe", str(rp),
           "-p", "0.f_lo:100:5000", "-p", "0.release:0.05:2", "--budget", "30", "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout)
    assert info["after"]["total"] <= info["before"]["total"]
    for f in ("fitted.json", "report.md", "report.json", "comparison.png", "fitted_preview.wav"):
        assert (out / f).exists()
    assert soundlab.validate(json.loads((out / "fitted.json").read_text())) == []
    r2 = subprocess.run([sys.executable, "-m", "soundlab", "compare", str(tmp_path / "ref.wav"),
                         str(out / "fitted_preview.wav"), "--json"], capture_output=True, text=True, cwd=ROOT)
    assert r2.returncode == 0, r2.stderr
    assert "total" in list(json.loads(r2.stdout).values())[0]
