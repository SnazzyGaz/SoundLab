"""Blind checkpoint kit: build a shuffled A/B set and score it."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from soundlab.audio_io import write_wav

spec = importlib.util.spec_from_file_location("checkpoint", Path(__file__).resolve().parents[1] / "tools" / "checkpoint.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


def _tone(f, a, sr=48000, d=0.5):
    t = np.arange(int(sr * d)) / sr
    return a * np.sin(2 * np.pi * f * t) * np.exp(-t * 4)


def test_build_and_score(tmp_path):
    eng, lib = tmp_path / "eng", tmp_path / "lib"
    (lib / "rain").mkdir(parents=True)
    eng.mkdir()
    for i in range(2):
        write_wav(eng / f"ambience_rain_v{i + 1:02d}.wav", _tone(300 + i, 0.9), 48000)
        write_wav(eng / f"door_door_slam_v{i + 1:02d}.wav", _tone(100, 0.9), 48000)
        write_wav(lib / "rain" / f"take{i}.wav", _tone(500, 0.05), 48000)
        write_wav(lib / f"door_slam__take{i}.wav", _tone(120, 0.2), 44100)
    out = tmp_path / "set"
    key = cp.build(eng, lib, out, per_sound=5, seed=3)
    assert len(key["trials"]) == 10 and {t["sound"] for t in key["trials"]} == {"rain", "door_slam"}
    assert len(list((out / "trials").glob("*.wav"))) == 20
    # loudness matched within 1 dB on the active part
    a, _ = cp.read_audio(out / "trials" / "trial_01_A.wav")
    b, _ = cp.read_audio(out / "trials" / "trial_01_B.wav")
    r = lambda x: 20 * np.log10(np.sqrt(np.mean(x[np.abs(x) > 0.01 * np.abs(x).max()] ** 2)))
    assert abs(r(a) - r(b)) < 1.5
    # a listener who always spots the engine fails; a coin-flipper passes
    rows = list(csv.DictReader(open(out / "answers.csv")))
    truth = {t["trial"]: t["engine"] for t in key["trials"]}
    for mode, expect in (("perfect", False), ("wrong", True)):
        for row in rows:
            e = truth[row["trial"]]
            row["engine_is (A/B)"] = e if mode == "perfect" else ("B" if e == "A" else "A")
        p = out / f"ans_{mode}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        res = cp.score(out / "key.json", p)
        assert res["overall_physical"]["pass"] is expect
        assert "Overall" in cp.report(res)


def test_stats():
    lo, hi = cp.wilson(5, 10)
    assert lo < 0.5 < hi
    assert abs(cp.binom_p_greater(10, 10) - 1 / 1024) < 1e-12
    assert cp.binom_p_greater(0, 10) == 1.0
