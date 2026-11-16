"""Bias-free scoring of single-clip packs (balanced accuracy, d', bias c, bootstrap interval) and answer codes
from the static listening page."""
import base64
import json

import pytest

from soundlab import cli
from soundlab.checkpoint import decode_code, import_codes, score_pack, sdt, verdict


def _pack(tmp_path, n_sounds=6):
    clips = []
    for s in range(n_sounds):
        for t in ("engine", "engine", "real", "real"):
            clips.append({"clip": f"clip_{len(clips) + 1:03d}", "sound": f"s{s}", "truth": t, "source": "x"})
    p = tmp_path / "pack"
    (p / "answers").mkdir(parents=True)
    (p / "key.json").write_text(json.dumps({"version": 2, "mode": "single", "exempt": [], "clips": clips}))
    return p, clips


def _answer(p, who, clips, rule):
    ans = {c["clip"]: {"answer": rule(c)} for c in clips}
    (p / "answers" / f"{who}.json").write_text(json.dumps({"listener": who, "answers": ans}))


def _code(n, answers, handle=""):
    nb = (n + 3) // 4
    b = bytearray([1, 1, n] + [0] * nb + [0]) + handle.encode()
    for i, a in enumerate(answers):
        b[3 + i // 4] |= {"real": 1, "synth": 2}.get(a, 0) << ((i % 4) * 2)
    b[3 + nb] = sum(x for i, x in enumerate(b) if i != 3 + nb) & 255
    return "SL1." + base64.urlsafe_b64encode(bytes(b)).decode().rstrip("=")


def test_sdt_basics():
    r = sdt(10, 10, 0, 10)
    assert r["balanced_accuracy"] == 1.0 and r["d_prime"] > 2 and abs(r["c"]) < 1e-9
    r = sdt(8, 10, 8, 10)                                    # leans 'synth', hears nothing
    assert r["balanced_accuracy"] == 0.5 and abs(r["d_prime"]) < 1e-9 and r["c"] < -0.5
    assert verdict([0.55, 0.8]) == "FAIL" and verdict([0.45, 0.58]) == "PASS" and verdict([0.4, 0.7]) == "pass?"


def test_synth_leaning_guesser_is_not_a_fail(tmp_path):
    p, clips = _pack(tmp_path)
    _answer(p, "lean", clips, lambda c: "synth")             # everything called synthesised
    o = score_pack(p)["pooled"]["overall_physical"]
    assert o["pct"] == 1.0 and o["old_pass"] is False         # the old figure failed this
    assert o["balanced_accuracy"] == 0.5 and o["pass"] is True


def test_discriminating_listeners_fail_and_bias_is_reported(tmp_path):
    p, clips = _pack(tmp_path, 10)
    _answer(p, "a", clips, lambda c: "synth" if c["truth"] == "engine" else "real")
    _answer(p, "b", clips, lambda c: "synth" if c["truth"] == "engine" or c["clip"] < "clip_010" else "real")
    res = score_pack(p, reps=1000)
    o = res["pooled"]["overall_physical"]
    assert o["verdict"] == "FAIL" and o["ba_ci95"][0] > 0.5
    assert res["per_listener"]["b"]["c"] < res["per_listener"]["a"]["c"]
    assert set(res["pooled"]["per_sound"]) == {f"s{i}" for i in range(10)}


def test_answer_codes_round_trip_and_cli(tmp_path, capsys):
    p, clips = _pack(tmp_path)
    picks = ["synth" if c["truth"] == "engine" else "real" for c in clips]
    picks[3] = None                                           # one unanswered
    code = _code(len(clips), picks, "@some_one")
    d = decode_code(code)
    assert d["handle"] == "@some_one" and d["n"] == len(clips) and len(d["answers"]) == len(clips) - 1
    bad = code[:-3] + ("A" if code[-3] != "A" else "B") + code[-2:]
    with pytest.raises(ValueError):
        decode_code(bad)
    other = _code(len(clips) + 4, picks, "x")                 # a different pack
    f = tmp_path / "codes.txt"
    f.write_text(f"thanks! here it is: {code}\nand {other}")
    assert cli.main(["blindtest", "score", str(p), "--codes", str(f), "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["listeners"] == ["some_one"] and res["pooled"]["overall_physical"]["balanced_accuracy"] == 1.0
    assert import_codes(p, code, log=lambda m: None) == []    # already imported
