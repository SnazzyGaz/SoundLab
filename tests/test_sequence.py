"""Sequences (soundlab/sequence.py): plan determinism, distinct takes per event, timing, the event cache (timing-only
edits re-mix), export (mix, one-shots, cue CSV), and the CLI."""
import csv
import json

import numpy as np
import pytest
from scipy.io import wavfile

from soundlab import sequence as S
from soundlab.cli import main as cli_main
from soundlab.core import RecipeError
from soundlab.models import door_stems, recstems
from soundlab.recipe import render as render_recipe

FAST = "impact_any"         # a quick synthesis entry for the engine tests


def seq(steps, **kw):
    return {"soundlab": 1, "type": "sequence", "name": "t", "category": "impact", "seed": 3,
            "rate": {"interval": 0.3}, "level": "peak", "steps": steps, **kw}


# ------------------------------------------------------------------ presets and planning (no audio)
def test_presets_are_valid_and_use_library_entries():
    ps = S.presets()
    assert {"walk", "knock", "door_open_close", "door_open_close_lock", "gunfire", "reload"} <= {p["id"] for p in ps}
    for p in ps:
        assert not S.validate(p["sequence"]), p["id"]
        for st in p["sequence"]["steps"]:
            raw = json.loads(S.default_resolver(st["entry"]).read_text())
            assert "layers" in raw and not raw.get("hidden"), (p["id"], st["entry"])
        assert S.plan(p["sequence"])


def test_plan_is_deterministic_for_the_seed():
    a = [(e["entry"], e["take"]) for e in S.plan(S.preset("gunfire"))]
    b = [(e["entry"], e["take"]) for e in S.plan(S.preset("gunfire"))]
    assert a == b
    s2 = S.preset("gunfire")
    s2["seed"] = 99
    assert [(e["entry"], e["take"]) for e in S.plan(s2)] != a


def test_timing_and_repeat_changes_keep_the_takes():
    s1 = S.preset("walk")
    s2 = json.loads(json.dumps(s1))
    s2["rate"] = {"spm": 150}
    s2["humanise"] = {"timing_ms": 40, "level_db": 4, "alternate": 0.7}
    s2["steps"][0]["repeat"] = 10
    t1 = [e["take"] for e in S.plan(s1)]
    t2 = [e["take"] for e in S.plan(s2)]
    assert t2[:len(t1)] == t1


@pytest.mark.parametrize("name,n", [("walk", 24), ("knock", 12), ("gunfire", 10)])
def test_no_two_consecutive_events_of_an_entry_share_a_take(name, n):
    s = S.preset(name)
    for st in s["steps"]:
        if int(st.get("repeat", 1)) > 1:
            st["repeat"] = n
    last = {}
    distinct_gestures = {}
    for e in S.plan(s):
        if e["entry"] in last:
            pv, pg = last[e["entry"]]
            assert e["take"] != pv
            assert e["gesture"] != pg, (name, e["entry"])      # these sources have several recorded gestures/hits
        last[e["entry"]] = (e["take"], e["gesture"])
        distinct_gestures.setdefault(e["entry"], set()).add(e["gesture"])
    assert all(len(g) >= 2 for g in distinct_gestures.values())


def test_single_gesture_entry_still_gets_fresh_takes():
    s = seq([{"entry": "doors/door_wood_close", "repeat": 4}])
    ev = S.plan(s)
    assert len({e["gesture"] for e in ev}) == 1          # one recorded gesture
    takes = [e["take"] for e in ev]
    assert all(a != b for a, b in zip(takes, takes[1:]))  # every event a different remix of it


def test_fixed_take_is_used_for_every_event():
    ev = S.plan(seq([{"entry": FAST, "repeat": 3, "take": 5}]))
    assert [e["take"] for e in ev] == [5, 5, 5]


def test_validation_errors():
    assert S.validate({"type": "sequence", "steps": []})
    assert S.validate(seq([{"entry": FAST, "at": 0, "gap": 1}]))
    assert S.validate(seq([{"entry": FAST, "take": "x"}]))
    with pytest.raises(RecipeError):
        S.plan(seq([{"entry": "../etc/passwd"}]))


def test_knock_hit_is_a_window_of_the_recorded_gesture():
    st = recstems.get_source(door_stems._data(), "knock_wood")
    g = recstems.kept_gestures(st)[0]
    ons = g["onsets"]
    rng = np.random.default_rng(1)
    for h in range(len(ons)):
        (a, e, t), = recstems.plan_pieces(st, {"count": 1, "hit": h, "which": -1}, rng)
        assert t == 0.0 and a <= ons[h] < e
        assert all(not (a < o < e) for j, o in enumerate(ons) if j != h)
    whole = recstems.plan_pieces(st, {"count": 1, "hit": -1, "which": -1}, rng)
    assert whole == [(float(g["a"]), float(g["end"]), 0.0)]


def test_take_key_is_what_render_plays():
    raw = json.loads(S.default_resolver("doors/knock_wood").read_text())
    r, seed = S.concrete(raw, {}, {"0.hit": -2}, 4242)
    key = S.take_key(r)
    render_recipe(r, seed=seed)
    assert key[0][1:] == tuple((round(a, 4), round(e, 4)) for a, e, _ in recstems.LAST["pieces"])


# ------------------------------------------------------------------ rendering, timing and the event cache
def test_render_is_deterministic_and_timing_lands_on_the_grid():
    S.EVENTS.clear()
    s = seq([{"entry": FAST, "repeat": 3},
             {"entry": FAST, "gap": 0.25},
             {"entry": "sword_swing", "at": 2.0}])
    y1, sr, i1 = S.render(s)
    y2, _, i2 = S.render(s)
    assert np.array_equal(y1, y2) and i1["events"] == i2["events"]
    t = [e["time"] for e in i1["events"]]
    lead = t[0]
    assert np.allclose(np.array(t[:3]) - lead, [0.0, 0.3, 0.6], atol=1e-3)
    end3 = i1["events"][2]["start"] + i1["events"][2]["seconds"]
    assert t[3] == pytest.approx(end3 + 0.25, abs=2e-3)
    assert t[4] - lead == pytest.approx(2.0, abs=1e-3)


def test_timing_only_edits_re_mix_cached_events():
    S.EVENTS.clear()
    s = seq([{"entry": FAST, "repeat": 4}, {"entry": "sword_swing", "gap": 0.1}])
    y1, _, i1 = S.render(s)
    assert i1["events_rendered"] == 5 and i1["events_cached"] == 0
    for edit in ({"rate": {"interval": 0.45}}, {"humanise": {"timing_ms": 30, "level_db": 3, "alternate": 0.5}}):
        s2 = dict(s, **edit)
        y2, _, i2 = S.render(s2)
        assert i2["events_rendered"] == 0 and i2["events_cached"] == 5
        assert y2.shape != y1.shape or not np.array_equal(y1, y2)
    s3 = json.loads(json.dumps(s))
    s3["steps"][0]["gain_db"] = -6
    s3["steps"][1]["pan"] = 0.5
    _, _, i3 = S.render(s3)
    assert i3["events_rendered"] == 0
    s4 = json.loads(json.dumps(s))
    s4["steps"][1]["take"] = 7            # only that step's event renders again
    _, _, i4 = S.render(s4)
    assert i4["events_rendered"] == 1 and i4["events_cached"] == 4


def test_jitter_and_alternation():
    s = seq([{"entry": FAST, "repeat": 6, "alternate": True}], humanise={"timing_ms": 20, "level_db": 2, "alternate": 0.4})
    y, _, info = S.render(s)
    assert y.ndim == 2 and y.shape[1] == 2
    pans = [e["pan"] for e in info["events"]]
    assert pans == [-0.4, 0.4] * 3
    t = np.array([e["time"] for e in info["events"]])
    dev = np.diff(t) - 0.3
    assert np.all(np.abs(dev) <= 0.040 + 1e-6) and np.any(np.abs(dev) > 1e-4)
    g = [e["step_gain_db"] for e in info["events"]]
    assert all(abs(v) <= 2 for v in g) and len(set(g)) > 1
    mono = seq([{"entry": FAST, "repeat": 2}])
    assert S.render(mono)[0].ndim == 1


def test_export_mix_oneshots_and_cue_csv_rebuild_the_mix(tmp_path):
    s = seq([{"entry": FAST, "repeat": 3, "gain_db": -3}, {"entry": "sword_swing", "gap": -0.05, "gain_db": 2}],
            humanise={"timing_ms": 10, "level_db": 1})
    res = S.export(s, tmp_path / "x.wav", oneshots=True, bits=32)
    sr, mix = wavfile.read(res["wav"])
    rows = list(csv.DictReader(open(res["cues"], encoding="utf-8")))
    assert [r["index"] for r in rows] == ["1", "2", "3", "4"]
    assert [r["entry"] for r in rows] == [FAST] * 3 + ["sword_swing"]
    assert [int(r["take"]) for r in rows] == [e["take"] for e in S.plan(s)]
    times = [float(r["time"]) for r in rows]
    assert times == sorted(times[:3]) + times[3:]
    rebuilt = np.zeros(len(mix) + sr)
    for r in rows:
        fsr, x = wavfile.read(tmp_path / "x_oneshots" / r["file"])
        assert fsr == sr
        a = int(round(float(r["start"]) * sr))
        rebuilt[a:a + len(x)] += x * 10 ** (float(r["gain_db"]) / 20)
        assert float(r["start"]) <= float(r["time"]) <= float(r["start"]) + float(r["seconds"])
    k = len(mix) - int(0.06 * sr)            # the mix's trimmed end has a short fade
    assert np.max(np.abs(rebuilt[:k] - mix[:k])) < 1e-4


def test_cli_list_and_render(tmp_path, capsys):
    assert cli_main(["sequence", "--list"]) == 0
    assert "walk" in capsys.readouterr().out
    p = tmp_path / "s.json"
    p.write_text(json.dumps(seq([{"entry": FAST, "repeat": 2}])))
    assert cli_main(["sequence", str(p), "-o", str(tmp_path / "o.wav"), "--oneshots", "--repeat", "3",
                     "--interval", "0.2"]) == 0
    assert (tmp_path / "o.wav").exists()
    rows = list(csv.DictReader(open(tmp_path / "o_oneshots" / "cues.csv", encoding="utf-8")))
    assert len(rows) == 3 and float(rows[1]["time"]) - float(rows[0]["time"]) == pytest.approx(0.2, abs=1e-3)
    assert cli_main(["render", str(p), "-o", str(tmp_path / "r.wav")]) == 0      # render takes a sequence too
    assert (tmp_path / "r.wav").exists()
