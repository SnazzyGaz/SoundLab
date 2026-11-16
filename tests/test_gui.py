"""Slider app + blind-test HTTP API (server on a free port, temp output folder)."""
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from soundlab.checkpoint import build_pack, score_pack
from soundlab.gui import make_server
from soundlab.gui.server import REPO_RECIPES


@pytest.fixture()
def srv(tmp_path):
    s = make_server(0, recipe_dirs=[REPO_RECIPES], out_dir=tmp_path / "out", pack_dir=tmp_path / "pack")
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{s.server_address[1]}"
    yield base, s, tmp_path
    s.shutdown()
    s.server_close()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url, body):
    rq = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_binds_localhost_only(srv):
    _, s, _ = srv
    assert s.server_address[0] == "127.0.0.1"


def test_pages_and_info(srv):
    base, _, _ = srv
    for path in ("/", "/blindtest"):
        code, ctype, body = get(base + path)
        assert code == 200 and ctype.startswith("text/html") and b"<html" in body
    info = json.loads(get(base + "/api/info")[2])
    assert "version" in info and info["out_dir"].endswith("out")


def test_static_traversal_blocked(srv):
    base, _, _ = srv
    with pytest.raises(urllib.error.HTTPError):
        get(base + "/static/../server.py")


def test_recipes_listed_by_category(srv):
    base, _, _ = srv
    rs = json.loads(get(base + "/api/recipes")[2])
    ids = {r["id"] for r in rs}
    assert "gunshot" in ids and "rain" in ids
    assert not any(i.startswith("fit/") for i in ids)
    cats = [r["category"] for r in rs]
    assert cats == sorted(cats)


def test_describe_has_macros_and_advanced(srv):
    base, _, _ = srv
    d = json.loads(get(base + "/api/recipe?id=gunshot")[2])
    names = {m["name"]: m for m in d["macros"]}
    assert "pistol" in names["calibre"]["choices"] and names["body"]["default"] == 0.5
    keys = [c["key"] for b in d["advanced"] for c in b["controls"]]
    assert any(k.startswith("0.") for k in keys)
    code, _ = post(base + "/api/render", {"id": "../etc/passwd"})
    assert code == 400


def test_render_and_audio(srv):
    base, _, _ = srv
    code, r = post(base + "/api/render", {"id": "coin", "macros": {}, "variation": 0})
    assert code == 200 and r["seconds"] > 0 and len(r["waveform"]) == 800
    assert r["spectrogram"]["frames"] == 240
    code, ctype, wav = get(base + r["url"])
    assert code == 200 and ctype == "audio/wav" and wav[:4] == b"RIFF"


def test_render_reacts_to_macros_variation_overrides(srv):
    base, s, _ = srv
    a = s.app.render({"id": "gunshot"})
    b = s.app.render({"id": "gunshot", "macros": {"calibre": "shotgun"}})
    c = s.app.render({"id": "gunshot", "variation": 2})
    d = s.app.render({"id": "gunshot", "overrides": {"0.body": 0.9}})
    keys = [x["key"] for x in (a, b, c, d)]
    wav = [s.app.cache[k][0] for k in keys]
    for other in wav[1:]:
        n = min(len(wav[0]), len(other))
        assert not np.allclose(wav[0][:n], other[:n])
    again = s.app.render({"id": "gunshot"})
    assert np.allclose(s.app.cache[again["key"]][0], wav[0])


def test_export_batch_names(srv):
    base, _, tmp = srv
    code, r = post(base + "/api/export", {"id": "coin", "macros": {}, "variation": 0})
    assert code == 200 and Path(r["wav"]).exists() and Path(r["wav"]).parent == tmp / "out" / "exports"
    assert Path(r["wav"]).name.endswith("_v01.wav")
    sr, x = wavfile.read(r["wav"])
    assert len(x) > 0
    code, r = post(base + "/api/batch", {"id": "coin", "n": 3, "overrides": {}})
    names = sorted(Path(f["wav"]).name for f in r["files"])
    assert len(names) == 3 and names[-1].endswith("_v03.wav")
    code, r = post(base + "/api/export", {"id": "coin", "overrides": {"0.diameter_mm": 30}})
    assert "_custom_" in Path(r["wav"]).name


def test_save_new_recipe_and_reload(srv):
    base, s, tmp = srv
    code, r = post(base + "/api/save", {"id": "gunshot", "macros": {"calibre": "rifle"}, "name": "My Rifle!"})
    assert code == 200 and r["id"] == "my/my_rifle"
    saved = json.loads(Path(r["path"]).read_text())
    assert saved["based_on"] == "gunshot" and "macros" in saved and saved["layers"][0]["params"]["calibre"] == "rifle"
    ids = {x["id"] for x in json.loads(get(base + "/api/recipes")[2])}
    assert "my/my_rifle" in ids and "gunshot" in ids
    a = s.app.render({"id": "my/my_rifle"})
    b = s.app.render({"id": "gunshot", "macros": {"calibre": "rifle"}})
    assert np.allclose(s.app.cache[a["key"]][0], s.app.cache[b["key"]][0])
    code, r = post(base + "/api/save", {"id": "gunshot", "name": "  "})
    assert code == 400


def _library(tmp: Path) -> Path:
    lib = tmp / "library"
    rng = np.random.default_rng(0)
    for snd in ("coin", "rain", "not_a_recipe"):
        (lib / snd).mkdir(parents=True)
        for i in range(3):
            x = (rng.standard_normal(48000) * np.exp(-np.linspace(0, 6, 48000)) * 0.3).astype(np.float32)
            wavfile.write(lib / snd / f"take{i}.wav", 48000, x)
    return lib


def test_blindtest_pack_and_answers(srv):
    base, s, tmp = srv
    assert json.loads(get(base + "/api/bt/pack")[2])["ok"] is False
    info = build_pack(_library(tmp), tmp / "pack", per_sound=2, max_s=1.0)
    assert info["sounds"] == ["coin", "rain"] and info["clips"] == 8 and info["skipped_no_recipe"] == ["not_a_recipe"]
    key = json.loads((tmp / "pack" / "key.json").read_text())
    assert sorted(c["truth"] for c in key["clips"]).count("engine") == 4
    pack = json.loads(get(base + "/api/bt/pack")[2])
    assert pack["ok"] and pack["clips"] == 8
    code, st = post(base + "/api/bt/start", {"listener": "Ann B"})
    assert code == 200 and st["listener"] == "Ann_B" and len(st["order"]) == 8 and st["answered"] == []
    assert "truth" not in json.dumps(st) and "engine" not in json.dumps(st)
    code, ctype, wav = get(base + f"/api/bt/clip/{st['order'][0]}.wav")
    assert ctype == "audio/wav" and wav[:4] == b"RIFF"
    truth = {c["clip"]: c["truth"] for c in key["clips"]}
    for cid in st["order"]:            # a perfect listener
        code, r = post(base + "/api/bt/answer", {"listener": "Ann B", "clip": cid,
                                                  "answer": "synth" if truth[cid] == "engine" else "real"})
        assert code == 200
    assert r == {"answered": 8, "total": 8}
    saved = json.loads((tmp / "pack" / "answers" / "Ann_B.json").read_text())
    assert "finished" in saved
    code, st2 = post(base + "/api/bt/start", {"listener": "Ann B"})     # resumable, same order
    assert st2["order"] == st["order"] and len(st2["answered"]) == 8
    res = score_pack(tmp / "pack")
    o = res["pooled"]["overall_physical"]
    assert o["n"] == 4 and o["pct"] == 1.0 and o["pass"] is False and o["false_alarm_pct"] == 0.0
    rep = json.loads(get(base + "/api/bt/results")[2])
    assert "FAIL" in rep["report"] and rep["result"]["listeners"] == ["Ann_B"]
    code, err = post(base + "/api/bt/answer", {"listener": "Ann B", "clip": "clip_999", "answer": "real"})
    assert code == 400
    code, err = post(base + "/api/bt/start", {"listener": "  "})
    assert code == 400


def test_blindtest_guessing_listener_passes(tmp_path):
    lib = _library(tmp_path)
    build_pack(lib, tmp_path / "pack", per_sound=3, max_s=0.5, sounds=["coin"])
    key = json.loads((tmp_path / "pack" / "key.json").read_text())
    ans = {c["clip"]: {"answer": "real"} for c in key["clips"]}       # fooled every time
    (tmp_path / "pack" / "answers" / "x.json").write_text(json.dumps({"answers": ans}))
    o = score_pack(tmp_path / "pack")["pooled"]["overall_physical"]
    assert o["n"] == 3 and o["pct"] == 0.0 and o["pass"] is True
