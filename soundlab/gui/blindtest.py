"""Blind listening test API (single clips: 'real or synthesised?'). Answers: <pack>/answers/<listener>.json."""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path


def _pack(handler) -> Path:
    return Path(handler.app.pack_dir)


def _listener(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", (name or "").strip())[:40].strip("_")
    if not s:
        raise ValueError("enter your name or initials")
    return s


def _file(pack: Path, who: str) -> Path:
    return pack / "answers" / f"{who}.json"


def _save(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(p)


def handle_get(h, path: str, q: dict):
    pack = _pack(h)
    if path == "/api/bt/pack":
        pj = pack / "pack.json"
        if not pj.exists():
            return h._json({"ok": False, "pack_dir": str(pack),
                            "message": "No listening pack here yet. Build one with `soundlab blindtest build --library <folder>` "
                                       "(see docs/checkpoint.md)."})
        info = json.loads(pj.read_text())
        return h._json({"ok": True, "clips": len(info["clips"]), "sounds": len(info.get("sounds", [])),
                        "pack_dir": str(pack)})
    m = re.fullmatch(r"/api/bt/clip/(clip_\d{3})\.wav", path)
    if m:
        p = pack / "clips" / f"{m.group(1)}.wav"
        if not p.exists():
            return h._json({"error": "no such clip"}, 404)
        return h._send(200, p.read_bytes(), "audio/wav")
    if path == "/api/bt/results":
        from ..checkpoint import report_pack, score_pack
        res = score_pack(pack)
        return h._json({"report": report_pack(res), "result": _jsonable(res)})
    return h._json({"error": "not found"}, 404)


def handle_post(h, path: str, req: dict):
    pack = _pack(h)
    info = json.loads((pack / "pack.json").read_text())
    who = _listener(req.get("listener", ""))
    f = _file(pack, who)
    if path == "/api/bt/start":
        if f.exists():
            data = json.loads(f.read_text())
        else:
            order = list(info["clips"])
            seed = int(hashlib.sha256(who.encode()).hexdigest()[:8], 16)
            random.Random(seed).shuffle(order)          # own random order per listener
            data = {"listener": who, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "order": order, "answers": {}}
            _save(f, data)
        return h._json({"listener": who, "order": data["order"], "answered": sorted(data["answers"])})
    if path == "/api/bt/answer":
        data = json.loads(f.read_text())
        cid = req.get("clip")
        if cid not in data["order"]:
            raise ValueError("unknown clip")
        ans = req.get("answer")
        if ans not in ("real", "synth"):
            raise ValueError("answer must be 'real' or 'synth'")
        data["answers"][cid] = {"answer": ans, "confidence": int(req.get("confidence") or 0),
                                "note": str(req.get("note") or "")[:500], "time": time.strftime("%H:%M:%S")}
        if len(data["answers"]) == len(data["order"]):
            data["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save(f, data)
        return h._json({"answered": len(data["answers"]), "total": len(data["order"])})
    return h._json({"error": "not found"}, 404)


def _jsonable(o):
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o
