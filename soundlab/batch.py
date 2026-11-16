"""Batch export, naming and audition (v0.7, roadmap step 3).

File names: <category>_<name>[_<macro><value>...]_v<NN>.wav, e.g. impact_hit_materialmetal_size0.8_v03.wav
(words inside a field are joined with '-': pickup_coin-pickup_v01.wav).
v01 is the recipe as written (at the given macros); v02.. are seeded variations, so re-running a batch gives
the same files, and adding variations never changes the earlier ones.
"""
from __future__ import annotations

import itertools
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from .recipe import load


def _slug(s: str) -> str:
    """Lower case, words joined by '-' (underscores separate the name's fields)."""
    return re.sub(r"[^a-z0-9.]+", "-", str(s).lower()).strip("-") or "x"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".") if v not in (0.0, 1.0) else str(int(v))
    return _slug(v)


def file_name(recipe: dict, stem: str, k: int, tags: dict | None = None, ext: str = ".wav") -> str:
    cat = _slug(recipe.get("category", "sfx"))
    parts = [cat, _slug(stem)] + [f"{_slug(m)}{_fmt(v)}" for m, v in (tags or {}).items()]
    return "_".join(parts) + f"_v{k + 1:02d}{ext}"


def collect(inputs: Iterable[str | Path], recursive: bool = False) -> list[Path]:
    """Recipe files from files, folders (their *.json) and list files (.txt, one path per line)."""
    out = []
    for i in inputs:
        p = Path(i)
        if p.is_dir():
            out += sorted(p.rglob("*.json") if recursive else p.glob("*.json"))
        elif p.suffix == ".txt":
            out += [Path(ln.strip()) for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        else:
            out.append(p)
    return out


def parse_sweep(items) -> dict:
    """['size=0,0.5,1', 'material=wood,metal'] -> {'size': [0, 0.5, 1], 'material': ['wood', 'metal']}."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        vals = []
        for x in v.split(","):
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(x.strip())
        out[k.strip()] = vals
    return out


def batch_export(inputs, out_dir, variations: int = 1, macros: dict | None = None, sweep: dict | None = None,
                 mp3: bool = False, sample_rate: int | None = None, recursive: bool = False,
                 log=print) -> list[dict]:
    """Render every recipe x every sweep combination x `variations` files into out_dir.
    Returns a manifest (also written to out_dir/manifest.json)."""
    from . import render_to_file
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    manifest = []
    sweep = sweep or {}
    for path in collect(inputs, recursive):
        rec = load(path)
        specs = rec.get("macros") or {}
        names = [m for m in sweep if m in specs]
        combos = list(itertools.product(*[sweep[m] for m in names])) or [()]
        fixed = {k: v for k, v in (macros or {}).items() if k in specs}
        for combo in combos:
            mv = dict(fixed, **dict(zip(names, combo)))
            tags = dict(fixed, **dict(zip(names, combo)))
            for k in range(max(1, int(variations))):
                fn = od / file_name(rec, path.stem, k, tags)
                t = time.time()
                info = render_to_file(rec, fn, mp3=mp3, sample_rate=sample_rate, macros=mv, variation=k)
                info.update({"recipe": str(path), "macros": mv, "variation": k, "render_s": round(time.time() - t, 3)})
                manifest.append(info)
                log(f"{fn.name}  ({info['render_s']} s)")
    mf = od / "manifest.json"
    old = json.loads(mf.read_text()) if mf.exists() else []
    new = {m["wav"] for m in manifest}
    mf.write_text(json.dumps([m for m in old if m.get("wav") not in new] + manifest, indent=1))
    return manifest


PLAYERS = (["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"], ["afplay"], ["paplay"], ["aplay", "-q"])


def play(path: str | Path) -> bool:
    for cmd in PLAYERS:
        if shutil.which(cmd[0]):
            subprocess.run(cmd + [str(path)], check=False)
            return True
    return False


def audition(recipe_path: str | Path, macros: dict | None = None, variation: int = 0, sample_rate: int = 24000,
             seconds: float | None = None, out_dir: str | Path = "/tmp/soundlab-audition", do_play: bool = True,
             watch: bool = False, log=print) -> Path:
    """Quick preview: render at a low rate (default 24 kHz), optionally cut to `seconds`, write to a scratch
    folder and play it (ffplay/afplay/paplay/aplay if present). watch=True re-renders whenever the recipe
    file changes (the edit-listen loop, without paying Python's start-up time again)."""
    from .audio_io import write_wav
    from .recipe import render
    p = Path(recipe_path)
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    out = od / f"{p.stem}.wav"
    last = None
    while True:
        m = p.stat().st_mtime
        if m != last:
            last = m
            try:
                rec = load(p)
                if seconds:
                    rec["duration"] = min(float(rec.get("duration", seconds)), float(seconds))
                t = time.time()
                y, sr = render(rec, sample_rate=sample_rate, macros=macros, variation=variation)
                write_wav(out, y, sr)
                log(f"rendered {out} in {time.time() - t:.2f} s ({len(y) / sr:.2f} s of audio at {sr} Hz)")
                if do_play and not play(out):
                    log("no audio player found (ffplay/afplay/paplay/aplay); open the file instead")
                    do_play = False
            except Exception as e:  # keep the loop alive while editing
                log(f"error: {e}")
        if not watch:
            return out
        time.sleep(0.3)
