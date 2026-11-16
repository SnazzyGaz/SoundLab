#!/usr/bin/env python3
"""Blind A/B checkpoint kit (see docs/checkpoint.md).

build: pair engine renders with real library recordings of the same sound, loudness-match and trim them,
       shuffle, and write trial_XX_A.wav / trial_XX_B.wav, an answer sheet (answers.csv) and a hidden key
       (key.json - keep it away from the listener).
score: read the filled answer sheet + key and report how often the listener picked the engine clip,
       per sound and overall, with 95% intervals and a pass/fail against the pass mark (default: under 60%
       correct = pass, i.e. the listener cannot reliably tell).

Folder layout (both --engine and --library): either one sub-folder per sound (rain/xyz.wav), or flat files
named <sound>__anything.wav. Engine renders from `soundlab batch` (<category>_<name>_vNN.wav) are also
understood when a manifest.json is present or with --engine-batch.

Library recordings are the user's own and are never copied into the repo; only the listening set (OUT)
contains them.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soundlab.audio_io import read_audio, write_wav  # noqa: E402

AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}
SR = 48000


def _batch_name(stem: str) -> str:
    """<category>_<name>[_macro value]_vNN -> name (best effort without a manifest)."""
    parts = stem.split("_")
    if parts and parts[-1].startswith("v") and parts[-1][1:].isdigit():
        parts = parts[:-1]
    return "_".join(parts[1:]) if len(parts) > 1 else stem


def norm(name: str) -> str:
    """Sound names compare case-, dash- and space-insensitively (batch files use dashes: creature-growl)."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def collect(folder: Path, batch: bool = False) -> dict[str, list[Path]]:
    folder = Path(folder)
    out: dict[str, list[Path]] = {}
    manifest = {}
    mf = folder / "manifest.json"
    if mf.exists():
        try:
            m = json.loads(mf.read_text())
            items = m.get("renders", m) if isinstance(m, dict) else m
            for it in (items if isinstance(items, list) else []):
                if not isinstance(it, dict):
                    continue
                name = it.get("name") or (Path(it["recipe"]).stem if it.get("recipe") else None)
                for k in ("file", "wav", "mp3"):
                    if it.get(k) and name:
                        manifest[Path(it[k]).stem] = name
        except (json.JSONDecodeError, AttributeError):
            pass
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in AUDIO or not p.is_file():
            continue
        rel = p.relative_to(folder)
        if len(rel.parts) > 1:
            key = rel.parts[0]
        elif "__" in p.stem:
            key = p.stem.split("__")[0]
        elif p.stem in manifest:
            key = manifest[p.stem]
        elif batch:
            key = _batch_name(p.stem)
        else:
            key = p.stem
        out.setdefault(norm(key), []).append(p)
    for k, v in out.items():                      # one entry per take: prefer the wav over its mp3 twin
        stems = {}
        for p in v:
            if p.with_suffix("") not in stems or p.suffix.lower() == ".wav":
                stems[p.with_suffix("")] = p
        out[k] = sorted(stems.values())
    return out


def _load(path: Path) -> np.ndarray:
    x, sr = read_audio(path)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.asarray(x, float)


def prepare(x: np.ndarray, max_s: float, target_db: float = -20.0) -> np.ndarray:
    """Trim leading silence, cap length, fade, and match loudness (RMS of the active part)."""
    env = np.convolve(np.abs(x), np.ones(240) / 240, "same")
    thr = env.max() * 0.02 if env.max() > 0 else 0
    on = np.nonzero(env > thr)[0]
    if len(on):
        x = x[max(0, on[0] - int(0.01 * SR)):]
    x = x[: int(max_s * SR)].copy()
    f = min(len(x) // 4, int(0.05 * SR))
    if f > 0:
        x[-f:] *= np.linspace(1, 0, f)
    env = np.convolve(x ** 2, np.ones(2400) / 2400, "same")
    act = env > env.max() * 0.01 if env.max() > 0 else np.ones(len(x), bool)
    rms = math.sqrt(float(np.mean(x[act] ** 2))) if np.any(act) else 0.0
    if rms > 0:
        x *= 10 ** (target_db / 20) / rms
    pk = np.max(np.abs(x)) if len(x) else 0
    if pk > 0.95:                                   # never clip: scale both later by the same rule
        x *= 0.95 / pk
    return x


def build(engine: Path, library: Path, out: Path, per_sound: int = 4, max_s: float = 4.0, seed: int = 0,
          engine_batch: bool = True, sounds: list[str] | None = None, exempt: list[str] | None = None) -> dict:
    rng = random.Random(seed)
    E, L = collect(engine, engine_batch), collect(library)
    names = sorted(set(E) & set(L))
    if sounds:
        names = [s for s in names if s in {norm(x) for x in sounds}]
    if not names:
        raise SystemExit(f"no sound names in common. engine: {sorted(E)}  library: {sorted(L)}")
    trials = []
    for nm in names:
        for _ in range(per_sound):
            trials.append((nm, rng.choice(E[nm]), rng.choice(L[nm])))
    rng.shuffle(trials)
    out = Path(out)
    (out / "trials").mkdir(parents=True, exist_ok=True)
    key = {"version": 1, "seed": seed, "exempt": sorted({norm(x) for x in (exempt or [])}), "trials": []}
    rows = []
    for i, (nm, ep, lp) in enumerate(trials, 1):
        e, lib = prepare(_load(ep), max_s), prepare(_load(lp), max_s)
        engine_side = rng.choice("AB")
        a, b = (e, lib) if engine_side == "A" else (lib, e)
        tid = f"trial_{i:02d}"
        write_wav(out / "trials" / f"{tid}_A.wav", a, SR)
        write_wav(out / "trials" / f"{tid}_B.wav", b, SR)
        key["trials"].append({"trial": tid, "sound": nm, "engine": engine_side, "engine_file": str(ep),
                              "library_file": str(lp)})
        rows.append({"trial": tid, "sound": nm, "engine_is (A/B)": "", "confidence (1-3)": "", "notes": ""})
    with open(out / "answers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "README.txt").write_text(
        "Blind listening test. For each trial, play A and B (any order, as often as you like) and write in\n"
        "answers.csv which one you think is the ENGINE (synthesised) sound: A or B. Guess if unsure; never\n"
        "leave a row blank. Confidence: 1 = guessing, 2 = leaning, 3 = sure. Do NOT open key.json.\n")
    return key


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_p_greater(k: int, n: int) -> float:
    """One-sided P(X >= k) for X ~ Binomial(n, 0.5): chance of doing this well by pure guessing."""
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n else 1.0


def score(key_path: Path, answers_path: Path, pass_mark: float = 0.60) -> dict:
    key = json.loads(Path(key_path).read_text())
    truth = {t["trial"]: t for t in key["trials"]}
    exempt = set(key.get("exempt", []))
    per: dict[str, list[int]] = {}
    missing = []
    with open(answers_path, newline="") as f:
        for row in csv.DictReader(f):
            tid = row.get("trial", "").strip()
            ans = next((v for k, v in row.items() if k and k.lower().startswith("engine")), "") or ""
            ans = ans.strip().upper()[:1]
            if tid not in truth:
                continue
            if ans not in ("A", "B"):
                missing.append(tid)
                continue
            per.setdefault(truth[tid]["sound"], []).append(int(ans == truth[tid]["engine"]))
    rows = {}
    for nm, v in sorted(per.items()):
        k, n = sum(v), len(v)
        lo, hi = wilson(k, n)
        rows[nm] = {"n": n, "correct": k, "pct": k / n, "ci95": [lo, hi], "p_guess": binom_p_greater(k, n),
                    "exempt": nm in exempt, "pass": (k / n) < pass_mark}
    phys = [x for nm, v in per.items() if nm not in exempt for x in v]
    k, n = sum(phys), len(phys)
    overall = {"n": n, "correct": k, "pct": (k / n) if n else float("nan"), "ci95": list(wilson(k, n)),
               "p_guess": binom_p_greater(k, n), "pass": bool(n) and (k / n) < pass_mark}
    return {"pass_mark": pass_mark, "per_sound": rows, "overall_physical": overall, "unanswered": missing}


def report(res: dict) -> str:
    lines = [f"# Blind checkpoint result (pass = under {res['pass_mark']:.0%} correctly identified)", "",
             "| sound | trials | engine spotted | 95% range | chance by guessing | verdict |", "|---|---|---|---|---|---|"]
    for nm, r in res["per_sound"].items():
        v = "exempt" if r["exempt"] else ("pass" if r["pass"] else "FAIL")
        lines.append(f"| {nm} | {r['n']} | {r['pct']:.0%} | {r['ci95'][0]:.0%}-{r['ci95'][1]:.0%} | "
                     f"{r['p_guess']:.2f} | {v} |")
    o = res["overall_physical"]
    lines += ["", f"Overall (physical sounds): {o['correct']}/{o['n']} = {o['pct']:.0%} spotted "
              f"(95% range {o['ci95'][0]:.0%}-{o['ci95'][1]:.0%}) -> {'PASS' if o['pass'] else 'FAIL'}"]
    if o["n"] and o["ci95"][1] >= res["pass_mark"] and o["pass"]:
        lines.append("Note: a pass, but the upper end of the range crosses the mark: run more trials to be sure.")
    if res["unanswered"]:
        lines.append(f"Unanswered trials (not counted): {', '.join(res['unanswered'])}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="make a shuffled A/B listening set")
    b.add_argument("--engine", required=True, type=Path, help="folder of engine renders")
    b.add_argument("--library", required=True, type=Path, help="folder of real library recordings (yours)")
    b.add_argument("-o", "--out", required=True, type=Path)
    b.add_argument("--per-sound", type=int, default=4, help="trials per sound (default 4)")
    b.add_argument("--max-s", type=float, default=4.0, help="cap clip length (s)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--sounds", nargs="*", help="only these sound names")
    b.add_argument("--exempt", nargs="*", default=[], help="sounds outside the pass mark (e.g. stylised ones)")
    s = sub.add_parser("score", help="score a filled answer sheet")
    s.add_argument("--key", required=True, type=Path)
    s.add_argument("--answers", required=True, type=Path)
    s.add_argument("--pass-mark", type=float, default=0.60)
    s.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        k = build(a.engine, a.library, a.out, a.per_sound, a.max_s, a.seed, True, a.sounds, a.exempt)
        print(f"{len(k['trials'])} trials -> {a.out}/trials ; answer sheet {a.out}/answers.csv ; "
              f"key {a.out}/key.json (hide it)")
    else:
        r = score(a.key, a.answers, a.pass_mark)
        print(json.dumps(r, indent=1) if a.json else report(r))


if __name__ == "__main__":
    main()
