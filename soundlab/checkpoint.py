"""Blind checkpoint kit (see docs/checkpoint.md). Two formats:

A/B sets (`build` / `score`, also `python tools/checkpoint.py`) and single-clip packs for the browser
listening page (`build_pack` / `score_pack`, `soundlab blindtest build|score`, page `/blindtest` in
`soundlab gui`): each clip is played alone and the listener answers "real or synthesised?".

build: pair engine renders with real library recordings of the same sound, loudness-match and trim them,
       shuffle, and write trial_XX_A.wav / trial_XX_B.wav, an answer sheet (answers.csv) and a hidden key
       (key.json - keep it away from the listener).
score: read the filled answer sheet + key and report how often the listener picked the engine clip,
       per sound and overall, with 95% intervals and a pass/fail against the pass mark (default: under 60%
       correct = pass, i.e. the listener cannot reliably tell).

Folder layout (both --engine and --library): either one sub-folder per sound (rain/xyz.wav), or flat files
named <sound>__anything.wav. Engine renders from `soundlab batch` (<category>_<name>_vNN.wav) are also
understood when a manifest.json is present or with --engine-batch.

Library recordings stay in their own folder and are never copied into the repo; only the listening set (OUT)
contains them.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

from .audio_io import read_audio, write_wav

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


def prepare(x: np.ndarray, max_s: float, target_lufs: float = -20.0, trim_db: float = 40.0) -> np.ndarray:
    """Make real and engine clips alike so neither level nor dead air gives them away: trim leading silence,
    cap the length, trim the decayed tail (once the 10 ms level is `trim_db` under its maximum, short fade),
    and match loudness (maximum momentary loudness, BS.1770) under a -1 dBTP ceiling. The tail threshold is
    40 dB rather than the export default of 60 so a recording's room tone and a render's clean silence end
    at the same point of the decay."""
    from .finish import normalize_loudness, trim_tail
    env = np.convolve(np.abs(x), np.ones(240) / 240, "same")
    thr = env.max() * 0.02 if env.max() > 0 else 0
    on = np.nonzero(env > thr)[0]
    if len(on):
        x = x[max(0, on[0] - int(0.01 * SR)):]
    x = np.array(x[: int(max_s * SR)], float)
    x = trim_tail(x, SR, trim_db=trim_db, min_s=0.2, pad_s=0.05, fade_s=0.05)
    f = min(len(x) // 4, int(0.05 * SR))
    if f > 0:
        x[-f:] *= np.linspace(1, 0, f)
    if len(x) and np.max(np.abs(x)) > 0:
        x, _ = normalize_loudness(x, SR, target_lufs, "momentary", -1.0)
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
        v = "exempt" if r["exempt"] else ("no data" if not r["n"] else ("pass" if r["pass"] else "FAIL"))
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


# ------------------------------------------------------------------------------ single-clip packs
def build_pack(library: Path, out: Path, recipes_dir: Path | None = None, per_sound: int = 2, max_s: float = 4.0,
               seed: int = 0, sounds: list[str] | None = None, exempt: list[str] | None = None) -> dict:
    """Make a listening pack for the browser page: for every sound that has real recordings in
    library/<sound>/ AND a recipe recipes/<sound>.json, take up to `per_sound` real takes and render the same
    number of engine variations, prepare all clips alike (trim, cap length, fade, loudness-match, 48 kHz),
    shuffle them and write clips/clip_NNN.wav (neutral names), key.json (hidden truth) and pack.json (what
    the page shows). Library audio is copied into the pack only - never into the repo."""
    from . import render as sl_render
    rng = random.Random(seed)
    if not recipes_dir:
        from .gui.server import REPO_RECIPES as recipes_dir
    L = collect(library)
    have = {norm(p.stem): p for p in Path(recipes_dir).glob("*.json")}
    names = sorted(nm for nm in L if nm in have)
    if sounds:
        names = [nm for nm in names if nm in {norm(x) for x in sounds}]
    skipped = sorted(nm for nm in L if nm not in have)
    if not names:
        raise SystemExit(f"no library sound matches a recipe. library: {sorted(L)}")
    items = []
    for nm in names:
        takes = list(L[nm])
        rng.shuffle(takes)
        takes = takes[:per_sound]
        for p in takes:
            items.append({"sound": nm, "truth": "real", "source": str(p), "audio": lambda p=p: _load(p)})
        for k in range(len(takes)):
            def eng(nm=nm, k=k):
                y, sr = sl_render(have[nm], variation=k + 1)
                if sr != SR:
                    from scipy.signal import resample_poly
                    g = math.gcd(int(sr), SR)
                    y = resample_poly(y, SR // g, int(sr) // g)
                return y
            items.append({"sound": nm, "truth": "engine", "source": f"{have[nm].name} variation {k + 1}",
                          "audio": eng})
    rng.shuffle(items)
    for it in items:
        it["x"] = prepare(it["audio"](), max_s)
    equalise_by_sound(items)
    out = Path(out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    old_answers = out / "answers"
    if old_answers.exists() and any(old_answers.glob("*.json")):   # answers belong to the old clips
        old_answers.rename(out / time.strftime("answers_old_%Y%m%d_%H%M%S"))
    (out / "answers").mkdir(exist_ok=True)
    for old in (out / "clips").glob("clip_*.wav"):
        old.unlink()
    key = {"version": 2, "mode": "single", "seed": seed, "exempt": sorted({norm(x) for x in (exempt or [])}),
           "clips": []}
    for i, it in enumerate(items, 1):
        cid = f"clip_{i:03d}"
        write_wav(out / "clips" / f"{cid}.wav", it["x"], SR, 16)
        key["clips"].append({"clip": cid, "sound": it["sound"], "truth": it["truth"], "source": it["source"]})
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "pack.json").write_text(json.dumps({"version": 2, "clips": [c["clip"] for c in key["clips"]],
                                               "sounds": names, "per_sound": per_sound, "max_s": max_s}, indent=1))
    (out / "README.txt").write_text(
        "SoundLab blind listening pack. Start `soundlab gui --pack <this folder>` and open /blindtest.\n"
        "Listeners must not open key.json or the clips folder. Answers are saved in answers/<listener>.json.\n"
        "Score: `soundlab blindtest score <this folder>` or the Results link on the page.\n")
    return {"sounds": names, "clips": len(items), "skipped_no_recipe": skipped, "out": str(out)}


def equalise_by_sound(items: list[dict], target_lufs: float = -20.0) -> None:
    """Within each sound, real and engine clips must not differ in length or level. Every clip of a sound is
    cut to the shorter of (longest real clip, longest engine clip); a clip that is longer keeps the window
    around its loudest moment (so a slam after a handle rattle is not cut off), with 50 ms fades. Each clip
    is then levelled again (BS.1770, -1 dBTP), and all clips of the sound are brought to the quietest one's
    loudness (gain down only)."""
    from .finish import momentary_max_lufs, normalize_loudness
    by: dict[str, list[dict]] = {}
    for it in items:
        by.setdefault(it["sound"], []).append(it)
    for group in by.values():
        longest = {t: max((len(i["x"]) for i in group if i["truth"] == t), default=0) for t in ("real", "engine")}
        cap = min(v for v in longest.values() if v > 0)
        for it in group:
            x = it["x"]
            if len(x) > cap:
                w = int(0.05 * SR)
                env = np.convolve(x ** 2, np.ones(w) / w, "same")
                start = int(np.clip(np.argmax(env) - int(0.03 * SR), 0, len(x) - cap))
                x = x[start:start + cap].copy()
                f = min(cap // 4, int(0.05 * SR))
                if start > 0:
                    x[:int(0.005 * SR)] *= np.linspace(0, 1, int(0.005 * SR))
                x[-f:] *= 0.5 * (1 + np.cos(np.linspace(0, math.pi, f)))
            it["x"], _ = normalize_loudness(x, SR, target_lufs, "momentary", -1.0)
        lv = [momentary_max_lufs(it["x"], SR) for it in group]
        low = min(lv)
        for it, v in zip(group, lv):
            it["x"] = it["x"] * 10 ** ((low - v) / 20)


def load_answers(pack: Path) -> dict[str, dict]:
    res = {}
    for p in sorted((Path(pack) / "answers").glob("*.json")):
        try:
            res[p.stem] = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
    return res


def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771201e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        return -_z(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def sdt(hits: int, n_engine: int, fas: int, n_real: int) -> dict:
    """Signal-detection summary of 'real or synthesised?' answers. Hit = an engine clip called synthesised,
    false alarm = a real clip called synthesised. Balanced accuracy = (hit rate + correct-rejection rate) / 2,
    which a listener cannot raise by leaning towards one answer. d' and the criterion c use the log-linear
    correction (add 0.5 to each count, 1 to each total; Hautus 1995), so 0 % and 100 % stay finite.
    c > 0: leans towards 'real'; c < 0: leans towards 'synthesised'."""
    H = hits / n_engine if n_engine else float("nan")
    F = fas / n_real if n_real else float("nan")
    ba = (H + 1 - F) / 2 if n_engine and n_real else float("nan")
    if n_engine and n_real:
        zh, zf = _z((hits + 0.5) / (n_engine + 1)), _z((fas + 0.5) / (n_real + 1))
        dp, c = zh - zf, -(zh + zf) / 2
    else:
        dp = c = float("nan")
    return {"hit_rate": H, "fa_rate": F, "balanced_accuracy": ba, "d_prime": dp, "c": c}


def _boot_ba(trials: list[tuple[str, str, int, int]], reps: int = 4000, seed: int = 0) -> list[float]:
    """95 % percentile interval of pooled balanced accuracy. Two-stage bootstrap: listeners are resampled
    with replacement, and clips are resampled with replacement separately among engine and real clips (the
    same clip draw applies to every listener in a replicate), so both listener and clip variability count.
    trials: (listener, clip, is_engine, said_synth)."""
    if not trials:
        return [float("nan"), float("nan")]
    listeners = sorted({t[0] for t in trials})
    eng = sorted({t[1] for t in trials if t[2]})
    real = sorted({t[1] for t in trials if not t[2]})
    if not eng or not real:
        return [float("nan"), float("nan")]
    li = {x: i for i, x in enumerate(listeners)}
    ci = {x: i for i, x in enumerate(eng + real)}
    M = np.full((len(listeners), len(eng) + len(real)), -1, int)       # -1 = not answered
    for who, clip, _, ss in trials:
        M[li[who], ci[clip]] = ss
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    ne = len(eng)
    for r in range(reps):
        L = rng.integers(0, len(listeners), len(listeners))
        E = rng.integers(0, ne, ne)
        R = ne + rng.integers(0, len(real), len(real))
        me, mr = M[np.ix_(L, E)], M[np.ix_(L, R)]
        ve, vr = me[me >= 0], mr[mr >= 0]
        if not len(ve) or not len(vr):
            out[r] = np.nan
            continue
        out[r] = (ve.mean() + 1 - vr.mean()) / 2
    out = out[np.isfinite(out)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if len(out) else [float("nan")] * 2


def verdict(ci: list[float], chance: float = 0.5, mark: float = 0.60) -> str:
    """'FAIL' when the 95 % interval of balanced accuracy lies above chance (listeners reliably tell real
    from engine); 'PASS' when the whole interval is under the mark (any ability to tell is small);
    'pass?' when the interval includes chance but reaches the mark (no reliable difference, not enough
    data to rule out a modest one); 'no data' otherwise."""
    lo, hi = ci
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "no data"
    if lo > chance:
        return "FAIL"
    return "PASS" if hi < mark else "pass?"


def score_pack(pack: Path, pass_mark: float = 0.60, listeners: list[str] | None = None, reps: int = 4000) -> dict:
    """Score single-clip answers ('real or synthesised?'). Primary figure: balanced accuracy, the mean of
    the share of engine clips called synthesised (hit rate) and the share of real clips called real. It is
    50 % for guessing whatever the listener's leaning, so answering 'synthesised' to everything cannot make a
    sound fail. Verdict from a two-stage bootstrap 95 % interval (listeners x clips, see `verdict`).
    `pass_mark` is the upper bound of the interval for a clear pass. Also reported, for information: the
    share of engine clips spotted (the pre-0.16 figure, pass = under the mark), false alarms, d', and the
    response bias c per listener."""
    pack = Path(pack)
    key = json.loads((pack / "key.json").read_text())
    truth = {c["clip"]: c for c in key["clips"]}
    exempt = set(key.get("exempt", []))
    answers = load_answers(pack)
    if listeners:
        answers = {k: v for k, v in answers.items() if k in listeners}
    trials = []                                      # (listener, clip, is_engine, said_synth, sound)
    for who, a in answers.items():
        for cid, rec in a.get("answers", {}).items():
            t = truth.get(cid)
            if not t or rec.get("answer") not in ("real", "synth"):
                continue
            trials.append((who, cid, int(t["truth"] == "engine"), int(rec["answer"] == "synth"), t["sound"]))

    def summarise(tr, boot=True):
        e = [x[3] for x in tr if x[2]]
        r = [x[3] for x in tr if not x[2]]
        k, n = sum(e), len(e)
        fa, nr = sum(r), len(r)
        st = sdt(k, n, fa, nr)
        ci = _boot_ba([x[:4] for x in tr], reps) if boot else [float("nan")] * 2
        acc = [int(x[3] == x[2]) for x in tr]
        return {"n": n, "correct": k, "pct": (k / n) if n else float("nan"), "ci95": list(wilson(k, n)),
                "p_guess": binom_p_greater(k, n), "old_pass": bool(n) and (k / n) < pass_mark,
                "real_n": nr, "false_alarms": fa, "false_alarm_pct": (fa / nr) if nr else float("nan"),
                "accuracy": (sum(acc) / len(acc)) if acc else float("nan"), "answered": len(acc),
                **st, "ba_ci95": ci, "verdict": verdict(ci, 0.5, pass_mark)}

    rows = {}
    for nm in sorted({x[4] for x in trials} | {c["sound"] for c in key["clips"]}):
        r = summarise([x for x in trials if x[4] == nm])
        r["exempt"] = nm in exempt
        r["pass"] = r["verdict"] in ("PASS", "pass?")
        rows[nm] = r
    phys = [x for x in trials if x[4] not in exempt]
    overall = summarise(phys)
    overall["pass"] = overall["verdict"] in ("PASS", "pass?")
    each = {}
    for who in sorted(answers):
        o = summarise([x for x in phys if x[0] == who])
        o["pass"] = o["verdict"] in ("PASS", "pass?")
        each[who] = o
    return {"pass_mark": pass_mark, "mode": "single", "measure": "balanced_accuracy", "listeners": sorted(answers),
            "pooled": {"per_sound": rows, "overall_physical": overall}, "per_listener": each,
            "total_clips": len(truth)}


def _pc(x: float) -> str:
    return "-" if not (isinstance(x, (int, float)) and math.isfinite(x)) else f"{x:.0%}"


def report_pack(res: dict) -> str:
    pm = res["pass_mark"]
    lines = [f"# Blind listening result (balanced accuracy; 50 % = cannot tell real from engine)", "",
             f"Verdict from the 95 % interval of balanced accuracy: FAIL = interval above 50 % (real and engine "
             f"are told apart reliably); PASS = whole interval under {pm:.0%}; pass? = interval includes 50 % "
             f"but reaches {pm:.0%} (no reliable difference yet, more answers would settle it).", "",
             f"Listeners: {', '.join(res['listeners']) or 'none yet'}", "",
             "| sound | engine clips | spotted as synth | real clips | real called synth | balanced accuracy "
             "| 95% interval | verdict (hint) |",
             "|---|---|---|---|---|---|---|---|"]
    for nm, r in res["pooled"]["per_sound"].items():
        v = "exempt" if r["exempt"] else r["verdict"]
        lines.append(f"| {nm} | {r['n']} | {_pc(r['pct'])} | {r['real_n']} | {_pc(r['false_alarm_pct'])} | "
                     f"{_pc(r['balanced_accuracy'])} | {_pc(r['ba_ci95'][0])}-{_pc(r['ba_ci95'][1])} | {v} |")
    o = res["pooled"]["overall_physical"]
    if o["n"]:
        lines += ["", f"Pooled (physical sounds): balanced accuracy {_pc(o['balanced_accuracy'])} "
                  f"(95% interval {_pc(o['ba_ci95'][0])}-{_pc(o['ba_ci95'][1])}) -> {o['verdict']}; "
                  f"d' {o['d_prime']:.2f}. Engine clips spotted {o['correct']}/{o['n']} = {_pc(o['pct'])}, "
                  f"real clips called synthesised {_pc(o['false_alarm_pct'])}, overall accuracy "
                  f"{_pc(o['accuracy'])}. (Pre-0.16 figure: spotted under {pm:.0%} -> "
                  f"{'pass' if o['old_pass'] else 'fail'}.)", "",
                  "| listener | answered | spotted (hit rate) | real called synth | balanced accuracy | 95% interval "
                  "| d' | bias c | verdict |", "|---|---|---|---|---|---|---|---|---|"]
        for name, ov in res["per_listener"].items():
            if ov["answered"]:
                lines.append(f"| {name} | {ov['answered']} | {_pc(ov['hit_rate'])} | {_pc(ov['fa_rate'])} | "
                             f"{_pc(ov['balanced_accuracy'])} | {_pc(ov['ba_ci95'][0])}-{_pc(ov['ba_ci95'][1])} | "
                             f"{ov['d_prime']:.2f} | {ov['c']:+.2f} | {ov['verdict']} |")
        lines.append("")
        lines.append("Bias c: negative = leans towards 'synthesised', positive = towards 'real'. Per-sound rows "
                     "rest on a few clips each: hints, not verdicts.")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- answer codes (static web page)
def decode_code(code: str) -> dict:
    """Decode an answer code from the static listening page: 'SL1.' + base64url of [version 1, pack number,
    clip count n, n 2-bit answers in clip-number order (0 none, 1 real, 2 synth), checksum (sum of all
    other bytes mod 256), handle as UTF-8]."""
    import base64
    s = code.strip().split(".", 1)[1]
    b = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    if len(b) < 4 or b[0] != 1:
        raise ValueError("not a version-1 answer code")
    n = b[2]
    nb = (n + 3) // 4
    if len(b) < 4 + nb:
        raise ValueError("code cut off")
    if sum(x for i, x in enumerate(b) if i != 3 + nb) & 255 != b[3 + nb]:
        raise ValueError("checksum mismatch (code damaged or cut off when copied)")
    ans = {}
    for i in range(n):
        v = (b[3 + (i >> 2)] >> ((i & 3) * 2)) & 3
        if v in (1, 2):
            ans[f"clip_{i + 1:03d}"] = "real" if v == 1 else "synth"
    return {"pack": b[1], "n": n, "handle": b[4 + nb:].decode("utf-8", "replace").strip(), "answers": ans}


def import_codes(pack: Path, text: str, dry_run: bool = False, log=print) -> list[Path]:
    """Find every answer code in `text` and write it to <pack>/answers/<handle>.json (the format the page
    and `score_pack` use). A code already imported is skipped; a different code with the same handle gets
    a numbered file."""
    import re
    pack = Path(pack)
    key = json.loads((pack / "key.json").read_text())
    valid = {c["clip"] for c in key["clips"]}
    out = pack / "answers"
    out.mkdir(exist_ok=True)
    existing = {}
    for p in out.glob("*.json"):
        try:
            existing[json.loads(p.read_text()).get("code")] = p
        except (json.JSONDecodeError, AttributeError):
            pass
    written = []
    for code in dict.fromkeys(re.findall(r"SL1\.[A-Za-z0-9_-]{8,}", text)):
        try:
            d = decode_code(code)
        except ValueError as e:
            log(f"skip {code[:20]}...: {e}")
            continue
        if d["n"] != len(valid) or not set(d["answers"]) <= valid:
            log(f"skip {code[:20]}...: made for a different pack ({d['n']} clips, this pack has {len(valid)})")
            continue
        if code in existing:
            log(f"already imported: {existing[code].name}")
            continue
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", d["handle"].lstrip("@")).strip("._")[:40] or "anon"
        p, k = out / f"{name}.json", 2
        while p.exists():
            p, k = out / f"{name}_{k}.json", k + 1
        rec = {"listener": d["handle"] or name, "code": code, "imported": time.strftime("%Y-%m-%d %H:%M:%S"),
               "order": sorted(d["answers"]), "answers": {c: {"answer": v} for c, v in d["answers"].items()}}
        log(f"{'(dry run) ' if dry_run else ''}{d['handle'] or '(no handle)'}: {len(d['answers'])}/{d['n']} "
            f"answered -> {p.name}")
        if not dry_run:
            p.write_text(json.dumps(rec, indent=1))
            existing[code] = p
        written.append(p)
    return written


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
