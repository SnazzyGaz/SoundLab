"""Sequences: library entries chained on a timeline and rendered offline into one buffer.

A sequence is plain JSON, like a recipe, with "type": "sequence" and a list of steps instead of layers:

{
  "soundlab": 1, "type": "sequence", "name": "walk", "title": "Walk", "category": "impact", "seed": 1,
  "rate": {"spm": 110},                          # base interval: {"interval": seconds} or {"spm": steps per minute}
  "humanise": {"timing_ms": 12, "level_db": 1.5, "alternate": 0.25},
  "level": "lufs",                               # the mix: "lufs" (loudness per category) or "peak"
  "steps": [
    {"entry": "footstep", "macros": {"surface": "gravel"}, "repeat": 8},
    {"entry": "doors/knock_wood", "overrides": {"0.hit": -2}, "gap": 0.6, "repeat": 3, "interval": 0.22,
     "gain_db": -3, "pan": 0.2, "take": "auto"}
  ]
}

Step keys
  entry      a library id (recipes/<id>.json, or my/<name> in the slider app)
  macros     the entry's controls (as in the slider app); overrides: exact model values ("L.param" keys)
  take       "auto" (default): every event draws its own take, never the same take as the previous event of
             the same entry; a number: that variation for every event (0 = the entry as designed)
  repeat     how many events this step plays (default 1), spaced by "interval" seconds (default: the base rate)
  start      "at": seconds from the start, or "offset": seconds after the previous step's last event starts
             (default: one base interval), or "gap": seconds after the previous step's last event ends
             (negative overlaps). The first step starts at 0 unless it says otherwise.
  gain_db, pan (-1 left .. 1 right), alternate (true/false: left/right alternation; default true for footsteps)
  key        optional number the step's random draws are keyed on (default: its position), so a moved step keeps
             its takes and jitter

Humanise: every event gets a uniform timing offset within +-timing_ms and a level offset within +-level_db; the
events of an alternating step swing left and right by "alternate" (pan units). The draws come from their own
generators (sequence seed, step key or index, purpose), so the takes stay put when only timing or levels change.

Events are rendered as the slider app plays a single entry (sources ring out, the silent tail is trimmed, then the
loudness gain for the entry's category), so a step at gain 0 sits at the level the entry has on its own. The peak
limiter runs once, on the finished mix. Event renders are cached by
their concrete recipe and seed: a timing, gain, pan or jitter change only re-mixes cached events. Event onsets (the
first sample within 30 dB of the peak) land on the timeline. Takes are real distinct takes: for the recording-stem
families the gesture each variation plays is checked, so consecutive events of an entry use different recorded
gestures where the recording has more than one (and always a different variation, i.e. a fresh remix of the stems).
Nothing is averaged, stretched or frozen.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .core import REGISTRY, RecipeError
from .fastpath import check

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "sequences"
TAKE_TRIES = 16
MAX_EVENTS = 512
STEMS_MODULES = {"door_stems": "door_stems", "foley_stems": "foley_stems", "foot_stems": "foot_stems",
                 "impact_stems": "impact_stems", "weapon_stems": "weapon_stems"}


def is_sequence(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("type") == "sequence"


def load(path_or_obj: Any) -> dict:
    if isinstance(path_or_obj, dict):
        return path_or_obj
    return json.loads(Path(path_or_obj).read_text(encoding="utf-8"))


def presets(folder: Path | None = None) -> list[dict]:
    out = []
    root = Path(folder) if folder else PRESETS
    for p in sorted(root.glob("*.json")) if root.is_dir() else []:
        s = json.loads(p.read_text(encoding="utf-8"))
        if is_sequence(s):
            out.append({"id": p.stem, "title": s.get("title") or p.stem, "doc": s.get("doc", ""),
                        "order": s.get("order", 99), "sequence": s})
    return sorted(out, key=lambda d: (d["order"], d["title"]))


def preset(name: str) -> dict:
    p = PRESETS / f"{name}.json"
    if not re.fullmatch(r"[a-z0-9_]+", name or "") or not p.is_file():
        raise RecipeError(f"no sequence preset {name!r}; have {[d['id'] for d in presets()]}")
    return load(p)


def default_resolver(entry: str) -> Path:
    """Library id -> recipe file (shipped recipes)."""
    if not re.fullmatch(r"[A-Za-z0-9_\-.]+(/[A-Za-z0-9_\-.]+)*", entry or "") or ".." in entry:
        raise RecipeError(f"bad entry id {entry!r}")
    p = ROOT / "recipes" / f"{entry}.json"
    if not p.is_file():
        raise RecipeError(f"no library entry {entry!r}")
    return p


def validate(seq: dict) -> list[str]:
    errs = []
    if not is_sequence(seq):
        errs.append('a sequence needs "type": "sequence"')
    steps = seq.get("steps")
    if not isinstance(steps, list) or not steps:
        errs.append("a sequence needs a non-empty 'steps' list")
        return errs
    rate = seq.get("rate") or {}
    if rate and not ({"interval", "spm"} & set(rate)):
        errs.append("rate needs 'interval' (seconds) or 'spm' (steps per minute)")
    total = 0
    for i, s in enumerate(steps):
        if not isinstance(s, dict) or not s.get("entry"):
            errs.append(f"step {i}: needs an 'entry'")
            continue
        if len({"at", "offset", "gap"} & set(s)) > 1:
            errs.append(f"step {i}: give one of at / offset / gap")
        t = s.get("take", "auto")
        if t not in (None, "auto") and not (isinstance(t, int) or (isinstance(t, str) and t.isdigit())):
            errs.append(f"step {i}: take must be 'auto' or a variation number")
        rep = int(s.get("repeat", 1) or 1)
        if rep < 1:
            errs.append(f"step {i}: repeat must be at least 1")
        total += rep
    if total > MAX_EVENTS:
        errs.append(f"too many events ({total} > {MAX_EVENTS})")
    return errs


def base_interval(seq: dict) -> float:
    rate = seq.get("rate") or {}
    if rate.get("spm"):
        return 60.0 / max(1.0, float(rate["spm"]))
    return max(0.01, float(rate.get("interval", 0.5)))


# --------------------------------------------------------------------------------------------- takes
def take_key(recipe: dict):
    """The recorded gestures a concrete recipe plays (stem families), or None (synthesis: every variation is a
    different take)."""
    import importlib
    keys = []
    for layer in recipe.get("layers", []):
        for b in (layer.get("chain") or [layer]):
            mod = STEMS_MODULES.get(b.get("model"))
            if mod is None:
                continue
            m = importlib.import_module(f".models.{mod}", __package__)
            try:
                keys.append(m.take_key(REGISTRY[b["model"]].resolve(b.get("params"))))
            except (RecipeError, KeyError, ValueError, TypeError):
                keys.append(None)
    return tuple(keys) if keys and any(k is not None for k in keys) else None


def concrete(raw: dict, macros: dict | None, overrides: dict | None, variation: int) -> tuple[dict, int | None]:
    """Entry + macros + variation + exact overrides, as the slider app does it. Returns (recipe, seed)."""
    from .recipe import apply_overrides, resolve
    r, vseed = resolve(raw, macros or {}, int(variation))
    r = copy.deepcopy(r)
    apply_overrides(r, overrides or {})
    seed = vseed if variation else None
    if seed is not None:
        r["seed"] = seed
    r["category"] = raw.get("category", r.get("category", "sfx"))
    return r, seed


# --------------------------------------------------------------------------------------------- event cache
class _EventCache:
    """Byte-bounded LRU of finished event buffers, keyed by the concrete recipe, seed, rate and level."""

    def __init__(self):
        self.d: OrderedDict = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def limit() -> int:
        try:
            return int(float(os.environ.get("SOUNDLAB_SEQ_CACHE_MB", "256")) * 2 ** 20)
        except ValueError:
            return 256 * 2 ** 20

    def get(self, key):
        with self.lock:
            v = self.d.get(key)
            if v is None:
                self.misses += 1
                return None
            self.d.move_to_end(key)
            self.hits += 1
            return v

    def has(self, key) -> bool:
        with self.lock:
            return key in self.d

    def put(self, key, val):
        nb = val[0].nbytes
        lim = self.limit()
        with self.lock:
            if key in self.d or nb > lim:
                return
            self.d[key] = val
            self.bytes += nb
            while self.bytes > lim and self.d:
                _, v = self.d.popitem(last=False)
                self.bytes -= v[0].nbytes

    def clear(self):
        with self.lock:
            self.d.clear()
            self.bytes = 0
            self.hits = self.misses = 0


EVENTS = _EventCache()


def onset_index(x: np.ndarray, sr: int, below_db: float = 30.0, pre_s: float = 0.002) -> int:
    a = np.abs(x)
    pk = float(a.max()) if len(a) else 0.0
    if pk <= 0:
        return 0
    i = int(np.argmax(a >= pk * 10 ** (-below_db / 20)))
    return max(0, i - int(pre_s * sr))


def event_key(r: dict, seed, sr, level) -> tuple:
    return (json.dumps(r, sort_keys=True, default=repr), seed, sr, level)


def render_event(r: dict, seed: int | None, sr: int | None = None, level: str = "lufs") -> tuple[np.ndarray, int, int]:
    """One event as the slider app plays the entry on its own: ring out, trim the silent tail, the loudness gain for
    the entry's category (level 'lufs'; no limiter here, so peaks may pass 0 dBFS until the mix is finished) or the
    raw render ('raw'). Cached. Returns (mono audio, sr, onset index)."""
    key = event_key(r, seed, sr, level)
    hit = EVENTS.get(key)
    if hit is not None:
        return hit
    from .finish import finish, momentary_max_lufs, target_for
    from .recipe import render
    y, rsr = render(r, sample_rate=sr, seed=seed, ring_out=True)
    y, _ = finish(np.asarray(y, float), rsr, r, level="peak", trim=True)        # trim only
    if level == "lufs":         # the entry's loudness gain, without its limiter: the mix is limited once, at the end
        now = momentary_max_lufs(y, rsr)
        if np.isfinite(now):
            y = y * 10 ** ((target_for(r) - now) / 20)
    y = np.ascontiguousarray(y, dtype=float)
    val = (y, rsr, onset_index(y, rsr))
    EVENTS.put(key, val)
    return val


def render_events(events: list[dict], sample_rate: int | None = None, level: str = "lufs") -> tuple[list, int]:
    """Every event's buffer, in order, and how many had to be rendered (the rest came from the cache). With more than one thread (SOUNDLAB_THREADS, or the slider app) the events
    not yet cached render side by side; each one's samples are the same either way."""
    from . import fastpath
    n = fastpath.threads()
    todo, seen = [], set()
    for e in events:
        k = event_key(e["recipe"], e["seed"], sample_rate, level)
        if k not in seen and not EVENTS.has(k):
            seen.add(k)
            todo.append(e)
    if n > 1 and len(todo) > 1:
        stale = getattr(fastpath._local, "is_stale", None)

        def one(e):
            with fastpath.cancel_scope(stale):
                check()
                return render_event(e["recipe"], e["seed"], sample_rate, level)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(n, len(todo)), thread_name_prefix="soundlab-seq") as ex:
            list(ex.map(one, todo))
    out = []
    for e in events:
        check()     # the slider app stops a render a newer one replaced
        out.append(render_event(e["recipe"], e["seed"], sample_rate, level))
    return out, len(todo)


# --------------------------------------------------------------------------------------------- planning
def step_key(step: dict, index: int) -> int:
    """What a step's random draws are keyed on: its "key" if it has one (the app gives every step a key, so
    moving a step keeps its takes and jitter), else its position."""
    k = step.get("key")
    return abs(int(k)) if isinstance(k, (int, float)) or (isinstance(k, str) and k.isdigit()) else index


def plan(seq: dict, resolver: Callable[[str], Path] | None = None) -> list[dict]:
    """The events of a sequence (entry, concrete recipe, seed, take), before timing. Deterministic for the
    sequence seed; depends only on the steps' entries, settings, take choices and repeat counts."""
    resolver = resolver or default_resolver
    errs = validate(seq)
    if errs:
        raise RecipeError("; ".join(errs))
    sseed = int(seq.get("seed", 1) or 0)
    raws: dict[str, dict] = {}
    prev: dict[str, tuple] = {}         # entry -> (variation, take key) of its previous event
    events = []
    for si, step in enumerate(seq["steps"]):
        entry = step["entry"]
        if entry not in raws:
            raws[entry] = json.loads(Path(resolver(entry)).read_text(encoding="utf-8"))
        raw = raws[entry]
        if "layers" not in raw:
            raise RecipeError(f"step {si}: {entry!r} is not a sound recipe")
        macros, overrides = step.get("macros") or {}, step.get("overrides") or {}
        take = step.get("take", "auto")
        rng = np.random.default_rng([sseed, step_key(step, si), 11])
        for k in range(int(step.get("repeat", 1) or 1)):
            if take not in (None, "auto"):
                v = int(take)
                r, seed = concrete(raw, macros, overrides, v)
                key = take_key(r)
            else:
                pv, pkey = prev.get(entry, (None, None))
                first = None
                for _ in range(TAKE_TRIES):
                    v = int(rng.integers(1, 100000))
                    if v == pv:
                        continue
                    r, seed = concrete(raw, macros, overrides, v)
                    key = take_key(r)
                    if first is None:
                        first = (v, r, seed, key)
                    if key is None or pkey is None or key != pkey:
                        break
                else:                   # one recorded gesture only: a fresh remix of it (a different variation)
                    v, r, seed, key = first
            prev[entry] = (v, key)
            events.append({"step": si, "k": k, "entry": entry, "title": raw.get("title") or raw.get("name") or entry,
                           "take": v, "recipe": r, "seed": seed, "gesture": key})
    return events


def schedule(seq: dict, events: list[dict], lengths: list[float], onsets: list[float]) -> list[dict]:
    """Onset time, gain and pan per event (timing-only: cheap). lengths/onsets in seconds per event."""
    base = base_interval(seq)
    hum = seq.get("humanise") or {}
    tj = float(hum.get("timing_ms", 0.0) or 0.0) / 1000.0
    lj = float(hum.get("level_db", 0.0) or 0.0)
    alt = float(hum.get("alternate", 0.0) or 0.0)
    sseed = int(seq.get("seed", 1) or 0)
    out = []
    by_step: dict[int, list[int]] = {}
    for i, e in enumerate(events):
        by_step.setdefault(e["step"], []).append(i)
    prev_on = prev_end = None
    for si, step in enumerate(seq["steps"]):
        idx = by_step.get(si, [])
        if not idx:
            continue
        if "at" in step:
            t0 = float(step["at"])
        elif "gap" in step and prev_end is not None:
            t0 = prev_end + float(step["gap"])
        elif prev_on is not None:
            t0 = prev_on + float(step.get("offset", base))
        else:
            t0 = float(step.get("offset", 0.0)) if "offset" in step else 0.0
        iv = float(step.get("interval", base))
        trng = np.random.default_rng([sseed, step_key(step, si), 22])
        lrng = np.random.default_rng([sseed, step_key(step, si), 33])
        alt_on = step.get("alternate")
        if alt_on is None:
            alt_on = str(events[idx[0]]["recipe"].get("library", "")).startswith("Foley/Footsteps")
        for k, i in enumerate(idx):
            u_t, u_l = float(trng.uniform(-1, 1)), float(lrng.uniform(-1, 1))
            t = t0 + k * iv + u_t * tj         # anything before 0 shifts the whole timeline (render)
            pan = float(step.get("pan", 0.0) or 0.0)
            if alt_on and alt:
                pan += alt * (-1 if k % 2 == 0 else 1)
            out.append({"i": i, "t": t, "gain_db": float(step.get("gain_db", 0.0) or 0.0) + u_l * lj,
                        "pan": max(-1.0, min(1.0, pan))})
        last = out[-1]
        prev_on = t0 + (len(idx) - 1) * iv
        prev_end = last["t"] + lengths[last["i"]] - onsets[last["i"]]
    return out


# --------------------------------------------------------------------------------------------- render
def render(seq: Any, resolver: Callable[[str], Path] | None = None, sample_rate: int | None = None,
           level: str | None = None, event_level: str = "lufs", with_events: bool = False):
    """Render a sequence to one buffer: mono (n,), or stereo (n, 2) when any event is panned.
    Returns (audio, sr, info) where info has the cue list ("events") and render counters."""
    from .finish import finish
    seq = load(seq)
    events = plan(seq, resolver)
    bufs, fresh = render_events(events, sample_rate, event_level)
    srs = {b[1] for b in bufs}
    if len(srs) != 1:
        raise RecipeError(f"entries render at different sample rates {sorted(srs)}; set a sample rate")
    sr = srs.pop()
    lengths = [len(b[0]) / sr for b in bufs]
    onsets = [b[2] / sr for b in bufs]
    cues = schedule(seq, events, lengths, onsets)
    lead = max([0.0] + [onsets[c["i"]] - c["t"] for c in cues])     # nothing starts before 0: shift all
    starts = [int(round((c["t"] + lead - onsets[c["i"]]) * sr)) for c in cues]
    n = max(s + len(bufs[c["i"]][0]) for s, c in zip(starts, cues)) if cues else 0
    stereo = any(abs(c["pan"]) > 1e-9 for c in cues)
    mix = np.zeros((n, 2) if stereo else n)
    for s, c in zip(starts, cues):
        x = bufs[c["i"]][0] * 10 ** (c["gain_db"] / 20)
        if stereo:
            th = (c["pan"] + 1) * np.pi / 4          # constant power; centre = -3 dB each side
            mix[s:s + len(x), 0] += x * np.cos(th) * np.sqrt(2)
            mix[s:s + len(x), 1] += x * np.sin(th) * np.sqrt(2)
        else:
            mix[s:s + len(x)] += x
    lvl = level or seq.get("level") or "lufs"
    raw_peak = float(np.max(np.abs(mix))) if n else 0.0
    if lvl == "lufs":
        out, fin = finish(mix, sr, {"category": seq.get("category", "sfx"),
                                    "loudness_lufs": seq.get("loudness_lufs")}, level="lufs", trim=True)
    else:
        out, fin = finish(mix, sr, None, level="peak", trim=True)
        pk = float(np.max(np.abs(out))) if len(out) else 0.0
        if pk > 0:
            out = out * (10 ** (-1 / 20) / pk)
    master_db = 20 * np.log10((float(np.max(np.abs(out))) + 1e-12) / (raw_peak + 1e-12)) if raw_peak > 0 else 0.0
    info = {"finish": fin, "events_rendered": fresh, "events_cached": len(events) - fresh,
            "master_db": round(float(master_db), 3), "stereo": stereo}
    cue_rows = []
    for j, (s, c) in enumerate(zip(starts, cues)):
        e = events[c["i"]]
        cue_rows.append({"index": j + 1, "time": round(c["t"] + lead, 6), "start": round(s / sr, 6),
                         "entry": e["entry"], "title": e["title"], "take": e["take"], "step": e["step"] + 1,
                         "gain_db": round(c["gain_db"] + master_db, 3), "step_gain_db": round(c["gain_db"], 3),
                         "pan": round(c["pan"], 3), "seconds": round(len(bufs[c["i"]][0]) / sr, 4)})
    info["events"] = cue_rows
    if with_events:
        info["_buffers"] = [bufs[c["i"]][0] for c in cues]
    return out, sr, info


CUE_FIELDS = ["index", "time", "start", "entry", "title", "take", "step", "gain_db", "pan", "seconds", "file"]


def write_cues(path: Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CUE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def export(seq: Any, out: str | Path, resolver: Callable[[str], Path] | None = None, mp3: bool = False,
           oneshots: bool = False, level: str | None = None, sample_rate: int | None = None, bits: int = 24) -> dict:
    """Write the mix (out.wav, and out.mp3 with mp3=True) and, with oneshots=True, a folder out_oneshots/ of
    numbered event files (as rendered, before step gain and pan, peaking at -1 dBFS at most) plus cues.csv: time (the event onset in the mix,
    s), start (where its file starts in the mix, s), entry, take (the entry's variation number: the same take in
    the slider app or `soundlab render --variation`), gain_db (gain to apply to the file to rebuild the mix,
    including the mix's level change), pan."""
    from .audio_io import write_mp3, write_wav
    seq = load(seq)
    audio, sr, info = render(seq, resolver, sample_rate, level, with_events=oneshots)
    out = Path(out)
    if out.suffix.lower() not in (".wav", ".mp3"):
        out = out.with_suffix(".wav")
    res = {"wav": str(write_wav(out.with_suffix(".wav"), audio, sr, bits)), "seconds": round(len(audio) / sr, 3),
           "events": len(info["events"]), "finish": info["finish"]}
    if mp3:
        try:
            res["mp3"] = str(write_mp3(out.with_suffix(".mp3"), audio, sr))
        except Exception as e:      # ffmpeg missing: WAV still written
            res["mp3_error"] = str(e)
    if oneshots:
        folder = out.with_name(out.stem + "_oneshots")
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.wav"):
            old.unlink()
        rows = []
        for row, buf in zip(info["events"], info.pop("_buffers")):
            slug = re.sub(r"[^a-z0-9]+", "_", Path(row["entry"]).name.lower()).strip("_")
            name = f"{row['index']:03d}_{slug}_t{row['take']}.wav"
            pk = float(np.max(np.abs(buf))) if len(buf) else 0.0
            k = min(1.0, 10 ** (-1 / 20) / pk) if pk > 0 else 1.0       # files peak at -1 dBFS at most
            write_wav(folder / name, buf * k, sr, bits)
            rows.append(dict(row, file=name, gain_db=round(row["gain_db"] - 20 * np.log10(k), 3)))
        res["oneshots"] = str(folder)
        res["cues"] = str(write_cues(folder / "cues.csv", rows))
    info.pop("_buffers", None)
    res["cue_list"] = info["events"]
    return res
