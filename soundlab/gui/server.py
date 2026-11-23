"""HTTP server + JSON API for the slider app and the blind test (standard library only)."""
from __future__ import annotations

import base64
import copy
import io
import json
import mimetypes
import os
import re
import threading
import time
import uuid
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from scipy.io import wavfile

from .. import __version__
from ..core import REGISTRY, RecipeError
from ..recipe import load as load_recipe
from ..recipe import render as render_recipe
from ..recipe import resolve

DEFAULT_PORT = 8766            # AIDAW uses 8765
STATIC = Path(__file__).resolve().parent / "static"


def _find_recipes() -> Path:
    """recipes/ next to the package (git clone + `pip install -e .`), else ./recipes in the current folder."""
    for p in (Path(__file__).resolve().parents[2] / "recipes", Path.cwd() / "recipes"):
        if p.is_dir():
            return p
    return Path(__file__).resolve().parents[2] / "recipes"


REPO_RECIPES = _find_recipes()


def default_out_dir() -> Path:
    return Path(os.environ.get("SOUNDLAB_OUT", Path.home() / "SoundLab"))


# --------------------------------------------------------------------------------------------- state
class App:
    """Everything the handlers need: recipe folders, output folder, render cache, blind-test pack."""

    def __init__(self, recipe_dirs=None, out_dir=None, pack_dir=None, cache_size: int = 24, results_size: int = 32,
                 fast: bool = False, threads: int | None = None):
        self.recipe_dirs = [Path(p) for p in (recipe_dirs or [REPO_RECIPES])]
        self.out_dir = Path(out_dir) if out_dir else default_out_dir()
        self.user_recipes = self.out_dir / "recipes"
        self.pack_dir = Path(pack_dir) if pack_dir else self.out_dir / "blindtest"
        self.cache: dict[str, tuple[np.ndarray, int]] = {}
        self.order: list[str] = []
        self.cache_size = cache_size
        self.lock = threading.Lock()
        # slider app speed: finished renders by request (Play again, undo a slider, step back a variation), a
        # generation counter so a render the page has replaced stops early, and one render at a time so the newest
        # gets every core instead of queueing behind stale ones
        self.results: OrderedDict[str, tuple] = OrderedDict()
        self.results_size = results_size
        self.gen = 0
        self.render_lock = threading.Lock()
        self.prefetch = fast
        self.prefetched = None
        if fast:            # the slider app: band loops on every core, and unchanged blocks reused between renders
            from .. import fastpath
            fastpath.set_threads(threads or os.cpu_count() or 1)
            fastpath.set_block_memo(True)

    # ---- recipes
    def recipes(self) -> list[dict]:
        out, seen = [], set()
        for root in self.recipe_dirs + [self.user_recipes]:
            if not root.exists():
                continue
            for p in sorted(root.rglob("*.json")):
                rel = p.relative_to(root).with_suffix("").as_posix()
                if root == self.user_recipes:
                    rel = "my/" + rel          # your saved recipes never shadow the shipped ones
                if rel.startswith("fit/") or rel in seen:
                    continue
                try:
                    r = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if "layers" not in r:
                    continue
                seen.add(rel)
                user = root == self.user_recipes
                out.append({"id": rel, "name": r.get("name") or Path(rel).name, "category": r.get("category", "sfx"),
                            "user": user, "macros": len(r.get("macros", {})),
                            "duration": r.get("duration", 3.0),
                            # library view: plain name, place in the tree, search words; hidden = an old id kept
                            # as an alias (still renders, not listed)
                            "title": r.get("title") or r.get("name") or Path(rel).name,
                            "library": r.get("library") or ("My recipes" if user else "Other"),
                            "tags": r.get("tags", []), "hidden": bool(r.get("hidden")),
                            "replaced_by": r.get("replaced_by")})
        return sorted(out, key=lambda d: (d["category"], d["id"]))

    def recipe_path(self, rid: str) -> Path:
        # relative ids only: no '..', and no leading '/' (Path("recipes") / "/etc/x.json" is "/etc/x.json")
        if not re.fullmatch(r"[A-Za-z0-9_\-.]+(/[A-Za-z0-9_\-.]+)*", rid or "") or ".." in rid:
            raise RecipeError(f"bad recipe id {rid!r}")

        def inside(root: Path, rel: str) -> Path | None:
            p = (root / f"{rel}.json").resolve()
            return p if p.is_file() and root.resolve() in p.parents else None

        if rid.startswith("my/"):
            p = inside(self.user_recipes, rid[3:])
            if p:
                return p
        for root in self.recipe_dirs:
            p = inside(root, rid)
            if p:
                return p
        raise RecipeError(f"no recipe {rid!r}")

    def describe(self, rid: str) -> dict:
        r = load_recipe(self.recipe_path(rid))
        macros = []
        for name, spec in r.get("macros", {}).items():
            d = {"name": name, "default": spec.get("default", 0.5), "doc": spec.get("doc", ""),
                 "label": spec.get("label", name), "ui": spec.get("ui", "front")}
            if "choices" in spec:
                d["choices"] = list(spec["choices"])
            elif "options" in spec:
                d["options"] = list(spec["options"])
                d["default_label"] = spec.get("default_label")
            macros.append(d)
        advanced = []
        for li, layer in enumerate(r["layers"]):
            chain = layer.get("chain") or [layer]
            for bi, b in enumerate(chain):
                prefix = f"{li}" if not layer.get("chain") else f"{li}.{bi}"
                advanced.append(self._block_controls(prefix, layer.get("name", f"layer{li}"), b))
        for j, b in enumerate(r.get("post", [])):
            advanced.append(self._block_controls(f"post.{j}", "post", b))
        return {"id": rid, "name": r.get("name") or Path(rid).name, "category": r.get("category", "sfx"),
                "title": r.get("title") or r.get("name") or Path(rid).name, "library": r.get("library", ""),
                "hidden": bool(r.get("hidden")), "replaced_by": r.get("replaced_by"),
                "duration": r.get("duration", 3.0), "seed": r.get("seed", 0), "macros": macros,
                "loopable": bool(r.get("loop")), "stereo": bool(r.get("stereo")),
                "advanced": advanced, "vary": r.get("vary", {})}

    @staticmethod
    def _block_controls(prefix: str, layer_name: str, block: dict) -> dict:
        m = REGISTRY.get(block.get("model"))
        params = block.get("params", {})
        ctrls = []
        if m is not None:
            for p in m.params:
                val = params.get(p.name, p.default)
                c = {"key": f"{prefix}.{p.name}", "name": p.name, "unit": p.unit, "doc": p.doc, "value": val}
                if p.choices:
                    c.update(kind="choice", choices=list(p.choices))
                elif isinstance(val, bool):
                    c.update(kind="bool")
                elif isinstance(val, (int, float)) and not p.is_list:
                    lo = p.lo if p.lo is not None else (0.0 if val >= 0 else 2 * val)
                    hi = p.hi if p.hi is not None else max(1.0, abs(val) * 4 if val else 1.0)
                    c.update(kind="number", lo=lo, hi=hi, log=bool(lo and lo > 0 and hi / lo >= 50))
                else:
                    c.update(kind="fixed", value=json.dumps(val)[:120])
                ctrls.append(c)
        return {"prefix": prefix, "layer": layer_name, "model": block.get("model"), "controls": ctrls}

    # ---- rendering
    def concrete(self, req: dict) -> tuple[dict, int | None]:
        """Recipe with macros + variation applied, then the advanced overrides (absolute values)."""
        raw = load_recipe(self.recipe_path(req["id"]))
        variation = int(req.get("variation") or 0)
        r, vseed = resolve(raw, req.get("macros") or {}, variation)
        apply_overrides(r, req.get("overrides") or {})
        seed = req.get("seed")
        if seed not in (None, ""):
            seed = int(seed)
        elif variation:
            seed = vseed
        else:
            seed = None
        if seed is not None:
            r["seed"] = seed
        r["category"] = raw.get("category", r.get("category", "sfx"))
        return r, seed

    def finished_audio(self, req: dict):
        """Render as it will be exported: tail trim + loudness per category (or raw peak), or a seamless loop."""
        from .. import render_loop, render_stereo, wants_stereo
        from ..finish import finish
        raw = load_recipe(self.recipe_path(req["id"]))
        r, seed = self.concrete(req)
        r["stereo"] = raw.get("stereo")
        loop = float(req.get("loop") or 0) or None
        trim = req.get("trim", True) is not False
        st = wants_stereo(raw, bool(loop), req.get("stereo"))
        if loop:
            loop = max(1.0, min(loop, 120.0))
            audio, sr = render_loop(r, loop, seed=seed, stereo=st)
        elif st:
            audio, sr = render_stereo(r, seed=seed, ring_out=trim)
        else:
            audio, sr = render_recipe(r, seed=seed, ring_out=trim)
        audio, fin = finish(audio, sr, raw, level=req.get("level") or "lufs", trim=req.get("trim", True) is not False,
                            loop=bool(loop))
        return audio, sr, fin

    def _result_key(self, req: dict) -> str:
        """The request plus the recipe file's identity (a saved recipe can be overwritten on disk)."""
        p = self.recipe_path(req["id"])
        st = p.stat()
        return json.dumps([str(p), st.st_mtime_ns, st.st_size, {k: v for k, v in req.items() if k != "supersede"}],
                          sort_keys=True, default=str)

    def _produce(self, req: dict, rkey: str, stale) -> tuple:
        """Finished render + page pictures for one request, one render at a time; raises Superseded once stale()."""
        from ..fastpath import Superseded, cancel_scope
        with self.render_lock:
            if stale():
                raise Superseded("superseded")
            with cancel_scope(stale):
                audio, sr, fin = self.finished_audio(req)
            mono = audio if audio.ndim == 1 else audio.mean(axis=1)
            hit = (audio, sr, fin, waveform(mono), spectrogram(mono, sr))
        with self.lock:
            self.results[rkey] = hit
            while len(self.results) > self.results_size:
                self.results.popitem(last=False)
        return hit

    def render(self, req: dict) -> dict:
        """Render for the page. With "supersede": true (the slider app) a newer render makes this one stop early and
        answer {"superseded": true}; renders run one at a time, newest first. Repeated requests come from a cache."""
        from ..fastpath import Superseded
        t0 = time.time()
        rkey = self._result_key(req)
        supersede = bool(req.get("supersede"))
        with self.lock:
            if supersede:
                self.gen += 1
            my = self.gen
            hit = self.results.get(rkey)
            if hit is not None:
                self.results.move_to_end(rkey)
        stale = (lambda: self.gen != my) if supersede else (lambda: False)
        if hit is None:
            try:
                hit = self._produce(req, rkey, stale)
            except Superseded:
                return {"superseded": True}
        if supersede and self.prefetch:
            self._start_prefetch(req, my)
        audio, sr, fin, wave, spec = hit
        key = uuid.uuid4().hex[:12]
        with self.lock:
            self.cache[key] = (audio, sr)
            self.order.append(key)
            while len(self.order) > self.cache_size:
                self.cache.pop(self.order.pop(0), None)
        return {"key": key, "url": f"/api/audio/{key}.wav", "sample_rate": sr, "seconds": len(audio) / sr,
                "channels": 1 if audio.ndim == 1 else audio.shape[1], "finish": fin,
                "render_s": round(time.time() - t0, 3), "peak": float(np.max(np.abs(audio)) if len(audio) else 0.0),
                "waveform": wave, "spectrogram": spec}

    # ---- warm-up while the page is idle
    def prefetch_requests(self, req: dict) -> list[dict]:
        """What to render ahead after the page showed `req`, for recipes built on recording stems (their rebuilt
        noise is the slow part, and it depends on the take, not on most sliders): each slider at both ends, then the
        next variation. Those leave every stem's noise for this take cached, so the slider moves after them only
        re-run the cheap stages. Synthesis recipes get nothing (their slow part changes with every slider value)."""
        raw = load_recipe(self.recipe_path(req["id"]))
        models = {b.get("model", "") for layer in raw.get("layers", []) for b in (layer.get("chain") or [layer])}
        if not any(str(m).endswith("_stems") for m in models):
            return []
        base = {k: v for k, v in req.items() if k != "supersede"}
        out = []
        for name, spec in raw.get("macros", {}).items():
            if "choices" in spec or "options" in spec or spec.get("ui") == "hidden" or \
                    any("variation" in str(m.get("to", "")) for m in spec.get("map", [])):
                continue            # a 'seed' slider: every position is a new take; dropdowns and hidden aliases
            for v in (1.0, 0.0):
                out.append(dict(base, macros={**(base.get("macros") or {}), name: v}))
        out.append(dict(base, variation=int(base.get("variation") or 0) + 1))
        return out

    def _start_prefetch(self, req: dict, my: int) -> None:
        sig = json.dumps({k: v for k, v in req.items() if k not in ("supersede", "macros")}, sort_keys=True, default=str)
        with self.lock:
            if sig == self.prefetched:
                return
            self.prefetched = sig
        threading.Thread(target=self._prefetch, args=(req, my), daemon=True, name="soundlab-prefetch").start()

    def _prefetch(self, req: dict, my: int) -> None:
        from ..fastpath import Superseded
        stale = lambda: self.gen != my          # noqa: E731  (any newer page render stops the warm-up)
        try:
            todo = self.prefetch_requests(req)
        except Exception:
            return
        for q in todo:
            try:
                if stale():
                    raise Superseded("superseded")
                rkey = self._result_key(q)
                with self.lock:
                    if rkey in self.results:
                        continue
                self._produce(q, rkey, stale)
            except Superseded:
                with self.lock:         # cut short: the next page render starts it again (done parts are cached)
                    self.prefetched = None
                return
            except Exception:
                continue

    def export_name(self, req: dict, k: int, ext: str) -> str:
        from ..batch import file_name
        raw = load_recipe(self.recipe_path(req["id"]))
        defaults = {n: s.get("default", 0.5) for n, s in raw.get("macros", {}).items()}
        tags = {n: v for n, v in (req.get("macros") or {}).items() if n in defaults and v != defaults[n]}
        if req.get("loop"):
            tags["loop"] = f"{max(1.0, min(float(req['loop']), 120.0)):g}s"
        if audio_channels(req, raw) == 2:
            tags["stereo"] = True
        name = file_name(raw, Path(req["id"]).name, k, tags, ext)
        if req.get("overrides") or req.get("seed") not in (None, ""):
            name = name.replace(f"_v{k + 1:02d}{ext}", f"_custom_v{k + 1:02d}{ext}")
        return name

    def export(self, req: dict, k: int | None = None) -> dict:
        from ..audio_io import write_mp3, write_wav
        k = int(req.get("variation") or 0) if k is None else k
        rq = dict(req, variation=k)
        audio, sr, fin = self.finished_audio(rq)
        folder = self.out_dir / "exports"
        wav = folder / self.export_name(rq, k, ".wav")
        out = {"wav": str(write_wav(wav, audio, sr, 24)), "seconds": round(len(audio) / sr, 3), "finish": fin}
        if req.get("mp3"):
            try:
                out["mp3"] = str(write_mp3(wav.with_suffix(".mp3"), audio, sr))
            except Exception as e:      # ffmpeg missing: WAV still written
                out["mp3_error"] = str(e)
        return out

    def batch(self, req: dict) -> dict:
        n = max(1, min(int(req.get("n") or 5), 100))
        files = [self.export(req, k) for k in range(n)]
        return {"folder": str(self.out_dir / "exports"), "files": files}

    def save(self, req: dict) -> dict:
        raw = load_recipe(self.recipe_path(req["id"]))
        r, seed = self.concrete(dict(req, variation=req.get("variation") or 0))
        name = re.sub(r"[^a-z0-9_]+", "_", (req.get("name") or "").strip().lower()).strip("_")
        if not name:
            raise RecipeError("give the new recipe a name (letters, digits, _)")
        r["name"] = name
        r["based_on"] = req["id"]
        for k in ("hidden", "replaced_by", "library", "tags"):
            r.pop(k, None)              # your copy is listed under My recipes, by its own name
        r["title"] = name
        if raw.get("macros"):
            r["macros"] = copy.deepcopy(raw["macros"])     # sliders now move relative to the saved sound
        if raw.get("vary"):
            r["vary"] = copy.deepcopy(raw["vary"])
        if seed is not None:
            r["seed"] = seed
        r.pop("macro_values", None)
        self.user_recipes.mkdir(parents=True, exist_ok=True)
        p = self.user_recipes / f"{name}.json"
        p.write_text(json.dumps(r, indent=1), encoding="utf-8")
        return {"path": str(p), "id": f"my/{name}"}


def audio_channels(req: dict, raw: dict) -> int:
    from .. import wants_stereo
    return 2 if wants_stereo(raw, bool(req.get("loop")), req.get("stereo")) else 1


def apply_overrides(r: dict, overrides: dict) -> None:
    """Set controls by key: 'L.name' (single-model layer), 'L.B.name' (block B of a chain), 'post.J.name'."""
    for key, val in overrides.items():
        parts = key.split(".")
        if parts[0] == "post":
            blk = r["post"][int(parts[1])]
        else:
            layer = r["layers"][int(parts[0])]
            blk = layer["chain"][int(parts[1])] if layer.get("chain") and len(parts) == 3 else layer
        blk.setdefault("params", {})[parts[-1]] = val


def waveform(x: np.ndarray, bins: int = 800) -> list:
    """[min, max] per bin. Bins tile x, so the non-empty ones are reduceat segments between their starts."""
    if len(x) == 0:
        return []
    edges = np.linspace(0, len(x), bins + 1).astype(int)
    a, b = edges[:-1], edges[1:]
    ok = b > a
    lo, hi = np.zeros(bins), np.zeros(bins)
    lo[ok] = np.minimum.reduceat(x, a[ok])
    hi[ok] = np.maximum.reduceat(x, a[ok])
    return [[round(float(l), 4), round(float(h), 4)] if k else [0, 0] for l, h, k in zip(lo, hi, ok)]


def spectrogram(x: np.ndarray, sr: int, frames: int = 240, bands: int = 96) -> dict:
    """Log-frequency magnitude spectrogram, 0..255 per cell, base64 (frames x bands, row = frame)."""
    nfft, hop = 1024, max(1, len(x) // frames)
    win = np.hanning(nfft)
    pad = np.concatenate([x, np.zeros(nfft)])
    starts = (np.arange(frames) * hop).clip(0, max(0, len(x) - 1))
    S = np.array([np.abs(np.fft.rfft(pad[s:s + nfft] * win)) for s in starts]) + 1e-9
    f = np.fft.rfftfreq(nfft, 1 / sr)
    edges = np.geomspace(30, min(20000, sr / 2), bands + 1)
    idx = np.clip(np.searchsorted(f, edges), 1, len(f) - 1)
    B = np.stack([S[:, a:max(b, a + 1)].max(axis=1) for a, b in zip(idx[:-1], idx[1:])], axis=1)
    db = 20 * np.log10(B / B.max())
    q = np.clip((db + 80) / 80 * 255, 0, 255).astype(np.uint8)
    return {"frames": frames, "bands": bands, "fmin": 30, "fmax": float(edges[-1]),
            "data": base64.b64encode(q.tobytes()).decode()}


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    from ..audio_io import to_int16
    wavfile.write(buf, sr, to_int16(np.clip(x, -1, 1)))
    return buf.getvalue()


# --------------------------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    app: App = None
    server_version = f"SoundLab/{__version__}"

    def log_message(self, fmt, *args):     # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass        # the page dropped the request (a newer slider move replaced it)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _origin_ok(self) -> bool:
        """Only this page may talk to the server: Host must be 127.0.0.1:<port> or localhost:<port> (blocks
        DNS rebinding) and an Origin header, when present, must be the same. Other web sites open in the
        browser are refused."""
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if (self.headers.get("Host") or "") not in allowed:
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{a}" for a in allowed}:
            return False
        return True

    def _forbidden(self, why: str):
        return self._json({"error": f"forbidden: {why}"}, 403)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        if not self._origin_ok():
            return self._forbidden("only http://127.0.0.1:<port> may use this server")
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        path = unquote(u.path)
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path in ("/blindtest", "/blindtest/"):
                return self._static("blindtest.html")
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/info":
                return self._json({"version": __version__, "out_dir": str(self.app.out_dir),
                                   "pack_dir": str(self.app.pack_dir)})
            if path == "/api/recipes":
                return self._json(self.app.recipes())
            if path == "/api/recipe":
                return self._json(self.app.describe(q.get("id", "")))
            m = re.fullmatch(r"/api/audio/([0-9a-f]{12})\.wav", path)
            if m:
                item = self.app.cache.get(m.group(1))
                if not item:
                    return self._json({"error": "expired; render again"}, 404)
                return self._send(200, wav_bytes(*item), "audio/wav")
            if path.startswith("/api/bt/"):
                from . import blindtest
                return blindtest.handle_get(self, path, q)
            return self._json({"error": "not found"}, 404)
        except (RecipeError, KeyError, ValueError, OSError) as e:
            return self._json({"error": str(e)}, 400)

    def do_POST(self):
        if not self._origin_ok():
            return self._forbidden("only http://127.0.0.1:<port> may use this server")
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            return self._forbidden("POST needs Content-Type: application/json")
        path = urlparse(self.path).path
        try:
            req = self._body()
            if path == "/api/render":
                return self._json(self.app.render(req))
            if path == "/api/export":
                return self._json(self.app.export(req))
            if path == "/api/batch":
                return self._json(self.app.batch(req))
            if path == "/api/save":
                return self._json(self.app.save(req))
            if path.startswith("/api/bt/"):
                from . import blindtest
                return blindtest.handle_post(self, path, req)
            return self._json({"error": "not found"}, 404)
        except (RecipeError, KeyError, ValueError, IndexError, TypeError, json.JSONDecodeError, OSError) as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    def _static(self, name: str):
        p = (STATIC / name).resolve()
        if STATIC.resolve() not in p.parents or not p.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if p.suffix in (".html", ".js", ".css"):
            ctype += "; charset=utf-8"
        return self._send(200, p.read_bytes(), ctype)


def make_server(port: int = DEFAULT_PORT, host: str = "127.0.0.1", **app_kw) -> ThreadingHTTPServer:
    app_kw.setdefault("fast", True)
    app = App(**app_kw)
    handler = type("SoundLabHandler", (Handler,), {"app": app})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.app = app
    return srv


def serve(port: int = DEFAULT_PORT, open_browser: bool = True, **app_kw) -> None:
    srv = make_server(port, **app_kw)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"SoundLab {__version__} slider app: {url}  (blind test: {url}blindtest)", flush=True)
    print(f"exports and saved recipes go to {srv.app.out_dir}   - Ctrl+C to stop", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
