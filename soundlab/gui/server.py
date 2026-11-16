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

    def __init__(self, recipe_dirs=None, out_dir=None, pack_dir=None, cache_size: int = 24):
        self.recipe_dirs = [Path(p) for p in (recipe_dirs or [REPO_RECIPES])]
        self.out_dir = Path(out_dir) if out_dir else default_out_dir()
        self.user_recipes = self.out_dir / "recipes"
        self.pack_dir = Path(pack_dir) if pack_dir else self.out_dir / "blindtest"
        self.cache: dict[str, tuple[np.ndarray, int]] = {}
        self.order: list[str] = []
        self.cache_size = cache_size
        self.lock = threading.Lock()

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
                out.append({"id": rel, "name": r.get("name") or Path(rel).name, "category": r.get("category", "sfx"),
                            "user": root == self.user_recipes, "macros": len(r.get("macros", {})),
                            "duration": r.get("duration", 3.0)})
        return sorted(out, key=lambda d: (d["category"], d["id"]))

    def recipe_path(self, rid: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_\-./]+", rid or "") or ".." in rid:
            raise RecipeError(f"bad recipe id {rid!r}")
        if rid.startswith("my/"):
            p = self.user_recipes / f"{rid[3:]}.json"
            if p.exists():
                return p
        for root in self.recipe_dirs:
            p = root / f"{rid}.json"
            if p.exists():
                return p
        raise RecipeError(f"no recipe {rid!r}")

    def describe(self, rid: str) -> dict:
        r = load_recipe(self.recipe_path(rid))
        macros = []
        for name, spec in r.get("macros", {}).items():
            d = {"name": name, "default": spec.get("default", 0.5), "doc": spec.get("doc", "")}
            if "choices" in spec:
                d["choices"] = list(spec["choices"])
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
                "duration": r.get("duration", 3.0), "seed": r.get("seed", 0), "macros": macros,
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

    def render(self, req: dict) -> dict:
        r, seed = self.concrete(req)
        t0 = time.time()
        audio, sr = render_recipe(r, seed=seed)
        dt = time.time() - t0
        key = uuid.uuid4().hex[:12]
        with self.lock:
            self.cache[key] = (audio, sr)
            self.order.append(key)
            while len(self.order) > self.cache_size:
                self.cache.pop(self.order.pop(0), None)
        return {"key": key, "url": f"/api/audio/{key}.wav", "sample_rate": sr, "seconds": len(audio) / sr,
                "render_s": round(dt, 3), "peak": float(np.max(np.abs(audio)) if len(audio) else 0.0),
                "waveform": waveform(audio), "spectrogram": spectrogram(audio, sr)}

    def export_name(self, req: dict, k: int, ext: str) -> str:
        from ..batch import file_name
        raw = load_recipe(self.recipe_path(req["id"]))
        defaults = {n: s.get("default", 0.5) for n, s in raw.get("macros", {}).items()}
        tags = {n: v for n, v in (req.get("macros") or {}).items() if n in defaults and v != defaults[n]}
        name = file_name(raw, Path(req["id"]).name, k, tags, ext)
        if req.get("overrides") or req.get("seed") not in (None, ""):
            name = name.replace(f"_v{k + 1:02d}{ext}", f"_custom_v{k + 1:02d}{ext}")
        return name

    def export(self, req: dict, k: int | None = None) -> dict:
        from ..audio_io import write_mp3, write_wav
        k = int(req.get("variation") or 0) if k is None else k
        rq = dict(req, variation=k)
        r, seed = self.concrete(rq)
        audio, sr = render_recipe(r, seed=seed)
        folder = self.out_dir / "exports"
        wav = folder / self.export_name(rq, k, ".wav")
        out = {"wav": str(write_wav(wav, audio, sr, 24))}
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
    if len(x) == 0:
        return []
    edges = np.linspace(0, len(x), bins + 1).astype(int)
    return [[round(float(x[a:b].min()), 4), round(float(x[a:b].max()), 4)] if b > a else [0, 0]
            for a, b in zip(edges[:-1], edges[1:])]


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
    wavfile.write(buf, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return buf.getvalue()


# --------------------------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    app: App = None
    server_version = f"SoundLab/{__version__}"

    def log_message(self, fmt, *args):     # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
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
