"""Parameter fitting by analysis-by-synthesis (roadmap step 8).

Render a recipe, compare it with a reference recording, nudge the recipe's controls, repeat.

Loss = weighted sum of four parts, each scaled so ~1.0 means "clearly different":
  spectral  - long-term 1/3-octave spectrum (LTAS) difference in dB / 10. Timing-free: says "the overall
              colour is right" (bassy vs tinny).
  stft      - multi-resolution log-spectrogram distance (FFT 256/1024/4096) on onset-aligned signals,
              pooled into 1/3-octave bands (cf. Engel et al. 2020 DDSP multi-scale spectral loss). Says "the
              same thing happens at the same time".
  envelope  - 10 ms level envelope in dB / 10: half onset-aligned difference (attack, decay, gaps), half
              difference of level quantiles (timing-free duty cycle / decay shape).
  pitch     - only when the reference has a confident f0: |log2(f0_candidate / f0_reference)| (octaves,
              capped at 1; 1 if the candidate is unpitched). Added because spectral losses alone happily trade
              pitch for brightness (we saw a foghorn fit raise its pitch to match the reference's centroid).
  texture   - McDermott & Simoncelli (2011) style texture statistics: per-band envelope mean, coefficient of
              variation, skewness, and modulation power in 0.5-4 / 4-16 / 16-64 / 64-256 Hz. Says "rough vs smooth,
              steady vs choppy" without needing alignment.

Spectral losses are poor at locating pitch (Turian & Henry 2020), so pitch-like controls are initialised from
the reference's analysed f0 when one exists, and the optimiser then fine-tunes. Optimisers are gradient-free:
a compact CMA-ES (Hansen 2016 tutorial, default) or scipy Nelder-Mead, over controls normalised to [0, 1]
(log-scaled where a range spans 10x or more). All renders use the same seed (common random numbers) so the
loss surface is not dominated by noise.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal, stats

from . import analysis, dsp
from .core import REGISTRY, is_automation
from .recipe import render as render_recipe

FIT_SR = 24000
DEFAULT_WEIGHTS = {"spectral": 1.0, "stft": 1.0, "envelope": 0.7, "texture": 0.7, "sparsity": 0.5, "pitch": 0.7}
THIRD_OCT = [25 * 2 ** (k / 3) for k in range(0, 27)]  # 25 Hz .. ~10 kHz
TEX_BANDS = [(60, 150), (150, 300), (300, 600), (600, 1000), (1000, 1600), (1600, 2500), (2500, 4000),
             (4000, 6300), (6300, 10000)]
MOD_BANDS = [(0.5, 4), (4, 16), (16, 64), (64, 256)]
FLOOR_DB = -45.0  # level differences below this (re peak) are ignored: background noise in field
#                  recordings should not dominate the distance


# ----------------------------------------------------------------------------- features
def sparsity_stats(x, sr, lo=300.0, hi=10000.0, nfft=512):
    """v0.19 bubble-aware term. STFT power (lo..hi Hz, frames within 30 dB of the loudest) is divided by its
    local average over 15 bins x 9 frames; the log10 quantiles (50 / 99 / 99.9 %) of the result measure how
    sparse the sound is in time and frequency. Gaussian noise, however it is shaped, has fixed, light-tailed
    statistics (power ~ exponential); bubbles, droplets and crackle are short ringing blobs with a heavy tail.
    Without this, the distance prefers band-shaped noise for water. Related: spectral flatness and kurtosis
    descriptors (Peeters 2004; the Sound Design Toolkit's spectral features)."""
    from scipy import ndimage
    f, t, Z = signal.stft(x, sr, nperseg=nfft)
    P = np.abs(Z[(f >= lo) & (f <= min(hi, sr / 2 - 1))]) ** 2 + 1e-20
    e = P.sum(0)
    P = P[:, e > e.max() * 1e-3]
    if P.shape[1] < 9:
        return np.zeros(3)
    N = P / (ndimage.uniform_filter(P, (15, 9), mode="nearest") + 1e-20)
    return np.log10(np.percentile(N, [50, 99, 99.9]))


def prepare(x, sr, start=None, end=None, sr_out=FIT_SR):
    """Crop, mono, resample to the fitting rate, peak-normalise."""
    x = np.asarray(x, float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    a = int((start or 0) * sr)
    b = int(end * sr) if end else len(x)
    x = x[a:b]
    if sr != sr_out:
        g = math.gcd(int(sr), int(sr_out))
        x = signal.resample_poly(x, sr_out // g, sr // g)
    return x / (np.max(np.abs(x)) + 1e-12)


def _band_edges(fc):
    return fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)


def _pool_third_octave(P, f):
    rows = []
    for fc in THIRD_OCT:
        lo, hi = _band_edges(fc)
        m = (f >= lo) & (f < hi)
        rows.append(P[m].sum(axis=0) if m.any() else np.zeros(P.shape[1:]))
    return np.array(rows)


def ltas_db(x, sr):
    f, P = signal.welch(x, sr, nperseg=4096 if len(x) >= 4096 else len(x))
    B = _pool_third_octave(P[:, None], f)[:, 0]
    d = 10 * np.log10(B / (B.sum() + 1e-20) + 1e-12)
    return np.maximum(d, FLOOR_DB)


def onset_index(x, sr, thresh_db=-20.0):
    env = analysis.envelope(x, sr, 5.0)
    i = np.argmax(env > env.max() * 10 ** (thresh_db / 20))
    return int(i)


def aligned(x, sr, n):
    """Signal starting 20 ms before its onset, padded/truncated to n samples."""
    i = max(0, onset_index(x, sr) - int(0.02 * sr))
    y = x[i:i + n]
    return np.pad(y, (0, n - len(y)))


def env_db(x, sr, hop_s=0.01):
    e = analysis.envelope(x, sr, 10.0)[:: int(hop_s * sr)]
    return np.maximum(dsp.db(e / (e.max() + 1e-12)), FLOOR_DB)


def mr_stft(x, sr, sizes=(256, 1024, 4096)):
    out = []
    for nf in sizes:
        if len(x) < nf:
            continue
        f, t, Z = signal.stft(x, sr, nperseg=nf, noverlap=nf * 3 // 4, boundary=None, padded=False)
        P = _pool_third_octave(np.abs(Z) ** 2, f)
        out.append(P)
    return out


def texture_stats(x, sr):
    """Per-band envelope statistics (McDermott & Simoncelli 2011, simplified)."""
    stats_ = []
    d = max(1, sr // 1000)
    for lo, hi in TEX_BANDS:
        if hi >= sr / 2:
            stats_.append(np.zeros(3 + len(MOD_BANDS)))
            continue
        sos = signal.butter(2, [lo, hi], btype="band", fs=sr, output="sos")
        b = signal.sosfilt(sos, x)
        e = np.abs(signal.hilbert(b)) ** 0.3  # compressive nonlinearity, as in the texture model
        e = signal.decimate(e, d) if d > 1 else e
        esr = sr / d
        mu = e.mean() + 1e-9
        cv = e.std() / mu
        sk = float(stats.skew(e)) if e.std() > 1e-9 else 0.0
        f, P = signal.welch(e - mu, esr, nperseg=min(len(e), 2048))
        tot = P.sum() + 1e-20
        mods = [P[(f >= a) & (f < c)].sum() / tot for a, c in MOD_BANDS]
        stats_.append(np.array([mu, cv, np.tanh(sk / 3)] + mods))
    S = np.array(stats_)
    S[:, 0] = S[:, 0] / (S[:, 0].max() + 1e-12)  # relative band levels
    return S


@dataclass
class Target:
    x: np.ndarray
    sr: int
    ltas: np.ndarray
    env: np.ndarray
    stft: list
    tex: np.ndarray
    n: int
    f0: float | None = None
    sparse: np.ndarray | None = None

    @classmethod
    def from_signal(cls, x, sr):
        xa = aligned(x, sr, len(x))
        f0 = analysis.f0_estimate(x, sr, return_info=True)["f0_hz"]
        return cls(x, sr, ltas_db(x, sr), env_db(xa, sr), mr_stft(xa, sr), texture_stats(x, sr), len(x), f0)


def _stft_dist(A, B):
    ds = []
    for a, b in zip(A, B):
        m = min(a.shape[1], b.shape[1])
        a, b = a[:, :m], b[:, :m]
        la = np.maximum(10 * np.log10(a / (a.max() + 1e-20) + 1e-12), FLOOR_DB - 5)
        lb = np.maximum(10 * np.log10(b / (b.max() + 1e-20) + 1e-12), FLOOR_DB - 5)
        ds.append(np.mean(np.abs(la - lb)) / 10)
    return float(np.mean(ds)) if ds else 0.0


def _env_dist(e_ref, e_c):
    """Half onset-aligned envelope difference, half difference of the sorted envelopes (level quantiles).
    The quantile half is timing-free: it compares how much of the time the sound is loud vs quiet (duty cycle,
    decay shape), so a quack series slightly out of step is not punished as if it were a different sound."""
    e_c = np.pad(e_c, (0, max(0, len(e_ref) - len(e_c))), constant_values=FLOOR_DB)
    aligned_d = float(np.mean(np.abs(e_c - e_ref)))
    q = np.linspace(0, 100, 41)
    quant_d = float(np.mean(np.abs(np.percentile(e_c, q) - np.percentile(e_ref, q))))
    return (0.5 * aligned_d + 0.5 * quant_d) / 10


def distance(target: Target, y: np.ndarray, sr: int | None = None, weights=None) -> dict:
    """Loss components between a Target and a candidate signal at the same rate."""
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    sr = sr or target.sr
    y = np.asarray(y, float)
    y = y / (np.max(np.abs(y)) + 1e-12)
    ya = aligned(y, sr, target.n)
    parts = {
        "spectral": float(np.mean(np.abs(ltas_db(y, sr) - target.ltas)) / 10),
        "stft": _stft_dist(target.stft, mr_stft(ya, sr)),
        "envelope": _env_dist(target.env, env_db(ya, sr)[: len(target.env)]),
    }
    tc = texture_stats(y, sr)
    tdiff = np.abs(tc - target.tex)
    tdiff[:, 3:] *= 2  # modulation shares are small numbers
    parts["texture"] = float(np.mean(tdiff))
    if getattr(target, "sparse", None) is None:
        target.sparse = sparsity_stats(target.x, target.sr)
    parts["sparsity"] = float(min(2.0, np.mean(np.abs(sparsity_stats(y, sr) - target.sparse)) / 0.1))
    keys = [k for k in DEFAULT_WEIGHTS if k != "pitch"]
    if target.f0:
        # pitch term (only when the reference has a confident pitch): distance in octaves, 1.0 if unpitched
        fc = analysis.f0_estimate(y, sr, return_info=True)["f0_hz"]
        parts["pitch"] = float(min(abs(math.log2(fc / target.f0)), 1.0)) if fc else 1.0
        keys.append("pitch")
    else:
        parts["pitch"] = 0.0
    parts["total"] = float(sum(w[k] * parts[k] for k in keys) / sum(w[k] for k in keys))
    return {k: round(v, 4) for k, v in parts.items()}


def compare_signals(ref, sr_ref, cand, sr_cand, ref_start=None, ref_end=None, weights=None):
    r = prepare(ref, sr_ref, ref_start, ref_end)
    c = prepare(cand, sr_cand)
    return distance(Target.from_signal(r, FIT_SR), c, FIT_SR, weights)


# ----------------------------------------------------------------------------- parameters
@dataclass
class FitParam:
    layer: int
    block: int
    name: str
    lo: float
    hi: float
    log: bool
    integer: bool

    @property
    def key(self):
        return f"{self.layer}.{self.block}.{self.name}" if self.block else f"{self.layer}.{self.name}"

    def to_unit(self, v):
        v = min(max(float(v), self.lo), self.hi)
        if self.log:
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def from_unit(self, u):
        u = min(max(float(u), 0.0), 1.0)
        v = math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo))) if self.log \
            else self.lo + u * (self.hi - self.lo)
        return int(round(v)) if self.integer else v


SKIP = {"start", "at", "seed"}


def _blocks(recipe):
    for li, layer in enumerate(recipe["layers"]):
        chain = layer.get("chain") or [layer]
        for bi, b in enumerate(chain):
            yield li, bi, b


def _get_block(recipe, li, bi):
    layer = recipe["layers"][li]
    return (layer.get("chain") or [layer])[bi]


def parse_param(spec: str, recipe) -> FitParam:
    """'L.name', 'L.B.name' or with bounds 'L.name:lo:hi'. Layer gain is 'L.gain_db'."""
    parts = spec.split(":")
    path = parts[0].split(".")
    li = int(path[0])
    bi = int(path[1]) if len(path) == 3 else 0
    name = path[-1]
    lo = hi = None
    if name == "gain_db":
        lo, hi, integer = -30.0, 10.0, False
    else:
        b = _get_block(recipe, li, bi)
        m = REGISTRY[b["model"]]
        p = next((q for q in m.params if q.name == name), None)
        if p is None:
            raise ValueError(f"{b['model']} has no control {name!r}")
        lo, hi = p.lo, p.hi
        integer = isinstance(p.default, int) and not isinstance(p.default, bool)
    if len(parts) == 3:
        lo, hi = float(parts[1]), float(parts[2])
    if lo is None or hi is None:
        raise ValueError(f"{spec}: give bounds as {parts[0]}:lo:hi")
    log = lo > 0 and hi / lo >= 10
    return FitParam(li, bi, name, float(lo), float(hi), log, integer)


def default_params(recipe) -> list[FitParam]:
    """Every scalar numeric control with a full range, except times (start/at) and counts."""
    out = []
    for li, bi, b in _blocks(recipe):
        m = REGISTRY[b["model"]]
        if m.stage not in ("sound", "excitation", "resonator", "radiation", "synth", "fx"):
            continue
        cur = m.resolve(b.get("params"))
        for p in m.params:
            if p.name in SKIP or p.lo is None or p.hi is None or p.choices or p.is_list:
                continue
            if isinstance(p.default, bool) or isinstance(p.default, int):
                continue
            if is_automation(cur.get(p.name)):
                continue
            out.append(FitParam(li, bi, p.name, float(p.lo), float(p.hi), p.lo > 0 and p.hi / p.lo >= 10, False))
    return out


def current_value(recipe, fp: FitParam):
    if fp.name == "gain_db":
        return float(recipe["layers"][fp.layer].get("gain_db", 0.0))
    b = _get_block(recipe, fp.layer, fp.block)
    return REGISTRY[b["model"]].resolve(b.get("params"))[fp.name]


def apply(recipe, fps, values):
    r = copy.deepcopy(recipe)
    for fp, v in zip(fps, values):
        if fp.name == "gain_db":
            r["layers"][fp.layer]["gain_db"] = round(float(v), 3)
        else:
            b = _get_block(r, fp.layer, fp.block)
            b.setdefault("params", {})[fp.name] = v if fp.integer else round(float(v), 5)
    return r


# ----------------------------------------------------------------------------- optimisers
def cmaes(f, x0, sigma0=0.2, budget=200, seed=0, callback=None):
    """Minimal (mu/mu_w, lambda)-CMA-ES on [0,1]^d (Hansen, 'The CMA Evolution Strategy: A Tutorial', 2016).
    Out-of-box samples are clipped for evaluation and penalised by their distance to the box."""
    rng = np.random.default_rng(seed)
    d = len(x0)
    lam = 4 + int(3 * math.log(d))
    mu = lam // 2
    wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wts /= wts.sum()
    mueff = 1 / np.sum(wts ** 2)
    cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
    cs = (mueff + 2) / (d + mueff + 5)
    c1 = 2 / ((d + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (d + 1)) - 1) + cs
    chiN = math.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d))
    m = np.array(x0, float)
    sigma = sigma0
    pc = np.zeros(d)
    ps = np.zeros(d)
    C = np.eye(d)
    best = (f(np.clip(m, 0, 1)), m.copy())
    evals = 1
    gen = 0
    while evals + lam <= budget:
        gen += 1
        evals_c, evecs = np.linalg.eigh(C)
        evals_c = np.maximum(evals_c, 1e-20)
        B, D = evecs, np.sqrt(evals_c)
        z = rng.standard_normal((lam, d))
        y = z * D @ B.T
        X = m + sigma * y
        fit = []
        for xi in X:
            xc = np.clip(xi, 0, 1)
            v = f(xc) + 1.0 * np.sum((xi - xc) ** 2)
            fit.append(v)
            if v < best[0]:
                best = (v, xc.copy())
        evals += lam
        idx = np.argsort(fit)
        yw = (wts[:, None] * y[idx[:mu]]).sum(axis=0)
        m = m + sigma * yw
        invsqrtC = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ yw
        hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN < 1.4 + 2 / (d + 1)
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * yw
        ymu = y[idx[:mu]]
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (not hsig) * cc * (2 - cc) * C) \
            + cmu * (wts[:, None] * ymu).T @ ymu
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = min(sigma, 0.5)
        if callback:
            callback(gen, evals, best[0])
        if sigma < 1e-3:
            break
    return best[1], best[0], evals


def nelder_mead(f, x0, budget=200, callback=None):
    from scipy.optimize import minimize
    count = [0]
    best = [f(np.clip(x0, 0, 1)), np.array(x0, float)]

    def g(x):
        xc = np.clip(x, 0, 1)
        v = f(xc) + np.sum((x - xc) ** 2)
        count[0] += 1
        if v < best[0]:
            best[0], best[1] = v, xc.copy()
        if callback and count[0] % 10 == 0:
            callback(count[0], count[0], best[0])
        return v

    d = len(x0)
    simplex = [np.array(x0, float)] + [np.clip(np.array(x0, float) + 0.15 * np.eye(d)[i], 0, 1) for i in range(d)]
    minimize(g, x0, method="Nelder-Mead", options={"maxfev": budget, "initial_simplex": np.array(simplex),
                                                   "xatol": 1e-3, "fatol": 1e-4})
    return best[1], best[0], count[0] + 1


# ----------------------------------------------------------------------------- driver
PITCH_NAMES = {"f0", "tone_hz", "freq"}
PITCH_TOL_CENTS = 5.0


def lock_pitch(work, fps, r, log=print, iterations=4):
    """Pitch-first rule (v0.6): measure the reference's fundamental, set every pitch control in the fit set to
    it, correct by re-rendering and measuring until within PITCH_TOL_CENTS, then take those controls OUT of the
    optimisation. Spectral losses get stuck trading pitch for brightness; fixing pitch first avoids that.
    Returns (work_with_locked_pitch, remaining_fps, locked list, notes)."""
    notes = []
    pfs = [fp for fp in fps if fp.name in PITCH_NAMES]
    if not pfs:
        return work, fps, [], notes
    info = analysis.f0_estimate(r, FIT_SR, return_info=True)
    f_ref = info["f0_hz"]
    if not f_ref:
        notes.append("pitch-first: reference has no confident pitch; pitch controls are fitted with the rest")
        return work, fps, [], notes
    fixed_seed = int(work.get("seed", 0))
    vals = [current_value(work, fp) for fp in pfs]
    # first guess: move the main (first) pitch control to the reference f0, keep the others' intervals
    ratio = f_ref / float(vals[0])
    cents = None
    for it in range(iterations):
        vals = [min(max(float(v) * ratio, fp.lo), fp.hi) for fp, v in zip(pfs, vals)]
        w = apply(work, pfs, vals)
        y, _ = render_recipe(w, FIT_SR, fixed_seed)
        f_c = analysis.f0_estimate(y, FIT_SR, return_info=True)["f0_hz"]
        if not f_c:
            notes.append("pitch-first: the render has no measurable pitch; used the reference f0 directly")
            break
        cents = 1200 * math.log2(f_ref / f_c)
        if abs(cents) <= PITCH_TOL_CENTS:
            break
        ratio = f_ref / f_c
    work = apply(work, pfs, vals)
    locked = [{"key": fp.key, "value": round(float(v), 3)} for fp, v in zip(pfs, vals)]
    notes.append(f"pitch-first: reference f0 {f_ref:.1f} Hz; locked " + ", ".join(f"{d['key']}={d['value']}" for d in locked)
                 + (f" (render within {abs(cents):.1f} cents)" if cents is not None else ""))
    log(notes[-1])
    return work, [fp for fp in fps if fp.name not in PITCH_NAMES], locked, notes


def fit(recipe, ref, ref_sr, params=None, ref_start=None, ref_end=None, method="cmaes", budget=150, weights=None,
        seed=None, init_from_analysis=True, log=print, pitch_first=True):
    """Fit `recipe` controls to a reference. Returns dict with fitted recipe, losses and history.

    pitch_first (default): pitch controls (f0, tone_hz, freq) are measured from the reference, set, verified on
    a render and locked before anything else is fitted (see lock_pitch)."""
    from . import models  # noqa: F401

    base = copy.deepcopy(recipe)
    r = prepare(ref, ref_sr, ref_start, ref_end)
    target = Target.from_signal(r, FIT_SR)
    dur_ref = len(r) / FIT_SR
    work = copy.deepcopy(base)
    work["sample_rate"] = FIT_SR
    fps = [parse_param(p, work) if isinstance(p, str) else p for p in params] if params else default_params(work)
    if not fps:
        raise ValueError("no controls to fit; pass params like ['0.f0', '0.rasp']")
    notes = []
    locked = []
    if pitch_first:
        work, fps, locked, pn = lock_pitch(work, fps, r, log)
        notes += pn
        if locked:
            base = apply(base, [parse_param(d["key"], base) for d in locked], [d["value"] for d in locked])
    if not fps:
        y0, _ = render_recipe(work, FIT_SR, int(work.get("seed", 0) if seed is None else seed))
        d0 = distance(target, y0, FIT_SR, weights)
        return {"recipe": base, "params": [], "locked": locked, "before": d0, "after": d0, "evaluations": 0,
                "method": method, "history": [], "notes": notes + ["only pitch controls were requested; nothing else fitted"],
                "ref_duration_s": round(dur_ref, 3), "target": target, "base_recipe": work}
    x0 = [fp.to_unit(current_value(work, fp)) for fp in fps]
    if init_from_analysis and not pitch_first:
        info = analysis.f0_estimate(r, FIT_SR, return_info=True)
        for i, fp in enumerate(fps):
            if fp.name in PITCH_NAMES and info["f0_hz"] and fp.lo <= info["f0_hz"] <= fp.hi:
                x0[i] = fp.to_unit(info["f0_hz"])
                notes.append(f"{fp.key} initialised from reference f0 {info['f0_hz']:.1f} Hz")
    fixed_seed = int(work.get("seed", 0) if seed is None else seed)
    cache = {}

    def render_vals(vals):
        rr = apply(work, fps, vals)
        y, _ = render_recipe(rr, FIT_SR, fixed_seed)
        return y

    def loss(u):
        key = tuple(np.round(u, 4))
        if key in cache:
            return cache[key]
        vals = [fp.from_unit(ui) for fp, ui in zip(fps, u)]
        try:
            v = distance(target, render_vals(vals), FIT_SR, weights)["total"]
        except Exception as e:  # a bad corner of parameter space
            log(f"  render failed at {vals}: {e}")
            v = 10.0
        cache[key] = v
        return v

    base_y, _ = render_recipe(work, FIT_SR, fixed_seed)
    before = distance(target, base_y, FIT_SR, weights)
    start_vals = [fp.from_unit(u) for fp, u in zip(fps, x0)]
    history = []

    def cb(gen, evals, best):
        history.append((evals, best))
        log(f"  eval {evals:4d}  best loss {best:.4f}")

    if method == "nelder-mead":
        ub, fb, ne = nelder_mead(loss, np.array(x0), budget, cb)
    else:
        ub, fb, ne = cmaes(loss, np.array(x0), 0.2, budget, fixed_seed, cb)
    vals = [fp.from_unit(u) for fp, u in zip(fps, ub)]
    fitted = apply(base, fps, vals)
    after = distance(target, render_vals(vals), FIT_SR, weights)
    if after["total"] > before["total"]:
        fitted, after, vals = copy.deepcopy(base), before, [current_value(work, fp) for fp in fps]
        notes.append("no improvement found; kept the original values")
    return {
        "recipe": fitted,
        "params": [{"key": fp.key, "lo": fp.lo, "hi": fp.hi, "log": fp.log,
                    "before": current_value(work, fp), "start": s, "after": v}
                   for fp, s, v in zip(fps, start_vals, vals)],
        "locked": locked,
        "before": before, "after": after, "evaluations": ne, "method": method, "history": history,
        "notes": notes, "ref_duration_s": round(dur_ref, 3), "target": target, "base_recipe": work,
    }


def holdout(result, ref, ref_sr, ref_start=None, ref_end=None, weights=None, name="holdout"):
    """Hold-out rule (v0.6): score the recipe before and after fitting on a SECOND recording of the same sound
    that the fit never saw. A fit that only helps on its own clip is overfitting that clip."""
    r = prepare(ref, ref_sr, ref_start, ref_end)
    tgt = Target.from_signal(r, FIT_SR)
    seed = int(result["base_recipe"].get("seed", 0))
    rb = copy.deepcopy(result["base_recipe"])
    rf = copy.deepcopy(result["recipe"])
    out = {"name": name}
    for k, rec in (("before", rb), ("after", rf)):
        rec = copy.deepcopy(rec)
        rec["duration"] = len(r) / FIT_SR
        y, _ = render_recipe(rec, FIT_SR, seed)
        out[k] = distance(tgt, y, FIT_SR, weights)
    out["improved"] = out["after"]["total"] <= out["before"]["total"]
    return out


def ear_renders(result, out_dir, shift=0.25, sr=48000, positions=(0.15, 0.85)):
    """Listening renders: the fitted recipe, then the fitted recipe at a few MACRO positions (each
    numeric macro at 0.15 and 0.85 with the others at default, every choice of a choice macro). A fit that
    only works at one setting shows up here. Recipes without macros fall back to every fitted control moved
    `shift` of its range toward its low and high end."""
    from .audio_io import write_wav
    od = Path(out_dir) / "ear"
    od.mkdir(parents=True, exist_ok=True)
    rec = result["recipe"]
    specs = rec.get("macros") or {}
    if specs:
        files = []
        jobs = [("fitted", {})]
        for name, spec in specs.items():
            if "choices" in spec:
                jobs += [(f"{name}_{c}", {name: c}) for c in spec["choices"] if c != spec.get("default")]
            else:
                jobs += [(f"{name}{v:g}", {name: v}) for v in positions]
        for tag, mv in jobs:
            y, _ = render_recipe(rec, sr, macros=mv)
            p = od / f"{tag}.wav"
            write_wav(p, y / (np.max(np.abs(y)) + 1e-12) * 0.9, sr)
            files.append(str(p))
        return files
    fps = [parse_param(p["key"] + f":{p['lo']}:{p['hi']}", rec) for p in result["params"]]
    files = []
    for tag, d in (("fitted", 0.0), ("low", -shift), ("high", shift)):
        vals = [fp.from_unit(fp.to_unit(current_value(rec, fp)) + d) for fp in fps]
        y, _ = render_recipe(apply(rec, fps, vals) if fps else rec, sr)
        p = od / f"{tag}.wav"
        write_wav(p, y / (np.max(np.abs(y)) + 1e-12) * 0.9, sr)
        files.append(str(p))
    return files


FEATURE_KEYS = ["f0_hz", "voiced_fraction", "spectral_centroid_hz", "rolloff85_hz", "spectral_flatness",
                "envelope_kind", "decay_t60_s", "event_rate_hz", "envelope_modulation_hz", "attack_s"]


def quick_features(x, sr):
    S = np.abs(analysis.stft(x, sr)[2])
    f = np.fft.rfftfreq(analysis.N_FFT, 1 / sr)
    cen, roll, flat = analysis.spectral_stats(S, f)
    p = analysis.f0_estimate(x, sr, return_info=True)
    e = analysis.envelope_class(x, sr)
    m = analysis.modulation_rate(x, sr)
    env = analysis.envelope(x, sr)
    pk = env.max() + 1e-12
    att = max(int(np.argmax(env > 0.9 * pk)) - int(np.argmax(env > 0.1 * pk)), 0) / sr
    out = {"f0_hz": p["f0_hz"], "voiced_fraction": p["voiced_fraction"], "spectral_centroid_hz": cen,
           "rolloff85_hz": roll, "spectral_flatness": flat, "envelope_kind": e["kind"],
           "decay_t60_s": e["decay_t60_s"], "event_rate_hz": e["event_rate_hz"], "envelope_modulation_hz": m,
           "attack_s": att}
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def report(result, ref_x, out_dir, ref_name="reference", recipe_name="recipe"):
    """Write fitted.json, report.json, report.md and comparison.png into out_dir."""
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "fitted.json").write_text(json.dumps(result["recipe"], indent=1))
    tgt = result["target"]
    yb, _ = render_recipe(result["base_recipe"], FIT_SR) if "base_recipe" in result else (None, None)
    ya, _ = render_recipe(result["recipe"], FIT_SR)
    fr = quick_features(tgt.x, FIT_SR)
    fa = quick_features(ya, FIT_SR)
    fb = quick_features(yb, FIT_SR) if yb is not None else {}
    rep = {k: v for k, v in result.items() if k not in ("recipe", "target", "base_recipe")}
    rep["features"] = {"reference": fr, "before": fb, "after": fa}
    (od / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    L = [f"# Fit report: {recipe_name} -> {ref_name}", "",
         f"Method {result['method']}, {result['evaluations']} renders, reference {result['ref_duration_s']} s.", "",
         "## Distance to the reference (lower is better)", "", "| part | before | after |", "|---|---|---|"]
    for k in list(DEFAULT_WEIGHTS) + ["total"]:
        L.append(f"| {k} | {result['before'][k]} | {result['after'][k]} |")
    L += ["", "## Controls", "", "| control | range | before | after |", "|---|---|---|---|"]
    for p in result["params"]:
        fmt = lambda v: f"{v:.4g}" if isinstance(v, (int, float)) else str(v)
        L.append(f"| {p['key']} | {fmt(p['lo'])} .. {fmt(p['hi'])}{' (log)' if p['log'] else ''} | "
                 f"{fmt(p['before'])} | {fmt(p['after'])} |")
    L += ["", "## Features", "", "| feature | reference | before | after |", "|---|---|---|---|"]
    for k in FEATURE_KEYS:
        L.append(f"| {k} | {fr.get(k)} | {fb.get(k)} | {fa.get(k)} |")
    if result.get("locked"):
        L += ["", "## Locked first (pitch-first rule)", ""] + [f"- {d['key']} = {d['value']}" for d in result["locked"]]
    hos = result.get("holdouts") or []
    L += ["", "## Hold-out check (second recording, not used for fitting)", ""]
    if hos:
        L += ["| recording | before | after | verdict |", "|---|---|---|---|"]
        for h in hos:
            L.append(f"| {h['name']} | {h['before']['total']} | {h['after']['total']} | "
                     f"{'ok' if h['improved'] else 'WORSE: likely overfitted to the fit clip'} |")
    else:
        L.append("**Missing.** No hold-out recording was given (`--holdout`). Do not adopt this fit until it has "
                 "been scored on a second recording of the same sound and listened to.")
    if result.get("ear"):
        L += ["", "Listening renders: " + ", ".join(Path(f).name for f in result["ear"]) +
              " (fitted, then macro positions; or, without macros, all fitted controls 25 % of their range lower / higher)."]
    if result["notes"]:
        L += ["", "## Notes", ""] + [f"- {n}" for n in result["notes"]]
    (od / "report.md").write_text("\n".join(L) + "\n")
    _plot(tgt, yb, ya, od / "comparison.png")
    from .audio_io import write_wav
    write_wav(od / "fitted_preview.wav", ya / (np.max(np.abs(ya)) + 1e-12) * 0.9, FIT_SR)
    return od


def _plot(tgt, yb, ya, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    ax[0].semilogx(THIRD_OCT, tgt.ltas, "k-", lw=2, label="reference")
    sr = FIT_SR
    if yb is not None:
        ax[0].semilogx(THIRD_OCT, ltas_db(yb / (np.max(np.abs(yb)) + 1e-12), sr), "C1--", label="before")
    ax[0].semilogx(THIRD_OCT, ltas_db(ya / (np.max(np.abs(ya)) + 1e-12), sr), "C0-", label="after")
    ax[0].set_title("Long-term 1/3-octave spectrum (dB re total)")
    ax[0].legend()
    t = np.arange(len(tgt.env)) * 0.01
    ax[1].plot(t, tgt.env, "k-", lw=2, label="reference")
    for y, st, lab in ((yb, "C1--", "before"), (ya, "C0-", "after")):
        if y is not None:
            e = env_db(aligned(y / (np.max(np.abs(y)) + 1e-12), sr, tgt.n), sr)[: len(t)]
            ax[1].plot(t[: len(e)], e, st, label=lab)
    ax[1].set_title("Onset-aligned level envelope (dB)")
    ax[1].set_ylim(FLOOR_DB - 2, 3)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
