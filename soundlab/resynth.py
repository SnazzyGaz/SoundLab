"""Analysis and resynthesis (v0.17): turn a recording into an editable, layered recipe.

Pipeline (see docs/resynthesis.md):
1. analyse_clip: envelope, spectral centroid over time, 1/3-octave band energies over time, onsets, f0 track,
   tonal vs noise share (harmonic / transient / residual split).
2. build_recipe: a layered recipe for the kind of sound:
   - impact:  a `bandnoise` body whose band x time grid follows the recording (breakpoints simplified to a
              tolerance), plus a physical model layer at each main onset (e.g. `splash`) for the detail.
   - texture: a stationary `bandnoise` (mean spectrum + random band wobble) and a `bubbles` layer, so the recipe
              loops and varies instead of copying the clip's timeline.
   - voice:   a `harmonic` layer on the measured f0 curve with formants read from the harmonic spectrum, its
              loudness envelope as automation, plus a `bandnoise` layer for the breath/noise residual.
   - choir:   `harmonic` with detuned unison voices on a harmonic-sum pitch track, and a reverb.
3. fit_and_check: pitch first (taken from the analysis and verified on a render, in cents), then the remaining
   free controls fitted with fit.fit on a fitting section, then scored on a held-out section (or a second
   recording) and with a different random seed.
Reference audio never enters the repo; write outputs outside it.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy import ndimage, signal

from . import analysis as A
from . import fit as F
from .models.resynth_blocks import THIRD_OCTAVES

AN_NFFT = 2048


# ----------------------------------------------------------------------------- analysis
def band_grid(x, sr, bands=THIRD_OCTAVES, hop_s=0.01, smooth=3, nfft=AN_NFFT):
    """1/3-octave band levels (dB of mean power spectral density) over time. Returns (times, grid[T, B])."""
    hop = max(1, int(hop_s * sr))
    f, t, Z = signal.stft(x, sr, nperseg=nfft, noverlap=nfft - hop if nfft > hop else 0)
    P = np.abs(Z) ** 2
    cols = []
    for c in bands:
        m = (f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))
        if not m.any():
            m = np.abs(f - c) == np.min(np.abs(f - c))
        cols.append(P[m].mean(0))
    G = np.stack(cols, 1)
    if smooth > 1:
        G = ndimage.uniform_filter1d(G, smooth, axis=0, mode="nearest")
    return t, 10 * np.log10(G + 1e-14)


def simplify_frames(t, G, tol_db=3.0, max_frames=60, floor_db=None):
    """Greedy breakpoint reduction of a band x time grid: keep the frames needed so linear interpolation stays
    within tol_db of the grid in every band (levels below floor_db are clipped first, so silence is cheap)."""
    if floor_db is not None:
        G = np.maximum(G, floor_db)
    keep = [0, len(t) - 1]
    while len(keep) < max_frames:
        ks = sorted(keep)
        approx = np.stack([np.interp(t, t[ks], G[ks, j]) for j in range(G.shape[1])], 1)
        err = np.abs(approx - G).max(1)
        i = int(np.argmax(err))
        if err[i] <= tol_db:
            break
        keep.append(i)
    ks = sorted(set(keep))
    return [[round(float(t[k]), 4), [round(float(v), 1) for v in G[k]]] for k in ks]


def centroid_track(x, sr, hop_s=0.05):
    hop = max(1, int(hop_s * sr))
    f, t, Z = signal.stft(x, sr, nperseg=AN_NFFT, noverlap=AN_NFFT - hop if AN_NFFT > hop else 0)
    S = np.abs(Z)
    return t, (S * f[:, None]).sum(0) / (S.sum(0) + 1e-12)


def harmonic_sum_f0(x, sr, fmin=60.0, fmax=400.0, hop_s=0.25, n_harm=8):
    """Pitch track for dense, reverberant or unison voices: per frame, the f0 whose first n_harm harmonics hold
    the most log-spectral energy (subharmonic summation, Hermes 1988). Returns (times, f0, strength)."""
    nfft = 8192
    hop = int(hop_s * sr)
    f, t, Z = signal.stft(x, sr, nperseg=nfft, noverlap=nfft - hop)
    L = np.log10(np.abs(Z) ** 2 + 1e-12)
    L = L - ndimage.median_filter(L, size=(41, 1), mode="nearest")  # whiten: peaks above local floor
    L = np.maximum(L, 0)
    cands = fmin * 2 ** (np.arange(0, np.log2(fmax / fmin) * 48) / 48)
    score = np.zeros((len(cands), len(t)))
    df = f[1]
    for i, c in enumerate(cands):
        for k in range(1, n_harm + 1):
            b = int(round(k * c / df))
            if b < len(f):
                score[i] += L[max(0, b - 1):b + 2].max(0) / k ** 0.5
    best = score.argmax(0)
    return t, cands[best], score.max(0)


def analyse_clip(x, sr):
    """Features used to build a recipe (also printed in the report)."""
    x = np.asarray(x, float)
    stems, _ = A.hpr_split(x, sr)
    E = lambda s: float(np.sum(np.asarray(s) ** 2)) + 1e-20  # noqa: E731
    tot = E(x)
    env = A.envelope(x, sr)
    ct, cv = centroid_track(x, sr)
    info = A.f0_estimate(x, sr, return_info=True)
    tt, grid = band_grid(x, sr, hop_s=0.02)
    return {
        "duration_s": round(len(x) / sr, 3),
        "peak_db": round(20 * np.log10(np.max(np.abs(x)) + 1e-12), 1),
        "onsets_s": [round(float(o), 3) for o in A.onsets(x, sr)][:12],
        "f0": info,
        "tonal_share": round(E(stems["harmonic"]) / tot, 3),
        "transient_share": round(E(stems["transient"]) / tot, 3),
        "noise_share": round(E(stems["noise"]) / tot, 3),
        "centroid_hz": {"times": [round(float(a), 2) for a in ct[::4]], "hz": [int(v) for v in cv[::4]]},
        "envelope_db_100ms": [round(float(v), 1) for v in 20 * np.log10(env[:: max(1, int(len(env) / (len(x) / sr * 10)))] + 1e-9)],
        "band_levels_db": dict(zip([str(b) for b in THIRD_OCTAVES], [round(float(v), 1) for v in
                                                                      10 * np.log10(np.mean(10 ** (grid / 10), 0))])),
        "stems": stems,
    }


def suppress_tonal(x, sr, fmin=1500.0, fmax=10000.0, prominence_db=12.0, kernel_hz=700.0):
    """Remove narrow tonal peaks (bird chirps, whistles) from a noisy texture: in fmin..fmax, STFT bins more than
    prominence_db above the local median spectrum (over kernel_hz) are set to that median, phase kept.
    Returns (cleaned, fraction of energy removed)."""
    nfft = 2048
    f, t, Z = signal.stft(x, sr, nperseg=nfft)
    S = np.abs(Z)
    k = max(3, int(kernel_hz / (f[1] - f[0])) | 1)
    med = ndimage.median_filter(S, size=(k, 1), mode="nearest")
    band = (f >= fmin) & (f <= fmax)
    mask = (S > med * 10 ** (prominence_db / 20)) & band[:, None]
    mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))  # neighbouring bins and frames too
    mask &= band[:, None]
    Zc = np.where(mask, Z / (S + 1e-12) * np.minimum(S, med), Z)
    _, y = signal.istft(Zc, sr, nperseg=nfft)
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    removed = 1 - float(np.sum(y ** 2)) / (float(np.sum(np.asarray(x) ** 2)) + 1e-20)
    return y, max(0.0, removed)


def estimate_formants(x, sr, f0_hz, n=4, fmax=8000.0):
    """Formants [[Hz, bandwidth Hz, dB]] from the harmonic peaks of a voiced sound: harmonic amplitudes are
    sampled at k*f0, smoothed over 1/3 octave, and the n strongest peaks (with -3 dB widths) are kept."""
    f, P = signal.welch(x, sr, nperseg=AN_NFFT)
    L = 10 * np.log10(P + 1e-14)
    ks = np.arange(1, int(min(fmax, sr / 2 - 1) // f0_hz) + 1)
    fk = ks * f0_hz
    amp = np.array([L[max(0, int(round(v / f[1])) - 2):int(round(v / f[1])) + 3].max() for v in fk])
    grid = np.geomspace(max(100.0, f0_hz * 0.8), fk[-1], 200)
    env = np.interp(np.log2(grid), np.log2(fk), amp)
    env = ndimage.gaussian_filter1d(env, 200 / (np.log2(grid[-1] / grid[0]) * 3) / 2.355)
    pk, _ = signal.find_peaks(env, distance=8)
    if not len(pk):
        pk = np.array([int(np.argmax(env))])
    pk = pk[np.argsort(env[pk])[::-1][:n]]
    top = env[pk].max()
    out = []
    for p in sorted(pk):
        lo = p
        while lo > 0 and env[lo] > env[p] - 3:
            lo -= 1
        hi = p
        while hi < len(env) - 1 and env[hi] > env[p] - 3:
            hi += 1
        out.append([int(grid[p]), int(max(60.0, grid[hi] - grid[lo])), round(float(env[p] - top), 1)])
    return out


def _curve(t, v, tol, max_points=40, log=False):
    """Breakpoints [[t, v], ...] approximating a curve within tol (in log2 units if log)."""
    vv = np.log2(v) if log else np.asarray(v, float)
    keep = [0, len(t) - 1]
    while len(keep) < max_points:
        ks = sorted(keep)
        err = np.abs(np.interp(t, t[ks], vv[ks]) - vv)
        i = int(np.argmax(err))
        if err[i] <= tol:
            break
        keep.append(i)
    ks = sorted(set(keep))
    return [[round(float(t[k]), 3), round(float(v[k]), 2 if log else 4)] for k in ks]


def _level_curve(x, sr, hop_s=0.02, tol_db=2.0, floor_db=-50.0, max_points=40, smooth_s=0.0):
    env = A.envelope(x, sr)
    if smooth_s > 0:  # RMS over smooth_s, so beating between voices does not become level dips
        k = max(1, int(round(smooth_s / (len(x) / sr) * len(env))))
        env = np.sqrt(ndimage.uniform_filter1d(env ** 2, k, mode="nearest"))
    n = len(x)
    t_env = np.linspace(0, n / sr, len(env))
    step = max(1, int(round(len(env) / (n / sr / hop_s))))
    t, e = t_env[::step], env[::step]
    d = np.maximum(20 * np.log10(e / (e.max() + 1e-12) + 1e-12), floor_db)
    pts = _curve(t, d, tol_db, max_points)
    return [[a, round(float(10 ** (b / 20)) if b > floor_db else 0.0, 4)] for a, b in pts]


# ----------------------------------------------------------------------------- recipes
def _macros(kind):
    m = {"level": {"default": 0.5, "doc": "quiet .. loud", "map": [{"to": "master.gain_db", "op": "add",
                                                                     "points": [[0, -12], [0.5, 0], [1, 0]]}]}}
    if kind in ("impact", "texture"):
        m["bubble_size"] = {"default": 0.5, "doc": "small bright bubbles .. big deep bubbles",
                            "map": [{"to": "0.r_max_mm", "op": "mul", "points": [[0, 0.5], [0.5, 1], [1, 2]]},
                                    {"to": "0.r_min_mm", "op": "mul", "points": [[0, 0.7], [0.5, 1], [1, 1.4]]}]}
        m["churn"] = {"default": 0.5, "doc": "few bubbles .. many bubbles",
                      "map": [{"to": "0.rate", "op": "mul", "points": [[0, 0.3], [0.5, 1], [1, 3]]}]}
    if kind in ("voice", "choir"):
        m["pitch"] = {"default": 0.5, "doc": "lower .. higher",
                      "map": [{"to": "0.pitch_st", "op": "add", "points": [[0, -7], [0.5, 0], [1, 7]]}]}
        m["size"] = {"default": 0.5, "doc": "bigger .. smaller (vocal tract)",
                     "map": [{"to": "0.formant_shift", "op": "mul", "points": [[0, 0.8], [0.5, 1], [1, 1.25]]}]}
    return m


def voice_frames(x, sr, f0_t, f0_v, stems, hop_s=0.02, fmax=10000.0):
    """WORLD-style decomposition for the `harmonic` block (Morise et al. 2016, method only):
    env_frames = harmonic amplitudes at k*f0 per frame, interpolated to 1/3-octave centres and normalised to
    0 dB per frame (the vowel; loudness goes to `level`); ap_frames = noise-to-harmonic ratio per band over time
    (aperiodicity), normalised to 0 dB at the noisiest point."""
    nfft = AN_NFFT
    hop = max(1, int(hop_s * sr))
    f, t, Z = signal.stft(stems["harmonic"], sr, nperseg=nfft, noverlap=nfft - hop)
    S = np.abs(Z)
    f0 = np.interp(t, f0_t, f0_v)
    lb = np.log2(THIRD_OCTAVES)
    rows = []
    for j, fj in enumerate(f0):
        ks = np.arange(1, int(min(fmax, sr / 2 - 1) // fj) + 1)
        bins = np.round(ks * fj / f[1]).astype(int)
        amp = np.array([S[max(0, b - 2):b + 3, j].max() for b in bins])
        d = 20 * np.log10(amp + 1e-9)
        row = np.interp(lb, np.log2(ks * fj), d)
        rows.append(row - row.max())
    E = ndimage.uniform_filter1d(np.array(rows), 3, axis=0, mode="nearest")
    env = simplify_frames(t, np.maximum(E, -60), tol_db=3.0, max_frames=40)
    _, Gh = band_grid(stems["harmonic"], sr, hop_s=hop_s)
    _, Gn = band_grid(stems["noise"] + stems["transient"], sr, hop_s=hop_s)
    m = min(len(Gh), len(Gn))
    tt = np.linspace(0, len(x) / sr, m)
    AP = np.clip(Gn[:m] - Gh[:m], -40, 20)
    AP = ndimage.uniform_filter1d(AP, 5, axis=0, mode="nearest")
    AP -= AP.max()
    ap = simplify_frames(tt, np.maximum(AP, -40), tol_db=4.0, max_frames=25)
    return env, ap


def _gated_attack(x, sr, onsets, keep_s=0.07, cut_db=30.0):
    """Band grid of only the first keep_s after each onset (the slap/crack); elsewhere cut_db lower."""
    t, G = band_grid(x, sr, hop_s=0.01, smooth=1, nfft=1024)
    near = np.zeros(len(t), bool)
    for o in onsets:
        near |= (t >= o - 0.01) & (t <= o + keep_s)
    gate = np.where(near, 0.0, -cut_db)
    gate = ndimage.uniform_filter1d(gate, 3, mode="nearest")
    return simplify_frames(t, np.maximum(G + gate[:, None], G.max() - 100), tol_db=3.0, max_frames=60)


def _strong_onsets(x, sr, feats, min_gap=0.25, within_db=12.0):
    env = A.envelope(x, sr)
    strong = []
    for o in feats["onsets_s"]:
        i = min(len(env) - 1, int(o / (len(x) / sr) * len(env)))
        seg = env[i:i + max(1, len(env) // 20)]
        if seg.size and 20 * np.log10(seg.max() / env.max() + 1e-12) > -within_db and \
                all(abs(o - s) > min_gap for s in strong):
            strong.append(o)
    return strong[:2] or [0.03]


def build_recipe(x, sr, kind, name="resynth", detail_model="splash", feats=None):
    """Layered recipe for `kind` in impact | texture | voice | choir (see module doc)."""
    x = np.asarray(x, float)
    feats = feats or analyse_clip(x, sr)
    dur = round(len(x) / sr, 3)
    rec = {"soundlab": 1, "name": f"Resynthesis: {name}", "category": "resynth", "sample_rate": 48000,
           "duration": dur, "seed": 17, "layers": [], "post": [{"model": "eq", "params": {"curve": []}}],
           "master": {"normalize_db": -1.0, "fade_out": 0.02}, "macros": _macros(kind)}
    # bubble sizes from the spectrum: Minnaert f = 3.26 / r, so the band holding most energy sets the radii
    bl = feats["band_levels_db"]
    fc = np.array([float(k) for k in bl])
    lv = np.array(list(bl.values()))
    live = fc[(lv > lv.max() - 15) & (fc >= 300)]
    f_lo, f_hi = (float(live.min()), float(live.max())) if live.size else (500.0, 6000.0)
    r_min, r_max = round(max(0.2, 3.26 / f_hi * 1e3), 2), round(min(40.0, 3.26 / f_lo * 1e3), 2)
    if kind == "impact":
        strong = _strong_onsets(x, sr, feats)
        lvl = _level_curve(x, sr, hop_s=0.01, tol_db=2.0, floor_db=-60.0, max_points=60)
        rec["layers"].append({"name": "bubbles", "model": "bubble_cloud",
                              "params": {"rate": 4000.0, "r_min_mm": r_min, "r_max_mm": r_max, "size_exponent": 2.5,
                                         "amp_exponent": 0.5, "xi": 0.5, "damping": 1.0, "level": lvl}})
        # a quiet noise body (kept at least 12 dB under the bubbles: noise in front is what makes water rushy)
        t, G = band_grid(x, sr, hop_s=0.01, smooth=1, nfft=1024)
        body = simplify_frames(t, G, tol_db=3.0, max_frames=60, floor_db=G.max() - 100)
        rec["layers"].append({"name": "body", "model": "bandnoise", "gain_db": -18.0,
                              "params": {"frames": body, "grain": 0.0}})
        rec["layers"].append({"name": "slap", "model": "bandnoise", "gain_db": -6.0,
                              "params": {"frames": _gated_attack(x, sr, strong), "grain": 0.0}})
        if detail_model == "splash":
            for k, o in enumerate(strong):
                rec["layers"].append({"name": f"splash{k + 1}", "model": "splash", "gain_db": -18.0,
                                      "start": round(max(0.0, o - 0.01), 3),
                                      "params": {"size": 0.15, "speed": 4.0, "bloop": 0.5, "droplets": 0.6,
                                                 "slosh": 0.5, "at": 0.0}})
    elif kind == "texture":
        t, G = band_grid(x, sr, hop_s=0.05)
        mean = 10 * np.log10(np.mean(10 ** (G / 10), 0))
        rec["layers"].append({"name": "bubbles", "model": "bubble_cloud",
                              "params": {"rate": 3000.0, "r_min_mm": r_min, "r_max_mm": r_max, "size_exponent": 2.5,
                                         "amp_exponent": 0.5, "xi": 0.5, "damping": 1.0}})
        rec["layers"].append({"name": "spray", "model": "bandnoise", "gain_db": -20.0,
                              "params": {"frames": [[0.0, [round(float(v), 1) for v in mean]]], "grain": 0.2,
                                         "grain_hz": 6.0}})
    elif kind in ("voice", "choir"):
        stems = feats["stems"]
        if kind == "voice":
            ft, f0 = A.f0_track(x, sr, fmin=150.0, fmax=1500.0)[:2]
            ok = np.isfinite(f0) & (f0 > 0)
            ft, f0 = np.asarray(ft)[ok], np.asarray(f0)[ok]
            f0 = ndimage.median_filter(f0, 5, mode="nearest")
            f0_pts = _curve(ft, f0, tol=20 / 1200, max_points=30, log=True)
        else:
            # two passes: a wide search for the overall pitch, then a narrow one around it (stops octave jumps)
            _, f0w, sw = harmonic_sum_f0(x, sr)
            med = float(np.median(f0w[sw >= np.median(sw)]))
            ft, f0, s = harmonic_sum_f0(x, sr, fmin=med / 1.5, fmax=med * 1.5)
            f0 = ndimage.median_filter(f0, 5, mode="nearest")
            f0_pts = _curve(ft, f0, tol=30 / 1200, max_points=40, log=True)
        fref = float(np.median(f0))
        env, ap = voice_frames(x, sr, ft, f0, stems, hop_s=0.02 if kind == "voice" else 0.1,
                               fmax=10000.0 if kind == "voice" else 5000.0)
        if kind == "choir":  # one sung line through a hall: a steady vowel colour is safer than tracking it
            env = [[0.0, [round(float(v), 1) for v in np.mean([fr[1] for fr in env], 0)]]]
        lv = _level_curve(x, sr, hop_s=0.02 if kind == "voice" else 0.1, floor_db=-60.0,
                          smooth_s=0.0 if kind == "voice" else 0.4)
        hb = {"model": "harmonic",
              "params": {"f0": f0_pts, "formants": [], "env_frames": env, "ap_frames": ap, "tilt_db": 0.0,
                         "breath": 0.1, "pulse_noise": 0.7, "level": lv, "vibrato_hz": 5.0,
                         "vibrato_cents": 5.0 if kind == "voice" else 15.0, "wander_cents": 0.0 if kind == "voice"
                         else 10.0, "voices": 1 if kind == "voice" else 4, "detune_cents": 15.0,
                         "max_hz": 10000.0 if kind == "voice" else 5000.0}}
        if kind == "voice":
            rec["layers"].append({"name": "voice", **hb})
        else:
            rec["layers"].append({"name": "choir", "chain": [hb, {"model": "reverb", "params": {
                "size": 0.9, "damp": 0.5, "predelay_ms": 20.0, "mix": 0.5}}]})
            # the hall's diffuse, noisy part (the residual after the harmonic split) as its own layer
            r = stems["noise"] + stems["transient"]
            t, G = band_grid(r, sr, hop_s=0.1)
            rec["layers"].append({"name": "hall_air", "model": "bandnoise", "gain_db": -12.0, "params": {
                "frames": simplify_frames(t, G, tol_db=4.0, max_frames=30, floor_db=G.max() - 60), "grain": 0.2}})
        rec["f0_reference_hz"] = round(fref, 2)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return rec


FIT_PARAMS = {
    "impact": ["0.rate:200:30000", "0.r_min_mm:0.2:4", "0.r_max_mm:1:20", "0.size_exponent:1:4",
               "0.damping:0.5:3", "0.xi:0:3", "1.gain_db:-30:-12", "2.gain_db:-30:0", "3.gain_db:-40:-6",
               "3.size:0.05:0.6"],
    "texture": ["0.rate:1000:30000", "0.r_min_mm:0.2:4", "0.r_max_mm:1:20", "0.size_exponent:1:4",
                "0.amp_exponent:0:1.5", "0.damping:0.5:3", "0.xi:0:3", "1.gain_db:-40:-12"],
    "voice": ["0.tilt_db:-6:6", "0.breath:0:0.6", "0.pulse_noise:0:1", "0.formant_shift:0.9:1.1",
              "0.vibrato_cents:0:40"],
    "choir": ["0.0.tilt_db:-6:6", "0.0.breath:0:0.6", "0.0.detune_cents:0:40", "0.1.size:0.5:0.99",
              "0.1.mix:0.1:0.9", "0.1.damp:0:0.9", "1.gain_db:-30:0"],
}



# ----------------------------------------------------------------------------- fit and check
def render(rec, sr=F.FIT_SR, seed=None, duration=None):
    from .recipe import render as _render
    r = copy.deepcopy(rec)
    if duration is not None:
        r["duration"] = duration
    y, _ = _render(r, sr, seed)
    return y


def pitch_error_cents(ref, sr_ref, y, sr_y, method="yin"):
    """Median |cents| between the f0 tracks of ref and y over frames voiced in both (None if unpitched).
    method 'hsum' compares harmonic-sum tracks (dense or reverberant voices) around the reference's pitch."""
    if method == "hsum":
        _, fw, sw = harmonic_sum_f0(ref, sr_ref)
        med = float(np.median(fw[sw >= np.median(sw)]))
        tr, fr, _ = harmonic_sum_f0(ref, sr_ref, fmin=med / 1.5, fmax=med * 1.5)
        ty, fy, _ = harmonic_sum_f0(y, sr_y, fmin=med / 1.5, fmax=med * 1.5)
        m = min(len(fr), len(fy))
        return round(float(np.median(np.abs(1200 * np.log2(fy[:m] / fr[:m])))), 1)
    try:
        tr, fr = A.f0_track(ref, sr_ref)[:2]
        ty, fy = A.f0_track(y, sr_y)[:2]
    except Exception:
        return None
    fr, fy = np.asarray(fr, float), np.asarray(fy, float)
    fy_on = np.interp(np.asarray(tr, float), np.asarray(ty, float), np.where(np.isfinite(fy), fy, 0))
    ok = np.isfinite(fr) & (fr > 0) & (fy_on > 0)
    if ok.sum() < 3:
        return None
    return round(float(np.median(np.abs(1200 * np.log2(fy_on[ok] / fr[ok])))), 1)


def eq_correct(rec, x, sr, fit_section=None, passes=2, max_db=18.0):
    """Spectral fine-tune: put the remaining long-term 1/3-octave difference between the reference (fit section)
    and the render, smoothed over neighbouring bands, into the recipe's post `eq` block. Kept only if the
    distance on the fit section improves."""
    s0, s1 = fit_section or (0.0, len(x) / sr)
    seg = x[int(s0 * sr):int(s1 * sr)]
    _, Gr = band_grid(seg, sr, hop_s=0.05)
    lr = 10 * np.log10(np.mean(10 ** (Gr / 10), 0))
    best = copy.deepcopy(rec)
    eqb = next((b for b in best.get("post", []) if b["model"] == "eq"), None)
    if eqb is None:
        return best
    yb = render(best, duration=round(s1 - s0, 3))
    dbest = F.compare_signals(x, sr, yb, F.FIT_SR, s0 if s0 > 0 else None, s1)["total"]
    cur = copy.deepcopy(rec)
    for _ in range(passes):
        y = render(cur, duration=round(s1 - s0, 3))
        _, Gy = band_grid(y, F.FIT_SR, hop_s=0.05)
        ly = 10 * np.log10(np.mean(10 ** (Gy / 10), 0))
        diff = ndimage.uniform_filter1d((lr - lr.max()) - (ly - ly.max()), 3, mode="nearest")
        p = next(b for b in cur["post"] if b["model"] == "eq").setdefault("params", {})
        old = dict((int(a), b) for a, b in p.get("curve", []))
        p["curve"] = [[b, round(float(np.clip(old.get(b, 0.0) + 0.8 * d, -max_db, max_db)), 1)]
                      for b, d in zip(THIRD_OCTAVES, diff)]
        d = F.compare_signals(x, sr, render(cur, duration=round(s1 - s0, 3)), F.FIT_SR, s0 if s0 > 0 else None,
                              s1)["total"]
        if d < dbest:
            best, dbest = copy.deepcopy(cur), d
    return best


# v0.19: the sparsity part (bubbles vs noise) matters most for liquid textures; pitched kinds leave it out
# (on voices it traded timbre for grain statistics).
FIT_WEIGHTS = {"texture": {**F.DEFAULT_WEIGHTS},
               "impact": {**F.DEFAULT_WEIGHTS},
               "voice": {**F.DEFAULT_WEIGHTS, "sparsity": 0.0},
               "choir": {**F.DEFAULT_WEIGHTS, "sparsity": 0.0}}


def fit_and_check(rec, x, sr, kind, fit_section=None, holdout=None, budget=80, log=print):
    """Fit the free controls of a built recipe and check it.

    fit_section: (start, end) seconds of x to fit on (default whole clip). holdout: list of (name, audio, sr,
    start, end, mode) where mode 'timeline' crops the full render at the same times (one-shots, fixed
    timelines) and 'fresh' renders the recipe from t=0 for the section length (stationary textures, or another
    recording of the same sound). Returns dict with the fitted recipe and every distance."""
    dur = len(x) / sr
    s0, s1 = fit_section or (0.0, dur)
    work = copy.deepcopy(rec)
    work["duration"] = round(s1 - s0, 3)
    params = [p for p in FIT_PARAMS[kind] if int(p.split(".")[0]) < len(work["layers"])]
    res = F.fit(work, x, sr, params=params, ref_start=s0 if s0 > 0 else None, ref_end=s1, budget=budget,
                weights=FIT_WEIGHTS.get(kind), log=lambda *a, **k: None)
    fitted = copy.deepcopy(res["recipe"])
    fitted["duration"] = rec["duration"]
    fitted = eq_correct(fitted, x, sr, fit_section=(s0, s1))
    out = {"recipe": fitted, "fit_before": res["before"], "fit_after": res["after"],
           "fit_section": [round(s0, 3), round(s1, 3)], "params": res["params"], "evaluations": res["evaluations"]}
    y = render(fitted)
    out["full"] = F.compare_signals(x, sr, y, F.FIT_SR)
    out["full_other_seed"] = F.compare_signals(x, sr, render(fitted, seed=1234), F.FIT_SR)
    out["holdout"] = []
    for name, hx, hsr, a, b, mode in holdout or []:
        if mode == "timeline":
            seg = y[int(a * F.FIT_SR):int(b * F.FIT_SR)]
            d = F.compare_signals(hx, hsr, seg, F.FIT_SR, a, b)
        else:
            d = F.compare_signals(hx, hsr, render(fitted, duration=round(b - a, 3)), F.FIT_SR, a, b)
        out["holdout"].append({"name": name, "section": [a, b], "mode": mode, **d})
    out["pitch_error_cents"] = (pitch_error_cents(x, sr, y, F.FIT_SR, "yin" if kind == "voice" else "hsum")
                                if kind in ("voice", "choir") else None)
    return out



def auto_kind(feats):
    """Guess the recipe kind from the analysis: pitched and voiced -> voice; tonal but no single clear pitch
    (several voices, heavy reverb) -> choir; long, steady and onset-free -> texture; otherwise impact."""
    f0 = feats["f0"]
    if f0.get("f0_hz") and f0.get("voiced_fraction", 0) > 0.4:
        return "voice"
    if feats["tonal_share"] > 0.5 and feats["duration_s"] > 3 and not feats["onsets_s"]:
        return "choir"
    if feats["duration_s"] > 4 and len(feats["onsets_s"]) <= 1:
        return "texture"
    return "impact"


def run(x, sr, kind="auto", name="resynth", fit_section=None, holdout=None, holdout_section=None,
        clean_tonal=False, budget=80, log=print):
    """Whole pipeline for the CLI: optional tonal clean-up, analysis, recipe, fit, checks."""
    x = np.asarray(x, float)
    removed = None
    if clean_tonal:
        x, removed = suppress_tonal(x, sr)
    feats = analyse_clip(x, sr)
    if kind == "auto":
        kind = auto_kind(feats)
    log(f"kind: {kind}")
    rec = build_recipe(x, sr, kind, name, feats=feats)
    hold = list(holdout or [])
    if holdout_section:
        a, b = holdout_section
        hold.append(("held-out section", x, sr, a, b, "fresh" if kind == "texture" else "timeline"))
    res = fit_and_check(rec, x, sr, kind, fit_section=fit_section, holdout=hold, budget=budget, log=log)
    res["kind"] = kind
    res["tonal_removed_fraction"] = removed
    res["features"] = {k: v for k, v in feats.items() if k != "stems"}
    return res
