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
    if kind == "growl":
        m["size"] = {"default": 0.5, "doc": "smaller .. bigger creature",
                     "map": [{"to": "0.tract_cm", "op": "mul", "points": [[0, 0.5], [0.5, 1], [1, 2]]},
                             {"to": "0.f0", "op": "mul", "points": [[0, 1.6], [0.5, 1], [1, 0.65]]}]}
        m["rough"] = {"default": 0.5, "doc": "smoother .. more chaotic",
                      "map": [{"to": "0.chaos", "op": "add", "points": [[0, -0.3], [0.5, 0], [1, 0.3]],
                               "clamp": [0, 1]}]}
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


def drone_pitch(x, sr, fref, rel_db=-12.0):
    """v0.20: a sustained lower pitch under a sung line (a drone). Looks for the strongest long-term spectral peak
    between 50 Hz and 0.85 x the melody's median f0 that is present in most 1 s frames. Returns (Hz, level curve)
    or None."""
    f, P = signal.welch(x, sr, nperseg=16384)
    m = (f >= 50) & (f <= 0.85 * fref)
    if m.sum() < 5:
        return None
    L = 10 * np.log10(P[m] + 1e-20)
    pk, _ = signal.find_peaks(L, prominence=6)
    if not len(pk):
        return None
    i = pk[np.argmax(L[pk])]
    if L[i] < 10 * np.log10(P[(f >= 50) & (f <= 2000)].max()) + rel_db:
        return None
    fd = float(f[m][i])
    # present in most frames? band energy around fd vs the frame's strongest peak
    h = sr
    sos = signal.butter(2, [fd * 0.94, fd * 1.06], "bandpass", fs=sr, output="sos")
    xb = signal.sosfiltfilt(sos, x)
    n = len(x) // h
    if n < 2:
        return None
    eb = np.sqrt(np.mean(xb[:n * h].reshape(n, h) ** 2, 1))
    ea = np.sqrt(np.mean(x[:n * h].reshape(n, h) ** 2, 1)) + 1e-12
    if np.mean(20 * np.log10(eb / ea + 1e-12) > -20) < 0.6:
        return None
    lv = _level_curve(xb, sr, hop_s=0.1, floor_db=-60.0, smooth_s=0.4)
    return fd, lv


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
        rec["strong_onsets_s"] = [round(float(o), 3) for o in strong]
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
        if kind == "choir":  # v0.20: moving vowels again, smoothed over ~0.5 s so beating does not become flutter
            M = np.array([fr[1] for fr in env], float)
            M = ndimage.uniform_filter1d(M, 5, axis=0, mode="nearest")
            M = M - 10 * np.log10(np.sum(10 ** (M / 10), 1, keepdims=True))  # colour only; level carries loudness
            env = simplify_frames(np.array([fr[0] for fr in env], float), M, tol_db=2.0, max_frames=30,
                                  floor_db=M.max() - 80)
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
            # v0.20: several human singers rather than one buzzy line plus hiss. Each voice gets its own pitch
            # drift, jitter and shimmer, vocal tract size and vibrato rate; a Rosenberg glottal pulse; vibrato that
            # grows in after each note; breath tied to the glottal cycle and kept low; a convolution hall instead
            # of the old noise layer (which was the hiss).
            hb["params"].update({"glottal": "rosenberg", "open_quotient": 0.6, "jitter_cents": 6.0,
                                 "shimmer_db": 0.6, "vibrato_cents": 25.0, "vibrato_onset": 0.5,
                                 "voice_spread": 0.05, "wander_cents": 12.0, "breath": 0.03, "pulse_noise": 0.95,
                                 "tilt_db": 0.0})
            hall = {"model": "conv_reverb", "params": {"rt60": 2.5, "hf_ratio": 0.5, "lf_ratio": 1.2,
                                                       "predelay_ms": 25.0, "early_ms": 80.0, "mix": 0.45}}
            rec["layers"].append({"name": "choir", "chain": [hb, hall]})
            # a held drone under the melody (common in chant and organum): its own singers, same hall
            dr = drone_pitch(x, sr, fref)
            if dr is not None:
                fd, dlev = dr
                hd = copy.deepcopy(hb)
                hd["params"].update({"f0": round(fd, 2), "level": dlev, "vibrato_cents": 12.0, "voices": 3})
                rec["layers"].append({"name": "drone", "gain_db": -6.0, "chain": [hd, copy.deepcopy(hall)]})
        rec["f0_reference_hz"] = round(fref, 2)
    elif kind == "growl":
        # v0.21: rough creature voice (creature_voice: CHIMERA glottis port + f0/2 ventricular layer + long tract)
        # the fold rate, not the f0/2 subharmonic (that is the ventricular layer's job): search from 60 Hz,
        # then pull frames more than ~0.6 octave off the median back to it
        ft, f0, _ = harmonic_sum_f0(x, sr, fmin=60.0, fmax=400.0)
        med = float(np.median(f0))
        f0 = np.where(np.abs(np.log2(f0 / med)) > 0.6, med, f0)
        f0 = ndimage.median_filter(f0, 3, mode="nearest")
        f0_pts = [[round(float(a), 3), round(float(b), 2)] for a, b in zip(ft, f0)]
        lv = _level_curve(x, sr, hop_s=0.02, floor_db=-60.0, smooth_s=0.05)
        rec["layers"].append({"name": "voice", "model": "creature_voice", "params": {
            "f0": f0_pts, "chaos": 0.3, "ventricular": 0.5, "rasp": 0.3, "breath": 0.1, "jitter": 0.02,
            "shimmer": 0.15, "open_quotient": 0.5, "sharpness": 0.8, "tract_cm": 30.0, "mouth": 0.5,
            "bandwidth": 1.5, "level": lv}})
        rec["f0_reference_hz"] = round(float(np.median(f0)), 2)
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
    "growl": ["0.chaos:0:1", "0.ventricular:0:2", "0.rasp:0:1.5", "0.breath:0:1", "0.tract_cm:10:120",
              "0.mouth:0:1", "0.open_quotient:0.3:0.9", "0.sharpness:0:1", "0.shimmer:0:0.5", "0.bandwidth:0.5:4"],
    "choir": ["0.0.tilt_db:-6:6", "0.0.breath:0:0.08", "0.0.detune_cents:0:30", "0.1.rt60:1:6",
              "0.1.mix:0.1:0.8", "0.1.hf_ratio:0.2:1", "1.gain_db:-20:6"],
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


def eq_correct(rec, x, sr, fit_section=None, passes=2, max_db=18.0, align="max", lo_hz=0.0):
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
        if align == "median":  # v0.20 (choir): align on the median band difference above lo_hz, so a few
            # spiky low bands (a low voice's sparse harmonics) do not tilt the whole correction
            raw = lr - ly
            use = np.array(THIRD_OCTAVES) >= lo_hz
            raw = np.where(use, raw - np.median(raw[use]), 0.0)
            diff = ndimage.uniform_filter1d(raw, 5, mode="nearest")
        else:
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


# ----------------------------------------------------------------------------- v0.20 splash tail
TAIL_PARAMS = ["0.rate:200:30000", "0.r_min_mm:0.2:4", "0.r_max_mm:1:20", "0.size_exponent:1:4",
               "0.damping:0.5:3", "0.xi:0:3"]


def tail_start(x, sr, onsets, min_after=0.3, drop_db=15.0, xfade=0.2):
    """Where the splash phase ends: after the last strong onset, once the level has fallen drop_db below the
    peak that followed it (at least min_after s later). None if too little tail is left."""
    dur = len(x) / sr
    last = max(onsets) if onsets else 0.0
    t, env = _env_db(x, sr)
    after = t >= last
    if not after.any():
        return None
    peak = env[after].max()
    cand = np.nonzero(after & (t >= last + min_after) & (env < peak - drop_db))[0]
    ts = float(t[cand[0]]) if cand.size else last + min_after
    return round(ts, 3) if ts < dur - 0.3 - xfade else None


def _env_db(x, sr, hop_s=0.01):
    h = int(hop_s * sr)
    m = len(x) // h
    e = np.sqrt(np.mean(np.asarray(x[:m * h], float).reshape(m, h) ** 2, 1)) + 1e-9
    e = ndimage.uniform_filter1d(e, 5, mode="nearest")
    return (np.arange(m) + 0.5) * hop_s, 20 * np.log10(e)


def _curve_at(pts, t):
    if not isinstance(pts, list):
        return np.full(len(t), float(pts))
    a = np.array(pts, float)
    return np.interp(t, a[:, 0], a[:, 1])


def _fade_out_points(pts, ts, xf):
    """Level automation `pts` faded to 0 (equal-power) between ts and ts + xf."""
    tk = sorted(set([p[0] for p in pts] if isinstance(pts, list) else [0.0]) | {ts, ts + xf / 2, ts + xf})
    tk = np.array(tk)
    g = np.cos(0.5 * np.pi * np.clip((tk - ts) / xf, 0, 1))
    v = _curve_at(pts, tk) * g
    return [[round(float(a), 4), round(float(b), 5)] for a, b in zip(tk, v) if a <= ts + xf] + [[round(ts + xf + 0.01, 4), 0.0]]


def _fade_out_frames(frames, ts, xf, floor=-60.0):
    """bandnoise frames ramped down by -floor dB between ts and ts + xf (and held down after)."""
    t = np.array([f[0] for f in frames], float)
    G = np.array([f[1] for f in frames], float)
    at = lambda tt: [round(float(np.interp(tt, t, G[:, b])), 1) for b in range(G.shape[1])] if len(t) > 1 else list(G[0])
    out = [f for f in frames if f[0] < ts]
    out.append([round(ts, 3), at(ts)])
    out.append([round(ts + xf, 3), [round(v + floor, 1) for v in at(ts + xf)]])
    return out


def add_bubble_tail(rec, x, sr, ts, budget=60, xfade=0.2, log=print):
    """v0.20 hybrid splash: the fitted splash recipe keeps the impact and splash phase; from `ts` its bubbles,
    body and slap layers cross-fade (over `xfade` s) into a separate `tail_bubbles` bubble cloud fitted on the
    tail only, with the sparsity part of the loss (so the onset is never pulled towards a gentle bubble)."""
    dur = len(x) / sr
    out = copy.deepcopy(rec)
    L = out["layers"]
    L[0]["params"]["level"] = _fade_out_points(L[0]["params"].get("level", 1.0), ts, xfade)
    for lay in L[1:3]:
        lay["params"]["frames"] = _fade_out_frames(lay["params"]["frames"], ts, xfade)
    seg = x[int(ts * sr):]
    lvl = _level_curve(seg, sr, hop_s=0.01, tol_db=2.0, floor_db=-60.0, max_points=40)
    tk = sorted({p[0] for p in lvl} | {0.0, xfade / 2, xfade})
    gin = np.sin(0.5 * np.pi * np.clip(np.array(tk) / xfade, 0, 1))
    lvl = [[round(float(a), 4), round(float(b), 5)] for a, b in zip(tk, _curve_at(lvl, np.array(tk)) * gin)]
    p0 = L[0]["params"]
    tail = {"name": "tail_bubbles", "model": "bubble_cloud", "start": round(ts, 3), "gain_db": -6.0,
            "params": {"rate": 3000.0, "r_min_mm": p0.get("r_min_mm", 1.0), "r_max_mm": max(p0.get("r_max_mm", 8.0), 2.0),
                       "size_exponent": 2.5, "amp_exponent": 0.5, "xi": 0.5, "damping": 1.0, "level": lvl}}
    solo = {k: copy.deepcopy(v) for k, v in out.items() if k not in ("layers", "macros")}
    solo["duration"] = round(dur - ts, 3)
    solo["layers"] = [dict(tail, start=0.0)]
    res = F.fit(solo, x, sr, params=TAIL_PARAMS, ref_start=ts, ref_end=dur, budget=budget,
                weights=FIT_WEIGHTS["texture"], log=lambda *a, **k: None)
    tail["params"].update({k: v for k, v in res["recipe"]["layers"][0]["params"].items() if k != "level"})
    L.append(tail)
    # level: match the tail's RMS (relative to the whole clip's peak) to the recording
    i0 = int(ts * sr)
    want = np.sqrt(np.mean(x[i0:] ** 2)) / (np.max(np.abs(x)) + 1e-12)
    for _ in range(3):
        y = render(out)
        j0 = int(ts * F.FIT_SR)
        got = np.sqrt(np.mean(y[j0:] ** 2)) / (np.max(np.abs(y)) + 1e-12)
        tail["gain_db"] = round(float(np.clip(tail["gain_db"] + 20 * np.log10(want / (got + 1e-12)), -40, 6)), 2)
    out["tail_start_s"] = round(ts, 3)
    return out


# v0.19: the sparsity part (bubbles vs noise) matters most for liquid textures; pitched kinds leave it out
# (on voices it traded timbre for grain statistics). v0.20: impacts leave it out too (it turned splashes into a
# wood knock and a blow of air); add_bubble_tail uses it on the tail window only.
FIT_WEIGHTS = {"texture": {**F.DEFAULT_WEIGHTS},
               "impact": {**F.DEFAULT_WEIGHTS, "sparsity": 0.0},  # v0.20: sparsity only on the splash tail
               "voice": {**F.DEFAULT_WEIGHTS, "sparsity": 0.0},
               "choir": {**F.DEFAULT_WEIGHTS, "sparsity": 0.0},
               "growl": {**F.DEFAULT_WEIGHTS, "sparsity": 0.25}}


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
    if kind == "choir":
        fitted = eq_correct(fitted, x, sr, fit_section=(s0, s1), max_db=12.0, align="median",
                            lo_hz=1.5 * float(rec.get("f0_reference_hz", 100.0)))
    else:
        fitted = eq_correct(fitted, x, sr, fit_section=(s0, s1))
    if kind == "impact" and s0 == 0:
        ts = tail_start(x, sr, rec.get("strong_onsets_s", []))
        if ts is not None:
            fitted = add_bubble_tail(fitted, x, sr, ts, budget=max(20, budget // 2), log=log)
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
                                if kind in ("voice", "choir", "growl") else None)
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
