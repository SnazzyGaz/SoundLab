"""Export finishing: loudness (ITU-R BS.1770-4), true peak, tail trim and seamless loops.

`soundlab.render` returns the raw mix (peak-normalised). Files meant for a game go through `finish()`:

1. **Trim** the near-silent tail: the end is the last moment the 10 ms RMS envelope is within `trim_db`
   (default 60 dB) of its maximum, plus a short pad and a cosine fade. At -60 dB below the loudest moment a
   reverb tail is already inaudible in a game mix, so nothing audible is cut. A minimum length is kept.
2. **Loudness**: K-weighted loudness per BS.1770-4. One-shots are measured on their maximum momentary
   loudness (400 ms window): integrated loudness of a 0.3 s hit with a long quiet tail depends mostly on the
   tail and the gate, so it is unreliable for short sounds. Loops and the `ambience` category use gated
   integrated loudness. Sounds shorter than the 400 ms window are measured inside one window (a 0.1 s
   click counts as quieter than a 0.4 s burst at the same level, which matches how short sounds are heard).
   If reaching the target would push the true peak over the ceiling, up to 6 dB of look-ahead peak
   limiting is used on the transients; beyond that the sound stays under target rather than being squashed.
   The true peak (4x oversampled) always stays under the ceiling (default -1 dBTP).
3. Targets per category (`TARGETS`), overridable per recipe with `"loudness_lufs"` or per call.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter, resample_poly

# Default targets (LUFS). One-shots: maximum momentary loudness; loops/ambiences: integrated loudness.
# Rough game-audio practice: UI and pickups sit under the action, weapons and explosions on top,
# ambience beds well below everything.
TARGETS = {
    "ui": -22.0, "pickup": -20.0, "powerup": -19.0, "coin": -20.0, "magic": -19.0, "scifi": -17.0,
    "whoosh": -19.0, "foley": -20.0, "footstep": -19.0, "door": -17.0, "water": -19.0, "glass": -16.0,
    "impact": -16.0, "combat": -15.0, "weapon": -12.0, "explosion": -11.0, "creature": -16.0,
    "animal": -18.0, "vehicle": -17.0, "ambience": -24.0,
}
DEFAULT_TARGET = -18.0
LOOP_TARGET = -24.0           # integrated loudness for loops of categories not listed above
CEILING_DBTP = -1.0


# ------------------------------------------------------------------------------------------- loudness
def _k_filter(sr: int):
    """BS.1770 K-weighting as two biquads for any sample rate (the libebur128 design; at 48 kHz it gives the
    coefficients printed in the standard)."""
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196      # stage 1: high shelf
    K = math.tan(math.pi * f0 / sr)
    Vh = 10 ** (G / 20)
    Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    b1 = np.array([Vh + Vb * K / Q + K * K, 2 * (K * K - Vh), Vh - Vb * K / Q + K * K]) / a0
    a1 = np.array([a0, 2 * (K * K - 1), 1 - K / Q + K * K]) / a0
    f0, Q = 38.13547087602444, 0.5003270373238773                            # stage 2: RLB high pass
    K = math.tan(math.pi * f0 / sr)
    a0 = 1 + K / Q + K * K
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([a0, 2 * (K * K - 1), 1 - K / Q + K * K]) / a0
    return (b1, a1), (b2, a2)


def _block_power(x: np.ndarray, sr: int, win_s: float, hop_s: float) -> np.ndarray:
    """Mean square of the K-weighted signal per block; stereo (n, 2): the sum over channels (BS.1770, L/R
    weights 1.0)."""
    (b1, a1), (b2, a2) = _k_filter(sr)
    y = lfilter(b2, a2, lfilter(b1, a1, np.asarray(x, float), axis=0), axis=0)
    p = y * y if y.ndim == 1 else np.sum(y * y, axis=1)
    w, h = int(round(win_s * sr)), max(1, int(round(hop_s * sr)))
    if len(p) < w:                       # shorter than one window: measure it inside one window
        p = np.concatenate([p, np.zeros(w - len(p))])
    c = np.concatenate([[0.0], np.cumsum(p)])
    starts = np.arange(0, len(p) - w + 1, h)
    return (c[starts + w] - c[starts]) / w


def _lk(z):
    return -0.691 + 10 * np.log10(np.maximum(z, 1e-20))


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """Gated integrated loudness (BS.1770-4: 400 ms blocks, 75 % overlap, -70 LUFS and -10 LU gates)."""
    z = _block_power(x, sr, 0.4, 0.1)
    z = z[_lk(z) > -70]
    if not len(z):
        return -float("inf")
    z = z[_lk(z) > _lk(z.mean()) - 10]
    return float(_lk(z.mean()))


def momentary_max_lufs(x: np.ndarray, sr: int) -> float:
    return float(_lk(_block_power(x, sr, 0.4, 0.01).max()))


def short_term_max_lufs(x: np.ndarray, sr: int) -> float:
    return float(_lk(_block_power(x, sr, 3.0, 0.1).max()))


def true_peak_db(x: np.ndarray) -> float:
    pk = np.max(np.abs(resample_poly(np.asarray(x, float), 4, 1))) if len(x) else 0.0
    return float(20 * np.log10(max(pk, 1e-12)))


def measure(x: np.ndarray, sr: int) -> dict:
    return {"integrated": integrated_lufs(x, sr), "momentary_max": momentary_max_lufs(x, sr),
            "short_term_max": short_term_max_lufs(x, sr), "true_peak": true_peak_db(x),
            "peak": float(20 * np.log10(np.max(np.abs(x)) + 1e-12)), "seconds": len(x) / sr}


def target_for(recipe: dict | None, loop: bool = False) -> float:
    r = recipe or {}
    if r.get("loudness_lufs") is not None:
        return float(r["loudness_lufs"])
    cat = str(r.get("category", "sfx"))
    if loop:
        return TARGETS.get(cat, LOOP_TARGET) if cat == "ambience" else min(TARGETS.get(cat, LOOP_TARGET), LOOP_TARGET + 4)
    return TARGETS.get(cat, DEFAULT_TARGET)


def limit(x: np.ndarray, sr: int, ceiling_db: float, lookahead_s: float = 0.0015, release_s: float = 0.06) -> np.ndarray:
    """Look-ahead peak limiter: the gain is already down `lookahead_s` before each peak (a smooth ramp, no
    clipped edges) and recovers over `release_s`."""
    from scipy.ndimage import maximum_filter1d
    c = 10 ** (ceiling_db / 20)
    amp = np.abs(x) if x.ndim == 1 else np.max(np.abs(x), axis=1)  # stereo: one linked gain for both sides
    red = 1.0 - np.minimum(1.0, c / (amp + 1e-12))                  # gain reduction needed per sample
    la = max(1, int(lookahead_s * sr))
    red = maximum_filter1d(red, size=2 * la + 1, mode="nearest")    # hold it over the look-ahead window
    a = math.exp(-1.0 / (release_s * sr))
    nz = np.nonzero(red > 0)[0]
    if len(nz):                                                     # release: decay only after the last peak
        i0, i1 = nz[0], min(len(red), nz[-1] + int(10 * release_s * sr))
        seg = red[i0:i1].copy()
        for i in range(1, len(seg)):
            v = seg[i - 1] * a
            if v > seg[i]:
                seg[i] = v
        red[i0:i1] = seg
    k = np.hanning(2 * la + 1)
    red = np.convolve(red, k / k.sum(), mode="same")                # round the attack corner
    g = 1.0 - np.clip(red, 0, 1)
    return x * (g if x.ndim == 1 else g[:, None])


def normalize_loudness(x: np.ndarray, sr: int, target: float, measure_by: str = "momentary",
                       ceiling_dbtp: float = CEILING_DBTP, max_limit_db: float = 6.0) -> tuple[np.ndarray, dict]:
    """Gain to `target` LUFS (measured by 'momentary' max or 'integrated'), with the true peak kept under the
    ceiling. If the gain would push the peaks over, up to `max_limit_db` of transient peak limiting is used;
    beyond that the gain is reduced (so very peaky sounds end up under target rather than squashed)."""
    meas = integrated_lufs if measure_by == "integrated" else momentary_max_lufs
    now = meas(x, sr)
    if not np.isfinite(now):
        return x, {"gain_db": 0.0, "measured_before": now, "limited_by_peak": False, "limiter_db": 0.0}
    gain = target - now
    tp = true_peak_db(x)
    over = tp + gain - ceiling_dbtp
    lim_db = 0.0
    if over > 0:
        lim_db = min(over, max_limit_db)
        gain -= over - lim_db                       # what the limiter may not take comes off the gain
    y = x * 10 ** (gain / 20)
    if lim_db > 0:
        y = limit(y, sr, ceiling_dbtp - 0.3)
        tp2 = true_peak_db(y)
        if tp2 > ceiling_dbtp:                     # inter-sample overs left: trim the gain a little
            y = y * 10 ** ((ceiling_dbtp - tp2) / 20)
    res = meas(y, sr)
    return y, {"gain_db": round(gain, 2), "measured_before": round(now, 2), "target": target,
               "result": round(res, 2), "limited_by_peak": bool(over > 0), "limiter_db": round(lim_db, 2),
               "measured_by": measure_by, "true_peak": round(true_peak_db(y), 2)}


# --------------------------------------------------------------------------------------------- trim
def trim_tail(x: np.ndarray, sr: int, trim_db: float = 60.0, min_s: float = 0.1, pad_s: float = 0.05,
              fade_s: float = 0.03) -> np.ndarray:
    """Cut the tail after the last moment the 10 ms RMS envelope is within `trim_db` of its maximum."""
    n = len(x)
    if n == 0:
        return x
    w = max(1, int(0.01 * sr))
    xx = np.asarray(x, float)
    c = np.concatenate([[0.0], np.cumsum(xx ** 2 if xx.ndim == 1 else np.mean(xx ** 2, axis=1))])
    env = np.sqrt(np.maximum((c[w:] - c[:-w]) / w, 0))          # env[i] = RMS of x[i:i+w]
    if env.max() <= 0:
        return x[: max(1, int(min_s * sr))]
    above = np.nonzero(env > env.max() * 10 ** (-trim_db / 20))[0]
    end = min(n, int(above[-1]) + w + int(pad_s * sr))
    end = max(end, min(n, int(min_s * sr)))
    if end >= n:
        return x
    y = np.array(x[:end], float)
    nf = min(int(fade_s * sr), end // 2)
    if nf > 0:
        f = 0.5 * (1 + np.cos(np.linspace(0, math.pi, nf)))
        y[-nf:] *= f if y.ndim == 1 else f[:, None]
    return y


# --------------------------------------------------------------------------------------------- loops
def fold_loop(y: np.ndarray, sr: int, seconds: float, xfade_s: float = 0.5) -> np.ndarray:
    """Make a seamless loop of `seconds` from a render at least seconds + xfade_s long: the part after the
    loop end is crossfaded (equal power) into the start, so the last sample flows into the first."""
    n, x = int(round(seconds * sr)), int(round(xfade_s * sr))
    if len(y) < n + x:
        raise ValueError("render too short for the loop")
    out = np.array(y[:n], float)
    t = np.linspace(0, math.pi / 2, x, endpoint=False)
    if out.ndim == 2:
        t = t[:, None]
    out[:x] = y[:x] * np.sin(t) + y[n:n + x] * np.cos(t)
    return out


def loop_seam_jump(y: np.ndarray) -> float:
    """Size of the jump at the loop point relative to the typical sample-to-sample step (about 1 = seamless)."""
    d = np.abs(np.diff(y, axis=0))
    return float(np.max(np.abs(y[0] - y[-1]) / (np.median(d, axis=0) + 1e-12)))


# --------------------------------------------------------------------------------------------- finish
def finish(x: np.ndarray, sr: int, recipe: dict | None = None, level: str = "lufs", trim: bool = True,
           loop: bool = False, target: float | None = None, ceiling_dbtp: float = CEILING_DBTP,
           trim_db: float = 60.0) -> tuple[np.ndarray, dict]:
    """Prepare a render for export. level: 'lufs' (default) or 'peak' (keep the render's peak level).
    Loops are never trimmed (their length is the loop)."""
    info = {"level": level, "seconds_in": round(len(x) / sr, 3)}
    if trim and not loop:
        x = trim_tail(x, sr, trim_db)
    info["seconds"] = round(len(x) / sr, 3)
    if level == "lufs":
        t = target if target is not None else target_for(recipe, loop)
        by = "integrated" if loop or (recipe or {}).get("category") == "ambience" else "momentary"
        x, li = normalize_loudness(x, sr, t, by, ceiling_dbtp)
        info.update(li)
    return x, info
