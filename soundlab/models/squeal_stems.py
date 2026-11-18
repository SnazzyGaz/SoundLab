"""squeal_stems (v0.29): squeals, squeaks and creaks rebuilt from stems of CC0 recordings.

Every source burst was analysed (outside the repo) into numbers, never audio: sinusoidal partial tracks (frequency
and level per 2.9 ms frame) and a residual noise envelope (96 log-spaced bands per frame). From those the model
derives five stems per burst, and recipes recombine real stems instead of averaging them:

1. tonal  - each partial's pitch and level contour
2. micro  - the real jitter (cents) and flutter (dB) of the strongest partial, kept as its own track and read at its
            own rate, so a borrowed wobble keeps its real speed
3. noise  - the residual time-frequency field with transients bridged out, plus its median spectral shape (a noise
            colour borrowed from another burst is applied as an EQ difference, so the base keeps its own movement)
4. trans  - the onset (first ~30 ms) and clicks (frames > 6 dB over a 30 ms median)
5. body   - the burst's 1/3-octave long-term spectrum minus a straight tilt (a colour swap is an EQ difference)

A recipe is a gesture timeline: a list of motions (open, close, step...), each with its own speed, pressure and
duration, playing one burst from the pool; pauses between motions are the motion stopping (only the ambient bed
carries on), not a cut. Speed shortens the burst (and raises the squeal slightly), pressure makes it louder and a
little brighter.

Lessons that shaped it (docs/friction.md):
- never average: an averaged contour or spectrum loses the jitter, flutter and mode hops that make a squeal real
- prefer each burst's own wobble; borrow one only when it measures like the base's own (a stronger borrowed
  flutter reads as a trill or a purr)
- per-frame noise gains are themselves random; replayed at 2.9 ms they add envelope flutter (bubbling), so the
  optional anti-bubble pass power-smooths the noise and folds tracker "birdies" back into it
- stray sustained partials (a hum or drone in the recording) are muted by rule, keeping random draws and gains
"""
from __future__ import annotations

import copy
import functools
import json
import lzma
import math
import struct
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d

from ..core import Ctx, Param, RecipeError, model

SR = 44100
H = 128
FR = SR / H
BODY_GRID = np.geomspace(100, 16000, 23)            # 1/3 octave
DATA = Path(__file__).with_name("squeal_stems.bin")
RAW_RENDER = dict(skip_end_clicks=False, end_fade_s=0.006)     # render settings of the plain (non-smoothed) path


def lp_noise(n, fc, rng, sr=SR):
    b, a = signal.butter(2, fc / (sr / 2))
    z = signal.lfilter(b, a, rng.standard_normal(n))
    return z / (np.std(z) + 1e-12)


def smooth(z, n):
    n = max(1, int(n)) | 1
    return uniform_filter1d(z, n, mode='nearest')

def decompose(b, edges):
    nfr = b['nfr']
    # --- tonal + micro
    trs = b['tracks']
    dom = max(trs, key=lambda t: float(np.sum(t[2] ** 2))) if trs else None
    tonal = []
    for s0, f, A in trs:
        lc = 1200 * np.log2(np.maximum(f, 1.0)); ld = 20 * np.log10(np.maximum(A, 1e-9))
        tonal.append((s0, smooth(lc, 0.06 * FR), smooth(ld, 0.04 * FR)))
    if dom is not None:
        s0, f, A = dom
        lc = 1200 * np.log2(np.maximum(f, 1.0)); ld = 20 * np.log10(np.maximum(A, 1e-9))
        micro_c = lc - smooth(lc, 0.06 * FR); micro_d = ld - smooth(ld, 0.04 * FR)
        micro_c = np.clip(micro_c, -150, 150); micro_d = np.clip(micro_d, -12, 12)
    else:
        micro_c = micro_d = np.zeros(8)
    # --- noise / transients
    E = b['E'].astype(np.float64) + 1e-9
    Ld = 20 * np.log10(E)
    tot = 10 * np.log10(np.sum(E ** 2, axis=0))
    med = median_filter(tot, size=int(0.03 * FR) | 1, mode='nearest')
    click = tot > med + 6
    onset = np.zeros(nfr, bool); onset[:min(nfr, int(0.03 * FR) + 8)] = True
    tmask = click | onset
    Ls = Ld.copy()
    for k in range(Ld.shape[0]):                          # noise with transients bridged by the local median
        Ls[k] = np.where(tmask, median_filter(Ld[k], size=int(0.03 * FR) | 1, mode='nearest'), Ld[k])
    nenv = 10 * np.log10(np.sum(10 ** (Ls / 10), axis=0))
    nshape = np.median(Ls - nenv[None, :], axis=1)
    trans = dict(onset=Ld[:, :int(0.03 * FR) + 8] - 0.0, clicks=[(i, Ld[:, i]) for i in np.where(click & ~onset)[0]])
    # --- body colour (1/3 octave LTAS of tonal+noise in band terms, minus a straight tilt)
    cen = np.sqrt(np.maximum(edges[:-1], 50) * edges[1:])
    P = np.mean(E ** 2, axis=1)
    for s0, lc, ld in tonal:                              # add partial energy into the band picture
        fz = 2 ** (lc / 1200); k = np.clip(np.searchsorted(edges, fz) - 1, 0, len(P) - 1)
        np.add.at(P, k, (10 ** (ld / 20)) ** 2 * 0.25 * len(lc) / nfr)
    Lb = 10 * np.log10(np.interp(np.log(BODY_GRID), np.log(cen), P) + 1e-20)
    tilt = np.polyval(np.polyfit(np.log2(BODY_GRID), Lb, 1), np.log2(BODY_GRID))
    body = Lb - tilt
    tonal_raw = [(s0, 1200 * np.log2(np.maximum(f, 1.0)), 20 * np.log10(np.maximum(A, 1e-9))) for s0, f, A in trs]
    return dict(tonal=tonal, tonal_raw=tonal_raw, Ls=Ls.astype(np.float32), micro_c=micro_c, micro_d=micro_d, nenv=nenv, nshape=nshape, trans=trans, body=body,
                nfr=nfr, src=b['src'], t0=b['t0'], f_dom=b['f_dom'], bands_cen=cen)

def lpz(z, fc):
    """zero-phase 2nd-order low-pass on a frame-rate contour (a box filter leaks 8-30 Hz straight through)."""
    if len(z) < 16 or fc <= 0: return z
    b, a = signal.butter(2, min(0.99, fc / (FR / 2)))
    return signal.filtfilt(b, a, z, padlen=min(len(z) - 1, 3 * max(len(a), len(b))))

def clean(s, edges, min_track_ms=60, rel_db=45, amp_ms=20, freq_ms=12, noise_ms=25, amp_fc=6.0, freq_fc=10.0):
    """Anti-bubble pass on a burst's stems.
    - partial tracks shorter than min_track_ms or > rel_db below the strongest are 'birdies' (tracker picking noise
      peaks); they are dropped and their power is returned to the noise field instead of being played as tones
    - own per-partial micro is low-passed: amplitude over amp_ms, log-frequency over freq_ms (tracker warble)
    - the noise field is power-smoothed over noise_ms: per-frame band gains of noise are themselves random (STFT of
      noise is Rayleigh), so playing them back at 2.9 ms doubles the envelope flutter -> bubbling. Transients are
      separate stems so smoothing does not blunt clicks or onsets."""
    s = copy.copy(s)
    peaks = [float(np.max(rd)) for _, _, rd in s['tonal_raw']] or [0.0]
    top = max(peaks)
    keep, tr, raw = [], [], []
    Lp = 10 ** (s['Ls'].astype(np.float64) / 10)
    cen = s['bands_cen']
    for (s0, lc, ld), (_, rc, rd), pk in zip(s['tonal'], s['tonal_raw'], peaks):
        if len(rc) < min_track_ms / 1000 * FR or pk < top - rel_db:
            k = np.clip(np.searchsorted(cen, 2 ** (rc / 1200)), 0, len(cen) - 1)
            fr = np.arange(s0, s0 + len(rc)); ok = fr < Lp.shape[1]
            np.add.at(Lp, (k[ok], fr[ok]), (10 ** (rd[ok] / 20)) ** 2 * 0.5 * 4)   # amp^2/2 into band power scale
            continue
        rc2 = lpz(rc, freq_fc); rd2 = lpz(rd, amp_fc)
        tr.append((s0, lc, ld)); raw.append((s0, rc2, rd2))
    s['tonal'], s['tonal_raw'] = tr, raw
    if noise_ms > 0:
        Lp = uniform_filter1d(Lp, max(1, int(noise_ms / 1000 * FR)) | 1, axis=1, mode='nearest')
    s['Ls'] = (10 * np.log10(Lp + 1e-30)).astype(np.float32)
    if raw:   # micro of the cleaned dominant partial
        s0, rc, rd = max(raw, key=lambda t: float(np.sum(10 ** (t[2] / 10))))
        s['micro_c'] = np.clip(rc - smooth(rc, 0.06 * FR), -150, 150); s['micro_d'] = np.clip(rd - smooth(rd, 0.04 * FR), -12, 12)
    return s

def micro_rms(s):
    return float(np.std(s['micro_c'])), float(np.std(s['micro_d']))

def _read_native(z, n):
    """A micro track read at its own rate for n frames (mirror-looped), so its wobble rate never changes."""
    if len(z) == 0: return np.zeros(n)
    idx = np.arange(n) % (2 * len(z)); idx = np.where(idx < len(z), idx, 2 * len(z) - 1 - idx)
    return z[idx]

def render_burst(base, edges, rng, stretch=1.0, semis=0.0, micro=None, noise_shape=None, onset=None, body=None,
                 tilt_db_oct=0.0, jitter_add_c=0.0, flutter_add_db=0.0, noise_db=0.0, body_w=0.7, flatten=1.0, tick_db=None, wobble=1.0, am_wobble=1.0,
                 micro_c_scale=1.0, micro_d_scale=1.0, noise_tail_db=0.0, noise_tail_s=0.0, noise_hi_db=0.0, noise_hi_hz=0.0,
                 skip_end_clicks=True, end_fade_s=0.02, partial_hi_db=0.0, partial_hi_hz=2500.0):
    """Recombine stems. base supplies the tonal contours and noise envelope; the other stems can come from other
    bursts (micro, noise_shape, onset, body). Returns audio."""
    n_out = int(round(base['nfr'] * stretch)); n = n_out * H
    pos = np.arange(n_out) / stretch                       # source frame position for the stretched contour
    mic = micro if micro is not None else base
    mc = _read_native(mic['micro_c'], n_out); md = _read_native(mic['micro_d'], n_out)
    if micro is None: mc = np.zeros(n_out); md = np.zeros(n_out)   # own per-partial micro is in tonal_raw
    mc = mc * micro_c_scale; md = md * micro_d_scale              # depth of a borrowed wobble track
    if jitter_add_c > 0:
        mc = mc + jitter_add_c * lp_noise(n_out, min(40.0, FR / 2.2), rng, sr=FR)
        md = md + flutter_add_db * lp_noise(n_out, 12.0, rng, sr=FR)
    out = np.zeros(n)
    t = np.arange(n) / H
    own = micro is None
    for (s0, lc, ld), (_, rc, rd) in zip(base['tonal'], base['tonal_raw']):
        if own: lc, ld = rc, rd
        if wobble != 1.0:   # scale the pitch wobble (deviation from a 1.5 Hz trend), contour shape kept
            tr_ = lpz(lc, 1.5); lc = tr_ + wobble * (lc - tr_)
        if am_wobble != 1.0:
            tr_ = lpz(ld, 1.5); ld = tr_ + am_wobble * (ld - tr_)
        a = int(np.ceil(s0 * stretch)); b = int(np.floor((s0 + len(lc) - 1) * stretch))
        if b - a < 3: continue
        q = pos[a:b + 1] - s0
        if flatten != 1.0:
            md_ = np.median(lc); lc = md_ + flatten * (lc - md_)
        c = np.interp(q, np.arange(len(lc)), lc) + 100 * semis + mc[a:b + 1]
        d = np.interp(q, np.arange(len(ld)), ld) + md[a:b + 1]
        f = 2 ** (c / 1200)
        if tilt_db_oct: d = d + tilt_db_oct * np.log2(f / 1000)
        if partial_hi_db: d = d + partial_hi_db * np.clip(np.log2(f / partial_hi_hz) + 0.5, 0, 1)   # smooth high shelf, tones only
        ii = np.arange(a * H, min(n, (b + 1) * H)); fr = ii / H
        fs = np.interp(fr, np.arange(a, b + 1), f); As = 10 ** (np.interp(fr, np.arange(a, b + 1), d) / 20)
        ramp = np.minimum(1, np.minimum((ii - a * H) / H, (b * H - ii) / H + 1))
        ok = fs < 0.45 * SR
        out[ii] += As * np.clip(ramp, 0, 1) * ok * np.sin(2 * np.pi * np.cumsum(fs) / SR + rng.uniform(0, 6.3))
    # noise: base time envelope (stretched) x chosen spectral shape, shifted with the pitch
    nb = len(base['nshape'])
    Lsrc = base['Ls'].astype(np.float64)
    G0 = np.array([np.interp(pos, np.arange(Lsrc.shape[1]), Lsrc[k_]) for k_ in range(nb)])
    shp = (noise_shape['nshape'] - base['nshape']) if noise_shape is not None else np.zeros(nb)
    bpo = (nb - 1) / np.log2(edges[-1] / 100)             # bands per octave on the residual grid
    k = np.arange(nb)
    shp = np.interp(k - semis / 12 * bpo, k, shp)
    G0 = np.array([np.interp(np.arange(nb) - semis / 12 * bpo, np.arange(nb), G0[:, j]) for j in range(n_out)]).T
    G = G0 + np.clip(shp, -9, 9)[:, None] + noise_db
    if noise_hi_db:                                       # noise layer only, above a frequency (over the tone)
        G = G + noise_hi_db * (base['bands_cen'] >= noise_hi_hz)[:, None]
    if noise_tail_db and noise_tail_s > 0:                # noise layer only, ramped down over the last noise_tail_s
        nt = min(n_out, int(noise_tail_s * FR)); G[:, -nt:] += np.linspace(0, noise_tail_db, nt)[None, :]
    if tilt_db_oct:
        G = G + tilt_db_oct * np.log2(np.maximum(base['bands_cen'], 100) / 1000)[:, None]
    # transients: onset (own or swapped) and the base's clicks (moved with the stretch), pitch-shifted bands
    on = (onset if onset is not None else base)['trans']['onset']
    if onset is not None:                                 # keep a donor onset inside the base's own bandwidth
        cap = np.max(base['Ls'], axis=1) + 3
        on = np.minimum(on, cap[:, None])
    on = np.array([np.interp(k - semis / 12 * bpo, k, on[:, j]) for j in range(on.shape[1])]).T
    m = min(on.shape[1], n_out)
    Gl = 10 ** (G / 20); Gl[:, :m] = np.sqrt(Gl[:, :m] ** 2 + (10 ** (on[:, :m] / 20)) ** 2 * (onset is not None) + 0)
    if onset is None:
        Gl[:, :m] = np.maximum(Gl[:, :m], 10 ** (on[:, :m] / 20))
    for i, col in base['trans']['clicks']:
        if skip_end_clicks and i >= base['nfr'] - 10: continue                # a 'click' in the end margin is the next event leaking in
        j = int(i * stretch)
        if j < n_out:
            Gl[:, j] = np.maximum(Gl[:, j], 10 ** (np.interp(k - semis / 12 * bpo, k, col) / 20))
    if tick_db is not None:                               # hard contact tick: 2 broadband frames at the onset
        j0 = int(np.argmax(np.sum(Gl[:, :max(2, m)] ** 2, axis=0)))
        lvl = np.sqrt(np.mean(Gl[:, j0] ** 2)) * 10 ** (tick_db / 20)
        Gl[:, j0:j0 + 2] = np.maximum(Gl[:, j0:j0 + 2], lvl)
    f_ = np.fft.rfftfreq(1024, 1 / SR)
    Gi = np.array([np.interp(np.log(np.maximum(f_, 1)), np.log(base['bands_cen']), Gl[:, j]) for j in range(n_out)]).T
    Z = Gi * np.exp(2j * np.pi * rng.random(Gi.shape))
    _, nz = signal.istft(Z, SR, nperseg=1024, noverlap=1024 - H)
    nz = nz * np.sqrt(1024 / H)
    nz = nz[:n] if len(nz) >= n else np.pad(nz, (0, n - len(nz)))
    y = out + nz
    if body is not None:                                   # body colour swap: EQ by (target - own), +-9 dB
        c = np.clip(body_w * (body['body'] - base['body']), -6, 6)
        f = np.concatenate([[0], BODY_GRID, [SR / 2]]) / (SR / 2)
        g = 10 ** (np.concatenate([[c[0]], c, [c[-1]]]) / 20)
        y = signal.fftconvolve(y, signal.firwin2(2049, f, g), 'same')
    e = np.ones(n); r = int(0.006 * SR); e[:r] = np.linspace(0, 1, r); r2 = int(end_fade_s * SR); e[-r2:] = np.linspace(1, 0, r2) ** 2
    return y * e

def merge_items(a, b, M=8):
    """Join two pool bursts that were contiguous in the source (the long-burst splitter cut them at a quiet point),
    so they render as one unbroken burst. Both carry M margin frames; cut each at the shared junction and join
    tracks that continue across it (within 3 % in frequency)."""
    na = a['nfr'] - M                                       # junction frame in a's coordinates == frame M in b
    E = np.concatenate([a['E'][:, :na], b['E'][:, M:]], axis=1)
    A_end, out = [], []
    for s0, f, A in a['tracks']:
        hi = min(s0 + len(f), na)
        if hi - s0 < 2: continue
        t = (s0, f[:hi - s0], A[:hi - s0])
        (A_end if s0 + len(f) >= na - 1 else out).append(t)
    used = set()
    for s0, f, A in b['tracks']:
        lo = max(s0, M)
        if s0 + len(f) - lo < 2: continue
        fb, Ab = f[lo - s0:], A[lo - s0:]
        joined = False
        if lo <= M + 1:
            for j, (sa, fa, Aa) in enumerate(A_end):
                if j not in used and abs(fa[-1] - fb[0]) < 0.03 * fb[0]:
                    A_end[j] = (sa, np.concatenate([fa, fb]), np.concatenate([Aa, Ab])); used.add(j); joined = True; break
        if not joined:
            out.append((lo - M + na, fb, Ab))
    out += A_end
    m = dict(a); m.update(E=E, tracks=out, nfr=E.shape[1], t1=b['t1'], level=float(np.sqrt((a['level'] ** 2 + b['level'] ** 2) / 2)))
    return m


# ---------------------------------------------------------------------------------------------------- stored data
@functools.lru_cache(maxsize=1)
def _data():
    raw = DATA.read_bytes()
    if raw[:8] != b"SLSTEMS1":
        raise RecipeError(f"{DATA.name}: not a soundlab stem file")
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    hdr = json.loads(p[4:4 + n].decode())
    blob = p[4 + n:]

    def arr(d):
        dt = np.dtype(d["dt"])
        return np.frombuffer(blob, dt, int(np.prod(d["sh"])), d["o"]).reshape(d["sh"])

    bursts = {}
    for k, b in hdr["bursts"].items():
        E = np.maximum(10 ** ((arr(b["E_q"]).astype(np.float64) * 0.6 - 175.0) / 20) - 1e-9, 0).astype(np.float32)
        t = b["tracks"]
        st, ln, c0 = arr(t["start"]), arr(t["length"]), arr(t["c0"])
        dc, db = arr(t["dc"]).astype(np.float64), arr(t["db"]).astype(np.float64)
        cut = np.concatenate([[0], np.cumsum(ln)])
        pk = arr(t["pk"]).astype(np.float64)        # each track's peak stored exactly, the rest as offsets from it
        tracks = [(int(st[j]), (2 ** ((c0[j] + dc[cut[j]:cut[j + 1]]) / 1200)).astype(np.float32),
                   (pk[j] * 10 ** (db[cut[j]:cut[j + 1]] / 20)).astype(np.float32)) for j in range(len(ln))]
        bursts[k] = dict(src=b["src"], t0=b["t0"], t1=b["t1"], nfr=int(b["nfr"]), f_dom=b["f_dom"], level=b["level"],
                         snr=b["snr"], E=E, tracks=tracks)
    donors = {k: {kk: (arr(v) if isinstance(v, dict) else v) for kk, v in d.items()} for k, d in hdr["donors"].items()}
    return dict(edges=arr(hdr["edges"]).astype(np.float64), bursts=bursts, donors=donors, files=hdr["files"])


def stem_keys() -> dict:
    """The stored bursts (playable bases) and donors (wobble / onset / noise colour / body only), by category."""
    d = _data()
    cat = lambda k: k.split("_")[0]
    out = {}
    for k in d["bursts"]:
        out.setdefault(cat(k), {"bursts": [], "donors": []})["bursts"].append(k)
    for k in d["donors"]:
        out.setdefault(cat(k), {"bursts": [], "donors": []})["donors"].append(k)
    return out


def _item(key):
    d = _data()
    keys = key if isinstance(key, (list, tuple)) else [key]
    for k in keys:
        if k not in d["bursts"]:
            raise RecipeError(f"squeal_stems: unknown burst {k!r}; stored: {sorted(d['bursts'])}")
    it = d["bursts"][keys[0]]
    for k in keys[1:]:                 # bursts the analysis split out of one continuous squeal play as one
        it = merge_items(it, d["bursts"][k])
    return it


_STEMS: dict = {}


def _stem(key, smoothing):
    ck = (json.dumps(key), json.dumps(smoothing, sort_keys=True))
    if ck not in _STEMS:
        it = _item(key)
        s = decompose(it, _data()["edges"])
        if smoothing is not None:
            s = clean(s, _data()["edges"], **smoothing)
        _STEMS[ck] = (s, it)
    return _STEMS[ck]


def _donor(key, smoothing):
    if not key:
        return None
    d = _data()
    if key in d["bursts"]:
        return _stem(key, smoothing)[0]
    if key not in d["donors"]:
        raise RecipeError(f"squeal_stems: unknown donor {key!r}")
    v = d["donors"][key]
    sfx = "_clean" if smoothing is not None else ""
    return dict(src=v["src"], body=v["body"].astype(np.float64), nshape=v["nshape"].astype(np.float64),
                trans=dict(onset=v["onset"].astype(np.float64)), micro_c=v["micro_c" + sfx].astype(np.float64),
                micro_d=v["micro_d" + sfx].astype(np.float64))


def _muted(s, rules, purr):
    """A copy of a burst's stems with stray partials muted by rule (kept in place so random draws match) and an
    optional clamp of one noise band at its median + over_db (a purring hum in the noise layer)."""
    if not rules and not purr:
        return s
    m = copy.copy(s)
    if rules:
        def hit(rc):
            f = 2 ** (np.median(rc) / 1200)
            for r in rules:
                if (r.get("lo", 0) < f < r.get("hi", 1e9) and len(rc) > r.get("min_s", 0.0) * FR
                        and np.ptp(rc) < r.get("max_range_cents", 1e9)):
                    return True
            return False
        flags = [hit(t[1]) for t in s["tonal_raw"]]
        mute = lambda t, f: (t[0], t[1], t[2] - 300.0) if f else t
        m["tonal"] = [mute(t, f) for t, f in zip(s["tonal"], flags)]
        m["tonal_raw"] = [mute(t, f) for t, f in zip(s["tonal_raw"], flags)]
    if purr:
        lo, hi, over = purr
        L = m["Ls"].astype(float).copy()
        b = (m["bands_cen"] >= lo) & (m["bands_cen"] < hi)
        L[b] = np.minimum(L[b], np.median(L[b], axis=1, keepdims=True) + over)
        m["Ls"] = L.astype(np.float32)
    return m


def bg_noise(prof, edges, n, rng):
    cen = np.sqrt(np.maximum(edges[:-1], 50) * edges[1:])
    f = np.fft.rfftfreq(1024, 1 / SR)
    g = np.interp(np.log(np.maximum(f, 1)), np.log(cen), prof)
    Z = g[:, None] * np.exp(2j * np.pi * rng.random((len(f), n // H + 8)))
    _, z = signal.istft(Z, SR, nperseg=1024, noverlap=1024 - H)
    return (z * np.sqrt(1024 / H))[:n]


def _gesture(motions, n, rng_pick):
    """motions_n > 0: repeat the recipe's motion pattern n times, extra motions playing other bursts of the same
    kind from the pool."""
    if n <= 0 or n == len(motions):
        return motions
    pool = stem_keys()
    out = []
    for i in range(n):
        m = dict(motions[i % len(motions)])
        if i >= len(motions):
            k0 = m["burst"] if isinstance(m["burst"], str) else m["burst"][0]
            cands = [k for k in pool[k0.split("_")[0]]["bursts"] if not k.startswith("gate_207140")] or [k0]
            m["burst"] = cands[int(rng_pick.integers(len(cands)))]
        out.append(m)
    return out


def _rng(seed, variation):
    return np.random.default_rng(seed if variation == 0 else [seed, abs(variation), int(variation < 0)])


def render_stems(p: dict, timeline: list | None = None) -> np.ndarray:
    """Render a squeal_stems parameter set at 44.1 kHz (peak 0.89). If `timeline` is a list, (start_s, end_s,
    action) of every motion is appended to it."""
    edges = _data()["edges"]
    seed, var = int(round(p["seed"])), int(round(p["variation"]))
    rng = _rng(seed, var)
    b = float(p["brightness"])
    smoothing = p["smoothing"] if p["smoothing"] else None
    motions = _gesture(list(p["motions"]), int(round(p["motions_n"])), _rng(seed + 1, var))
    if not motions:
        raise RecipeError("squeal_stems: needs at least one motion")
    k0 = motions[0]["burst"] if isinstance(motions[0]["burst"], str) else motions[0]["burst"][0]
    src0 = k0.split("/")[0]
    gaps = [g for g in _data()["files"][src0]["gaps"] if 0.06 < g < 1.5] or [0.25]
    micro_d, onset_d = _donor(p["micro"], smoothing), _donor(p["onset"], smoothing)
    noise_d, body_d = _donor(p["noise"], smoothing), _donor(p["body"], smoothing)
    kw = dict(RAW_RENDER) if smoothing is None else {}
    kw.update(noise_db=p["noise_db"], noise_hi_db=p["noise_hi_db"], noise_hi_hz=p["noise_hi_hz"],
              noise_tail_db=p["noise_tail_db"] - 10.0 * b, noise_tail_s=p["noise_tail_s"] if (p["noise_tail_db"] or b) else 0.0,
              micro_c_scale=p["micro_depth"], micro_d_scale=p["micro_depth"], wobble=p["wobble"], am_wobble=p["am_wobble"],
              partial_hi_db=p["partial_hi_db"] + 3.0 * b, partial_hi_hz=p["partial_hi_hz"],
              jitter_add_c=p["jitter_cents"], flutter_add_db=p["flutter_db"])
    if kw["noise_tail_db"] and not kw["noise_tail_s"]:
        kw["noise_tail_s"] = 0.12
    parts, gains, mg = [], [], []
    for i, m in enumerate(motions):
        s, it = _stem(m["burst"], smoothing)
        sm = _muted(s, p["mute"], p["purr_gate"])
        speed, press = float(m.get("speed", 1.0)), float(m.get("pressure", 1.0))
        stretch = p["stretch"] / max(speed, 1e-3)
        if float(m.get("duration", 0) or 0) > 0:
            stretch = float(m["duration"]) / (s["nfr"] / FR)
        semis = p["pitch"] + 2.0 * b + 1.2 * math.log2(max(speed, 1e-3))
        tilt = p["tilt"] + 2.0 * b + 1.5 * math.log2(max(press, 1e-3))
        st = stretch * rng.uniform(0.95, 1.05)
        se = semis + rng.uniform(-0.3, 0.3)
        mic = micro_d
        if mic is not None and p["borrow"] == "if_similar":     # borrow a wobble only if it measures like our own
            bc, bd = micro_rms(s)
            dc, dd = micro_rms(mic)
            mic = mic if (abs(dc - bc) <= 0.25 * bc and abs(dd - bd) <= 0.25 * bd) else None
        args = dict(stretch=st, semis=se, micro=mic, noise_shape=noise_d, onset=onset_d if i == 0 else None,
                    body=body_d, tilt_db_oct=tilt, flatten=p["flatten"], **kw)
        level = it["level"]
        if sm is not s:      # muted partials must not turn into a gain change on the rest: keep the full burst's gain
            full = render_burst(s, edges, copy.deepcopy(rng), **args)
            y = render_burst(sm, edges, rng, **args)
            level = level * np.sqrt(np.mean(y ** 2)) / max(np.sqrt(np.mean(full ** 2)), 1e-12)
        else:
            y = render_burst(s, edges, rng, **args)
        if p["trim_s"] > 0 and len(y) > int(p["trim_s"] * SR):
            y = y[:int(p["trim_s"] * SR)].copy()
            fl = int(p["trim_fade_s"] * SR)
            y[-fl:] *= np.linspace(1, 0, fl) ** 2
        g = level / (np.sqrt(np.mean(y ** 2)) + 1e-12)
        gains.append(g)
        parts.append(y * g * press)
        if timeline is not None:
            t0 = 0.25 + sum(len(q) for q in parts[:-1]) / SR
            timeline.append((t0, t0 + len(y) / SR, m.get("action", "")))
        mg.append(np.concatenate([sm["Ls"][:, :5], sm["Ls"][:, -5:]], axis=1))
        if i < len(motions) - 1:
            gap = float(rng.choice(gaps)) * p["stretch"] * p["gap_scale"]     # the motion stops: no squeal
            ga = m.get("gap_after", "auto")
            parts.append(np.zeros(int((gap if ga in (None, "auto") else float(ga)) * SR)))
    y = np.concatenate([np.zeros(int(0.25 * SR))] + parts + [np.zeros(int(0.4 * SR))])
    bed = 10 ** (median_filter(np.percentile(np.concatenate(mg, axis=1), 20, axis=1), size=9, mode="nearest") / 20)
    bed_gain = (10 ** (p["bed_db"] / 20) if p["bed_db"] > -120 else 0.0) * (1.0 - b)
    y = y + bg_noise(bed, edges, len(y), rng) * float(np.mean(gains)) * bed_gain
    y = signal.sosfiltfilt(signal.butter(2, 40 / (SR / 2), "high", output="sos"), y)
    y = y / (np.abs(y).max() + 1e-12) * 0.89
    if p["latch"]:
        y = _with_latch(y, p["latch_gap_s"], p["latch_level"])
    return y


def _with_latch(g, gap_s, level):
    """A real latch lift and drop (two clean clank bursts of a CC0 gate recording, 30 ms apart) after the squeal."""
    rng = np.random.default_rng(7)
    edges = _data()["edges"]
    out = []
    for i, k in enumerate(("gate_207140/0", "gate_207140/1")):
        s, it = _stem(k, None)
        y = render_burst(s, edges, rng, end_fade_s=0.01)
        out.append(y * it["level"] / (np.sqrt(np.mean(y ** 2)) + 1e-12))
        if i == 0:
            out.append(np.zeros(int(0.03 * SR)))
    L = np.concatenate(out)
    last = int(np.nonzero(np.abs(g) > 10 ** (-50 / 20))[0][-1])
    L *= level * np.abs(g).max() / (np.abs(L).max() + 1e-12)
    y = np.concatenate([g[:last], np.zeros(int(gap_s * SR)), L, np.zeros(int(0.3 * SR))])
    return y / (np.abs(y).max() + 1e-12) * 0.89


def stems_length_s(p: dict) -> float:
    return len(render_stems(p)) / SR


@model("squeal_stems", "sound",
       "Squeals, squeaks and creaks rebuilt from stems of CC0 recordings (numbers only, no audio): partial tracks, "
       "the real wobble, the noise layer, onset/clicks and body colour per burst, recombined across recordings. A "
       "gesture timeline of motions (each with speed, pressure, duration) plays one burst per motion; the pause "
       "between motions is the motion stopping. Brightness turns a hinge into a gate (higher, brighter, more "
       "metallic, dry); an optional latch clank ends it.",
       [
           Param("motions", [{"burst": "hinge_535530/9"}], "", doc="gesture timeline: list of {burst, action, speed, "
                 "pressure, duration, gap_after}; burst is a stored burst key or a list of keys that play joined",
                 is_list=True),
           Param("micro", "", "", doc="donor of the wobble track (stored key, '' = each burst's own)"),
           Param("onset", "", "", doc="donor of the first motion's onset ('' = own)"),
           Param("noise", "", "", doc="donor of the noise colour ('' = own)"),
           Param("body", "", "", doc="donor of the body colour ('' = own)"),
           Param("borrow", "always", "", choices=("always", "if_similar"),
                 doc="if_similar: use the wobble donor only when it measures within 25 % of the burst's own"),
           Param("stretch", 1.0, "", 0.5, 2.0, "time stretch of every burst (>1 slower)"),
           Param("pitch", 0.0, "st", -6, 6, "pitch shift in semitones (noise shaping follows)"),
           Param("tilt", 0.0, "dB/oct", -6, 6, "spectral tilt about 1 kHz (tones and noise)"),
           Param("brightness", 0.0, "", 0, 1, "0 as recorded .. 1 gate: +2 st, +2 dB/oct, +3 dB upper partials, "
                 "no ambient bed, noise tails pulled down (dry, outdoor)"),
           Param("jitter_cents", 0.0, "cents", 0, 30, "extra pitch jitter on top of the real wobble"),
           Param("flutter_db", 0.0, "dB", 0, 3, "extra level flutter on top of the real wobble"),
           Param("micro_depth", 1.0, "", 0, 2, "depth of a borrowed wobble track"),
           Param("wobble", 1.0, "", 0, 1.5, "scale of pitch wobble around a 1.5 Hz trend"),
           Param("am_wobble", 1.0, "", 0, 1.5, "scale of level wobble around a 1.5 Hz trend"),
           Param("flatten", 1.0, "", 0, 1, "scale of the pitch contour around its median (hooks, glides)"),
           Param("smoothing", None, "", doc="null = stems as analysed; or the anti-bubble pass {min_track_ms, "
                 "rel_db, noise_ms, amp_fc, freq_fc}", is_list=True),
           Param("noise_db", 0.0, "dB", -40, 20, "noise layer level"),
           Param("noise_hi_db", 0.0, "dB", -40, 20, "noise layer above noise_hi_hz"),
           Param("noise_hi_hz", 0.0, "Hz", 0, 20000),
           Param("noise_tail_db", 0.0, "dB", -40, 0, "noise pulled down over the last noise_tail_s of each burst"),
           Param("noise_tail_s", 0.12, "s", 0, 2),
           Param("partial_hi_db", 0.0, "dB", -20, 20, "smooth high shelf on the partials above partial_hi_hz"),
           Param("partial_hi_hz", 2500.0, "Hz", 200, 16000),
           Param("mute", [], "", doc="stray-partial rules: list of {lo, hi, min_s, max_range_cents}", is_list=True),
           Param("purr_gate", [], "", doc="[lo_hz, hi_hz, over_db]: clamp that noise band at its median + over_db",
                 is_list=True),
           Param("trim_s", 0.0, "s", 0, 30, "cut each burst at this time (0 = whole burst)"),
           Param("trim_fade_s", 0.3, "s", 0.005, 2),
           Param("bed_db", 0.0, "dB", -200, 20, "ambient bed (the band levels just outside the bursts)"),
           Param("latch", False, "", doc="end with a real latch lift and drop"),
           Param("latch_gap_s", 0.35, "s", 0, 3),
           Param("latch_level", 0.5, "", 0, 2, "latch peak re the squeal peak"),
           Param("motions_n", 0, "", 0, 16, "number of motions (0 = as listed; more repeat the pattern with other "
                 "bursts of the same kind)"),
           Param("gap_scale", 1.0, "", 0, 4, "scales the pauses between motions"),
           Param("seed", 0, "", 0, 2**31 - 1, "the model's own seed (the recipe default reproduces its reference)"),
           Param("variation", 0, "", -1e6, 1e6, "0 = the reference render; other values reseed"),
           Param("at", 0.0, "s", 0, None),
       ],
       ["Serra & Smith 1990 (spectral modelling synthesis: sinusoids + residual)",
        "McAulay & Quatieri 1986 (sinusoidal tracking)",
        "Akay 2002 (acoustics of friction); Woodhouse 2004 (bowed-string Helmholtz motion)",
        "Freesound CC0 recordings 766155, 563136, 575206, 455324, 268574, 124289, 535530, 429130, 788701, 784964, "
        "204759, 88502, 397687, 483166, 207140 (analysed numbers only; THIRD_PARTY.md)"],
       min_s=lambda p: 0.1)
def squeal_stems(ctx: Ctx, x, **p):
    p = dict(p)
    at = float(p.pop("at"))
    y = render_stems(p)
    if ctx.sr != SR:
        g = math.gcd(int(ctx.sr), SR)
        y = signal.resample_poly(y, int(ctx.sr) // g, SR // g)
    out = np.zeros(ctx.n)
    i0 = int(at * ctx.sr)
    if i0 < ctx.n:
        m = min(len(y), ctx.n - i0)
        out[i0:i0 + m] = y[:m]
    return out
