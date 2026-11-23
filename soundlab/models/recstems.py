"""Shared renderer for impact_stems, weapon_stems, foot_stems, foley_stems and door_stems.

Numbers-only recording stems: body-mode partial tracks, real transient events as
1/8-octave envelopes (rebuilt with an impulse-plus-noise vocoder), and a 1/8-octave
1 ms sub-band vocoder bed per inter-hit segment. No band pitch-shifts (size is tape
speed), no freeze frames, no time-squeeze, no smoothing that adds hiss.

This module is the engine; each family module loads its own .bin and registers a model.
See docs/impact_stems.md, docs/weapon_stems.md, docs/foot_stems.md, docs/foley_stems.md and
docs/door_stems.md.
"""
from __future__ import annotations

import json
import lzma
import math
import struct
from pathlib import Path

import numpy as np
from scipy import signal

from .air_stems import GRID6, _eq_to
from ..fastpath import memo_arrays
from .liquid_stems import _interp_rows, _vocoder_passes
from .machine_stems import SR, _bank, _q, _shelf, _tape, _vocoder

LAST = {}


def _load_bin(path: Path, magic: bytes):
    """Load a stems bin. Source arrays are unpacked lazily (float32, few at a time) so many
    workers can share a machine without each holding every envelope as float64."""
    raw = path.read_bytes()
    if isinstance(magic, str):
        magic = magic.encode("ascii")
    if raw[:8] != magic:
        raise ValueError(f"{path.name}: not a soundlab stem file")
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    hdr = json.loads(p[4:4 + n].decode())
    blob = p[4 + n:]
    return {"sr": hdr.get("sr", SR), "family": hdr.get("family"), "files": hdr.get("files", {}),
            "sources": hdr["sources"], "_blob": blob, "_cache": {}, "_cache_order": []}


_CACHE_N = 3  # unpacked sources kept per family


def _arr(blob, d):
    return np.frombuffer(blob, np.dtype(d["dt"]), int(np.prod(d["sh"])), d["o"]).reshape(d["sh"])


def _unpack_source(s, blob):
    st = {kk: s[kk] for kk in s if not isinstance(s[kk], dict) or "dt" not in s[kk]}
    st["n"] = int(s["n"])
    if "bed_env" in s:
        st["bed_env"] = _q(_arr(blob, s["bed_env"]), s["bed_lo"]).astype(np.float32, copy=False)
        st["bed_a0"] = _arr(blob, s["bed_a0"]).astype(np.int32)
        st["bed_b1"] = _arr(blob, s["bed_b1"]).astype(np.int32)
        st["bed_off"] = _arr(blob, s["bed_off"]).astype(np.int32)
        st["bed_sr"] = int(s.get("bed_sr", 1000))
    if "bed_eq6" in s:
        st["bed_eq6"] = _arr(blob, s["bed_eq6"]).astype(np.float32)
    if "ev_env" in s:
        st["ev_env"] = _q(_arr(blob, s["ev_env"]), s["ev_lo"]).astype(np.float32, copy=False)
        st["ev_t"] = _arr(blob, s["ev_t"]).astype(np.int32)
        st["ev_len"] = _arr(blob, s["ev_len"]).astype(np.int32)
        st["ev_pad"] = _arr(blob, s["ev_pad"]).astype(np.int32)
        st["ev_off"] = _arr(blob, s["ev_off"]).astype(np.int32)
        st["ev_energy"] = _arr(blob, s["ev_energy"]).astype(np.float32)
    if "ev_eq6" in s:
        st["ev_eq6"] = _arr(blob, s["ev_eq6"]).astype(np.float32)
    if "ev_cap" in s:       # optional (foley_stems, door_stems): grain lengths and the grain level cap
        st["ev_glen"] = _arr(blob, s["ev_glen"]).astype(np.int32)
        st["ev_cap"] = _q(_arr(blob, s["ev_cap"]), s["ev_cap_lo"]).astype(np.float32, copy=False)
        st["ev_cap_off"] = _arr(blob, s["ev_cap_off"]).astype(np.int32)
    if "md_amp" in s:
        st["md_nom"] = _arr(blob, s["md_nom"]).astype(np.float32)
        st["md_t0"] = _arr(blob, s["md_t0"]).astype(np.float32)
        st["md_dt"] = _arr(blob, s["md_dt"]).astype(np.float32)
        st["md_len"] = _arr(blob, s["md_len"]).astype(np.int32)
        st["md_off"] = _arr(blob, s["md_off"]).astype(np.int32)
        st["md_f"] = _arr(blob, s["md_f"]).astype(np.float32)
        st["md_amp"] = _q(_arr(blob, s["md_amp"]), s["md_lo"]).astype(np.float32, copy=False)
        st["md_ph"] = np.unwrap(_arr(blob, s["md_ph"]).astype(np.float64) * (2 * np.pi / 256)).astype(np.float32)
    if "hf_f" in s:
        st["hf_f"] = _arr(blob, s["hf_f"]).astype(np.float32)
        st["hf_db"] = _arr(blob, s["hf_db"]).astype(np.float32)
    return st


def get_source(data, name):
    if name not in data["sources"]:
        raise KeyError(name)
    cache = data.setdefault("_cache", {})
    if name in cache:
        order = data.setdefault("_cache_order", [])
        if name in order:
            order.remove(name)
        order.append(name)
        return cache[name]
    st = _unpack_source(data["sources"][name], data["_blob"])
    cache[name] = st
    order = data.setdefault("_cache_order", [])
    order.append(name)
    while len(order) > _CACHE_N:
        old = order.pop(0)
        cache.pop(old, None)
    return st


def minphase_eq(y, d6):
    """Static 1/6-octave EQ as a minimum-phase FIR (no pre-ring ahead of a hit)."""
    N = 16384
    f = np.fft.rfftfreq(N, 1 / SR)
    G = np.interp(np.log(np.maximum(f, 10)), np.log(GRID6), np.clip(d6, -40, 20)) / 20 * np.log(10)
    c = np.fft.irfft(G, N)
    w = np.zeros(N); w[0] = 1; w[1:N // 2] = 2; w[N // 2] = 1
    h = np.fft.irfft(np.exp(np.fft.rfft(c * w)), N)[:N // 2]
    h *= np.r_[np.ones(N // 4), np.cos(np.linspace(0, np.pi / 2, N // 4)) ** 2]
    return signal.fftconvolve(y, h)[:len(y)]


def minphase_lin(y, f, gdb, N=16384):
    fr = np.fft.rfftfreq(N, 1 / SR)
    G = np.interp(fr, f, gdb) / 20 * np.log(10)
    c = np.fft.irfft(G, N)
    w = np.zeros(N); w[0] = 1; w[1:N // 2] = 2; w[N // 2] = 1
    h = np.fft.irfft(np.exp(np.fft.rfft(c * w)), N)[:N // 2]
    h *= np.r_[np.ones(N // 4), np.cos(np.linspace(0, np.pi / 2, N // 4)) ** 2]
    return signal.fftconvolve(y, h)[:len(y)]


@memo_arrays
def voc_ev(env, env_sr, n, seed, i0, iters=8, per_oct=8, imp=1.0):
    rng = np.random.default_rng(seed)
    Hs, _ = _bank(n, per_oct)
    t = np.arange(n) / SR
    te = np.arange(env.shape[1]) / env_sr
    y = rng.standard_normal(n)
    if imp:
        y[max(0, min(n - 1, int(i0)))] += imp * np.sqrt(n)
    E = _interp_rows(t, te, env)
    return _vocoder_passes(y, Hs, E, n, iters)


def synth_modes(st, n, a=0, e=None):
    if "md_amp" not in st:
        return np.zeros(n)
    e = n if e is None else e
    ta = np.arange(n) / SR
    y = np.zeros(n)
    nmd = int(st.get("md_n", 0))
    for i in range(nmd):
        o = int(st["md_off"][i]); ln = int(st["md_len"][i])
        t0 = float(st["md_t0"][i]); dt = float(st["md_dt"][i])
        te = t0 + dt * np.arange(ln)
        amp = st["md_amp"][o:o + ln]
        fr = st["md_f"][o:o + ln]
        ph = st["md_ph"][o:o + ln]
        # shift so window start a is t=0 in the output
        t_abs = ta + a / SR
        phi = 2 * np.pi * np.cumsum(np.interp(t_abs, te, fr)) / SR
        if st.get("md_abs") and a > 0:
            # families whose mode phases are measured from the start of the recording (foley, doors)
            phi = phi + 2 * np.pi * np.sum(np.interp(np.arange(a) / SR, te, fr)) / SR
        y += np.interp(t_abs, te, amp) * np.cos(phi + np.interp(t_abs, te, ph))
    return y


def _event_slice(st, j):
    o0 = int(st["ev_off"][j]); o1 = int(st["ev_off"][j + 1])
    return st["ev_env"][:, o0:o1], int(st["ev_t"][j]), int(st["ev_len"][j]), int(st["ev_pad"][j])


def render_events(st, n, a, e, seed, evg=1.0, identity=True):
    if "ev_env" not in st:
        return np.zeros(n)
    y = np.zeros(n)
    ev_rate = int(st.get("ev_rate", 8000))
    ev_imp = float(st.get("ev_imp", 1.0))
    n_ev = len(st["ev_t"])
    for j in range(n_ev):
        E, s, ne, P = _event_slice(st, j)
        # event audio spans s-P .. s-P+ne in source samples; keep if it overlaps [a, e)
        s0 = s - P
        if s0 + ne <= a or s0 >= e:
            continue
        sd = (7919 + j) if identity else int(seed) + 17 * j + 3
        w = voc_ev(E, ev_rate, ne, sd, P + int(0.002 * SR), imp=ev_imp)
        mk = np.ones(len(w)); mk[:P] = 0
        fd = int(0.0005 * SR)
        if P + fd <= len(mk):
            mk[P:P + fd] = np.sin(np.linspace(0, np.pi / 2, fd)) ** 2
        if "ev_cap" in st:
            w = _cap_event(st, j, w, mk, P)
            mk = 1.0
        # place into window: source s0 maps to output s0-a
        o = s0 - a
        w = w * mk
        if o < 0:
            w, o = w[-o:], 0
        m = min(len(w), n - o)
        if m > 0:
            y[o:o + m] += w[:m] * evg
    if st.get("ev_eq6") is not None and np.any(y):
        y = minphase_eq(y, st["ev_eq6"])
    return y


def _cap_event(st, j, w, mk, P):
    """Event end and level as the approved rebuilds (families that store a grain cap): a 3 ms fade at the end
    of the real grain, then the 1 ms RMS envelope of the rebuilt event held at or under the grain's own."""
    from scipy.ndimage import uniform_filter1d
    e0 = P + int(st["ev_glen"][j]); fo = int(0.003 * SR)
    mk = mk.copy()
    k = len(mk[e0:e0 + fo])
    if k > 0:
        mk[e0:e0 + fo] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    mk[e0 + fo:] = 0
    w = w * mk
    hop = int(st.get("ev_cap_hop", 11))
    c = st["ev_cap"][int(st["ev_cap_off"][j]):int(st["ev_cap_off"][j + 1])]
    eg = np.interp(np.arange(len(w)), np.arange(len(c)) * hop, c)
    ew = np.sqrt(np.maximum(uniform_filter1d(w ** 2, 44), 0)) + 1e-12
    cg = np.minimum(1.0, eg / ew)
    cg = np.minimum(cg, uniform_filter1d(cg, 44))
    return w * cg


def render_bed(st, n, a, e, seed, bdg=1.0, identity=True):
    if "bed_env" not in st:
        return np.zeros(n)
    y = np.zeros(n)
    nseg = len(st["bed_a0"])
    sr_env = int(st.get("bed_sr", 1000))
    for k in range(nseg):
        a0 = int(st["bed_a0"][k]); b1 = int(st["bed_b1"][k])
        if b1 <= a or a0 >= e:
            continue
        o0 = int(st["bed_off"][k]); o1 = int(st["bed_off"][k + 1])
        E = st["bed_env"][:, o0:o1]
        sd = k if identity else int(seed) + 101 * k + 5
        w = _vocoder(E, sr_env, b1 - a0, None, sd, iters=4)
        # mask as in run_*: fade in at the segment start (relative to a0), zero before it
        # window of this vocoder is [a0, b1) in source time
        t = np.arange(a0, b1) / SR
        # original used pads and wk from residual; here E already includes pads.
        # Zero before the inner start if we can infer it; use a short fade at the edges of the overlap.
        fd = int(0.0005 * SR)
        if st.get("bed_exact") and len(st.get("bounds") or []) > k + 1:
            # the approved rebuilds' mask: zero before the segment start, 0.5 ms in, 4 ms out ending 1 ms early
            ba, bb = int(st["bounds"][k]), int(st["bounds"][k + 1])
            mk = np.clip((t - ba / SR) / 0.001 + 1, 0, 1) * np.clip((bb / SR - 0.001 - t) / 0.004, 0, 1)
            i0 = max(0, ba - a0); mk[:i0] = 0
            mk[i0:i0 + fd] = np.sin(np.linspace(0, np.pi / 2, fd))[:len(mk[i0:i0 + fd])] ** 2
        else:
            mk = np.ones(len(w))
            if fd > 1:
                mk[:fd] *= np.sin(np.linspace(0, np.pi / 2, fd)) ** 2
                mk[-fd:] *= np.cos(np.linspace(0, np.pi / 2, fd)) ** 2
        o = a0 - a
        if o < 0:
            w, mk, o = w[-o:], mk[-o:], 0
        m = min(len(w), n - o)
        if m > 0:
            y[o:o + m] += (w * mk)[:m] * bdg
    if st.get("bed_eq6") is not None and np.any(y):
        y = minphase_eq(y, st["bed_eq6"])
    return y


def soften_events(ev, soft):
    if soft <= 0:
        return ev
    fc = min(14000.0, 12000.0 / (1.0 + 0.8 * soft))
    return signal.sosfilt(signal.butter(2, fc, "low", fs=SR, output="sos"), ev)


def post(y, force=0.0, soft=0.0, cents=0.0, dist=1.0):
    if force:
        y = signal.sosfilt(_shelf(3000.0, 0.7 * force, "high"), y) * 10 ** (0.5 * force / 20)
    if soft:
        y = signal.sosfilt(_shelf(120.0, -4.0 * soft, "low"), y)
    if cents:
        y = _tape(y, cents)
    if dist != 1.0:
        ld = math.log2(max(dist, 1e-6))
        y = signal.sosfilt(_shelf(2500.0, -5 * ld, "high"), y)
        y = signal.sosfilt(signal.butter(2, min(0.99 * SR / 2, 14000.0 / dist ** 0.5), fs=SR, output="sos"), y)
        y = y * 10 ** (-6 * ld / 20)
    return y


def kept_gestures(st, early=False):
    gs = [g for g in (st.get("gestures") or []) if not g.get("drop")]
    if early and gs:
        nkeep = max(2, int(math.ceil(len(gs) * 0.5)))
        gs = gs[:nkeep] or gs
    return gs


def plan_pieces(st, p, rng):
    """Return a list of (src_a, src_end, out_t) in seconds for count gestures."""
    early = bool(p.get("early"))
    gs = kept_gestures(st, early=early)
    k = int(p.get("count", 1) or 1)
    tempo = float(p.get("tempo", 1.0) or 1.0)
    if not gs:
        return [(0.0, st["n"] / SR, 0.0)]
    hit = int(p.get("hit", -1) if p.get("hit") is not None else -1)
    if hit != -1 and k <= 1:
        return _plan_hit(st, gs, p, rng, hit)
    if p.get("which") is not None and int(p.get("which")) >= 0 and k <= 1:
        j = int(p["which"]) % len(gs)
        g = gs[j]
        return [(float(g["a"]), float(g["end"]), 0.0)]
    if k <= 1:
        # pick a representative untruncated gesture, or the first
        good = [i for i, g in enumerate(gs)
                if not g.get("truncated") and float(g.get("peak_db", 0)) > -14
                and 0.12 <= float(g.get("dur_s") or (g["end"] - g["a"])) <= 3.0]
        # default recipes should be deterministic: first good, else first
        j = good[0] if good else 0
        # variation/seed may pick another
        if int(p.get("variation", 0) or 0) != 0 and good:
            j = int(rng.choice(good))
        g = gs[j]
        return [(float(g["a"]), float(g["end"]), 0.0)]
    # k > 1: a contiguous run if possible
    jmax = max(0, len(gs) - k)
    j = 0
    if int(p.get("variation", 0) or 0) != 0:
        j = int(rng.integers(0, jmax + 1)) if jmax else 0
    j = min(j, jmax)
    ii = list(range(j, j + min(k, len(gs) - j)))
    if len(ii) == 1:
        g = gs[ii[0]]
        return [(float(g["a"]), float(g["end"]), 0.0)]
    # if tight, one window; else space by onsets / tempo
    def onset(g):
        ons = g.get("onsets") or []
        return float(ons[0]) if ons else float(g["a"])
    tight = all(gs[ii[n + 1]]["a"] - gs[ii[n]]["end"] <= 0.15
                or onset(gs[ii[n + 1]]) - onset(gs[ii[n]]) <= 1.2
                for n in range(len(ii) - 1))
    if tight and tempo == 1.0:
        return [(float(gs[ii[0]]["a"]), float(gs[ii[-1]]["end"]), 0.0)]
    out, t = [], 0.0
    for n, i in enumerate(ii):
        g = gs[i]
        on = onset(g)
        out.append((float(g["a"]), float(g["end"]), t - (on - float(g["a"]))))
        if n < len(ii) - 1:
            d = onset(gs[ii[n + 1]]) - on
            rest = float(g["end"]) - on + 0.12
            gap = (d if 0.15 < d <= 1.4 else rest) / tempo
            t += gap
    sh = -min(pce[2] for pce in out)
    return [(a, e, t + sh) for a, e, t in out]


def _plan_hit(st, gs, p, rng, hit):
    """One hit of a multi-hit gesture (a knock of a knock-knock-knock): the gesture as usual, then the source
    window from just before hit `hit` to just before the next one (the last hit keeps its whole tail). hit -2:
    hit 0 for variation 0, else a hit drawn with the variation's generator. A cut of the real recording; nothing
    is averaged or stretched."""
    (a, e, _), = plan_pieces(st, dict(p, hit=-1), rng)
    g = next((g for g in gs if abs(float(g["a"]) - a) < 1e-9), None)
    ons = [float(o) for o in ((g or {}).get("onsets") or []) if a <= float(o) < e]
    if len(ons) < 2:
        return [(a, e, 0.0)]
    if hit >= 0:
        h = hit % len(ons)
    else:
        h = int(rng.integers(0, len(ons))) if int(p.get("variation", 0) or 0) != 0 else 0
    a2 = max(a, ons[h] - 0.012, (ons[h - 1] + 0.004) if h else a)
    e2 = ons[h + 1] - 0.004 if h + 1 < len(ons) else e
    return [(a2, e2, 0.0)]


def take_key(data, p, default_source):
    """What makes a take a different take, without rendering it: the stored source and the gesture windows
    render() would play for params p (same seed and generator as render)."""
    p = dict(p)
    src = p.pop("source", default_source)
    p.pop("at", None)
    if src not in data["sources"]:
        return None
    seed = int(p.get("seed", 0) or 0)
    var = int(p.get("variation", 0) or 0)
    if var:
        seed = seed + 10007 * var
    seed = int(seed) % (2 ** 31)
    rng = np.random.default_rng(seed if seed else 1)
    pieces = plan_pieces(get_source(data, src), p, rng)
    return (src,) + tuple((round(a, 4), round(e, 4)) for a, e, _ in pieces)


def render_window(st, a, e, seed, force=0.0, soft=0.0, identity=True):
    """Render source samples [a, e) (seconds) into a new buffer."""
    A, E = int(round(a * SR)), int(round(e * SR))
    E = min(E, st["n"]); A = max(0, min(A, E - 8))
    n = E - A
    evg = 10 ** ((3 * force - 1.5 * soft) / 20)
    bdg = 10 ** ((1.5 * force) / 20)
    mdg = 10 ** ((1.0 * force) / 20)
    ev = soften_events(render_events(st, n, A, E, seed, evg=evg, identity=identity), soft)
    bed = render_bed(st, n, A, E, seed, bdg=bdg, identity=identity)
    mo = synth_modes(st, n, a=A) * mdg
    y = (bed + mo + ev) * float(st.get("stem_gain", 1.0) or 1.0)
    if "hf_f" in st:
        y = minphase_lin(y, st["hf_f"], st["hf_db"])
    if st.get("sil_fr"):
        y = y * _silence_gate(st, A, n)
    return y


def _silence_gate(st, A, n):
    """Gain mask for source samples [A, A+n): 0 where the recording itself is digital silence (stored as
    frame ranges, >= 100 dB under its peak for >= 20 ms), 1 elsewhere, 5 ms ramps inside the silent stretch.
    Sources without silent stretches skip this entirely."""
    hop = int(st.get("sil_hop", 44))
    g = np.ones(n)
    hit = False
    for a, b in st["sil_fr"]:
        i0, i1 = max(0, a * hop - A), min(n, b * hop - A)
        if i1 > i0:
            g[i0:i1] = 0.0; hit = True
    if not hit:
        return 1.0
    k = int(0.005 * SR)
    return np.maximum(g, np.convolve(g, np.ones(k) / k, "same"))


def render(st, p, rng=None):
    """Render one take from a packed source. p: count, tempo, which, variation, seed,
    force, soft, cents, dist, early, identity (True when variation==0 and force==0 and soft==0)."""
    p = dict(p)
    seed = int(p.get("seed", 0) or 0)
    var = int(p.get("variation", 0) or 0)
    if var:
        seed = seed + 10007 * var
    # vary abs on 0.variation can be negative; the generator needs a uint seed
    seed = int(seed) % (2 ** 31)
    if seed < 0:
        seed += 2 ** 31
    rng = rng or np.random.default_rng(seed if seed else 1)
    force = float(p.get("force", 0.0) or 0.0)
    soft = float(p.get("soft", 0.0) or 0.0)
    cents = float(p.get("cents", p.get("speed_cents", 0.0)) or 0.0)
    dist = float(p.get("distance", p.get("dist", 1.0)) or 1.0)
    identity = (var == 0 and abs(force) < 1e-12 and abs(soft) < 1e-12)
    pieces = plan_pieces(st, p, rng)
    T = max(t + (e - a) for a, e, t in pieces) + 0.02
    lead = 0.005
    n = int((T + lead) * SR) + 1
    y = np.zeros(n)
    for i, (a, e, t) in enumerate(pieces):
        w = render_window(st, a, e, seed + i, force=force, soft=soft, identity=identity)
        o = int((t + lead) * SR)
        m = min(len(w), n - o)
        if m > 0:
            # short fades at piece edges
            fi, fo = int(0.002 * SR), int(0.03 * SR)
            ww = w[:m].copy()
            k = min(fi, m); ww[:k] *= np.sin(np.linspace(0, np.pi / 2, k)) ** 2
            k = min(fo, m); ww[m - k:] *= np.cos(np.linspace(0, np.pi / 2, k)) ** 2
            y[o:o + m] += ww
    y = post(y, force=force, soft=soft, cents=cents, dist=dist)
    # trim trailing silence, keep a short tail
    env = np.abs(y); thr = 1e-4 * (env.max() + 1e-12)
    last = np.where(env > thr)[0]
    if len(last):
        y = y[: min(len(y), int(last[-1] + 0.08 * SR))]
    # match the identity rebuild's RMS when this is an identity full-source render
    if identity and p.get("count", 1) in (0, None) and not p.get("which") and st.get("y_rms"):
        r = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-30))
        if r > 0:
            y = y * (float(st["y_rms"]) / r)
    LAST.clear(); LAST.update(source=p.get("source"), pieces=pieces, n=len(y))
    return y.astype(np.float64)


def render_into_ctx(ctx, p, data, default_source, family_sr=SR):
    from ..core import RecipeError
    p = dict(p)
    src = p.pop("source", default_source)
    if src not in data["sources"]:
        raise RecipeError(f"unknown source {src!r}; have {sorted(k for k in data['sources'] if not str(k).startswith('_'))}")
    at = float(p.pop("at", 0.0) or 0.0)
    y = render(get_source(data, src), p)
    if ctx.sr != family_sr:
        g = math.gcd(int(ctx.sr), family_sr)
        y = signal.resample_poly(y, int(ctx.sr) // g, family_sr // g)
    out = np.zeros(ctx.n)
    i0 = int(at * ctx.sr)
    if i0 < ctx.n:
        m = min(len(y), ctx.n - i0)
        out[i0:i0 + m] = y[:m]
    return out
