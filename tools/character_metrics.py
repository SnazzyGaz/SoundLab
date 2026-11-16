"""Targeted 'listening note' metrics: one or two numbers per complaint, for refs vs versions.

usage: python tools/character_metrics.py TARGETS.json REF_DIR OUT.md LABEL=DIR [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

from soundlab import analysis as A
from soundlab.audio_io import read_audio
from soundlab.fit import prepare

SR = 24000


def band(x, lo, hi):
    sos = signal.butter(4, [lo, hi] if lo else hi, btype="band" if lo else "low", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x)


def share(x, lo, hi):
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(X[(f >= lo) & (f < hi)].sum() / (X.sum() + 1e-20))


def centroid(x):
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float((f * X).sum() / (X.sum() + 1e-20))


def loudest(x, win):
    e = np.convolve(x ** 2, np.ones(int(win * SR)), "valid")
    i = int(np.argmax(e))
    return x[i:i + int(win * SR)]


def duck(x):
    T, F, Ap, R = A.f0_track(x, SR, 80, 1000, threshold=0.35)
    ev = A.envelope_class(x, SR)
    # pitch movement inside loud stretches: semitone range p90-p10 of relaxed-YIN f0 within each event
    env = A.envelope(x, SR, 20)[(T * SR).astype(int)]
    loud = env > env.max() * 0.25
    arcs, cur = [], []
    for f, l in zip(F, loud):
        if l and np.isfinite(f):
            cur.append(f)
        elif cur:
            if len(cur) >= 4:
                st = 12 * np.log2(np.array(cur) / np.median(cur))
                arcs.append(np.percentile(st, 90) - np.percentile(st, 10))
            cur = []
    p = A.f0_estimate(x, SR, return_info=True)
    return {"pitch movement in a quack (semitones)": round(float(np.median(arcs)), 2) if arcs else None,
            "periodicity (1 = pure tone)": p["periodicity"], "centroid Hz": round(centroid(x)),
            "quack rate /s": ev["event_rate_hz"]}


def engine(x):
    seg = loudest(x, 0.5)
    stems, _ = A.hpr_split(seg, SR)
    tot = sum(float((v ** 2).sum()) for v in stems.values()) + 1e-20
    return {"high-rpm centroid Hz": round(centroid(seg)), "high-rpm share <300 Hz": round(share(seg, 20, 300), 3),
            "high-rpm noise share": round(float((stems["noise"] ** 2).sum()) / tot, 3)}


def foghorn(x):
    p = A.f0_estimate(x, SR, return_info=True)
    e = A.envelope(x, SR, 10.0)
    act = np.nonzero(e > e.max() * 0.1)[0]
    tail = e[act[0] + int(0.6 * (act[-1] - act[0])): act[-1]] if len(act) else e
    d = 20 * np.log10(tail / (tail.max() + 1e-12) + 1e-9)
    return {"f0 Hz": round(p["f0_hz"], 1) if p["f0_hz"] else None, "share <250 Hz": round(share(x, 20, 250), 3),
            "centroid Hz": round(centroid(x)), "choppiness (dB std, late part)": round(float(np.std(d)), 2)}


def explosion(x):
    e = A.envelope(x, SR, 2.0)
    i10, i90 = int(np.argmax(e > 0.1 * e.max())), int(np.argmax(e > 0.9 * e.max()))
    on = i10
    first = x[on:on + int(0.5 * SR)]
    h = A.envelope(band(x, 2000, 10000), SR, 20.0)
    hiss_t = float((h > h.max() * 0.1).sum() / SR)
    return {"rise ms": round((i90 - i10) / SR * 1000, 1), "share <200 Hz (first 0.5 s)": round(share(first, 20, 200), 3),
            "hiss time >2 kHz within 20 dB (s)": round(hiss_t, 2)}


def match(x):
    e = A.envelope(x, SR, 10.0)
    ip = int(np.argmax(e))
    late = e[ip + int(0.25 * SR): ip + int(0.6 * SR)]
    return {"centroid Hz": round(centroid(x)),
            "flame level re burst (dB)": round(float(20 * np.log10(np.median(late) / e.max() + 1e-9)), 1) if len(late) else None}


FN = {"duck": duck, "engine": engine, "foghorn": foghorn, "explosion": explosion, "match": match}


def main(argv):
    targets = json.loads(Path(argv[1]).read_text())
    ref_dir = Path(argv[2])
    versions = [a.split("=", 1) for a in argv[4:]]
    lines, res = [], {}
    for s, refs in targets.items():
        rows = []
        for rf, a, b in refs:
            x, sr = read_audio(ref_dir / rf)
            rows.append((f"ref {rf} [{a or 0}-{b or 'end'}]", FN[s](prepare(x, sr, a, b))))
        for v, d in versions:
            x, sr = read_audio(Path(d) / f"{s}.wav")
            rows.append((v, FN[s](prepare(x, sr))))
        keys = list(rows[0][1])
        lines += [f"### {s}", "", "| source | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        lines += [f"| {n} | " + " | ".join(str(r[k]) for k in keys) + " |" for n, r in rows]
        lines.append("")
        res[s] = rows
    Path(argv[3]).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv)
