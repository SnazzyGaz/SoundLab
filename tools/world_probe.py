"""WORLD analysis-resynthesis probe: how much of a vocal sound's realism is frame-wise detail?

Analyses a recording with WORLD (f0, spectral envelope, aperiodicity every 5 ms), resynthesises it,
then resynthesises it again with the envelope and aperiodicity smoothed over 50/200/800 ms and held
static, and scores every render against the input with soundlab.fit.compare_signals (lower is better).
Also writes one edit (down 4 semitones, formants x0.85) to show track editing.

Needs the optional dependency pyworld (WORLD: modified BSD; pyworld: MIT):
    pip install "soundlab[world]"

usage: python tools/world_probe.py INPUT.wav OUT_DIR
Write OUT_DIR outside the repository: rendered audio must never be committed.
See docs/research/creature-research.md, section 2.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from soundlab import fit as F
from soundlab.audio_io import read_audio, write_wav

FRAME_MS = 5.0


def _pyworld():
    try:
        import pyworld
    except ImportError:
        sys.exit("tools/world_probe.py needs pyworld: pip install \"soundlab[world]\"")
    return pyworld


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    pw = _pyworld()
    src, out = Path(argv[0]), Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    x, sr = read_audio(str(src))
    x = np.asarray(x, float)
    x = np.ascontiguousarray(x.mean(1) if x.ndim > 1 else x)
    res = {}

    def run(name, f0, sp, ap):
        y = pw.synthesize(*(np.ascontiguousarray(a) for a in (f0, sp, ap)), sr, FRAME_MS)[:len(x)]
        y = y / (np.max(np.abs(y)) + 1e-9) * np.max(np.abs(x))
        write_wav(str(out / f"{name}.wav"), y, sr)
        res[name] = {k: round(v, 3) for k, v in F.compare_signals(x, sr, y, sr).items() if isinstance(v, float)}

    def analyse(floor):
        f0, t = pw.harvest(x, sr, f0_floor=floor, f0_ceil=400, frame_period=FRAME_MS)
        return f0, pw.cheaptrick(x, f0, t, sr), pw.d4c(x, f0, t, sr)

    # a floor above a strong f0/2 subharmonic stops the tracker locking onto it
    for floor in (30, 60):
        f0, sp, ap = analyse(floor)
        res[f"_voiced_frac_floor{floor}"] = round(float((f0 > 0).mean()), 3)
        res[f"_median_f0_floor{floor}"] = round(float(np.median(f0[f0 > 0])), 1) if (f0 > 0).any() else 0.0
        run(f"world_floor{floor}", f0, sp, ap)

    f0, sp, ap = analyse(30)
    lsp = np.log(sp)
    for ms in (50, 200, 800):
        n = max(1, int(ms / FRAME_MS))
        run(f"smooth_{ms}ms", f0, np.exp(uniform_filter1d(lsp, n, axis=0)), np.clip(uniform_filter1d(ap, n, axis=0), 0, 1))
    loud = lsp.mean(1, keepdims=True)
    static = lsp.mean(0, keepdims=True) - lsp.mean()
    ap_static = np.repeat(ap.mean(0, keepdims=True), len(f0), 0)
    run("static_env_keep_loudness", f0, np.exp(static + loud), ap_static)
    run("static_f0_too", np.where(f0 > 0, np.median(f0[f0 > 0]), 0), np.exp(static + loud), ap_static)

    # edit: down 4 semitones, formants scaled by 0.85 (a larger animal)
    nb = sp.shape[1]
    idx = np.clip(np.arange(nb) * 0.85, 0, nb - 1)
    warp = lambda m: np.array([np.interp(idx, np.arange(nb), r) for r in m])
    y = pw.synthesize(np.ascontiguousarray(f0 * 2 ** (-4 / 12)), warp(sp), warp(ap), sr, FRAME_MS)
    write_wav(str(out / "edit_down4_formant085.wav"), y / (np.max(np.abs(y)) + 1e-9) * 0.9, sr)

    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
