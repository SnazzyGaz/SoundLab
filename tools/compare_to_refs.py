"""Side-by-side distance of rendered versions to reference recordings.

usage: python tools/compare_to_refs.py TARGETS.json REF_DIR OUT.md LABEL=DIR [LABEL=DIR ...]

TARGETS.json maps a sound name to a list of [ref_file, start_s|null, end_s|null]. Each DIR must contain
<sound>.wav. Writes a markdown table (and OUT.json) with spectral/stft/envelope/texture/pitch/total per
reference and the mean per version. Lower is better; ~0 = indistinguishable by these measures.
"""
import json
import sys
from pathlib import Path

import numpy as np

from soundlab.audio_io import read_audio
from soundlab.fit import DEFAULT_WEIGHTS, compare_signals


def main(argv):
    targets = json.loads(Path(argv[1]).read_text())
    ref_dir = Path(argv[2])
    out = Path(argv[3])
    versions = [a.split("=", 1) for a in argv[4:]]
    parts = list(DEFAULT_WEIGHTS) + ["total"]
    res = {}
    lines = ["| sound | reference (crop) | " + " | ".join(f"{v} total" for v, _ in versions) + " |",
             "|---|---|" + "---|" * len(versions)]
    for sound, refs in targets.items():
        cands = {v: read_audio(Path(d) / f"{sound}.wav") for v, d in versions}
        res[sound] = {}
        for rf, a, b in refs:
            x, sr = read_audio(ref_dir / rf)
            key = f"{rf} [{a or 0}-{b or 'end'}]"
            res[sound][key] = {v: compare_signals(x, sr, y, ysr, a, b) for v, (y, ysr) in cands.items()}
            lines.append(f"| {sound} | {key} | " + " | ".join(f"{res[sound][key][v]['total']:.3f}" for v, _ in versions) + " |")
        means = {v: {p: float(np.mean([res[sound][k][v][p] for k in res[sound]])) for p in parts} for v, _ in versions}
        res[sound]["_mean"] = means
        lines.append(f"| **{sound}** | **mean** | " + " | ".join(f"**{means[v]['total']:.3f}**" for v, _ in versions) + " |")
    lines += ["", "Mean per part (averaged over the references of each sound):", "",
              "| sound | version | " + " | ".join(parts) + " |", "|---|---|" + "---|" * len(parts)]
    for sound in targets:
        for v, _ in versions:
            m = res[sound]["_mean"][v]
            lines.append(f"| {sound} | {v} | " + " | ".join(f"{m[p]:.3f}" for p in parts) + " |")
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv)
