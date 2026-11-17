"""Turn a blind-test pack (soundlab blindtest build) into a self-contained web page folder: index.html + clips/.

    python tools/blindtest_artifact.py <pack folder> <out folder> [pack number]

The page plays the clips in a random order per listener, takes "real or synthesised?" answers (keys R / S,
Space to play), saves them (the artifact's database when published with `db`, else the browser), and shows
the result by sound at the end, plus an answer code (SL1....) that `soundlab blindtest score --codes`
imports. The truth is in the page only lightly scrambled (not secret: a listener who reads the source can
find it); the clips have neutral names.
"""
from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("blindtest_artifact.html")


def mask(n: int) -> list[int]:
    """Same byte stream as the page's scramble() (32-bit LCG)."""
    s, out = 0x5EED, []
    for _ in range(n):
        s = (s * 1103515245 + 12345) & 0xFFFFFFFF
        out.append((s >> 16) & 0xFF)
    return out


LABELS = {"hc_bounce": "metal ball bouncing", "hc_knock": "knock on wood", "drumstick": "drumstick click",
          "metal_clang": "metal clang (anvil, iron bar)", "squeak_friction": "rubber shoe squeak",
          "creak_friction": "creaking wooden floor", "scrape_sdt": "blade scraped on stone", "foghorn": "foghorn",
          "wind_howl": "howling, whistling wind", "dog_growl": "dog growling"}


def build(pack: Path, out: Path, pack_no: int = 1) -> Path:
    key = json.loads((pack / "key.json").read_text())
    clips = sorted(key["clips"], key=lambda c: c["clip"])
    sounds = sorted({c["sound"] for c in clips})
    raw = [(sounds.index(c["sound"]) << 1) | (c["truth"] == "engine") for c in clips]
    m = mask(len(raw))
    data = {"pack": pack_no, "clips": [c["clip"] for c in clips], "sounds": sounds,
            # v0.25: the page names what each clip is meant to be (the sound is not secret, only real vs engine),
            # so a clip is judged against its target, not only for being plausible as some real recording
            "labels": [LABELS.get(c["sound"], c["sound"].replace("_", " ")) for c in clips],
            "k": base64.b64encode(bytes(b ^ x for b, x in zip(raw, m))).decode()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir(exist_ok=True)
    for c in clips:
        shutil.copyfile(pack / "clips" / f"{c['clip']}.wav", out / "clips" / f"{c['clip']}.wav")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*PACK_DATA*/null", json.dumps(data))
    (out / "index.html").write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    print(f"wrote {p / 'index.html'} and {len(list((p / 'clips').glob('*.wav')))} clips")
