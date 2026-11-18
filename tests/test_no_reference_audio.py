"""Rule: never ship reference clips or audio stems derived from them.

Reference recordings are used for tuning during development only and live
OUTSIDE the repository (soundlab-research/refs). This test fails if any audio file is tracked by git or sits
anywhere in the working tree, except paths listed in tests/audio_allowlist.txt (empty: nothing is allowed).
Analysis numbers of credited CC0 recordings are allowed (v0.28 creak_data.py, v0.29 squeal_stems.bin, v0.30 air_stems.bin, v0.31 liquid_stems.bin:
partial tracks and band envelopes at frame rate, see THIRD_PARTY.md); no file may hold sample data, whatever its name.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXT = {".wav", ".wave", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".aif", ".aiff", ".aifc",
             ".wma", ".caf", ".mid", ".midi", ".npy", ".npz"}
AUDIO_MAGIC = (b"RIFF", b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"fLaC", b"OggS", b"FORM", b"\x93NUMPY",
               b"MThd", b"caff", b"\x30\x26\xb2\x75")
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules"}


def _allowlist():
    p = ROOT / "tests" / "audio_allowlist.txt"
    if not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith("#")}


def _walk():
    for p in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            yield p.relative_to(ROOT).as_posix()


def test_no_audio_in_working_tree():
    bad = sorted(set(_walk()) - _allowlist())
    assert not bad, ("audio files in the repo tree (reference clips / stems must stay outside the repo; "
                     f"render outputs go to an external folder): {bad}")


def test_no_audio_tracked_by_git():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return  # not a git checkout (e.g. an sdist)
    bad = sorted(f for f in out.splitlines() if Path(f).suffix.lower() in AUDIO_EXT and f not in _allowlist())
    assert not bad, f"audio files tracked by git: {bad}"


def test_no_file_holds_audio_under_another_name():
    """Content check: no file in the package or tests starts like an audio container or a numpy array."""
    bad = []
    for d in ("soundlab", "tests", "recipes", "tools", "docs", "demos"):
        for p in (ROOT / d).rglob("*"):
            if not p.is_file() or any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
                continue
            with open(p, "rb") as f:
                head = f.read(16)
            if head.startswith(AUDIO_MAGIC) or head[8:12] in (b"WAVE", b"AIFF", b"AIFC"):
                bad.append(p.relative_to(ROOT).as_posix())
    assert not sorted(set(bad) - _allowlist()), bad


def test_stem_data_has_no_sample_rate_arrays():
    """squeal_stems.bin: every array is per frame (hop 128) or per band, never per sample."""
    import json, lzma, struct
    import numpy as np
    raw = (ROOT / "soundlab" / "models" / "squeal_stems.bin").read_bytes()
    p = lzma.decompress(raw[8:])
    hdr = json.loads(p[4:4 + struct.unpack("<I", p[:4])[0]])
    blob = memoryview(p)[4 + struct.unpack("<I", p[:4])[0]:]
    for k, b in hdr["bursts"].items():
        assert b["nfr"] <= (b["t1"] - b["t0"]) * hdr["sr"] / hdr["hop"] + 160, k     # 2.9 ms frames (+ margins)
        t = b["tracks"]
        ln = np.frombuffer(blob, t["length"]["dt"], t["length"]["sh"][0], t["length"]["o"])
        assert ln.max() <= b["nfr"] and t["dc"]["sh"][0] == t["db"]["sh"][0] == ln.sum(), k


def test_air_stem_data_has_no_sample_rate_arrays():
    """air_stems.bin (v0.30): band envelopes at 2.9 ms frames, onset envelopes at 1 ms, tracks per frame."""
    import json, lzma, struct
    raw = (ROOT / "soundlab" / "models" / "air_stems.bin").read_bytes()
    p = lzma.decompress(raw[8:])
    hdr = json.loads(p[4:4 + struct.unpack("<I", p[:4])[0]])
    for k, s in hdr["segments"].items():
        nfr = s["n"] / 128 + 16
        for name, a in s.items():
            if isinstance(a, dict) and "sh" in a:
                assert max(a["sh"]) <= s["n"] / 40, (k, name)      # never one value per sample
        assert s["E"]["sh"][1] <= nfr and s["fine"]["sh"][1] <= s["n"] / 44 + 2, k
        assert s["t_f"]["sh"][0] <= s["t_len"]["sh"][0] * nfr, k


def test_liquid_stem_data_has_no_sample_rate_arrays():
    """liquid_stems.bin (v0.31): band envelopes at 2 ms (or 2.9 ms STFT frames), onset envelopes at 1 ms, bubbles as
    chirp settings with one amplitude per 1.45 ms frame, tracks per frame."""
    import json, lzma, struct
    raw = (ROOT / "soundlab" / "models" / "liquid_stems.bin").read_bytes()
    p = lzma.decompress(raw[8:])
    hdr = json.loads(p[4:4 + struct.unpack("<I", p[:4])[0]])
    for k, s in hdr["sources"].items():
        for name, a in s.items():
            if isinstance(a, dict) and "sh" in a:
                assert max(a["sh"]) <= s["n"] / 40, (k, name)      # never one value per sample
        if s["engine"] == "env":
            assert s["env"]["sh"][1] <= s["n"] / 88 + 2, k
            assert s["c_db"]["sh"][0] <= max(1, s["c_len"]["sh"][0]) * (s["n"] / 64 + 16), k
        else:
            assert s["E"]["sh"][1] <= s["n"] / 128 + 16 and s["fine"]["sh"][1] <= s["n"] / 44 + 2, k
