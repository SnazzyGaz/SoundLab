"""Rule: never ship reference clips or stems derived from them.

Reference recordings are used for tuning during development only and live
OUTSIDE the repository (soundlab-research/refs). This test fails if any audio file is tracked by git or sits
anywhere in the working tree, except paths listed in tests/audio_allowlist.txt (empty: nothing is allowed).
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXT = {".wav", ".wave", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".aif", ".aiff", ".aifc",
             ".wma", ".caf", ".mid", ".midi", ".npy", ".npz"}
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
