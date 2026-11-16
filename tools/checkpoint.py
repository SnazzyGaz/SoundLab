#!/usr/bin/env python3
"""A/B blind checkpoint sets (wrapper; the code lives in soundlab/checkpoint.py). See docs/checkpoint.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soundlab.checkpoint import *  # noqa: F401,F403,E402
from soundlab.checkpoint import main  # noqa: E402

if __name__ == "__main__":
    main()
