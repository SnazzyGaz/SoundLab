#!/usr/bin/env bash
# Render the five reference recipes to WAV + MP3 and analyse them.
set -euo pipefail
OUT=${1:-/workspace/soundlab-out}
cd "$(dirname "$0")"
for r in duck engine foghorn explosion match; do
  .venv/bin/soundlab render recipes/$r.json -o "$OUT/$r.wav" --mp3 >/dev/null
  mkdir -p "$OUT/analysis"
  .venv/bin/soundlab analyze "$OUT/$r.wav" -o "$OUT/analysis/$r" > "$OUT/analysis/$r.summary.json"
  echo "$r done"
done
