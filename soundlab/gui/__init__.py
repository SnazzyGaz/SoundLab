"""SoundLab slider app: a small local web app (Python standard library only).

    soundlab gui                     # opens http://127.0.0.1:8766/
    soundlab gui --port 9000 --out D:/SoundLabOut --no-browser --pack D:/SoundLabOut/blindtest

Pages: `/` (recipes, macro sliders, advanced controls, play, waveform/spectrogram, export, batch, save) and
`/blindtest` (blind listening test: one clip at a time, 'real or synthesised?', answers saved per listener).
The server binds to 127.0.0.1 only (nothing is reachable from other machines).
"""
from __future__ import annotations

from .server import DEFAULT_PORT, make_server, serve  # noqa: F401
