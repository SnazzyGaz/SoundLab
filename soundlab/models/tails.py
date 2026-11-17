"""How long each processing block keeps sounding after its input stops (v0.22).

`soundlab.render(..., ring_out=True)` (used by every export) lets sources stop at the recipe's `duration` and
then runs the processing blocks this much longer, so a room's reverb, a bar's ring or an outdoor echo is not
cut off at the recipe length. Export then trims the tail at -60 dB as before, so an over-estimate only costs
render time; an under-estimate cuts the tail. Each estimate follows the block's own physics or code.
"""
from __future__ import annotations

import math

from ..core import REGISTRY, SPEED_OF_SOUND
from .room import room_geometry
from .synth import _COMBS

T60_PER_DB = 60.0  # dB


def _feedback_t60(loop_s: float, gain: float) -> float:
    """T60 of a feedback loop: round trip `loop_s`, gain per trip `gain` (60 dB / dB lost per trip)."""
    gain = abs(gain)
    if gain <= 0:
        return loop_s
    if gain >= 1:
        return 30.0
    return loop_s * T60_PER_DB / (-20 * math.log10(gain))


def _room(p):
    _, T = room_geometry(float(p["volume"]), float(p["absorption"]), float(p["rt60"]))
    return 1.2 * min(T, 10.0) + 0.1           # late IR is min(1.2 T, 12 s) long, plus early reflections


def _outdoor(p):
    echoes = [2.0 * float(d) / SPEED_OF_SOUND + 0.03 + 0.1 * min(float(d) / 1000.0, 1.0)
              for d, _ in (p["reflectors"] or [])]
    tail = 1.2 * float(p["tail_s"]) if float(p["tail_s"]) > 0 and float(p["tail_level"]) > 0 else 0.0
    return max(echoes + [0.0]) + tail


def _modal(p):
    # upper bound: the frequency-independent loss alone (material loss only shortens it), capped by max_t60
    return min(float(p["max_t60"]), 6.91 / max(float(p["extra_damping"]), 1e-3))


def _reverb(p):
    fb = 0.7 + 0.28 * float(p["size"])
    return float(p["predelay_ms"]) * 1e-3 + _feedback_t60(max(_COMBS) / 44100.0, fb)


def _conv_reverb(p):
    return float(p["predelay_ms"]) * 1e-3 + min(1.5 * float(p["rt60"]) * max(1.0, float(p["lf_ratio"])), 12.0)


def _delay(p):
    t = float(p["time_ms"]) * 1e-3
    return t + _feedback_t60(t, float(p["feedback"]))


TAILS = {
    "room": _room,
    "outdoor": _outdoor,
    "modal": _modal,
    "string": lambda p: float(p["t60"]),
    "tube": lambda p: _feedback_t60(2 * float(p["length"]) / SPEED_OF_SOUND, float(p["loss"])),
    "horn": lambda p: _feedback_t60(2 * float(p["length"]) / SPEED_OF_SOUND, float(p["resonance"])),
    "helmholtz": lambda p: 6.91 * float(p["q"]) / (math.pi * SPEED_OF_SOUND / (2 * math.pi) * math.sqrt(
        float(p["neck_area"]) / (float(p["volume"]) * float(p["neck_length"])))),
    "formant": lambda p: 0.05,
    "filter": lambda p: 0.05,
    "svf": lambda p: 0.05,
    "distance": lambda p: float(p["distance"]) / SPEED_OF_SOUND if p["delay"] else 0.0,
    "ground": lambda p: 0.02,
    "chorus": lambda p: (float(p["base_ms"]) + float(p["depth_ms"])) * 1e-3,
    "reverb": _reverb,
    "conv_reverb": _conv_reverb,
    "delay": _delay,
}

for _name, _fn in TAILS.items():
    if _name in REGISTRY:
        REGISTRY[_name].tail_s = _fn
