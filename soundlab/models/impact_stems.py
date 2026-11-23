"""impact_stems (v0.34): Impacts and breaking rebuilt from stems of CC0 recordings (numbers only, no audio): knocks, a dowel, drumsticks, a wood drop, a bouncing ball, a metal clang, a glass clink, pickaxe hits, a coin drop and jingle, a body punch, a ceramic crack and a glass break.

See docs/impact_stems.md. Numbers only, no audio; sources are credited in THIRD_PARTY.md.
"""
from __future__ import annotations

import functools
from pathlib import Path

from ..core import Ctx, Param, RecipeError, model
from . import recstems as R

DATA = Path(__file__).with_name("impact_stems.bin")
MAGIC = b'SLIMP001'
SOURCES = ("ball_bounce", "body_punch", "ceramic_break", "coin_drop", "coin_jingle", "dowel", "drumstick", "glass_break", "glass_clink", "metal_clang", "pickaxe_earth", "pickaxe_rock", "wood_drop", "wood_knock")


@functools.lru_cache(maxsize=1)
def _data():
    try:
        return R._load_bin(DATA, MAGIC)
    except ValueError as e:
        raise RecipeError(str(e)) from e


def sources() -> list:
    return list(SOURCES)


def render_impact(p: dict) -> "object":
    import numpy as np
    p = dict(p)
    src = p.get("source", "wood_knock")
    data = _data()
    if src not in data["sources"]:
        raise RecipeError(f"unknown source {src!r}")
    return R.render(R.get_source(data, src), p)


@model("impact_stems", "sound", """Impact and breaking sounds rebuilt from stems of CC0 recordings. Body-mode tracks, real hit events as band envelopes, a per-hit noise bed. Fresh takes remix the real stems (never averaged). Speed is tape speed. force, soft and distance are causal.""",
       [
           Param("source", "wood_knock", "", None, None, "stored source", choices=["ball_bounce", "body_punch", "ceramic_break", "coin_drop", "coin_jingle", "dowel", "drumstick", "glass_break", "glass_clink", "metal_clang", "pickaxe_earth", "pickaxe_rock", "wood_drop", "wood_knock"]),
           Param("count", 1, "", 1, 8, "number of real gestures (steps, hits or landings) in this take"),
           Param("tempo", 1.0, "", 0.4, 2.5, "spacing of a short count (1 = the recording's own gaps)"),
           Param("which", -1, "", -1, 64, "which stored gesture to play (0-based; -1 = pick a typical one)"),
           Param("early", 0, "", 0, 1, "prefer the earlier gestures of the recording (1 = yes)"),
           Param("force", 0.0, "", -3.0, 3.0, "event / body level and a 3 kHz shelf; 0 = as recorded"),
           Param("soft", 0.0, "", 0.0, 1.2, "softer attack: event-layer low-pass and a 120 Hz low shelf (heel / thump)"),
           Param("speed_cents", 0.0, "cents", -1200, 1200, "tape speed (pitch and time together)"),
           Param("distance", 1.0, "", 0.25, 4.0, "1 = as recorded; further is quieter and duller; nearer has more low end"),
           Param("hs_hz", 3000.0, "Hz", 200, 16000, "high shelf corner used with force"),
           Param("legacy_choice", "", "", None, None, "accepted so older choice macros still parse; unused"),
           Param("seed", 0, "", 0, 2**31 - 1, "noise seed (0 with force 0 and soft 0 rebuilds the stored take)"),
           Param("variation", 0, "", -1e6, 1e6, "0 = the stored take; others reseed a fresh remix of the same stems"),
           Param("at", 0.0, "s", 0, None),
       ],
       ["McDermott & Simoncelli 2011 (sound texture statistics)",
        "McAulay & Quatieri 1986 (sinusoidal modelling)",
        "Freesound CC0 recordings (numbers only; THIRD_PARTY.md)"],
       min_s=lambda p: 0.08)
def impact_stems(ctx: Ctx, x, **p):
    return R.render_into_ctx(ctx, p, _data(), "wood_knock")


def take_key(p: dict):
    return R.take_key(_data(), p, "wood_knock")
