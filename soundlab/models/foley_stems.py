"""foley_stems: Everyday handling sounds (cloth, zips, keys, paper, cups, glasses, bottles, cutlery, bags, coins, phones, chairs, pens, switches) rebuilt from stems of CC0 recordings (numbers only, no audio). One gesture by default; force,
softness, size (tape speed) and distance are optional macros.

See docs/foley_stems.md. Numbers only, no audio; sources are credited in THIRD_PARTY.md.
"""
from __future__ import annotations

import functools
from pathlib import Path

from ..core import Ctx, Param, RecipeError, model
from . import recstems as R

DATA = Path(__file__).with_name("foley_stems.bin")
MAGIC = b'SLFOL001'
SOURCES = ('cloth_handling', 'cloth_heavy', 'zipper_bag', 'zipper_jacket', 'keys_jingle', 'keys_down', 'paper_rustle', 'paper_crumple', 'page_turn', 'book_close', 'cup_down', 'glass_down', 'bottle_down', 'cutlery', 'bag_down', 'backpack_handle', 'cupboard', 'coin_drop', 'phone_down', 'chair_scrape', 'pen_click', 'light_switch', 'plastic_bag')


@functools.lru_cache(maxsize=1)
def _data():
    try:
        return R._load_bin(DATA, MAGIC)
    except ValueError as e:
        raise RecipeError(str(e)) from e


def sources() -> list:
    return list(SOURCES)


@model("foley_stems", "sound", """Everyday handling sounds rebuilt from stems of CC0 recordings. Real events as band envelopes, a per-gesture noise bed, body-mode tracks where the object rings. Fresh takes remix the real stems (never averaged). Speed is tape speed. force, soft and distance are causal.""",
       [
           Param("source", "cloth_handling", "", None, None, "stored source", choices=list(SOURCES)),
           Param("count", 1, "", 1, 8, "number of real gestures in this take"),
           Param("tempo", 1.0, "", 0.4, 2.5, "spacing of a short count (1 = the recording's own gaps)"),
           Param("which", -1, "", -1, 64, "which stored gesture to play (0-based; -1 = pick a typical one)"),
           Param("early", 0, "", 0, 1, "prefer the earlier gestures of the recording (1 = yes)"),
           Param("force", 0.0, "", -3.0, 3.0, "event / body level and a 3 kHz shelf; 0 = as recorded"),
           Param("soft", 0.0, "", 0.0, 1.2, "softer attack: event-layer low-pass and a 120 Hz low shelf"),
           Param("speed_cents", 0.0, "cents", -1200, 1200, "tape speed (pitch and time together)"),
           Param("distance", 1.0, "", 0.25, 4.0, "1 = as recorded; further is quieter and duller; nearer has more low end"),
           Param("hs_hz", 3000.0, "Hz", 200, 16000, "high shelf corner used with force"),
           Param("seed", 0, "", 0, 2**31 - 1, "noise seed (0 with force 0 and soft 0 rebuilds the stored take)"),
           Param("variation", 0, "", -1e6, 1e6, "0 = the stored take; others reseed a fresh remix of the same stems"),
           Param("at", 0.0, "s", 0, None),
       ],
       ["McDermott & Simoncelli 2011 (sound texture statistics)",
        "McAulay & Quatieri 1986 (sinusoidal modelling)",
        "Freesound CC0 recordings (numbers only; THIRD_PARTY.md)"],
       min_s=lambda p: 0.08)
def foley_stems(ctx: Ctx, x, **p):
    return R.render_into_ctx(ctx, p, _data(), "cloth_handling")
