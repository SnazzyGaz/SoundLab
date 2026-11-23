"""door_stems: Doors, locks, knocks, gates, drawers and a garage door rebuilt from stems of CC0 recordings (numbers only, no audio). One gesture by default; force,
softness, size (tape speed) and distance are optional macros.

See docs/door_stems.md. Numbers only, no audio; sources are credited in THIRD_PARTY.md.
"""
from __future__ import annotations

import functools
from pathlib import Path

from ..core import Ctx, Param, RecipeError, model
from . import recstems as R

DATA = Path(__file__).with_name("door_stems.bin")
MAGIC = b'SLDOR001'
SOURCES = ('door_wood_open', 'door_wood_close', 'door_wood_slam', 'door_creak', 'door_creak_small', 'door_heavy', 'door_metal_close', 'door_metal_slam', 'door_handle', 'key_lock', 'deadbolt', 'knock_wood_muffled', 'knock_wood', 'knock_metal', 'door_sliding', 'door_glass', 'door_screen', 'car_door', 'cupboard_door', 'wardrobe_door', 'gate_wood', 'gate_metal', 'drawer', 'garage_door')


@functools.lru_cache(maxsize=1)
def _data():
    try:
        return R._load_bin(DATA, MAGIC)
    except ValueError as e:
        raise RecipeError(str(e)) from e


def sources() -> list:
    return list(SOURCES)


@model("door_stems", "sound", """Doors, locks, knocks and gates rebuilt from stems of CC0 recordings. Real events as band envelopes, a per-gesture noise bed, body-mode tracks for metal rings and a harmonic hinge-squeak voice (as mode tracks). Fresh takes remix the real stems (never averaged). Speed is tape speed. force, soft and distance are causal.""",
       [
           Param("source", "door_wood_close", "", None, None, "stored source", choices=list(SOURCES)),
           Param("count", 1, "", 1, 8, "number of real gestures in this take"),
           Param("tempo", 1.0, "", 0.4, 2.5, "spacing of a short count (1 = the recording's own gaps)"),
           Param("which", -1, "", -1, 64, "which stored gesture to play (0-based; -1 = pick a typical one)"),
           Param("early", 0, "", 0, 1, "prefer the earlier gestures of the recording (1 = yes)"),
           Param("hit", -1, "", -2, 16, "one hit of a multi-hit gesture such as a knock (0-based; -1 = the whole gesture; -2 = a hit picked by the variation)"),
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
def door_stems(ctx: Ctx, x, **p):
    return R.render_into_ctx(ctx, p, _data(), "door_wood_close")


def take_key(p: dict):
    return R.take_key(_data(), p, "door_wood_close")
