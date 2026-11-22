"""foot_stems (v0.34): Footsteps rebuilt from stems of CC0 recordings (numbers only, no audio): walks and runs on many surfaces, boots, heels, trainers, barefoot on wood, and jump landings. Recipes default to a soft, natural step; force is an optional macro.

See docs/foot_stems.md. Numbers only, no audio; sources are credited in THIRD_PARTY.md.
"""
from __future__ import annotations

import functools
from pathlib import Path

from ..core import Ctx, Param, RecipeError, model
from . import recstems as R

DATA = Path(__file__).with_name("foot_stems.bin")
MAGIC = b'SLFTS001'
SURFACE = {
    "wood_hard": "wood_floor", "wood": "wood_floor", "stone": "stone_walk", "steel": "metal_walk",
    "metal": "metal_walk", "dirt": "dirt_walk", "gravel": "gravel_walk", "grass": "grass_walk",
    "leaves": "leaves_walk", "concrete": "concrete_walk", "sand": "sand_walk", "snow": "snow_walk",
    "mud": "mud_walk", "tile": "tile_walk", "carpet": "carpet_walk",
}
SHOE = {"trainer": "trainers_walk", "trainers": "trainers_walk", "boot": "boots_walk",
        "boots": "boots_walk", "heel": "heels_walk", "heels": "heels_walk",
        "barefoot": "barefoot_wood"}

SOURCES = ("concrete_walk", "concrete_run", "stone_walk", "wood_floor", "wood_creaky", "wood_stairs", "gravel_walk", "dirt_walk", "grass_walk", "grass_run", "sand_walk", "snow_walk", "mud_walk", "puddle_walk", "metal_walk", "metal_stairs", "tile_walk", "carpet_walk", "leaves_walk", "boots_walk", "heels_walk", "trainers_walk", "barefoot_wood", "jump_land", "jump_land_rock")


@functools.lru_cache(maxsize=1)
def _data():
    try:
        return R._load_bin(DATA, MAGIC)
    except ValueError as e:
        raise RecipeError(str(e)) from e


def sources() -> list:
    return list(SOURCES)


def render_foot(p: dict) -> "object":
    import numpy as np
    p = dict(p)
    src = p.get("source", "wood_floor")
    data = _data()
    if src not in data["sources"]:
        raise RecipeError(f"unknown source {src!r}")
    return R.render(R.get_source(data, src), p)


@model("foot_stems", "sound", """Footsteps rebuilt from stems of CC0 recordings. Body-mode tracks, real step events as band envelopes, a per-step noise bed. Recipes default to a softer step than the (often forceful) recording. Fresh takes remix the real stems (never averaged). Speed is tape speed. force, soft and distance are causal.""",
       [
           Param("source", "wood_floor", "", None, None, "stored source", choices=["concrete_walk", "concrete_run", "stone_walk", "wood_floor", "wood_creaky", "wood_stairs", "gravel_walk", "dirt_walk", "grass_walk", "grass_run", "sand_walk", "snow_walk", "mud_walk", "puddle_walk", "metal_walk", "metal_stairs", "tile_walk", "carpet_walk", "leaves_walk", "boots_walk", "heels_walk", "trainers_walk", "barefoot_wood", "jump_land", "jump_land_rock"]),
           Param("count", 1, "", 1, 8, "number of real gestures (steps, hits or landings) in this take"),
           Param("tempo", 1.0, "", 0.4, 2.5, "spacing of a short count (1 = the recording's own gaps)"),
           Param("which", -1, "", -1, 64, "which stored gesture to play (0-based; -1 = pick a typical one)"),
           Param("early", 0, "", 0, 1, "prefer the earlier gestures of the recording (1 = yes)"),
           Param("force", 0.0, "", -3.0, 3.0, "event / body level and a 3 kHz shelf; 0 = as recorded"),
           Param("soft", 0.0, "", 0.0, 1.2, "softer attack: event-layer low-pass and a 120 Hz low shelf (heel / thump)"),
           Param("speed_cents", 0.0, "cents", -1200, 1200, "tape speed (pitch and time together)"),
           Param("distance", 1.0, "", 0.25, 4.0, "1 = as recorded; further is quieter and duller; nearer has more low end"),
           Param("hs_hz", 3000.0, "Hz", 200, 16000, "high shelf corner used with force"),
           Param("seed", 0, "", 0, 2**31 - 1, "noise seed (0 with force 0 and soft 0 rebuilds the stored take)"),
           Param("variation", 0, "", -1e6, 1e6, "0 = the stored take; others reseed a fresh remix of the same stems"),
           Param("surface", "", "", None, None, "optional surface name for the generic footstep recipe (wood, stone, dirt, ...)"),
           Param("shoe", "", "", None, None, "optional shoe name for the generic footstep recipe (trainer, boot, heel, barefoot)"),
           Param("at", 0.0, "s", 0, None),
       ],
       ["McDermott & Simoncelli 2011 (sound texture statistics)",
        "McAulay & Quatieri 1986 (sinusoidal modelling)",
        "Freesound CC0 recordings (numbers only; THIRD_PARTY.md)"],
       min_s=lambda p: 0.08)
def foot_stems(ctx: Ctx, x, **p):
    p = dict(p)
    shoe = str(p.pop("shoe", "") or "").strip().lower()
    surface = str(p.pop("surface", "") or "").strip().lower()
    if shoe in SHOE:
        p["source"] = SHOE[shoe]
        if p["source"] == "trainers_walk":
            p.setdefault("early", 1)
    elif surface in SURFACE:
        p["source"] = SURFACE[surface]
    return R.render_into_ctx(ctx, p, _data(), "wood_floor")
