"""weapon_stems (v0.34): Weapons and combat rebuilt from stems of CC0 recordings (numbers only, no audio): shots, pumps, reloads, blade work, explosions and related one-shots.

See docs/weapon_stems.md. Numbers only, no audio; sources are credited in THIRD_PARTY.md.
"""
from __future__ import annotations

import functools
from pathlib import Path

from ..core import Ctx, Param, RecipeError, model
from . import recstems as R

DATA = Path(__file__).with_name("weapon_stems.bin")
MAGIC = b'SLWEP001'
SOURCES = ("arrow_impact", "assault_rifle_burst", "assault_rifle_shot", "axe_chop", "blade_slide", "bolt_cycle", "bow_release", "brass_casings", "charging_handle", "club_hit", "crossbow", "distant_gunfire", "dry_fire", "explosion_close", "explosion_distant", "firework_bang", "hammer_cock", "knife_draw", "knife_flick", "knife_slash", "knife_stab", "magazine_change", "pistol_reload", "pistol_shot", "revolver_shot", "ricochet", "rifle_reload", "rifle_shot", "safety_click", "shield_block_metal", "shotgun_pump", "shotgun_shot", "slide_rack", "smg_burst", "steel_target_ping", "suppressed_shot", "sword_clash", "sword_draw", "sword_sheathe", "tree_break")


@functools.lru_cache(maxsize=1)
def _data():
    try:
        return R._load_bin(DATA, MAGIC)
    except ValueError as e:
        raise RecipeError(str(e)) from e


def sources() -> list:
    return list(SOURCES)


def render_weapon(p: dict) -> "object":
    import numpy as np
    p = dict(p)
    src = p.get("source", "pistol_shot")
    data = _data()
    if src not in data["sources"]:
        raise RecipeError(f"unknown source {src!r}")
    return R.render(R.get_source(data, src), p)


@model("weapon_stems", "sound", """Weapon and combat sounds rebuilt from stems of CC0 recordings. Body-mode tracks, real events as band envelopes, a per-hit noise bed. Fresh takes remix the real stems (never averaged). Speed is tape speed. force, soft and distance are causal.""",
       [
           Param("source", "pistol_shot", "", None, None, "stored source", choices=["arrow_impact", "assault_rifle_burst", "assault_rifle_shot", "axe_chop", "blade_slide", "bolt_cycle", "bow_release", "brass_casings", "charging_handle", "club_hit", "crossbow", "distant_gunfire", "dry_fire", "explosion_close", "explosion_distant", "firework_bang", "hammer_cock", "knife_draw", "knife_flick", "knife_slash", "knife_stab", "magazine_change", "pistol_reload", "pistol_shot", "revolver_shot", "ricochet", "rifle_reload", "rifle_shot", "safety_click", "shield_block_metal", "shotgun_pump", "shotgun_shot", "slide_rack", "smg_burst", "steel_target_ping", "suppressed_shot", "sword_clash", "sword_draw", "sword_sheathe", "tree_break"]),
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
def weapon_stems(ctx: Ctx, x, **p):
    return R.render_into_ctx(ctx, p, _data(), "pistol_shot")


def take_key(p: dict):
    return R.take_key(_data(), p, "pistol_shot")
