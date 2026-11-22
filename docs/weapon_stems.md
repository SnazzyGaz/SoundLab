# weapon_stems

Weapons and combat rebuilt from stems of CC0 recordings (v0.34). The package holds analysis numbers only: body-mode partial tracks, the 1/8-octave band envelopes of each recorded event, a 1/8-octave 1 ms sub-band noise bed, a 1/6-octave colour curve and a static treble ceiling. No samples of the recordings are stored. Sources are listed in `THIRD_PARTY.md`.

Recipes play a single real gesture or a short natural count (a burst, a reload, two pumps). Macros are the same as `impact_stems`: `count`, `tempo`, `force`, `soft`, `size` (tape speed), `distance`, `seed` / `variation`. Stems are remixed, never averaged.

## Recipes

Shots: `pistol_shot`, `revolver_shot`, `rifle_shot`, `assault_rifle_shot`, `assault_rifle_burst`, `shotgun_shot`, `suppressed_shot`, `distant_gunfire`, `dry_fire`, `smg_burst`.

Actions: `shotgun_pump`, `slide_rack`, `charging_handle`, `bolt_cycle`, `safety_click`, `hammer_cock`, `brass_casings`, `magazine_change`, `pistol_reload`, `rifle_reload`.

Blades and melee: `sword_clash`, `blade_slide`, `sword_draw`, `sword_sheathe`, `knife_draw`, `knife_flick`, `knife_slash`, `knife_stab`, `axe_chop`, `club_hit`, `shield_block_metal`.

Bows and others: `bow_release`, `arrow_impact`, `crossbow`, `ricochet`, `steel_target_ping`, `firework_bang`, `explosion_close`, `explosion_distant`, `tree_break`.

`sword_clash` now uses `weapon_stems`. Its older macro names (object, intensity, scrape, ring, distance) still parse; intensity maps lightly onto force, and unused names are accepted as no-ops. Other procedural recipes (`gunshot`, `blade_clash`, `explosion`, and the rest of the fitted combat set) are unchanged.

The previous procedural `sword_clash` recipe stays in `recipes/impact_model/` as a fallback.
