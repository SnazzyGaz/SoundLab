# foot_stems

Footsteps rebuilt from stems of CC0 recordings (v0.34). The package holds analysis numbers only: body-mode partial tracks, the 1/8-octave band envelopes of each recorded step, a 1/8-octave 1 ms sub-band noise bed, a 1/6-octave colour curve and a static treble ceiling. No samples of the recordings are stored. Sources are listed in `THIRD_PARTY.md`.

Recipes play a single real step or landing by default, and default to a softer, more natural step than the often-forceful recordings: lower event gain, a causal low-pass on the attack, less heel weight (a 120 Hz low shelf) and a gentler 3 kHz shelf. `force` is an optional macro (higher is heavier). Carpet and leaves default softer still; trainers prefer the first steps of that recording; `jump_land_rock` is one landing at a time.

`count` plays a short walk or run of real steps; `tempo` changes their spacing; `size` is tape speed; `distance` is quieter and duller when greater than 1; `seed` / `variation` remixes the same real stems (never averaged).

## Recipes

Surfaces: `concrete_walk`, `concrete_run`, `stone_walk`, `wood_floor`, `wood_creaky`, `wood_stairs`, `gravel_walk`, `dirt_walk`, `grass_walk`, `grass_run`, `sand_walk`, `snow_walk`, `mud_walk`, `puddle_walk`, `metal_walk`, `metal_stairs`, `tile_walk`, `carpet_walk`, `leaves_walk`.

Shoes: `boots_walk`, `heels_walk`, `trainers_walk`, `barefoot_wood`.

Landings: `jump_land`, `jump_land_rock`.

In the app's library these are one **Footsteps** entry (the `footstep` recipe) with Surface, Shoe and Pace
dropdowns; each surface or shoe name plays the same sound as its recipe above (same soft default and length),
and Pace run uses `concrete_run` / `grass_run` on concrete and grass. The single-surface names stay as hidden
aliases and render as before. Stairs and the two landings stay separate entries.

The older `footstep`, `footstep_wood`, `footstep_dirt`, `footstep_grass`, `footstep_gravel`, `footstep_metal` and `footstep_stone` recipe names still work and now use `foot_stems` (soft defaults, the same macros plus the previous names, which still parse). The generic `footstep` recipe also has `surface` and `shoe` macros that select among the stored sources. New work should use the dedicated names above.

The previous procedural `footstep` / `footstep_*` recipes stay in `recipes/impact_model/` as a fallback, the same way `recipes/creak_model/` keeps the older creak recipes.
