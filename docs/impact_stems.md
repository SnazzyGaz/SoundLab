# impact_stems

Impacts and breaking rebuilt from stems of CC0 recordings (v0.34). The package holds analysis numbers only: body-mode partial tracks, the 1/8-octave band envelopes of each recorded hit, a 1/8-octave 1 ms sub-band noise bed between hits, a 1/6-octave colour curve and a static treble ceiling. No samples of the recordings are stored; renders use new noise and the stored envelopes. Sources are listed in `THIRD_PARTY.md`.

Recipes play a single real gesture by default (one knock, one drop, one clink). `count` plays a short natural number of real hits; `tempo` changes their spacing. `force` raises or lowers the event, bed and body layers and a 3 kHz shelf; `soft` applies a causal low-pass to the event layer and a 120 Hz low shelf; `size` is tape speed (pitch and time together); `distance` is quieter and duller when greater than 1; `seed` / `variation` remixes the same real stems (never averaged). Nothing is band-pitch-shifted, frozen, time-squeezed or smoothed in a way that adds hiss.

## Recipes

`wood_knock`, `drumstick`, `dowel`, `wood_drop`, `ball_bounce`, `metal_clang`, `glass_clink`, `pickaxe_earth`, `pickaxe_rock`, `coin_drop`, `coin_jingle`, `body_punch`, `ceramic_break`, `glass_break`.

Recipes that previously used a procedural impact or hit model under the same filename (`ceramic_break`, `coin_jingle`, `dowel`, `drumstick`, `glass_break`, `glass_clink`, `metal_clang`, `pickaxe_earth`, `pickaxe_rock`) now use `impact_stems`. The older macro names still parse: hardness, muffle and intensity map lightly onto force and softness; material and other unused names are accepted as no-ops. Other procedural recipes (`hit`, `knock`, `impact_any`, and the rest of the fitted impact set) are unchanged.

The previous procedural recipes under these filenames (`ceramic_break`, `coin_jingle`, `dowel`, `drumstick`, `glass_break`, `glass_clink`, `metal_clang`, `pickaxe_earth`, `pickaxe_rock`) stay in `recipes/impact_model/` as a fallback, the same way `recipes/creak_model/` keeps the older creak recipes.
