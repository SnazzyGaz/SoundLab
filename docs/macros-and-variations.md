# Macros, variations, batch export and audition (0.7.0; 0.8.0 notes; 0.9.0 library)

Roadmap steps 2 and 3. **Macros are the primary interface**: every recipe declares a few sliders (0-1) or
named choices, and each one moves several underlying controls with a sensible curve. The physical controls
stay underneath as the reference and for fitting.

```bash
soundlab describe recipes/hit.json                        # macros first, then layers
soundlab render recipes/hit.json -m material=metal -m size=0.8 -o metal_bar.wav
soundlab render recipes/footstep.json --variations 5 -o step.wav   # step_v01.wav .. step_v05.wav
soundlab batch recipes/blocks recipes/hit.json -o out/ --variations 5 --mp3 --zip
soundlab batch recipes/explosion.json -o out/ --sweep size=0,0.5,1
soundlab audition recipes/blocks/coin_pickup.json -m pitch=0.7 --watch   # re-renders + plays on save
```

```python
y, sr = soundlab.render("recipes/hit.json", macros={"material": "glass", "intensity": 0.8}, variation=2)
soundlab.macros("recipes/engine.json")          # the macro specs
soundlab.batch(["recipes/blocks"], "out/", variations=5, sweep={"pitch": [0.2, 0.8]})
```

## How macros map

At its default a macro leaves the recipe **exactly as written** (a test checks every shipped recipe), so
fitted values stay valid and macros move around them. Mappings (`mul`, `add`, `stretch`, `set`, piecewise
`points`, `clamp`) and target paths are documented at the top of `soundlab/macros.py` and in
[recipe-format.md](recipe-format.md). `stretch` scales breakpoint *times*, which is how "length" and
"speed" macros lengthen sweeps and envelopes. `master.gain_db` is applied after peak normalising, so
`distance` makes a sound quieter as well as duller/roomier (it never makes it louder than written).

## Variations

`--variations N` writes N files: `v01` is the recipe as written (at the chosen macros), `v02..` are
seeded variations. Each variation (1) jitters every numeric macro by its `jitter` (default +-0.04),
(2) applies the recipe's `vary` list (uniform `cents`, `rel`, `db` or `abs` offsets per control: e.g. hit
velocity +-20 %, strike position +-0.06, onset +-4 ms), and (3) renders with its own seed so every random
part (noise, grains, crackle timing) differs. Variation k of a recipe is fully determined by (seed, k):
re-running a batch gives identical files, and asking for more variations never changes the earlier ones.

## Naming

`<category>_<name>[_<macro><value>...]_v<NN>.wav`, words inside a field joined by `-`:
`impact_hit_materialmetal_v03.wav`, `pickup_coin-pickup_v01.wav`, `vehicle_engine_rpm0.85_load0.8_v01.wav`.
When category and name are the same the name appears once (`footstep_v01.wav`); loops get a `loop10s` tag.
Exported files are tail-trimmed and loudness-levelled per category; see [export.md](export.md).
`category` comes from the recipe. `batch` also writes `manifest.json` (file, recipe, macros, variation,
render time).

## Audition

`soundlab audition RECIPE` renders at 24 kHz into `/tmp/soundlab-audition/` and plays it with the first
player found (ffplay, afplay, paplay, aplay). `--watch` keeps the process alive and re-renders on every save,
which avoids Python's ~0.7 s start-up; most recipes render in 0.01-0.2 s. `--seconds` cuts long recipes.

## Default choices and fitting (0.8.0)

A choice macro at its default is now skipped, like a numeric macro at its default (the test suite forces the
default choice's list and checks it is the identity). Before 0.8 a default choice re-applied its `set`
values on every render, so `soundlab match` could not move a control that the default choice names (found
while fitting `hit.json`, whose `object: board` choice names the board's dimensions).

## Shipped macros

| recipe | category | macros |
|---|---|---|
| `blaster.json` | scifi | `size`, `intensity`, `length` |
| `bullet_impact.json` | weapon | `material` (concrete, wood, metal), `intensity`, `debris`, `ricochet`, `distance` |
| `car_passby.json` | vehicle | `speed`, `distance` |
| `cloth_rustle.json` | foley | `intensity`, `length`, `distance` |
| `coin.json` | coin | `floor` (tile, wood, table, stone), `metal` (cupronickel, bronze, steel, brass, aluminium), `size`, `height`, `spin`, `distance` |
| `coin_jingle.json` | coin | `metal` (cupronickel, bronze, steel, brass, aluminium), `coins`, `shake`, `length`, `distance` |
| `creature_growl.json` | creature | `size`, `roughness`, `snarl`, `length`, `distance`, `uvula`, `wet` |
| `door_creak.json` | door | `speed`, `length`, `pitch`, `irregularity`, `body`, `squeal`, `distance` |
| `door_slam.json` | door | `intensity`, `size`, `rattle`, `distance` |
| `dowel.json` | impact | `material` (wood, metal, glass, plastic, stone, cardboard), `size`, `intensity`, `hardness`, `distance` |
| `drip.json` | water | `size`, `intensity`, `distance` |
| `duck.json` | animal | `size`, `intensity`, `distance` |
| `engine.json` | vehicle | `rpm`, `load`, `size`, `intensity`, `distance` |
| `engine_blips.json` | vehicle | `rpm`, `load`, `size`, `intensity`, `distance` |
| `engine_idle.json` | vehicle | `rpm`, `load`, `size`, `intensity`, `distance` (loopable) |
| `explosion.json` | explosion | `size`, `intensity`, `debris`, `distance` |
| `explosion_small.json` | explosion | `size`, `intensity`, `debris`, `distance` |
| `fire.json` | ambience | `size`, `intensity`, `distance` |
| `foghorn.json` | ambience | `size`, `intensity`, `length`, `distance` |
| `foghorn_pattern.json` | ambience | `size`, `intensity`, `length`, `distance` |
| `footstep.json` | footstep | `shoe` (trainer, boot, heel), `surface` (wood, stone, gravel), `size`, `intensity`, `pace`, `creak`, `distance` |
| `glass_break.json` | glass | `size`, `intensity`, `floor` (hard, wood, soft), `distance` |
| `glass_hit_room.json` | impact | `size`, `intensity`, `distance` |
| `gunshot.json` | weapon | `calibre` (pistol, rifle, shotgun), `space` (street, room, open), `body`, `crack`, `distance` |
| `hit.json` | impact | `object` (board, door, dowel), `material` (wood, metal, glass, plastic, stone, cardboard), `size`, `intensity`, `hardness`, `distance` |
| `match.json` | foley | `size`, `intensity`, `wetness`, `distance` |
| `matchbox.json` | foley | `intensity`, `size`, `wetness`, `distance` |
| `metal_clang.json` | impact | `object` (pipe, plate, pan, sword), `intensity`, `size`, `ring`, `distance` |
| `paper_rustle.json` | foley | `material` (paper, plastic, leaves, cloth), `intensity`, `length`, `distance` |
| `punch.json` | combat | `target` (body, face, belly), `glove` (bare, glove), `intensity`, `sweeten`, `distance` |
| `rain.json` | ambience | `intensity`, `surface` (ground, leaves, roof, window, puddle), `closeness`, `distance` |
| `splash.json` | water | `size`, `intensity`, `distance`, `spray` |
| `sword_clash.json` | weapon | `object` (sword, pipe, plate, pan), `intensity`, `scrape`, `ring`, `distance` |
| `sword_swing.json` | whoosh | `speed`, `object`, `zing`, `distance` |
| `water.json` | water | `size`, `intensity`, `distance` |
| `wind.json` | ambience | `strength`, `gusts`, `trees`, `howl`, `distance` |
| `wind_gap.json` | ambience | `strength`, `pitch`, `whistle`, `distance` |
| `blocks/coin_pickup.json` | pickup | `pitch`, `length`, `brightness` |
| `blocks/laser_zap.json` | scifi | `pitch`, `length`, `intensity` |
| `blocks/magic_shimmer.json` | magic | `pitch`, `density`, `size` |
| `blocks/power_up.json` | powerup | `pitch`, `length`, `brightness`, `size` |
| `blocks/ui_click.json` | ui | `pitch`, `length`, `brightness` |
| `blocks/ui_confirm.json` | ui | `pitch`, `length`, `brightness` |
| `blocks/whoosh.json` | whoosh | `speed`, `pitch`, `intensity` |
