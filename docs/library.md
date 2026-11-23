# The library

The app lists sounds by what people look for, with plain names: Domain › Thing (`Foley › Doors & gates ›
Wood door, close`). Search matches the name, the place in the tree and a few plain words per sound (`gun`,
`steps`, `slam`). Engine names are not shown.

Every recipe file carries this in non-audio keys ([recipe-format.md](recipe-format.md#library-keys)):
`title` (the plain name), `library` (`"Foley/Doors & gates"`), `tags`, and for old ids `hidden` plus
`replaced_by`. `category` is unchanged: it still sets the export loudness.

## One entry per need

Variants are choices inside one entry rather than separate list items:

- **Footsteps**: Surface (wood, creaky wood, concrete, stone, tile, metal, carpet, dirt, gravel, grass, leaves,
  sand, snow, mud, puddle), Shoe (normal, trainer, boot, heel, barefoot) and Pace (walk, run; run is recorded
  on concrete and grass, other surfaces walk). Each surface and shoe name plays the same sound as its
  single-surface recipe (`carpet_walk`, `trainers_walk`, ...), including that recipe's softer default and length.
  The old 0..1 surface and shoe values still work.
- **Machines**: Power drill, Lawn mower, Sewing machine and Small motor have a Shape choice: running, on-off
  (and two speeds for the drill). At default settings each shape is identical to the previous separate recipe.
- **Splash**: one entry with Splash small / medium / large (separate recordings), identical to the previous
  three recipes at every setting.
- **Coin drop**: Shape single (one coin) or spin (the longer spinning drop).
- **Parts**: where one recording holds two different actions they are a Part choice: wood door latch / swing,
  deadbolt lock / unlock, bag zip open / close, cupboard open / close.
- **Rain**: Surface (open ground, a roof, or a window) and Intensity (light to heavy on open ground, crossfading
  real recordings). An Intensity curve under Advanced can swell or fade over the length of the bed.
- **Thunder**: Kind near crack or distant roll.
- **Wind**: Intensity from a gentle breeze to gusty (crossfading real recordings). Wind howl stays a separate
  entry (a whistling, moaning wind). `wind_gust` and `wind_trees` stay as `air_stems` recipes.

## Controls

Each sound shows a few plain controls; the same names mean the same thing everywhere: **Intensity**, **Size**
(tape speed), **Distance**, and for short counted gestures **Count** and **Rate**. Fine controls that still do
something (softness, tape-speed cents, count and rate on one-shots) are under **Advanced**, with every model
control. Sliders that no longer change the sound, and older names that only repeat another slider, are hidden;
they stay in the recipe so existing calls and saved recipes still work.

**Variation** is the one control for new takes: 0 is the sound as designed, every other number is a different
take (◀ ▶ step through them, 🎲 picks one). It replaces the separate seed box and the per-recipe seed slider
(the slider stays in the files, hidden, for older calls).

## Sequences

The Sequence tab chains entries on a timeline: walk, run, knock xN, door open-close(-lock), gunfire with
casings, reload, or your own. Every event gets its own take, and export gives one mix or numbered one-shots with
a cue CSV. See [sequences.md](sequences.md).

## Old ids

Recipes that were merged or replaced stay as hidden files: they are not listed, but they still load, render
exactly as before and open by link, and the app names the entry that replaced them. The single-surface footstep
recipes are folded into Footsteps the same way. `glass_hit_room` is kept as a hidden engine demo.

## Next steps

Weather and ambience beds (`weather_stems`, [weather.md](weather.md)) cover rain (light, steady, heavy, on a
roof or a window, plus drips), thunder (near and distant), a gentle breeze and a howling wind, shore waves,
a forest afternoon, night insects, a city street and a quiet room. Rain and wind can follow an intensity curve
that crossfades real recordings (no stretching). `wind_gust` and `wind_trees` stay as `air_stems` recipes.
Sequences themselves are built: see [sequences.md](sequences.md).

Not built yet:

- **Gaps**: a storm build as a sequence; crowd and office beds; animals (dog, cat, birds, horse);
  non-verbal people (body fall, clothing movement, grabs); bullet impacts on concrete, wood and metal, sword
  swing with a ring, heavy body hits; car start/stop, idle loop, horn, plane, helicopter and train; household
  (typing, phone ring and vibrate, doorbell, kettle, tap, microwave, toilet flush, alarm clock); more UI sounds
  (error, back, hover, notification, toggle).
- Several older synthesis recipes stay in the library until stem rebuilds replace them (the creature growls,
  the animals, engine idle and blips, the scrapes, bullet impact, sword swing). The previous procedural rain
  and wind-howl recipes remain under `recipes/synth_model/`.
