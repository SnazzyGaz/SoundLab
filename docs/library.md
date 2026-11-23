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

## Controls

Each sound shows a few plain controls; the same names mean the same thing everywhere: **Intensity**, **Size**
(tape speed), **Distance**, and for short counted gestures **Count** and **Rate**. Fine controls that still do
something (softness, tape-speed cents, count and rate on one-shots) are under **Advanced**, with every model
control. Sliders that no longer change the sound, and older names that only repeat another slider, are hidden;
they stay in the recipe so existing calls and saved recipes still work.

**Variation** is the one control for new takes: 0 is the sound as designed, every other number is a different
take (◀ ▶ step through them, 🎲 picks one). It replaces the separate seed box and the per-recipe seed slider
(the slider stays in the files, hidden, for older calls).

## Old ids

Recipes that were merged or replaced stay as hidden files: they are not listed, but they still load, render
exactly as before and open by link, and the app names the entry that replaced them. The single-surface footstep
recipes are folded into Footsteps the same way. `glass_hit_room` is kept as a hidden engine demo.

## Next steps

Not built yet:

- **Sequences**: a timeline of entries with their own settings and offsets (walk: N steps at a rate; knock xN;
  open, close, lock; gunfire plus casings; reload), with timing and level humanisation, a fresh take per event
  (never the same take twice in a row), long beds with an intensity curve (crossfading real recordings, no
  stretching), and export as one mix or as numbered one-shots with a cue list.
- **Gaps**: real rain (light, heavy, on a roof, on a window), thunder, a storm build, sea waves; ambience beds
  (room tone, traffic, forest, crowd, office); animals (dog, cat, birds, horse, insects); non-verbal people
  (body fall, clothing movement, grabs); bullet impacts on concrete, wood and metal, sword swing with a ring,
  heavy body hits; car start/stop, idle loop, horn, plane, helicopter and train; household (typing, phone ring
  and vibrate, doorbell, kettle, tap, microwave, toilet flush, alarm clock); more UI sounds (error, back, hover,
  notification, toggle).
- Several older synthesis recipes stay in the library until stem rebuilds replace them (rain, wind howl and
  whistle, the creature growls, the animals, engine idle and blips, the scrapes, bullet impact, sword swing).
