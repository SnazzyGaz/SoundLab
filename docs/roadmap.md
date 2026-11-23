# Roadmap

What is planned next. For what exists today see the [README](../README.md).

Ground rules:
- **One implementation, Python/NumPy.** No PyTorch copy of the engine (two copies drift apart). If real-time
  playback is needed later, generate it from a single source (e.g. Faust), do not hand-port every block.
- **Physics models are kept as they are** (impact, modal, tubes, engine, foghorn, explosion, match, shaker,
  ...). Deeper physics only comes after the checkpoint, and only for sounds a game steers live.
- Offline render to WAV is the main use.

## Now: interaction models and resynthesis (v0.17 to v0.27)

v0.35: household foley and doors from recording stems (`foley_stems`, `door_stems`; numbers only; 23 foley and 24
door recipes); the library is one tree of plain names, with old recipe ids kept as hidden aliases (see
[library.md](library.md), [foley_stems.md](foley_stems.md) and [door_stems.md](door_stems.md)). The slider app
re-renders in a near-instant pass when the audio is unchanged, with named macro options, display labels and an
Advanced panel. Sequences put library entries on a timeline (`soundlab sequence`, a Sequence tab in the slider app),
with presets for a walk, a run, knocks, a door, gunfire and a reload, and export a mix or numbered one-shots with a
cue CSV; see [sequences.md](sequences.md).

v0.36: weather and ambience from recording stems (`weather_stems`: rain, drips, thunder, wind, ocean waves,
and looping forest, night, city and room-tone beds; an intensity curve can swell or fade by crossfading real
recordings; numbers only) and everyday interaction stems (`interact_stems`: camera, drinks and household
handling; 18 recipes), plus typing and can-and-pour sequence presets. See [weather.md](weather.md) and
[interact_stems.md](interact_stems.md).

v0.34: impacts, weapons and footsteps from recording stems (`impact_stems`, `weapon_stems`, `foot_stems`: body-mode
partial tracks, each recorded hit or step as its own 1/8-octave band envelopes, a 1/8-octave 1 ms noise bed, a
1/6-octave colour curve and a static treble ceiling; numbers only). 14 impact, 40 weapon and 25 footstep recipes
(a single real gesture or a short natural count by default), with mild macros (`force`, `soft`, `size` by tape
speed, `distance`, `count`, `tempo`, `seed`). Footstep recipes default to a softer step than the often-forceful
recordings. The older `footstep` / `footstep_*` names, and a few impact / weapon filenames that already existed,
now use the stem models and still accept their previous macro names. See [impact_stems.md](impact_stems.md),
[weapon_stems.md](weapon_stems.md) and [foot_stems.md](foot_stems.md).

v0.33: fire, crackle and electricity from recording stems (`fire_stems`: band-envelope noise beds at 1 ms, every
recorded crackle / pop / click as its own band envelopes at 0.5 ms (0.125 ms for igniter clicks), tracked harmonics
and lines at 4 ms, a static treble ceiling; numbers only); new recipes `campfire`, `large_blaze`, `embers`,
`match_strike`, `gas_stove_ignite`, `sparkler`, `fuse`, `transformer_hum`, `electric_arc`, `fluorescent_flicker`,
each with mild macros (fuse speed by tape speed), and fresh takes from forward-only pieces of the stems replayed in
a new order with their events. Next for this family: ignite / burn / die-down gestures, a torch, stereo from paired
takes; see [fire.md](fire.md).

v0.32: machines and vehicles from recording stems (`machine_stems`: tracked harmonics with their real amplitude and
phase wobble at 4 ms, persistent lines, band-envelope noise at 1 ms, per-cycle firing pulses, band-envelope ticks;
numbers only); new recipes `petrol_rev`, `diesel_idle`, `motorbike_passby`, `car_drive_by`, `servo_motor`,
`power_drill`, `ac_unit`, `clock_tick`, `tram`, `lawn_mower`, `sewing_machine`, each with mild macros (speed by tape
speed), plus gesture timelines (start, hold, change speed, stop) from real spin-up, steady and spin-down stems.
Next for this family: recorded starts and stops for the motor and sewing machine, engine start / shut-down,
stereo from paired takes; see [machines.md](machines.md).

v0.31: liquids from recording stems (`liquid_stems`: per-band noise envelopes on 1/8-octave bands at 2 ms imposed
on band-limited noise, bubbles as chirps, onset envelopes and colour, with the `air_stems` STFT engine for the
stream, boiling and slosh; numbers only); new recipes `splash_small`, `splash_medium`, `splash_large`, `drip_tub`,
`pour_glass`, `stream`, `river`, `waterfall`, `boiling`, `slosh`, each with mild macros; variations remix the real
stems. Next for this family: a smoother pour jet, cleaner references
for the pour and the large splash, stereo from paired takes; see [liquids.md](liquids.md).

v0.30: air from recording stems (`air_stems`: band-noise envelopes smoothed across frequency, tonal tracks,
onset envelopes, colour, a fan's blade cycle, fly-by passes with their Doppler geometry; numbers only); new recipes
`wind_gust`, `wind_trees` (gale by default, `strength` down to a breeze), `whoosh`, `arrow_flyby`, `fan`; the
generic `wind`, `wind_gale` and `sword_swing` recipes stay as physical fallbacks. Next for this family: calmer
wind recordings for the low end of `strength` (it is extrapolated from a gale today), flags and cloth flapping
from a better reference, howling gaps, stereo from paired takes; see [air.md](air.md).

v0.29: squeals, squeaks and creaks rebuilt from stems of CC0 recordings (`squeal_stems`: partial tracks, the
real wobble, noise layer, onsets and body colour per burst, stored as numbers, recombined across recordings and
played as a gesture timeline of motions with pauses between them); new recipes `door_squeal`, `hinge_squeak_2`,
`gate_squeal`, `gate_squeal_long`, `shoe_squeak_2`; the v0.28 `creak` recipes moved to `recipes/creak_model/` as the
physical fallback. Next for this family: more motions per recording in the pool (squeaky chairs, beds, ropes,
wet glass), speed and pressure read from a game's door angle, stems for scrapes; see [friction.md](friction.md).

v0.28: creaks and squeaks rebuilt as the fitted `creak` model (groan: slip clicks through fixed body
resonances; squeal: a near-pure tone locked to a resonance that hops between bursts), measured from CC0
recordings; `rope_creak` removed; see [friction.md](friction.md).

v0.27.1: door creak, hinge squeak, shoe squeak and rope creak rebuilt as stick-slip squeals from the excitation
(no ringing body, no knock at the stroke end); see [friction.md](friction.md).

v0.27: friction as one model (`friction_family`: elasto-plastic stick-slip, scrape and rub driving the
`material_impact` bodies) with 11 recipes; see [friction.md](friction.md). Gravel footsteps lost their ringing
pebbles (dull stone grains). Next families: air and whooshes, liquids, machines, fire and electricity, creatures.

v0.26: impacts and materials as one model (`material_impact`: Hunt-Crossley contact, shape x material modes,
per-material damping laws, residual noise, ground grains, fracture) with 16 derived recipes; see
[impacts.md](impacts.md). Next families: friction (creaks, squeaks, scrapes), air and whooshes, liquids,
machines, fire and electricity, creatures (frame-wise resynthesis, see research/creature-research.md).

v0.21: creature voices (`creature_voice`, CHIMERA glottis port + f0/2 ventricular layer + lossy-tube tract) and
a `growl` resynthesis kind. Next for creatures: a moving tract (mouth automation is averaged today), a syrinx for
birds, CHIMERA's pulsed seal calls and the elephant's trunk-valve trumpet.
Research for the next creature step (frame-wise analysis-resynthesis with subharmonic and chaos tracks, macros
that edit analysed tracks) is in [research/creature-research.md](research/creature-research.md).


v0.20: splashes keep the v0.18 impact and get a separately fitted bubble tail; choirs lose the noise layer and gain
per-singer jitter, shimmer, drift, glottal shape, vibrato onset, a drone and a convolution hall. Open: the choir's
400 Hz-3 kHz region is still 8-12 dB thin; the double splash's tail droplets are sparser than the model.


v0.19 went through every model in the Sound Design Toolkit (method only, see [sdt-methods.md](sdt-methods.md))
and added Hunt-Crossley contact, elasto-plastic friction (squeaks and creaks), rolling and scraping, crumpling and
breaking, wind cavities, DC motors and mains hum, plus a `sparsity` part in the fit loss so water fits choose
bubbles over noise. Backlog from that study and the wider survey:
- Hunt-Crossley contact inside the `hit`, `sword_clash` and `bullet_impact` sound models (they still use a
  fixed contact time).
- Engine: exhaust waveguides with runner reflections and a muffler, plus backfire (SDT Motor, engine-sim).
- Voices: ventricular-fold growl at f0/2 (CHIMERA), an LF glottal pulse, a syrinx for birds, a waveguide vocal
  tract for speech-like creatures.
- STK-style bowed string and blown pipe/reed (bowed creaks, whistles, horns).
- Explosion: FDN scattering after the Friedlander pulse.
- Analysis: McLeod pitch, spectral flux onsets, a structure-tensor harmonic/percussive/residual split.
- Water: bubble depth and "blooping" over time for pouring and filling.


v0.18 rebuilt water on bubbles (`bubble_cloud`) and voices on harmonic + envelope + aperiodicity, following
open-source projects and papers (see [resynthesis.md](resynthesis.md) and THIRD_PARTY.md). Next from that survey:
a bubble-aware distance term; CHIMERA-style ventricular-fold growl (f0/2) and a syrinx for birds; an
engine-sim-style exhaust waveguide with runner reflections; an LF glottal pulse; bubble depth / "blooping" over
time for pouring.

### v0.17

`soundlab resynth` turns a recording into an editable layered recipe (see [resynthesis.md](resynthesis.md)).
New generic blocks: `bandnoise` (band x time noise grid) and `harmonic` (formant/vocal source). Next:
- Moving formants for voices (meows, growls, vowels): automatable `formant_shift` or per-formant curves.
- Water textures: individual bubble and gurgle events on top of the stationary noise (`stft` is the weakest part).
- Impacts: fit the physical detail layer more and the copied grid less, so a recipe generalises to other
  recordings of the same kind of sound (hold-out on a second recording, not just a later section).
- Choirs: vowel changes, several melodic lines, and a measured hall instead of Freeverb.

## Next: the blind checkpoint

Run the blind listening test ([checkpoint.md](checkpoint.md)) with real library samples for every physical
recipe sound and at least two listeners who did not tune the recipes.

1. **Material.** For each recipe sound, SoundLab renders and library samples of the same sound (licensed
   sample library, own recordings or CC0). Loudness-matched, similar length, 48 kHz WAV, neutral file names
   (`soundlab blindtest build` does this).
2. **Listeners.** At least two people who did not tune the recipes, each scored separately and pooled.
   Headphones or decent speakers, same playback chain for all items.
3. **Procedure.** Random order per listener. For each clip: "real or synthesised?", optionally what gave it
   away. No replay limit, no time pressure.
4. **Score.** Per listener and pooled: balanced accuracy (the mean of the share of SoundLab clips called
   synthesised and the share of real clips called real; 50 % = cannot tell them apart), separately for
   **physical sounds** (impacts, engines, fire, water, explosions, footsteps, foghorn, duck, match, ...) and
   stylised ones (UI, magic, sci-fi, retro), where "sounds synthesised" is fine. See
   [checkpoint.md](checkpoint.md#4-read-the-result).
5. **Pass mark.** Physical sounds pass if the 95 % interval of pooled balanced accuracy includes 50 % (no
   reliable difference); a clear pass when the whole interval is under 60 %. A physical sound told apart by
   every listener is a failure to fix before any deeper physics work.
6. **Record.** Item list, answers and "what gave it away" notes stay with the pack, outside the repo
   (results and audio both).

## Sound to-dos

- **Gunshot**: pistol, rifle and shotgun sound too alike. The `calibre` choice should change the character
  (crack, body, length of the tail), not only small numbers.
- **Creature growl**: growls range from purr-like rumbles to bright, open snarls; the recipe sits at the deep
  end. A snarl preset (raised formants, more breath, less chest) would cover the bright end.
- **Splash**: v0.16 `slosh` brought the texture statistics closer (hold-out 0.82 -> 0.66), but the
  onset-aligned fit stays ~0.72 (0.68 in v0.14); the plateau of continuous splashing is not modelled.
- **Paper rustle**: v0.16 crackling-noise model; the earlier short reference (a single 0.4 s crumple) still
  fits worse (1.03) than the hold-outs (0.82). Plastic and leaves still use the older branch.
- **Glass break**: still furthest from its hold-out recordings (fit 0.87); those are brighter at 4-8 kHz with
  less 0.5-1 kHz than the engine.

## Known gaps

- Engine: 200-400 Hz still ~6 dB hot against the V8 reference; the long `engine.json` rev to 4500 rpm has no
  reference at that speed.
- Explosion: the cinematic `explosion` no longer has a long tail: its roar decays in 4 s (T60) and the sound ends
  at the -60 dB floor by about 7.5 s (it was 13 s). Short blasts are `explosion_close` and `explosion_distant`.
- Matchbox shake ~4 dB bright at 6.4-12.8 kHz.
- The sword clash and bullet impacts follow the CC0 refs' bright balance; check them for tinny tops.
- The pipe clang rings long (a free pipe does) and is very tonal.
- Door creaks vary hugely in real life (CC0 refs from ~26 Hz ticking to a ~1.1 kHz squeal); ours sits at 260 Hz
  with slip-rate jumps of 0.6-1.6x.
- Wind (v0.23) defaults to a steady breeze in trees fitted to windshielded field recordings; wind on a bare
  microphone (sub-60 Hz buffeting) needs `rumble` up and `low_cut_hz` down.
- The `room` block's early reflections colour close, broadband sounds slightly (comb lines); close foley
  recipes keep its level low.
- `passby` reads its delay line up to c/(c - v) times faster than real time while the source approaches
  (3.7x at 250 m/s) with linear interpolation and no anti-aliasing, so very fast flybys alias their top end.
  Car speeds are unaffected (1.06x at 20 m/s).
- Stereo/binaural output and per-layer propagation.

## Sequences: what is left

Sequences (entries on a timeline, presets, a fresh take per event, export as a mix or as one-shots with a cue
CSV) are built; see [sequences.md](sequences.md). Still to do:

- Intensity-curve beds on the weather recipes themselves are built; see [weather.md](weather.md). A per-entry
  intensity envelope in the sequence editor is not.
- Separate magazine-out and magazine-in pieces for the reload preset. The current recording has both in one
  gesture.
- Single-hit cuts (`hit`) for other multi-hit foley sources (coins, keys), as door knocks have now.

## Later, if the checkpoint says so

Deeper physics for live-steered sounds (engine following real revs, impacts that follow material and speed),
real-time via a single-source generator if needed, optional ML-based analysis (stem separation, learned
embeddings).
