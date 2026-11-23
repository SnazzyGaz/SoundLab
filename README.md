# SoundLab

A physical-modelling sound engine. You describe the object and the event (a 0.3 m steel bar struck hard, a
4-cylinder engine at 3000 rpm under load, a diaphone foghorn heard from 800 m) and SoundLab simulates what
vibrates and how the sound travels. Where full physics would be too slow (crackle, rain, debris) it uses
physically informed stochastic models whose controls still map to real quantities.

By Garyck Arntzen ([@_mintydoge](https://x.com/_mintydoge)). MIT licence. Standalone project; AIDAW uses it as
a library/plugin.

What's in it:

- **One impact model for every material** (`material_impact`, v0.26): wood, metal, glass, ceramic, stone,
  plastic, rubber, flesh, dirt, gravel, grass, leaves on plate/bar/rod/beam/box/shell/ball/block shapes, with
  size, force, hardness, hollowness, muffle and contact-point sliders; see [docs/impacts.md](docs/impacts.md).
- **Friction through the same materials** (`friction_family`, v0.27): creaks, squeaks, scrapes and rubs with
  pressure, speed, speed-change, irregularity, size and hollowness sliders; see [docs/friction.md](docs/friction.md).
- **Squeals, squeaks and creaks from recording stems** (`squeal_stems`, v0.29): door squeal, door creak (open,
  pause, close), hinge squeaks, gate squeals (with an optional latch), shoe squeaks and floorboard creak rebuilt
  from stems of CC0 recordings (partial tracks, real wobble, noise layer, onsets, body colour; numbers only, no
  audio) and played as a gesture timeline of motions; macros brightness, stretch, pitch, motions, gaps, seed.
  The fitted physical `creak` model (v0.28) stays as a fallback in `recipes/creak_model/`; previous procedural impact, footstep and clash recipes that now use recording stems stay in `recipes/impact_model/`; see
  [docs/friction.md](docs/friction.md).
- **Wind, whooshes, arrow fly-bys and fans from recording stems** (`air_stems`, v0.30): `wind_gust`, `wind_trees`
  (a gale by default; a `strength` macro takes it down to a breeze that barely registers), `whoosh` (speed, size,
  pitch), `arrow_flyby` (count, spacing, speed, closeness: real passes re-flown with the Doppler re-drawn) and `fan`
  (speed, size: blade rate and tip-speed brightness), rebuilt from band-noise envelopes, tonal tracks, onset
  envelopes, colour and blade cycles of CC0 recordings (numbers only, no audio). The generic physical `wind`,
  `wind_gale` and `sword_swing` recipes stay as fallbacks; see [docs/air.md](docs/air.md).
- **Splashes, drips, a pour, flowing water, boiling and sloshing from recording stems** (`liquid_stems`, v0.31):
  `splash_small`, `splash_medium`, `splash_large` (size, distance, tail; size by band shift or tape speed),
  `drip_tub` (rate, pitch: the real drops re-sequenced), `pour_glass` (fill, vessel), `stream`, `river`,
  `waterfall` (strength: quieter, darker, slower, with no pitch shift), `boiling` (intensity) and `slosh` (amount),
  rebuilt from per-band noise envelopes, bubble chirps, onset envelopes and colour of public-domain and CC0
  recordings (numbers only, no audio). The procedural `drip`, `splash` and `water` recipes stay as fallbacks; see
  [docs/liquids.md](docs/liquids.md).
- **Engines, pass-bys, motors and machines from recording stems** (`machine_stems`, v0.32): `petrol_rev` and
  `lawn_mower` (throttle), `diesel_idle` (load), `motorbike_passby`, `car_drive_by` and `tram` (distance, speed),
  `servo_motor`, `power_drill` and `sewing_machine` (speed), `ac_unit` (fan) and `clock_tick` (tick rate, room),
  rebuilt from tracked harmonics with their real amplitude and phase wobble, persistent lines, band-envelope noise,
  per-cycle firing pulses and band-envelope ticks of CC0 and public-domain recordings (numbers only, no audio).
  Gesture timelines (start, hold, change speed, stop) play the drill, motor, sewing machine and mower from their
  real spin-up, steady and spin-down stems: `power_drill_on_off`, `power_drill_two_speeds`, `servo_motor_on_off`,
  `sewing_machine_on_off`, `lawn_mower_on_off`; see [docs/machines.md](docs/machines.md).
- **Fire, crackle and electricity from recording stems** (`fire_stems`, v0.33): `campfire` (size, distance),
  `large_blaze` (intensity, distance), `embers` (pop density), `match_strike` (force), `gas_stove_ignite` (click
  count, flame), `sparkler` (pop density, distance), `fuse` (burn speed by tape speed), `transformer_hum` (load),
  `electric_arc` (intensity) and `fluorescent_flicker` (clang density), rebuilt from band-envelope noise beds, every
  recorded crackle, pop and click as its own band envelopes, and tracked harmonics and lines of CC0 recordings
  (numbers only, no audio). Fresh takes replay forward-only pieces of the stems in a new order with the events
  riding along; see [docs/fire.md](docs/fire.md).
- **Impacts and breaking from recording stems** (`impact_stems`, v0.34): `wood_knock`, `drumstick`, `dowel`,
  `wood_drop`, `ball_bounce`, `metal_clang`, `glass_clink`, `pickaxe_earth`, `pickaxe_rock`, `coin_drop`,
  `coin_jingle`, `body_punch`, `ceramic_break` and `glass_break`, rebuilt from body-mode tracks, each recorded hit
  as band envelopes, and a 1/8-octave 1 ms noise bed of CC0 recordings (numbers only, no audio). Recipes play a
  single real gesture by default. `force`, `soft`, `size` (tape speed), `distance`, `count`, `tempo` and `seed`;
  see [docs/impact_stems.md](docs/impact_stems.md).
- **Weapons and combat from recording stems** (`weapon_stems`, v0.34): shots, pumps, reloads, blade work,
  explosions and related one-shots (40 recipes), rebuilt the same way from CC0 recordings (numbers only, no audio).
  A single real gesture or a short natural count by default; the same macros as `impact_stems`;
  see [docs/weapon_stems.md](docs/weapon_stems.md).
- **Footsteps from recording stems** (`foot_stems`, v0.34): walks and runs on concrete, stone, wood (floor, creaky,
  stairs), gravel, dirt, grass, sand, snow, mud, puddles, metal (walk, stairs), tile, carpet and leaves, plus boots,
  heels, trainers and barefoot on wood, and two jump landings (25 recipes). Rebuilt the same way from CC0 recordings
  (numbers only, no audio). Defaults are a softer, more natural step than the often-forceful recordings; `force` and
  `soft` stay as macros. The older `footstep` / `footstep_*` names still work;
  see [docs/foot_stems.md](docs/foot_stems.md).
- **Household foley from recording stems** (`foley_stems`, v0.35): cloth, zips, bags, paper, a page turn, a book, cups,
  glasses, bottles, cutlery, keys, a coin, a phone, a pen click, a light switch, a chair scrape and a cupboard
  (23 recipes in `recipes/foley/`), rebuilt the same way from CC0 recordings (numbers only, no audio);
  see [docs/foley_stems.md](docs/foley_stems.md).
- **Doors, locks, knocks and gates from recording stems** (`door_stems`, v0.35): wood, heavy, metal, glass, screen,
  sliding, car, cupboard and wardrobe doors, a drawer, a garage door, two gates, a key in a lock, a deadbolt, three
  knocks and a hinge squeak (24 recipes in `recipes/doors/`); see [docs/door_stems.md](docs/door_stems.md).
- **Sequences** (v0.35): library sounds on a timeline (`soundlab sequence`, and a Sequence tab in the slider app).
  Presets for a walk, a run, knocks, a door, gunfire and a reload. Export a mix or numbered one-shots with a cue CSV;
  see [docs/sequences.md](docs/sequences.md).
- **Weather and ambience from recording stems** (`weather_stems`): `rain` (surface, intensity, and an intensity
  curve that can swell or fade), `rain_drips`, `thunder` (near or distant), `ocean_waves`, `wind_breeze` (intensity from a gentle breeze to gusty),
  `wind_howl`, `forest_day`, `night_insects`, `city_street` and `room_tone`. Beds loop to any length. See
  [docs/weather.md](docs/weather.md). The previous procedural rain and wind-howl recipes remain under
  `recipes/synth_model/`. `wind_gust` and `wind_trees` stay as `air_stems` recipes.
- **Featured recipes** for game sound (295 recipe files in all; the app lists 197, the rest are old ids kept as
  hidden aliases, see [docs/library.md](docs/library.md)): footsteps on many surfaces and shoes, knocks, coins, wind, whooshes, arrows, fans,
  rain, thunder, ocean waves, forest and night beds, a city street, room tone, glass break, sword swing and clash, metal clang, door creak, squeal and slam, hinge, gate and shoe squeaks, punch, bullet impacts, gunshot
  (pistol/rifle/shotgun), creature growl, paper and cloth rustle, splashes, drips, a pour, a stream, a river, a waterfall, boiling and sloshing, engines, explosions, fire, water,
  a duck, a foghorn, a match, plus 7 demos of the generic synth blocks.
- **Macro sliders** on every recipe (size, distance, intensity, material, ...), **seeded variations** and
  **batch export** with sensible file names.
- **Game-ready exports**: loudness per category (BS.1770 LUFS, -1 dBTP ceiling), reverb and ringing tails
  played out then silent tails trimmed, DC blocked, 24-bit (16-bit dithered), and **seamless loops** for rain,
  wind, fire, water and engine idle (`--loop 10`), exported as **stereo beds**; see
  [docs/export.md](docs/export.md).
- A **slider app** in the browser (`soundlab gui`) with a library tree, one Variation control, play,
  waveform/spectrogram preview and export; slider edits re-render in a near-instant pass when the audio is unchanged.
- A **blind listening test** in the browser (`/blindtest`, "real or synthesised?") and an A/B kit.
- Fitting tools: analyse a recording, fit a recipe's controls to it (`soundlab match`), compare renders.
- A generic synth toolkit (oscillators, FM, noise, grains, filters, envelopes, distortion, delay, chorus,
  reverb) for non-physical sounds.

See [docs/roadmap.md](docs/roadmap.md) for what is planned.

## The five stages

```
excitation -> resonator -> radiation -> propagation -> room
(hit, scrape,  (modes, tubes,  (source size,   (distance, air,   (rooms, outdoor
 airflow, valve, strings,       horns)          Doppler, ground)  echoes)
 burst, grains) formants)
```

Every block has physical controls in real units with realistic ranges and sources (`soundlab describe <model>`).
Complete sound models (`duck`, `engine`, `foghorn`, `explosion`, `match`, `fire`, `water`, `hit`, `footstep`, `coin`) are built from
the same blocks.

## Install

### Windows

You need Python 3.10 or newer (from [python.org](https://www.python.org/downloads/) — tick "Add python.exe to
PATH" — or `winget install Python.Python.3.12`) and Git (`winget install Git.Git`). In PowerShell or Command
Prompt:

```bat
git clone https://github.com/SnazzyGaz/SoundLab.git
cd SoundLab
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
soundlab gui
```

The app opens in your browser at http://127.0.0.1:8766/. Next time: `cd SoundLab`, `.venv\Scripts\activate`,
`soundlab gui`.

- If PowerShell refuses to run `activate` ("running scripts is disabled"), run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt instead.
- MP3 export and reading MP3 reference files need ffmpeg (optional; WAV works without it):
  `winget install Gyan.FFmpeg`, then open a new terminal.
- For the tests: `pip install -e ".[test]"`, then `pytest -q`.
- Keep the clone: install with `-e` (editable) so the app finds the shipped `recipes/` folder.

### macOS / Linux

```bash
git clone https://github.com/SnazzyGaz/SoundLab.git && cd SoundLab
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[plot,test]"      # numpy + scipy; matplotlib for analysis plots; ffmpeg (optional) for MP3
soundlab gui
```

## Slider app

`soundlab gui` starts a small local web app (Python standard library only; it listens on 127.0.0.1, so nothing
is reachable from other computers) and opens your browser. Options: `--port 8766` (AIDAW uses 8765), `--out
<folder>` (default `~/SoundLab`, or the `SOUNDLAB_OUT` environment variable), `--no-browser`.

- Sounds listed by what you look for (Foley › Doors & gates › Wood door, close), plain names and a search box
  that matches plain words; your own saved recipes appear under My recipes. See [docs/library.md](docs/library.md).
- A few plain controls per sound (Intensity, Size, Distance, choices such as Surface or Shape); an "Advanced"
  panel with the finer controls and every underlying model control.
- Variation: 0 = the sound as designed, any other number = another take (◀ ▶, 🎲 for a random one).
- Play renders fast and plays in the browser; moving a slider re-renders and plays when you let go.
- Waveform and spectrogram of the current sound.
- Export WAV (and MP3 with ffmpeg) to `<out>/exports/`, named like `weapon_gunshot_calibreshotgun_v01.wav`;
  batch-export N variations; "Save as new recipe" writes `<out>/recipes/<name>.json` with the sliders kept.
- Sequence tab: chain library entries on a timeline (walk, knock xN, door open-close-lock, gunfire with
  casings, reload, or your own), each event its own take, then export one mix or numbered one-shots with a cue
  CSV. Timing-only edits re-mix cached events. See [docs/sequences.md](docs/sequences.md).
- `/blindtest`: the blind listening test (see [docs/checkpoint.md](docs/checkpoint.md)).

## Sequences

A sequence chains library entries on a timeline and renders them into one file: a walk on gravel, knock xN,
a door opened, closed and locked, shots with casings, a reload, or your own steps. Every event draws its own
real take, never the same as the previous one. Timing and level can be humanised, and footsteps alternate left
and right. Export one mix (WAV/MP3) or numbered one-shots with a cue CSV (time, entry, take, gain). Use the
Sequence tab in the slider app or `soundlab sequence`. See [docs/sequences.md](docs/sequences.md).

## Command line

```bash
soundlab gui                            # slider app in the browser (http://127.0.0.1:8766/)
soundlab blindtest build --library my_library/   # listening pack for /blindtest; `soundlab blindtest score`
soundlab models                         # list all blocks and sound models by stage
soundlab describe engine                # controls, units, ranges, sources
soundlab materials                      # material table (density, E, loss factor, T60 at 1 kHz)
soundlab render recipes/engine.json -o engine.wav --mp3
soundlab sequence walk -m surface=gravel --repeat 12 -o walk.wav   # entries on a timeline (see --list)
soundlab sequence knock --repeat 5 --oneshots -o knock.wav  # + knock_oneshots/ (numbered files, cues.csv)
soundlab render recipes/rain.json --loop 10 -o rain_loop.wav   # seamless 10 s stereo loop (--mono for mono)
soundlab render recipes/hit.json --level peak --no-trim   # raw level, full length (default: LUFS per category, trimmed)
soundlab render recipes/engine.json --set 0.cylinders=6 --seed 4 -o six.wav
soundlab describe recipes/hit.json      # a recipe's macro sliders (the main controls)
soundlab render recipes/hit.json -m material=metal -m size=0.8 -o bar.wav
soundlab batch recipes/blocks -o out/ --variations 5 --sweep pitch=0.2,0.8 --mp3 --zip
soundlab audition recipes/footstep.json --watch   # quick 24 kHz preview, re-renders on save
soundlab validate recipes/duck.json
soundlab analyze sample.wav -o sample_stems/   # harmonic / transient / noise / sines stems + features.json + plot
soundlab match ref.wav --recipe recipes/fit/duck_fit.json -p 0.f0:120:350 -p 0.rasp -o duck_fit/
                                        # fit controls to a recording: fitted.json + report.md + comparison.png
soundlab resynth ref.wav --kind auto --holdout-section 1.2:2.0 -o ../fits/ref
                                        # recording -> editable layered recipe + fit + hold-out check
soundlab compare ref.wav a.wav b.wav    # distance of renders to a reference (lower = closer)
```

## Python API (what AIDAW calls)

```python
import soundlab

audio, sr = soundlab.render("recipes/duck.json")            # numpy float array, mono
audio, sr = soundlab.render({"duration": 2, "layers": [{"model": "hit", "params": {"material": "glass"}}]})
audio, sr = soundlab.render("recipes/hit.json", ring_out=True)  # let the room ring past `duration`
lr, sr = soundlab.render_loop("recipes/rain.json", 10, stereo=True)   # (n, 2) seamless stereo bed
soundlab.render_to_file(recipe, "out.wav", mp3=True, seed=7)
soundlab.list_models("excitation"); soundlab.describe("modal")
feats = soundlab.analyze_file("sample.wav", out_dir="stems/")
res = soundlab.match("recipes/fit/match_fit.json", "strike.wav", ["0.flare_s", "0.flame_db"], out_dir="fit/")
d = soundlab.compare("strike.wav", "render.wav")   # {"spectral", "stft", "envelope", "texture", "pitch", "total"}
```

Renders are deterministic: the same recipe and seed always give the same samples.

## Recipes

Plain JSON. A layer is a single model or a chain of blocks; `post` blocks run on the mix. Any `automatable`
control takes a breakpoint list `[[time_s, value], ...]`.

```json
{
  "soundlab": 1, "sample_rate": 48000, "duration": 3.0, "seed": 1,
  "layers": [
    {"name": "tap", "chain": [
      {"model": "impact", "params": {"velocity": 1.2, "contact_ms": 0.2, "bounces": 3}},
      {"model": "modal",  "params": {"material": "glass", "shape": "bar", "length": 0.25}},
      {"model": "radiation", "params": {"radius": 0.01}}]}
  ],
  "post": [{"model": "room", "params": {"volume": 40, "absorption": 0.3}}],
  "master": {"normalize_db": -1.0}
}
```

Shipped recipes: `duck`, `engine`, `foghorn`, `explosion`, `match` (reference sounds), `matchbox` (shake +
strike + flame), `engine_blips`, `explosion_small`, `foghorn_pattern`, `glass_hit_room`, `car_passby`, `fire`, `drip`, `blaster` (examples), `hit`, `dowel`, `footstep`, `coin`, `coin_jingle`, `water`; the v0.34 stem one-shots (`wood_knock`, `pistol_shot`, `wood_floor` and the rest of `impact_stems`, `weapon_stems` and `foot_stems`); the 0.9 game library: `wind`, `wind_gap`, `rain`, `glass_break`, `sword_swing`, `sword_clash`, `metal_clang`, `door_creak`, `door_slam`, `punch`, `bullet_impact`, `gunshot`, `creature_growl`, `paper_rustle`, `cloth_rustle`, `splash`; v0.21 creature voices: `demon_growl`, `monster_roar`, `dog_growl`, `alligator_bellow`, `elephant_rumble`; v0.19 interaction models: `hc_knock`, `hc_bounce`, `scrape_sdt`, `roll_sdt`, `creak_friction`, `squeak_friction`, `crumple_paper`, `break_glass_sdt`, `wind_howl`, `drill_motor`, `mains_hum`; `recipes/blocks/*` (generic-synth demos: UI click/confirm, coin, power-up, laser,
magic shimmer, whoosh), and `recipes/fit/*_fit.json` (dry recipes
matched to the reference recordings, the starting points for `soundlab match`). Full format: [docs/recipe-format.md](docs/recipe-format.md).

## Documentation

- [docs/playing-guide.md](docs/playing-guide.md): how to play each sound, what each control does physically,
  and what makes it read as real.
- [docs/physics-reference.md](docs/physics-reference.md): formulas, material and mode tables, sources.
- [docs/models.md](docs/models.md): every block and control (generated from the code).
- [docs/analysis.md](docs/analysis.md): the stem-split / feature tool and where it is going.
- [docs/artefacts.md](docs/artefacts.md): synthesis artefacts (grain, hiss, tinny, digital,
  "plastic bucket"), how they were measured and fixed.
- [docs/export.md](docs/export.md): loudness targets, tail trim, DC blocker, seamless loops.
- [docs/macros-and-variations.md](docs/macros-and-variations.md): macro sliders, variations, batch export,
  naming, audition.
- [docs/synth-blocks.md](docs/synth-blocks.md): the generic synthesis blocks (non-physical sounds, layering).
- [docs/fitting.md](docs/fitting.md): fitting recipes to recordings (`soundlab match`), the loss, and how to read
  a fit report.
- [docs/resynthesis.md](docs/resynthesis.md): analysis and resynthesis (`soundlab resynth`), the recording
  -> layered recipe pipeline, its new blocks (`bandnoise`, `harmonic`) and first results.
- [docs/impacts.md](docs/impacts.md): the impact and material model, its recipes and checks.
- [docs/sdt-methods.md](docs/sdt-methods.md): every Sound Design Toolkit model mapped to SoundLab blocks
  (method only), and the v0.19 interaction blocks.
- [docs/architecture.md](docs/architecture.md): code layout and how AIDAW plugs in.
- [docs/licensing.md](docs/licensing.md): what we may copy, what is reference-only.
- [docs/checkpoint.md](docs/checkpoint.md): the blind test: browser "real or synthesised?" page with your own
  recordings, plus the A/B kit (build a listening set, score it).
- [docs/sequences.md](docs/sequences.md): sequences: entries on a timeline, presets, takes, cue CSV export.
- [docs/roadmap.md](docs/roadmap.md): build order and status.

## Tests

```bash
pytest -q
```

The tests check the physics against the formulas (bar mode ratios, loss-factor decay, Minnaert bubble pitch,
tube resonances, Aeolian tone, firing frequency, Doppler shift, ISO air absorption) and the five reference sounds
against their recipe expectations.

## Licence and credit

MIT licence (see [LICENSE](LICENSE)). Made by Garyck Arntzen — [x.com/_mintydoge](https://x.com/_mintydoge).
Reference recordings used for tuning and the blind test are not part of this repository.
