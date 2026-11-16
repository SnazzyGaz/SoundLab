# SoundLab

A physical-modelling sound engine. You describe the object and the event (a 0.3 m steel bar struck hard, a
4-cylinder engine at 3000 rpm under load, a diaphone foghorn heard from 800 m) and SoundLab simulates what
vibrates and how the sound travels. Where full physics would be too slow (crackle, rain, debris) it uses
physically informed stochastic models whose controls still map to real quantities.

By Garyck Arntzen ([@_mintydoge](https://x.com/_mintydoge)). MIT licence. Standalone project; AIDAW uses it as
a library/plugin.

What's in it:

- **37 recipes** for game sound: footsteps (trainer/boot/heel on wood, stone, gravel), knocks, coins, wind,
  rain, glass break, sword swing and clash, metal clang, door creak and slam, punch, bullet impacts, gunshot
  (pistol/rifle/shotgun), creature growl, paper and cloth rustle, splash, engines, explosions, fire, water,
  a duck, a foghorn, a match, plus 7 demos of the generic synth blocks.
- **Macro sliders** on every recipe (size, distance, intensity, material, ...), **seeded variations** and
  **batch export** with sensible file names.
- **Game-ready exports**: loudness per category (BS.1770 LUFS, -1 dBTP ceiling), silent tails trimmed,
  DC blocked, and **seamless loops** for rain, wind, fire, water and engine idle (`--loop 10`); see
  [docs/export.md](docs/export.md).
- A **slider app** in the browser (`soundlab gui`) with play, waveform/spectrogram preview and export.
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

- Recipes listed by category with a search box; your own saved recipes appear with a ★.
- Each recipe's macro sliders and choice menus; an "Advanced" panel with every underlying model control.
- Variation number (0 = as written, 1, 2, 3... = random takes) and seed.
- Play renders fast and plays in the browser; moving a slider re-renders and plays when you let go.
- Waveform and spectrogram of the current sound.
- Export WAV (and MP3 with ffmpeg) to `<out>/exports/`, named like `weapon_gunshot_calibreshotgun_v01.wav`;
  batch-export N variations; "Save as new recipe" writes `<out>/recipes/<name>.json` with the sliders kept.
- `/blindtest`: the blind listening test (see [docs/checkpoint.md](docs/checkpoint.md)).

## Command line

```bash
soundlab gui                            # slider app in the browser (http://127.0.0.1:8766/)
soundlab blindtest build --library my_library/   # listening pack for /blindtest; `soundlab blindtest score`
soundlab models                         # list all blocks and sound models by stage
soundlab describe engine                # controls, units, ranges, sources
soundlab materials                      # material table (density, E, loss factor, T60 at 1 kHz)
soundlab render recipes/engine.json -o engine.wav --mp3
soundlab render recipes/rain.json --loop 10 -o rain_loop.wav   # seamless 10 s loop
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
soundlab compare ref.wav a.wav b.wav    # distance of renders to a reference (lower = closer)
```

## Python API (what AIDAW calls)

```python
import soundlab

audio, sr = soundlab.render("recipes/duck.json")            # numpy float array, mono
audio, sr = soundlab.render({"duration": 2, "layers": [{"model": "hit", "params": {"material": "glass"}}]})
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
strike + flame), `engine_blips`, `explosion_small`, `foghorn_pattern`, `glass_hit_room`, `car_passby`, `fire`, `drip`, `blaster` (examples), `hit`, `dowel`, `footstep`, `coin`, `coin_jingle`, `water`; the 0.9 game library: `wind`, `wind_gap`, `rain`, `glass_break`, `sword_swing`, `sword_clash`, `metal_clang`, `door_creak`, `door_slam`, `punch`, `bullet_impact`, `gunshot`, `creature_growl`, `paper_rustle`, `cloth_rustle`, `splash`; `recipes/blocks/*` (generic-synth demos: UI click/confirm, coin, power-up, laser,
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
- [docs/architecture.md](docs/architecture.md): code layout and how AIDAW plugs in.
- [docs/licensing.md](docs/licensing.md): what we may copy, what is reference-only.
- [docs/checkpoint.md](docs/checkpoint.md): the blind test: browser "real or synthesised?" page with your own
  recordings, plus the A/B kit (build a listening set, score it).
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
