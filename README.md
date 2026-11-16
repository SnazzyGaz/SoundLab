# SoundLab

A physical-modelling sound engine. You describe the object and the event (a 0.3 m steel bar struck hard, a
4-cylinder engine at 3000 rpm under load, a diaphone foghorn heard from 800 m) and SoundLab simulates what
vibrates and how the sound travels. Where full physics would be too slow (crackle, rain, debris) it uses
physically informed stochastic models whose controls still map to real quantities.

By Garyck Arntzen ([@_mintydoge](https://x.com/_mintydoge)). MIT licence. Standalone project; AIDAW uses it as
a library/plugin.

Status: **0.1, first working core** (build steps 1-6 of the plan, plus a first analysis tool). See
[docs/roadmap.md](docs/roadmap.md).

## The five stages

```
excitation -> resonator -> radiation -> propagation -> room
(hit, scrape,  (modes, tubes,  (source size,   (distance, air,   (rooms, outdoor
 airflow, valve, strings,       horns)          Doppler, ground)  echoes)
 burst, grains) formants)
```

Every block has physical controls in real units with realistic ranges and sources (`soundlab describe <model>`).
Complete sound models (`duck`, `engine`, `foghorn`, `explosion`, `match`, `fire`, `water`, `hit`) are built from
the same blocks.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[plot,test]"      # numpy + scipy; matplotlib for analysis plots; ffmpeg for MP3
```

## Command line

```bash
soundlab models                         # list all blocks and sound models by stage
soundlab describe engine                # controls, units, ranges, sources
soundlab materials                      # material table (density, E, loss factor, T60 at 1 kHz)
soundlab render recipes/engine.json -o engine.wav --mp3
soundlab render recipes/engine.json --set 0.cylinders=6 --seed 4 -o six.wav
soundlab validate recipes/duck.json
soundlab analyze sample.wav -o sample_stems/   # harmonic / transient / noise / sines stems + features.json + plot
```

## Python API (what AIDAW calls)

```python
import soundlab

audio, sr = soundlab.render("recipes/duck.json")            # numpy float array, mono
audio, sr = soundlab.render({"duration": 2, "layers": [{"model": "hit", "params": {"material": "glass"}}]})
soundlab.render_to_file(recipe, "out.wav", mp3=True, seed=7)
soundlab.list_models("excitation"); soundlab.describe("modal")
feats = soundlab.analyze_file("sample.wav", out_dir="stems/")
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

Shipped recipes: `duck`, `engine`, `foghorn`, `explosion`, `match` (reference sounds) and `glass_hit_room`,
`car_passby`, `fire`, `drip`, `blaster` (examples). Full format: [docs/recipe-format.md](docs/recipe-format.md).

## Documentation

- [docs/playing-guide.md](docs/playing-guide.md): how to play each sound, what each control does physically,
  and what makes it read as real.
- [docs/physics-reference.md](docs/physics-reference.md): formulas, material and mode tables, sources.
- [docs/models.md](docs/models.md): every block and control (generated from the code).
- [docs/analysis.md](docs/analysis.md): the stem-split / feature tool and where it is going.
- [docs/architecture.md](docs/architecture.md): code layout and how AIDAW plugs in.
- [docs/licensing.md](docs/licensing.md): what we may copy, what is reference-only.
- [docs/roadmap.md](docs/roadmap.md): build order and status.

## Tests

```bash
pytest -q
```

The tests check the physics against the formulas (bar mode ratios, loss-factor decay, Minnaert bubble pitch,
tube resonances, Aeolian tone, firing frequency, Doppler shift, ISO air absorption) and the five reference sounds
against their recipe expectations.
