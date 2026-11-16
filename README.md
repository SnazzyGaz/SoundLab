# SoundLab

A physical-modelling sound engine. You describe the object and the event (a 0.3 m steel bar struck hard, a
4-cylinder engine at 3000 rpm under load, a diaphone foghorn heard from 800 m) and SoundLab simulates what
vibrates and how the sound travels. Where full physics would be too slow (crackle, rain, debris) it uses
physically informed stochastic models whose controls still map to real quantities.

By Garyck Arntzen ([@_mintydoge](https://x.com/_mintydoge)). MIT licence. Standalone project; AIDAW uses it as
a library/plugin.

Status: **0.10.0**: gunshot, splash, door slam, creature growl, wind and rain reworked after a listening check (see docs/playing-guide.md). 0.9.0: a game-ready library of 36 recipes + 7 generic-block demos. New in 0.9: wind (gusts, trees, whistling gap), rain (light to heavy on ground/leaves/roof/window/puddle), glass break (fracture + shard debris), sword swing (with Doppler), sword clash and metal clang, door creak (hinge stick-slip) and slam, punch, bullet impacts (wood/metal/concrete), a simple gunshot, a creature growl (rough source + vocal tract), paper and cloth rustle, splash; plus the blind checkpoint kit (`tools/checkpoint.py`). In 0.8.1 heel and toe flam into one step (35 ms gap). New in 0.8: a body-weight `footstep` model (shoe: trainer/boot/heel; surface:
wood/stone/gravel), the default `hit` is now a damped board knock (the old quick resonant sound is
`dowel.json` / `-m object=dowel`), and a physical `coin` (drop with bounces and Euler's-disk wobble, plus a
jingle; the retro blip stays as `blocks/coin_pickup.json`). Every recipe has macro sliders (size, distance, intensity, material, ...), seeded
variations, batch export with sensible names and a quick audition loop. Physical models for the five reference sounds tuned by ear and to reference clips, plus a
generic synth toolkit (oscillators, FM, noise, grains, filters, envelopes, distortion, delay, chorus, reverb)
for non-physical game sounds. Next: a listening check of the new sounds, then the blind checkpoint (step 5). See
[docs/roadmap.md](docs/roadmap.md).

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
- [docs/artefacts.md](docs/artefacts.md): the synthesis artefacts found in v2 (grain, hiss, tinny, digital,
  "plastic bucket"), how they were measured and fixed in 0.3.0.
- [docs/macros-and-variations.md](docs/macros-and-variations.md): macro sliders, variations, batch export,
  naming, audition.
- [docs/synth-blocks.md](docs/synth-blocks.md): the generic synthesis blocks (non-physical sounds, layering).
- [docs/fitting.md](docs/fitting.md): fitting recipes to recordings (`soundlab match`), the loss, and how to read
  a fit report.
- [docs/architecture.md](docs/architecture.md): code layout and how AIDAW plugs in.
- [docs/licensing.md](docs/licensing.md): what we may copy, what is reference-only.
- [docs/checkpoint.md](docs/checkpoint.md): the blind A/B checkpoint kit (build a listening set, score it).
- [docs/roadmap.md](docs/roadmap.md): build order and status.

## Tests

```bash
pytest -q
```

The tests check the physics against the formulas (bar mode ratios, loss-factor decay, Minnaert bubble pitch,
tube resonances, Aeolian tone, firing frequency, Doppler shift, ISO air absorption) and the five reference sounds
against their recipe expectations.
