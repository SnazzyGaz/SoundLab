# Blind test (roadmap step 5)

The stop-and-judge test: can a listener tell SoundLab renders from real recordings
of the same sound? Pass mark for **physical sounds: under 60 % correctly identified** (50 % = pure guessing).

There are two ways to run it. The **browser test** is the easy one: listeners hear one clip at a time and
answer "real or synthesised?". The **A/B kit** (further down) plays pairs and asks which one is the engine.

## Browser test ("real or synthesised?")

### 1. Put your real recordings in a library folder

Make a folder anywhere outside the SoundLab folder, with one sub-folder per sound. Name each sub-folder
exactly like the recipe it should be compared with (the file names in `recipes/`, without `.json`):

```
my_library\
  gunshot\       pistol_01.wav  rifle_close.wav  ...
  door_slam\     slam_a.wav  slam_b.wav
  footstep\      wood_walk.wav
  rain\          roof_rain.wav
```

- WAV always works; MP3/FLAC/OGG work when ffmpeg is installed.
- One sound per file is best (one shot, one slam, a few steps); clips are cut to 4 seconds from the first
  sound, the decayed tail is trimmed (40 dB under the loudest moment), and every clip is matched in loudness
  (BS.1770, -20 LUFS max momentary, -1 dBTP ceiling). The engine clips get exactly the same treatment, so
  level and dead air give nothing away. Within each sound, all clips are also cut to the same
  length (keeping the loudest part) and set to the same loudness.
- Use recordings you are allowed to use (your own, a licensed sample library, CC0). Never put them in the
  SoundLab folder: the repo has a test that fails if any audio is found there.
- Sounds without a matching recipe name are skipped and listed. Useful names: `bullet_impact`,
  `cloth_rustle`, `coin`, `creature_growl`, `door_creak`, `door_slam`, `footstep`, `glass_break`, `gunshot`,
  `hit` (knock), `metal_clang`, `paper_rustle`, `punch`, `rain`, `splash`, `sword_clash`, `sword_swing`,
  `wind`, `wind_gap`, plus `engine`, `explosion`, `fire`, `water`, `match`, `duck`, `foghorn`, `drip`,
  `car_passby`.

### 2. Build the listening pack

```bat
soundlab blindtest build --library C:\path\to\my_library
```

This takes up to 2 real takes per sound (`--per-sound 3` for more), renders the same number of SoundLab
variations of the matching recipe, prepares every clip the same way, shuffles them and gives them neutral
names (`clip_001.wav`, ...). The pack goes to `~/SoundLab/blindtest` (on Windows
`C:\Users\<you>\SoundLab\blindtest`), or `--out <folder>`. Options: `--sounds gunshot rain` (only these),
`--seed 3` (another draw), `--max-s 3` (shorter clips), `--exempt blaster` (stylised sounds, not counted).
`key.json` in the pack holds the answers: don't let listeners open the pack folder.

Rebuilding the pack replaces its clips; answers from the old pack are moved to `answers_old_<date-time>` so they
are not scored against the new clips.

### 3. Get listeners to take it

1. Start the app: `soundlab gui` (or `soundlab gui --pack <folder>` for a pack elsewhere).
2. Open http://127.0.0.1:8766/blindtest on the same computer, give the listener headphones and leave them to it.
3. They type their name or initials, press Play, answer Real or Synthesised (plus how sure, and optionally
   what gave it away), and move on. Each listener gets their own random order. Answers are saved after every
   clip in `<pack>/answers/<name>.json`, so they can stop and continue later with the same name.
4. Aim for at least two listeners who did not tune the sounds. Answers are listed separately per name.

The app only listens on this computer (127.0.0.1). To test someone remotely, zip the pack and have them
run SoundLab themselves, then copy their `answers/<name>.json` back into your pack.

### 4. Read the result

Open http://127.0.0.1:8766/blindtest#results, or run `soundlab blindtest score` (add the pack folder if it is
not the default). Per sound and pooled: how many synthesised clips were spotted as synthesised, a 95 % range,
the verdict (pass = spotted under 60 %), how often real clips were called synthesised (someone who answers
"synthesised" to everything shows up here), and overall accuracy; then one line per listener.
`--json` for machine use. The same reading-it-honestly rules as below apply: one sound's result from a few
clips is only a hint, the pooled figure is the verdict.

### A starter pack from CC0 recordings

CC0 recordings (e.g. from Freesound) make a good first library: 19 of the recipe sounds can be covered with
2-3 CC0 takes each. Keep them outside the repo, with a note of each file's source and licence.

## A/B kit

### What you need

- **Engine renders**: any folder from `soundlab batch` (its `manifest.json` names each file by recipe), or
  sub-folders per sound.
- **Library samples (yours)**: real recordings of the same sounds, from a licensed sample library or your
  own recordings. Lay them out as one sub-folder per sound
  (`library/rain/take1.wav`) or flat files named `<sound>__anything.wav` (`rain__heavy_roof.wav`). The
  sound names must match recipe names (`rain`, `door_slam`, `creature_growl`, ...; dashes and case are
  ignored). These files stay outside the repo; only the listening set contains them.

### Build the listening set

```bash
python tools/checkpoint.py build --engine soundlab-out/v9 --library ~/my_library -o ~/checkpoint_set \
    --per-sound 4 --max-s 4 --seed 1 [--exempt blaster ui_click]   # exempt = stylised sounds
```

For every trial it picks one engine render and one library take of the same sound, trims leading silence,
caps the length (`--max-s`), fades the end, matches loudness (RMS of the active part, -20 dBFS, never
clipping), resamples to 48 kHz, and writes `trials/trial_XX_A.wav` + `trial_XX_B.wav` in shuffled order with
the engine on a random side. It also writes:

- `answers.csv`: the answer sheet (trial, sound, `engine_is (A/B)`, confidence 1-3, notes);
- `key.json`: the hidden key (which side was the engine, source files). **Keep it away from the listener**;
- `README.txt`: the listener's instructions.

### Run it

1. At least two listeners who did not tune the recipes, each scored separately (one
   answer sheet per listener; build with a different `--seed` per listener for a fresh order).
2. Same playback chain for everyone (headphones or decent speakers), no time pressure, replay freely.
3. For each trial: which one is the engine, A or B? Guess when unsure; never leave a row blank. Notes on
   what gave it away are the most useful part.

### Score it

```bash
python tools/checkpoint.py score --key ~/checkpoint_set/key.json --answers answers_listener1.csv
```

Prints, per sound and overall (physical sounds only; `--exempt` ones are listed but not counted): trials,
how often the engine was spotted, a 95 % range (Wilson interval), the chance of doing that well by
guessing (one-sided binomial), and pass/fail against the mark (`--pass-mark 0.6`). `--json` for machine use.

Reading it honestly:
- With 4 trials per sound, a single sound's result is only a hint (4/4 by guessing happens 1 time in 16).
  The overall figure is the verdict; use more trials for sounds that look like failures.
- If the overall result passes but the upper end of the 95 % range crosses 60 %, the tool says so: run more
  trials before calling it.
- A sound spotted by every listener is a failure to fix before deeper physics work.
- Record the item list, answers and "what gave it away" notes in `docs/checkpoint-results.md` (results only;
  audio stays outside the repo).
