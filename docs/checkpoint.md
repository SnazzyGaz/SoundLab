# Blind checkpoint kit (roadmap step 5)

The stop-and-judge test agreed in the plan review: can a listener tell SoundLab renders from real recordings
of the same sound? Pass mark for **physical sounds: under 60 % correctly identified** (50 % = pure guessing).

## What you need

- **Engine renders**: any folder from `soundlab batch` (its `manifest.json` names each file by recipe), or
  sub-folders per sound.
- **Library samples (yours)**: real recordings of the same sounds, from a licensed sample library or your
  own recordings. Never the private reference clips. Lay them out as one sub-folder per sound
  (`library/rain/take1.wav`) or flat files named `<sound>__anything.wav` (`rain__heavy_roof.wav`). The
  sound names must match recipe names (`rain`, `door_slam`, `creature_growl`, ...; dashes and case are
  ignored). These files stay outside the repo; only the listening set contains them.

## Build the listening set

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

## Run it

1. At least two listeners who did not tune the recipes; a tuner may take part but is scored separately (one
   answer sheet per listener; build with a different `--seed` per listener for a fresh order).
2. Same playback chain for everyone (headphones or decent speakers), no time pressure, replay freely.
3. For each trial: which one is the engine, A or B? Guess when unsure; never leave a row blank. Notes on
   what gave it away are the most useful part.

## Score it

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
