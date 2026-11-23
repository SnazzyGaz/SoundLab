# Sequences

A sequence chains library entries on a timeline: a walk across gravel, a knock-knock-knock, a door opened,
closed and locked, a burst of shots with their casings, a reload. It renders offline into one file, or into
numbered one-shots with a cue list for a game engine or a DAW. The engine is `soundlab/sequence.py`; the presets
are in `sequences/`.

## Presets

| preset | what it is |
|---|---|
| `walk` | N footsteps at a walking rate (110 steps per minute). Surface, shoe and pace are the Footsteps entry's own choices. Steps alternate left and right. |
| `run` | Running footsteps on concrete (165 steps per minute). Runs were recorded on concrete and grass; other surfaces play their walk steps at a running rate. |
| `knock` | Knocks on a wooden door, any count. Each knock is one real hit cut from the recorded knock-knock-knock (`hit` below). |
| `door_open_close` | Wood door: latch, swing open, closed. |
| `door_open_close_lock` | The same, then the deadbolt locked. |
| `gunfire` | Pistol shots, each followed by a brass casing. |
| `reload` | Magazine out and in, then the slide racked. The magazine recording has out and in in one gesture, so they are one step, using the take that has both. |

## Format

```json
{
  "soundlab": 1, "type": "sequence", "name": "walk", "title": "Walk", "category": "impact", "seed": 1,
  "rate": {"spm": 110},
  "humanise": {"timing_ms": 15, "level_db": 1.5, "alternate": 0.2},
  "level": "lufs",
  "steps": [
    {"entry": "footstep", "macros": {"surface": "gravel", "shoe": "boot"}, "repeat": 8}
  ]
}
```

- `rate`: the base interval between events, as `{"interval": seconds}` or `{"spm": steps per minute}`.
- `humanise`: `timing_ms` moves each event by a random amount within ± that many milliseconds. `level_db`
  changes each event's level by up to ± that many dB. `alternate` swings the events of an alternating step left
  and right by that amount (pan units, 0 to 1).
- `seed`: picks the takes and the jitter. The same seed always gives the same file.
- `level`: the level of the mix. `lufs` (the default) uses the loudness for the sequence's `category`, as single
  exports do (see [export.md](export.md)). `peak` gives a raw -1 dBFS peak.
- `category`: picks the loudness target.

Step keys:

| key | meaning |
|---|---|
| `entry` | a library id (`footstep`, `doors/knock_wood`, `pistol_shot`; `my/<name>` for your saved recipes in the app) |
| `macros` | the entry's controls, as in the slider app (`{"surface": "snow"}`, `{"part": "swing"}`, `{"force": 0.7}`) |
| `overrides` | exact model values (`{"0.hit": -2}`), like the app's Advanced panel |
| `repeat` | number of events in this step (default 1) |
| `interval` | seconds between this step's events (default: the base rate) |
| `at` / `offset` / `gap` | when the step starts. `at` is seconds from the start. `offset` is seconds after the previous step's last event starts (default: one base interval). `gap` is seconds after the previous step's last event ends; a negative gap overlaps. The first step starts at 0. |
| `gain_db`, `pan` | level, and position from -1 (left) to 1 (right) |
| `alternate` | left/right alternation for this step (default: on for footsteps, off otherwise) |
| `take` | `"auto"` (the default) or a variation number used for every event (0 = the entry as designed) |
| `key` | optional number the step's random draws are keyed on (default: its position). The app gives every step one, so moving a step keeps its takes and jitter. |

## Takes

With `take: "auto"` every event draws its own take, and an event never uses the same take as the previous event
of the same entry, even when that event was in an earlier step. For the recording-stem families (footsteps,
doors, foley, impacts, weapons) the engine also checks which recorded gesture each take would play, without
rendering it. If the recording has several gestures, consecutive events play different ones. If it has only one
(a door close, say), each event is still a different variation, which is a fresh remix of the same real stems.
Takes are never averaged, stretched or frozen.

The cue list's `take` is the entry's variation number. The same number in the slider app's Variation box, or
`soundlab render recipes/<entry>.json --variation N`, gives the same take, provided the step's own macros and
overrides (such as `0.hit`) are also applied.

The draws come from separate generators keyed on the sequence seed, the step (its `key`, else its position) and
their purpose (takes, timing, level). Changing the rate, jitter, gain or pan, or adding events at the end of a step, keeps every
earlier event's take.

### One hit of a knock

`door_stems` has a `hit` control. `-1` (the default) plays the whole gesture, exactly as before. `0, 1, ...`
plays only that hit of a multi-hit gesture. That is a window of the recording from just before the hit to just
before the next one; the last hit keeps its whole tail. `-2` picks the hit with the variation's generator. The
`knock` preset uses `-2`, so any number of knocks can be built from the recorded triple, with no two consecutive
knocks using the same hit. The cut ends just before the next hit, so earlier hits have a shorter tail than the
last one.

## Rendering and speed

Each event renders the way the slider app plays the entry on its own: sources ring out, the silent tail is
trimmed, and the loudness gain for the entry's category is applied. A step at gain 0 therefore sits at the level
the entry has alone. The peak limiter is not applied per event; it runs once, on the finished mix, so transients
are not limited twice. Each event's first sample within 30 dB of its peak lands on the timeline. If timing jitter or a
long lead-in would start an event before 0, the whole timeline moves later by the same amount.

Event audio is cached by its concrete recipe and seed (`SOUNDLAB_SEQ_CACHE_MB`, default 256). Edits that only
change timing, gain, pan or jitter re-mix the cached events. A changed step, a new step or a new seed renders
only the events that changed. With more than one thread (`SOUNDLAB_THREADS`; the slider app uses every core),
the uncached events render side by side. The samples are the same either way.

## Export

- One mixed file: WAV (24-bit), plus MP3 with ffmpeg. The mix is mono unless an event is panned, in which case
  it is stereo.
- Numbered one-shots: a folder `<name>_oneshots/` with `001_footstep_t78500.wav`, ... (each event as rendered,
  before its step gain and pan, peaking at -1 dBFS at most) and `cues.csv`.

`cues.csv` columns:

| column | meaning |
|---|---|
| `index` | event number, in timeline order per step |
| `time` | the event's onset in the mix (s) |
| `start` | where the one-shot file starts in the mix (s) |
| `entry`, `title` | library id and plain name |
| `take` | the entry's variation number |
| `step` | the step it came from (1-based) |
| `gain_db` | gain to apply to the one-shot to rebuild the mix, including the mix's own level change |
| `pan` | -1 left .. 1 right (constant-power) |
| `seconds`, `file` | one-shot length and file name |

Placing every file at `start` with `gain_db` and `pan` rebuilds the mix, apart from the mix's final tail trim and
any limiting.

## Command line

```bash
soundlab sequence --list
soundlab sequence walk -m surface=snow -m shoe=boot --repeat 12 --spm 100 -o walk_snow.wav --mp3
soundlab sequence knock --repeat 5 --oneshots -o knock5.wav      # knock5.wav + knock5_oneshots/cues.csv
soundlab sequence gunfire --seed 7 --jitter-ms 40 -o burst.wav
soundlab sequence my_sequence.json --oneshots -o out/scene.wav
soundlab render my_sequence.json -o scene.wav                     # render also takes a sequence file
```

`-m NAME=VALUE` sets that control on every step whose entry has it. `--repeat` sets the count of every step
that repeats.

## Slider app

The **Sequence** tab, next to **Sounds**, opens the sequence panel:

- **Start from**: a preset, or your saved sequences. **Empty** starts with no steps.
- Click a sound in the library to add it as a step. Each step row has its start (after the previous step
  starts, after it ends, or at a fixed time), the number of events and their spacing, gain, pan, a fixed take
  (blank = auto), ⚙ for the entry's own controls (Surface, Shoe, Pace, Intensity, ...), ↑ ↓ to reorder and ✕
  to remove.
- Rate (steps per minute or seconds), Timing jitter, Level jitter, Left / right, and Variation (the sequence
  seed: ◀ ▶ 🎲).
- Every change re-renders after a 60 ms pause. A newer change cancels the render in flight, as with the sliders.
  The waveform marks each event with its number. The status line says how many events were rendered and how
  many came from the cache.
- Export writes `<out>/exports/sequences/<name>.wav` (plus MP3, plus one-shots and cues.csv). **Save
  sequence** writes `<out>/sequences/<name>.json`, which then appears under Start from.

## Not yet

Long beds that follow an intensity curve (rain from light to heavy) are not part of sequences yet. They need real
recordings at several intensities to crossfade, which waits on the weather rebuild; see
[roadmap.md](roadmap.md).
