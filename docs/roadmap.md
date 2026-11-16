# Roadmap

Status as of 0.9.0 (16 Nov 2026): steps 1-4 done, step 5 kit ready. 0.8 fixed footsteps, knocks and coins after a listening check; 0.9 grew the library to 36 recipes + 7 generic-block demos (wind, rain, glass break, sword swing and clash, metal clang, door creak and slam, punch, bullet impacts, gunshot, creature growl, paper and cloth rustle, splash) and added the blind checkpoint kit ([checkpoint.md](checkpoint.md)). The new sounds still need a listening check. Re-ordered after a review of the plan with an outside critique: the
physics models stay as they are, and the next work is what makes SoundLab useful for game sound: generic
blocks, macro sliders, variations and batch export, a recipe library, and a measured checkpoint.

Ground rules:
- **One implementation, Python/NumPy.** No PyTorch copy of the engine (two copies drift apart). If real-time
  playback is needed later, generate it from a single source (e.g. Faust), do not hand-port every block.
- **Physics models are kept as they are** (impact, modal, tubes, engine, foghorn, explosion, match, shaker,
  ...). Deeper physics only comes after the checkpoint, and only for sounds a game steers live.
- Offline render to WAV is the main use.

## Build order

| step | content | status |
|---|---|---|
| done before 0.6 | Recipe format + render CLI; physical blocks (exciters, modal, strings, tubes, horns, radiation, propagation, rooms); five reference sounds tuned to listening and reference clips; analysis and `soundlab match` | done (0.1-0.5) |
| 1 | **Generic block toolkit** alongside the physics models: band-limited oscillators (sine/saw/square/triangle/pulse, unison, FM), noise colours (+ sample-and-hold), resonant SVF (LP/HP/BP/notch/peak), envelopes (ADSR + breakpoints), distortion/waveshaping/bitcrush, pitch and filter sweeps (automation), random grains, delay/chorus/reverb, layering with offsets. Same JSON recipe format. For UI blips, magic, sci-fi, retro, power-ups. | **done in 0.6.0** (see [synth-blocks.md](synth-blocks.md)) |
| 2 | **Macro sliders per recipe** (size, distance, intensity, material where relevant) mapping onto the underlying controls. Macros are the primary interface; the physical controls stay underneath as the reference. | **done in 0.7.0** ([macros-and-variations.md](macros-and-variations.md)) |
| 3 | **Seeded variations** (`--variations N`: small randomisation of the controls a recipe declares as variable), **batch export**, sensible naming (`category_name_v01.wav`), fast audition loop. | **done in 0.7.0** |
| 4 | **About 30 recipes** including non-physical ones (impacts, footsteps, whooshes, UI, pickups, power-ups, magic, sci-fi, fire, water, engines, creatures, explosions), tuned by listening. | **done in 0.9.0** (43 recipes; 16 new ones awaiting a listening check) |
| 5 | **Stop-and-judge checkpoint** with a defined pass mark (below). | kit ready in 0.9.0 (`tools/checkpoint.py`, [checkpoint.md](checkpoint.md)); needs library samples + listeners |
| after 5 | Only then: deeper physics for live-steered sounds (engine following real revs, impacts that follow material and speed), real-time via a single-source generator if needed, optional ML-based analysis (stem separation, learned embeddings). | later, if the checkpoint says so |

## Checkpoint protocol (step 5)

A blind test, decided now so the checkpoint is not skipped.

1. **Material.** For each of the ~30 recipe sounds, one SoundLab render and one library sample of the same
   sound (licensed sample library or own recordings; never the private reference clips). Match loudness
   (LUFS within 1 dB), trim to similar length, same format (48 kHz WAV), no names in the file names.
2. **Listeners.** At least two people who did not tune the recipes (a tuner may take part but is scored
   separately). Headphones or decent speakers, same playback chain for all items.
3. **Procedure.** Shuffle all items (random order per listener). For each item the listener answers
   "synthesised or recorded?" and, optionally, notes what gave it away. No replay limit, no time pressure.
4. **Score.** For each listener and pooled: the share of SoundLab renders correctly identified as synthesised,
   reported separately for **physical sounds** (impacts, engines, fire, water, explosions, footsteps,
   foghorn, duck, match, ...) and for stylised ones (UI, magic, sci-fi, retro), where "sounds synthesised" is
   fine.
5. **Pass mark.** Physical sounds pass if **under 60 %** of SoundLab renders are correctly identified as
   synthesised (pooled over listeners; 50 % = indistinguishable). Each physical sound identified by every
   listener is listed as a failure to fix before any deeper physics work.
6. **Record.** Keep the item list, the answers and the "what gave it away" notes in `docs/checkpoint-results.md`
   (results only; the audio stays outside the repo).

The kit (0.9.0) automates 1 and 4 as an A/B variant ("which of these two is the engine?", same pass mark):
`python tools/checkpoint.py build ...` then `... score ...`; see [checkpoint.md](checkpoint.md).

## Fitting rules (adopted in 0.6.0)

- **Pitch first.** `soundlab match` measures the reference's fundamental, sets every pitch control in the fit
  (`f0`, `tone_hz`, `freq`), checks it on a render and locks it before fitting anything else
  (`--no-pitch-first` to opt out).
- **Hold-out check.** Every fitted recipe is scored on a second recording of the same sound it was not fitted
  to (`--holdout`), and rendered at a few slider positions for listening (`ear/` renders at macro positions). A fit report without a hold-out says so in bold.
- **Never ship reference clips or stems derived from them.** They live outside the repo
  (`soundlab-research/refs`); a test fails if any audio file is in the repo tree or tracked by git
  (`tests/test_no_reference_audio.py`, allowlist intentionally empty). Tuned settings are our own work.

## Known gaps carried over

- Engine: 200-400 Hz still ~6 dB hot against the V8 clip; the long `engine.json` rev to 4500 rpm has no
  reference at that speed.
- Explosion: one recipe cannot match a 2.5 s blast and a 7 s cinematic one; `explosion_small` covers the first.
- Matchbox shake ~4 dB bright at 6.4-12.8 kHz.
- Stereo/binaural output and per-layer propagation.
- 0.9 library: ambiences (wind, rain) are 8 s one-offs, not seamless loops; the sword clash and bullet
  impacts follow the CC0 refs' bright balance and need a check for tinny tops; the pipe
  clang rings long (a free pipe does) and is very tonal; door creaks vary hugely in real life (CC0 refs from
  ~26 Hz ticking to a ~1.1 kHz squeal), ours sits at 260 Hz.
