# Roadmap

What is planned next. For what exists today see the [README](../README.md).

Ground rules:
- **One implementation, Python/NumPy.** No PyTorch copy of the engine (two copies drift apart). If real-time
  playback is needed later, generate it from a single source (e.g. Faust), do not hand-port every block.
- **Physics models are kept as they are** (impact, modal, tubes, engine, foghorn, explosion, match, shaker,
  ...). Deeper physics only comes after the checkpoint, and only for sounds a game steers live.
- Offline render to WAV is the main use.

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
4. **Score.** Per listener and pooled: the share of SoundLab clips identified as synthesised, separately for
   **physical sounds** (impacts, engines, fire, water, explosions, footsteps, foghorn, duck, match, ...) and
   stylised ones (UI, magic, sci-fi, retro), where "sounds synthesised" is fine.
5. **Pass mark.** Physical sounds pass if **under 60 %** of SoundLab clips are identified as synthesised
   (pooled; 50 % = indistinguishable). A physical sound spotted by every listener is a failure to fix before
   any deeper physics work.
6. **Record.** Item list, answers and "what gave it away" notes in `docs/checkpoint-results.md` (results only;
   audio stays outside the repo).

## Sound to-dos

- **Gunshot**: pistol, rifle and shotgun sound too alike. The `calibre` choice should change the character
  (crack, body, length of the tail), not only small numbers.
- **Splash**: the water part is a bit fast; slow the crown/sheet and spray down.
- **Creature growl**: still reads as an object pushed along a floor rather than a living voice. Needs breath
  noise, irregular pulses, and throat and mouth shape changing over time.
- **Rain**: a bit static. Add slow variation (gusts of heavier drops, occasional near drops).

## Known gaps

- Engine: 200-400 Hz still ~6 dB hot against the V8 reference; the long `engine.json` rev to 4500 rpm has no
  reference at that speed.
- Explosion: one recipe cannot match a 2.5 s blast and a 7 s cinematic one; `explosion_small` covers the first.
- Matchbox shake ~4 dB bright at 6.4-12.8 kHz.
- The sword clash and bullet impacts follow the CC0 refs' bright balance; check them for tinny tops.
- The pipe clang rings long (a free pipe does) and is very tonal.
- Door creaks vary hugely in real life (CC0 refs from ~26 Hz ticking to a ~1.1 kHz squeal); ours sits at 260 Hz.
- Stereo/binaural output and per-layer propagation.

## Later, if the checkpoint says so

Deeper physics for live-steered sounds (engine following real revs, impacts that follow material and speed),
real-time via a single-source generator if needed, optional ML-based analysis (stem separation, learned
embeddings).
