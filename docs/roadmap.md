# Roadmap

What is planned next. For what exists today see the [README](../README.md).

Ground rules:
- **One implementation, Python/NumPy.** No PyTorch copy of the engine (two copies drift apart). If real-time
  playback is needed later, generate it from a single source (e.g. Faust), do not hand-port every block.
- **Physics models are kept as they are** (impact, modal, tubes, engine, foghorn, explosion, match, shaker,
  ...). Deeper physics only comes after the checkpoint, and only for sounds a game steers live.
- Offline render to WAV is the main use.

## Now: interaction models and resynthesis (v0.17 to v0.19)

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
6. **Record.** Item list, answers and "what gave it away" notes in `docs/checkpoint-results.md` (results only;
   audio stays outside the repo).

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
- Explosion: one recipe cannot match a 2.5 s blast and a 7 s cinematic one; `explosion_small` covers the first.
- Matchbox shake ~4 dB bright at 6.4-12.8 kHz.
- The sword clash and bullet impacts follow the CC0 refs' bright balance; check them for tinny tops.
- The pipe clang rings long (a free pipe does) and is very tonal.
- Door creaks vary hugely in real life (CC0 refs from ~26 Hz ticking to a ~1.1 kHz squeal); ours sits at 260 Hz
  with slip-rate jumps of 0.6-1.6x.
- Wind recorded on a bare microphone is mostly sub-60 Hz buffeting; ours only gets that from the `rumble`
  control, which stays low by default.
- The `room` block's early reflections colour close, broadband sounds slightly (comb lines); close foley
  recipes keep its level low.
- Stereo/binaural output and per-layer propagation.

## Later, if the checkpoint says so

Deeper physics for live-steered sounds (engine following real revs, impacts that follow material and speed),
real-time via a single-source generator if needed, optional ML-based analysis (stem separation, learned
embeddings).
