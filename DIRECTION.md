# Direction (v0.25)

Where SoundLab stands, and what to do next, in order.

## State

- 71+ blocks, 60+ recipes, fitting and resynthesis, slider app, blind-test kit (pack builder, scorer and a
  self-contained web page, `tools/blindtest_artifact.py`).
- v0.21.2 fixed crashes (matchbox variations, short renders, late events), physics (bar and membrane modes,
  string/tube tuning, exact Doppler, stiff friction), real 24-bit WAV, and GUI recipe ids.
- v0.22 added ring-out tails, stereo beds and dither.
- v0.23 refitted water and wind to real recordings (loss 14.8 -> 3.9 and 9.7 -> 3.4).
- v0.24 fixed water combing, wind flicker, fire swell, splash body and match boom, each pinned by a measured cue.
- v0.25 fitted ten new-domain recipes to reference recordings with hold-out takes; v0.25.1-0.25.3 added the
  Godot walk demo, a strong-wind recipe, rain without excess crackle, a groaning door creak and per-surface
  footstep eq.

## Next, in order

1. **Blind checkpoint.** Run the blind test ([docs/checkpoint.md](docs/checkpoint.md)) with listeners who did
   not tune the recipes; results stay outside the repository. Sounds most in need of work next: paper,
   crumple, door creak, drill, engine.
2. ~~**Export what a game actually loads.**~~ Done in v0.22: stereo beds for the 7 loopable ambiences
   (two-seed or allpass-decorrelated, mono bass), ring-out of reverb/resonator/echo tails before the trim,
   seeded TPDF dither on 16-bit. See [docs/export.md](docs/export.md).
3. **Variation quality.** Games play 10-30 takes of a footstep or impact. Measure that v02..vN differ enough
   to avoid machine-gunning but keep their identity (pairwise spectral/envelope distance band), and add it
   to the tests.
4. **Refit the untuned waveguides** (foghorn pipe, engine exhaust, horn) with `tune=True`, then re-run
   `soundlab match` on their references.
5. **Engineering.** CI on Windows/macOS/Linux; optional numba for the per-sample loops (`ep_friction`,
   moving-cutoff SVF, limiter release) so the slider app stays responsive; anti-aliased Doppler read in
   `passby`.
6. **Real-time in Godot.** `demos/godot_walk` (v0.25.1): a first-person yard in Godot 4 where wind/rain
   parameters crossfade SoundLab beds, an indoor area low-passes them, 3D emitters loop fire/stream/howl and
   one-shot banks play footsteps per surface, door, knocks, clangs and splashes (`tools/godot_export.py`
   renders the set). Next: port the cheap, parametric models (wind, rain, water, fire, engine, footsteps,
   impacts) to a GDExtension that renders live from the same JSON recipes and macros, so game state drives
   the sliders; heavy models stay offline as exported banks.

## Not doing

PyTorch port, deeper physics before the blind checkpoint says which sounds need it (same rules as
[docs/roadmap.md](docs/roadmap.md)).
