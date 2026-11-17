# Game-ready export: loudness, tails, DC, loops, stereo, dither

`soundlab.render()` returns the raw mix, peak-normalised to -1 dBFS. Peak level says little about how loud a
sound is: at the same peak, a coin pickup and a match strike differ by about 20 dB in loudness. Every export
(`soundlab render`, `soundlab batch`, the slider app's Play/Export) therefore goes through
`soundlab.finish` by default:

0. **Ring-out** (v0.22). Sources stop at the recipe's `duration` (with the master fade-out), but rooms,
   reverbs, ringing resonators and echoes keep running until they have decayed: `render(ring_out=True)`
   extends the render by the processing blocks' own tails (room RT60, reverb decay, modal/string T60, echo
   distance; `soundlab/models/tails.py`), capped at 20 s, and the trim below cuts the silence. Before v0.22 a
   reverb was cut wherever the recipe ended (the elephant's hall 2.3 s early, magic_shimmer 2.2 s,
   explosion_small lost its 1.1 km hill echo). `--no-trim` keeps exactly the recipe length. A plain
   `soundlab.render()` is unchanged.
1. **Tail trim.** Fixed recipe lengths leave near-silent tails (a glass tap in a room: 2.2 s of nothing). The
   file ends once the 10 ms RMS level has fallen 60 dB under its loudest moment, plus 50 ms and a 30 ms
   cosine fade. Reverb tails are kept down to -60 dB, where they are inaudible in a game mix. Minimum length
   0.1 s. `--no-trim` keeps the full length.
2. **Loudness** (ITU-R BS.1770-4 K-weighting, implemented in `soundlab/finish.py`, no extra dependency).
   - One-shots are measured on their **maximum momentary loudness** (400 ms window). Integrated loudness is
     unreliable for a 0.3 s hit (it depends on the tail and the gate).
   - Loops and the `ambience` category use **gated integrated loudness**.
   - **True peak** (4x oversampled) stays under **-1 dBTP**. If the target needs more, up to 6 dB of
     look-ahead peak limiting is applied to the transients; beyond that the sound stays under its target
     rather than being squashed (very spiky sounds such as a ui click or a bullet hit end a few dB low).
   - `--level peak` keeps the raw -1 dBFS peak level.
3. **Targets per category** (LUFS; one-shots max momentary, ambience integrated). Defaults follow common
   game practice: UI and pickups under the action, weapons and explosions on top, beds well below:

   | category | target | category | target |
   |---|---|---|---|
   | ui | -22 | impact, glass, creature | -16 |
   | pickup, coin, foley | -20 | combat | -15 |
   | powerup, magic, whoosh, footstep, water | -19 | weapon | -12 |
   | animal | -18 | explosion | -11 |
   | scifi, door, vehicle | -17 | ambience (integrated) | -24 |

   Other categories: -18. A recipe can set its own with `"loudness_lufs": -21`. In a game, set the mix
   with the engine's volume controls; these targets only make the files start out consistent.
4. **DC blocker.** Every render passes a 7 Hz second-order high pass on the master (`"master":
   {"dc_block_hz": 0}` turns it off). Offsets cost headroom and click at edits.

## Seamless loops

Stationary recipes are tagged `"loop": {"lead_s": 1.0, "xfade_s": 0.5}`: `rain`, `wind`, `wind_gap`,
`fire`, `water`, `drip`, `engine_idle`.

```bash
soundlab render recipes/rain.json --loop 10 -o rain_loop.wav
soundlab batch recipes -o out/ --loop 20          # loopable recipes as loops, the rest as one-shots
```

How: the recipe is rendered for lead-in + loop + crossfade; the lead-in (onsets, build-up of a fire) is
dropped; the part after the loop end is crossfaded into the start with equal-power curves, so the last sample
runs straight into the first. Loops are not trimmed and are levelled on integrated loudness. Names get a
`loop<seconds>s` tag (`ambience_rain_loop10s_v01.wav`). In the slider app, loopable recipes show a
"seamless loop" option; Play then repeats the loop so you can listen to the seam.

## Stereo beds (v0.22)

A game pans one-shots itself, so they stay mono. Ambience beds are played as stereo files. Loops of recipes
with a `"stereo"` config (rain, fire, water, drip, wind, wind_gap, engine_idle) export in stereo by default;
`--stereo` / `--mono` (CLI and batch), `stereo=True/False` (Python) and the slider app's "stereo" box override.
File names get a `_stereo` tag.

```json
"stereo": {"mode": "seeds", "width": 0.8, "mono_below_hz": 150}
```

- `seeds`: a second take with another seed, so each drop, crackle or bubble lands somewhere else on each side
  (a spaced microphone pair over a field of independent sources). Rain, fire, water, drip.
- `decorrelate`: one take and a copy through an allpass with a smooth random group delay (1-20 ms; flat
  magnitude, so neither side and not the mono fold-down is coloured). Gusts and engine rhythm stay common to
  both sides. Wind, engine idle.
- `width` is 1 - the inter-channel correlation above `mono_below_hz` (measured: rain 0.20 for width 0.8, wind
  0.40 for 0.6). Below `mono_below_hz` both channels are identical (in-phase bass, safe mono fold-down).
- Every step is circular for loops, so the stereo loop is as seamless as the mono one.
- Loudness follows BS.1770 for stereo (channel powers summed), so a stereo bed at -24 LUFS integrated has the
  same loudness as a mono one at -24.

## Dither

16-bit files (and the slider app's preview) get TPDF dither of +-1 LSB, so fades and reverb tails decay into
a steady noise floor near -96 dBFS instead of truncation distortion. The dither is seeded, so the same render
always gives the same file. 24-bit (the default) and 32-bit float exports are not dithered.

