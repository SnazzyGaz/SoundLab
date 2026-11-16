# Game-ready export: loudness, tails, DC, loops

`soundlab.render()` returns the raw mix, peak-normalised to -1 dBFS. Peak level says little about how loud a
sound is: at the same peak, a coin pickup and a match strike differ by about 20 dB in loudness. Every export
(`soundlab render`, `soundlab batch`, the slider app's Play/Export) therefore goes through
`soundlab.finish` by default:

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
