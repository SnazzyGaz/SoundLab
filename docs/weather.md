# Weather and ambience from recording stems

`weather_stems` rebuilds rain, thunder, wind, ocean and ambience beds from stems of CC0 recordings (all credited in
[THIRD_PARTY.md](../THIRD_PARTY.md)). `soundlab/models/weather_stems.bin` holds only analysis numbers. Every
render uses new noise, so no sample of a recording is played back. Each recipe's default reproduces its rebuild.

| Recipe | Source | Stems | Macros |
|---|---|---|---|
| `rain` | light rural rain, steady rain on a hard surface, and heavy rain on leaves (24 s each); also rain on a roof and on a window | noise | `surface`, `intensity`, `distance`; Advanced: intensity `curve` |
| `rain_drips` | water dripping from a gutter (23.5 s) | events, noise | `intensity`, `distance` |
| `thunder` | a close crack (15.1 s) or a distant roll (14.0 s) | noise | `kind`, `distance` |
| `ocean_waves` | shore waves on the Baltic Sea (24 s) | noise | `intensity`, `distance` |
| `wind_breeze` | a gentle breeze and gusty wind in trees (24 s each) | noise | `intensity`, `distance`; Advanced: intensity `curve` |
| `wind_howl` | a moaning whistle wind (24 s) | noise | `intensity`, `distance` |
| `forest_day` | a spring afternoon forest (24 s) | noise | `intensity`, `distance` |
| `night_insects` | night crickets (24 s) | noise | `intensity`, `distance` |
| `city_street` | inner-city traffic (24 s) | noise | `intensity`, `distance` |
| `room_tone` | a quiet room with closed windows, light traffic outside (24 s) | noise | `intensity`, `distance` |

The earlier procedural `rain`, `rain_roof`, `rain_steady` and `wind_howl` recipes stay in `recipes/synth_model/`.
`wind_gust` and `wind_trees` stay as `air_stems` recipes.

Beds loop or run to any length (`length_s`, 0 = as recorded).

## Stems

Analysis runs at 44.1 kHz on each clip, scaled to a 0.89 peak.

1. **Events** (rain drips). Transients that stand over a running median of the high-passed envelope are cut out
   with short fades. Each event is stored as its own band envelopes (76 bands at 1/8 octave) and rebuilt from
   noise. No waveform is stored.
2. **Noise.** The residual: each band's own amplitude envelope (76 raised-cosine bands at 1/8 octave, every
   1 ms) imposed on fresh noise carriers, then the residual's 1/6-octave colour.
3. **Treble ceiling.** One static zero-phase cut above 10 kHz wherever the rebuild sits over the recording (it
   never boosts, so it can only remove hiss).

Measured spectral mean-absolute error of each rebuild against its recording is in the tests; waveform
correlation against a noise-vocoded bed is not a useful metric.

## Fresh takes and any length

Any variation (the `seed` macro), a `length_s` other than the recording's, or a non-zero `bias` plays a fresh
take:

- The recording is cut at quiet points into forward-only pieces of random length (1.2-2.4 s; longer for waves
  and traffic so a gesture stays whole). Cuts avoid any recorded transient within 60 ms of either end of a
  crossfade.
- `bias` 0: every piece once, in a new order (then again in another order for longer takes). `bias` != 0:
  pieces drawn by their transient activity, with no piece back within three picks.
- Pieces join with 100 ms power-complementary crossfades of the noise envelopes.

Nothing is averaged, nothing is held or frozen, no band is pitch-shifted, nothing is time-squeezed and no
smoothing is added. `speed_cents` is tape speed of the whole render (pitch and time together).

## Intensity over time

Where a family has recordings at more than one intensity (open-ground rain: light, steady, heavy), a curve
crossfades those takes with equal-power weights. The recordings themselves are not stretched or frozen.

- `curve` `constant` (default): the `intensity` slider alone. At a value that matches one stored recording
  exactly, the default take is that recording's rebuild.
- `curve` `linear`: intensity moves from `curve_start` to `curve_end` over the length of the take.
- `curve` `swell`: a raised-cosine rise from `curve_start` to `curve_peak` at `curve_peak_at` (a fraction of the
  length), then a raised-cosine fall to `curve_end`.

Roof and window rain are separate recordings, not points on the open-ground intensity scale, so the intensity
slider does not crossfade them. Thunder is a one-shot (`kind` near or distant), not a looping bed.

## Macros and controls

Macros run 0..1 with 0.5 = the recording, unless a recipe says otherwise (open-ground rain defaults to light).

- `surface` (rain): open ground, a roof, or a window.
- `intensity`: for open-ground rain, light (0) .. heavy (1), crossfading the stored recordings. For a family
  with one recording it is reserved for a later grouping and does not change the bed on its own.
- `distance`: 0.7 .. 2. Further: -6 dB per doubling (master gain), high shelf -5 dB per doubling at 2.5 kHz and
  a soft low-pass at 14 kHz / sqrt(d). Nearer: low shelf +3 dB per halving.
- `kind` (thunder): near crack or distant roll.
- `curve`, `curve_start`, `curve_end`, `curve_peak`, `curve_peak_at`: Advanced, as above.
- `seed` (hidden): reseeds the noise; looping sources play a fresh take. **Variation** 0 is the reference.

Length is the model control `length_s` under Advanced (0 = as recorded).
