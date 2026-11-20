# Fire, crackle and electricity from recording stems (v0.33)

`fire_stems` rebuilds fires, embers, a match, a gas stove, a sparkler, a fuse and three electrical sounds from stems
of ten CC0 recordings (all credited in [THIRD_PARTY.md](../THIRD_PARTY.md)). `soundlab/models/fire_stems.bin`
(about 5 MB) holds only analysis numbers. Every render uses new noise, so no sample of a recording is played back.
Each recipe's default reproduces its rebuild; the fingerprints in `tests/data/fire_fingerprints.json` check this.

| Recipe | Source | Stems | Macros |
|---|---|---|---|
| `campfire` | small campfire, dense crackle, 12 s | events (131), noise | `size`, `distance`, `seed` |
| `large_blaze` | large fire roaring, 12 s | events (125), noise | `intensity`, `distance`, `seed` |
| `embers` | slow-burning wood charcoal ticking and popping, 12 s | events (140), noise | `pop_density`, `seed` |
| `match_strike` | match struck, flaring and burning, 6.2 s | events (29), noise | `force`, `seed` |
| `gas_stove_ignite` | gas stove: knob, 7 igniter clicks, burner catching, 7.5 s | events (9, fine resolution), noise | `clicks`, `flame`, `seed` |
| `sparkler` | sparkler fizzing, 12 s | events (35), noise | `pop_density`, `distance`, `seed` |
| `fuse` | firework fuse burning, 9 s | noise | `burn_speed`, `seed` |
| `transformer_hum` | electrical transformer humming, 12 s | harmonics (66), noise | `load`, `seed` |
| `electric_arc` | Jacob's ladder arc, 12 s | harmonics (99), noise | `intensity`, `seed` |
| `fluorescent_flicker` | fluorescent tube flickering, 12 s | harmonics (59), lines (5), noise | `clang_density`, `seed` |

The procedural `fire`, `match`, `matchbox` and `mains_hum` recipes stay as physical fallbacks.

## Stems

Analysis runs at 44.1 kHz on each clip, scaled to a 0.89 peak.

1. **Events** (fires, embers, match, gas stove, sparkler). Transients that stand 8-12 dB over a 400 ms running
   median of the high-passed (1.5 kHz) envelope, at least 8-30 ms apart, are cut out with short fades. Each event
   is stored as its own band envelopes (76 bands at 1/8 octave, every 0.5 ms, 0.5 dB steps) over its window plus
   10 ms either side, and rebuilt from noise with 8 band-limiting passes. The gas stove's igniter clicks are stored
   every 0.125 ms and rebuilt with nothing before each click's own start, so their attacks stay sharp. One static
   1/6-octave EQ per take (60 numbers) matches the rebuilt event layer to the recording's events. No waveform is
   stored.
2. **Harmonics** (transformer, arc, tube). The fundamental is tracked by subharmonic summation with a Viterbi path
   within a set range (transformer 100-140 Hz, arc and tube 90-110 Hz). Each harmonic is demodulated at k x the
   tracked phase and stored as amplitude (0.5 dB steps) and phase (1/256 cycle) every 4 ms; the transformer uses a
   27 Hz demodulation bandwidth so its slow hum wobble is kept. Rebuild: amp x cos(k x phi + phase track).
3. **Lines** (tube). Persistent narrow lines off the harmonic series, demodulated the same way.
4. **Noise.** The residual after events, harmonics and lines: each band's own amplitude envelope (76 raised-cosine
   bands at 1/8 octave, every 1 ms) imposed on fresh noise carriers, then the residual's 1/6-octave colour. The
   fuse and the arc's crackle live entirely in this layer.
5. **Treble ceiling.** One static zero-phase cut above 10 kHz wherever the rebuild sits over the recording (it never
   boosts, so it can only remove hiss). The tube matches the recording two-way above 3 kHz (boost capped at
   +6 dB) to keep its narrow resonances; its recording is low-passed at 17.5 kHz first (the preview's floor above
   that is removed from both).

Measured: fingerprint distance from each recipe's default to its rebuild 0.02-0.15 dB; rebuild to recording
0.58-1.38 dB.

## Fresh takes

Any variation (the `seed` macro), a `length_s` other than the recording's, or a non-zero `bias` plays a fresh take:

- The recording is cut at quiet points into forward-only pieces of random length (0.6-1.5 s; 1.2-2.8 s for the
  blaze so its slow swells stay whole). For sources without separate events, cuts avoid any recorded transient
  within 60 ms of either end of a crossfade.
- `bias` 0: every piece once, in a new order (then again in another order for longer takes). `bias` != 0: pieces
  drawn by their transient activity (weight exp(bias x z-score)), with no piece back within three picks.
- Pieces join with 100 ms power-complementary crossfades: noise envelopes and harmonic amplitudes in power, f0 and
  line frequencies linearly, phase tracks shifted by whole cycles to meet across the crossfade.
- Each real event plays with the piece it belongs to (crossfades split at their middle); a piece used twice brings
  its events twice.

Nothing is averaged, nothing is held or frozen, no band is pitch-shifted, nothing is time-squeezed and no smoothing
is added.

## Macros and controls

Macros run 0..1 with 0.5 = the recording.

- `size` (campfire): larger = bed +2.5 dB, crackles +1.5 dB, about 1.3x the crackles, low shelf +2 dB at 250 Hz;
  smaller is the mirror image.
- `intensity` (blaze): bed +-2 dB, crackles +-1.5 dB, crackle density +-0.25, low shelf +-1.5 dB.
- `distance` (campfire, blaze, sparkler): 0.7 .. 2. Further: -6 dB per doubling (master gain), high shelf -5 dB
  per doubling at 2.5 kHz and a soft low-pass at 14 kHz / sqrt(d). Nearer: low shelf +3 dB per halving.
- `pop_density` (embers): fewer = a fresh take drawn towards the quieter stretches (bias -1, about 15% fewer pops);
  more = the pops of a second, independent take added with probability density - 1 (up to 1.4x).
  (sparkler): density 0.6 .. 1.6, the fizz +1 dB at the top.
- `density` (control): below 1 the events thin out with keep probability density x w / mean(w), where w is the
  event's energy re the median to the power 0.3, so small ticks go first and the big pops mostly stay.
- `force` (match): harder = strike crackles +2.5 dB, bed +1 dB; softer = crackles -4 dB, bed -1.5 dB, high shelf
  -3 dB at 3 kHz.
- `clicks` (gas stove): 4 .. 10 igniter clicks before the burner catches (7 as recorded; the control takes 4 .. 13).
  Fewer drop recorded click cycles; more repeat recorded cycles, never one twice in a row.
- `flame` (gas stove): burner -2.5 .. +2.5 dB from when it catches (applied to the bed's envelopes before the bed
  takes the recording's overall colour, so the burner's balance against the gas hiss moves by about +-0.9 dB).
- `burn_speed` (fuse): tape speed -/+300 cents (pitch and time together); slower adds a 2nd-order roll-off at
  16 kHz x speed x 0.8 (`slow_mode` `tape_lp`) because the recording's ~16 kHz edge moves down with the tape.
- `load` (transformer): heavier = harmonics +2 dB, tilt +0.03, bed +1.5 dB; lighter = harmonics -3 dB, tilt -0.15,
  bed -3 dB.
- `intensity` (arc): stronger = harmonics +1 dB and a fresh take drawn towards the crackliest stretches (bias +1);
  weaker = harmonics -2 dB, tilt -0.1, bed -2 dB, envelope contrast 0.85.
- `clang_density` (tube): a fresh take drawn towards the quieter (bias -1.2) or busier (bias +0.6) stretches.
- `seed`: reseeds the noise and the rebuilt events; looping sources play a fresh take.

Variations and macros are screened against the default render (at the same tape speed): no new narrow tones,
spectral bumps, comb ripple, wobble, frozen stretches, shimmer, hiss, buzz, chop or flapping, and no clicks at the
joins of a fresh take.

## Next

Gestures for fire (ignite, burn, die down), a torch, stereo from paired takes.
