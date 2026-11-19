# Machines and vehicles from recording stems (v0.32)

`machine_stems` rebuilds engines, pass-bys, motors and household machines from stems of eleven CC0 and
public-domain recordings, plus two more CC0 takes for gestures (all credited in [THIRD_PARTY.md](../THIRD_PARTY.md)).
`soundlab/models/machine_stems.bin` (about 6 MB) holds only analysis numbers. Every render uses new noise, so no
sample of a recording is played back. Each recipe's default reproduces its rebuild; the fingerprints in
`tests/data/machine_fingerprints.json` check this.

| Recipe | Source | Stems | Macros |
|---|---|---|---|
| `petrol_rev` | V8 idle into a rev and back, 12 s | harmonics, lines, noise | `throttle`, `seed` |
| `diesel_idle` | diesel van idling, 10 s | harmonics, noise, cycle pulses | `load`, `seed` |
| `motorbike_passby` | 650 cc motorbike passing, 9.5 s | harmonics, noise | `distance`, `speed`, `seed` |
| `car_drive_by` | car approaching, passing, receding, 12 s | harmonics, noise | `distance`, `speed`, `seed` |
| `servo_motor` | small electric motor whine, 12 s | harmonics (to 19 kHz), noise | `speed`, `seed` |
| `power_drill` | cordless drill trigger runs (revving, as recorded), 12 s | harmonics, lines, noise | `speed`, `seed` |
| `ac_unit` | window air conditioner, 10 s | harmonics, noise | `fan`, `seed` |
| `clock_tick` | clock ticking, 10 s | events, noise | `tick_rate`, `room`, `seed` |
| `tram` | tram approaching and passing, 12 s | harmonics, noise | `distance`, `speed`, `seed` |
| `lawn_mower` | petrol push mower running, 12 s | harmonics, noise, cycle pulses | `throttle`, `seed` |
| `sewing_machine` | sewing machine stitching, 11 s | harmonics, events, noise | `speed`, `seed` |
| `power_drill_on_off` | gesture: start, hold 3 s, stop | real spin-up / steady / spin-down | `speed`, `seed` |
| `power_drill_two_speeds` | gesture: start at 0.8, change up to 1.15, stop | real spin-up / steady / spin-down | `speed`, `seed` |
| `servo_motor_on_off` | gesture: start, hold 5 s, stop | steady run, ramps | `speed`, `seed` |
| `sewing_machine_on_off` | gesture: start, hold 5 s, stop | steady run, ramps | `speed`, `seed` |
| `lawn_mower_on_off` | gesture: pull-start, hold 4 s, shut down | real start / steady / shut-down | `speed`, `seed` |

The procedural `engine`, `engine_idle`, `engine_blips`, `car_passby`, `drill_motor`, `fan` and `mains_hum` recipes
stay as physical fallbacks.

## Stems

Analysis runs at 44.1 kHz on each clip, scaled to a 0.89 peak.

1. **Events** (clock, sewing machine). Transients that stand 8-10 dB over a 400 ms running median of the
   high-passed (1.5 kHz) envelope are cut out with short fades. Each event is stored as its own band envelopes
   (76 bands at 1/8 octave, every 0.5 ms, 0.5 dB steps) over its window plus 10 ms either side, and rebuilt from
   noise with 8 band-limiting passes. No waveform is stored.
2. **Harmonics.** The fundamental is tracked by subharmonic summation (8192-sample frames, 256 hop, 12
   subharmonics) with a Viterbi path that penalises jumps over 60 cents. Each harmonic k is demodulated at
   k x the tracked phase with a Gaussian low-pass (bandwidth 0.3 x the 5th-percentile f0, 3-40 Hz), kept only
   where it stands 6 dB over the half-harmonic floors (20 ms soft gate), and stored as amplitude (0.5 dB steps) and
   phase (1/256 cycle) every 4 ms. Rebuild: amp x cos(k x phi + phase track). The phase track carries the real
   wobble; no smoothing is added (smoothed tracks rang as steady tones).
3. **Lines** (petrol engine, drills). Persistent narrow lines off the harmonic series (present in 15 % of frames)
   are tracked within +-3 % of their nominal frequency and demodulated the same way.
4. **Noise.** The residual after events, harmonics and lines: each band's own amplitude envelope (76 raised-cosine
   bands at 1/8 octave, 30 Hz-20 kHz, every 1 ms) imposed on fresh noise carriers with four band-limiting passes,
   then the residual's 1/6-octave colour.
5. **Cycle pulses** (diesel, mower). Per octave band (125 Hz-16 kHz), the residual's envelope folded on each firing
   cycle (24 phase bins, smoothed over 0.7 cycles). At render the noise is reshaped pitch-synchronously, so each
   cycle carries its own measured pulse shape (gain = target / the noise's own profile, 0.1-10).

Cleaning is off: a machine's hum is often the sound. Measured on the rebuilds: fingerprint distance to the
reference 0.75-1.23 dB for the harmonic sources. The clock and sewing events are band envelopes rather than the
waveform grains of the research rebuilds: distance to those rebuilds 1.32 dB (clock; two noise seeds of that rebuild
differ by 1.29) and 0.92 dB (sewing).

## Macros and controls

Macros run 0..1 with 0.5 = the recording (the clock's `room` starts at 0 = dry).

- **Speed** is tape speed (pitch and time together), never a band pitch shift. Slower renders need care because
  the recordings' ~16 kHz edge moves down with the tape: `slow_mode` `tape_lp` (motors, drill, sewing, AC) adds a
  2nd-order roll-off at 16 kHz x speed x 0.8; `split` (engines, pass-bys, tram, mower) tapes only the tonal stems
  and stretches the noise envelopes in time (an expansion, read backwards past the end, never held), with the
  cycle pulses following the slowed fundamental.
- `throttle` (petrol, mower): tape speed -/+300 or 250 cents, master gain -/+2 to 2.5 dB, harmonic tilt -/+0.1 to
  0.19 (sharper firing when higher), a little more noise when higher.
- `load` (diesel): lighter = +60 cents, softer pulses (shape exponent 0.6), tilt -0.15, harmonics -1.5 dB;
  heavier = -80 cents, harmonics +2 dB, low shelf +2.5 dB at 150 Hz.
- `distance` (pass-bys, tram): 0.6 .. 2 (2.5 for the tram). Further: -6 dB per doubling (master gain), high shelf
  -5 dB per doubling at 2.5 kHz, a soft low-pass at 14 kHz / sqrt(d), flatter noise envelope. Nearer: low shelf
  +3 dB per halving at 200 Hz and a deeper envelope.
- `fan` (AC): -/+200 cents, quieter (-3 dB) with a flatter envelope when low, +2 dB when high.
- `tick_rate` (clock): 0.5 .. 2 ticks per recorded tick. The real ticks are re-timed, each carrying its own
  following stretch of the noise stem from 15 ms before it; a stretch shorter than the new period is read
  backwards, never held, with 10 ms power crossfades.
- `room` (clock): a smooth exponentially decaying noise tail (0 .. 0.6 s) excited by the real ticks with their
  narrow peaks flattened, so the room does not ring into a tone (no comb or feedback delays).
- `seed`: reseeds the noise carriers (the clock also re-sequences its ticks).

## Gestures

The `gesture` control is a timeline that plays a drill, the servo motor, the sewing machine or the mower under
control instead of as recorded:

```json
"gesture": [{"action": "start", "speed": 0.8}, {"action": "hold", "duration": 2.5},
            {"action": "change", "speed": 1.15, "duration": 0.6}, {"action": "hold", "duration": 2.5},
            {"action": "stop"}]
```

- `start` (optional `speed`, 1 = as recorded), `hold` (`duration` s, optional `speed`: a 0.2 s ramp to it),
  `change` (`speed` reached over `duration` s, log-linear), `stop`, `pause` (`duration` s of silence while
  stopped; start again after it). Speeds run 0.2 .. 3; `speed_cents` scales them all.
- The drill and the mower use their own gesture takes from the same recordings as their takes: the drill a
  free-running trigger burst later in the file (55.6-58.8 s; spin-up 2.17-2.36 s, steady 2.36-2.71 s at full
  speed, spin-down 2.71-2.95 s), the mower its pull-start (start 0.1-1.3 s, steady 1.3-7.3 s, shut-down
  7.3-11.0 s). Spin-ups and spin-downs are the real stretches, tape-scaled to the speed.
- The servo motor (steady 0.3-3.0 s) and the sewing machine (steady 0.3-5.2 s) have no recorded start or stop:
  they start with a tape ramp from 0.3x to the speed over 0.6 s with a fade-in and stop with a ramp down to 0.3x
  over 0.9 s with a fade-out.
- Steady runs are the real steady stretch read forwards and backwards between its ends (never a frozen frame).
  The drill's full-speed stretch is short (0.35 s), so it is read forwards only: the hold is cut into chunks of
  0.14-0.24 s (random lengths, so there is no fixed repeat rate), each read forwards from a random point of the
  stretch, crossfading over 100 ms; the seed picks the chunks. Harmonic and line phase tracks are read by their
  rate, so a backwards read keeps each partial's own frequency offset. Where the reading jumps (between chunks,
  or into a spin-down from the middle of the steady run) the two readings crossfade (60 ms unless the take sets
  its own): noise envelopes in power, amplitudes and phase rates linearly. The noise layer's long-term spectrum
  follows the envelope power the gesture actually reads, band by band, so a gesture that is mostly running keeps
  the running machine's noise level rather than the clip average (which includes the take's silences). Events (stitches) play forwards
  whenever the reading passes their recorded time, in either direction.
- Speed is a variable tape speed applied to the whole render (Kaiser-windowed sinc reads); slower stretches get the
  matching treble roll-off, crossfaded with the speed.
- An empty timeline (the default) plays the recording as taken; `power_drill` keeps its revving trigger take.

Gestures are screened against the same gesture played on audio (the as-recorded rebuild read through the identical
map): no new lines, bumps, comb, wobble, frozen stretches, shimmer, hiss, buzz, chop or flapping, and no clicks at
loop turns or crossfades.

## Variations

Variations reseed the noise carriers and remix the real stems; nothing is averaged. The clock's variations also
re-sequence its real ticks.

## Next

More gesture takes with recorded starts and stops (servo, sewing machine), a recorded engine start and shut-down,
Doppler-aware pass-by distance, stereo from paired takes.
