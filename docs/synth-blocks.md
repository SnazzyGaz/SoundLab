# Generic synthesis blocks (0.6.0)

Not everything a game needs has a physical source: UI blips, coins, power-ups, lasers, magic. These blocks
are a small, general synth toolkit that works in the same JSON recipes as the physical models, so a recipe can
mix both (a physical impact plus a synth sparkle, say). Every control is listed in [models.md](models.md).

Two new stages:
- **`synth`** blocks are sources: they start a chain (or add into one).
- **`fx`** blocks process a signal: in a chain after a source, or in `post` on the mix.

| block | stage | what it is |
|---|---|---|
| `osc` | synth | band-limited sine / saw / square / pulse (PWM) / triangle; unison voices + detune; vibrato |
| `fm` | synth | two-operator FM (ratio, index, feedback); 2x oversampled |
| `noisegen` | synth | white / pink / brown / blue / violet noise; optional sample-and-hold (`hold_hz`) for retro chip noise |
| `grains` | synth | random grains (sine / noise / saw), rate, length, pitch spread, scale snapping, percussive or smooth window |
| `svf` | fx | resonant state-variable filter: lp / hp / bp / notch / peak, per-sample cutoff and resonance |
| `env` | fx | ADSR (with gate) or breakpoints, exponential or linear |
| `shaper` | fx | tanh / hard clip / wavefolder / asymmetric (4x oversampled), bitcrush, sample-rate reduction |
| `delay` | fx | feedback echo with damping per repeat |
| `chorus` | fx | multi-voice modulated delay (chorus / flanger) |
| `reverb` | fx | mono Freeverb-style algorithmic reverb (for stylised space; physical rooms: `room`, `outdoor`) |
| `gain` | fx | automatable gain + tremolo |

**Sweeps** are automation: any control marked automatable takes `[[t, value], ...]` (seconds from the layer
start). Pitch (`osc.freq`, `fm.freq`, `grains.freq`) and filter cutoff are interpolated in log-frequency, so a
sweep from 200 Hz to 3200 Hz moves evenly in octaves. A step (arpeggio, coin) is two points a hair apart:
`[[0, 988], [0.065, 988], [0.0655, 1318.5]]`.

**Layering** uses the recipe's layers: each layer has its own `start` (offset in s) and `gain_db`, and is
normalised before its gain unless `"normalize": false`. `post` blocks (e.g. `chorus`, `reverb`) run on the mix.

```json
{"soundlab": 1, "name": "coin", "duration": 0.6, "layers": [
  {"chain": [
    {"model": "osc", "params": {"wave": "pulse", "freq": [[0, 987.8], [0.065, 987.8], [0.0655, 1318.5]]}},
    {"model": "svf", "params": {"mode": "lp", "cutoff": 9000}},
    {"model": "env", "params": {"points": [[0, 0], [0.003, 0.8], [0.063, 0.75], [0.068, 1], [0.5, 0.0001]]}},
    {"model": "delay", "params": {"time_ms": 90, "feedback": 0.25, "mix": 0.15}}]}]}
```

## Demo recipes (`recipes/blocks/`)

Generic blocks only (a test enforces it): `ui_click`, `ui_confirm`, `coin_pickup`, `power_up`, `laser_zap`,
`magic_shimmer`, `whoosh`. Each has a `category` field ready for step 3's naming.

## Quality notes

- Oscillators: PolyBLEP at 2x with a steep decimation FIR. Aliasing for a 1.2 kHz saw is about -43 dB re the
  energy above 2 kHz (naive saw -11 dB, plain 1x PolyBLEP -26 dB); it is the same as an ideal additive saw on
  this measure. Very high notes (above ~5 kHz) with saw/square still carry some residual aliasing.
- FM at high index x high carrier can still fold back above the 2x rate; keep `index` x `freq` x `ratio`
  under ~40 kHz.
- The SVF is a per-sample Python loop: about 0.1-0.2 s per second of audio per filter. Fine offline; a compiled
  core (later) would fix it.
- The reverb is mono and generic. It is good enough for stylised sounds; it is not a physical room.
