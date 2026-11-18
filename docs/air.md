# Air: wind, whooshes, fly-bys and fans from recording stems (v0.30)

`air_stems` rebuilds noise-dominant air sounds from stems of five CC0 recordings (credited in
[THIRD_PARTY.md](../THIRD_PARTY.md)). `soundlab/models/air_stems.bin` (about 620 KB) holds only analysis numbers.
Every render uses new noise, so no sample of a recording is played back. Each recipe's default reproduces its
reference render; the fingerprints in `tests/data/air_fingerprints.json` check this to within 0.05-0.15 dB.

| Recipe | Source | Macros |
|---|---|---|
| `wind_gust` | wind_352421 from 10 s (8 s) | `strength`, `air`, `seed` |
| `wind_trees` | wind_760990 from 0 s, 3% slower, -25 cents, -6 dB above 3.5 kHz | `strength` (1 = gale), `air`, `seed` |
| `whoosh` | whoosh_507466 (0.92 s) | `speed`, `size`, `pitch`, `seed` |
| `arrow_flyby` | the five passes of flyby_675821 | `count`, `spacing`, `speed`, `closeness`, `seed` |
| `fan` | fan_761929 from 1 s (6 s) | `speed`, `size`, `seed` |

The older generic models are kept as physical fallbacks. Use `wind` and `wind_gale` for scenes that steer wind
speed live (the Godot demo uses both), and `sword_swing` for a swing with an edge tone.

## Stems

Analysis runs at 44.1 kHz with an STFT of N = 1024 and hop 128 (2.9 ms frames).

1. **Noise.** Band power on 49 log bands from 40 Hz to 20 kHz, measured after the tones are removed (a 31-bin
   median spectrum baseline, with tone bins clamped to it). The envelope is smoothed **only 10 ms in time**; fans
   use one blade period in time instead. (Correction, v0.31: earlier versions of this page said the envelope was
   also smoothed over 7 bands across frequency. The analysis meant to do that, but its time smoothing was applied
   to the unsmoothed envelope and replaced the frequency-smoothed one, and the fan segments were analysed with
   frequency smoothing off. No stored segment is smoothed across frequency. The fingerprints were measured on
   the data as stored, so the renders are unchanged.)
2. **Tonal tracks.** Partials from the sinusoidal tracker, kept if they last at least 60 ms and stand 10 dB above
   the local noise: whistles, blade tones, motor buzz. The stored stretch of the gust also contains a bird
   (about 6.9 kHz). Tracks above 5 kHz are muted by rule (`mute_above_hz`), and the tests check that no stray
   narrow peak is left.
3. **Onsets.** 1 ms envelopes in four bands (20-500 Hz, 500 Hz-2 kHz, 2-6 kHz, 6-20 kHz). They are applied only
   within ±15 ms of a detected high-band onset (a jump of more than 3x over the 21 ms median). This brings back
   twig snaps and leaf ticks without adding grain to the steady noise.
4. **Colour.** A 1/6-octave long-term spectrum, applied as a final static EQ. Without it, the lows came out
   15-30 dB too strong: a 23 ms frame smears about ±86 Hz, which is several bands at the bottom.
5. **Rotor (fans).** The blade rate is the envelope peak (3-60 Hz) of the upper bands, refined by ±3% to the rate
   that gives the deepest fold. The model stores a 32-phase shape of one blade cycle per band, averaged over
   about 60 cycles, and applies it to the smooth part of the envelope.
6. **Passes (fly-bys).** Each pass is stored as its own segment with its peak level and peak time. The geometry
   is pooled across the recording: 71 m/s, closest distance 0.56 m. Fitting each pass alone was too noisy
   (centroid error of 170-500 cents).

**Resynthesis.** Complex Gaussian noise is shaped by the band envelope, with one calibration pass that compares
the measured envelope to the target (25 ms smoothing). Then the model adds the tracks (cubic-phase synthesis),
the onset correction and the colour EQ.

## Controls

- **Speed and size (whoosh):** time scales with sqrt(size) / speed. Pitch moves +600 cents per doubling of speed
  and -840 cents per doubling of size.
- **Fan:** speed raises the blade rate and the spectrum (+1200 cents per doubling) and the level (+15 dB per
  doubling, since sound power goes with tip speed^5). Size lowers the spectrum (-960 cents per doubling) and
  slows the blades.
- **Fly-by:** a take draws passes (never the same pass twice running) and re-flies each one with its speed ±8%,
  closeness ±15% and spacing ±30%.
  - Time scales with distance / speed.
  - The Doppler curve (f' = f·c / (c - v_r)) of the new geometry replaces the old one, frame by frame.
  - Level goes as 1/r, and a high shelf of +3 dB per doubling of closeness models less air absorption.
- **Strength (wind):** **the trees recording is gale force.** Strength 1, the default, is that gale. Lower values
  extrapolate down from it, so that a light breeze barely registers in trees. With r = 0.25 + 0.75·strength:
  - level falls by 60·log10(r) dB (sound power ~ wind speed^6): -9 dB at 0.6, -19 dB at 0.3, -36 dB at 0;
  - the spectrum shifts down by 600·log2(r) cents, and a further high shelf of -8·(1 - strength) dB above
    3.5 kHz makes the leaves smaller and darker;
  - gusts get slower (time ×(1 + 0.8·(1 - strength))) and shallower (band envelopes pulled towards their mean by a
    factor of 0.35 + 0.65·strength).

  Strength 1 is exactly the stored segment. The tests check that level and brightness fall monotonically as
  strength falls.
- **Seed / variations:** wind and fan takes pick another stored stretch of the same recording and reseed the
  noise. Fly-by takes draw other passes. A transform with neutral settings is skipped, so the default keeps the
  tracker's measured phases.

## Lessons

- **Do not smooth noise envelopes over long times.** The first attempt smoothed over 70 ms in time and wiped out
  the fan's blade beat and the leaf flutter. The stored stems use 10 ms (one blade period for fans). (Correction,
  v0.31: this lesson used to say "smooth across frequency, not time" and credit a 7-band average across
  frequency; that average never reached the stored data, see Stems above. The liquid work in v0.31 found that
  smoothing across frequency is not needed when the per-band envelopes are measured on narrow bands directly:
  [liquids.md](liquids.md).)
- **Tonal parts need a stray-tone check.** One tracked whistle the listener does not expect (a bird in a wind
  recording) ruins the take. Mute it by rule and test for narrow peaks.
- **Time compression raises 8-30 Hz modulation.** Making a gust faster raises the envelope rate and reads as
  more "AM", but it is not bubbling. The bubbling check (`am_depth_db`) compares like with like, and single
  gestures (whoosh, fly-by) have a looser limit.
- **Fit geometry across passes, not per pass.** Each Doppler fit alone is unreliable; a pooled speed and distance,
  with each pass's own peak time, re-draws the curves well. Too close (closeness 1.6) reads as a flutter, so the
  default is 1.25.
- **Mind the source's scale.** A recording is one point on a physical range. The trees reference is a gale, so
  strength is a macro that maps down from it, rather than treating the recording as typical wind.
- A flag/cloth flap was tried from one recording and dropped: the reference was unsuitable (stiff), and a better one is needed.
