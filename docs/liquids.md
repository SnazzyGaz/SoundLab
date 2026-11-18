# Liquids: splashes, drips, a pour, flowing water, boiling and sloshing from recording stems (v0.31)

`liquid_stems` rebuilds liquid sounds from stems of nine public-domain, Public Domain Mark and CC0 recordings
(credited in [THIRD_PARTY.md](../THIRD_PARTY.md)). `soundlab/models/liquid_stems.bin` (about 970 KB) holds only
analysis numbers. Every render uses new noise, so no sample of a recording is played back. Each recipe's default
reproduces its reference render; the fingerprints in `tests/data/liquid_fingerprints.json` check this.

| Recipe | Source | Engine | Cleaning | Macros |
|---|---|---|---|---|
| `splash_small` | bathtub hand splash, 45.3-47.6 s | band envelopes | on | `size`, `distance`, `tail`, `seed` |
| `splash_medium` | bathtub hand splash, 30.2-33.0 s | band envelopes | on | `size`, `distance`, `tail`, `seed` |
| `splash_large` | dive splash (3 s) | band envelopes | off | `size`, `size_by`, `distance`, `tail`, `seed` |
| `drip_tub` | drops into a bathtub, 0-8 s | band envelopes | on | `rate`, `pitch`, `seed` |
| `pour_glass` | pouring into a cup (7.8 s) | band envelopes | off | `fill`, `vessel`, `seed` |
| `stream` | trickling brook, 2-10 s | STFT stems | off | `strength`, `seed` |
| `river` | rapids, 10-18 s | band envelopes | on | `strength`, `seed` |
| `waterfall` | small waterfall, 10-18 s | band envelopes | on | `strength`, `seed` |
| `boiling` | boiling lentils, 0.5-8.5 s | STFT stems | off | `intensity`, `seed` |
| `slosh` | sloshing canister (9.9 s) | STFT stems | off | `amount`, `seed` |

The older procedural `drip`, `splash` and `water` recipes stay as physical fallbacks (the `water` bubble model and
`bubble_cloud`).

## Stems

Analysis runs at 44.1 kHz.

**Band envelopes** (splashes, drip, pour, river, waterfall).

1. **Bubbles.** A sinusoidal tracker (512-sample window, 1.45 ms hop) finds narrow partials that stand 15 dB over
   the local spectrum for at least 10 ms. Only those above 400 Hz whose pitch moves less than 150 cents are kept
   (Minnaert chirps and steady rings). They are subtracted from the signal sample by sample, so the noise analysis
   has no spectral holes, and stored as chirps: start, length, a straight line in log frequency, a start phase and
   an amplitude per frame (0.5 dB steps).
2. **Noise.** The residual is split into 78 raised-cosine bands, 1/8 octave wide, from 30 Hz to 20 kHz (their
   squares sum to one). Each band's own amplitude envelope (analytic signal, low-passed at half the band width,
   at most 200 Hz) is stored every 2 ms in 0.5 dB steps. There is no smoothing across frequency or time beyond
   that low-pass. For the drip, a steady floor (20th percentile + 3 dB) is taken off so the room tone between
   drops is not rebuilt.
3. **Colour.** A 1/6-octave long-term spectrum, applied as a final static EQ.

Resynthesis imposes each band's envelope on a fresh Gaussian noise carrier and repeats the band-limiting four
times, so the envelopes and the band limits agree. Then the chirps are added (cubic-phase synthesis) and the
colour EQ is applied.

**STFT stems** (stream, boiling, slosh): the `air_stems` engine (N = 1024, hop 128, 49 log bands smoothed over
5 bands across frequency and 5 ms in time, raw bubble tracks, 1 ms onset envelopes in four bands applied around
detected onsets, one calibration pass). The band-envelope rebuilds of these three were measurably further from the
approved renders, so these keep the engine they were approved with.

**Storage.** The envelopes were checked at three resolutions: 1/8-octave bands at 2 ms reproduce the 1 ms
analysis within 0.04-0.28 dB (fingerprint distance); 1/6-octave bands at 2 ms were 0.67-1.85 dB off, so the
bands stay at 1/8 octave.

**Cleaning.** Some references carry steady hum lines, codec whines and short narrow "dings" (a room, a fridge, a
camera). Before analysis, steady narrow lines (15th-percentile spectrum peaks 10 dB over their neighbours, or any
narrow line present in at least 20% of frames for textures) are notched (IIR notch, Q 20, up to three passes),
and narrow events of 250 ms or more (15 dB prominence) or narrow whines above 8 kHz (12 dB, 100 ms or more) are
replaced by the local floor (the drip had an earlier, simpler pass of the same kind). Cleaning is on for the small and medium splash, the drip, the river and the waterfall,
and off for the pour and the large splash, whose rebuilds were approved uncleaned.

## Controls

All macros are 1 (or 0 cents) at the recording.

- **Splash size** (0.5-2): bigger is lower, longer and louder. By default (`size_mode` "shift") the stored bands
  move by -700 cents per doubling, the time stretches by size^0.5 (size^0.3 below 1), the bubble rings lengthen by
  the same factor and the level rises 9 dB per doubling (through the recipe's master gain, never above 0 dB).
  `size_by` = "tape" on `splash_large` changes the tape speed instead (pitch and time together, the render
  resampled; no band interpolation).
- **Splash distance** (0.5-4): -6 dB per doubling, a high shelf of -5 dB per doubling above 2.5 kHz, and a softer
  onset (band envelopes pulled towards their 150 ms mean).
- **Splash tail** (0.5-2): above 1 the tail after the peak is slowed; below 1 it decays faster (an extra
  exponential decay). Compressing the tail in time sped its sloshing up into a wobble, so it is not done.
- **Drip rate and pitch:** the real drops (five in the stored stretch, about 1.6 s apart) are cut out with their
  ring bubbles and re-sequenced: rate scales the drop rate, pitch shifts each drop (smaller drop = higher, shorter
  ring).
- **Pour fill and vessel:** fill speeds up or slows the whole fill (the cavity note rises faster); a bigger vessel
  is lower (-700 cents per doubling) and fills for longer (vessel^1.5).
- **Flow strength** (stream, river, waterfall; 0.2-1.5): **no pitch shift.** Lower is quieter (8 dB per halving),
  darker (a high shelf of 6 dB per halving above 3 kHz and, below 1, a soft 2nd-order low-pass at
  16 kHz·strength^0.7), slower (an envelope stretch of strength^-0.3 at a fixed length) and gentler (shallower
  envelopes). Past the end of the stored stretch the stems are read backwards, never held.
- **Boiling intensity** (0.3-2): lower drops real bubbles at random and softens the rest; higher overlays a
  time-offset copy of the same bubbles and noise.
- **Slosh amount** (0.5-1.6): faster and harder (time ×amount^-0.4, +120 cents and +7 dB per doubling).
- **Seed / variations:** a take reseeds the noise. Flows also reorder 1.6 s blocks of their real stems (80 ms
  power crossfades, the last block runs on to the end); the drip re-sequences its drops. Nothing is averaged.

## Lessons

- **Measure envelopes on narrow bands instead of smoothing.** Per-band envelopes on 1/8-octave bands, imposed on
  band-limited noise, keep a splash's crash and bubbly tail without the smeared, chorus-like grain of a smoothed
  STFT envelope.
- **Never hold the last frame.** A stretch that reads past the end and holds the last frame turns into a steady
  tone within a second. Read the stems backwards instead.
- **Do not pitch-shift textures by interpolating bands.** A fractional band shift smears narrow lumps into a
  noise spectrum (metallic), and tape speed moves a recording's codec edge (about 16 kHz) down into hearing range
  (a high whine). Darken with a shelf and a soft low-pass instead.
- **Fade, do not compress.** Shortening a tail by time compression turns its slow slosh into an 8-60 Hz wobble.
- **Check every change against the approved render**, not only against the reference: shimmer (musical noise),
  1/3-octave hiss from 4 to 16 kHz and 30-300 Hz envelope buzz, each within 1.5 dB.
