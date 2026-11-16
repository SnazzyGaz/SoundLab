# Analysis: from a sample to stems and features

`soundlab analyze sample.wav -o out/` (or `soundlab.analyze_file`) writes:

| file | contents |
|---|---|
| `harmonic.wav` | sustained, pitched/resonant part (horizontal ridges in the spectrogram) |
| `transient.wav` | attacks and clicks (vertical ridges) |
| `noise.wav` | everything else: the stochastic residual |
| `sines.wav` | resynthesis of the 20 strongest tracked partials (sinusoidal model) |
| `features.json` | measurements (below) |
| `analysis.png` | spectrogram, envelope and long-term spectrum |

## Method

1. Harmonic/percussive/residual split by median filtering of the magnitude spectrogram along time and along
   frequency, with a separation margin of 2 (FitzGerald, DAFx 2010; Driedger, Mueller & Disch, ISMIR 2014).
   Our own implementation (librosa, ISC, has an equivalent).
2. Sinusoidal tracking on the harmonic part: spectral peaks with parabolic interpolation, linked frame to frame
   by nearest frequency (Serra & Smith 1990, SMS). For each partial: median frequency, peak level, start/end, and
   T60 from a straight-line fit to its dB track after the peak. This is the input the modal fitter will use.
3. Features: spectral centroid, 85 % roll-off, flatness, octave-band energy, onsets (spectral flux), attack
   time (10-90 %), each stem's energy share, plus the robust measures below.

## Robust pitch, decay and modulation (0.2)

v0.1 reported a pitch for everything (an explosion "had" 1042 Hz), a decay for sustained sounds (an engine
"decayed" over 27 s), and an envelope rate that locked onto the exhaust-pipe comb. v0.2 changes this:

- **Pitch** (`f0_hz`, `voiced_fraction`, `periodicity`) runs YIN on the full signal, frame by frame.
  - A frame counts as voiced if its YIN dip is under 0.15.
  - The sound gets a pitch only if at least 25 % of its loud frames (within 30 dB of the peak) are voiced. The
    reported value is the loudness-weighted median.
  - `periodicity` = 1 - median YIN dip over the loud frames. A pure tone is about 1, noise 0.5 or less. Our
    mallard refs score 0.36-0.50; v1's duck scored 0.61.
- **Envelope class** (`envelope_kind`, `decay_t60_s`, `events`, `event_rate_hz`, `active_s`):
  - `repeated` means 3 or more separate events (rises above -15 dB with dips below -25 dB between them).
  - Otherwise the Schroeder energy-decay curve after the attack peak decides. A free decay takes about twice as
    long from -10 to -30 dB as from 0 to -10 dB, while a held tone that stops does not. Those cases are labelled
    `decaying` and `sustained`.
  - T60 is reported only for `decaying`.
  - Truncated recordings can look `sustained`.
- **Envelope modulation** (`envelope_modulation_hz`, `modulation_strength`):
  - Log envelopes are taken in four broad bands and their autocorrelations averaged, so one ringing resonance
    cannot dominate.
  - This runs in 1 s windows, so rpm changes do not smear it. Confident windows vote.
  - Within a window, the shortest lag with at least 85 % of the best peak wins.
  - Returns None when under a quarter of the windows are confident.

## Known limits

- No separation of overlapping sources yet (Demucs, MIT, is the planned optional front end; it was trained on
  music, not Foley).
- Stems are mask-based, so they sum back to the original but each one carries some bleed.
- The partial tracker is basic (no birth/death hysteresis, no harmonic grouping).
- Pitch is a single summary number. Two-tone sounds (a foghorn's high then low) report the louder or longer part.
- Envelope class is a heuristic. Rev-then-idle engines read as `decaying`, and truncated files can read as
  `sustained`.

## Next

Fitting is now in (`soundlab match`, [fitting.md](fitting.md)). Next: initialise `modal` frequencies and decays
straight from the partial list, so impacts can be fitted mode by mode.
