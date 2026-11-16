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
3. Features: f0 (YIN, de Cheveigne & Kawahara 2002), spectral centroid, 85 % roll-off, flatness, octave-band
   energy, onsets (spectral flux), attack time (10-90 %), decay T60 (Schroeder backward integration), envelope
   modulation rate (finds e.g. firing frequencies and wing-beat rates), and each stem's energy share.

## Known limits (0.1)

- No separation of overlapping sources yet (Demucs, MIT, is the planned optional front end; it was trained on
  music, not Foley).
- Stems are mask-based, so they sum back to the original but each one carries some bleed.
- The partial tracker is basic (no birth/death hysteresis, no harmonic grouping).
- Envelope modulation can lock onto a pipe comb or a sub-multiple instead of the true pulse rate.
- No parameter fitting yet: that is build step 8 (analysis-by-synthesis with a differentiable twin).

## Next

Fit recipes to a recording: initialise frequencies and decays from the partial list (spectral losses give poor
pitch gradients, Turian & Henry 2020), fit noise layers by matching cochlear envelope statistics (McDermott &
Simoncelli 2011), then refine with a multi-resolution STFT loss.
