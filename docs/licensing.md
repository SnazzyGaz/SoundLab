# Licensing

SoundLab is MIT-licensed (Copyright (c) 2026 Garyck Arntzen). Policy:

1. **Copy/vendor only permissive code** (MIT, BSD, ISC, Apache-2.0, or the Faust GRAME exception for generated
   code), keeping its licence and attribution in `THIRD_PARTY.md`.
2. **GPL/AGPL/LGPL code is reference-only**: read the papers and documentation, reimplement from the maths, do not
   copy code. This covers the SDT (GPLv3), SuperCollider (GPL-3), sms-tools (AGPL-3), Csound (LGPL), k-Wave (LGPL).
3. **Non-commercial models are optional external hooks**, never bundled or required: RAVE (CC BY-NC 4.0),
   AudioLDM/AudioLDM2 (CC BY-NC-SA 4.0), AudioCraft and MMAudio weights (CC-BY-NC 4.0), Stable Audio Open weights
   (Stability AI Community License).
4. Code without a licence (DiffSound, NeuralSound repos) has no reuse rights: papers only.

## Status of 0.1

No third-party code is vendored. All code was written for this project. Algorithms are implemented from
published descriptions:

| algorithm | published source |
|---|---|
| biquad filters | R. Bristow-Johnson, Audio EQ Cookbook |
| state-variable filter | Chamberlin; Zavalishin, The Art of VA Filter Design (TPT form) |
| modal synthesis | van den Doel, Kry & Pai, SIGGRAPH 2001; Cook 2002 |
| waveguides, dispersion, Doppler | J. O. Smith, Physical Audio Signal Processing |
| stochastic grains | Cook, PhISM (CMJ 1997); Farnell, Designing Sound (2010) |
| bubbles | van den Doel, ACM TAP 2005 |
| friction | stick-slip (textbook); Avanzini, Serafin & Rocchesso, DAFx 2002 (reference) |
| engine | Baldan et al., SIVE 2015; Farnell practical 22 |
| explosion | Farnell practical 31; Friedlander waveform |
| image-source rooms | Allen & Berkley 1979 |
| air absorption | ISO 9613-2:1996 Table 2 |
| HPSS / residual | FitzGerald, DAFx 2010; Driedger et al., ISMIR 2014 |
| sinusoidal tracking | Serra & Smith, CMJ 1990 |
| pitch | de Cheveigne & Kawahara, YIN (JASA 2002) |

Runtime dependencies: NumPy (BSD-3), SciPy (BSD-3); optional matplotlib (PSF-based, BSD-compatible) and the
ffmpeg executable (called as an external program for MP3).

- Freeverb (Jezar at Dreampoint) is public domain; `reverb` reimplements its structure and tunings from the published description (0.6.0).
