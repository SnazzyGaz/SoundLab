# Third-party code

None vendored as of 0.1.0. Runtime dependencies (installed, not copied): NumPy (BSD-3-Clause), SciPy
(BSD-3-Clause); optional: matplotlib (matplotlib licence, BSD-style), pytest (MIT), ffmpeg (external executable).

## Methods adopted from other projects and papers (no code copied)

v0.18 resynthesis blocks follow published methods. No source code was copied. The work they build on:

- **Bubble-population liquid sound** (`bubble_cloud`): M. Minnaert (1933); K. van den Doel, "Physically based
  models for liquid sounds", ACM TAP 2(4), 2005; W. Moss, H. Yeh, J.-M. Hong, M. Lin, D. Manocha, "Sounding
  liquids", ACM TOG 2010; C. Zheng and D. L. James, "Harmonic fluids", ACM TOG 2009; the liquids module of the
  Sound Design Toolkit (SkAT-VG, https://github.com/SkAT-VG/SDT, GPL-3.0: method only).
- **Harmonic + envelope + aperiodicity voice model** (`harmonic` `env_frames` / `ap_frames`): M. Morise,
  F. Yokomori, K. Ozawa, "WORLD: a vocoder-based high-quality speech synthesis system", IEICE 2016
  (https://github.com/mmorise/World, modified BSD); X. Serra and J. O. Smith, "Spectral modeling synthesis", CMJ
  1990; J. Engel et al., "DDSP", ICLR 2020 (https://github.com/magenta/ddsp, Apache-2.0).
- **Breath noise modulated by the glottal cycle** (`pulse_noise`): D. H. Klatt and L. C. Klatt, JASA 87(2), 1990;
  Pink Trombone by Neil Thapen (https://dood.al/pinktrombone/; modular port https://github.com/chdh/pink-trombone-mod,
  MIT); CHIMERA by Jameson Hodge (https://github.com/jamesondh/chimera-vocal-synth, MIT): turbulence noise tied
  to airflow.
- **Noise filterbank synthesis** (`bandnoise`, v0.17): H. Dudley's channel vocoder (1939); A. Barahona-Ríos and
  T. Collins, "NoiseBandNet" (https://github.com/adrianbarahona/noisebandnet, AGPL-3.0: method only).
