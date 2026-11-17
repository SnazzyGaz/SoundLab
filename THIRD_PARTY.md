# Third-party code

None vendored as of 0.1.0. Runtime dependencies (installed, not copied): NumPy (BSD-3-Clause), SciPy
(BSD-3-Clause); optional: matplotlib (matplotlib licence, BSD-style), pytest (MIT), ffmpeg (external executable).

## Methods adopted from other projects and papers (no code copied)

v0.20 choir voices: glottal pulse shape after A. E. Rosenberg, "Effect of glottal pulse shape on the quality of
natural vowels", JASA 49 (1971), the pulse family also used by klatt-syn (https://github.com/chdh/klatt-syn, MIT)
and Pink Trombone (pink-trombone-mod, MIT), written from the paper, no code copied; jitter, shimmer and vibrato
after I. Titze, *Principles of Voice Production* (1994). `conv_reverb` follows J. A. Moorer, "About this
reverberation business", CMJ 1979, and J.-M. Jot and A. Chaigne (AES 1991) for frequency-dependent decay.

v0.19 interaction blocks (`hc_contact`, `ep_friction`, `surface`, `crumple`, `wind_cavity`, `dc_motor`) follow the
models of the **Sound Design Toolkit** (S. Baldan, S. Delle Monache, D. Rocchesso et al., SkAT-VG,
https://github.com/SkAT-VG/SDT, GPL-3.0). SDT was studied from its headers, documentation and papers; **no SDT code
was copied or translated**, the blocks are written from the published equations: K. H. Hunt and F. R. E. Crossley
(1975) and F. Avanzini and D. Rocchesso (DAFx 2001) for contact; P. Dupont, V. Hayward, B. Armstrong and
F. Altpeter (IEEE TAC 2002) and F. Avanzini, S. Serafin and D. Rocchesso (IEEE TSAP 2005) for elasto-plastic
friction; K. van den Doel, P. Kry and D. Pai, "FoleyAutomatic" (SIGGRAPH 2001) and M. Rath and D. Rocchesso
(IEEE Multimedia 2005) for rolling and scraping; J. Sethna, K. Dahmen and C. Myers, "Crackling noise" (Nature
2001) for crumpling and breaking; A. Farnell, *Designing Sound* (MIT Press 2010) for wind cavities, motors and
hum. The fit-loss `sparsity` part (v0.19) is a standard spectro-temporal statistic (G. Peeters, CUIDADO audio
descriptors, 2004). Details: [docs/sdt-methods.md](docs/sdt-methods.md).

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
