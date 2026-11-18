# Third-party code

No packages vendored (ported code is listed under "Ported code" below). Runtime dependencies (installed, not copied): NumPy (BSD-3-Clause), SciPy
(BSD-3-Clause); optional: matplotlib (matplotlib licence, BSD-style), pytest (MIT), ffmpeg (external executable), pyworld
(MIT, wrapping WORLD by M. Morise, modified BSD; used only by `tools/world_probe.py`, extra `soundlab[world]`).

## Ported code

### CHIMERA vocal synth (MIT) -> `soundlab/models/creature_voice.py` (v0.21)

The glottal source of `creature_voice` is ported (Rust to NumPy) from CHIMERA by Jameson Hodge,
https://github.com/jamesondh/chimera-vocal-synth, file `crates/chimera/src/sources/glottis.rs` at commit
24bd2e4: the area pulse `shape()` (raised-cosine opening, closure blended between soft and abrupt) and the
per-cycle perturbations of `ParametricGlottis::next` (Gaussian jitter and shimmer, alternating diplophonia, the
logistic-map bifurcation control `r = 2.9 + 1.1 chaos` that scales period and amplitude, vocal-fry double pulses).
The alligator and elephant recipes take their pitch, chaos, jitter/shimmer and tract settings from CHIMERA's
creature definitions. CHIMERA's airway network was not ported (SoundLab uses a lossy-tube comb). Licence:

```
MIT License

Copyright (c) 2026 Jameson Hodge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Recordings measured for fitted numbers (no audio included)

v0.28 `creak` (`soundlab/models/creak_data.py`): resonances, squeal pitches and harmonic levels, click rates,
slip timing, burst lengths and decay rates were measured from these CC0 1.0 recordings on Freesound. Only the
numbers are in the repository. Further CC0 door, gate, hinge, shoe and floor recordings from Freesound were
looked at for ranges.

| Freesound id | Title | Author | URL |
|---|---|---|---|
| 455324 | Heavy Door Creaking_05 | rambler52 | https://freesound.org/people/rambler52/sounds/455324/ |
| 766155 | creaking door | dianegray26 | https://freesound.org/people/dianegray26/sounds/766155/ |
| 397687 | Creaking Gate | piermic | https://freesound.org/people/piermic/sounds/397687/ |
| 788701 | Soft_sqeuak_door_hinge | T.I.B | https://freesound.org/people/T.I.B/sounds/788701/ |
| 88502 | sneaker squeak rubber | shakaharu | https://freesound.org/people/shakaharu/sounds/88502/ |
| 124289 | squeaky-floor-loud | alienistcog | https://freesound.org/people/alienistcog/sounds/124289/ |
| 506665 | Wood Creak Single V10 | Rudmer_Rotteveel | https://freesound.org/people/Rudmer_Rotteveel/sounds/506665/ |

## Methods adopted from other projects and papers (no code copied)

v0.28 `creak`: the creaking door as stick-slip events through a fixed wood formant bank (A. Farnell, *Designing
Sound*, MIT Press 2010, practical "creaking door"; C. Heinrichs, A. McPherson, A. Farnell, "Human performance of
computational sound models for immersive environments", The New Soundtrack 4(2), 2014). Method only; the
resonances and event statistics here are our own fits.

v0.27 `friction_family`: elasto-plastic friction (P. Dupont, V. Hayward, B. Armstrong, F. Altpeter, "Single
state elastoplastic friction models", IEEE TAC 47(5), 2002) as in the Sound Design Toolkit friction model
(F. Avanzini, S. Serafin, D. Rocchesso, "Interactive simulation of rigid body interaction with friction-induced
sound generation", IEEE TSAP 13(5), 2005; GPL-3.0: method only); friction sound categories (A. Akay, "Acoustics
of friction", JASA 111(4), 2002); scraping through modal bodies (van den Doel, Kry, Pai 2001, as above).

v0.26 `material_impact`: modal synthesis with material parameters (K. van den Doel, P. G. Kry, D. K. Pai,
"FoleyAutomatic", SIGGRAPH 2001); material categories from frequency-dependent damping (M. Aramaki, M. Besson,
R. Kronland-Martinet, S. Ystad, "Controlling the perceived material in an impact sound synthesizer", IEEE TASLP
19(2), 2011); perceived material from decay (R. L. Klatzky, D. K. Pai, E. P. Krotkov, Presence 9(4), 2000);
residual compensation (Z. Ren, H. Yeh, M. C. Lin, "Example-guided physically based modal sound synthesis", ACM
TOG 32(1), 2013); plate, ring and spherical-shell mode formulas (A. W. Leissa, *Vibration of Plates* and
*Vibration of Shells*, NASA SP-160/288); granular ground and fracture as in the crumpling and walking models
of the Sound Design Toolkit (GPL-3.0: method only) and R. Bresin et al. 2010.

v0.21 `creature_voice`, beyond the CHIMERA port: ventricular folds vibrating at half the vocal-fold rate in growl
singing (L. Bailly, N. Henrich, X. Pelorson, "Vocal fold and ventricular fold vibration in period-doubled
phonation", JASA 127, 2010); vocal tract length and formant spacing in animals (W. T. Fitch, JASA 102, 1997;
T. Riede and W. T. Fitch, J. Exp. Biol. 1999); the tube as a lossy feedback comb (J. L. Kelly and C. Lochbaum,
1962; J. O. Smith, *Physical Audio Signal Processing*).

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
