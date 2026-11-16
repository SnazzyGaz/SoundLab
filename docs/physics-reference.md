# Physics reference

Formulas and data used in the code, with sources. See the knowledge-base Page for the full research.

## Formulas

| quantity | formula | where used | source |
|---|---|---|---|
| Modal decay | y = sum a_k e^(-t/tau_k) sin(2 pi f_k t), tau_k = 1/(pi f_k eta), T60 = 6.91 tau | `modal` | van den Doel et al. 2001; Klatzky et al. 2000 |
| Klatzky decay | tau_d = 1/(pi tan phi); decay to 1/e at f is tau_d / f | `modal klatzky=` | Klatzky, Pai & Krotkov 2000 |
| Free-free bar | f1 = 4.730^2/(2 pi L^2) h sqrt(E/(12 rho)); ratios 1, 2.76, 5.40, 8.93, 13.34 | `modal shape=bar` | Euphonics (Woodhouse) |
| Simply supported plate | f_mn = (pi/2) sqrt(D/(rho h)) ((m/Lx)^2 + (n/Ly)^2), D = E h^3 / 12(1 - nu^2) | `modal shape=plate` | standard plate theory |
| Circular membrane | ratios = Bessel zeros / 2.405: 1, 1.59, 2.14, 2.30, ... | `modal shape=membrane` | Rochester PHY103 |
| Tubes | closed-open: (2k-1) c/4L; open-open / cone: k c/2L | `tube`, `horn` | J. O. Smith, PASP |
| Helmholtz | f = c/(2 pi) sqrt(A/(V L)) | `helmholtz` | Wikipedia: Helmholtz resonance |
| Minnaert bubble | f = 3.26/r (r in m) | `bubbles`, `water` | Wikipedia: Minnaert resonance |
| Bubble damping | d = 0.13/r + 0.0072 r^-1.5; pitch rise f(t) = f0 (1 + xi d t), xi = 0.1 | `bubbles` | van den Doel 2005; Cirio et al. 2013; dougjam Bubble Sound Bank |
| Aeolian tone | f = St U / D, St ~ 0.2 | `turbulence` | Wikipedia: Aeolian sound |
| Firing frequency | f = RPM x cylinders / 120 (4-stroke); even interval 720 deg / cylinders | `engine` | Borla; Wikipedia: Firing order |
| Friedlander blast | P(t) = Ps e^(-t/t*) (1 - t/t*) | `combustion`, `explosion` | Wikipedia: Blast wave |
| Sabine | T60 = 0.161 V / (S a) | `room` | Wikipedia: Reverberation |
| Critical distance | r_c ~ 0.057 sqrt(V/T60) | `room` (direct/reverb balance) | standard diffuse-field result (not in our fetched sources) |
| Speed of sound | 343 m/s at 20 C | everywhere | Wikipedia: Speed of sound |
| Doppler | listener hears emission time te where t = te + r(te)/c | `passby` | J. O. Smith, PASP (variable delay) |
| Dispersion | allpass chain in the waveguide loop; low frequencies delayed more | `string dispersion=` | J. O. Smith, PASP |

## Air absorption (ISO 9613-2:1996, Table 2), dB/km

| atmosphere | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k |
|---|---|---|---|---|---|---|---|---|
| 10 C, 70 % | 0.1 | 0.4 | 1.0 | 1.9 | 3.7 | 9.7 | 32.8 | 117 |
| 20 C, 70 % | 0.1 | 0.3 | 1.1 | 2.8 | 5.0 | 9.0 | 22.9 | 76.6 |
| 30 C, 70 % | 0.1 | 0.3 | 1.0 | 3.1 | 7.4 | 12.7 | 23.1 | 59.3 |
| 15 C, 20 % | 0.3 | 0.6 | 1.2 | 2.7 | 8.2 | 28.2 | 88.8 | 202 |
| 15 C, 50 % | 0.1 | 0.5 | 1.2 | 2.2 | 4.2 | 10.8 | 36.2 | 129 |
| 15 C, 80 % | 0.1 | 0.3 | 1.1 | 2.4 | 4.1 | 8.3 | 23.7 | 82.8 |

Interpolated log-log between bands; above 8 kHz extrapolated as f^2 (our approximation).

## Materials

Cremer & Heckl, *Structure-Borne Sound*, via the vibrationdata damping tutorial. `soundlab materials` prints the
table with the material-only T60 at 1 kHz. Values marked "estimate" in `soundlab/materials.py` (E for concrete,
brick, fir, plywood; cardboard entirely) are typical values not taken from that table.

Material-only T60 is an upper bound: real objects also lose energy by radiation and through their mounting,
which `modal.extra_damping` models (1/s, frequency independent).

## Perceptual decay (Klatzky et al. 2000)

| category | critical log10 tau_d | tan phi | T60 at 1 kHz |
|---|---|---|---|
| rubber | 0.46 | ~0.11 | ~20 ms |
| wood | 0.50 | ~0.10 | ~22 ms |
| glass | 2.10 | ~0.0025 | ~0.87 s |
| steel | 2.65 | ~0.0007 | ~3.1 s |

## Reference numbers per sound

- Mallard: hen quack series 2-10 notes (typ. 5-6), decrescendo; f0 reported ~179 Hz (Greenewalt 1958) to
  ~585 Hz; strongest band 1.5-2.5 kHz; drake raeb ~0.3 s, peak ~2 kHz; bulla widens 500-2000 Hz.
- Diaphone: Type F ~250 Hz at 35-40 psi; grunt 93-150 Hz; F2T sustains the low tone.
- Engine: 4 cyl at 3000 rpm = 100 Hz firing.
- Explosion: Farnell shockwave ~10 ms cosine burst; blast band swept down from ~7 kHz; fireball ~70 Hz.
- Thunder: claps 0.2-2 s, rumbles can exceed 30 s.
- Water: the dripping-tap "plink" is the trapped bubble driving the surface; impact and splash are nearly
  silent (Phillips, Agarwal & Jordan 2018).
