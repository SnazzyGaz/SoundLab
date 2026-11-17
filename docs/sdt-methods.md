# Interaction models from the Sound Design Toolkit (v0.19)

The [Sound Design Toolkit](https://github.com/SkAT-VG/SDT) (SDT, SkAT-VG / University of Verona and IUAV,
GPL-3.0) is the most complete open library of physically informed everyday-sound models. SoundLab is MIT, so
**no SDT code is used**: the models were studied from SDT's headers, its documentation and the papers behind it,
then written again from the published equations. This page lists every SDT model, where it lands in SoundLab,
and what is still to do.

## Map: SDT model -> SoundLab

| SDT model | Method | SoundLab | Status |
|---|---|---|---|
| Impact (interactor) | Hunt-Crossley contact force `f = k x^a + l x^a v` between two resonators | `hc_contact` (new excitation) -> any resonator (`modal`, `string`, `tube`) | **v0.19.** Harder or faster hits get a shorter, brighter contact on their own; replaces fixed `contact_ms` where speed matters |
| Bouncing (control) | Ball flight under gravity, restitution, irregularity | `hc_contact` `bounces`, `restitution`, `irregularity` | **v0.19**; old `impact` keeps its geometric bounce series |
| Friction (interactor) | Elasto-plastic bristle model (Dupont et al. 2002): stiffness, internal damping, viscosity, noise, Stribeck curve, break-away | `ep_friction` (new excitation, modes built in) | **v0.19.** Rigid drive = squeal / squeak / bowed tone; `drive_hz` > 0 = springy drive that gives stick-slip creaks |
| Rolling (control) | Ground trace of a rough surface seen through a ball of finite size; micro-flights | `surface` `mode=roll` | **v0.19** |
| Scraping (control) | Probe over a fractal profile; force ~ v^2 x bump | `surface` `mode=scrape` | **v0.19** |
| Crumpling (control) | Crackling noise: power-law event energies, fragment size sets pitch | `crumple` `mode=crumple` | **v0.19** (`rustle` stays for leaves and cloth) |
| Breaking (control) | Stored energy spent by fragments; rate and size fall as energy runs out | `crumple` `mode=break` | **v0.19** |
| Modal resonator | Mass-spring-damper modes with pickups | `modal` (material tables, Klatzky damping) | already had (richer than SDT's) |
| Inertial mass | Free mass for impacts | `hc_contact` `mass_g` | v0.19 |
| Bubble | Minnaert pitch, rising chirp, depth | `bubbles`, `bubble_cloud` | v0.18; depth/blooping on the backlog |
| FluidFlow | Bubble voices, power-law radius and depth, quietest voice replaced | `bubble_cloud` (population form) | v0.18 |
| WindFlow | Band-pass noise whose level and centre follow speed | `wind` | already had |
| WindKarman | Narrow band noise at the Strouhal frequency | `turbulence` (Aeolian tone) | already had |
| WindCavity | Edge tone locking onto cavity modes (comb + resonant bands, each mode with a best speed) | `wind_cavity` (new) | **v0.19**: howling pipes, bottles, gaps |
| Explosion | Friedlander pulse + FDN scattering | `explosion` (Friedlander) | had; FDN scattering on the backlog |
| Motor | Waveguides for intake, cylinders, exhaust, muffler, outlet; piston, ignition, valves; backfire | `engine` v2 | had (simpler exhaust); waveguide exhaust and backfire on the backlog |
| DCMotor | Rotor partials, load FM, gear and brush noise, chassis comb | `dc_motor` (new) | **v0.19**: drills, fans, servos, toys |
| (none) | Mains hum and transformer buzz | `mains_hum` (new) | v0.19 (SDT has no hum model) |
| (none) | Fire | `fire` | SDT has no fire model; ours stays |
| Analysis: zero-crossing, spectral moments, flatness, flux, onset, McLeod pitch | Feature extraction for control | `analysis`, `fit` | flatness/sparsity in the fit loss (v0.19); McLeod pitch and flux on the backlog |
| Demix | Harmonic / percussive / residual split from the spectrogram structure tensor | `hpr_split` (median filtering) | had a simpler split; structure tensor on the backlog |
| Reverb (FDN), pitch shift | Rocchesso FDN | `room`, `reverb` | already had |

## The new blocks in one line each

- `hc_contact`: one or more Hunt-Crossley contacts, integrated per sample, with gravity flight between bounces.
  Feed it into `modal` for knocks, taps and drops.
- `ep_friction`: elasto-plastic friction driving up to a few modes; `velocity` and `normal_force` are automatable.
- `surface`: `scrape` or `roll` over a 1/f^beta surface profile; feed it into a resonator.
- `crumple`: crumpling or breaking as a stream of small modal grains with power-law energies.
- `wind_cavity`: wind across an opening; the edge tone (Strouhal number) picks the nearest cavity mode, so the
  pitch jumps between modes as `wind_speed` rises.
- `dc_motor`: small electric motor; `rpm` automatable for spin-up and spin-down.
- `mains_hum`: 50/60 Hz hum, 100/120 Hz magnetostriction buzz, optional rattle.

New recipes: `hc_knock`, `hc_bounce`, `scrape_sdt`, `roll_sdt`, `creak_friction`, `squeak_friction`,
`crumple_paper`, `break_glass_sdt`, `wind_howl`, `drill_motor`, `mains_hum`.

## Fit loss: sparsity term

Resynthesis kept choosing band-shaped noise for water, because the old loss parts (spectrum, STFT, envelope,
texture statistics) score smooth noise well. v0.19 adds a `sparsity` part: STFT power is divided by its local
average, and the 50 / 99 / 99.9 % quantiles of the result are compared. Shaped Gaussian noise always has the same
light tail; bubbles, drops and crackle have a heavy one. Weight 0.5 (see [fitting.md](fitting.md)).

## Sources (methods only)

- SDT: S. Delle Monache, D. Rocchesso, F. Avanzini, et al., Sound Design Toolkit, https://github.com/SkAT-VG/SDT
  (GPL-3.0, studied, not copied).
- K. H. Hunt and F. R. E. Crossley, "Coefficient of restitution interpreted as damping in vibroimpact", J. Appl.
  Mech. 1975; F. Avanzini and D. Rocchesso, "Modeling collision sounds: non-linear contact force", DAFx 2001.
- P. Dupont, V. Hayward, B. Armstrong, F. Altpeter, "Single state elastoplastic friction models", IEEE TAC 2002;
  F. Avanzini, S. Serafin, D. Rocchesso, "Interactive simulation of rigid body interaction with friction-induced
  sound generation", IEEE TSAP 2005.
- K. van den Doel, P. Kry, D. Pai, "FoleyAutomatic", SIGGRAPH 2001; M. Rath and D. Rocchesso, "Informative sonic
  feedback for continuous human-machine interaction: controlling a sound model of a rolling ball", IEEE
  Multimedia 2005.
- J. Sethna, K. Dahmen, C. Myers, "Crackling noise", Nature 2001; M. Houben, A. Kohlrausch, D. Hermes,
  "Perception of the size and speed of rolling balls by sound", Speech Communication 2004.
- S. Baldan, S. Delle Monache, D. Rocchesso, "The sound design toolkit", SoftwareX 2017; A. Farnell, Designing
  Sound, MIT Press 2010 (wind, motors, hum); M. S. Howe, Acoustics of Fluid-Structure Interactions (edge tones).
