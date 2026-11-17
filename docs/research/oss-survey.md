# Open-source sound generation: knowledge base (SoundLab v0.19)

Snapshot from v0.19; later work is in creature-research.md.

Research notes. Organised by sound domain. Each row: project or paper, the method
worth taking, where it lands (or would land) in SoundLab, and the licence. **Licence rule:** MIT / BSD / Apache /
Unlicense / MIT-style code may be ported with attribution in THIRD_PARTY.md; GPL / LGPL / AGPL / non-commercial
/ unlicensed projects are **method only** (read, then write again from the equations). Status: done = in
SoundLab, v19 = added this release, backlog = planned, skip = not worth it.

Search coverage (v0.18 + v0.19): GitHub repo, code and topic search (until the search API rate-limited; repo
reads continued through the core API), SDT source and docs read in full, STK, Faust libraries, Pd/Farnell,
Nemisindo/Reiss papers, DDSP family, DAFx/ICASSP/NIME/SMC papers, X/Twitter (several query sets plus author
queries; very little technical signal there beyond CHIMERA's buzz-source thread).

## 1. Contact: impacts, bounces, hits

| Source | Method | SoundLab | Licence | Status |
|---|---|---|---|---|
| SDT Impact (Avanzini & Rocchesso DAFx 2001) | Hunt-Crossley force k x^a + l x^a v; contact time and brightness follow speed and hardness | `hc_contact` -> `modal` | GPL-3 (method) | v19 |
| SDT Bouncing | gravity flight, restitution, irregularity | `hc_contact` bounces | GPL-3 (method) | v19 |
| SDT / van den Doel modal synthesis | modes with material damping, pickups | `modal` (material tables, Klatzky damping) | GPL-3 / papers | done |
| RipplerX (tiagolr) | modal + waveguide pairs, mallet noise, coupled resonators | `modal`, `string` coupling idea | GPL-3 (method) | backlog (coupling) |
| Faust physmodels.lib | modal and waveguide percussion building blocks | `modal`, `tube` | LGPL (method) | done-ish |
| NeuralResonator (Diaz et al.) | learned modes from shape | not needed offline | Apache-2.0 | skip |
| Nemisindo (Reiss group) | procedural impacts in the browser, parameter sets per material | presets | papers / closed engine | done (presets) |
| Hunt-Crossley in sound models | -- | `hit`, `sword_clash`, `bullet_impact` still use fixed contact time | -- | backlog |

## 2. Continuous contact: friction, scraping, rolling, crumpling, breaking

| Source | Method | SoundLab | Licence | Status |
|---|---|---|---|---|
| SDT Friction (Dupont et al. 2002; Avanzini, Serafin, Rocchesso 2005) | elasto-plastic bristle friction, Stribeck, break-away, pre-sliding | `ep_friction` (squeak; creak with springy drive) | GPL-3 (method) | v19 |
| STK BowTable / Bowed | bow friction table on a waveguide string | bowed string block | MIT-style | backlog |
| SDT Scraping / Rolling (van den Doel 2001; Rath & Rocchesso 2005) | fractal surface profile, ground trace, micro-flights | `surface` | GPL-3 (method) | v19 |
| SDT Crumpling / Breaking (Sethna 2001 crackling noise) | power-law energies, fragment size -> pitch, energy budget | `crumple` | GPL-3 (method) | v19 |
| Farnell, Designing Sound (Pd) | creaks, footsteps on gravel/snow, paper | `door_creak`, `footstep`, `rustle` | book / Pd patches (method) | done |
| Bresin et al. SDT walking | footsteps as crumpling + impacts on ground types | `footstep` | method | partly done |

## 3. Liquids: bubbles, splashes, pouring, rain

| Source | Method | SoundLab | Licence | Status |
|---|---|---|---|---|
| Minnaert 1933; van den Doel 2005 | bubble pitch 3.26/r, damping, rising chirp; populations | `bubbles`, `bubble_cloud` | papers | done (v18) |
| SDT Bubble / FluidFlow | power-law radius and depth, voice stealing | `bubble_cloud` | GPL-3 (method) | done; depth/bloop backlog |
| Moss et al. 2010 "Sounding liquids"; Zheng & James 2009 "Harmonic fluids" | bubbles from simulated fluid; splash = bubble burst + impact | `impact` resynth kind | papers | done |
| drip (various), Farnell water chapter | drop = impact + bubble | `drip`, `rain` | unlicensed / book (method) | done |
| Sparsity statistic (Peeters 2004 descriptors; this work) | heavy tail of locally normalised STFT power separates bubbles from noise | fit `sparsity` part | -- | v19 |

## 4. Air and gases: wind, whooshes, whistles, explosions, fire

| Source | Method | SoundLab | Licence | Status |
|---|---|---|---|---|
| SDT WindFlow / WindKarman | speed-driven band noise; Aeolian tone at the Strouhal frequency | `wind`, `turbulence` | GPL-3 (method) | done |
| SDT WindCavity; Howe edge tones | edge tone locks onto cavity modes | `wind_cavity` | GPL-3 (method) | v19 |
| Nemisindo / Selfridge, Moffat, Reiss (aeroacoustic swords, propellers, whistles) | compact aeroacoustic sources: Aeolian, cavity, edge tones | `sword_swing`, `wind_gap` | papers | done |
| SDT Explosion; Friedlander | blast pulse + FDN scattering | `explosion` | GPL-3 (method) | done; FDN scattering backlog |
| thunder-synthesis (Nemisindo group) | N-wave chains along a tortuous channel | thunder block | no licence (method) | backlog |
| Fire: Farnell; Nemisindo fire | crackle (Poisson pops), hiss, lapping roar | `fire` | book / papers | done (SDT has no fire) |

## 5. Machines: engines, motors, hum

| Source | Method | SoundLab | Licence | Status |
|---|---|---|---|---|
| engine-sim (ange-yaghi) | per-cylinder combustion, intake/exhaust runners as waveguides, convolution exhaust | `engine` v2; exhaust waveguide backlog | MIT (portable) | partly done |
| SDT Motor | waveguides for intake, cylinders, extractor, exhaust, muffler, outlet; backfire | `engine` | GPL-3 (method) | backlog (muffler, backfire) |
| engine-sound-generator (Antonio-R1), markeasting/engine-audio | WebAudio cylinder pulses + waveguide pipes; sample crossfades | `engine` | MIT | done (ideas) |
| Rc_Engine_Sound_ESP32 | sample-based RC engine with rpm crossfades | -- | no licence | skip |
| SDT DCMotor | rotor partials, load FM, gear, brush noise, chassis comb | `dc_motor` | GPL-3 (method) | v19 |
| Mains hum | 50/60 Hz + 2x magnetostriction | `mains_hum` | physics | v19 |

## 6. Voice: speech, singing, creatures, animals

| Source | Method | SoundLab | Licence | Status |
|---|---|---|---|---|
| WORLD (Morise), pyworld | f0 + spectral envelope + aperiodicity | `harmonic` env/ap frames | BSD / MIT | done (v18) |
| Klatt (klatt-syn), Pink Trombone (pink-trombone-mod) | cascade/parallel formants; waveguide tract; noise tied to glottal flow | `harmonic` pulse_noise; tract backlog | MIT | done / backlog |
| CHIMERA vocal synth | ventricular folds at f0/2 for growls, subharmonics, roughness | creature growl | MIT | done in v0.21 (`creature_voice`); see creature-research.md |
| fdtd-vocaltract-model, VocalTractLab | 2D/1D tract acoustics, articulatory control | waveguide tract | MIT / GPL | backlog |
| golf (DDSP glottal flow), DDSP | differentiable LF glottal source, harmonic + noise | LF pulse | MIT / Apache | backlog |
| Trio (PaulBatchelor), sndkit | small singing-voice / tract DSP | voice blocks | MIT / Unlicense | ideas taken |
| OpenUtau, espeak-ng | singing / speech front ends | -- | MIT / GPL | skip |
| creature-voice-synth | formant + growl creature voices | creature presets | no licence (method) | ideas |
| Syrinx models (Mindlin; Smyth & Smith) | bird song source | syrinx block | papers | backlog |

## 7. Synthesis toolkits, retro and UI sounds

| Source | Method | SoundLab | Licence | Status |
|---|---|---|---|---|
| sfxr / jsfxr, ZzFX | parametric retro sfx | `recipes/blocks` | Unlicense / MIT | done |
| STK (Cook & Scavone) | waveguide instruments: bowed, blown, plucked, shakers (PhISEM) | `shaker` done; bowed/blown backlog | MIT-style | partly done |
| Faust libraries | filters, reverbs, physical models | reference | LGPL (method) | done-ish |
| SuperCollider, Csound, pyo | unit generators | reference | GPL / LGPL (method) | skip |
| earshot (Leonxlnx) | small JS sfx generator | ideas | MIT | skip |

## 8. Analysis, fitting and control

| Source | Method | SoundLab | Licence | Status |
|---|---|---|---|---|
| auraloss | multi-resolution STFT losses | fit `stft` | Apache-2.0 | done |
| torchsynth, diffsptk, DDSP | differentiable synth / fitting | -- (CMA-ES offline) | Apache-2.0 | skip |
| McDermott & Simoncelli 2011 | texture statistics | fit `texture` | paper | done |
| SDT analysis | zero-crossing, spectral moments, flatness, flux, onset, McLeod NSDF pitch | fit `sparsity` (v19); McLeod pitch, flux backlog | GPL-3 (method) | partly |
| SDT Demix | structure-tensor harmonic/percussive/residual split | `hpr_split` | GPL-3 (method) | backlog |
| sms-tools | sinusoidal + residual modelling | `harmonic` | AGPL (method) | done (method) |
| NoiseBandNet | filterbank noise | `bandnoise` | AGPL (method) | done (method) |
| RAVE | neural timbre transfer | -- | CC BY-NC | skip |
| pyroomacoustics | image-source rooms | `room` reference | MIT | done (method) |

## Cross-domain transfers worth doing

- **Friction drives everything that squeaks, creaks, bows and squeals**: hinges, shoes on floors, wet glass,
  violin-like creature whines. `ep_friction` covers it; an STK-style bowed waveguide would add sustained tones.
- **Hunt-Crossley contact for every impact model**: speed-dependent brightness for free (footsteps, swords,
  bullets, coins).
- **Waveguide tubes for exhausts, vocal tracts and horns**: one block, three domains (engine exhaust, creature
  throat, foghorn/brass).
- **Crackling-noise control for paper, fire pops, ice, gravel and glass**: `crumple` + `fire` share the law.
- **Edge tones and cavity locking for wind gaps, bottles, whistles, sword swings and flutes.**
- **Bubble populations for water, boiling, fizz and lava.**

## Backlog, ranked by expected impact

1. CHIMERA-style ventricular-fold growl (f0/2) for creatures (done in v0.21; see creature-research.md for why it is not enough).
2. Hunt-Crossley contact inside `hit`, `sword_clash`, `bullet_impact`, `footstep`.
3. Engine exhaust waveguides with runner reflections, muffler, backfire.
4. STK bowed and blown waveguides.
5. Explosion FDN scattering; thunder N-wave chains.
6. LF glottal pulse; waveguide vocal tract; syrinx.
7. McLeod pitch, spectral flux onsets, structure-tensor HPR split.
8. Bubble depth / blooping over time (pouring, filling).
