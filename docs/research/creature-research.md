# Creature and growl research (after v0.21.0)

Research notes, November 2026. No code in this document is part of the library yet.
Why: the v0.21 creature presets (demon_growl, monster_roar, dog_growl, alligator_bellow, elephant_rumble) did not sound convincing, so this pass looked at how creature sounds are actually made and reproduced.

## 1. Short answer

Our growl is bad because it is a **static parametric guess**. About 15 knobs are each held at one value for the whole sound, and on top of that we add a CMA-ES fit against a metric that cannot hear what makes a growl a growl.

What works, in both practice and the literature, is one of two things:
- **analysis-resynthesis**: frame-wise tracks taken from a real recording, with edits applied on top;
- **hybrid layering**: real or human performances that are pitched, formant-shifted and layered.

Pure synthesis only convinces when its controls change frame by frame and come from recordings, and even then mainly for harsh, noisy calls (grunts, screams, roars).

Rebuild plan: a frame-wise harmonic + subharmonic + noise resynthesis core, driven by tracks analysed from real recordings, with macros that edit those tracks rather than replace them.

## 2. Measured evidence on a recorded growl (test recording, about 3.4 s, 44.1 kHz)

Script: `tools/world_probe.py` (needs the optional `pyworld` package: `pip install soundlab[world]`). It uses WORLD through pyworld. Lower scores are better under `soundlab.fit.compare_signals`. The test recording itself is not in the repo.

| Render | total | note |
|---|---|---|
| v0.21 fitted creature_voice | 0.277 (held-out 0.400) | what was shipped |
| v0.20 creature_growl recipe | 0.824 | |
| Plain WORLD resynthesis, f0 floor 60 Hz | **0.214** | no fitting at all; 5 ms frames |
| WORLD, f0 floor 30 Hz | 0.318 | the tracker drops to the subharmonic |
| WORLD with envelope + aperiodicity smoothed over 50 ms | 0.481 | |
| ... smoothed over 200 ms | 0.535 | |
| ... smoothed over 800 ms | 0.526 | |
| One static envelope + aperiodicity, loudness kept | 0.542 | about what our model can express |
| ... and a static f0 as well | 0.534 | |

Energy per band, in dB relative to total 20 Hz–8 kHz:

| | 34–46 Hz (f0/2) | 65–90 Hz (f0) | 140–170 Hz | 1–4 kHz |
|---|---|---|---|---|
| Recording | −15.5 | −2.8 | −17.1 | −13.8 |
| v0.21 | **−7.8** | −4.1 | **−9.3** | −15.5 |
| WORLD (floor 60) | −24.2 | −1.9 | −16.3 | −14.2 |

What these show:
1. **Time variation is most of the realism.** Smoothing the same envelope and aperiodicity over just 50 ms more than doubles the error. Sparsity, our roughness and graininess term, goes from 0.38 to 1.6–2.0. The grit lives in cycle-to-cycle and frame-to-frame change. Our model has one `chaos`, one `mouth`, and one uniform tube for the whole sound, so it sits in the "static" rows by construction.
2. **v0.21's subharmonic is about 8 dB too loud and fixed.** The f0/2 ventricular layer, and the f0×2 region it feeds, are constant. In the real growl the subharmonic is weaker and comes and goes. A buzzy, constant octave-down drone is very likely the "fake" sound being heard.
3. **Plain vocoders lose the subharmonic.** WORLD is about 9 dB short at 40 Hz because it rebuilds one pulse train at f0. So the new core needs an **explicit subharmonic track**: a per-frame f0/2 (or f0/3) depth, which is soundgen's approach.
4. **Our metric is weak.** WORLD beat our fitted model by only 0.06 although it is a near-copy. The fit loss has no term for f0/2 depth over time or for period irregularity. The fitter was optimising the wrong thing.

The probe also writes `edit_down4_formant085.wav` (down 4 semitones, formants ×0.85, i.e. a bigger animal) to its output folder. It shows that edits on analysed tracks are cheap, though a plain WORLD edit of this kind still sounds noticeably processed.

## 3. Diagnosis: wrong assumptions in v0.21

| Assumption | Why it is wrong |
|---|---|
| A handful of global params can describe a growl | Real growls change regime within a call: periodic, then subharmonic, then chaos (Fitch, Neubauer & Herzel 2002; Wilden et al. 1998). Soundgen models the regime as a random walk because it is never constant. Our probe shows that static envelopes double the error. |
| Chaos = one logistic-map amount | Deterministic chaos in folds comes from asymmetric or desynchronised oscillators (Herzel; Neubauer et al. 2004). It has intermittent episodes and onsets that are bifurcations. One fixed chaos setting just sounds like constant jitter noise. |
| Ventricular f0/2 layer = subharmonic | Only partly true. Kargyraa-style period doubling is cycle-locked and modulates the main pulse (Bailly, Henrich & Pelorson 2010). It is not a separate sine-ish layer added in parallel, and its depth varies. Ours is 8 dB too loud and constant. |
| A uniform lossy tube with one mouth value is a creature tract | Real tracts move: larynx retraction in deer, mouth opening sweeps during a lion roar (Reby & McComb 2003; Titze). Formant motion carries a lot of the realism. |
| CMA-ES on spectral/STFT/envelope distance will find a growl | The loss is blind to subharmonic depth over time, period-to-period irregularity and formant motion. Fitted results converge to "average-shaped noise". |
| Pure synthesis can match a recording | Practitioners don't do this for hero creature sounds (see §4). Even soundgen, the strongest parametric vocal synth, is rated about 10% less authentic than real recordings overall. It matches them only for gasps, grunts, moans and screams. |

## 4. What practitioners do

- **Film and game creature design** (Krotos Dehumaniser tutorials, CRI middleware blog, typical GDC talks): performances come first. Designers record expressive human growls and real animals, then pitch them down, formant-shift them and layer them. A typical stack is a low pitched-down body, a saturated mid rasp, and a breath or noise layer, sometimes vocoded or granulated. Separate idle, aggressive, rumble and breath layers are blended by an "aggressiveness" parameter rather than switched.
- **VocAlien** (No Man's Sky, Paul Weir, GDC 2017): a physically modelled tract plugin that designers *perform* live. That means time-varying input, not static presets.
- **Farnell, Designing Sound**: Practical 28 Birds and Practical 29 Mammals (cow, lion). Pd patches build source-filter models from tract length, width and impedance plus glottal frequency. Farnell himself presents these as starting points that need performed envelopes.
- **Nemisindo / Reiss (Wilkinson & Reiss 2016, AES 61st)**: a source-filter model, with FM for nonlinear roughness and LPC mapped to a waveguide tract. In listening tests, harsh dense lion roars came out acceptable. Every control is a time contour (duration, rise and fall, freq1→freq2, smooth/coarse, harsh, bright, nasality, timbre modulation).
- **Honest boundary**: pure synthesis is fine for distant, harsh, noisy, or invented (alien) creatures. For anything the listener knows (dog, big cat, demon-from-human-voice), hybrid analysis-resynthesis or sample layering wins.

## 5. What bioacoustics says

- **Source-filter plus nonlinear phenomena** (Fitch, Neubauer & Herzel 2002, *Animal Behaviour*): subharmonics, biphonation, deterministic chaos and frequency jumps are everywhere in mammal calls and come from the vocal folds themselves. They switch on and off within one call.
- **Big cats** (Klemuk et al. 2011, *PLoS ONE*; Titze): Panthera folds are square and flat, with a large fat-filled pad. Roar f0 is low (about 40–200 Hz, often 36–81 Hz). At full effort the oscillation becomes irregular and aperiodic ("overblowing"). That regime is effort-dependent, so it is a time track, not a constant.
- **Deer** (Reby & McComb 2003; Fitch & Reby 2001): a retracted larynx lengthens the tract and lowers the formants during the roar. PSOLA formant rescaling of real roars is the standard playback method. In other words, the bioacousticians themselves resynthesise from recordings.
- **Alligator**: infrasonic bellow, with water dance from body resonance. Fit models are mostly descriptive. No good open synth exists.
- **Canids**: growl formant dispersion encodes body size (Riede & Fitch 1999; Taylor, Reby & McComb). This argues for a "size" macro that scales formant dispersion, not a filter brightness knob.
- **Code**: soundgen (R, GPL; Anikin 2019) is the reference implementation. It has vectorised time-varying f0, amplitude, formants and noise; a random-walk nonlinear regime (0 none, 1 subharmonics, 2 subharmonics + jitter + shimmer); subharmonics as an extra stack at f0/subRatio with depth subDep; and `matchPars` to fit a recording. It is GPL, so we port the method only.

## 6. Resynthesis methods (the key question)

**Is our problem over-parameterisation with too few, too static controls compared with frame-wise envelopes?** Yes. §2 measures it: static envelopes give about 0.53, while frame-wise envelopes with no fitting give 0.21.

| Method | What it gives us | Licence | Verdict |
|---|---|---|---|
| WORLD (Morise) / pyworld | f0, spectral envelope and aperiodicity every 5 ms; robust; easy edits | WORLD modified BSD-3; pyworld MIT | **Use** as the analysis front end, or port the method. Add our own subharmonic track. |
| STRAIGHT | Ancestor of WORLD | Academic licence, not open | Skip |
| SMS / Loris / SPEAR | Sinusoidal tracks + residual | SMS-tools AGPL; Loris GPL; SPEAR freeware closed | Method only. The partial tracking idea is fine; WORLD + harmonic sampling is simpler. |
| PSOLA (TD-PSOLA via Parselmouth/Praat) | Pitch and formant shifting that keeps grit | Parselmouth GPL-3 | Method only. TD-PSOLA is easy to write ourselves. Good for "real growl, bigger beast". |
| DDSP harmonic + filtered noise | Per-frame harmonic amplitudes + noise filter; works on animals (arXiv 2210.10857; DDSP-SFX DAFx24 gates harmonics by pitch confidence) | Apache-2.0 | **Use the method** (no neural net needed): per-frame harmonic amplitudes from analysis drive an additive synth. |
| NoiseBandNet | Filterbank noise synthesis for textures | AGPL-3.0 | Method only. Noisy-texture reference. |
| HiFi-GAN, BigVGAN | Neural vocoders, realism ceiling | MIT | Reference only. Too heavy for this; useful to measure "how close can it get". |
| RAVE | Neural timbre transfer | CC BY-NC 4.0 | Avoid (non-commercial). |
| soundgen | Time-varying parametric vocals with nonlinear regimes | GPL | Method only |
| CHIMERA | Physical airway | MIT | Already ported (glottis). Its forks are below. |

## 7. CHIMERA forks (jamesondh/chimera-vocal-synth)

Upstream has about 2.6k stars and 324 forks; 246 unique forks were listed and checked (November 2026). Every fork checked is MIT, inherited from upstream. I walked them all (sorted by stars and by recent push), compared each default branch and every non-main branch with upstream main, and also read open and closed upstream PRs.

**Bottom line:** no fork has animal voices refined and tested by ear. Forks have almost no stars (the top fork is cupOJoseph at 55, with 2 commits ahead and nothing animal-related). The animal work there is new physically modelled creatures checked against published measurements, not against listening. The useful parts are the cat models, the self-oscillating folds and the ventricular-fold (kargyraa) work.

### Worth studying

| Fork / PR | Ahead | What it adds | Demos | Use to us |
|---|---|---|---|---|
| **fazilaltinel**, branch `bestiary-domestic-cat` (upstream PR #123, open) | 1 commit, 16 files, cat.rs +378, tests +173 | Domestic cat from published anatomy: meow (jaw moves F1), kitten mew, trill, hiss, **growl, snarl**, purr that also sounds on the in-breath (gated valve at 25 Hz). Formants are within 5% of a published cat tube model (Ekström et al. 2024: 629/2707/5165 vs 602/2675/5105 Hz). Purr timing and level match Schötz & Eklund 2011; hiss and growl numbers match Schötz 2015 (growl about 130 Hz, 2.5 s; snarl about 475 Hz). | Yes: per-call WAVs and a spectrogram are attached to PR #123 | **Best in show.** It is the only fork that validates calls against numbers and includes a growl/snarl. Its tables of growl f0, duration and hiss centre of gravity make good targets for our presets. |
| **huntsinger-c** main | 6 ahead / 24 behind, 18 files, cat.rs +420 | House cat (meow, trill, purr, hiss); stroke to purr, tap to meow; oversampling fix for short airways below 44.1 kHz; a documented "rules for adding a creature" process with tests | Hosted UI only (vercel); no WAVs | Second cat. Its oversampling fix matters for short tracts. |
| **CypherNaught-0x**, branch `self-oscillating-folds` | 6 ahead, 33 files, folds.rs +731 | Three-mass body-cover folds (Story & Titze 1995; Titze & Story 2002) driven by real airway pressures; zipper closure; falsetto vs chest; **flapping false folds**; pressed rough tone with push-roughness that saturates at 40 cmH₂O; `scream_probe` example | None | Best physical source. Roughness, chaos and false-fold flapping emerge from pressure and tension instead of being painted on. Good as an optional "physical" source mode later; heavy. |
| **dwilliams27**, branch `living-folds` | 1 ahead, 21 files, folds.rs +606, docs/living-folds.md | Opt-in self-oscillating folds (chest, falsetto, the break) | None | Same idea, better documented |
| **knonode** PR #45 (closed, unmerged), branch `throat-singing` | +1262, 12 files | **Kargyraa via ventricular folds closing on every other glottal cycle** (Bailly, Henrich & Pelorson 2010): an area waveform locked to the glottal cycle that caps the pharynx area, giving Borda–Carnot loss and turbulence | Measurement table in docs | **Directly fixes our subharmonic mistake.** Period doubling should be a cycle-locked modulation of the main airflow, not an added f0/2 layer. |
| **robertDouglass** PR #103 (closed) | +1010 | Two-mass folds (Ishizaka–Flanagan, as in VocalTractLab); a raised larynx presses the folds | | Lighter alternative to the three-mass model |
| **rtalbot** main | 8 ahead / 74 behind, 58 files | Ten invented aliens: stalker (inner-jaw predator: hiss through teeth plus a pressed shriek with an aerodynamic whistle, tube lengths that change as it strikes), skywhale, swarm, ventboiler, whistler, chorister, glasssinger, accordion, bellows, resonant; alien languages | Record-to-video button; no WAVs | Good for invented monsters. "Stalker" is the closest to a horror creature. The idea to borrow is **tract geometry that moves during the call**. |
| **samshapley** PR #36 (closed) | +1138 | Alien body: scaled human airway plus organs (throat sac Helmholtz cavity, second voicebox, crest) | | "Size" and "extra organ" macros for monsters |
| **schani** PR #66 (closed) | +5603 | Flatulence creature checked against 31 real recordings | Video on X | The only creature **checked against real recordings**. A model for how to validate a creature (method, not use). |
| **hed0rah**, branch `exo-atmospheres` | 9 ahead | A gas model (speech in other atmospheres) | | Niche: an alien-atmosphere macro |
| **markschroedr/chimera-radio** | 14 ahead | Render a "creature score" from timed params, gates and calls | | Confirms the point: creatures need **timed parameter tracks** |
| **Franzlazarus/chimera-lab** | 3 ahead, 19 stars | VST3 adaptation, "strange throat textures" | | Plugin packaging only |

### Checked, nothing animal-related
- **Description says creature, nothing pushed:** rockinghelvetica ("adapt this to a cat") and hrspwdr/chimera-horse are both 0 ahead.
- **Language and music forks:** schani main (hill climbs, MusicXML), soundworkshop (MPE), verusferro (covers), SaturnityGL (MIDI), alfarouq637 (Arabic), gattocci (Spanish), lumia1998 (Mandarin), monomyth (Russian), Denjin-K (bagpipe), robert-gilliam (tape).
- **Unrelated small changes:** KaiMiyazawa, evergreenstrategies, CDC6600, yuejunzhang, cupOJoseph, sohlis, WZehan, hugoriosbrito.

**On X:** almost nothing beyond launch hype ("push it into animal voices") and a Chinese post noting the repo had 40 forks within three days. No creature showcases. GitHub search for creature/animal/growl synthesis also turned up nothing of quality; the open field really is soundgen plus CHIMERA plus Farnell.

## 8. Rebuild plan (proposal, nothing built yet)

**Stage A: analysis (new `soundlab/analysis/voice_tracks.py`)**
- Every 5 ms, from a recording: f0 (searched above the subharmonic, then octave-checked as in v21) plus voicing confidence; spectral envelope (CheapTrick method, or pyworld if we accept the dependency); aperiodicity per band (D4C method).
- New tracks WORLD doesn't give us:
  - **subharmonic depth** at f0/2 and f0/3, measured per frame as the energy at half-harmonics against harmonics, like soundgen's subDep;
  - **period jitter and shimmer per cycle** from the epochs;
  - **regime label** per frame (0 periodic, 1 subharmonic, 2 chaotic).
- Tracks are stored as compact breakpoint curves so recipes stay small JSON.

**Stage B: synthesis core (a new `voice_tracks` model)**
- A pulse-synchronous harmonic source driven by the f0 track.
- Period doubling is applied as a **cycle-locked modulation**, alternating amplitude and timing at subharmonic depth. This follows the kargyraa / ventricular-fold method from PR #45 and Bailly 2010; no parallel f0/2 layer.
- Chaos episodes come from the regime track: per-cycle jitter and shimmer bursts, so the logistic map is used only *inside* those episodes.
- The spectral envelope and aperiodicity are applied frame by frame (harmonic amplitudes sampled from the envelope, as in DDSP; noise shaped by aperiodicity).
- Validation target: before any editing, it should at least match plain WORLD (≤0.21 on the test growl) **and** match the reference's 40 Hz band within 3 dB.

**Stage C: macros that edit tracks, never replace them**
- **size**: formant dispersion scaling plus an f0 shift, as in deer PSOLA practice and canid size cues.
- **aggression / effort**: raises subharmonic depth and the chaos-episode rate where the reference already has them; adds rasp aperiodicity in the 1–4 kHz band.
- **rasp**: per-band aperiodicity gain. **throat**: tract length scaling over time.
- **morph**: blend the tracks of two analysed calls (dog growl → demon).

**Stage D: presets from real recordings**
- Each preset is the analysed tracks of a licence-clean recording plus macro settings. Recordings themselves never go in the repo, only derived curves.
- Clean sources: our own recorded human performances (growls pitched down, the creature-designer standard), CC0 Freesound, and published call statistics as targets (the fazilaltinel cat tables, Klemuk lion f0, Schötz cat growl).

**Stage E: fixing the fit metric**
- Add three terms to the loss: subharmonic-depth-over-time distance, period irregularity distance, and formant-trajectory distance.
- Once stage A exists, CMA-ES only tunes macros, never raw tracks.

**Optional later**
- A physical source mode (two- or three-mass folds with flapping false folds) ported from the MIT forks, with attribution in THIRD_PARTY.md.
- A sample-layer slot (user-supplied file) for hero sounds.

## 9. Licence summary

| Thing | Licence | How we'd use it |
|---|---|---|
| CHIMERA and every fork above | MIT | Code port allowed, with attribution in THIRD_PARTY.md |
| WORLD | Modified BSD-3 | Port or depend on it, with attribution |
| pyworld | MIT | Optional dependency |
| DDSP | Apache-2.0 | Method (or code, with NOTICE) |
| HiFi-GAN, BigVGAN | MIT | Reference only |
| soundgen | GPL | Method only |
| Loris | GPL | Method only |
| Parselmouth | GPL-3 | Method only |
| SMS-tools, NoiseBandNet | AGPL | Method only |
| RAVE | CC BY-NC 4.0 | Avoid |
| SPEAR | Closed freeware | Avoid |
| STRAIGHT | Not open | Avoid |
| Farnell Pd patches | Book companion code | Method only |
