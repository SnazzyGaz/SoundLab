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

## Recordings decomposed into stems for `squeal_stems` (v0.29, numbers only, no audio included)

`soundlab/models/squeal_stems.bin` holds analysis numbers of bursts from these CC0 1.0 recordings on Freesound:
partial tracks (start frame, pitch in cents, level in dB per 2.9 ms frame), 96-band residual envelopes, wobble
tracks, onset and body spectra, and the pauses between bursts. No samples of the recordings are stored; the sound
is resynthesised from the numbers. Which burst of which recording each recipe plays is listed in the recipe's
`notes` and in `docs/friction.md`.

| Freesound id | Title | Author | URL | Licence |
|---|---|---|---|---|
| 88502 | sneaker squeak rubber | shakaharu | https://freesound.org/people/shakaharu/sounds/88502/ | CC0 1.0 |
| 124289 | squeaky-floor-loud | alienistcog | https://freesound.org/people/alienistcog/sounds/124289/ | CC0 1.0 |
| 204759 | hinge slow motion | Yuval | https://freesound.org/people/Yuval/sounds/204759/ | CC0 1.0 |
| 207140 | OldGate | ahill86 | https://freesound.org/people/ahill86/sounds/207140/ | CC0 1.0 |
| 268574 | Hollow_Door_Creak | 7h3_lark | https://freesound.org/people/7h3_lark/sounds/268574/ | CC0 1.0 |
| 397687 | Creaking Gate | piermic | https://freesound.org/people/piermic/sounds/397687/ | CC0 1.0 |
| 429130 | Spring Loaded Metal Door Creak Open Close 02 | leonelmail | https://freesound.org/people/leonelmail/sounds/429130/ | CC0 1.0 |
| 455324 | Heavy Door Creaking_05 | rambler52 | https://freesound.org/people/rambler52/sounds/455324/ | CC0 1.0 |
| 483166 | Gate Creak 2 | recklessgoomba | https://freesound.org/people/recklessgoomba/sounds/483166/ | CC0 1.0 |
| 535530 | hinge-squeaks | cyoung510 | https://freesound.org/people/cyoung510/sounds/535530/ | CC0 1.0 |
| 563136 | door creak | ollitunes | https://freesound.org/people/ollitunes/sounds/563136/ | CC0 1.0 |
| 575206 | cold room door creaks opening close 08 | TRP | https://freesound.org/people/TRP/sounds/575206/ | CC0 1.0 |
| 766155 | creaking door | dianegray26 | https://freesound.org/people/dianegray26/sounds/766155/ | CC0 1.0 |
| 784964 | Squeaky Door | apintofmild | https://freesound.org/people/apintofmild/sounds/784964/ | CC0 1.0 |
| 788701 | Soft_sqeuak_door_hinge | T.I.B | https://freesound.org/people/T.I.B/sounds/788701/ | CC0 1.0 |

## Recordings decomposed into stems for `air_stems` (v0.30, numbers only, no audio included)

`soundlab/models/air_stems.bin` holds analysis numbers (band-noise envelopes at 2.9 ms frames, tonal tracks,
1 ms onset envelopes, a 1/6-octave spectrum, a fan's blade cycle, fly-by pass geometry) of stretches of these
CC0 1.0 recordings on Freesound. No samples are stored; renders use new noise.

| Freesound id | Title | Author | URL | Licence |
|---|---|---|---|---|
| 352421 | wind_gust.aif | joseph.larralde | https://freesound.org/people/joseph.larralde/sounds/352421/ | CC0 1.0 |
| 507466 | Clean fast Swoosh.aiff | Danjocross | https://freesound.org/people/Danjocross/sounds/507466/ | CC0 1.0 |
| 675821 | S27-05 Five arrows whoosh by.wav | craigsmith | https://freesound.org/people/craigsmith/sounds/675821/ | CC0 1.0 |
| 760990 | 5. Wind in trees | mathiaslyhne1 | https://freesound.org/people/mathiaslyhne1/sounds/760990/ | CC0 1.0 |
| 761929 | large indoor fan on low - 1 | FOSSarts | https://freesound.org/people/FOSSarts/sounds/761929/ | CC0 1.0 |

## Recordings decomposed into stems for `liquid_stems` (v0.31, numbers only, no audio included)

`soundlab/models/liquid_stems.bin` holds analysis numbers (per-band noise envelopes at 2 ms on 1/8-octave bands,
bubbles as chirp settings, STFT band envelopes, tonal tracks, onset envelopes, a 1/6-octave spectrum) of stretches
of these public-domain, Public Domain Mark and CC0 recordings. No samples are stored; renders use new noise.

| Source | Title | Author | Where | URL | Licence |
|---|---|---|---|---|---|
| splash_small (45.3-47.6 s), splash_medium (30.2-33.0 s) | Bathtub water splashes.ogg | gradha | Wikimedia Commons | https://commons.wikimedia.org/wiki/File:Bathtub_water_splashes.ogg | Public domain |
| splash_large (whole) | WATRSplsh_Water Splash, Big, Dive Into Water, Person or Animal | Nicholas A. Judy (The Designer's Choice) | Internet Archive | https://archive.org/details/Designers-Choice-Collection-Water | CC0 1.0 |
| drip (0-8 s) | WATRDrip_Bathtub Dripping | Nicholas A. Judy (The Designer's Choice) | Internet Archive | https://archive.org/details/Designers-Choice-Collection-Water | CC0 1.0 |
| pour_glass (whole) | WATRPour_Water Pouring, Into Cup | Nicholas A. Judy (The Designer's Choice) | Internet Archive | https://archive.org/details/Designers-Choice-Collection-Water | CC0 1.0 |
| stream (2-10 s) | 363120 fractalstudios water-trickle.wav | FractalStudios | Wikimedia Commons | https://commons.wikimedia.org/wiki/File:363120_fractalstudios_water-trickle.wav | CC0 |
| river (10-18 s) | Georgia Rapids.ogg | Jarrod stanley | Wikimedia Commons | https://commons.wikimedia.org/wiki/File:Georgia_Rapids.ogg | CC0 |
| waterfall (10-18 s) | 26081101391556.flac (radio aporee 73066) | t.fille (radio aporee) | Internet Archive | https://archive.org/details/aporee_73066_85299 | Public Domain Mark 1.0 |
| boiling (0.5-8.5 s) | Boiling lentils.ogg | hugh | Wikimedia Commons | https://commons.wikimedia.org/wiki/File:Boiling_lentils.ogg | Public domain |
| slosh (whole) | Sloshing canister of water.ogg | stephan | Wikimedia Commons | https://commons.wikimedia.org/wiki/File:Sloshing_canister_of_water.ogg | Public domain |

## Recordings decomposed into stems for `machine_stems` (v0.32, numbers only, no audio included)

`soundlab/models/machine_stems.bin` holds analysis numbers (a tracked fundamental, harmonic and line amplitude and
phase tracks at 4 ms, per-band noise envelopes at 1 ms on 1/8-octave bands, per-cycle pulse profiles, tick / stitch
band envelopes at 0.5 ms, a 1/6-octave spectrum) of stretches of these CC0 and public-domain recordings. No samples
are stored; renders use new noise. The `_gesture` takes supply the real spin-up, steady and spin-down stretches for
gesture timelines.

| Source | Title | Author | Where | URL | Licence |
|---|---|---|---|---|---|
| ac_unit (10-20 s) | Window Air Conditioner in Apartment | benhabrams | Freesound | https://freesound.org/people/benhabrams/sounds/521033/ | CC0 |
| car_drive_by (4-16 s) | car_passing_by | zamoranqk | Freesound | https://freesound.org/people/zamoranqk/sounds/808404/ | CC0 |
| clock_tick (2-12 s) | Clock, ticking | natalie (pdsounds.org) | Wikimedia Commons | https://commons.wikimedia.org/wiki/File:Clock_ticking.ogg | Public domain |
| diesel_idle (10-20 s) | engine idle van diesel exhaust CBC microwave truck side4 nearby | kyles | Freesound | https://freesound.org/people/kyles/sounds/453995/ | CC0 |
| lawn_mower (26-38 s) | 166cc gas-powered push mower; starting, pulling cord, playing with control lever | Mihacappy | Freesound | https://freesound.org/people/Mihacappy/sounds/774661/ | CC0 |
| lawn_mower_gesture (0-12 s) | 166cc gas-powered push mower; starting, pulling cord, playing with control lever | Mihacappy | Freesound | https://freesound.org/people/Mihacappy/sounds/774661/ | CC0 |
| motorbike_passby (2-11.5 s) | 10_motorbike_650_passing_by | 16HPanskaFantys_Tomas | Freesound | https://freesound.org/people/16HPanskaFantys_Tomas/sounds/499237/ | CC0 |
| petrol_rev (14-26 s) | auto performance corvette ext top open shift into gear, start, idle, rev from engine | kyles | Freesound | https://freesound.org/people/kyles/sounds/637184/ | CC0 |
| power_drill (24-36 s) | cordless_drill | ondondvo | Freesound | https://freesound.org/people/ondondvo/sounds/393422/ | CC0 |
| power_drill_gesture (55.6-58.8 s) | cordless_drill | ondondvo | Freesound | https://freesound.org/people/ondondvo/sounds/393422/ | CC0 |
| servo_motor (2-14 s) | Small electronic motor | derjuli | Freesound | https://freesound.org/people/derjuli/sounds/418013/ | CC0 |
| sewing_machine (0.5-11.5 s) | SewingMachineLong.wav | Infinita08 | Freesound | https://freesound.org/people/Infinita08/sounds/546441/ | CC0 |
| tram (0.5-12.5 s) | TRAM_001.wav | rollingmill | Freesound | https://freesound.org/people/rollingmill/sounds/340456/ | CC0 |

## Recordings decomposed into stems for `fire_stems` (v0.33, numbers only, no audio included)

`soundlab/models/fire_stems.bin` holds analysis numbers (per-band noise envelopes at 1 ms on 1/8-octave bands, the
band envelopes of each recorded crackle / pop / click at 0.5 ms (0.125 ms for the gas stove's clicks) inside its own
window, a tracked fundamental with harmonic and line amplitude and phase tracks at 4 ms, a 1/6-octave spectrum, a
static treble ceiling and transient times) of stretches of these CC0 recordings. No samples are stored; renders use
new noise.

| Source | Title | Author | Where | URL | Licence |
|---|---|---|---|---|---|
| campfire (20-32 s) | crackling campfire | aerror | Freesound #767107 | https://freesound.org/people/aerror/sounds/767107/ | CC0 |
| electric_arc (1-13 s) | Jacob's ladder 2.wav | jensfelger | Freesound #367455 | https://freesound.org/people/jensfelger/sounds/367455/ | CC0 |
| embers (81-93 s) | Sound of wood charcoal slow burning | tadiyan | Freesound #645460 | https://freesound.org/people/tadiyan/sounds/645460/ | CC0 (as listed in Freesound's CC0-filtered search) |
| fluorescent_flicker (30-42 s) | fluorescent lamp flickering.flac | arseniiv | Freesound #380660 | https://freesound.org/people/arseniiv/sounds/380660/ | CC0 |
| fuse (16.5-25.5 s) | Firework Fuse | Joblez | Freesound #646928 | https://freesound.org/people/Joblez/sounds/646928/ | CC0 |
| gas_stove_ignite (0.2-7.7 s) | turning on gas stove 2 | FOSSarts | Freesound #740092 | https://freesound.org/people/FOSSarts/sounds/740092/ | CC0 |
| large_blaze (8-20 s) | R27-25-Large Fire Burning.wav | craigsmith | Freesound #483318 | https://freesound.org/people/craigsmith/sounds/483318/ | CC0 |
| match_strike (0.1-6.3 s) | Match Ignite.aif | DiArchangeli | Freesound #462111 | https://freesound.org/people/DiArchangeli/sounds/462111/ | CC0 |
| sparkler (2-14 s) | sparkler_fuse_nm.wav | Alex_hears_things | Freesound #316682 | https://freesound.org/people/Alex_hears_things/sounds/316682/ | CC0 |
| transformer_hum (24-36 s) | transformer at night.wav | ideaph | Freesound #641208 | https://freesound.org/people/ideaph/sounds/641208/ | CC0 |

## Recordings decomposed into stems for `impact_stems` (v0.34, numbers only, no audio included)

`soundlab/models/impact_stems.bin` holds analysis numbers of gestures from these CC0 1.0 recordings on Freesound. No samples are stored; renders use new noise and the stored envelopes.

| Source (cut) | Title | Author | Site | URL | Licence |
|---|---|---|---|---|---|
| ball_bounce (12-23 s) | Bouncy ball bouncing on floor | 89o_dev | Freesound #869172 | https://freesound.org/people/89o_dev/sounds/869172/ | CC0 |
| body_punch (0.5-13.2 s) | Punches/hits | Wigglesworth | Freesound #853588 | https://freesound.org/people/Wigglesworth/sounds/853588/ | CC0 |
| ceramic_break (0-13.4 s) | pottery-dish-smash-4x-samples.wav | ragamuffin | Freesound #186363 | https://freesound.org/people/ragamuffin/sounds/186363/ | CC0 |
| coin_drop (0-15 s) | coin(s) spin drop.wav | Yuval | Freesound #197214 | https://freesound.org/people/Yuval/sounds/197214/ | CC0 |
| coin_jingle (0-4.3 s) | coin jangle_in hand_fast.wav | ekfink | Freesound #235487 | https://freesound.org/people/ekfink/sounds/235487/ | CC0 |
| dowel (0.2-6.8 s) | CLAVES | SamuelGremaud | Freesound #517609 | https://freesound.org/people/SamuelGremaud/sounds/517609/ | CC0 |
| drumstick (0-11 s) | Drum Sticks.wav | MioMeg | Freesound #707012 | https://freesound.org/people/MioMeg/sounds/707012/ | CC0 |
| glass_break (8-20 s) | various window breaks smashes.flac | TRP | Freesound #575283 | https://freesound.org/people/TRP/sounds/575283/ | CC0 |
| glass_clink (3-15 s) | Glasses clinking | esmelane | Freesound #806416 | https://freesound.org/people/esmelane/sounds/806416/ | CC0 |
| metal_clang (0-12 s) | clang_metal_(hit).wav | Godowan | Freesound #240463 | https://freesound.org/people/Godowan/sounds/240463/ | CC0 |
| pickaxe_earth (2-16 s) | digging into mud.wav | Stefan21100190 | Freesound #593113 | https://freesound.org/people/Stefan21100190/sounds/593113/ | CC0 |
| pickaxe_rock (0-14 s) | Garden pickaxe hitting rock | Sadiquecat | Freesound #855629 | https://freesound.org/people/Sadiquecat/sounds/855629/ | CC0 |
| wood_drop (11-22.4 s) | wood plank impact ground hits forest bouncing | unfa | Freesound #857862 | https://freesound.org/people/unfa/sounds/857862/ | CC0 |
| wood_knock (0-12 s) | Door knocks wood close 77mel 191026.wav | TRP | Freesound #573835 | https://freesound.org/people/TRP/sounds/573835/ | CC0 |

## Recordings decomposed into stems for `weapon_stems` (v0.34, numbers only, no audio included)

`soundlab/models/weapon_stems.bin` holds analysis numbers of gestures from these CC0 1.0 recordings on Freesound. No samples are stored; renders use new noise and the stored envelopes.

| Source (cut) | Title | Author | Site | URL | Licence |
|---|---|---|---|---|---|
| arrow_impact (0-1.5 s) | ARROW_WOOD_IMPACT_SINGLE_ARCHERY_01.wav | JoeDinesSound | Freesound #534956 | https://freesound.org/people/JoeDinesSound/sounds/534956/ | CC0 |
| assault_rifle_burst (0.1-4.2 s) | M16 burst in street.aif | Franki-01234 | Freesound #201668 | https://freesound.org/people/Franki-01234/sounds/201668/ | CC0 |
| assault_rifle_shot (0-2.6 s) | AR15 rifle shot | michorvath | Freesound #427596 | https://freesound.org/people/michorvath/sounds/427596/ | CC0 |
| axe_chop (0-7.1 s) | chopping wood_medium [459] | ElSchorscho | Freesound #847818 | https://freesound.org/people/ElSchorscho/sounds/847818/ | CC0 |
| blade_slide (0-4.9 s) | Escrime swords hits in practice room | YannSauvin | Freesound #778321 | https://freesound.org/people/YannSauvin/sounds/778321/ | CC0 |
| bolt_cycle (0-12 s) | Mosin Nagant Bolt Action Cycle | Zott820 | Freesound #370345 | https://freesound.org/people/Zott820/sounds/370345/ | CC0 |
| bow_release (0-0.3 s) | Longbow Release 2.wav | saturdaysoundguy | Freesound #394179 | https://freesound.org/people/saturdaysoundguy/sounds/394179/ | CC0 |
| brass_casings (0-12 s) | Brass bullet shell casing drop onto concrete, multiple takes | GrayJoy | Freesound #210102 | https://freesound.org/people/GrayJoy/sounds/210102/ | CC0 |
| charging_handle (0-1.2 s) | Wasr RPK Charge Foreward1.mp3 | Pjkasinski3 | Freesound #171631 | https://freesound.org/people/Pjkasinski3/sounds/171631/ | CC0 |
| club_hit (0-0.6 s) | Bat_hit2 | hashtagsmcgee | Freesound #851452 | https://freesound.org/people/hashtagsmcgee/sounds/851452/ | CC0 |
| crossbow (19.5-23.5 s) | EK Cobra Pistol Crossbow.mp3 | Tairblenn | Freesound #540406 | https://freesound.org/people/Tairblenn/sounds/540406/ | CC0 |
| distant_gunfire (0-7.6 s) | Distant Gunfire | iainmccurdy | Freesound #842326 | https://freesound.org/people/iainmccurdy/sounds/842326/ | CC0 |
| dry_fire (0-0.3 s) | 9mm Handgun Being Dry Fired | serøutōnin--deprivəd | Freesound #674568 | https://freesound.org/people/ser%C3%B8ut%C5%8Dnin--depriv%C9%99d/sounds/674568/ | CC0 |
| explosion_close (0.4-1.1 s) | petarda_jaka.wav | mandzo18 | Freesound #60880 | https://freesound.org/people/mandzo18/sounds/60880/ | CC0 |
| explosion_distant (12.4-14.8 s) | Three Outdoor Clean Explosions (Fireworks on New Year's Eve 2021) | unfa | Freesound #613673 | https://freesound.org/people/unfa/sounds/613673/ | CC0 |
| firework_bang (0.5-3.2 s) | tro_petarda.wav | mandzo18 | Freesound #60885 | https://freesound.org/people/mandzo18/sounds/60885/ | CC0 |
| hammer_cock (0.1-0.6 s) | revolvercock1.wav | Dredile | Freesound #111676 | https://freesound.org/people/Dredile/sounds/111676/ | CC0 |
| knife_draw (0-5.9 s) | Knife Block Unsheath Kitchen Knife | loganzsound | Freesound #873758 | https://freesound.org/people/loganzsound/sounds/873758/ | CC0 |
| knife_flick (0-12 s) | opening and closing switchblade.wav | 170129 | Freesound #408259 | https://freesound.org/people/170129/sounds/408259/ | CC0 |
| knife_slash (11.1-19.5 s) | Blade swipes knives etc | spaghettiSFX | Freesound #795665 | https://freesound.org/people/spaghettiSFX/sounds/795665/ | CC0 |
| knife_stab (0-12 s) | Knife Stabs into Chicken Breasts #2 | CHallSmith | Freesound #870739 | https://freesound.org/people/CHallSmith/sounds/870739/ | CC0 |
| magazine_change (0.2-7.2 s) | A Glock handgun having its magazine removed and re-inserted // 3 takes | serøutōnin--deprivəd | Freesound #854460 | https://freesound.org/people/ser%C3%B8ut%C5%8Dnin--depriv%C9%99d/sounds/854460/ | CC0 |
| pistol_reload (0-4.3 s) | Browning Hi-Power handgun being reloaded while empty. Full sequence | serøutōnin--deprivəd | Freesound #674569 | https://freesound.org/people/ser%C3%B8ut%C5%8Dnin--depriv%C9%99d/sounds/674569/ | CC0 |
| pistol_shot (0.1-12.5 s) | gun_.22mm_7.5mm_9mm_close_single_shots_m10.wav | johanwestling | Freesound #377786 | https://freesound.org/people/johanwestling/sounds/377786/ | CC0 |
| revolver_shot (0-8.8 s) | Revolver Gunshots | C-V | Freesound #835200 | https://freesound.org/people/C-V/sounds/835200/ | CC0 |
| ricochet (1.1-1.6 s) | 22 caliber gunfire with ricochet.wav | C-V | Freesound #523404 | https://freesound.org/people/C-V/sounds/523404/ | CC0 |
| rifle_reload (0-7.1 s) | An M16 Rifle being reloaded while empty; full sequence | serøutōnin--deprivəd | Freesound #725397 | https://freesound.org/people/ser%C3%B8ut%C5%8Dnin--depriv%C9%99d/sounds/725397/ | CC0 |
| rifle_shot (0.1-0.8 s) | gun lee enfield 303 rifle fire shot loud badass crispy.wav | kyles | Freesound #450852 | https://freesound.org/people/kyles/sounds/450852/ | CC0 |
| safety_click (0-4.7 s) | Pocket Pistol safety switch | mahecic | Freesound #803265 | https://freesound.org/people/mahecic/sounds/803265/ | CC0 |
| shield_block_metal (0.1-4.8 s) | Shield / sword hits.wav | PixelsphereStudios | Freesound #182112 | https://freesound.org/people/PixelsphereStudios/sounds/182112/ | CC0 |
| shotgun_pump (0-12 s) | Remington 870 12 gauge shotgun pump racking cha chock | TotallyPhilip | Freesound #755495 | https://freesound.org/people/TotallyPhilip/sounds/755495/ | CC0 |
| shotgun_shot (0-1.7 s) | 20 gauge shotgun gunshot | michorvath | Freesound #427595 | https://freesound.org/people/michorvath/sounds/427595/ | CC0 |
| slide_rack (0-10 s) | Pocket Pistol Slide rack | mahecic | Freesound #803266 | https://freesound.org/people/mahecic/sounds/803266/ | CC0 |
| smg_burst (0-2.4 s) | MP5 submachine gun down the street.aif | Franki-01234 | Freesound #201671 | https://freesound.org/people/Franki-01234/sounds/201671/ | CC0 |
| steel_target_ping (0.3-1.4 s) | Anvil Hit 2 | michorvath | Freesound #270588 | https://freesound.org/people/michorvath/sounds/270588/ | CC0 |
| suppressed_shot (0-0.6 s) | Springfield Armory 1911 Operator (Suppressed) | areniporgen | Freesound #827945 | https://freesound.org/people/areniporgen/sounds/827945/ | CC0 |
| sword_clash (0-12 s) | Escrime swords hits in practice room | YannSauvin | Freesound #778321 | https://freesound.org/people/YannSauvin/sounds/778321/ | CC0 |
| sword_draw (0-1.8 s) | Sword drawing | xkeril | Freesound #666581 | https://freesound.org/people/xkeril/sounds/666581/ | CC0 |
| sword_sheathe (4.9-5.4 s) | OWI Sword Draw and sheath.wav | matthewHoldenSound | Freesound #542516 | https://freesound.org/people/matthewHoldenSound/sounds/542516/ | CC0 |
| tree_break (2.5-8.2 s) | Heavy explosion in forest debris tree falling.flac | wyskoj | Freesound #530163 | https://freesound.org/people/wyskoj/sounds/530163/ | CC0 |

## Recordings decomposed into stems for `foot_stems` (v0.34, numbers only, no audio included)

`soundlab/models/foot_stems.bin` holds analysis numbers of gestures from these CC0 1.0 recordings on Freesound. No samples are stored; renders use new noise and the stored envelopes.

| Source (cut) | Title | Author | Site | URL | Licence |
|---|---|---|---|---|---|
| concrete_walk (0-6.7 s) | Footsteps - Stone, Rock, Concrete, Cement | SecureSubset | Freesound #813622 | https://freesound.org/people/SecureSubset/sounds/813622/ | CC0 |
| concrete_run (5.2-14.9 s) | paving_18c_trekkingboots_run.wav | sturmankin | Freesound #273117 | https://freesound.org/people/sturmankin/sounds/273117/ | CC0 |
| stone_walk (0-3.5 s) | Hiking Boot Footsteps on Stone | Fission9 | Freesound #521590 | https://freesound.org/people/Fission9/sounds/521590/ | CC0 |
| wood_floor (2-8.1 s) | Footsteps on Wood | snakebarney | Freesound #138123 | https://freesound.org/people/snakebarney/sounds/138123/ | CC0 |
| wood_creaky (0-10.1 s) | Heavy Footsteps on a Squeaky Floor | deleted_user_7146007 | Freesound #383731 | https://freesound.org/people/deleted_user_7146007/sounds/383731/ | CC0 |
| wood_stairs (0-12.4 s) | walking up wood stairs in shoes | Bastard Fish | Freesound #326985 | https://freesound.org/people/Bastard%20Fish/sounds/326985/ | CC0 |
| gravel_walk (0-9 s) | Footsteps_Mountain_Boots_Gritty_Ground_Stones_Pebbles_Mono.wav | Nox_Sound | Freesound #543806 | https://freesound.org/people/Nox_Sound/sounds/543806/ | CC0 |
| dirt_walk (0-9 s) | Footsteps_On_Dirt.wav | lzmraul | Freesound #389454 | https://freesound.org/people/lzmraul/sounds/389454/ | CC0 |
| grass_walk (0-4.4 s) | Hiking Boot Footsteps on Grass | Fission9 | Freesound #521587 | https://freesound.org/people/Fission9/sounds/521587/ | CC0 |
| grass_run (20.8-30.5 s) | Footsteps_Mountain_Boots_Grass_Mono.wav | Nox_Sound | Freesound #556042 | https://freesound.org/people/Nox_Sound/sounds/556042/ | CC0 |
| sand_walk (16.4-24 s) | Soft sand steps.wav | 200221-WeanBekker | Freesound #543714 | https://freesound.org/people/200221-WeanBekker/sounds/543714/ | CC0 |
| snow_walk (0-11.3 s) | Footsteps_Mountain_Boots_Snow_Walk_Mono.wav | Nox_Sound | Freesound #613849 | https://freesound.org/people/Nox_Sound/sounds/613849/ | CC0 |
| mud_walk (0-5.5 s) | Footsteps_Mud_01.wav | BenDrain | Freesound #488068 | https://freesound.org/people/BenDrain/sounds/488068/ | CC0 |
| puddle_walk (0-7 s) | puddlewalk.wav | j1987 | Freesound #106395 | https://freesound.org/people/j1987/sounds/106395/ | CC0 |
| metal_walk (0-5.4 s) | Footsteps On Metal | IENBA | Freesound #834029 | https://freesound.org/people/IENBA/sounds/834029/ | CC0 |
| metal_stairs (0-11.2 s) | Metal stairs .WAV | DDT197 | Freesound #445776 | https://freesound.org/people/DDT197/sounds/445776/ | CC0 |
| tile_walk (0-6.1 s) | Footsteps On Tile | IENBA | Freesound #834027 | https://freesound.org/people/IENBA/sounds/834027/ | CC0 |
| carpet_walk (0-5.5 s) | Footsteps on carpet.wav | theshuggie | Freesound #371017 | https://freesound.org/people/theshuggie/sounds/371017/ | CC0 |
| leaves_walk (0-4.1 s) | Footsteps_Leaves_Stereo.wav | Nox_Sound | Freesound #496420 | https://freesound.org/people/Nox_Sound/sounds/496420/ | CC0 |
| boots_walk (0-9.3 s) | WoodType_1_Footstep_LeatherBoot.wav | monte32 | Freesound #405112 | https://freesound.org/people/monte32/sounds/405112/ | CC0 |
| heels_walk (0-4.4 s) | 15_Walking_high_heel.wav | Juaner_ | Freesound #471416 | https://freesound.org/people/Juaner_/sounds/471416/ | CC0 |
| trainers_walk (0-5.1 s) | WoodType_1_Footstep_Sneakers.wav | monte32 | Freesound #405114 | https://freesound.org/people/monte32/sounds/405114/ | CC0 |
| barefoot_wood (0-4.6 s) | walk_indoor_barefoot.mp3 | dragonmin | Freesound #554592 | https://freesound.org/people/dragonmin/sounds/554592/ | CC0 |
| jump_land (1.3-2.1 s) | Jumping Onto Wooden Floor.wav | krnash | Freesound #389800 | https://freesound.org/people/krnash/sounds/389800/ | CC0 |
| jump_land_rock (0-4.9 s) | Footsteps_Mountain_Boots_Rock_Jump_Sequence_Mono.wav | Nox_Sound | Freesound #558477 | https://freesound.org/people/Nox_Sound/sounds/558477/ | CC0 |



## Recordings decomposed into stems for `foley_stems` (numbers only, no audio included)

`soundlab/models/foley_stems.bin` holds analysis numbers of gestures from these CC0 1.0 recordings on Freesound. No samples are stored; renders use new noise and the stored envelopes, tracks and level caps. The foley drawer was not kept (the door_stems drawer is used).

| Source (cut) | Title | Author | Site | URL | Licence |
|---|---|---|---|---|---|
| cloth_handling (4.35-6.75 s) | Clothing Ruffle 01.wav | WhiteFire43 | Freesound #641383 | https://freesound.org/people/WhiteFire43/sounds/641383/ | CC0 |
| cloth_heavy (1.65-4.6 s) | Jacket / Campera de Cuerina | IENBA | Freesound #483943 | https://freesound.org/people/IENBA/sounds/483943/ | CC0 |
| zipper_bag (0.25-4.15 s) | Opening & closing zipper on a bag | randbsoundbites | Freesound #829757 | https://freesound.org/people/randbsoundbites/sounds/829757/ | CC0 |
| zipper_jacket (2.12-3.05 s) | Zipper_01.wav | RobSp1derp1g | Freesound #615428 | https://freesound.org/people/RobSp1derp1g/sounds/615428/ | CC0 |
| keys_jingle (1.55-4.25 s) | Keys Jingling | bwarpus99 | Freesound #443863 | https://freesound.org/people/bwarpus99/sounds/443863/ | CC0 |
| keys_down (0.35-1.37 s) | Put Down Keys.wav | sophiehall3535 | Freesound #248046 | https://freesound.org/people/sophiehall3535/sounds/248046/ | CC0 |
| paper_rustle (0.25-2.56 s) | rustling paper.wav | keweldog | Freesound #181774 | https://freesound.org/people/keweldog/sounds/181774/ | CC0 |
| paper_crumple (0-2.19 s) | Crumpling Paper | OwlStorm | Freesound #151231 | https://freesound.org/people/OwlStorm/sounds/151231/ | CC0 |
| page_turn (0-1.78 s) | Book Page Turning | EminYILDIRIM | Freesound #535358 | https://freesound.org/people/EminYILDIRIM/sounds/535358/ | CC0 |
| book_close (1.18-2.4 s) | Closing book | ondondvo | Freesound #195795 | https://freesound.org/people/ondondvo/sounds/195795/ | CC0 |
| cup_down (0.95-2.4 s) | Mug On Table: Setting Down and Picking Up | haus.the.child | Freesound #839346 | https://freesound.org/people/haus.the.child/sounds/839346/ | CC0 |
| glass_down (0-1 s) | Put Down Glass Object.flac | qubodup | Freesound #171939 | https://freesound.org/people/qubodup/sounds/171939/ | CC0 |
| bottle_down (3.25-5.2 s) | glass bottle on a wooden table | ventousawins | Freesound #681410 | https://freesound.org/people/ventousawins/sounds/681410/ | CC0 |
| cutlery (0.1-3.5 s) | Dropping silverware | cb9382 | Freesound #873865 | https://freesound.org/people/cb9382/sounds/873865/ | CC0 |
| bag_down (5.1-6.22 s) | Bag Of Flour Dropped On Counter Top.wav | bbrocer | Freesound #382642 | https://freesound.org/people/bbrocer/sounds/382642/ | CC0 |
| backpack_handle (0-2.4 s) | Backpack Pickup Putdown | j1987 | Freesound #237046 | https://freesound.org/people/j1987/sounds/237046/ | CC0 |
| cupboard (0-1.75 s) | Magnetic latch cupboard door | wlabarron | Freesound #509112 | https://freesound.org/people/wlabarron/sounds/509112/ | CC0 |
| coin_drop (0.15-2.65 s) | Dropping Single Coin On Floor 4.wav | keatonmarek | Freesound #533774 | https://freesound.org/people/keatonmarek/sounds/533774/ | CC0 |
| phone_down (0-1.08 s) | phone cordless plastic put down on table.flac | kyles | Freesound #637747 | https://freesound.org/people/kyles/sounds/637747/ | CC0 |
| chair_scrape (0.15-1.7 s) | Wooden Chair Move.mp3 | harrietniamh | Freesound #219677 | https://freesound.org/people/harrietniamh/sounds/219677/ | CC0 |
| pen_click (0.3-1.6 s) | pen_clicking_sound_effect(custom) | 8bid9537 | Freesound #753411 | https://freesound.org/people/8bid9537/sounds/753411/ | CC0 |
| light_switch (0.45-1.4 s) | Light switch.wav | boop_7 | Freesound #684372 | https://freesound.org/people/boop_7/sounds/684372/ | CC0 |
| plastic_bag (0-2.86 s) | Noisy Little Plastic Bag | TheWandermiles | Freesound #697256 | https://freesound.org/people/TheWandermiles/sounds/697256/ | CC0 |

## Recordings decomposed into stems for `door_stems` (numbers only, no audio included)

`soundlab/models/door_stems.bin` holds analysis numbers of gestures from these CC0 1.0 recordings on Freesound. No samples are stored; renders use new noise and the stored envelopes, tracks and level caps.

| Source (cut) | Title | Author | Site | URL | Licence |
|---|---|---|---|---|---|
| door_wood_open (0-1.55 s) | wooden door open, close | doppido01 | Freesound #805974 | https://freesound.org/people/doppido01/sounds/805974/ | CC0 |
| door_wood_close (4.76-5.4 s) | Door Slam | AlyssaB2043 | Freesound #466850 | https://freesound.org/people/AlyssaB2043/sounds/466850/ | CC0 |
| door_wood_slam (4.05-5.35 s) | A wooden door being opened and closed | serøutōnin--deprivəd | Freesound #725396 | https://freesound.org/people/ser%C3%B8ut%C5%8Dnin--depriv%C9%99d/sounds/725396/ | CC0 |
| door_creak (0-3.84 s) | Heavy Door Creaking_05.wav | rambler52 | Freesound #455324 | https://freesound.org/people/rambler52/sounds/455324/ | CC0 |
| door_creak_small (1.6-2.6 s) | Door squeak | TiesWijnen | Freesound #455539 | https://freesound.org/people/TiesWijnen/sounds/455539/ | CC0 |
| door_heavy (0.15-3.12 s) | Heavy_wooden_door_close_mansion.wav | elliott.klein | Freesound #367221 | https://freesound.org/people/elliott.klein/sounds/367221/ | CC0 |
| door_metal_close (5.06-5.83 s) | Metal Shed Door Close 2 | Yoyodaman234 | Freesound #334682 | https://freesound.org/people/Yoyodaman234/sounds/334682/ | CC0 |
| door_metal_slam (0.15-1.17 s) | metaldoorslam | saha213131 | Freesound #730196 | https://freesound.org/people/saha213131/sounds/730196/ | CC0 |
| door_handle (0.05-2.07 s) | Opening Door Handle | amholma | Freesound #344355 | https://freesound.org/people/amholma/sounds/344355/ | CC0 |
| key_lock (0-1.13 s) | Locking Door with Key.wav | baidonovan | Freesound #187352 | https://freesound.org/people/baidonovan/sounds/187352/ | CC0 |
| deadbolt (0.1-1.6 s) | Light Metal Deadbolt Lock Unlock | loganzsound | Freesound #873133 | https://freesound.org/people/loganzsound/sounds/873133/ | CC0 |
| knock_wood_muffled (0-1.2 s) | Door Knock - 42.wav | Ligidium | Freesound #193873 | https://freesound.org/people/Ligidium/sounds/193873/ | CC0 |
| knock_wood (0.25-1.35 s) | Apartment Door Knock | Ultra-Edward | Freesound #828194 | https://freesound.org/people/Ultra-Edward/sounds/828194/ | CC0 |
| knock_metal (0-1.2 s) | door knock.wav | clothespeg | Freesound #628576 | https://freesound.org/people/clothespeg/sounds/628576/ | CC0 |
| door_sliding (0-3.6 s) | Door sliding along a metal track | wlabarron | Freesound #509114 | https://freesound.org/people/wlabarron/sounds/509114/ | CC0 |
| door_glass (0.2-1.73 s) | Wooden/Glass door slammed  48000 Hz 24 bit Stereo | diogorusso | Freesound #612598 | https://freesound.org/people/diogorusso/sounds/612598/ | CC0 |
| door_screen (6.1-8.4 s) | screen door slam.wav | 3dm0nd_M | Freesound #380036 | https://freesound.org/people/3dm0nd_M/sounds/380036/ | CC0 |
| car_door (8.7-9.9 s) | car door.wav | pawsound | Freesound #154832 | https://freesound.org/people/pawsound/sounds/154832/ | CC0 |
| cupboard_door (5.2-6.6 s) | Cupboard.wav | willsauntsounddesign | Freesound #564031 | https://freesound.org/people/willsauntsounddesign/sounds/564031/ | CC0 |
| wardrobe_door (4.6-6.5 s) | Door_wood_wardrobe_1.WAV | o_ciz | Freesound #475488 | https://freesound.org/people/o_ciz/sounds/475488/ | CC0 |
| gate_wood (1.99-3.05 s) | Wooden Gate | RealSquink | Freesound #788015 | https://freesound.org/people/RealSquink/sounds/788015/ | CC0 |
| gate_metal (0-1.5 s) | Gate made out of metal and wire close | zxz41 | Freesound #241931 | https://freesound.org/people/zxz41/sounds/241931/ | CC0 |
| drawer (1.6-2.9 s) | Drawer Open Close.wav | ThunderQuads | Freesound #467194 | https://freesound.org/people/ThunderQuads/sounds/467194/ | CC0 |
| garage_door (0.45-3.1 s) | Garage door closing | Walking.With.Microphones | Freesound #624397 | https://freesound.org/people/Walking.With.Microphones/sounds/624397/ | CC0 |

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
