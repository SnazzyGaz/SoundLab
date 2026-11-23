# interact_stems

Everyday interaction sounds rebuilt from stems of CC0 recordings: camera shutters, a film advance, drink
handling, velcro, a lighter, a plastic lid, a phone vibrating, typing, a mouse, tape, a stapler, scissors and
a spray can. The package holds analysis numbers only: the 1/8-octave band envelopes of each recorded event
with the event's own length and 1 ms level, a 1/8-octave 1 ms sub-band noise bed per segment, body-mode
partial tracks where the object rings (a glass of ice, a beer bottle), a 1/6-octave colour curve, a static
treble ceiling and, where the recording itself is digitally silent between gestures, those silent stretches.
No samples of the recordings are stored. Sources are listed in `THIRD_PARTY.md`.

Recipes (in `recipes/interact/`) play one real gesture by default. Intensity (`force`), Size (tape speed) and
Distance are the front controls; softness, count and rate are under Advanced, except where Count and Rate sit
on the front for a short run of takes (typing, scissors, a stapler, a mouse, a spray, a phone vibrate).
Variation remixes the same real stems (never averaged) and, where the recording holds several takes of the
same action, plays another take.

| Library | Recipes |
|---|---|
| Camera | `interact/camera_shutter`, `interact/film_advance`, `interact/phone_shutter` |
| Kitchen & tableware | `interact/can_open`, `interact/bottle_cap`, `interact/cork_pop`, `interact/bottle_twist`, `interact/ice_cubes`, `interact/plastic_lid` |
| Cloth & bags | `interact/velcro` (Part: open / close) |
| Household | `interact/lighter`, `interact/phone_vibrate`, `interact/keyboard`, `interact/mouse_click`, `interact/tape_peel`, `interact/stapler`, `interact/scissors`, `interact/spray_can` |

These do not replace sounds the library already has: the foley pen click, light switch, page turn, book close,
zips, drawer, cup / glass / bottle put down, or the liquid-stems pour into a glass.

The renderer is the shared one in `soundlab/models/recstems.py`, with the same optional parts used by
`foley_stems` and `door_stems`: each event ends with its real grain (3 ms fade) and is held at or under the
grain's own 1 ms RMS level; the bed of each segment fades in at the segment start; mode phases are measured
from the start of the recording; and the stored silent stretches are kept silent. Nothing is pitch-shifted per
band, frozen or time-squeezed.
