# door_stems

Doors, locks, knocks, gates, a drawer and a garage door rebuilt from stems of CC0 recordings. Numbers only, as
in [foley_stems.md](foley_stems.md): event band envelopes with each event's length and 1 ms level, a 1 ms
sub-band noise bed per segment, body-mode tracks for metal rings, a colour curve, a treble ceiling and the
recording's own silent stretches. The small hinge squeak (`door_creak_small`) also has a tracked harmonic voice:
each harmonic is stored as a mode track that follows k x f0(t) with its own measured amplitude and phase. No
samples are stored. Sources are listed in `THIRD_PARTY.md`.

Recipes are in `recipes/doors/` and play one real gesture by default (the knocks are one natural set: three on
wood, three muffled, two on metal). Front controls: Intensity, Size (tape speed), Distance; knocks, the door
handle and the key in a lock also show Count and Rate.

| Library | Recipes |
|---|---|
| Wood door | `doors/door_wood_open` (Part: latch / swing), `doors/door_wood_close`, `doors/door_wood_slam`, `doors/door_creak`, `doors/door_handle` |
| Other doors | `doors/door_heavy`, `doors/door_metal_close`, `doors/door_metal_slam`, `doors/door_glass`, `doors/door_screen`, `doors/door_sliding`, `doors/car_door`, `doors/cupboard_door`, `doors/wardrobe_door`, `doors/drawer`, `doors/garage_door` |
| Gates | `doors/gate_wood`, `doors/gate_metal` |
| Locks and keys | `doors/key_lock`, `doors/deadbolt` (Part: lock / unlock) |
| Knocks | `doors/knock_wood`, `doors/knock_wood_muffled`, `doors/knock_metal` |
| Hinge | `doors/door_creak_small` |

The older door recipes stay as hidden aliases that still render exactly as before: `wood_knock`, `knock`,
`hc_knock` and `hit` (now listed as `doors/knock_wood`), `door_slam` (`doors/door_wood_slam`), `door_creak` and
`creak_model/door_creak` (`doors/door_creak`), `door_squeal` (`doors/door_creak_small`) and `gate_squeal`
(`doors/gate_metal`). `gate_squeal_long` and `hinge_squeak` stay in the library.
