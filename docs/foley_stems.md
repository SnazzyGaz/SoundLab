# foley_stems

Everyday handling sounds rebuilt from stems of CC0 recordings. The package holds analysis numbers only: the
1/8-octave band envelopes of each recorded event with the event's own length and 1 ms level, a 1/8-octave 1 ms
sub-band noise bed per segment, body-mode partial tracks where the object rings (cups, glasses, bottles,
cutlery, coins), a 1/6-octave colour curve, a static treble ceiling and, where the recording itself is digitally
silent between gestures, those silent stretches. No samples of the recordings are stored. Sources are listed in
`THIRD_PARTY.md`.

Recipes (in `recipes/foley/`) play one real gesture by default. Intensity (`force`), Size (tape speed) and
Distance are the front controls; softness, count and rate are under Advanced. Variation remixes the same real
stems (never averaged) and, where the recording holds several takes of the same action, plays another take.

| Library | Recipes |
|---|---|
| Cloth & bags | `foley/cloth_handling`, `foley/cloth_heavy` (three flops as takes), `foley/zipper_bag` (Part: open / close), `foley/zipper_jacket`, `foley/bag_down`, `foley/backpack_handle`, `foley/plastic_bag` |
| Paper & books | `foley/paper_rustle`, `foley/paper_crumple`, `foley/page_turn` (gentle by default; two takes), `foley/book_close` |
| Kitchen & tableware | `foley/cup_down`, `foley/glass_down`, `foley/bottle_down`, `foley/cutlery` (two takes) |
| Household | `foley/keys_jingle` (four takes; Count and Rate on the front), `foley/keys_down`, `foley/coin_drop`, `foley/phone_down`, `foley/pen_click` (two takes), `foley/light_switch`, `foley/chair_scrape`, `foley/cupboard` (Part: open with its magnetic latch / close) |

`foley/coin_drop` is one Coin drop entry with a Shape choice: `single` (a single coin, this family) or `spin`
(the longer spinning drop from `impact_stems`, identical to the previous `coin_drop` recipe at every setting).

The renderer is the shared one in `soundlab/models/recstems.py`. This family turns on three optional parts of
it that the earlier stem families do not use: each event ends with its real grain (3 ms fade) and is held at or
under the grain's own 1 ms RMS level; the bed of each segment fades in at the segment start; mode phases are
measured from the start of the recording; and the stored silent stretches are kept silent (5 ms ramps inside
them). Nothing is pitch-shifted per band, frozen or time-squeezed.
