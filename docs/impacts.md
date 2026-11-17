# Impacts and materials (`material_impact`, v0.26)

One contact + resonance model covers knocks, drops, bounces, footsteps, tool strikes, clashes, clinks, break
onsets and body hits. Instead of a hand-built recipe per sound, a recipe picks a **material**, a **shape** and a
**size**, and its macros move the physics: size, force, hardness, hollowness, muffle, contact point, material
(or surface for footsteps), distance.

## How it works

| Stage | Method | Source |
|---|---|---|
| Contact | Hunt-Crossley force F = k x^1.5 (1 + 0.3 xdot) between striker and target; their stiffnesses add in series (rubber sole on oak is soft, steel pick on granite is hard). Slower or softer contacts last longer and are duller by themselves; bounces reuse the rebound speed. | Hunt & Crossley 1975; Avanzini & Rocchesso 2001 (via `hc_contact`) |
| Body | Modal synthesis. Frequencies from the shape's mode formula and the material's E, rho, nu, so pitch scales as 1/size: free-free beam (`bar`, `rod`, `beam`), simply supported plate with golden-ratio sides (`plate`; no near-harmonic coincidences), thin-ring bending modes n(n^2-1)/sqrt(n^2+1) (`shell`: cups, bowls, pots, pipes), closed thin spherical shell (`ball`), thick solid from the shear wave speed (`block`), thin-walled plate + air-cavity modes c/2L (`box`). | van den Doel, Kry & Pai 2001 (FoleyAutomatic); Leissa (plates, rings); thin-shell sphere result (Silbiger/Kalnins) |
| Damping | alpha(f) = (d0 + pi eta f) exp(h f / 5 kHz) per mode. Material is heard mostly from how decay grows with frequency, so each material has its own eta (loss factor), d0 (mounting/radiation) and h (high-frequency growth). | Aramaki, Besson, Kronland-Martinet & Ystad 2011; Klatzky, Pai & Krotkov 2000; Cremer & Heckl (eta) |
| Not metallic | Non-metals get a steep frequency law, a ring-time cap (wood 0.25-0.4 s, stone 0.18 s, plastic 0.3 s), 2 % mode scatter and random mode levels; only steel, aluminium and iron ring freely. Test: wood, stone, plastic, rubber and flesh plates stay under 0.6 s T60; steel and aluminium are above 1.5 s. | |
| Residual | A short noise burst shaped by the object's own spectrum (the modal response cut to ~4 ms), scaled per material: stone and soft wood are half noise, metal and glass almost none. | Ren, Yeh & Lin 2013 (residual compensation in example-guided modal sound) |
| Ground | `dirt`, `gravel`, `grass`, `leaves`: no modes; a damped low thud plus granular grains (power-law energies, dense first then thinning; pebble clicks are tonal, soil and grass are noise). | Sethna et al. 2001; Bresin et al. 2010 (walking on surfaces); SDT crumpling: method only |
| Fracture | An energy budget spent in shards pitched by the material (`fracture`). | crumple `break` method |
| Gestures | `hit` (with `bounces` and `restitution`) or `footstep` (heel, ball of the foot 60-110 ms later at about half the speed, and a scuff). | |

## Materials

`wood_soft`, `wood_hard`, `steel`, `aluminium`, `iron`, `glass`, `ceramic`, `stone`, `plastic`, `rubber`,
`flesh`, `dirt`, `gravel`, `grass`, `leaves`. Strikers: `none` (rigid), `flesh`, `wood`, `steel`, `rubber`,
`stone`, `glass`, `shoe`. `striker_ring` adds the striker's own ring (pick, sword, spoon, glass).

## Recipes

`impact_any` (every material and shape on choice macros), `knock`, `drop`, `bounce`, `footstep_wood`,
`footstep_stone`, `footstep_metal`, `footstep_dirt`, `footstep_gravel`, `footstep_grass` (each with a `surface`
choice macro, leaves included), `pickaxe_earth`, `pickaxe_rock`, `blade_clash`, `glass_clink`, `ceramic_break`,
`body_hit`. Every recipe has seeded variations (speed, contact point and size jitter plus a new seed).

## Checks against recordings (outside the repo)

Loss is `soundlab.fit.compare_signals` (lower is better). Fits tune 7-10 controls with CMA-ES on one take, then a
residual eq limited to +-6 dB; the other takes are held out.

| Recipe | Fitted take | Held-out takes | Older recipe on the same takes |
|---|---|---|---|
| `knock` | 0.70 | 0.86, 1.21 | `hit` 0.84 / 0.76, 1.12; `hc_knock` 1.04 / 0.85, 1.09 |
| `footstep_wood` | 0.86 | 0.50 | `footstep` 1.04 / 0.46 |
| `blade_clash` | 0.50 | 0.61 | `sword_clash` 0.43 / 0.49 |
| `body_hit` | 0.46 | 1.12 | `punch` 0.61 / 1.07 |
| `pickaxe_earth` | 0.22 (0.26-0.32 over variations) | | |

The one general model is about level with the older hand-built recipes: better on fitted takes, roughly equal or
slightly worse on held-out ones, and worse than `sword_clash` (whose dispersive blade ring the bar modes lack).
Its value is coverage: one set of controls for every material, shape and gesture.

## v0.26.1 fixes

- **Footsteps are normal walking, not stomping.** Heel speed ~1 m/s, a softer heel, the foot's weight damps the
  floor (+60 1/s on every mode, ring capped at 50 ms on non-metals, 150 ms on metal), the ball of the foot comes
  in at 40 % speed and a quarter of the level, the scuff is a quarter of before, a sole low-pass (6 kHz hard floors,
  10 kHz ground) and a 50 Hz high-pass, and the recipes sit 10 dB under full scale. Tests: every footstep peaks at
  least 9 dB under a knock, nothing rings 150 ms after the heel.
- **A break is a crack, not a ding.** With `fracture` the body's modes stop within ~12-110 ms (less the harder the
  break), a 1-2 ms broadband snap and a short low floor thud mark the crack, and shards land 20-250 ms later,
  bounce 1-4 times with shrinking gaps and levels, and settle. Shard character is per material: ceramic and stone
  are dull, short and partly noisy; glass and metal ring. Test: no partial stands more than 20 dB above its
  neighbours after 50 ms (a glass clink scores above 40).
- **A body hit is an untuned thump.** Flesh no longer goes through modes or a long contact pulse with rising
  noise: two smooth pulses (an instant ~1 ms attack and the heavier mass behind it), a short decaying low tail
  and a faint slap. Tests: the envelope peaks within 5 ms and never swells afterwards; no spectral peak stands
  more than 11 dB above its neighbours (a blade clash scores above 25). The v0.26 residual eq on `body_hit`
  (a +6 dB bump at 125-200 Hz) rang like a tone and was removed.

## Known gaps

- Inflated balls (membrane tension) and wooden doors' low boom need larger, tension-aware models; knock fits push
  to the speed and hardness limits.
- Residual eqs on several fits sit at the +-6 dB limit, so the spectra are still off by more than that in places.
- No contact nonlinearity between two ringing bodies (both ring independently), no rolling or scraping here
  (use `surface`), no radiation efficiency below the coincidence frequency.
