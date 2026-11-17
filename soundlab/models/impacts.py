"""material_impact (v0.26): one contact + resonance model for knocks, drops, bounces, footsteps, tool strikes,
clashes, clinks, breaks and body hits.

Method (all original code; sources in the model's `sources` list and THIRD_PARTY.md):
- Contact: Hunt-Crossley force between striker and target (Hunt & Crossley 1975; Avanzini & Rocchesso 2001),
  reusing `hc_pulse`. The two contact stiffnesses combine in series, so a rubber sole on oak is soft and a steel
  pick on granite is hard; softer or slower contacts are longer and duller by themselves.
- Body: modal synthesis (van den Doel, Kry & Pai 2001, FoleyAutomatic). Mode frequencies come from the shape's
  textbook mode formula and the material's E, rho and nu, so size scales pitch as 1/size: free-free beam (bar,
  rod, beam), simply supported plate, thin ring/shell bending modes n(n^2-1)/sqrt(n^2+1) (cups, bowls, pots,
  pipes), closed spherical shell (balls), thick block, and a box = thin wall plate + air-cavity modes c/(2 L).
- Damping: per-mode decay alpha(f) = (d0 + pi eta f) exp(h f / 5 kHz). The material loss factor eta and the
  frequency law follow Aramaki, Kronland-Martinet et al. (2006-2011): material is heard mainly from how decay
  grows with frequency. Wood, stone, plastic and other non-metals get a steep law, a ring-time cap and a few
  per cent of mode scatter, so they never ring like metal.
- Residual: a short noise burst shaped by the object's own spectrum (the "residual" that Ren, Yeh & Lin 2013
  add to modal fits for realism), scaled per material: wood and stone are partly noise, metal and glass are not.
- Ground (dirt, gravel, grass, leaves): no modes; a damped low thud plus granular grains (power-law sizes,
  as in crumpling/crackling control: Sethna et al. 2001; Bresin et al. 2010 walking on surfaces).
- Fracture: an energy budget spent in shards that come fast and thin out (crumple `break` method), pitched by
  the material.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .. import dsp
from ..core import Ctx, Param, model
from .sdt_methods import G, hc_pulse

C_AIR = 343.0

# rho kg/m3, E Pa, nu, eta (loss factor), d0 1/s (mounting/radiation), h (Aramaki-style HF damping growth),
# k contact stiffness N/m^1.5, resid (noise share), cap (max ring time s; metals uncapped), shard Hz, metal
# Values: Cremer & Heckl via vibrationdata (rho, E, eta), Klatzky et al. 2000 (perceived decay order), tuned so
# each class keeps its decay character; ground materials have no modes.
IMPACT_MATERIALS: dict[str, dict] = {
    "wood_soft":  dict(rho=450, E=9e9, nu=0.3, eta=0.025, d0=30, h=1.6, k=2e7, resid=0.45, cap=0.25, shard=1800),
    "wood_hard":  dict(rho=750, E=12e9, nu=0.3, eta=0.013, d0=20, h=1.1, k=8e7, resid=0.3, cap=0.4, shard=2500),
    "steel":      dict(rho=7800, E=210e9, nu=0.31, eta=3e-4, d0=1.5, h=0.05, k=5e10, resid=0.04, cap=None,
                       shard=6000, metal=True),
    "aluminium":  dict(rho=2700, E=70e9, nu=0.34, eta=2e-4, d0=2.0, h=0.05, k=3e10, resid=0.05, cap=None,
                       shard=6000, metal=True),
    "iron":       dict(rho=7200, E=120e9, nu=0.26, eta=1.5e-3, d0=4.0, h=0.25, k=3e10, resid=0.1, cap=None,
                       shard=5000, metal=True),
    "glass":      dict(rho=2500, E=62e9, nu=0.24, eta=1e-3, d0=3.0, h=0.3, k=4e10, resid=0.05, cap=2.5, shard=6500),
    "ceramic":    dict(rho=2300, E=60e9, nu=0.22, eta=2.5e-3, d0=8.0, h=0.5, k=3e10, resid=0.15, cap=1.0, shard=4500),
    "stone":      dict(rho=2500, E=40e9, nu=0.22, eta=8e-3, d0=40, h=0.9, k=2e10, resid=0.5, cap=0.18, shard=3500),
    "plastic":    dict(rho=1050, E=2.3e9, nu=0.35, eta=0.03, d0=25, h=1.3, k=3e7, resid=0.3, cap=0.3, shard=3000),
    "rubber":     dict(rho=1100, E=0.05e9, nu=0.49, eta=0.15, d0=60, h=2.0, k=5e5, resid=0.2, cap=0.12, shard=1000),
    "flesh":      dict(rho=1050, E=0.02e9, nu=0.45, eta=0.35, d0=90, h=2.0, k=4e5, resid=0.6, cap=0.08, shard=800),
    "dirt":       dict(ground=True, k=1e6, thud_hz=140, grain=(500, 4500), grain_ms=2.5, n=45, spread=0.07,
                       tonal=0.5, resid=0.5),        # packed soil: crumbs and small pebbles click
    "gravel":     dict(ground=True, k=6e6, thud_hz=220, grain=(1500, 7000), grain_ms=3.0, n=70, spread=0.12,
                       tonal=0.8, resid=0.3),
    "grass":      dict(ground=True, k=3e5, thud_hz=110, grain=(1200, 9000), grain_ms=6.0, n=60, spread=0.15,
                       tonal=0.0, resid=0.7),
    "leaves":     dict(ground=True, k=3e5, thud_hz=120, grain=(2500, 12000), grain_ms=1.5, n=110, spread=0.18,
                       tonal=0.35, resid=0.4),
}
STRIKERS = {"none": None, "flesh": "flesh", "wood": "wood_hard", "steel": "steel", "rubber": "rubber",
            "stone": "stone", "glass": "glass", "shoe": "rubber"}
STRIKER_K = {"shoe": 4e6}           # a shoe heel is stiffer than soft rubber
SHAPES = ("plate", "bar", "rod", "beam", "box", "shell", "ball", "block")
# thickness / size for a solid object of each shape (hollowness thins the walls)
T_RATIO = {"plate": 0.03, "bar": 0.03, "rod": 0.02, "beam": 0.1, "box": 0.012, "shell": 0.02, "ball": 0.03, "block": 0.5}
BAR_BL = [4.7300, 7.8532, 10.9956, 14.1372, 17.2788]


def mode_table(shape: str, m: dict, size: float, hollow: float, pos: float, n: int, rng) -> tuple:
    """(freqs Hz, amps) of the target's modes from its shape, size and material."""
    E, rho, nu = m["E"], m["rho"], m["nu"]
    t = T_RATIO[shape] * size * (1.4 - 0.8 * hollow)
    k = np.arange(1, n + 1)
    if shape in ("bar", "rod", "beam"):
        rg = t / 4 if shape == "rod" else t / math.sqrt(12)          # radius of gyration
        bl = np.array(BAR_BL + [(2 * j + 1) * math.pi / 2 for j in range(6, n + 2)])[:n]
        f = bl ** 2 / (2 * math.pi * size ** 2) * rg * math.sqrt(E / rho)
        amp = np.abs(np.sin((k + 0.5) * math.pi * (0.03 + 0.94 * pos))) + 0.05
    elif shape == "block":  # thick solid: dense low-order modes from the shear wave speed (~0.4 c_s / size)
        c = math.sqrt(E / (2 * (1 + nu) * rho))
        r = np.cumsum(np.concatenate([[1.0], rng.uniform(0.12, 0.35, n - 1)]))
        f = 0.4 * c / size * r
        amp = 1 / np.sqrt(k) * rng.uniform(0.5, 1.0, n)
    elif shape in ("plate", "box"):
        L, W = size, size * 0.618               # golden ratio sides: no near-harmonic mode coincidences
        D = E * t ** 3 / (12 * (1 - nu ** 2))
        c0 = (math.pi / 2) * math.sqrt(D / (rho * t))
        mm = [(a, b) for a in range(1, 14) for b in range(1, 14)]
        fr = np.array([c0 * ((a / L) ** 2 + (b / W) ** 2) for a, b in mm])
        o = np.argsort(fr)[:n]
        f = fr[o]
        sp = (0.05 + 0.9 * pos, 0.37 + 0.2 * pos)
        amp = np.array([abs(math.sin(mm[i][0] * math.pi * sp[0]) * math.sin(mm[i][1] * math.pi * sp[1])) + 0.05
                        for i in o])
    elif shape == "ball":  # closed thin spherical shell: membrane lower branch below the ring frequency, plus
        # bending (Kalnins / Silbiger thin-shell result, simplified): W^2 = m_n + beta (n(n+1) - 2)^2 / (n(n+1))
        R = size / 2
        fr = math.sqrt(E / (rho * (1 - nu ** 2))) / (2 * math.pi * R)
        nn = np.arange(2, n + 2)
        q = nn * (nn + 1)
        W2 = (q - 2) * (1 + nu) / (q + 1 + nu) + (t / R) ** 2 / 12 * (q - 2) ** 2 / q
        f = fr * np.sqrt(W2)
        amp = 1 / np.sqrt(nn - 1)
    else:  # shell: thin ring bending modes (n, 0), plus weaker axial (n, 1) partners
        R = size / 2
        base = t / (2 * math.pi * R ** 2) * math.sqrt(E / (12 * rho * (1 - nu ** 2)))
        nn = np.arange(2, n // 2 + 3)
        f0 = base * nn * (nn ** 2 - 1) / np.sqrt(nn ** 2 + 1)
        f = np.concatenate([f0, f0 * 1.45])[:n]
        amp = np.concatenate([1 / np.sqrt(nn - 1), 0.35 / np.sqrt(nn - 1)])[:n]
        o = np.argsort(f)
        f, amp = f[o], amp[o]
    if not m.get("metal") and m["E"] < 100e9:
        amp = amp * rng.uniform(0.6, 1.0, len(amp))
    return np.asarray(f, float), np.asarray(amp, float), t


def cavity_modes(size: float) -> np.ndarray:
    dims = (size, size * 0.7, size * 0.5)
    f = sorted({C_AIR / 2 * math.sqrt((a / dims[0]) ** 2 + (b / dims[1]) ** 2 + (c / dims[2]) ** 2)
                for a in range(3) for b in range(3) for c in range(2) if a + b + c > 0})
    return np.array(f[:8])


def decay_rates(f, m, muffle, extra=0.0):
    a = (m["d0"] + extra + 120 * muffle ** 1.5 + math.pi * m["eta"] * f) * np.exp(m["h"] * (1 + muffle) * f / 5000)
    return a


def modal_ir(f, amp, alpha, sr, max_t60, cap, rng, scatter):
    t60 = 6.91 / alpha
    if cap:
        t60 = np.minimum(t60, cap)
    t60 = np.minimum(t60, max_t60)
    keep = (f < 0.45 * sr) & (f > 20)
    f, amp, t60 = f[keep], amp[keep], t60[keep]
    if len(f) == 0:  # every mode above Nyquist (tiny, stiff objects): the contact click itself
        ir = np.zeros(16)
        ir[0] = 1.0
        return ir
    if scatter:
        f = f * (1 + scatter * rng.uniform(-1, 1, len(f)))
    L = int(max(t60.max(), 0.01) * sr) + 1
    t = np.arange(L) / sr
    ir = (amp[:, None] * np.exp(-6.91 / t60[:, None] * t) *
          np.sin(2 * math.pi * f[:, None] * t + rng.uniform(0, 0.4, len(f))[:, None])).sum(0)
    return ir / (np.abs(ir).max() + 1e-12)


def _contact_k(material_k, striker, hardness):
    ks = STRIKER_K.get(striker) or (IMPACT_MATERIALS[STRIKERS[striker]]["k"] if STRIKERS.get(striker) else 1e12)
    k = 1.0 / (1.0 / material_k + 1.0 / ks)
    return k * 10 ** (2.0 * (hardness - 0.5))


def _grains(sr, rng, n_events, spread, band, grain_ms, tonal, size_exp=1.5, decay_rate=None):
    """Granular burst: n_events grains over `spread` s (dense first, thinning), power-law energies."""
    L = int((spread * 2.5 + 0.03) * sr)
    out = np.zeros(L)
    lo, hi = band
    for _ in range(int(n_events)):
        tt = rng.exponential(spread / 2.2) if decay_rate is None else rng.exponential(decay_rate)
        i = int(min(tt, spread * 2.4) * sr)
        e = min(6.0, (1 - rng.random()) ** (-1 / size_exp)) / 3
        fc = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        T = grain_ms * 1e-3 * (0.5 + rng.random()) * (2000 / fc) ** 0.3
        Lg = int(T * 6 * sr) + 16
        tg = np.arange(Lg) / sr
        env = np.exp(-tg / T)
        if rng.random() < tonal:
            g = sum(np.sin(2 * math.pi * fc * r * (1 + 0.03 * rng.standard_normal()) * tg + rng.uniform(0, 6.28)) / (j + 1)
                    for j, r in enumerate((1.0, 2.32, 4.25)) if fc * r < 0.45 * sr) * env
        else:
            nz = rng.standard_normal(Lg)
            g = dsp.bandpass(nz, min(fc, 0.4 * sr), sr, q=1.2) * env
        dsp.place(out, g, i, math.sqrt(e))
    return out


# shard character per material: (band lo, hi Hz, grain ms, tonal share). Ceramic and stone fragments are dull,
# short and partly noisy; glass and metal pieces ring (the sparkle belongs to them, not to pottery).
SHARD = {"glass": (2500, 11000, 5.0, 0.9), "ceramic": (900, 5000, 1.2, 0.35), "stone": (700, 4500, 0.9, 0.25),
         "steel": (2500, 10000, 8.0, 0.95), "aluminium": (2500, 10000, 6.0, 0.95), "iron": (2000, 8000, 4.0, 0.9)}


def _shards(sr, rng, n_shards, m, room_s, name):
    """Fragments thrown off by a break: each lands 20-250 ms later and bounces 1-4 times with shrinking gaps and
    levels (restitution ~0.45), then settles; a few skitter. Pitch from the material, decay short for pottery."""
    lo, hi, gms, tonal = SHARD.get(name, (m["shard"] * 0.4, m["shard"] * 1.6, 1.5, 0.4))
    L = int(min(room_s, 1.6) * sr)
    out = np.zeros(max(L, 16))
    for _ in range(n_shards):
        t = rng.uniform(0.0, 0.03) + rng.exponential(0.06) + 0.02
        gap = rng.uniform(0.04, 0.16)
        e = min(4.0, (1 - rng.random()) ** (-1 / 1.5)) / 3
        size = rng.uniform(0.4, 1.0)
        fc = math.exp(rng.uniform(math.log(lo), math.log(min(hi, 0.42 * sr)))) / size ** 0.5
        for b in range(int(rng.integers(1, 5))):
            i = int(t * sr)
            if i >= L:
                break
            T = gms * 1e-3 * (0.6 + 0.8 * rng.random())
            Lg = int(T * 6 * sr) + 16
            tg = np.arange(Lg) / sr
            env = np.exp(-tg / T)
            if rng.random() < tonal and fc * 2.3 < 0.45 * sr:
                g = sum(np.sin(2 * math.pi * fc * r * tg + rng.uniform(0, 6.28)) / (j + 1)
                        for j, r in enumerate((1.0, 1.59, 2.31))) * env
            else:
                g = dsp.bandpass(rng.standard_normal(Lg), min(fc, 0.4 * sr), sr, q=0.9) * env
            dsp.place(out, g, i, math.sqrt(e) * 0.45 ** b)
            t += gap
            gap *= 0.45
    return out


def _body_thump(out, i0, lvl, size, hardness, residual, sr, rng):
    """A blow to a body: soft tissue over a heavy, lossy mass. Instant attack, then a short low-frequency thump
    (low-passed noise under a decaying envelope: no modes, so no pitch) and a faint dull slap of skin or cloth.
    Nothing rises after the onset (a long Hunt-Crossley pulse into flesh plus noise reads like reversed air)."""
    ta = 1.0 / (2 * math.pi * (110.0 + 160.0 * hardness))       # pulse width sets how low the thump sits
    tb = 4 * ta * (0.6 + 0.8 * min(size, 1.0))                     # the heavier mass behind it
    tau = 0.012 + 0.025 * min(size, 1.0)
    L = int(max(12 * tb, 7 * tau) * sr)
    t = np.arange(L) / sr
    body = (t / ta) * np.exp(1 - t / ta) + 0.6 * (t / tb) * np.exp(1 - t / tb)   # single smooth pulses: no pitch
    tail = dsp.lowpass(dsp.lowpass(rng.standard_normal(L), 300.0, sr), 300.0, sr) * np.exp(-t / tau)
    body = body / body.max() + 0.25 * tail / (np.abs(tail).max() + 1e-12)
    dsp.place(out, body / (np.abs(body).max() + 1e-12), i0, lvl)
    Ls = int(0.012 * sr)
    slap = dsp.lowpass(dsp.highpass(rng.standard_normal(Ls), 500.0, sr), 2500.0, sr) * np.exp(-np.arange(Ls) / (0.0018 * sr))
    dsp.place(out, slap / (np.abs(slap).max() + 1e-12), i0, lvl * 0.18 * residual * (0.4 + hardness))


@model("material_impact", "sound",
       "One contact + resonance model for knocks, drops, bounces, footsteps, tool strikes, clashes, clinks, "
       "breaks and body hits. Pick a `material` (wood, metal, glass, ceramic, stone, plastic, rubber, flesh, "
       "dirt, gravel, grass, leaves), a `shape` and `size`; a Hunt-Crossley contact between `striker` and target "
       "excites the target's modes (shape formula x material, decay law per material, so non-metals never ring "
       "metallic) plus a material-shaped noise residual. `hollowness` thins the walls and adds air-cavity modes; "
       "`muffle` damps and darkens; `contact_point` moves the strike from edge to middle. Ground materials give a "
       "thud plus grains. `gesture` footstep plays heel, ball and a scuff; `fracture` adds shards.",
       [
           Param("material", "wood_hard", "", choices=tuple(IMPACT_MATERIALS)),
           Param("shape", "plate", "", choices=SHAPES),
           Param("size", 0.3, "m", 0.01, 5.0, "largest dimension of the struck object (pitch ~ 1/size)"),
           Param("at", 0.01, "s", 0, None),
           Param("velocity", 2.0, "m/s", 0.05, 30, "impact speed (louder, shorter, brighter contact)"),
           Param("striker", "wood", "", choices=tuple(STRIKERS), doc="what hits it (none = rigid)"),
           Param("striker_mass_g", 200.0, "g", 1, 100000, "mass of the striking object"),
           Param("hardness", 0.5, "", 0, 1, "contact stiffness trim: 0 = padded, 1 = hard edge"),
           Param("hollowness", 0.3, "", 0, 1, "0 = solid, 1 = thin-walled and hollow (cavity modes)"),
           Param("muffle", 0.0, "", 0, 1, "damping by hands, mounting or soft contact (shorter, darker)"),
           Param("contact_point", 0.3, "", 0, 1, "0 = edge/end .. 1 = middle"),
           Param("residual", 1.0, "", 0, 3, "scale of the material's noise residual (crack and grain)"),
           Param("gesture", "hit", "", choices=("hit", "footstep")),
           Param("bounces", 1, "count", 1, 30, "contacts (drops and bounces)"),
           Param("restitution", 0.5, "", 0, 0.95, "speed kept per bounce"),
           Param("fracture", 0.0, "", 0, 1, "energy spent breaking into shards"),
           Param("striker_ring", 0.0, "", 0, 1, "level of the striker's own ring (pick, sword, spoon)"),
           Param("striker_size", 0.4, "m", 0.02, 3, "length of the striker (its ring pitch)"),
           Param("grains", 1.0, "", 0, 4, "grain count scale (ground and debris)"),
           Param("n_modes", 40, "count", 4, 120, "modes of the target"),
           Param("max_t60", 4.0, "s", 0.02, 20, "cap on ring time"),
       ],
       ["Hunt & Crossley 1975; Avanzini & Rocchesso 2001 (contact)",
        "van den Doel, Kry & Pai 2001, FoleyAutomatic (modal synthesis, SIGGRAPH)",
        "Aramaki, Besson, Kronland-Martinet & Ystad 2011, IEEE TASLP (material categories from damping laws)",
        "Klatzky, Pai & Krotkov 2000, Presence (perceived material from decay)",
        "Ren, Yeh & Lin 2013, ACM TOG (example-guided modal sound: residual compensation)",
        "Cremer & Heckl, Structure-Borne Sound (material data via vibrationdata)",
        "Leissa, Vibration of Plates / thin ring bending modes (shape formulas)",
        "Sethna et al. 2001; Bresin et al. 2010 (granular crackle, walking on surfaces); SDT: method only"])
def material_impact(ctx: Ctx, x, material, shape, size, at, velocity, striker, striker_mass_g, hardness, hollowness,
                    muffle, contact_point, residual, gesture, bounces, restitution, fracture, striker_ring,
                    striker_size, grains, n_modes, max_t60):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    m = IMPACT_MATERIALS[material]
    out = np.zeros(n)
    ground = m.get("ground", False)
    k = _contact_k(m["k"], striker, hardness)
    mass = striker_mass_g / 1000.0

    # the target's response to a unit force (shared by every contact)
    if not ground:
        f, amp, _t = mode_table(shape, m, size, hollowness, contact_point, int(n_modes), rng)
        extra = 2.5 * m["d0"] if shape == "block" else 0.0
        alpha = decay_rates(f, m, muffle, extra)
        if hollowness > 0 and shape in ("box", "shell", "ball", "plate"):
            fc = cavity_modes(size if shape not in ("shell", "ball") else size * 0.6)
            f = np.concatenate([f, fc])
            amp = np.concatenate([amp, hollowness * 1.2 / np.arange(1, len(fc) + 1)])
            alpha = np.concatenate([alpha, np.full(len(fc), 25.0 + 60 * muffle + 20 * (1 - hollowness))])
        scatter = 0.0 if m.get("metal") else 0.02
        cap = m["cap"]
        if gesture == "footstep":   # the foot's weight stays on the floor and damps it: a thud, not a ring
            alpha = alpha + 60.0
            cap = min(cap or 0.15, 0.15 if m.get("metal") else 0.05)
        if fracture > 0:            # the body breaks apart: its modes stop instead of ringing on as a ding
            cap = min(cap or 1.0, 0.012 + 0.1 * (1 - fracture))
        ir = modal_ir(f, amp, alpha, sr, max_t60, cap, rng, scatter)
        tir = np.arange(len(ir)) / sr
        ir_res = ir * np.exp(-tir / 0.004)                          # the object's spectrum, a few ms long
    if striker_ring > 0 and STRIKERS.get(striker):
        sm = IMPACT_MATERIALS[STRIKERS[striker]]
        sf, sa, _ = mode_table("bar", sm, striker_size, 0.0, 0.1, 16, rng)
        ir_s = modal_ir(sf, sa, decay_rates(sf, sm, 0.3, extra=15.0), sr, 1.5, sm["cap"], rng,
                        0.0 if sm.get("metal") else 0.02)

    def one_contact(v, t0, gain):
        pulse, vout = hc_pulse(v, mass, k, 1.5, 0.3, sr)
        if len(pulse) == 0:
            return vout
        p = pulse / (pulse.max() + 1e-12)
        i0 = int(t0 * sr)
        lvl = gain * min(1.0, v / 3.0) ** 0.8
        if material == "flesh":
            _body_thump(out, i0, lvl, size, hardness, residual, sr, rng)
        elif ground:
            g = m
            thud = dsp.lowpass(np.concatenate([p, np.zeros(int(0.08 * sr))]), g["thud_hz"] * (1.5 - muffle), sr, q=1.4)
            thud = thud / (np.abs(thud).max() + 1e-12)
            dsp.place(out, thud, i0, lvl)
            ne = g["n"] * grains * min(2.0, 0.4 + v / 2.0)
            if gesture == "footstep":
                ne *= 0.5
            gr = _grains(sr, rng, ne, g["spread"], g["grain"], g["grain_ms"], g["tonal"])
            gr = gr / (np.abs(gr).max() + 1e-12)
            dsp.place(out, gr, i0 + len(p) // 3, lvl * g["resid"] * residual * (0.8 if gesture == "footstep" else 1.0))
        else:
            y = signal.fftconvolve(p, ir)
            dsp.place(out, y / (np.abs(y).max() + 1e-12), i0, lvl * (1 - 0.6 * fracture))
            if m["resid"] * residual > 0:
                nb = rng.standard_normal(len(p) + int(0.003 * sr)) * np.concatenate([p, np.exp(-np.arange(int(0.003 * sr)) / (0.0008 * sr)) * p[-1]])
                r = signal.fftconvolve(nb, ir_res)
                dsp.place(out, r / (np.abs(r).max() + 1e-12), i0, lvl * m["resid"] * residual)
        if striker_ring > 0 and STRIKERS.get(striker):
            ys = signal.fftconvolve(p, ir_s)
            dsp.place(out, ys / (np.abs(ys).max() + 1e-12), i0, lvl * striker_ring)
        return vout

    if gesture == "footstep":
        heel_t = at
        one_contact(velocity, heel_t, 1.0)
        one_contact(velocity * 0.4, heel_t + rng.uniform(0.07, 0.12), 0.25)       # ball of the foot, rolled onto
        L = int(0.12 * sr)
        sc = dsp.bandpass(rng.standard_normal(L), 2500 if not ground else 4000, sr, q=0.7)
        sc *= np.sin(np.linspace(0, math.pi, L)) ** 2
        dsp.place(out, sc / (np.abs(sc).max() + 1e-12), int((heel_t + 0.02) * sr), 0.03 * (1 - muffle) * residual)
        out = dsp.highpass(dsp.lowpass(out, 10000.0 if ground else 6000.0, sr), 50.0, sr)   # sole: no hard click
    else:
        t, v = float(at), float(velocity)
        for b in range(int(bounces)):
            if v < 0.03 or t >= ctx.duration:
                break
            vout = one_contact(v, t, 1.0)
            v = vout * restitution * (1 - 0.2 * rng.random())
            t += 2 * v / G + 0.002

    if fracture > 0 and not ground:
        i0 = int(at * sr)
        # the crack: a broadband snap of 1-2 ms (no pitch), and the floor's short low thud under it
        Ls = int(0.004 * sr)
        snap = rng.standard_normal(Ls) * np.exp(-np.arange(Ls) / (0.0007 * sr))
        dsp.place(out, dsp.highpass(snap, 800.0, sr), i0, 1.2 * fracture)
        Lt = int(0.12 * sr)
        thud = dsp.lowpass(dsp.lowpass(rng.standard_normal(Lt) * np.exp(-np.arange(Lt) / (0.018 * sr)), 160.0, sr), 160.0, sr)
        dsp.place(out, thud / (np.abs(thud).max() + 1e-12), i0, 0.7 * fracture)
        sh = _shards(sr, rng, int(4 + 26 * fracture * grains), m, ctx.duration - at, material)
        dsp.place(out, sh / (np.abs(sh).max() + 1e-12), i0 + int(0.006 * sr), 0.6 * fracture)
    if muffle > 0:
        out = dsp.lowpass(out, 16000 * (1 - muffle) + 1500 * muffle, sr)
    return dsp.highpass(out, 30.0, sr)          # a compact body radiates no DC (the bare force pulse has some)
