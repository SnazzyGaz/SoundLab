"""Material table. Density, Young's modulus and loss factor from Cremer & Heckl,
'Structure-Borne Sound', as reproduced in the vibrationdata damping tutorial
(https://www.vibrationdata.com/tutorials_alt/damping.pdf). Where a range is given we
store [low, high] and use the geometric mean as the default loss factor.

Perceptual decay (Klatzky, Pai & Krotkov 2000, Presence 9(4)): critical log10(tau_d)
rubber 0.46, wood 0.50, glass 2.10, steel 2.65, with tan(phi) = 1/(pi * tau_d).
"""
from __future__ import annotations

import math

# name: (density kg/m3, E Pa, nu, eta_lo, eta_hi, note)
MATERIALS: dict[str, dict] = {
    "steel":     dict(rho=7800, E=210e9, nu=0.31, eta=(0.2e-4, 3e-4)),
    "aluminium": dict(rho=2700, E=72e9, nu=0.34, eta=(0.3e-4, 1e-4)),
    "brass":     dict(rho=8500, E=95e9, nu=0.35, eta=(0.2e-3, 1e-3)),
    "copper":    dict(rho=8900, E=125e9, nu=0.34, eta=(2e-3, 2e-3)),
    "lead":      dict(rho=11300, E=17e9, nu=0.44, eta=(5e-2, 30e-2)),
    "glass":     dict(rho=2500, E=60e9, nu=0.24, eta=(0.6e-3, 2e-3)),
    "concrete":  dict(rho=2300, E=26e9, nu=0.2, eta=(4e-3, 8e-3), note="E not in table; typical value"),
    "brick":     dict(rho=2000, E=16e9, nu=0.2, eta=(1e-2, 2e-2), note="E not in table; typical value"),
    "oak":       dict(rho=850, E=6e9, nu=0.3, eta=(1e-2, 1e-2), note="E range 2-10 GPa"),
    "fir":       dict(rho=550, E=8e9, nu=0.3, eta=(8e-3, 8e-3), note="E not in table; typical value"),
    "plywood":   dict(rho=600, E=6e9, nu=0.3, eta=(1.3e-2, 1.3e-2), note="E not in table; typical value"),
    "plexiglas": dict(rho=1150, E=5.6e9, nu=0.35, eta=(2e-2, 4e-2)),
    "cardboard": dict(rho=700, E=3e9, nu=0.3, eta=(5e-2, 5e-2), note="not in table; estimate for match boxes"),
    "rubber":    dict(rho=1100, E=0.05e9, nu=0.49, eta=(0.1, 0.1), note="eta = 2 x damping ratio ~0.05"),
    "cork":      dict(rho=200, E=0.02e9, nu=0.0, eta=(0.13, 0.17)),
    "sand":      dict(rho=1600, E=0.1e9, nu=0.3, eta=(0.06, 0.12), note="dry sand"),
}

KLATZKY_LOG_TAU_D = {"rubber": 0.46, "wood": 0.50, "glass": 2.10, "steel": 2.65}


def get(name: str) -> dict:
    try:
        m = dict(MATERIALS[name])
    except KeyError:
        raise KeyError(f"unknown material {name!r}; known: {sorted(MATERIALS)}") from None
    lo, hi = m["eta"]
    m["eta_default"] = math.sqrt(lo * hi)
    m["name"] = name
    return m


def t60_at(eta: float, f: float) -> float:
    """Material-only T60 (s) for a mode at f Hz with loss factor eta: 6.91/(pi f eta)."""
    return 6.91 / (math.pi * f * eta)


def eta_from_klatzky(category: str) -> float:
    tau_d = 10 ** KLATZKY_LOG_TAU_D[category]
    return 1.0 / (math.pi * tau_d)


def table() -> list[dict]:
    rows = []
    for k in MATERIALS:
        m = get(k)
        rows.append({
            "material": k, "density_kg_m3": m["rho"], "E_GPa": m["E"] / 1e9,
            "eta": list(m["eta"]), "t60_at_1kHz_s": round(t60_at(m["eta_default"], 1000.0), 3),
            "note": m.get("note", ""),
        })
    return rows
