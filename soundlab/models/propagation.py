"""Stage 4: propagation (distance, air, motion, ground)."""
from __future__ import annotations


import numpy as np

from .. import dsp
from ..core import Ctx, Param, ctrl, model

# ISO 9613-2 Table 2 atmospheric attenuation (dB/km) at octave centres 63..8000 Hz
ISO_9613_2 = {
    "10C_70RH": [0.1, 0.4, 1.0, 1.9, 3.7, 9.7, 32.8, 117.0],
    "20C_70RH": [0.1, 0.3, 1.1, 2.8, 5.0, 9.0, 22.9, 76.6],
    "30C_70RH": [0.1, 0.3, 1.0, 3.1, 7.4, 12.7, 23.1, 59.3],
    "15C_20RH": [0.3, 0.6, 1.2, 2.7, 8.2, 28.2, 88.8, 202.0],
    "15C_50RH": [0.1, 0.5, 1.2, 2.2, 4.2, 10.8, 36.2, 129.0],
    "15C_80RH": [0.1, 0.3, 1.1, 2.4, 4.1, 8.3, 23.7, 82.8],
}
OCT = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], float)


def air_alpha_db_per_m(f: np.ndarray, atmosphere: str) -> np.ndarray:
    """Interpolate ISO 9613-2 Table 2 in log-log; extrapolate above 8 kHz with f^2."""
    a = np.array(ISO_9613_2[atmosphere]) / 1000.0
    lf = np.log(np.clip(f, 20, None))
    out = np.exp(np.interp(lf, np.log(OCT), np.log(a)))
    hi = f > 8000
    out[hi] = a[-1] * (f[hi] / 8000) ** 2
    return out


def air_absorb(x: np.ndarray, sr: int, distance: float, atmosphere: str = "20C_70RH") -> np.ndarray:
    return dsp.fft_filter(x, sr, lambda f: 10 ** (-air_alpha_db_per_m(f, atmosphere) * distance / 20))


@model("distance", "propagation",
       "Static distance: spherical spreading (1/r re 1 m), ISO 9613-2 air absorption, optional travel delay.",
       [
           Param("distance", 10.0, "m", 0.1, 20000, "source-listener distance"),
           Param("atmosphere", "20C_70RH", "", choices=tuple(ISO_9613_2)),
           Param("delay", False, "", None, None, "include the travel delay (2.9 ms per m)"),
           Param("spreading", True, "", None, None, "apply 1/r level loss"),
       ], ["ISO 9613-2:1996 Table 2", "J. O. Smith, PASP: Air absorption", "Wikipedia: Speed of sound"])
def distance(ctx: Ctx, x, distance, atmosphere, delay, spreading):
    y = air_absorb(x, ctx.sr, distance, atmosphere)
    if spreading:
        y = y / max(distance, 1.0)
    if delay:
        d = int(distance / ctx.c * ctx.sr)
        y = np.concatenate([np.zeros(d), y])[: len(x)]
    return y


@model("passby", "propagation",
       "Moving source on a straight line past the listener: time-varying delay gives the Doppler shift; "
       "level follows 1/r; air absorption at the closest distance.",
       [
           Param("speed", 20.0, "m/s", 0, 300, "source speed"),
           Param("closest", 10.0, "m", 0.5, 2000, "distance at closest approach"),
           Param("t_closest", 2.0, "s", 0, None, "time of closest approach"),
           Param("atmosphere", "20C_70RH", "", choices=tuple(ISO_9613_2)),
       ], ["J. O. Smith, PASP (Doppler, variable delay lines)"])
def passby(ctx: Ctx, x, speed, closest, t_closest, atmosphere):
    t = ctx.t
    c = ctx.c
    v = min(float(speed), 0.999 * c)        # subsonic only (a supersonic source has no single emission time)
    d = float(closest)
    # emission time te: the listener hears at t the sound emitted at te with c (t - te) = r(te). With
    # T = t - t_closest and u = te - t_closest this is (c^2 - v^2) u^2 - 2 c^2 T u + c^2 T^2 - d^2 = 0; the root
    # with te <= t is exact for any subsonic speed. (Until v0.21.1: 6 fixed-point steps, which converge at the
    # rate v/c and were audibly off for fast flybys: ~15 % of the travel time left at 250 m/s.)
    T = t - t_closest
    u = (c * c * T - np.sqrt(c * c * v * v * T * T + (c * c - v * v) * d * d)) / (c * c - v * v)
    te = t_closest + u
    r = np.sqrt((v * u) ** 2 + d * d)
    y = np.interp(te * ctx.sr, np.arange(len(x)), x, left=0.0, right=0.0)
    y = y * (closest / r)
    return air_absorb(y, ctx.sr, closest, atmosphere)


@model("ground", "propagation",
       "Ground reflection: a delayed copy from the mirror-image source interferes with the direct sound "
       "(comb filtering; moving heights give a sweep).",
       [
           Param("source_height", 1.0, "m", 0, 1000, "source height", automatable=True),
           Param("listener_height", 1.7, "m", 0, 1000, "listener height"),
           Param("distance", 20.0, "m", 0.5, 20000, "horizontal distance"),
           Param("reflection", 0.7, "", -1, 1, "ground reflection coefficient (hard ~0.9, grass ~0.5)"),
           Param("ground_hz", 2000.0, "Hz", 100, 20000, "above this the ground reflects less (porous/grassy "
                 "ground has a finite impedance; asphalt ~8000, grass ~800; estimate after Delany-Bazley)"),
           Param("turbulence", 0.5, "", 0, 1, "air turbulence: random wobble of the reflected path that smears "
                 "the comb notches at high frequencies (a perfectly static comb sounds phasey and synthetic)"),
       ], ["Farnell 2010 practical 31 (ground comb)", "ISO 9613-2 (ground effect)",
           "Delany & Bazley 1970 (porous ground impedance)", "Salomons 2001, Computational Atmospheric Acoustics "
           "(coherence loss from turbulence)"])
def ground(ctx: Ctx, x, source_height, listener_height, distance, reflection, ground_hz, turbulence):
    hs = ctrl(ctx, source_height)
    d_direct = np.sqrt(distance ** 2 + (hs - listener_height) ** 2)
    d_refl = np.sqrt(distance ** 2 + (hs + listener_height) ** 2)
    extra = (d_refl - d_direct) / ctx.c * ctx.sr
    if turbulence > 0:
        # path-length wobble: ~0.1-0.3 ms at turbulence 1, slow (a few Hz); grows with distance (estimate)
        wob = dsp.lowpass(ctx.rng.standard_normal(ctx.n), 3.0, ctx.sr)
        wob /= np.std(wob) + 1e-12
        sig = turbulence * (0.1e-3 + 0.2e-3 * min(distance / 100.0, 1.0)) * ctx.sr
        extra = extra + sig * wob
    refl = dsp.variable_delay(x, np.maximum(extra, 0.0)) * reflection * (d_direct / d_refl)
    refl = dsp.lowpass(refl, ground_hz, ctx.sr, 0.5)
    return x + refl
