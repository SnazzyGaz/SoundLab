"""creak (v0.28): door creaks, hinge squeaks, shoe squeaks and floorboard creaks in two regimes, rebuilt from CC0
recordings (fitted numbers in creak_data.py).

What the recordings show, and what the model does:
- Groan: a train of short slip clicks (not a voice-like pulse) heard through the door's or floor's fixed
  resonances, which stay put while the click rate moves. Big slips every ~0.1-0.2 s jump the level by several dB
  and decay. (Farnell 2010 "creaking door": stick-slip events through a wooden-door formant bank.)
- Squeal: the stick-slip locks onto one resonance and sings a near-pure tone with a few harmonics, gliding with
  the motion; between bursts it can hop to another resonance. No noise bed.
- Both arrive in bursts (tens to hundreds of ms) separated by sticks. A slow motion groans; above `squeal_above`
  the bursts squeal.
- Shoe squeak: a contact tick and a short scuff, then a ~2.4 kHz tone with a weak inharmonic partial that drifts
  down and decays.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .. import dsp
from ..core import Ctx, Param, model
from .creak_data import BODIES, BODY_TILT, GROAN, VOICES


def _smooth(rng, n, sr, rate):
    k = max(4, int(n / sr * rate) + 4)
    return np.interp(np.linspace(0, k - 3, n), np.arange(k), rng.standard_normal(k))


def _body_filter(x, body, sr):
    out = np.zeros_like(x)
    for f, bw, g in BODIES[body]:
        if f < 0.45 * sr:
            out += 10 ** (g / 20) * dsp.biquad(x, "bandpass", f, max(0.3, f / bw), sr)
    out = out + 0.01 * x
    lp = BODY_TILT.get(body)
    return dsp.lowpass(dsp.lowpass(out, lp, sr), lp, sr) if lp and lp < 0.45 * sr else out


def _body_gain(fr, body):
    """Magnitude of the body's resonances at frequencies fr (for squeal harmonics), peak-normalised."""
    g = np.zeros_like(fr)
    for f, bw, gd in BODIES[body]:
        g += 10 ** (gd / 20) / np.sqrt(1 + ((fr - f) / (bw / 2)) ** 2)
    return g / max(1e-9, g.max())


def _env(n, sr, att, rel):
    e = np.ones(n)
    a, r = min(n // 2, int(att * sr)), min(n // 2, int(rel * sr))
    if a > 0:
        e[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    if r > 0:
        e[-r:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r))
    return e


@model("creak", "sound",
       "Door creaks, hinge squeaks, shoe squeaks and floorboard creaks rebuilt from CC0 recordings. Bursts of slip "
       "clicks through fixed body resonances (groan) or a near-pure squeal locked to a resonance that glides with "
       "the motion and hops between bursts. Speed sets the regime (slow groans, fast squeals), the click rate and "
       "the squeal glide.",
       [
           Param("at", 0.05, "s", 0, None),
           Param("length_s", 1.5, "s", 0.05, 30, "length of the motion"),
           Param("speed", 0.5, "", 0.02, 2.0, "how fast it moves (slow groans .. fast squeals)"),
           Param("pitch", 1.0, "", 0.25, 4.0, "scales the squeal lock pitches and the click rate"),
           Param("speed_change", 0.0, "", -1, 1, "-1 slowing down .. +1 speeding up over the motion"),
           Param("irregularity", 0.4, "", 0, 1, "wobble of the motion"),
           Param("regime", "auto", "", choices=("auto", "groan", "squeal")),
           Param("squeal_above", 0.6, "", 0, 1.5, "auto: speed above which bursts squeal"),
           Param("body", "door_heavy", "", choices=tuple(BODIES)),
           Param("voice", "door", "", choices=tuple(VOICES)),
           Param("burst_ms", 250.0, "ms", 10, 3000, "median burst length"),
           Param("gap_ms", 60.0, "ms", 0, 2000, "median gap between bursts"),
           Param("burst_spread", 0.45, "", 0, 1.5, "log-normal spread of burst lengths"),
           Param("squeal_burst_ms", 0.0, "ms", 0, 5000, "median squeal burst length (0 = burst_ms)"),
           Param("squeal_gap_ms", -1.0, "ms", -1, 2000, "median gap after a squeal burst (-1 = gap_ms)"),
           Param("pulse_hz", GROAN["pulse_hz"], "Hz", 2, 600, "groan click rate at speed 0.5"),
           Param("pulse_jitter", GROAN["pulse_jitter"], "", 0, 0.8, "groan click timing irregularity"),
           Param("slip_interval_s", GROAN["slip_interval_s"], "s", 0.02, 2, "groan: time between big slips"),
           Param("slip_jump_db", GROAN["slip_jump_db"], "dB", 0, 20, "groan: level jump of a big slip"),
           Param("slip_decay_db_s", GROAN["slip_decay_db_s"], "dB/s", 5, 1000, "groan: decay after a big slip"),
           Param("squeal_hz", 0.0, "Hz", 0, 8000, "squeal lock pitch (0 = the voice's pitches)"),
           Param("hop", 0.4, "", 0, 1, "chance a new burst hops to another lock pitch"),
           Param("glide_follow", 1.0, "", 0, 3, "squeal pitch ~ speed^follow within a burst"),
           Param("rise_cents", -1.0, "cents", -1, 1200, "onset hook below the lock pitch (-1 = voice default)"),
           Param("drift_cents", 0.0, "cents", -1200, 1200, "pitch drift over each squeal burst"),
           Param("tone_decay_db_s", 0.0, "dB/s", 0, 1000, "squeal decays within a burst (0 = sustains)"),
           Param("squeal_body", 0.0, "", 0, 1, "how much the body resonances colour the squeal harmonics"),
           Param("squeal_db", 0.0, "dB", -40, 20, "squeal level re groan"),
           Param("tick", 0.0, "", 0, 2, "contact tick at each squeal onset (shoes)"),
           Param("scuff", 0.0, "", 0, 2, "short scuff noise after the tick (shoes)"),
       ],
       ["Farnell 2010, Designing Sound, practical 'Creaking door' (stick-slip events through a door formant bank)",
        "Heinrichs & McPherson 2014 (performable creaking-door model)",
        "Avanzini, Serafin & Rocchesso 2005 (friction locking to a resonance)",
        "Freesound CC0 recordings 455324, 766155, 397687, 88502, 124289, 506665, 788701 (fitted numbers only)"])
def creak(ctx: Ctx, x, at, length_s, speed, pitch, speed_change, irregularity, regime, squeal_above, body, voice, burst_ms,
          gap_ms, burst_spread, squeal_burst_ms, squeal_gap_ms, pulse_hz, pulse_jitter, slip_interval_s, slip_jump_db, slip_decay_db_s, squeal_hz, hop,
          glide_follow, rise_cents, drift_cents, tone_decay_db_s, squeal_body, squeal_db, tick, scuff):
    sr, n, rng = ctx.sr, ctx.n, ctx.rng
    t = np.arange(n) / sr
    u = np.clip((t - at) / max(length_s, 1e-3), 0, 1)
    v = speed * np.exp(speed_change * math.log(4.0) * (u - 0.5))
    if irregularity > 0:
        v = v * np.clip(1 + 0.35 * irregularity * _smooth(rng, n, sr, 3.0), 0.3, None)
    V = dict(VOICES[voice])
    pitches = [p * pitch for p in ([squeal_hz] if squeal_hz > 0 else V["pitches"])]
    rise = V["rise"] if rise_cents < 0 else rise_cents
    out = np.zeros(n)
    end = min(n, int((at + length_s) * sr))
    i = int(at * sr)
    lock = pitches[0]
    first = True
    sq_out = np.zeros(n)
    gr_out = np.zeros(n)
    sq_ms = squeal_burst_ms or burst_ms
    sq_gap = gap_ms if squeal_gap_ms < 0 else squeal_gap_ms
    while i < end:
        vb = float(np.mean(v[i:min(end, i + int(0.05 * sr))]))
        if regime == "auto":
            squeal = rng.random() < np.clip((vb - squeal_above) / 0.15 + 0.5, 0, 1)
        else:
            squeal = regime == "squeal"
        nom = (sq_ms if squeal else burst_ms) * 1e-3 * sr
        if end - i < max(int(0.012 * sr), int(0.35 * nom)) and not first:
            break                                       # no clipped stub of a burst at the end
        d = int(nom * math.exp(burst_spread * rng.standard_normal()))
        d = max(int(0.012 * sr), min(d, end - i))
        seg = slice(i, i + d)
        tt = np.arange(d) / sr
        d0 = d
        if squeal and tone_decay_db_s > 0:              # a decaying squeal rings out past its burst (shoe)
            d = int(min(n - i, max(d, min(0.8, 55.0 / tone_decay_db_s) * sr)))
            seg = slice(i, i + d)
            tt = np.arange(d) / sr
        if squeal:
            if not first and len(pitches) > 1 and rng.random() < hop:
                lock = pitches[int(rng.integers(len(pitches)))]
            vv = np.maximum(v[seg], 1e-6) / max(v[i], 1e-6)
            cents = -rise * np.exp(-tt / 0.04) + drift_cents * tt / max(tt[-1], 1e-3) \
                + V["wobble"] * _smooth(rng, d, sr, 6.0)
            f0 = lock * vv ** glide_follow * 2 ** (cents / 1200)
            ph = 2 * np.pi * np.cumsum(f0) / sr
            y = np.zeros(d)
            parts = [(k + 1, 10 ** (db / 20)) for k, db in enumerate(V["harm"])]
            if V["extra"]:
                parts.append((V["extra"][0], 10 ** (V["extra"][1] / 20)))
            for ratio, a in parts:
                xdec = V["extra"][2] if (V["extra"] and ratio == V["extra"][0]) else 0.0
                fk = ratio * f0
                if a < 1e-3 or fk.min() > 0.45 * sr:
                    continue
                g = a * (fk < 0.45 * sr)
                if squeal_body > 0:
                    g = g * ((1 - squeal_body) + squeal_body * _body_gain(fk, body))
                if xdec:
                    g = g * 10 ** (-xdec * tt / 20)
                y += g * np.sin(ratio * ph + rng.uniform(0, 2 * np.pi))
            e = _env(d, sr, 0.006, 0.015) * np.sqrt(np.clip(vv, 0, 2))
            if tone_decay_db_s > 0:
                e *= 10 ** (-tone_decay_db_s * tt / 20)
            y *= e
            if tick > 0 or scuff > 0:
                L = int(0.004 * sr)
                tk = rng.standard_normal(L) * np.exp(-np.arange(L) / (0.0008 * sr))
                y[:L] += tick * 1.2 * dsp.highpass(tk, 400.0, sr)
                Ls = d
                sc = dsp.lowpass(dsp.highpass(rng.standard_normal(Ls), 1500.0, sr), 9000.0, sr)
                y[:Ls] += scuff * 0.12 * sc * (0.3 * np.exp(-np.arange(Ls) / (0.01 * sr)) + np.exp(-np.arange(Ls) / (0.09 * sr)))
            sq_out[seg] += y
        else:
            # slip clicks: a sample impulse with a 0.1 ms tail (not a glottal ramp), rate following the speed
            clicks = np.zeros(d)
            nxt_slip = rng.uniform(0, slip_interval_s)
            slip_t = -1.0
            jump = 10 ** (slip_jump_db / 20) - 1
            k = 0.0
            while True:
                rate = pitch * pulse_hz * (v[min(n - 1, i + int(k))] / 0.5) ** 0.8
                k += sr / max(rate, 1.0) * max(0.25, 1 + pulse_jitter * rng.standard_normal())
                if k >= d:
                    break
                tk = k / sr
                if tk >= nxt_slip:
                    slip_t = tk
                    nxt_slip = tk + slip_interval_s * math.exp(0.3 * rng.standard_normal())
                g = 1.0 + (jump * math.exp(-(tk - slip_t) * slip_decay_db_s / 8.686) if slip_t >= 0 else 0.0)
                clicks[int(k)] += g * (0.6 + 0.4 * rng.random()) * (1 if rng.random() < 0.5 else -1)
            tail = np.exp(-np.arange(int(0.0006 * sr)) / (0.0001 * sr))
            y = _body_filter(np.convolve(clicks, tail)[:d], body, sr)
            y *= _env(d, sr, 0.003, 0.01) * np.clip(v[seg] / 0.5, 0, 2) ** 0.7   # louder as it moves faster
            gr_out[seg] += y
        first = False
        i += d0 + int((sq_gap if squeal else gap_ms) * 1e-3 * sr * math.exp(0.6 * rng.standard_normal()))
    gpk, spk = np.abs(gr_out).max(), np.abs(sq_out).max()
    if gpk > 0:
        out += 0.5 * gr_out / gpk * (np.clip(np.max(v) / 0.5, 0.2, 1.5) if spk > 0 else 1.0)
    if spk > 0:
        out += 0.6 * 10 ** (squeal_db / 20) * sq_out / spk
    return dsp.highpass(out, 40.0, sr)
