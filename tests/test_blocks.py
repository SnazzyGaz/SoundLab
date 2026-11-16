"""Physics checks for the building blocks against the formulas in the knowledge base."""
import math

import numpy as np
import pytest

import soundlab
from soundlab import materials
from soundlab.analysis import f0_estimate, track_partials
from soundlab.core import REGISTRY
from soundlab.models.excitation import bubble_grain
from soundlab.models.resonator import bar_modes
from conftest import SR, peak_freq, run


def test_every_model_renders_with_defaults(impulse):
    for name, m in REGISTRY.items():
        x = None
        if m.needs_input:
            x = np.zeros(SR)
            x[100] = 1.0
            x[1000:3000] += np.random.default_rng(0).standard_normal(2000) * 0.1
        y = run(name, 1.0, x)
        assert y.shape == (SR,), name
        assert np.all(np.isfinite(y)), name
        assert np.max(np.abs(y)) > 0, f"{name} is silent"


def test_describe_has_units_and_docs():
    for name in REGISTRY:
        d = soundlab.describe(name)
        assert d["doc"] and d["params"], name


def test_bar_mode_ratios_match_euphonics():
    r = bar_modes(5)
    for got, ref in zip(r, [1, 2.76, 5.40, 8.93, 13.34]):
        assert abs(got - ref) / ref < 0.01


def test_modal_bar_partials(impulse):
    y = run("modal", 1.0, impulse, material="steel", shape="bar", f0=400.0, n_modes=4, detune=0.0, strike_pos=0.22)
    tracks, _ = track_partials(y, SR)
    freqs = sorted(t["freq_hz"] for t in tracks[:6])
    for target in [400, 400 * 2.756, 400 * 5.404]:
        assert min(abs(f - target) for f in freqs) / target < 0.02


def test_modal_bar_fundamental_from_geometry():
    # f1 = 4.730^2 / (2 pi L^2) * h * sqrt(E / (12 rho)); steel 0.3 m x 6 mm -> ~ 349 Hz
    m = materials.get("steel")
    f1 = 4.730 ** 2 / (2 * math.pi * 0.3 ** 2) * 0.006 * math.sqrt(m["E"] / (12 * m["rho"]))
    x = np.zeros(SR)
    x[0] = 1
    y = run("modal", 1.0, x, material="steel", shape="bar", length=0.3, thickness=0.006, n_modes=1, detune=0)
    assert abs(peak_freq(y) - f1) < 3


@pytest.mark.parametrize("mat,f", [("glass", 1000.0), ("oak", 500.0)])
def test_modal_decay_follows_loss_factor(mat, f):
    x = np.zeros(2 * SR)
    x[0] = 1
    y = run("modal", 2.0, x, material=mat, shape="custom", ratios=[1.0], f0=f, n_modes=1, extra_damping=0,
            detune=0, max_t60=60)
    expected = materials.t60_at(materials.get(mat)["eta_default"], f)
    env = np.abs(y)
    # fit log-envelope slope over peaks
    seg = int(SR / f)
    pk = np.array([env[i:i + seg].max() for i in range(seg, len(env) - seg, seg)])
    t = np.arange(len(pk)) * seg / SR
    good = pk > pk.max() * 1e-3
    slope = np.polyfit(t[good], 20 * np.log10(pk[good]), 1)[0]
    assert abs(-60 / slope - expected) / expected < 0.1


def test_klatzky_material_order(impulse):
    t60 = {}
    for cat in ["rubber", "wood", "glass", "steel"]:
        t60[cat] = materials.t60_at(materials.eta_from_klatzky(cat), 1000)
    assert t60["rubber"] < t60["wood"] < t60["glass"] < t60["steel"]
    assert abs(t60["steel"] - 3.09) < 0.1 and abs(t60["glass"] - 0.87) < 0.05


def test_minnaert_bubble_frequency():
    g = bubble_grain(SR, 1e-3, xi=0.0)
    assert abs(peak_freq(g, lo=500) - 3260) / 3260 < 0.03


def test_tube_closed_open_resonance(impulse):
    y = run("tube", 1.0, impulse, length=0.5, ends="closed_open", loss=0.95, damp_hz=8000)
    f = peak_freq(y, lo=50, hi=400)
    assert abs(f - 343 / (4 * 0.5)) < 8  # 171.5 Hz


def test_tube_open_open_resonance(impulse):
    y = run("tube", 1.0, impulse, length=0.5, ends="open_open", loss=0.95, damp_hz=8000)
    f = peak_freq(y, lo=200, hi=500)
    assert abs(f - 343 / (2 * 0.5)) < 10  # 343 Hz


def test_string_pitch(impulse):
    y = run("string", 1.0, impulse, f0=220.0, t60=2.0)
    f0, _ = f0_estimate(y, SR)
    assert abs(f0 - 220) < 3


def test_aeolian_tone():
    y = run("turbulence", 2.0, flow_speed=20.0, obstacle_d=0.004, tone=1.0, tone_q=30)
    assert abs(peak_freq(y, lo=200, hi=5000) - 1000) < 60  # f = 0.2 * 20 / 0.004


def test_valve_pulse_pitch_and_rasp_subharmonic():
    y = run("valve_pulse", 1.0, f0=300.0, jitter=0.0, shimmer=0.0)
    assert abs(f0_estimate(y, SR)[0] - 300) < 2
    yr = run("valve_pulse", 1.0, f0=300.0, jitter=0.0, shimmer=0.0, rasp=1.0, aspiration=0.0)
    Y = np.abs(np.fft.rfft(yr))
    f = np.fft.rfftfreq(len(yr), 1 / SR)
    sub = Y[(f > 145) & (f < 155)].max()
    assert sub > 0.05 * Y[(f > 295) & (f < 305)].max()  # period doubling puts energy at f0/2


def test_friction_stick_slip_oscillates():
    y = run("friction", 0.5, velocity=0.3, normal_force=3.0, roughness=0.0, f_osc=800)
    seg = y[SR // 10:]
    assert np.std(seg) > 0
    f = peak_freq(seg, lo=50, hi=5000)
    assert 50 < f < 2000  # relaxation oscillation below/around the contact resonance


def test_friction_no_motion_is_silent():
    y = run("friction", 0.3, velocity=0.0, normal_force=3.0, roughness=0.0)
    assert np.max(np.abs(y[SR // 20:])) < 1e-6


def test_chopper_pitch():
    y = run("chopper", 1.0, freq=250.0, jitter=0.0, flow_noise=0.0)
    assert abs(f0_estimate(y, SR)[0] - 250) < 2


def test_impact_bounces_get_closer():
    y = run("impact", 2.0, bounces=5, restitution=0.6, first_gap=0.3, contact_ms=0.5)
    on = np.nonzero(np.diff((y > 1e-6).astype(int)) == 1)[0]
    gaps = np.diff(on)
    assert len(on) == 5 and all(gaps[i + 1] < gaps[i] for i in range(len(gaps) - 1))


def test_air_absorption_iso_table():
    from soundlab.models.propagation import air_alpha_db_per_m
    a = air_alpha_db_per_m(np.array([1000.0, 8000.0]), "20C_70RH")
    assert abs(a[0] * 1000 - 5.0) < 1e-6 and abs(a[1] * 1000 - 76.6) < 1e-6


def test_distance_darkens_sound():
    x = np.random.default_rng(0).standard_normal(SR)
    y = run("distance", 1.0, x, distance=1000.0, spreading=False)
    def band(z, lo, hi):
        Z = np.abs(np.fft.rfft(z)) ** 2
        f = np.fft.rfftfreq(len(z), 1 / SR)
        return Z[(f > lo) & (f < hi)].sum()
    drop_hi = 10 * np.log10(band(y, 7000, 9000) / band(x, 7000, 9000))
    drop_lo = 10 * np.log10(band(y, 400, 600) / band(x, 400, 600))
    assert drop_lo > -5 and drop_hi < -50


def test_passby_doppler():
    t = np.arange(4 * SR) / SR
    x = np.sin(2 * np.pi * 1000 * t)
    y = run("passby", 4.0, x, speed=30.0, closest=5.0, t_closest=2.0)
    f_app = peak_freq(y[int(0.3 * SR):int(1.0 * SR)], lo=800, hi=1300)
    f_rec = peak_freq(y[int(3.0 * SR):int(3.7 * SR)], lo=700, hi=1200)
    assert abs(f_app - 1000 * 343 / (343 - 30)) < 15  # ~1096 Hz approaching
    assert abs(f_rec - 1000 * 343 / (343 + 30)) < 15  # ~920 Hz receding


def test_room_sabine_tail(impulse):
    y = run("room", 2.0, np.pad(impulse, (0, SR)), volume=500, absorption=0.2)
    assert np.sum(y[SR // 2:] ** 2) > 0  # has a tail
    from soundlab.analysis import schroeder_t60
    t = schroeder_t60(y, SR)
    assert t is not None and 0.3 < t < 4.0


def test_dispersive_string_chirps_down():
    x = np.zeros(SR)
    x[0] = 1.0
    y = run("string", 1.0, x, f0=12.0, t60=1.0, brightness=16000, dispersion=0.7, dispersion_stages=24)
    first = y[int(0.05 * SR):int(0.16 * SR)]  # first round trip
    hi = np.abs(dsp_band(first, 4000, 8000))
    lo = np.abs(dsp_band(first, 200, 400))
    # energy-weighted arrival time: high band arrives earlier than low band (falling 'pew')
    t = np.arange(len(first))
    assert (hi ** 2 @ t) / (hi ** 2).sum() < (lo ** 2 @ t) / (lo ** 2).sum()


def dsp_band(x, lo, hi):
    from soundlab import dsp
    return dsp.bandpass(x, np.sqrt(lo * hi), SR, np.sqrt(lo * hi) / (hi - lo))


def test_rough_pulses_rate_and_jitter():
    from soundlab.analysis import f0_estimate
    clean = run("rough_pulses", 1.0, f0=200, jitter=0.0, period_doubling=0.0, shimmer=0.0, aspiration=0.0)
    rough = run("rough_pulses", 1.0, f0=200, jitter=0.2, period_doubling=0.6, shimmer=0.5, aspiration=0.3)
    # clean train: about 200 pulses (closure spikes are the negative peaks)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(-clean, height=0.5)
    assert 190 <= len(pk) <= 210
    ic = f0_estimate(clean, SR, return_info=True)
    ir = f0_estimate(rough, SR, return_info=True)
    assert abs(ic["f0_hz"] - 200) < 3
    assert ir["periodicity"] < ic["periodicity"] - 0.2
