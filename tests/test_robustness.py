"""v0.21.2 regressions: physics tables, waveguide tuning, exact Doppler, short renders, late events, negative
layer starts, 24-bit WAV, GUI recipe ids, stiff friction."""
import json
import math
import wave

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import jn_zeros

import soundlab
from soundlab import dsp
from soundlab.core import REGISTRY, RecipeError
from soundlab.models.resonator import MEMBRANE_J, bar_modes

from conftest import SR, run


# ----------------------------------------------------------------------------- physics tables
def test_bar_modes_match_free_free_beam_roots():
    roots = [brentq(lambda x: math.cos(x) * math.cosh(x) - 1, (k + 0.5) * math.pi - 0.5, (k + 0.5) * math.pi + 0.5)
             for k in range(1, 13)]
    exact = (np.array(roots) / roots[0]) ** 2
    ours = np.array(bar_modes(12))
    assert np.all(np.diff(ours) > 0), "no repeated modes"
    assert np.allclose(ours, exact, rtol=2e-4)


def test_membrane_modes_are_the_ordered_bessel_zeros():
    allz = sorted(z for m in range(12) for z in jn_zeros(m, 5))
    assert np.allclose(MEMBRANE_J, allz[: len(MEMBRANE_J)], atol=2e-3)


# ----------------------------------------------------------------------------- waveguide tuning
def _peak_hz(y, f0, sr=SR):
    nfft = 1 << 20
    Y = np.abs(np.fft.rfft(y, nfft))
    f = np.fft.rfftfreq(nfft, 1 / sr)
    band = (f > 0.7 * f0) & (f < 1.3 * f0)
    return f[band][np.argmax(Y[band])]


@pytest.mark.parametrize("f0", [110.0, 440.0, 1000.0, 2000.0, 3000.0])
def test_string_is_in_tune(f0, impulse):
    y = run("string", x=impulse, f0=f0, t60=2.0)
    assert abs(1200 * math.log2(_peak_hz(y, f0) / f0)) < 3.0          # was -38 c at 1 kHz, -100 c at 3 kHz


@pytest.mark.parametrize("ends,f0", [("open_open", 500.0), ("open_open", 1500.0), ("closed_open", 400.0)])
def test_tube_is_in_tune(ends, f0, impulse):
    L = 343.0 / (2 * f0) if ends == "open_open" else 343.0 / (4 * f0)
    y = run("tube", x=impulse, length=L, ends=ends, loss=0.98)
    assert abs(1200 * math.log2(_peak_hz(y, f0) / f0)) < 3.0


def test_untuned_waveguide_is_unchanged_for_fitted_models(impulse):
    """foghorn / engine / horn keep the loop their reference fits were made with."""
    y = dsp.comb_waveguide(impulse, 48.0, 0.9, SR, 5000.0, 1.0)
    p = math.exp(-2 * math.pi * 5000.0 / SR)
    b, a = np.array([1.0, -p]), np.zeros(49)
    a[0], a[1], a[48] = 1.0, -p, -0.9 * (1 - p)
    from scipy.signal import lfilter
    assert np.array_equal(y, lfilter(b, a, impulse))


def test_string_t60_is_the_fundamentals_when_reachable(impulse):
    from scipy.signal import stft
    x = np.zeros(3 * SR)
    x[0] = 1.0
    y = run("string", dur=3.0, x=x, f0=220.0, t60=2.0)
    f, t, Z = stft(y, SR, nperseg=4096, noverlap=3072)
    m = 20 * np.log10(np.abs(Z[np.argmin(abs(f - 220.0))]) + 1e-30)
    sel = (t > 0.1) & (t < 2.0)
    t60 = -60 / np.polyfit(t[sel], m[sel], 1)[0]
    assert abs(t60 - 2.0) < 0.1


# ----------------------------------------------------------------------------- Doppler
@pytest.mark.parametrize("speed", [20.0, 250.0, 300.0])
def test_passby_arrival_times_are_exact(speed):
    """A pulse emitted at te must arrive at te + r(te) / c, also for fast flybys (the old 6-step iteration was
    ~0.8 s off at 250 m/s). A smooth 2 ms pulse, not one sample: approaching at 250 m/s the delay line is read
    ~3.7x faster than real time, so a single-sample click can fall between two reads."""
    c, d, tc = 343.0, 10.0, 2.0
    n = int(5.0 * SR)
    L = int(0.002 * SR)
    for te in (0.5, 1.5, 2.0, 3.0):
        x = np.zeros(n)
        i = int(te * SR)
        x[i - L // 2:i - L // 2 + L] = np.hanning(L)
        y = run("passby", dur=5.0, x=x, speed=speed, closest=d, t_closest=tc)
        expect = te + math.hypot(speed * (te - tc), d) / c
        got = np.argmax(np.abs(y)) / SR
        assert abs(got - expect) < 2e-3, (speed, te, got, expect)


# ----------------------------------------------------------------------------- robustness
SOURCES = [n for n, m in REGISTRY.items() if not m.needs_input]


@pytest.mark.parametrize("dur", [0.001, 0.005, 0.05])
def test_every_model_renders_very_short(dur):
    for name, m in REGISTRY.items():
        chain = ([{"model": "impact"}] if m.needs_input else []) + [{"model": name}]
        if name == "modal":
            chain[-1]["params"] = {"shape": "bar"}
        y, sr = soundlab.render({"duration": dur, "layers": [{"chain": chain}]})
        assert len(y) == int(round(dur * sr)) and np.all(np.isfinite(y)), name


@pytest.mark.parametrize("name", [n for n in SOURCES if any(p.name in ("at", "start") for p in REGISTRY[n].params)])
def test_events_at_or_after_the_end_do_not_crash(name):
    key = "at" if any(p.name == "at" for p in REGISTRY[name].params) else "start"
    for t in (0.95, 1.0, 1.5, 4.0):
        y, _ = soundlab.render({"duration": 1.0, "layers": [{"model": name, "params": {key: t}}]})
        assert np.all(np.isfinite(y))


def test_negative_layer_start_is_preroll():
    base = {"duration": 1.0, "seed": 2, "master": {"normalize_db": None, "dc_block_hz": 0, "fade_out": 0},
            "layers": [{"model": "hit", "params": {}, "normalize": False}]}
    early = json.loads(json.dumps(base))
    early["layers"][0]["start"] = -0.1
    y0, _ = soundlab.render(base)
    y1, _ = soundlab.render(early)
    assert len(y1) == len(y0) and np.all(np.isfinite(y1))
    assert np.argmax(np.abs(y1)) < np.argmax(np.abs(y0))      # the hit arrives earlier


def test_matchbox_variations_render():
    """matchbox jitters its layer start by +-50 ms; negative starts used to crash variations 2-5 and 8."""
    for k in range(1, 9):
        y, _ = soundlab.render("recipes/matchbox.json", variation=k)
        assert np.all(np.isfinite(y))


def test_skipped_layer_does_not_reseed_later_layers():
    lay = [{"model": "hit", "start": 5.0}, {"model": "noise", "params": {"hold": 0.2}}]
    # layer 1 sees the same random stream whether layer 0 was skipped (start past the end) or not
    ra, _, sa = soundlab.render({"duration": 1.0, "seed": 4, "layers": lay}, return_layers=True)
    rb, _, sb = soundlab.render({"duration": 1.0, "seed": 4, "layers": [{"model": "hit", "start": 0.99}, lay[1]]},
                                return_layers=True)
    assert np.array_equal(sa["layer1"], sb["layer1"])


def test_crackle_band_order_does_not_matter():
    a = run("crackle", seed=3, f_lo=500.0, f_hi=4000.0)
    b = run("crackle", seed=3, f_lo=4000.0, f_hi=500.0)
    assert np.array_equal(a, b)


def test_friction_stable_at_max_stiffness():
    for st in (1e5, 1e6):
        y = run("ep_friction", dur=0.5, stiffness=st)
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e3


# ----------------------------------------------------------------------------- file output
def test_wav_24_bit_is_real_24_bit(tmp_path):
    from soundlab.audio_io import read_audio, write_wav
    x = 0.9 * np.sin(np.linspace(0, 300, SR))
    x[:2] = [1.0, -1.0]
    p = write_wav(tmp_path / "a.wav", x, SR, 24)
    with wave.open(str(p)) as w:
        assert (w.getsampwidth(), w.getnchannels(), w.getframerate(), w.getnframes()) == (3, 1, SR, SR)
    y, sr = read_audio(p)
    assert sr == SR and np.max(np.abs(y - x)) < 2e-7
    from soundlab.audio_io import to_int16
    assert np.max(np.abs(to_int16(x, dither=False) / 32767 - x)) <= 0.5 / 32767 + 1e-9   # rounded, not truncated
    y16, _ = read_audio(write_wav(tmp_path / "b.wav", x, SR, 16))           # dithered: within 1.5 LSB
    assert np.max(np.abs(y16[2:] - x[2:])) <= 1.5 / 32767 + 1e-9
    with pytest.raises(ValueError):
        write_wav(tmp_path / "c.wav", x, SR, 20)


def test_gui_recipe_ids_stay_inside_the_recipe_folders(tmp_path):
    from soundlab.gui.server import REPO_RECIPES, App
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "x.json").write_text(json.dumps({"layers": [{"model": "hit"}]}))
    app = App(recipe_dirs=[REPO_RECIPES], out_dir=tmp_path / "out")
    for rid in (str(outside / "x"), "my/" + str(outside / "x"), "my/../x", "x//y", "/hit"):
        with pytest.raises(RecipeError):
            app.recipe_path(rid)
    assert app.recipe_path("hit").name == "hit.json"
    assert app.recipe_path("blocks/ui_click").name == "ui_click.json"
