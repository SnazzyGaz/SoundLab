"""v0.21 creature_voice: CHIMERA glottis port (MIT), f0/2 ventricular layer, lossy-tube tract."""
import numpy as np
from scipy import signal

import soundlab.recipe as R
from soundlab.models.creature_voice import chimera_shape, glottal_cycles

SR = 24000


def render(p, dur=3.0, seed=4):
    y, _ = R.render({"duration": dur, "seed": seed, "sample_rate": SR, "master": {"normalize_db": None},
                     "layers": [{"model": "creature_voice", "params": p}]}, SR)
    y = np.asarray(y, float)
    assert np.isfinite(y).all()
    return y


def level_at(y, hz):
    f, P = signal.welch(y, SR, nperseg=16384)
    pk = P[np.abs(f - hz) < 3].max()
    return 10 * np.log10(pk / P[(f > 20) & (f < 400)].max())


def test_chimera_pulse_shape():
    ph = np.linspace(0, 1, 1001)
    a = chimera_shape(ph, 0.5, 0.6, 0.8)
    assert a[0] == 0 and a[-1] == 0 and np.all(a[ph >= 0.5] == 0)
    assert abs(ph[np.argmax(a)] - 0.3) < 0.01  # peak at skew x open quotient


def test_logistic_map_walks_into_period_doubling_and_chaos():
    rng = np.random.default_rng(0)
    f0 = np.full(SR * 2, 100.0)
    _, p0, _, _ = glottal_cycles(f0, SR, rng, np.zeros_like(f0), 0, 0, 0)
    _, p2, _, _ = glottal_cycles(f0, SR, rng, np.full_like(f0, 0.35), 0, 0, 0)
    _, pc, _, _ = glottal_cycles(f0, SR, rng, np.full_like(f0, 1.0), 0, 0, 0)
    assert np.std(p0) < 1e-9
    tail = p2[50:]
    assert len(set(np.round(tail, 3))) == 2                  # period two: alternating cycles
    assert len(set(np.round(pc[50:], 3))) > 20               # chaos: no repeating pattern


def test_ventricular_layer_adds_the_f0_over_2_subharmonic():
    plain = render({"f0": 80, "chaos": 0.0, "jitter": 0, "shimmer": 0, "drift_cents": 0})
    vent = render({"f0": 80, "chaos": 0.0, "jitter": 0, "shimmer": 0, "drift_cents": 0, "ventricular": 1.0})
    assert level_at(vent, 40) > level_at(plain, 40) + 15


def test_tract_length_sets_the_round_trip_time():
    # the tube is a comb: its cepstrum peaks at the round trip 2L/c
    def round_trip_ms(cm):
        y = render({"f0": 8, "chaos": 0, "jitter": 0, "shimmer": 0, "drift_cents": 0, "breath": 1.0,
                    "tract_cm": cm, "mouth": 0.0, "bandwidth": 0.5}, 2.0)
        f, P = signal.welch(y, SR, nperseg=4096)
        c = np.abs(np.fft.irfft(np.log(P + 1e-30)))
        q = np.arange(len(c)) / SR * 1000
        m = (q > 0.6) & (q < 8)
        return q[m][np.argmax(c[m])]
    for cm in (20, 35, 60):
        assert abs(round_trip_ms(cm) - 2 * cm / 100 / 350 * 1000) < 0.15


def test_presets_render():
    for name in ("demon_growl", "alligator_bellow", "elephant_rumble", "monster_roar", "dog_growl"):
        import json
        d = json.load(open(f"recipes/{name}.json"))
        y, sr = R.render(d, SR)
        assert np.isfinite(y).all() and np.max(np.abs(y)) > 0.1
