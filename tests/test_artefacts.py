"""v3: tests for the synthesis artefacts behind the reference 'grain / hiss / tinny / digital / plastic bucket' notes."""
import numpy as np

from conftest import SR, run


def inharmonic_db(y, f0, fmin=3000.0, tol=4.0):
    """Energy above fmin that is NOT within +-tol Hz of a harmonic of f0, re the total energy above fmin.
    For a clean periodic source this is tiny; aliasing folds partials onto inharmonic frequencies."""
    y = y[SR // 2:] * np.hanning(len(y) - SR // 2)
    Y = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    sel = f > fmin
    h = np.abs(f / f0 - np.round(f / f0)) * f0 < tol
    return 10 * np.log10(Y[sel & ~h].sum() / (Y[sel].sum() + 1e-30) + 1e-30)


def test_rough_pulses_has_no_sample_grid_grain():
    # v2 snapped each pulse to the sample grid: about -19 dB of inharmonic 'grain' above 3 kHz
    y = run("rough_pulses", 3.0, f0=123.4, level=1.0, jitter=0.0, period_doubling=0.0, shimmer=0.0,
            pulse_ms=0.35, aspiration=0.0)
    assert inharmonic_db(y, 123.4) < -38


def test_valve_pulse_is_oversampled():
    # v2 (no oversampling): about -33 dB
    y = run("valve_pulse", 3.0, f0=123.4, level=1.0, open_quotient=0.5, speed_quotient=3.0, jitter=0.0,
            shimmer=0.0, rasp=0.0, aspiration=0.0)
    assert inharmonic_db(y, 123.4) < -42


def test_chopper_is_clean():
    y = run("chopper", 3.0, freq=123.4, pressure=1.0, open_fraction=0.4, edge=0.05, jitter=0.0, flow_noise=0.0)
    assert inharmonic_db(y, 123.4) < -55


def test_noise_level_is_sample_rate_independent():
    # oversampled blocks scale their noise so the in-band level does not drop 6 dB at 4x the rate
    from soundlab.models.excitation import noise_scale
    assert abs(noise_scale(48000) - 1.0) < 1e-12
    assert abs(noise_scale(192000) - 2.0) < 1e-12


def _notch_depth(y, lo=500, hi=6000):
    """Depth (dB) of the deepest notches below the running median of the long-term spectrum."""
    import scipy.signal as ss
    from scipy.ndimage import median_filter
    f, P = ss.welch(y, SR, nperseg=8192)
    sel = (f > lo) & (f < hi)
    d = 10 * np.log10(P[sel] + 1e-30)
    return np.percentile(median_filter(d, 121) - d, 99)


def test_ground_comb_is_softened():
    """Engine at 4 m: v2's static, perfectly coherent reflection dug notches at ~0.8, 2.3, 3.9 kHz (a phasey
    hiss). v3: frequency-dependent ground and turbulent path wobble."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8 * SR)
    kw = dict(source_height=0.3, listener_height=1.6, distance=4.0, reflection=0.8)
    v2 = run("ground", 8.0, x, ground_hz=6000.0, turbulence=0.0, **kw)
    v3 = run("ground", 8.0, x, **kw)
    assert _notch_depth(v3) < _notch_depth(v2) - 3


def test_explosion_has_no_tonal_boom():
    """The v2 boom was a ringing 2-pole resonator ('plastic bucket'). The low band must not have a narrow peak."""
    import scipy.signal as ss
    y = run("explosion", 6.0, size=0.7, debris=0.0, at=0.05)
    f, P = ss.welch(y, SR, nperseg=16384)
    sel = (f > 30) & (f < 400)
    d = 10 * np.log10(P[sel])
    trend = np.convolve(d, np.ones(15) / 15, mode="same")
    assert np.max((d - trend)[10:-10]) < 8


def test_explosion_roar_lasts():
    from soundlab.analysis import envelope
    y = run("explosion", 8.0, size=0.8, fireball_s=5.0, roar=1.6, debris=0.0, at=0.05)
    e = envelope(y, SR, 10.0)
    assert (e > 0.1 * e.max()).sum() / SR > 2.0  # refs: big blasts stay within -20 dB for several seconds


def test_foghorn_pitch_is_steady_but_alive():
    from soundlab.analysis import f0_track
    y = run("foghorn", 4.0, tone_hz=110.0, blast_s=3.0, start=0.1)
    f = f0_track(y[int(0.6 * SR):int(2.8 * SR)], SR)
    f = np.asarray(f[1] if isinstance(f, tuple) else f, float)
    f = f[np.isfinite(f) & (f > 0)]
    st = 12 * np.log2(f / np.median(f))
    assert abs(np.median(f) - 110) < 3
    assert np.std(st) < 0.3  # steady like the barge ref (no glides)
    assert np.std(st) > 0.005  # but not frozen


def test_match_flame_has_no_mid_honk():
    import scipy.signal as ss
    y = run("match", 2.5, flame_s=2.0, flame_size=0.5, flame_db=-20, start=0.05)
    flame = y[int(1.0 * SR):int(2.0 * SR)]
    f, P = ss.welch(flame, SR, nperseg=4096)
    d = 10 * np.log10(P + 1e-30)
    def band(a, b):
        return d[(f > a) & (f < b)].mean()
    assert band(400, 700) < band(150, 300) + 3  # v2 woof peaked at ~530 Hz


def test_shaker_rattles():
    from soundlab.analysis import envelope
    y = run("shaker", 1.5, rate_hz=4.0, shake_s=1.0, start=0.1)
    e = envelope(y, SR, 60.0)
    pk = ss_peaks(e)
    assert 5 <= pk <= 14  # ~2 reversals per stroke x 4 strokes/s x 1 s


def ss_peaks(e):
    import scipy.signal as ss
    p, _ = ss.find_peaks(e, height=0.15 * e.max(), distance=int(0.06 * SR))
    return len(p)


# ---------------------------------------------------------------- v4 (the v3 notes)
from pathlib import Path

RECIPES = Path(__file__).resolve().parent.parent / "recipes"


def _hf_periodicity(y, lo=2000, hi=10000):
    """Autocorrelation peak (lag 2.5-50 ms) of the high-band envelope: 1 = the top is a strict pulse train."""
    import scipy.signal as ss
    hf = ss.sosfilt(ss.butter(4, [lo, hi], "band", fs=SR, output="sos"), y)
    e = ss.sosfilt(ss.butter(2, 800, fs=SR, output="sos"), np.abs(ss.hilbert(hf)))
    e = ss.resample_poly(e, 1, 24)
    e = e - e.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    ac /= ac[0] + 1e-20
    return float(np.max(ac[5:100]))


def test_engine_top_is_not_a_buzz():
    """v3: identical cracks + metronomic valve ticks on every firing made the top a pulse train at the firing
    rate ('a buzzing on top'; periodicity 0.75 at idle). v4 varies each crack and tick."""
    y = run("engine", 2.0, rpm=900, load=0.35, cylinders=8, firing="crossplane_v8", mechanical=0.8,
            exhaust_crack=0.8, pops=0)
    assert _hf_periodicity(y[SR // 4:]) < 0.55


def _hf(y):
    import scipy.signal as ss
    return ss.sosfilt(ss.butter(4, 2000, "high", fs=SR, output="sos"), y)


def test_match_ignition_is_not_a_fry():
    """v3's ignition crackle ran at up to 500 grains/s: 'bacon frying on high heat'. v4 is a short spray of
    sparks. Measure: kurtosis of the >2 kHz signal in 50 ms frames during the flare (spiky = high), averaged
    over seeds (v3: ~4, v4: ~1.1)."""
    import scipy.stats as st
    def fk(x, w=0.05):
        n = int(w * SR)
        return np.median([st.kurtosis(x[i:i + n]) for i in range(0, len(x) - n, n)])
    ks = [fk(_hf(run("match", 1.5, seed=sd, flare_s=0.28, flame_s=1.0, start=0.05)[int(0.1 * SR):int(0.45 * SR)]))
          for sd in range(1, 7)]
    assert np.mean(ks) < 2.0


def test_shaker_is_grouped_rattle_not_spray():
    """v3: needle-sharp 0.05-0.25 ms clicks plus 1-sample knocks, so the top was a continuous spiky spray
    (busy 41 % of the time, kurtosis ~2300). v4: soft contacts grouped per stroke."""
    import scipy.signal as ss
    import scipy.stats as st
    y = run("shaker", 2.0, rate_hz=3.0, shake_s=1.5, start=0.1)[int(0.1 * SR):int(1.6 * SR)]
    h = _hf(y)
    e = ss.sosfilt(ss.butter(2, 200, fs=SR, output="sos"), np.abs(ss.hilbert(h)))
    busy = np.mean(e > np.percentile(e, 99) * 10 ** (-10 / 20))
    assert busy < 0.2
    assert st.kurtosis(h) < 600


def test_foghorn_scene_tail_is_short():
    """v3's scene had a constant 'sea' noise layer and far, diffused echoes: a wash that lasted seconds after
    the horn stopped ('like ocean waves'). v4: the tail is a short echo of the tone."""
    import soundlab
    from soundlab.analysis import envelope
    y, sr = soundlab.render(RECIPES / "foghorn.json")
    e = envelope(y, sr, 20.0)
    db = 20 * np.log10(e / e.max() + 1e-9)
    stop = np.nonzero(db > -6)[0][-1]
    i = stop + int(1.5 * sr)
    assert i >= len(db) or db[i] < -40
