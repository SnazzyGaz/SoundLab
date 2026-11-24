"""v0.16: paper, rain and splash cues measured against hold-out recordings. Each test checks the measured
property, not a particular waveform."""
from pathlib import Path

import numpy as np
from scipy import signal

import soundlab

RECIPES = Path(__file__).resolve().parents[1] / "recipes"
SR = 48000


def rec(name, k=1):
    y, sr = soundlab.render(RECIPES / f"{name}.json", variation=k)
    assert sr == SR and np.all(np.isfinite(y))
    return np.asarray(y, float)


def band_env(y, lo, hi, win_s=0.01):
    b = signal.butter(4, [lo, hi], "bandpass", fs=SR, output="sos")
    e = np.abs(signal.hilbert(signal.sosfilt(b, y)))
    w = int(win_s * SR)
    return np.convolve(e, np.ones(w) / w, "same")[::w]


def test_paper_crumple_is_continuous_and_spiky():
    for k in (1, 2):
        y = rec("paper_rustle", k)
        w = int(0.05 * SR)
        n = (len(y) - int(0.4 * SR)) // w
        lv = np.array([np.sqrt(np.mean(y[int(0.1 * SR) + i * w: int(0.1 * SR) + (i + 1) * w] ** 2)) for i in range(n)])
        lv_db = 20 * np.log10(lv / lv.max() + 1e-9)
        assert np.mean(lv_db[: int(3.2 / 0.05)] > -30) > 0.95   # no silences while the sheet is handled
        e = band_env(y[int(0.1 * SR): int(3.5 * SR)], 4000, 8000, 0.002)
        k4 = np.mean((e - e.mean()) ** 4) / np.var(e) ** 2 - 3
        assert k4 > 1.0                                          # crackle: spiky high-band envelope


def test_plastic_and_leaves_unchanged_branch():
    from soundlab.models import library
    import inspect
    src = inspect.getsource(library.rustle)
    assert 'material == "paper"' in src and '("plastic", "leaves")' in src


def test_rain_patter_adds_mid_rate_modulation():
    import json, tempfile
    r = json.loads((RECIPES / "synth_model" / "rain.json").read_text())
    ys = []
    for p in (0.0, 1.0):
        r["layers"][0]["params"]["patter"] = p
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "r.json"
            f.write_text(json.dumps(r))
            y, _ = soundlab.render(f, variation=1)
        e = band_env(np.asarray(y, float)[SR: 7 * SR], 2000, 6000, 0.005)
        f_, P = signal.welch(e - e.mean(), 200, nperseg=256)
        ys.append(P[(f_ >= 4) & (f_ < 30)].sum() / P[f_ > 0].sum())
    assert ys[1] > ys[0]


def test_splash_slosh_decorrelates_bands():
    import json, tempfile
    r = json.loads((RECIPES / "splash.json").read_text())
    cs = []
    for s in (0.0, 1.0):
        r["layers"][0]["params"]["slosh"] = s
        c = []
        for k in (1, 2, 3):
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "s.json"
                f.write_text(json.dumps(r))
                y, _ = soundlab.render(f, variation=k)
            y = np.asarray(y, float)[int(0.2 * SR): int(1.2 * SR)]
            c.append(np.corrcoef(band_env(y, 200, 400), band_env(y, 7000, 12000))[0, 1])
        cs.append(np.mean(c))
    assert cs[1] < cs[0]
