import numpy as np
import pytest

from soundlab.core import Ctx, REGISTRY

SR = 48000


def run(name, dur=1.0, x=None, seed=1, **params):
    m = REGISTRY[name]
    ctx = Ctx(SR, int(dur * SR), np.random.default_rng(seed))
    return m.fn(ctx, x, **m.resolve(params))


def peak_freq(y, sr=SR, lo=20, hi=20000):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    m = (f >= lo) & (f <= hi)
    return f[m][np.argmax(Y[m])]


@pytest.fixture
def impulse():
    x = np.zeros(SR)
    x[0] = 1.0
    return x
