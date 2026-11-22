"""Slider-app speed paths (soundlab.fastpath, the app's render cache and warm-up): every one must leave the samples
exactly as the plain render makes them."""
import math

import numpy as np
import pytest

from soundlab import fastpath, recipe
from soundlab.finish import _release, limit
from soundlab.gui.server import REPO_RECIPES, App, waveform
from soundlab.models import machine_stems, recstems


@pytest.fixture()
def threads():
    yield fastpath.set_threads
    fastpath.set_threads(None)


def test_threaded_band_loop_is_bit_identical(threads):
    rng = np.random.default_rng(3)
    env = np.abs(rng.standard_normal((76, 300))).astype(np.float32)
    threads(1)
    a = machine_stems._vocoder.uncached(env, 1000, 12001, None, 5)
    b = recstems.voc_ev.uncached(env, 1000, 9001, 7, 40)
    threads(4)
    assert np.array_equal(a, machine_stems._vocoder.uncached(env, 1000, 12001, None, 5))
    assert np.array_equal(b, recstems.voc_ev.uncached(env, 1000, 9001, 7, 40))


def test_memo_hands_out_copies():
    env = np.abs(np.random.default_rng(1).standard_normal((76, 120)))
    a = machine_stems._vocoder(env, 1000, 5000, None, 2)
    a[:] = 0
    b = machine_stems._vocoder(env, 1000, 5000, None, 2)
    assert np.array_equal(b, machine_stems._vocoder.uncached(env, 1000, 5000, None, 2)) and np.any(b)
    assert not np.array_equal(b, machine_stems._vocoder(env, 1000, 5000, None, 3))      # the seed is in the key


def _release_loop(seg, a):
    seg = seg.copy()
    for i in range(1, len(seg)):
        v = seg[i - 1] * a
        if v > seg[i]:
            seg[i] = v
    return seg


def test_limiter_release_matches_the_loop():
    from scipy.ndimage import maximum_filter1d
    a = math.exp(-1.0 / (0.06 * 48000))
    for k in range(12):
        rng = np.random.default_rng(k)
        x = np.abs(rng.standard_normal(int(rng.integers(2, 60000)))) * (0.5 + 2 * rng.random())
        red = maximum_filter1d(1.0 - np.minimum(1.0, 1 / (x + 1e-12)), 145, mode="nearest")
        if k % 3 == 0:
            red = np.round(red, 2)          # plateaus and exact ties
        assert np.array_equal(_release(red, a), _release_loop(red, a))
    y = limit(np.random.default_rng(0).standard_normal((4000, 2)), 48000, -1.0)
    assert y.shape == (4000, 2) and np.all(np.isfinite(y))


def test_waveform_bins_match_slices():
    for n in (1, 7, 800, 801, 12345):
        x = np.random.default_rng(n).standard_normal(n)
        edges = np.linspace(0, n, 801).astype(int)
        ref = [[round(float(x[a:b].min()), 4), round(float(x[a:b].max()), 4)] if b > a else [0, 0]
               for a, b in zip(edges[:-1], edges[1:])]
        assert waveform(x) == ref


@pytest.mark.parametrize("rid", ["door_slam", "punch"])
def test_block_memo_keeps_samples(rid):
    path = REPO_RECIPES / f"{rid}.json"
    plain = [recipe.render(path, variation=k, ring_out=True)[0] for k in (0, 1, 0)]
    fastpath.set_block_memo(True)
    try:
        fast = [recipe.render(path, variation=k, ring_out=True)[0] for k in (0, 1, 0, 1)]
    finally:
        fastpath.set_block_memo(False)
    for a, b in zip(plain + plain[1:2], fast):
        assert np.array_equal(a, b)


def test_page_render_superseded_over_http(tmp_path):
    """Two page renders back to back: the first one is dropped (or done), the second always answers."""
    import json
    import threading
    import urllib.request
    from soundlab.gui import make_server
    s = make_server(0, recipe_dirs=[REPO_RECIPES], out_dir=tmp_path / "out", fast=False)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{s.server_address[1]}/api/render"

    def post(body):
        rq = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(rq) as r:
            return json.loads(r.read())
    try:
        got = {}
        t = threading.Thread(target=lambda: got.update(a=post({"id": "rain", "supersede": True})))
        t.start()
        import time
        t0 = time.time()
        while s.app.gen < 1 and time.time() - t0 < 30:       # the first request has reached the server
            time.sleep(0.005)
        b = post({"id": "punch", "supersede": True})
        t.join()
        assert "url" in b and (got["a"] == {"superseded": True} or "url" in got["a"])
    finally:
        s.shutdown()
        s.server_close()


def test_app_caches_and_supersedes(tmp_path):
    app = App(recipe_dirs=[REPO_RECIPES], out_dir=tmp_path)
    calls = []
    real = app.finished_audio

    def counted(req):
        calls.append(req["id"])
        return real(req)
    app.finished_audio = counted
    a = app.render({"id": "punch", "supersede": True})
    b = app.render({"id": "punch"})                       # same request: from the cache
    assert calls == ["punch"] and a["waveform"] == b["waveform"]
    assert np.array_equal(app.cache[a["key"]][0], app.cache[b["key"]][0])

    def replaced(req):                                    # a newer page render arrives mid-way
        app.gen += 1
        fastpath.check()
        return real(req)
    app.finished_audio = replaced
    assert app.render({"id": "door_slam", "supersede": True}) == {"superseded": True}
    with fastpath.cancel_scope(lambda: False):
        fastpath.check()


def test_warm_up_only_for_stem_recipes(tmp_path):
    app = App(recipe_dirs=[REPO_RECIPES], out_dir=tmp_path)
    assert app.prefetch_requests({"id": "rain"}) == []
    qs = app.prefetch_requests({"id": "concrete_walk", "macros": {"force": 0.6}})
    assert qs and qs[-1]["variation"] == 1
    moved = {k for q in qs[:-1] for k in q["macros"] if q["macros"][k] in (0.0, 1.0)}
    assert "force" in moved and "seed" not in moved
    assert all(q["macros"].get("force", 0.6) in (0.6, 0.0, 1.0) for q in qs[:-1])


def test_batched_band_terms_match_the_band_loop():
    from soundlab.models.liquid_stems import _analytic, _band_terms
    for n in (4800, 4801):
        rng = np.random.default_rng(n)
        Y = np.fft.rfft(rng.standard_normal(n))
        H, _ = recstems._bank(n, 8)
        E = np.abs(rng.standard_normal((len(H), n)))
        ref = []
        for h, e in zip(H, E):
            a = _analytic(Y, h, n)
            c = a / (np.abs(a) + 1e-9 * (np.abs(a).max() + 1e-30))
            ref.append(np.fft.rfft((e * c).real) * h)
        assert np.array_equal(np.concatenate([_band_terms(Y, H[i:i + 5], E[i:i + 5], n) for i in range(0, len(H), 5)]),
                              np.array(ref))


def test_liquid_env_noise_matches_the_plain_loop(threads):
    from soundlab.models import liquid_stems as L
    env = np.abs(np.random.default_rng(2).standard_normal((40, 90)))
    threads(1)
    a = L._env_noise.uncached(env, 1000, 4000, 4, 9, 4)
    threads(3)
    assert np.array_equal(a, L._env_noise(env, 1000, 4000, 4, 9, 4))
