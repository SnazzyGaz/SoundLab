"""v0.31: liquid_stems -- splashes, drips, a pour, a stream, a river, a waterfall, boiling and sloshing rebuilt from
stems of public-domain / CC0 recordings (docs/liquids.md).

The stored data is numbers only (per-band noise envelopes, bubble chirps, STFT band envelopes, tracks, onset
envelopes, colour); every default recipe must reproduce its approved rebuild (fingerprints in tests/data, no
audio). Macros must move the sound the way they say, and variations and macro settings must not add narrow
tones, frozen stretches, shimmer, hiss or buzz relative to the default."""
import json
import lzma
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

import soundlab
from soundlab.core import REGISTRY, RecipeError
from soundlab.models import liquid_stems as L

from liquid_checks import artefacts, frozen, new_lines
from squeal_fp import distance, fingerprint

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "recipes"
FP = json.loads((Path(__file__).parent / "data" / "liquid_fingerprints.json").read_text())
RECIPES = ["splash_small", "splash_medium", "splash_large", "drip_tub", "pour_glass", "stream", "river", "waterfall",
           "boiling", "slosh"]
SR = L.SR
_cache = {}


def render(name, k=0, **mac):
    key = (name, k, json.dumps(mac, sort_keys=True))
    if key not in _cache:
        _cache[key] = soundlab.render(R / f"{name}.json", variation=k, macros=mac or None)
    return _cache[key]


def model(source, **p):
    key = ("model", source, json.dumps(p, sort_keys=True))
    if key not in _cache:
        _cache[key] = L.render_liquid(REGISTRY["liquid_stems"].resolve(dict(p, source=source)))
    return _cache[key]


def db(y):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(y) ** 2)) + 1e-12)


def centroid(y, sr):
    f, P = signal.welch(y, sr, nperseg=4096)
    return float((f * P).sum() / P.sum())


def onsets(y, sr):
    e = 20 * np.log10(np.sqrt(signal.convolve(y ** 2, np.ones(int(0.005 * sr)) / int(0.005 * sr), "same")) + 1e-9)
    d = e[int(0.02 * sr):] - e[:-int(0.02 * sr)]
    pk, _ = signal.find_peaks(d, height=20, distance=int(0.2 * sr))
    return len(pk)


# ---- the stored data ------------------------------------------------------------------------------------------

def test_liquid_data_is_compact_numbers_with_credited_public_domain_sources():
    raw = L.DATA.read_bytes()
    assert raw[:8] == b"SLLIQ001" and len(raw) < 1_100_000
    p = lzma.decompress(raw[8:])
    hdr = json.loads(p[4:4 + struct.unpack("<I", p[:4])[0]])
    assert set(hdr["sources"]) == set(L.FAMILY)
    third = (ROOT / "THIRD_PARTY.md").read_text()
    for f in hdr["files"].values():
        assert f["license"] in ("Public domain", "Public Domain Mark 1.0", "CC0", "CC0 1.0"), f
        assert f["url"] in third and f["author"] in third, f["url"]
    for k, s in hdr["sources"].items():
        if s["engine"] == "env":
            assert s["env_sr"] == 500 and s["per_oct"] == 8 and s["env"]["sh"][1] <= s["n"] / 88 + 2, k   # 2 ms steps
        for name, a in s.items():
            if isinstance(a, dict) and "sh" in a:
                assert max(a["sh"]) <= s["n"] / 40, (k, name)      # never one value per sample
    clean = {k: s["cleaning"] for k, s in hdr["sources"].items() if s["engine"] == "env"}
    assert clean == {"splash_small": "on", "splash_medium": "on", "river": "on", "waterfall": "on", "drip": "on",
                     "pour_glass": "off", "splash_large": "off"}


def test_unknown_source_is_a_recipe_error():
    with pytest.raises(RecipeError):
        L.render_liquid(dict(REGISTRY["liquid_stems"].resolve({}), source="lake"))


def test_model_has_no_smoothing_control():
    names = {p.name for p in REGISTRY["liquid_stems"].params}
    assert not any("smooth" in n for n in names)


# ---- reproduction ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", RECIPES)
def test_default_recipe_reproduces_its_approved_rebuild(name):
    y, sr = render(name)
    d = distance(FP[name]["fp"], fingerprint(y, sr))
    assert d < 0.5, (name, d)                    # measured: 0.05-0.35 dB


def test_tape_speed_splash_take_reproduces_its_reference():
    e = FP["splash_large_tape"]
    y = L.render_liquid(REGISTRY["liquid_stems"].resolve(e["params"]))
    d = distance(e["fp"], fingerprint(y, e["sr"]))
    assert d < 0.5, d                            # measured: 0.05 dB


def test_reproduction_distance_tells_takes_apart():
    assert distance(FP["splash_large"]["fp"], FP["splash_large_tape"]["fp"]) > 0.5
    assert distance(FP["river"]["fp"], FP["waterfall"]["fp"]) > 1.0


@pytest.mark.parametrize("name", RECIPES)
def test_recipe_renders_and_varies(name):
    a, sr = render(name)
    b, _ = render(name, 1)
    assert sr == 48000 and np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.abs(a).max() > 0.5
    n = min(len(a), len(b))
    assert np.abs(a[:n] - b[:n]).max() > 0.01


@pytest.mark.parametrize("name", ["stream", "river", "waterfall"])
def test_flow_variations_have_no_silent_end(name):
    y, sr = render(name, 1)
    tail = y[-int(0.5 * sr) - int(0.05 * sr):-int(0.05 * sr)]   # before the master fade
    assert db(tail) > db(y) - 6, (db(tail), db(y))


# ---- macros ---------------------------------------------------------------------------------------------------

def test_splash_size_lowers_and_lengthens():
    small, sr = render("splash_medium", size=0.0)
    mid, _ = render("splash_medium")
    big, _ = render("splash_medium", size=1.0)
    assert len(small) < len(mid) < len(big)
    assert centroid(small, sr) > centroid(mid, sr) > centroid(big, sr)
    assert db(small) < db(mid)


def test_splash_size_by_tape_speed_is_available_on_the_large_splash():
    a, sr = render("splash_large", size=0.6)
    b, _ = render("splash_large", size=0.6, size_by="tape")
    assert centroid(b, sr) < centroid(a, sr) * 1.2 and np.abs(a[:len(b)] - b[:len(a)]).max() > 0.01


def test_splash_distance_is_quieter_and_duller():
    near, sr = render("splash_medium")
    far, _ = render("splash_medium", distance=1.0)
    assert db(far) < db(near) - 6 and centroid(far, sr) < 0.7 * centroid(near, sr)


def test_splash_tail_lengthens_or_fades():
    short, sr = render("splash_medium", tail=0.0)
    mid, _ = render("splash_medium")
    long_, _ = render("splash_medium", tail=1.0)
    assert len(long_) > len(mid)
    late = slice(int(1.0 * sr), int(1.9 * sr))
    assert db(short[late]) < db(mid[late]) - 3


def test_drip_rate_and_pitch():
    slow, sr = render("drip_tub", rate=0.0)
    fast, _ = render("drip_tub", rate=1.0)
    assert onsets(fast, sr) > onsets(slow, sr) + 2
    lo, _ = render("drip_tub", pitch=0.0)
    hi, _ = render("drip_tub", pitch=1.0)
    assert centroid(hi, sr) > centroid(lo, sr)


def test_pour_fill_and_vessel():
    slow, sr = render("pour_glass", fill=0.0)
    fast, _ = render("pour_glass", fill=1.0)
    assert len(slow) > 1.8 * len(fast)
    small, _ = render("pour_glass", vessel=0.0)
    big, _ = render("pour_glass", vessel=1.0)
    assert len(big) > 1.8 * len(small) and centroid(small, sr) > centroid(big, sr)


@pytest.mark.parametrize("name", ["stream", "river", "waterfall"])
def test_flow_strength_is_quieter_and_darker_without_a_pitch_shift(name):
    lv, ce = [], []
    for m in (0.8, 0.4, 0.0):
        y, sr = render(name, strength=m)
        lv.append(db(y))
        ce.append(centroid(y, sr))
        assert len(y) == len(render(name)[0])        # fixed length: slower envelopes, read backwards past the end
    assert lv[0] > lv[1] > lv[2] and ce[0] > ce[1] > ce[2], (lv, ce)
    p = REGISTRY["liquid_stems"].resolve({"source": name, "strength": 0.5})
    s = L.flow(L._data()["sources"][name][0], 0.5)
    assert np.allclose(s["ltas6"][:10], L._data()["sources"][name][0]["ltas6"][:10] + 8 * np.log2(0.5), atol=0.5), p
    assert "tape" not in s


def test_boiling_intensity_and_slosh_amount():
    lo, sr = render("boiling", intensity=0.0)
    hi, _ = render("boiling", intensity=1.0)
    assert centroid(hi, sr) > centroid(lo, sr)
    gentle, _ = render("slosh", amount=0.0)
    hard, _ = render("slosh", amount=1.0)
    assert len(gentle) > len(hard) and centroid(hard, sr) > centroid(gentle, sr)


def test_macro_defaults_leave_the_recipe_as_written():
    for name in RECIPES:
        r = json.loads((R / f"{name}.json").read_text())
        mac = {k: v["default"] for k, v in r["macros"].items()}
        a, _ = render(name)
        b, _ = soundlab.render(R / f"{name}.json", macros=mac)
        assert np.allclose(a, b), name


# ---- artefacts (narrow tones that persist, frozen stretches, shimmer / hiss / buzz) ---------------------------

# (source, settings, allowances). Default limits: shimmer <= max(1.25x, +10/s), hiss and buzz <= +1.5 dB re the default.
# Allowances cover changes the settings are meant to make, measured on the approved takes with these settings.
ART = [("splash_large", dict(size=1.05, size_mode="tape", seed=1), {}),
       ("splash_small", dict(distance=4.0), {}),
       ("splash_medium", dict(tail=2.0), {}),
       ("splash_medium", dict(variation=1), {}),
       ("splash_medium", dict(size=0.6), {"hiss": 7.5, "buzz": 6.5}),          # smaller = higher: more highs (6.5-6.9 / 5.2-5.8)
       ("drip", dict(variation=4, rate=1.5, pitch=200), {"shimmer": 60, "hiss": 9.5}),   # smaller drops ring higher (8.4-8.6)
       ("drip", dict(variation=5, rate=0.7, pitch=-250), {"hiss": 6.0}),       # measured 5.0-5.1
       ("pour_glass", dict(variation=1), {}),
       ("stream", dict(variation=1), {}),
       ("stream", dict(variation=10, strength=0.8), {}),
       ("river", dict(variation=1), {}),
       ("river", dict(variation=7), {}),
       ("river", dict(strength=0.35), {}),
       ("river", dict(strength=1.4), {"hiss": 3.5}),                            # brighter by design (+6 dB/doubling shelf): 2.8
       ("waterfall", dict(variation=1), {}),
       ("waterfall", dict(variation=8), {}),
       ("waterfall", dict(variation=9, strength=0.6), {"buzz": 4.0}),         # the low-pass narrows the top bands: 2.9
       ("boiling", dict(variation=1), {}),
       ("boiling", dict(variation=12, intensity=0.6), {}),
       ("boiling", dict(intensity=0.3), {}),
       ("slosh", dict(variation=1), {}),
       ("slosh", dict(amount=0.5), {})]


def _id(s, kw):
    return f"{s}-{'-'.join(f'{k}{v}' for k, v in kw.items())}"


@pytest.mark.parametrize("source,kw,allow", ART, ids=[_id(s, kw) for s, kw, _ in ART])
def test_no_new_shimmer_hiss_or_buzz(source, kw, allow):
    r, _ = artefacts(model(source, **kw), model(source), SR)
    my, md = r["shimmer"]
    assert my <= allow.get("shimmer", max(1.25 * md, md + 10)), (source, kw, r)
    assert r["hiss_db"] <= allow.get("hiss", 1.5), (source, kw, r)
    assert r["buzz_db"] <= allow.get("buzz", 1.5), (source, kw, r)


TEX = [("stream", dict(variation=1)), ("stream", dict(strength=0.3)), ("river", dict(variation=1)),
       ("river", dict(strength=0.3)), ("river", dict(strength=1.5)), ("waterfall", dict(variation=1)),
       ("waterfall", dict(strength=0.3)), ("boiling", dict(intensity=0.3)), ("boiling", dict(intensity=2.0)),
       ("slosh", dict(amount=0.5)), ("slosh", dict(amount=1.6)), ("splash_large", dict(size=1.05, size_mode="tape", seed=1))]


@pytest.mark.parametrize("source,kw", TEX, ids=[_id(s, kw) for s, kw in TEX])
def test_no_narrow_tones_that_persist(source, kw):
    cents = -700 * np.log2(kw["size"]) if "size" in kw else (120 * np.log2(kw["amount"]) if "amount" in kw else 0.0)
    assert not new_lines(model(source, **kw), model(source), SR, cents), (source, kw)


@pytest.mark.parametrize("source,kw", [t for t in TEX if t[0] != "splash_large"],
                         ids=[_id(s, kw) for s, kw in TEX if s != "splash_large"])
def test_no_frozen_stretch(source, kw):
    fz, fd = frozen(model(source, **kw), SR), frozen(model(source), SR)
    assert fz > min(-2.5, fd - 1.5), (source, kw, fz, fd)   # a held frame reads far below the clip's own spread


def test_stretch_reads_backwards_past_the_end_never_holds():
    A = np.arange(10.0)[None, :]
    out = L._warp_cols(A, 1.0, 15.0, 10.0, lambda t: t)
    assert list(out[0, 9:15]) == [9, 8, 7, 6, 5, 4]
