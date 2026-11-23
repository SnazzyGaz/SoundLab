"""weather_stems: rain, thunder, wind, ocean and ambience beds rebuilt from stems of CC0 recordings
(docs/weather.md).

The stored data is numbers only (band envelopes of the noise bed and of any recorded events, a static treble
ceiling); every default source must reproduce its rebuild (fingerprints in tests/data, no audio). Fresh takes
replay forward-only pieces of the stems in a new order; an intensity curve crossfades recordings at several
intensities without stretching or freezing them."""
import json
import lzma
import struct
from pathlib import Path

import numpy as np
import pytest

import soundlab
from soundlab.core import REGISTRY, RecipeError
from soundlab.models import weather_stems as M

from squeal_fp import distance, fingerprint

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "recipes"
FP = json.loads((Path(__file__).parent / "data" / "weather_fingerprints.json").read_text())
SOURCES = list(FP["sources"])
SR = M.SR
_cache = {}


def model(source, **p):
    key = ("model", source, json.dumps(p, sort_keys=True))
    if key not in _cache:
        y = M.render_weather(REGISTRY["weather_stems"].resolve(dict(p, source=source)))
        _cache[key] = (y, M.LAST.get("timeline"), list(M.LAST.get("members") or []))
    return _cache[key]


def render(name, k=0, **mac):
    key = (name, k, json.dumps(mac, sort_keys=True))
    if key not in _cache:
        _cache[key] = soundlab.render(R / f"{name}.json", variation=k, macros=mac or None)
    return _cache[key]



def centroid(y, sr):
    from scipy import signal
    f, P = signal.welch(y, sr, nperseg=4096)
    return float((f * P).sum() / P.sum())


def _hdr():
    raw = M.DATA.read_bytes()
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    return raw, json.loads(p[4:4 + n]), p[4 + n:]


def test_weather_data_is_compact_numbers_with_credited_cc0_sources():
    raw, hdr, blob = _hdr()
    assert raw[:8] == b"SLWTH001"
    assert len(raw) < 16_000_000
    assert b"RIFF" not in raw and b"ID3" not in raw and b"fLaC" not in raw
    assert set(SOURCES).issubset(hdr["sources"])
    for name, s in hdr["sources"].items():
        assert "env" in s and s["n"] > 1000
        assert s["kind"] in ("loop", "shot")
    for fn, cred in hdr["files"].items():
        lic = str(cred.get("license") or cred.get("licence") or "")
        assert "CC0" in lic, (fn, cred)
        assert cred.get("url") and cred.get("author") and cred.get("title")
        assert "freesound.org" in cred["url"]


def test_model_has_no_smoothing_control():
    names = {p.name for p in REGISTRY["weather_stems"].params}
    assert not any("smooth" in n for n in names)


@pytest.mark.parametrize("source", SOURCES)
def test_default_source_reproduces_its_rebuild(source):
    rec = FP["sources"][source]
    y, _, _ = model(source, **rec.get("params", {}))
    d = distance(rec["fp"], fingerprint(y, SR))
    assert d < 0.5, (source, d)
    assert rec["rebuild_distance"] < 0.35, (source, rec["rebuild_distance"])


@pytest.mark.parametrize("name", ["rain", "thunder", "ocean_waves", "wind_breeze"])
def test_recipe_renders_varies_and_is_finite(name):
    a, sr = render(name)
    b, _ = render(name, 1)
    assert sr == 48000 and np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.abs(a).max() > 0.3
    n = min(len(a), len(b))
    assert np.abs(a[:n] - b[:n]).max() > 0.01


def test_macro_defaults_leave_the_recipe_as_written():
    for name in ["rain", "thunder", "ocean_waves", "wind_breeze"]:
        r = json.loads((R / f"{name}.json").read_text())
        mac = {k: v["default"] for k, v in r["macros"].items() if "default" in v}
        a, _ = render(name)
        b, _ = soundlab.render(R / f"{name}.json", macros=mac)
        assert np.allclose(a, b), name


def test_fresh_take_pieces_are_forward_only_random_and_crossfaded():
    rng = np.random.default_rng(5)
    ch = M.random_timeline(24.0, 30.0, rng, clen=(1.2, 2.4))
    L = np.array([c[2] for c in ch]) - M.XF
    assert all(c[0] >= 0 and c[0] + c[2] <= 24.0 + 1e-9 for c in ch)
    assert all(abs(ch[i + 1][1] - (ch[i][1] + ch[i][2] - M.XF)) < 1e-9 for i in range(len(ch) - 1))
    assert ch[-1][1] + ch[-1][2] >= 30.0 and L.min() > 0.29 and L.std() > 0.1
    starts = [round(c[0], 3) for c in ch]
    first = starts[:len(set(starts))]
    assert len(set(starts)) >= 8 and len(set(first)) == len(first) and first != sorted(first)
    ch2 = M.random_timeline(24.0, 30.0, np.random.default_rng(6), clen=(1.2, 2.4))
    assert [c[0] for c in ch2] != [c[0] for c in ch]


def test_longer_take_is_the_requested_length():
    y, timeline, _ = model("rain_light", variation=2, length_s=30.0, intensity=0.0, curve="constant")
    assert abs(len(y) / SR - 30.0) < 0.08
    assert timeline is not None and timeline.get("rain_light") is not None


def test_intensity_swell_crossfades_without_stretching():
    # Heavier rain is not reliably louder in short windows (the two takes have close RMS).
    p = dict(source="rain_light", length_s=8.0, curve="swell", curve_start=0.0, curve_peak=1.0,
             curve_end=0.2, curve_peak_at=0.5, variation=2, intensity=0.0)
    p = REGISTRY["weather_stems"].resolve(p)
    y = M.render_weather(p)
    assert abs(len(y) / SR - 8.0) < 0.08
    members = M.LAST.get("members") or []
    assert "rain_light" in members and "rain_heavy" in members
    group = M._group_members("rain_light")
    names = [n for n, _ in group]
    I = M._curve(p, len(y), SR)
    w0 = M._weights_at(group, float(I[0]))
    wmid = M._weights_at(group, float(I[len(y) // 2]))
    i_light, i_heavy = names.index("rain_light"), names.index("rain_heavy")
    assert w0[i_light] > 0.99 and w0[i_heavy] < 0.01
    assert wmid[i_heavy] > 0.99 and wmid[i_light] < 0.01
    data = M._data()
    for name in ("rain_light", "rain_heavy"):
        ch = (M.LAST.get("timeline") or {}).get(name)
        assert ch, name
        T_src = data["sources"][name][0]["n"] / SR
        assert all(c[0] >= 0 and c[0] + c[2] <= T_src + 1e-6 and c[2] > 0 for c in ch)


def test_constant_exact_intensity_uses_the_identity_path():
    y, timeline, members = model("rain_light", intensity=0.0, length_s=0.0, variation=0, curve="constant")
    assert members == ["rain_light"]
    ch = None if not timeline else timeline.get("rain_light")
    assert ch is None


def test_distance_is_duller_when_further():
    near, _, _ = model("rain_light", distance=0.7, variation=0, length_s=0.0, intensity=0.0)
    far, _, _ = model("rain_light", distance=2.0, variation=0, length_s=0.0, intensity=0.0)
    assert centroid(far, SR) < centroid(near, SR) - 50, (centroid(near, SR), centroid(far, SR))


def test_bad_settings_are_recipe_errors():
    with pytest.raises(RecipeError):
        M.render_weather(REGISTRY["weather_stems"].resolve(dict(source="not_a_source")))
    with pytest.raises(RecipeError):
        M.render_weather(REGISTRY["weather_stems"].resolve(dict(source="rain_light", curve="zigzag")))


LOOP_RECIPES = [
    "rain", "rain_drips", "ocean_waves", "wind_breeze", "wind_howl",
    "forest_day", "night_insects", "city_street", "room_tone",
]


@pytest.mark.parametrize("name", LOOP_RECIPES)
def test_beds_are_loopable(name):
    assert soundlab.is_loopable(R / f"{name}.json")


def test_thunder_is_not_loopable():
    assert not soundlab.is_loopable(R / "thunder.json")


def test_a_rain_loop_is_seamless_and_the_requested_length():
    y, sr = soundlab.render_loop(R / "rain.json", 6.0)
    assert sr == 48000 and len(y) == int(6.0 * sr)
    jump = abs(float(y[0] - y[-1]))
    step = float(np.percentile(np.abs(np.diff(y)), 99))
    assert jump <= step
    mid = float(np.std(y[sr:5 * sr]))
    assert np.std(y[: sr // 10]) > 0.3 * mid and np.std(y[-sr // 10:]) > 0.3 * mid


def test_wind_intensity_swell_crossfades_without_stretching():
    p = dict(source="wind_breeze", length_s=16.0, curve="swell", curve_start=0.0, curve_peak=1.0,
             curve_end=0.2, curve_peak_at=0.5, variation=2, intensity=0.0)
    y = M.render_weather(REGISTRY["weather_stems"].resolve(p))
    assert abs(len(y) / SR - 16.0) < 0.08
    members = M.LAST.get("members") or []
    assert "wind_breeze" in members and "wind_gusty" in members
