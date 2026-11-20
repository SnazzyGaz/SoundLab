"""v0.33: fire_stems -- a campfire, a large blaze, embers, a match strike, a gas stove igniting, a sparkler, a fuse, a
transformer hum, an electric arc and a flickering fluorescent tube rebuilt from stems of CC0 recordings
(docs/fire.md).

The stored data is numbers only (band envelopes of the noise bed and of every recorded event, harmonic amplitude /
phase tracks, lines, a static treble ceiling); every default recipe must reproduce its rebuild (fingerprints in
tests/data, no audio). Fresh takes replay forward-only pieces of the stems in a new order with the events riding
along; they and the macros must move the sound the way they say without adding narrow tones, shimmer, hiss, buzz,
frozen stretches or clicks at the joins."""
import json
import lzma
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

import soundlab
from soundlab.core import REGISTRY, RecipeError
from soundlab.models import fire_stems as M

from liquid_checks import artefacts, frozen, new_lines
from squeal_fp import distance, fingerprint

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "recipes"
FP = json.loads((Path(__file__).parent / "data" / "fire_fingerprints.json").read_text())
TAKES = ["campfire", "large_blaze", "embers", "match_strike", "gas_stove_ignite", "sparkler", "fuse", "transformer_hum",
         "electric_arc", "fluorescent_flicker"]
SR = M.SR
_cache = {}


def render(name, k=0, **mac):
    key = (name, k, json.dumps(mac, sort_keys=True))
    if key not in _cache:
        _cache[key] = soundlab.render(R / f"{name}.json", variation=k, macros=mac or None)
    return _cache[key]


def model(source, **p):
    """(audio, placed event times, timeline) of a model render."""
    key = ("model", source, json.dumps(p, sort_keys=True))
    if key not in _cache:
        y = M.render_fire(REGISTRY["fire_stems"].resolve(dict(p, source=source)))
        _cache[key] = (y, list(M.LAST.get("events", [])), M.LAST.get("timeline"))
    return _cache[key]


def db(y):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(y) ** 2)) + 1e-12)


def centroid(y, sr):
    f, P = signal.welch(y, sr, nperseg=4096)
    return float((f * P).sum() / P.sum())


def _hdr():
    raw = M.DATA.read_bytes()
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    return raw, json.loads(p[4:4 + n]), p[4 + n:]


# ---- the stored data ------------------------------------------------------------------------------------------

def test_fire_data_is_compact_numbers_with_credited_cc0_sources():
    raw, hdr, blob = _hdr()
    assert raw[:8] == b"SLFIR001" and len(raw) < 6_000_000
    assert set(M.sources()) == set(TAKES) == set(hdr["sources"])
    third = (ROOT / "THIRD_PARTY.md").read_text()
    for f in hdr["files"].values():
        assert f["license"] in ("CC0", "CC0 1.0"), f
        assert f["url"] in third and f["author"] in third, f["url"]
    for k, s in hdr["sources"].items():
        assert s["env_sr"] == 1000 and s["per_oct"] == 8 and s["env"]["sh"][1] <= s["n"] / 44 + 2, k   # 1 ms steps
        if "h_amp" in s:
            assert s["h_sr"] == 250 and s["h_amp"]["sh"][1] <= s["n"] / 176 + 2, k                    # 4 ms steps
        if "ev_env" in s:
            off, ln = (np.frombuffer(blob, "<i4", s[k2]["sh"][0], s[k2]["o"]) for k2 in ("ev_off", "ev_len"))
            assert s["ev_rate"] <= 8000 and np.all(np.diff(off) <= ln * s["ev_rate"] / SR + 2), k   # frames inside event windows only
        for name, a in s.items():
            if isinstance(a, dict) and "sh" in a and name != "ev_env":
                assert max(a["sh"]) <= s["n"] / 40, (k, name)      # never one value per sample


def test_bad_settings_are_recipe_errors():
    p = REGISTRY["fire_stems"].resolve({})
    with pytest.raises(RecipeError):
        M.render_fire(dict(p, source="volcano"))
    with pytest.raises(RecipeError):
        M.render_fire(dict(p, source="gas_stove_ignite", clicks=3))
    with pytest.raises(RecipeError):
        M.render_fire(dict(p, source="fuse", variation=1, chunk_min_s=4.0, chunk_max_s=5.0))


def test_model_has_no_smoothing_control():
    names = {p.name for p in REGISTRY["fire_stems"].params}
    assert not any("smooth" in n for n in names)


# ---- reproduction ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", TAKES)
def test_default_recipe_reproduces_its_rebuild(name):
    y, sr = render(name)
    d = distance(FP[name]["fp"], fingerprint(y, sr))
    assert d < 0.5, (name, d)
    assert FP[name]["rebuild_distance"] < 0.25, name        # measured 0.02-0.15 dB


def test_reproduction_distance_tells_takes_apart():
    assert distance(FP["campfire"]["fp"], FP["large_blaze"]["fp"]) > 1.0
    assert distance(FP["transformer_hum"]["fp"], FP["fluorescent_flicker"]["fp"]) > 1.0


@pytest.mark.parametrize("name", TAKES)
def test_recipe_renders_varies_and_ends_in_silence(name):
    a, sr = render(name)
    b, _ = render(name, 1)
    assert sr == 48000 and np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.abs(a).max() > 0.5
    n = min(len(a), len(b))
    assert np.abs(a[:n] - b[:n]).max() > 0.01
    assert np.abs(a[-int(0.15 * sr):]).max() < 1e-3          # the take's 30 ms end fade, then 0.3 s of silence


def test_macro_defaults_leave_the_recipe_as_written():
    for name in ["campfire", "gas_stove_ignite", "fuse"]:
        r = json.loads((R / f"{name}.json").read_text())
        mac = {k: v["default"] for k, v in r["macros"].items()}
        a, _ = render(name)
        b, _ = soundlab.render(R / f"{name}.json", macros=mac)
        assert np.allclose(a, b), name


# ---- fresh takes ----------------------------------------------------------------------------------------------

def test_fresh_take_pieces_are_forward_only_random_and_crossfaded():
    rng = np.random.default_rng(5)
    ch = M.random_timeline(12.0, 30.0, rng)
    L = np.array([c[2] for c in ch]) - M.XF
    assert all(c[0] >= 0 and c[0] + c[2] <= 12.0 for c in ch)                # inside the recording, read forwards
    assert all(abs(ch[i + 1][1] - (ch[i][1] + ch[i][2] - M.XF)) < 1e-9 for i in range(len(ch) - 1))   # 100 ms overlaps
    assert ch[-1][1] + ch[-1][2] >= 30.0 and L.min() > 0.29 and L.std() > 0.1   # covers the take, no fixed length
    starts = [round(c[0], 3) for c in ch]
    first = starts[:len(set(starts))]
    assert len(set(starts)) >= 8 and len(set(first)) == len(first) and first != sorted(first)   # every piece once, new order
    ch2 = M.random_timeline(12.0, 30.0, np.random.default_rng(6))
    assert [c[0] for c in ch2] != [c[0] for c in ch]


def test_biased_take_never_brings_a_piece_back_within_three_picks():
    st = M._data()["sources"]["fluorescent_flicker"][0]
    for bias in (-1.2, 0.6):
        ch = M.random_timeline(12.0, 60.0, np.random.default_rng(1), onsets=st["onsets"], act=M._act(st), bias=bias)
        s = [round(c[0], 4) for c in ch]
        assert all(s[i] not in s[max(0, i - 3):i] for i in range(len(s))), bias


def test_cuts_avoid_recorded_transients_in_bed_only_sources():
    st = M._data()["sources"]["electric_arc"][0]
    on = st["onsets"]
    ch = M.random_timeline(12.0, 40.0, np.random.default_rng(2), onsets=on, act=M._act(st))
    def hit(t):
        return np.any(np.abs(on - t) < 0.06) or np.any(np.abs(on - t - M.XF) < 0.06)
    joins = [c[0] for c in ch[1:]]
    rnd = np.random.default_rng(0).uniform(0.5, 11.0, 2000)
    ours, chance = np.mean([hit(t) for t in joins]), np.mean([hit(t) for t in rnd])
    assert ours <= 0.5 * chance, (ours, chance)          # cuts land on quiet points far more often than chance


def test_events_ride_with_their_pieces_even_when_a_piece_is_reused():
    st = M._data()["sources"]["campfire"][0]
    one = M.place_events(st, [(0.0, 0.0, 6.1)], 6.0)
    two = M.place_events(st, [(0.0, 0.0, 6.1), (0.0, 6.0, 6.0)], 12.0)
    assert len(two) == 2 * len(one) - sum(1 for j, s in one if abs(s / SR - 6.0) < M.XF / 2) or len(two) >= 1.9 * len(one)


def test_identity_take_places_every_recorded_event_once():
    _, ev, ch = model("embers")
    assert ch is None and len(ev) == len(M._data()["sources"]["embers"][0]["events"])


def test_density_thins_and_adds_events():
    _, ev0, _ = model("sparkler")
    _, lo, _ = model("sparkler", density=0.6)
    _, hi, _ = model("sparkler", density=1.6)
    assert len(lo) < 0.85 * len(ev0) and len(hi) > 1.35 * len(ev0)


def test_gas_stove_clicks_and_flame():
    st = M._data()["sources"]["gas_stove_ignite"][0]
    T = st["n"] / SR
    for k in (4, 7, 10, 13):
        seq = M.click_intervals(st, k, np.random.default_rng(k))
        ch = M.seq_timeline(seq)
        ev = M.place_events(st, ch, sum(b - a for a, b in seq))
        assert len(ev) == 2 + k, (k, len(ev))                    # knob + catch events, then k clicks
        rep = [s for s in seq[1:-1]]
        assert all(rep[i] != rep[i + 1] for i in range(len(rep) - 1)), k   # never one cycle twice in a row
    few, _ = render("gas_stove_ignite", clicks=0.0)
    many, sr = render("gas_stove_ignite", clicks=1.0)
    mid, _ = render("gas_stove_ignite")
    assert len(few) < len(mid) - 0.5 * sr and len(many) > len(mid) + 0.5 * sr
    y0, _, _ = model("gas_stove_ignite")
    big, _, _ = model("gas_stove_ignite", flame_db=2.5)
    small, _, _ = model("gas_stove_ignite", flame_db=-2.5)

    def burner(v):      # the burning flame re the gas hiss before the first click
        return db(v[int(3.5 * SR):]) - db(v[int(0.3 * SR):int(1.2 * SR)])
    # the bed keeps the take's overall colour, so +-2.5 dB on the burner moves this balance by about +-0.9 dB
    assert burner(big) > burner(y0) + 0.5 and burner(small) < burner(y0) - 0.5 and abs(T - len(y0) / SR) < 0.01


# ---- macros ---------------------------------------------------------------------------------------------------

def test_distance_is_quieter_and_duller():
    far, sr = render("campfire", distance=1.0)
    mid, _ = render("campfire")
    assert db(far) < db(mid) - 5 and centroid(far, sr) < 0.8 * centroid(mid, sr)


def test_burn_speed_is_tape_speed():
    slow, sr = render("fuse", burn_speed=0.0)
    mid, _ = render("fuse")
    assert len(slow) > 1.15 * len(mid) and centroid(slow, sr) < centroid(mid, sr)


def test_transformer_load_moves_the_hum():
    hi, _, _ = model("transformer_hum", harm_db=2.0, tilt=0.03, bed_db=1.5)
    lo, _, _ = model("transformer_hum", harm_db=-3.0, tilt=-0.15, bed_db=-3.0)
    y, _, _ = model("transformer_hum")

    def hum(v):      # level of the harmonics re the 2-8 kHz bed
        f, P = signal.welch(v, SR, nperseg=16384)
        return 10 * np.log10(P[(f > 100) & (f < 1000)].max() / P[(f > 2000) & (f < 8000)].mean())
    assert centroid(hi, SR) > centroid(y, SR) > centroid(lo, SR), (centroid(hi, SR), centroid(y, SR), centroid(lo, SR))
    assert hum(hi) > hum(lo) - 6, (hum(hi), hum(lo))


def test_clang_density_and_pop_density_move_the_events():
    st = M._data()["sources"]["fluorescent_flicker"][0]
    act = M._act(st)

    def busy(ch):
        return np.mean([act(c[0], c[0] + c[2] - M.XF) for c in ch])
    _, _, lo = model("fluorescent_flicker", bias=-1.2, variation=3)
    _, _, hi = model("fluorescent_flicker", bias=0.6, variation=3)
    assert busy(lo) < busy(hi)
    _, ev0, _ = model("embers")
    _, more, _ = model("embers", density=1.4)
    assert len(more) > 1.25 * len(ev0)


# ---- artefacts of fresh takes and macros (narrow tones that persist, frozen stretches, shimmer / hiss / buzz, joins)

# (source, settings, allowances). Default limits: shimmer <= max(1.25x, +10/s), hiss and buzz <= +1.5 dB re the
# default at the same tape speed.
ART = [("campfire", dict(variation=1), {"lines": [140]}),     # the recording's own ~140 Hz component (in the recording)
       ("large_blaze", dict(variation=1), {}),
       ("embers", dict(variation=1), {}),
       ("sparkler", dict(density=1.6, bed_db=1.0, variation=1), {}),
       ("fuse", dict(speed_cents=-300, slow_mode="tape_lp", variation=1), {}),
       ("transformer_hum", dict(variation=1), {}),
       ("electric_arc", dict(variation=1, bias=1.0, harm_db=1.0), {}),
       ("fluorescent_flicker", dict(variation=1), {})]


def _id(s, kw):
    return f"{s}-{'-'.join(f'{k}{v}' for k, v in kw.items())}"


@pytest.mark.parametrize("source,kw,allow", ART, ids=[_id(s, kw) for s, kw, _ in ART])
def test_no_new_shimmer_hiss_buzz_or_frozen_stretches(source, kw, allow):
    c = kw.get("speed_cents", 0)
    base = model(source)[0]
    if c:
        base = signal.resample(base, int(round(len(base) / 2 ** (c / 1200))))
        if c < 0:
            base = signal.sosfiltfilt(signal.butter(2, 16000 * 2 ** (c / 1200) * 0.8, fs=SR, output="sos"), base)
    y = model(source, **kw)[0]
    n = min(len(y), len(base))
    r, _ = artefacts(y[:n], base[:n], SR)
    my, md = r["shimmer"]
    assert my <= allow.get("shimmer", max(1.25 * md, md + 10)), (source, kw, r)
    assert r["hiss_db"] <= allow.get("hiss", 1.5), (source, kw, r)
    assert r["buzz_db"] <= allow.get("buzz", 1.5), (source, kw, r)
    nl = [f for f in new_lines(y[:n], base[:n], SR) if not any(abs(f - a) < 5 for a in allow.get("lines", []))]
    assert not nl, (source, kw, nl)
    fz, fd = frozen(y[:n], SR), frozen(base[:n], SR)
    assert fz > min(-2.5, fd - 1.5), (source, kw, fz, fd)


@pytest.mark.parametrize("source", ["fuse", "transformer_hum", "electric_arc", "fluorescent_flicker"])
def test_no_clicks_at_the_joins_of_a_fresh_take(source):
    y, _, ch = model(source, variation=1)
    h = signal.sosfiltfilt(signal.butter(4, 3000, "high", fs=SR, output="sos"), y)
    fr = int(0.002 * SR)
    e = 10 * np.log10(np.add.reduceat(h ** 2, np.arange(0, len(h), fr)) / fr + 1e-14)

    def bump(t):
        i = int(t / 0.002)
        return e[max(0, i - 3):i + 4].max() - np.median(e[max(0, i - 50):i + 50])
    joins = [o + M.XF / 2 for _, o, _ in ch[1:] if o + M.XF / 2 < len(y) / SR - 0.2]
    rnd = np.random.default_rng(0).uniform(0.2, len(y) / SR - 0.2, 300)
    ours, typical = max(bump(t) for t in joins), np.percentile([bump(t) for t in rnd], 95)
    assert ours <= max(6.0, typical + 2.0), (source, ours, typical)
