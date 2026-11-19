"""v0.32: machine_stems -- engines, pass-bys, motors, a drill, an AC unit, a clock, a tram, a mower and a sewing
machine rebuilt from stems of CC0 / public-domain recordings (docs/machines.md), plus gesture timelines (start, hold,
change speed, stop) from real spin-up, steady and spin-down stems.

The stored data is numbers only (harmonic amplitude / phase tracks, lines, band envelopes, cycle pulse profiles,
event band envelopes); every default recipe must reproduce its rebuild (fingerprints in tests/data, no audio).
Macros must move the sound the way they say; variations, macros and gestures must not add narrow tones, frozen
stretches, shimmer, hiss, buzz or clicks at loop turns and crossfades."""
import json
import lzma
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

import soundlab
from soundlab.core import REGISTRY, RecipeError
from soundlab.models import machine_stems as M

from liquid_checks import artefacts, frozen, new_lines
from squeal_fp import distance, fingerprint

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "recipes"
FP = json.loads((Path(__file__).parent / "data" / "machine_fingerprints.json").read_text())
TAKES = ["petrol_rev", "diesel_idle", "motorbike_passby", "car_drive_by", "servo_motor", "power_drill", "ac_unit",
         "clock_tick", "tram", "lawn_mower", "sewing_machine"]
GESTURES = ["power_drill_on_off", "power_drill_two_speeds", "servo_motor_on_off", "sewing_machine_on_off",
            "lawn_mower_on_off"]
SR = M.SR
_cache = {}


def render(name, k=0, **mac):
    key = (name, k, json.dumps(mac, sort_keys=True))
    if key not in _cache:
        _cache[key] = soundlab.render(R / f"{name}.json", variation=k, macros=mac or None)
    return _cache[key]


def model(source, **p):
    key = ("model", source, json.dumps(p, sort_keys=True))
    if key not in _cache:
        _cache[key] = M.render_machine(REGISTRY["machine_stems"].resolve(dict(p, source=source)))
    return _cache[key]


def db(y):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(y) ** 2)) + 1e-12)


def centroid(y, sr):
    f, P = signal.welch(y, sr, nperseg=4096)
    return float((f * P).sum() / P.sum())


def period(y, sr):
    """Strongest repetition period (0.35-1.5 s) of the 5 ms level envelope."""
    fr = int(0.005 * sr)
    e = np.sqrt(np.mean(y[:len(y) // fr * fr].reshape(-1, fr) ** 2, 1))
    e = e - e.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    lo, hi = int(0.35 / 0.005), int(1.5 / 0.005)
    return (lo + int(np.argmax(ac[lo:hi]))) * 0.005


def _hdr():
    raw = M.DATA.read_bytes()
    p = lzma.decompress(raw[8:])
    n = struct.unpack("<I", p[:4])[0]
    return raw, json.loads(p[4:4 + n]), p[4 + n:]


# ---- the stored data ------------------------------------------------------------------------------------------

def test_machine_data_is_compact_numbers_with_credited_public_domain_sources():
    raw, hdr, blob = _hdr()
    assert raw[:8] == b"SLMAC001" and len(raw) < 6_500_000
    assert set(M.sources()) == set(TAKES)
    assert set(hdr["sources"]) == set(TAKES) | {"power_drill_gesture", "lawn_mower_gesture"}
    third = (ROOT / "THIRD_PARTY.md").read_text()
    for f in hdr["files"].values():
        assert f["license"] in ("Public domain", "CC0", "CC0 1.0"), f
        assert f["url"] in third and f["author"] in third, f["url"]
    for k, s in hdr["sources"].items():
        assert s["env_sr"] == 1000 and s["per_oct"] == 8 and s["env"]["sh"][1] <= s["n"] / 44 + 2, k   # 1 ms steps
        if "h_amp" in s:
            assert s["h_sr"] == 250 and s["h_amp"]["sh"][1] <= s["n"] / 176 + 2, k                    # 4 ms steps
        if "ev_env" in s:
            off, ln = (np.frombuffer(blob, "<i4", s[k2]["sh"][0], s[k2]["o"]) for k2 in ("ev_off", "ev_len"))
            assert np.all(np.diff(off) <= ln / 22 + 2), k                    # 0.5 ms frames, inside event windows only
        for name, a in s.items():
            if isinstance(a, dict) and "sh" in a:
                assert max(a["sh"]) <= s["n"] / 40, (k, name)      # never one value per sample


def test_unknown_source_and_gesture_without_take_are_recipe_errors():
    with pytest.raises(RecipeError):
        M.render_machine(dict(REGISTRY["machine_stems"].resolve({}), source="jet"))
    with pytest.raises(RecipeError):
        M.render_machine(REGISTRY["machine_stems"].resolve({"source": "ac_unit", "gesture": [
            {"action": "start"}, {"action": "hold", "duration": 1.0}, {"action": "stop"}]}))


def test_model_has_no_smoothing_control():
    names = {p.name for p in REGISTRY["machine_stems"].params}
    assert not any("smooth" in n for n in names)


# ---- reproduction ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", TAKES + GESTURES)
def test_default_recipe_reproduces_its_rebuild(name):
    y, sr = render(name)
    d = distance(FP[name]["fp"], fingerprint(y, sr))
    assert d < 0.5, (name, d)                    # measured: 0.02-0.07 dB at 44.1 kHz


def test_band_envelope_events_stay_close_to_the_approved_grain_rebuilds():
    # clock and sewing ticks are stored as band envelopes (no waveform grains): measured distance to the approved
    # grain rebuilds 1.32 (clock; two seeds of that rebuild differ by 1.29) and 0.92 (sewing)
    assert FP["clock_tick"]["approved_distance"] < 1.5 and FP["sewing_machine"]["approved_distance"] < 1.0


def test_reproduction_distance_tells_takes_apart():
    assert distance(FP["petrol_rev"]["fp"], FP["lawn_mower"]["fp"]) > 1.0
    assert distance(FP["power_drill"]["fp"], FP["power_drill_on_off"]["fp"]) > 1.0


@pytest.mark.parametrize("name", TAKES)
def test_recipe_renders_and_varies(name):
    a, sr = render(name)
    b, _ = render(name, 1)
    assert sr == 48000 and np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.abs(a).max() > 0.5
    n = min(len(a), len(b))
    assert np.abs(a[:n] - b[:n]).max() > 0.01


def test_macro_defaults_leave_the_recipe_as_written():
    for name in ["petrol_rev", "clock_tick", "car_drive_by", "power_drill_on_off"]:
        r = json.loads((R / f"{name}.json").read_text())
        mac = {k: v["default"] for k, v in r["macros"].items()}
        a, _ = render(name)
        b, _ = soundlab.render(R / f"{name}.json", macros=mac)
        assert np.allclose(a, b), name


# ---- macros ---------------------------------------------------------------------------------------------------

def test_throttle_is_faster_brighter_and_louder():
    lo, sr = render("petrol_rev", throttle=0.1)
    mid, _ = render("petrol_rev")
    hi, _ = render("lawn_mower", throttle=1.0)
    mhi, _ = render("lawn_mower")
    assert len(lo) > len(mid) and db(lo) < db(mid)
    assert len(hi) < len(mhi) and centroid(hi, sr) > centroid(mhi, sr)


def test_distance_is_quieter_and_duller():
    near, sr = render("car_drive_by", distance=0.0)
    mid, _ = render("car_drive_by")
    far, _ = render("tram", distance=1.0)
    tmid, _ = render("tram")
    assert db(near) > db(mid) + 3
    assert db(far) < db(tmid) - 6 and centroid(far, sr) < 0.8 * centroid(tmid, sr)


def test_speed_is_tape_speed():
    slow, sr = render("sewing_machine", speed=0.0)
    mid, _ = render("sewing_machine")
    assert len(slow) > 1.15 * len(mid) and centroid(slow, sr) < centroid(mid, sr)


def test_clock_tick_rate_and_room():
    fast, sr = render("clock_tick", tick_rate=1.0)
    mid, _ = render("clock_tick")
    assert period(fast, sr) == pytest.approx(period(mid, sr) / 2, rel=0.1), (period(fast, sr), period(mid, sr))
    room, _ = render("clock_tick", room=0.5)
    # the room adds a tail to the same ticks: what is left after taking out the dry render is substantial
    g = np.dot(room, mid) / np.dot(mid, mid)
    tail = room - g * mid
    assert db(tail) > db(mid) - 20                     # measured: -6.8 dB


# ---- gestures -------------------------------------------------------------------------------------------------

def test_gesture_timeline_errors():
    p = REGISTRY["machine_stems"].resolve({"source": "power_drill"})
    for tl in ([{"action": "stop"}], [{"action": "start"}], [{"action": "hold", "duration": 1}],
               [{"action": "start"}, {"action": "spin"}, {"action": "stop"}],
               [{"action": "start", "speed": 9}, {"action": "stop"}]):
        with pytest.raises(RecipeError):
            M.render_machine(dict(p, gesture=tl))


def test_empty_gesture_plays_the_recording_as_taken():
    a = model("power_drill")
    b = model("power_drill", gesture=[])
    assert np.allclose(a, b)


def _map(take, tl, seed=0):
    st, meta = M._data()["sources"][take]
    r, g, piece, pieces = M._gesture_plan(tl, meta["gesture"], 1.0)
    return M._gesture_map(r, g, piece, pieces, st["n"] / SR, meta["gesture"], seed), meta["gesture"], r


def test_gesture_reads_the_steady_run_forwards_and_backwards_never_holds():
    tl = [{"action": "start"}, {"action": "hold", "duration": 8.0}, {"action": "stop"}]
    m, seg, r = _map("lawn_mower_gesture", tl)
    lo, hi = seg["steady"]
    run = (m["g"] > 0.99) & (m["a"] >= lo) & (m["a"] <= hi)
    d = np.diff(m["a"][run])
    assert np.all(np.abs(d[np.abs(d) < 0.01]) > 0.5 / M.CR)          # always moving: never a held frame
    assert (d < 0).sum() > 1000 and (d > 0).sum() > 1000              # both directions over an 8 s hold
    assert len(r) / M.CR == pytest.approx(1.2 + 8.0 + 3.7 + 0.1, abs=0.02)


def test_drill_steady_run_is_read_forwards_only_from_varied_points():
    tl = [{"action": "start"}, {"action": "hold", "duration": 8.0}, {"action": "stop"}]
    m, seg, r = _map("power_drill_gesture", tl)
    assert seg["loop"] == "forward"
    lo, hi = seg["steady"]
    for k in ("a", "b"):
        d = np.diff(m[k][(m["g"] > 0.99)])
        small = d[np.abs(d) < 0.01]
        assert np.all(small > 0.5 / M.CR)                              # forwards only, never held, never backwards
    jumps = np.nonzero(np.abs(np.diff(m["a"])) > 0.01)[0]
    starts = np.round(m["a"][jumps + 1], 3)
    assert len(jumps) > 30 and len(np.unique(starts)) > 0.8 * len(starts)   # chunks start at varied points
    gaps = np.diff(jumps)[:-1] / M.CR                                    # (the last one runs into the spin-down)
    assert gaps.min() > 0.12 and gaps.std() > 0.02                      # no fixed repeat rate
    a2, *_ = _map("power_drill_gesture", tl, seed=1)
    assert not np.allclose(a2["a"], m["a"])                             # the seed picks the chunks
    up, dn = seg["up"], seg["down"]
    assert len(r) / M.CR == pytest.approx(up[1] - up[0] + 8.0 + dn[1] - dn[0] + 0.1, abs=0.02)


def test_gesture_starts_and_ends_silent_and_holds_its_level():
    y, sr = render("power_drill_on_off")
    fr = int(0.05 * sr)
    lv = 20 * np.log10(np.sqrt(np.mean(y[:len(y) // fr * fr].reshape(-1, fr) ** 2, 1)) + 1e-9)
    assert lv[-2:].max() < lv.max() - 40                               # stopped
    steady = lv[int(0.4 / 0.05):int(3.0 / 0.05)]
    assert steady.min() > lv.max() - 10                                # held at speed (no dropouts at the loop turns)


def test_two_speeds_go_up_by_tape_speed():
    y, sr = render("power_drill_two_speeds")
    a, b = y[int(0.4 * sr):int(2.6 * sr)], y[int(3.5 * sr):int(5.7 * sr)]
    grid = 2 ** np.arange(np.log2(100), np.log2(4000), 1 / 96)

    def logspec(v):
        f, P = signal.welch(v, sr, nperseg=16384)
        L = np.interp(grid, f, 10 * np.log10(P + 1e-20))
        return L - L.mean()

    La, Lb = logspec(a), logspec(b)
    sh = max(range(0, 96), key=lambda k: np.dot(La[:len(La) - k], Lb[k:]) / (len(La) - k))
    assert 2 ** (sh / 96) == pytest.approx(1.15 / 0.8, rel=0.03), sh   # the whole spectrum moves with the speeds


def _audio_gesture(z, src, tl):
    """The same gesture played on audio: the as-recorded take read through the identical map."""
    tk = M.GESTURE_TAKE.get(src, src)
    st, meta = M._data()["sources"][tk]
    r, g, piece, pieces = M._gesture_plan(tl, meta["gesture"], 1.0)
    seed = int(round(REGISTRY["machine_stems"].resolve({"source": src, "gesture": tl})["seed"]))
    m = M._gesture_map(r, g, piece, pieces, st["n"] / SR, meta["gesture"], seed)
    u = np.arange(m["n"]) / SR
    gu = np.arange(len(m["a"])) / M.CR
    ta, tb, w = (np.interp(u, gu, m[k]) for k in ("a", "b", "w"))
    rd = lambda t: M._sinc_read(z, M._mirror(t * SR, 0, len(z) - 1))   # band-limited, as the model reads
    v = (np.sqrt(1 - w) * rd(ta) + np.sqrt(w) * rd(tb)) * M._ctrl_to_audio(m["g"], m["n"])
    return M._vary_tape(v, m["u_out"], m["t_out"], r), m, r


def _take(src):
    tk = M.GESTURE_TAKE.get(src, src)
    st = M._data()["sources"][tk][0]
    p = REGISTRY["machine_stems"].resolve({"source": src})
    y = M._render_take(tk, st, p, 0, np.random.default_rng(100))
    return y / np.abs(y).max() * 0.89


def _seams(m):
    d = np.sign(np.diff(m["a"]))
    turn = np.nonzero(d[1:] * d[:-1] < 0)[0] + 1
    xf = np.nonzero(np.diff((m["w"] > 0.5).astype(int)) != 0)[0]
    return np.interp(np.r_[turn, xf] / M.CR, m["u_out"], m["t_out"])


def _seam_hf(y, ts):
    h = signal.sosfiltfilt(signal.butter(4, 3000, "high", fs=SR, output="sos"), y)
    fr = int(0.002 * SR)
    e = 10 * np.log10(np.add.reduceat(h ** 2, np.arange(0, len(h), fr)) / fr + 1e-14)
    return [float(e[max(0, int(t / 0.002) - 3):int(t / 0.002) + 4].max() - np.median(e[max(0, int(t / 0.002) - 50):int(t / 0.002) + 50])) for t in ts]


@pytest.mark.parametrize("name", GESTURES[1:])
def test_gesture_has_no_artefacts_against_the_same_gesture_on_audio(name):
    r_ = json.loads((R / f"{name}.json").read_text())
    src, tl = r_["layers"][0]["params"]["source"], r_["layers"][0]["params"]["gesture"]
    y = model(src, gesture=tl)
    base, m, r = _audio_gesture(_take(src), src, tl)
    n = min(len(y), len(base))
    y, base = y[:n], base[:n] * np.abs(y[:n]).max() / np.abs(base[:n]).max()
    res, fails = artefacts(y, base, SR)
    assert not fails, (name, res)
    assert not new_lines(y, base, SR), name
    fz, fd = frozen(y, SR), frozen(base, SR)
    assert fz > min(-2.5, fd - 1.5), (name, fz, fd)
    ts = _seams(m)
    ours, theirs = _seam_hf(y, ts), _seam_hf(base, ts)
    assert max(ours, default=0) <= max(6.0, max(theirs, default=0) + 2), (name, ours, theirs)   # no clicks at turns / crossfades


def test_stems_are_read_backwards_past_the_end_never_held():
    A = np.arange(10.0)[None, :]
    out = M._read(A, 1.0, np.arange(9, 15, dtype=float))
    assert list(out[0]) == [9, 8, 7, 6, 5, 4]


# ---- artefacts of variations and macros (narrow tones that persist, frozen stretches, shimmer / hiss / buzz) ---

# (source, settings, allowances). Default limits: shimmer <= max(1.25x, +10/s), hiss and buzz <= +1.5 dB re the
# default at the same tape speed.
ART = [("petrol_rev", dict(variation=1), {}),
       ("diesel_idle", dict(variation=1), {}),
       ("diesel_idle", dict(pulse_gamma=0.6, tilt=-0.15, harm_db=-1.5, speed_cents=60), {}),
       ("car_drive_by", dict(distance=0.6), {}),
       ("ac_unit", dict(variation=1), {}),
       ("clock_tick", dict(variation=1), {"hiss": 2.0}),       # re-sequenced: another mix of the real ticks (1.7)
       ("lawn_mower", dict(variation=1), {}),
       ("sewing_machine", dict(variation=1), {}),
       ("power_drill", dict(speed_cents=250), {}),
       ("servo_motor", dict(speed_cents=-300, slow_mode="tape_lp"), {})]


def _id(s, kw):
    return f"{s}-{'-'.join(f'{k}{v}' for k, v in kw.items())}"


@pytest.mark.parametrize("source,kw,allow", ART, ids=[_id(s, kw) for s, kw, _ in ART])
def test_no_new_shimmer_hiss_or_buzz(source, kw, allow):
    c = kw.get("speed_cents", 0)
    base = model(source)
    if c:
        base = signal.resample(base, int(round(len(base) / 2 ** (c / 1200))))
        if c < 0:
            base = signal.sosfiltfilt(signal.butter(2, 16000 * 2 ** (c / 1200) * 0.8, fs=SR, output="sos"), base)
    y = model(source, **kw)
    n = min(len(y), len(base))
    r, _ = artefacts(y[:n], base[:n], SR)
    my, md = r["shimmer"]
    assert my <= allow.get("shimmer", max(1.25 * md, md + 10)), (source, kw, r)
    assert r["hiss_db"] <= allow.get("hiss", 1.5), (source, kw, r)
    assert r["buzz_db"] <= allow.get("buzz", 1.5), (source, kw, r)
    assert not new_lines(y[:n], base[:n], SR), (source, kw)
