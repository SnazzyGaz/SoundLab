"""v0.22 export: ring-out tails, stereo beds, 16-bit dither."""
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import soundlab
from soundlab import stereo as st
from soundlab.audio_io import read_audio, to_int16, write_wav
from soundlab.finish import momentary_max_lufs, trim_tail
from soundlab.recipe import ring_tail_s

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "recipes"
SR = 48000

ROOM = {"duration": 0.5, "seed": 1,
        "layers": [{"model": "noise", "params": {"hold": 0.05, "release": 0.02}}],
        "post": [{"model": "room", "params": {"volume": 2000, "absorption": 0.2, "rt60": 2.0}}]}


# ----------------------------------------------------------------------------- ring-out
def test_plain_render_keeps_its_length():
    y, sr = soundlab.render(ROOM)
    assert len(y) == int(0.5 * sr)


def test_ring_out_lets_the_room_finish():
    a, sr = soundlab.render(ROOM)
    b, _ = soundlab.render(ROOM, ring_out=True)
    assert ring_tail_s(ROOM) == pytest.approx(1.2 * 2.0 + 0.1)
    assert len(b) == len(a) + int(round(ring_tail_s(ROOM) * sr))
    env = np.sqrt(np.convolve(b ** 2, np.ones(480) / 480, "valid"))
    assert 20 * np.log10(env[len(a):].max() / env.max()) > -20          # the reverb carries on after 0.5 s
    assert 20 * np.log10(env[-2400:-960].max() / env.max()) < -60       # ... and has died away by the end


def test_ring_out_cap_and_sources_stop_at_duration():
    b, sr = soundlab.render(ROOM, ring_out=0.3)
    assert len(b) == int(0.8 * sr)
    dry = dict(ROOM, post=[])
    y, _ = soundlab.render(dry, ring_out=True)
    assert len(y) == int(0.5 * sr)                                       # no processing blocks: no tail


def test_export_keeps_the_whole_tail(tmp_path):
    info = soundlab.render_to_file(ROOM, tmp_path / "a.wav")
    assert info["seconds"] > 1.0                                         # was cut at 0.5 s before v0.22
    info = soundlab.render_to_file(ROOM, tmp_path / "b.wav", trim=False)
    assert info["seconds"] == pytest.approx(0.5)                         # --no-trim: exactly the recipe length


@pytest.mark.parametrize("name", ["hit", "metal_clang", "dog_growl", "foghorn"])
def test_ring_out_does_not_change_the_head(name):
    a, sr = soundlab.render(R / f"{name}.json")
    b, _ = soundlab.render(R / f"{name}.json", ring_out=True)
    h = len(a) - int(0.03 * sr)
    assert 10 * np.log10(np.sum((a[:h] - b[:h]) ** 2) / np.sum(a[:h] ** 2) + 1e-30) < -50


def test_every_ring_out_tail_decays_before_the_end():
    for p in sorted(R.glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("loop") or p.stem in ("car_passby", "engine", "engine_blips"):  # sources still going at the end
            continue
        # weather one-shots (thunder): the recorded take still has energy at the cut, and there is no
        # resonant tail after duration. Looping weather beds are already skipped by the loop key above.
        models = {layer.get("model") for layer in r.get("layers", [])}
        if "weather_stems" in models:
            continue
        y, sr = soundlab.render(p, ring_out=True)
        w = int(0.01 * sr)
        c = np.concatenate([[0], np.cumsum(y ** 2)])
        env = np.sqrt((c[w:] - c[:-w]) / w)
        assert 20 * np.log10(env[-int(0.07 * sr):-int(0.02 * sr)].max() / env.max() + 1e-15) < -55, p.stem


# ----------------------------------------------------------------------------- stereo
def test_decorrelator_is_allpass_and_short():
    h = st.decorrelator(SR)
    H = np.abs(np.fft.rfft(h))
    assert np.allclose(H, 1.0, atol=1e-9)
    assert np.sum(h[:int(0.025 * SR)] ** 2) / np.sum(h ** 2) > 0.99


@pytest.mark.parametrize("name", ["rain", "fire", "wind"])
def test_stereo_loop_width_bass_and_seam(name):
    cfg = json.loads((R / f"{name}.json").read_text())["stereo"]
    y, sr = soundlab.render_loop(R / f"{name}.json", 6, stereo=True)
    m, _ = soundlab.render_loop(R / f"{name}.json", 6)
    assert y.shape == (len(m), 2)
    assert st.correlation(y, sr, 500) == pytest.approx(1 - cfg["width"], abs=0.1)
    assert st.correlation(y, sr, 20, 100) > 0.97                         # low end in phase
    for ch in (0, 1):     # the step across the loop point is an ordinary step for that channel
        d = np.abs(np.diff(y[:, ch]))
        assert np.mean(d < abs(y[0, ch] - y[-1, ch])) < 0.995
    lr = 20 * np.log10(np.std(y[:, 0]) / np.std(y[:, 1]))
    assert abs(lr) < 2.5


def test_seeds_mode_keeps_the_mono_take_on_its_left_low_band():
    """The second take only enters above mono_below_hz; the first take is the low band on both sides."""
    x1 = np.random.default_rng(1).standard_normal(SR)
    x2 = np.random.default_rng(2).standard_normal(SR)
    y = st.combine(x1, x2, SR, 0.7, 150.0, circular=True)
    lo1, _ = st._split(x1, SR, 150.0, True)
    lo_l, _ = st._split(y[:, 0], SR, 150.0, True)
    D = np.fft.rfft(y[:, 0] - y[:, 1])
    f = np.fft.rfftfreq(SR, 1 / SR)
    assert np.max(np.abs(D[f < 100])) < 1e-6 * np.max(np.abs(D))          # identical below the crossover
    assert np.corrcoef(lo1, lo_l)[0, 1] > 0.98


def test_stereo_loudness_is_bs1770_channel_sum():
    x = 0.3 * np.random.default_rng(3).standard_normal(SR)
    assert momentary_max_lufs(np.stack([x, x], 1), SR) == pytest.approx(momentary_max_lufs(x, SR) + 3.01, abs=0.02)


def test_stereo_export_and_auto_rule(tmp_path):
    a = soundlab.render_to_file(R / "rain.json", tmp_path / "rain.wav", loop=3)
    b = soundlab.render_to_file(R / "rain.json", tmp_path / "rain_m.wav", loop=3, stereo=False)
    c = soundlab.render_to_file(R / "hit.json", tmp_path / "hit.wav")
    d = soundlab.render_to_file(R / "hit.json", tmp_path / "hit_s.wav", stereo=True)
    assert (a["channels"], b["channels"], c["channels"], d["channels"]) == (2, 1, 1, 2)
    with wave.open(a["wav"]) as w:
        assert (w.getnchannels(), w.getsampwidth()) == (2, 3)
    y, _ = read_audio(a["wav"], mono=False)
    assert y.shape[1] == 2 and a["finish"]["true_peak"] <= -1.0


def test_cli_stereo_and_mono_flags(tmp_path):
    for flag, ch in (("--mono", 1), (None, 2)):
        out = tmp_path / f"w{ch}.wav"
        cmd = [sys.executable, "-m", "soundlab", "render", str(R / "wind.json"), "-o", str(out), "--loop", "2"]
        r = subprocess.run(cmd + ([flag] if flag else []), capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        with wave.open(str(out)) as w:
            assert w.getnchannels() == ch


# ----------------------------------------------------------------------------- dither
def test_dither_is_deterministic_and_keeps_signal_below_one_lsb(tmp_path):
    t = np.arange(SR) / SR
    quiet = 0.4 / 32767 * np.sin(2 * np.pi * 440 * t)                    # under half an LSB
    assert not np.any(to_int16(quiet, dither=False))                      # plain rounding: silence
    q = to_int16(quiet).astype(float)
    assert np.corrcoef(q, quiet)[0, 1] > 0.1                              # dithered: the tone survives in the noise
    p1, p2 = write_wav(tmp_path / "a.wav", quiet, SR, 16), write_wav(tmp_path / "b.wav", quiet, SR, 16)
    assert p1.read_bytes() == p2.read_bytes()


def test_trim_and_fold_accept_stereo():
    y = np.zeros((SR, 2))
    y[:4800] = 0.5
    assert trim_tail(y, SR).shape[1] == 2 and len(trim_tail(y, SR)) < SR
