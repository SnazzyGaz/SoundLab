"""WAV/MP3 file I/O (scipy for WAV; ffmpeg, if installed, for MP3 and other formats)."""
from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def write_wav(path: str | Path, x: np.ndarray, sr: int, bits: int = 24) -> Path:
    """WAV, mono (n,) or multichannel (n, channels): 16-bit (TPDF-dithered, seeded), 24-bit integer PCM or
    32-bit float."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(x, float)
    if y.ndim > 2 or (y.ndim == 2 and y.shape[1] > 8):
        raise ValueError(f"audio must be (n,) or (n, channels), got {y.shape}")
    y = np.clip(y, -1.0, 1.0)
    ch = 1 if y.ndim == 1 else y.shape[1]
    if bits == 16:
        wavfile.write(path, sr, to_int16(y))
    elif bits == 32:
        wavfile.write(path, sr, y.astype(np.float32))
    elif bits == 24:
        # real packed 24-bit PCM (3 bytes per sample, little endian): the low 3 bytes of each int32. Until
        # v0.21.1 this wrote 32-bit integer PCM, which some game engines and older tools reject.
        q = np.round(y * 8388607).astype("<i4")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(3)
            w.setframerate(int(sr))
            w.writeframes(q.view(np.uint8).reshape(-1, 4)[:, :3].tobytes())
    else:
        raise ValueError(f"bits must be 16, 24 or 32 (float), not {bits}")
    return path


def to_int16(y: np.ndarray, dither: bool = True) -> np.ndarray:
    """Float in [-1, 1] -> int16. dither: TPDF dither of +-1 LSB, so quiet fades and reverb tails decay into a
    steady noise floor at about -96 dBFS instead of truncation distortion. The dither noise is seeded from the
    signal's length, so the same render always gives the same file."""
    q = np.asarray(y, float) * 32767.0
    if dither:
        rng = np.random.default_rng([0x5D, q.size])
        q = q + rng.uniform(-0.5, 0.5, q.shape) + rng.uniform(-0.5, 0.5, q.shape)
    return np.clip(np.round(q), -32768, 32767).astype(np.int16)


def read_audio(path: str | Path, mono: bool = True) -> tuple[np.ndarray, int]:
    path = Path(path)
    if path.suffix.lower() != ".wav":
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is needed to read non-WAV files")
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-"],
                             capture_output=True, check=True)
        sr = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                 "stream=sample_rate", "-of", "csv=p=0", str(path)],
                                capture_output=True, text=True, check=True).stdout.split()[0].strip(", "))
        return np.frombuffer(out.stdout, np.float32).astype(float), sr
    sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x / float(np.iinfo(x.dtype).max)
    elif x.dtype.kind == "u":
        x = (x.astype(float) - 128) / 128
    x = x.astype(float)
    if x.ndim > 1 and mono:
        x = x.mean(axis=1)
    return x, sr


def write_mp3(path: str | Path, x: np.ndarray, sr: int, bitrate: str = "320k") -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found; MP3 export needs it")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, float)
    ch = 1 if x.ndim == 1 else x.shape[1]
    pcm = to_int16(np.clip(x, -1, 1)).astype("<i2").tobytes()       # (n, ch) C order = interleaved
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", str(ch), "-i", "-",
                    "-codec:a", "libmp3lame", "-b:a", bitrate, str(path)], input=pcm, check=True)
    return path
