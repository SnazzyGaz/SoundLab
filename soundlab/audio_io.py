"""WAV/MP3 file I/O (scipy for WAV; ffmpeg, if installed, for MP3 and other formats)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def write_wav(path: str | Path, x: np.ndarray, sr: int, bits: int = 24) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(np.asarray(x, float), -1.0, 1.0)
    if bits == 16:
        wavfile.write(path, sr, (y * 32767).astype(np.int16))
    elif bits == 32:
        wavfile.write(path, sr, y.astype(np.float32))
    else:  # 24-bit packed as int32 container is not portable; write 32-bit int scaled to 24-bit range
        wavfile.write(path, sr, (y * (2 ** 31 - 1)).astype(np.int32))
    return path


def read_audio(path: str | Path, mono: bool = True) -> tuple[np.ndarray, int]:
    path = Path(path)
    if path.suffix.lower() != ".wav":
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is needed to read non-WAV files")
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-"],
                             capture_output=True, check=True)
        sr = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                 "stream=sample_rate", "-of", "csv=p=0", str(path)],
                                capture_output=True, text=True, check=True).stdout.strip())
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
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-",
                    "-codec:a", "libmp3lame", "-b:a", bitrate, str(path)], input=pcm, check=True)
    return path
