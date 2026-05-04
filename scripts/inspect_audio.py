"""Inspect audio files for characteristics relevant to a transcription pipeline.

Produces a JSON report per file. Run with venv python.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa


def _safe_db(x: float) -> float:
    if x <= 0 or not math.isfinite(x):
        return float("-inf")
    return 20.0 * math.log10(x)


def inspect(path: Path) -> dict:
    info = sf.info(str(path)) if path.suffix.lower() in {".wav", ".flac", ".ogg"} else None
    # m4a/aac route through librosa (which uses audioread/soundfile/ffmpeg as available)
    audio, sr = librosa.load(str(path), sr=None, mono=False)
    if audio.ndim == 1:
        channels = 1
        mono = audio
    else:
        channels = audio.shape[0]
        mono = audio.mean(axis=0)

    duration_s = float(len(mono)) / sr
    rms = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.max(np.abs(mono)))

    # Crude noise-floor estimate: median of bottom 10% of frame energies
    frame_len = int(sr * 0.025)  # 25 ms
    hop = int(sr * 0.010)         # 10 ms
    if frame_len < 1 or hop < 1:
        frames = np.array([rms])
    else:
        rms_frames = librosa.feature.rms(y=mono, frame_length=frame_len, hop_length=hop)[0]
        frames = rms_frames

    sorted_frames = np.sort(frames)
    noise_floor_rms = float(np.mean(sorted_frames[: max(1, int(len(sorted_frames) * 0.1))]))
    speech_floor_rms = float(np.mean(sorted_frames[-max(1, int(len(sorted_frames) * 0.1)):]))
    snr_estimate_db = _safe_db(speech_floor_rms) - _safe_db(noise_floor_rms)

    # Voice activity ratio: fraction of frames above noise_floor + 6 dB
    threshold = noise_floor_rms * (10 ** (6 / 20))
    voiced_fraction = float(np.mean(frames > threshold))

    # Spectral centroid (mean) — high values = thin/high-pass; low values = warm/bass-heavy
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=mono, sr=sr)))

    # Dynamic range: peak vs RMS (in dB)
    crest_db = _safe_db(peak) - _safe_db(rms)

    # Clipping check: fraction of samples within 0.5 dB of 0 dBFS
    clip_threshold = 10 ** (-0.5 / 20)
    clip_fraction = float(np.mean(np.abs(mono) >= clip_threshold))

    return {
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "duration_seconds": duration_s,
        "duration_human": f"{int(duration_s // 60)}:{int(duration_s % 60):02d}",
        "sample_rate_hz": sr,
        "channels": channels,
        "rms": rms,
        "rms_dbfs": _safe_db(rms),
        "peak": peak,
        "peak_dbfs": _safe_db(peak),
        "crest_factor_db": crest_db,
        "noise_floor_rms": noise_floor_rms,
        "noise_floor_dbfs": _safe_db(noise_floor_rms),
        "speech_floor_rms": speech_floor_rms,
        "speech_floor_dbfs": _safe_db(speech_floor_rms),
        "estimated_snr_db": snr_estimate_db,
        "voiced_fraction": voiced_fraction,
        "spectral_centroid_hz": centroid,
        "clip_fraction": clip_fraction,
        "soundfile_info": str(info) if info else None,
    }


def main(paths: list[str]) -> int:
    if not paths:
        # Default: glob the recordings folder for any audio.
        paths = [str(p) for p in Path("recordings").glob("meeting*.m4a")]
        print(f"globbed {len(paths)} files", file=sys.stderr)

    out = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            continue
        try:
            report = inspect(path)
        except Exception as e:  # noqa: BLE001
            print(f"failed to inspect {path}: {e}", file=sys.stderr)
            report = {"file": str(path), "error": str(e)}
        out.append(report)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
