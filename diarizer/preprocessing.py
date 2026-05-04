"""v2 audio preprocessing — measure first, then decide.

Two public functions:

  measure(path) -> AudioMeasurement
      Fast librosa-based metrics: duration, sample rate, channels, RMS,
      peak, noise floor, voiced fraction, spectral centroid. Substrate for
      validation gates and for the adaptive preprocessing decision.

  preprocess(input, output, config, measurement) -> Path
      FFmpeg invocation with filter selection driven by the measurement.
      Always decodes to 16 kHz mono PCM. Loudnorm only when RMS is below
      threshold. Denoise default OFF (denoising hurts Whisper WER on
      already-clean audio per RESEARCH 230).

The pipeline orchestrator (v2.pipeline) is responsible for re-measuring
the preprocessed output and feeding the comparison to a damage gate.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from diarizer.config import PreprocessingConfig

logger = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioMeasurement:
    """Fast metrics about an audio file. Computed once at ingest, reused by gates."""
    duration_s: float
    sample_rate: int
    channels: int
    rms_dbfs: float
    peak_dbfs: float
    noise_floor_dbfs: float
    speech_floor_dbfs: float
    voiced_fraction: float
    spectral_centroid_hz: float

    @property
    def estimated_snr_db(self) -> float:
        """Coarse SNR estimate. Inf when noise floor is digital silence."""
        if self.noise_floor_dbfs == float("-inf"):
            return float("inf")
        return self.speech_floor_dbfs - self.noise_floor_dbfs

    def to_dict(self) -> dict:
        return {
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "rms_dbfs": self.rms_dbfs,
            "peak_dbfs": self.peak_dbfs,
            "noise_floor_dbfs": self.noise_floor_dbfs,
            "speech_floor_dbfs": self.speech_floor_dbfs,
            "voiced_fraction": self.voiced_fraction,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "estimated_snr_db": self.estimated_snr_db,
        }


def _safe_dbfs(linear: float) -> float:
    if linear <= 0 or not math.isfinite(linear):
        return float("-inf")
    return 20.0 * math.log10(linear)


def measure(path: Path | str) -> AudioMeasurement:
    """Decode and measure an audio file. Lazy-imports librosa for fast cold start.

    Cost: a few hundred ms per minute of audio on CPU. Cheap relative to the ASR
    stage; the orchestrator runs this once at ingest.
    """
    import librosa  # heavy import; defer until we know we'll measure

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio input not found: {path}")

    audio, sr = librosa.load(str(path), sr=None, mono=False)
    if audio.ndim == 1:
        channels = 1
        mono = audio
    else:
        channels = audio.shape[0]
        mono = audio.mean(axis=0)

    duration_s = float(len(mono)) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(mono ** 2))) if len(mono) else 0.0
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0

    frame_len = int(sr * 0.025) if sr else 0
    hop = int(sr * 0.010) if sr else 0
    if frame_len < 1 or hop < 1:
        frames = np.array([rms]) if rms else np.array([0.0])
    else:
        frames = librosa.feature.rms(y=mono, frame_length=frame_len, hop_length=hop)[0]

    sorted_frames = np.sort(frames)
    n = max(1, int(len(sorted_frames) * 0.1))
    noise_floor_rms = float(np.mean(sorted_frames[:n]))
    speech_floor_rms = float(np.mean(sorted_frames[-n:]))

    threshold = noise_floor_rms * (10 ** (6 / 20))
    voiced_fraction = float(np.mean(frames > threshold)) if len(frames) else 0.0

    centroid_arr = librosa.feature.spectral_centroid(y=mono, sr=sr) if sr else np.array([[0.0]])
    centroid = float(np.mean(centroid_arr))

    return AudioMeasurement(
        duration_s=duration_s,
        sample_rate=int(sr),
        channels=int(channels),
        rms_dbfs=_safe_dbfs(rms),
        peak_dbfs=_safe_dbfs(peak),
        noise_floor_dbfs=_safe_dbfs(noise_floor_rms),
        speech_floor_dbfs=_safe_dbfs(speech_floor_rms),
        voiced_fraction=voiced_fraction,
        spectral_centroid_hz=centroid,
    )


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _select_filters(config: PreprocessingConfig, m: Optional[AudioMeasurement]) -> list[str]:
    """Decide which FFmpeg `-af` filters to apply, driven by measurement.

    Returns a list of filter-graph fragments to comma-join into a single -af.
    Empty list = no -af flag, raw decode/resample only.
    """
    filters: list[str] = []
    if not config.enabled:
        return filters

    # Loudness normalisation: apply only when input is too quiet (RMS below threshold).
    # Threshold default -32 dBFS — File 2 in RESEARCH 220 (-41 dBFS) gets normalised;
    # File 1 (-26.7) and Teams (-20.9) do not.
    if config.normalise:
        if m is None or m.rms_dbfs < -32.0:
            filters.append("loudnorm=I=-23:LRA=7:TP=-2")

    # Denoise: default OFF (config.denoise default False per RESEARCH 230).
    # Even when enabled, only apply if noise floor is high AND SNR low —
    # don't denoise audio that's already clean.
    if config.denoise:
        if m is None or (m.noise_floor_dbfs > -40.0 and m.estimated_snr_db < 20.0):
            filters.append("afftdn=nf=-25")

    return filters


def preprocess(
    input_path: Path | str,
    output_path: Path | str,
    config: PreprocessingConfig,
    measurement: Optional[AudioMeasurement] = None,
) -> Path:
    """Decode + adaptive-filter to 16 kHz mono PCM WAV.

    Always decodes and ensures channels/sample-rate are at Whisper's preferred
    format. Filter selection is driven by `measurement`; pass None to apply
    config flags unconditionally (used for unit tests / forced overrides).

    Raises FFmpegError on missing ffmpeg or non-zero exit.
    """
    if not _ffmpeg_available():
        raise FFmpegError("ffmpeg not found on PATH; preprocessing requires ffmpeg.")

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio input not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Spectral-centroid sanity warning — informational, doesn't block.
    if measurement is not None and measurement.spectral_centroid_hz > 6000.0:
        logger.warning(
            "Input spectral centroid is %.0f Hz (typical speech ~1500-3000 Hz). "
            "Audio has likely been through aggressive denoise upstream; "
            "transcription quality may be degraded.",
            measurement.spectral_centroid_hz,
        )

    filters = _select_filters(config, measurement)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", str(config.target_sample_rate),
    ]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += [str(output_path)]

    logger.info("FFmpeg preprocess: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
        )

    return output_path


# Back-compat alias for existing tests that still reference preprocess_audio.
preprocess_audio = preprocess
