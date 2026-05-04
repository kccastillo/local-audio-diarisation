"""Diarizer — speaker diarisation + transcription on faster-whisper + pyannote.audio 3.3.

Pipeline: adaptive preprocessing → voice-presence gate → diarisation (pyannote)
→ ASR (faster-whisper, large-v3-turbo, int8_float16) → attribution. Validation
gates between stages fail fast on cheap pre-checks and warn on quality.

The previous five-stage processor architecture (v1) was retired in PLAN 300;
the v1 baseline is recoverable via `git checkout v1-final`.
"""

from diarizer.config import Config, load_config
from diarizer.pipeline import Pipeline, TranscriptionResult

__all__ = ["Config", "load_config", "Pipeline", "TranscriptionResult"]
