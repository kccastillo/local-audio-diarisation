"""Test fixtures for the transcript-review webapp."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest


def _generate_silent_opus(out_path: Path) -> None:
    """1-second silent Opus at 16 kHz mono via ffmpeg lavfi anullsrc."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", "1",
        "-c:a", "libopus", "-b:a", "16k",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg silent-opus generation failed:\n{result.stderr}")


@pytest.fixture
def stub_session(tmp_path: Path) -> Path:
    """Materialise a minimal session dir with session.json + transcript.json + source.opus.

    The smoke test exercises webapp behaviour, NOT real pipeline output —
    that path is covered by existing pipeline tests.
    """
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not on PATH")

    session = tmp_path / "stub_session_20260506_010101"
    session.mkdir()

    _generate_silent_opus(session / "source.opus")

    transcript = {
        "source_path": "stub.mp4",
        "model_name": "stub",
        "language": "en",
        "duration_seconds": 1.0,
        "speaker_count": 2,
        "segments": [
            {"start": 0.0, "end": 0.4, "text": "hello world", "speaker": "SPEAKER_00", "words": []},
            {"start": 0.4, "end": 0.8, "text": "second segment", "speaker": "SPEAKER_01", "words": []},
            {"start": 0.8, "end": 1.0, "text": "third", "speaker": "SPEAKER_00", "words": []},
        ],
    }
    (session / "transcript.json").write_text(
        json.dumps(transcript, indent=2), encoding="utf-8"
    )

    manifest = {
        "version": 1,
        "source_basename": "stub",
        "created": datetime.now().isoformat(timespec="seconds"),
        "audio_opus": "source.opus",
        "audio_wav": "source.wav",
        "transcript_original": "transcript.json",
        "edits": [],
    }
    (session / "session.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return session
