"""v2 output formatters — TXT, JSON, SRT, plus Opus playback artefact."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from diarizer.config import OutputConfig
from diarizer.pipeline import TranscriptionResult


class OpusExportError(RuntimeError):
    pass


def write_opus(source_audio_path: Path | str, out_path: Path | str) -> Path:
    """Encode source audio (or video) to 16 kHz mono Opus/OGG at ~16 kbps VBR.

    This is the webapp's playback artefact — small, browser-native, lossy.
    The model-canonical input remains the 16 kHz WAV from `preprocessing.py`.
    """
    src = Path(source_audio_path)
    out = Path(out_path)
    if not src.exists():
        raise FileNotFoundError(f"Source audio not found: {src}")
    if shutil.which("ffmpeg") is None:
        raise OpusExportError("ffmpeg not found on PATH; Opus export requires ffmpeg.")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", "16k", "-vbr", "on",
        "-vn",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise OpusExportError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}")
    return out


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_txt(result: TranscriptionResult, path: Path | str, config: OutputConfig) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for seg in result.segments:
        prefix_parts = []
        if config.include_speakers and seg.speaker:
            prefix_parts.append(f"[{seg.speaker}]")
        if config.include_timestamps:
            prefix_parts.append(_format_timestamp(seg.start))
        prefix = " ".join(prefix_parts)
        lines.append(f"{prefix} {seg.text}".strip() if prefix else seg.text)
    body = "\n".join(lines) + ("\n" if lines else "")
    # Explicit newline="\n" — without this, Path.write_text on Windows
    # translates LF to CRLF, breaking byte-equivalence with the webapp's
    # /api/transcript/export.txt and producing platform-dependent output.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    return path


def write_json(result: TranscriptionResult, path: Path | str, config: OutputConfig) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_srt(result: TranscriptionResult, path: Path | str, config: OutputConfig) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for idx, seg in enumerate(result.segments, start=1):
        start = _format_srt_timestamp(seg.start)
        end = _format_srt_timestamp(seg.end)
        text = seg.text
        if config.include_speakers and seg.speaker:
            text = f"[{seg.speaker}] {text}"
        blocks.append(f"{idx}\n{start} --> {end}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


_WRITERS = {
    "txt": write_txt,
    "json": write_json,
    "srt": write_srt,
}


def write_output(result: TranscriptionResult, path: Path | str, config: OutputConfig) -> Path:
    fmt = config.format.lower()
    if fmt not in _WRITERS:
        raise ValueError(f"Unsupported output format: {fmt}. Supported: {sorted(_WRITERS)}")
    return _WRITERS[fmt](result, path, config)
