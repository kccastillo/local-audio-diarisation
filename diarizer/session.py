"""Session-dir layout for the transcript-review webapp.

A session dir groups together the artefacts a single pipeline run produces:
the model-canonical WAV, an Opus playback copy, the immutable original
transcript, and any later edit sidecars. The webapp reads all of these via
`session.json`, the manifest written here.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


SESSION_MANIFEST_VERSION = 1


def create_session_dir(base_output_dir: Path | str, source_path: Path | str) -> Path:
    """Create `<base_output_dir>/<source_basename>_<timestamp>/` and write `session.json`.

    The manifest references file *names* (not absolute paths) — the webapp
    resolves them relative to the session dir at read time. This keeps the
    session dir relocatable.
    """
    base = Path(base_output_dir)
    src = Path(source_path)
    stem = src.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    session_dir = base / f"{stem}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": SESSION_MANIFEST_VERSION,
        "source_basename": stem,
        "created": datetime.now().isoformat(timespec="seconds"),
        "audio_opus": "source.opus",
        "audio_wav": "source.wav",
        "transcript_original": "transcript.json",
        "edits": [],
    }
    (session_dir / "session.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return session_dir


def find_completed_session(base_output_dir: Path | str, stem: str) -> Path | None:
    """Return the lexicographically-newest completed session directory for *stem*, or None.

    A session directory is considered complete when it contains ALL of:
    ``session.json``, ``transcript.json``, and ``source.opus``.

    Notes:
    - A partial or crashed session (missing any of the three artefacts) does NOT
      count as complete; a subsequent run can safely overwrite it.
    - Matching is stem-based and extension-insensitive: ``foo.wav`` and ``foo.m4a``
      both have stem ``foo`` and will match the same set of session directories.
    - Returns ``None`` when ``base_output_dir`` does not exist or contains no
      qualifying directories.
    """
    base = Path(base_output_dir)
    if not base.exists():
        return None

    _TIMESTAMP_SUFFIX = re.compile(r"^\d{8}_\d{6}$")
    required_artefacts = {"session.json", "transcript.json", "source.opus"}

    candidates: list[Path] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        # Name must be exactly "<stem>_<timestamp>" where timestamp = YYYYMMDD_HHMMSS
        name = entry.name
        prefix = f"{stem}_"
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if not _TIMESTAMP_SUFFIX.match(suffix):
            continue
        # All three artefacts must be present.
        if all((entry / artefact).exists() for artefact in required_artefacts):
            candidates.append(entry)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def load_manifest(session_dir: Path | str) -> dict:
    p = Path(session_dir) / "session.json"
    return json.loads(p.read_text(encoding="utf-8"))


def append_edit(session_dir: Path | str, filename: str) -> dict:
    """Append a new edit-sidecar entry to `session.json`. Returns the updated manifest."""
    p = Path(session_dir) / "session.json"
    manifest = json.loads(p.read_text(encoding="utf-8"))
    manifest.setdefault("edits", []).append(
        {"filename": filename, "created": datetime.now().isoformat(timespec="seconds")}
    )
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
