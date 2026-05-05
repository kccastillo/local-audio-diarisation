"""Build a webapp session dir from an existing TXT transcript + source audio.

Used when the original pipeline run predates the session-dir convention.
Parses lines of form `[SPEAKER_NN] hh:mm:ss text...` into segments. End times
are inferred from the next segment's start, with a 5-second tail on the last.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

LINE_RE = re.compile(r"^\[(?P<speaker>[^\]]+)\]\s+(?P<ts>\d{1,2}:\d{2}:\d{2})\s+(?P<text>.*)$")


def parse_ts(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_txt(path: Path) -> list[dict]:
    segs: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        segs.append({
            "speaker": m.group("speaker"),
            "start": parse_ts(m.group("ts")),
            "text": m.group("text"),
            "words": [],
        })
    # Fill in `end` from next start; last segment gets +5s.
    for i, s in enumerate(segs):
        s["end"] = segs[i + 1]["start"] if i + 1 < len(segs) else s["start"] + 5
        if s["end"] <= s["start"]:
            s["end"] = s["start"] + 0.5
    return segs


def main(txt_path: str, audio_path: str, out_root: str = "output") -> None:
    txt = Path(txt_path)
    audio = Path(audio_path)
    out = Path(out_root)
    if not txt.exists():
        raise FileNotFoundError(txt)
    if not audio.exists():
        raise FileNotFoundError(audio)

    segments = parse_txt(txt)
    if not segments:
        raise ValueError(f"No segments parsed from {txt}")

    stem = txt.stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = out / f"{stem}_{ts}"
    session.mkdir(parents=True, exist_ok=True)

    # Encode source.opus
    cmd = [
        "ffmpeg", "-y", "-i", str(audio),
        "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-b:a", "16k", "-vbr", "on", "-vn",
        str(session / "source.opus"),
    ]
    print("Encoding Opus…")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr}")

    # transcript.json
    transcript = {
        "source_path": str(audio),
        "model_name": "imported-from-txt",
        "language": "en",
        "duration_seconds": segments[-1]["end"],
        "speaker_count": len({s["speaker"] for s in segments}),
        "segments": segments,
    }
    (session / "transcript.json").write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")

    # session.json
    manifest = {
        "version": 1,
        "source_basename": stem,
        "created": datetime.now().isoformat(timespec="seconds"),
        "audio_opus": "source.opus",
        "audio_wav": "source.wav",
        "transcript_original": "transcript.json",
        "edits": [],
    }
    (session / "session.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(session)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python scripts/build_session_from_txt.py <txt> <audio>", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
