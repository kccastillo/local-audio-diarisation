"""FastAPI app for the transcript-review webapp.

Loopback-only by construction — `serve` CLI binds to 127.0.0.1 and the
host-header middleware rejects anything else as a DNS-rebinding defence.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import math

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from diarizer.session import append_edit, load_manifest
from diarizer import minutes as minutes_mod


def _sanitise_json(obj):
    """Replace non-finite floats with None so Starlette's strict encoder accepts them.

    Pipeline measurements occasionally produce -Infinity (e.g. noise_floor_dbfs on
    digital silence). The pipeline writes them as JSON 'Infinity' tokens, which
    Python's json.loads reads back as Python `float('inf')` — but Starlette's
    JSONResponse uses strict mode (allow_nan=False) and rejects them on serialise.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitise_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise_json(v) for v in obj]
    return obj


# Keep in sync with diarizer/output.py:_format_timestamp (and the formatting loop
# in write_txt at lines 70-77). Step 6's byte-identity test enforces this.
def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_segments_as_txt(segments: list[dict]) -> str:
    """Mirror diarizer/output.py:write_txt formatting EXACTLY (lines 70-77).

    Per-segment line shape: prefix joins '[<speaker>]' and 'hh:mm:ss' (when present),
    then 'f"{prefix} {seg.text}".strip() if prefix else seg.text'.
    """
    lines = []
    for seg in segments:
        prefix_parts = []
        speaker = seg.get("speaker") if isinstance(seg, dict) else getattr(seg, "speaker", None)
        if speaker:
            prefix_parts.append(f"[{speaker}]")
        # webapp default: timestamps always on (matches the existing pipeline default
        # for transcripts that include both speakers and timestamps).
        start = seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0)
        prefix_parts.append(_format_timestamp(start))
        prefix = " ".join(prefix_parts)
        text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")
        lines.append(f"{prefix} {text}".strip() if prefix else text)
    return "\n".join(lines) + ("\n" if lines else "")


STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}


class HostHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        host = request.headers.get("host", "")
        host_only = host.split(":", 1)[0].lower()
        if host_only and host_only not in ALLOWED_HOSTS:
            return JSONResponse({"detail": "host header rejected"}, status_code=421)
        return await call_next(request)


def _resolve_session_dir(app: FastAPI) -> Path:
    d = app.state.session_dir
    if not isinstance(d, Path):
        d = Path(d)
    return d


def create_app(session_dir: Path | str) -> FastAPI:
    """Create the FastAPI app bound to a specific session directory.

    The session dir must already contain `session.json`, `source.opus`, and
    `transcript.json`; the CLI validates this before calling create_app.
    """
    session_dir = Path(session_dir)
    app = FastAPI(title="Diarizer transcript-review", docs_url=None, redoc_url=None)
    app.state.session_dir = session_dir
    app.add_middleware(HostHeaderMiddleware)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def root():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/session")
    async def get_session():
        d = _resolve_session_dir(app)
        manifest = load_manifest(d)
        # Refresh the edits list from disk in case the operator hand-dropped files
        edits_on_disk = sorted(p.name for p in d.glob("transcript_edit_*.json"))
        return {"manifest": manifest, "edits_on_disk": edits_on_disk}

    @app.get("/api/audio")
    async def get_audio(request: Request):
        d = _resolve_session_dir(app)
        audio_path = d / "source.opus"
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="source.opus missing")
        file_size = audio_path.stat().st_size
        range_header = request.headers.get("range")
        if range_header:
            # parse "bytes=START-END"
            try:
                units, rng = range_header.split("=", 1)
                if units.strip().lower() != "bytes":
                    raise ValueError
                start_s, end_s = rng.split("-", 1)
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else file_size - 1
            except ValueError:
                raise HTTPException(status_code=416, detail="bad range")
            end = min(end, file_size - 1)
            length = end - start + 1
            with open(audio_path, "rb") as f:
                f.seek(start)
                data = f.read(length)
            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(length),
                "Content-Type": "audio/ogg",
            }
            return Response(content=data, status_code=206, headers=headers)
        return FileResponse(audio_path, media_type="audio/ogg")

    @app.get("/api/transcript")
    async def get_transcript():
        d = _resolve_session_dir(app)
        p = d / "transcript.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="transcript.json missing")
        return JSONResponse(_sanitise_json(json.loads(p.read_text(encoding="utf-8"))))

    @app.get("/api/transcript/edit/{filename}")
    async def get_edit(filename: str):
        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="invalid filename")
        d = _resolve_session_dir(app)
        p = d / filename
        if not p.exists() or not filename.startswith("transcript_edit_"):
            raise HTTPException(status_code=404, detail="edit not found")
        return JSONResponse(_sanitise_json(json.loads(p.read_text(encoding="utf-8"))))

    @app.post("/api/transcript/save")
    async def save_transcript(payload: dict):
        d = _resolve_session_dir(app)
        candidate, wrapped = _save_numbered_snapshot(d, "transcript_edit_", payload)
        append_edit(d, candidate.name)
        return {"filename": candidate.name, "wrapped": wrapped}

    @app.get("/api/waveform")
    async def get_waveform():
        d = _resolve_session_dir(app)
        peaks_path = d / "waveform_peaks.json"
        if peaks_path.exists():
            return JSONResponse(json.loads(peaks_path.read_text(encoding="utf-8")))
        # Lazy-generate from source.opus if available.
        opus_path = d / "source.opus"
        if not opus_path.exists():
            raise HTTPException(
                status_code=404,
                detail="source.opus missing — cannot generate waveform peaks",
            )
        from diarizer.webapp.peaks import compute_peaks
        data = compute_peaks(opus_path)
        try:
            peaks_path.write_text(json.dumps(data), encoding="utf-8")
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Lazily generated waveform_peaks.json for %s", d
            )
        except OSError:
            # Read-only session dir; serve in-memory result anyway.
            pass
        return JSONResponse(data)

    @app.get("/api/transcript/export.txt")
    async def export_txt_get(from_: Optional[str] = None):
        d = _resolve_session_dir(app)
        if from_:
            if "/" in from_ or "\\" in from_ or ".." in from_:
                raise HTTPException(status_code=400, detail="invalid filename")
            src = d / from_
            if not src.exists() or not from_.startswith("transcript_edit_"):
                raise HTTPException(status_code=404, detail="edit not found")
        else:
            edits = sorted(d.glob("transcript_edit_*.json"))
            src = edits[-1] if edits else (d / "transcript.json")
        payload = json.loads(src.read_text(encoding="utf-8"))
        body = _format_segments_as_txt(payload.get("segments", []))
        manifest = load_manifest(d)
        filename = f"{manifest.get('source_basename', 'export')}_export_{datetime.now().strftime('%Y%m%d')}.txt"
        return PlainTextResponse(
            body,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/transcript/export.txt")
    async def export_txt_post(payload: dict):
        d = _resolve_session_dir(app)
        body = _format_segments_as_txt(payload.get("segments", []))
        manifest = load_manifest(d)
        filename = f"{manifest.get('source_basename', 'export')}_export_{datetime.now().strftime('%Y%m%d')}.txt"
        return PlainTextResponse(
            body,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/transcript/diff")
    async def get_diff(against: str = "original", from_: Optional[str] = None):
        d = _resolve_session_dir(app)
        original = json.loads((d / "transcript.json").read_text(encoding="utf-8"))
        if from_:
            edit_path = d / from_
        else:
            edits = sorted(d.glob("transcript_edit_*.json"))
            if not edits:
                return {"segments": [], "note": "no edits yet"}
            edit_path = edits[-1]
        edited = json.loads(edit_path.read_text(encoding="utf-8"))
        return {"segments": _diff_segments(original.get("segments", []), edited.get("segments", []))}

    @app.get("/api/minutes")
    async def get_minutes():
        d = _resolve_session_dir(app)
        p = d / "minutes.json"
        if not p.exists():
            raise HTTPException(status_code=404, detail="minutes.json missing")
        return JSONResponse(_sanitise_json(json.loads(p.read_text(encoding="utf-8"))))

    @app.get("/api/minutes/edit/{filename}")
    async def get_minutes_edit(filename: str):
        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="invalid filename")
        d = _resolve_session_dir(app)
        p = d / filename
        if not p.exists() or not filename.startswith("minutes_edit_"):
            raise HTTPException(status_code=404, detail="minutes edit not found")
        return JSONResponse(_sanitise_json(json.loads(p.read_text(encoding="utf-8"))))

    @app.post("/api/minutes/generate")
    async def generate_minutes(payload: dict):
        d = _resolve_session_dir(app)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "ANTHROPIC_API_KEY is not set. Minutes generation is opt-in "
                    "and requires a key from console.anthropic.com."
                ),
            )
        transcript_path = d / "transcript.json"
        if not transcript_path.exists():
            raise HTTPException(status_code=404, detail="transcript.json missing")
        transcript = json.loads(transcript_path.read_text(encoding="utf-8")).get("segments", [])

        peaks_path = d / "waveform_peaks.json"
        peaks = (
            json.loads(peaks_path.read_text(encoding="utf-8"))
            if peaks_path.exists()
            else {"bins": 0, "duration_s": 0, "peaks": []}
        )

        attendees = payload.get("attendees") or []
        model = payload.get("model") or "claude-sonnet-4-6"
        top_n = int(payload.get("top_n_emotive") or 15)

        try:
            result = minutes_mod.generate(
                transcript, attendees, model, peaks, top_n_emotive=top_n
            )
        except minutes_mod.MissingAPIKeyError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except minutes_mod.SchemaError as e:
            raise HTTPException(status_code=502, detail=f"LLM response invalid: {e}")
        except minutes_mod.APIError as e:
            raise HTTPException(status_code=502, detail=f"Anthropic API error: {e}")

        result["generated_at"] = datetime.now().isoformat(timespec="seconds")
        return JSONResponse(_sanitise_json(result))

    @app.post("/api/minutes/save")
    async def save_minutes(payload: dict):
        d = _resolve_session_dir(app)
        candidate, wrapped = _save_numbered_snapshot(d, "minutes_edit_", payload)
        # Mirror canonical minutes.json to the latest save.
        canonical = d / "minutes.json"
        try:
            canonical.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass
        return {"filename": candidate.name, "wrapped": wrapped}

    return app


def _save_numbered_snapshot(d: Path, prefix: str, payload: dict) -> tuple[Path, bool]:
    """Write `payload` as JSON to the next-numbered snapshot file under `d`.

    Naming: `<prefix><YYYYMMDD>_<NN>.json` with NN per-day, zero-padded to 2 digits.
    O_EXCL atomic create with retry on collision; wraps to _00 on overflow above
    99 with deliberate overwrite (matches transcript-edit naming convention).

    Returns (file_path, wrapped). Raises HTTPException(507) if retries exhaust.
    """
    today = datetime.now().strftime("%Y%m%d")
    existing = sorted(d.glob(f"{prefix}{today}_*.json"))
    existing_nns: list[int] = []
    for p in existing:
        tail = p.stem.rsplit("_", 1)[-1]
        if tail.isdigit() and len(tail) == 2:
            existing_nns.append(int(tail))
    next_nn = (max(existing_nns) + 1) if existing_nns else 0
    wrapped = False
    if next_nn > 99:
        next_nn = 0
        wrapped = True
    attempts = 0
    while attempts < 100:
        candidate = d / f"{prefix}{today}_{next_nn:02d}.json"
        try:
            fd = os.open(
                str(candidate),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError:
            if wrapped:
                candidate.unlink()
                fd = os.open(
                    str(candidate),
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o644,
                )
            else:
                next_nn += 1
                if next_nn > 99:
                    next_nn = 0
                    wrapped = True
                attempts += 1
                continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            candidate.unlink(missing_ok=True)
            raise
        return candidate, wrapped
    raise HTTPException(status_code=507, detail="exhausted save retries")


def _diff_segments(orig: list[dict], edit: list[dict]) -> list[dict]:
    """Per-index segment diff — speaker change + text change classification.

    Assumes segment order/count is preserved (true for edits in scope: speaker
    labels + text only). For mismatched lengths, surplus segments are reported.
    """
    out = []
    n = max(len(orig), len(edit))
    for i in range(n):
        o = orig[i] if i < len(orig) else None
        e = edit[i] if i < len(edit) else None
        if o is None:
            out.append({"index": i, "kind": "added", "edit": e})
            continue
        if e is None:
            out.append({"index": i, "kind": "removed", "original": o})
            continue
        speaker_changed = o.get("speaker") != e.get("speaker")
        text_changed = o.get("text") != e.get("text")
        if not speaker_changed and not text_changed:
            out.append({"index": i, "kind": "unchanged"})
            continue
        ratio = SequenceMatcher(None, o.get("text", ""), e.get("text", "")).ratio() if text_changed else 1.0
        out.append({
            "index": i,
            "kind": "changed",
            "speaker_changed": speaker_changed,
            "text_changed": text_changed,
            "original": {"speaker": o.get("speaker"), "text": o.get("text")},
            "edit": {"speaker": e.get("speaker"), "text": e.get("text")},
            "text_ratio": ratio,
        })
    return out


# Default app for `uvicorn diarizer.webapp.app:app` — the CLI overrides via
# DIARIZER_SESSION_DIR env var. This makes `serve` a thin wrapper.
def _default_session_dir() -> Path:
    raw = os.environ.get("DIARIZER_SESSION_DIR")
    if raw:
        return Path(raw)
    raise RuntimeError(
        "DIARIZER_SESSION_DIR not set. Use `diarizer serve <session-dir>` instead of importing app directly."
    )


def app_factory() -> FastAPI:
    return create_app(_default_session_dir())
