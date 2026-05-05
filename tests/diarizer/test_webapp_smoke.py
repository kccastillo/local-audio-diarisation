"""Smoke tests for diarizer.webapp.app.

Covers the contract spelled out in the PLAN's Step 7 acceptance line:
- /api/session returns valid JSON
- /api/transcript returns the original
- /api/audio honours Range: bytes=0-99 → 206 with 100-byte body
- forged Host header → rejected (421/403)
- POST /api/transcript/save writes a versioned sidecar matching the pattern
- O_EXCL behaviour: pre-create _NN, save picks _NN+1
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest

from diarizer.webapp.app import create_app


@pytest.fixture
def client(stub_session: Path):
    app = create_app(stub_session)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1")


@pytest.mark.asyncio
async def test_get_session_returns_manifest(client, stub_session):
    async with client as c:
        r = await c.get("/api/session")
    assert r.status_code == 200
    body = r.json()
    assert body["manifest"]["version"] == 1
    assert body["manifest"]["source_basename"] == "stub"
    assert isinstance(body["edits_on_disk"], list)


@pytest.mark.asyncio
async def test_get_transcript_returns_original(client):
    async with client as c:
        r = await c.get("/api/transcript")
    assert r.status_code == 200
    body = r.json()
    assert len(body["segments"]) == 3
    assert body["segments"][0]["text"] == "hello world"


@pytest.mark.asyncio
async def test_audio_range_returns_206(client, stub_session):
    async with client as c:
        r = await c.get("/api/audio", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206, r.text
    assert len(r.content) == 100
    assert "Content-Range" in r.headers


@pytest.mark.asyncio
async def test_audio_no_range_returns_full(client):
    async with client as c:
        r = await c.get("/api/audio")
    assert r.status_code == 200
    assert len(r.content) > 100  # 1 sec of opus is at least a few hundred bytes


@pytest.mark.asyncio
async def test_forged_host_rejected(client):
    async with client as c:
        r = await c.get("/api/session", headers={"Host": "evil.example"})
    assert r.status_code in (403, 421), r.text


@pytest.mark.asyncio
async def test_save_creates_first_sidecar(client, stub_session):
    payload = {"segments": [{"start": 0.0, "end": 1.0, "text": "edited", "speaker": "Alice"}]}
    async with client as c:
        r = await c.post("/api/transcript/save", json=payload)
    assert r.status_code == 200, r.text
    j = r.json()
    today = datetime.now().strftime("%Y%m%d")
    assert j["filename"] == f"transcript_edit_{today}_00.json"
    assert j["wrapped"] is False
    saved = stub_session / j["filename"]
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8"))["segments"][0]["text"] == "edited"


@pytest.mark.asyncio
async def test_save_o_excl_picks_next_nn(stub_session):
    """If _05 exists, the next save must produce _06 — O_EXCL behaviour."""
    today = datetime.now().strftime("%Y%m%d")
    pre = stub_session / f"transcript_edit_{today}_05.json"
    pre.write_text('{"segments": []}', encoding="utf-8")

    app = create_app(stub_session)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
        r = await c.post("/api/transcript/save", json={"segments": []})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["filename"] == f"transcript_edit_{today}_06.json"


@pytest.mark.asyncio
async def test_root_serves_html(client):
    async with client as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text or "<html" in r.text.lower()


@pytest.mark.asyncio
async def test_diff_no_edits(client):
    async with client as c:
        r = await c.get("/api/transcript/diff")
    assert r.status_code == 200
    j = r.json()
    assert j["segments"] == []


# ---------- v2 affordances: waveform, lazy-gen, TXT export ----------


@pytest.mark.asyncio
async def test_waveform_lazy_generates_when_missing(stub_session: Path):
    """First /api/waveform request when peaks file is absent should compute + persist."""
    peaks_path = stub_session / "waveform_peaks.json"
    assert not peaks_path.exists()
    app = create_app(stub_session)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
        r = await c.get("/api/waveform")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 1
    assert body["bins"] == len(body["peaks"]) > 0
    assert all(0.0 <= p <= 1.0 for p in body["peaks"])
    assert peaks_path.exists()  # lazy-write happened


@pytest.mark.asyncio
async def test_waveform_returns_existing_peaks(stub_session: Path):
    pre = {"version": 1, "bins": 3, "duration_s": 1.0, "peaks": [0.1, 0.5, 0.9]}
    (stub_session / "waveform_peaks.json").write_text(json.dumps(pre))
    app = create_app(stub_session)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
        r = await c.get("/api/waveform")
    assert r.status_code == 200
    assert r.json() == pre


@pytest.mark.asyncio
async def test_waveform_404_when_source_missing(stub_session: Path):
    """If both waveform_peaks.json and source.opus are absent, 404."""
    (stub_session / "source.opus").unlink()
    app = create_app(stub_session)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
        r = await c.get("/api/waveform")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_export_txt_get_returns_plaintext(client, stub_session: Path):
    # Pin assertion to whatever speaker label the fixture's transcript actually uses.
    fixture_transcript = json.loads((stub_session / "transcript.json").read_text(encoding="utf-8"))
    expected_speaker = fixture_transcript["segments"][0]["speaker"]

    async with client as c:
        r = await c.get("/api/transcript/export.txt")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "attachment" in r.headers.get("content-disposition", "")
    assert f"[{expected_speaker}]" in r.text


@pytest.mark.asyncio
async def test_export_txt_post_uses_body_segments(client):
    custom = {
        "segments": [
            {"start": 7.0, "end": 8.0, "text": "custom body wins", "speaker": "Alice"},
        ]
    }
    async with client as c:
        r = await c.post("/api/transcript/export.txt", json=custom)
    assert r.status_code == 200
    assert "[Alice]" in r.text
    assert "custom body wins" in r.text


@pytest.mark.asyncio
async def test_export_txt_format_includes_timestamps(client):
    import re
    payload = {
        "segments": [
            {"start": 4.0, "end": 5.0, "text": "hello", "speaker": "S0"},
            {"start": 65.0, "end": 66.0, "text": "world", "speaker": "S1"},
        ]
    }
    async with client as c:
        r = await c.post("/api/transcript/export.txt", json=payload)
    body = r.text.rstrip("\n")
    for line in body.split("\n"):
        assert re.match(r"^\[.+\] \d{2}:\d{2}:\d{2}", line), f"bad line: {line!r}"


@pytest.mark.asyncio
async def test_export_txt_matches_write_txt_byte_for_byte(client, tmp_path):
    """Webapp's TXT export must produce identical bytes to diarizer.output.write_txt
    on the same segment list, with speakers + timestamps both enabled.
    """
    from diarizer.config import OutputConfig
    from diarizer.output import write_txt
    from diarizer.pipeline import Segment, TranscriptionResult

    segs_data = [
        {"start": 0.0, "end": 1.0, "text": "first line here", "speaker": "SPEAKER_00"},
        {"start": 4.5, "end": 5.0, "text": "no speaker line", "speaker": ""},
        {"start": 12.0, "end": 13.0, "text": "third with caps", "speaker": "Alice"},
        {"start": 3700.5, "end": 3701.0, "text": "over an hour", "speaker": "Bob"},
    ]
    result = TranscriptionResult(
        source_path="x", model_name="x", language="en",
        segments=[Segment(**s, words=[]) for s in segs_data],
    )
    cfg = OutputConfig(format="txt", include_speakers=True, include_timestamps=True)
    out_path = tmp_path / "expected.txt"
    write_txt(result, out_path, cfg)
    expected_bytes = out_path.read_bytes()

    async with client as c:
        r = await c.post("/api/transcript/export.txt", json={"segments": segs_data})
    actual_bytes = r.content
    assert actual_bytes == expected_bytes, (
        f"webapp export diverges from output.py:write_txt\n"
        f"expected: {expected_bytes!r}\nactual:   {actual_bytes!r}"
    )
