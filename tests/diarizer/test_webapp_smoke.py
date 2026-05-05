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
