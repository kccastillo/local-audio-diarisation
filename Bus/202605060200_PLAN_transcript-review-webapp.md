---
title: "Transcript-review webapp: synced playback, live editing, versioned saves"
type: bus-plan
status: ready
assigned_to: sonnet
priority: medium
created: 2026-05-06
created_by: opus
created_month: 202605
log_month: 202605
due: ""
repeatable: false
linked_decisions:
  - "Local FastAPI webapp + vanilla JS frontend (no build step)"
  - "Live spectrogram via Web Audio AnalyserNode on a MediaElementAudioSourceNode"
  - "Versioned sidecar transcripts: <base>_edit_YYYYMMDD_NN.json; NN auto-increments per save; wraps to 00 after 99 (overwriting same-day _00)"
  - "Persisted playback artefact: 16 kHz mono Opus in OGG, libopus VBR ~16 kbps; co-located with original WAV"
  - "Pipeline model-input format unchanged — preprocessing.py already emits 16 kHz mono PCM WAV"
  - "Edit scope v1: speaker labels (global rename + per-segment reassign) + transcript text. Segment boundaries and audio zoom DEFERRED."
  - "Undo/redo within session DEFERRED (versioned saves provide coarse undo)"
  - "Webapp binds to 127.0.0.1 only on default port 8765 (overridable via --port); no LAN exposure"
linked_inputs: []
blocked_by: ""
depends_on_plans: []
rollover_count: 0
pipeline_phase: drafted
audit_state:
  sufficiency_iterations: 1
  plan_safety_iterations: 0
  last_stage: sufficiency
  last_outcome: revision_needed
---

## Objective

Build a local browser-based transcript review tool that, after the diarisation+ASR pipeline finishes, lets the operator: play the source audio (extracted from video if needed), watch a live spectrogram with a moving playhead synchronised to the active transcript segment, edit speaker labels (globally and per-segment) and transcript text live during playback, save versioned sidecar transcripts without ever overwriting the original, and view a realtime diff between current edits and the original.

## Context

**What exists today.**
- Pipeline lives in `diarizer/` — `cli.py`, `pipeline.py`, `preprocessing.py`, `output.py`, `gates.py`, `config.py`.
- `preprocessing.py:168-220` (function `preprocess`) already invokes ffmpeg with `-ac 1 -ar 16000` to produce 16 kHz mono PCM WAV. This IS the model-standard input format — no pipeline change is needed for the "standard format" requirement, only verification + a one-line note in ARCHITECTURE.md.
- `output.py` has writers for txt/json/srt. JSON output (via `TranscriptionResult.to_dict`) contains segments with `start`, `end`, `text`, `speaker`, `words` — sufficient for playhead sync.
- `output/` directory holds finished transcripts; no current convention for grouping a transcript with its source audio.

**What is being added.**
- New module `diarizer/webapp/` containing FastAPI app + static frontend (vanilla JS, no build step).
- New output-stage step that emits a 16 kHz mono Opus/OGG (libopus, ~16 kbps VBR) artefact alongside the existing WAV. For video inputs this is extracted from the source video via ffmpeg in a single pass.
- New "session directory" convention: each pipeline run produces a directory containing `source.opus`, `source.wav` (the existing model-input WAV), `transcript.json` (the immutable original output), and any `transcript_edit_YYYYMMDD_NN.json` sidecars created later by the webapp.
- New CLI subcommand `diarizer serve <session-dir>` that boots the FastAPI app on 127.0.0.1:8765 (overridable via `--port`), opens the browser, and serves the webapp.

**Decisions locked this session (Human, 2026-05-06):**
- D1: Local FastAPI + vanilla JS frontend; no build tooling.
- D2: Spectrogram is live (Web Audio AnalyserNode), not pre-rendered.
- D3: Player controls limited to play/pause, +5s, -5s, volume.
- D4: Edit scope is speaker labels + transcript text only. Segment boundaries and audio zoom deferred to a follow-up PLAN.
- D5: Sidecar filename `<base>_edit_YYYYMMDD_NN.json`; NN is a per-day counter starting at `00`; on the 100th same-day save the counter wraps and overwrites `_00`. Original `<base>.json` is never written by the webapp.
- D6: Persisted audio is Opus mono 16 kHz ~16 kbps (libopus VBR) in OGG container.
- D7: Undo/redo within an editing session is DEFERRED. Versioned saves provide coarse undo.
- D8: Webapp binds to 127.0.0.1 only — no `--host 0.0.0.0` option, no LAN exposure. Default port 8765.

**Out of scope (explicitly deferred):**
- Drag-to-adjust segment boundaries.
- Audio zoom / waveform horizontal scaling.
- In-session undo/redo stack.
- Multi-user / collaborative editing.
- Authentication (loopback-only binding makes this unnecessary for v1).

## Design Decisions Classification

*n/a — ad-hoc PLAN, not produced via ideate. All design decisions were locked by the Human in the session on 2026-05-06 (see linked_decisions and D1–D8 above).*

**Already locked** (Human affirmed during ideation; require no answer at design-review checkpoint):
- FastAPI + vanilla JS frontend (D1)
- Live Web Audio spectrogram, not pre-rendered (D2)
- Player controls: play/pause, +5s, -5s, volume only (D3)
- Edit scope: speaker labels + text; boundaries + audio zoom deferred (D4)
- Sidecar versioning scheme with NN wrap-to-00 (D5)
- Opus/OGG 16 kHz ~16 kbps libopus VBR (D6)
- Undo/redo within session deferred (D7)
- 127.0.0.1 only, default port 8765 (D8)

**Mechanically forced** (no meaningful alternative):
- ffmpeg for Opus encoding (already used for WAV — same invocation path)
- `<audio>` element + MediaElementAudioSourceNode for spectrogram (no other way to analyse an `<audio>` tag in real time)
- HTTP range request support on `/api/audio` (browsers require range requests for seekable `<audio>`)

**Real judgement calls** (none outstanding — all resolved above):
- *(none)*

## Steps

Number each step. Each step lists files touched, the change, and a `verify:` shell line where mechanically possible.

### Step 1 — Define session-dir layout + manifest

- New helper `diarizer/session.py` (or extend `output.py`): function `create_session_dir(base_output_dir, source_path) -> Path` that creates `output/<source_basename>_<timestamp>/` and writes a `session.json` manifest listing `source_audio_opus`, `source_audio_wav`, `transcript_original`, `edits: []`.
- `session.json` schema: `{version: 1, source_basename, created, audio_opus, audio_wav, transcript_original, edits: [{filename, created}]}`.
- `verify:` `python -c "from diarizer.session import create_session_dir; import tempfile, pathlib; d = create_session_dir(pathlib.Path(tempfile.mkdtemp()), pathlib.Path('foo.mp4')); assert (d/'session.json').exists()"`

### Step 2 — Add Opus-export to output stage

- Extend `diarizer/output.py` with `write_opus(source_audio_path, out_path) -> Path` invoking ffmpeg: `ffmpeg -y -i <src> -ac 1 -ar 16000 -c:a libopus -b:a 16k -vbr on <out.opus>`.
- Wire into `cli.py` so a successful pipeline run also emits `source.opus` into the session dir alongside the existing WAV.
- `verify:` ffprobe of the output reports 1 channel, 16000 Hz, codec opus.

### Step 3 — Video-input handling

- Confirm current behaviour in `preprocessing.py:202-211` (the ffmpeg invocation drops the video stream because no `-vn` is needed when only audio output is requested, but the WAV is the only artefact). For video inputs, the same source file is also passed to the Step-2 Opus encoder.
- Document in ARCHITECTURE.md (one paragraph) that the webapp's playback artefact is Opus, not WAV; the WAV remains the model-canonical input.
- No code change in preprocessing.py — verification only.

### Step 4 — FastAPI backend `diarizer/webapp/app.py`

Endpoints (all bound to 127.0.0.1):
- `GET /` — serves `static/index.html`.
- `GET /static/*` — static asset passthrough.
- `GET /api/session` — returns `session.json` plus a list of currently-existing `_edit_*.json` files.
- `GET /api/audio` — streams the session's `source.opus` with HTTP range support (browsers need range requests for `<audio>` seeking).
- `GET /api/transcript` — returns the original `transcript.json`.
- `GET /api/transcript/edit/<filename>` — returns a specific edit sidecar.
- `POST /api/transcript/save` — body is the current edited transcript JSON; computes today's `YYYYMMDD`, scans existing `<base>_edit_<today>_NN.json` files, picks next `NN` (wrapping to 00 if NN==99 already exists, overwriting), writes file, appends to `session.json` `edits` list, returns the new filename.
- `GET /api/transcript/diff?against=original` — returns a structured diff (per-segment) between the latest edit (or supplied `?from=<filename>`) and the original.

Constraints:
- App MUST bind to `127.0.0.1`; the `host` parameter is not exposed via CLI.
- Reject any request whose `Host:` header is not `127.0.0.1` or `localhost` (defence against DNS rebinding).
- No external network calls.

### Step 5 — Frontend `diarizer/webapp/static/`

Files: `index.html`, `app.js`, `style.css`. Vanilla JS only — no bundler, no framework.

Components:
- **Player bar:** play/pause toggle, −5s, +5s, volume slider (0-1), elapsed/total time. Backed by a single `<audio>` element with `src=/api/audio`.
- **Spectrogram canvas:** `<canvas>` rendered each frame from a Web Audio `AnalyserNode` connected via `MediaElementAudioSourceNode` from the `<audio>` element. FFT size 1024; log-frequency Y axis; magnitude mapped to a viridis-like colour ramp; horizontal scroll = time, with a vertical playhead line at the current time. (Note: the analyser is realtime — only past audio is rendered as the user plays.)
- **Transcript pane:** vertical list of segments. Each segment row shows `[speaker]` (clickable — opens speaker-rename modal), `start time` (clickable — seeks audio), and the segment text in a `contenteditable` cell. The row whose `[start, end]` brackets the current playhead is highlighted (CSS class).
- **Speaker controls:** "Rename speaker" modal — choose a speaker (e.g. `SPEAKER_00`), enter a new label, confirm — applies to every segment with that speaker. Per-segment reassign is done by clicking the speaker pill on a single segment and picking from a dropdown.
- **Save button:** posts current state to `/api/transcript/save`. On success, shows the new filename and adds it to a "Versions" dropdown.
- **Diff toggle:** swaps the transcript pane into diff view rendered from `/api/transcript/diff` — per-segment additions/deletions/changes shown inline.

### Step 6 — CLI subcommand `diarizer serve`

- Extend `diarizer/cli.py` with subparser `serve <session-dir> [--port 8765]`.
- Validates the session dir contains `session.json`, `source.opus`, and `transcript.json`.
- Boots `uvicorn` against `diarizer.webapp.app:app`, host `127.0.0.1`, port from `--port`.
- Optionally opens default browser at `http://127.0.0.1:<port>/` (use `webbrowser.open`).
- `verify:` `diarizer serve --help` exits 0.

### Step 7 — Verification + smoke test

- Add `tests/test_webapp_smoke.py`: starts the FastAPI app via `httpx.AsyncClient`, asserts `/api/session` returns valid JSON, `/api/transcript` returns the original, `/api/transcript/save` writes a sidecar with the expected filename pattern.
- Manual UI walkthrough (verify: human): open an existing session in Chrome and Firefox; confirm Opus plays; confirm spectrogram renders; confirm play/pause/+5/-5/volume work; confirm active segment highlights; confirm global rename and per-segment reassign both work; confirm save creates `_edit_YYYYMMDD_00.json`, second save creates `_NN+1`; confirm diff view renders.
- `acceptance:` `pytest tests/test_webapp_smoke.py -q`

## Verification

- [ ] Session dir helper creates directory and `session.json` with correct schema.
      `verify: python -c "from diarizer.session import create_session_dir; import tempfile, pathlib; d = create_session_dir(pathlib.Path(tempfile.mkdtemp()), pathlib.Path('foo.mp4')); assert (d/'session.json').exists()"`
- [ ] Opus output has 1 channel, 16000 Hz, codec opus.
      `verify: ffprobe -v error -show_entries stream=codec_name,channels,sample_rate -of default=noprint_wrappers=1 <session-dir>/source.opus`
- [ ] ARCHITECTURE.md updated with Opus playback artefact paragraph.
      `verify: python -c "assert 'Opus' in open('ARCHITECTURE.md').read()"`
- [ ] FastAPI app starts and responds on loopback.
      `verify: python -c "import httpx, asyncio; r = asyncio.run(httpx.AsyncClient(app=__import__('diarizer.webapp.app', fromlist=['app']).app, base_url='http://127.0.0.1').get('/')); assert r.status_code == 200"`
- [ ] `diarizer serve --help` exits 0.
      `verify: diarizer serve --help`
- [ ] Smoke tests pass.
      `acceptance: pytest tests/test_webapp_smoke.py -q`
- [ ] Manual UI walkthrough: playback, spectrogram, segment highlight, global rename, per-segment reassign, save/versioning, diff view.
      `verify: human`

## Executor Notes

*Populated after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:**
**What was done:**
**Blockers (if any):**
**Files modified:**
