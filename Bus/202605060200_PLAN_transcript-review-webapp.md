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
  - "Save endpoint uses O_EXCL atomic create with retry on collision; wrap (NN=99 → 00) returns `wrapped: true` so the frontend can warn"
  - "Frontend aesthetic: simple/functional for v1 — system fonts, no CSS frameworks, native controls, minimal palette. Polish deferred."
linked_inputs: []
blocked_by: ""
depends_on_plans: []
rollover_count: 0
pipeline_phase: drafted
audit_state:
  sufficiency_iterations: 3
  plan_safety_iterations: 0
  last_stage: sufficiency
  last_outcome: success
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
- New "session directory" convention: each pipeline run produces a directory containing `source.opus`, `source.wav` (the existing model-input WAV), `transcript.json` (the immutable original output, copied from `output/<stem>.json`), and any `transcript_edit_YYYYMMDD_NN.json` sidecars created later by the webapp.
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
  - `transcript_original` points to `transcript.json` **inside the session directory** — this is the immutable canonical reference for the webapp. It does NOT point to the legacy `output/<stem>.json` path; that file remains in place and is unchanged, but the webapp always reads from the session-dir copy.
- `verify:` `python -c "from diarizer.session import create_session_dir; import tempfile, pathlib; d = create_session_dir(pathlib.Path(tempfile.mkdtemp()), pathlib.Path('foo.mp4')); assert (d/'session.json').exists()"`

### Step 2 — Add Opus-export and transcript copy to output stage

- Extend `diarizer/output.py` with `write_opus(source_audio_path, out_path) -> Path` invoking ffmpeg: `ffmpeg -y -i <src> -ac 1 -ar 16000 -c:a libopus -b:a 16k -vbr on <out.opus>`.
- Wire into `cli.py` so a successful pipeline run:
  1. Emits `source.opus` into the session dir (the Opus export).
  2. **Copies `output/<stem>.json` to `<session-dir>/transcript.json`** as part of the same finalisation block. This copy is the immutable canonical transcript reference for the webapp. The original `output/<stem>.json` is left in place — existing pipeline behaviour is unchanged.
- `verify:` ffprobe of the output reports 1 channel, 16000 Hz, codec opus.
- `verify:` after a pipeline run, both `output/<stem>.json` and `<session-dir>/transcript.json` exist and are byte-identical.

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
- `POST /api/transcript/save` — body is the current edited transcript JSON. Save behaviour:
  1. Resolves today's date as `YYYYMMDD`.
  2. Scans the session dir for existing `<base>_edit_<today>_NN.json` files; picks the next `NN` candidate.
  3. Attempts to create the file atomically using `open(path, "x")` (equivalent to `O_EXCL | O_CREAT | O_WRONLY`).
  4. On `FileExistsError`, increments `NN` and retries. Bounded retry: max 100 attempts (covers the full `00`–`99` daily counter range).
  5. On wrap (`NN` would exceed `99`), wraps to `00` per locked D5 behaviour — this is a deliberate overwrite (operator choice). Response payload includes `"wrapped": true` when this path fires, so the frontend can surface a toast warning.
  6. Appends the new filename to the `session.json` `edits` list.
  7. Returns the new filename (and `wrapped` flag).
- `GET /api/transcript/diff?against=original` — returns a structured diff (per-segment) between the latest edit (or supplied `?from=<filename>`) and the original.

Constraints:
- App MUST bind to `127.0.0.1`; the `host` parameter is not exposed via CLI.
- Reject any request whose `Host:` header is not `127.0.0.1` or `localhost` (defence against DNS rebinding). Return HTTP 421 or 403 — implementation choice, but rejection must occur.
- No external network calls.

`verify:` unit test that pre-creates `<session-dir>/<base>_edit_<today>_05.json`, calls `POST /api/transcript/save`, asserts the newly written file is `<base>_edit_<today>_06.json`.

### Step 5 — Frontend `diarizer/webapp/static/`

Files: `index.html`, `app.js`, `style.css`. Vanilla JS only — no bundler, no framework.

Components:
- **Player bar:** play/pause toggle, −5s, +5s, volume slider (0-1), elapsed/total time. Backed by a single `<audio>` element with `src=/api/audio`.
- **Spectrogram canvas:** `<canvas>` driven by Web Audio. Wiring:
  - Create `audioCtx = new AudioContext()` lazily on the first user-gesture (play click).
  - `source = audioCtx.createMediaElementSource(audioElement)`.
  - `analyser = audioCtx.createAnalyser(); analyser.fftSize = 1024;`
  - **Connect both** `source.connect(analyser)` AND `analyser.connect(audioCtx.destination)` — without the destination connection the operator hears no audio.
  - On the `play` click handler: `if (audioCtx.state === 'suspended') await audioCtx.resume()` — autoplay-policy gates require this on first gesture.
  - Render loop reads `analyser.getByteFrequencyData()` per frame; magnitude mapped to a viridis-like colour ramp; horizontal scroll = time, vertical playhead line at current `audio.currentTime`.
- **Transcript pane:** vertical list of segments. Each segment row shows `[speaker]` (clickable — opens speaker-rename modal), `start time` (clickable — seeks audio), and the segment text in a `contenteditable` cell. The row whose `[start, end]` brackets the current playhead is highlighted (CSS class).
- **Speaker controls:** "Rename speaker" modal — choose a speaker (e.g. `SPEAKER_00`), enter a new label, confirm — applies to every segment with that speaker. Per-segment reassign is done by clicking the speaker pill on a single segment and picking from a dropdown.
- **Save button:** posts current state to `/api/transcript/save`. On success, shows the new filename and adds it to a "Versions" dropdown. If the response includes `"wrapped": true`, surfaces a toast warning to the operator.
- **Diff toggle:** swaps the transcript pane into diff view rendered from `/api/transcript/diff` — per-segment additions/deletions/changes shown inline.

#### UI aesthetic constraints

The frontend prioritises function over visual polish for v1:
- Plain, system-font-based UI (`font-family: system-ui, sans-serif`). No CSS frameworks, no icon libraries, no custom theming.
- Native HTML controls (`<button>`, `<input type="range">`, `<textarea>` / `contenteditable`) styled minimally.
- Layout: simple flex/grid; clear separation between player bar (top), spectrogram (mid), transcript pane (main scroll area). No animations beyond the playhead movement and the active-segment highlight.
- Colour palette: 3–4 colours max (background, text, accent for active segment, warning/wrap toast). High contrast.
- The aim is "looks like a tool, not a product". Polish is explicitly deferred.

### Step 6 — CLI restructure + `diarizer serve` subcommand

`cli.py` is currently flat-arg (required `--input`). Adding `serve` requires real restructuring:

1. Convert `cli.py` to use `argparse` subparsers with two subcommands:
   - `run` — current behaviour (takes `--input`, produces a session dir as new finalisation).
   - `serve <session-dir> [--port 8765]` — boots the FastAPI app on 127.0.0.1.
2. Back-compat: when `cli.py` is invoked with no subcommand but `--input` is present, route to the `run` subcommand. This preserves existing invocation patterns. Pure no-arg invocations now print help.
3. `serve` validates the session dir contains `session.json`, `source.opus`, and `transcript.json`.
4. Boots `uvicorn` against `diarizer.webapp.app:app`, host `127.0.0.1`, port from `--port`.
5. Optionally opens default browser at `http://127.0.0.1:<port>/` (use `webbrowser.open`).
6. Add a one-line note to `README.md` documenting the subparser change and the `serve` invocation.

- `verify:` `diarizer serve --help` exits 0.
- `verify:` `diarizer run --help` exits 0.
- `verify:` `diarizer --input foo.wav` (legacy flat-arg form) routes correctly without error.

### Step 7 — Verification + smoke test

**Fixture strategy.**
- The smoke test uses a **synthetic stub session** built under `tests/fixtures/webapp_session/` containing:
  - `session.json` — hand-written minimal manifest, `version=1`.
  - `source.opus` — a 1-second silent Opus file generated at test setup time using ffmpeg (not committed as a binary; ffmpeg is already a project dependency so generation is reliable on the test runner).
  - `transcript.json` — minimal hand-written transcript with 2–3 segments.
- A `pytest` fixture in `tests/conftest.py` (or a dedicated fixture file) materialises this stub session into a temp dir per test invocation.
- The smoke test explicitly does **not** exercise real pipeline output → save round-trip; that path is covered by existing pipeline tests.

**Test file:** `tests/test_webapp_smoke.py` — starts the FastAPI app via `httpx.AsyncClient` and asserts:
- `GET /api/session` returns valid JSON.
- `GET /api/transcript` returns the original transcript.
- `GET /api/audio` with a `Range: bytes=0-99` header returns HTTP 206 with a 100-byte body.
- `GET /api/session` with a forged `Host: evil.example` header returns HTTP 421 or 403 (exact code is implementation choice; rejection must occur).
- `POST /api/transcript/save` writes a sidecar with the expected filename pattern.

**Manual UI walkthrough** (`verify: human — operator runs the affordance checklist (1–18 above); all items must pass`):

For each item, the operator must observe the stated behaviour. If any item fails, the walkthrough fails — outcome-verification will revert to drafted.

1. Audio loads and plays from `/api/audio` in Chrome and Firefox.
2. Play/pause button toggles playback (clicked while playing → pauses; clicked while paused → resumes).
3. `+5s` button advances `audio.currentTime` by exactly 5 seconds (verify by reading the elapsed-time readout before/after).
4. `-5s` button rewinds by exactly 5 seconds; clamps at 0 when called near start.
5. Volume slider changes perceived loudness; setting to 0 silences; setting back to previous value restores level.
6. Spectrogram canvas is non-empty during playback (visible spectral content updating each frame).
7. Spectrogram playhead line moves left-to-right in sync with audio.
8. Transcript pane renders all segments with `[speaker]` pill, start time, and editable text cell.
9. Active-segment highlight follows the playhead — as audio crosses each segment's `[start, end]`, the corresponding row gains the highlight class and earlier rows lose it.
10. Clicking a segment's start-time seeks audio to that timestamp.
11. Editing a segment's text cell persists across blur/focus — clicking out of an edited cell and back in shows the edited text, not the original. Edits are not silently reverted.
12. Speaker rename modal applies globally — renaming `SPEAKER_00` to `Alice` updates every segment with that speaker in the visible pane.
13. Per-segment speaker reassign — clicking a single segment's speaker pill and choosing a different speaker updates only that segment.
14. Save button POSTs to `/api/transcript/save` and surfaces the new filename (e.g. `transcript_edit_20260506_00.json`).
15. Second save the same day produces `_01`, third produces `_02`, etc.
16. Diff toggle swaps the transcript pane into a diff view showing per-segment changes vs the original.
17. Wrap-warning toast appears when the save endpoint returns `wrapped: true` (force this by pre-creating the necessary fixture files in a manual test path).
18. Reloading the page restores the latest edit version (verify the dropdown lists all `_edit_*.json` files and selecting one loads its contents).

`acceptance:` `pytest tests/test_webapp_smoke.py -q` — covers session JSON, transcript fetch, audio range (206/100-byte), rebinding rejection, and save round-trip.

## Verification

- [ ] Session dir helper creates directory and `session.json` with correct schema; `transcript_original` points to session-dir `transcript.json`.
      `verify: python -c "from diarizer.session import create_session_dir; import tempfile, pathlib; d = create_session_dir(pathlib.Path(tempfile.mkdtemp()), pathlib.Path('foo.mp4')); assert (d/'session.json').exists()"`
- [ ] After pipeline run, `output/<stem>.json` and `<session-dir>/transcript.json` exist and are byte-identical.
      `verify: python -c "import json, pathlib; s = list(pathlib.Path('output').glob('*/session.json'))[0]; m = json.loads(s.read_text()); o = pathlib.Path('output') / (m['source_basename'] + '.json'); t = s.parent / 'transcript.json'; assert o.read_bytes() == t.read_bytes()"`
- [ ] Opus output has 1 channel, 16000 Hz, codec opus.
      `verify: ffprobe -v error -show_entries stream=codec_name,channels,sample_rate -of default=noprint_wrappers=1 <session-dir>/source.opus`
- [ ] ARCHITECTURE.md updated with Opus playback artefact paragraph.
      `verify: python -c "assert 'Opus' in open('ARCHITECTURE.md').read()"`
- [ ] FastAPI app starts and responds on loopback.
      `verify: python -c "import httpx, asyncio; r = asyncio.run(httpx.AsyncClient(app=__import__('diarizer.webapp.app', fromlist=['app']).app, base_url='http://127.0.0.1').get('/')); assert r.status_code == 200"`
- [ ] Save endpoint O_EXCL behaviour: pre-create `_edit_<today>_05.json`, call save, assert new file is `_edit_<today>_06.json`.
      `verify: unit test in tests/test_webapp_smoke.py`
- [ ] `diarizer serve --help` exits 0.
      `verify: diarizer serve --help`
- [ ] `diarizer run --help` exits 0.
      `verify: diarizer run --help`
- [ ] Legacy flat-arg invocation `diarizer --input foo.wav` routes to `run` subcommand without error.
      `verify: human / integration test`
- [ ] README.md updated with one-line note on subparser change and `serve` invocation.
      `verify: python -c "assert 'serve' in open('README.md').read()"`
- [ ] Smoke tests pass (session JSON, transcript, audio range 206, rebinding rejection, save round-trip).
      `acceptance: pytest tests/test_webapp_smoke.py -q`
- [ ] Manual UI walkthrough: operator runs the affordance checklist (items 1–18); all items must pass.
      `verify: human — operator runs the affordance checklist (1–18 above); all items must pass`

## Executor Notes

*Populated after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:**
**What was done:**
**Blockers (if any):**
**Files modified:**
