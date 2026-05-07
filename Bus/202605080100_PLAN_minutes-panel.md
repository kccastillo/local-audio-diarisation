---
title: "Meeting-minutes side panel: generate, display, and anchor-sync minutes in the diarizer webapp"
type: bus-plan
status: ready
assigned_to: "haiku"
priority: medium
created: 2026-05-08
created_by: "sonnet"
created_month: 202605
log_month: 202605
due: ""
repeatable: false
repeat_cadence: ""
linked_inputs: []
blocked_by: ""
depends_on_plans: []
rollover_count: 0
triggers_plans: []
branch: ""
files_touched:
  - diarizer/webapp/app.py
  - diarizer/webapp/static/index.html
  - diarizer/webapp/static/app.js
  - diarizer/webapp/static/style.css
  - diarizer/minutes.py
  - pyproject.toml
  - config/config.yaml
  - README.md
  - ARCHITECTURE.md
verification_commands: []
tag_after: ""
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: ""
pipeline_phase: "drafting"
---

## Objective

Add a second panel beside the running transcript in the diarizer webapp that displays generated meeting minutes. Each minute statement is anchored to one or more transcript segments so that clicking a statement highlights the corresponding waveform region and scrolls the transcript to the primary anchor. Minutes are generated via an opt-in external Anthropic API call and persisted using the same numbered-snapshot convention as transcript edits.

## Context

The webapp currently shows a transcript panel with synced waveform playback and speaker-label editing. This feature adds a companion minutes panel to the right, making the webapp useful for post-meeting review without leaving the tool. The feature explicitly breaks the project's offline-by-default guarantee — this is intentional, opt-in, and must be clearly documented.

**Offline guarantee caveat:** Using the Generate function sends the full transcript to the Anthropic API. This is opt-in (button press + `ANTHROPIC_API_KEY` env var required). The webapp must not call the API unless explicitly triggered, and the README and ARCHITECTURE must document the caveat prominently.

## Design Decisions Classification

**Already locked** (proposed and affirmed during ideation):

- External Anthropic API generation, opt-in, offline-guarantee broken when used
- Default model `claude-sonnet-4-6`; user-selectable `claude-opus-4-7` via UI dropdown
- Attendees pre-filled from distinct transcript speaker labels, manually editable before Generate
- Primary anchor (`primary_ref`) + secondary refs (`secondary_refs`) per statement
- Content-driven tension detection with crosstalk amplitude windows as supplementary signal
- Standard meeting-minutes sections (`agenda`, `discussion`, `decisions`, `action_items`, `next_steps`) plus a `tensions` section
- `minutes.json` canonical file + `minutes_edit_YYYYMMDD_NN.json` numbered snapshots in session dir
- Full-snapshot saves (no in-place diff machinery — `_diff_segments` is not reused; minutes have add/remove/reorder semantics incompatible with segment-aligned diff)
- Top-N (~20) high-amplitude crosstalk windows sent to LLM as `[{start_seg, end_seg, indicator_value}, ...]`
- Split-pane with draggable vertical divider; both panels visible simultaneously
- Synchronous generation for MVP (block UI with spinner; ~10–30 s expected)

**Mechanically forced** (downstream consequences of locked decisions):

- `ANTHROPIC_API_KEY` env var; `anthropic` SDK dependency added to `pyproject.toml`
- New `diarizer/minutes.py` module for API call, prompt construction, and response validation
- Endpoints: `POST /api/minutes/generate`, `GET /api/minutes`, `POST /api/minutes/save`, `GET /api/minutes/edit/<filename>`
- `POST /api/minutes/generate` returns 4xx when `ANTHROPIC_API_KEY` is absent
- Clicking a minute statement syncs both transcript scroll and waveform highlight (flows from `primary_ref` + split-pane design)
- Tensions section rendered with visually distinct styling (amber accent)
- README and ARCHITECTURE offline-caveat update

**Real judgement calls** (for Human design-review checkpoint before execution):

- Whether Generate requires a confirmation dialog warning that transcript content will be sent to Anthropic (privacy friction vs friction cost)
- Whether the model selector is a UI dropdown or config-only setting
- Whether `primary_ref` is determined by the LLM emitting segment indices directly vs LLM emitting quoted snippets that are post-hoc matched to segment indices

## Steps

1. **Add `anthropic` dependency and config defaults.** Update `pyproject.toml` to add `anthropic` to the dependency list. Extend `config/config.yaml` with a `minutes:` block containing: `default_model`, `opus_model`, `top_n_amplitude_windows`, `generation_enabled`.

2. **Build `diarizer/minutes.py`.** Implement:
   - `build_prompt(transcript, attendees, crosstalk_summary) -> messages` — constructs the messages list for the Anthropic API call. Transcript is formatted as `[{index, speaker, text, start, end}, ...]`; crosstalk_summary is the top-N amplitude windows.
   - `extract_top_n_crosstalk(crosstalk_data, n) -> list` — selects and returns the N highest-amplitude windows from the session crosstalk data.
   - `generate(transcript, attendees, model) -> minutes_dict` — calls the Anthropic API, parses the JSON response, and validates the schema (all required sections present; each statement has `id`, `text`, `primary_ref`; all refs are valid segment indices). Raises typed exceptions on missing API key, schema violation, and API error.

3. **Wire backend endpoints in `app.py`.** Add four routes:
   - `POST /api/minutes/generate` — body `{attendees: [...], model: "..."}`. Reads current session transcript + crosstalk data, calls `diarizer.minutes.generate`, returns the generated minutes object. Does NOT persist — the UI save step is explicit. Returns 4xx with a clear error message if `ANTHROPIC_API_KEY` is not set.
   - `GET /api/minutes` — returns canonical `minutes.json` from the current session dir if present; 404 otherwise.
   - `POST /api/minutes/save` — body is the full minutes object. Writes the next numbered `minutes_edit_YYYYMMDD_NN.json` snapshot; overwrites canonical `minutes.json` to match. Returns the snapshot filename.
   - `GET /api/minutes/edit/<filename>` — fetches a specific named snapshot from the session dir.

4. **Implement split-pane layout in `index.html` and `style.css`.** Refactor the existing single-panel layout to a horizontal split pane. The transcript panel occupies the left side; the minutes panel occupies the right. A draggable vertical divider separates them. Divider position is persisted in `localStorage`. The minutes panel is visible by default (not hidden behind a toggle) so that click-to-sync is immediately usable when minutes are loaded.

5. **Build the minutes panel UI in `app.js`.** Implement:
   - Empty state: attendee editor (chip or textarea, pre-filled from `distinctSpeakers()`), model dropdown (`claude-sonnet-4-6` default, `claude-opus-4-7` option), Generate button.
   - On Generate: disable button, show spinner, call `POST /api/minutes/generate`. On success, render result. On error (including missing API key), show an inline error message.
   - Render: each section rendered as a labelled collapsible block. Each statement rendered as a clickable line. The `tensions` section uses amber accent styling. Statement click handler: call the existing waveform highlight API for `primary_ref` (full region highlight) and add small tick marks for each `secondary_ref`; scroll the transcript panel to the segment at `primary_ref`.
   - Save button: call `POST /api/minutes/save` with the current in-memory minutes object. Show the returned snapshot filename as a success indicator.
   - Session load: on page load, call `GET /api/minutes`; if a response arrives (200), render the existing minutes and skip the empty state.

6. **Documentation update.** Add a "Generating meeting minutes (online)" section to `README.md` covering: what the feature does, how to enable it (`ANTHROPIC_API_KEY`), and an explicit note that using it sends the full transcript text to Anthropic, breaking the project's offline guarantee. Add the same caveat (one sentence with a pointer to the README section) to `ARCHITECTURE.md`.

## Verification

- [ ] `anthropic` SDK installed and importable.
      `verify: python -c "import anthropic"`
- [ ] `diarizer/minutes.py` module importable with expected public API.
      `verify: python -c "from diarizer.minutes import generate, build_prompt, extract_top_n_crosstalk"`
- [ ] Webapp still serves after layout refactor.
      `verify: python -m diarizer.cli serve <session-dir>` — GET / returns 200.
- [ ] Generate endpoint returns 4xx when API key is absent.
      `verify: python -m pytest tests/ -k "minutes" -x` (if unit tests are written for the endpoint)
- [ ] With `ANTHROPIC_API_KEY` set, clicking Generate on a small session produces a valid `minutes.json` with all expected sections and at least one statement carrying `primary_ref` and `secondary_refs`.
      `acceptance: human`
- [ ] Clicking a minute statement highlights the corresponding waveform region and scrolls the transcript to the `primary_ref` segment.
      `acceptance: human`
- [ ] Save creates a numbered snapshot file matching the transcript-edit naming convention (`minutes_edit_YYYYMMDD_NN.json`).
      `acceptance: human`
- [ ] Without `ANTHROPIC_API_KEY` set, the Generate button shows a clear inline error and the server does not crash.
      `acceptance: human`
- [ ] README documents the feature and explicitly notes that using it sends transcript content to Anthropic, breaking the offline guarantee.
      `acceptance: human`

## Executor Notes

*Populated after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:**
**What was done:**
**Blockers (if any):**
**Files modified:**
