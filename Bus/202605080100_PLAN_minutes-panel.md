---
title: "Meeting-minutes side panel: generate, display, and anchor-sync minutes in the diarizer webapp"
type: bus-plan
status: cancelled
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
pipeline_phase: "executing"
audit_state:
  sufficiency_iterations: 2
  plan_safety_iterations: 1
  last_stage: plan_safety
  last_outcome: success
---

## Cancellation note (2026-05-08)

Implementation completed and verified, then reverted at the Human's request. The Anthropic API key requirement (separate from the Human's existing Claude.ai subscription) was the deciding factor — the Human chose not to ship the feature rather than maintain a second billing relationship for occasional use. Implementation is preserved in git history (commit `2a0a7d3`, reverted by `dc630ce`) and can be reinstated by reverting the revert if the cost question changes. PLAN retained for the audit trail.

## Objective

Add a second panel beside the running transcript in the diarizer webapp that displays generated meeting minutes. Each minute statement is anchored to one or more transcript segments so that clicking a statement highlights the corresponding waveform region and scrolls the transcript to the primary anchor. Minutes are generated via an opt-in external Anthropic API call and persisted using the same numbered-snapshot convention as transcript edits.

## Context

The webapp currently shows a transcript panel with synced waveform playback and speaker-label editing. This feature adds a companion minutes panel to the right, making the webapp useful for post-meeting review without leaving the tool. The feature explicitly breaks the project's offline-by-default guarantee — this is intentional, opt-in, and must be clearly documented.

**Offline guarantee caveat:** Using the Generate function sends the full transcript to the Anthropic API. This is opt-in (button press + `ANTHROPIC_API_KEY` env var required). The webapp must not call the API unless explicitly triggered, and the README and ARCHITECTURE must document the caveat prominently.

**Statement schema** — two distinct shapes:

**LLM-emitted (on-wire) shape** — what the LLM returns in its JSON response:

```
{ id, text, quote, cited_index, secondary: [{quote, cited_index}], kind? }
```

**Post-`resolve_refs` canonical shape** — stored in `minutes.json` after reference resolution:

```
{ id, text, quote, primary_ref, secondary_refs, kind?, ref_resolution?: "exact"|"fuzzy" }
```

- `quote` — short verbatim snippet from the transcript that the statement is grounded in; emitted by the LLM, used by `resolve_refs` for post-hoc index validation.
- `cited_index` — integer segment index as emitted by the LLM; input to `resolve_refs`.
- `primary_ref` — integer segment index after resolution; replaces `cited_index` in the stored shape.
- `secondary_refs` — list of integer segment indices resolved from the on-wire `secondary` list; each resolved independently by `resolve_refs`.
- `kind` — optional; used for the `tensions` section to distinguish tension statements.
- `ref_resolution` — optional; present only when fuzzy-match fallback was used for `primary_ref`. Absent (or `"exact"`) when substring match succeeded.

## Design Decisions Classification

**Already locked** (proposed and affirmed during ideation):

- External Anthropic API generation, opt-in, offline-guarantee broken when used
- Default model `claude-sonnet-4-6`; user-selectable `claude-opus-4-7` via UI dropdown
- Attendees pre-filled from distinct transcript speaker labels, manually editable before Generate
- Primary anchor (`primary_ref`) + secondary refs (`secondary_refs`) per statement
- Content-driven tension detection with emotive-segment signal as supplementary input to the LLM
- Standard meeting-minutes sections (`agenda`, `discussion`, `decisions`, `action_items`, `next_steps`) plus a `tensions` section
- `minutes.json` canonical file + `minutes_edit_YYYYMMDD_NN.json` numbered snapshots in session dir
- Full-snapshot saves (no in-place diff machinery — `_diff_segments` is not reused; minutes have add/remove/reorder semantics incompatible with segment-aligned diff)
- Top-N emotive segments (default N=15, configurable in `config/config.yaml`) selected by amplitude energy score derived from the existing `waveform_peaks.json` in each session dir; passed to the LLM as `[{seg_index, speaker, energy_score, text}, ...]` with prompt note that these segments are louder than the surrounding meeting; LLM is instructed to weight tension/statements-of-interest detection accordingly. Note: the crosstalk indicator (`webapp/crosstalk.py`, speaker-swap count over a sliding window, drives waveform highlight stripes) is a separate, unrelated signal — it is NOT an input to the LLM and NOT involved in prompt construction.
- Split-pane with draggable vertical divider; both panels visible simultaneously
- Synchronous generation for MVP (block UI with spinner; ~10–30 s expected)
- **Generate confirmation:** First-use confirmation modal (per session). Modal text: "Generating minutes will send the full transcript to Anthropic, breaking the offline guarantee. Continue?" Consent remembered for the rest of the session via `sessionStorage`. (Resolved 2026-05-08)
- **Model selector placement:** Dropdown in the minutes panel UI next to the Generate button. User picks per-run from `claude-sonnet-4-6` (default) or `claude-opus-4-7`. (Resolved 2026-05-08)
- **`primary_ref` determination:** Hybrid validation. LLM emits both `quote` (short snippet) and `cited_index` per statement. Post-hoc validation: if the quote substring is found in `transcript[cited_index].text`, accept the index. Otherwise fuzzy-match the quote against all segments and use the best match; record `ref_resolution: "fuzzy"` on the statement when fallback fired. Drop statement entirely (omit from minutes) only if no segment scores above a similarity threshold. Same logic applies to `secondary_refs`. (Resolved 2026-05-08)

**Mechanically forced** (downstream consequences of locked decisions):

- `ANTHROPIC_API_KEY` env var; `anthropic` SDK dependency added to `pyproject.toml`
- New `diarizer/minutes.py` module for API call, prompt construction, and response validation
- Endpoints: `POST /api/minutes/generate`, `GET /api/minutes`, `POST /api/minutes/save`, `GET /api/minutes/edit/<filename>`
- `POST /api/minutes/generate` returns 4xx when `ANTHROPIC_API_KEY` is absent
- Clicking a minute statement syncs both transcript scroll and waveform highlight (flows from `primary_ref` + split-pane design)
- Tensions section rendered with visually distinct styling (amber accent)
- README and ARCHITECTURE offline-caveat update

**Real judgement calls** (for Human design-review checkpoint before execution):

All resolved at design-review checkpoint 2026-05-08.

## Steps

1. **Add `anthropic` dependency and config defaults.** Update `pyproject.toml` to add `anthropic` to the dependency list. Extend `config/config.yaml` with a `minutes:` block containing: `default_model`, `opus_model`, `top_n_emotive_segments`, `generation_enabled`.

2. **Build `diarizer/minutes.py`.** Implement:
   - `extract_emotive_segments(transcript, peaks, n) -> list[dict]` — ranks transcript segments by audio amplitude energy score. For each segment, compute energy score as `mean(abs(peaks))` over the segment's `[start, end]` time range using the existing `waveform_peaks.json` samples (no audio re-decoding). Take the top-N segments by score (default N=15, configurable via `config/config.yaml`). Returns `[{seg_index, speaker, energy_score, text}, ...]`. Future-replaceable with a small emotion-detection model; this is v1.
   - `build_prompt(transcript, attendees, emotive_segments) -> messages` — constructs the messages list for the Anthropic API call. Transcript is formatted as `[{index, speaker, text, start, end}, ...]`; `emotive_segments` is the list returned by `extract_emotive_segments`. The prompt instructs the LLM that the emotive segments are louder than the surrounding meeting and to weight tension/statements-of-interest detection accordingly. The prompt instructs the LLM to emit both a `quote` (short verbatim snippet) and a `cited_index` (segment index) for every statement, plus `secondary` as a list of `{quote, cited_index}` objects.
   - `generate(transcript, attendees, model, peaks) -> minutes_dict` — calls the Anthropic API, parses the JSON response, calls `resolve_refs` to validate and normalise all references, and validates the schema (all required sections present; each statement has `id`, `text`, `primary_ref`; all refs are valid segment indices). Raises typed exceptions on missing API key, schema violation, and API error.
   - `resolve_refs(statements, transcript) -> statements` — post-hoc reference validation. For each statement's `cited_index`: if `quote` is a substring of `transcript[cited_index].text`, accept the index (exact match). Otherwise, fuzzy-match `quote` against all segments using `difflib.SequenceMatcher`; use the best-matching segment index and set `ref_resolution: "fuzzy"` on the statement. Drop the statement entirely (exclude from output) if the best ratio is below the similarity threshold (0.6). Apply the same exact-then-fuzzy logic to each entry in `secondary`; drop individual secondary refs that fall below threshold rather than dropping the whole statement. Output uses `primary_ref` / `secondary_refs` (canonical shape) rather than `cited_index` / `secondary` (on-wire shape).

3. **Wire backend endpoints in `app.py`.** Add four routes:
   - `POST /api/minutes/generate` — body `{attendees: [...], model: "..."}`. Reads current session transcript and `waveform_peaks.json`, calls `diarizer.minutes.generate`, returns the generated minutes object. Does NOT persist — the UI save step is explicit. Returns 4xx with a clear error message if `ANTHROPIC_API_KEY` is not set.
   - `GET /api/minutes` — returns canonical `minutes.json` from the current session dir if present; 404 otherwise.
   - `POST /api/minutes/save` — body is the full minutes object. Mirror `save_transcript`'s semantics verbatim: O_EXCL atomic create, per-day NN counter, wrap-overwrite at >99. Writes the next numbered `minutes_edit_YYYYMMDD_NN.json` snapshot; overwrites canonical `minutes.json` to match. Returns the snapshot filename. Factor a shared `_next_numbered_path` helper in `app.py` if convenient (refactor existing `save_transcript` to use it as well).
   - `GET /api/minutes/edit/<filename>` — fetches a specific named snapshot from the session dir.

4. **Implement split-pane layout in `index.html` and `style.css`.** Refactor the existing single-panel layout to a horizontal split pane. The transcript panel occupies the left side; the minutes panel occupies the right. A draggable vertical divider separates them. Divider position is persisted in `localStorage`. The minutes panel is visible by default (not hidden behind a toggle) so that click-to-sync is immediately usable when minutes are loaded.

5. **Build the minutes panel UI in `app.js`.** Implement:
   - Empty state: attendee editor (chip or textarea, pre-filled from `distinctSpeakers()`), model selector dropdown (`claude-sonnet-4-6` default, `claude-opus-4-7` option) placed next to the Generate button, Generate button.
   - First-use confirmation modal: on the first Generate click each session (checked via `sessionStorage` flag), show a modal with the text "Generating minutes will send the full transcript to Anthropic, breaking the offline guarantee. Continue?" If the user confirms, set the `sessionStorage` flag so the modal is not shown again for the rest of the session. If the user cancels, abort the Generate call.
   - On Generate (after confirmation): disable button, show spinner, call `POST /api/minutes/generate` with the selected model. On success, render result. On error (including missing API key), show an inline error message.
   - Render: each section rendered as a labelled collapsible block. Each statement rendered as a clickable line. The `tensions` section uses amber accent styling. Statements where `ref_resolution` is `"fuzzy"` are flagged with a visual indicator (e.g. a small warning icon or muted tooltip) to communicate that the anchor was resolved by fuzzy match. Statement click handler: call the existing waveform highlight API for `primary_ref` (full region highlight) and add small tick marks for each `secondary_ref`; scroll the transcript panel to the segment at `primary_ref`.
   - Save button: call `POST /api/minutes/save` with the current in-memory minutes object. Show the returned snapshot filename as a success indicator.
   - Session load: on page load, call `GET /api/minutes`; if a response arrives (200), render the existing minutes and skip the empty state.

6. **Documentation update.** Add a "Generating meeting minutes (online)" section to `README.md` covering: what the feature does, how to enable it (`ANTHROPIC_API_KEY`), and an explicit note that using it sends the full transcript text to Anthropic, breaking the project's offline guarantee. Add the same caveat (one sentence with a pointer to the README section) to `ARCHITECTURE.md`.

## Verification

- [ ] `anthropic` SDK installed and importable.
      `verify: python -c "import anthropic"`
- [ ] `diarizer/minutes.py` module importable with expected public API.
      `verify: python -c "from diarizer.minutes import generate, build_prompt, extract_emotive_segments, resolve_refs"`
- [ ] Webapp `create_app` instantiates without binding to a port.
      `verify: python -c "from diarizer.webapp.app import create_app; from pathlib import Path; create_app(Path('output/<known-test-session>'))"`
- [ ] Generate endpoint returns 4xx when API key is absent.
      `acceptance: ANTHROPIC_API_KEY= python -c "from fastapi.testclient import TestClient; from diarizer.webapp.app import create_app; from pathlib import Path; c = TestClient(create_app(Path('output/<known-test-session>'))); r = c.post('/api/minutes/generate', json={'attendees': [], 'model': 'claude-sonnet-4-6'}); assert 400 <= r.status_code < 500, r.status_code"`
- [ ] `resolve_refs` falls back to fuzzy match and tags `ref_resolution: "fuzzy"` when `cited_index` is wrong but quote text matches a different segment.
      `acceptance: python -c "from diarizer.minutes import resolve_refs; transcript = [{'index': 0, 'text': 'hello world', 'speaker': 'A', 'start': 0.0, 'end': 1.0}, {'index': 1, 'text': 'goodbye world', 'speaker': 'B', 'start': 1.0, 'end': 2.0}]; stmts = [{'id': 's1', 'text': 'farewell', 'quote': 'goodbye world', 'cited_index': 0, 'secondary': []}]; result = resolve_refs(stmts, transcript); assert result[0]['primary_ref'] == 1, result[0]; assert result[0].get('ref_resolution') == 'fuzzy', result[0]"`
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
