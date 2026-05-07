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
pipeline_phase: "drafted"
audit_state:
  sufficiency_iterations: 1
  plan_safety_iterations: 0
  last_stage: sufficiency
  last_outcome: revision_needed
---

## Objective

Add a second panel beside the running transcript in the diarizer webapp that displays generated meeting minutes. Each minute statement is anchored to one or more transcript segments so that clicking a statement highlights the corresponding waveform region and scrolls the transcript to the primary anchor. Minutes are generated via an opt-in external Anthropic API call and persisted using the same numbered-snapshot convention as transcript edits.

## Context

The webapp currently shows a transcript panel with synced waveform playback and speaker-label editing. This feature adds a companion minutes panel to the right, making the webapp useful for post-meeting review without leaving the tool. The feature explicitly breaks the project's offline-by-default guarantee — this is intentional, opt-in, and must be clearly documented.

**Offline guarantee caveat:** Using the Generate function sends the full transcript to the Anthropic API. This is opt-in (button press + `ANTHROPIC_API_KEY` env var required). The webapp must not call the API unless explicitly triggered, and the README and ARCHITECTURE must document the caveat prominently.

**Statement schema** (canonical shape for each statement emitted by the LLM and stored in `minutes.json`):

```
{ id, text, quote, primary_ref, secondary_refs, kind?, ref_resolution?: "exact"|"fuzzy" }
```

- `quote` — short verbatim snippet from the transcript that the statement is grounded in; emitted by the LLM, used by `resolve_refs` for post-hoc index validation.
- `primary_ref` — integer segment index; resolved and validated by `resolve_refs`.
- `secondary_refs` — list of integer segment indices; each resolved independently by `resolve_refs`.
- `kind` — optional; used for the `tensions` section to distinguish tension statements.
- `ref_resolution` — optional; present only when fuzzy-match fallback was used for `primary_ref`. Absent (or `"exact"`) when substring match succeeded.

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

1. **Add `anthropic` dependency and config defaults.** Update `pyproject.toml` to add `anthropic` to the dependency list. Extend `config/config.yaml` with a `minutes:` block containing: `default_model`, `opus_model`, `top_n_amplitude_windows`, `generation_enabled`.

2. **Build `diarizer/minutes.py`.** Implement:
   - `build_prompt(transcript, attendees, crosstalk_summary) -> messages` — constructs the messages list for the Anthropic API call. Transcript is formatted as `[{index, speaker, text, start, end}, ...]`; crosstalk_summary is the top-N amplitude windows. The prompt instructs the LLM to emit both a `quote` (short verbatim snippet) and a `cited_index` (segment index) for every statement, plus `secondary_refs` as a list of `{quote, cited_index}` objects.
   - `extract_top_n_crosstalk(crosstalk_data, n) -> list` — selects and returns the N highest-amplitude windows from the session crosstalk data.
   - `generate(transcript, attendees, model) -> minutes_dict` — calls the Anthropic API, parses the JSON response, calls `resolve_refs` to validate and normalise all references, and validates the schema (all required sections present; each statement has `id`, `text`, `primary_ref`; all refs are valid segment indices). Raises typed exceptions on missing API key, schema violation, and API error.
   - `resolve_refs(statements, transcript) -> statements` — post-hoc reference validation. For each statement's `cited_index`: if `quote` is a substring of `transcript[cited_index].text`, accept the index (exact match). Otherwise, fuzzy-match `quote` against all segments using `difflib.SequenceMatcher`; use the best-matching segment index and set `ref_resolution: "fuzzy"` on the statement. Drop the statement entirely (exclude from output) if the best ratio is below the similarity threshold (0.6). Apply the same exact-then-fuzzy logic to each entry in `secondary_refs`; drop individual secondary refs that fall below threshold rather than dropping the whole statement.

3. **Wire backend endpoints in `app.py`.** Add four routes:
   - `POST /api/minutes/generate` — body `{attendees: [...], model: "..."}`. Reads current session transcript + crosstalk data, calls `diarizer.minutes.generate`, returns the generated minutes object. Does NOT persist — the UI save step is explicit. Returns 4xx with a clear error message if `ANTHROPIC_API_KEY` is not set.
   - `GET /api/minutes` — returns canonical `minutes.json` from the current session dir if present; 404 otherwise.
   - `POST /api/minutes/save` — body is the full minutes object. Writes the next numbered `minutes_edit_YYYYMMDD_NN.json` snapshot; overwrites canonical `minutes.json` to match. Returns the snapshot filename.
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
