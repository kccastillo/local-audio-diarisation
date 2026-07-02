---
schema_version: 2
title: Refresh ARCHITECTURE.md for post-PLAN-AA0 freshness (config default, drag-drop
  launcher, test count)
type: plan
status: ready
assigned_to: sonnet
priority: low
created: 2026-07-02
created_by: Claude (Opus 4.8)
created_month: 202607
log_month: 202607
due: ''
repeatable: false
repeat_cadence: ''
linked_decisions: []
linked_inputs: []
blocked_by: ''
rollover_count: 0
triggers_plans: []
closes_thread: ''
advances_thread: ''
parent_plan_of_plans: ''
pipeline_phase: drafted
ideate_phase: ''
ideate_critique_addressed: []
ideate_iteration_count:
  self_critique: 0
  spec_refine: 0
ideate_reconcile_outcome: ''
tags:
- docs
- architecture
- freshness
files_touched:
- ARCHITECTURE.md
substrate_files:
- ARCHITECTURE.md
- README.md
- diarizer/cli.py
- diarizer/session.py
audit_acknowledgements: []
audit_disputes: []
audit_overrides: []
audit_extracted: null
pipeline_overrides: []
halt_log: []
audit_state:
  sufficiency_iterations: 1
  plan_safety_iterations: 0
  last_stage: sufficiency
  last_outcome: revision_needed
  last_audit_commit: fb7b8829
  preferred_model_override: ''
verification_state:
  state_pass: 0
  state_fail: 0
  acceptance_pass: 0
  acceptance_fail: 0
  human_pending: []
  human_verdict: pending
  human_diagnostics: ''
  human_acknowledged_failures: []
  failure_logs: {}
  human_passed: false
---

## Objective

Refresh ARCHITECTURE.md so it accurately reflects the current CLI and test surface. Three freshness corrections, no design changes: (1) `config/config.yaml` is now the built-in default config (loaded automatically when `--config` is omitted, as of PLAN-AA0), not merely "the example"; (2) document the drag-drop launcher `Transcribe.cmd` and the skip-already-done / `--force` behaviour added by PLAN-AA0; (3) reconcile the stated unit-test count (currently "79") with the real collected count and make the per-file breakdown sum to it. Scope is documentation only — ARCHITECTURE.md is the sole file edited; no source or test changes.

## Context

ARCHITECTURE.md is the design leg of a deliberate three-doc split established by `Bus/202605040100_PLAN_integrate-harness-claude-md.md` — CLAUDE.md (directives), README.md (usage), ARCHITECTURE.md (design), with the explicit rule "No content duplicated across the three files." It is load-bearing: CLAUDE.md:7 and README.md:7/:120 both point to it for design, and the `maintain-claude-md` skill audits it. It is kept, not retired; this PLAN only refreshes stale spots.

Three drifts were identified (2026-07-02):
- **Config default (ARCHITECTURE.md:114, :127).** Line 127 says "`config/config.yaml` is the example." and the module-layout tree comment (line 114) says "example config". PLAN-AA0 made `config/config.yaml` the built-in default (`DEFAULT_CONFIG` in `diarizer/cli.py`; `--config` is now an optional override). The doc should say so.
- **New CLI surface (unmentioned).** PLAN-AA0 added the repo-root `Transcribe.cmd` drag-drop launcher, a skip-already-done policy (a source file whose stem already has a completed session directory — all three of `session.json`, `transcript.json`, `source.opus` — is skipped), and a `--force` flag to reprocess. The check is `find_completed_session` in `diarizer/session.py`. None of this appears in ARCHITECTURE.md.
- **Test count (ARCHITECTURE.md:116, :163).** ARCHITECTURE states "79 unit tests" / "79 tests"; README.md:120 says "85 tests". The per-file breakdown (test_config.py (9), test_gates.py (28), test_measurement.py (8), test_output.py (9), test_pipeline.py (16), test_preprocessing.py (9)) sums to 79. Ground truth is whatever `pytest --collect-only` reports today; the doc should match it and the breakdown should sum to the total. This drift predates PLAN-AA0.

## Design Decisions Classification

**Already locked:** keep ARCHITECTURE.md (not retire); documentation-only refresh; the three-doc split (no design content moved into README/CLAUDE.md) is preserved.

**Mechanically forced:** the config-default wording follows directly from PLAN-AA0's shipped behaviour; the true test count is whatever `pytest --collect-only` collects (not a choice).

**Real judgement calls:** exact placement and wording of the launcher + skip/`--force` note (module-layout tree entry plus a one-to-two-sentence description vs a dedicated subsection) — this PLAN adds a compact tree entry plus a short note, avoiding a new heavyweight section; phrasing must match the doc's terse design voice.

## Steps

1. **Correct the config-default description (`ARCHITECTURE.md`).** In the "## Configuration" section, change the sentence "`config/config.yaml` is the example." so it states that `config/config.yaml` is the built-in default config, loaded automatically when `--config` is omitted (as of PLAN-AA0), and also serves as the example. Keep the rest of that paragraph (the "All sections are optional…" text and the section list) unchanged. In the "## Module layout" ASCII tree, update the `config.yaml` comment from "example config" to note it is the default (loaded when `--config` omitted) and example.

2. **Document the drag-drop launcher and skip/`--force` behaviour (`ARCHITECTURE.md`).** In the "## Module layout" section, add a repo-root entry for `Transcribe.cmd` — a Windows drag-drop launcher that runs one dropped file through `diarizer.cli run`. Add a concise note (one to two sentences, matching the doc's voice) — placed near the `cli.py` entry or adjacent to the session-directory description in the webapp section — stating that the CLI skips a source file whose stem already has a completed session directory (containing all three of `session.json`, `transcript.json`, `source.opus`) and that `--force` reprocesses; name `find_completed_session` in `diarizer/session.py` as the implementing check.

3. **Reconcile the unit-test count (`ARCHITECTURE.md`).** Determine the real collected test count by running `python -m pytest tests/diarizer --collect-only -q` and counting collected tests. Update both stated counts — the "## Module layout" tree line ("79 unit tests") and the "## Testing" lead-in ("79 tests covering") — to the actual number. Adjust the per-file breakdown counts (test_config.py, test_gates.py, test_measurement.py, test_output.py, test_pipeline.py, test_preprocessing.py) so they reflect current per-file collection and sum to the new total. Do NOT modify any test file or any source file — this Step edits only ARCHITECTURE.md.

## Verification

- [ ] Config-default correction present.
      `verify: grep -q "built-in default" ARCHITECTURE.md`
- [ ] Launcher documented.
      `verify: grep -q "Transcribe.cmd" ARCHITECTURE.md`
- [ ] Skip / --force behaviour documented.
      `verify: grep -q -- "--force" ARCHITECTURE.md`
- [ ] Skip-detection helper referenced.
      `verify: grep -q "find_completed_session" ARCHITECTURE.md`
- [ ] Acceptance — config-default phrasing corrected and the stale "is the example." sentence removed (requirement 1).
      `acceptance: python -c "import pathlib; t=pathlib.Path('ARCHITECTURE.md').read_text(encoding='utf-8'); assert 'built-in default' in t, 'missing built-in default'; assert 'is the example. All sections' not in t, 'stale sentence remains'; print('OK')"`
- [ ] Acceptance — stated test count equals the real collected count and appears in both places (requirement 3).
      `acceptance: python -c "import subprocess,re,pathlib; r=subprocess.run(['python','-m','pytest','tests/diarizer','--collect-only','-q'],capture_output=True,text=True); c=len([l for l in r.stdout.splitlines() if '::' in l]); assert c>0, (r.stdout[-300:]+r.stderr[-300:]); t=pathlib.Path('ARCHITECTURE.md').read_text(encoding='utf-8'); assert re.search(rf'{c} unit tests', t), f'no \"{c} unit tests\"'; assert re.search(rf'{c} tests', t), f'no \"{c} tests\"'; print('OK', c)"`

## Executor Notes

*Populated after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**
