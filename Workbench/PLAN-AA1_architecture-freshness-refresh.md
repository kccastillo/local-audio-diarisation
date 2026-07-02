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
  sufficiency_iterations: 2
  plan_safety_iterations: 2
  last_stage: plan_safety
  last_outcome: success
  last_audit_commit: 9cdaa873
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

Refresh ARCHITECTURE.md so it accurately reflects the current CLI and test surface. Three freshness corrections, no design changes: (1) `config/config.yaml` is now the built-in default config (loaded automatically when `--config` is omitted, as of PLAN-AA0), not merely "the example"; (2) document the drag-drop launcher `Transcribe.cmd` and the skip-already-done / `--force` behaviour added by PLAN-AA0; (3) reconcile the stated unit-test count with the real collected count and rebuild the per-file breakdown to enumerate all current test files (8 files / ~111 tests as of 2026-07-02, including test_crosstalk.py and test_webapp_smoke.py which were absent from the prior breakdown) so every per-file count sums exactly to the live total. Scope is documentation only — ARCHITECTURE.md is the sole file edited; no source or test changes.

## Context

ARCHITECTURE.md is the design leg of a deliberate three-doc split established by `Bus/202605040100_PLAN_integrate-harness-claude-md.md` — CLAUDE.md (directives), README.md (usage), ARCHITECTURE.md (design), with the explicit rule "No content duplicated across the three files." It is load-bearing: CLAUDE.md:7 and README.md:7/:120 both point to it for design, and the `maintain-claude-md` skill audits it. It is kept, not retired; this PLAN only refreshes stale spots.

Three drifts were identified (2026-07-02):
- **Config default (ARCHITECTURE.md:114, :127).** Line 127 says "`config/config.yaml` is the example." and the module-layout tree comment (line 114) says "example config". PLAN-AA0 made `config/config.yaml` the built-in default (`DEFAULT_CONFIG` in `diarizer/cli.py`; `--config` is now an optional override). The doc should say so.
- **New CLI surface (unmentioned).** PLAN-AA0 added the repo-root `Transcribe.cmd` drag-drop launcher, a skip-already-done policy (a source file whose stem already has a completed session directory — all three of `session.json`, `transcript.json`, `source.opus` — is skipped), and a `--force` flag to reprocess. The check is `find_completed_session` in `diarizer/session.py`. None of this appears in ARCHITECTURE.md.
- **Test count (ARCHITECTURE.md:116, :163).** ARCHITECTURE states "79 unit tests" / "79 tests"; the per-file breakdown lists only 6 files (test_config.py (9), test_gates.py (28), test_measurement.py (8), test_output.py (9), test_pipeline.py (16), test_preprocessing.py (9)) summing to 79. Live ground truth (verified 2026-07-02 via `python -m pytest tests/diarizer --collect-only -q`, pytest 9.0.3 / Python 3.12.10): 111 tests across 8 files — test_config.py (14), test_crosstalk.py (10), test_gates.py (29), test_measurement.py (8), test_output.py (9), test_pipeline.py (16), test_preprocessing.py (9), test_webapp_smoke.py (16). test_crosstalk.py and test_webapp_smoke.py are entirely absent from the current breakdown. NOTE: README.md's "85 tests" figure is a separate, known drift and is OUT OF SCOPE for this PLAN — README.md is not edited here.

**Test count context — H4 calling-convention note.** The acceptance checks for requirement (3) rely on pytest 9.0.3's `--collect-only -q` output format. Under this version, `--collect-only -q` emits a tree of `<Module>/<Function>/<Coroutine>` entries followed by a terminal summary line of the form `N tests collected in ...s`. The acceptance parses the `N tests collected` summary line — it does NOT rely on `::` node-ID lines (which are absent in this pytest version's tree collect format). Test-runner async/sync posture: the existing tests are a mix; no new tests are added by this PLAN so async/sync posture changes are out of scope.

## Design Decisions Classification

**Already locked:** keep ARCHITECTURE.md (not retire); documentation-only refresh; the three-doc split (no design content moved into README/CLAUDE.md) is preserved.

**Mechanically forced:** the config-default wording follows directly from PLAN-AA0's shipped behaviour; the true test count is whatever `pytest --collect-only` collects (not a choice).

**Real judgement calls:** exact placement and wording of the launcher + skip/`--force` note (module-layout tree entry plus a one-to-two-sentence description vs a dedicated subsection) — this PLAN adds a compact tree entry plus a short note, avoiding a new heavyweight section; phrasing must match the doc's terse design voice.

## Steps

1. **Correct the config-default description (`ARCHITECTURE.md`) and bump `last_updated`.** In the "## Configuration" section, change the sentence "`config/config.yaml` is the example." so it states that `config/config.yaml` is the built-in default config, loaded automatically when `--config` is omitted (as of PLAN-AA0), and also serves as the example. Keep the rest of that paragraph (the "All sections are optional…" text and the section list) unchanged. In the "## Module layout" ASCII tree, update the `config.yaml` comment from "example config" to note it is the default (loaded when `--config` omitted) and example. Also update ARCHITECTURE.md's frontmatter `last_updated:` field from `2026-05-04` to `2026-07-03`.

2. **Document the drag-drop launcher and skip/`--force` behaviour (`ARCHITECTURE.md`).** In the "## Module layout" section, add a repo-root entry for `Transcribe.cmd` — a Windows drag-drop launcher that runs one dropped file through `diarizer.cli run`. Add a concise note (one to two sentences, matching the doc's voice) — placed near the `cli.py` entry or adjacent to the session-directory description in the webapp section — stating that the CLI skips a source file whose stem already has a completed session directory (containing all three of `session.json`, `transcript.json`, `source.opus`) and that `--force` reprocesses; name `find_completed_session` in `diarizer/session.py` as the implementing check.

3. **Reconcile the unit-test count (`ARCHITECTURE.md`).** Using the verified ground-truth numbers recorded in the Context 'Test count' bullet — total 111 tests across 8 files: test_config.py (14), test_crosstalk.py (10), test_gates.py (29), test_measurement.py (8), test_output.py (9), test_pipeline.py (16), test_preprocessing.py (9), test_webapp_smoke.py (16) — edit ARCHITECTURE.md with Read/Edit only. Do NOT run pytest or any shell command; you have no shell. These numbers were verified live by the orchestrator on 2026-07-03, and the acceptance: items in the Verification section re-derive the live count in the orchestrator's parent context to catch any drift.

   (a) Update BOTH stated totals to 111: the "## Module layout" tree line ("79 unit tests" -> "111 unit tests") and the "## Testing" lead-in ("79 tests covering" -> "111 tests covering").

   (b) Rebuild the per-file breakdown under "## Testing" to enumerate all eight files with the counts above — this means adding the two currently-absent files (test_crosstalk.py, test_webapp_smoke.py) and correcting the stale counts (test_config.py 9->14, test_gates.py 28->29) — so the per-file counts sum exactly to 111. For each of the two added files, write a short accurate parenthetical description by reading that test file's contents (test_crosstalk.py: cross-talk / speaker-overlap handling; test_webapp_smoke.py: webapp smoke tests) — reading source/test files is allowed; only editing them is forbidden.

   (c) Do NOT modify any test file or source file — this Step edits ARCHITECTURE.md only.

## Verification

- [ ] Config-default correction present.
      `verify: python -c "import pathlib; assert 'built-in default' in pathlib.Path('ARCHITECTURE.md').read_text(encoding='utf-8'); print('OK')"`
- [ ] Launcher documented.
      `verify: python -c "import pathlib; assert 'Transcribe.cmd' in pathlib.Path('ARCHITECTURE.md').read_text(encoding='utf-8'); print('OK')"`
- [ ] Skip / --force behaviour documented.
      `verify: python -c "import pathlib; assert '--force' in pathlib.Path('ARCHITECTURE.md').read_text(encoding='utf-8'); print('OK')"`
- [ ] Skip-detection helper referenced.
      `verify: python -c "import pathlib; assert 'find_completed_session' in pathlib.Path('ARCHITECTURE.md').read_text(encoding='utf-8'); print('OK')"`
- [ ] Acceptance — config-default phrasing corrected and the stale "is the example." sentence removed (requirement 1).
      `acceptance: python -c "import pathlib; t=pathlib.Path('ARCHITECTURE.md').read_text(encoding='utf-8'); assert 'built-in default' in t, 'missing built-in default'; assert 'is the example. All sections' not in t, 'stale sentence remains'; print('OK')"`
- [ ] Acceptance — stated total equals live collected count, in both the layout line and the Testing lead-in (requirement 3).
      `acceptance: python -c "import subprocess,re,pathlib; r=subprocess.run(['python','-m','pytest','tests/diarizer','--collect-only','-q'],capture_output=True,text=True); m=re.search(r'(\d+) tests? collected', r.stdout); assert m, (r.stdout[-300:]+r.stderr[-300:]); c=int(m.group(1)); t=pathlib.Path('ARCHITECTURE.md').read_text(encoding='utf-8'); assert re.search(rf'{c} unit tests', t), f'no {c} unit tests'; assert re.search(rf'{c} tests covering', t), f'no {c} tests covering'; print('OK', c)"`
- [ ] Acceptance — per-file breakdown sums to the live collected total (requirement 3).
      `acceptance: python -c "import subprocess,re,pathlib; r=subprocess.run(['python','-m','pytest','tests/diarizer','--collect-only','-q'],capture_output=True,text=True); c=int(re.search(r'(\d+) tests? collected', r.stdout).group(1)); t=pathlib.Path('ARCHITECTURE.md').read_text(encoding='utf-8'); per=[int(n) for n in re.findall(r'test_\w+\.py \((\d+)\)', t)]; assert per, 'no per-file counts'; assert sum(per)==c, f'breakdown {sum(per)} != collected {c}'; print('OK', sum(per))"`

## Executor Notes

*Populated after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**
