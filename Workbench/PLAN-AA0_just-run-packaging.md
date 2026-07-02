---
schema_version: 2
title: Package the pipeline to 'just run' — drag-drop launcher, default config, skip-already-done
type: plan
status: ready
assigned_to: sonnet
priority: medium
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
- packaging
- cli
- dx
files_touched:
- diarizer/cli.py
- diarizer/session.py
- Transcribe.cmd
- README.md
- CLAUDE.md
substrate_files:
- diarizer/cli.py
- diarizer/config.py
- diarizer/session.py
- config/config.yaml
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
  last_audit_commit: 171a06bc
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
Make transcription "just run" against a single source audio file with no flags to remember. Three concrete changes: (1) drag an audio file onto a launcher and it processes that one file; (2) `config/config.yaml` becomes the built-in default so `--config` is never required — closing the footgun where an omitted flag silently disables diarisation and speaker attribution; (3) if the dropped file already has completed output, skip it rather than reprocess. This turns a two-command, flag-sensitive workflow into a single drag-and-drop gesture that a non-technical operator can use safely.

## Context
Today the pipeline runs one file at a time via `python -m diarizer.cli run --input <file> --config config/config.yaml`. Two facts make this fragile:

- **The `--config` footgun.** `_add_run_args` sets `--config` default to `None` (`diarizer/cli.py:30`), and `load_config(None)` returns baked-in dataclass defaults (`diarizer/config.py:125-126`) rather than the project YAML. The baked-in `AuthConfig.token_path` is `None`, so the Hugging Face token at `D:/Projects/Tokens/kc_diariser.txt` (declared only in `config/config.yaml`) never resolves, diarisation silently skips, and the transcript has no speaker labels. This is recorded as a standing project rule ("always pass `--config config/config.yaml`"). Making the project config the default removes the rule's reason to exist.
- **No entry point for a non-technical run.** Processing two files this session was two manual background runs. The operator's stated want is: drag one audio file onto something → it runs.

Requirement was agreed with the operator in conversation (2026-07-02) before this PLAN was authored, per requirement-before-solution:
- **Entry point:** drag an audio file onto a launcher (Windows drag-drop passes the file path as `%1`).
- **Scope:** process that one dragged file (not a batch of `./input/`).
- **Config:** `config/config.yaml` is the default; `--config` becomes an optional override only.
- **Re-run:** skip a file that already has completed output; provide `--force` as the escape hatch to reprocess.

Substrate ground truth (read during authoring): a completed run is identifiable by a directory `<output_dir>/<stem>_<timestamp>/` containing all three of `session.json`, `transcript.json`, `source.opus` (written by `create_session_dir` + `_run`, `diarizer/session.py:19-46`, `diarizer/cli.py:126-151`). `diarizer/session.py` imports only the standard library, so the skip-detection helper is placed there to keep it unit-testable without importing the heavy transcription pipeline (`diarizer/cli.py:21` pulls in `Pipeline`).

**Launcher environment:** the launcher invokes the venv interpreter directly by absolute path. The canonical environment is the `venv` directory at the project root; `venv-old` and `venv-v2-test` are preserved but not the target.

Platform note: the launcher (`Transcribe.cmd`) and the drag-drop gesture are Windows-specific by nature; the corresponding human-verification item is annotated `# platform: windows`. All automated `verify:`/`acceptance:` items are portable Python or POSIX `test`.

## Design Decisions Classification

**Already locked** (operator affirmed during the 2026-07-02 requirement conversation):
- Drag-drop launcher as the entry point.
- Single dragged file is the unit of work (not batch-the-folder).
- `config/config.yaml` is the default config; `--config` is override-only.
- Skip a file that already has completed output.

**Mechanically forced** (no meaningful alternative given the locked decisions):
- Resolve the default config path from `__file__` (`Path(__file__).resolve().parents[1] / "config" / "config.yaml"`), not from the current working directory — drag-drop launches with an unpredictable CWD, so a CWD-relative default would break.
- A `--force` flag — a skip-by-default policy is only usable if there is a deterministic way to override it.
- "Done" is defined as a session directory holding all three artefacts; a partial directory from a crashed run must NOT count as done, so a re-run can recover it.

**Real judgement calls** (surfaced for design-review; operator may prefer otherwise):
- Launcher filename `Transcribe.cmd` at the project root (vs `run.cmd`, `Diarize.cmd`, or a `.lnk` shortcut).
- On multiple files dropped at once: process only the first and print a note that the rest were ignored (honours the single-file scope) — vs looping over all dropped files. This PLAN takes the first-only behaviour.
- Placing `find_completed_session` in `diarizer/session.py` (chosen, for stdlib-only testability) vs in `diarizer/cli.py` alongside its caller.

## Steps

1. **Default the config to the project YAML (`diarizer/cli.py`).** Add a module-level constant near the top of the file: `DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"`. In `_add_run_args`, change the `--config` argument's `default=None` to `default=DEFAULT_CONFIG` and update its help text to `"Path to YAML config (default: <repo>/config/config.yaml)."`. Do not change any other argument.

2. **Guard a missing config file in `_run` (`diarizer/cli.py`).** Immediately after the existing `if not args.input.exists()` block in `_run`, add: `if not args.config.exists(): print(f"Config not found: {args.config}", file=sys.stderr); return 2`. This prevents an unhandled `FileNotFoundError` traceback from `load_config` when the default file has been deleted or an explicit `--config` path is mistyped. (`args.config` is always a `Path` now that the default is a `Path`.)

3. **Add the skip-detection helper (`diarizer/session.py`).** Add a function `find_completed_session(base_output_dir: Path | str, stem: str) -> Path | None`. It must: iterate directories directly under `base_output_dir` whose name matches `f"{stem}_"` followed by the timestamp pattern `\d{8}_\d{6}` (use `re` — mirror the `"%Y%m%d_%H%M%S"` format written at `session.py:29`); of those, return the lexicographically-newest directory that contains ALL of `session.json`, `transcript.json`, and `source.opus`; return `None` if none qualifies or `base_output_dir` does not exist. Import only from the standard library (`re`, `pathlib` — already imported). Add a docstring noting: (1) a partial/crashed session (missing any of the three artefacts) does not count as complete; (2) matching is stem-based and extension-insensitive (e.g. `foo.wav` and `foo.m4a` share a stem).

4. **Wire skip + `--force` into the run path (`diarizer/cli.py`).** In `_add_run_args`, add `p.add_argument("--force", action="store_true", help="Reprocess even if a completed session for this file already exists.")`. In `_run`, after `config = _apply_overrides(config, args)` (so `output_dir` is known) and before `pipeline = Pipeline(config)`, add: if `not args.force`, call `find_completed_session(Path(config.paths.output_dir), args.input.stem)`; if it returns a directory, `print(f"Already processed: {existing}. Use --force to reprocess.")` and `return 0`. Import `find_completed_session` from `diarizer.session` (extend the existing `from diarizer.session import create_session_dir` line).

5. **Create the drag-drop launcher `Transcribe.cmd` at the project root.** Batch behaviour: `@echo off`; `cd /d "%~dp0"` (project root = the launcher's own directory); if `"%~1"==""` print a one-line usage message ("Drag an audio file onto this launcher to transcribe it."), `pause`, and exit; otherwise invoke the venv interpreter by absolute path — `"%~dp0venv\Scripts\python.exe" -m diarizer.cli run --input "%~1"` (no `--config` needed now) — then `pause` so the console window stays open to read progress and any error. Process only the first dropped file (`%~1`); if additional arguments are present, print a note that only the first file is processed. Do not activate a shell environment; calling the venv python directly is sufficient for double-click / drag-drop.

6. **Document the "just run" path in `README.md`.** Add a section with the literal heading `## Just run (drag-and-drop)` stating: drag an audio file onto `Transcribe.cmd` to transcribe it; `--config` is no longer required because `config/config.yaml` is the default; an already-processed file is skipped, and `--force` reprocesses it. Keep it to a few lines consistent with the surrounding README voice and AU spelling.

7. **Update the stale CLAUDE.md project rule.** In `CLAUDE.md`, under "## Project operating rules", edit the "**Diarizer CLI:**" bullet. It currently reads: "always pass `--config config/config.yaml` to `diarizer.cli run` — otherwise the HF token (`auth.token_path`) does not resolve and diarisation silently skips with no speaker attribution." Replace it with a bullet stating that `config/config.yaml` is now the built-in default and `--config` is an optional override only, so the token resolves without any flag; note that the old always-pass-`--config` footgun is closed as of this PLAN. Keep AU spelling and the existing terse bullet voice. **CRITICAL:** The "## Project operating rules" section is in the project-owned area ABOVE the `plan-foundry:init-plan-foundry:start` sentinel block. Do NOT edit anything inside or below that sentinel-managed block — plan_foundry owns those lines and will overwrite any changes on the next `/plan-foundry-sync`. **Out-of-scope for this executor:** The operator's auto-memory file `feedback_diarizer_config_flag.md` (under `~/.claude/projects/`) becomes stale with this change. Flag this for the orchestrator/operator to update or retire at completion—the file is outside the repo, so it is not an executor edit.

## Verification
- [ ] `DEFAULT_CONFIG` constant added to the CLI.
      `verify: grep -q "DEFAULT_CONFIG" diarizer/cli.py`
- [ ] Skip-detection helper exists.
      `verify: grep -q "def find_completed_session" diarizer/session.py`
- [ ] `--force` flag added to the CLI.
      `verify: grep -q "force" diarizer/cli.py`
- [ ] Launcher file exists at the project root.
      `verify: test -f Transcribe.cmd`
- [ ] README documents the drag-drop path.
      `verify: grep -qi "drag" README.md`
- [ ] Acceptance — the `run` parser defaults `--config` to the project `config.yaml` (requirement 3).
      `acceptance: python -c "from diarizer.cli import _build_parser; import pathlib; a=_build_parser().parse_args(['run','--input','x.wav']); assert a.config is not None and pathlib.Path(a.config).name=='config.yaml', a.config; print('OK', a.config)"`
- [ ] Acceptance — skip-detection recognises a completed session and ignores a non-existent / partial one (requirement 4).
      `acceptance: python -c "import tempfile,pathlib,diarizer.session as s; d=pathlib.Path(tempfile.mkdtemp()); sd=d/'foo_20260101_000000'; sd.mkdir(); [ (sd/n).write_text('x') for n in ('session.json','transcript.json','source.opus') ]; part=d/'bar_20260101_000000'; part.mkdir(); (part/'session.json').write_text('x'); assert s.find_completed_session(d,'foo')==sd; assert s.find_completed_session(d,'bar') is None; assert s.find_completed_session(d,'nope') is None; print('OK')"`
- [ ] Human — drag a real `.m4a` onto `Transcribe.cmd`: it runs end-to-end and the transcript carries speaker labels (diarisation ran); dragging an already-processed file prints the skip message and exits without reprocessing; dragging the same already-processed file with `--force` (or via a `--force`-enabled invocation) reprocesses it end-to-end.
      `verify: human # platform: windows`

## Executor Notes
*Populated after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**
