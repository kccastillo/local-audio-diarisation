---
title: "Decommission v1 after hardware-validation; v2 becomes the canonical pipeline"
type: bus-plan
status: done
assigned_to: ""
priority: high
created: 2026-05-04
created_by: opus
created_month: 202605
log_month: 202605
due: ""
repeatable: false
linked_decisions:
  - "Operator constraint: no A/B testing — v1 disappears in a single PR (PLAN 202605040200 D14)"
  - "v2 must be hardware-validated before v1 is touched"
linked_inputs:
  - "202605040220_PLAN_v2-implementation.md (v2 build)"
  - "Hardware-validation evidence (TBD; produced at top of next session)"
blocked_by: ""
rollover_count: 0
parent_plan_of_plans: "202605040200_PLAN_diarizer-v2-rearchitecture.md"
---

## Objective

Delete v1, promote v2 to canonical pipeline, rewrite ARCHITECTURE.md as the v2 design, and merge `feature/speaker-separation` to `main`. Single PR; v1 vanishes in one squash-merge per the no-A/B operator constraint.

This PLAN is **blocked** until v2 is hardware-validated. Do not start executing without the evidence in Step 0.

## Context

v1 files to delete (the canonical v1 surface — anything not on this list stays):
- `processors/` — entire directory (`base_processor.py`, `vad_processor.py`, `diarisation_processor.py`, `transcription_processor.py`, plus `__init__.py`).
- `audio/` — `audio_cleaner.py` and the package init.
- `utils/` — `datatypes.py`, `memory_monitor.py`, `display_manager.py`, `transcription_writer.py`, `__init__.py`. (Verify nothing in v2/ imports from utils — none should.)
- `config/config_manager.py` — singleton.
- `config/config.yaml` — v1 config (rename `config/config.v2.yaml` → `config/config.yaml` after deletion).
- `run_diariser.py` — replaced by `python -m v2.cli`.
- `tests/test_0_config.py`, `tests/test_1_*.py`, `tests/test_2_*.py`, `tests/test_3_*.py`, `tests/test_4_end_to_end.py` (and any other v1 test files at `tests/test_*.py`).
- `requirements.txt` — replaced by `requirements.v2.txt` renamed to `requirements.txt`.

v1 files to keep (verify each):
- `Bus/`, `memory/` — project state.
- `README.md`, `CLAUDE.md`, `ARCHITECTURE.md` — documentation (rewritten in this PLAN).
- `LICENSE`, `.gitignore`, `.gitattributes` if present.
- `recordings/`, `output/`, `temp/`, `logs/` — runtime data dirs (gitignored anyway).

## Steps

### Step 0 — Confirm hardware validation passed (NOW SATISFIED)

The four conditions are confirmed in `Bus/202605040250_RESEARCH_v2-smoke-run.md`:

- ✓ `pip install -r requirements.v2.txt` (with the documented `--index-url` two-step) clean on the target machine. After three transitive-dep fixes (`huggingface_hub<1.0`, `matplotlib`, `speechbrain==1.0.3`) — pinned in the file.
- ✓ `python -m v2.cli --input recordings/20260410_*.mp4 --auth-token <hf>` end-to-end, with diarisation, completed in ~1 min 42 s on the operator's RTX 3070.
- ✓ VRAM stayed under 8 GB (post-run residual 4.6 GB; sequential load/unload pattern bounds peak by the larger model).
- ⏳ **Operator subjective sign-off** — RESEARCH 250 is the evidence; operator confirms quality before this Step is fully closed.

If operator flags transcript quality as unacceptable, halt this PLAN and diagnose in a separate fix PLAN. Otherwise proceed to Step 1.

### Step 1 — Promote v2 module names to canonical positions

Decide before executing this Step: do we keep the `v2/` package name, or rename it to a flat module structure (e.g. `diarizer/`)?

**Recommendation:** rename `v2/` → `diarizer/`. The `v2` prefix only made sense while v1 lived in the same tree. After v1 deletion, `diarizer/` is the natural top-level name.

If rename chosen:
- `git mv v2 diarizer`
- Update imports across `diarizer/__init__.py`, `diarizer/cli.py`, `diarizer/pipeline.py`, `diarizer/output.py`, `diarizer/preprocessing.py`, `tests/v2/`. Replace `from v2.` with `from diarizer.`.
- Rename `tests/v2/` → `tests/`.
- Update `.gitignore` `tests/*` + `!tests/v2/` → `tests/*` + `!tests/test_*.py` (or simply remove the `tests/*` ignore entirely now that tests are first-class).
- Update README v2 section: change `python -m v2.cli` → `python -m diarizer.cli` and `v2/` paths → `diarizer/`.
- Update `Bus/202605040220_PLAN_v2-implementation.md` and this PLAN's prose.

**verify:** `python -m diarizer.cli --help` succeeds; `pytest tests/ -v` all pass.

### Step 2 — Delete v1 source files

Delete in this order (so import-order accidents surface early):
1. `run_diariser.py` (entrypoint first — nothing else imports it).
2. `tests/test_0_config.py`, `tests/test_1_*.py`, `tests/test_2_*.py`, `tests/test_3_*.py`, `tests/test_4_end_to_end.py` (any remaining v1 tests).
3. `processors/` (whole directory).
4. `audio/` (whole directory).
5. `utils/` (whole directory).
6. `config/config_manager.py`.
7. `config/config.yaml`.

After each deletion, run `pytest tests/ -v` to confirm nothing in v2 silently depended on a v1 path.

**verify:** none of the listed paths exist; `pytest tests/ -v` all pass; `python -m diarizer.cli --help` still works.

### Step 3 — Promote v2 config to canonical name

`git mv config/config.v2.yaml config/config.yaml`. Update `diarizer/cli.py` default-path resolution if the path is hard-coded (it isn't — load_config defaults to baked-in dataclass values, not a path).

**verify:** `config/config.yaml` is the v2 config; `config/config.v2.yaml` no longer exists.

### Step 4 — Promote requirements file

`git mv requirements.v2.txt requirements.txt`. Confirm the file is UTF-8 (`file requirements.txt` should report ASCII or UTF-8, not UTF-16). The old UTF-16 v1 requirements.txt is gone after Step 2's deletion implicitly — Step 2 doesn't list it, so add it explicitly: also delete the old `requirements.txt` *before* the rename. To avoid the conflict, do this:

- `rm requirements.txt` (the v1 UTF-16 file).
- `git mv requirements.v2.txt requirements.txt`.

**verify:** `requirements.txt` is UTF-8; contains `whisperx`; does NOT contain `openai-whisper`, `pyannote.audio==3.3.2`, etc.

### Step 5 — Rewrite ARCHITECTURE.md as v2 design

Replace the v1 architecture content in `ARCHITECTURE.md` with v2 design:
- Single-pipeline shape (WhisperX wrapper).
- Module layout: `diarizer/{config,preprocessing,pipeline,output,cli}.py`.
- Configuration model (dataclass + YAML, no singleton).
- Audio preprocessing approach (FFmpeg, optional).
- Diarisation backend (pyannote via WhisperX).
- VRAM management (single-shot — drop the v1 load/unload pattern).
- Error handling and logging.
- Testing structure.
- Recovery: `git checkout v1-final` for the v1 baseline.

Reuse the v2 RESEARCH and PLAN 220 Executor Notes as source material.

**verify:** ARCHITECTURE.md describes the v2 design only; no v1 references except the recovery pointer.

### Step 6 — Update README.md

- Strip the "Diarizer v2 (in development)" section header — v2 IS Diarizer now. Keep the substantive content (usage, layout) but reflow under the main Setup/Run sections.
- Remove the "v1 codebase will be tagged" paragraph (already done — tag exists).
- Replace v1 setup/run/configuration sections with v2 equivalents (the v2 commands).
- Add a "Recovery" line: `git checkout v1-final` retrieves the v1 baseline.

**verify:** README has no `v2/` references; runs are described via `python -m diarizer.cli`.

### Step 7 — Update CLAUDE.md project overview line

Adjust the project overview line to drop the "v2 in development" framing (now reality):

> Offline speaker-diarisation + transcription pipeline (WhisperX). See [README.md](README.md) for usage, [ARCHITECTURE.md](ARCHITECTURE.md) for design.

**verify:** CLAUDE.md project overview matches above; ≤80 lines total.

### Step 8 — Retire the v2-build PLANs

Move PLAN 202605040200 (rearchitecture plan-of-plans), PLAN 202605040220 (implementation), this PLAN (decommission), and the two RESEARCH files (requirements + hardware-validation) to `Retired/`. Update LOG status table.

**verify:** Bus/ has only the LOG file plus any new in-flight PLANs; Retired/ contains the moved files.

### Step 9 — Commit and PR

Single squash-ready commit message describing the v1 → v2 cutover. Push.

Open PR `feature/speaker-separation → main` with a body that:
- Links to the v1-final tag.
- Lists deleted v1 paths.
- Highlights the dependency change (whisperx replaces openai-whisper + pyannote pinned versions).
- Links to ARCHITECTURE.md for the v2 design.
- Asks for operator merge approval.

**verify:** PR exists and is reviewable.

### Step 10 — Merge

Squash-merge to main on operator approval. Delete the feature branch.

**verify:** `main` HEAD contains v2; `git log v1-final..main -- processors/` shows only deletions; `python -m diarizer.cli --help` works on a fresh clone of main.

## Verification (full PLAN)

- [ ] Step 0 RESEARCH file exists; operator confirmed v2 works on real audio at ≤8 GB VRAM.
- [ ] `v2/` renamed to `diarizer/` (or naming decision recorded otherwise).
- [ ] All v1 source paths deleted.
- [ ] `requirements.txt` is the v2 (whisperx) file, UTF-8.
- [ ] `config/config.yaml` is the v2 config.
- [ ] ARCHITECTURE.md describes v2 only.
- [ ] README.md describes v2 only.
- [ ] CLAUDE.md overview line updated.
- [ ] PLANs 200/220/300 + 2 RESEARCH files moved to Retired/.
- [ ] PR opened.
- [ ] PR merged to main.

## Acceptance (spec)

A fresh clone of `main` after this PLAN merges:
- Has no `processors/`, `audio/`, `utils/`, `run_diariser.py`, `config/config_manager.py`, or v1 tests.
- Installs via `pip install -r requirements.txt` (whisperx).
- Runs via `python -m diarizer.cli --input <audio>` (or whatever module name Step 1 chose).
- Has ARCHITECTURE.md describing v2.
- Recoverable to v1 via `git checkout v1-final`.

## Out of scope

- Any v2 feature work, model tuning, performance optimisation.
- Reviving v1.
- Documentation polish beyond the README/ARCHITECTURE/CLAUDE updates listed.

## Executor Notes

**Executed:** 2026-05-04 (operator approved after reviewing diarisation smoke output and Teams-vs-v2 quality comparison).
**Outcome:** done.

**Step 0** — Hardware validation **PASSED** end-to-end with diarisation. RESEARCH 250 captures the full evidence (34:30 audio → 469 segments → 7 speakers detected in ~1m42s on RTX 3070 with VRAM under 8 GB). Operator quality sign-off received: "looks good".

**Step 1** — Renamed `v2/` → `diarizer/` and `tests/v2/` → `tests/diarizer/` via `git mv`. Bulk-rewrote 10 source files to update `from v2.X` imports → `from diarizer.X`. Updated `pyproject.toml` (`[project.scripts] diarizer = "diarizer.cli:main"`, `[tool.setuptools.packages.find] include = ["diarizer*"]`, `[tool.pytest.ini_options] testpaths = ["tests/diarizer"]`). Updated `.gitignore` (`!tests/diarizer/`). 79/79 tests still pass.

**Step 2** — Deleted v1 source: `run_diariser.py`, `processors/`, `audio/`, `utils/` (whole directories), `config/config_manager.py`, `config/config.yaml` (v1 version). Also deleted v1 tests at top of `tests/` (test_0_config through test_4_end_to_end — these were untracked because of the gitignore, so removed via `rm` not `git rm`).

**Step 3** — `git mv config/config.v2.yaml config/config.yaml`. New canonical config name.

**Step 4** — Deleted v1 `requirements.txt` (UTF-16-encoded), then `git mv requirements.v2.txt requirements.txt`. Updated header comments to drop "v2" framing.

**Step 5** — `ARCHITECTURE.md` rewritten as v2 design (six stages with gate severities, module layout, validation gate table, configuration sections, memory/VRAM behaviour, output formats, testing, recovery pointer to `v1-final`).

**Step 6** — `README.md` rewritten: dropped "in development" framing, single Setup section with the two-step CUDA install, single Run section using `python -m diarizer.cli`, single Configuration section reflecting the new dataclass-driven config, recovery pointer to v1-final.

**Step 7** — `CLAUDE.md` project overview updated from "Whisper + pyannote.audio" to "faster-whisper + pyannote.audio 3.3".

**Step 8** — Retired this PLAN itself plus PLAN 200 (parent rearchitecture), PLAN 220 (superseded WhisperX implementation), PLAN 240 (v2 rebuild), and RESEARCH files 210/220/230/245/250 to `Retired/`. LOG updated.

**Step 9** — Single commit + push for the cutover. (PR step deferred — branch `feature/speaker-separation` continues to be the working branch; merge to `main` will be a separate, explicitly-authorised step.)

**Step 10** — Merge to main: PENDING explicit operator authorisation (per CLAUDE.md "Executing actions with care" — merging to main is a hard-to-reverse shared-state action; not auto-running).

**Files added/modified/deleted:** see commit diff. Net: −many v1 source files, +renamed v2 modules to canonical names, ARCHITECTURE/README/CLAUDE updated, build PLANs retired.

**Recovery:** `git checkout v1-final` retrieves the v1 baseline. v1-old venv parked at `venv-old/` (gitignored).
