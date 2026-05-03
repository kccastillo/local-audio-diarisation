---
title: "Diarizer v2 re-architecture (clean replacement, plan-of-plans)"
type: bus-plan
status: ready
assigned_to: ""
priority: high
created: 2026-05-04
created_by: opus
created_month: 202605
log_month: 202605
due: ""
repeatable: false
linked_decisions:
  - "Clean replacement, no A/B coexistence (operator, 2026-05-04)"
  - "Tag v1-final before rewrite begins"
linked_inputs:
  - "ARCHITECTURE.md (v1 design baseline, 2026-05-04)"
blocked_by: ""
rollover_count: 0
parent_plan_of_plans: ""
---

## Objective

Plan the v2 re-architecture of the Diarizer pipeline as a clean replacement of v1. **No A/B testing, no coexistence.** Tag the v1 baseline (`v1-final`) for recoverability, then rebuild the pipeline against an updated requirement set. This PLAN is a plan-of-plans: it sequences the requirement → design → decomposition → execution arc and spawns child PLANs at each phase. Concrete mechanism decisions are deferred to their respective phases — per CLAUDE.md "requirement before solution".

## Context

**Why rewrite v1?** The v1 architecture (see [ARCHITECTURE.md](../ARCHITECTURE.md)) works but accumulated friction. Concrete pain points must be elicited from the operator in Step 2 before any v2 design is locked. Likely candidates (to be confirmed, not assumed):

- Speaker-attribution heuristic (highest-overlap window) is coarse — short utterances and overlapping speech default to "Unknown Speaker".
- Sequential model load/unload is reliable but slow on consumer GPUs; warm-cache strategies unexplored.
- Pyannote 3.3.2 and Whisper 2024-09-30 are both pinned; newer model lines (pyannote 3.4+, whisper-large-v3-turbo, distil-whisper) may justify rebase.
- Singleton `ConfigManager` couples lifecycle to module import; testing requires careful fixture ordering (visible in numbered test files).
- `BaseProcessor` interface assumes synchronous, single-file processing; batch / streaming use cases not natively supported.
- No first-class diarisation-aware decoding (e.g. forced alignment, speaker-conditioned Whisper prompts).

These are **hypotheses, not requirements**. Step 2 confirms which (if any) actually drive the rewrite.

**Constraint reaffirmed (operator, 2026-05-04):** clean replacement, no A/B. v2 lands on a branch; v1 dies in the same PR. Tag `v1-final` at the head of `feature/speaker-separation` (or `main`, depending on merge state) before deletion to keep v1 recoverable via `git checkout v1-final`.

**Harness scope.** The `.claude/` harness is production / stable upstream. This PLAN does not modify it. v2 design decisions live in Bus PLANs and `ARCHITECTURE.md` updates only.

## Steps

### Step 1 — Tag the v1 baseline

Before any v2 design work begins, tag the v1 codebase:

```
git tag -a v1-final -m "Diarizer v1 baseline before v2 rewrite"
git push origin v1-final
```

Tag location: head of the branch that contains the canonical v1 code. Operator confirms the branch in Step 1 dispatch.

**verify:** `git tag -l v1-final` returns `v1-final`; `git ls-remote --tags origin v1-final` shows the tag on the remote.

**acceptance:** v1 is recoverable from any future commit via `git checkout v1-final`.

### Step 2 — Capture v1 pain points and v2 goals (operator-driven)

Produce `Bus/202605040210_RESEARCH_v2-requirements.md` via the `write-bus-input` skill, capturing:

- Operator's lived pain points with v1 (specific files, specific failure modes, specific runs).
- v2 success criteria — qualitative ("attribution should handle 3+ speaker overlap better") and quantitative where measurable ("end-to-end runtime < 0.5× audio duration on RTX 4070").
- Hard constraints (must run offline, must fit 8 GB VRAM, must support audio formats X/Y/Z).
- Soft preferences (preferred model lines, preferred libraries, deployment style — CLI only / library / both).
- Out-of-scope items (what v2 explicitly does NOT need to do).

Step 2 is a **conversation between operator and a sounding-board agent**, not a unilateral draft. Use Opus sounding-board for strategic framing; produce the RESEARCH file as the durable artefact.

**verify:** `Bus/202605040210_RESEARCH_v2-requirements.md` exists with non-empty Goals, Constraints, Out-of-scope sections.

**acceptance:** Operator signs off on the RESEARCH file; downstream design decisions can cite specific lines.

### Step 3 — Design decisions PLAN

Draft `Bus/202605040220_PLAN_v2-design-decisions.md` once Step 2 RESEARCH lands. The PLAN walks through the open design questions and records operator decisions in a `linked_decisions:` list. Expected question categories:

- **Pipeline shape.** Same five stages, or restructured (e.g. diarisation-conditioned transcription, end-to-end speaker-aware ASR)?
- **Model versions.** Whisper variant (large-v3 / large-v3-turbo / distil-whisper), pyannote version (3.3 / 3.4+ / alternatives like NeMo / Sortformer).
- **Processor abstraction.** Keep `BaseProcessor` shape, replace with async/streaming interface, or collapse into a pipeline DAG framework?
- **Configuration.** Keep YAML+singleton, switch to Pydantic settings, or per-module dataclass configs?
- **Attribution.** Keep highest-overlap heuristic, replace with forced alignment, or rebuild on speaker embeddings?
- **VRAM management.** Keep sequential load/unload, add explicit memory budget per stage, or move to a single multi-task model?
- **I/O.** Keep local-only filesystem, or expose a Python API / HTTP service?
- **Testing.** Keep numbered-file ordering, switch to pytest markers + dependency declarations, or full fixture rework?

Decisions in this PLAN drive Step 4 decomposition. Each decision recorded with **chosen option + reason + rejected alternatives**.

**verify:** `linked_decisions:` contains at least one entry per question category; PLAN passes `audit-sufficiency` (Opus) and `audit-haiku-safe` (Sonnet) — but only if the harness audit pipeline is engaged for this project; if not, operator review is the gate.

**acceptance:** Operator signs off on the decisions PLAN.

### Step 4 — Decompose v2 into child PLANs

Once Step 3 decisions are locked, draft per-module child PLANs:

- `Bus/202605040230_PLAN_v2-config-and-bootstrap.md` — config module rewrite, project skeleton, dependency pins.
- `Bus/202605040240_PLAN_v2-audio-preprocessing.md` — preprocessing replacement.
- `Bus/202605040250_PLAN_v2-vad.md` — VAD stage.
- `Bus/202605040260_PLAN_v2-diarisation.md` — speaker diarisation.
- `Bus/202605040270_PLAN_v2-transcription.md` — Whisper stage.
- `Bus/202605040280_PLAN_v2-attribution-and-output.md` — attribution + transcript writer.
- `Bus/202605040290_PLAN_v2-cli-and-entrypoint.md` — `run_diariser.py` replacement.
- `Bus/202605040300_PLAN_v2-tests.md` — test suite rewrite.
- `Bus/202605040310_PLAN_v2-decommission-v1.md` — delete v1 files; final ARCHITECTURE.md rewrite; merge to main.

Exact list adjusts based on Step 3 decisions (e.g. if diarisation+transcription merge into one stage, those PLANs collapse). Each child PLAN declares `parent_plan_of_plans: 202605040200_PLAN_diarizer-v2-rearchitecture.md`.

**verify:** Each child PLAN exists in `Bus/`; each has frontmatter pointing back to this PLAN.

**acceptance:** All v2 work is captured by child PLANs; nothing is left implicit.

### Step 5 — Execute child PLANs in dependency order

Default sequence (subject to Step 3 decisions):

1. config-and-bootstrap (others depend on it).
2. audio-preprocessing → VAD → diarisation → transcription → attribution-and-output (pipeline order; each stage testable against v1 reference outputs from `v1-final` while v2 is being built; **this is not A/B testing — v1 is the dev-time correctness reference, not a production fallback**).
3. cli-and-entrypoint (wires the new pipeline).
4. tests (rebuild against v2 module surfaces).
5. decommission-v1 (last — deletes v1 code and updates ARCHITECTURE.md).

Each child PLAN runs to completion before its dependents start. Concurrency permitted only between truly independent stages (e.g. preprocessing and tests-skeleton).

**verify:** All child PLANs reach `complete` status; each retired to `Retired/` after completion.

**acceptance:** v2 pipeline runs end-to-end on the operator's test audio set with results meeting Step 2 success criteria.

### Step 6 — Decommission v1 and merge

The final child PLAN (Step 4's `v2-decommission-v1.md`) handles:

- Delete v1 source files: `processors/`, `audio/audio_cleaner.py`, old `utils/*` modules, old `config/config_manager.py`, old `run_diariser.py` — replaced atomically by v2 equivalents in the same PR.
- Rewrite `ARCHITECTURE.md` from "v1 baseline" to "v2 design".
- Update `README.md` for any usage changes.
- Open PR `feature/speaker-separation → main` (or operator-named target branch) with a single squash-merge commit.

**verify:** Post-merge `main` contains v2 only; `git log v1-final..main -- processors/` shows only deletions; `python run_diariser.py --help` reflects v2 CLI.

**acceptance:** v2 ships; v1 lives only behind `git checkout v1-final`.

## Verification (full PLAN)

- [ ] v1-final tag created and pushed (Step 1).
- [ ] v2 requirements RESEARCH written and operator-signed-off (Step 2).
- [ ] v2 design decisions PLAN written, all categories covered, operator-signed-off (Step 3).
- [ ] All child PLANs drafted and present in `Bus/` (Step 4).
- [ ] All child PLANs complete and retired (Step 5).
- [ ] v1 source files deleted; v2 merged to target branch (Step 6).
- [ ] ARCHITECTURE.md rewritten as v2 design.

## Acceptance (spec)

A user runs `python run_diariser.py --input <audio>` against the v2 codebase and gets a transcript that meets the success criteria recorded in Step 2 RESEARCH. The v1 codebase is no longer in the working tree but is recoverable via `git checkout v1-final`. ARCHITECTURE.md describes the v2 design, not v1. No A/B feature flag, no parallel pipeline, no v1/v2 branching at runtime.

## Out of scope

- Any modification to the `.claude/` harness (production / stable).
- Any RESEARCH or experimentation that doesn't feed a Step 2/3 decision (no exploratory model benchmarking unless operator explicitly requests it).
- Multi-language transcription, real-time streaming, GUI — unless captured as Step 2 goals.

## Executor Notes

*Populated after execution. Leave blank.*
