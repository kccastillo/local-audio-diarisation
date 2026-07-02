---
title: "Research — plan_foundry build_brief.py re-audit prior-findings section renders empty"
type: research
created: 2026-07-02
feeds_plan: ""
from: "consumer-side observation during PLAN-AA0 plan-pipeline run"
question_asked: "Why did the sufficiency re-audit brief show 0 prior findings when iteration 1 recorded 5?"
integration_status: pending
lifecycle_mode: reference
---

## Findings

**plan_foundry consumer-side observation for upstream handoff** (this is NOT a diarizer workstream item — it is a plan_foundry bundle observation captured per the CLAUDE.md detection-then-handoff procedure; `feeds_plan` intentionally empty, to be filled by the plan_foundry maintainer on transfer).

During plan-pipeline execution of PLAN-AA0 (2026-07-02), the sufficiency re-audit brief produced by `.claude/skills/plan-pipeline/lib/build_brief.py` for iteration 2 rendered:

```
## Prior Findings (0 total, 0 shown after excluding acknowledged)

_No findings in prior iteration._
```

— even though iteration 1 had recorded 5 structured findings (1 error S403, 2 warnings S999/S204, 2 notes S001/S201) into `Workbench/.audit/PLAN-AA0-sufficiency-1.json` via `audit_loop.py`. The prior-findings section of the re-audit brief was therefore empty; the auditor had to work purely from the embedded PLAN diff instead of statusing the concrete prior findings by code. `audit_loop.py` reported `stripped_count: 0` and `recurring_fingerprints: []` for that iteration.

**Impact: low-to-moderate.** The re-audit still functioned (the diff was present and the auditor correctly confirmed resolution), but the "status each prior finding (resolved / still-present)" contract in re-audit mode was not backed by the actual prior-findings list — it relied on the orchestrator manually re-listing prior findings in the dispatch prompt. If the orchestrator had not re-listed them, the auditor would have had no structured prior-findings to reconcile.

**Hypothesis for upstream triage:** `build_brief.py` may only carry forward findings that are unresolved/unacknowledged via the `ack` mechanism; since iteration-1 findings were resolved by PLAN revision (not via an explicit `ack` action), they were excluded from the prior-findings section rather than shown with a `status`. Alternatively `build_brief.py` reads prior findings from a path/shape that `audit_loop.py` did not populate as expected. Worth verifying which `.audit` file(s) `build_brief.py` consumes for the prior-findings block, and whether "resolved-by-revision" findings should still be surfaced for statusing.

**In-conversation workaround applied:** the orchestrator re-listed the prior findings verbatim in the re-audit dispatch prompt, so the auditor could status them. No blocker to the run.

## Sources

- Live plan-pipeline run of PLAN-AA0_just-run-packaging (diarizer repo, branch `main`).
- Repro anchor: commits around `171a06bc` (sufficiency:revision_needed) → `0feab64e` (sufficiency:success).
- Artefacts at observation time: `Workbench/.audit/PLAN-AA0-sufficiency-{1,2}.json` (transient, gitignored, since cleaned on retire); PLAN retired to `Retired/PLAN-AA0_just-run-packaging.md`.
- Relevant bundle code: `.claude/skills/plan-pipeline/lib/build_brief.py`, `.claude/skills/plan-pipeline/lib/audit_loop.py`.

## Caveats

- Consumer-side capture only — not verified against `build_brief.py` source internals; the hypothesis is unconfirmed.
- The `.audit/` JSON snapshots referenced were deleted during the PLAN's retire (orphaned-audit cleanup), so exact reproduction requires re-running an audit-revise-reaudit cycle.
- Handoff target: copy to the plan_foundry repo `Workbench/` (https://github.com/kccastillo/plan_foundry) for maintainer triage; retire the consumer-side copy afterwards.
