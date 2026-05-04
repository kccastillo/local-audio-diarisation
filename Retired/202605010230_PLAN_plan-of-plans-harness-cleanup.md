---
title: "Plan-of-plans: harness cleanup sequence (April carry-forward)"
type: bus-plan
status: ready
assigned_to: ""
priority: high
created: 2026-05-01
created_by: sonnet
created_month: 202605
log_month: 202605
due: ""
repeatable: false
repeat_cadence: ""
linked_decisions: []
linked_inputs: []
blocked_by: ""
rollover_count: 0
triggers_plans: []
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: ""
---

## Objective

Coordinate execution order for the four active plans carried forward from April 2026, record the two sequencing decisions made by Ken (2026-05-01), and note the one pre-execution plan revision still required before the main sequence begins.

## Context

**Decisions recorded (Ken, 2026-05-01):**

**Decision 1 — architecture single-source:** Option (c). Both `ROADMAP.md` and `.claude/references/architecture.md` are kept. `ROADMAP § Working architecture` holds challenge-annotated principles (planning context). `references/architecture.md` holds the descriptive current-state snapshot (cold-session orientation). `CLAUDE.md` points to `references/`. Maintenance cost is small and caught by the monthly audit.

**Decision 2 — jq vs Python for skill-usage logging:** Python (already resolved). The prior bash+jq script was rewritten in Python by the executor of PLAN 202604300210 (committed 2026-05-01). jq dependency dropped; no PATH issue.

**Status at plan-of-plans creation:**
- `202604300210_PLAN_skill-usage-logging.md` — **done** (committed and pushed; Python hook live)
- `202604300220_PLAN_skill-usage-audit-skill.md` — **blocked** (deferred; no action)
- Four plans remain active: adapt-imported-skills, rationalise-claude-md, rewrite-roadmap, RECUR-monthly-claude-md-audit

## Pre-execution revision (Sonnet, before any Haiku run)

One revision is required before the execution sequence starts:

**Revise `202604291200_PLAN_rewrite-roadmap.md`:**
- Drop Step 9 entirely (the CLAUDE.md architecture trim — `Architecture: see ROADMAP.md § Working architecture`). That pointer is now owned by `rationalise-claude-md` which points to `references/architecture.md` instead.
- Narrow Step 12 to cover ROADMAP.md `last_updated` only (remove the `CLAUDE.md` part of the step).
- Update Verification checklist: remove the two bullets that check for the CLAUDE.md architecture pointer and the grep for `load_model`/`BaseProcessor`/`singleton` in CLAUDE.md.
- After revision: rewrite-roadmap and rationalise-claude-md no longer touch the same CLAUDE.md section and can execute in any order.

## Execution sequence

### Step 1 — adapt-imported-skills (`202604291800_PLAN_adapt-imported-skills.md`)
**Must run first.** Edits `execute-plan/SKILL.md` and `execute-steps.md` — the procedure that subsequent plans run under. Also renames `create-agent-skills → create-skill` and updates the CLAUDE.md skills table, so rationalise-claude-md slims an already-correct file. Assigned: Haiku.

### Step 2 — rewrite-roadmap + rationalise-claude-md (after Step 1 completes)
After the pre-execution revision above, these two plans no longer collide on CLAUDE.md and can execute in either order. If the executor supports parallelism, they may run concurrently. Assigned: Haiku (both).

- `202604291200_PLAN_rewrite-roadmap.md`: rewrites ROADMAP.md structure and Working architecture section. Touches ROADMAP.md and CLAUDE.md `last_updated` only.
- `202604291950_PLAN_rationalise-claude-md.md`: creates `.claude/CONTEXT_CONSTITUTION.md` + `.claude/references/`, slims CLAUDE.md. Touches CLAUDE.md and the new reference files.

### Step 3 — RECUR-monthly-claude-md-audit (after rationalise-claude-md completes)
`202604291940_PLAN_RECUR-monthly-claude-md-audit.md`. Depends on `.claude/CONTEXT_CONSTITUTION.md` existing (created by rationalise-claude-md). Due 2026-05-01. Invokes `Skill("maintain-claude-md")` in audit mode.

### Deferred
`202604300220_PLAN_skill-usage-audit-skill.md` — stays blocked. Trigger condition: 15+ project skills OR felt misfire signal.

## Tracking

| Child Plan | Step | Status | Notes |
|---|---|---|---|
| pre-execution: revise rewrite-roadmap | pre | pending | Drop Step 9; narrow Step 12; update Verification |
| 202604291800_PLAN_adapt-imported-skills.md | 1 | pending | — |
| 202604291200_PLAN_rewrite-roadmap.md | 2 | pending | Requires pre-execution revision first |
| 202604291950_PLAN_rationalise-claude-md.md | 2 | pending | — |
| 202604291940_PLAN_RECUR-monthly-claude-md-audit.md | 3 | pending | Requires rationalise-claude-md done first |
| 202604300210_PLAN_skill-usage-logging.md | — | done | Committed 2026-05-01; Python hook live |
| 202604300220_PLAN_skill-usage-audit-skill.md | — | blocked | Deferred; no action |

## Verification
- [ ] Pre-execution revision applied to `202604291200_PLAN_rewrite-roadmap.md` (Step 9 dropped, Step 12 narrowed, Verification updated)
- [ ] adapt-imported-skills completes and verifies before rewrite-roadmap or rationalise-claude-md begin
- [ ] rewrite-roadmap completes without touching CLAUDE.md architecture section
- [ ] rationalise-claude-md completes; `.claude/CONTEXT_CONSTITUTION.md` and `.claude/references/architecture.md` exist
- [ ] RECUR audit runs only after `.claude/CONTEXT_CONSTITUTION.md` exists
- [ ] Tracking table above updated to `done` for each child plan as it completes
- [ ] This plan-of-plans row in LOG_202605 Status Table updated to `done` when all four active plans complete

## Executor Notes
*Populated after execution. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**
