---
title: "F1 Option C smoke test — plan-executor creates tmp/smoke.txt via filesystem tools"
type: bus-plan
status: done
assigned_to: haiku
priority: low
created: 2026-05-01
created_by: opus
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
parent_plan_of_plans: 202605011900_PLAN_post-dogfood-fixes.md
pipeline_phase: checked
last_executor_outcome:
  outcome: success
  outcome_subtype: done
---

## Objective

Smoke-test the F1 Option C rewrite (commit 11077b0). Confirm a haiku plan-executor subagent — with `Bash` in `disallowedTools` — can complete a trivial mechanical PLAN end-to-end using filesystem tools only. Deliverable: a file `tmp/smoke.txt` with the exact content `hello`, plus the executor populating Executor Notes and flipping `status: ready → done` and `last_executor_outcome.outcome: success`.

## Context

PLAN 1900 step B/5 calls for a smoke test of the F1 fix. The original A+ permissionMode hack was reverted (commit f90411e); Option C (filesystem-tools-only, hard-enforced via `disallowedTools: [Bash, ...]`) landed in commit 11077b0. This PLAN exercises the new contract: a single mechanical Step that creates a file. If the executor can complete this PLAN without Bash, the C rewrite is validated for the simplest case.

Throwaway: retire after a successful run.

## Steps

1. Use the Write tool to create the file `tmp/smoke.txt` with the exact content `hello` (no trailing newline, no surrounding whitespace). The Write tool auto-creates the `tmp/` parent directory if it does not exist.

## Verification

- [ ] `tmp/smoke.txt` exists in the repo working tree
      verify: test -f tmp/smoke.txt
- [ ] `tmp/smoke.txt` contains exactly `hello`
      acceptance: test "$(cat tmp/smoke.txt)" = "hello"

## Executor Notes

*Populated after execution. Leave blank.*

**Executed:** 2026-05-01 by haiku
**Outcome:** done
**What was done:** Created `tmp/smoke.txt` with exact content `hello` (no trailing newline) using Write tool. File created successfully and verified via Read tool to contain correct content.
**Blockers (if any):** None
**Files modified:** tmp/smoke.txt (created)
