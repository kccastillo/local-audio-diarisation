---
title: "Maintain CLAUDE.md — prune: <one-line summary of removals>"
type: bus-plan
status: ready
assigned_to: ""
priority: medium
created: YYYY-MM-DD
created_by: ""
created_month: YYYYMM
log_month: YYYYMM
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
Remove the following from <CLAUDE.md | .claude/CONTEXT_CONSTITUTION.md>: <one-paragraph summary>.

## Context
**Trigger:** [what prompted the prune — e.g., over hard cap, dead reference flagged, user request]
**Budget impact:** current line count → proposed line count.

## Removal candidates

For each candidate:
- **Location:** filename:line-range
- **Content:** [excerpt or summary of what's being removed]
- **Reason:** [why removal is safe — dead reference, duplicated elsewhere, drifted, anti-pattern]
- **Migration target (if any):** [where the content moves to, e.g., `.claude/references/X.md`]
- **Impact assessment:** [what could break if this is removed]

## Steps
1. For each candidate with a migration target: create / update the target file first.
2. Remove the content from the source file.
3. Verify all pointers still resolve.

## Verification
- [ ] Each removal applied
- [ ] Migrated content (if any) is reachable via pointer
- [ ] No previously-passing audit check now fails
- [ ] Line count reduced as expected

## Executor Notes
*Populated after execution. Leave blank.*

**Executed:**
**Outcome:**
**What was done:**
**Files modified:**
