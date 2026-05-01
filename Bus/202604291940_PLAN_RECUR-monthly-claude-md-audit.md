---
title: "Monthly audit of CLAUDE.md and CONTEXT_CONSTITUTION.md via maintain-claude-md skill"
type: bus-plan
status: ready
assigned_to: ""
priority: medium
created: 2026-04-29
created_by: ""
created_month: 202604
log_month: 202604
due: 2026-05-01
repeatable: true
repeat_cadence: "monthly — first of month, when new monthly LOG is created"
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
Run `maintain-claude-md` in audit mode against CLAUDE.md and `.claude/CONTEXT_CONSTITUTION.md`. The audit produces its own follow-up PLAN with findings — this RECUR- task tracks the monthly cadence and history.

## Context
camelCase principle: CLAUDE.md is living configuration; static maintenance is an anti-pattern. Monthly cadence keeps it current with the codebase. Trigger: when a new monthly LOG is created in `Bus/`, this RECUR- task should be visible in that LOG's Recurring Task Tracker as due.

## Steps
1. Invoke `Skill("maintain-claude-md")` with phrasing "audit CLAUDE.md and the constitution".
2. The skill produces an audit-results PLAN. File the PLAN under that month's LOG.
3. Append a row to the History table below: cycle number, completed date, outcome (clean / N findings), link to the produced audit PLAN.

## Verification
- [ ] An audit-results PLAN exists in `Bus/` for the current month
- [ ] History table has a new row
- [ ] Recurring Task Tracker in current LOG shows last_done updated, next_due advanced one month

## History

| Cycle | Completed | Outcome | Audit PLAN |
|---|---|---|---|
| 2026-05 | 2026-05-01 | skipped — insufficient data, CLAUDE.md/CONTEXT_CONSTITUTION.md churn since last audit too low to surface meaningful findings | — |

## Executor Notes
*Populated after execution. Leave blank.*

**Executed:**
**Outcome:**
**What was done:**
**Files modified:**
