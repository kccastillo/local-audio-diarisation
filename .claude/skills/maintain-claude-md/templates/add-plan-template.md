---
title: "Maintain CLAUDE.md — add: <one-line summary of addition>"
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
Add the following to <CLAUDE.md | .claude/CONTEXT_CONSTITUTION.md>: <one-paragraph summary>.

## Context
**Trigger:** [what the user asked for, verbatim]
**Target file:** [filename]
**Target section:** [section name + insertion point]
**Budget impact:** current line count → proposed line count (delta). Vs. soft cap (150) and hard cap (300).
**Dedupe check:** [list any existing rules that overlap; explain why this is still distinct or recommend modifying the existing rule instead]

## Proposed diff

```diff
[exact unified-diff-style proposal: which lines are added, in surrounding context]
```

## Steps
1. Apply the diff above to [filename].
2. Verify line count is within caps.
3. Run audit (mode: audit) to confirm no new anti-pattern introduced.

## Verification
- [ ] [filename] contains the new content at the specified location
- [ ] Line count remains ≤ 150 (or hard cap exception documented)
- [ ] Subsequent audit shows no new anti-pattern matches

## Executor Notes
*Populated after execution. Leave blank.*

**Executed:**
**Outcome:**
**What was done:**
**Files modified:**
