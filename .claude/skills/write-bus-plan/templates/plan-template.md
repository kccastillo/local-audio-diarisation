---
title: "[Plan title]"
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
closes_thread: ""        # ROADMAP.md thread ID this PLAN fully closes (e.g. "T01"); empty if none
advances_thread: ""      # ROADMAP.md thread ID this PLAN partially progresses; empty if none
parent_plan_of_plans: "" # Path to parent plan-of-plans file if part of a coordinated effort; empty if standalone
pipeline_phase: ""       # plan-pipeline orchestration state; empty for ad-hoc PLANs (see bus-conventions.md)
---

## Objective
[One paragraph: what this plan accomplishes and why it matters now.]

## Context
[What prompted this plan. Link to relevant Wiki pages via [[wikilinks]]. Note any constraints or dependencies.]

## Steps
[Numbered steps to execute via the `execute-plan` skill. Each step must be independently verifiable.
Mark [Ken] for steps that require human action before execution can proceed.
Mark [blocked-on-input] for steps waiting on a RESEARCH or ADVICE file (note which file — will be created via `write-bus-input`).]

1. 
2. 
3. 

## Verification
- [ ] [State check — e.g. "file X exists"]
      `verify: test -f path/to/file`
- [ ] [State check — e.g. "frontmatter status field is 'active'"]
      `verify: grep -q "^status: active" path/to/file`
- [ ] [Acceptance check — exercises the deliverable's behaviour. AT LEAST ONE per PLAN.]
      `acceptance: <shell command that runs the deliverable on a representative input and checks its output>`
- [ ] [Subjective item — surfaced for Human eyeball; no auto-fail]
      `verify: human`

## Recurring Task
*(Remove this section if repeatable: false)*

- **Cadence:** monthly | quarterly | after-event: [description]
- **Next due:** YYYY-MM-DD
- **Trigger condition:** [For event-driven tasks: what event fires the next cycle]

## Executor Notes
*Populated after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**

## History
*(For recurring tasks only — RECUR- slugs. Append one row per completed cycle.)*

| Cycle | Completed | Outcome | Notes |
|---|---|---|---|
