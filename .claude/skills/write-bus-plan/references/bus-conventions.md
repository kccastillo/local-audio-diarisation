# Bus File Conventions

This file is the canonical source for Bus/ file conventions. AGENT_RULES.md and CLAUDE.md should point here rather than duplicate.

## File Naming

`{YYYYMMDDHHMI}_{TYPE}_{slug}.md` — see `references/bus-naming-convention.md` for full pattern, type tokens, and examples.

## Monthly LOG Path

`Bus/{YYYYMM}010000_LOG_{YYYYMM}.md` — first-of-month midnight timestamp regardless of when created.

## Recurring PLAN Prefix

Slugs start with `RECUR-`. One persistent file per recurring task; each cycle appends to `## History`.

## Bus Types

`LOG` | `PLAN` | `RESEARCH` | `ADVICE`.

## PLAN Status Lifecycle

Frontmatter `status` field:
```
ready → in-progress → done | partially-complete | blocked | needs-revision
```

- `ready` — transcribed, not yet started. Set on creation.
- `in-progress` — execute-plan skill has started. Set when execution begins.
- `done` — all verification criteria pass. Terminal.
- `partially-complete` — some steps done, others blocked or deferred. Terminal for this cycle.
- `blocked` — cannot proceed; `blocked_by` holds the reason. Cleared automatically by `write-bus-input` when the resolving RESEARCH/ADVICE lands, or manually by Ken. Non-terminal until unblocked.
- `needs-revision` — plan itself is faulty; return to Sonnet. Non-terminal.

LOG Status Table "Status" column mirrors the plan's frontmatter `status` at end of day. Rollover treats `done`, `cancelled`, `closed` as complete; everything else rolls over.

## PLAN Input Linkage

Frontmatter `linked_inputs: []` contains one array of filenames — both RESEARCH and ADVICE. Type is recoverable from the `_RESEARCH_` / `_ADVICE_` token in each filename. No separate `linked_research` / `linked_advice` / `requires_opus` / `requires_research` fields — `status: blocked` + `blocked_by` carries the "needs input before running" signal.

## Roadmap Linkage

PLANs may reference items in `.claude/ROADMAP.md` via three optional frontmatter fields:

- `closes_thread: T{ID}` — this PLAN's successful execution fully closes the named thread. On PLAN closure (outcome=done), `execute-plan` Step 4.5 appends a closure bullet to the thread body and sets the thread's `Status` to `closed`. The thread stays in its pillar for historical context; grep for `Status:.*closed` to inventory closed work.
- `advances_thread: T{ID}` — this PLAN partially progresses the thread. On closure, `execute-plan` appends a progress note to the thread body but leaves the thread in its pillar.
- `parent_plan_of_plans: <path>` — this PLAN is part of a coordinated effort tracked by another file. On closure, `execute-plan` updates the parent plan-of-plans tracking section per that file's schema.

Set these on PLAN creation (in `write-bus-plan`). Empty string means no linkage. A PLAN closes or advances at most one thread; for multi-thread work, decompose into multiple PLANs or introduce a plan-of-plans.
