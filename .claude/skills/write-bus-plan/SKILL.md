---
name: write-bus-plan
description: Haiku's Bus transcription skill. Use when Sonnet has finalised a plan and needs it written into System/Bus/ with correct schema. Also handles monthly LOG creation, LOG updates, and rollover file updates. Trigger phrases: "write this plan to bus", "create plan file", "update the log", "write bus file", "create the log".
---

**Bus file conventions:** See [references/bus-conventions.md](references/bus-conventions.md) — canonical source for naming, status lifecycle, and input linkage.

<essential_principles>
Haiku transcribes. It does not think strategically or modify plan content.
Write exactly what Sonnet provided — no additions, no omissions, no interpretation.
Always check if the current month LOG exists before writing any PLAN file. Create it first if missing.
After writing any PLAN file, update the monthly LOG's Status Table.
For recurring PLAN files: append to the existing file's History table — do not create a new file.
Report back: filename written, LOG updated, ready for next step.
</essential_principles>

**Plan writing procedure:** See [workflows/write-plan.md](workflows/write-plan.md)

<constraints>
- Never modify plan content — transcribe exactly what Sonnet provides
- Never write to Wiki/ — that is Antigravity's domain
- Never create a new RECUR- PLAN file if one already exists for that slug
- Always create the monthly LOG before writing the first PLAN of a month
- When creating a new monthly LOG, always run rollover (Step 2a) against the prior month's LOG before writing any new PLAN
- Always update the LOG Status Table after every PLAN file write
- The created_month field is set once and never changed — rollover updates log_month and rollover_count only
- Never move or rename PLAN files on rollover
- Status Table rows must be ordered: rows whose Status is not in {done, cancelled, closed} appear first sorted by filename descending; remaining rows follow sorted by filename descending. Apply this order on every LOG write — when adding a new row, when updating a row's Status, or when rolling over.
</constraints>

<success_criteria>
- PLAN file exists at the correct path with valid frontmatter
- Monthly LOG Status Table has the new row
- Recurring tasks appear in Recurring Task Tracker
- RECUR- files have a ## History table
- On month rollover: prior LOG is closed; new LOG's Rollover and Status tables list all incomplete plans; each rolled plan's frontmatter has log_month updated and rollover_count incremented
- Ken has been given the PLAN filename for handoff to Haiku
- LOG Status Table rows are ordered correctly: non-terminal statuses before terminal statuses ({done, cancelled, closed}); within each group, filename descending.
</success_criteria>
