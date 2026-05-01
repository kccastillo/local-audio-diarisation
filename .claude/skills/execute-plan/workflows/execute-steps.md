# Plan Execution Steps

## Step 1: Read the PLAN file in full
- Confirm `status: ready`
- Flip frontmatter `status: ready` → `status: in-progress`
- Read Objective, Context, Steps, Verification
- Note any cross-references to other files (e.g. "see harry_feedback_context.md")

## Step 2: Execute Steps in order

**Tool-choice rule (F1 Option C, PLAN 202605011900):** the executor uses filesystem tools — Read, Write, Edit, Glob, Grep — for ALL mechanical Steps. Never call Bash for `mkdir`, `cp`, `mv`, `rm`, `touch`, `cat`, `echo`, `grep`, `sed`, `awk`, or any other filesystem operation. Reason: subagents inherit only a subset of parent permissions; shell calls fail with permission denials. Filesystem tools work in every subagent context. The plan-executor agents enforce this with `disallowedTools: [Bash, ...]`. Mapping:
- `mkdir foo/` → use Write to create a file inside `foo/` (Write creates parent directories) or skip if not needed
- `cp src dst` → Read src, then Write dst
- `mv src dst` → Read src, Write dst, then (the orchestrator handles cleanup of src)
- `cat file > out` → Read file, Write out
- `echo "x" >> file` → Read file, Edit/Write with appended content
- `grep pat file` → Grep tool
- `ls dir/` → Glob tool

If a PLAN Step is shell-shaped, translate the intent to the equivalent filesystem-tool call. If translation is genuinely impossible (e.g. running a test suite, invoking a compiler), halt with `outcome: exception` — that Step belongs in the orchestrator's outcome-verifying phase (`acceptance:` shell command), not the executor.

- One step at a time; verify each completes before moving to the next
- If a step is marked [Ken]: halt, surface to Ken with the specific action required, and wait for response
- If a step is marked [blocked-on-input] (waits on a RESEARCH or ADVICE file): halt, flag the required input; it will be commissioned and `write-bus-input` will unblock the PLAN when it lands
- If a step fails, errors, or would be unsafe: halt, capture as blocker, skip to Step 4 with outcome = blocked or partially-complete
- On halt-on-failure (any of the conditions above — step failure, ambiguity, [Ken]/[Human] marker requiring approval, destructive operation without consent): populate `last_executor_outcome` in PLAN frontmatter with `outcome: exception` and a `diagnostics_summary`; populate Executor Notes; return. Do NOT commit (caller owns git per PLAN 202605011400 decision 13). The orchestrator (or Human in bootstrap) reads the frontmatter outcome on completion and applies the kanban full-stop.

## Step 3: Run the Verification checklist
- For each checkbox, independently confirm the condition is true
- If any fail: outcome becomes partially-complete or needs-revision
- Do not tick boxes that have not been verified

## Step 4: Populate the PLAN's Executor Notes section
Modify ONLY Executor Notes section + frontmatter `status`:
- **Executed:** {today YYYY-MM-DD}
- **Outcome:** done | partially-complete | blocked | needs-revision
- **What was done:** bullets — one per meaningful action
- **Blockers (if any):** specific step numbers and reasons; leave blank if none
- **Files modified:** bullets listing created/modified/renamed paths
- **Flip frontmatter `status`** to match the Outcome exactly (done / partially-complete / blocked / needs-revision). PLAN frontmatter `status` and LOG Status column must always agree.
- Also write `last_executor_outcome` frontmatter on completion (any outcome): outcome (enum: success | revision_needed | exception), outcome_subtype (existing values: done | partially-complete | blocked | needs-revision), executed (today YYYY-MM-DD), diagnostics_summary (one-line; empty if outcome=success). Per PLAN 202605011400 decision 24.

## Step 4.5: Roadmap and plan-of-plans sync

Only runs if PLAN outcome is `done` AND any of `closes_thread`, `advances_thread`, or `parent_plan_of_plans` frontmatter is non-empty.

**If `closes_thread: T{ID}` is set:**
1. Read `.claude/ROADMAP.md`
2. Locate the thread block beginning `**T{ID} — ` (in any pillar; do not move the block).
3. Within the thread body, find the existing `- Status: ` bullet and replace its value with `closed` (so the bullet reads `- Status: closed`). If no Status bullet exists, insert one as the first bullet.
4. Append a final bullet to the end of the thread body (immediately before the next blank line or next `**T` heading): `- **Closed:** {today YYYY-MM-DD} — see Bus/{this PLAN filename}. {one-line outcome derived from Executor Notes "What was done"}.`
5. Update ROADMAP.md frontmatter `last_updated` to today.

The thread stays in its pillar for historical context.

**If `advances_thread: T{ID}` is set:**
1. Read `.claude/ROADMAP.md`
2. Locate the thread block beginning `**T{ID} — `
3. Append a bullet to the thread body: `- **Progress:** {today YYYY-MM-DD} — {one-line outcome}; remaining: {what's left to close the thread}.`
4. Update ROADMAP.md frontmatter `last_updated` to today
5. Thread stays in its pillar

**If `parent_plan_of_plans: <path>` is set:**
1. Read the parent file at the given path
2. Locate this PLAN's tracking entry (by filename or T-ID convention used in that file)
3. Update its status/outcome line per that file's schema
4. Update parent file's frontmatter `last_updated` to today

**If multiple are set:** apply in order — closes_thread, then advances_thread, then parent_plan_of_plans.

**If a field is set but the thread or parent file cannot be located:** halt with `needs-revision`. Do not proceed to Step 5. Record the failure in Executor Notes.

## Step 5: Bump `last_updated` on touched files
- For every State/ file modified during execution: update its frontmatter `last_updated` to today's date ({YYYY-MM-DD})
- Applies to any file with a `last_updated` frontmatter field that this execution actually changed — do not touch files that were only read

## Step 6: Update the monthly LOG Status Table
- Path: `Bus/{YYYYMM}010000_LOG_{YYYYMM}.md`
- Find the row for this PLAN filename
- Update the Status column to match the Executor Notes outcome (and PLAN frontmatter `status`)
- Update `last_updated` in the LOG frontmatter to today

## Step 7: Report to Ken
```
Executed:    {PLAN filename}
Outcome:     {outcome}
LOG:         {LOG filename} → status: {status}
last_executor_outcome: {outcome enum} (subtype: {outcome_subtype})
```
Caller (plan-pipeline orchestrator or Human) handles git commit + push per parent PLAN 202605011400 decisions 13, 22.