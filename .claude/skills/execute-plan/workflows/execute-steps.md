# Plan Execution Steps

## Step 1: Read the PLAN file in full
- Confirm `status: ready`
- Flip frontmatter `status: ready` → `status: in-progress`
- Read Objective, Context, Steps, Verification
- Note any cross-references to other files (e.g. "see harry_feedback_context.md")

## Step 2: Execute Steps in order
- One step at a time; verify each completes before moving to the next
- If a step is marked [Ken]: halt, output MODEL SWITCH NEEDED to Sonnet requesting Ken's action
- If a step is marked [blocked-on-input] (waits on a RESEARCH or ADVICE file): halt, flag the required input; Sonnet will commission it and `write-bus-input` will unblock the PLAN when it lands
- If a step fails, errors, or would be unsafe: halt, capture as blocker, skip to Step 4 with outcome = blocked or partially-complete

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

## Step 7: Git commit and push
- Run `git status` to confirm expected changes
- `git add -A`
- Commit via HEREDOC with format:
  ```
  {Subject: ≤72 chars, adapted from PLAN title, reflects what was done}

  - {bullet from Executor Notes "What was done"}
  - {...}

  Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
  ```
- `git push`
- If commit or push fails: diagnose root cause and fix. Never use `--no-verify`, `--force`, or bypass signing.

## Step 8: Report to Sonnet/Ken
```
Executed:    {PLAN filename}
Outcome:     {outcome}
LOG:         {LOG filename} → status: {status}
Commit:      {hash} on {branch}
Pushed:      {branch} → origin
```