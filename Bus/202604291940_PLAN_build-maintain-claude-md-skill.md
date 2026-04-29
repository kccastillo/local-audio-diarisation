---
title: "Build maintain-claude-md skill: audit/add/prune CLAUDE.md and CONTEXT_CONSTITUTION.md, output as Bus PLANs"
type: bus-plan
status: done
assigned_to: ""
priority: high
created: 2026-04-29
created_by: sonnet
created_month: 202604
log_month: 202604
due: ""
repeatable: false
repeat_cadence: ""
linked_decisions: []
linked_inputs: []
blocked_by: ""
rollover_count: 0
triggers_plans: ["202604291950_PLAN_rationalise-claude-md.md"]
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: ""
---

## Objective
Create the `maintain-claude-md` skill at `.claude/skills/maintain-claude-md/`. The skill audits CLAUDE.md and `.claude/CONTEXT_CONSTITUTION.md` against camelCase CLAUDE.md hygiene principles and the context-rot framework. Three modes — audit, add, prune — all infer from the user's phrasing and all produce a Bus PLAN as output (never direct edits, never chat-only output). A RECUR- PLAN provides monthly cadence visibility.

## Context

Two source frameworks drive this skill:

**camelCase principles** (Stop Writing Bad CLAUDE.md Files):
- Soft warn at 150 lines, hard fail at 300 lines (Ken's policy)
- Trinity required at top: project oneliner, key commands, caveats
- Lost-in-the-middle: critical content goes at top or bottom, never buried
- IMPORTANT / MUST markers signal-boost critical rules
- Anti-patterns: AI-init bloat, natural-language linting (e.g., "use 2 spaces"), static maintenance
- Progressive disclosure: long content moves to references, CLAUDE.md links to it

**Context-rot framework** (Poisoning / Distraction / Confusion / Clash + Write / Select / Compress / Isolate fixes): the skill audits whether CLAUDE.md positions Claude well to apply these — e.g., are external references labelled, is there a scratchpad pattern, are subagent triggers documented.

All three modes produce a Bus PLAN: audit produces an audit-results PLAN, add produces an addition PLAN (with the proposed diff embedded), prune produces a removal PLAN. Ken approves each PLAN before execution.

Monthly cadence: a RECUR- PLAN sits in the Recurring Task Tracker so the audit shows up as "due" each month when the new LOG is created. Soft trigger — visible reminder, not auto-execution.

## Steps

**Structure note for the executor:** The numbered procedure at the bottom ("Step-by-step procedure") is the execution order. The "File N:" content blocks below are the verbatim file contents that the numbered procedure writes to disk. Do not treat the "File N:" blocks as steps themselves — they are reference content.

### File 1: `.claude/skills/maintain-claude-md/SKILL.md`

Create with this exact content:

```markdown
---
name: maintain-claude-md
description: Audit, propose additions to, or propose removals from CLAUDE.md and .claude/CONTEXT_CONSTITUTION.md. Triggered by phrasing like "audit CLAUDE.md", "add X to CLAUDE.md", "prune CLAUDE.md", "is CLAUDE.md still good", or as the monthly RECUR- task. All output lands as a Bus PLAN — never direct edits, never chat-only.
---

# maintain-claude-md

Base directory: `d:\Projects\Diarizer\.claude\skills\maintain-claude-md`

## What this skill does

Maintains CLAUDE.md and `.claude/CONTEXT_CONSTITUTION.md` as living configuration. Three modes, all dispatched by phrasing:

- **audit** — "audit CLAUDE.md", "is CLAUDE.md still good", "review the constitution", or the monthly RECUR- trigger. Scans both files against the checklist and produces a punch-list PLAN.
- **add** — "add X to CLAUDE.md", "should we add Y to the constitution". Proposes the addition as a PLAN with the exact diff embedded; flags placement, dedupe risk, and budget impact.
- **prune** — "prune CLAUDE.md", "remove the dead reference to X", "trim CLAUDE.md". Proposes specific removals as a PLAN.

All three modes write a PLAN to `Bus/` via the `write-bus-plan` skill. Ken approves the PLAN before any file edit happens.

## Files in scope

- `CLAUDE.md`
- `.claude/CONTEXT_CONSTITUTION.md`

The skill does NOT audit `.claude/references/*.md` (those are progressive-disclosure targets — bulk is fine there). It does check that pointers in CLAUDE.md to those files resolve.

## Line-cap policy

- **Soft warn:** 150 lines. Above this, audit flags it as "approaching cap" and prune mode is suggested.
- **Hard fail:** 300 lines. Above this, audit blocks new additions and forces a prune PLAN before any add PLAN can land.

Both files have the same caps unless overridden in their frontmatter.

## Workflow

See [workflows/produce-plan.md](workflows/produce-plan.md) for the unified procedure.

## Audit checklist

See [references/audit-checklist.md](references/audit-checklist.md) — camelCase + context-rot checks.

## Anti-patterns

See [references/anti-patterns.md](references/anti-patterns.md) — what to flag and why.

## PLAN templates

See [templates/audit-plan-template.md](templates/audit-plan-template.md), [templates/add-plan-template.md](templates/add-plan-template.md), [templates/prune-plan-template.md](templates/prune-plan-template.md).

<essential_principles>
Output is always a Bus PLAN — never direct edits to CLAUDE.md or CONTEXT_CONSTITUTION.md, never chat-only findings.
Audits use the checklist verbatim; do not invent new rules ad-hoc — propose new rules through an add-mode PLAN against the checklist itself.
Mode is inferred from user phrasing; if ambiguous, ask once before producing anything.
The monthly RECUR- task is a visibility mechanism, not auto-execution — it shows in the LOG's Recurring Task Tracker; Ken or the active session triggers it.
</essential_principles>

<constraints>
- Never edit CLAUDE.md or .claude/CONTEXT_CONSTITUTION.md directly from this skill — always via a Bus PLAN
- Never write findings to chat only — always to a Bus PLAN
- Never audit `.claude/references/*.md` — those are intentionally bulky
- Soft warn at 150 lines, hard fail at 300 lines per file
- If CLAUDE.md is over hard cap, block all add-mode PLANs until a prune-mode PLAN executes
- Do not modify the audit checklist from inside an audit run — propose changes via add mode
</constraints>

<success_criteria>
- A Bus PLAN exists at the correct path with valid frontmatter
- The PLAN body contains specific, actionable findings (file + line range + verdict + recommended fix)
- Monthly LOG Status Table has a row for the new PLAN
- Ken can approve the PLAN and hand it to `execute-plan` without further design
</success_criteria>
```

### File 2: `.claude/skills/maintain-claude-md/references/audit-checklist.md`

Create with this exact content:

```markdown
# Audit Checklist

Run every check on every audit. Group findings by severity: **blocker** (hard-cap breach, dead reference, anti-pattern) → **warn** (soft-cap, structural risk) → **suggestion** (progressive-disclosure opportunity).

## A. Size and budget
- A1. Line count vs. soft cap (150). Flag if over.
- A2. Line count vs. hard cap (300). Block adds if over.
- A3. Estimated instruction count (count each bullet, table row, and emphasised rule). Flag if approaching 100 (camelCase ceiling for the user-budget portion of the model's instruction window).

## B. Trinity present (top of file)
- B1. Project oneliner — single paragraph, names framework + tech stack + domain.
- B2. Key Commands — only the most-frequent (≤3 commands inline; rest as pointer).
- B3. Caveats — non-obvious project quirks (e.g., "do not modify schema.prisma directly").
Missing any: blocker.

## C. Instruction weighting (Lost-in-the-Middle)
- C1. Most critical rules in the top ~30 lines or bottom ~20 lines, not buried mid-file.
- C2. IMPORTANT / MUST / NEVER markers used on load-bearing rules.
- C3. Pointer to CONTEXT_CONSTITUTION.md sits in top 10 lines.
Buried critical content: blocker.

## D. Anti-pattern detection
See [anti-patterns.md](anti-patterns.md). Flag every match.

## E. Reference health
- E1. Every pointer to `.claude/references/*.md`, `AGENT_RULES.md`, or other in-repo files resolves to an existing file.
- E2. Every pointer to a skill (`Skill("name")` or `.claude/skills/name/`) resolves.
- E3. No pointers to retired files (check `Retired/` for matching basenames).
Dead reference: blocker.

## F. Progressive-disclosure opportunities
- F1. Any inline section ≥20 lines that documents codebase-derivable facts (architecture, schemas, command lists) → suggest moving to `.claude/references/`.
- F2. Any inline section ≥10 lines that's only relevant for specific file types → suggest path-based rule loading (`.claude/rules/<pattern>.md`).
- F3. Any large block of caveats specific to one subsystem → suggest a subsystem-specific reference.

## G. Static-maintenance (drift)
- G1. Tool/library version numbers inline — flag for verification against `requirements.txt` / `package.json`.
- G2. File paths inline — verify each exists; dead path is blocker.
- G3. References to people, projects, or external systems — flag for review (cannot auto-verify).

## H. Context-rot positioning
- H1. Does CLAUDE.md document the scratchpad pattern (Bus/ for plans, memory for cross-session)? Required.
- H2. Does CLAUDE.md document subagent triggers (when to delegate vs. handle inline)? Required.
- H3. Does CLAUDE.md establish content labelling (e.g., naming external-source content distinctly)? Suggestion if missing.
- H4. Does CONTEXT_CONSTITUTION.md exist and is pointed to from the top of CLAUDE.md? Required.

## I. Constitution-specific (only when auditing CONTEXT_CONSTITUTION.md)
- I1. All four rot modes named (Poisoning / Distraction / Confusion / Clash).
- I2. All four fixes named (Write / Select / Compress / Isolate).
- I3. Project-specific imperative rules present (≥6 rules tied to actual project workflows).
- I4. Recovery protocol on rot detection documented.
```

### File 3: `.claude/skills/maintain-claude-md/references/anti-patterns.md`

Create with this exact content:

```markdown
# Anti-patterns

Patterns that, when found in CLAUDE.md or CONTEXT_CONSTITUTION.md, are flagged as blockers (must be fixed) or warns (should be reviewed).

## Blockers

### Natural-language linting
Rules that describe formatting/style enforceable by tooling. Examples:
- "Use 2 spaces for indentation"
- "Use single quotes for strings"
- "Always include a trailing newline"
**Why blocker:** wastes instruction budget on a job an LLM does worse than a linter. Move to `.eslintrc`, `.prettierrc`, `pyproject.toml`, or a `post_tool_use` hook in `settings.json`.

### AI-generated bloat
Verbose generic guidance with no project specificity. Tells:
- Sentences starting with "It's important to..." / "Remember to..." / "Always make sure to..."
- Bullet lists of generic best practices ("Write clean code", "Add tests", "Handle errors")
- Hedging modifiers ("might", "should consider", "could be useful")
**Why blocker:** zero per-token signal. Replace with project-specific imperatives or delete.

### Codebase duplication
Documentation of facts authoritative elsewhere:
- Dependency lists that duplicate `requirements.txt` / `package.json`
- Config schemas that duplicate the actual config file
- Module-by-module architecture descriptions Claude can derive by reading the code
**Why blocker:** drifts silently when the source updates. Move to `.claude/references/` as a pointer, or delete and let Claude read source.

### Dead references
Pointers to files, skills, or sections that no longer exist.
**Why blocker:** poisons context — Claude follows the pointer, finds nothing, may invent content.

## Warns

### Static maintenance smell
- Version numbers inline (likely to drift)
- "As of YYYY-MM" timestamps without an audit cadence
- "We recently switched to X" (will become stale)
**Why warn:** date the project, schedule a review, or move to a versioned reference.

### Lost-in-the-middle risk
- Critical rules buried 60-80% into the file with no signal-boost markers
- Long mid-file sections with no headers (model attention drifts)
**Why warn:** restructure or add IMPORTANT markers.

### Caveat creep
- Caveats section growing past ~10 items
**Why warn:** likely contains some that are now obvious from code (delete) and some that are subsystem-specific (move to subsystem reference).

### Subagent / delegation rules missing
- No documented threshold for when to delegate to a subagent
**Why warn:** Claude defaults to handling everything inline, blowing context on big searches.

## Not anti-patterns (don't flag)

- Working-style preferences (AU spelling, tone, review format) — these ARE the user's signal.
- Project-specific imperatives ("never modify schema.prisma directly") — these are the Trinity caveats; they belong here.
- Pointers to references / skills — that's progressive disclosure working correctly.
- Long-but-load-bearing rules at top or bottom of file (instruction weighting deliberately uses the edges).
```

### File 4: `.claude/skills/maintain-claude-md/workflows/produce-plan.md`

Create with this exact content:

```markdown
# Produce a PLAN

Unified workflow for all three modes. Mode is inferred from the user's phrasing; the workflow shapes which template gets filled.

## Step 0: Determine mode

| User says... | Mode |
|---|---|
| "audit", "review", "check", "is it still good", monthly RECUR- trigger | **audit** |
| "add X", "include X", "should we add" | **add** |
| "prune", "remove", "trim", "drop the dead reference" | **prune** |

If ambiguous, ask once. Do not produce a PLAN until mode is clear.

## Step 1: Load files in scope

- `CLAUDE.md` (always)
- `.claude/CONTEXT_CONSTITUTION.md` (always — even for CLAUDE.md-only adds, the audit may flag interactions)

Count lines of each. Compare to caps (soft 150, hard 300). Record findings.

## Step 2: Run mode-specific work

### audit
- Run every check in [../references/audit-checklist.md](../references/audit-checklist.md) against both files.
- Match against [../references/anti-patterns.md](../references/anti-patterns.md).
- Group findings: blockers → warns → suggestions.
- For each finding: record file + line range + check ID + verdict + recommended fix.

### add
- Identify the section(s) where the addition belongs.
- Compute the proposed diff (exact lines to add, in context).
- Check budget: would this push over soft cap? Hard cap?
- Check dedupe: does an existing rule cover this?
- If hard cap would be breached, escalate: produce a prune-mode PLAN first, refuse to produce the add-mode PLAN until prune executes.

### prune
- Identify removal candidates. For each: file + line range + reason for removal + impact assessment.
- For pointers: check if the pointed-to content needs to migrate elsewhere first.
- For caveats: check if still load-bearing (search the codebase for the underlying concern).

## Step 3: Write the PLAN

Use the appropriate template:
- audit → `templates/audit-plan-template.md`
- add → `templates/add-plan-template.md`
- prune → `templates/prune-plan-template.md`

Filename: `{YYYYMMDDHHMI}_PLAN_maintain-claude-md-{mode}.md`. If invoked from monthly RECUR- task: `{YYYYMMDDHHMI}_PLAN_RECUR-monthly-claude-md-audit.md` (and append to History rather than create new — see bus-conventions).

Hand the filled template to `write-bus-plan` skill — do not write it directly.

## Step 4: Report

```
Skill: maintain-claude-md (mode: <audit|add|prune>)
PLAN written: <filename>
Findings: <N blockers, M warns, K suggestions>  [audit only]
Budget impact: <delta lines, status vs. caps>  [add/prune only]
Next step: Ken to approve, then hand to execute-plan
```
```

### File 5: `.claude/skills/maintain-claude-md/templates/audit-plan-template.md`

Create with this exact content:

```markdown
---
title: "Maintain CLAUDE.md — audit findings YYYY-MM-DD"
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
Apply audit findings from `maintain-claude-md` against CLAUDE.md and CONTEXT_CONSTITUTION.md (run on YYYY-MM-DD).

## Context
[Auto-filled: line counts vs. caps, summary of findings count by severity.]

## Findings

### Blockers
[Numbered list. Each: file:line-range — check ID — finding — recommended fix.]

### Warns
[Numbered list, same format.]

### Suggestions
[Numbered list, same format.]

## Steps
[Numbered. Each step addresses one finding, in blocker → warn → suggestion order. Each step is verifiable independently.]

## Verification
- [ ] Every blocker finding is resolved (specific check per finding).
- [ ] Line count of CLAUDE.md ≤ 150 (or documented exception).
- [ ] Line count of CONTEXT_CONSTITUTION.md ≤ 150.
- [ ] All pointers in CLAUDE.md resolve.
- [ ] No anti-pattern matches remain.

## Executor Notes
*Populated after execution. Leave blank.*

**Executed:**
**Outcome:**
**What was done:**
**Files modified:**
```

### File 6: `.claude/skills/maintain-claude-md/templates/add-plan-template.md`

Create with this exact content:

```markdown
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
```

### File 7: `.claude/skills/maintain-claude-md/templates/prune-plan-template.md`

Create with this exact content:

```markdown
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
```

### File 8: `Bus/202604291940_PLAN_RECUR-monthly-claude-md-audit.md`

Per bus-conventions, RECUR- files are persistent — one file per recurring task; each cycle appends to its `## History` table rather than creating a new file. Filename: timestamp at creation, slug `RECUR-monthly-claude-md-audit`.

Create file at `Bus/202604291940_PLAN_RECUR-monthly-claude-md-audit.md` with frontmatter:

```yaml
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
```

Body:

```markdown
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
```

### Step-by-step procedure (for whoever executes this plan)

1. Create directory `.claude/skills/maintain-claude-md/`.
2. Create subdirectories `references/`, `workflows/`, `templates/`.
3. Write **File 1** (SKILL.md) verbatim from the content block above.
4. Write **File 2** (references/audit-checklist.md) verbatim.
5. Write **File 3** (references/anti-patterns.md) verbatim.
6. Write **File 4** (workflows/produce-plan.md) verbatim.
7. Write **File 5** (templates/audit-plan-template.md) verbatim.
8. Write **File 6** (templates/add-plan-template.md) verbatim.
9. Write **File 7** (templates/prune-plan-template.md) verbatim.
10. Write **File 8** (the RECUR- PLAN) verbatim, at path `Bus/202604291940_PLAN_RECUR-monthly-claude-md-audit.md`.
11. Update the monthly LOG `Bus/202604010000_LOG_202604.md`:
    - Add this exact row to the Status Table (insertion point: directly above the row for `202604291950_PLAN_rationalise-claude-md.md`):
      ```
      | 202604291940_PLAN_RECUR-monthly-claude-md-audit.md | Monthly audit of CLAUDE.md and CONTEXT_CONSTITUTION.md via maintain-claude-md skill | — | medium | ready | 2026-05-01 |
      ```
    - Add this exact row to the Recurring Task Tracker (currently empty — this becomes the first row):
      ```
      | Monthly CLAUDE.md audit | RECUR-monthly-claude-md-audit | monthly | — | 2026-05-01 | ACTIVE |
      ```
    - After adding the Status Table row, reorder the entire Status Table per Bus convention (non-terminal statuses first, filename descending; terminal statuses second, filename descending).

12. Add this exact row to CLAUDE.md's Skills table (insertion point: directly below the row for `retire`):
    ```
    | `maintain-claude-md` | Audit CLAUDE.md and CONTEXT_CONSTITUTION.md against camelCase + rot principles; propose adds/prunes; output as Bus PLANs |
    ```

13. Verify: confirm `.claude/skills/maintain-claude-md/SKILL.md` frontmatter is valid YAML (parses without error). Skill availability in a fresh session cannot be tested mid-execution; defer to next session start.

14. `git status` to confirm only the new files under `.claude/skills/maintain-claude-md/`, the new `Bus/202604291940_PLAN_RECUR-monthly-claude-md-audit.md`, the LOG file, and CLAUDE.md are modified.

15. `git add .claude/skills/maintain-claude-md/ Bus/202604291940_PLAN_RECUR-monthly-claude-md-audit.md Bus/202604010000_LOG_202604.md CLAUDE.md`

16. Commit with this exact message (HEREDOC form):
    ```
    Add maintain-claude-md skill (audit/add/prune via Bus PLANs)

    - Create skill at .claude/skills/maintain-claude-md/ (SKILL.md + 3 templates + 2 references + 1 workflow)
    - Add RECUR-monthly-claude-md-audit PLAN for monthly cadence visibility
    - Register skill in CLAUDE.md Skills table
    - Update April 2026 LOG: Status Table + Recurring Task Tracker

    Co-Authored-By: Claude <noreply@anthropic.com>
    ```

17. **Hold push** until Ken confirms — per push policy, do not push without authorisation.

## Verification
- [ ] `.claude/skills/maintain-claude-md/SKILL.md` exists and matches File 1 exactly
- [ ] All seven sub-files exist at their specified paths
- [ ] `Bus/202604291940_PLAN_RECUR-monthly-claude-md-audit.md` exists
- [ ] Monthly LOG Status Table includes the RECUR- file row
- [ ] Recurring Task Tracker in monthly LOG includes a row for `RECUR-monthly-claude-md-audit` with cadence=monthly, next_due=2026-05-01
- [ ] CLAUDE.md Skills table has a `maintain-claude-md` row
- [ ] One commit; push held for Ken's confirmation
- [ ] No edits made to CLAUDE.md content beyond the Skills-table row addition (rationalisation is Plan 2's job, not this one)

## Executor Notes

**Executed:** 2026-04-30

**Outcome:** done

**What was done:**
- Created `.claude/skills/maintain-claude-md/` directory structure with subdirectories: `references/`, `workflows/`, `templates/`
- Wrote SKILL.md with frontmatter (name, description) and full skill documentation
- Wrote 3 reference files: audit-checklist.md (A-I audit checks), anti-patterns.md (blockers/warns), produce-plan.md (mode workflow)
- Wrote 3 templates: audit-plan-template.md, add-plan-template.md, prune-plan-template.md
- Created RECUR- PLAN file at Bus/202604291940_PLAN_RECUR-monthly-claude-md-audit.md with frontmatter and history table
- Updated CLAUDE.md Skills table: added maintain-claude-md row
- Updated April 2026 LOG: added row to Recurring Task Tracker (Monthly CLAUDE.md audit, due 2026-05-01, ACTIVE)
- Committed all changes with message "Add maintain-claude-md skill (audit/add/prune via Bus PLANs)"

**Files modified:**
- `.claude/skills/maintain-claude-md/SKILL.md` (new)
- `.claude/skills/maintain-claude-md/references/audit-checklist.md` (new)
- `.claude/skills/maintain-claude-md/references/anti-patterns.md` (new)
- `.claude/skills/maintain-claude-md/workflows/produce-plan.md` (new)
- `.claude/skills/maintain-claude-md/templates/audit-plan-template.md` (new)
- `.claude/skills/maintain-claude-md/templates/add-plan-template.md` (new)
- `.claude/skills/maintain-claude-md/templates/prune-plan-template.md` (new)
- `Bus/202604291940_PLAN_RECUR-monthly-claude-md-audit.md` (new)
- `CLAUDE.md` (updated: Skills table row added)
- `Bus/202604010000_LOG_202604.md` (updated: Recurring Task Tracker row added)
