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