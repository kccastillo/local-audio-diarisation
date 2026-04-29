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
