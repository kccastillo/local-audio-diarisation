---
title: assigned_to Field — Valid Values
type: reference
description: Frontmatter field specifying which agent executes a PLAN file
---

# assigned_to Field

Specifies which agent will execute the plan steps.

## Valid Values

| Value | Agent | Context |
|---|---|---|
| `haiku` | Haiku (Claude Code) | **Default.** Haiku executes plan steps directly. Use this for most plans that involve Wiki writes, edits, or file operations within the vault. |
| `ken` | Ken (human) | Plan requires human action/decision before execution can begin. Plan status should be `blocked` with `blocked_by` explaining what Ken needs to do. |
| `antigravity` | Antigravity (Gemini) | **Optional override only.** Ken can request Antigravity instead of Haiku at execution time via Step 8 of `new-plan.md`. Do not use as the default `assigned_to` value in the PLAN file — always use `haiku`. |

## Workflow Notes

- **Default execution path** (new-plan.md Step 8): Haiku receives the PLAN and executes. 
- **When Sonnet drafts the PLAN**: Always set `assigned_to: haiku` unless the plan is blocked (set `assigned_to: ken` with `blocked_by` reason).

## Examples

```yaml
# Most plans (Haiku executes)
assigned_to: haiku
status: ready

# Plan blocked on Ken action
assigned_to: ken
status: blocked
blocked_by: "Awaiting confirmation of Matt's lease entitlements from employer"

# Plan ready but Haiku will execute (Ken may override to Antigravity at Step 8)
assigned_to: haiku
status: ready
```
