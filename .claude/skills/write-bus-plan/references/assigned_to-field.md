---
title: assigned_to Field — Valid Values
type: reference
description: Frontmatter field specifying who or what executes a PLAN file
---

# assigned_to Field

Free-text optional field naming the executor of the plan. Default empty (the active session executes).

## Valid Values

| Value | Meaning |
|---|---|
| `""` (empty) | **Default.** The active session executes the plan steps directly. |
| `ken` | Plan requires human action/decision before execution can begin. Plan status should be `blocked` with `blocked_by` explaining what Ken needs to do. |
| (free-text other) | Optional: a specific tool, sub-agent, or external system that should execute. Document the value's meaning where it's used. |

## Examples

```yaml
# Default — active session executes
assigned_to: ""
status: ready

# Plan blocked on Ken action
assigned_to: ken
status: blocked
blocked_by: "Awaiting confirmation of which two commands to keep inline in CLAUDE.md"
```
