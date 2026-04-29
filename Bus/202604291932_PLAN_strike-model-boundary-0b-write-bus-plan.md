---
title: "Strike model-boundary 0b: write-bus-plan skill (5 files)"
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
triggers_plans: []
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: ""
---

## Objective
Strike model-role protocol from the `write-bus-plan` skill. Five files: SKILL.md, workflows/write-plan.md, templates/plan-template.md, references/assigned_to-field.md, references/bus-conventions.md, references/bus-naming-convention.md.

## Context
Sub-plan 0b of the strike-model-boundary split. Targets the most-affected skill — every file in this skill carries some model-role framing. Sub-decision (from the parent plan): keep the `assigned_to` frontmatter field as free-text optional owner; default empty.

## Steps

### File 1: `.claude/skills/write-bus-plan/SKILL.md`

1. Replace the frontmatter `description` field. Find this exact line:
   ```
   description: Haiku's Bus transcription skill. Use when Sonnet has finalised a plan and needs it written into System/Bus/ with correct schema. Also handles monthly LOG creation, LOG updates, and rollover file updates. Trigger phrases: "write this plan to bus", "create plan file", "update the log", "write bus file", "create the log".
   ```
   Replace with:
   ```
   description: Bus transcription skill. Use when a plan has been finalised and needs to be written into Bus/ with correct schema. Also handles monthly LOG creation, LOG updates, and rollover file updates. Trigger phrases: "write this plan to bus", "create plan file", "update the log", "write bus file", "create the log".
   ```

2. Replace the `<essential_principles>` block. Find this exact block:
   ```
   <essential_principles>
   Haiku transcribes. It does not think strategically or modify plan content.
   Write exactly what Sonnet provided — no additions, no omissions, no interpretation.
   Always check if the current month LOG exists before writing any PLAN file. Create it first if missing.
   After writing any PLAN file, update the monthly LOG's Status Table.
   For recurring PLAN files: append to the existing file's History table — do not create a new file.
   Report back: filename written, LOG updated, ready for next step.
   </essential_principles>
   ```
   Replace with:
   ```
   <essential_principles>
   Transcribe plan content accurately. Do not invent, summarise, or modify content.
   Always check if the current month LOG exists before writing any PLAN file. Create it first if missing.
   After writing any PLAN file, update the monthly LOG's Status Table.
   For recurring PLAN files: append to the existing file's History table — do not create a new file.
   Report back: filename written, LOG updated, ready for next step.
   </essential_principles>
   ```

3. In the `<constraints>` block, find this exact line:
   ```
   - Never modify plan content — transcribe exactly what Sonnet provides
   ```
   Replace with:
   ```
   - Never modify plan content — transcribe exactly as specified
   ```

4. In the `<success_criteria>` block, find this exact line:
   ```
   - Ken has been given the PLAN filename for handoff to Haiku
   ```
   Replace with:
   ```
   - Ken has been given the PLAN filename for next-step handoff
   ```

### File 2: `.claude/skills/write-bus-plan/workflows/write-plan.md`

5. Find this exact line:
   ```
   Step 1: Receive from Sonnet:
   ```
   Replace with:
   ```
   Step 1: Receive plan content:
   ```

6. Find this exact heading and bullet:
   ```
   Step 5: Report to Ken:
   ```
   Leave heading unchanged (it's referring to Ken the human, not a model).

### File 3: `.claude/skills/write-bus-plan/templates/plan-template.md`

7. Find this exact line:
   ```
   assigned_to: haiku
   ```
   Replace with:
   ```
   assigned_to: ""
   ```

8. Find this exact line:
   ```
   created_by: haiku
   ```
   Replace with:
   ```
   created_by: ""
   ```

9. Find this exact line in the Executor Notes section:
   ```
   *Populated by Haiku after execution via `execute-plan`. Leave blank.*
   ```
   Replace with:
   ```
   *Populated after execution via `execute-plan`. Leave blank.*
   ```

10. Find this exact line in the Steps section:
    ```
    [Numbered steps for Haiku to execute via the `execute-plan` skill. Each step must be independently verifiable.
    ```
    Replace with:
    ```
    [Numbered steps to execute via the `execute-plan` skill. Each step must be independently verifiable.
    ```

11. Find this exact line:
    ```
    Mark [Ken] for steps that require human action before Haiku can proceed.
    ```
    Replace with:
    ```
    Mark [Ken] for steps that require human action before execution can proceed.
    ```

### File 4: `.claude/skills/write-bus-plan/references/assigned_to-field.md`

12. Replace the entire file contents with this exact content:

    ```markdown
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
    ```

### File 5: `.claude/skills/write-bus-plan/references/bus-conventions.md`

13. Find this exact bullet line in the "PLAN Status Lifecycle" section:
    ```
    - `ready` — transcribed, not yet started. Set by Haiku on creation.
    ```
    Replace with:
    ```
    - `ready` — transcribed, not yet started. Set on creation.
    ```

14. Find this exact bullet line:
    ```
    - `in-progress` — execute-plan skill has started. Set by Haiku on execution start.
    ```
    Replace with:
    ```
    - `in-progress` — execute-plan skill has started. Set when execution begins.
    ```

### File 6: `.claude/skills/write-bus-plan/references/bus-naming-convention.md`

15. Replace the entire "Type Tokens" table. Find this exact block:
    ```
    | Token | What it is | Created by | Consumed by |
    |---|---|---|---|
    | `LOG` | Monthly rollup log | Haiku | All agents for status |
    | `PLAN` | Actionable task for Haiku | Haiku | Haiku (executes) |
    | `RESEARCH` | Haiku data drop | Haiku | Sonnet (integrates) |
    | `ADVICE` | Opus strategic note (Ken pastes) | Ken | Sonnet (integrates) |
    ```
    Replace with:
    ```
    | Token | What it is |
    |---|---|
    | `LOG` | Monthly rollup log |
    | `PLAN` | Actionable task to execute |
    | `RESEARCH` | Data drop — feeds a PLAN |
    | `ADVICE` | Strategic note (e.g., Ken pastes Opus output) — feeds a PLAN |
    ```

16. Find this exact line in "RESEARCH and ADVICE files":
    ```
    Written via the `write-bus-input` skill — see [.claude/skills/write-bus-input/SKILL.md](.claude/skills/write-bus-input/SKILL.md) for content rules. Sonnet pre-generates the target filename and gives it to Haiku before the input is written. This allows PLANs to pre-link via `linked_research` / `linked_advice` before the file exists.
    ```
    Replace with:
    ```
    Written via the `write-bus-input` skill — see [.claude/skills/write-bus-input/SKILL.md](.claude/skills/write-bus-input/SKILL.md) for content rules. The target filename is generated when the PLAN is drafted, so the PLAN can pre-link via `linked_inputs` before the input file exists.
    ```

### Verify and commit

17. Verify by grep across the skill directory: `grep -rin "haiku\|sonnet thinks\|opus reviews" .claude/skills/write-bus-plan/` returns no matches.

18. `git add .claude/skills/write-bus-plan/`

19. Commit with this exact message:
    ```
    Strike model-boundary from write-bus-plan skill (0b)

    - Remove "Haiku transcribes" framing from SKILL.md essential_principles
    - Default plan-template.md assigned_to and created_by to empty strings
    - Rewrite assigned_to-field.md as free-text optional owner
    - Strip "Set by Haiku" lines from bus-conventions.md
    - Drop model-name columns from bus-naming-convention.md Type Tokens table
    - Part 2 of 5 (sub-plans 0a–0e)

    Co-Authored-By: Claude <noreply@anthropic.com>
    ```

20. Hold push for Ken's confirmation.

## Verification
- [ ] `grep -rin "haiku\|sonnet thinks\|opus reviews" .claude/skills/write-bus-plan/` returns no matches
- [ ] `.claude/skills/write-bus-plan/templates/plan-template.md` line for `assigned_to:` reads `assigned_to: ""`
- [ ] `.claude/skills/write-bus-plan/SKILL.md` `<essential_principles>` no longer mentions Haiku or Sonnet
- [ ] `.claude/skills/write-bus-plan/references/bus-naming-convention.md` Type Tokens table has 2 columns (Token + What it is), not 4
- [ ] One commit; push held

## Executor Notes

**Executed:** 2026-04-30
**Outcome:** done
**What was done:**
1. Edited 6 files in .claude/skills/write-bus-plan/: SKILL.md, workflows/write-plan.md, templates/plan-template.md, assigned_to-field.md, bus-conventions.md, bus-naming-convention.md
2. Removed all model-role protocol references (Haiku, Sonnet, Opus) from skill descriptions and templates
3. Changed assigned_to defaults to empty string (active session executes)
4. Rewrote assigned_to-field.md to describe it as free-text optional owner
5. Simplified Type Tokens table from 4 columns to 2 columns
6. Updated all workflow references from model names to generic role descriptions
7. Verified all changes via grep: no model-role phrases remain
8. Committed changes: 556100c
9. Push held for Ken confirmation

**Files modified:** 6 files in .claude/skills/write-bus-plan/
