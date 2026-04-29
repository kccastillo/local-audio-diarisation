---
title: "Strike model-boundary 0d: execute-plan skill (2 files)"
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
Strike model-role protocol from the `execute-plan` skill. Two files: SKILL.md (heavy edits — `<haiku_safe_definition>` block must be renamed and rewritten) and workflows/execute-steps.md.

**Sub-decision (made in this plan, not deferred):** rename the `<haiku_safe_definition>` block to `<plan_safe_definition>`. The concept (executable mechanically with no design work) is useful and model-agnostic; just the name carries the model assumption.

## Context
Sub-plan 0d. The `execute-plan` skill is the most heavily model-bound — its description, essential_principles, the haiku_safe_definition block, and the commit template all reference Haiku. Every reference is rewritten with exact replacement text.

## Steps

### File 1: `.claude/skills/execute-plan/SKILL.md`

1. Replace the frontmatter `description` field. Find this exact line:
   ```
   description: Haiku's plan execution skill. Runs a PLAN from Bus/ end-to-end — executes steps in order, populates Executor Notes, updates the monthly LOG Status Table, commits, and pushes to origin. Trigger phrases: "execute this plan", "implement the plan", "run PLAN_x", "ok implement".
   ```
   Replace with:
   ```
   description: Plan execution skill. Runs a PLAN from Bus/ end-to-end — executes steps in order, populates Executor Notes, updates the monthly LOG Status Table, commits, and pushes to origin. Trigger phrases: "execute this plan", "implement the plan", "run PLAN_x", "ok implement".
   ```

2. Replace the `<essential_principles>` block. Find this exact block:
   ```
   <essential_principles>
   Haiku executes — does not redesign, re-scope, or improve the plan mid-flight.
   Execute steps in order. Verify each before moving on.
   If a step is ambiguous, unsafe, or marked [Ken]: halt and flag. Do not improvise.
   Always populate Executor Notes AND update the LOG before git commit.
   Git commit + push is the final step — never skip; never use --no-verify or --force.
   On closure (outcome=done), if frontmatter sets closes_thread / advances_thread / parent_plan_of_plans, apply the roadmap sync (workflows/execute-steps.md Step 4.5) before LOG update.
   </essential_principles>
   ```
   Replace with:
   ```
   <essential_principles>
   Execute the plan as written — do not redesign, re-scope, or improve mid-flight.
   Execute steps in order. Verify each before moving on.
   If a step is ambiguous, unsafe, or marked [Ken]: halt and flag. Do not improvise.
   Always populate Executor Notes AND update the LOG before git commit.
   Git commit + push is the final step — never skip; never use --no-verify or --force.
   On closure (outcome=done), if frontmatter sets closes_thread / advances_thread / parent_plan_of_plans, apply the roadmap sync (workflows/execute-steps.md Step 4.5) before LOG update.
   </essential_principles>
   ```

3. In `<preconditions>`, find this exact line:
   ```
   - Ken has authorised execution (trigger phrase in this skill's description)
   ```
   Leave unchanged (refers to Ken the human).

4. Replace the `<haiku_safe_definition>` block in full. Find this exact block:
   ```
   <haiku_safe_definition>
   A plan is "Haiku-safe" when every step is:
   - **Concrete:** specific file paths, exact command syntax, no "likely" or "probably"
   - **Unambiguous:** no judgment calls; Haiku executes, not redesigns
   - **Atomic:** one step at a time; clear success/failure condition
   - **Safe:** no destructive operations without explicit Ken approval; no bypasses (--no-verify, --force)
   - **Testable:** verification criteria are independent and checkable

   Example of Haiku-safe: "Read SKILL.md from `.claude/skills/atomise/SKILL.md`. Verify frontmatter `name: atomise`. Extract `<process>` block (lines 43–68) to `workflows/atomise-steps.md`. Commit with message 'chore: trim atomise SKILL.md'"

   Example of NOT Haiku-safe: "Audit SKILL.md and extract any residual content. Use your judgment." (ambiguous, requires interpretation)

   **When authoring plans for Haiku:** Sonnet must refer to [workflows/execute-steps.md](workflows/execute-steps.md) for the execution protocol and [references/log-rules.md](references/log-rules.md) for the LOG contract, and ensure every step passes this definition.
   </haiku_safe_definition>
   ```
   Replace with:
   ```
   <plan_safe_definition>
   A plan is "plan-safe" (executable mechanically, without design work) when every step is:
   - **Concrete:** specific file paths, exact command syntax, no "likely" or "probably"
   - **Unambiguous:** no judgment calls; the executor runs the steps, not redesigns them
   - **Atomic:** one step at a time; clear success/failure condition
   - **Safe:** no destructive operations without explicit Ken approval; no bypasses (--no-verify, --force)
   - **Testable:** verification criteria are independent and checkable

   Example of plan-safe: "Read SKILL.md from `.claude/skills/atomise/SKILL.md`. Verify frontmatter `name: atomise`. Extract `<process>` block (lines 43–68) to `workflows/atomise-steps.md`. Commit with message 'chore: trim atomise SKILL.md'"

   Example of NOT plan-safe: "Audit SKILL.md and extract any residual content. Use your judgment." (ambiguous, requires interpretation)

   **When authoring plans:** refer to [workflows/execute-steps.md](workflows/execute-steps.md) for the execution protocol and [references/log-rules.md](references/log-rules.md) for the LOG contract, and ensure every step passes this definition.
   </plan_safe_definition>
   ```

5. In `<skill_invocation_semantics>`, find this exact block:
   ```
   <skill_invocation_semantics>
   **Invoking a skill from a PLAN step:**

   When a PLAN step says "Invoke `Skill("skill-name", "args")`", Haiku's role is:
   1. Call `Skill("skill-name", "args")`
   2. Read the returned SKILL.md documentation (skill call returns the skill's SKILL.md)
   3. Execute the documented workflow steps yourself using your tools (Read, Write, Edit, Bash, Glob, Grep, etc.)
   4. The Skill() call *loads the instructions*; you are the executor of those instructions

   The skill framework does not self-execute. Haiku reads the skill's workflow files and implements the steps using the available tools.

   **Example:** A PLAN step says "Run atomise on break_glass_requirement.md". Haiku:
   - Calls `Skill("atomise", "mode:production file:Production/Staging/break_glass_requirement.md")`
   - Reads the returned SKILL.md, which references [workflows/atomise-steps.md](workflows/atomise-steps.md)
   - Reads atomise-steps.md to see the detailed workflow (Steps 1–7)
   - Executes each step manually using Read, Write, Bash, etc., creating atom files as directed
   </skill_invocation_semantics>
   ```
   Replace with:
   ```
   <skill_invocation_semantics>
   **Invoking a skill from a PLAN step:**

   When a PLAN step says "Invoke `Skill("skill-name", "args")`", the executor's role is:
   1. Call `Skill("skill-name", "args")`
   2. Read the returned SKILL.md documentation (skill call returns the skill's SKILL.md)
   3. Execute the documented workflow steps using available tools (Read, Write, Edit, Bash, Glob, Grep, etc.)
   4. The Skill() call *loads the instructions*; the executor runs them

   The skill framework does not self-execute. The executor reads the skill's workflow files and implements the steps using the available tools.

   **Example:** A PLAN step says "Run atomise on break_glass_requirement.md". The executor:
   - Calls `Skill("atomise", "mode:production file:Production/Staging/break_glass_requirement.md")`
   - Reads the returned SKILL.md, which references [workflows/atomise-steps.md](workflows/atomise-steps.md)
   - Reads atomise-steps.md to see the detailed workflow (Steps 1–7)
   - Executes each step manually using Read, Write, Bash, etc., creating atom files as directed
   </skill_invocation_semantics>
   ```

6. In `<constraints>`, find this exact line:
   ```
   - If a PLAN step requires tools or permissions Haiku lacks: halt, flag, escalate to Sonnet
   ```
   Replace with:
   ```
   - If a PLAN step requires tools or permissions the executor lacks: halt, flag, escalate to Ken
   ```

7. In `<success_criteria>`, find this exact line:
   ```
   - Sonnet/Ken has been given the final report: filename, outcome, LOG path, commit hash
   ```
   Replace with:
   ```
   - Ken has been given the final report: filename, outcome, LOG path, commit hash
   ```

### File 2: `.claude/skills/execute-plan/workflows/execute-steps.md`

8. In Step 2, find this exact line:
   ```
   - If a step is marked [Ken]: halt, output MODEL SWITCH NEEDED to Sonnet requesting Ken's action
   ```
   Replace with:
   ```
   - If a step is marked [Ken]: halt, surface to Ken with the specific action required, and wait for response
   ```

9. In Step 7, find this exact block in the commit template:
   ```
   - Commit via HEREDOC with format:
     ```
     {Subject: ≤72 chars, adapted from PLAN title, reflects what was done}

     - {bullet from Executor Notes "What was done"}
     - {...}

     Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
     ```
   ```
   Replace with:
   ```
   - Commit via HEREDOC with format:
     ```
     {Subject: ≤72 chars, adapted from PLAN title, reflects what was done}

     - {bullet from Executor Notes "What was done"}
     - {...}

     Co-Authored-By: Claude <noreply@anthropic.com>
     ```
   ```

10. In Step 8, find this exact heading:
    ```
    ## Step 8: Report to Sonnet/Ken
    ```
    Replace with:
    ```
    ## Step 8: Report to Ken
    ```

### Verify and commit

11. Verify: `grep -rin "haiku\|sonnet" .claude/skills/execute-plan/` returns no matches.

12. Verify the renamed block: `grep -n "plan_safe_definition\|haiku_safe_definition" .claude/skills/execute-plan/SKILL.md`. Expected: two matches for `plan_safe_definition` (open and close tags), zero for `haiku_safe_definition`.

13. `git add .claude/skills/execute-plan/`

14. Commit with this exact message:
    ```
    Strike model-boundary from execute-plan skill (0d)

    - Remove "Haiku executes" framing from SKILL.md essential_principles
    - Rename <haiku_safe_definition> to <plan_safe_definition>
    - Update skill_invocation_semantics to model-agnostic phrasing
    - Drop Haiku reference from commit Co-Authored-By template
    - Part 4 of 5 (sub-plans 0a–0e)

    Co-Authored-By: Claude <noreply@anthropic.com>
    ```

15. Hold push for Ken's confirmation.

## Verification
- [ ] `grep -rin "haiku\|sonnet" .claude/skills/execute-plan/` returns no matches
- [ ] `.claude/skills/execute-plan/SKILL.md` contains `<plan_safe_definition>` (open and close), no `<haiku_safe_definition>`
- [ ] Commit Co-Authored-By template no longer mentions Claude Haiku 4.5
- [ ] One commit; push held

## Executor Notes

**Executed:** 2026-04-30
**Outcome:** done
**What was done:**
1. Edited SKILL.md: removed "Haiku executes" framing from essential_principles
2. Edited SKILL.md: replaced <haiku_safe_definition> block with <plan_safe_definition> (renamed tag, updated all references from "Haiku-safe" to "plan-safe")
3. Edited SKILL.md: updated skill_invocation_semantics to use "executor" instead of "Haiku"
4. Edited SKILL.md: replaced constraints line about "Haiku lacks" with "executor lacks"
5. Edited SKILL.md: updated success_criteria from "Sonnet/Ken" to just "Ken"
6. Edited workflows/execute-steps.md: changed Step 2 Ken handling text from MODEL SWITCH to surface to Ken
7. Edited workflows/execute-steps.md: updated commit Co-Authored-By from "Claude Haiku 4.5" to "Claude"
8. Edited workflows/execute-steps.md: changed heading "Step 8: Report to Sonnet/Ken" to "Step 8: Report to Ken"
9. Edited log-rules.md and workflows execute-steps.md to remove remaining Sonnet references
10. Verified all changes: no haiku/sonnet references remain, plan_safe_definition renamed correctly
11. Committed changes: e84becb
12. Push held for Ken confirmation

**Files modified:** 3 files in .claude/skills/execute-plan/
