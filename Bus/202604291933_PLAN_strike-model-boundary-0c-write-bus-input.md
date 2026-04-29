---
title: "Strike model-boundary 0c: write-bus-input skill (3 files)"
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
Strike model-role protocol from the `write-bus-input` skill. Three files: SKILL.md, workflows/write-input.md, templates/research-template.md. Preserve the RESEARCH vs ADVICE distinction (data vs strategy) — that's model-agnostic and useful.

## Context
Sub-plan 0c of the strike-model-boundary split.

## Steps

### File 1: `.claude/skills/write-bus-input/SKILL.md`

1. Replace the frontmatter `description` field. Find this exact line:
   ```
   description: Haiku's Bus input transcription skill. Writes RESEARCH (Haiku data drops) and ADVICE (Opus strategic notes, pasted by Ken) files into Bus/, updates the monthly LOG's Context Inputs table, and clears any PLAN's blocked state that was waiting on this input. Trigger phrases: "write this research to bus", "create research file", "write this advice", "paste opus advice", "record this input".
   ```
   Replace with:
   ```
   description: Bus input transcription skill. Writes RESEARCH (data drops) and ADVICE (strategic notes, e.g. Opus output pasted by Ken) files into Bus/, updates the monthly LOG's Context Inputs table, and clears any PLAN's blocked state that was waiting on this input. Trigger phrases: "write this research to bus", "create research file", "write this advice", "paste advice", "record this input".
   ```

2. Replace the `<essential_principles>` block. Find this exact block:
   ```
   <essential_principles>
   Haiku transcribes. It does not interpret, summarise, or modify the content Sonnet/Ken provides.
   RESEARCH and ADVICE are the same shape: context inputs that unblock PLANs. One skill handles both; the Type field distinguishes them.
   Always check the input's `feeds_plan` (RESEARCH) or `advises_plan` (ADVICE) frontmatter — if that PLAN is `blocked` waiting on this input, clear the blocked state.
   Always update the monthly LOG's Context Inputs table after writing an input file.
   Report back: filename written, LOG updated, PLAN(s) unblocked if any.
   </essential_principles>
   ```
   Replace with:
   ```
   <essential_principles>
   Transcribe input content accurately. Do not interpret, summarise, or modify content.
   RESEARCH and ADVICE are the same shape: context inputs that unblock PLANs. One skill handles both; the Type field distinguishes them.
   Always check the input's `feeds_plan` (RESEARCH) or `advises_plan` (ADVICE) frontmatter — if that PLAN is `blocked` waiting on this input, clear the blocked state.
   Always update the monthly LOG's Context Inputs table after writing an input file.
   Report back: filename written, LOG updated, PLAN(s) unblocked if any.
   </essential_principles>
   ```

3. In `<preconditions>`, find this exact line:
   ```
   - Sonnet has provided the content, target filename, and target plan (if any)
   ```
   Replace with:
   ```
   - Content, target filename, and target plan (if any) have been provided
   ```

4. In `<constraints>`, find this exact line:
   ```
   - Never modify input content — transcribe exactly what Sonnet/Ken provided
   ```
   Replace with:
   ```
   - Never modify input content — transcribe exactly as provided
   ```

5. In `<constraints>`, find this exact line:
   ```
   - An input file's `integration_status` stays `pending` until Sonnet integrates it into a PLAN; this skill does not mark integration complete
   ```
   Replace with:
   ```
   - An input file's `integration_status` stays `pending` until it is integrated into a PLAN; this skill does not mark integration complete
   ```

### File 2: `.claude/skills/write-bus-input/workflows/write-input.md`

6. Find this exact block:
   ```
   Step 1: Receive from Sonnet/Ken:
     - Type: RESEARCH | ADVICE
     - Content (full body)
     - Target filename (Sonnet generates per bus naming convention)
     - Target plan filename (if this input feeds/advises a specific plan)
     - Frontmatter values: question_asked, from (for RESEARCH), etc.
   ```
   Replace with:
   ```
   Step 1: Receive input content:
     - Type: RESEARCH | ADVICE
     - Content (full body)
     - Target filename (per bus naming convention)
     - Target plan filename (if this input feeds/advises a specific plan)
     - Frontmatter values: question_asked, from (for RESEARCH), etc.
   ```

7. In Step 4, find this exact line:
   ```
   | [input filename] | RESEARCH or ADVICE | [from — e.g. "Haiku", "Opus via Ken"] | [target plan filename or —] | pending |
   ```
   Replace with:
   ```
   | [input filename] | RESEARCH or ADVICE | [source — e.g. "websearch", "Opus via Ken", "user paste"] | [target plan filename or —] | pending |
   ```

8. In Step 5d, find this exact line:
   ```
       - If `assigned_to: ken` was set because of this block, flip back to `assigned_to: haiku`
   ```
   Replace with:
   ```
       - If `assigned_to: ken` was set because of this block, flip back to `assigned_to: ""`
   ```

### File 3: `.claude/skills/write-bus-input/templates/research-template.md`

9. Find this exact line:
   ```
   from: "Haiku"
   ```
   Replace with:
   ```
   from: ""
   ```

10. Find this exact line in the title:
    ```
    title: "Haiku Research — [topic]"
    ```
    Replace with:
    ```
    title: "Research — [topic]"
    ```

### Verify and commit

11. Verify: `grep -rin "haiku\|sonnet" .claude/skills/write-bus-input/` returns no matches.

12. `git add .claude/skills/write-bus-input/`

13. Commit with this exact message:
    ```
    Strike model-boundary from write-bus-input skill (0c)

    - Remove "Haiku transcribes" framing from SKILL.md
    - Update workflow Step 1 to be model-agnostic
    - Default research-template.md from field to empty string
    - Preserve RESEARCH vs ADVICE distinction (data vs strategy)
    - Part 3 of 5 (sub-plans 0a–0e)

    Co-Authored-By: Claude <noreply@anthropic.com>
    ```

14. Hold push for Ken's confirmation.

## Verification
- [ ] `grep -rin "haiku\|sonnet" .claude/skills/write-bus-input/` returns no matches
- [ ] `.claude/skills/write-bus-input/templates/research-template.md` `from:` field is empty string
- [ ] RESEARCH and ADVICE distinction preserved in SKILL.md essential_principles
- [ ] One commit; push held

## Executor Notes

**Executed:** 2026-04-30
**Outcome:** done
**What was done:**
1. Edited 3 files in .claude/skills/write-bus-input/: SKILL.md, workflows/write-input.md, research-template.md
2. Removed "Haiku transcribes" framing from SKILL.md essential_principles
3. Updated workflow Step 1 from "Receive from Sonnet/Ken" to "Receive input content"
4. Made workflow references model-agnostic (removed Sonnet references)
5. Changed research-template.md title from "Haiku Research" to "Research"
6. Changed research-template.md from field from "Haiku" to empty string
7. Preserved RESEARCH vs ADVICE distinction (data vs strategy)
8. Verified all changes via grep: no model-role phrases remain
9. Committed changes: ec25bb0
10. Push held for Ken confirmation

**Files modified:** 3 files in .claude/skills/write-bus-input/
