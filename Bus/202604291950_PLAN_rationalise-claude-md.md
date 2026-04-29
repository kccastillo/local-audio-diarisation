---
title: "Rationalise CLAUDE.md using maintain-claude-md skill: write Context Constitution, externalise to .claude/references/, slim CLAUDE.md"
type: bus-plan
status: ready
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
Apply the first real run of `maintain-claude-md` against CLAUDE.md. The work has three concrete outputs: (1) a new `.claude/CONTEXT_CONSTITUTION.md` codifying context-rot mitigation rules; (2) `.claude/references/` directory with four extracted reference files; (3) a slimmed CLAUDE.md (~80 lines, well under the 150-line soft cap) consisting of pointers + Trinity + Working style + Agent execution rules + Skills table + Caveats + Further Documentation.

## Context

**Depends on:**
- Plan 202604291931_PLAN_strike-model-boundary-0a-claude-md.md (must execute first — CLAUDE.md's "Agent execution rules" section is being edited by both plans; 0a must land first to avoid merge conflicts). Plans 0b–0e do not touch CLAUDE.md and are not blockers for this plan.
- Plan 202604291940_PLAN_build-maintain-claude-md-skill.md (must execute first — this plan invokes the skill).

**Confirmed inputs from Ken (2026-04-29):**
- Two commands kept inline in slimmed CLAUDE.md: `python run_diariser.py --input <file>` and `pytest tests/ -v`. (Confirmed: "those are the two that would be run the most often, cheers.")
- Recovery on rot detection: flag and stop.
- Compression thresholds: single tool result ≥10k tokens triggers summarisation; conversation at ≥80% context triggers proactive compression. (Sonnet's decision per Ken's "check standard practice and decide".)
- Isolate / subagent triggers: any of (a) >3 search queries, (b) raw result would exceed ~10k tokens, (c) well-scoped task with conclusion returnable in <500 words. (Sonnet's decision.)
- Add mode: diff proposed as a Bus PLAN, not direct edit. (Already encoded in the skill.)
- Audit output: Bus PLAN, not chat or ADVICE.
- Line cap: soft 150, hard 300.
- Cadence: monthly via RECUR- task. (Already created in Plan 202604291940.)

## Steps

### Phase A — Audit pass

1. Invoke `Skill("maintain-claude-md")` with phrasing "audit CLAUDE.md". Confirm the skill produces an audit-results PLAN file in `Bus/`. If the skill returns no PLAN file, or returns only chat output: halt this plan, set its frontmatter `status: needs-revision`, record in Executor Notes that Plan 202604291940 (build skill) appears broken because the skill did not produce a PLAN file, and stop. Do not attempt subsequent steps.

2. Compare the skill's audit output to the **expected findings list below**. If the skill's findings substantially match (the listed blockers/warns appear, plus or minus minor differences in how they're phrased): proceed to Phase B. If the skill's findings diverge meaningfully (a listed blocker is missing, or new blockers appear that aren't listed): halt the plan, set `status: needs-revision`, record the divergence in Executor Notes, and stop. Do not attempt Phase B if the skill produced unexpected output — investigate the skill first.

   **Expected findings (audit of CLAUDE.md as it stands at plan-execution time):**

   *Blockers (per audit-checklist Section B and E):*
   - **B1/B2 — Trinity partial.** Project Oneliner present (line ~7); Caveats present ("Important Implementation Notes", lines ~153–161); Key Commands section is bloated (~47 lines) — fails the "≤3 most-frequent commands inline" check.
   - **H4 — No Context Constitution pointer.** `.claude/CONTEXT_CONSTITUTION.md` is not referenced from CLAUDE.md (file does not yet exist; will be created in Phase B).
   - **D — Anti-pattern: codebase duplication.** "Key Dependencies" section duplicates `requirements.txt`. Architecture, Configuration, and Testing sections duplicate codebase-derivable facts.

   *Warns:*
   - **F1 — Progressive-disclosure opportunities.** Architecture (lines ~46–75), Commands (~86–132), Configuration (~134–144), Testing (~146–151) are all ≥10-line inline sections that should move to `.claude/references/`.

   *Suggestions:*
   - **C — Instruction weighting.** Critical content (Working style, Agent execution rules) sits in the upper third — acceptable but not optimised. Top-of-file pointer to Constitution will improve weighting.
   - **H1, H2 — Context-rot positioning.** Bus/ scratchpad pattern is documented (good); subagent triggers are not explicitly documented in CLAUDE.md (acceptable — will live in CONTEXT_CONSTITUTION.md instead).

   *No anti-pattern matches expected for:*
   - Working style (legitimate user preferences, not natural-language linting).
   - Skills table (small, current).
   - Agent execution rules (post-Plan-0a edits — model-boundary content already removed).

### Phase B — Write the Context Constitution

3. Create `.claude/CONTEXT_CONSTITUTION.md` with this exact content:

```markdown
---
title: Context Constitution
purpose: Rules for how Claude operates within a session to mitigate context rot.
last_reviewed: 2026-04-29
---

# Context Constitution

**IMPORTANT.** This file is referenced from the top of CLAUDE.md. It governs how Claude manages its own context window during a session. The four rot modes name failure patterns; the four fixes name interventions; the project-specific rules below translate them into actions.

## The four rot modes

- **Poisoning** — a hallucinated fact enters context and gets reinforced. One bad assumption snowballs into a chain of derivations.
- **Distraction** — long conversation history dilutes attention to the system prompt and earliest instructions.
- **Confusion** — too many concurrent documents or tool results piled into one window; signals get crossed.
- **Clash** — old and new versions of the same fact both present; which is true becomes ambiguous.

## The four fixes

- **Write** — externalise state. Plans, scratch notes, decisions go to files (Bus/, memory) instead of being held in chat.
- **Select** — just-in-time retrieval. Pull only what's needed for the current step; don't pre-load.
- **Compress** — summarise old or oversized content; preserve meaning, drop bulk.
- **Isolate** — delegate well-scoped work to a subagent so its raw outputs never enter the main context.

## Project-specific rules

### R1. Plans go to Bus/, not chat
**IMPORTANT.** Any planned work — multi-step implementations, refactors, audits, decisions with consequences — is written as a Bus PLAN before execution. Chat is for negotiation; Bus/ is the scratchpad. This is the project's primary **Write** discipline.

### R2. Memory system for cross-session facts
User profile, feedback, project state, and external references go to `memory/` files via the auto-memory system. Do not re-derive these each session.

### R3. Compress oversized tool results
**MUST.** Any single tool output ≥10k tokens (≈40KB or several screens) is summarised or extracted before being reasoned over. Save the raw output as a working note (`Bus/` or scratchpad file) and continue from the summary. This is **Compress** at the per-result level.

### R4. Compress conversation at 80% context
When the conversation approaches 80% of the model's context window, proactively summarise the earliest content into a brief and continue from the summary. Do not wait for auto-truncation — that's a **Distraction** failure mode in slow motion.

### R5. Isolate broad searches
**MUST.** Delegate to a subagent (typically the Explore agent) when any of: (a) the task needs >3 search/grep queries to answer, (b) raw results would exceed ~10k tokens before filtering, (c) the task is scoped enough that a worker can return its conclusion in <500 words. The Manager session never sees the raw bulk — only the worker's summary.

### R6. Label external content
When integrating quoted user pastes, web-fetch results, or third-party documentation into reasoning, mark the source explicitly (e.g., "Per the user-pasted spec:", "From the fetched docs at <url>:"). This prevents **Confusion** between user instruction, author opinion, and external claim.

### R7. Halt on Clash
**IMPORTANT.** If two facts in context contradict — file says X, earlier conversation says Y, memory says Z — stop and surface the clash. Do not silently pick one. Naming the conflict to the user is the recovery; never paper over it by guessing.

### R8. Halt on suspected rot
**IMPORTANT.** If mid-task evidence suggests a previous step was based on a hallucinated fact (Poisoning) or that a chain of reasoning has drifted from grounded facts, stop. Surface the suspicion to the user. Do not attempt self-recovery silently. Per Ken's directive (2026-04-29): "flag to me and stop."

### R9. Verify memory before relying on it
A memory record names a fact at the time of writing. Before acting on a memory record that names a specific file, function, or flag — verify it still exists. Memory is a hint, not a guarantee.

### R10. Edge-place critical instructions
**Lost-in-the-middle** is real. When writing CLAUDE.md, instructions, or PLAN files: critical content goes in the top ~30 lines or bottom ~20 lines, never buried mid-document. Use IMPORTANT / MUST / NEVER markers on load-bearing rules.

## Recovery protocol

When R7 or R8 fires:
1. Stop the current work step immediately.
2. State plainly to the user: which fact is suspect, why, what depended on it.
3. Wait for direction. Options the user can give: confirm the fact (resume), correct the fact (replan), restart conversation (preserve nothing), or abandon the task.
4. After resolution, write a short note to memory if the rot was caused by a stale or wrong memory entry — so future sessions don't hit the same trap.

## Maintenance

This file is in scope for the `maintain-claude-md` skill. Soft cap 150 lines, hard cap 300 lines. Audited monthly via the RECUR-monthly-claude-md-audit task.
```

### Phase C — Create reference directory

4. Create directory `.claude/references/`.

**Note on extraction anchors:** Plan 0a edits CLAUDE.md's "Agent execution rules" section before this plan runs, so absolute line numbers will have shifted. Use **section-header anchors** (find the heading line, copy through the line before the next `## ` heading at the same level) rather than line numbers throughout this plan.

5. Create `.claude/references/architecture.md`. Content (in this order):
    - H1 title: `# Architecture & Core Design`
    - One-line preamble: `Extracted from CLAUDE.md (2026-04-29) — see CLAUDE.md for project overview and entry-points.`
    - Blank line
    - Verbatim copy of the section in CLAUDE.md beginning with the line `## Architecture & Core Design` and ending at the line immediately before `## Key Dependencies`. Copy the section body but NOT the `## Architecture & Core Design` heading itself (the new H1 above replaces it).

6. Create `.claude/references/commands.md`. Content (in this order):
    - H1 title: `# Commands`
    - Preamble: `Extracted from CLAUDE.md (2026-04-29) — full command reference. CLAUDE.md keeps only the two most-frequent.`
    - Blank line
    - Verbatim copy of the section in CLAUDE.md beginning with the line `## Commands` and ending at the line immediately before `## Configuration`. Copy the section body but NOT the `## Commands` heading itself.

7. Create `.claude/references/configuration.md`. Content (in this order):
    - H1 title: `# Configuration`
    - Preamble: `Extracted from CLAUDE.md (2026-04-29) — config schema reference. config/config.yaml is authoritative.`
    - Blank line
    - Verbatim copy of the section in CLAUDE.md beginning with the line `## Configuration` and ending at the line immediately before `## Testing Structure`. Copy the section body but NOT the `## Configuration` heading itself.

8. Create `.claude/references/testing.md`. Content (in this order):
    - H1 title: `# Testing Structure`
    - Preamble: `Extracted from CLAUDE.md (2026-04-29) — pytest conventions and structure.`
    - Blank line
    - Verbatim copy of the section in CLAUDE.md beginning with the line `## Testing Structure` and ending at the line immediately before `## Important Implementation Notes`. Copy the section body but NOT the `## Testing Structure` heading itself.

### Phase D — Slim CLAUDE.md

9. Edit CLAUDE.md as a series of discrete edits. Apply these in order. Anchors are section headings (`## Heading` lines), not line numbers.

   **9a. Insert Context Constitution pointer.** Find the line `## Project Overview`. Immediately before it (and after the file's opening paragraph that begins "This file provides guidance..."), insert this exact block followed by a blank line:

   ````
   ## Context Constitution

   **IMPORTANT.** Read [.claude/CONTEXT_CONSTITUTION.md](.claude/CONTEXT_CONSTITUTION.md) at session start. It governs how I manage context (rot modes, fixes, project-specific rules R1–R10). The most load-bearing rules: plans go to Bus/, compress oversized results, isolate broad searches, halt on clash or suspected rot.

   ````

   **9b. Project Overview, Working style, Agent execution rules, Skills:** leave these sections unchanged. (Agent execution rules has already been edited by Plan 0a; Skills table has already been edited by Plan 202604291940 to add the `maintain-claude-md` row.)

   **9c. Replace Architecture section.** Find the section starting with `## Architecture & Core Design` and ending at the line immediately before `## Key Dependencies`. Replace the entire range (heading + body) with this exact block:

   ````
   ## Architecture
   See [.claude/references/architecture.md](.claude/references/architecture.md) — pipeline stages, processor pattern, config singleton.
   ````

   **9d. Delete Key Dependencies section.** Find the section starting with `## Key Dependencies` and ending at the line immediately before `## Commands`. Delete the entire range (heading + body, plus the trailing blank line if present). Do not replace.

   **9e. Replace Commands section.** Find the section starting with `## Commands` and ending at the line immediately before `## Configuration`. Replace the entire range (heading + body) with this exact block (note: the inner ```bash fence is part of the replacement content):

   ````
   ## Key Commands

   ```bash
   python run_diariser.py --input <file>   # main pipeline
   pytest tests/ -v                        # full test run
   ```

   Full command reference: [.claude/references/commands.md](.claude/references/commands.md).
   ````

   **9f. Replace Configuration section.** Find the section starting with `## Configuration` and ending at the line immediately before `## Testing Structure`. Replace the entire range (heading + body) with this exact block:

   ````
   ## Configuration
   Schema: [.claude/references/configuration.md](.claude/references/configuration.md). Authoritative source: [config/config.yaml](config/config.yaml).
   ````

   **9g. Replace Testing Structure section.** Find the section starting with `## Testing Structure` and ending at the line immediately before `## Important Implementation Notes`. Replace the entire range (heading + body) with this exact block:

   ````
   ## Testing
   See [.claude/references/testing.md](.claude/references/testing.md) — numbering convention, fixtures, end-to-end test position.
   ````

   **9h. Important Implementation Notes:** leave this section unchanged — it stays inline as the Trinity's third element (caveats).

   **9i. Append Further Documentation section.** At the end of the file (after the last line of Important Implementation Notes), insert this exact block:

   ````

   ## Further Documentation
   - [.claude/CONTEXT_CONSTITUTION.md](.claude/CONTEXT_CONSTITUTION.md) — context-rot rules
   - [.claude/references/architecture.md](.claude/references/architecture.md)
   - [.claude/references/commands.md](.claude/references/commands.md)
   - [.claude/references/configuration.md](.claude/references/configuration.md)
   - [.claude/references/testing.md](.claude/references/testing.md)
   - [AGENT_RULES.md](AGENT_RULES.md) — plan execution lifecycle (post strike-model-boundary)
   - [.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md](.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md)
   ````

### Phase E — Verify and commit

10. Verify CLAUDE.md final line count:
    - If line count ≤ 120: pass; proceed.
    - If line count > 120 and ≤ 150: pass with note (record actual count in Executor Notes); proceed.
    - If line count > 150: halt the plan, set `status: needs-revision`, record the count in Executor Notes, and stop. Do not improvise pruning — the rationalisation steps above produce a deterministic output, and a value over 150 means a step was misapplied.

11. Verify each pointer in slimmed CLAUDE.md resolves to an existing file. Run `ls` against each of these exact paths:
    - `.claude/CONTEXT_CONSTITUTION.md`
    - `.claude/references/architecture.md`
    - `.claude/references/commands.md`
    - `.claude/references/configuration.md`
    - `.claude/references/testing.md`
    - `config/config.yaml`
    - `AGENT_RULES.md`
    - `.claude/skills/SKILLS_IMPLEMENTATION_GUIDE.md`

    If any path returns "No such file": halt this plan, set `status: needs-revision`, record which path is missing in Executor Notes.

12. Re-run `Skill("maintain-claude-md")` audit pass against the new CLAUDE.md and `.claude/CONTEXT_CONSTITUTION.md`. Expected outcome: zero blockers. If blockers appear: halt the plan, `status: needs-revision`, record the blockers in Executor Notes. Do not commit. Do not improvise fixes.

13. Commit in two logical commits, both via HEREDOC.

    **Commit 1** — content additions only. `git status` to confirm only the new files are staged so far:
    - `.claude/CONTEXT_CONSTITUTION.md`
    - `.claude/references/architecture.md`
    - `.claude/references/commands.md`
    - `.claude/references/configuration.md`
    - `.claude/references/testing.md`

    `git add .claude/CONTEXT_CONSTITUTION.md .claude/references/`

    Commit with this exact message:
    ```
    Add Context Constitution and externalised reference docs

    - Create .claude/CONTEXT_CONSTITUTION.md with rot framework + R1–R10
    - Extract Architecture, Commands, Configuration, Testing sections
      from CLAUDE.md into .claude/references/ (verbatim)

    Co-Authored-By: Claude <noreply@anthropic.com>
    ```

    **Commit 2** — CLAUDE.md slimming. `git status` to confirm only `CLAUDE.md` is now staged.

    `git add CLAUDE.md`

    Commit with this exact message:
    ```
    Slim CLAUDE.md: pointers + Trinity + caveats

    - Add Context Constitution pointer at top (IMPORTANT-prefixed)
    - Replace Architecture, Commands, Configuration, Testing sections with pointers
    - Delete Key Dependencies section (requirements.txt is authoritative)
    - Append Further Documentation list at bottom for instruction weighting
    - Final line count: <fill in actual count> (was 161)

    Co-Authored-By: Claude <noreply@anthropic.com>
    ```

    **Hold push** for Ken's confirmation per push policy.

14. Update Bus monthly LOG: mark this plan's row status as `done` after Phase E completes.

## Verification
- [ ] `Skill("maintain-claude-md")` audit ran successfully (Phase A) and findings logged
- [ ] `.claude/CONTEXT_CONSTITUTION.md` exists with R1–R10 plus rot framework + recovery protocol
- [ ] `.claude/references/architecture.md` exists with verbatim extracted content
- [ ] `.claude/references/commands.md` exists with verbatim extracted content
- [ ] `.claude/references/configuration.md` exists with verbatim extracted content
- [ ] `.claude/references/testing.md` exists with verbatim extracted content
- [ ] CLAUDE.md final line count ≤ 150 (target ≤ 120; counts between 121 and 150 pass with note per Step 10)
- [ ] CLAUDE.md no longer contains "Key Dependencies" section
- [ ] CLAUDE.md "Context Constitution" pointer is in top 10 lines with IMPORTANT prefix
- [ ] CLAUDE.md "Further Documentation" section lists all 4 references + constitution + AGENT_RULES + skills guide
- [ ] Every pointer in CLAUDE.md resolves to an existing file
- [ ] Re-audit (Phase E step 12) shows zero blockers
- [ ] Two commits; push held for Ken's confirmation
- [ ] LOG row for this plan marked `done`

## Executor Notes
*Populated after execution. Leave blank.*

**Executed:**
**Outcome:**
**What was done:**
**Files modified:**
