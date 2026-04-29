---
name: execute-plan
description: Plan execution skill. Runs a PLAN from Bus/ end-to-end — executes steps in order, populates Executor Notes, updates the monthly LOG Status Table, commits, and pushes to origin. Trigger phrases: "execute this plan", "implement the plan", "run PLAN_x", "ok implement".
---

<essential_principles>
Execute the plan as written — do not redesign, re-scope, or improve mid-flight.
Execute steps in order. Verify each before moving on.
If a step is ambiguous, unsafe, or marked [Ken]: halt and flag. Do not improvise.
Always populate Executor Notes AND update the LOG before git commit.
Git commit + push is the final step — never skip; never use --no-verify or --force.
On closure (outcome=done), if frontmatter sets closes_thread / advances_thread / parent_plan_of_plans, apply the roadmap sync (workflows/execute-steps.md Step 4.5) before LOG update.
</essential_principles>

<preconditions>
Before starting, confirm:
- PLAN file exists in Bus/ with `status: ready`
- Monthly LOG exists and contains the PLAN in its Status Table
- PLAN is not in `status: blocked` and `blocked_by` is empty — blocks are cleared by `write-bus-input` when the requisite RESEARCH/ADVICE lands, or by Ken explicitly saying "proceed without" (Ken clears `blocked_by` manually in that case)
- Ken has authorised execution (trigger phrase in this skill's description)
</preconditions>

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

<constraints>
- Never modify the PLAN's Objective, Context, Steps, or Verification sections — only Executor Notes and the frontmatter `status` field
- PLAN frontmatter `status` must always match the Executor Notes Outcome and the LOG Status column — never leave them in disagreement
- Never tick a Verification checkbox without confirming the condition
- Never skip the LOG update — Ken needs monthly visibility
- Never skip git commit + push — the source of truth must be in origin
- Never use `--no-verify`, `--force`, `--force-with-lease`, or bypass signing
- If a PLAN step requires tools or permissions the executor lacks: halt, flag, escalate to Ken
- If outcome is not "done": still commit and push, but clearly mark blockers in the commit body
- **Halt-on-failure protocol:** If any step fails or produces output that does not match the step's verification criteria, halt the PLAN immediately. Do not attempt subsequent steps. Set frontmatter `status: needs-revision`. Populate Executor Notes with: which step failed, actual output, suspected cause. Update LOG Status Table row to `needs-revision` AND reorder entire Status Table per the sort rule in write-bus-plan/SKILL.md (non-terminal statuses first by filename descending, then terminal by filename descending). Commit + push partial work with message beginning `WIP:` and clearly notes the halt. Report to Ken.
</constraints>

<success_criteria>
- PLAN's Executor Notes populated with execution details and today's date
- PLAN frontmatter `status` flipped to match Outcome (not left on `ready` or `in-progress`)
- Every State/ file modified has its frontmatter `last_updated` bumped to today
- Monthly LOG Status Table row updated with the final outcome (matches PLAN frontmatter `status`)
- Git commit created with subject ≤72 chars + bulleted body + Co-Authored-By trailer
- Commit pushed to origin on the current branch
- Ken has been given the final report: filename, outcome, LOG path, commit hash
- Halt-on-failure protocol applied: any failed step results in PLAN status `needs-revision` and a `WIP:` commit; Ken is notified before any further steps are attempted
- Roadmap sync applied if applicable: thread Status flipped to `closed` and closure bullet appended in pillar (closes_thread), or progress bullet appended in pillar (advances_thread), or parent plan-of-plans updated (parent_plan_of_plans). ROADMAP.md frontmatter last_updated bumped to today.
</success_criteria>