---
name: execute-plan
description: Plan execution skill. Runs a PLAN from Bus/ end-to-end — executes steps in order, populates Executor Notes, updates the monthly LOG Status Table, writes last_executor_outcome to PLAN frontmatter. Caller (plan-pipeline orchestrator or Human) commits and pushes. Trigger phrases: "execute this plan", "implement the plan", "run PLAN_x", "ok implement".
---

<essential_principles>
Execute the plan as written — do not redesign, re-scope, or improve mid-flight.
Execute steps in order. Verify each before moving on.
If a step is ambiguous, unsafe, or marked [Ken]: halt and flag. Do not improvise.
Always populate Executor Notes AND update the LOG before returning to the caller (caller commits + pushes per parent PLAN 202605011400 decisions 13, 22).
On closure (outcome=done), if frontmatter sets closes_thread / advances_thread / parent_plan_of_plans, apply the roadmap sync (workflows/execute-steps.md Step 4.5) before LOG update.
Git commit and push are the caller's responsibility (e.g. plan-pipeline orchestrator, or the Human during bootstrap) — execute-plan no longer commits or pushes.
Retirement of the PLAN file is the caller's responsibility — execute-plan no longer auto-retires.
On any halt (success or failure), write `last_executor_outcome` to PLAN frontmatter so callers can route deterministically (parent PLAN 202605011400 decision 24).
Wire format: end response with literal `<pipeline-result>` containing JSON code fence per parent decision 23. No XML payload, no HTML escaping.
**Executor uses filesystem tools, never Bash, for mechanical Steps (F1 Option C, PLAN 202605011900).** Create/copy/move/read/grep/edit operations go through Read, Write, Edit, Glob, Grep — not `mkdir`/`cp`/`mv`/`cat`/`grep`/`sed` shell calls. Reason: subagents inherit only a subset of the parent's permissions (Anthropic GH #37730 closed-not-planned), so Bash calls fail with permission denials at runtime. Filesystem tools work in every subagent context. The plan-executor agents enforce this structurally via `disallowedTools: [Bash, ...]`. Bash is NOT needed for self-verification of `verify:`/`acceptance:` items — the orchestrator's outcome-verifying phase (decision 25) re-runs those in parent context, where the allowlist works correctly. The executor self-ticks Verification boxes based on filesystem-tool observations only.
</essential_principles>

<preconditions>
Before starting, confirm:
- PLAN file exists in Bus/ with `status: ready`
- Monthly LOG exists and contains the PLAN in its Status Table
- PLAN is not in `status: blocked` and `blocked_by` is empty — blocks are cleared by `write-bus-input` when the requisite RESEARCH/ADVICE lands, or by Ken explicitly saying "proceed without" (Ken clears `blocked_by` manually in that case)
- Ken has authorised execution (trigger phrase in this skill's description)
</preconditions>

**Plan-safe definition:** See [../_shared/plan-safe.md](../_shared/plan-safe.md) — single source of truth shared with check-plan.

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
- Caller must commit and push after execute-plan returns; never use --no-verify, --force, --force-with-lease, or bypass signing
- If a PLAN step requires tools or permissions the executor lacks: halt, flag, escalate to Ken
- If outcome is not "done": still populate Executor Notes and update LOG; the caller decides commit cadence and content per parent PLAN 202605011400 decision 22.
- **Halt-on-failure protocol:** If any step fails or produces output that does not match the step's verification criteria, halt the PLAN immediately. Do not attempt subsequent steps. Set frontmatter `status: needs-revision` AND write `last_executor_outcome` with `outcome: exception` + diagnostics_summary. Populate Executor Notes with: which step failed, actual output, suspected cause. Update LOG Status Table row to `needs-revision` AND reorder entire Status Table per the sort rule in write-bus-plan/SKILL.md (non-terminal statuses first by filename descending, then terminal by filename descending). Return to caller — caller (orchestrator or Human in bootstrap) handles commit/push of partial work with `WIP:` prefix per decision 22. Report to Ken.
</constraints>

<success_criteria>
- PLAN's Executor Notes populated with execution details and today's date
- PLAN frontmatter `status` flipped to match Outcome (not left on `ready` or `in-progress`)
- Every State/ file modified has its frontmatter `last_updated` bumped to today
- Monthly LOG Status Table row updated with the final outcome (matches PLAN frontmatter `status`)
- last_executor_outcome frontmatter populated with outcome enum + diagnostics_summary.
- Ken has been given the final report: filename, outcome, LOG path
- Halt-on-failure protocol applied: any failed step results in PLAN status `needs-revision` AND `last_executor_outcome.outcome: exception`; caller is notified to handle commit/push and recovery
- Roadmap sync applied if applicable: thread Status flipped to `closed` and closure bullet appended in pillar (closes_thread), or progress bullet appended in pillar (advances_thread), or parent plan-of-plans updated (parent_plan_of_plans). ROADMAP.md frontmatter last_updated bumped to today.
</success_criteria>