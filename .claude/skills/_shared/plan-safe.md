# Plan-safe definition

A plan is "plan-safe" (executable mechanically, without design work) when every step is:
- **Concrete:** specific file paths, exact command syntax, no "likely" or "probably"
- **Unambiguous:** no judgment calls; the executor runs the steps, not redesigns them
- **Atomic:** one step at a time; clear success/failure condition
- **Safe:** no destructive operations without explicit Human approval; no bypasses (--no-verify, --force)
- **Testable:** verification criteria are independent and checkable

Example of plan-safe: "Read SKILL.md from `.claude/skills/atomise/SKILL.md`. Verify frontmatter `name: atomise`. Extract `<process>` block (lines 43–68) to `workflows/atomise-steps.md`. Commit with message 'chore: trim atomise SKILL.md'"

Example of NOT plan-safe: "Audit SKILL.md and extract any residual content. Use your judgment." (ambiguous, requires interpretation)

**When authoring plans:** refer to [.claude/skills/execute-plan/workflows/execute-steps.md](../execute-plan/workflows/execute-steps.md) for the execution protocol and [.claude/skills/execute-plan/references/log-rules.md](../execute-plan/references/log-rules.md) for the LOG contract, and ensure every step passes this definition.

**Mechanical Steps must be filesystem-tool-shaped (F1 Option C, PLAN 202605011900).** Express mechanical Steps as Read/Write/Edit/Glob/Grep operations, not shell-shaped (`mkdir`, `cp`, `echo > file`, `cat file`, `sed -i`, `grep ... file`). The plan-executor agents disallow Bash structurally, so shell-shaped Steps will fail at runtime. Shell commands belong in the Verification section as `verify:`/`acceptance:` items — those run in the orchestrator's outcome-verifying phase (parent context, allowlist works correctly).

## Verification format requirement (per PLAN 202605011400 decision 25)

Every PLAN's `## Verification` section item must be shell-runnable, with one of the following annotations on the line directly below the prose checkbox:

- `verify: <shell command>` — state assertion (file exists, grep matches, command exit code). Exit 0 = pass.
- `acceptance: <shell command>` — spec-level behavioural check that exercises the deliverable. Exit 0 = pass. **Every PLAN must include at least one `acceptance:` item.**
- `verify: human` — genuinely subjective item; surfaced for Human eyeball but does not auto-fail.

The orchestrator runs all `verify:` and `acceptance:` commands as a separate outcome-verification phase (after plan-executor returns success, before advancing to complete). Failures override the executor's self-reported success.

### skills-as-deliverable carve-out (per PLAN 202605011900 F3)

PLANs whose deliverable is a Claude skill (or other artefact not invokable from a shell) MAY satisfy the at-least-one-`acceptance:` requirement via an artefact-property check (frontmatter parses, workflow content present, gate-clauses present) PLUS a `verify: human` for the actual invocation behaviour. The skills-as-deliverable carve-out keeps the mechanical gate active without forcing a fictional shell-acceptance command. Claude skills are invoked by Claude, not by a shell; a true behavioural acceptance command cannot exist for them. The artefact-property `acceptance:` keeps the mechanical gate active; the `verify: human` keeps invocation fidelity in the loop.
