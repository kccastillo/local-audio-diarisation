---
name: plan-pipeline
description: End-to-end planning orchestrator. Walks a PLAN through five phases — drafting (ideation) → drafted (audit loop) → checked → executing → outcome-verifying → complete (retire) — by dispatching pinned-model subagents in `.claude/agents/` and tracking state on disk via the PLAN's `pipeline_phase` frontmatter. The orchestrator is the brain; subagents do domain work. The orchestrator owns all git operations (subagents never commit). Activation triggers (parent-session only): "let's make a plan", "let's plan X", "I want to plan Y", "plan out Z", "run the pipeline on PLAN_x", "resume the pipeline". Disambiguation: "create plan file" routes to `write-bus-plan` (transcription primitive), NOT here — this skill orchestrates a full lifecycle, write-bus-plan only writes one file.
---

**Plan-safe definition:** See [../_shared/plan-safe.md](../_shared/plan-safe.md).

**Companion files:**
- [workflows/dispatch.md](workflows/dispatch.md) — the procedural walk-through for one orchestrator invocation.
- [references/phase-state-machine.md](references/phase-state-machine.md) — routing tables: `(phase, audit_state, outcome) → action`, agent dispatch table, commit-message templates, frontmatter mutation cheat sheet.

<essential_principles>
Orchestrator is the brain. Skills are organs. Subagents are dispatched workers. The orchestrator does no domain work itself — it routes data, applies HITL checkpoints, and owns git. (Parent PLAN 202605011400 decision 16.)
Pipeline state lives on disk in PLAN frontmatter (`pipeline_phase`, `audit_state`, `last_executor_outcome`, `verification_state`), not in conversation history. Every invocation reads disk first; transitions are durable. (Decision 1.)
Re-entry is idempotent. If the disk state shows the action for the current phase has already been taken and no new outcome is recorded, no-op and return. Do not double-dispatch. (Decision 18.)
Hybrid background mode. Only `plan-executor`/`plan-executor-sonnet`/`plan-executor-opus` dispatch with `run_in_background: true`. Every other subagent dispatch (`plan-writer`, `sufficiency-auditor`, `plan-safety-auditor`, `plan-retirer`) is foreground and synchronous. (Decision 18.)
Single conversational re-entry point. Background executor's completion message is the one cue requiring parent-Claude to re-invoke this skill. All other transitions happen synchronously inside one orchestrator invocation. (Decision 18.)
The `outcome` enum (`success | revision_needed | exception`) is the load-bearing contract for every subagent return. Route on it; never re-interpret prose. (Decisions 19, 20.)
Kanban full-stop on `outcome: exception` from any subagent OR from the orchestrator's own MAX_ITERATIONS check OR from a git failure. Halt the pipeline; commit + push the WIP state with diagnostics; do not advance. The Human decides recovery. (Decisions 19, 22.)
Audit loops are synchronous, durable, and bounded. Sufficiency runs first; plan-safety runs only after sufficiency returns success. Each iteration is announced to the Human (no silent re-runs). MAX_ITERATIONS=5 per stage. (Decision 21.)
Outcome-verification is a separate phase between executing and complete. Run all `verify:`/`acceptance:` shell commands; surface `verify: human` items via a structured prompt. Failures override the executor's self-reported success. (Decision 25.)
Decision-15 triage on every Human-facing surface. Classify items Already-locked / Mechanically-forced / Real-judgement-call before asking. Only Real-judgement-call items become questions. (Decision 15.)
Orchestrator commits + pushes at every milestone. Subagents never touch git. Fail-loud on git failure (kanban stop). Never use `--no-verify`/`--force`/`--force-with-lease`/bypass signing. (Decisions 13, 22.)
Ideation is interactive and runs in the parent session only — invoke `Skill("ideate")` directly, never via a subagent. The structural safeguard is the absence of an `ideate-runner` agent file. (Decision 10.)
Subagent-return wire format is a `<pipeline-result>` block containing JSON. Parse the LAST such block in the agent's return. If the block is absent/malformed → `outcome: exception` with diagnostics. (Decision 23.)
PLAN-driven executor tier selection. Read the target PLAN's `assigned_to:` frontmatter; route to `plan-executor` (haiku, default), `plan-executor-sonnet`, or `plan-executor-opus`. `assigned_to` is a guideline (decision 8) — Human can override per PLAN.
Subagents do not inherit the parent's skill registry. Each agent file declares `skills: [...]` to preload. Honour that — do not assume an agent has access to the parent's wider registry. (Decision 17.)
Bootstrap exception. During the bootstrap of parent PLAN 202605011400, the Human commits manually after each `execute-plan`. The orchestrator owns git only for *future* PLANs.
"Human", not "Ken", in any new prose surfaced to the operator. (Decision 9.)
</essential_principles>

<preconditions>
- Running in the parent session (interactive). The orchestrator surfaces Human-facing prompts and dispatches subagents; both require parent-session context.
- The repository is a git working tree (orchestrator commits at every milestone). If git is unavailable, the orchestrator emits `exception` immediately.
- For invocations on an existing PLAN: the PLAN file exists at the supplied path under `Bus/`.
- Required subagent files exist in `.claude/agents/`: `plan-writer`, `sufficiency-auditor`, `plan-safety-auditor`, `plan-executor` (and optionally `-sonnet`, `-opus` variants), `plan-retirer`. Required leaf skills exist: `ideate`, `audit-sufficiency`, `audit-haiku-safe`, `write-bus-plan`, `execute-plan`, `retire`, `write-bus-input`.
- Shared `.claude/skills/_shared/plan-safe.md` exists.
</preconditions>

<inputs>
- `request: string (optional)` — Human's natural-language request for a fresh ideation entry (e.g. "let's plan the speaker separation rewrite"). Used when no PLAN exists yet.
- `plan_path: string (optional)` — repo-relative path to an existing PLAN file (e.g. `Bus/202605020900_PLAN_<slug>.md`). Used when resuming or operating on a known PLAN.
- `human_reply: string (optional)` — the Human's most recent message text, supplied by parent-Claude when re-entering the skill in response to a surfaced prompt (audit revision-needed reply, outcome-verification human-pending reply, or a kanban-halt recovery instruction). The orchestrator uses this only to interpret a previously-surfaced prompt; it never silently mines the conversation otherwise.

Exactly one of `request` or `plan_path` is required at first entry. On re-entry triggered by a background executor's completion, parent-Claude supplies `plan_path` (the executor's return references it).
</inputs>

<output_schema>
**This skill does not return a `<pipeline-result>` block** — it is the top-level orchestrator and is not itself invoked by another orchestrator. Its "output" is:

1. **Filesystem changes** — PLAN frontmatter mutations (`pipeline_phase`, `audit_state`, `last_executor_outcome`, `verification_state`), monthly LOG row updates, and any subagent-produced files.
2. **Conversational surfacing** — phase-transition narration, audit findings, outcome-verification prompts, kanban-halt diagnostics. Always with decision-15 triage applied.
3. **Git milestones** — one commit + push per milestone (see commit-message templates in references/phase-state-machine.md).
4. **Control hand-off**:
   - **Synchronous return to parent** when the orchestrator has no further work for this invocation (audit `revision_needed` awaiting Human revision; outcome-verifying with human_pending; kanban halt; `complete` after retire).
   - **Background dispatch + return** when entering `executing` — parent-Claude resumes work; the executor's completion message is the cue to re-invoke this skill.
</output_schema>

<exception_conditions>
The orchestrator emits its own `outcome: exception` (kanban full-stop, commit + push WIP) when:
- Required agent or skill file is missing on disk (preconditions violated).
- PLAN file at `plan_path` is unreadable or has malformed frontmatter.
- A subagent returns no `<pipeline-result>` block, or the block's JSON does not parse, or `outcome` is not in the enum.
- An audit stage's iteration counter exceeds MAX_ITERATIONS=5 ("audit loop did not converge").
- A git operation (add, commit, push) fails. Never bypass with `--no-verify`/`--force`.
- The Human's reply to a surfaced prompt is ambiguous after one re-prompt (orchestrator does not infer beyond a single clarifying question).

Subagent-emitted `outcome: exception` propagates as kanban halt — same handling as orchestrator-emitted.
</exception_conditions>

**Dispatch procedure:** See [workflows/dispatch.md](workflows/dispatch.md) for the per-invocation walk-through. See [references/phase-state-machine.md](references/phase-state-machine.md) for the routing tables and commit templates.

<constraints>
- Never write *new* PLAN body content to disk yourself. Dispatch `plan-writer` (foreground subagent) for every create-or-substantive-update of a PLAN file. The orchestrator may write PLAN *frontmatter* fields (`pipeline_phase`, `audit_state`, `last_executor_outcome`, `verification_state`) directly because those are orchestration-owned. Body content (Objective, Context, Steps, Verification, Executor Notes) authoring or re-authoring flows through `plan-writer` or `plan-executor`.
- **Surgical-edit carve-out (F4 from PLAN 202605011900, 2026-05-01):** the orchestrator MAY use the Edit tool directly for one-line surgical body-content revisions — commit-hash citations, audit-resolution notes, and similar single-line adjustments to existing body text. Full body re-authoring (Objective/Steps/Verification rewrite) MUST still route through plan-writer. Boundary test: if the change is ≤1 line and does not introduce new structural sections, Edit is permitted; otherwise, dispatch plan-writer.
- Never run `ideate` as a subagent. Invoke `Skill("ideate", ...)` in the parent session.
- **Phase-boundary chaining (F9 from PLAN 202605011900, 2026-05-01):** the orchestrator MAY chain phase transitions within a single invocation (e.g. drafted → checked → executing) when running in a continuous parent session, provided each transition produces a milestone commit + push. Re-entry idempotency is preserved by reading disk on every invocation. The earlier "one phase per invocation" rule is relaxed: chaining is permitted, but the per-milestone commit + push contract is non-negotiable. Audit-loop iterations within the same phase (`drafted`'s sufficiency → plan-safety chain) are unchanged — they have always been allowed within one invocation.
- Never silently skip a milestone commit. If `git push` fails, halt with `exception` — never retry with destructive flags.
- Never poll a background executor. The single re-entry trigger is parent-Claude observing the executor's completion message and re-invoking this skill.
- Never re-interpret a `<pipeline-result>` outcome based on prose. The enum value is canonical.
- Never alter a PLAN's Objective / Context / Steps / Verification — only frontmatter and (via `plan-executor`) Executor Notes.
- Never advance a parent PLAN's phase while its `triggers_plans:` children are unfinished. Detect via children's `status:` (terminal = `done`/`partially-complete`/`cancelled`/`closed`); non-terminal children block parent advancement. Surface "paused for children" to the Human and return.
- Never write "Ken" into newly-authored content — use "Human". (Decision 9. Pre-existing references in unrelated files are out of scope.)
- Never bypass decision-15 triage when surfacing to the Human. Even single-item surfaces apply the classification.
</constraints>

<success_criteria>
- Every invocation reads PLAN frontmatter from disk before deciding any action.
- Every phase transition that produces filesystem changes is followed by a commit + push with a descriptive message per the templates in references/phase-state-machine.md.
- Audit loop tracked via durable `audit_state` frontmatter; iterations bounded at 5 per stage; each iteration narrated to the Human.
- Executor dispatched with `run_in_background: true`; all other subagent dispatches foreground.
- Outcome-verification phase always runs between executing and complete; never skipped on `outcome: success`.
- `verify: human` items surfaced via the structured prompt described in references/phase-state-machine.md; Human reply interpreted with at most one clarifying re-prompt before falling to `exception`.
- Decision-15 triage applied to every Human-facing prompt — only Real-judgement-call items become questions.
- Re-entry idempotency: a duplicate invocation on the same disk state with no new outcome recorded is a no-op (returns "already at <phase>; nothing to do").
- Kanban full-stop on any `exception`: commit + push WIP with diagnostics; never advance.
- `<pipeline-result>` JSON parsed via "last block, JSON within fence" — fails loud on malformed.
</success_criteria>
