---
title: "Build plan-pipeline orchestrator and supporting skill ecosystem"
type: bus-plan
status: partially-complete
assigned_to: sonnet
priority: high
created: 2026-05-01
created_by: opus
created_month: 202605
log_month: 202605
due: ""
repeatable: false
repeat_cadence: ""
linked_decisions: []
rollover_count: 0
triggers_plans:
  - 202605011410_PLAN_create-ideate-skill.md
  - 202605011420_PLAN_create-audit-haiku-safe-skill.md
  - 202605011425_PLAN_create-audit-sufficiency-skill.md
  - 202605011430_PLAN_create-plan-pipeline-skill.md
  - 202605011440_PLAN_dogfood-plan-pipeline.md
linked_inputs:
  - 202605011500_ADVICE_sufficiency-audit-exemplar.md
closes_thread: ""
advances_thread: ""
parent_plan_of_plans: ""
pipeline_phase: checked
blocked_by: ""
---

## Objective
Build an end-to-end orchestrator skill (`plan-pipeline`) that walks the planning workflow through five phases — ideation, incremental plan write, sufficiency check, execution, retirement — invoking the right skill at each phase boundary. Ideation runs in the parent session where the Human is interacting (the only interactive phase); the non-interactive phases dispatch to **pinned-model subagents** in `.claude/agents/` via the Agent tool (Sonnet-pinned for plan writing and review, Haiku-pinned for mechanical execution and retirement), so tier-mixing happens without asking the Human to switch parent-session models mid-flow. Create the missing leaf skills (`ideate`, `check-plan`), extract a shared plan-safe definition so `check-plan` and `execute-plan` enforce the same contract, mutate existing skills where the pipeline requires it, and dogfood the result. The pipeline activates on natural-language triggers like "let's make a plan", "let's plan X", "I want to plan Y".

## Context

**Why now.** The five-phase planning workflow (ideation → write-bus-plan → check → execute → retire) currently has gaps: no `ideate` skill, no `check-plan` skill, no orchestrator. Without an orchestrator, the model switch between Opus (ideation) and Haiku/Sonnet (execution) is ad-hoc, and Opus context compression can elide ideation work before it's captured to disk. Without `check-plan`, plans reach `execute-plan` without a sufficiency or plan-safety pass — relying on `execute-plan`'s halt-on-failure as the only safety net.

**Key design decisions** (locked during ideation, this conversation):

1. **Pipeline state lives on disk, not in conversation.** A new orthogonal `pipeline_phase` frontmatter field on PLAN files (`drafting | drafted | checked | executing | complete`) tracks orchestration state independent of `status`. This survives context resets, parent-session interruptions, and background-subagent runs — any subsequent invocation of `plan-pipeline` reads the field and resumes from the right phase.
2. **Single source of truth for plan-safety.** The `<plan_safe_definition>` block currently lives in `execute-plan/SKILL.md` only. Extract to `.claude/skills/_shared/plan-safe.md`; both `check-plan` and `execute-plan` reference it. Eliminates drift.
3. **Decouple auto-retire from `execute-plan`.** Recent commit `a19c326` added auto-retire to `execute-plan`. The orchestrator owns lifecycle; `execute-plan` becomes a clean primitive. Final phase of `plan-pipeline` invokes `Skill("retire", ...)` explicitly.
4. **Loosen `write-bus-plan` for incremental writes.** Current description says "use when a plan has been finalised" — incompatible with checkpoint-during-ideation. Loosen description and `<essential_principles>` to accept in-flight provisional writes.
5. **Sufficiency audit and Haiku-safety audit are separate skills run by separate model tiers.** ⚠️ *Revised from earlier "one skill, two sections" framing after Human-led Opus pass surfaced architectural issues that mechanical review had missed (see ADVICE 202605011500_ADVICE_sufficiency-audit-exemplar.md).*
    - **`audit-sufficiency`** (Opus-pinned subagent `sufficiency-auditor`) — interrogates assumptions, surfaces missing edge cases, audits the validation path, asks meta-questions ("is this over-engineered? does the dogfood actually test real friction?"). The kind of pass an experienced reviewer does. Operates with high reasoning bandwidth.
    - **`audit-haiku-safe`** (Sonnet-pinned subagent `plan-safety-auditor`) — mechanical, contract-based: each step concrete, atomic, line-numbers accurate, exact-text complete, no judgement calls. References the shared `_shared/plan-safe.md` definition.
    - **Sequence:** sufficiency runs first; only if it returns `Blockers: 0` does `audit-haiku-safe` run. No point checking Haiku-safety on an insufficient plan.
    - Both must return `Blockers: 0` before `pipeline_phase: drafted → checked` advances (per decision 14's gate semantics, applied twice in sequence).
    - Output format for both: CLAUDE.md "Reviews" rule + machine-readable `Blockers: N` summary.
    - The exemplar ADVICE file (202605011500) shows the *kind* of output `audit-sufficiency` produces — qualitatively different reasoning from a plan-safety pass.
6. **`ideate` is a conversational arc with three explicit phases** (Clarify → Survey → Converge). Imposes the discipline already named in CLAUDE.md "Working style" (requirement before solution; options-with-recommendation). Does not write to disk itself; orchestrator handles checkpoint writes.
7. **Behavioural regression accepted:** removing auto-retire from `execute-plan` means one-shot uses lose that convenience. Retirement is now an explicit step. Clean contracts beat compatibility shims.
8. **Agent assignments are guidelines, not requirements.** `assigned_to` (e.g. `sonnet`, `haiku`) and any "typically Opus" / "typically Haiku" annotations in skill docs are advisory — they describe the tier the work is *expected* to suit, not a hard binding. Any sufficiently capable agent may execute a given step.
9. **Manual-intervention markers use "Human", not "Ken".** Skill docs, PLAN markers (e.g. `[Human]`), and orchestrator prompts refer to the generic operator role rather than a specific person, to keep the skill ecosystem portable.
10. **Model-tier mixing via pinned-model subagents.** The orchestrator dispatches each non-interactive phase to a model-pinned agent defined in `.claude/agents/` (frontmatter `model: opus | sonnet | haiku`) via the Agent tool, rather than asking the Human to switch parent-session models. Suggested agents: `plan-writer` (sonnet), `plan-checker` (sonnet), `plan-executor` (haiku), `plan-retirer` (haiku — or inline the retire call). The pinned model on each agent is a guideline per decision 8: any sufficiently capable model may be substituted by editing the agent frontmatter.

   **Hard rule — interactive phases stay in the parent session.** Subagents are one-shot and have no channel back to the Human mid-run. Therefore: `ideate` (the only interactive phase) always runs in the parent session where the Human is interacting; it is *never* dispatched to a subagent. Skills invoked from subagents must themselves be non-interactive and one-shot. If a phase needs Human clarification, it halts and returns to the parent, which surfaces the question — it does not try to ask from inside a subagent.
11. **Activation triggers.** `plan-pipeline`'s skill description includes natural-language triggers: "let's make a plan", "let's plan X", "I want to plan Y", "plan out Z", and similar phrasings. The Skill matcher routes these to `plan-pipeline` automatically.
12. **Ideation may produce RESEARCH/ADVICE inputs, not just PLAN content.** During the Clarify or Survey phases of `ideate`, a question may surface that requires external data lookup (→ RESEARCH file) or a strategic decision the Human wants persisted as a reference (→ ADVICE file). The `ideate` skill should recognise these moments and invoke `write-bus-input` (in the parent session, since it's interactive) to drop the appropriate Bus file. The resulting filename then lands in the PLAN's `linked_inputs: []` frontmatter array. If RESEARCH is needed but not yet available, the PLAN may be written with `status: blocked` and `blocked_by` referencing the pending input — `write-bus-input` clears the block automatically when the resolving file lands (per existing convention in `bus-conventions.md`).
13. **Subagents do not perform git operations.** Git commit and push are the orchestrator's responsibility (or the Human's during bootstrap), never a subagent's. Subagents do filesystem work — write files, move files, populate Notes, update LOG — and return. The orchestrator collects the work and commits/pushes once per phase (or once at pipeline completion, depending on dispatch design). Affects two existing skills: `execute-plan` (currently commits + pushes at workflow Step 7) and `retire` (currently commits + pushes after moving). Both must be decoupled. The previous memory note `feedback_retire_push.md` (which reminded Claude to commit+push after retire) is superseded by codified orchestrator behaviour and is deleted in step 3c — the orchestrator's stepwise milestone-commit logic (decision 22) is the new source of truth.
14. **`drafted → checked` is a hard gate, not advisory.** The transition happens only when `check-plan` returns **zero blockers** (in either the Sufficiency or Plan-safety section). Nits are not blockers and do not gate the transition. To enable mechanical enforcement, `check-plan`'s output must structurally separate blockers from nits (per the existing CLAUDE.md "Reviews" rule format — explicit "Blockers" and "Not blockers" subgroups in each section) AND state a machine-readable summary line at the end such as `Blockers: 0` or `Blockers: 3 (sufficiency: 1, plan-safety: 2)`. The orchestrator reads that summary line, applies the gate, and either flips `pipeline_phase: drafted → checked` (zero) or stays at `drafted` and surfaces the punch list (>0). The same gate semantics apply at the bootstrap level: the Human reading a manual `check-plan` output should not advance until the Blockers count is 0.

   **Definition of "blocker":** any finding that would cause the executor (Haiku, Sonnet, or Human in bootstrap) to halt, error, or be forced into a judgement call mid-execution. Anything that can be safely deferred or that is operator-procedural (e.g. "Human commit cadence during bootstrap") is a nit, not a blocker.

15. **Triage decisions before requesting Human input.** Whenever a skill or the orchestrator surfaces design decisions to the Human for review (at a `[Human]` checkpoint, at the convergence of ideation, when escalating a check-plan blocker, or anywhere else Human discretion is sought), classify each decision into one of three categories before asking:
    - **Already locked** — Human proposed the decision directly, or explicitly affirmed it earlier in this conversation / PLAN history.
    - **Mechanically forced** — no meaningful alternative exists; the decision is a necessary downstream consequence of an earlier locked decision, an infrastructure constraint, or an obvious-best per established working style.
    - **Real judgement call** — genuinely has alternatives the Human might prefer.

    **Only the "real judgement call" category is presented to the Human as a question.** The other two categories are listed once, briefly, for transparency (so the Human can spot-check the classification) but require no answer. This protects the Human's attention budget and avoids forcing re-confirmation of decisions the Human already made.

    Applies to: parent PLAN's `[Human — design-review checkpoint]` step, `ideate` skill's converge phase output, `plan-pipeline` orchestrator's Human-facing surfacing at any phase, and `check-plan`'s output when it identifies items needing Human input. Recursive: future iterations of the pipeline use the same triage.

16. **Skill System architecture: Sequential Workflow Orchestration framework.** The orchestrator (`plan-pipeline`) and the skills it composes must implement the five fundamental layers of a Skill System:
    1. **Orchestrator (Brain)** — manages sequence, routes data, applies HITL checkpoints; does not perform domain work itself. → `plan-pipeline`.
    2. **Modular Building Blocks (Skill Library)** — each skill is single-purpose, narrow-context, standalone-capable. The "10–30 high-quality skills" rule applies: prefer reuse over re-authoring.
    3. **Clean Handoffs** — each skill declares an explicit input schema (preconditions: what it needs to start) and output schema (what it returns). Inter-skill handoffs are routed by the orchestrator per these schemas, not by ad-hoc prose interpretation.
    4. **Human-in-the-Loop (HITL)** — strategic checkpoints at phase boundaries and on blockers, with decision-triage (decision 15) applied. Already covered by decisions 14 and 15.
    5. **Visual Feedback** — the system must render its internal state. Already partially covered by `pipeline_phase` frontmatter and the monthly LOG Status Table; orchestrator must echo phase transitions to the Human conversationally so progress is visible mid-flow.

    **Anti-patterns explicitly avoided:**
    - **Isolationism** — orchestrator removes the Human as manual connector between skills.
    - **Monolithic Design** — when a skill exceeds reasonable scope, decompose into children (parent → child pattern already established by this PLAN itself).
    - **Context Bloat** — pinned-model subagents (decision 10) keep each agent's context narrow.
    - **Brittle Logic** — each skill is invocable standalone, not only from within `plan-pipeline`.

    **Concrete requirements for each new skill** (children 1, 2, 3 absorb):
    - SKILL.md must contain a `<preconditions>` block (input schema — what the skill needs to start) and an `<output_schema>` block (what it returns to the caller).
    - Handoff schemas between skills must be documented in `plan-pipeline`'s SKILL.md so routing is explicit.

    **Existing skills (write-bus-plan, execute-plan, retire) are partially compliant** — `execute-plan` already has `<preconditions>`. Retrofitting the others to add explicit preconditions/output_schema is **out of scope** for this PLAN and is captured as a follow-up sweep alongside the Ken→Human conversion.

**Human confirmations recorded at design checkpoint (2026-05-01):** Decision 5: ⚠️ **superseded** by Human-led Opus pass — split into two skills (audit-sufficiency / audit-haiku-safe) with different model tiers. Decision 6 (ideate as three-phase Clarify → Survey → Converge): confirmed. Decision 11 (current trigger phrase set for plan-pipeline): confirmed. Decisions 6 and 11 are Already-locked for future triage; decision 5's revised form is itself Human-proposed and locked.

17. **Subagents preload their required skill(s) via frontmatter.** Probe finding: subagents do NOT inherit the parent's skill registry at runtime. Workaround: each agent file declares `skills: [<skill-name>]` in its frontmatter, which preloads the skill content into the agent's context at startup. Each agent's `skills:` list includes its primary skill plus any skills that primary skill itself dispatches to mid-flight (transitive closure). Mechanical correction; no real choice.

18. **Background mode used only for `plan-executor` (Human choice — Option δ hybrid).** Probe-2 (2026-05-01) found that mid-execution streaming, cancellation, and re-entry hooks are not documented as supported. Rather than load-bearing on undocumented behaviour everywhere, we limit `background: true` in agent frontmatter to **only `plan-executor.md`** — the one phase where parent responsiveness during a long run actually matters. All other subagents (`plan-writer`, `sufficiency-auditor`, `plan-safety-auditor`, `plan-retirer`) run **foreground** (synchronous): orchestrator dispatches, parent blocks, agent returns, orchestrator continues — deterministic and platform-supported. The single async transition is `executing → complete`, where the parent Claude re-invokes the orchestrator on the executor's completion message; idempotent re-entry via on-disk `pipeline_phase` makes a missed cue recoverable.

   **Amendment 2026-05-01 (surfaced during dogfood iteration 2 of child 1440):** the Agent tool documents `run_in_background: true` as a tool-call parameter (originally noted as undocumented under probe-2; now documented). Orchestrator passes `run_in_background: true` explicitly on the Agent invocation when dispatching the executor; the agent's frontmatter `background: true` is preserved as authoring documentation but is not the load-bearing mechanism. The architectural sense of decision 18 (the executor is the *only* background dispatch; everything else is foreground) is preserved — only the mechanism is updated. All other clauses of decision 18 (no streaming, no cancellation, single conversational re-entry, idempotent re-entry from disk) stand unchanged.

19. **Subagent-driven exception handling — "let it crash, supervisor decides".** ⚠️ *Replaces earlier "chatty subagents with milestones + anomaly halt during execution" framing after the platform was found not to support mid-execution coordination.*
    - Every subagent skill declares an `<exception_conditions>` block in SKILL.md listing the conditions under which the skill self-terminates early (e.g. PLAN file unreadable, upstream output unparseable, scope-creep detected, destructive operation lacks Human approval).
    - On detecting any exception condition during its run, the subagent stops doing work, populates its result with `outcome: exception` + diagnostics, and returns cleanly. **The subagent has agency over its own termination — the orchestrator does not need to cancel it externally.**
    - The orchestrator receives one transactional result per dispatch (the agent's completion message). It reads the result's `outcome` field and routes:
      - `success` → advance phase (or proceed to next stage in a loop).
      - `revision_needed` → audit found blockers; surface findings, await revision, re-dispatch (loop).
      - `exception` → **kanban full stop**: halt the entire pipeline, surface diagnostics, do not advance. Human decides recovery path.
    - No mid-execution milestones (not platform-supported). No external cancellation (not platform-supported). The transactional boundary at completion is the protocol.

20. **The `outcome` enum is the load-bearing contract.** ⚠️ *Sharpened from earlier framing.* Every subagent declares `<inputs>` and `<outputs>` blocks in SKILL.md body (XML-tagged, matching existing skill convention). The output schema **must** include:
    ```
    outcome: enum [success, revision_needed, exception]
    payload: { skill-specific data — varies per skill }
    diagnostics: { populated when outcome != success — what happened, what state was left, what's recoverable }
    ```
    Skill-specific output fields go into `payload`. Examples:
    - `audit-sufficiency` payload: `{blockers_count: int, review_text: string, triaged_human_items: list}`.
    - `plan-writer` payload: `{filename_written: string, action: enum[created, updated], log_updated: bool}`.
    - `plan-executor` payload: `{outcome_subtype: enum[done, partially-complete, blocked, needs-revision], executor_notes: string, files_modified: list}`.
    The orchestrator's SKILL.md contains a routing table: `outcome × phase → next action`. Inter-skill handoffs are validated against schemas (orchestrator asserts upstream `payload` matches downstream `inputs` before dispatching).

21. **Audit loop conditions, explicit.** The orchestrator drives the audit loop synchronously, with state durably stored on disk so the loop is recoverable across orchestrator re-invocations (which happen between iterations when the Human revises in response to `revision_needed`).

    **Durable loop state in PLAN frontmatter:**
    ```
    audit_state:
      sufficiency_iterations: int   # count of sufficiency-auditor dispatches so far; default 0
      plan_safety_iterations: int   # count of plan-safety-auditor dispatches so far; default 0
      last_stage: enum [none, sufficiency, plan_safety]
      last_outcome: enum [none, success, revision_needed, exception]
    ```
    `MAX_ITERATIONS` per stage = 5 (configurable). Exceeding triggers `outcome: exception` from the orchestrator itself ("audit loop did not converge — Human review required").

    **Orchestrator behaviour on each invocation while `pipeline_phase == drafted`:**
    1. Read `audit_state` from PLAN frontmatter.
    2. **Determine which audit to dispatch:**
       - If `last_outcome == none` (fresh entry) OR `last_outcome == revision_needed AND last_stage == sufficiency`: dispatch `sufficiency-auditor`.
       - If `last_outcome == success AND last_stage == sufficiency`: dispatch `plan-safety-auditor`.
       - If `last_outcome == revision_needed AND last_stage == plan_safety`: dispatch `plan-safety-auditor` (re-run the failing one only — sufficiency already passed and didn't need to re-run).
       - If `last_outcome == success AND last_stage == plan_safety`: both audits passed → flip `pipeline_phase: checked`, exit loop.
    3. **Increment the appropriate iteration counter** before dispatch.
    4. **If counter > MAX_ITERATIONS:** do NOT dispatch; orchestrator emits `outcome: exception` itself; halt with "audit loop did not converge".
    5. **Dispatch foreground; receive `outcome`:**
       - `success` → update `audit_state` (`last_stage` = the stage just run, `last_outcome: success`); commit+push (decision 22); re-enter from step 1 to determine next stage.
       - `revision_needed` → update `audit_state` (`last_stage` = the failed stage, `last_outcome: revision_needed`); commit+push the audit_state update; surface `payload.review_text` + iteration counts to the Human with the message "Revise the PLAN and re-invoke me to re-audit"; orchestrator returns control. The next orchestrator invocation (after Human revision) re-enters at step 1.
       - `exception` → kanban full stop: surface diagnostics, do not advance, do not commit (PLAN unchanged). Human decides recovery.

    **The loop's "rerun-or-continue" signal:**
    - **Continue signal** = `outcome: success` from the current stage → orchestrator advances to next stage (or exits loop on `last_stage == plan_safety AND outcome == success`).
    - **Rerun signal** = `outcome: revision_needed` → orchestrator pauses for revision, re-dispatches the same stage on next invocation (the audit that flagged is the one re-run; the audit that already passed is not re-run unnecessarily).
    - **Stop signal** = `outcome: exception` (from subagent OR from orchestrator's own MAX_ITERATIONS check) → kanban full stop.

    Each iteration is announced to the Human in the parent conversation (no silent re-runs). The same `outcome`-routed pattern applies to plan-executor's transition: `success` → advance to complete; `revision_needed` → revert `pipeline_phase: drafted` and surface findings; `exception` → kanban stop.

22. **Orchestrator commits + pushes at every milestone — subagents never touch git.** Reaffirms decision 13 with explicit checkpoint discipline: any state transition that produces filesystem changes (or advances `pipeline_phase`/`audit_state` durable state) must be followed by an orchestrator-driven `git add -A && git commit -m "<descriptive message>" && git push`. Specifically:
    - After plan-writer success → commit + push (PLAN written/updated).
    - After Human revision input that modifies the PLAN (detected on next orchestrator invocation by file mtime or explicit Human signal) → commit + push *before* re-dispatching the audit.
    - After every audit `outcome` write to `audit_state` frontmatter → commit + push (so loop state itself is durable in origin, not just locally).
    - After plan-executor completes → commit + push the execution work.
    - After plan-retirer completes → commit + push the retirement.
    - On `outcome: exception` halt → commit + push current state with message `WIP: pipeline halted at <phase> for <plan> — see diagnostics`.

    Rationale: work is never lost. Even mid-loop revisions are durable in origin. If the parent session crashes or the conversation is abandoned mid-pipeline, the state is recoverable from the remote. Failure handling: if commit/push fails, orchestrator emits `outcome: exception` itself with diagnostics; never use `--no-verify`/`--force`/`--force-with-lease`/bypass signing.

23. **Subagent return serialisation — `<pipeline-result>` tag with JSON code fence.** Every subagent's response (the message returned to the orchestrator via the Agent tool) ends with a structured block in this exact format:
    ```
    <pipeline-result>
    ```json
    {
      "outcome": "success" | "revision_needed" | "exception",
      "payload": { /* skill-specific fields per the SKILL.md output_schema */ },
      "diagnostics": { /* populated when outcome != success */ }
    }
    ```
    </pipeline-result>
    ```
    The agent body may include prose / explanation / progress narrative *before* the `<pipeline-result>` block — that's for Human reading. The orchestrator finds and parses the **last** `<pipeline-result>` block in the agent's return (text-scan: locate the last opening tag, extract content up to the matching closing tag, parse the JSON code fence within). Fails loud — if no closing tag, orchestrator emits `outcome: exception` itself with diagnostics: "subagent return malformed".

24. **Plan-executor outcome lives in PLAN frontmatter, not just conversation history.** On completion, plan-executor writes a `last_executor_outcome:` field to the PLAN frontmatter (in addition to populating Executor Notes per the existing execute-plan workflow). Schema:
    ```
    last_executor_outcome:
      outcome: success | revision_needed | exception
      outcome_subtype: done | partially-complete | blocked | needs-revision
      executed: YYYY-MM-DD
      diagnostics_summary: "<one-line summary if not success>"
    ```
    This makes orchestrator re-entry deterministic: orchestrator reads PLAN frontmatter on re-entry, sees `last_executor_outcome.outcome`, branches accordingly. Avoids fragile "scan conversation history for the agent return". The `<pipeline-result>` block (decision 23) is the canonical wire format; the durable disk record is the frontmatter field.

25. **Spec verification by an independent verifier — not the executor.** Plan-executor (Haiku) runs the steps and self-reports `outcome: success`. Self-report is not enough — the *right things* must be independently verified to have been done. The orchestrator runs an **outcome-verification phase** between `executing` and `complete`, separate from the executor's own self-tick.

    **Two tiers of verification, both shell-runnable (zero LLM tokens for the happy path):**

    1. **State assertions** — `verify:` shell commands per Verification item. Mechanical: file exists, grep matches, command exit code. Catches "executor said done but didn't actually edit the file." Cheap, broad.

    2. **Spec acceptance sampling** — `acceptance:` shell commands that exercise the deliverable's *behaviour* against the Objective. Sampled, not exhaustive — one or two critical-path behavioural checks per PLAN. Catches "we did the steps but the result doesn't actually work." Examples: invoke the new skill on a representative input and grep the output for an expected pattern; trigger the new orchestrator on a tiny target and check the resulting frontmatter state; for a refactor, run a focused command and check it now does/doesn't do something.

    **Mandatory format for every PLAN's Verification section** (per parent decision 16's contract orientation):
    ```
    - [ ] <prose description>
          verify: <shell command, exit 0 = pass>      # state-level
    - [ ] <prose description>
          acceptance: <shell command, exit 0 = pass>  # spec-level (at least one per PLAN)
    - [ ] <prose>
          verify: human                                # for genuinely subjective items
    ```
    Each PLAN MUST include at least one `acceptance:` item — non-negotiable. Items marked `verify: human` are surfaced to the Human at outcome-verification time but do not auto-fail the pipeline (they're inconclusive, not failed).

    **Skills-as-deliverable carve-out (F3 from PLAN 202605011900, 2026-05-01):** PLANs whose deliverable is a Claude skill (or other artefact not invokable from a shell) MAY satisfy the at-least-one-`acceptance:` requirement via an artefact-property check (frontmatter parses, workflow content present, gate-clauses present) PLUS a `verify: human` for the actual invocation behaviour. Rationale: Claude skills are invoked by Claude, not by a shell; a true behavioural acceptance command cannot exist for them. The artefact-property `acceptance:` keeps the mechanical gate active; the `verify: human` keeps invocation fidelity in the loop.

    **Orchestrator's outcome-verification phase:**
    ```
    on plan-executor return with outcome: success:
      pipeline_phase = outcome-verifying  # new transient state
      run all verify: <shell> commands, tally pass/fail
      run all acceptance: <shell> commands, tally pass/fail

      collect all verify: human items into a "human_pending" list and write
        verification_state to PLAN frontmatter:
          verification_state:
            state_pass: N
            state_fail: N
            acceptance_pass: N
            acceptance_fail: N
            human_pending: [list of item descriptions]
            human_verdict: pending | all_pass | rejected
      commit + push the verification_state update (decision 22)

      branch:
        if any verify: or acceptance: shell command failed:
          outcome = revision_needed (override executor's success)
          diagnostics: list of failed assertions with commands and exit codes
          revert pipeline_phase to drafted, surface, halt
        else if human_pending is non-empty:
          surface the human items to the parent conversation with explicit prompt:
            "Outcome-verification: <N> auto-checks passed. <M> items need
             your eyeball:
             - [item 1 prose]
             - [item 2 prose]
             Reply 'all good' to mark them passed, or describe what is wrong."
          orchestrator returns control. pipeline_phase stays at outcome-verifying.
        else (no human items, all auto-checks passed):
          flip pipeline_phase = complete, advance to retire
    ```

    **`verify: human` flow — how it transmits to the orchestrator:**
    1. **Storage.** During the verification phase the orchestrator writes a `verification_state.human_pending` list to PLAN frontmatter (the `verify: human` items' prose descriptions). State is durable on disk so a missed re-entry is recoverable.
    2. **Surfacing.** The orchestrator surfaces those items to the parent conversation in a structured prompt: "These items need your eyeball: [list]. Reply 'all good' to mark them passed, or describe what is wrong."
    3. **Re-entry trigger.** The Human's reply is the trigger. The parent Claude, seeing a reply that's a yes/no/critique to the prompt, re-invokes `Skill("plan-pipeline", "<plan>")`.
    4. **Interpretation.** Orchestrator on re-entry reads `pipeline_phase: outcome-verifying` AND `verification_state.human_verdict: pending`. It examines the Human's most recent message. If unambiguously affirming ("all good", "yes", "approved", "pass"), set `human_verdict: all_pass` and continue; if rejecting/critiquing, set `human_verdict: rejected` with the Human's text as diagnostics. If ambiguous: re-prompt explicitly ("Please reply with 'pass' or 'fail [reason]'"). Update verification_state, commit + push.
    5. **Branch from there:**
       - `human_verdict: all_pass` → flip `pipeline_phase: complete`, advance to retire.
       - `human_verdict: rejected` → outcome = revision_needed, revert `pipeline_phase: drafted`, surface diagnostics.
    6. **Token cost:** parsing the Human's reply is the only LLM cost in the happy path (a single short read). Everything else is bash.

    **Where the criteria land:**
    - `_shared/plan-safe.md` (created in step 1): adds a clause that every Verification item must be shell-runnable (`verify:` or `acceptance:` or `verify: human`).
    - `audit-haiku-safe` (child 1420): one of its mechanical checks is "every Verification item has the right shell-command format and at least one acceptance: item exists".
    - `audit-sufficiency` (child 1425): adds a seventh lens — "does the acceptance sample actually exercise what the Objective claims, or is it only incidental?"
    - `plan-pipeline` orchestrator (child 1430): new transient `outcome-verifying` phase between `executing` and `complete`.
    - `plan-template.md` (updated in step 5): example Verification items show the new format.

    **Token cost analysis:** state and acceptance commands are bash — zero LLM tokens for execution. Only the failure path costs tokens (orchestrator surfaces diagnostics + Human resolves). Happy path is essentially free. Sampling principle: acceptance doesn't need full coverage; if the sample exercises a representative critical-path behaviour, the rest is presumed coherent.

**Constraints.**
- Several steps below ("create the `ideate` skill") are inherently design-heavy and not Haiku-safe by the strict definition. This PLAN's `assigned_to: sonnet` is a guideline (per decision 8). Expect `check-plan` (when it exists, and when run on this PLAN dogfood-style) to flag the design-heavy steps for decomposition into per-skill sub-plans.
- Dogfood phase requires `plan-pipeline` to exist before it can run, so the dogfood test is sequenced last (child PLAN 4) and targets the real-but-small `note-jot` utility skill — chosen so the ideation phase has genuine surface to exercise.
- **Bootstrap chicken-and-egg.** This PLAN builds the orchestrator that future PLANs will use. Therefore *this* parent cannot itself be orchestrated by `plan-pipeline` — the orchestrator doesn't exist until child PLAN 3 lands. Parent execution is **manually orchestrated by the Human** through the bootstrap: Human runs mechanical steps (1–6a) via `execute-plan` directly; pauses at step 6 checkpoint; runs children sequentially (each via `execute-plan` directly); resumes parent at step 8 once children are done; runs step 9 (unblock dogfood) which itself runs via `execute-plan` directly. The orchestrator takes over only for *future* PLANs after the bootstrap completes.
- Existing skills currently contain many "Ken" references and some hard-tier assignments. Converting those is **out of scope** for this PLAN — captured here as a follow-up: a separate sweep PLAN should propagate the Human/guideline conventions across all existing skill files **and their `workflows/` files** (e.g. `execute-steps.md` Step 9 "Report to Ken" survives even after this PLAN runs).

## Learnings captured during sharpening

Discoveries made while sharpening this PLAN to be Haiku-safe — to feed forward into child PLANs and future iterations.

- **Sharpening requires reading source files.** Embedding exact line numbers (e.g. step 3's "delete lines 84–87 of execute-steps.md") is only possible after reading the file once. Future PLANs targeted at Haiku should bake this read-then-embed step into their authoring workflow.
- **Auto-retire lives only in `workflows/execute-steps.md`** (Step 8, lines 84–87 at time of authoring), not in `execute-plan/SKILL.md`'s `<success_criteria>`. The original step 3 wording about "drop mention from success_criteria" was inaccurate; the sharpened version replaces it with adding a principle to `<essential_principles>`.
- **Workflow files contain Ken references too.** `execute-steps.md` Step 9 has "Report to Ken" — the follow-up sweep PLAN must scope `.claude/skills/*/workflows/*.md`, not just SKILL.md files.
- **Trigger-phrase collision is a real risk.** `write-bus-plan`'s existing triggers include "create plan file" (close to "let's make a plan"). Child PLAN for `plan-pipeline` must verify Skill matcher routes correctly and may need to disambiguate trigger phrases.
- **`parent_plan_of_plans` traversal is a real orchestrator concern.** When `plan-pipeline` becomes a parent (this PLAN's situation), it must know how to traverse to children, wait for their completion, and resume the parent. This needs explicit handling in the `plan-pipeline` SKILL design (child PLAN 3).
- **`pipeline_phase` advancement vs. spawned-children blocking.** When a parent pauses at step 6 (Human checkpoint) and spawns children, the parent's pipeline_phase shouldn't auto-advance until children complete. The orchestrator dispatch logic needs a "waiting on children" state or equivalent. Land in child PLAN 3.
- **Ideation can produce RESEARCH/ADVICE, not just PLAN content** (decision 12). The original conception of `ideate` as "purely conversational, no disk output" was incomplete — it should be able to invoke `write-bus-input` mid-arc when external data or decisions need persisting. Land in child PLAN 1 (create-ideate-skill): document the trigger conditions and the handoff to `write-bus-input`.
- **Subagents must not touch git** (decision 13). Surfaced after the agent layout was drafted. Means `execute-plan` and `retire` both need their git operations stripped — they currently perform commit/push themselves. The orchestrator (or Human in bootstrap) takes over. Affects the design of child PLAN 3 (`plan-pipeline` skill must explicitly handle git after each phase). Memory note about retire-then-commit remains valid at the user-observed level.
- **Bootstrap implication of decision 13.** During bootstrap, `execute-plan` no longer commits/pushes. The Human must run `git add -A && git commit && git push` manually after each `execute-plan` invocation (or after a batch of mechanical steps). This is friction — but only during the bootstrap; once `plan-pipeline` exists, the orchestrator handles git automatically.
- **Gate semantics matter** (decision 14). The original framing of `check-plan` as "produces a review" was too loose — review output is consumed by a gate, so the format must support mechanical enforcement (explicit Blockers/Not-blockers subgroups + a summary count). Land in child PLAN 2 (`create-check-plan-skill`): output schema must include a final `Blockers: N` line. Land in child PLAN 3 (`create-plan-pipeline-skill`): dispatch logic reads the count and enforces the gate, not interprets prose.
- **Human attention is a scarce resource — triage before asking** (decision 15). Surfaced when the Human asked which decisions actually needed input out of 14 listed. Most were already-locked or mechanically-forced; only 3 were real judgement calls. The pattern of "list everything and ask the Human to confirm all of it" wastes attention. Land in child PLAN 1 (ideate — converge output triages decisions), child PLAN 2 (check-plan — Human-input items get triaged), child PLAN 3 (plan-pipeline — applies triage at every Human-facing surface).
- **Sequential Workflow Orchestration framework affirms the existing direction** (decision 16). Most of the framework's five layers and four anti-patterns were already implicitly addressed by prior decisions; the framework just makes the architecture explicit. Two new concrete requirements emerged: every new skill must declare `<preconditions>` and `<output_schema>` blocks so handoffs are schema-routed, not prose-interpreted. Existing skills are partial-compliance and retrofitting is captured as follow-up.
- **Mechanical review and conceptual review are different operations and need different model tiers** (decision 5 revised). After six rounds of mechanical check-plan came back `Blockers: 0`, a Human-led Opus pass surfaced six architectural issues that none of the prior reviews had touched (load-bearing assumption verification, bootstrap validation timing, dogfood fidelity, orchestrator edge cases, memory note staleness, decision-overhead meta-concern). The lesson: a passing plan-safety check is not a sufficiency check. They use different lenses (assumption/validation/fidelity/edges/freshness/meta vs. concreteness/atomicity/line-numbers) and benefit from different model tiers (Opus vs. Sonnet/Haiku). The exemplar ADVICE file 202605011500 captures the shape of an Opus-pass output for the design of `audit-sufficiency`.

- **Probe findings constrained the architecture** (decisions 17, 18). Subagents do not inherit parent's skill registry → must preload via `skills:` frontmatter. `run_in_background` as a tool-call parameter is undocumented; only `background: true` in subagent frontmatter is documented → all subagent dispatches use frontmatter-level background, and the orchestrator accepts that completion-triggered transitions are conversational (not skill-internal). This is a meaningful architectural constraint inherited from the platform. **Amendment 2026-05-01 (surfaced during dogfood iteration 2 of child 1440):** `run_in_background: true` is now documented as a tool-call parameter on the Agent tool — the probe-2 finding was point-in-time accurate but became stale within hours. Decision 18's *intent* (only the executor runs background; one conversational re-entry; idempotent from disk) survives; the *mechanism* updates — orchestrator passes the parameter explicitly. Lesson: probe findings are perishable; re-verify before treating them as load-bearing in a long-lived PLAN.

- **Chatty subagents emit milestones and anomaly halts** (decision 19). Background execution costs visibility — without milestones, the parent has no idea what a 5-minute plan-executor run is doing. Solution: skills declare milestone boundaries and emit structured messages; orchestrator processes them, surfaces progress to Human, and pulls a full stop if any subagent reports anomaly. The pattern is novel — we're inventing message-passing semantics for skills. Reliability will need a second probe before bootstrap execution.

- **Audit phases loop, not gate-once** (decision 21). Original framing of `drafted → checked` as a single-pass gate was incomplete. In practice, blockers trigger revision, which triggers re-audit. The orchestrator drives the loop with iteration count, surfaces each iteration to the Human, and triggers anomaly halt on max iterations to prevent runaway loops. Same pattern applies to `executing` (halt-on-failure → revise → re-execute), already covered by execute-plan's halt protocol but worth stating uniformly.

- **Subagent-driven exception is cleaner than orchestrator-driven cancellation** (decisions 18, 19 simplified). After probe-2 found platform doesn't support mid-execution streaming or cancellation, the user reframed: "subagents terminate early on problems; output surfaces only on completion is a feature, not a limitation; orchestrator decides whether to pull kanban full stop based on the outcome enum". This collapsed three earlier complications (streaming, cancellation, mid-execution coordination) into one clean transactional contract — `outcome: success | revision_needed | exception` — with the subagent owning its own anomaly detection and termination. The orchestrator becomes a result-router, not a supervisor. Significant simplification.

- **Loop conditions are explicit and durable** (decision 21 sharpened). Audit loop state lives in PLAN frontmatter (`audit_state: {sufficiency_iterations, plan_safety_iterations, last_stage, last_outcome}`) so re-invocations between Human-revision pauses pick up correctly. Per-stage iteration counters mean only the failing audit re-runs after revision (sufficiency doesn't re-run if only plan-safety failed). The "rerun signal" is `outcome: revision_needed`; the "continue signal" is `outcome: success`; the "stop signal" is `outcome: exception` (from subagent or orchestrator's MAX_ITERATIONS check).

- **Orchestrator commits at every milestone** (decision 22). Subagents can't push (decision 13); the orchestrator owns git. Every state transition that produces filesystem changes — including audit_state frontmatter writes during loops, Human revisions between iterations, and exception halts — gets committed and pushed before continuing. Work is never lost; even mid-loop revisions are durable in origin.

- **Wire format and durable record are different concerns** (decisions 23, 24). The agent's return message is the *wire format* (a `<pipeline-result>` tag with JSON code fence — greppable, parseable, fails-loud-on-malformed). The PLAN frontmatter (`last_executor_outcome` field, `audit_state` field, etc.) is the *durable disk record* — what the orchestrator reads on re-entry. Wire vs. disk are intentionally separate so re-entry doesn't depend on scanning conversation history.

- **Codified orchestrator behaviour replaces memory crutches** (step 3c revised). Originally we planned to *update* the `feedback_retire_push.md` memory note to reflect decision 13. After review, the memory note's purpose (reminding future Claude to commit+push after retire) is fully superseded by decision 22's stepwise milestone-commit logic. We *delete* the memory note instead — the orchestrator's codified behaviour is the new source of truth. Standalone retire callers (Human ad-hoc) own git themselves; no memory note needed.

- **Spec verification ≠ step-tick verification** (decision 25). The original Verification section had Sonnet/Haiku read each item and judge "is this true?" — vulnerable to executor laziness/hallucination, and only checks intermediate state. Decision 25 splits into mechanical state assertions (`verify:`) and behavioural acceptance samples (`acceptance:`), both shell-runnable, executed by the orchestrator (not the executor) as a separate `outcome-verifying` phase between executing→complete. Token cost is near-zero on the happy path; only the failure path costs LLM tokens. The `verify: human` annotation handles genuinely subjective items via a structured prompt → human reply → orchestrator interpretation cycle, with verification_state on disk for re-entry idempotency.

## Steps

1. **Create shared plan-safe reference (with verification-format clause).**
   - Create new file `.claude/skills/_shared/plan-safe.md`.
   - Copy the content of the `<plan_safe_definition>` block (lines 23–36 of `.claude/skills/execute-plan/SKILL.md`) into it, formatted as a standalone reference doc with H1 title "Plan-safe definition".
   - **Append a new section** at the end:
     ```markdown
     ## Verification format requirement (per PLAN 202605011400 decision 25)

     Every PLAN's `## Verification` section item must be shell-runnable, with one of the following annotations on the line directly below the prose checkbox:

     - `verify: <shell command>` — state assertion (file exists, grep matches, command exit code). Exit 0 = pass.
     - `acceptance: <shell command>` — spec-level behavioural check that exercises the deliverable. Exit 0 = pass. **Every PLAN must include at least one `acceptance:` item.**
     - `verify: human` — genuinely subjective item; surfaced for Human eyeball but does not auto-fail.

     The orchestrator runs all `verify:` and `acceptance:` commands as a separate outcome-verification phase (after plan-executor returns success, before advancing to complete). Failures override the executor's self-reported success.
     ```
   - Verification:
     - [ ] file exists
           `verify: test -f .claude/skills/_shared/plan-safe.md`
     - [ ] file references the verification format clause
           `verify: grep -q "Verification format requirement" .claude/skills/_shared/plan-safe.md`
     - [ ] file's plan-safe definition body matches the original (sampled)
           `acceptance: grep -q "Concrete: specific file paths" .claude/skills/_shared/plan-safe.md`

2. **Update `execute-plan/SKILL.md` to reference the shared file.**
   - Replace lines 23–36 of `.claude/skills/execute-plan/SKILL.md` with a one-line reference: `**Plan-safe definition:** See [../_shared/plan-safe.md](../_shared/plan-safe.md) — single source of truth shared with check-plan.`
   - Verification: SKILL.md no longer contains the inline definition; reference link resolves.

3. **Decouple auto-retire AND git operations from `execute-plan`; map halt-on-failure to outcome enum.**
   - Open `.claude/skills/execute-plan/workflows/execute-steps.md`.
   - Delete lines 69–87 in one operation. This removes the entire `## Step 7: Git commit and push` block (lines 69–82) AND the entire `## Step 8: Retire the PLAN file` block (lines 84–87). The blank line 83 between them is included in the deletion.
   - Renumber the remaining `## Step 9: Report to Ken` heading to `## Step 7: Report to Ken`.
   - In `## Step 2: Execute Steps in order` (lines 9–13 of the workflow), append a new bullet immediately after the existing "If a step fails..." bullet:
     ```
     - On halt-on-failure (any of the conditions above — step failure, ambiguity, [Ken]/[Human] marker requiring approval, destructive operation without consent): populate `last_executor_outcome` in PLAN frontmatter with `outcome: exception` and a `diagnostics_summary`; populate Executor Notes; return. Do NOT commit (caller owns git per decision 13). Do NOT mark `WIP:` commit (no commit happens here). The orchestrator (or Human in bootstrap) reads the frontmatter outcome on completion and applies the kanban full-stop.
     ```
   - In the existing `## Step 4: Populate the PLAN's Executor Notes section` (around lines 20–27), add a sub-bullet under the section to also write `last_executor_outcome` frontmatter (per parent decision 24):
     ```
     - Also write `last_executor_outcome` frontmatter on completion (any outcome): outcome (enum: success | revision_needed | exception), outcome_subtype (existing values: done | partially-complete | blocked | needs-revision), executed (today YYYY-MM-DD), diagnostics_summary (one-line; empty if outcome=success).
     ```
   - Open `.claude/skills/execute-plan/SKILL.md`. In the `<essential_principles>` block (lines 6–13), add three final lines:
     ```
     Git commit and push are the caller's responsibility (e.g. plan-pipeline orchestrator, or the Human during bootstrap) — execute-plan no longer commits or pushes.
     Retirement of the PLAN file is the caller's responsibility — execute-plan no longer auto-retires.
     On any halt (success or failure), write `last_executor_outcome` to PLAN frontmatter so callers can route deterministically (parent PLAN decision 24).
     ```
   - In `<success_criteria>`, delete the two lines beginning `- Git commit created with` and `- Commit pushed to origin`. Add: `- last_executor_outcome frontmatter populated with outcome enum + diagnostics_summary.`
   - In `<constraints>`, replace the line `Never skip git commit + push — the source of truth must be in origin` with `Caller must commit and push after execute-plan returns; never use --no-verify, --force, --force-with-lease, or bypass signing.`
   - Update `<plan_safe_definition>` (already extracted to `_shared/plan-safe.md` in step 1) — no change here, just confirming it stayed shared.
   - Update the agent return: subagent's response message ends with a `<pipeline-result>` block per parent decision 23 (the wire format) summarising the outcome. The Executor Notes + frontmatter + LOG remain the durable record.
   - Verification: `grep -n "Git commit and push" .claude/skills/execute-plan/workflows/execute-steps.md` returns no matches; `grep -n "Retire the PLAN file" .claude/skills/execute-plan/workflows/execute-steps.md` returns no matches; `grep -n "last_executor_outcome" .claude/skills/execute-plan/workflows/execute-steps.md` returns the new bullet; SKILL.md essential_principles contains all three new lines; SKILL.md success_criteria contains the new last_executor_outcome line and no longer mentions commit hash or push.

3b. **Decouple git operations from `retire` skill.**
   - Open `.claude/skills/retire/workflows/retire-file.md`.
   - Delete line 9 (the `5. **Commit and push**: ...` step).
   - Renumber the remaining steps: 6 → 5 (Confirm), 7 → 6 (Plan note).
   - In the (now) Step 5 "Confirm" line, remove the trailing `, and commit hash` so it reads `Return success message with source and destination paths.`
   - Update the **Return** example block (lines 20–24 originally) to remove any commit-hash line if present.
   - Open `.claude/skills/retire/SKILL.md`.
   - In the description (line 3), remove the trailing sentence `Always commits and pushes after moving.`
   - In the `<quick_start>` `Returns:` line, replace `Confirmation that file has been retired, committed, and pushed.` with `Confirmation that file has been retired. Caller is responsible for git commit and push.`
   - In `<success_criteria>`, delete the line `- Git commit created and pushed to origin`.
   - Verification: `grep -in "commit\|push" .claude/skills/retire/SKILL.md .claude/skills/retire/workflows/retire-file.md` returns no matches related to git operations within the retire skill (the "git" string should not appear).

3c. **Memory note `feedback_retire_push.md` deletion — completed out-of-band 2026-05-01 before phase A.**
   - Memory file `C:/Users/Ken/.claude/projects/d--Projects-Diarizer/memory/feedback_retire_push.md` was deleted directly by the agent (`Bash rm`) and MEMORY.md was rewritten to remove the index entry. Done before bootstrap phase A to avoid user-space-memory permission risk during automated execute-plan.
   - Rationale: the orchestrator handles git deterministically per decision 22; the memory note's purpose (reminding future Claude to commit+push after retire) is superseded by codified orchestrator behaviour.
   - This step is therefore a **no-op during execute-plan** but remains documented here for traceability and verification.
   - Verification:
     - [ ] memory file no longer exists
           `verify: ! test -f "C:/Users/Ken/.claude/projects/d--Projects-Diarizer/memory/feedback_retire_push.md"`
     - [ ] MEMORY.md does not reference feedback_retire_push
           `verify: ! grep -q "feedback_retire_push" "C:/Users/Ken/.claude/projects/d--Projects-Diarizer/memory/MEMORY.md"`

4. **Loosen `write-bus-plan` description and principles.**
   - Open `.claude/skills/write-bus-plan/SKILL.md`.
   - Replace line 3 (the `description:` frontmatter line) with exactly:
     ```
     description: Bus transcription skill. Writes PLAN files (and updates monthly LOG, rollover files) at any point in a plan's lifecycle — including incremental in-flight writes during ideation, draft updates, and final transcription. Overwrites of existing PLAN files with revised content are permitted. Trigger phrases: "write this plan to bus", "create plan file", "update the plan", "update the log", "write bus file", "create the log".
     ```
   - In the `<essential_principles>` block (lines 8–14), insert a new line before the closing tag, exactly:
     ```
     Overwriting an existing PLAN file with updated content is permitted during the drafting/ideation phase — preserve frontmatter `created`, `created_month`, and `created_by`; refresh body content and any `last_updated`-style fields.
     ```
   - Verification: `grep -n "incremental in-flight writes" .claude/skills/write-bus-plan/SKILL.md` returns the description line; `grep -n "Overwriting an existing PLAN file" .claude/skills/write-bus-plan/SKILL.md` returns the new principle line.

5. **Document `pipeline_phase` field in bus-conventions and template.**
   - Open `.claude/skills/write-bus-plan/references/bus-conventions.md`.
   - After line 35 (the line ending "everything else rolls over.") and before line 37 (the "## PLAN Input Linkage" heading), insert exactly:
     ```

     ## PLAN Pipeline Phase

     Frontmatter `pipeline_phase` field, orthogonal to `status`. Tracks orchestration state for PLANs managed by the `plan-pipeline` skill.

     ```
     drafting → drafted → checked → executing → complete
     ```

     - `drafting` — ideation underway; PLAN being written/updated incrementally.
     - `drafted` — ideation closed; PLAN written; awaiting sufficiency + plan-safety review.
     - `checked` — review passed; ready to execute. Aligns with `status: ready`.
     - `executing` — `plan-executor` subagent has been dispatched.
     - `complete` — execution done; ready for retirement.

     **Default for ad-hoc PLANs** (created directly via `write-bus-plan` outside the pipeline): field absent or empty. When `plan-pipeline` is invoked against such a PLAN, it treats absent/empty as `drafted` (assume the PLAN is already written and ready for review), not `drafting` — to avoid re-ideating an already-authored plan.
     ```
   - Open `.claude/skills/write-bus-plan/templates/plan-template.md`. After the `parent_plan_of_plans:` frontmatter line (line 21), insert exactly:
     ```
     pipeline_phase: ""       # plan-pipeline orchestration state; empty for ad-hoc PLANs (see bus-conventions.md)
     ```
   - Replace the existing `## Verification` section example items (lines 39–43 of plan-template.md) with this updated example, demonstrating the verify:/acceptance: format (per parent decision 25):
     ```markdown
     ## Verification
     - [ ] [State check — e.g. "file X exists"]
           `verify: test -f path/to/file`
     - [ ] [State check — e.g. "frontmatter status field is 'active'"]
           `verify: grep -q "^status: active" path/to/file`
     - [ ] [Acceptance check — exercises the deliverable's behaviour. AT LEAST ONE per PLAN.]
           `acceptance: <shell command that runs the deliverable on a representative input and checks its output>`
     - [ ] [Subjective item — surfaced for Human eyeball; no auto-fail]
           `verify: human`
     ```
   - Verification:
     - [ ] bus-conventions.md contains pipeline_phase section
           `verify: grep -q "PLAN Pipeline Phase" .claude/skills/write-bus-plan/references/bus-conventions.md`
     - [ ] template includes pipeline_phase frontmatter
           `verify: grep -q "^pipeline_phase:" .claude/skills/write-bus-plan/templates/plan-template.md`
     - [ ] template Verification section references verify:/acceptance: format
           `verify: grep -q "acceptance:" .claude/skills/write-bus-plan/templates/plan-template.md`
     - [ ] template format actually parses as the new format (acceptance check)
           `acceptance: grep -E '^\s*(verify|acceptance):' .claude/skills/write-bus-plan/templates/plan-template.md | wc -l | awk '{exit !($1>=3)}'`

6. **[Human — design-review checkpoint]** Pause here.
   - **Apply the decision-triage rule (decision 15) first.** Classify each design decision (1–N) into Already-locked, Mechanically-forced, or Real-judgement-call.
   - Surface only the Real-judgement-call items to the Human as questions. Already-locked and Mechanically-forced are listed briefly for transparency, no answer required.
   - If any judgement call shifts, revise this PLAN, revert `pipeline_phase: drafted`, and re-run `check-plan` before continuing.
   - If no shifts, advance to step 6a.

6a. **Create pinned-model subagent definitions in `.claude/agents/`.**
   - Create the directory `.claude/agents/` if it does not exist.
   - Create `.claude/agents/plan-writer.md` with exactly this content:
     ```
     ---
     name: plan-writer
     model: sonnet
     skills: [write-bus-plan]
     description: Foreground subagent that runs the write-bus-plan skill to transcribe or update a PLAN file. Invoked by plan-pipeline at draft and checkpoint moments. Per parent decisions 8, 17, 18 (foreground; no background:true since runs are fast).
     ---

     # plan-writer

     Single-purpose foreground subagent. Inputs (decision 20): `{plan_content: string, target_filename: string, mode: enum[create, update]}`. Outputs: `{outcome: enum[success, revision_needed, exception], payload: {filename_written, action, log_updated}, diagnostics}`.

     Exception conditions (decision 19): target filename collides with an existing PLAN created in a different month; frontmatter missing required fields; rollover detected mid-write; LOG file unreachable.

     Does not handle ideation, review, execution, or retirement. Does not commit/push (decision 13).
     ```
   - Create `.claude/agents/sufficiency-auditor.md` with exactly this content:
     ```
     ---
     name: sufficiency-auditor
     model: opus
     skills: [audit-sufficiency]
     description: Foreground subagent that runs the audit-sufficiency skill (Opus-grade conceptual review). Invoked by plan-pipeline at the drafted phase loop, before plan-safety-auditor. Per decisions 17, 18.
     ---

     # sufficiency-auditor

     Inputs: `{plan_path: string}`. Outputs: `{outcome: enum[success, revision_needed, exception], payload: {blockers_count, review_text, triaged_human_items}, diagnostics}`.

     Outcome semantics: `success` → blockers_count == 0, advance to plan-safety-auditor. `revision_needed` → blockers_count > 0, surface review_text to Human for revision. `exception` → see exception_conditions below.

     Exception conditions: PLAN file unreadable; referenced linked_inputs files missing; PLAN structurally malformed (no Steps section, no Verification section); review process itself fails (e.g. shared `_shared/plan-safe.md` reference missing — would block plan-safety-auditor next anyway).
     ```
   - Create `.claude/agents/plan-safety-auditor.md` with exactly this content:
     ```
     ---
     name: plan-safety-auditor
     model: sonnet
     skills: [audit-haiku-safe]
     description: Foreground subagent that runs the audit-haiku-safe skill (Sonnet-grade mechanical review against shared plan-safe definition). Invoked by plan-pipeline at the drafted phase loop, after sufficiency-auditor passes. Per decisions 17, 18.
     ---

     # plan-safety-auditor

     Inputs: `{plan_path: string}`. Outputs: `{outcome: enum[success, revision_needed, exception], payload: {blockers_count, review_text}, diagnostics}`.

     Outcome semantics: `success` → blockers_count == 0, advance pipeline_phase to `checked`. `revision_needed` → blockers_count > 0, surface to Human. `exception` → see below.

     Exception conditions: invoked before sufficiency-auditor passed (returns `outcome: exception`); PLAN file unreadable; referenced source `.claude/skills/_shared/plan-safe.md` missing.
     ```
   - Create `.claude/agents/plan-executor.md` with exactly this content:
     ```
     ---
     name: plan-executor
     model: haiku
     background: true
     skills: [execute-plan]
     description: Background subagent that runs the execute-plan skill against a checked PLAN. The ONLY background subagent — long-running phase where parent responsiveness matters. Invoked at checked→executing. Per decisions 17, 18.
     ---

     # plan-executor

     Inputs: `{plan_path: string}`. Outputs: `{outcome: enum[success, revision_needed, exception], payload: {outcome_subtype: enum[done, partially-complete, blocked, needs-revision], executor_notes: string, files_modified: list}, diagnostics}`.

     Outcome semantics: `success` → outcome_subtype == done, all verification passed. `revision_needed` → outcome_subtype in [partially-complete, blocked, needs-revision]; orchestrator reverts pipeline_phase to drafted. `exception` → see below.

     Exception conditions: step requires Human approval (per [Human] marker — terminate early, return for Human input); destructive operation lacks explicit approval; tools/permissions unavailable; unsigned commit attempted; halt-on-failure trigger fires; upstream PLAN file modified mid-execution.

     Note: `skills:` should be expanded to include any skills that `execute-plan` itself dispatches to mid-flight (per the existing `<skill_invocation_semantics>` block in execute-plan/SKILL.md). Audit at execution time and add as needed.

     Does not commit/push (decision 13). Does not retire (decision 3).
     ```
   - Create `.claude/agents/plan-retirer.md` with exactly this content:
     ```
     ---
     name: plan-retirer
     model: haiku
     skills: [retire]
     description: Foreground subagent that runs the retire skill against a completed PLAN. Invoked by plan-pipeline at the complete phase. Per decisions 17, 18.
     ---

     # plan-retirer

     Inputs: `{plan_path: string}`. Outputs: `{outcome: enum[success, revision_needed, exception], payload: {retired_path, gitignore_updated}, diagnostics}`.

     Outcome semantics: `success` → file retired cleanly. `revision_needed` → not used by this skill. `exception` → see below.

     Exception conditions: source file does not exist; destination `Retired/<filename>` collides with existing file; `.gitignore` unreachable.

     Does not commit/push (decision 13). Optional — plan-pipeline may inline the retire call instead.
     ```
   - **No `planning-ideator` agent** — ideation is interactive and always runs in the parent session per decision 10's hard rule.
   - Verification: `ls .claude/agents/` lists all five files (plan-writer, sufficiency-auditor, plan-safety-auditor, plan-executor, plan-retirer); `head -5 .claude/agents/plan-writer.md` (and each other agent file) shows valid frontmatter with the correct `model:` value (sonnet/opus/sonnet/haiku/haiku respectively).

**Note on steps 7 and 9 (bootstrap orchestration).** Steps 7 and 9 contain "wait for children to reach `status: done`" — this wait is performed by the Human, not by `execute-plan`. The Human flips children's `status: blocked` → `status: ready` (mechanical, per these steps), then triggers each child via `execute-plan` directly, then resumes this parent at the next step once children are done. See Constraints section "Bootstrap chicken-and-egg" for context.

7. **Unblock child PLANs 1–4 (skill creation).**
   - Confirm the following child PLAN files exist in `Bus/`:
     - `202605011410_PLAN_create-ideate-skill.md`
     - `202605011420_PLAN_create-audit-haiku-safe-skill.md`
     - `202605011425_PLAN_create-audit-sufficiency-skill.md`
     - `202605011430_PLAN_create-plan-pipeline-skill.md`
   - In each child file, change frontmatter `status: blocked` → `status: ready` and clear `blocked_by: ""`.
   - Wait for all four children to reach `status: done` before proceeding to step 8. (Children will be picked up by execution agents per their own pipeline_phase. They are themselves design-heavy; each child's own check-plan pass will determine its execution path.)
   - Verification: `grep -l "status: ready" Bus/202605011410_PLAN_*.md Bus/202605011420_PLAN_*.md Bus/202605011425_PLAN_*.md Bus/202605011430_PLAN_*.md` lists all four after the flip; on completion check, `grep -l "status: done"` lists all four.

8. **Update CLAUDE.md skill table.**
   - Open `CLAUDE.md`. Locate the skill table (under the "## Skills" section, after the introduction line referencing `SKILLS_IMPLEMENTATION_GUIDE.md`).
   - Append the following four rows immediately after the existing `retire` row:
     ```
     | `ideate` | Three-phase ideation arc (Clarify → Survey → Converge); imposes requirement-before-mechanism discipline; runs in parent session only |
     | `audit-sufficiency` | Opus-grade conceptual audit of a PLAN — assumptions, edge cases, validation path, meta-design. Returns structured review + Blockers count. Run before audit-haiku-safe |
     | `audit-haiku-safe` | Sonnet-grade mechanical audit of a PLAN against the shared plan-safe definition; per-step concreteness/atomicity/exact-text checks. Returns structured review + Blockers count |
     | `plan-pipeline` | End-to-end orchestrator: walks ideation → write → audit-sufficiency → audit-haiku-safe → execute → retire; dispatches to pinned-model subagents; activates on "let's make a plan" |
     ```
   - Verification: `grep -n "plan-pipeline" CLAUDE.md` returns the new row; table is well-formed (pipe count consistent with neighbours).

9. **Unblock child PLAN 4 (dogfood test).**
   - Confirm `Bus/202605011440_PLAN_dogfood-plan-pipeline.md` exists.
   - Change its frontmatter `status: blocked` → `status: ready` and clear `blocked_by: ""`.
   - Wait for it to reach `status: done`. Its outcome and findings will be captured in its own Executor Notes; reference back here under this PLAN's Executor Notes.
   - Verification: dogfood PLAN reaches `status: done` and is retired.

## Verification

- [ ] `.claude/skills/_shared/plan-safe.md` exists with the extracted definition (parent step 1)
- [ ] `execute-plan/SKILL.md` references the shared file (no inline definition) (parent step 2)
- [ ] `execute-plan` workflow no longer contains `## Step 7: Git commit and push` or `## Step 8: Retire the PLAN file`; remaining step renumbered to Step 7 (parent step 3)
- [ ] `execute-plan/SKILL.md` `<essential_principles>` notes both git AND retirement are caller's responsibility; `<success_criteria>` no longer references commit/push; `<constraints>` updated (parent step 3)
- [ ] `retire/workflows/retire-file.md` no longer contains commit/push step; `retire/SKILL.md` description, quick_start return, and success_criteria no longer reference git operations (parent step 3b)
- [ ] Memory file `feedback_retire_push.md` deleted; MEMORY.md index entry removed (parent step 3c)
- [ ] `execute-plan` halt-on-failure protocol writes `last_executor_outcome` frontmatter with `outcome: exception`; SKILL.md success_criteria includes last_executor_outcome line (parent step 3, per decisions 19, 24)
- [ ] `_shared/plan-safe.md` includes the verification-format clause requiring verify:/acceptance:/human annotations (parent step 1, per decision 25)
- [ ] `plan-template.md` Verification section demonstrates the verify:/acceptance: format with at least one acceptance: example (parent step 5, per decision 25)
      `acceptance: grep -E '^\s*acceptance:' .claude/skills/write-bus-plan/templates/plan-template.md`
- [ ] `write-bus-plan/SKILL.md` description and `<essential_principles>` updated per step 4 exact text
- [ ] `bus-conventions.md` contains "## PLAN Pipeline Phase" section; `plan-template.md` contains `pipeline_phase: ""` (parent step 5)
- [ ] `.claude/agents/` contains five pinned-model subagent files (plan-writer, sufficiency-auditor, plan-safety-auditor, plan-executor, plan-retirer); no `planning-ideator` (parent step 6a)
- [ ] All four child skill-creation PLANs reach `status: done` (parent step 7)
- [ ] CLAUDE.md skill table contains rows for `ideate`, `audit-sufficiency`, `audit-haiku-safe`, `plan-pipeline` (parent step 8)
- [ ] Dogfood child PLAN reaches `status: done` and is retired (parent step 9)
- [ ] All four children's outcomes summarised in this PLAN's Executor Notes

## Executor Notes

**Executed:** 2026-05-01 (Phase A — bootstrap mechanical setup)
**Outcome:** partially-complete (steps 1–6a done; halted at step 7 awaiting children)
**What was done:**
- Step 1: created `.claude/skills/_shared/plan-safe.md` with the extracted plan-safe definition + verification format clause (decision 25)
- Step 2: replaced inline plan_safe_definition in execute-plan/SKILL.md with reference to shared file
- Step 3: deleted Step 7 (git) + Step 8 (retire) blocks from execute-plan workflow; added halt-on-failure → outcome:exception bullet to Step 2; added `last_executor_outcome` write to Step 4; updated SKILL.md essential_principles, success_criteria, constraints
- Step 3b: removed Commit-and-push step from retire workflow; updated retire SKILL.md description, quick_start return, success_criteria, objective to drop git references
- Step 3c: no-op (memory file `feedback_retire_push.md` already deleted out-of-band before Phase A; MEMORY.md cleared)
- Step 4: loosened write-bus-plan/SKILL.md description and added incremental-overwrite principle
- Step 5: added "PLAN Pipeline Phase" section to bus-conventions.md; added `pipeline_phase` frontmatter line + verify:/acceptance: Verification format guidance to plan-template.md
- Step 6: Human checkpoint — approved in conversation (Decisions 1–25 locked at design-review point on 2026-05-01)
- Step 6a: created 5 agent files in `.claude/agents/`: plan-writer (sonnet), sufficiency-auditor (opus), plan-safety-auditor (sonnet), plan-executor (haiku, background:true), plan-retirer (haiku)

**Blockers (if any):** Step 7 contains "Wait for all four children to reach `status: done`" — non-mechanical coordination point. Bootstrap pattern: Human runs children sequentially via execute-plan, then resumes parent at step 8.

**Files modified:**
- Created: `.claude/skills/_shared/plan-safe.md`
- Modified: `.claude/skills/execute-plan/SKILL.md`
- Modified: `.claude/skills/execute-plan/workflows/execute-steps.md`
- Modified: `.claude/skills/retire/SKILL.md`
- Modified: `.claude/skills/retire/workflows/retire-file.md`
- Modified: `.claude/skills/write-bus-plan/SKILL.md`
- Modified: `.claude/skills/write-bus-plan/references/bus-conventions.md`
- Modified: `.claude/skills/write-bus-plan/templates/plan-template.md`
- Created: `.claude/agents/plan-writer.md`
- Created: `.claude/agents/sufficiency-auditor.md`
- Created: `.claude/agents/plan-safety-auditor.md`
- Created: `.claude/agents/plan-executor.md`
- Created: `.claude/agents/plan-retirer.md`
- Deleted (out-of-band): `C:/Users/Ken/.claude/projects/d--Projects-Diarizer/memory/feedback_retire_push.md`
- Modified (out-of-band): `C:/Users/Ken/.claude/projects/d--Projects-Diarizer/memory/MEMORY.md`

**Note on Verification:** the inline per-step verification grep for "Concrete: specific file paths" (step 1) failed due to markdown bold formatting (`**Concrete:**` vs literal `Concrete:`); body content is correct, confirmed by direct inspection. All other phase A verification commands passed.

**last_executor_outcome:**
  outcome: revision_needed
  outcome_subtype: partially-complete
  executed: 2026-05-01
  diagnostics_summary: "Phase A complete (steps 1-6a). Halted at step 7 'Wait for children to complete' — bootstrap pattern requires Human to run children sequentially before resuming parent at step 8."
