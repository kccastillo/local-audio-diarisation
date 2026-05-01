---
name: ideate
description: Three-phase ideation arc (Clarify → Survey → Converge) for shaping a problem into a plan-ready idea. Imposes "requirement before mechanism" discipline; presents options with a recommendation rather than jumping to solution. Runs ONLY in the parent session — never as a subagent (interactive). May produce RESEARCH/ADVICE Bus inputs mid-arc when external data or persisted decisions are needed. Trigger phrases: "let's ideate", "let's think through X", "help me think about Y", "ideate Z", "what should we do about W".
---

**Plan-safe definition:** See [../_shared/plan-safe.md](../_shared/plan-safe.md).

<essential_principles>
Run only in the parent session — never inside a subagent (per parent PLAN 202605011400 decision 10). The conversational arc requires Human turn-taking which subagents cannot offer.
Walk three explicit phases in order: Clarify → Survey → Converge. Do not skip; do not merge.
Refuse to discuss mechanism until the requirement is stated and the Human has acknowledged it (Working style: "Requirement before solution").
Present options in full with tradeoffs, then state which one you lean toward and explain why (Working style: "When offering options").
Exit on Human signal — no automatic detection of "ideation complete".
This skill does NOT write the PLAN file itself. The orchestrator (or Human) dispatches `plan-writer` to transcribe at checkpoints.
When a question genuinely needs external data or a strategic decision worth persisting, invoke `write-bus-input` to drop a RESEARCH or ADVICE Bus file (decision 12).
At Converge close, classify decisions touched per decision 15 (Already-locked / Mechanically-forced / Real-judgement-call) so downstream `[Human]` checkpoints can triage cleanly.
</essential_principles>

<preconditions>
- Running in the parent session (Human-interactive). NOT runnable from a subagent context (no channel back to the Human mid-run).
- A user request or trigger that initiates ideation about a problem to be shaped into a plan.
- Optionally: an existing PLAN draft if iterating on prior ideation (will be re-ideated in place rather than re-authored from scratch).
</preconditions>

<inputs>
- `seed: string` — the user's initial problem statement, question, or topic for ideation.
- `existing_plan_path: string (optional)` — if provided, ideation iterates on the existing PLAN's Objective and Context rather than starting fresh.
</inputs>

<output_schema>
**This skill does not return a structured `<pipeline-result>` block** — it is conversational and runs in the parent session, not as a one-shot subagent. Its "output" is:

1. **Conversational state in the parent** — the Clarify→Survey→Converge arc unfolds as a normal Human/Claude exchange.
2. **Checkpoint side-effects** — at clarify-locked and survey-converged moments, the Human (or orchestrator) dispatches `plan-writer` to write or update a PLAN file in `Bus/`. The skill itself does not write to disk.
3. **Optional Bus inputs** — when the Clarify or Survey phase surfaces a question requiring external data (RESEARCH) or a persisted decision (ADVICE), invoke `write-bus-input` to land the file. The PLAN's `linked_inputs` array gets the filename when `plan-writer` next writes.
4. **Decision classification at Converge close** — produces a triage of decisions touched during ideation:
   - **Already locked:** Human proposed/affirmed.
   - **Mechanically forced:** no alternative.
   - **Real judgement call:** needs Human answer.
   This classification accompanies the final PLAN write so the subsequent `[Human]` design-review checkpoint can surface only the third class (per decision 15).

When invoked from `plan-pipeline`'s `drafting` phase, the orchestrator monitors the conversation for the Human signalling "done ideating" and then advances `pipeline_phase: drafting → drafted`. The skill itself does not flip phase state.
</output_schema>

<exception_conditions>
- Invoked from inside a subagent (no Human channel) — surface and refuse; the orchestrator must dispatch this skill in the parent only.
- Existing PLAN at `existing_plan_path` is unreadable or malformed — surface and ask the Human whether to start fresh or fix the file first.
- Human goes silent across the arc (mid-ideation pause without a returning prompt) — not technically an exception; ideation simply waits for the next Human turn.
</exception_conditions>

**Ideation procedure:** See [workflows/ideate-arc.md](workflows/ideate-arc.md).

<constraints>
- Never propose a mechanism before the requirement is acknowledged. If Clarify hasn't closed, redirect Survey-phase prompting back to Clarify.
- Never present a single option as if it were the only one. If only one option fits, state that explicitly with "no real alternative" and skip Survey.
- Never write to disk yourself. Hand off to `plan-writer` (foreground subagent dispatch) for any PLAN write or update; hand off to `write-bus-input` for RESEARCH/ADVICE writes.
- Never run as a subagent. If invoked via the Agent tool with `subagent_type: ideate-runner` or similar, return `outcome: exception` immediately.
- Never silently exit ideation. The Human signals exit; the skill does not infer it.
</constraints>

<success_criteria>
- All three phases (Clarify, Survey, Converge) were walked, in order, with explicit transitions visible in conversation.
- Requirement was stated and acknowledged before any mechanism was discussed.
- At Survey, ≥2 options were presented in full with tradeoffs and a stated lean.
- At Converge, the chosen approach was sharpened to plan-ready specificity.
- Decisions touched during the arc were classified per decision 15 at Converge close.
- If RESEARCH or ADVICE was needed, the appropriate Bus input was written via `write-bus-input` and the resulting filename is captured for `plan-writer`'s `linked_inputs` array.
- Plan-writer was dispatched at checkpoint moments (clarify-locked, survey-converged) so the PLAN file is durable on disk before ideation closes.
- Skill ran entirely in the parent session.
</success_criteria>
