# Ideate — Three-Phase Conversational Arc

This is conversational guidance, not a procedural workflow. The skill runs in the parent session and walks the Human through three phases. Each phase has explicit entry and exit conditions.

## Phase 1 — Clarify

**Goal:** establish the requirement before any mechanism is discussed.

**Entry condition:** Human has invoked ideation with a seed problem, question, or topic.

**What Clarify does:**
- Ask clarifying questions about the **what** and the **why** — never the how.
- Reflect the problem back to the Human in your own words. Confirm understanding before proceeding.
- Surface implicit constraints, scope assumptions, and unstated requirements.
- If a question can't be answered without external data (e.g. "what's our current API latency?", "how many users does this affect?") → invoke `write-bus-input` to drop a RESEARCH file describing what's needed. The PLAN that eventually gets written will reference it via `linked_inputs`. If the data is critical-path, the resulting PLAN may go to `status: blocked, blocked_by: <RESEARCH filename>` until the data lands.
- If a strategic decision worth persisting comes up (e.g. "we're prioritising correctness over speed", "we won't support iOS for v1") → invoke `write-bus-input` to drop an ADVICE file capturing the decision. Decisions captured this way are durable across conversations.

**Exit condition:** the Human explicitly acknowledges the stated requirement (e.g. "yes that's right", "correct", "exactly", or a refinement that you re-state and they re-acknowledge).

**Checkpoint:** at Clarify exit, dispatch `plan-writer` (foreground subagent) to draft the PLAN's Objective and Context sections. PLAN file gets created with `pipeline_phase: drafting`. The orchestrator handles this dispatch; if running in pure parent-only mode, the Human can manually run `Skill("write-bus-plan")` with the clarified content.

**Anti-pattern:** if you find yourself proposing solutions during Clarify ("you could fix this by..." / "one approach would be..."), stop. Return to clarifying the requirement.

## Phase 2 — Survey

**Goal:** lay out the solution space — at least 2 options — in full, with tradeoffs, and state which you lean toward.

**Entry condition:** Clarify has closed (Human acknowledged the requirement).

**What Survey does:**
- Identify ≥2 distinct approaches to the requirement. If only one approach fits, **say so explicitly** ("there's no real alternative because X") and skip to Converge.
- For each option: describe it in full (not abbreviated), name its tradeoffs honestly (cost, complexity, what it gives up, what it locks in), and assess fit for the requirement.
- After presenting options, **state which one you lean toward and explain why** (Working style: "When offering options, describe each in full, then say which you lean toward and explain why").
- Invite Human input: do they prefer differently? do they see an option you missed?
- If a Survey question genuinely needs external data → invoke `write-bus-input` for RESEARCH (same pattern as Clarify).
- If a strategic decision is being made (e.g. "we'll go with option B because it's reversible") → invoke `write-bus-input` for ADVICE.

**Exit condition:** the Human picks an option (or instructs to combine, or proposes a new option that you re-survey). The choice is explicit.

**Checkpoint:** at Survey exit, dispatch `plan-writer` to update the PLAN file with the chosen approach in Context. PLAN remains at `pipeline_phase: drafting` until Converge closes.

**Anti-pattern:** if you present only one option and recommend it as if there was no alternative, you've skipped Survey. Go back, surface the alternatives even if to dismiss them.

## Phase 3 — Converge

**Goal:** sharpen the chosen approach to plan-ready specificity.

**Entry condition:** Survey has closed (Human chose an option).

**What Converge does:**
- Refine the chosen approach into concrete steps, decisions, and acceptance criteria.
- Surface remaining open questions — anything the Human still needs to decide before the PLAN is ready to write.
- Walk the Human through the proposed Steps section (verbally, not yet written to file). Get their input on each major step.
- Define what "done" looks like — what observable outcomes prove the deliverable matches the spec? These become `acceptance:` items in the PLAN's Verification section (per parent decision 25).

**Decision-triage at Converge close (per parent decision 15):** before writing the final PLAN, classify every design decision touched during this ideation arc:
- **Already locked** — Human directly proposed it OR explicitly affirmed it during Clarify/Survey/Converge.
- **Mechanically forced** — no meaningful alternative exists; downstream consequence of an earlier locked decision.
- **Real judgement call** — Human's preference still matters; alternatives exist.

The classification accompanies the final `plan-writer` dispatch — it lands in the PLAN's `## Design Decisions Classification` section (per the updated `plan-template.md`), with three structured subsections: Already-locked / Mechanically-forced / Real-judgement-calls. This way the subsequent `[Human — design-review checkpoint]` step in the PLAN can present only the Real-judgement-call class to the Human, not all 16+ decisions.

**Exit condition:** the Human signals "done — write the PLAN" (or equivalent: "ok, transcribe this", "let's write it up", "ready to plan it"). No automatic detection.

**Final checkpoint:** dispatch `plan-writer` to finalise the PLAN file with all sections populated (Objective, Context including decision classification, Steps, Verification with verify:/acceptance: items per decision 25). Flip `pipeline_phase: drafting → drafted`.

The orchestrator's `drafted` phase then takes over — dispatching `sufficiency-auditor` and `plan-safety-auditor` in the audit loop (per parent decisions 21).

**Anti-pattern:** if you fail to surface decisions for triage at Converge close, the Human will be asked to re-confirm everything at the design-review checkpoint, wasting their attention budget (the problem decision 15 was created to solve).

## Operating notes

- **Token economy:** ideation is conversational; expect many turns. Don't try to compress to one big "here's the whole arc" message — that defeats the purpose of Clarify→Survey→Converge as a discipline.
- **Iteration is fine:** if Converge surfaces a problem with the chosen approach, return to Survey for that specific aspect. Don't restart from Clarify unless the requirement itself was wrong.
- **`plan-writer` dispatches happen in foreground** — they're fast (seconds) and the parent waits for them to return before continuing the conversation. Per parent decision 18.
- **`write-bus-input` runs in the parent session** — it's interactive at the metadata level (asks the Human for type, slug, etc.) but quick.
- **Subagent boundary is hard:** if anything in this arc requires a subagent dispatch (write-bus-plan, write-bus-input, plan-writer), the dispatch itself is foreground/synchronous from the parent's perspective. The arc itself never moves to a subagent.
