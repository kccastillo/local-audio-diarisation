---
title: "Rewrite ROADMAP.md and consolidate architecture into single source"
type: bus-plan
status: ready
assigned_to: haiku
priority: medium
created: 2026-04-29
created_by: haiku
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

Rewrite ROADMAP.md to drop the v1-vs-v2 framing, restructure into Phase 1 ("make it work") and Phase 2 ("improve"), refine individual threads based on the critique, and fold the architectural-principles content out of CLAUDE.md so there is a single source of truth in ROADMAP.md.

## Context

Ken reviewed the current ROADMAP.md and accepted the critique developed in conversation. Key decisions from that exchange:
- No benchmarking against v1 — just build something that works, then improve it.
- Architectural principles currently duplicated in CLAUDE.md should live in ROADMAP.md, with CLAUDE.md pointing to them.
- Gender-prefix speaker labels (M01/F01) considered and dropped.
- ASR backend desk research already completed: faster-whisper-large-v3 is the recommended default for Ken's audio profile (English, AU + Indian-English + US accents, Teams + phone recordings).

This is a documentation-only plan — no code changes.

## Steps

1. Rewrite ROADMAP.md: drop v1-vs-v2 framing entirely, including the "v2 rewrite" mission language and the `v1-final` tag as a load-bearing milestone.

2. Restructure ROADMAP.md threads into two sections: **Phase 1 — make it work** and **Phase 2 — improve**.

3. Promote Q1 (WhisperX wholesale vs piecewise improvements) to a top-level gating decision at the start of Phase 1, since it determines what "make something that works" looks like. Expand options from binary to three: (a) WhisperX wholesale, (b) piecewise with native Whisper word timestamps, (c) piecewise with our own faster-whisper + ctc-forced-aligner.

4. Pin T01 specifically as "swap to faster-whisper-large-v3" (not generic backend swap). Add note: large-v3-turbo is a fallback if speed is insufficient, with documented Indian-English regression risk.

5. Reframe T03 (word-level speaker attribution) to try Whisper native word timestamps first (`word_timestamps=True`) before committing to forced alignment with wav2vec2 / ctc-forced-aligner.

6. Re-justify or demote T05 (long-audio chunking) — note that Whisper processes in 30s windows internally, so the original VRAM premise is partially wrong; the real motivation is failure isolation and progress visibility on long files.

7. Re-justify or demote T06 (batch mode) — note that batch mode contradicts the low-VRAM ethos for an 8GB target with a single user; load/unload overhead amortised over N files vs holding intermediate state for all files.

8. Add a new top section to ROADMAP.md titled "Working architecture" listing current architectural principles in force, with each principle annotated by which thread (if any) challenges it. Principles to include and their challenge status:
   - Sequential load/unload of models — challenged conditionally; right for 8GB today, may not hold after T01.
   - Singleton ConfigManager — challenged; pass config explicitly to processors instead.
   - BaseProcessor inheritance with load_model/unload_model/process — challenged; processors are stateless functions wearing OOP clothing, a context manager is closer to the truth.
   - Five-stage pipeline with separate VAD — challenged by T02; pipeline is four stages, VAD redundant with pyannote 3.1.
   - Speaker attribution by segment-overlap dominance — challenged by T03.
   - Immutable TranscriptionSegment dataclass — kept; cheap, prevents accidental mutation.
   - Memory monitor as cross-cutting infra — kept conditionally; useful while VRAM pressure is the design constraint.

9. Trim CLAUDE.md "Architecture & Core Design" section to a one-line pointer: `Architecture: see ROADMAP.md § Working architecture`. Remove the duplicated content so ROADMAP.md is the single source of truth.

10. Drop the gender-prefix speaker label idea (T07) — Ken decided against it. Do not add this thread to the rewritten ROADMAP.md.

11. Mark exactly one thread in the rewritten ROADMAP.md with `Status: next` to indicate the work to start tomorrow.

12. Update `last_updated` frontmatter in both ROADMAP.md and CLAUDE.md to today's date.

## Verification

- [ ] ROADMAP.md no longer references "v1 baseline", "v2 rewrite", or `v1-final` as a load-bearing milestone.
- [ ] ROADMAP.md has a Phase 1 / Phase 2 structure.
- [ ] ROADMAP.md has a "Working architecture" top section listing principles with challenge annotations.
- [ ] ROADMAP.md has Q1 expanded to three options (WhisperX wholesale, native-piecewise, our-own-piecewise).
- [ ] ROADMAP.md T01 explicitly names `faster-whisper-large-v3`.
- [ ] ROADMAP.md T03 mentions trying Whisper native word timestamps before forced alignment.
- [ ] ROADMAP.md T05 and T06 either re-justified with corrected premises or demoted to Phase 2 with a note.
- [ ] ROADMAP.md contains no T07 or gender-prefix content.
- [ ] Exactly one thread in ROADMAP.md has `Status: next`.
- [ ] CLAUDE.md "Architecture & Core Design" section is replaced by a single-line pointer to ROADMAP.md.
- [ ] CLAUDE.md no longer duplicates architecture content (grep for "load_model", "BaseProcessor", "singleton" in CLAUDE.md returns no matches in the architecture section).
- [ ] `last_updated` frontmatter on ROADMAP.md and CLAUDE.md is `2026-04-29`.
- [ ] No code changes outside `.md` files (`git diff --name-only` shows only markdown).

## Executor Notes
*Populated by Haiku after execution via `execute-plan`. Leave blank.*

**Executed:**
**Outcome:** done | partially-complete | blocked | needs-revision
**What was done:**
**Blockers (if any):**
**Files modified:**
