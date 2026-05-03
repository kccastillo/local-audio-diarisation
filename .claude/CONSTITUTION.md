# Constitution

Standing rules every Claude session in this project must obey. Auto-loads at session start alongside CLAUDE.md.

## Section load classification

Tagged here so a future split (when this file passes ~200 lines) is mechanical. Until then everything auto-loads together.

- **HOT** (must auto-load — governs in-flight behaviour every session):
  - Domain neutrality
  - CLAUDE.md content rule
  - Token economy: compress hot, write plain cold
  - Communication with the operator
  - Discussion-vs-work-order
  - Review shape
  - Bash compounds
  - Decision autonomy and sounding-board delegation
  - Subagent capability boundaries
  - Context-rot recovery protocol
  - Memory and persistence
  - Plan lifecycle and halt conditions
  - Skill registry
- **COLD** (rationale, source-of-truth pointers, expanded examples — readable on demand): none yet. When sections grow rationale tails too long for hot-load, move the tail into `references/` and leave the directive line in the hot section.

## Domain neutrality

**IMPORTANT:** The harness must work for any domain — code, writing, BA, data, research, anything. Skill descriptions, templates, examples, and verification snippets must be **domain-neutral** by default. Do not overfit to code (no `pytest`, `npm`, `python -c` examples in templates), to BA (no `Wiki/`, `atomise`, `librarian`), or to any other single domain.

Concrete rules:
- **Templates** (`plan-template.md`, `research-template.md`, `advice-template.md`) use placeholder verification examples (`<shell command that exercises the deliverable>`) or domain-neutral concrete predicates (`test -f path`, `grep -q "expected" path`). Per-domain test invocations can be mentioned as bracketed alternatives, never as the prescribed default.
- **Skill SKILL.md examples** describe the shape of the operation, not a domain-specific instance. If a concrete example is needed, prefer harness-self-referential ones (e.g. "Edit ROADMAP.md to add a thread") over single-domain ones.
- **Optional domain-specific skills** (e.g. `debug-like-expert` for code) live as installable add-ons per project, not as part of the base harness deployment via `initiate-harness`.
- **The cleanup test:** if a skill or template would feel awkward when applied to a writing project, a research project, or a BA project, it's overfit. Generalise.

## CLAUDE.md content rule

**IMPORTANT:** CLAUDE.md contains directives only — no explanations, no examples, no justifications, no rationale prose. If a directive needs context to be applied correctly, the context lives in this file or in a referenced sub-doc; CLAUDE.md gets only the terse imperative plus a pointer for further reading. The operator scans CLAUDE.md frequently; every word that isn't a directive is a tax on his attention.

## Token economy: compress hot, write plain cold

**IMPORTANT:** Compression discipline is surface-dependent. The principle: **compress what's read every turn; write plainly what's read on demand.** Mis-applying compression to on-demand surfaces costs more than it saves.

Three surface classes:

- **Hot surfaces** (always-on context — CLAUDE.md, this file once auto-loaded by SessionStart hook, anything injected at session start): every word is paid every interaction. Tighten ruthlessly — directives only, no padding, no rationale prose. The CLAUDE.md content rule above is one expression of this.
- **Runtime traffic** (dispatch prompts to subagents, parsed agent results, agent-to-agent handoffs): high volume across a session, not human-facing. Established project shorthand and bare references permitted per the Subagent traffic rule below; novel encoding still default-OFF for the silent-misread reason given there.
- **Cold surfaces** (`.claude/skills/*/SKILL.md`, `.claude/agents/*.md`, `workflows/`, `references/`, `templates/`): read on demand at invocation, audited by humans, depended on for correct rule application by Haiku doing mechanical work. **Stay in plain English.** Cut padding and redundancy as a clarity goal, not as a token-economy goal. Do not strip articles, abbreviate identifiers, or compress prose for token savings on these surfaces. The costs — audit-time readability for the operator, model interpretation drift (text further from the training distribution costs attention budget elsewhere), rule-fidelity risk for Haiku doing mechanical work — outweigh the ~15-20% input-token saving per invocation.

This rule and the Subagent traffic shorthand rule below are two halves of the same principle: compression where the surface is high-volume and not human-read; plain prose where the surface is read on demand and depended on for correctness. They reinforce each other rather than being independent decisions.

## Communication with the operator

**IMPORTANT:** Translate jargon and identifiers into plain language at roughly a 9th grade reading level whenever talking to the operator (Ken). He does not have my immediate session-recall ability — every numeric ID, finding code, decision number, or skill name is opaque to him without a gloss.

- **PLAN files at decision moments** — when asking for a greenlight, judgement call, sequencing choice, or halt-or-proceed, pair the timestamp ID with a 2-5 word handle: `PLAN 1945 (CLAUDE.md prune)`, `PLAN 2000 (executor capability boundary)`. Status-only mentions can use bare IDs.
- **Decision references / F-numbers / blocker codes** — gloss at first mention per response. "Decision 15 (the triage rule that splits items into already-locked / mechanically-forced / real-judgement)", "F1 Option C (the fix where we stripped Bash from the executor)".
- **Skill and agent names** — gloss with a parenthetical at first mention in a status update; subsequent references in the same response can use the bare name.
- **Internal thinking blocks**: shorthand encouraged. Token economy matters; the translation rule is for output, not internals.
- **Subagent traffic** (dispatch prompts to agents, parsing agent results, agent-to-agent handoffs): two distinct cases, treated differently.
    - **De-glossing established project references**: explicitly permitted. Bare PLAN IDs (`202605020100`), bare decision numbers (`Decision 15`), bare F-numbers, bare blocker codes (`B2`, `W3`), bare skill/agent names (`plan-writer`, `audit-haiku-safe`) — drop the operator-facing parenthetical gloss when sending to or receiving from agents. These resolve through the shared project substrate (Bus/, .claude/skills/, .claude/agents/, MAINTENANCE.md, ROADMAP.md); the agent has the same ground truth the orchestrator does.
    - **Novel encoding / invented shorthand / neuralese**: default is terse English or structured JSON. Inventing notation for a specific dispatch is permitted only when the orchestrator can state, in the dispatch prompt itself, what the encoding means. If the encoding needs explaining, the explanation lives in the prompt — at which point the token saving has usually evaporated and plain language wins. The failure mode this rule closes: subagent confidently produces output from a misread of invented notation, and the misread isn't visible at the orchestrator level.

The principle: optimise reading effort for the operator in user-facing surfaces; optimise tokens everywhere else (thinking, agent prompts, agent results) — but only by removing redundant glosses, not by inventing private codes. The two surfaces don't conflict.

## Discussion-vs-work-order

**IMPORTANT:** Treat phrasings like "X is broken", "we should Y", "how about Z", or "what if we did W" as **discussion openers**, not work orders. Pause. Ask whether Ken wants you to act, propose, or just think out loud. Acting on a thought-out-loud as if it were a directive wastes effort and can do harm.

Concrete cues that something is a discussion opener, not a work order:
- Hypothetical framing (`what if`, `how about`, `we could`).
- Diagnostic framing without an explicit ask (`X is broken`, `Y feels off`).
- Open-ended exploratory verbs (`consider`, `think about`, `look into`).

Concrete cues that something **is** a work order:
- Imperative verbs with a clear object (`change X to Y`, `delete Z`, `commit and push`).
- Explicit go-aheads after a prior options-and-recommendation exchange (`go with option B`, `do it`, `proceed`).
- Standing instructions in CLAUDE.md or memory.

When in genuine doubt, ask one short clarifying question. Cost of a clarification is low; cost of a wrong action is high.

## Review shape

**IMPORTANT:** Reviews (of code, plans, documents, anything) follow a fixed five-element shape. Do not skip elements; do not reorder.

1. **Verification preamble** — one short paragraph naming what was checked against (the actual files/specs/tests, not just the document under review). Establishes ground truth.
2. **One-line overall verdict** — ship / don't ship / ship-with-fixes, in one sentence.
3. **Priority-ordered numbered punch list** — blockers first, then majors, then nits. Each item: location (`file.py:42`), one-sentence problem, one-sentence proposed fix.
4. **"Not blockers" subgroup** — items observed but explicitly classified as out-of-scope or below the bar for this review. Prevents scope creep without losing the observation.
5. **Net verdict** — what's ready to land, what needs fixing, suggested next step.

The shape exists because reviews without a verification preamble drift into impressionistic feedback, and reviews without a "Not blockers" subgroup smuggle scope creep into the punch list.

## Bash compounds

**IMPORTANT:** Avoid combining commands in a single Bash call via `&&`, `;`, `||`, or pipes when the same work can be done as separate calls. One operation per call.

**Why.** Two failure modes:
1. **PowerShell parser quirks.** This project's primary shell is PowerShell (Windows). Compound POSIX-style chains parse unpredictably; `&&` is unsupported in PowerShell 5.1.
2. **Hook brittleness.** `.claude/` hooks (PreToolUse, PostToolUse, telemetry logging) inspect the Bash command string. Compound commands either get logged ambiguously or trip permission prompts that single calls would avoid.

**Allowed.** Pipes inside a single semantic operation (`cat file | grep pattern`) — that's one operation. Heredocs for multi-line strings to a single command (e.g. `git commit -m "$(cat <<EOF ... EOF)"`).

**Forbidden by default.** Chains of distinct semantic operations (`pytest && git add . && git commit`). Split into separate calls — they parallelise where independent and give clearer failure attribution where sequential.

## Decision autonomy

**IMPORTANT:** Default behaviour is **decide and proceed**. When facing a decision about whether to surface to the operator vs decide autonomously, apply Decision-15 triage and act per `Skill('decide-and-proceed')` — the skill encodes the triage logic, the practical-first-conservative-second pattern, the sounding-board delegation rules (strategic→Opus, tactical→Sonnet), and the post-decision reporting shape. Ken's standing direction (2026-05-02): "happy for you to make the edits within this project as you see fit ... don't ask for my input unless absolutely necessary."

The skill sits in `.claude/skills/decide-and-proceed/SKILL.md` (migrated out of this file 2026-05-02 because constitution prose alone wasn't reliably activating the rule mid-session — skill-style trigger phrasing in the description performs the activation that always-on prose missed).

## Subagent capability boundaries

See `.claude/skills/_shared/plan-safe.md` § Executor capability boundaries. Cite-only — do not re-state.

## Context-rot recovery protocol

**IMPORTANT:** If the session starts feeling drifty (instructions forgotten, prior decisions re-litigated, output style sliding, repeated mistakes), do NOT attempt self-recovery by re-reading files or summarising. Self-recovery from rot tends to compound the problem.

**Required action:** full stop + surface to the operator. Tell Ken plainly that the session feels rotted and ask him to run `/clear` to reset context. He will run `/clear` (a Claude Code slash command that clears the conversation) and start fresh. Standing rules in CLAUDE.md, this file, and `memory/` will reload automatically; in-flight PLAN state is durable on disk via frontmatter.

The harness is designed so that any in-flight work — PLANs in any phase, audit_state, executor outcomes, monthly LOG — survives a `/clear`. Re-entry is a `Skill("plan-pipeline")` invocation against the live PLAN file; the orchestrator reads disk and resumes from the recorded phase. There is no in-memory state worth preserving past detected rot.

## Memory and persistence

- **Cross-session memory** lives at `memory/MEMORY.md` (auto-loaded). Save user preferences, project facts, feedback corrections there — not chat-only.
- **Skills and this file** are codified standing rules. Anything that should apply across sessions becomes a skill or a rule, not a memory crutch.
- **Episodic / month-bound observations** live in the Lessons Learned section of the monthly LOG; the `lessons-learned` skill curates them forward at rollover.

## Maintenance loop

`MAINTENANCE.md` is the harness self-maintenance record. It holds the drift queue (observed potholes, deferred items without a ROADMAP thread), codified lessons, and a dated change log.

The maintenance loop runs via the `maintenance` skill and `maintenance-agent`:
- **On-demand:** invoke `Skill('maintenance', 'audit')` or dispatch `maintenance-agent` to run a read-only drift audit and receive structured findings.
- **Monthly:** `Bus/202605020210_PLAN_RECUR-monthly-maintenance-sweep.md` triggers the agent at the start of each month alongside the LOG creation.

After an audit run, the parent session applies findings to MAINTENANCE.md by invoking `Skill('maintenance', 'update-log')` or editing directly from the `<pipeline-result>` payload. The maintenance-agent never writes files — the structural split is enforced by `disallowedTools: [Edit]`.

## Three-record split for harness state

The harness maintains three distinct state records. Never confuse their roles:

- **ROADMAP.md** — strategic threads (multi-month, sticky). A thread lives here from identification through to closure. ROADMAP is canonical for thread status; other files should not duplicate thread definitions.
- **MAINTENANCE.md** — drift queue: one-off detections, observed potholes, and sweep findings that don't yet have a PLAN or a ROADMAP thread home. Items in MAINTENANCE.md "Deferred items" that appear as ROADMAP threads (e.g. T04, T05, T06, T07, T08, T10, T12) should be removed from MAINTENANCE.md; ROADMAP is canonical for those. MAINTENANCE.md "Deferred items" is for queue-class drift findings without a strategic-thread home.
- **Monthly LOG** (`Bus/{YYYYMM}010000_LOG_{YYYYMM}.md`) — month-bound execution record: PLANs executed this month, lessons learned, recurring task tracker. Scope is strictly the current month; prior months roll forward via `lessons-learned` curate-forward at rollover.

**The split rule:** a strategic thread (multi-PLAN, multi-month, has a Done-when criterion) belongs in ROADMAP. A one-off pothole or drift detection belongs in MAINTENANCE.md until it gets a PLAN. Month-bound execution state belongs in the LOG. If you're unsure, ask: will this be relevant next month? Yes → ROADMAP. Maybe → MAINTENANCE.md. No → LOG.

## Always-loaded budget

Combined `CLAUDE.md` + `.claude/CONSTITUTION.md` + `MEMORY.md` index ≤ 2000 tokens. Audited monthly by `maintain-claude-md` (see `references/audit-checklist.md` § J). On warn: prune or migrate content out of the always-loaded surface. The system-reminder skill listing is budgeted separately via `skillListingMaxDescChars` and `skillOverrides` (see Campaign C1's R5.3 child for skill demotions). Baseline snapshot: `.claude/research/baseline-always-loaded-tokens-202605.md`.

## Plan lifecycle and halt conditions

PLAN status lifecycle: `ready → in-progress → done | partially-complete | blocked | needs-revision`

- `ready` — transcribed, not yet started.
- `in-progress` — execution has started.
- `done` — all verification criteria pass. Terminal.
- `partially-complete` — some steps done, others blocked or deferred. Terminal for this cycle.
- `blocked` — cannot proceed; `blocked_by` holds the reason. Cleared by `write-bus-input` when resolving input lands, or manually by Ken.
- `needs-revision` — plan itself is faulty; halt and surface to Ken.

Execution phases: Draft → Transcribe → Review (optional) → Revise (if needed) → Execute → Closeout → Post-execution review.

**Halt conditions:**
- **Interrupted runs:** capture description as a hiccup entry in `.claude/skills/_hiccups.md` before resuming.
- **Ambiguity:** if a step is ambiguous, unsafe, or marked [Ken]: halt, set `status: needs-revision`, surface to Ken. Do not improvise.
- **Verification failure:** if any verification check fails, halt before commit; do not tick boxes that have not been verified.

## Skill registry

Skills live in `.claude/skills/<name>/SKILL.md`. Invoke via `Skill("name")`. Full authoring guide: see `.claude/skills/create-skill/references/` (skill structure, YAML frontmatter, common patterns).

| Skill | Purpose |
|---|---|
| `initiate-harness` | Bootstrap a fresh project with the Bus/PLAN harness |
| `create-skill` | Expert guidance for creating and refining skills |
| `ideate` | Three-phase ideation arc for shaping a problem into a plan-ready idea |
| `audit-sufficiency` | Conceptual plan audit (seven lenses) |
| `audit-haiku-safe` | Mechanical plan-safety audit (concrete, atomic, unambiguous, safe, testable) |
| `plan-pipeline` | End-to-end planning orchestrator (drafting through retired) |
| `write-bus-plan` | Transcribe plans to Bus/ files; manage monthly LOG and status tables |
| `write-bus-input` | Write RESEARCH/ADVICE files to Bus/; unblock plans waiting on input |
| `execute-plan` | Execute PLAN steps in order; populate Executor Notes; update LOG; commit + push |
| `maintain-claude-md` | Audit CLAUDE.md and CONSTITUTION.md; propose adds/prunes |
| `maintain-skills` | Audit the skill ecosystem for context rot; propose adds/prunes |
| `maintain-log` | Audit the monthly Bus LOG; propose adds/prunes |
| `retire` | Move files to gitignored Retired/ folder when no longer needed |
| `lessons-learned` | Capture decision rationales and learnings from completed work |
| `maintenance` | Harness drift audit (seven classes) + MAINTENANCE.md updates |
| `decide-and-proceed` | Autonomous decision-making — decide and proceed unless operator input genuinely required |
| `architecture-review` | Harness architectural review — snapshot → literature → synthesis → decomposition → reconciliation chain |
