# Audit Checklist

Run every check on every audit. Group findings by severity: **blocker** (hard-cap breach, dead reference, anti-pattern) → **warn** (soft-cap, structural risk) → **suggestion** (progressive-disclosure opportunity).

## A. Size and budget
- A1. Line count vs. soft cap (150). Flag if over.
- A2. Line count vs. hard cap (300). Block adds if over.
- A3. Estimated instruction count (count each bullet, table row, and emphasised rule). Flag if approaching 100 (camelCase ceiling for the user-budget portion of the model's instruction window).

## B. Trinity present (top of file)
- B1. Project oneliner — single paragraph, names framework + tech stack + domain.
- B2. Key Commands — only the most-frequent (≤3 commands inline; rest as pointer).
- B3. Caveats — non-obvious project quirks (e.g., "do not modify schema.prisma directly").
Missing any: blocker.

## C. Instruction weighting (Lost-in-the-Middle)
- C1. Most critical rules in the top ~30 lines or bottom ~20 lines, not buried mid-file.
- C2. IMPORTANT / MUST / NEVER markers used on load-bearing rules.
- C3. Pointer to CONTEXT_CONSTITUTION.md sits in top 10 lines.
Buried critical content: blocker.

## D. Anti-pattern detection
See [anti-patterns.md](anti-patterns.md). Flag every match.

## E. Reference health
- E1. Every pointer to `.claude/references/*.md`, `AGENT_RULES.md`, or other in-repo files resolves to an existing file.
- E2. Every pointer to a skill (`Skill("name")` or `.claude/skills/name/`) resolves.
- E3. No pointers to retired files (check `Retired/` for matching basenames).
Dead reference: blocker.

## F. Progressive-disclosure opportunities
- F1. Any inline section ≥20 lines that documents codebase-derivable facts (architecture, schemas, command lists) → suggest moving to `.claude/references/`.
- F2. Any inline section ≥10 lines that's only relevant for specific file types → suggest path-based rule loading (`.claude/rules/<pattern>.md`).
- F3. Any large block of caveats specific to one subsystem → suggest a subsystem-specific reference.

## G. Static-maintenance (drift)
- G1. Tool/library version numbers inline — flag for verification against `requirements.txt` / `package.json`.
- G2. File paths inline — verify each exists; dead path is blocker.
- G3. References to people, projects, or external systems — flag for review (cannot auto-verify).

## H. Context-rot positioning
- H1. Does CLAUDE.md document the scratchpad pattern (Bus/ for plans, memory for cross-session)? Required.
- H2. Does CLAUDE.md document subagent triggers (when to delegate vs. handle inline)? Required.
- H3. Does CLAUDE.md establish content labelling (e.g., naming external-source content distinctly)? Suggestion if missing.
- H4. Does CONTEXT_CONSTITUTION.md exist and is pointed to from the top of CLAUDE.md? Required.

## I. Constitution-specific (only when auditing CONTEXT_CONSTITUTION.md)
- I1. All four rot modes named (Poisoning / Distraction / Confusion / Clash).
- I2. All four fixes named (Write / Select / Compress / Isolate).
- I3. Project-specific imperative rules present (≥6 rules tied to actual project workflows).
- I4. Recovery protocol on rot detection documented.
