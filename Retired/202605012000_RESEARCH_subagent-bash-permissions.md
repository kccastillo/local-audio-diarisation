---
title: "Subagent Bash permission resolution — probe findings for F1"
type: bus-research
status: integrated
created: 2026-05-01
created_by: opus
created_month: 202605
log_month: 202605
feeds_plan: 202605011900_PLAN_post-dogfood-fixes.md
---

## Problem

Plan-executor (haiku subagent) gets `Permission('Bash(...)') denied` on every Bash call, even patterns explicitly present in project `.claude/settings.json` `permissions.allow`. Surfaced as F1 in PLAN 202605011900. Blocks every Haiku-executable PLAN flowing through the orchestrator pipeline.

## Documentation findings

- Subagent permission docs say custom subagents have "specific tool access, and **independent permissions**" — *independent*, not inherited.
- Available agent frontmatter fields: `tools`, `disallowedTools`, `permissionMode`, `mcpServers`, `hooks`, `skills`, `model`, `background`. **No `permissions:` or `allow:` field.**
- `tools:` is coarse (tool names only, no Bash patterns like `Bash(mkdir *)`).
- `permissionMode:` accepts `default | acceptEdits | auto | dontAsk | bypassPermissions | plan`.
- Settings precedence (Managed > CLI args > Local > Project > User) is documented but does NOT explicitly say project allow flows into subagents.

## Acknowledged Anthropic defect

- **GitHub issue #37730** ("Subagents (Agent tool) don't inherit permission settings"): exact match. **Closed as not planned.** No workarounds posted.
- Issue #27661 (open feature request, same observation).
- Issue #10906 (built-in Plan agent, same root cause).

Anthropic is not fixing this. Design around it.

## Empirical probe (2026-05-01)

Dispatched plan-executor (haiku, background) with four Bash calls. Each pattern was present in project `.claude/settings.json` `permissions.allow` at probe time:

```
1. bash -c "mkdir -p /tmp/probe-test-dir"   →  Permission denied
2. bash -c "date +%Y-%m-%d"                 →  Permission denied
3. bash -c "echo hello"                     →  Permission denied
4. bash -c "ls /tmp"                        →  Permission denied   ← Bash(ls *) IS in parent allow
```

**All four denied.** `ls` denial confirms the inheritance gap is total — not pattern-shape-dependent. The subagent's Bash tool is blocked at the system level regardless of allowlist contents.

**Implication:** ANY workaround that depends on the subagent inheriting `permissions.allow` is non-viable. Including the previously-listed Option C (bare `Bash` in allowlist).

## Fix-shape options remaining (A and B; C eliminated)

### Option A — `permissionMode: bypassPermissions` on plan-executor variants

Add `permissionMode: bypassPermissions` to `.claude/agents/plan-executor.md`, `plan-executor-sonnet.md`, `plan-executor-opus.md`. One line per file.

**Pros:**
- Documented field; supported by harness.
- Matches trust model — plan-executor only sees PLANs that have already passed sufficiency-auditor and plan-safety-auditor, so the executor is internal trusted infrastructure.
- Minimal blast radius — only the three executor agent files change.
- Other audit/writer/retirer subagents continue to use default permission resolution; they don't need bypass.

**Cons:**
- Bypasses ALL permissions, not just allowlisted ones. A malicious or buggy PLAN slipping past audit could run anything.
- Mitigation: add `disallowedTools:` to deny `WebFetch`, `WebSearch`, etc. so no egress is possible. But that's now a denylist.

**Scope:** 3 agent files; no settings.json changes; no skill workflow changes.

### Option B — Rewrite execute-plan to avoid Bash-pattern dependency

Replace shell calls in `execute-plan` workflow with tool-native equivalents:
- `mkdir` → Write tool (creates parent dirs implicitly).
- `date` → environment-injected timestamp (current_date / current_time variables).
- File existence checks → Read or Glob.
- `acceptance:` shell lines still use Bash — but the harness inheritance gap (above) means even bare `Bash` doesn't inherit, so this approach hits the same wall on `acceptance:`.

**Pros:**
- Sidesteps the bug entirely for filesystem ops.
- Works even if Anthropic never fixes #37730.
- No permissive escape hatches.

**Cons:**
- Large rewrite in `.claude/skills/execute-plan/SKILL.md` and the workflow file.
- The PLAN pipeline has been built around Bash-shaped `acceptance:` lines (`bash -c '...'`) — those are load-bearing in PLANs (parent decision 25 mandates shell-runnable acceptance). Either acceptance: lines also degrade (running outside subagent context, by orchestrator in parent — works, but changes who runs them), or `acceptance:` shape itself must change.
- Touches several files; harder to validate didn't break audit-haiku-safe's expectations.

**Scope:** `execute-plan` SKILL.md + workflow; possibly `_shared/plan-safe.md`; possibly `audit-haiku-safe` if its checks reference Bash; possibly the orchestrator's outcome-verifying phase (which runs `acceptance:` items — but does it run them in subagent or parent context? If parent, no impact).

### Option A+ — Hybrid (refinement)

Apply Option A (`permissionMode: bypassPermissions` + `disallowedTools:` denylist) plus a partial Option B move: have the orchestrator's outcome-verifying phase run `acceptance:` shell commands in the **parent** context (where allowlist works), not via a subagent. Decouples PLAN's `acceptance:` syntax from the executor's permission scope.

**Pros:** keeps shell-shaped `acceptance:` while avoiding the subagent inheritance gap for the verification phase too. Clean separation: executor does mechanical work (now bypass-permitted); orchestrator runs `verify:`/`acceptance:` (parent permission context).

**Cons:** requires confirming the outcome-verifying phase already runs in parent (likely does per current orchestrator design).

## Recommendation

**Lean: A+ hybrid.** `permissionMode: bypassPermissions` on the three plan-executor variants + `disallowedTools: [WebFetch, WebSearch, ...]` to scope egress; outcome-verifying phase confirmed to run `acceptance:` in parent context (already the case per current dispatch.md).

Rationale: minimal change, maximum compatibility with existing PLAN conventions, matches the trust model, doesn't depend on a defect being fixed.

Trade-off accepted: if a malicious PLAN passes both audits, plan-executor would happily run it. This is the same trade-off any orchestrator with a trusted internal executor faces; we mitigate by keeping audits tight, not by adding bash-allowlist friction that doesn't actually work anyway.

## Next steps for PLAN 1900

- Phase A Step 2 (Survey): present A vs B vs A+ to Human. Decide.
- Phase A Step 3 (Converge): document chosen approach in 1900's Context.
- Phase B Step 4: implement.
- Phase B Step 5: smoke-test.

## Sources

- [Create custom subagents — Claude Code Docs](https://code.claude.com/docs/en/sub-agents)
- [Claude Code Settings](https://code.claude.com/docs/en/settings)
- [Issue #37730 — Subagents don't inherit permission settings](https://github.com/anthropics/claude-code/issues/37730)
- [Issue #27661 — Subagents should inherit parent session hooks and permission rules](https://github.com/anthropics/claude-code/issues/27661)
- [Issue #10906 — Built-in Plan agent ignores parent settings.json permissions](https://github.com/anthropics/claude-code/issues/10906)
