---
name: initiate-harness
description: Bootstrap a fresh project with the Bus/PLAN harness. Run this from the root of a new project that has had only `.claude/` selectively pulled in from the harness repo. The skill creates the missing directories and root files (Bus/, Retired/, AGENT_RULES.md, CLAUDE.md, ROADMAP.md, current-month LOG, .gitignore entries, settings.json fallback) from templates shipped inside this skill. Idempotent — checks each item and skips if already present. Trigger phrases: "initiate the harness", "bootstrap the harness", "set up the harness here", "transplant complete, finish setup".
---

<objective>
Take a project that contains only the transplanted `.claude/` directory and bring it up to a working harness state — Bus/ with the current month's LOG, Retired/, root files (AGENT_RULES.md, CLAUDE.md, ROADMAP.md), .gitignore entries, settings.json fallback. All content comes from templates inside this skill so the bootstrap is self-contained and offline.
</objective>

<preconditions>
- cwd is the root of the target project
- `.claude/skills/initiate-harness/templates/` exists and contains the template files
- `.claude/scripts/log-skill-usage.py` exists if the settings.json hook is to function (ships inside `.claude/` from the same selective pull)
</preconditions>

<inputs>
- None. Reads from cwd. Uses today's date for the current-month LOG filename.
</inputs>

**Bootstrap procedure:** See [workflows/bootstrap.md](workflows/bootstrap.md)

<essential_principles>
Idempotent. Each item is checked before creation; existing files are never overwritten.
Templates only. Never invent content — copy from `templates/` verbatim, substituting only the date/month placeholders.
Project-specific placeholders stay as `<!-- PROJECT: ... -->` markers in the new CLAUDE.md and ROADMAP.md. Do not fill them in — the user does that after bootstrap.
Report what was created vs skipped, and list the placeholder sections the user still needs to fill in.
</essential_principles>

<constraints>
- Never overwrite an existing root file (CLAUDE.md, AGENT_RULES.md, ROADMAP.md, settings.json) — skip if present.
- Never modify an existing .gitignore line — only append missing harness entries.
- Never create a LOG if one already exists for the current month.
- Never write outside cwd.
- Never invoke other skills (write-bus-plan etc.) — those depend on the harness being already in place. This skill is the bootstrap that precedes them.
</constraints>

<success_criteria>
- `Bus/` directory exists and contains a `{YYYYMM}010000_LOG_{YYYYMM}.md` for the current month
- `Retired/` directory exists
- `.gitignore` exists and contains the harness entries (Retired/, .claude/settings.local.json, .claude/_skill_usage.jsonl)
- `AGENT_RULES.md`, `CLAUDE.md`, `ROADMAP.md` exist at project root
- `.claude/settings.json` exists
- A summary report is returned listing every action taken (created, skipped, appended) and every `<!-- PROJECT: ... -->` placeholder the user still needs to fill in
</success_criteria>

<exception_conditions>
- cwd has no `.claude/skills/initiate-harness/templates/` — the templates directory is missing from the selective pull. Halt and report.
- cwd is not writable — halt and report.
</exception_conditions>