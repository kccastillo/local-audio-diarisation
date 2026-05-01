# Bootstrap procedure

Run from the root of the target project. Each step is idempotent — check first, act only if missing.

## 1. Verify templates are present

Check `.claude/skills/initiate-harness/templates/` exists. If missing, halt with the message: `templates/ missing — selective pull of .claude/ was incomplete; cannot bootstrap.`

Required template files:
- `AGENT_RULES.md`
- `CLAUDE.md`
- `ROADMAP.md`
- `LOG_template.md`
- `settings.json`
- `gitignore_entries.txt`

If any are missing, halt and list which ones.

## 2. Ensure directories

For each of `Bus/`, `Retired/`:
- If exists: skip (note in report).
- Else: create.

## 3. Ensure .gitignore entries

Read `templates/gitignore_entries.txt` — one entry per line. For each line:
- If `.gitignore` does not exist: create it with all entries.
- Else: read existing lines; append only entries not already present (exact-match, ignoring whitespace).

## 4. Ensure current-month LOG

Compute `YYYYMM` from today's date. Target path: `Bus/{YYYYMM}010000_LOG_{YYYYMM}.md`.

- If exists: skip.
- Else: copy `templates/LOG_template.md`, substituting:
  - `{YYYY-MM}` → today's year-month (e.g. `2026-05`)
  - `{YYYY-MM-DD}` → today's date (e.g. `2026-05-01`)

## 5. Ensure root files

For each of `AGENT_RULES.md`, `CLAUDE.md`, `ROADMAP.md`:
- If exists: skip (note in report).
- Else: copy verbatim from `templates/{filename}` to project root. **Do not** substitute the `<!-- PROJECT: ... -->` placeholders — those are for the user to fill in.

## 6. Ensure .claude/settings.json

- If exists: skip.
- Else: copy `templates/settings.json` to `.claude/settings.json`.

## 7. Report

Return a summary structured as:

```
Created:
- <list of files / directories created>

Skipped (already present):
- <list>

Appended to .gitignore:
- <list of new lines, or "none">

Placeholders to fill in:
- CLAUDE.md: Project Overview, Architecture, Dependencies, Commands, Configuration, Testing, Implementation Notes
- ROADMAP.md: Mission, Threads (T01..)
- (if applicable) any other <!-- PROJECT: ... --> markers found
```

Do not commit. The user does that after filling in placeholders.