---
name: bus-naming-convention
description: Complete naming rules, type tokens, and worked examples for Bus/ files.
---

# Bus Naming Convention

## Pattern

```
{YYYYMMDDHHMI}_{TYPE}_{slug}.md
```

- `{YYYYMMDDHHMI}` — 12-digit compact datetime: year(4) + month(2) + day(2) + hour(2) + minute(2)
- `{TYPE}` — one of four tokens (uppercase)
- `{slug}` — lowercase-hyphenated descriptor, max 5 words, no spaces

---

## Type Tokens

| Token | What it is | Created by | Consumed by |
|---|---|---|---|
| `LOG` | Monthly rollup log | Haiku | All agents for status |
| `PLAN` | Actionable task for Haiku | Haiku | Haiku (executes) |
| `RESEARCH` | Haiku data drop | Haiku | Sonnet (integrates) |
| `ADVICE` | Opus strategic note (Ken pastes) | Ken | Sonnet (integrates) |

---

## Special Rules

### Monthly LOG files
Always use first-of-month midnight as timestamp, regardless of when created:
```
{YYYYMM}010000_LOG_{YYYYMM}.md
```
This makes them predictable and lexicographically first in the month's directory listing.

### Recurring PLAN files
Slug must start with `RECUR-`:
```
{YYYYMMDDHHMI}_PLAN_RECUR-{slug}.md
```
Recurring PLAN files are **persistent** — one file per recurring task. Each completed cycle appends a row to the `## History` table. Do not create a new file each cycle.

### RESEARCH and ADVICE files
Written via the `write-bus-input` skill — see [.claude/skills/write-bus-input/SKILL.md](.claude/skills/write-bus-input/SKILL.md) for content rules. Sonnet pre-generates the target filename and gives it to Haiku before the input is written. This allows PLANs to pre-link via `linked_research` / `linked_advice` before the file exists.

### Slugs
- Lowercase, hyphenated
- Max 5 words
- Descriptive enough to understand without opening the file
- No month/year in slug (the timestamp carries that)
- No spaces, no underscores (hyphens only)

---

## Worked Examples

```
202604010000_LOG_202604.md
  → April 2026 monthly log (first-of-month timestamp)

202604191430_PLAN_matt-lease-insurance.md
  → One-off plan: insurance implications for Matt's lease

202604191500_PLAN_RECUR-hormuz-tracker.md
  → Recurring plan: monthly Hormuz signals tracker update

202604191600_PLAN_RECUR-fbt-monitor.md
  → Recurring plan: monthly FBT exemption status check

202604191700_RESEARCH_fbt-exemption-apr26.md
  → Haiku research drop: FBT exemption status April 2026

202604191800_ADVICE_hormuz-portfolio-strategy.md
  → Opus advice: Hormuz scenario portfolio strategy

202605010000_LOG_202605.md
  → May 2026 monthly log
```

---

## Directory Listing Behaviour

Files sort lexicographically by timestamp first, making the monthly LOG always appear before any same-month PLAN files. Within a month, files sort chronologically by creation time.

Recurring PLAN files (RECUR-) sort by their original creation timestamp — they do not get new timestamps on rollover.
