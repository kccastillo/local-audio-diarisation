# Skill Structure Examples

## Simple Skill (Single File)

```yaml
---
name: skill-name
description: What it does and when to use it.
---

<objective>What this skill does</objective>
<quick_start>Immediate actionable guidance</quick_start>
<process>Step-by-step procedure</process>
<success_criteria>How to know it worked</success_criteria>
```

## Complex Skill (Router Pattern)

```
skill-name/
├── SKILL.md              # Router + principles
├── workflows/            # Step-by-step procedures (FOLLOW)
├── references/           # Domain knowledge (READ)
├── templates/            # Output structures (COPY + FILL)
└── scripts/              # Reusable code (EXECUTE)
```

**SKILL.md:**
- `<essential_principles>` - Always applies
- `<intake>` - Question to ask
- `<routing>` - Maps answers to workflows

**workflows/:**
- `<required_reading>` - Which refs to load
- `<process>` - Steps
- `<success_criteria>` - Done when...

**references/:**
- Domain knowledge, patterns, examples

**templates/:**
- Output structures Claude copies and fills (plans, specs, configs, documents)

**scripts/:**
- Executable code Claude runs as-is (deploy, setup, API calls, data processing)

## When to Use Each Folder

- **workflows/** - Multi-step procedures Claude follows
- **references/** - Domain knowledge Claude reads for context
- **templates/** - Consistent output structures Claude copies and fills
- **scripts/** - Executable code Claude runs as-is