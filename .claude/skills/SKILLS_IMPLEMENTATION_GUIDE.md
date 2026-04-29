# Skills Implementation Guide

Skills are reusable, agent-callable capabilities that extend Claude's abilities. They live in `.claude/skills/<name>/SKILL.md` and are invoked via the `Skill()` tool, either by Claude programmatically or by another skill in a pipeline.

---

## When to Use a Skill vs a Command

**Use a SKILL when:**
- Claude (or another skill) invokes it programmatically via `Skill()` tool
- It's one step in a larger agent workflow (e.g. atomise → librarian → wiki-map-update)
- It carries domain expertise or performs a reusable capability
- It has its own principles, workflows, references, or templates
- It's chainable — can invoke or be invoked by other skills

**Use a COMMAND when:**
- Ken triggers it by typing `/name` in chat
- It's a human-facing entry point to a workflow
- It orchestrates other tools/skills for something Ken invokes regularly
- It has access to the full conversation context

**Decision rule:**
- If a human triggers it → **command** (even if trivial)
- If an agent triggers it → **skill**
- If both apply → primary is a **command**; it invokes skills internally
- If only ever one step inside a larger workflow → **skill**

See `.claude/commands/COMMANDS_IMPLEMENTATION_GUIDE.md` for command structure and examples.

---

## Skill Folder Structure

```
.claude/skills/
├── skill-name/
│   ├── SKILL.md              # Main skill file with frontmatter + instructions
│   ├── workflows/            # Step-by-step procedures (optional)
│   │   ├── workflow-1.md
│   │   └── workflow-2.md
│   ├── references/           # Domain knowledge, patterns, examples (optional)
│   │   ├── topic-1.md
│   │   └── topic-2.md
│   ├── templates/            # Output structures Claude copies and fills (optional)
│   │   ├── output-structure-1.md
│   │   └── output-structure-2.md
│   └── scripts/              # Executable code Claude runs as-is (optional)
│       ├── script-1.sh
│       └── script-2.py
```

### SKILL.md Template

```yaml
---
name: skill-name                          # lowercase-with-hyphens
description: What it does and when to use it (third person)
---

<objective>What this skill does</objective>
<quick_start>Immediate actionable guidance</quick_start>
<process>Step-by-step procedure</process>
<success_criteria>How to know it worked</success_criteria>
```

**Naming conventions:**
- Directory and filename: lowercase with hyphens (e.g. `create-skill`, `consolidate`, `authority-detect`)
- Description should answer: "What does it do?" and "When should I use it?"

---

## Key Principles

1. **Essential principles are always loaded.** If it's non-negotiable, put it in SKILL.md `<essential_principles>`.
2. **Progressive disclosure.** SKILL.md under 500 lines. Detailed content lives in `/workflows`, `/references`, `/templates`, `/scripts`.
3. **Router pattern for complex skills.** Ask "what do you want to do?" and route to appropriate workflow.
4. **Pure XML structure in body.** No markdown headings (#, ##, ###) in the skill body — use `<objective>`, `<process>`, etc.
5. **One workflow per file.** If a skill has 3+ workflows, create separate files in `/workflows/`.

---

## Creating a New Skill

Use the `create-agent-skills` skill to guide you:

```
Skill("create-agent-skills")
```

This will:
1. Ask: "Task-execution skill or domain expertise skill?"
2. Route you to the appropriate workflow
3. Validate your SKILL.md against best practices

---

## Integrating a Skill with the Project

After creating a new skill:

1. **Register in CLAUDE.md.** Add to the Skills table with name, description, and what triggers it.
2. **If it integrates with a plan:** Note in the plan's Steps which skill is invoked and when.
3. **If it's called by another skill:** Document the call in both the calling skill and the called skill.
4. **Test invocation.** Confirm `Skill("name")` works and returns expected results.

---

## Existing Project Skills

| Skill | Purpose | Callable By |
|---|---|---|
| `retire` | Move files to gitignored Retired/ folder | plans, other skills |
| `consolidate` | Find canonical content, replace with pointer refs | plans, other skills |
| `authority-detect` | Detect if content is authoritative | atomise |
| `create-agent-skills` | Guide creation of new skills | user (invoked manually) |
| `create-commands` | Guide creation of new commands | user (invoked manually) |
| `write-bus-plan` | Transcribe plans into Bus/ structure | Haiku on Ken request |
| `write-bus-input` | Write RESEARCH/ADVICE to Bus/ | Ken/Sonnet on handoff |
| `pandoc-convert` | Convert Clippings source files to MD | Haiku in atomise pipeline |
| `atomise` | Split MD into atomic notes | Haiku after pandoc-convert |
| `librarian` | Route atomic notes into Wiki/ | Haiku after atomise |
| `wiki-map-update` | Rebuild Wiki_Map.md Mermaid diagram | librarian, consolidate (after migration) |
| `execute-plan` | Run a PLAN step-by-step | Haiku on Ken request |

---

## Best Practices

- **Defaults to "no"** for ambiguous decisions. If uncertain, leave content unmarked or ask.
- **Fail loudly.** If a skill can't proceed, halt and flag. Don't silently degrade.
- **Verify before acting.** Especially when writing files or invoking other skills.
- **Keep state in frontmatter.** Use SKILL.md frontmatter for configuration, not environment variables.
- **Document dependencies.** If your skill calls another, state that clearly.
- **Test in the project context.** Invoke the skill with real project files; verify it works.

---

## Troubleshooting

**"Skill not found"** — Confirm the directory path is `.claude/skills/<name>/SKILL.md` with exact spelling.

**"My skill is too long"** — Split into workflows or references. SKILL.md should stay under 500 lines.

**"Other skill can't call my skill"** — Check that your skill's name matches what the caller is using. Skill names are case-sensitive.

**"Frontmatter validation failed"** — Ensure YAML syntax is correct: no bare colons, proper indentation, quoted strings with special chars.