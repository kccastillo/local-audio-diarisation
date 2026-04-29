# Anti-patterns

Patterns that, when found in CLAUDE.md or CONTEXT_CONSTITUTION.md, are flagged as blockers (must be fixed) or warns (should be reviewed).

## Blockers

### Natural-language linting
Rules that describe formatting/style enforceable by tooling. Examples:
- "Use 2 spaces for indentation"
- "Use single quotes for strings"
- "Always include a trailing newline"
**Why blocker:** wastes instruction budget on a job an LLM does worse than a linter. Move to `.eslintrc`, `.prettierrc`, `pyproject.toml`, or a `post_tool_use` hook in `settings.json`.

### AI-generated bloat
Verbose generic guidance with no project specificity. Tells:
- Sentences starting with "It's important to..." / "Remember to..." / "Always make sure to..."
- Bullet lists of generic best practices ("Write clean code", "Add tests", "Handle errors")
- Hedging modifiers ("might", "should consider", "could be useful")
**Why blocker:** zero per-token signal. Replace with project-specific imperatives or delete.

### Codebase duplication
Documentation of facts authoritative elsewhere:
- Dependency lists that duplicate `requirements.txt` / `package.json`
- Config schemas that duplicate the actual config file
- Module-by-module architecture descriptions Claude can derive by reading the code
**Why blocker:** drifts silently when the source updates. Move to `.claude/references/` as a pointer, or delete and let Claude read source.

### Dead references
Pointers to files, skills, or sections that no longer exist.
**Why blocker:** poisons context — Claude follows the pointer, finds nothing, may invent content.

## Warns

### Static maintenance smell
- Version numbers inline (likely to drift)
- "As of YYYY-MM" timestamps without an audit cadence
- "We recently switched to X" (will become stale)
**Why warn:** date the project, schedule a review, or move to a versioned reference.

### Lost-in-the-middle risk
- Critical rules buried 60-80% into the file with no signal-boost markers
- Long mid-file sections with no headers (model attention drifts)
**Why warn:** restructure or add IMPORTANT markers.

### Caveat creep
- Caveats section growing past ~10 items
**Why warn:** likely contains some that are now obvious from code (delete) and some that are subsystem-specific (move to subsystem reference).

### Subagent / delegation rules missing
- No documented threshold for when to delegate to a subagent
**Why warn:** Claude defaults to handling everything inline, blowing context on big searches.

## Not anti-patterns (don't flag)

- Working-style preferences (AU spelling, tone, review format) — these ARE the user's signal.
- Project-specific imperatives ("never modify schema.prisma directly") — these are the Trinity caveats; they belong here.
- Pointers to references / skills — that's progressive disclosure working correctly.
- Long-but-load-bearing rules at top or bottom of file (instruction weighting deliberately uses the edges).
