# Retire a File

## Process

1. **Validate input**: Confirm file path exists and is readable.
2. **Create Retired folder**: If `Retired/` doesn't exist, create it at repo root.
3. **Update .gitignore**: Add `Retired/` to .gitignore if not already present.
4. **Move file**: Move the target file to `Retired/` preserving filename and relative structure where reasonable.
5. **Confirm**: Return success message with source and destination paths.
6. **Plan note**: When used in a plan, the plan execution log should include "retire skill invoked on [filename]".

## Examples

**Usage in a skill:**
```
Skill("retire", "old_document.md")
```

**Return:**
```
✓ Retired old_document.md → Retired/old_document.md
  Retired/ added to .gitignore
```

**In plan execution:**
The execute-plan skill will note in Executor Notes: "retire skill invoked on old_document.md"