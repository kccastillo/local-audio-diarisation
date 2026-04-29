# LOG Update Rules

## Executor Notes Fields

Populate the PLAN's Executor Notes section with these fields:

- **Executed:** today's date in YYYY-MM-DD format
- **Outcome:** one of: `done` | `partially-complete` | `blocked` | `needs-revision`
- **What was done:** bulleted list, one item per meaningful action
- **Blockers (if any):** specific step numbers and reasons; leave blank if none
- **Files modified:** bulleted list of created/modified/renamed paths

## Frontmatter Status

The PLAN frontmatter `status` field must match the Executor Notes Outcome exactly:
- `done` → execution completed, all verification passed
- `partially-complete` → some steps done, others blocked or deferred
- `blocked` → cannot proceed; `blocked_by` field specifies reason
- `needs-revision` → plan itself is faulty; needs review

## LOG Status Table Update

Path: `Bus/{YYYYMM}010000_LOG_{YYYYMM}.md`

1. Find the row for this PLAN filename in the Status Table
2. Update the Status column to match PLAN frontmatter `status`
3. Update LOG frontmatter `last_updated` to today's date (YYYY-MM-DD)

Status column must always match PLAN frontmatter `status` and Executor Notes Outcome — never leave in disagreement.

## State File Updates

For every State/ file modified during execution:
- Update its frontmatter `last_updated` to today's date ({YYYY-MM-DD})
- Only touch files that were actually changed — do not modify read-only accesses

## Success Criteria

- Executor Notes populated with execution details and today's date
- Frontmatter `status` flipped to match Outcome (not left on `ready` or `in-progress`)
- Every State/ file modified has `last_updated` bumped to today
- LOG Status Table row updated with final outcome (matches PLAN frontmatter `status`)