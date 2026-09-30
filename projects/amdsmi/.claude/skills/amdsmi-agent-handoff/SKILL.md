---
name: amdsmi-agent-handoff
description: "Use when handing work from one amd-smi agent to another (planning→development, planning→review, development→review) or compacting a long session into a fresh one. Produces a compact handoff doc referencing artifacts by path instead of duplicating them."
---

# Agent Handoff — amd-smi

The single hand-off contract between amd-smi agents and across session boundaries.
One shape for every hop, so no agent restates its own dispatch/return format.

**Core principle:** Reference artifacts by path. Never paste file contents, diffs,
or plans into the handoff — point to them. Redact secrets.

## Nesting Budget

Subagents may dispatch their own subagents, but the chain is capped by the
**`agent-max-nesting-depth`** setting (default **3**, declared in
`.github/copilot-instructions.md` — the single source of truth).

Rules:

- Depth is counted from the first orchestrator (Planning / Development / Review).
- Every dispatch carries a **`Nesting depth: N/3`** line (see the template).
  The first orchestrator runs at `1/3`.
- When you dispatch, pass `N+1` to the receiver.
- If you are running at depth `N == MAX`, you are a **leaf**: do the work
  yourself, do **not** dispatch any subagent. (Dispatching parallel subagents at
  the same level all share your `N+1` — width is unlimited, only depth is capped.)
- If a leaf genuinely cannot proceed without going deeper, return
  `STATUS: BLOCKED` explaining why, rather than exceeding the budget.

## When to Use

- Planning dispatches a task to the Development or Review agent
- Development returns results to Planning, or hands a branch to Review
- A session is getting long and work must continue in a fresh session

**Don't use when:** answering a trivial inline request Planning can handle itself.

## Where the Doc Goes

Write to the OS temp dir, **not** the workspace:

```bash
HANDOFF="${TMPDIR:-/tmp}/amdsmi-handoff-$(date +%Y%m%d-%H%M%S).md"
```

Pass the path to the receiver. Workspace stays clean (no stray handoff files in git).

## Handoff Template

```markdown
# Handoff: <one-line goal>

**From:** <planning | development>  **To:** <development | review>
**Date:** <YYYY-MM-DD>
**Nesting depth:** <N>/3   <!-- receiver's depth; leaf when N == 3, do not dispatch further -->

## Goal
<one or two sentences — what the receiver must accomplish>

## Scope
- In: <files/dirs the receiver may change>
- Out: <files/dirs that are OFF LIMITS>

## Constraints
- <e.g., do not regenerate the wrapper unless adding a C API function>
- <project rules: TDD first, verification-before-completion, no push w/o approval>

## Artifacts (by path — do NOT inline)
- Spec: ${TMPDIR:-/tmp}/amdsmi-agent-specs/<file>.md
- Plan: ${TMPDIR:-/tmp}/amdsmi-agent-plans/<file>.md (task N, lines X–Y)
- Worktree: <abs path>
- Related: <PR URL, issue, prior handoff path>

## Suggested Skills
- <skill the receiver should load — e.g., test-driven-development, systematic-debugging>

## Expected Return
- <what the sender needs back — STATUS, files changed, tests run, blockers>
```

## Expected Return (receiver → sender)

Receivers report back in this shape (replaces per-agent return formats):

```markdown
STATUS: DONE | BLOCKED | NEEDS_CONTEXT
FILES CHANGED:
- <path>:<lines> — <summary>
TESTS RUN:
- <command> → <result>
VERIFICATION:
- <what was run from verification-before-completion>
BLOCKERS (if any):
- <description + which skill/agent resolves it>
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Pasting the plan/diff into the handoff | Reference it by path |
| Writing the handoff into the workspace | Write to `${TMPDIR:-/tmp}` |
| Omitting the Out-of-scope list | Always state what NOT to touch |
| Including tokens/passwords in Artifacts | Redact; reference the secret's location instead |
| Restating a skill's steps | List it under Suggested Skills — the receiver loads it |
| Dispatching a subagent while at max depth | You're a leaf — do the work, or return BLOCKED |
| Omitting the `Nesting depth` line | Always carry it; the receiver needs its budget |
