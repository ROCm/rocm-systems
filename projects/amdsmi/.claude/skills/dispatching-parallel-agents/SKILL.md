---
name: dispatching-parallel-agents
description: "Use when facing 2+ independent problems with no shared state — e.g., unrelated test failures in different subsystems, multiple independent bug investigations, parallel research tasks. Dispatch one focused subagent per domain instead of investigating sequentially."
---

# Dispatching Parallel Agents — amd-smi

When you have multiple independent problems, sequential investigation wastes time. Dispatch one focused subagent per problem domain, integrate the results.

**Core principle:** One subagent per independent problem. Hand-craft each one's context — never let them inherit your session history.

## When to Use

| Use when | Don't use when |
|----------|---------------|
| 2+ test files fail with unrelated root causes | One failure may cascade into others |
| Independent subsystems broken (C lib + Python + CLI) by separate causes | You don't yet know what's broken |
| Multiple unrelated research questions | Investigation requires shared system state |
| Need to read/summarize many files in parallel | Agents would edit the same files |

## Decision Flow

```
Multiple problems?
   └─ Independent (no shared state, no cascade between them)?
        ├─ yes → dispatch one subagent per problem domain
        └─ no  → investigate together (single agent, full context)
```

## The Pattern

### 1. Identify Independent Domains

Group failures or tasks by what's broken or what's being investigated. Each domain must be understandable without context from the others.

### 2. Craft Focused Subagent Prompts

Each prompt must contain:

- **Scope** — exactly one problem domain, one test file, or one subsystem
- **Context** — paste relevant error messages, file paths, line numbers, recent diffs
- **Constraints** — what the subagent must NOT touch
- **Expected output** — what the subagent must return (don't accept narrative summaries)

### 3. Dispatch Concurrently

Use the `runSubagent` tool, issuing all dispatches in the same response. Choose the most appropriate amd-smi review/development agent per task — don't default to the general-purpose agent if a specialized one exists.

### 4. Integrate Results

When subagents return:

1. **Verify the work** — subagent reports describe intent, not necessarily what happened. Read the actual file changes or command output before accepting.
2. **Check for conflicts** — did two subagents edit the same file?
3. **Run the affected test suites** to confirm the integrated state passes
4. **If any subagent flagged a blocker** → that branch of work stops until resolved; other branches may continue

## Subagent Prompt Template

```
TASK: <specific problem in one sentence>

CONTEXT:
- Failing test: <path>:<line>
- Error message: <paste verbatim>
- Recent changes in this area: <git log oneline of last 3 commits touching this file>
- Related files: <paths>

CONSTRAINTS:
- Do NOT modify <list of files outside scope>
- Do NOT regenerate amdsmi_wrapper.py (it's auto-generated)
- Follow the project-layout rule

EXPECTED OUTPUT:
- Root cause (one paragraph)
- Files changed (list with line ranges)
- Verification command run and its output
- Any new findings/blockers
```

## Common Mistakes

| ❌ Bad | ✅ Good |
|-------|--------|
| "Fix all the tests" | "Fix the 3 failures in `tests/python_unittest/test_metrics.py`" |
| Letting the subagent inherit your full session context | Hand-craft exactly the context the subagent needs |
| Trusting the subagent's "Done!" report | Read the diff and run the verification command yourself |
| Dispatching subagents to edit overlapping files | One subagent per file/subsystem; serialize edits to shared files |
| Dispatching for problems that might be related | Investigate together first; split only after confirming independence |

## Verification After Integration

- Re-run the broader test suite (`amdsmi-test-runner`), not just the per-subagent slice
- Check `git status` for files no subagent claimed to touch
- Look for systematic errors — if two subagents made the same wrong assumption, the prompt template was the problem

## Red Flags — STOP

- Subagent prompts that read "investigate X and fix anything related"
- Two subagents told to edit the same file
- No verification command in the prompt
- Accepting "looks good" without reading the diff
