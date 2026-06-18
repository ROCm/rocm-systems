---
name: executing-plans
description: "Use when a written implementation plan exists at ${TMPDIR:-/tmp}/amdsmi-agent-plans/ and you need to execute it task-by-task in the current session with verification at each step."
---

# Executing Plans — amd-smi

Load the plan, review it critically, execute every task in order, verify at every step, report when done.

**Announce at start:** "I'm using the `executing-plans` skill to implement this plan."

If subagent dispatch is available and the plan has 3+ independent tasks, prefer the `amdsmi-development` agent's subagent-driven mode — it isolates per-task context and gives higher-quality output. Use this skill when subagents aren't available or the plan is small enough to run inline.

## Process

### Step 1: Load and Review the Plan

1. Read the plan file in full
2. Read the linked spec
3. Identify any gaps, contradictions, or unclear instructions
4. If concerns exist → raise them with the user before any code change
5. If no concerns → create a todo for each task and proceed

### Step 2: Workspace Setup

Verify you're in an isolated workspace. If not, invoke `amdsmi-using-git-worktrees` before touching code.

### Step 3: Execute Each Task

For each task in the plan:

1. Mark the task `in-progress` in the todo list
2. Follow each step in the order written — do not skip verification steps
3. Run the exact commands specified; compare output to "Expected"
4. Commit at the step the plan says to commit (don't batch commits across tasks)
5. Mark the task `completed`

**After every task:** invoke `verification-before-completion` before marking the task done. Compilation success is not verification.

### Step 4: Final Verification

After all tasks complete:

- Run the full test suite (`amdsmi-test-runner` skill)
- Run pre-commit: `pre-commit run --all-files`
- Verify cascade integrity (see `project-layout` rule's "Quick Check" grep)

### Step 5: Finish the Branch

Invoke the `amdsmi-restructure-commits` skill — the "Finishing" section handles merge/PR/discard options.

## When to Stop and Ask

STOP immediately when:

- A step's expected output doesn't match actual output and you don't understand why
- A test fails after Step 3 and the plan doesn't address that failure
- A plan instruction is ambiguous
- Verification fails twice on the same task
- You hit an architectural problem the plan didn't anticipate (3+ fix attempts → `systematic-debugging` Phase 4.5)

**Ask the user. Don't guess. Don't improvise.**

## When to Revisit Earlier Steps

Return to Step 1 (re-review) if:

- The user updates the plan
- A discovered architectural problem invalidates later tasks

Do NOT batch-edit later tasks to compensate for an earlier task's gaps — fix the plan, then re-execute.

## Red Flags

| Symptom | Action |
|---------|--------|
| Skipping a verification step "to save time" | Stop. Verification steps are mandatory. |
| Editing files outside the plan's "Files" list | Stop. Ask whether the plan needs updating. |
| Batching multiple tasks into one commit | Stop. Commit per the plan. |
| "The test failure looks unrelated" | Run `systematic-debugging`. Don't dismiss. |
| Starting on `develop`/`main` without a feature branch | Stop. Create a branch (or worktree) first. |

## Required Companion Skills

- **`amdsmi-using-git-worktrees`** — set up isolated workspace before Step 3
- **`test-driven-development`** — when the plan says "write failing test", follow strict RED-GREEN-REFACTOR
- **`verification-before-completion`** — after every task
- **`systematic-debugging`** — when a step fails unexpectedly
- **`amdsmi-restructure-commits`** — to finish the branch (Step 5)
