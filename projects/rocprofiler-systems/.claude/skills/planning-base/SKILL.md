---
name: planning-base
description: Base planning skill with shared rules - do not invoke directly, use specific planning skills instead
---

# Base Planning Rules

<IMPORTANT>
This is a base skill containing shared planning rules. Do not invoke this skill directly.
Instead, use the appropriate specialized skill:
- `planning-feature` - for new features
- `planning-bugfix` - for bug fixes
- `planning-refactor` - for refactoring
- `planning-docs` - for documentation
- `review-architecture` - for architecture documentation
</IMPORTANT>

## Core Principles

Planning MUST happen before ANY implementation. A task is non-trivial if it requires more than 2 distinct steps.

**Every plan leads to a Pull Request.** Keep PR reviewability in mind from the start.

## Planning Mode Workflow

<IMPORTANT>
Complete all planning phases (0-4) before implementation. After completing the plan:
1. Save the plan to `planning/` folder (Phase 5)
2. Then begin implementation

**Platform-specific:**

- **Cursor:** Use Plan Mode for phases 0-4, then switch to Agent Mode
- **Claude Code:** Use `EnterPlanMode` tool or simply complete planning before coding

This ensures the plan is persisted before any code changes begin.
</IMPORTANT>

## Phase 0: Check for Existing Plans

Before creating a new plan, check the `planning/` folder in project root:

1. **List files** in `planning/` directory
2. **Look for related plans** - similar features, continuation of previous work
3. **If found:**
   - Read the existing plan
   - Check which tasks are already completed (marked with `[x]`)
   - Resume from where it left off OR adapt the plan for the new request
4. **If not found:** proceed to Phase 1

## Phase 1: Analyze

Before writing ANY code or making ANY changes:

1. **Understand the request** - What exactly is being asked?
2. **Identify the scope** - What files/components are affected?
3. **Find dependencies** - What existing code/patterns should be followed?
4. **Spot risks** - What could go wrong? What needs extra attention?

## Phase 2: Assess PR Scope

<IMPORTANT>
Before decomposing, assess if this should be ONE PR or MULTIPLE PRs.
Invoke `git-create-pull-request` skill for detailed PR guidelines.
</IMPORTANT>

### PR Size Guidelines

- **< 400 lines** - Ideal, easy to review
- **400-800 lines** - Acceptable for complex work
- **> 800 lines** - Must split into multiple PRs

### When to Split

- Changes touch multiple unrelated systems
- Refactoring can be separated from feature work
- Infrastructure changes can land independently
- Tests can be added before implementation

### If Splitting Required

Plan each PR separately:

```markdown
## PR Strategy

### PR 1: [Title]
**Scope:** [What's included]
**Dependencies:** None

### PR 2: [Title]
**Scope:** [What's included]
**Dependencies:** PR 1
```

## Phase 3: Decompose

Break the task into concrete, actionable steps:

- Each step should be independently verifiable
- Steps should be ordered by dependency (what must come first)
- Keep steps small enough to track progress meaningfully
- Include validation/testing steps where appropriate
- **Group steps by PR** if multiple PRs are planned

## Phase 4: Create Task List

Use your platform's task tracking tool to create a structured task list:

- **Claude Code:** `TaskCreate`, `TaskUpdate`, `TaskList` tools
- **Cursor:** `TodoWrite`

```text
Example task structure:
1. [pending] Analyze existing code structure
2. [pending] Create/modify necessary files
3. [pending] Implement core logic
4. [pending] Add error handling
5. [pending] Verify changes work correctly
```

**Task Status Rules:**

- `pending` - Not yet started
- `in_progress` - Currently working on (only ONE at a time)
- `completed` - Finished and verified
- `cancelled` - No longer needed

## Phase 5: Persist the Plan

<IMPORTANT>
This phase happens **immediately after planning is complete** - before any implementation begins.
</IMPORTANT>

Save the plan to the `planning/` folder in the project root:

1. **Create folder** if it doesn't exist: `planning/`
2. **Save plan** as a markdown file with descriptive name (see specific skill for naming)
3. **Include in the plan file:**
   - Original request/goal
   - Analysis summary
   - Todo list with status markers
   - Any relevant context or decisions made

## Phase 6: Confirm (Optional)

For complex or risky changes, present the plan to the user:

> "Before I begin, here's my plan:
>
> 1. [step 1]
> 2. [step 2]
> ...
> Should I proceed?"

## During Execution

<IMPORTANT>
After completing EACH task or solution, you MUST:
1. Mark the task as `completed` using your platform's task tracking tool
2. Update the plan file - change `- [ ]` to `- [x]` for the completed task
</IMPORTANT>

- Mark each task as `in_progress` when you start working on it
- Mark as `completed` immediately after finishing each task
- Update the plan file in `planning/` to reflect current progress
- If a step reveals new requirements, add new tasks AND update the plan file
- If a step becomes unnecessary, mark as `cancelled`

## What NOT to Include in Tasks

Do NOT create tasks for:

- Reading/searching the codebase (this is implicit)
- Running linters (this is automatic)
- Basic validation steps that are part of normal workflow

## After Execution: Create Test Plan

Once implementation is complete:

1. **Create test plan file:** `planning/testplan-<feature-name>.md`
2. **Invoke `testing-testplan` skill** for the template
3. **Fill in:**
   - Automated tests (what tests exist or will be written)
   - Manual verification scenarios (for developer and QA)
   - Regression check (related features to verify)
4. **Verify manually** and check off items

## After Test Plan: Create Pull Request

Once all tasks are complete and tests are written (if requested):

1. **Ask user** if they want to create a PR
2. **Invoke `git-create-pull-request` skill** for PR template
3. **Create PR** with required sections:
   - **Motivation** - Why is this change needed?
   - **Technical Details** - What changed and how?
   - **Test Plan** - Reference `planning/testplan-<name>.md`

PR creation is the final step of every feature/bugfix/refactor workflow.
