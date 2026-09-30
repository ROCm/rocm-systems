---
description: Review the current local branch using the AMD-SMI Review Agent
allowed-tools: Bash(git:*), Read, Write, Glob, Grep, Task
argument-hint: "[review-type ...] — style, tests, docs, architecture, security, performance, build, skeptic (or blank for comprehensive). Add 'fast' to skip rebuttal. Add 'no-build' to skip build step."
---

Review the current local branch using the AMD-SMI Review Agent.

## Arguments

- `$ARGUMENTS` contains: `[review-type ...]`
- Valid types: `style`, `tests`, `docs`, `architecture`, `security`, `performance`, `build`, `skeptic`
- Special modifier: `fast` — skips the rebuttal round (comprehensive mode only)
- Special modifier: `no-build` — skips the build & install step
- If no types specified, perform a **comprehensive** review (all subagents, with rebuttal)

## Process

### 1. Get Branch Information

```bash
git branch --show-current
git log --oneline main..HEAD
git diff --stat main..HEAD
```

Determine: branch name, base branch, commit count, files changed.

### 2. Get the Diff

```bash
git diff main..HEAD
```

### 3. Gather CI Evidence (if available)

If a CI system is configured, attempt to fetch recent run data for this branch to pass to test and performance subagents.

### 4. Dispatch Review

Invoke the **AMD-SMI Review Agent** with:

- The diff
- The review type(s) from `$ARGUMENTS` (or "comprehensive" if none)
- Any CI evidence gathered

The agent will dispatch to the appropriate subagent(s) and produce a formatted review.
By default, comprehensive reviews include a rebuttal round (Round 2) with the skeptic subagent. If `fast` was specified, the rebuttal round is skipped.

### 5. Report Results

- Present the review inline
- Summarize the overall assessment
- List any blocking issues found
- If saving requested, use: `reviews/local_{COUNTER}_{branch-name}[_{TYPE}].md`

## Examples

``` bash
/amdsmi-review-branch
/amdsmi-review-branch style
/amdsmi-review-branch tests security
/amdsmi-review-branch build
/amdsmi-review-branch fast
/amdsmi-review-branch skeptic
```
