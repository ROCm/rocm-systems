---
description: Review a GitHub pull request using the AMD-SMI Review Agent
allowed-tools: Bash(gh:*, git:*), Read, Write, Glob, Grep, WebFetch, Task
argument-hint: "<PR_NUMBER> [review-type ...] — e.g. 1234 style tests (or just the number for comprehensive). Add 'fast' to skip rebuttal. Add 'no-build' to skip build step."
---

Review a GitHub pull request using the AMD-SMI Review Agent.

## Arguments

- `$ARGUMENTS` contains: `<PR_NUMBER> [review-type ...]`
- First argument: PR number (required). Construct the full URL as `https://github.com/ROCm/rocm-systems/pull/<PR_NUMBER>`
- Remaining: optional review types: `style`, `tests`, `docs`, `architecture`, `security`, `performance`, `build`, `skeptic`, `spec`, `maintainer`
- Special modifier: `fast` — skips the rebuttal round (comprehensive mode only)
- Special modifier: `no-build` — skips the build & install step
- If no types specified, perform a **comprehensive** review (all subagents, with rebuttal)

## Process

### 1. Construct PR URL

Set `PR_URL=https://github.com/ROCm/rocm-systems/pull/<PR_NUMBER>` from the first argument.

### 2. Fetch PR Information

```bash
gh pr view $PR_URL --json number,title,author,body,files,additions,deletions,state,baseRefName,headRefName,comments,reviews
```

### 3. Set Up Worktree

Check out the PR in its own worktree with the `amdsmi-using-git-worktrees` skill (`new-worktree.sh -p <PR_NUMBER>`). If `rocm-systems-pr<PR_NUMBER>` already exists, refresh it from there:

```bash
git fetch origin "pull/<PR_NUMBER>/head" && git checkout --detach FETCH_HEAD
```

Run every later step (build, tests, file reads) inside this worktree.

### 4. Fetch the Diff

```bash
gh pr diff $PR_URL
```

### 5. Gather CI Evidence

Check for linked CI runs and fetch step-level data:

```bash
# Get CI check runs for the PR
gh pr checks $PR_URL

# For each failed or interesting run, get details
gh run view <RUN_ID> --json jobs
```

Find a recent baseline run on `develop` for comparison:

```bash
gh run list --branch develop --workflow <WORKFLOW> --limit 1 --json databaseId,conclusion
```

Compare step timings and status between PR run and baseline. Pass this evidence to the test and performance subagents.

### 6. Fetch Unresolved Comments

```bash
gh pr view $PR_URL --json comments,reviews,reviewRequests
```

Extract unresolved review comments for the unresolved comments analysis.

### 7. Dispatch Review

Invoke the **AMD-SMI Review Agent** with:

- PR metadata (title, author, files, +/- lines)
- The diff
- The review type(s) from `$ARGUMENTS` (or "comprehensive" if none)
- CI evidence (test results, step timings, baseline comparison)
- Unresolved comments
- The worktree path so subagents build, test and read the PR's code

The agent will dispatch to the appropriate subagent(s) and produce a formatted review.
By default, comprehensive reviews include a rebuttal round (Round 2) with the skeptic subagent. If `fast` was specified, the rebuttal round is skipped.

### 8. Report Results

- Present the review inline
- Summarize the overall assessment
- List any blocking issues found
- If saving requested, use: `reviews/pr_{NUMBER}[_{TYPE}].md`

## Examples

``` bash
/amdsmi-review-pr https://github.com/ROCm/rocm-systems/pull/123
/amdsmi-review-pr https://github.com/ROCm/rocm-systems/pull/123 style
/amdsmi-review-pr https://github.com/ROCm/rocm-systems/pull/123 tests security performance
/amdsmi-review-pr https://github.com/ROCm/rocm-systems/pull/123 fast
```
