---
description: "Review a GitHub pull request using the AMD-SMI Review Agent"
agent: "AMD-SMI Review Agent"
argument-hint: "<PR_NUMBER> [review-type ...] — e.g. 1234 style tests (or just the number for comprehensive). Add 'fast' to skip rebuttal. Add 'no-build' to skip build step. Add 'inherit' to use the orchestrator's model for subagents instead of the default (Sonnet 4.6)."
tools: [execute, read, edit, search, agent, web, todo]
---

Review a GitHub pull request using the AMD-SMI Review Agent.

## Arguments

- `$ARGUMENTS` contains: `<PR_NUMBER> [review-type ...]`
- First argument: PR number (required)
- Remaining: optional review types: `style`, `tests`, `docs`, `architecture`, `security`, `performance`, `build`, `skeptic`
- Special modifier: `fast` — skips the rebuttal round (comprehensive mode only)
- Special modifier: `no-build` — skips the build & install step
- Special modifier: `inherit` — makes all subagents inherit the orchestrator's model (the model you selected in the VS Code model picker) instead of their default (Sonnet 4.6)
- If no types specified, perform a **comprehensive** review (all subagents, with rebuttal)

## Process

### 1. Fetch PR Information

Use the PR number directly — `gh` resolves the repo from the local git remote:

```bash
gh pr view <PR_NUMBER> --json number,title,author,body,files,additions,deletions,state,baseRefName,headRefName,comments,reviews
```

### 2. Fetch the Diff

```bash
gh pr diff <PR_NUMBER>
```

### 3. Gather CI Evidence

Check for linked CI runs and fetch step-level data:

```bash
# Get CI check runs for the PR
gh pr checks <PR_NUMBER>

# For each failed or interesting run, get details
gh run view <RUN_ID> --json jobs
```

Find a recent baseline run on `develop` for comparison:
```bash
gh run list --branch develop --workflow <WORKFLOW> --limit 1 --json databaseId,conclusion
```

Compare step timings and status between PR run and baseline. Pass this evidence to the test and performance subagents.

### 5. Fetch Unresolved Comments

```bash
gh pr view $PR_URL --json comments,reviews,reviewRequests
```

Extract unresolved review comments for the unresolved comments analysis.

### 6. Dispatch Review

Invoke the **AMD-SMI Review Agent** with:
- PR metadata (title, author, files, +/- lines)
- The diff
- The review type(s) from `$ARGUMENTS` (or "comprehensive" if none)
- CI evidence (test results, step timings, baseline comparison)
- Unresolved comments
- If `inherit` was specified, tell the agent to have subagents inherit the orchestrator's model (ignore their frontmatter `model` field)

The agent will dispatch to the appropriate subagent(s) and produce a formatted review.
By default, comprehensive reviews include a rebuttal round (Round 2) with the skeptic subagent. If `fast` was specified, the rebuttal round is skipped.

### 7. Report Results

- Present the review inline
- Summarize the overall assessment
- List any blocking issues found
- If saving requested, use: `reviews/pr_{NUMBER}[_{TYPE}].md`

## Examples

```
/amdsmi-review-pr 123
/amdsmi-review-pr 123 style
/amdsmi-review-pr 123 tests security performance
/amdsmi-review-pr 123 fast
```
