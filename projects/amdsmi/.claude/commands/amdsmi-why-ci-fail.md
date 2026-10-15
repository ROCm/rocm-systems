---
description: Diagnose CI failures for an AMDSMI pull request — determines whether failures are caused by this PR's changes or are pre-existing/unrelated
allowed-tools: Bash(gh:*, git:*)
argument-hint: "[PR_NUMBER] — defaults to the PR associated with the current branch"
---

Diagnose CI failures for an AMDSMI pull request.

## Arguments

- `$ARGUMENTS`: optional PR number. If omitted, detect from current branch.

## Process

### 1. Resolve PR Number

If `$ARGUMENTS` is provided, use it directly.

Otherwise, detect from the current branch:

```bash
gh pr view --repo ROCm/rocm-systems --json number,headRefName 2>/dev/null
```

Set `PR_URL=https://github.com/ROCm/rocm-systems/pull/<NUMBER>`.

### 2. Fetch PR Metadata and Changed Files

```bash
gh pr view $PR_URL --repo ROCm/rocm-systems \
  --json number,title,author,baseRefName,headRefName,commits,files
```

Key outputs to capture:
- `files[].path` — all files changed in this PR
- `commits[].oid` — all commit SHAs in this PR
- `baseRefName` — the base branch (usually `develop`)

### 3. Fetch CI Check Results

```bash
gh pr checks $PR_URL --repo ROCm/rocm-systems
```

Separate results into:
- **Failed** checks (conclusion: `failure`)
- **Passed** checks
- **Skipped** checks

If no failures exist, report "All CI checks passed — nothing to diagnose" and stop.

### 4. Get Failure Logs for Each Failed Check

For each failed check, extract the run ID from its URL
(`https://github.com/ROCm/rocm-systems/actions/runs/<RUN_ID>/job/<JOB_ID>`),
then fetch the failed step logs:

```bash
gh run view <RUN_ID> --repo ROCm/rocm-systems --log-failed 2>&1 | head -200
```

Capture the error messages, failed test names, and file paths mentioned in the logs.

### 5. Check Baseline on Base Branch

For each failed workflow, find the most recent run on the base branch to detect
pre-existing failures:

```bash
gh run list --repo ROCm/rocm-systems --branch <BASE_REF> \
  --workflow <WORKFLOW_NAME> --limit 3 \
  --json databaseId,conclusion,headBranch
```

A failure that also appears in the last 3 base branch runs is pre-existing.

### 6. Correlate Failures with PR Changes

For each failed check, determine the verdict by comparing:

**Changed files** (from step 2) vs **files/modules mentioned in failure logs** (from step 4).

Verdict rules:
- **Your issue** — failure log references a file or module that appears in the PR diff,
  OR a test that was added/modified in this PR is now failing.
- **Pre-existing** — same check fails on the base branch (from step 5).
- **Unrelated/Flaky** — failure is in a completely different subsystem with no
  overlap with changed files, and does not appear on base branch consistently.
- **Needs investigation** — logs are insufficient to determine cause.

### 7. Report

Output a structured diagnosis:

```
## CI Diagnosis — PR #<NUMBER>: <TITLE>

### Changed Files (<N> files)
- <list key changed files, grouped by subsystem>

### Failed Checks (<N> failures)

#### ❌ <Check Name>
- **Verdict**: <Your issue | Pre-existing | Unrelated/Flaky | Needs investigation>
- **Reason**: <one or two sentences explaining why>
- **Key error**: <the most relevant error line from the log>
- **Overlap**: <which changed file(s) relate to this failure, if any>

### Summary
- **Your issues**: <N> — <brief description of what needs fixing>
- **Pre-existing**: <N> — <these exist on base branch, not your problem>
- **Unrelated/Flaky**: <N> — <safe to ignore or re-run>

### Recommended Next Steps
<actionable list: what to fix, what to ignore, what to re-run>
```

## Notes

- Windows `hip-tests (ROCR)` failures marked `(xfail)` are **expected failures by design** —
  they are not caused by your PR. Treat them as pre-existing/unrelated.
- `therock-pr-bot` failures are usually a **secondary symptom**: the bot reports failure
  because other checks (like the xfail ones above) are failing. Look at its log output —
  it dumps a JSON map of all check statuses. If all the failing checks it lists are
  themselves xfail or pre-existing, the bot failure is not your fault.
- If logs are truncated or unavailable, note it and give best-effort analysis.
- Check names containing `(xfail)` should never be blamed on the PR author.
