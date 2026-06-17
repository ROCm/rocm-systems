---
name: amdsmi-using-git-worktrees
description: "Use when starting amd-smi feature work, executing an implementation plan, or reviewing a PR that needs an isolated workspace away from the main checkout. Sets up a worktree following the rocm-systems-pr<PR#> convention."
---

# Using Git Worktrees — amd-smi

Ensure work happens in an isolated workspace so the main checkout is never disturbed.

**Core principle:** Detect existing isolation first. Then use the project convention. Never fight the harness.

**Announce at start:** "I'm using the `amdsmi-using-git-worktrees` skill to set up an isolated workspace."

## amd-smi Worktree Convention

Worktrees of `rocm-systems` are created as **sibling directories** of the main
checkout. The prefix is always `rocm-systems-`, lowercase (never `amd-smi-…` or
uppercase `ROCM-…`). Pick the name by this tiered preference:

```
<parent>/rocm-systems-pr<PR#>                  # 1. preferred — a PR exists
<parent>/rocm-systems-<ticket#>                # 2. ticket work, no PR yet (e.g. rocm-systems-26211)
<parent>/rocm-systems-rocm-<ticket#>-<desc>    # 3. ticket + short desc when one name isn't enough
<parent>/rocm-systems-<branch-name>            # local feature branches with no ticket/PR
```

Derive the parent dynamically — do not hard-code it:

```bash
MAIN_CHECKOUT=$(git -C "$(git rev-parse --git-common-dir)/.." rev-parse --show-toplevel)
WORKTREE_PARENT=$(dirname "$MAIN_CHECKOUT")
```

Never create worktrees inside the main checkout (e.g., `./worktrees/`) for this repo — the convention is siblings.

## Step 0: Detect Existing Isolation

Before creating anything, check whether you're already in a linked worktree:

```bash
GIT_DIR=$(cd "$(git rev-parse --git-dir)" 2>/dev/null && pwd -P)
GIT_COMMON=$(cd "$(git rev-parse --git-common-dir)" 2>/dev/null && pwd -P)
```

**Submodule guard** — `GIT_DIR != GIT_COMMON` is also true inside submodules. Verify:

```bash
git rev-parse --show-superproject-working-tree 2>/dev/null
```

If that returns a path → you're in a submodule, treat as a normal repo.

**If `GIT_DIR != GIT_COMMON` and not a submodule:** Already in an isolated worktree. Report path + branch. Skip to Step 3.

**If `GIT_DIR == GIT_COMMON`:** Normal checkout. Continue to Step 1.

## Step 1: Confirm Consent

If the user hasn't already asked for isolation in the current task:

> "Would you like me to set up an isolated worktree at `<parent>/rocm-systems-pr<PR#>`? It protects your current checkout from changes."

If declined → work in place, skip to Step 3.

## Step 2: Create the Worktree

For PR-based work:

```bash
PR_NUM=<number>
BRANCH=<branch-name>   # the PR's branch

# Derive parent from the current checkout
MAIN_CHECKOUT=$(git rev-parse --show-toplevel)
WORKTREE_PARENT=$(dirname "$MAIN_CHECKOUT")
WORKTREE="${WORKTREE_PARENT}/rocm-systems-pr${PR_NUM}"

cd "$MAIN_CHECKOUT"

# Fetch the PR branch if not already local
git fetch origin "${BRANCH}:${BRANCH}" 2>/dev/null || git fetch origin "pull/${PR_NUM}/head:${BRANCH}"

git worktree add "${WORKTREE}" "${BRANCH}"
cd "${WORKTREE}/projects/amdsmi"
```

For a new local feature branch:

```bash
BRANCH=<branch-name>

MAIN_CHECKOUT=$(git rev-parse --show-toplevel)
WORKTREE_PARENT=$(dirname "$MAIN_CHECKOUT")
WORKTREE="${WORKTREE_PARENT}/rocm-systems-${BRANCH}"

cd "$MAIN_CHECKOUT"
git worktree add "${WORKTREE}" -b "${BRANCH}" origin/develop
cd "${WORKTREE}/projects/amdsmi"
```

**Sandbox fallback:** If `git worktree add` fails with a permission error, tell the user and work in the current directory instead.

## Step 3: Project Setup

Worktrees of `rocm-systems` already have the full source tree — no `npm install` / `pip install` needed for amd-smi work.

If the worktree is for a clean build, invoke the `amdsmi-build-install` skill to produce a baseline build.

## Step 4: Verify Clean Baseline

If the worktree is new, build once to confirm a clean starting state:

```bash
# From the amdsmi/ subdirectory inside the worktree
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j"$(nproc)"
```

If the build fails on `origin/develop` → report the failure, ask whether to proceed or investigate. Do NOT proceed with feature work on top of a broken baseline.

## Quick Reference

| Situation | Action |
|-----------|--------|
| Already in a linked worktree (`GIT_DIR != GIT_COMMON`) | Skip creation, report path + branch |
| In a submodule | Treat as a normal repo |
| PR-based work | `rocm-systems-pr<PR#>` sibling dir |
| Local feature branch | `rocm-systems-<branch>` sibling dir |
| Sandbox blocks `worktree add` | Fall back to in-place, tell the user |
| Build fails on baseline | Report, ask before proceeding |

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Creating a worktree inside the main checkout | Sibling-directory convention only |
| Skipping Step 0 detection | Always detect existing isolation first |
| Creating a worktree without consent | Ask first (unless user already said so) |
| Forgetting to `cd` into `projects/amdsmi` after worktree add | The worktree root is `rocm-systems-pr<N>`, but amd-smi work happens in `projects/amdsmi` |
| Proceeding past a failing baseline build | Stop and report |

## Red Flags — STOP

- Creating a worktree without confirming current isolation status
- Building feature work on a baseline that doesn't compile
- Branching from anything other than `origin/develop` (unless explicitly told otherwise)
- Force-deleting a worktree without consulting `amdsmi-restructure-commits`' finishing flow

## Cleanup

Cleanup happens at the end of the workflow — see the `amdsmi-restructure-commits` skill's "Finishing the Branch" section. Don't remove worktrees inline; that's the finishing skill's job.
