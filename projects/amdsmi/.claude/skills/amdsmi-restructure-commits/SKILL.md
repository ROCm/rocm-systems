---
name: amdsmi-restructure-commits
description: "Use when finishing an amd-smi development branch — consolidating commits into logical groups with clean messages AND deciding how to integrate the work (merge to develop, push and open PR, keep as-is, or discard). Covers commit restructuring plus the merge/PR/cleanup workflow."
---

# Restructure Commits & Finish a Branch — amd-smi

Consolidates a branch's commits into a minimal set of logical, independent
commits with descriptive messages and bullet-pointed bodies, following
rocm-systems / amd-smi conventions.

## When to Use

- Branch has many small "WIP", "Updates", or "Addressed comments" commits
- Commits have tangled dependencies that should be separated
- Preparing a PR for merge with clean history
- Reviewer requests commit cleanup

## Scope — Current Worktree/Branch by Default

By default this skill operates on **the branch checked out in the current working
directory** and nothing else. Confirm the target up front:
`git rev-parse --show-toplevel` and `git branch --show-current`.

Wandering to other worktrees, sibling checkouts, or other open PRs is allowed
**only with explicit user approval for that specific scope**:

- Default to the current branch. Do not `cd` into another worktree or restructure
  another branch/PR unless the user has clearly asked you to.
- If restructuring would help across several branches/PRs, **propose it and wait
  for approval** — name each branch/worktree you'd touch — before leaving the
  current one. Silence is not approval.
- When approved to span multiple branches, treat it as **one pass per branch**,
  run from that branch's own worktree, restructuring only that branch's commits.
- If the current branch isn't the one the user meant, STOP and ask rather than
  switching to it yourself.

---

## Workflow

### Step 1: Analyze Current Commits

Run the full analysis in one shot:

```bash
# Commit summary + full messages + per-commit files + total diff + sign-offs
git log --oneline origin/develop..HEAD && echo '---' && \
git log --format="%H%n%s%n%b%n---" origin/develop..HEAD && echo '---' && \
for commit in $(git rev-list --reverse origin/develop..HEAD); do \
  echo "=== $(git log --oneline -1 $commit) ==="; \
  git diff-tree --no-commit-id --name-only -r $commit; echo; \
done && echo '---' && \
git diff --stat origin/develop..HEAD && echo '---' && \
git log origin/develop..HEAD | grep '^Signed-off-by' | sort -u
```

Then inspect per-file diffs for files that need hunk-level splitting:

```bash
git diff origin/develop..HEAD -- path/to/file
```

### Step 2: Plan Logical Grouping

Group changes by **independence** — minimize inter-commit dependencies.

#### amd-smi API Cascade Ordering (use for API / feature work)

When a PR touches multiple layers of the codebase, order commits to
follow the cascade so each commit builds on the previous and can compile
independently:

| Order | Layer | Directories |
|-------|-------|-------------|
| 1 | Public + internal headers | `include/amd_smi/amdsmi.h`, `rocm_smi/include/rocm_smi/*.h` |
| 2 | C/C++ implementation | `rocm_smi/src/`, `src/amd_smi/amd_smi.cc`, `src/nic/` |
| 3 | Python bindings | `py-interface/amdsmi_wrapper.py`, `py-interface/amdsmi_interface.py` |
| 4 | CLI tool | `amdsmi_cli/` (parser, commands, helpers) |
| 5 | Tests & examples | `tests/`, `example/` |
| 6 | Docs & changelog | `CHANGELOG.md`, `docs/` |
| 7 | Build / packaging | `CMakeLists.txt`, `cmake_modules/`, `RPM/`, `DEBIAN/`, `pyproject.toml` |

#### Common Grouping Strategies

| PR Type | Strategy | Typical # Commits |
|---------|----------|-------------------|
| New API feature | By cascade layer | 4–6 |
| CLI bug fixes | By independent fix | 2–4 |
| Cross-cutting refactor | By subsystem | 2–3 |
| Single focused fix | Squash all into 1 | 1 |
| Mixed feature + docs + tests | Feature, tests, docs/changelog | 3 |

#### Principles

- Each commit should compile independently when possible
- Commits should be revertable without breaking other commits
- Related changes to the same logical unit belong together
- `CHANGELOG.md` is usually its own commit (easy to amend during review)
- If a file was changed then reverted across commits (net-zero diff), skip it during restaging

### Step 3: Create Safety Backup

```bash
git branch backup/BRANCH_NAME
```

### Step 4: Soft Reset to Merge Base

```bash
MERGE_BASE=$(git merge-base origin/develop HEAD)
git reset --soft "$MERGE_BASE"
git reset HEAD          # unstage everything to working tree
```

### Step 5: Selectively Stage and Commit

**Whole files** (when every hunk belongs to the same commit):

```bash
git add include/amd_smi/amdsmi.h rocm_smi/include/rocm_smi/rocm_smi.h
```

**Partial files** (when a single file has hunks for different commits):

```bash
# Interactive — y=stage, n=skip, s=split hunk, e=edit hunk
git add -p path/to/file
```

For complex splits where hunks are interleaved, `git add -e` opens the
diff in an editor for precise line-by-line control — more reliable than
`git add -p` when hunk boundaries don't align with logical groupings.

**Commit** using the message format below. Repeat for each logical group.

### Step 6: Verify Integrity

```bash
# CRITICAL — total diff must be identical (no output = pass)
diff <(git diff origin/develop..HEAD) \
     <(git diff origin/develop..backup/BRANCH_NAME)

# Confirm expected commit count
echo "Commits: $(git log --oneline origin/develop..HEAD | wc -l)"

# No leftover unstaged or staged changes
git status --short
git diff --cached --stat
```

### Step 7: Cleanup

```bash
git branch -D backup/BRANCH_NAME   # only after Step 6 passes
```

---

## Commit Message Format

**REQUIRED SUB-SKILL:** Use the `amdsmi-commit-and-pr-conventions` skill for the commit
title and body format — it is the single source of truth for the
Conventional Commits `type(amdsmi):` title, the bulleted body, the
no-ticket-in-body rule, and brevity caps. Apply it to every commit you create here.

Quick reminder while restaging (see the skill for the full convention):

- Subject: `type(amdsmi): imperative summary`, ≤72 chars, no trailing period
- Body: blank line, verb-first bullets, no `ROCM-NNNNN` refs in the body
- `Signed-off-by` required; preserve original authors and `Co-authored-by:`

---

## Handling Complex Cases

### Overlapping hunks in the same file

Common in amd-smi: `amdsmi_parser.py`, `amdsmi_commands.py`, `amd_smi.cc`,
and `rocm_smi_gpu_metrics.cc` often accumulate fixup hunks across many commits.
Use `git add -p` with split (`s`) or edit (`e`) to separate hunks that belong
to different logical commits. Review the full per-file diff first to plan which
hunks go where.

### Net-zero changes (added then reverted)

These files won't appear in the final `git diff` — skip them during restaging.
Verify by checking `git status --short` produces no output at the end.

### Pre-commit hooks

amd-smi uses pre-commit with: **codespell**, **trailing whitespace**,
**clang-format**, **ruff format**, and **gersemi**. Each `git commit` triggers
them automatically. If hooks modify staged files, those modifications are
included in the commit. This is normal — the hooks enforce project style.

**If a hook reformats staged content** and you need to amend:

```bash
git add -u && git commit --amend --no-edit
```

Do **not** use `--no-verify` to skip hooks — the formatting they enforce is required.

### Preserving authorship

The `Signed-off-by` trailer must match the original author(s). The git
author email may differ from the sign-off (e.g., corporate vs personal).
Extract the actual trailers from the existing commits:

```bash
git log origin/develop..HEAD | grep '^Signed-off-by' | sort -u
```

If multiple people authored changes being squashed together, keep each
sign-off and add `Co-authored-by` trailers.

---

## Safety Notes

- **Always** create a backup branch before starting
- **Always** verify with `diff` that the total changeset is identical after restructuring
- **Never** force-push without user confirmation — ask first
- If something goes wrong: `git reset --hard backup/BRANCH_NAME`

---

# Finishing the Branch

After commit restructuring (or any time implementation is complete), guide the user through integration. Verify tests → detect environment → present options → execute → clean up.

**Announce at start:** "Implementation complete. Running through the finishing checklist."

## Step F1: Verify Tests Pass

Before offering any integration option, confirm a clean test baseline:

```bash
# C++ GTest (if hardware available)
cd build && ./tests/amd_smi_test/amdsmitst --gtest_brief=1

# Python
cd tests/python_unittest && python3 -m pytest -q

# Pre-commit
pre-commit run --all-files
```

If any of these fail → STOP. Report the failures. Do not present integration options until tests are green. Loop the user back to `systematic-debugging` if the failures are unexpected.

## Step F2: Detect Environment

```bash
GIT_DIR=$(cd "$(git rev-parse --git-dir)" 2>/dev/null && pwd -P)
GIT_COMMON=$(cd "$(git rev-parse --git-common-dir)" 2>/dev/null && pwd -P)
BRANCH=$(git branch --show-current)
```

| State | Menu | Cleanup |
|-------|------|---------|
| `GIT_DIR == GIT_COMMON` (normal checkout) | Standard 4 options | No worktree to clean up |
| `GIT_DIR != GIT_COMMON`, named branch | Standard 4 options | Worktree cleanup if it matches the `rocm-systems-*` sibling-dir convention |
| `GIT_DIR != GIT_COMMON`, detached HEAD | Reduced 3 options (no local merge) | No cleanup (externally managed) |

## Step F3: Present Options

**Normal / named-branch worktree — exactly these 4 options:**

```
Implementation complete and tests pass. What would you like to do?

1. Merge back to develop locally
2. Push and create a Pull Request (gh pr create)
3. Keep the branch as-is (I'll handle it later)
4. Discard this work

Which option?
```

**Detached HEAD — 3 options:**

```
Implementation complete. You're on a detached HEAD (externally managed workspace).

1. Push as a new branch and create a Pull Request
2. Keep as-is
3. Discard this work

Which option?
```

Keep it concise — no extra explanation. The user knows what each option does.

## Step F4: Execute the Choice

### Option 1 — Merge to develop Locally

Note: per `CLAUDE.md`, PRs target `develop` (not `main`). Local merges follow the same convention.

```bash
MAIN_ROOT=$(git -C "$(git rev-parse --git-common-dir)/.." rev-parse --show-toplevel)
cd "$MAIN_ROOT"

git checkout develop
git pull --ff-only origin develop
git merge --no-ff "$BRANCH"

# Re-verify tests on the merged result
cd projects/amdsmi  # adjust if not in rocm-systems
# Run the test suite again
```

Then run Step F5 (cleanup) and finally:

```bash
git branch -d "$BRANCH"
```

### Option 2 — Push and Open PR

Per user rules: **never push without explicit approval**. Confirm before pushing.

```bash
git push -u origin "$BRANCH"

gh pr create \
  --base develop \
  --title "<title>" \
  --body "$(cat <<'EOF'
## Summary
<2-3 bullets of what changed>

## Test Plan
- [ ] <verification steps>

## Cascade
<which layers changed, if API work>
EOF
)"
```

**Do NOT clean up the worktree** — the user needs it alive to iterate on PR feedback.

### Option 3 — Keep As-Is

Report: `Keeping branch <name>. Worktree preserved at <path>.` Stop here.

### Option 4 — Discard

**Require typed confirmation** (per user rules — destructive operation):

```
This will permanently delete:
- Branch <name>
- All commits: <list from git log --oneline origin/develop..HEAD>
- Worktree at <path>

Type 'discard' to confirm.
```

Wait for exact `discard`. If confirmed, run Step F5 then:

```bash
git branch -D "$BRANCH"
```

## Step F5: Cleanup Worktree

**Runs for Options 1 and 4 only. Options 2 and 3 always preserve the worktree.**

```bash
WORKTREE_PATH=$(git rev-parse --show-toplevel)
```

- If `GIT_DIR == GIT_COMMON` → normal repo, no worktree. Done.
- If `WORKTREE_PATH` matches the `rocm-systems-*` sibling-dir convention from `amdsmi-using-git-worktrees` (i.e. `$(dirname "$WORKTREE_PATH")/rocm-systems` exists as the main checkout) → we own cleanup:

```bash
MAIN_ROOT=$(git -C "$(git rev-parse --git-common-dir)/.." rev-parse --show-toplevel)
cd "$MAIN_ROOT"
git worktree remove "$WORKTREE_PATH"
git worktree prune
```

- Otherwise → the harness owns it. Do not remove.

## Finishing Quick Reference

| Option | Merge | Push | Keep Worktree | Cleanup Branch |
|--------|-------|------|---------------|----------------|
| 1. Merge locally | yes | — | — | yes (after merge) |
| 2. Create PR | — | yes | yes | — |
| 3. Keep as-is | — | — | yes | — |
| 4. Discard | — | — | — | yes (force) |

## Red Flags — STOP

- `cd`-ing into another worktree/checkout, or restructuring a branch other than the current one, without explicit user approval for that scope
- Tests failing when presenting options (must be green first)
- Pushing without explicit user approval (user rule)
- Force-push without explicit user approval (user rule)
- Merging without re-running tests on the merged result
- Deleting the branch before removing the worktree (`git branch -d` will fail)
- Running `git worktree remove` while CWD is inside that worktree
- Cleaning up a worktree that doesn't match the `rocm-systems-*` convention (harness-owned)
- Discarding without typed `discard` confirmation
