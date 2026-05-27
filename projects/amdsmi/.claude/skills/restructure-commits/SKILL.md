---
name: restructure-commits
description: "Consolidate branch commits into logical, independent groups with clean messages. Use when: squashing commits, rewriting history, cleaning up PR commits, restructuring commit order, preparing for merge."
---

# Restructure Commits — amd-smi

Consolidates a branch's commits into a minimal set of logical, independent
commits with descriptive messages and bullet-pointed bodies, following
rocm-systems / amd-smi conventions.

## When to Use

- Branch has many small "WIP", "Updates", or "Addressed comments" commits
- Commits have tangled dependencies that should be separated
- Preparing a PR for merge with clean history
- Reviewer requests commit cleanup

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

### rocm-systems Convention

The monorepo uses a **bracketed project/ticket prefix** on the subject line.
Commits touching amd-smi code should use one of:

| Prefix | When to Use |
|--------|-------------|
| `[AMD-SMI]` | General amd-smi work with no JIRA ticket |
| `[ROCM-NNNNN]` | Linked to a ROCM JIRA ticket |
| `[SWDEV-NNNNNN]` | Linked to a SWDEV JIRA ticket |

#### Template

```
[AMD-SMI] Short imperative summary

- Bullet describing what changed and why
- Another bullet for a distinct logical change
- Keep bullets concise but informative

Signed-off-by: Full Name <email@amd.com>
```

#### Rules

- **Subject line**: `[PREFIX] imperative summary`, ≤72 chars, no trailing period
- **Body**: blank line after subject, wrapped at 72 chars
- Each bullet starts with a verb (Add, Fix, Remove, Refactor, Implement)
- Include "why" when the change isn't self-evident
- **Signed-off-by** is required — preserve the original author's sign-off
- If multiple authors contributed, include `Co-authored-by:` trailers


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
