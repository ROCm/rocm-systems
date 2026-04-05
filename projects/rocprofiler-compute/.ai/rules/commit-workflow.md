# Commit Workflow

Follow this methodology precisely when creating git commits to produce clean,
reviewable commits.

## Workflow

### 1. Assess changes

Run these commands in parallel to understand the current state:

- `git status` (never use `-uall` flag)
- `git diff` to see staged and unstaged changes
- `git log --oneline -10` to understand the repo's commit message style

### 2. Identify the correct files to stage

- Only stage files relevant to the logical change. Prefer `git add <file1> <file2>`
  over `git add -A` or `git add .`.
- Never stage files that may contain secrets (`.env`, credentials, tokens, etc.).
- Never stage unrelated modifications (e.g., dirty submodules, editor configs).

### 3. Draft the commit message and confirm with user

Draft a commit message following the repo's conventions. Present it to the user
and wait for approval before proceeding. Do NOT commit without confirmation.

**Commit message rules for this repo:**

- Prefix with the project name in brackets: `[rocprofiler-compute]`
- First line: concise summary under 72 characters
- Body: explain the "why", not the "what" — the diff shows the what
- Separate logical points as bullet list items
- Highlight bugfixes explicitly and distinctly from feature changes
- Keep descriptions concise — one to two lines per point, no verbose paragraphs
- Do not include refactoring commentary that is obvious from the diff
- If referencing another project's code as inspiration, keep it to a one-liner
- End with: `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`

**Tone guidelines (learned from user):**

- Do not complain about previous behavior. Describe what the change does, not
  what was wrong before.
- Be concise. Three bullet points are better than six.
- Only highlight changes that a reviewer needs to know: bugfixes, new features,
  and notable design decisions.

### 4. Create the commit

- Use a HEREDOC to pass the commit message to avoid shell escaping issues:

```bash
git commit -m "$(cat <<'EOF'
[rocprofiler-compute] Your title here

- Bullet point one
- Bullet point two

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

### 5. Handle pre-commit hooks

- If pre-commit hooks fail (e.g., ruff, formatting), read the output to
  understand what was auto-fixed or what needs manual fixing.
- Re-stage the fixed files with `git add <fixed-files>`.
- Create a NEW commit — never use `--amend` after a hook failure, as the
  original commit did not succeed and amending would modify the previous commit.
- Repeat until all hooks pass.

### 6. Verify success

- Run `git log --oneline -3` to confirm the commit landed correctly.
- Run `git status` to confirm a clean working tree (for staged files).

## Branch safety

- Never commit directly to `main` or `develop` unless the user explicitly
  instructs you to do so. If the user doesn't specify a branch, ask.
- If a commit was accidentally made on the wrong branch, move it:
  1. `git branch <correct-branch>` (create branch at current HEAD)
  2. `git reset --hard HEAD~1` (reset the wrong branch back)
  3. `git checkout <correct-branch>` (switch to the correct branch)

## What NOT to do

- Do not push to remote unless explicitly asked.
- Do not use `--no-verify` or `--no-gpg-sign` to bypass hooks.
- Do not use interactive flags (`-i`) with git commands.
- Do not create empty commits.
- Do not batch multiple unrelated changes into one commit.
