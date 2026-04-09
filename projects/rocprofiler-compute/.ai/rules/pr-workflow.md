# PR Workflow

Follow this methodology precisely when creating GitHub pull requests to produce
clean, reviewable PRs.

## Workflow

### 1. Understand the full scope of changes

Run these in parallel:

- `git status` (never use `-uall`)
- `git diff <base-branch>...HEAD` to see the full diff from base
- `git log --oneline <base-branch>..HEAD` to see all commits in the branch
- `git remote -v` to identify the correct repository

Look at ALL commits, not just the latest one.

### 2. Infer the repo's PR template from recent PRs

Never hardcode the PR template. Always infer it from recent PRs:

- Use GitHub MCP tools to search for recent rocprofiler-compute PRs in the repo.
- Read 2-3 recent PR bodies to identify the required sections and format.
- Match the template exactly — section names, ordering, and checklist format.

The template may change over time. Always check recent PRs for the latest format.

### 3. Ask the user for JIRA ID

Before drafting, ask the user:
- "Do you have a JIRA ID for this PR?"
- If yes, include it in the JIRA ID section.
- If no, leave the JIRA ID section empty (keep the heading, leave body blank).
- JIRA ID should never be specified using url, it should always be just the ticket id without the hyperlink part

### 4. Draft the PR title and body, then confirm with user

Present the full PR (title + body with all sections) to the user and wait for
approval. Do NOT create the PR without confirmation.

**Title rules:**

- Prefix with the project name: `[rocprofiler-compute]`
- Keep under 72 characters
- Use imperative mood ("Add X" not "Added X")

**Body rules:**

- **Motivation**: Describe what this PR does and why, in 2-3 concise lines.
  Do not complain about previous behavior — focus on the positive framing of
  the current change. Highlight bugfixes as a distinct line starting with
  "Bugfix:" to make them stand out during review.
- **Technical Details**: Bullet list of key changes. Only include what a
  reviewer needs to know — omit obvious refactoring details. Keep each point
  to one line where possible.
- **JIRA ID**: The JIRA ticket ID (e.g., `AIPROFCOMP-123`), or empty if none.
- **Test Plan**: Concrete steps to verify the change works.
- **Test Result**: Fill in if tests were run, leave empty if pending.
- **Submission Checklist**: Check off items that apply.

**Reference style:**

- When referencing code from other projects as inspiration, use a hyperlink
  to the specific file and line on GitHub. Format as:
  [`function_name()`](https://github.com/org/repo/blob/branch/path/to/file.py#L127)
- Keep references concise — one to two lines max.

### 5. Identify the correct repository and create the PR

- Identify the correct remote repository from `git remote -v`. Do not assume
  the repo name — verify it. The rocprofiler-compute project lives within the
  `rocm-systems` monorepo.
- Ensure the branch is pushed to the remote before creating the PR.
- Use GitHub MCP tools to create the PR.
- When using MCP tools, pass the body as a plain string with actual newlines,
  NOT with escaped `\n` characters — escaped newlines render as literal `\n`
  in the PR body and break markdown formatting.

### 6. Verify and fix formatting

- After creation, read back the PR to verify markdown renders correctly.
- If formatting is broken (e.g., literal `\n`, HTML-escaped characters),
  update the PR body immediately using the update PR tool.

### 7. Return the PR URL

Always return the PR URL so the user can
review it.

## Tone guidelines (learned from user)

- Be concise. If you can say it in one line, don't use three.
- Do not use negative framing about old code ("The old function was broken").
  Use positive framing about the new behavior ("Add total ranks detection").
- Separate bugfixes from feature work — reviewers scan for these differently.
- Do not add refactoring commentary that is obvious from the diff.
- Do not duplicate the same information in Motivation and Technical Details.

## What NOT to do

- Do not create a PR without user confirmation of title and body.
- Do not create a PR to the wrong base branch (default: `develop`).
- Do not create a PR on the wrong repository.
- Do not use escaped `\n` in PR body strings passed to MCP tools.
- Do not hardcode the PR template — always infer it from recent PRs.
- Do not skip asking the user for JIRA ID.
- Do not include sensitive information (API keys, tokens) in PR descriptions.
- Do not force-push or modify commits as part of PR creation.
