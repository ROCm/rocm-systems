# Code Review Skill

Follow **[`AGENTS.md`](../../AGENTS.md)** and the full redirect chain it references.

If given a PR URL or number, first check out its exact `headRefName` in the
current worktree: stop if `git status --porcelain` is non-empty or another
worktree holds that branch, else fetch the head ref and fast-forward onto it
(never `-B`/`reset --hard`). Leave it checked out. No argument: review HEAD.

Understand the changes made in this PR branch. Use the `gh` command to obtain
the PR metadata and base branch, and to fetch the context of all review
comments on the PR.
