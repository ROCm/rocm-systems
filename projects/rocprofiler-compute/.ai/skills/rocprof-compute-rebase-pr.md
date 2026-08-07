# Rebase PR onto origin/rocprofiler-compute-develop

Follow **[`AGENTS.md`](../../AGENTS.md)** and the full redirect chain it references.

## Goal

Cherry-pick the user's commits onto the latest `origin/rocprofiler-compute-develop` so the PR has a clean, linear history. Only the user's own commits should appear in the PR — no merge commits, no commits from other contributors.

> **Note:** We call this "rebasing" but the procedure is actually reset + cherry-pick. A true `git rebase` is impractical when the branch history contains merge commits that cause cascading conflicts.

## Steps

1. **Detect the current branch.**
   - Run: `git rev-parse --abbrev-ref HEAD`
   - If the branch is `rocprofiler-compute-develop` or `develop`, stop — the user must checkout their PR branch first.

2. **Ensure the PR base branch is `rocprofiler-compute-develop`.**
   - Run: `gh pr view --json number,baseRefName` (uses the current branch).
   - If the command fails (no open PR), tell the user to create a PR first or proceed without this check if they confirm.
   - If `baseRefName` is NOT `rocprofiler-compute-develop`, tell the user:
     > The PR base is currently `<baseRefName>`. Please change it to `rocprofiler-compute-develop` on GitHub:
     > PR page → Edit (next to title) → base dropdown → select `rocprofiler-compute-develop` → Save.
   - Wait for confirmation before continuing.

3. **Fetch the latest upstream.**
   - Run: `git fetch origin`

4. **Identify the user's commits.**
   - Run: `git log --reverse --oneline origin/rocprofiler-compute-develop..<branch-name> --no-merges`
   - Present the list and ask the user to confirm which commits are theirs. Do NOT assume — other contributors' commits may be mixed in.
   - Collect the confirmed hashes in the same order shown (oldest-to-newest) for step 6.

5. **Create a safety backup, then reset the branch to the upstream base.**
   - Run: `git branch <branch-name>-backup`
   - Run: `git reset --hard origin/rocprofiler-compute-develop`
6. **Cherry-pick the confirmed commits (oldest first).**
   - Run: `git cherry-pick <hash1> <hash2> ... <hashN>` (all in one command).
   - If a commit is already in `rocprofiler-compute-develop`, the cherry-pick will report "empty commit" — run `git cherry-pick --skip` to continue.
   - If there is a conflict, help the user resolve it, stage the file(s), then run `git cherry-pick --continue` to proceed to the next commit.

7. **Verify the result.**
   - Run: `git log --oneline origin/rocprofiler-compute-develop..HEAD`
   - Confirm only the user's commits appear and there are no merge commits.

8. **Force-push.**
   - Run: `git push --force-with-lease origin <branch-name>`
   - Tell the user: the PR on GitHub may take some time to sync — the file-changed count and diff may not immediately reflect the rebase.
