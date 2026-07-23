---
name: emulation-rebase
description: Safely analyze and rebase or rebuild an emulation branch when some commits, squash merges, cherry-picks, or equivalent changes have already landed upstream. Use for partial-merge rebases, dropping already-merged commits, resolving replay conflicts, updating a PR branch, or proving that the rebased rocjitsu and Mirage diff preserves the intended changes.
compatibility: Requires git and a clean worktree; network access is needed only when upstream refs must be fetched.
metadata:
  author: ROCm
  version: "1.0"
---

# Emulation partial-merge rebase

Rewrite history only after proving which changes are already upstream. Preserve
the intended patch, provide a recovery point, and validate affected rocjitsu and
Mirage behavior afterward.

## Hard rules

1. Never infer equivalence from commit subject alone. Use ancestry, patch IDs,
   diffs, and final tree behavior.
2. Require a clean worktree, including relevant untracked files, before a
   history rewrite. Do not stash implicitly.
3. Create and report a durable local recovery ref for the original tip before
   rebasing. Do not delete it in this workflow.
4. Never push any commit, branch, tag, rebased history, or other content—and
   never modify a pull request or other GitHub state—without explicit user
   approval immediately before that specific write. A request to rebase,
   commit, update a branch, or prepare a pull request is not publication
   approval. Approval for one write does not authorize another. For an approved
   rewritten-branch push, use
   `--force-with-lease=<branch>:<expected-remote-oid>`, never plain `--force`.
5. Never continue a conflicted rebase without inspecting the full file and the
   intent of both sides. Never resolve generated files by choosing one side or
   hand-editing output; resolve generator inputs and regenerate.
6. Follow `emulation/AGENTS.md`, enforce rocjitsu style, and validate Mirage
   integration when affected.
7. Never leak confidential references or their contents while inspecting
   history or resolving conflicts. Do not quote, name, link, copy, summarize,
   upload, or expose them through paths, metadata, screenshots, logs, prompts,
   generated artifacts, internal terminology, issues, pull requests, reviews,
   tests, commits, or chat. Base recorded rationale on repository history,
   public sources, and reproducible tests; treat uncertain publication status
   as confidential.

## 1. Capture immutable state

Record:

- current branch and tip OID;
- upstream/default branch name, local OID, and remote OID;
- configured tracking branch and expected remote topic OID;
- merge base, worktree status, submodules if any, and recent graph;
- open PR base/head OIDs when the task concerns a PR.

Fetch only the required refs. If fetching would overwrite no local state but
network access is unavailable, continue against recorded local refs and label
them potentially stale.

Create a clearly named recovery branch or tag at the original tip, such as
`backup/<topic>-pre-rebase-YYYYMMDD`. Confirm it resolves to the recorded OID.

## 2. Classify every topic commit

Use [the equivalence checklist](references/commit-equivalence.md). Start from the
old merge base and list topic commits in topological order. For each commit,
classify one of:

- **ancestor:** exact commit is already reachable from new upstream;
- **patch-equivalent:** stable patch identity is upstream, including a
  cherry-pick;
- **squashed/absorbed:** behavior is upstream as part of a different patch;
- **partially absorbed:** only some hunks or behavior landed;
- **topic-only:** still needs replay;
- **uncertain:** evidence is insufficient.

`git cherry` and stable patch IDs are useful signals, not final truth: conflict
resolution, file movement, generated output, or refactoring can alter patch
identity while preserving behavior. For absorbed and partially absorbed
commits, compare the relevant old-parent-to-commit diff against new upstream and
inspect tests/callers.

Present the classification table before rewriting when any commit is partially
absorbed or uncertain. Ask the user only for intent that history and code cannot
answer.

## 3. Choose the least risky transformation

- Use a normal rebase when Git reliably skips exact/equivalent commits and the
  remaining commits are independent.
- Use an interactive rebase when specific commits are proven absorbed and can
  be dropped, reordered, or edited cleanly.
- Use `rebase --onto <new-upstream> <old-base> <topic>` when the old base is
  known and the complete remaining series should replay.
- Preserve intentional merge structure with `--rebase-merges`.
- For deeply interdependent, partially absorbed, or repeatedly conflicted
  history, rebuild the branch from new upstream by replaying selected commits
  or reconstructing the final patch. Keep authorship and meaningful commit
  boundaries where practical.

Before execution, state the proposed command/sequence, commits to drop or edit,
and why each is safe.

## 4. Resolve replay conflicts by intent

For each conflict:

1. inspect the original commit and its parent, current upstream implementation,
   full conflicted files, callers, tests, and later topic commits;
2. decide whether the topic change is already present, still required, or must
   be adapted to an upstream refactor;
3. make the smallest resolution that preserves both upstream fixes and remaining
   topic intent;
4. run a focused check when feasible before continuing;
5. verify no conflict markers or unintended generated/manual edits remain.

Abort rather than guess when behavior or confidential domain intent cannot be
established. The recovery ref must remain available whether the rebase succeeds
or aborts.

## 5. Prove the result

After the rewrite, compare old and new history with `range-diff` and compare
trees against their respective bases. Account for every old topic commit and
every material hunk as upstream-provided, replayed, intentionally removed, or
reworked.

Check specifically for:

- accidentally lost tests, docs, build files, generated outputs, or fixups;
- duplicated changes that Git failed to skip;
- reverted upstream fixes caused by conflict resolution;
- generated output not matching the current generator;
- commits outside `emulation/` that were part of the topic;
- changed public behavior through Mirage.

Run formatting and focused tests first, then the affected component suites from
`emulation/AGENTS.md`. For rocjitsu-Mirage boundaries, run an integration path
through Mirage. Record unavailable tests rather than hiding them.

## 6. Report and optionally publish

Report:

- old tip, new upstream, old merge base, new tip, and recovery ref;
- commit-classification table and transformation used;
- conflict resolutions and any intent assumptions;
- `range-diff`/tree-diff conclusion;
- tests and formatting run, failures, and residual risk;
- whether the branch was pushed.

If a remote update is desired, first record the current remote topic OID and
show the user the local tip, destination, commits/diff to publish, and exact
push mode. Obtain explicit approval immediately before the push. For rewritten
history, use an OID-qualified force-with-lease. Stop if the lease fails; fetch
and reassess instead of overriding another contributor's work.
