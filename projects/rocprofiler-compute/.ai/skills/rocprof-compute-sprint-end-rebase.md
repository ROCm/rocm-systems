# Sprint-end rebase: linearize rocprofiler-compute-develop onto develop

Follow **[`AGENTS.md`](../../AGENTS.md)** and the full redirect chain it references.

## Goal

Linearize `rocprofiler-compute-develop` (rcd) onto `develop` at sprint end, dropping the
`develop -> rcd` sync merge commits, then force-push in place.

`develop` requires linear history. rcd is the ongoing integration branch and is kept current
by periodically merging `develop` into it, so it accumulates
`Merge branch 'develop' into rocprofiler-compute-develop` commits. The sprint drop PR
(head=rcd, base=develop) can only merge once rcd is flattened. Preserve authorship (no squash).

## Key facts

- Every sync merge on rcd is `develop -> rcd`. Linearizing = replaying only rcd's non-merge
  commits (the sprint work) onto the latest `develop`.
- The rebase-from point is the merge-base of `develop` and rcd, NOT `develop`'s current tip.
  Get it authoritatively with `git merge-base origin/develop rocprofiler-compute-develop`.
  (`gh pr view <PR> --json baseRefOid` returns `develop`'s tip, which only equals the
  merge-base when develop has not advanced past rcd's last sync merge, so do not rely on it.)
- The final rocprofiler-compute tree at the linearized tip MUST byte-match the pre-rebase tip.
  Linearizing changes history shape, not content. This is the oracle that gates the push.

## DO NOT use `-X theirs` (or `-X ours`)

A blanket merge strategy silently corrupts content. In a rebase, `-X theirs` resolves each
conflicted hunk toward the commit being replayed and discards the new-base side with no stop.
When a prior partial drop already landed some sprint PRs on develop under different SHAs, those
PRs' lines conflict on replay and `-X theirs` reverts develop's newer content to an older
intermediate. Nothing re-applies it, so the tip ships a regression that passes the rebase but
fails tests / pre-commit. Resolve conflicts per commit instead (see step 5).

## Steps

1. `git fetch origin develop rocprofiler-compute-develop`
2. `MERGE_BASE=$(git merge-base origin/develop rocprofiler-compute-develop)`
3. Backup: `git tag backup-rcd-<date> rocprofiler-compute-develop` (this tag is the oracle and
   the restore point. Keep it until the push is validated.)
4. Sanity-list before rewriting:
   - sprint commits (will replay): `git log --no-merges --oneline $MERGE_BASE..rocprofiler-compute-develop`
   - merges (will drop): `git log --merges --oneline $MERGE_BASE..rocprofiler-compute-develop`
5. Flatten, resolving conflicts commit-by-commit (NO `-X` strategy):
   `git rebase --onto origin/develop $MERGE_BASE rocprofiler-compute-develop`
   - On each stop, resolve sensibly. Most conflicts are the handful of sprint PRs whose end
     state already reached develop via a prior partial drop. Reconcile so the result keeps the
     sprint's final intent, then `git rebase --continue`.
   - `git config rerere.enabled true` first so repeated resolutions replay automatically.
6. Verify (all must pass BEFORE pushing):
   - ORACLE (must be empty): `git diff backup-rcd-<date> HEAD -- projects/rocprofiler-compute/`
     If non-empty, a conflict was mis-resolved. Fix it. Do not push.
   - merges == 0: `git rev-list --merges --count origin/develop..rocprofiler-compute-develop`
   - commit count == the non-merge count from step 4
   - linear on develop: `git merge-base --is-ancestor origin/develop rocprofiler-compute-develop`
   - authorship intact (original PR authors, not the rebaser):
     `git log --format='%an <%ae>' origin/develop..rocprofiler-compute-develop | sort -u`
7. Force-push in place (branch is rewritten, NOT deleted):
   `git push --force-with-lease origin rocprofiler-compute-develop`

## Notes

- Never squash; plain rebase keeps author fields (new SHAs are expected and fine).
- The oracle in step 6 is non-negotiable. Any resolution strategy is safe behind it; none is
  safe without it. The empty rocprofiler-compute diff proves the tip content is unchanged.
