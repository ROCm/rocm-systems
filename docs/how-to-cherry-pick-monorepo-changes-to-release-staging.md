# How to cherry-pick super-repo changes into release-staging branches

> [!IMPORTANT]
> This document is currently in **draft** and may be subject to change.

When a project has been migrated into the ROCm super-repo, day-to-day work happens on the super-repo’s `develop` branch.
Down-stream teams, however, still consume the original (pre-super-repo) repositories, particularly their `release-staging/rocm-rel-x.y` branches, through a variety of mechanisms.
This document explains how to move a change from the super-repo into those release-staging branches while guaranteeing that every commit on a release-staging branch also exists in the super-repo.

## 1. Land the change in the super-repo's develop branch

1. Create a pull request in `ROCm/rocm-systems` that targets `develop`.
2. When merging, choose **Squash & Merge** (if the change can be represented as a single logical commit).
   Why? A single commit is easier to cherry-pick later.

Result: The commit is now on `ROCm/rocm-systems:develop`.

## 2. Cherry-pick into the super-repo's release-staging branch

1. Create a local branch based on the release-staging branch:

```
$ git checkout -b cherry-pick-foo-rel-x.y origin/release-staging/rocm-rel-x.y
```

2. Cherry-pick the commit:

```
$ git cherry-pick abcd1234
```

3. Resolve any merge conflicts (rare if the branch is close to develop).
4. Push the branch and open a PR that targets `ROCm/rocm-systems:release-staging/rocm-rel-x.y`.
5. Request reviews, obtain approvals, and merge.

## 3. Wait for the automatic “fan-out” sync

Every ~15 minutes, a CI job copies new commits from the super-repo back into the corresponding standalone repositories.

After merging your PR:

1. Monitor the CI job or simply wait ~15 minutes.
2. Go to the original (pre-super-repo) repository and verify the commits have been reflected onto the `develop` and `release-staging/rocm-rel-x.y` branches.

## FAQ

Q : Can I cherry-pick multiple commits at once?
A : Yes, but prefer a squash merge in the super-repo so you only need to pick one.

Q : What if the auto-sync hasn’t copied the commit?
A : Verify the CI status in `rocm-systems`. If failed, ask the infra team; the commit will re-sync after a successful run.

Q : Can I push directly to the release-staging branch?
A : No. Always go through a PR so CI and reviewers can validate the cherry-pick.

Q : What if commits have been pushed to develop that make a cherry-pick incompatible with release-staging?
A : It's likely that this fix/change will also be landed in develop at some point, else we risk divergent features/support. So, it's recommended to still land the change in develop first, and cherry-pick to release-staging, resolving any merge conflicts that arise. If for some reason the develop branch has diverged so far from the release-staging for your component that a cherry-pick is irreconcilable, land the changes in develop and release-staging using fully separate PRs, and add references to the other for traceability.

## Summary

In short:

1. Merge change to super-repo `develop`.
2. Cherry-pick to super-repo `release-staging/rocm-rel-x.y`.
3. Wait for the fan-out sync and verify the changes are reflected in the original repository.

Following this process keeps release branches in sync with the super-repo while allowing critical fixes to flow to down-stream consumers.
