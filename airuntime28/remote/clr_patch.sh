#!/bin/bash
# Regenerate airuntime28-nt-blit.patch from the commit, and verify the two agree.
#
# The change used to live only as uncommitted working-tree edits plus a loose
# .patch file, which a stray `git checkout` would have destroyed and which had no
# way to stay in step with each other. Now the commit is the source of truth and
# this script derives the patch from it. Run it after amending the commit; never
# hand-edit the patch.
set -euo pipefail
cd "$(dirname "$0")"

WT=${WT:-~/airuntime28-clr}
BRANCH=users/victzhan/AIRUNTIME-28-nt-blit
PATCH=$(cd "$(dirname "$0")" && pwd)/airuntime28-nt-blit.patch
UPSTREAM=${UPSTREAM:-~/rocm-systems}

cd "$WT"
head=$(git rev-parse --abbrev-ref HEAD)
if [ "$head" != "$BRANCH" ]; then
  echo "worktree $WT is on '$head', expected '$BRANCH'"
  exit 1
fi

# The base is the merge-base with develop, not HEAD^.
#
# HEAD^ was right while the branch was a single commit on a fixed base, and wrong
# the moment anyone pressed "Update branch" on the PR: that lands a merge commit
# whose first parent is our change, so HEAD^ would have made the patch contain
# everything develop had gained instead of what we added. The merge-base is what
# a PR diffs against and stays correct however the branch is updated.
git fetch -q "$(git config --get branch."$BRANCH".remote || echo origin)" develop 2>/dev/null || true
base=$(git merge-base origin/develop HEAD)
{
  echo "# Generated from $BRANCH by remote/clr_patch.sh."
  echo "# Do not hand-edit: regenerate instead, or the branch and this file will drift."
  echo "# base:   $base"
  echo "# tip:    $(git rev-parse HEAD)"
  echo "#"
  echo "# The tip may be a merge of develop; the base is the merge-base with"
  echo "# develop, so this patch is what the PR adds and nothing else."
  echo "#"
  git diff "$base" HEAD
} > "$PATCH"
echo "wrote $PATCH ($(wc -l < "$PATCH") lines)"
# The patch's own stat, not `git show HEAD` - on a merge commit that would print
# the combined diff, i.e. everything develop brought in, which looks alarming and
# says nothing about the change.
git diff --stat "$base" HEAD | sed 's/^/  /'

# Applying it to a clean tree at the base must reproduce the commit exactly.
# Without this the patch could be stale and nobody would find out until a build
# silently measured the wrong kernel.
TEST=/tmp/airuntime28_patch_test
rm -rf "$TEST"
git -C "$UPSTREAM" worktree add --detach "$TEST" "$base" >/dev/null 2>&1
trap 'cd /tmp; git -C "$UPSTREAM" worktree remove --force "$TEST" >/dev/null 2>&1 || true; git -C "$UPSTREAM" worktree prune' EXIT

cd "$TEST"
git apply --check "$PATCH"
git apply "$PATCH"
if diff <(git diff) <(git -C "$WT" diff "$base" HEAD) >/dev/null; then
  echo "verified: the patch applies to $base and reproduces the branch's change exactly"
else
  echo "MISMATCH: the patch does not reproduce the branch's change"
  exit 1
fi
