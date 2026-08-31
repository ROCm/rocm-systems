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

base=$(git rev-parse HEAD^)
{
  echo "# Generated from $BRANCH by remote/clr_patch.sh."
  echo "# Do not hand-edit: regenerate instead, or the branch and this file will drift."
  echo "# base:   $base"
  echo "# commit: $(git rev-parse HEAD)"
  echo "#"
  git diff HEAD^ HEAD
} > "$PATCH"
echo "wrote $PATCH ($(wc -l < "$PATCH") lines)"
git show --stat --oneline HEAD | sed 's/^/  /'

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
if diff <(git diff) <(git -C "$WT" diff HEAD^ HEAD) >/dev/null; then
  echo "verified: the patch applies to $base and reproduces the commit exactly"
else
  echo "MISMATCH: the patch does not reproduce the commit"
  exit 1
fi
