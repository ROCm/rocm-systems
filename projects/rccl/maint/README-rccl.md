# RCCL Maintenance Scripts

`README.md` in this directory is a verbatim copy of NVIDIA's and describes
NVIDIA's workflow. This file covers the scripts RCCL adds, and the places where
RCCL's tooling deliberately does not match what that file says.

Run everything below from the repository root unless stated otherwise.

## Which files RCCL formats

`projects/rccl/.pre-commit-config.yaml` is the single owner of the answer. The
`clang-format` hook's `files:` pattern defines the suffixes
(`.c .cc .cpp .cu .cuh .h .hpp`) and limits the scope to `src/`.
`.clang-format-ignore` still owns which *paths* within that scope are exempt,
because only `clang-format` itself can apply that file.

Two other places look like they answer the same question and do not:

- `README.md` lists `.cc .cu .cuh .h .h.in`.
- `run-clang-format.sh` collects `.cc .cu .cuh .h`.

Both are byte-identical copies of upstream. They disagree with each other —
`.h.in` cannot be matched by that script's `find -name '*.h'`, and in any case
`.clang-format-ignore` exempts `src/nccl.h.in` explicitly — and they disagree
with the hook, which also covers `.c`, `.cpp` and `.hpp`. The fix belongs
upstream. Forking either file here would buy a cosmetic correction in exchange
for a merge conflict on every NCCL sync, so they are left alone and the hook is
authoritative.

`.h.in` is formatted by nothing. `clang-format` cannot infer a language from
that suffix, and `src/nccl.h.in`, the one that matters, is ignored anyway.

## Running the checks

Install `pre-commit` once (`pip install pre-commit`, or
`uv tool install pre-commit`), then:

```sh
# run the hooks automatically on `git commit`, against your staged files
pre-commit install --config projects/rccl/.pre-commit-config.yaml

# full sweep
pre-commit run --config projects/rccl/.pre-commit-config.yaml --all-files

# just what you changed
pre-commit run --config projects/rccl/.pre-commit-config.yaml \
  --files $(git diff --name-only origin/develop...HEAD)
```

From inside `projects/rccl` the `--config` becomes unnecessary, because
`pre-commit` picks up the `.pre-commit-config.yaml` sitting in the directory it
is run from, and with no arguments it checks your staged changes:

```sh
cd projects/rccl
# staged changes
pre-commit
```

`pre-commit install` writes `.git/hooks/pre-commit`, which is not tracked, so
each person installs it for themselves. It is per *clone*, not per working tree:
hooks live in the common git directory, so every worktree of a clone shares the
one hook, and installing from any of them installs for all of them. The hook
records the `--config` path it was given, so install it from the worktree whose
path you want baked in. `git commit --no-verify` skips the hooks;
`SKIP=clang-format git commit` skips one.

Expect `--all-files` to report things. Most hooks here are advisory on purpose
and say what they would change without changing it; the config explains why for
each one.

The baseline whitespace hooks are the exception: they block, and they fix rather
than report. A failure from one means it has already rewritten the file in your
working tree, so there is nothing to go and edit by hand — `git diff` to see
what it did, then `git add` the result and commit again.

## `ci-pre-commit.sh`

The CI wrapper around the ordinary-stage hooks, used by the `pre-commit` job in
`.github/workflows/rccl-format-and-lint.yml`. It runs them over the range a pull
request changed and writes a summary to `$GITHUB_STEP_SUMMARY`. Most of those
hooks are advisory and so always pass, which is the point of the summary:
otherwise their output only exists inside a collapsed log of a green step.

```sh
projects/rccl/maint/ci-pre-commit.sh              # origin/develop..HEAD
projects/rccl/maint/ci-pre-commit.sh BASE HEAD    # an explicit range
```

Unlike `ci-clang-tidy.sh` it installs nothing and adds no git remote, so it is
safe to run in your normal clone; it does expect `pre-commit` on `PATH` and
stops with a clear message if it is missing. The summary is written from an
`EXIT` trap, so a run that dies on a bad config or an unreachable hook
repository still says so rather than leaving an empty summary.

## `run-clang-tidy.sh`

Runs `clang-tidy` over RCCL's hipified sources and reports only findings in
files RCCL owns.

```sh
projects/rccl/maint/run-clang-tidy.sh                      # RCCL-owned sources
projects/rccl/maint/run-clang-tidy.sh --build-dir DIR      # reuse a build tree
projects/rccl/maint/run-clang-tidy.sh --strict             # exit non-zero on findings
projects/rccl/maint/run-clang-tidy.sh --include-upstream   # also report NCCL-owned code
projects/rccl/maint/run-clang-tidy.sh src/graph/tuning.cc  # matching files only
```

Findings are advisory: without `--strict` the script exits 0 even when
`clang-tidy` reports something. The enabled checks live in
`projects/rccl/.clang-tidy`.

Unlike `clang-format`, `clang-tidy` needs a parsed translation unit, so the
script configures a build and runs the hipify step if `--build-dir` has no
compile database. That takes a few minutes, but needs no GPU and compiles
nothing. Results land in `projects/rccl/build/clang-tidy/` by default:

| File                      | Contents                                         |
|---------------------------|--------------------------------------------------|
| `clang-tidy-all.txt`      | every finding, RCCL-owned and NCCL-owned         |
| `clang-tidy-findings.txt` | the RCCL-owned subset, which is what is reported |
| `clang-tidy-stderr.txt`   | the driver's stderr, for when a run dies         |

Filtering out NCCL-owned files needs the `nccl` remote, and the script stops
rather than silently widening its scope if that remote is missing:

```sh
git remote add nccl git@github.com:NVIDIA/nccl.git
git remote set-url --push nccl no_push
git fetch nccl
```

Set `NCCL_REF` to compare against something other than `nccl/master`, or pass
`--include-upstream` to skip the filter entirely.

The script prefers ROCm's `clang-tidy` (`/opt/rocm/lib/llvm/bin/clang-tidy`); a
stock upstream build cannot parse ROCm's HIP headers. Set `CLANG_TIDY` to
override.

Reported paths are mapped back out of the hipify staging tree, so they name
files you can actually open. The two renames CMake applies — `.cu` to `.cu.cpp`
and a `_tmp` suffix for colliding basenames — are accounted for on both sides of
the ownership comparison.

This is also the `clang-tidy` pre-commit hook, which is manual-stage precisely
because of the build step: it has no business running on `git commit`, and
`pre-commit` would hand it a changed-file list it ignores anyway.

```sh
pre-commit run --config projects/rccl/.pre-commit-config.yaml \
  --hook-stage manual clang-tidy
```

## `ci-clang-tidy.sh`

The sibling of `ci-pre-commit.sh` for the manual-stage hook, used by the
`clang-tidy` job in `.github/workflows/rccl-format-and-lint.yml`. It installs
the build dependencies, adds and fetches the `nccl` remote, installs
`pre-commit` into a virtualenv, runs the hook, and writes a summary to
`$GITHUB_STEP_SUMMARY`. The summary is written from an `EXIT` trap, so a run
that dies while installing packages or fetching upstream still says so rather
than leaving an empty summary.

It mutates the checkout — it adds a git remote and installs apt packages — so do
not point it at your working clone. Reproduce a CI run in a container instead:

```sh
podman run \
    --interactive \
    --rm \
    --tty \
    --volume "$PWD:/src" \
    --workdir /src \
    rocm/dev-ubuntu-26.04:10.1.0-full \
    projects/rccl/maint/ci-clang-tidy.sh
```

To run `clang-tidy` in your own tree, use `run-clang-tidy.sh` directly.
