# Relocate rocprofiler-compute from `projects/` to `profilers/`

## Problem statement

This plan describes how to move the rocprofiler-compute subproject inside the
`rocm-systems` monorepo from `projects/rocprofiler-compute` to
`profilers/rocprofiler-compute`, next to the existing `profilers/profiler-hub`.

The move is mechanically simple in isolation but touches three independent systems
that must stay consistent: the monorepo CI, the documentation build pipeline, and the
separate TheRock build repository. A single in-place rename would break CI, because
TheRock builds the live monorepo tree and currently expects the old path.

The plan therefore uses a copy-first sequence:
1. Stand up the new location and repoint CI, config, and docs at it, leaving the old
   location in place.
2. Update TheRock to build the new location, then point rocm-systems at that TheRock
   commit.
3. Freeze the old location, move contributions to the new one, and delete the old
   location.

This keeps CI green at every merged step, because the old path keeps working until
every consumer has moved to the new one.

What does not change: the rocprofiler-compute application logic (no `.py` or `.cpp`
file references the directory path), the CMake build (all paths are computed relative
to the source dir), and all package, binary, install-prefix, container-image, RTD
slug, and published-docs-URL identities (these are names, not directory paths).

## Strategic risks

### R1: rocprofiler-compute would be the first `profilers/` subproject in TheRock
TheRock today references only `projects/<name>` subprojects. `profiler-hub`, the only
existing `profilers/` resident, is not part of TheRock at all, and there is no
`profilers/` path anywhere in TheRock. So there is no established convention for how a
`profilers/` subproject is wired into TheRock; we would be inventing it. Confirm with
the TheRock owners whether `profilers/` subprojects are in scope, and ideally wait
until profiler-hub (or any first `profilers/` resident) is onboarded into TheRock, then
follow that convention. This is a precondition for Part 2.

### R2: the docs auto-rebuild silently watches only `projects/`
The Read the Docs auto-rebuild trigger (`update-docs.yml`) fires only on changes under
`projects/**` and derives the project name from the path. After the move, doc changes
under `profilers/rocprofiler-compute` stop triggering a docs rebuild, with no failing
CI to signal it. This needs a change to the shared docs pipeline, coordinated with the
documentation team. Covered in Part 1.1.

### R3: TheRock builds the live monorepo tree, so a plain rename turns CI red
rocm-systems presubmit CI checks out the live PR tree and builds it through TheRock
(pinned to a fixed TheRock commit) which hardcodes `projects/rocprofiler-compute`. A
straight rename turns that CI red until TheRock is updated and re-pinned. The
copy-first sequence (Parts 1 to 3) avoids this: the old path keeps building until
TheRock is updated to the new path and rocm-systems points at the new TheRock commit.

## Impact at a glance

| Component | What changes | Effort | Risk |
|---|---|---|---|
| rocprofiler-compute application logic (.py/.cpp) | Nothing | None | None |
| CMake build chain | Nothing (paths are relative) | None | None |
| rocprofiler-compute in-tree self-references (docs, docker, readme, `.readthedocs.yaml`, inner pre-commit, `copyright_header_check.py`) | Path strings, 26 files | Low | Low |
| 6 `rocprofiler-compute-*` CI workflows | Repoint path references | Medium | Low, mechanical |
| Root config (`.gitmodules`, root pre-commit, dependabot, labeler, `repos-config.json` category, `therock_matrix.py` + test, therock skip globs) | Repoint path references | Low | Low |
| Docs auto-rebuild pipeline (`update-docs.yml`) | Add `profilers/` support | Low | Medium, silent if missed (R2) |
| TheRock build repo | 1 path edit + submodule bump + re-pin in rocm-systems | Medium | High, cross-repo (R1, R3) |

## Part 1: the actual move (establish `profilers/rocprofiler-compute`)

Goal: create the new location and make it the one that CI builds, lints, and packages,
while leaving `projects/rocprofiler-compute` in place so TheRock (still pinned to its
current commit) keeps building successfully.

Branch handling: the relocation lands on `develop`, because `develop` is the branch
TheRock builds and the monorepo mainline. Populate the new
`profilers/rocprofiler-compute` from the current `rocprofiler-compute-develop` content
so the new location starts from the up-to-date subproject state.

Steps in this PR (single PR to `develop`):

1. Copy the subproject tree to `profilers/rocprofiler-compute` (including its
   submodule directories: `src/vendored/pyyaml`,
   `src/lib/external/{googletest,fmt,json}`). Keep `projects/rocprofiler-compute` in
   place for now.
2. Root `.gitmodules`: add the four submodule entries for the new
   `profilers/rocprofiler-compute/...` paths (the old ones remain until Part 3).
3. Repoint the 6 standalone workflows to the new location (path filters,
   sparse-checkout, working-directory, artifact paths, the formatting `--config`).
   Keep filenames and job/check names unchanged so branch-protection required checks
   stay valid.
   - `rocprofiler-compute-continuous-integration.yml`: 21,29-33,37,45-49,69,73,113,162,180,205
   - `rocprofiler-compute-formatting.yml`: 8,12,51
   - `rocprofiler-compute-ghcr.yml`: 12,16,28,31,54,109,126
   - `rocprofiler-compute-rhel-8.yml`: 9,17,18,22,30,31,97,105,111,119,125
   - `rocprofiler-compute-tarball.yml`: 9,17,18,22,30,31,61,87,90,96,104
   - `rocprofiler-compute-ubuntu-jammy.yml`: 9,17,18,22,30,31,59,67,73,81,87
4. Root config edits:
   - `.pre-commit-config.yaml` (root): the `exclude` prefix for the subproject.
   - `.github/repos-config.json:106`: `category` projects -> profilers (read by TheRock
     CI change-detection and docs-preview tooling to recognize the subproject).
   - `.github/dependabot.yml:35`: docs `directory`.
   - `.github/labeler.yml:65`: glob (and optional label name `:63`).
   - `.github/scripts/therock_matrix.py:23`: the path key in the subtree-to-project map.
   - `.github/scripts/tests/therock_configure_ci_test.py:174,194,258`: fixtures.
   - `.github/scripts/therock_configure_ci.py:132-133`: widen `projects/*` skip globs to
     also cover `profilers/*`.
   - `.github/CODEOWNERS`: auto-generated from the subproject CODEOWNERS; regenerates
     with the new prefix, no manual edit.
5. In-tree self-references inside the new copy (read by tooling or shown to users, so
   they must point at the new path):
   - `.readthedocs.yaml:7,16`
   - inner `.pre-commit-config.yaml:8,10,12,22,24,31,38,39,50,57,59,67,78,81`
   - `tools/copyright_header_check.py:25,26`
   - `docker/Dockerfile.standalone:3`, `Dockerfile.doctest:25`,
     `Dockerfile.therock.tarball:41-44,47`, `Dockerfile.therock.wheel:42-45,48`,
     `Dockerfile.therock.wheel.pytorch:46-49,52`
   - `docker/docker-compose.*.yml:5` (5 files)
   - `docs/install/source-install.rst:109,118` (sparse-checkout / cd instructions)
   - `README.md:32,35`, `CONTRIBUTING.md:5,20`

After this PR: the new location is fully exercised by standalone CI; the old location
is an inert copy that only TheRock (at its current pinned commit) still builds.

## Part 1.1: ensure docs is being built

The published docs URL does not change (it is derived from the project name, not the
directory), and `docs-config.json` is name-keyed, so neither needs editing. The
`/docs-preview` PR workflow auto-adapts because it derives the project from
`repos-config.json` (updated in Part 1). The one required change is the auto-rebuild
trigger:

- `.github/workflows/update-docs.yml`: it triggers only on `paths: projects/**` and
  extracts the project name with `grep '^projects/' | awk -F/ '{print $2}'`. Add
  `profilers/**` to the path filter and handle the `profilers/` prefix in the
  extraction. Without this, doc changes under the new path silently stop rebuilding the
  docs (R2). This file is part of the shared docs pipeline: coordinate the change with
  the documentation team.

Verify by making a trivial docs change under the new path and confirming an RTD build
is triggered.

## Part 2: TheRock changes

Precondition: resolve R1 (confirm the `profilers/` convention with TheRock owners).

TheRock locates the subproject at exactly one place:
`/TheRock/profiler/CMakeLists.txt:223`
(`EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/rocprofiler-compute"`).
Everything else in TheRock (source fetch, artifact descriptors, packaging, test
runners) keys off the submodule as a whole or the install-tree layout, not this path.

Sequence:

1. In TheRock, change `profiler/CMakeLists.txt:223` to point at
   `profilers/rocprofiler-compute`, and bump TheRock's `rocm-systems` submodule to the
   commit from Part 1 (where the new location exists). This produces a new TheRock
   commit that builds cleanly against the new location.
2. In rocm-systems, bump the pinned TheRock commit used by the therock CI workflows
   (the `ref:` in `therock-ci-linux.yml` and `therock-ci-windows.yml`, and the setup
   job in `therock-ci.yml`) to the new TheRock commit from step 1. rocm-systems therock
   CI now builds the new location; the old location still exists, so nothing is red.
3. Update the cosmetic TheRock docs that mention the old path
   (`/TheRock/docs/development/windows_support.md:63`,
   `/TheRock/docs/rfcs/RFC0010-Test-Scripts-Migration.md:179`).

## Part 2.1: freeze the old location and cut contributions to the new one

Once TheRock builds the new location (end of Part 2), the new location becomes
canonical. From this point:

- Freeze `projects/rocprofiler-compute`: no further changes land there.
- All new contributions, including the `rocprofiler-compute-develop` integration
  branch, target `profilers/rocprofiler-compute`.

This ordering matters: contributions only move after TheRock builds the new path, so no
work is ever stranded on a path that is not being built. Announce the cutover so
contributors retarget their in-flight branches.

## Part 3: verify everything uses the new path, then delete the old one

Only after Parts 1, 2, and 2.1 are complete and green:

1. Verify the new location is the sole path in use:
   - The 6 standalone workflows run green on `profilers/rocprofiler-compute`.
   - `git submodule update --init` works for the new submodule paths.
   - Root and inner pre-commit run clean; `.github/CODEOWNERS` shows `/profilers/...`.
   - TheRock CI (at the re-pinned commit) builds rocprofiler-compute from the new path.
   - An RTD build triggers from a docs change under the new path (Part 1.1).
   - The subtree change-detector matches `profilers/rocprofiler-compute`
     (e.g. a dry run of `pr_detect_changed_subtrees.py`).
2. Delete `projects/rocprofiler-compute` and remove its four old submodule entries from
   root `.gitmodules`. This PR targets `develop`. After it merges, the old TheRock
   commit (which built the old path) is no longer referenced by rocm-systems, so
   nothing builds the deleted path.
3. Clean up the remaining cosmetic GitHub repo-path URLs that still say
   `tree/develop/projects/rocprofiler-compute`: `CMakeLists.txt:50,1217`,
   `cmake/rocprofcompute.lua.in:15`, `docs/conf.py:174`, `docs/index.rst:18`,
   `docs/install/quickstart.rst:86,213,216,219`, `docs/how-to/analyze/cli.rst:134`,
   `docs/how-to/profile/mode.rst:63`,
   `docs/tutorial/includes/vector-memory-operation-counting.rst:626`,
   `docs/tutorial/profiling-by-example.rst:10`, `README.md:159`, `CHANGELOG.md:168`.
