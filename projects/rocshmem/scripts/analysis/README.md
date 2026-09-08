# scripts/analysis

Two commit-vs-commit comparison tools for AMDGPU compiler-codegen regressions, plus the Python helpers they drive:

- **`resource_usage_compare.sh`** — per-kernel GPU resource usage (VGPR/SGPR/AGPR/scratch/LDS/occupancy) from `-Rpass-analysis=kernel-resource-usage` compiler remarks. Answers "did this change increase register pressure or drop occupancy for kernel X?".
- **`lto_inline_remarks_compare.sh`** — the whole-program AMDGPU LTO inliner's optimization-record remarks (`-Xoffload-linker --plugin-opt=opt-remarks-filename=...`). Answers "did this change flip an inlining decision the LTO inliner was making?".

Both scripts build each commit under test in its own isolated `git worktree` (under `/tmp`), so your working tree — including uncommitted changes — is never touched, and drive the same pipeline: build → extract remarks → CSV → diff → a self-contained interactive HTML dashboard (plus PNG charts for quick viewing). The Python helpers (`resource_usage_to_csv.py`/`_diff.py`/`_dashboard.py`, `lto_inline_remarks_to_csv.py`/`_diff.py`/`_dashboard.py`/`_report.py`) are plain CSV/YAML parsers and chart/dashboard generators with no project-specific logic — they only need whatever CSV/YAML the shell drivers hand them.

## Prerequisites

- The target project must be a **git repository** (both scripts isolate builds via `git worktree add`/`remove` and resolve commits via `git rev-parse`/`merge-base`).
- A ROCm `clang++`/`hipcc` toolchain with the AMDGPU backend.
- `python3` (stdlib only for most of the pipeline).
- `PyYAML` — only needed by `lto_inline_remarks_compare.sh` (parses the linker's opt-remarks YAML).
- `matplotlib` — optional; PNG chart generation is skipped with a warning if it's not installed, everything else still runs.
- `llvm-cxxfilt` or `c++filt` on `PATH` — used to demangle kernel names.

## Using this in another project

These tools don't actually know anything about rocSHMEM — they call out to *a build script you provide* for the actual compile step. To adopt them in a different CMake-based project:

1. Copy the whole `scripts/analysis/` directory so it sits next to your project's top-level `CMakeLists.txt`:
   ```
   <your-project>/
     CMakeLists.txt
     scripts/
       analysis/          <- copied verbatim from here
         resource_usage_compare.sh
         lto_inline_remarks_compare.sh
         *.py
   ```
   Both scripts locate your project root as two directories up from themselves (`scripts/analysis/../..`), so this nesting matters — it's what lets them find `git` and place cache/output dirs correctly.

2. Point the scripts at your build, with either option:

   - **`--config-path PATH`** (recommended for a new adopter) — a direct path to an executable build script, wherever it already lives in your repo. No required directory layout. Relative paths are resolved *inside each commit's isolated worktree* (so the version of the script checked in at that commit is what actually runs); absolute paths are used as-is.
   - **`--build-config NAME`** — rocSHMEM's own convention: `NAME` is searched for as an executable at `scripts/build_configs/NAME` inside the worktree. Use this if you'd rather organize multiple build variants as named scripts under that directory, the way rocSHMEM does.

   Either way, your script is invoked with cwd set to a fresh, empty build directory and is expected to behave like rocSHMEM's own wrappers (see `scripts/build_configs/all_backends` in this repo for a full example):

   ```bash
   #!/bin/bash
   set -e
   # Locate your own source tree relative to this script's checked-out
   # location — adjust the number of "../" to match where you put this
   # script relative to your top-level CMakeLists.txt.
   src_path=$(dirname "$(realpath "$0")")/..

   cmake \
     -DCMAKE_BUILD_TYPE=Release \
     -DGPU_TARGETS=... \
     "$@" "$src_path"          # "$@" carries e.g. -DGPU_TARGETS=<...> forwarded by the compare script
   cmake --build . --parallel "$(nproc)"
   ```

3. Run it:
   ```bash
   ./scripts/analysis/resource_usage_compare.sh \
     --commit1 HEAD~1 --commit2 HEAD \
     --gpu-target gfx942 --config-path ./scripts/my_build.sh

   ./scripts/analysis/lto_inline_remarks_compare.sh \
     --commit1 HEAD~1 --commit2 HEAD \
     --gpu-target gfx942 --config-path ./scripts/my_build.sh
   ```

For rocSHMEM itself, nothing changes — no new options are required; the existing `--build-config all_backends` default keeps working exactly as before.

## Option reference

### `resource_usage_compare.sh`

| Option               | Default                   | Description                                                                       |
| -------------------- | ------------------------- | --------------------------------------------------------------------------------- |
| `--commit1 REF`      | `HEAD` (or PR merge-base) | First commit/branch to measure                                                    |
| `--commit2 REF`      | *(none)*                  | Second commit to diff against `commit1`; omit for a single-commit snapshot        |
| `--pr NUM`           |                           | Fetch GitHub PR #NUM and diff it against its merge-base with `--base-branch`      |
| `--base-branch NAME` | `origin/develop`          | Base branch for `--pr` merge-base resolution                                      |
| `--gpu-target ARCH`  | `gfx950`                  | GPU target architecture                                                           |
| `--build-config CFG` | `all_backends`            | Named build script under `scripts/build_configs/`                                 |
| `--config-path PATH` |                           | Direct path to an executable build script; takes precedence over `--build-config` |
| `--skip-build`       | on                        | No-op flag kept for explicitness — cached builds are reused by default            |
| `--force-rebuild`    | off                       | Force a fresh rebuild even if this commit is cached                               |
| `--match REGEX`      |                           | Pin kernels matching this regex to the top of every report/chart                  |
| `--top N`            | `50`                      | Rows shown per generated chart/CSV                                                |
| `--output-dir DIR`   | See below                 | Where to write the report                                                         |

Output DIR default value is: `<parent-of-project>/resource-usage/<gpu>-<config>-<sha1>-vs-<sha2>/`

```bash
./resource_usage_compare.sh --commit1 d48c64f6e --commit2 3caf8d080 --build-config all_backends
./resource_usage_compare.sh --pr 42 --build-config all_backends
./resource_usage_compare.sh --commit1 HEAD~1 --commit2 HEAD --config-path ./scripts/my_build.sh
```

### `lto_inline_remarks_compare.sh`

| Option               | Default        | Description                                                                       |
| -------------------- | -------------- | --------------------------------------------------------------------------------- |
| `--commit1 REF`      | `HEAD`         | First commit/branch to measure                                                    |
| `--commit2 REF`      | *(none)*       | Second commit to diff against `commit1`; omit for a single-commit snapshot        |
| `--gpu-target ARCH`  | `gfx950`       | GPU target architecture                                                           |
| `--build-config CFG` | `all_backends` | Named build script under `scripts/build_configs/`                                 |
| `--config-path PATH` |                | Direct path to an executable build script; takes precedence over `--build-config` |
| `--force-rebuild`    | off            | Force a fresh rebuild even if cached (same non-determinism caveat as above)       |
| `--top N`            | `20`           | Rows shown in dashboard charts/report panels                                      |
| `--output-dir DIR`   | See below      | Where to write the report                                                         |

Output DIR default value is: `<parent-of-project>/lto-inline-remarks/<gpu>-<config>-<sha1>[-vs-<sha2>]/`

```bash
./lto_inline_remarks_compare.sh --commit1 HEAD~1 --commit2 HEAD --build-config ipc_single
./lto_inline_remarks_compare.sh --commit1 HEAD~1 --commit2 HEAD --config-path ./scripts/my_build.sh
```

## Output layout

Everything lands as a **sibling** of your project directory, never inside your source tree:

```
<parent-of-project>/
  build-cache/<gpu>-<config>-<sha>/                 # scratch cmake build dirs (safe to delete)
  resource-usage/cache/<gpu>-<config>-<sha>/         # durable per-commit CSV + build logs
  resource-usage/<gpu>-<config>-<sha1>-vs-<sha2>/    # report: res-*.csv, res_diff_*.{csv,png}, resource_dashboard.html
  build-cache-lto-remarks/<gpu>-<config>-<sha>/       # scratch build dirs for the LTO tool
  lto-inline-remarks/cache/<gpu>-<config>-<sha>/      # durable per-commit remarks CSV/YAML
  lto-inline-remarks/<gpu>-<config>-<sha1>[-vs-<sha2>]/  # report + lto_inline_dashboard.html
```

Each report directory's `*_dashboard.html` is fully self-contained (data inlined) — open it directly in a browser, no server needed.

## Notes & caveats

- **`--force-rebuild`**: AMDGPU LTO codegen (register allocation, scheduling, symbol layout) is *not* build-to-build deterministic, even for identical source. A real before/after regression decision needs a matched-fresh-pair rebuild of both sides — pass `--force-rebuild` rather than trusting a fresh build against a stale cached one.
- **Device-bitcode measurement** (`resource_usage_compare.sh` only): this script additionally backend-compiles rocSHMEM's whole-program device bitcode artifact (`librocshmem_device_<arch>.bc`) for a second, ungated-inliner set of numbers. This is rocSHMEM-specific — it just prints a "skipping device-bitcode resource-usage measurement" note and continues for any project that doesn't produce a matching artifact.
- **Forced CMake flags** (`lto_inline_remarks_compare.sh` only): this script always appends `-DBUILD_TOOLS=OFF -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF` to whichever build script it resolves. These are rocSHMEM CMake option names that guarantee exactly one `-fgpu-rdc` LTO-linked executable gets built — required so the single `opt-remarks-filename` isn't clobbered by concurrent parallel links. CMake silently ignores unrecognized `-D` flags, so this is harmless for another project, but *your* build script is responsible for the same one-LTO-executable guarantee if you need it.
- **`--config-path` vs `--build-config` caching**: each is cached independently (keyed by the resolved config), so switching between them for the same commit never serves a stale/mismatched result.
- `"Cannot find <name> in worktree"` means your named `--build-config` script is missing or not executable at that path in the checked-out worktree; `"--config-path ... is not an executable file"` means the same for a `--config-path` script.
