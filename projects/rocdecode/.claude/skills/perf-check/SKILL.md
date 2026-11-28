---
name: perf-check
description: Run the rocDecode performance test and compare to baseline to catch regressions
allowed-tools:
  - Bash(cmake *)
  - Bash(test/perf_regression.py *)
  - Bash(python3 test/perf_regression.py *)
  - Bash(python test/perf_regression.py *)
  - Bash(nproc)
  - Bash(grep ^ROCM_PATH samples/videoDecodePerf/build/CMakeCache.txt)
  - Bash(cmd //c "rmdir *)
---

Run the rocDecode performance regression check: measure decode FPS and compare it against
the GPU-specific baseline. Execute the steps in order, stopping only if a step fails. Run
from the rocDecode project root (the directory containing `build/`, `samples/`, and
`test/`).

IMPORTANT: Run each command as a SEPARATE Bash tool call. Do not chain commands with && or |.

The skill accepts optional arguments:
- `quick` — fast one-stream-per-subfolder check (≤4K only), e.g. `/perf-check quick`
- `--gpu <name>` — specify the GPU name for baseline lookup (e.g. `/perf-check --gpu Navi31`)
- `m0` / `m1` / `m2` / `m3` — output surface memory type for `videoDecodePerf` (`-m`):
  0 device internal, 1 device copied, 2 host copied, 3 decode only (the default).
  e.g. `/perf-check m1`
- `save-baseline` — record the local baseline for this GPU and memory type instead of
  comparing (e.g. `/perf-check m1 save-baseline`); add `force` to overwrite an existing one
- Arguments can be combined: `/perf-check quick m1 --gpu Navi31`

Parse the arguments and pass them to `perf_regression.py` in Step 3: `quick` → `--quick`,
`--gpu <name>` as is, `m<N>` → `--mem-type <N>`, `save-baseline` → `--save-baseline`,
`force` → `--force`. If `save-baseline` and `quick` are both given, stop and tell the user
they cannot be combined (a baseline must cover every stream) instead of running.

First determine the OS and follow the matching section below. Under Git Bash on Windows,
`uname -s` returns `MINGW64_NT-...`; on Linux it returns `Linux`.

---

## Linux

The GPU is detected automatically (via `amd-smi`/`rocm-smi`/`hipInfo`/KFD) and mapped to
the matching column in the baseline (`MI250X`, `MI300X`, `MI300A`, `MI350`, `MI355`,
`Navi31`, `Navi48`). Override with `ROCDECODE_PERF_GPU=<column>` if detection is wrong.

Perf streams and the baseline file are located via `ROCDECODE_PERF_DIR` (default
`$HOME/rocDecodePerformance`). That directory must contain the per-codec stream
subdirectories `AvcPerformance`, `Av1Performance`, `HevcPerformance`, `Vp9Performance`, and
the baseline `rocDecode_perf_baseline.html` (download it from SharePoint first). The
regression threshold is 5% Avg FPS drop, overridable via `ROCDECODE_PERF_TOLERANCE`.

The SharePoint baseline holds decode-only (`m3`) numbers. `m0`, `m1` and `m2` are compared
against a local baseline, `$ROCDECODE_PERF_DIR/local_baselines/<GPU>_m<N>.csv`, which is
recorded on this machine with `save-baseline`. If it does not exist yet, the run is
measure-only and prints the command to record one; pass that on to the user rather than
recording a baseline unasked, because a baseline should come from a known-good build.

### Step 1 — Verify the ROCm toolchain

Run: `test/perf_regression.py --check-rocm`

Confirms `ROCM_PATH` points at a usable ROCm install (default `/opt/rocm`). If it fails,
set `ROCM_PATH` — in `~/.profile`, since skills run cmake in a non-interactive shell that
does not source `~/.bashrc` — and re-run before continuing.

### Step 2 — Build the performance sample

Each sample builds in its own directory. Run these two commands separately:

1. `cmake -S samples/videoDecodePerf -B samples/videoDecodePerf/build`
2. `cmake --build samples/videoDecodePerf/build`

This requires rocDecode to be installed already (`make install`); run the `/validate` skill
first if you have not built and installed the library this session.

Stale-build-dir pitfall: the sample caches `ROCM_PATH` on first configure, and re-running
cmake does not update it. After switching install trees (for example to a newer TheRock
build), `samples/videoDecodePerf/build` keeps building and linking against the old tree, so
the check silently measures the old library. If
`grep ^ROCM_PATH samples/videoDecodePerf/build/CMakeCache.txt` does not match `ROCM_PATH`,
delete `samples/videoDecodePerf/build` before Step 2.

### Step 3 — Run the regression check

Run: `test/perf_regression.py`

Add the options parsed from the skill arguments (e.g.
`test/perf_regression.py --quick --gpu Navi31` or `test/perf_regression.py --mem-type 1`).

`--quick` measures one stream per leaf subfolder capped at ≤4K (the 8K streams dominate
runtime, so they are skipped). Use it for a fast sanity check; use the full run before
finalizing. Each run prints its own elapsed time.

Streams within tolerance pass on a single run; streams that appear to regress are
re-measured (3-run average) to rule out noise before being reported.

Report the final summary box from perf_regression.py to the user, state the memory type
that was measured, and list any streams marked REGRESSED (with their measured vs. baseline
FPS and delta%). For `save-baseline`, report where the baseline was saved instead.

---

## Windows

On Windows the shell is Git Bash, but the build toolchain is MSVC: cmake auto-selects the
Visual Studio multi-config generator, so the build step takes an explicit `--config Release`.
The perf exe lands at `samples/videoDecodePerf/build/Release/videodecodeperf.exe`.

### Environment

Before running the pipeline, the developer's shell must have:

- `ROCM_PATH` pointing at the rocDecode install prefix (default `C:\opt\rocm`). This is
  the tree the samples resolve rocDecode from.
- The install's runtime DLL directories on `PATH`: `%ROCM_PATH%\bin` (holds
  `rocdecode.dll`) and `%ROCM_PATH%\lib\rocm_sysdeps\bin` (holds `rocm_sysdeps_va*.dll`
  and `vaon12_drv_video.dll`).
- FFmpeg's `bin` on `PATH` at BOTH configure and runtime. The samples' `FindFFmpeg.cmake`
  auto-discovers FFmpeg from PATH, so do NOT set `FFMPEG_ROOT` when ffmpeg's bin is already
  on PATH.
- `pandas` and `lxml` installed in the Python interpreter used by the script
  (`pip install pandas lxml`). Without `lxml`, the baseline HTML parser fails.

The GPU is detected via `hipInfo` (extracting `Name:` and `gcnArchName:` fields) since
`amd-smi` and `rocm-smi` may not work on Windows. Override with
`ROCDECODE_PERF_GPU=<column>` if detection produces the wrong label or if the baseline has
no column for the detected GPU.

Perf streams and baseline are located via `ROCDECODE_PERF_DIR` (same variable as Linux).
The default is `$HOME/rocDecodePerformance`; on the dev machine the streams are at
`C:\Movies\rocDecodePerformance` — set the variable accordingly.

### Step 1 — Verify the install prefix

Run: `python test/perf_regression.py --check-rocm`

On Windows use `python` (not `python3`, which is usually the inert Store shim). This
confirms `ROCM_PATH` exists. If it fails, fix `ROCM_PATH` and re-run before continuing.

### Step 2 — Build the performance sample

Run these two commands separately:

1. `cmake -S samples/videoDecodePerf -B samples/videoDecodePerf/build`
2. `cmake --build samples/videoDecodePerf/build --config Release --parallel`

This requires rocDecode to be installed already. If it is not, build and install it first
(run each separately):

1. `cmake -B build`
2. `cmake --build build --config Release --parallel`
3. `cmake --install build --config Release`

CRITICAL stale-build-dir pitfall: cmake configure is a no-op when a `build/` cache already
exists, so a STALE `samples/videoDecodePerf/build` dir silently keeps building against
whatever `ROCM_PATH` it was first configured with. When switching install trees, delete the
stale directory first. On Windows `rm -rf` may be blocked by permissions; use
`cmd //c "rmdir /s /q samples\videoDecodePerf\build"`.

### Step 3 — Run the regression check

Run: `python test/perf_regression.py`

Add the options parsed from the skill arguments (e.g.
`python test/perf_regression.py --quick --gpu Navi31` or
`python test/perf_regression.py --mem-type 1`).

Report the final summary box from perf_regression.py to the user, state the memory type
that was measured, and list any streams marked REGRESSED (with their measured vs. baseline
FPS and delta%). For `save-baseline`, report where the baseline was saved instead.
