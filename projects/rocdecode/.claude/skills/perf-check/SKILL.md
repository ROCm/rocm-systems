---
name: perf-check
description: Run the rocDecode performance test and compare to baseline to catch regressions
allowed-tools:
  - Bash(cmake *)
  - Bash(test/perf_regression.py *)
  - Bash(python3 test/perf_regression.py *)
  - Bash(python test/perf_regression.py *)
  - Bash(nproc)
  - Bash(cmd *)
---

Run the rocDecode performance regression check: measure decode FPS and compare it against
the GPU-specific baseline. Execute the steps in order, stopping only if a step fails. Run
from the rocDecode project root (the directory containing `build/`, `samples/`, and
`test/`).

IMPORTANT: Run each command as a SEPARATE Bash tool call. Do not chain commands with && or |.

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

### Step 3 — Run the regression check

Run: `test/perf_regression.py`

If the skill was invoked with a `quick` argument (e.g. `/perf-check quick`), run the fast
variant instead — `test/perf_regression.py --quick` — which measures one stream per leaf
subfolder capped at ≤4K (the 8K streams dominate runtime, so they are skipped) instead of
every stream. Use it for a fast sanity check; use the full run before finalizing. Each run
prints its own elapsed time.

Streams within tolerance pass on a single run; streams that appear to regress are
re-measured (3-run average) to rule out noise before being reported.

Report the final summary box from perf_regression.py to the user, and list any streams
marked REGRESSED (with their measured vs. baseline FPS and delta%).

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
`cmd //c "rmdir /s /q samples/videoDecodePerf/build"`.

### Step 3 — Run the regression check

Run: `python test/perf_regression.py`

If the skill was invoked with a `quick` argument (e.g. `/perf-check quick`), run the fast
variant instead — `python test/perf_regression.py --quick`.

Report the final summary box from perf_regression.py to the user, and list any streams
marked REGRESSED (with their measured vs. baseline FPS and delta%).
