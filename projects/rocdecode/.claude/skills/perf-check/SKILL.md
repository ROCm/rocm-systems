---
name: perf-check
description: Run the rocDecode performance test and compare to baseline to catch regressions
allowed-tools:
  - Bash(cmake *)
  - Bash(test/perf_regression.py *)
  - Bash(python3 test/perf_regression.py *)
  - Bash(nproc)
---

Run the rocDecode performance regression check: measure decode FPS and compare it against
the GPU-specific baseline. Execute the steps in order, stopping only if a step fails. Run
from the rocDecode project root (the directory containing `build/`, `samples/`, and
`test/`).

IMPORTANT: Run each command as a SEPARATE Bash tool call. Do not chain commands with && or |.

The GPU is detected automatically (via `amd-smi`/`rocm-smi`/KFD) and mapped to the matching
column in the baseline (`MI250X`, `MI300X`, `MI300A`, `MI350`, `MI355`, `Navi31`, `Navi48`).
Override with `ROCDECODE_PERF_GPU=<column>` if detection is wrong.

Perf streams and the baseline file are located via `ROCDECODE_PERF_DIR` (default
`$HOME/rocDecodePerformance`). That directory must contain the per-codec stream
subdirectories `AvcPerformance`, `Av1Performance`, `HevcPerformance`, `Vp9Performance`, and
the baseline `rocDecode_perf_baseline.html` (download it from SharePoint first). The
regression threshold is 5% Avg FPS drop, overridable via `ROCDECODE_PERF_TOLERANCE`.

## Step 1 — Verify the ROCm toolchain

Run: `test/perf_regression.py --check-rocm`

Confirms `ROCM_PATH` points at a usable ROCm install (default `/opt/rocm`). If it fails,
set `ROCM_PATH` — in `~/.profile`, since skills run cmake in a non-interactive shell that
does not source `~/.bashrc` — and re-run before continuing.

## Step 2 — Build the performance sample

Each sample builds in its own directory. Run these two commands separately:

1. `cmake -S samples/videoDecodePerf -B samples/videoDecodePerf/build`
2. `cmake --build samples/videoDecodePerf/build`

This requires rocDecode to be installed already (`make install`); run the `/validate` skill
first if you have not built and installed the library this session.

## Step 3 — Run the regression check

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
