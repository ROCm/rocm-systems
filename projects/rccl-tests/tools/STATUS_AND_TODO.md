# RCCL rocTX + rocprofv3 Performance Tooling

## Instrumentation (`src/common.cu`)

- rocTX push/pop markers bracket the timed launch region in `BenchTime()`.
- Runtime-gated by `RCCL_TESTS_ROCTX=1` (off by default, zero overhead when disabled).
- Marker message encodes `size`, `count`, `type`, `op`, `in_place`, `proc`, `thread`, `ngpus`, `graph`.
- rocTX symbols are loaded via `dlsym`/`dlopen` at first use -- no link-time dependency.

## Tools

### `roctx_perf_run.py` -- Profiled Performance Runner

Launches `rocprofv3 --marker-trace --kernel-trace` around `mpirun <test>_perf` for a
matrix of tests x dtypes x repeats.

Key features:
- Auto-detects GPU count for `--np` via `rocm_agent_enumerator`.
- Derives `mpirun` from `build/CMakeCache.txt` (`MPIEXEC_EXECUTABLE` or `MPI_HOME`),
  falling back to `$MPI_HOME` env var, then `$PATH`.
- Resolves `rocprofv3` from `$ROCM_PATH/bin` first (falling back to `PATH`), avoiding
  stale system installs under `/usr/bin`. Override with `--rocprofv3`.
- Uses a `.tmp-*` staging directory during the run; renames to `YYYYMMDD-HHMMSS` only on
  clean exit. Interrupted runs leave a clearly-marked temp directory.
- Captures comprehensive `metadata.json`: command, environment, `rocm-smi`, git status,
  `ldd` output, and `librccl.so` provenance (path, realpath, md5, RCCL version,
  ROCm/HIP build IDs, embedded git hash).
- Strips ANSI escape codes from profiler log output.
- Output format: CSV (`-f csv`).

Usage:
```
python3 tools/roctx_perf_run.py --test all_reduce [--dtypes float,half] [--repeats 1]
python3 tools/roctx_perf_run.py --list-tests
```

Default perf args: `-b 8 -e 1G -f 2 -w 5 -n 50 -A 1`

### `roctx_analyze.py` -- Trace Correlation and Statistics

Reads rocprofv3 CSV output from a run directory, correlates kernel dispatches to rocTX
markers via timestamp containment, and reports per-(size, place) statistics.

Key features:
- Auto-discovers all `*_marker_api_trace.csv` / `*_kernel_trace.csv` pairs recursively.
- Identifies collective kernels by `ncclDev` in the kernel name.
- Aggregates kernel durations across all ranks and repetitions per (size, in_place) tuple.
- Outlier detection: MAD (modified Z-score, default threshold 3.5) or IQR (default factor 1.5).
- Reports `#kept`, `#outliers`, `min`, `median`, `max` on inliers.
- Computes algorithm bandwidth (`size/time`) and bus bandwidth (with per-collective scaling
  factor) from median kernel duration when the collective type and rank count are known.
- **Multi-run directory support**: when pointed at a top-level run directory containing
  subdirs named `{collective}_{dtype}_rep{N}`, automatically groups by (collective, dtype)
  and produces a separate headed section for each -- no more mixing kernels across
  different collectives. `np` is read correctly from `metadata.json` so bus bandwidth
  is always shown.

Usage:
```
python3 tools/roctx_analyze.py perf-runs/20260306-012345          # single or multi-run
python3 tools/roctx_analyze.py perf-runs/20260306-012345 --outlier iqr
```

## librccl.so Provenance

Each run records in `metadata.json`:
- `path`: the `librccl.so` resolved by `ldd` on the test binary.
- `realpath`: canonical path after symlink resolution.
- `md5`: hash of the actual `.so` file.
- `rccl_version`, `rocm_build`, `hip_build`: extracted from embedded strings.
- `git_hash`: the `rcclGitHash` value (e.g. `branch:shorthash+`), extracted from strings.

## TODO

1. **Visual aids** -- bar-and-whisker charts, side-by-side comparison plots across builds.
2. **Multi-collective runs** -- extend beyond all-reduce to all-gather, reduce-scatter, etc.
3. **Multi-platform** -- run the same battery on at least two platforms.
4. **Per-test `-b` selection** -- some tests require minimum message sizes > 8 bytes.
5. **Algorithm/protocol diagnostics** -- capture algo/proto/channel decisions and correlate
   to bandwidth inflection points.

## Notes

- Timestamp comparison is rank-local only; do not compare absolute timestamps across ranks.
- `LD_LIBRARY_PATH` must be set to use a non-system `librccl.so`.
- The test binary must be built with MPI support (`make -C src MPI=1` or cmake with `-DUSE_MPI=ON`).
