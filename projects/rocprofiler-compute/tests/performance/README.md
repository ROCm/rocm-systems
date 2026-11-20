# Performance tests

Manual GPU harnesses for timing `rocprof-compute` profile and analyze.
These are **not** part of the `quick` / `standard` / `full` CTest categories.

## Fibonacci vcopy + nbody (`test_perf_vcopy_nbody`)

One HipRTC app ([`fib_vcopy_nbody_app.py`](fib_vcopy_nbody_app.py)) launches
`vecCopy` and mini-nbody `update_bodies` in a Fibonacci schedule
(`1v,1n,2v,2n,3v,3n,5v,…`) until `--iters` kernel launches (default 5000).
Device buffers ping-pong: nbody input is vcopy output; nbody output is the
next vcopy input.

Profile selects every top-level block for the live GPU arch except
roofline (block `4` / `roof`), PC sampling, and membw — equivalent to a full
counter profile without the deprecated `--no-roof`.

### CMake / CTest

```bash
cmake -S . -B build \
  -DENABLE_TESTS=ON \
  -DENABLE_PERFORMANCE_TESTS=ON
cmake --build build
ctest --test-dir build -L performance -V --output-on-failure
```

`ENABLE_PERFORMANCE_TESTS` defaults to **OFF**, so ordinary CI `ctest` runs
do not register or execute this test.

### Driver CLI

```bash
# Single instance (this tree) — prints profile / analyze / total seconds
python3 tests/performance/run_perf_vcopy_nbody.py --iters 5000 --keep

# Local gfx1201 / lean env (skip native-tool cmake build)
python3 tests/performance/run_perf_vcopy_nbody.py \
  --experimental \
  --iters 5000 \
  --bodies 81920 \
  --no-native-tool \
  --keep

# Multi-instance comparison — table with Δ vs the first path
python3 tests/performance/run_perf_vcopy_nbody.py \
  --rocprof-compute /path/to/checkout-A \
  --rocprof-compute /path/to/checkout-B \
  --iters 5000 \
  --keep
```

| Flag | Meaning |
|------|---------|
| `--rocprof-compute PATH` | Entrypoint or project root (repeatable) |
| `--iters` | Total kernel launches (default 5000) |
| `--bodies` | N-body particle count (default 81920) |
| `-d` / `--work-dir` | Workload parent directory |
| `--keep` | Retain workload output after the run |
| `--device` | HIP device id for the app |
| `--experimental` | Forward `--experimental` to profile/analyze (e.g. gfx1201) |
| `--no-native-tool` | Forward `--no-native-tool` to profile |

The profiled app always comes from this tree; only the `rocprof-compute`
tool path varies across `--rocprof-compute` instances.
