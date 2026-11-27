# Performance tests

Manual GPU harnesses for timing `rocprof-compute` profile and analyze.
These are **not** part of the `quick` / `standard` / `full` CTest categories.

## Fibonacci tiled transpose + nbody (`test_perf_transpose_nbody`)

One HipRTC app ([`fib_transpose_nbody_app.py`](fib_transpose_nbody_app.py))
launches an LDS tiled matrix transpose (`transpose_lds`, tile 16x16) and
mini-nbody `update_bodies` in a Fibonacci schedule
(`1t,1n,2t,2n,3t,3n,5t,…`) until `--iters` kernel launches (default 5000).
Device buffers ping-pong: nbody input is transpose output; nbody output is
the next transpose input. The transpose kernel follows the LDS tiling
approach from
[rocm-examples tiling_matrix_transpose](https://github.com/ROCm/rocm-examples/tree/amd-staging/Programming-Guide/Tutorials/Performance-Optimization/tiling_matrix_transpose).
Default `--matrix-dim 32768` allocates two float buffers (~8 GiB) and launches
a `2048x2048` grid of `16x16` blocks so every matrix element is covered; the
app checks this against the device `maxGridSize` before running.

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
do not register or execute this test. The harness is source-tree only and is
not registered when `TEST_FROM_INSTALL=ON`.

### Driver CLI

```bash
# Single instance (this tree) — prints profile / analyze / total seconds
python3 tests/performance/run_perf_transpose_nbody.py --iters 5000 --keep

# Multi-instance comparison — table with delta vs the first path
python3 tests/performance/run_perf_transpose_nbody.py \
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
| `--matrix-dim` | Square matrix side for tiled transpose (default 32768, ~8 GiB for two buffers) |
| `-d` / `--work-dir` | Workload parent directory |
| `--keep` | Retain workload output after the run |
| `--device` | HIP device id for the app |
| `--experimental` | Forward `--experimental` to profile/analyze (e.g. gfx1201) |

The profiled app always comes from this tree; only the `rocprof-compute`
tool path varies across `--rocprof-compute` instances.
