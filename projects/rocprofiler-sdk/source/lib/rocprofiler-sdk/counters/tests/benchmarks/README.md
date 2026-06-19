# Kernel Replay Benchmarks

Small benchmarks for the kernel-replay memory save/restore path (`memory_tracker` +
`memory_snapshot`). Scope is intentionally narrow: directly-allocated device memory only
(`hipMalloc`), no unified / managed memory.

## Benchmarks

| Name | Kernel | What it validates |
|------|--------|-------------------|
| `VecAdd` | `c[i] = a[i] + b[i]` | Baseline save/restore round-trip; correctness across passes |
| `Saxpy`  | `y[i] = a*x[i] + y[i]` | In-place buffer is fully reverted by restore between passes |

Each benchmark snapshots its device buffers, runs the kernel several times restoring memory between
passes, checks correctness, and prints snap/restore timing.

## Building

```bash
cmake --build build/rocprofiler-sdk --target replay-benchmarks
```

## Running

```bash
cd build/rocprofiler-sdk
./bin/replay-benchmarks
```

Or with CTest:

```bash
ctest -R replay-benchmarks --output-on-failure
```
