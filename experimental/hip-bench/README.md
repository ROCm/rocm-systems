# hip-bench

Tiny HIP benchmark that times kernel-dispatch and HIP-graph workloads at
increasing batch sizes. Designed to be paired with an LD_PRELOAD tracer
(such as `experimental/hip-trace-lite`) so you can measure the
end-to-end runtime cost of tracing.

## What it measures

Each outer iteration runs 7 segments. The number after `=` is the
launch count for that segment.

| Segment | Description |
|---|---|
| `kernel,1`    | 1 noop kernel launch + sync |
| `kernel,10`   | 10 noop kernel launches + sync |
| `kernel,1000` | 1000 noop kernel launches + sync |
| `graph,1`     | Pre-built graph of 1 chained kernel node — launch + sync |
| `graph,10`    | Pre-built graph of 10 chained kernel nodes — launch + sync |
| `graph,100`   | Pre-built graph of 100 chained kernel nodes — launch + sync |
| `graph,1000`  | Pre-built graph of 1000 chained kernel nodes — launch + sync |

Outer loop: 10 iterations. Total: 70 CSV rows.

Timing is `std::chrono::steady_clock` around the launch loop + sync.
This captures launch overhead, GPU execution, and sync wait — i.e.,
exactly what a user feels.

Warmup pass runs one full sweep before the timed loop and is discarded.

## Build

Requires a HIP/ROCm install (uses hipcc via CMake's HIP language).

```bash
cmake -B build experimental/hip-bench
cmake --build build -j
```

Artifact: `build/hipbench`.

## Run

Baseline (no tracer):
```bash
./build/hipbench > baseline.csv 2> baseline.log
```

With hip-trace-lite preloaded:
```bash
LD_PRELOAD=/path/to/libhiptracelite.so \
HTL_OUTPUT_FILE=/tmp/run.htl \
./build/hipbench > traced.csv 2> traced.log
```

CSV format on stdout:
```
iteration,mode,count,ns
0,kernel,1,12345
0,kernel,10,67890
...
9,graph,1000,1234567
```

Status messages (warmup / per-iteration / complete) go to stderr.

## Comparing tracers

Run baseline and traced runs separately, then diff:
```bash
./build/hipbench > a.csv
LD_PRELOAD=…/libhiptracelite.so HTL_OUTPUT_FILE=/tmp/run.htl ./build/hipbench > b.csv
# join on (mode,count), compute per-segment mean over the 10 iterations,
# subtract → tracing overhead.
```

## Caveats

- Each segment uses the same single-threaded null stream (`stream=0`).
- The graph nodes are *chained* (linear dependency), so 1000 nodes
  serialise on the GPU. That makes the timing dominated by per-node
  launch and dependency-walk cost, not by GPU peak throughput.
- The kernel is `__global__ void noop() {}`. The host-side launch path
  is the only meaningful work per packet; the device-side work is ~0.
