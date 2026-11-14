# hip-bench

Tiny HIP benchmark that times API-call, kernel-dispatch and HIP-graph workloads at
increasing batch sizes. Designed to be paired with a tracer so you can measure the
end-to-end runtime cost of tracing: an `LD_PRELOAD` tracer such as
`experimental/hip-trace-lite` on Linux, or an ETW session on Windows.

## What it measures

Each outer iteration runs 9 segments. The number after the mode is the call or
launch count for that segment.

| Segment | Description |
|---|---|
| `api,1000`    | 1000 `hipGetDevice` calls, no GPU work |
| `api,100000`  | 100000 `hipGetDevice` calls, no GPU work |
| `kernel,1`    | 1 noop kernel launch + sync |
| `kernel,10`   | 10 noop kernel launches + sync |
| `kernel,1000` | 1000 noop kernel launches + sync |
| `graph,1`     | Pre-built graph of 1 chained kernel node — launch + sync |
| `graph,10`    | Pre-built graph of 10 chained kernel nodes — launch + sync |
| `graph,100`   | Pre-built graph of 100 chained kernel nodes — launch + sync |
| `graph,1000`  | Pre-built graph of 1000 chained kernel nodes — launch + sync |

Outer loop: 100 iterations by default, overridable as `hipbench <iterations>`.
A traced run can emit an event per API call, so the default sweep hands a tracer
roughly 10M of them; lower the count when that is too many.

The `kernel` and `graph` segments include dispatch and sync time, which is what a
user feels but also swamps the per-call cost of a trace call site. The `api`
segments are the isolator: `hipGetDevice` reads thread-local state and returns, so
the difference between a traced and an untraced run is almost entirely the call
site. Reach for `api` to price a tracer's instrumentation and for `kernel`/`graph`
to price its effect on a real workload.

Timing is `std::chrono::steady_clock` around the launch loop + sync.

Warmup pass runs one full sweep before the timed loop and is discarded.

If no device is visible the `kernel` and `graph` segments are skipped and only the
`api` segments run — a runtime that cannot enumerate a GPU still executes and
traces API call sites, so that measurement stays meaningful.

## Build

Requires a HIP/ROCm install. The ROCm prefix is taken from `ROCM_PATH`, then
`HIP_PATH`, then `/opt/rocm`; override with `-DROCM_PATH=...`.

```bash
cmake -B build experimental/hip-bench
cmake --build build -j
```

Artifact: `build/hipbench` (`build/hipbench.exe` on Windows).

On Windows, configure from a shell with the MSVC environment loaded
(`vcvars64.bat`) and use a single-config generator:

```bat
cmake -G Ninja -B build experimental\hip-bench
cmake --build build
```

The architecture comes from `amdgpu-arch` where `rocm_agent_enumerator` is not
available, so `-DCMAKE_HIP_ARCHITECTURES` is not normally needed.

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

Windows has no `LD_PRELOAD`. HIP tracing there is out-of-process: the runtime is
an ETW provider and the consumer lives in a separate process, so the benchmark
binary is launched unmodified while that session is open. Nothing in this
directory is specific to that arrangement.

CSV format on stdout:
```
iteration,mode,count,ns
0,api,1000,11400
0,kernel,1,12345
0,kernel,10,67890
...
99,graph,1000,1234567
```

A blank line then separates a stats footer carrying min/max/mean/stddev per
`(mode,count)` tuple:
```
section,mode,count,n,min_ns,max_ns,mean_ns,stddev_ns
stats,api,1000,100,10800,12200,11400,721
```

Status messages (device count / warmup / per-iteration / complete) go to stderr.

## Comparing tracers

Run baseline and traced runs separately, then diff:
```bash
./build/hipbench > a.csv
LD_PRELOAD=…/libhiptracelite.so HTL_OUTPUT_FILE=/tmp/run.htl ./build/hipbench > b.csv
# join on (mode,count), compute per-segment mean over the iterations,
# subtract → tracing overhead.
```

Run each A/B pair in both orders. A tracing session degrades as its buffers fill,
so whichever build runs second is otherwise charged for the first.

## Caveats

- Each segment uses the same single-threaded null stream (`stream=0`), so none of
  this says anything about how a tracer scales across threads.
- The graph nodes are *chained* (linear dependency), so 1000 nodes
  serialise on the GPU. That makes the timing dominated by per-node
  launch and dependency-walk cost, not by GPU peak throughput.
- The kernel is `__global__ void noop() {}`. The host-side launch path
  is the only meaningful work per packet; the device-side work is ~0.
- The `api` segments call one API in a tight loop. A tracer whose per-call cost
  varies by API — argument serialisation, say — is not characterised by this.
