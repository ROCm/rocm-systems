# hip-bench Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use checkbox (`- [ ]`) syntax. Mirror the smoke-test pattern from `experimental/hip-trace-lite/`.

**Goal:** A standalone HIP benchmark that times kernel-dispatch and HIP-graph workloads at increasing batch sizes and emits CSV results. Designed to be paired with `hip-trace-lite` (or any LD_PRELOAD tracer) so we can compare end-to-end runtime with vs. without tracing.

**Architecture:** Single-binary app `hipbench` built via hipcc (HIP language CMake). One `__global__ void noop()` kernel and a driver that runs each segment in an outer loop of 10 iterations. CSV to stdout, status to stderr.

**Tech Stack:** C++17, CMake 3.16+, HIP (find_package(hip)), hipcc for the .hip source.

**Workload definition:**

| Mode | Counts | What runs |
|------|--------|-----------|
| kernel | 1, 10, 1000 | Loop of `hipLaunchKernelGGL(noop)` then `hipDeviceSynchronize()` |
| graph | 1, 10, 100, 1000 | Pre-built `hipGraph_t` of N chained kernel nodes; `hipGraphLaunch` then sync |

**Segments per outer iteration:** 3 + 4 = 7. Outer iterations: 10. Total CSV rows: 70.

**Output format (stdout):**
```
iteration,mode,count,ns
0,kernel,1,12345
0,kernel,10,67890
...
9,graph,1000,1234567
```

Headers and totals printed once at top; one row per (iteration, segment).

**Status to stderr:** "warming up", "iteration N/10 done", "exit summary".

**Timing source:** `std::chrono::steady_clock`, ns resolution. Each segment's timer starts immediately before the launch loop and stops immediately after `hipDeviceSynchronize()`. This captures launch overhead + GPU exec + sync wait — exactly what a user feels.

**Warmup:** One full pass (all 7 segments) before the timed loop, to amortize first-launch / module-load / driver init costs. Warmup output is discarded.

**Graphs are pre-built once** at startup (not inside the timed loop). The graph_exec is also instantiated once. Each graph segment times `hipGraphLaunch + hipDeviceSynchronize` only.

---

## File structure

```
experimental/hip-bench/
├── CMakeLists.txt
├── README.md
├── plan.md            (this file)
└── src/
    ├── main.cpp       driver, timing, CSV output
    └── kernels.hip    __global__ void noop()
```

Single .cpp + single .hip — keeps the build minimal.

---

## Task 1: CMake skeleton + README stub

**Files:**
- Create: `experimental/hip-bench/CMakeLists.txt`
- Create: `experimental/hip-bench/README.md`

- [ ] **Step 1: Write CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.16)
project(hip-bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find HIP. We need it because the .hip translation unit must be compiled
# with hipcc and linked against hip::host.
list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
find_package(hip REQUIRED CONFIG)

enable_language(HIP)

add_executable(hipbench
    src/main.cpp
    src/kernels.hip
)

set_source_files_properties(src/kernels.hip PROPERTIES LANGUAGE HIP)
target_link_libraries(hipbench PRIVATE hip::host)
target_compile_options(hipbench PRIVATE -Wall -Wextra)
```

- [ ] **Step 2: README stub**

```
# hip-bench — see plan.md
```

(Full README in Task 6.)

- [ ] **Step 3: Configure**

```bash
rm -rf /tmp/hb-build
cmake -B /tmp/hb-build experimental/hip-bench
```

Configure should succeed if HIP is in /opt/rocm. (On a non-ROCm host the configure will fail at find_package(hip REQUIRED) — that's expected; this benchmark is GPU-only.)

- [ ] **Step 4: Commit**

```bash
git add experimental/hip-bench/CMakeLists.txt experimental/hip-bench/README.md
git commit -m "$(cat <<'EOF'
[hip-bench] CMake skeleton + README stub

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: noop kernel (`src/kernels.hip`)

**Files:**
- Create: `experimental/hip-bench/src/kernels.hip`

- [ ] **Step 1: Write kernel**

```cpp
// kernels.hip — minimal kernel for benchmark dispatch overhead.
#include <hip/hip_runtime.h>

extern "C" __global__ void noop() { }
```

`extern "C"` so we can take its address in main.cpp without name-mangling friction (hipcc and clang++ may mangle differently).

- [ ] **Step 2: Sanity build**

```bash
cmake --build /tmp/hb-build 2>&1 | tail -3
```

Will fail at link because main.cpp doesn't exist yet. That's fine — confirm that the .hip TU at least compiles with hipcc (look for `Building HIP object src/kernels.hip.o` line).

- [ ] **Step 3: Commit**

```bash
git add experimental/hip-bench/src/kernels.hip
git commit -m "$(cat <<'EOF'
[hip-bench] add noop kernel

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Driver (`src/main.cpp`)

**Files:**
- Create: `experimental/hip-bench/src/main.cpp`

- [ ] **Step 1: Write the full driver**

```cpp
// main.cpp — hip-bench driver.
//
// Runs a sweep of "submit N kernels + sync" and "launch graph of N nodes + sync"
// segments, repeated 10 times. Emits CSV on stdout.

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" __global__ void noop();

#define HIP_CHECK(expr)                                                                   \
    do {                                                                                  \
        hipError_t _err = (expr);                                                         \
        if (_err != hipSuccess) {                                                         \
            std::fprintf(stderr, "hip error %d (%s) at %s:%d: %s\n",                      \
                         _err, hipGetErrorString(_err), __FILE__, __LINE__, #expr);       \
            std::exit(2);                                                                 \
        }                                                                                 \
    } while (0)

using clk = std::chrono::steady_clock;

static uint64_t ns_since(clk::time_point t0) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count());
}

// --------------------------------------------------------------------------
// Kernel-mode segment: launch N noop kernels back-to-back, then sync.
// --------------------------------------------------------------------------
static uint64_t time_kernel_segment(int n) {
    auto t0 = clk::now();
    for (int i = 0; i < n; ++i) {
        hipLaunchKernelGGL(noop, dim3(1), dim3(1), 0, 0);
    }
    HIP_CHECK(hipDeviceSynchronize());
    return ns_since(t0);
}

// --------------------------------------------------------------------------
// Graph-mode helpers: build a chained graph of N kernel nodes.
// --------------------------------------------------------------------------
struct GraphHolder {
    hipGraph_t      graph     = nullptr;
    hipGraphExec_t  exec      = nullptr;
    int             node_count = 0;
};

static GraphHolder build_graph(int n) {
    GraphHolder gh;
    gh.node_count = n;
    HIP_CHECK(hipGraphCreate(&gh.graph, 0));

    hipKernelNodeParams params{};
    params.func           = reinterpret_cast<void*>(&noop);
    params.gridDim        = dim3(1);
    params.blockDim       = dim3(1);
    params.sharedMemBytes = 0;
    params.kernelParams   = nullptr;
    params.extra          = nullptr;

    std::vector<hipGraphNode_t> nodes;
    nodes.reserve(n);
    for (int i = 0; i < n; ++i) {
        hipGraphNode_t node = nullptr;
        const hipGraphNode_t* deps = nodes.empty() ? nullptr : &nodes.back();
        size_t              ndeps = nodes.empty() ? 0 : 1;
        HIP_CHECK(hipGraphAddKernelNode(&node, gh.graph, deps, ndeps, &params));
        nodes.push_back(node);
    }
    HIP_CHECK(hipGraphInstantiate(&gh.exec, gh.graph, nullptr, nullptr, 0));
    return gh;
}

static void destroy_graph(GraphHolder& gh) {
    if (gh.exec)  HIP_CHECK(hipGraphExecDestroy(gh.exec));
    if (gh.graph) HIP_CHECK(hipGraphDestroy(gh.graph));
    gh.exec  = nullptr;
    gh.graph = nullptr;
}

static uint64_t time_graph_segment(const GraphHolder& gh) {
    auto t0 = clk::now();
    HIP_CHECK(hipGraphLaunch(gh.exec, 0));
    HIP_CHECK(hipDeviceSynchronize());
    return ns_since(t0);
}

// --------------------------------------------------------------------------
// Driver
// --------------------------------------------------------------------------
int main() {
    constexpr int kKernelCounts[] = {1, 10, 1000};
    constexpr int kGraphCounts[]  = {1, 10, 100, 1000};
    constexpr int kIterations     = 10;

    // Pre-build graphs once.
    std::vector<GraphHolder> graphs;
    graphs.reserve(sizeof(kGraphCounts) / sizeof(kGraphCounts[0]));
    for (int n : kGraphCounts) graphs.push_back(build_graph(n));

    // Warmup: one full pass, discarded.
    std::fprintf(stderr, "hipbench: warmup\n");
    for (int n : kKernelCounts) (void)time_kernel_segment(n);
    for (auto& gh : graphs)     (void)time_graph_segment(gh);

    // CSV header.
    std::printf("iteration,mode,count,ns\n");

    for (int it = 0; it < kIterations; ++it) {
        for (int n : kKernelCounts) {
            uint64_t ns = time_kernel_segment(n);
            std::printf("%d,kernel,%d,%llu\n", it, n, (unsigned long long)ns);
        }
        for (auto& gh : graphs) {
            uint64_t ns = time_graph_segment(gh);
            std::printf("%d,graph,%d,%llu\n", it, gh.node_count, (unsigned long long)ns);
        }
        std::fflush(stdout);
        std::fprintf(stderr, "hipbench: iteration %d/%d done\n", it + 1, kIterations);
    }

    for (auto& gh : graphs) destroy_graph(gh);
    std::fprintf(stderr, "hipbench: complete\n");
    return 0;
}
```

- [ ] **Step 2: Build**

Configure if not already, then build:
```bash
cmake -B /tmp/hb-build experimental/hip-bench   # if needed
cmake --build /tmp/hb-build -j 2>&1 | tail -5
```

Expected: `[100%] Built target hipbench` and a binary at `/tmp/hb-build/hipbench`.

NOTE: This task can only be built/run on a host with HIP/ROCm installed. On the WSL2 dev box the build will fail at `find_package(hip)` — that is expected. The build/run validation happens in Task 4 on the remote container.

- [ ] **Step 3: Commit**

```bash
git add experimental/hip-bench/src/main.cpp
git commit -m "$(cat <<'EOF'
[hip-bench] driver: kernel + graph sweep, CSV output

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Build and validate on `bewelton_htl_rock` (banff TheRock container)

**Container:** `bewelton_htl_rock` on `banff-ccs-aus-g05-05.cs-aus.dcgpu`, ROCm 7.13.0.

- [ ] **Step 1: Push source into container**

```bash
ssh bewelton@banff-ccs-aus-g05-05.cs-aus.dcgpu "docker exec bewelton_htl_rock mkdir -p /root/hip-bench"
tar c -C experimental hip-bench | ssh bewelton@banff-ccs-aus-g05-05.cs-aus.dcgpu \
    "docker exec -i bewelton_htl_rock tar x --strip-components=1 -C /root/hip-bench"
```

- [ ] **Step 2: Build inside the container**

```bash
ssh bewelton@banff-ccs-aus-g05-05.cs-aus.dcgpu "docker exec bewelton_htl_rock bash -c '
  cd /root/hip-bench && rm -rf build && cmake -B build . 2>&1 | tail -5 &&
  cmake --build build -j 2>&1 | tail -5'"
```

Expected: clean build, binary at `/root/hip-bench/build/hipbench`.

- [ ] **Step 3: Run baseline (no tracer)**

```bash
ssh bewelton@banff-ccs-aus-g05-05.cs-aus.dcgpu "docker exec bewelton_htl_rock bash -c '
  /root/hip-bench/build/hipbench > /tmp/baseline.csv 2> /tmp/baseline.log
  echo === stderr ===
  cat /tmp/baseline.log
  echo === stdout ===
  cat /tmp/baseline.csv'"
```

Expected: 1 header + 70 data rows, no errors. Iteration N/10 messages on stderr.

- [ ] **Step 4: Run with hip-trace-lite preloaded (sanity check the pairing)**

```bash
ssh bewelton@banff-ccs-aus-g05-05.cs-aus.dcgpu "docker exec bewelton_htl_rock bash -c '
  rm -f /tmp/htl-bench.htl
  LD_PRELOAD=/root/hip-trace-lite/build/libhiptracelite.so \
  HTL_OUTPUT_FILE=/tmp/htl-bench.htl \
  /root/hip-bench/build/hipbench > /tmp/traced.csv 2> /tmp/traced.log
  echo === stderr === && cat /tmp/traced.log
  echo === records === && /root/hip-trace-lite/build/htl_dump /tmp/htl-bench.htl | wc -l
  echo === stdout sample === && head -5 /tmp/traced.csv && echo ... && tail -5 /tmp/traced.csv'"
```

Expected: same 70 CSV rows; the .htl trace contains thousands of records (warmup pass alone = 1+10+1000+1+10+100+1000 = 2122 records, then 10x more for the timed iterations); decoded record count should be on the order of tens of thousands.

- [ ] **Step 5: Commit any tweaks discovered during validation**

If Task 4 surfaces real issues (unlikely for this scope; likely only minor things like "graph build of 1000 nodes is slow — split out of warmup"), commit them. Otherwise no commit for this task.

---

## Task 5: README with build + run instructions

**Files:**
- Modify: `experimental/hip-bench/README.md`

- [ ] **Step 1: Replace stub with full content**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add experimental/hip-bench/README.md
git commit -m "$(cat <<'EOF'
[hip-bench] README: build, run, comparison recipe

Co-Authored-By: Claude Opus 4 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist

- [ ] All 5 tasks have actual code/commands; no placeholders.
- [ ] CMake builds via `find_package(hip)` and `enable_language(HIP)` — same pattern as the existing `hip-trace-lite` smoke binary.
- [ ] Warmup precedes the timed loop and is discarded.
- [ ] Graphs are pre-built once; per-iteration timing covers `hipGraphLaunch + hipDeviceSynchronize` only.
- [ ] CSV has a single header followed by exactly 70 data rows in canonical order.
- [ ] `HIP_CHECK` aborts on any HIP error so silent failures can't poison results.
- [ ] No dependency on hip-trace-lite — `hipbench` runs standalone; tracing is purely an LD_PRELOAD overlay.
