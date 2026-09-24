---
myst:
    html_meta:
        "description": "Tutorial showing how to detect an LDS data race in a HIP kernel using the ConSan sanitizer in rocJITsu."
        "keywords": "rocJITsu, ConSan, LDS, race detection, DBI, shared memory, HIP, AMD, ROCm, GPU sanitizer"
---
# Detect an LDS data race with ConSan

ConSan is a dynamic binary instrumentation (DBI) sanitizer shipped with
rocJITsu that detects data races on AMD GPU Local Data Share (LDS, also
known as shared memory). It intercepts GPU code-object loads through the
HSA tools interface, patches final native machine code at load time, and
reports conflicts at runtime. ConSan does not require rebuilding your
application or having the source code.

This tutorial walks through a complete cycle: writing a HIP kernel with
a deliberate missing barrier, running it under ConSan, interpreting the
race report, fixing the bug, and verifying the clean output.

```{note}
ConSan patches native machine code at load time and runs on a physical
GPU. To detect the same class of LDS races without a physical GPU, use
the built-in race detector inside the rocJITsu emulator instead---see
[Detect a missing barrier with the race detector](race-detection-walkthrough.md).
```

## What ConSan instruments

ConSan instruments **LDS read and write operations** (`ds_load_*`,
`ds_store_*`, and admitted flat/VFLAT accesses whose address provenance
reaches the LDS aperture). For each instrumented site it uses VGPR
temporaries to hold intermediate state such as duplicate-load results,
shadow metadata, or sampled watchpoint entries.

ConSan does **not** detect global-memory races. Selected atomics and
fences provide ordering evidence for LDS communication, but they are not
general global-memory race instrumentation. For details on ConSan's
scope and the four instrumentation profiles, see
[ConSan GPU LDS sanitizer reference](../reference/consan.md).

## Write a kernel with a missing barrier

The following HIP kernel deliberately omits `__syncthreads()` between a
shared-memory write and a cross-lane read. Each thread writes its value
to `tile[tid]`, then reads from `tile[127 - tid]`---without a barrier,
those reads race with writes from other waves.

``` cpp
// race_example.hip
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void transpose_lds(const int *in, int *out) {
  __shared__ int tile[128];
  int tid = threadIdx.x;
  tile[tid] = in[tid];
  // BUG: missing __syncthreads() — the read below may see
  // another thread's write before it has completed.
  out[tid] = tile[127 - tid];
}

int main() {
  int *d_in, *d_out;
  hipMalloc(&d_in, 128 * sizeof(int));
  hipMalloc(&d_out, 128 * sizeof(int));
  transpose_lds<<<1, 128>>>(d_in, d_out);
  hipDeviceSynchronize();
  hipFree(d_in);
  hipFree(d_out);
  printf("done\n");
}
```

## Build the hook and compile the kernel

Build the ConSan DBI hooks library from your rocJITsu build directory:

``` bash
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_dbi_hooks -j4

export CONSAN_HOOK="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
```

Compile the example kernel. The `--offload-arch` must match the GPU that
will execute the kernel:

``` bash
hipcc -o /tmp/race_example race_example.hip --offload-arch=gfx1201
```

## Run under ConSan

Load the hook through the `HSA_TOOLS_LIB` environment variable and
select a ConSan flavor. SuperCollider is the simplest profile---it
repeats or reads back each supported LDS access and sets an automatic
marker on mismatch:

``` bash
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=supercollider \
  RJ_CONSAN_LOG=1 \
  /tmp/race_example
```

`RJ_CONSAN_LOG=1` enables compact diagnostic output. You can also run
the three MOI engines for structured race detection:

``` bash
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=moi \
  RJ_CONSAN_MOI_ENGINE=inline_shadow \
  RJ_CONSAN_LOG=1 \
  /tmp/race_example
```

None of these profiles require you to choose a register number,
report-buffer size, patch limit, or sampling stride. Ordinary defaults
instrument all admitted supported sites and manage resources
automatically.

## Interpret the output

At `RJ_CONSAN_LOG=1`, first confirm that ConSan loaded transformed
bytes:

``` text
ConSan patch end ... outcome=modified-valid ... patches=N modified=true
ConSan summary ... patches=N modified=true
```

Then check coverage and completeness:

``` text
ConSan coverage ... access=... barrier=... atomic=... fence=...
ConSan analysis verdict ... static_complete=... dynamic_complete=...
```

When a race is detected, the Inline Shadow engine emits bounded
diagnostic records containing:

-   **LDS byte range** --- the four-byte-aligned cell range of the
    conflicting access.
-   **Owner** --- the workitem-derived identity of the wave that
    performed the access.
-   **Epoch** --- a monotonically incrementing value that advances after
    barrier synchronization. Two accesses in the same epoch with
    different owners and at least one write constitute a conflict.
-   **Access kind** --- read or write.
-   **Instruction offset** --- the `.text` byte offset of the
    conflicting instruction.

SuperCollider instead reports that its automatic mismatch marker
changed, indicating that a duplicated or read-back LDS value differed
from the original. It does not attribute a specific happens-before
violation.

For focused workloads, add acceptance guards to turn a vacuous result
into a failure:

``` bash
export RJ_CONSAN_REQUIRE_PATCH=1          # reject if no patches were applied
export RJ_CONSAN_MOI_REQUIRE_RECORDS=1    # MOI: require visible runtime evidence
```

## Fix the bug and verify

Add `__syncthreads()` between the write and the read:

``` cpp
__global__ void transpose_lds(const int *in, int *out) {
  __shared__ int tile[128];
  int tid = threadIdx.x;
  tile[tid] = in[tid];
  __syncthreads();          // barrier: all writes visible before any read
  out[tid] = tile[127 - tid];
}
```

Recompile and rerun:

``` bash
hipcc -o /tmp/race_example race_example.hip --offload-arch=gfx1201

env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=moi \
  RJ_CONSAN_MOI_ENGINE=inline_shadow \
  RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1 \
  RJ_CONSAN_LOG=1 \
  /tmp/race_example
```

`RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1` causes the process to fail if any
conflict diagnostic is emitted, giving you a clear pass or fail signal.
With the barrier in place, the output should show `modified=true` with
patched sites and zero diagnostics.

## Next steps

-   Read [ConSan GPU LDS sanitizer reference](../reference/consan.md) for the full
    set of environment variables, profile descriptions, and coverage
    signals.
-   See [Execution plugin system](../conceptual/execution-plugins.md)
    for an overview of rocJITsu's plugin architecture and how ConSan
    fits into the DBI hooks system.
-   Explore the other instrumentation profiles (Record/Replay, Sampled)
    to understand the precision and overhead tradeoffs, described in
    [ConSan GPU LDS sanitizer reference](../reference/consan.md).
