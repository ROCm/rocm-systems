---
myst:
    html_meta:
        "description": "Tutorial walking through race detection with rocJITsu: write a HIP kernel with a missing barrier, run it under the emulator, interpret the race report, and fix the bug."
        "keywords": "rocJITsu, ROCm, race detection, LDS, data race, __syncthreads, barrier, GPU debugging, HIP"
---
# Detect a missing barrier with the race detector

When multiple waves in a workgroup access local data share (LDS) memory
without proper synchronization, the values read are ambiguous. This
tutorial walks you through writing a HIP kernel that has a missing
`__syncthreads()` barrier, running it under the rocJITsu race detector,
interpreting the diagnostic output, and confirming the fix.

You do not need a physical AMD GPU for this workflow. The rocJITsu
emulator runs the kernel entirely on the CPU.

```{note}
This tutorial uses the built-in race detector, an execution plugin that
runs inside the rocJITsu emulator (`RJ_RACE=1`). To detect the same
class of LDS races on a physical GPU without the emulator, use the
ConSan DBI sanitizer instead---see
[Detect an LDS data race with ConSan](consan-detect-lds-race.md).
```

## Write the kernel

The `transpose_lds` kernel loads data into a shared-memory tile and
reads it back in reverse order. Without a barrier between the write and
the cross-lane read, one wave might read from a slot that another wave
has not yet written.

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

The key detail is `out[tid] = tile[127 - tid]`. Thread 0 reads
`tile[127]`, which is written by thread 127. If threads 0 and 127 are in
different waves, the read can execute before the write completes.

## Compile the kernel

Compile with `hipcc`, targeting `gfx950` (the default emulated GPU
architecture):

``` bash
hipcc -o /tmp/race_example race_example.hip --offload-arch=gfx950
```

## Run under the race detector

Set `RJ_RACE=1` to activate the race detection plugin, then launch the
binary under rocJITsu:

``` bash
RJ_RACE=1 build/tools/rocjitsu/rocjitsu -- /tmp/race_example
```

## Interpret the race report

The race detector prints a report to stderr that looks like this:

``` text
RACE type=LDS reg=508 wave=0 lane=0 wg=0,0,0 conflict=unknown
Race on LDS byte 508 [workgroup (0, 0, 0), wave 0, lane 0]
  ==>  ds_write_b32 v0, v1  ; <-- wave 1
       v_sub_u32_e32 v1, 0, v0
  ==>  ds_read_b32 v1, v1  ; <-- wave 0 lane 0
END_RACE
```

Each field tells you something specific:

-   **type=LDS** --- the race is on local data share memory (shared
    memory in HIP terms).
-   **reg=508** --- the LDS byte address where the conflict occurred.
-   **wave=0, lane=0** --- the wave and lane that performed the
    conflicting read.
-   **wg=0,0,0** --- the workgroup coordinates.
-   **ds_write_b32** --- the store instruction from wave 1 that wrote to
    LDS byte 508.
-   **ds_read_b32** --- the load instruction from wave 0 that read the
    same address without an intervening barrier.

The detector tracks three lifecycle states for each in-flight memory
operation: *active* (in flight --- accessing the destination is a race),
*wave-complete* (retired by `s_waitcnt` for the owning wave, but still a
race for other waves), and *retired* (synchronized by `s_barrier` ---
safe for all waves). Because no barrier separates the write and the
read, the write event is still active when the read executes.

## Fix the kernel

Add `__syncthreads()` between the write and the cross-lane read:

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
hipcc -o /tmp/race_example race_example.hip --offload-arch=gfx950
RJ_RACE=1 build/tools/rocjitsu/rocjitsu -- /tmp/race_example
```

The output should complete with `done` and no `RACE` reports.

## Save reports to a file

For CI pipelines or scripted workflows, redirect race reports to a file
using the sink system:

``` bash
RJ_RACE=1 RJ_SINKS=file RJ_SINK_DIR=/tmp/output \
  build/tools/rocjitsu/rocjitsu -- /tmp/race_example
```

Reports are written to `/tmp/output/race.log`.

## Next steps

-   Learn how the execution plugin system and sink configuration work in
    [Execution plugin system](../conceptual/execution-plugins.md).
-   Analyze final GPU code objects for missing wait instructions with
    [waitcheck hazard analyzer reference](../reference/waitcheck.md).
-   Detect LDS data races on physical hardware using the ConSan DBI
    sanitizer described in [ConSan GPU LDS sanitizer reference](../reference/consan.md).
