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
runs inside the rocJITsu emulator.
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

From `emulation/rocjitsu`, copy `configs/gfx950_mi350x.json` to
`my_config.json` and add a `plugins` section to activate the race detector:

``` json
{ "plugins": { "race": {} } }
```

Launch the binary with that configuration, where `BUILD_DIR` is your
rocJITsu build directory:

``` bash
$BUILD_DIR/tools/rocjitsu/rocjitsu --config my_config.json -- /tmp/race_example
```

## Interpret the race report

The following report was captured from the gfx950 example above. PCs,
register assignments, and dispatch IDs can vary between builds and runs.

``` text
RACE kernel=transpose_lds symbol=_Z13transpose_ldsPKiPi dispatch=2 type=LDS access=read reg=508 wave=0 lane=0 wg=0,0,0 conflict=unknown
Race on LDS byte 508 [workgroup (0, 0, 0), wave 0, lane 0]: read-after-write (RAW).
  Current access: read at PC 0x73aee4951728.
  Conflict: transfer from VGPRs into LDS at PC 0x73aee495171c, wave 1.
  Completion for wave 1 is not guaranteed; pending counters: lgkmcnt.
  Waves 1 and 0 have not synchronized this access.
       PC              Instruction
  ==>  0x73aee495171c  ds_write_b32 v0, v1  ; <-- wave 1
       Execution trace below is for wave 0.
       0x73aee495171c  ds_write_b32 v0, v1
       0x73aee4951724  v_sub_u32_e32 v1, 0, v0
  ==>  0x73aee4951728  ds_read_b32 v1, v1 offset:508  ; <-- wave 0 lane 0
END_RACE
```

Each field tells you something specific:

-   **type=LDS** --- the race is on local data share memory (shared
    memory in HIP terms).
-   **reg=508** --- the LDS byte address where the conflict occurred.
-   **wave=0, lane=0** --- the wave and lane that performed the
    conflicting read.
-   **wg=0,0,0** --- the workgroup coordinates.
-   **Marked ds_write_b32** --- the store instruction from wave 1 that
    wrote to LDS byte 508. The unmarked row is wave 0's own execution of
    that store, at the beginning of its trace.
-   **ds_read_b32** --- the load instruction from wave 0 that read the
    same address without an intervening barrier.

The completion line says that wave 1's store still has an unsatisfied
`lgkmcnt` wait requirement. The synchronization line says that waves 1 and 0
have not synchronized this access. These are separate conditions: satisfying
the store's wait requirement in wave 1 would not order wave 0's read after
the store. A missing barrier alone does not tell us whether the store's
wait requirements have been satisfied. The kernel needs `__syncthreads()`
between the write and the cross-wave read either way.

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
$BUILD_DIR/tools/rocjitsu/rocjitsu --config my_config.json -- /tmp/race_example
```

The output should complete with `done` and no `RACE` reports.

## Save reports to a file

For CI pipelines or scripted workflows, create the output directory with
`mkdir -p /tmp/output` and add a `sinks` section to `my_config.json`:

``` json
{ "sinks": { "types": ["file"], "dir": "/tmp/output" } }
```

Rerun with the same command. Reports are written to `/tmp/output/race.log`.

## Next steps

-   Learn how the execution plugin system and sink configuration work in
    [Execution plugin system](../conceptual/execution-plugins.md).
-   Analyze final GPU code objects for missing wait instructions with
    [waitcheck hazard analyzer reference](../reference/waitcheck.md).
