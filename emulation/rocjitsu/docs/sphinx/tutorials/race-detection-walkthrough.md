---
myst:
    html_meta:
        "description": "Tutorial walking through race detection with rocjitsu: write a HIP kernel with a missing barrier, run it under the emulator, interpret the race report, and fix the bug."
        "keywords": "rocjitsu, ROCm, race detection, LDS, data race, __syncthreads, barrier, GPU debugging, HIP"
---
# Detect a missing barrier with the race detector

When multiple waves in a workgroup access local data share (LDS) memory
without proper synchronization, the values read are ambiguous. This
tutorial walks you through writing a HIP kernel that has a missing
`__syncthreads()` barrier, running it under the rocjitsu race detector,
interpreting the diagnostic output, and confirming the fix.

You do not need a physical AMD GPU for this workflow. The rocjitsu
emulator runs the kernel entirely on the CPU.

```{note}
This tutorial uses the built-in race detector, an execution plugin that
runs inside the rocjitsu emulator.
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
#include <cstdlib>

__global__ void transpose_lds(const int *in, int *out) {
  __shared__ int tile[128];
  int tid = threadIdx.x;
  tile[tid] = in[tid];
  // BUG: missing __syncthreads() — the read below may see
  // another thread's write before it has completed.
  out[tid] = tile[127 - tid];
}

void check_hip(hipError_t status) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "HIP error: %s\n", hipGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  int *d_in, *d_out;
  check_hip(hipMalloc(&d_in, 128 * sizeof(int)));
  check_hip(hipMalloc(&d_out, 128 * sizeof(int)));
  transpose_lds<<<1, 128>>>(d_in, d_out);
  check_hip(hipGetLastError());
  check_hip(hipDeviceSynchronize());
  check_hip(hipFree(d_in));
  check_hip(hipFree(d_out));
  std::puts("done");
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

Launch the binary with the race detector enabled, where `BUILD_DIR` is your
rocjitsu build directory:

``` bash
$BUILD_DIR/tools/rocjitsu/rocjitsu --plugin race -- /tmp/race_example
```

`--plugin race` requires the race detector to load for this launch. The example
checks HIP errors and exits with a failure status, so a skipped detector cannot
be mistaken for a successful race check.

## Interpret the race report

The following report was captured from the gfx950 example above. PCs,
register assignments, and dispatch IDs can vary between builds and runs.

``` text
RACE kernel=transpose_lds symbol=_Z13transpose_ldsPKiPi dispatch=2 type=LDS access=read reg=508 wave=0 lane=0 wg=0,0,0 conflict=unknown
Race on LDS byte 508 [workgroup (0, 0, 0), wave 0, lane 0]: read-after-write (RAW).
  Current access: read at PC 0x73aee4951728.
  Conflict: transfer from VGPRs into LDS at PC 0x73aee495171c, wave 1.
  Completion for wave 1 is not guaranteed; pending counters: lgkmcnt.
  The detector observed no synchronization between waves 1 and 0 for this access.
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
`lgkmcnt` wait requirement. The synchronization line says that the detector
observed no synchronization between waves 1 and 0 for this access. These are
separate conditions: satisfying the store's wait requirement in wave 1 would
not order wave 0's read after the store. A missing barrier alone does not tell us whether the store's
wait requirements have been satisfied. The kernel needs `__syncthreads()`
between the write and the cross-wave read either way.

Other kernels can use synchronization that the detector does not observe,
such as an LDS barrier phase wait. The synchronization line can then accompany
a false positive; see the race detector's
[limitations](https://rocm.github.io/rocm-systems/rocjitsu/race-detector/#limitations).

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
$BUILD_DIR/tools/rocjitsu/rocjitsu --plugin race -- /tmp/race_example
```

The run should exit successfully with `done` and no `RACE` reports. A HIP
error or a failed process exit does not establish that the kernel is race-free.

## Save reports to a file

For CI pipelines or scripted workflows, create the output directory with
`mkdir -p /tmp/output`, copy `configs/gfx950_mi350x.json` to `my_config.json`,
and add plugin sink settings:

``` json
{ "plugins": { "race": {} }, "sinks": { "types": ["file"], "dir": "/tmp/output" } }
```

Rerun with the config file. Reports are written to `/tmp/output/race.log`:

``` bash
$BUILD_DIR/tools/rocjitsu/rocjitsu --config my_config.json -- /tmp/race_example
```

The plugin writes its summary of emitted reports and suppressed observations
at VM teardown. A successful application can emit complete `RACE` blocks
without a final summary, so a missing summary does not mean zero races or
zero suppressed observations.

## Next steps

-   See the JSON plugin and sink settings in the
    [plugin output examples](https://rocm.github.io/rocm-systems/rocjitsu/plugins/#examples).
-   Analyze final GPU code objects for missing wait instructions with
    [waitcheck hazard analyzer reference](../reference/waitcheck.md).
