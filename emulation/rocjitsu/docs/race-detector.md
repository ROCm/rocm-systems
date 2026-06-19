# Race Detector

Detects synchronization hazards in AMD GPU kernels. Tracks in-flight
memory events (loads, stores) and reports races when a register or LDS
byte is accessed before the operation that produced it has been properly
synchronized.

Currently detects intra-workgroup races — cases where the value read
from a register or LDS location is ambiguous due to missing `s_waitcnt`
or `s_barrier` instructions.

## Quick start

This section is a standalone guide to getting up and running with race
detection. For general rocjitsu build and usage instructions, see the
[README](../README.md). For an overview of the plugin system and sink
configuration, see [plugins.md](plugins.md).

Build rocjitsu (see [building.md](building.md) for details). You don't
need a physical GPU — the emulator runs entirely on the CPU.

```bash
cd emulation/rocjitsu
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Example: detecting a missing barrier

Here's a small HIP kernel with a missing `__syncthreads()`. Each thread
writes to shared memory and then reads from another thread's slot —
without a barrier, those reads race with the writes.

```c++
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

Compile it with `hipcc` or `amdclang++` (the `--offload-arch` must match
the emulated GPU, which is gfx950 by default). If using `amdclang++`,
pass `-O1` or higher — the emulator does not currently support
unoptimized (`-O0`) GPU code objects. `hipcc` defaults to `-O3` so this
isn't an issue there.

```bash
hipcc -o /tmp/race_example race_example.hip --offload-arch=gfx950
# or: amdclang++ -O2 -o /tmp/race_example race_example.hip --offload-arch=gfx950
```

Run it under the emulator with `RJ_RACE=1` to enable the race detector:

```bash
RJ_RACE=1 build/tools/rocjitsu/rocjitsu -- /tmp/race_example
```

You should see output like:

```
RACE type=LDS reg=508 wave=0 lane=0 wg=0,0,0 conflict=unknown
Race on LDS byte 508 [workgroup (0, 0, 0), wave 0, lane 0]
  ==>  ds_write_b32 v0, v1  ; <-- wave 1
       v_sub_u32_e32 v1, 0, v0
  ==>  ds_read_b32 v1, v1  ; <-- wave 0 lane 0
END_RACE
```

This tells you that wave 1 wrote to LDS (`ds_write_b32`) and wave 0
read from the same address (`ds_read_b32`) without a barrier in between.
The fix is to add `__syncthreads()` between the write and the read.

### Running your own application

Replace the binary path with your application. This works with any
ROCm workload — compiled HIP/HSA binaries, Python scripts using PyTorch
or JAX, multi-process launchers like `torchrun`, etc.

```bash
RJ_RACE=1 build/tools/rocjitsu/rocjitsu -- ./my_app
RJ_RACE=1 build/tools/rocjitsu/rocjitsu -- python my_script.py
```

To capture reports to a file instead of stderr (useful for CI or
scripted workflows), set `RJ_SINKS=file` and `RJ_SINK_DIR`:

```bash
RJ_RACE=1 RJ_SINKS=file RJ_SINK_DIR=/tmp/output \
  build/tools/rocjitsu/rocjitsu -- ./my_app
# Reports are written to /tmp/output/race.log
```

## What is a race condition?

A race occurs when the value read from a register or LDS byte is
ambiguous due to unsynchronized access. On AMD GPUs, correct use of
`s_waitcnt` (to wait for a wave's own memory operations to complete)
and `s_barrier` (to synchronize waves within a workgroup) is required
to avoid races. Some examples:

1. A wave issues a global load into a VGPR, then reads that VGPR
   before issuing `s_waitcnt vmcnt(0)`. The load may not have completed,
   so the read value is undefined.
2. Two waves in the same workgroup write to the same LDS address without
   an intervening `s_barrier`. A subsequent read of that address produces
   an ambiguous value because the order of the writes is unspecified.

## What it detects

- **VGPR races**: a vector register is read before a pending global or
  LDS load has completed (`s_waitcnt vmcnt` / `s_waitcnt lgkmcnt`
  insufficient).
- **SGPR races**: a scalar register is read before a pending scalar load
  has completed (`s_waitcnt lgkmcnt` insufficient).
- **LDS races**: an LDS byte is read or written by one wave while
  another wave has an outstanding write to the same byte, without an
  intervening `s_barrier`.

Detection is at byte granularity: D16 (half-register) loads only flag
races on the affected bytes, and LDS races are tracked per byte.

## How it works

Each in-flight memory operation creates an **event** with a lifecycle:

1. **ACTIVE** — the operation is in flight. Accessing the destination
   register or LDS byte is a race.
2. **WAVE_COMPLETE** — `s_waitcnt` has retired the event for the owning
   wave. Safe for the owning wave, but still a race if another wave
   accesses the same LDS byte.
3. **RETIRED** — `s_barrier` has synchronized all waves. The event is
   fully retired and safe for everyone.

**LDS race detection** uses per-byte counters for fast-path checks, with
interval-based overlap scanning as a fallback. Live events are split by
direction so that RAW and WAR hazards are checked independently.

**VGPR race detection** tracks events per register, using the stored
exec mask to determine which lanes are affected. Tracking is at byte
granularity within each 32-bit VGPR so that D16 instructions do not
cause false positives when the other half is accessed independently.

## Directory layout

Source files are under `lib/rocjitsu/src/rocjitsu/vm/plugins/race_detector/`:

```
race_detector/
├── plugin.h/.cpp          rocjitsu plugin adapter (translates hooks to core API)
└── core/                  detection algorithm (does not depend on rocjitsu types)
    ├── race_detector.h/.cpp   main detector: event allocation, validation, retirement
    ├── wave_race_state.h/.cpp per-wave state: register tracking, waitcnt resolution
    ├── event_registry.h       append-only event store with prefix trimming
    ├── interval_set.h         half-open byte range tracking for LDS
    ├── types.h                core enum types and structs
    ├── common_register.h      SGPR/VGPR classification helpers
    ├── dim3d.h                3D coordinate helpers
    └── profiler_interface.h   optional hook profiling
```

## Tests

Tests are part of the rocjitsu test suite (`emulation/rocjitsu/tests/`):

- `race_detector_tests.cpp` — drives `RaceDetector` and `WaveRaceState`
  directly via `race_test_builder.h`, covering VGPR, SGPR, LDS, D16,
  DTL, exec mask, multi-workgroup, and mixed counter scenarios.
- `interval_set_tests.cpp` — unit tests for `IntervalSet`.
- `hip_race_tests.hip` — end-to-end HIP kernel tests run under the
  emulator with `RJ_RACE=1`.

```bash
# Core detection tests
ctest --test-dir build -R "RaceDetector|IntervalSet"

# End-to-end HIP tests (RJ_RACE=1 is set automatically by ctest)
ctest --test-dir build -R "RaceTest"
```

## Limitations

- **Performance**: rocjitsu emulates GPU execution on the CPU, which is
  orders of magnitude slower than running on actual hardware. Race
  detection adds further overhead on top of emulation. This is a
  correctness tool, not a performance tool — use small inputs and
  targeted test cases rather than full production workloads.

- **Intra-workgroup only**: the detector tracks races within a single
  workgroup (missing `s_waitcnt` and `s_barrier`). It does not detect
  inter-workgroup races, races between dispatches, or host-device
  synchronization issues.

- **No WAW detection**: write-after-write hazards are not currently
  flagged. This includes both LDS WAW (two waves writing to the same
  LDS byte without a barrier) and VGPR WAW (an ALU instruction
  overwriting a register that has a pending global load).

- **Kernel name resolution**: kernel names in race reports may show as
  `"?"` if symbol information is not available in the code object.

## History

The race detection logic was originally developed as part of
**race-emulator**, a standalone CPU-side GPU assembly emulator that
parsed `.s` assembly text files.

The detection logic is independent of any particular emulation
approach — it operates on abstract memory events (register loads, LDS
accesses, waitcnt, barrier) regardless of how those events are produced.
The emulation part of race-emulator is no longer under development;
rocjitsu is used for emulation.
