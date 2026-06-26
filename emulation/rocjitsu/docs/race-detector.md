# Race Detector

Detects synchronization hazards in AMD GPU kernels. Tracks in-flight memory
events (loads, stores) and reports races when a register or LDS memory is
accessed before the operation that produced it has been properly synchronized.

Currently detects intra-workgroup races — cases where the value read from a
register or LDS is not deterministic due to missing `s_waitcnt` or `s_barrier`
instructions.

## Quick start

This section is a standalone guide to getting up and running with race
detection. For general rocjitsu build and usage instructions, see the
[README](../README.md). For an overview of the plugin system and sink
configuration, see [plugins.md](plugins.md).

Build rocjitsu (see [building.md](building.md) for details). You don't need a
physical GPU — the emulator runs entirely on the CPU.

```bash
cd emulation/rocjitsu
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Example: detecting a missing barrier

Here's a small HIP kernel with a missing `__syncthreads()`. Each thread writes
to shared memory (LDS) and then reads from where another thread wrote — without
a barrier. The reads race with the writes.

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

Compile it with `hipcc` or `amdclang++`. The `--offload-arch` must match the
emulated GPU, which depends on the config file you pass to rocjitsu (e.g.
`amdgpu_cdna4.json` emulates gfx950). If using `amdclang++`, pass `-O1` or
higher — the emulator does not currently support unoptimized (`-O0`) GPU code
objects. `hipcc` defaults to `-O3` so this isn't an issue there.

```bash
hipcc -o /tmp/race_example race_example.hip --offload-arch=gfx950
# or: amdclang++ -O2 -o /tmp/race_example race_example.hip --offload-arch=gfx950
```

Run it under the emulator with `RJ_RACE=1` to enable the race detector:

```bash
RJ_RACE=1 build/tools/rocjitsu/rocjitsu --config configs/amdgpu_cdna4.json -- /tmp/race_example
```

You should see output:

```
RACE type=LDS reg=508 wave=0 lane=0 wg=0,0,0 conflict=unknown
Race on LDS byte 508 [workgroup (0, 0, 0), wave 0, lane 0]
  ==>  ds_write_b32 v0, v1  ; <-- wave 1
       v_sub_u32_e32 v1, 0, v0
  ==>  ds_read_b32 v1, v1  ; <-- wave 0 lane 0
END_RACE
```

This tells you that wave 1 wrote to LDS (`ds_write_b32`) and wave 0 read from
the same address (`ds_read_b32`) without a barrier in between. The fix is to add
`__syncthreads()` between the write and the read.

### Running your own application

Replace the binary path with your application. This works with any ROCm workload
— compiled HIP/HSA binaries, Python scripts using PyTorch or JAX, multi-process
launchers like `torchrun`, etc.

```bash
RJ_RACE=1 build/tools/rocjitsu/rocjitsu --config configs/amdgpu_cdna4.json -- ./my_app
RJ_RACE=1 build/tools/rocjitsu/rocjitsu --config configs/amdgpu_cdna4.json -- python my_script.py
```

To capture reports to a file instead of stderr, set `RJ_SINKS=file` and
`RJ_SINK_DIR`:

```bash
RJ_RACE=1 RJ_SINKS=file RJ_SINK_DIR=/tmp/output \
  build/tools/rocjitsu/rocjitsu --config configs/amdgpu_cdna4.json -- ./my_app
# Reports are written to /tmp/output/race.log
```

## What is a race?

A race occurs when the value read from a register or LDS memory is 'ambiguous'
due to unsynchronized access. By 'ambiguous' we mean that it is theoretically
possible to observe different values on different runs. On AMD GPUs, correct use
of `s_waitcnt` (to wait for a wave's own memory operations to complete) and
`s_barrier` (to synchronize waves within a workgroup) is required to avoid
races. Some examples:

1. A wave issues a global load into a VGPR, then reads that VGPR before issuing
   `s_waitcnt vmcnt(0)`. The load may not have completed, so the read value is
   undefined.
1. One wave in a workgroup writes to an LDS address. Another wave reads from
   the same address without an intervening `s_barrier`. The read may see stale
   data because the write may not have completed from the reader's perspective.

## What this plugin detects

- **VGPR races**: a vector register is read before a pending global or LDS load
  has completed (`s_waitcnt vmcnt` / `s_waitcnt lgkmcnt` insufficient).
- **SGPR races**: a scalar register is read before a pending scalar load has
  completed (`s_waitcnt lgkmcnt` insufficient).
- **LDS races**: an LDS byte is read or written by one wave while another wave
  has an outstanding write to the same byte, without an intervening `s_barrier`.

Detection is at byte granularity: D16 (half-register) loads only flag races on
the affected bytes, and LDS races are tracked per byte.

## How it works

Every in-flight memory operation has an **event** that goes through the
following lifecycle:

1. **ACTIVE** — the operation is in flight.
1. **WAVE_COMPLETE** — `s_waitcnt` has retired the event for the owning wave.
   This means the event is no longer in flight from the perspective of the wave
   that issued the operation, but is still in flight from the perspective of
   other waves in the same workgroup.
1. **RETIRED** — `s_barrier` has synchronized all waves. The event is fully
   retired and, from the perspective of all threads in all wavefronts, the
   operation is complete.

The plugin keeps track, for all registers and LDS memory bytes, of which memory
operations are in flight. When an instruction in the emulator accesses an LDS
byte, there is a check to see what memory events are still in flight that
read/write that byte, from the perspective of the accessing thread. In this way,
RAW (read-after-write) and WAR (write-after-read) hazards can be detected.
Similar logic applies for VGPR and SGPR accesses.

**LDS race detection** uses coarse-grained counters (one per 16-byte chunk) for
fast-path checks, with interval-based overlap scanning as a fallback. Live
events are split by direction so that RAW and WAR hazards are checked
independently.

**VGPR race detection** tracks events per register, using the stored exec mask
to determine which lanes are affected. Tracking is at byte granularity within
each 32-bit VGPR so that D16 instructions do not cause false positives when the
other half is accessed independently.

### Worked example

The transpose kernel above compiles to something like this (simplified):

```asm
ds_write_b32 ...                ; tile[tid] = in[tid]
s_waitcnt lgkmcnt(0)            ; wait for own LDS write
                                ; BUG: no s_barrier here
ds_read_b32 ...                 ; out[tid] = tile[127 - tid]
```

Two waves in the same workgroup execute this code. Each wave's write address
overlaps the other wave's read address (because `127 - tid` crosses the wave
boundary). Below is one possible interleaving — the detector will find the race
regardless of the order the waves execute in:

1. **Wave 0 executes `ds_write_b32`.** The detector allocates an event for the
   LDS write. Status: **ACTIVE**. The event records the byte range written
   (lane 0 writes bytes 0–3 for `tile[0]`) and is added to the live write list.
   Coarse-grained write counts (one counter per 16-byte chunk) for the affected
   region are incremented.

1. **Wave 0 executes `s_waitcnt lgkmcnt(0)`.** This drains wave 0's lgkmcnt
   counter. The write event transitions from **ACTIVE** to **WAVE_COMPLETE**. It
   is now safe for wave 0 to access those bytes, but the event remains in the
   live write list — other waves have not synchronized yet.

1. **Wave 1 executes `ds_write_b32`.** A separate event is allocated for wave
   1's write (lane 63 writes bytes 508–511 for `tile[127]`). Status: **ACTIVE**.

1. **Wave 1 executes `s_waitcnt lgkmcnt(0)`.** Wave 1's event transitions to
   **WAVE_COMPLETE**.

1. **Wave 0 executes `ds_read_b32`.** Lane 0 reads `tile[127 - 0]` = bytes
   508–511, the address wave 1 wrote. The detector validates that no live writes
   overlap:

   - *Fast path*: the write count for the 16-byte chunk containing byte 508 is
     non-zero (wave 1's write is still live). Falls through to slow path.
   - *Slow path*: scans live write events. Finds wave 1's event covering bytes
     508–511. The event is **WAVE_COMPLETE**, not **RETIRED**, and the accessing
     wave (0) differs from the owning wave (1).
   - **Race reported.**

1. **What `s_barrier` would fix.** If an `s_barrier` had appeared between steps
   4 and 5, the detector would flush all **WAVE_COMPLETE** events to
   **RETIRED**: they are removed from the live write list and the chunk-level
   write counts are decremented back to zero. The subsequent read validation
   finds no live writes — no race.

## Directory layout

Source files are under `lib/rocjitsu/src/rocjitsu/vm/plugins/race_detector/`:

```
race_detector/
├── plugin.h/.cpp            rocjitsu plugin adapter (translates hooks to core API)
└── core/                    detection algorithm (does not depend on rocjitsu types)
    ├── race_detector.h/.cpp  main detector: event allocation, validation, retirement
    ├── wave_race_state.h/.cpp  per-wave state: register tracking, waitcnt resolution
    ├── event_registry.h      append-only event store with prefix trimming
    ├── interval_set.h        half-open byte range tracking for LDS
    ├── types.h               core enum types and structs
    ├── common_register.h     SGPR/VGPR classification helpers
    ├── dim3d.h               3D coordinate helpers
    └── profiler_interface.h  optional hook profiling
```

## Tests

Tests are part of the rocjitsu test suite (`emulation/rocjitsu/tests/`):

- `race_detector_tests.cpp` — drives `RaceDetector` and `WaveRaceState` directly
  via `race_test_builder.h`, covering VGPR, SGPR, LDS, D16, DTL, exec mask,
  multi-workgroup, and mixed counter scenarios.
- `interval_set_tests.cpp` — unit tests for `IntervalSet`.
- `hip_race_tests.hip` — end-to-end HIP kernel tests run under the emulator with
  `RJ_RACE=1`.

```bash
# Core detection tests
ctest --test-dir build -R "RaceDetector|IntervalSet"

# End-to-end HIP tests (RJ_RACE=1 is set automatically by ctest)
ctest --test-dir build -R "RaceTest"
```

## Limitations

- **Performance**: rocjitsu emulates GPU execution on the CPU, which is orders
  of magnitude slower than running on actual hardware. Race detection adds
  further overhead on top of emulation. This is a correctness tool, not a
  performance tool — use small inputs and targeted test cases rather than full
  production workloads.

- **Intra-workgroup only**: the detector tracks races within a single workgroup
  (missing `s_waitcnt` and `s_barrier`). It does not detect inter-workgroup
  races, races between dispatches, or host-device synchronization issues.

- **No WAW detection**: write-after-write hazards are not currently flagged.
  This includes both LDS WAW (two waves writing to the same LDS byte without a
  barrier) and VGPR WAW (an ALU instruction overwriting a register that has a
  pending global load).

- **Kernel name resolution**: kernel names in race reports may show as `"?"` if
  symbol information is not available in the code object.

## History

The race detection logic was originally developed as part of **race-emulator**,
a standalone CPU-side GPU assembly emulator that parsed `.s` assembly text
files.

The detection logic is independent of any particular emulation approach — it
operates on abstract memory events (register loads, LDS accesses, waitcnt,
barrier) regardless of how those events are produced. The emulation part of
race-emulator is no longer under development; rocjitsu is used for emulation.
