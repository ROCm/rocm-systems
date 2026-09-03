# Race Detector

Detects synchronization hazards in AMD GPU kernels. Tracks in-flight memory
events (loads, stores) and reports races when a register or LDS memory is
accessed before the operation that produced it has been properly synchronized.

Currently detects intra-workgroup races — cases where the value read from a
register or LDS is not deterministic due to missing `s_waitcnt` or `s_barrier`
instructions.

## Target scope

The race detector focuses on pre-GFX12 architectures. Its current end-to-end
coverage exercises gfx950 (GFX9/CDNA4) and gfx1151 (GFX11.5/RDNA3.5). GFX12
and later architectures are not supported; their counter, scheduling, and
writeback rules will be modeled when that support is added rather than being
partially anticipated here.

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
`gfx950_mi355x.json` emulates gfx950). If using `amdclang++`, pass `-O1` or
higher — the emulator does not currently support unoptimized (`-O0`) GPU code
objects. `hipcc` defaults to `-O3` so this isn't an issue there.

```bash
hipcc -o /tmp/race_example race_example.hip --offload-arch=gfx950
# or: amdclang++ -O2 -o /tmp/race_example race_example.hip --offload-arch=gfx950
```

Enable the race detector by adding it to the `plugins` section of your
rocjitsu config file (`my_config.json`):

```json
{ "plugins": { "race": {} } }
```

Run it under the emulator:

```bash
$BUILD_DIR/tools/rocjitsu/rocjitsu --config my_config.json -- /tmp/race_example
```

You should see output:

```
RACE kernel=transpose_lds symbol=_Z13transpose_ldsPKiPi dispatch=1 type=LDS access=read reg=508 wave=0 lane=0 wg=0,0,0 conflict=unknown
Race on LDS byte 508 [workgroup (0, 0, 0), wave 0, lane 0]
  ==>  ds_write_b32 v0, v1  ; <-- wave 1
       v_sub_u32_e32 v1, 0, v0
  ==>  ds_read_b32 v1, v1  ; <-- wave 0 lane 0
END_RACE
```

This tells you that dispatch 1 of `transpose_lds` reported a race: wave 1 wrote
to LDS (`ds_write_b32`) and wave 0 read from the same address (`ds_read_b32`)
without a barrier in between. The `kernel` field is a compact display name, and
`symbol` is the exact ELF symbol when rocjitsu can resolve it. If a name cannot
be resolved, rocjitsu reports `?` for the unresolved field. The fix is to add
`__syncthreads()` between the write and the read.

### Running your own application

Replace the binary path with your application. This works with any ROCm workload
— compiled HIP/HSA binaries, Python scripts using PyTorch or JAX, multi-process
launchers like `torchrun`, etc.

```bash
$BUILD_DIR/tools/rocjitsu/rocjitsu --config my_config.json -- ./my_app
$BUILD_DIR/tools/rocjitsu/rocjitsu --config my_config.json -- python my_script.py
```

To capture reports to a file instead of stderr (useful for CI or
scripted workflows), add a `sinks` section to your config:

```json
{
  "plugins": { "race": {} },
  "sinks": { "types": ["file"], "dir": "/tmp/output" }
}
```

```bash
$BUILD_DIR/tools/rocjitsu/rocjitsu --config my_config.json -- ./my_app
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
1. A wave issues a load into a VGPR, then an instruction overwrites that VGPR
   before the load completes. The load may complete later and clobber the
   instruction result.
1. One wave in a workgroup writes to an LDS address. Another wave reads from
   the same address without an intervening `s_barrier`. The read may see stale
   data because the write may not have completed from the reader's perspective.

## What this plugin detects

- **VGPR races**: a vector register is read or overwritten by an instruction
  before a pending global or LDS load has completed (`s_waitcnt vmcnt` /
  `s_waitcnt lgkmcnt` insufficient).
- **SGPR races**: a scalar register is read before a pending scalar load has
  completed (`s_waitcnt lgkmcnt` insufficient).
- **LDS races**: an LDS byte is read by one wave while another wave has an
  outstanding write to the same byte, or written by one wave while another
  wave has an outstanding read of the same byte, without an intervening
  `s_barrier`;
  or a wave reads bytes targeted by its own outstanding direct-to-LDS operation
  before the required `s_waitcnt vmcnt`.

Detection is at byte granularity: D16 (half-register) loads only flag races on
the affected bytes, and LDS races are tracked per byte.

## How it works

Every in-flight memory operation has an **event** that goes through the
following lifecycle:

1. **ACTIVE** — the operation is in flight. Ordinary DS operations issued by
   the same wave remain ordered with respect to each other, so a later
   same-wave DS read or write does not race solely because the earlier DS event
   is still active. Direct-to-LDS VMEM writes still require the owning wave to
   wait for `vmcnt` before reading the destination bytes.
1. **WAVE_COMPLETE** — `s_waitcnt` has retired the event for the owning wave.
   This means the event is no longer in flight from the perspective of the wave
   that issued the operation, but is still in flight from the perspective of
   other waves in the same workgroup.
1. **RETIRED** — `s_barrier` has synchronized all waves. The event is fully
   retired and, from the perspective of all threads in all wavefronts, the
   operation is complete.

An all-ones wait-count field is the architectural “do not wait” value. CDNA's
four-bit `lgkmcnt(15)` and six-bit `vmcnt(63)` therefore retire no events;
`lgkmcnt(14)` and `vmcnt(62)` are the largest values that can impose an
explicit wait. Hardware also prevents counter overflow by stalling issue.

The physical LGKMCNT capacity is shared by every event accounted to LGKMCNT,
including LDS, GDS, scalar-memory, and message operations. Sharing that counter
does not mean those event classes complete in order with each other. The
detector therefore retires only the oldest prefix that is provably complete in
an ordered class. For example, a new LGKM-counted instruction issued after 15
pending local-LDS operations proves that the oldest LDS operation completed,
even when the new instruction is scalar memory or GDS. Mixed-class pressure
that does not identify a completed event remains conservatively pending. VMCNT
is handled the same way for its 63-entry ordered non-FLAT VMEM class.

For example, CDNA cannot issue the final scalar load below while all 15 earlier
LGKM tokens remain outstanding:

```asm
ds_read_b32 v0, v16
ds_read_b32 v1, v16
; ... 13 more ordered LDS reads, through v14 ...
s_load_dword s4, s[2:3], 0
```

The scalar load can issue only after the LGKMCNT value drops below its
four-bit capacity. Because all preceding operations are ordered LDS reads, this
proves that the oldest read into `v0` completed. If those pending operations
belonged to different or unordered classes, the capacity stall would not prove
which individual event completed.

The detector records the target-specific wait-counter family on every event.
It handles both the combined wait fields and the standalone counter forms on
supported targets, for example:

```asm
s_waitcnt vmcnt(0) lgkmcnt(0)  ; combined fields
s_waitcnt_vmcnt null, 0        ; standalone VMCNT form
s_waitcnt_lgkmcnt null, 0      ; standalone LGKMCNT form
```

Counter-capacity backpressure is modeled for the VMCNT and LGKMCNT domains on
CDNA1 through CDNA4 and GFX11. CDNA uses a four-bit LGKMCNT and six-bit VMCNT;
GFX11 uses six-bit fields for both. GFX11 vector stores use the separate VSCNT
domain and therefore do not create VMCNT pressure.

The same completion-order distinction is used for nonzero partial waits; a
zero wait still completes every event on the selected counter. The detector
also uses these classes to determine whether two asynchronous writes to the
same VGPR are ordered.

The plugin keeps track, for all registers and LDS memory bytes, of which memory
operations are in flight. When an instruction in the emulator accesses an LDS
byte, there is a check to see what memory events are still in flight that
read/write that byte, from the perspective of the accessing thread. In this way,
RAW (read-after-write) and WAR (write-after-read) hazards can be detected.
For VGPRs, the detector also flags WAW when an instruction write can be
clobbered by a pending asynchronous load. Similar logic applies for VGPR and
SGPR reads.

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
- `hip_race_gfx950_test.hip` and `hip_race_gfx1151_test.hip` — end-to-end HIP
  kernel tests run under the emulator with the `race` plugin enabled in the
  config file.

```bash
# Core detection tests
ctest --test-dir build -R "RaceDetector|IntervalSet"

# End-to-end HIP tests (the test config enables the race plugin)
ctest --test-dir $BUILD_DIR -R "RaceTest"
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

- **Limited WAW detection**: VGPR WAW covers instruction writes and
  asynchronous memory writes that overlap a pending load. LDS WAW is not
  currently reported.

- **Conservative DPP/SDWA write masks**: WAW precision depends on the execution
  plugin's instruction-write lane and byte masks. DPP destinations can currently
  report all active lanes, and SDWA preserve-mode destinations can report a full
  dword, so writes to architecturally preserved lanes or bytes may be
  conservatively reported as races.

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
