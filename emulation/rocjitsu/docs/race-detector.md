# Race Detector

Detects synchronization hazards in AMD GPU kernels. The emulator's core checks
register dependencies at instruction issue. The race plugin reports those
findings with kernel context and instruction traces, and separately checks LDS
accesses against pending operations and unsynchronized peer waves.

These checks cover missing waits and barriers within a workgroup. They do not
model global-memory races or synchronization between workgroups.

## Target scope

The race detector supports gfx942 (GFX9/CDNA3), gfx950 (GFX9/CDNA4) and
gfx1151 (GFX11.5/RDNA3.5), with end-to-end tests for each. Support for gfx1250
(CDNA5) is limited and under active development; end-to-end tests cover split
wait counters and tensor DMA races. See the specific limits below.

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
`gfx950_mi350x.json` emulates gfx950). If using `amdclang++`, pass `-O1` or
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
RACE kernel=transpose_lds symbol=_Z13transpose_ldsPKiPi dispatch=1 type=LDS access=read reg=508 wave=0 lane=0 wg=0,0,0 conflict=unknown producer_pc=0x7f6a1e801010 consumer_pc=0x7f6a1e80101c checker=plugin
Race on LDS byte 508 [workgroup (0, 0, 0), wave 0, lane 0]
  ==>  0x7f6a1e801010  ds_write_b32 v0, v1  ; <-- wave 1
       0x7f6a1e801018  v_sub_u32_e32 v1, 0, v0
  ==>  0x7f6a1e80101c  ds_read_b32 v1, v1  ; <-- wave 0 lane 0
END_RACE
```

This tells you that dispatch 1 of `transpose_lds` reported a race: wave 1 wrote
to LDS (`ds_write_b32`) and wave 0 read from the same address (`ds_read_b32`)
without a barrier in between. The `kernel` field is a compact display name, and
`symbol` is the exact ELF symbol when rocjitsu can resolve it. If a name cannot
be resolved, rocjitsu reports `?` for the unresolved field. The fix is to add
`__syncthreads()` between the write and the read.

Each trace line shows an instruction's GPU virtual address, which varies
between runs, and its disassembly. The trace lists instructions that the
reporting wave (here wave 0) executed, oldest first. `==>` marks entries matching
the PC and captured encoding of either instruction involved in the race, and
the `<--` annotation names the wave and lane that performed that access. When the
conflicting instruction ran in another wave, as with wave 1's `ds_write_b32`
above, the marker is placed where the reporting wave executed the same bytes
at the same address, if such an entry is present. Repeated executions of identical
bytes at the same address are not distinguished; the displayed span ends at the
first matching racing access after its first marker.

The race detector retains the last 256 instructions of each wave. If the
reporting wave has no entry matching the conflicting PC and encoding, the report
prints the retained producer instruction first with a `(before trace window)`
note. This can mean the instruction has left the trace, but it can also mean that
another wave executed an instruction the reporting wave never executed. The note
does not establish that the conflict is more than 256 instructions old.

For example, if another wave's producer is at `0x1000` and the reporting wave's
trace contains only its racing access at `0x3000`, the producer appears with
`(before trace window)` and `0 instructions not recorded`. That count is the
number of retained reporting-wave entries skipped before the first displayed
marker, not the number of instructions since the producer ran.

Traces longer than 32 instructions show only their first and last 16. Each trace
row is disassembled from its own captured bytes, so it remains accurate if the
kernel's code is later overwritten or unmapped.

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

### Core-backed register reports

Enabling `"race": {}` requests the core's issue-time register findings, including
when stderr warnings are off. The core checks each instruction once for both
warnings and plugin reports; the plugin does not subscribe to register-access
hooks or maintain another set of register dependencies. XCNT replay-source
findings require XCNT checking to be explicitly enabled. See
[memory wait diagnostics](memory-wait-diagnostics.md) for target coverage,
recovery policy, and replay-mode limits.

Every `RACE` record includes `producer_pc`, `consumer_pc`, and `checker`.
Register records have `checker=core`; LDS records have `checker=plugin`.
Register records additionally include:

| Field | Meaning |
|---|---|
| `counter`, `required_wait` | Counter and sufficient maximum wait threshold for this dependency. |
| `reg`, `width` | First overlapping wave-relative physical register and number of dwords. |
| `lanes`, `bytes` | Actual overlapping lane mask and byte mask within a dword, in hexadecimal. Scalar lanes are all ones. |
| `lane` | First overlapping vector lane, or -1 for a scalar/special register. |

The trace uses the instruction encodings captured at execution, not code reread
at a saved PC. The issuer reports before the consumer's instruction notification;
the reporter includes the consumer's owned encoding in the displayed trace. If
either encoding is unavailable, the report keeps both PCs and prints explicit
`encoding unavailable` witnesses instead of a reconstructed trace.

The plugin emits one representative register finding per dispatch,
producer/consumer PC pair, register class/index, access kind, and counter.
Repeated visits to that site are counted but suppressed, including visits with
different lanes or widths. Different producers and consumers remain separate
sites. A read-modify-write can report both a read and an overwrite, and a later
unwaited consumer can report the same producer again: reporting does not retire
the dependency. Only a wait or another architectural guarantee does.

Core warnings have a separate per-CU budget. LDS reports use broader PC-based
suppression: within a dispatch, an emitted pair suppresses further LDS reports
involving either PC. The final summary counts emitted records and suppressed
register findings. Neither a clean nor an interrupted run proves complete coverage.

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
- **Scalar-register races**: an SGPR, TTMP, or supported special register is
  read or overwritten before its pending result is ready. A later scalar load
  to the same destination is also checked because scalar-memory results can
  complete out of order.
- **Replay-source hazards** (gfx1250, opt-in): a protected memory source is
  overwritten before address translation is known complete. See the core's
  [XCNT coverage](memory-wait-diagnostics.md#xcnt-replay-source-diagnostics).
- **LDS races**: an LDS byte is read by one wave while another wave has an
  outstanding write to the same byte, or written by one wave while another
  wave has an outstanding read of the same byte, without an intervening
  `s_barrier`;
  or a wave reads bytes targeted by its own outstanding direct-to-LDS operation
  before the required `s_waitcnt vmcnt`. A direct-to-LDS write over bytes of
  the same wave's pending DS read is also reported, on every target with
  direct-to-LDS, until a DS wait completes the read. That is a conservative
  rule: the ISA does not establish that the write waits for the read. Compilers
  can emit such a pair without a DS wait (on gfx1250, LLVM may place only
  `s_wait_kmcnt 0` between a `ds_load_b32` and a `global_load_async_to_lds_b32`
  to the same dword), and the detector reports it.
- **Tensor DMA races** (gfx1250): `tensor_load_to_lds` writes and
  `tensor_store_from_lds` reads are LDS events on `tensorcnt`. Any wave,
  including the issuing one, that reads a load's destination or overwrites a
  store's source before `s_wait_tensorcnt` is reported; other waves also need an
  `s_barrier`. A load covers every element of its tile, including elements
  outside the tensor, which it zero-fills; padding bytes are excluded. A store
  covers only the elements inside the tensor. Either way, only bytes inside the
  wave's LDS allocation are tracked, because bytes past it are not transferred:
  an i32 at offset 254 of a 256-byte allocation contributes only bytes 254 and
  255. Transfers from one wave complete
  on `tensorcnt` in issue order (CDNA5 ISA 10.11.1), so a partial
  `s_wait_tensorcnt` retires the oldest. The ISA does not specify the relative
  timing of their LDS reads and writes, so as a conservative rule the detector
  reports a tensor load and a tensor store from the same wave that touch the
  same LDS byte, in either order, until `s_wait_tensorcnt` retires the earlier
  one. Two stores of the same tile only read it, which is safe; two loads over
  the same bytes are an LDS write/write conflict, which the detector does not
  check (see Limitations). Each instruction is registered when it issues, as
  other memory instructions are, so an instruction that moves no data still
  occupies its `tensorcnt` slot, and a transfer that waits for memory is pending
  from the moment it issues.
- **Async copy races** (gfx1250): `global_load_async_to_lds_*` and
  `cluster_load_async_to_lds_*` writes and `global_store_async_from_lds_*` reads
  are LDS events on `asynccnt`, one per instruction, with each lane's bytes at
  its LDS address. Any wave, including the issuing one, that reads a load's
  destination or overwrites a store's source before `s_wait_asynccnt` is
  reported; other waves also need an `s_barrier`. Synchronizing other waves
  through an LDS barrier cell instead is not modeled (see False positives with
  LDS barrier synchronization). Async copies may access LDS out of order with
  each other (CDNA5 ISA 10.8), so a store is also reported when it reads bytes
  of the same wave's pending async load, and an async load when it overwrites
  the source of the same wave's pending async store. The ISA does not
  establish the LDS access order of an async copy relative to DS instructions
  or tensor DMA, so as a conservative rule the detector also
  reports a store that reads bytes of the same wave's pending DS write or tensor
  load, until a DS or tensor wait completes it. Compilers can emit such a pair
  without a DS wait (on gfx1250, Clang places no `s_wait_dscnt` between the
  `ds_store_b32` of a shared-memory assignment and a following
  `global_store_async_from_lds_b32` builtin that reads the same bytes), and the
  detector reports it. Async loads complete in issue order with other async
  loads and async stores with other async stores, but loads and stores
  complete out of order with each other, so a partial `s_wait_asynccnt`
  retires the oldest of each kind only when the count guarantees it: after a
  load and then a store, `s_wait_asynccnt 1` retires neither. `ds_atomic_async_barrier_arrive_b64` also completes on `asynccnt`, in
  order with async loads (10.8), so partial waits count it with the loads. The
  ISA specifies out-of-range LDS behavior for async loads only: their LDS write
  is dropped (CDNA5 ISA 3.3.3). As modeling choices, a load lane with any byte
  past the wave's LDS allocation writes nothing and is not tracked, while a
  store lane reads, and is tracked for, the bytes of its source inside the
  allocation, and zeros past it, so every byte a store may read is tracked. A
  copy with no active lanes touches no LDS but issues and counts on
  `asynccnt`; the hardware may skip it only when no load, store, async or
  tensor operation is pending (3.2.2.1), and counting it then retires no later
  copy early. Either way, the instruction occupies its `asynccnt` slot.

Detection is at byte granularity. Register findings use the core's lane and
byte masks, including target-dependent D16 writeback rules; LDS conflicts identify
a byte shared by the two accesses.

## How it works

### One register checker, separate LDS history

The [core register scoreboard](memory-wait-diagnostics.md#access-planning-and-cost)
checks decoded operands before execution or asynchronous MMA publication. It
tracks returned register bytes and, when enabled, replay sources. Register
accessors and memory writeback do not run the checker. Its
[counter coverage](memory-wait-counter-coverage.md) defines producer families,
target-specific waits, completion ordering, mixed-FLAT lane readiness, and
capacity proofs.

The plugin consumes structured core findings for reporting. Its own analysis
state contains only LDS footprints and counter progress. This state cannot be
replaced by register readiness: a wave's wait can release a register while its
completed LDS access still conflicts with an unsynchronized peer.

### LDS events and completion

The plugin records an event for each observed memory issue with counter
obligations. An event has an optional LDS read or write footprint, a producing
instruction snapshot, and its counter obligations. Operations without an LDS
effect still occupy queue positions. Otherwise a partial wait could incorrectly
identify which LDS operations completed.

Counter completion order and LDS access order answer different questions.
For example, tensor DMA completes in issue order on `tensorcnt`, so a partial
wait can complete a prefix. That does not prove that an overlapping tensor load
and store access LDS in order before the wait. Their LDS accesses are therefore
checked as unordered.

LDS events have three states:

1. **ACTIVE** — not every counter obligation has been satisfied. A conflicting
   peer access is checked regardless of whether execution already wrote the
   data eagerly. Ordinary DS operations from the same wave remain ordered with
   each other. Direct-to-LDS VMEM, async copies and tensor DMA use their own ordering
   rules and may require a wait even for an access from the issuing wave.
1. **WAVE_COMPLETE** — all counter obligations have been satisfied. The issuing
   wave can reuse the bytes, but completed LDS evidence remains available for
   checks by unsynchronized peers. This is not a claim that execution is still
   physically in flight.
1. **RETIRED** — a resolved workgroup barrier orders the completed access with
   peer waves, so the event leaves the LDS read/write lists. A single-wave
   workgroup retires completed LDS events immediately. Events without LDS bytes
   also need no barrier once complete.

The event registry trims only a completed or retired prefix. One older pending
event can therefore retain many later completed records even after those records
have left the LDS conflict lists. Logical completion and reclaiming registry
storage are separate.

The adapter uses the shared `MemoryIssueInfo` obligations and
`WaitCounterPolicy` explicit-wait decoding. A zero wait completes every event
on its selected counter; an all-ones field means no wait. A nonzero wait retires
only the provable prefix within each independently ordered completion class.
Sharing a counter does not make scalar memory, GDS, and indexed LDS one FIFO.

The plugin models counter-capacity admission for VMCNT and LGKMCNT on CDNA1–4
and GFX11. FLAT admission is applied after routing selects the domains actually
used; before address resolution, neither domain is guaranteed to need a slot.
These are LDS-analysis limits, not limits on the core's separate register
capacity model.

### LDS footprints

Memory routing supplies actual lane masks and addresses. Global/scratch-only
issues contribute counters without LDS intervals. Mixed global/LDS FLAT issues
contribute one event, with both obligations and only the ordinary LDS lanes'
intervals. The plugin conservatively keeps that whole event pending until both
domains complete; unlike the core's register dependencies, its LDS footprint
does not yet have independent per-domain readiness.

LDS ranges are bounded by the CU backing capacity. RDNA4 FLAT additionally clips
to the captured workgroup allocation, keeping only complete memory elements.
Its bound is the modeled `lds_base + lds_size`, including scheduler rounding,
not a new target-allocation model. Tensor DMA observations instead describe
byte ranges and can include a partial element at the allocation boundary.
Rejected LDS addresses record no intervals but retain their counter participation.
DDS has no emulator backing and faults before execution.

Conflict checks use counts per 16-byte chunk before scanning overlapping
intervals. Read and write histories are separate: a read checks prior writes
(RAW), and a write checks prior reads (WAR). Chunks containing only ordered DS
accesses from the same wave can bypass the scan. Cross-wave evidence remains
until synchronization; LDS write/write conflicts are not currently checked.

For a DS or FLAT instruction, the plugin first merges the active lanes' byte
ranges into intervals and checks the chunk counts under them. When no chunk can
hold a conflict, it skips the per-lane checks. The same intervals become the
instruction's event.

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
├── plugin.h/.cpp             core-finding reports and LDS observation adapter
└── core/                     LDS analysis
    ├── race_detector.h/.cpp   workgroup footprints, conflict checks, retirement
    ├── wave_race_state.h/.cpp  per-wave counter progress and wait resolution
    ├── event_registry.h       append-only event store with prefix trimming
    ├── interval_set.h         half-open byte ranges
    ├── types.h                LDS access, event, and violation types
    └── dim3d.h                3D coordinate helpers
```

Explicit wait decoding is shared with the core checker in
`lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h`.

## Tests

Tests are part of the rocjitsu test suite (`emulation/rocjitsu/tests/`):

- `race-detector/` — LDS overlap, bounds, counter progress, partial waits,
  capacity, barrier lifetime, event storage, and interval tests.
- `execution_plugin_race_test.cpp` — LDS routing, tensor DMA, async copies, workgroup
  lifecycle, and owned report traces through the plugin's hooks.
- `execution_plugin_register_hazard_test.cpp` — core findings through the
  actual issuer, report fields, owned witnesses, suppression, waits, and LDS
  controls.
- `memory_wait_scoreboard_test.cpp` — register readiness and counter proofs,
  including instruction footprint comparisons against execution observations.
- `wait_hazard_test.cpp` — explicit-wait conformance shared by independent
  core-register and plugin-LDS backends.
- The `race` suite in
  [ROCm/rocjitsu-test-corpus](https://github.com/ROCm/rocjitsu-test-corpus)
  owns the gfx942, gfx950, gfx1151 and gfx1250 end-to-end HIP programs; the
  gfx1250 program covers split wait counters and tensor DMA. The `rocjitsu-test-corpus` workflow builds
  them with the corpus HIP toolchain and runs each case under the emulator with
  the `race` plugin enabled.

```bash
# Core detection tests
ctest --test-dir build -R "RaceDetector|IntervalSet"
```

To run the HIP cases locally, follow the corpus's
[race-suite instructions](https://github.com/ROCm/rocjitsu-test-corpus#rocjitsu-race-detector-integration)
and point the wrapper at the RocJITsu build being tested. Set
`ROCJITSU_RACE_CONFIG` to `tests/race-detector/race_test_gfx942_config.json` for gfx942,
`tests/race-detector/race_test_config.json` for gfx950, `configs/gfx1151.json` for
gfx1151 or `configs/gfx1250_mi455x.json` for gfx1250, using absolute paths from
this source tree. The original
`RaceTest.<target>_<name>` selectors remain available through the corpus's
`--case` option. Shared atomic and FLAT-to-LDS cases use
`RaceTest.vmem_<target>_<name>`, for example
`--case RaceTest.vmem_gfx942_vmcnt_only`. Positive detection controls use the
`_race` suffix.

## Limitations

- **Performance**: rocjitsu emulates GPU execution on the CPU, which is orders
  of magnitude slower than running on actual hardware. Race detection adds
  further overhead on top of emulation. This is a correctness tool, not a
  performance tool — use small inputs and targeted test cases rather than full
  production workloads.

- **Intra-workgroup only**: the detector tracks races within a single workgroup
  (missing `s_waitcnt` and `s_barrier`). It does not detect inter-workgroup
  races, races between dispatches, or host-device synchronization issues.

- **Limited WAW detection**: the core checks register overwrites against pending
  results. The LDS checker reports RAW and WAR, but not LDS write/write conflicts.

- **Tensor DMA scope**: only the LDS side of a transfer is tracked. Its global
  memory accesses are not checked.

- **Async copy scope**: only the LDS side of an async copy is tracked. Its
  global memory accesses are not checked.

- **Async copy LDS alignment**: async copies use their LDS addresses as
  computed. DS instructions have their LDS address aligned in every alignment
  mode except unaligned (CDNA5 ISA 3.3.4.1), but the ISA does not say whether
  async copies follow that rule, so neither the emulator nor the detector
  applies it to them.

- **False positives with LDS barrier synchronization**: a producer can hand
  LDS data to another wave through an LDS barrier cell instead of a wait and
  `s_barrier`. A tensor load can request an arrival on a barrier cell when it
  completes, and `ds_atomic_async_barrier_arrive_b64` advances a barrier only
  after the producer's earlier async loads complete (CDNA5 ISA 11.2.2). A
  consumer wave can then wait for the barrier's phase and read the tile, or
  copy it out with an async store, without the producer waiting on
  `s_wait_tensorcnt` or `s_wait_asynccnt` or the workgroup passing `s_barrier`.
  The detector does not observe this synchronization: it retires a transfer
  only through wait-counter instructions and `s_barrier`. A consumer that is
  correctly synchronized through the barrier cell is therefore reported as
  racing with the tensor or async load. These reports are false positives, not
  missing coverage.

- **NULL tensor descriptors**: like the emulator, the detector counts a NULL
  descriptor (`count = 0`) on `tensorcnt`. That is an emulator assumption:
  hardware behavior for this encoding is not established. If hardware does not
  count it, a nonzero `s_wait_tensorcnt` that the detector accepts can leave an
  earlier transfer pending, so a real race would go unreported.

- **LDS counter-capacity coverage**: the plugin models admission only for VMCNT
  and LGKMCNT on CDNA1–4 and GFX11. On gfx1250, code relying on a full tensor
  counter can get a false LDS report: admitting a 64th ordered transfer proves
  the first complete, but the plugin still keeps it pending without an explicit
  wait. This does not limit the core's register-capacity checks.

- **Incomplete LDS counter occupancy**: pipeline memory issues and tensor DMA
  are retained. A narrow `s_sendmsg*` fallback applies issue-time LGKMCNT
  pressure, but does not retain message occupancy. Inline timestamp and other
  non-pipeline producers are not fully represented. Do not infer complete LDS
  partial-wait coverage from the core's broader register-producer coverage.

- **Mixed-FLAT LDS readiness**: the plugin attaches both domains' obligations
  to one mixed event. An already-completed LDS portion can therefore remain
  pending until the global portion also completes, producing conservative LDS
  reports. Core register readiness tracks the lane groups separately.

- **Cluster LDS multicast**: peer writes are not tracked. The plugin warns when
  an observed transfer targets peer workgroups and checks only the local LDS
  effect.

- **Kernel name resolution**: kernel names in race reports may show as `"?"` if
  symbol information is not available in the code object.

## History

The race detection logic was originally developed as part of **race-emulator**,
a standalone CPU-side GPU assembly emulator that parsed `.s` assembly text
files.

The plugin's LDS analysis operates on memory events, counter progress, and
barriers. Register analysis belongs to rocjitsu's core scoreboard. The original
race-emulator execution engine is no longer under development.
