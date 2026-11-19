# Race Detector

Detects synchronization hazards in AMD GPU kernels. Tracks in-flight memory
events (loads, stores) and reports races when a register or LDS memory is
accessed before the operation that produced it has been properly synchronized.

Currently detects intra-workgroup races — cases where the value read from a
register or LDS is not deterministic due to missing `s_waitcnt` or `s_barrier`
instructions.

## Target scope

The race detector supports gfx942 (GFX9/CDNA3), gfx950 (GFX9/CDNA4) and
gfx1151 (GFX11.5/RDNA3.5), with end-to-end tests for each. Support for gfx1250
(CDNA5) is limited and under active development; end-to-end tests currently
cover its tensor DMA races (see Tensor DMA races below) and its counter-capacity
backpressure. The plugin also tracks partial waits and FLAT destination lanes on RDNA4.
Native tests cover CDNA1/2 counter capacity and EXEC-zero VMCNT accounting.

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

Compile it with `hipcc` or `amdclang++`. The `--offload-arch` must match the
emulated GPU, which depends on the config file you pass to rocjitsu (e.g.
`gfx950_mi350x.json` emulates gfx950). If using `amdclang++`, pass `-O1` or
higher — the emulator does not currently support unoptimized (`-O0`) GPU code
objects. `hipcc` defaults to `-O3` so this isn't an issue there.

```bash
hipcc -o /tmp/race_example race_example.hip --offload-arch=gfx950
# or: amdclang++ -O2 -o /tmp/race_example race_example.hip --offload-arch=gfx950
```

Copy `configs/gfx950_mi350x.json` to `my_config.json` and enable the race
detector by adding these top-level settings to that configuration:

```json
{ "require_all_plugins": true, "plugins": { "race": {} } }
```

`require_all_plugins` makes a missing or invalid plugin fail initialization.
The example checks HIP errors and exits with a failure status in that case,
so a skipped detector cannot look like a successful race check.

Run it under the emulator:

```bash
$BUILD_DIR/tools/rocjitsu/rocjitsu --config my_config.json -- /tmp/race_example
```

A report identifies the conflicting instructions and their owning waves. This
report was captured from the gfx950 example above; PCs, register assignments,
and dispatch IDs can vary between builds and runs:

```
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

This tells you that dispatch 2 of `transpose_lds` reported a race: wave 1 wrote
to LDS (`ds_write_b32`) and wave 0 read from the same address (`ds_read_b32`)
without a barrier in between. The `kernel` field is a compact display name, and
`symbol` is the exact ELF symbol when rocjitsu can resolve it. If a name cannot
be resolved, rocjitsu reports `?` for the unresolved field. The fix is to add
`__syncthreads()` between the write and the read.

The completion line refers to the conflicting memory operation. Here, wave 1
has not satisfied its `lgkmcnt` wait requirement. Even if that requirement
were satisfied by an `s_waitcnt`, wave 0's read would still need to be ordered
after wave 1's write. The synchronization line describes that separate
requirement. The unmarked `ds_write_b32` is wave 0's own execution of the
store; the marked write above it belongs to wave 1.

### Running your own application

Replace the binary path with your application. This works with any ROCm workload
— compiled HIP/HSA binaries, Python scripts using PyTorch or JAX, multi-process
launchers like `torchrun`, etc.

```bash
$BUILD_DIR/tools/rocjitsu/rocjitsu --config my_config.json -- ./my_app
$BUILD_DIR/tools/rocjitsu/rocjitsu --config my_config.json -- python my_script.py
```

To capture reports to a file instead of stderr (useful for CI or
scripted workflows), create the output directory with `mkdir -p /tmp/output`
and add a `sinks` section to your config:

```json
{
  "require_all_plugins": true,
  "plugins": { "race": {} },
  "sinks": { "types": ["file"], "dir": "/tmp/output" }
}
```

```bash
$BUILD_DIR/tools/rocjitsu/rocjitsu --config my_config.json -- ./my_app
# Reports are written to /tmp/output/race.log
```

## Reading a report

The human-readable heading names the affected register or LDS byte and identifies
read-after-write (RAW), write-after-read (WAR), or write-after-write (WAW). The
next lines identify the current access and the conflicting memory event by
where its data moves, such as from VGPRs into LDS. Register-load conflicts name
the destination (VGPRs, SGPRs, or TTMPs) without guessing the source memory space.
Direct-to-LDS loads likewise say “load into LDS” because they can read global
memory or scratch. A FLAT instruction can load from different spaces in
different lanes, and scalar memory loads can read scratch. These events do
not retain that source. For LDS conflicts, atomic operations are described as
updates of LDS, with returned VGPR values mentioned when applicable.

For a pending event, the report lists the completion counters that remain unsatisfied, using
the architectural wait names for the target. Internal GFX11 subsets are
displayed as their covering `vmcnt`, `vscnt`, or `lgkmcnt` domains. When the
event belongs to another wave, the report also states that the detector observed
no synchronization between the waves for this access. This can be a false
positive when the kernel uses a synchronization mechanism the detector does not
observe, such as an LDS barrier phase wait; see [Limitations](#limitations).
Completion and inter-wave synchronization are separate conditions; one wait
instruction is not a universal fix for every reported hazard.

Each trace row is disassembled from its own captured bytes, so code rewrites or
unmapping do not change the reported instructions. Within the reporting wave,
the displayed span starts at the first matching producer PC and encoding and
ends at the last matching current access. Repeated producer executions with
identical PCs and encodings are not distinguished.

When the conflict belongs to another wave, the reporting wave's trace starts at
its own latest execution of the same instruction. Waves usually run the same
code, so this shows the path from that instruction to the reported access,
including any intervening waits or barriers. If the reporting wave never ran
that instruction, the report shows up to its last 16 instructions, limited by
`trace_context`.

The instruction table follows the reporting wave's execution order. An event
from another wave, or one older than the trace window, is shown separately.

Each wave retains 256 instruction records. The `trace_context` plugin option
controls how much of an excerpt is selected before eliding its middle. It
defaults to 128 records, split equally between the start and end. It accepts
integers from 2 to 256; an odd value gives the extra record to the end. Setting
it to 256 disables middle elision. For example:

```json
{ "require_all_plugins": true, "plugins": { "race": { "trace_context": 256 } } }
```

Keep `require_all_plugins` enabled when changing this option. An invalid value,
such as 257, otherwise warns and skips the detector instead of failing the run.

Synchronization and control-flow instructions remain visible in the middle,
even when they exceed the configured context. A gap of one instruction is
shown in full. The number of elided instructions is shown at each larger gap.
The fallback for a conflict never executed by the reporting wave remains the
last 16 instructions, or `trace_context` if smaller. Older unrecorded history
is identified separately; this option does not increase history storage.

Consecutive identical `s_nop` instructions in the selected context share a row
showing their execution count and final PC. Runs stop at a changed encoding,
nonsequential PC, elision gap, or marked instruction. This is display compaction:
context and gap counts still count individual executed instructions. Waits,
barriers (including split-barrier signals and LDS barrier arrivals), GWS
operations, and control-flow instructions are always shown individually.

The plugin writes a summary when it is destroyed during VM teardown. It counts
emitted reports and observations suppressed by the existing PC-based
deduplication, not affected lanes or independent bugs. A successful application
exit can occur without that teardown, so complete race reports can appear
without a summary. Its absence does not mean zero races or zero suppressed
observations. The structured `RACE` header and `END_RACE` delimiter remain
available for tools that consume individual reports.

The separate memory-wait warning states the threshold required immediately
before its reported access. That threshold need not be sufficient at an earlier
wait instruction, because more operations may have been issued in between.

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
  `s_waitcnt lgkmcnt` insufficient). On GFX10, `DS_ORDERED_COUNT` supplies a
  whole-wave result under LGKMCNT and an outgoing contribution from the first
  active lane under EXPCNT. The detector warns when an instruction reads or
  overwrites the pending result. It also warns when an instruction overwrites
  the contributing lane of the outgoing VGPR before its EXPCNT event retires.
  Waiting for one counter leaves the other obligation in effect.
- **Scalar-register races**: an SGPR or TTMP is read or overwritten before a
  pending scalar-memory load has completed. A later scalar load to the same
  destination is also checked because scalar-memory results can complete out
  of order.
- **LDS races**: an LDS byte is read by one wave while another wave has an
  outstanding write to the same byte, or written by one wave while another
  wave has an outstanding read of the same byte, without an intervening
  barrier that orders them (see Barriers below);
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
  store's source before the transfer completes, through `s_wait_tensorcnt` or a
  full `tensorcnt`, is reported; other waves also need an `s_barrier`. A load
  covers every element of its tile, including elements outside the tensor, which
  it zero-fills; padding bytes are excluded. A store covers only the elements
  inside the tensor. Either way, only bytes inside the wave's LDS allocation are
  tracked, because bytes past it are not transferred: an i32 at offset 254 of a
  256-byte allocation contributes only bytes 254 and 255. Transfers from one
  wave complete on `tensorcnt` in issue order (CDNA5 ISA 10.11.1), so a partial
  `s_wait_tensorcnt` or a full `tensorcnt` retires the oldest. The ISA does not
  specify the relative timing of their LDS reads and writes, so as a
  conservative rule the detector reports a tensor load and a tensor store from
  the same wave that touch the same LDS byte, in either order, until
  `s_wait_tensorcnt` or a full `tensorcnt` retires the earlier one. Two stores
  of the same tile only read it, which is safe; two loads over the same bytes
  are an LDS write/write conflict, which the detector does not check (see
  Limitations). Each instruction is registered when it issues, as other memory
  instructions are, so an instruction that moves no data still occupies its
  `tensorcnt` slot, and a transfer that waits for memory is pending from the
  moment it issues.
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

Detection is at byte granularity: D16 (half-register) loads only flag races on
the affected bytes, and LDS races are tracked per byte.

## How it works

Every in-flight memory operation has an **event** that goes through the
following lifecycle:

1. **ACTIVE** — the operation is in flight. Ordinary DS operations issued by
   the same wave remain ordered with respect to each other, so a later
   same-wave DS read or write does not race solely because the earlier DS event
   is still active. The detector applies this same-wave LDS-byte ordering to a
   FLAT lane routed through the LDS aperture. On RDNA1/2 and CDNA1–4 this is
   a simulator model without independent hardware qualification. The GFX10
   FLAT-store/DS-read regression exercises the model. Returned VGPRs have
   separate, target-specific writeback and wait rules.
   Direct-to-LDS VMEM writes still require the owning wave to
   wait for `vmcnt` before reading the destination bytes. Tensor DMA transfers
   complete in issue order, but as a conservative rule an active tensor load or
   store still conflicts with a later, overlapping transfer in the other
   direction from the same wave (see Tensor DMA races above). An active async
   store's LDS source likewise conflicts with any later, overlapping LDS write
   from the same wave (see Async copy races above).
1. **WAVE_COMPLETE** — waits have satisfied every counter obligation for the
   event. This means the event is no longer in flight from the perspective of
   the wave that issued the operation, but is still in flight from the
   perspective of other waves in the same workgroup.
1. **RETIRED** — every other live wave of the workgroup has passed a barrier
   that orders the event (see Barriers below). The event is fully retired and,
   from the perspective of all threads in all wavefronts, the operation is
   complete. A wave whose workgroup has no other live wave, because it is the
   only wave or the others have ended, has no other wave to race with, so its
   LDS events are retired as soon as they are wave-complete, without a
   barrier. They then leave the LDS lists that race checks scan, so
   those checks do not slow down as completed events add up in long
   single-wave kernels. Retired events stay in the event registry until it
   trims them, and it trims only from the oldest event: an older event that is
   still pending, such as a global load waiting on `loadcnt`, keeps every
   later retired event in the registry.

### Barriers

On gfx1250 a barrier is split into `s_barrier_signal`, which does not block and
waits for no memory counter, and `s_barrier_wait`, which blocks until the
barrier completes: for a workgroup barrier, once every wave has signalled or
ended; for a named barrier, once it has counted as many signals as its member
count (CDNA5 ISA 5.6 and 15.3). For a direct handoff from a producer to a
consumer, the detector recognizes ordering when:

- the issuing wave completed the event (by a wait or counter-capacity
  backpressure) before its own signal, and
- the accessing wave issued the access after its own wait.

The detector models each round of a barrier as a release/acquire channel. A
signal releases what the wave has completed, and what it has acquired from
earlier rounds; passing the wait acquires everything released into the round
that set the wave's completion bit. Ordering is transitive, so waves can pass
ordering through named barriers, and an LDS event is retired once every other
live wave has acquired it. A wave that ends releases its completed events into
every later workgroup round it had not signalled, because such a round cannot
complete until the wave ends; named and cluster rounds, which can complete
without it, do not. A round the wave had signalled before ending can also
complete without its end, so it orders only what that signal released. A named
round orders events only for the waves that joined it. A cluster round counts
one signal from each workgroup, so within a workgroup it releases what the
signalling wave had completed or acquired, for example through an earlier
workgroup barrier; ordering between workgroups is not modeled. The simulator
reports each counted signal, each completed round and each passed wait to
plugins.

These rules describe the detector's explicit signal/wait model. Other
protocols, such as polling a publication flag or a necessary reduction of a
named barrier's threshold by `s_barrier_leave`, can order LDS accesses but are
not recognized as handoffs by this model.

A completed LDS event stays in the history, and is still checked, until every
other live wave of the workgroup has acquired it. A workgroup in which some
wave never passes a barrier wait, or whose waves synchronize only through
named barriers among subsets, therefore keeps its completed events until those
waves end. Each wave's completed events are kept in the order of the signals
that release them, and a cross-wave check scans only the ones the accessing
wave has not acquired: waves that keep exchanging LDS data through a named
barrier do not rescan each other's ordered accesses, which remain in the
history for the waves outside the barrier.

The simulator also reports RDNA4's separate signal and wait through this
model. Targets with a single `s_barrier` instruction are handled with
every member's signal and wait at the point where the barrier resolves.

The detector reports an access in the window between a wave's signal and its
wait, and an event completed after its wave's signal, on every interleaving of
the waves:

```asm
; wave 0                         ; wave 1
ds_store_b32 v0, v1
s_barrier_signal -1              s_barrier_signal -1
s_wait_dscnt 0                   s_barrier_wait -1
s_barrier_wait -1                ds_load_b32 v2, v0   ; race: the store
                                                      ; completed after the signal
```

On RDNA and CDNA5 every issued generic `FLAT_*` instruction occupies a
vector-memory and an LDS counter position, including when EXEC is empty. On
RDNA1 through RDNA 3.5 a live destination needs sufficient waits on both
counters. RDNA4 and CDNA5 track the returned lanes by their address domain.
For CDNA1–4 the simulator models counter participation and
result readiness by address domain. The CDNA3 and CDNA4 ISA guides, section
10.2, describe joint VM and LGKM completion. CDNA hardware qualification is
still needed to establish when one wait suffices. The conservative wait on
gfx950 is `s_waitcnt vmcnt(0) lgkmcnt(0)`. LDS byte conflicts between waves
also require workgroup synchronization.

Routing preserves the decoded completion ordering for each issued counter.
Generated VMEM/FLAT stores and atomics do not carry an EXP completion
obligation. CDNA4 FLAT completion stays unordered in the model.

On targets with conditional counter participation, functional counter acquisition
and both runtime checkers use the selected domains. FLAT capacity constraints are
applied after addresses are resolved.
Before reading the address operands, neither domain is guaranteed to need a
counter slot. A full LDS counter followed by a global-only FLAT therefore cannot
prove an older LDS result complete.

Global and scratch lanes share the VMEM domain. On RDNA1 through RDNA3.5 a FLAT
result needs both counter waits. On targets with independent completion, each lane
only waits for the pipeline that returned its result. Functional execution completes
retryable global/scratch work before accessing LDS.
LDS validation and interval registration are bounded by the CU backing capacity.
For RDNA4 FLAT LDS, they also stop at the captured workgroup allocation end,
retaining only complete memory elements; a lane whose first element does not fit
records no LDS bytes, and an atomic requires its whole element. This end is the
modeled `lds_base + lds_size`, including current 256-byte scheduler rounding;
PM4 sizes may already be 512-byte multiples, and target allocation and encoding
granules remain unmodeled. Out-of-bounds addresses preserve register and counter
dependencies without recording LDS bytes. DDS has no emulator backing and faults
before execution.

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

The detector's persistent counter accounting is limited to operations routed
through rocJITsu's memory pipelines. A narrow `s_sendmsg*` fallback applies
issue-time pressure: one LGKMCNT token on the pre-GFX12 capacity targets (CDNA1
through CDNA4 and GFX11), and on gfx1250 one KMcnt token, or two for
`s_sendmsg_rtn_*`, which also counts its returned data (CDNA5 ISA 5.4). On
gfx1250 that pressure cannot complete anything, because the only events the
detector registers on KMcnt are scalar loads, which complete out of order.
Message occupancy and returning message results are not retained as pending
events. Timestamp-query operations such as `s_memtime` and `s_memrealtime` are
not accounted for, nor are the gfx1250 barrier instructions that use KMcnt:
`s_barrier_signal_isfirst`, `s_get_barrier_state` and `s_barrier_leave` (CDNA5
ISA 5.6.3). These operations therefore remain unsupported rather than being
treated as complete message, timestamp or barrier modeling. Leaving out a
counter's operations only makes its capacity model conservative: the detector
can miss a completion, never retire too much.

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

On RDNA1 and RDNA2, GDS loads and returning atomics expose their destination through
LGKMCNT. EXPCNT separately protects the source VGPRs and EXEC from overwrite.
A wait on only one counter releases only its associated dependency.

The detector records the target-specific wait-counter family on every event.
It handles both the combined wait fields and the standalone counter forms on
supported targets, for example:

```asm
s_waitcnt vmcnt(0) lgkmcnt(0)  ; combined fields
s_waitcnt_vmcnt null, 0        ; standalone VMCNT form
s_waitcnt_lgkmcnt null, 0      ; standalone LGKMCNT form
```

On CDNA1 through CDNA4, VMCNT holds up to 63 positions and LGKMCNT up to 15.
On RDNA1 through RDNA3.5, both hold up to 63. Vector stores use a separate VSCNT
on these RDNA targets. On RDNA4 and CDNA5 the detector applies the limits of
the individual counters used by the operation. LOADCNT, STORECNT, DSCNT,
SAMPLECNT, ASYNCCNT and TENSORCNT hold up to 63 positions where implemented,
KMCNT holds up to 31, and BVHCNT and EXPCNT hold up to 7. Generic FLAT
uses the target-specific admission rules above.

On CDNA1 through CDNA4, the detector retains ordered VMCNT slots for buffer,
global and scratch instructions issued with EXEC zero. These slots count
toward backpressure and partial waits but have no register or LDS accesses.
For example, after an active buffer load and an EXEC-zero buffer load,
`s_waitcnt vmcnt(1)` completes the older load.

EXEC-zero DS and generic FLAT instructions remain outside this accounting.

On gfx1250 it is modeled for each split counter: LOADcnt, STOREcnt, DScnt,
ASYNCcnt and TENSORcnt are 6-bit counters and KMcnt is a 5-bit counter (the
counter table in CDNA5 ISA 5.7.1), so each holds 63 tokens, or 31 for KMcnt.
These are the capacities the core memory-wait checker uses. Each counter
constrains only its own events: STOREcnt pressure does not complete a pending
load, and a scalar load does not complete a pending DS operation. Async loads
and async stores share ASYNCcnt but complete in order only within their own type
(CDNA5 ISA 10.8), so a full ASYNCcnt completes the oldest async load only when
async loads alone fill it, and likewise for stores. Scalar loads complete out of
order (CDNA5 ISA 5.7.1.1), so a full KMcnt never completes one. A memory
instruction issued with wave EXEC zero counts unless the counters its encoding
names are all zero (CDNA5 ISA 3.2.2.1): LOADcnt, STOREcnt, ASYNCcnt and
TENSORcnt for VMEM, DScnt for LDS, and all of them for FLAT. On those counters
it applies backpressure and keeps its slot, recorded as an event with no
registers or LDS bytes, so later capacity checks and partial waits count it.
That is safe even if the instruction was skipped: those counters were then zero,
so the earlier operations on them had completed. Its other counters are left
out: `ds_atomic_async_barrier_arrive_b64` is an LDS instruction that counts on
ASYNCcnt, so with EXEC zero it may be skipped while async loads are pending, and
it proves none of them complete. An ordinary buffer access whose lanes are all cleared while
EXEC is active keeps no slot. A VBUFFER instruction whose resource descriptor
TYPE is not a buffer is ignored (CDNA5 ISA 9.6), so it applies no backpressure
and keeps no slot, whether EXEC is active or zero. If hardware still counted it,
leaving it out only keeps older operations pending longer. `global_inv` counts on
LOADcnt, and `global_wb` and `global_wbinv` on STOREcnt (CDNA5 ISA 5.7.1.2), and
none of them is skipped for EXEC zero (CDNA5 ISA 3.2.2.1), so each applies
backpressure on its counter when it issues. Each then keeps its slot, recorded
as an event with no registers, in order with the VMEM operations on its
counter, so later capacity checks and partial waits count it. The LDS lane
permutes (`ds_permute_b32`, `ds_bpermute_b32`, `ds_bpermute_fi_b32`) and
`ds_swizzle_b32` use the LDS hardware without LDS memory (CDNA5 ISA 11.2.3),
and LDS operations count on DScnt and are kept in order (CDNA5 ISA 5.7.1.4), so
each applies DScnt backpressure when it issues and keeps its slot in order with
LDS operations. `ds_nop` does the same, following the core memory-wait
checker's model, since the ISA describes it only as "Do nothing". Tensor
DMA instructions apply TENSORcnt backpressure when they issue, before their LDS
bytes are checked, so the oldest transfer a full TENSORcnt completes does not
conflict with the new one. Every generic FLAT applies backpressure to both DScnt
and LOADcnt or STOREcnt when it issues, whichever memory its addresses resolve
to, because it increments both (CDNA5 ISA 5.7.1.3). The slot it takes in a
domain its addresses did not use is recorded as an event with no registers or
LDS bytes, in order with that domain's other operations, so later capacity
checks and partial waits count it as hardware does. The FLAT's own result still
depends only on the domains its addresses used.

A zero wait completes every event on the selected counter. On RDNA1/2,
ordinary VMEM loads and image samples advance VMCNT in issue order, so a
partial wait counts younger loads of either kind. Their register writes can
complete out of order across the two kinds. The detector checks overlapping
destinations separately when deciding whether to report an overwrite.

Inline `IMAGE_GET_LOD` on GFX10/11 and `IMAGE_GET_RESINFO` occupy VMCNT
(LOADCNT for GFX12 RESINFO). The detector tracks the VGPRs and lanes written
by each successful query. GET_LOD uses sample writeback order and GET_RESINFO
uses ordinary load writeback order.

The plugin keeps track, for all registers and LDS memory bytes, of which memory
operations are in flight. When an instruction in the emulator accesses an LDS
byte, there is a check to see what memory events are still in flight that
read/write that byte, from the perspective of the accessing thread. In this way,
RAW (read-after-write) and WAR (write-after-read) hazards can be detected.
For VGPRs and scalar registers, the detector also flags WAW when an instruction
write can be clobbered by a pending asynchronous load. The detector also reports
WAW between scalar loads targeting the same SGPR or TTMP.

On architectures where scalar-memory and data-share operations use a combined
`lgkmcnt`, a nonzero partial wait cannot identify which scalar destination has
completed. Those scalar destinations remain pending until `lgkmcnt(0)`. The
same partial wait can still retire older data-share operations that are provably
complete from their in-order completion rule.

**LDS race detection** uses coarse-grained counters (one per 16-byte chunk) for
fast-path checks, with interval-based overlap scanning as a fallback. Live
events are split by direction so that RAW and WAR hazards are checked
independently. An RDNA2 LDS atomic places the same event in both histories.
Naturally aligned atomics of the same width can operate on the same word.
A returning atomic also protects its result VGPR until its counter completes. Chunks containing only ordinary LDS accesses from the accessing
wave can bypass the interval scan because those operations are ordered. A
mixed chunk uses the scan until its count drains. The history remains available
for checks by other waves until a barrier orders it for all of them. Direct-to-LDS and unordered
memory operations still require their own completion checks.
For a DS or FLAT instruction, the plugin first merges the active lanes' byte
ranges into intervals and checks the chunk counts under them. When no chunk can
hold a conflict, it skips the per-lane checks. The same intervals become the
instruction's event.

For indexed RDNA1/2 LDS instructions, the byte checks use the same workgroup
allocation bounds as execution. The two addresses of a dual operation are
checked independently. On RDNA1/2, an `LDS_DIRECT` source checks the actual
byte, halfword or DWORD read by the instruction against pending writes. A
completed read also protects those bytes against a conflicting write from
another wave until workgroup synchronization.

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
└── core/                    event tracking and lifetime
    ├── race_detector.h/.cpp  main detector: event allocation, validation, retirement
    ├── wave_race_state.h/.cpp  per-wave state: register tracking, waitcnt resolution
    ├── event_registry.h      append-only event store with prefix trimming
    ├── interval_set.h        half-open byte range tracking for LDS
    ├── types.h               core enum types and structs
    ├── common_register.h     memory event types and ordering
    └── dim3d.h               3D coordinate helpers
```

Explicit wait decoding is shared with the core checker in
`lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h`.

## Tests

Tests are part of the rocjitsu test suite (`emulation/rocjitsu/tests/`):

- `race_detector_tests.cpp` — drives `RaceDetector` and `WaveRaceState` directly
  via `race_test_builder.h`, covering VGPR, SGPR, LDS, D16, DTL, exec mask,
  multi-workgroup, and mixed counter scenarios.
- `race_detector_counter_capacity_tests.cpp` — counter-capacity backpressure,
  for the combined counters before GFX12 and the gfx1250 split counters.
- `interval_set_tests.cpp` — unit tests for `IntervalSet`.
- `execution_plugin_race_test.cpp` — drives the plugin's hooks, including
  tensor DMA transfers and async copies on gfx1250.
- `wait_hazard_test.cpp` — wait/readiness tests shared with the core checker.
- The `race` suite in
  [ROCm/rocjitsu-test-corpus](https://github.com/ROCm/rocjitsu-test-corpus)
  owns the gfx942, gfx950, gfx1151 and gfx1250 end-to-end HIP programs; the
  gfx1250 program covers tensor DMA and counter capacity. The
  `rocjitsu-test-corpus` workflow builds them with the corpus HIP toolchain and
  runs each case under the emulator with the `race` plugin enabled.

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

- **Limited WAW detection**: VGPR WAW covers instruction writes and
  asynchronous memory writes that overlap a pending load. Scalar-register WAW
  covers instruction writes and scalar loads that overlap pending scalar-memory
  destinations. LDS WAW is not currently reported.

- **Embedded waits**: the race detector applies the VINTERP EXPCNT threshold before
  checking its VGPR sources on GFX11, GFX11.5 and GFX12. A threshold of 7 leaves
  the prior exports pending. An explicit EXPCNT wait and an embedded zero threshold
  both retire them. The typed wait-hazard suite covers the two checkers.

- **Tensor DMA scope**: only the LDS side of a transfer is tracked. Its global
  memory accesses are not checked.

- **Async copy scope**: only the LDS side of an async copy is tracked. Its
  global memory accesses are not checked.

- **Async copy LDS alignment**: async copies use their LDS addresses as
  computed. DS instructions have their LDS address aligned in every alignment
  mode except unaligned (CDNA5 ISA 3.3.4.1), but the ISA does not say whether
  async copies follow that rule, so neither the emulator nor the detector
  applies it to them.

- **False positives with LDS barrier synchronization**: a producer can hand LDS
  data to another wave through an LDS barrier cell instead of a wait and
  `s_barrier`. A tensor load can request an arrival on a barrier cell when it
  completes, and `ds_atomic_async_barrier_arrive_b64` advances a barrier only
  after the producer's earlier async loads complete (CDNA5 ISA 11.2.2). A
  consumer wave can then wait for the barrier's phase and read the tile, or copy
  it out with an async store, without the producer waiting on `s_wait_tensorcnt`
  or `s_wait_asynccnt` or the workgroup passing `s_barrier`. The detector does
  not observe this synchronization: it retires a transfer only through
  wait-counter instructions, counter capacity and `s_barrier`. A consumer that
  is correctly synchronized through the barrier cell is therefore reported as
  racing with the tensor or async load. These reports are false positives, not
  missing coverage.

- **NULL tensor descriptors and NOP tiles**: like the emulator, the detector
  counts a NULL descriptor (`count = 0`) and a NOP tile (`tile_dim0 = 0`) on
  `tensorcnt`. That is an emulator assumption: hardware behavior for these
  encodings is not established. If hardware does not count them, a nonzero
  `s_wait_tensorcnt` that the detector accepts, or a `tensorcnt` the detector
  considers full, can leave an earlier transfer pending, so a real race would
  go unreported.

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
