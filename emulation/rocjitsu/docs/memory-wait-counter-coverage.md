# Memory wait diagnostic coverage

The core checker accounts for completion-counter producers independently of the
functional memory pipelines. Generated instruction metadata selects producers, then the
shared waitcheck target model determines their counter families. Register values and
memory effects are still computed eagerly.

## Architectures and queues

| Architecture | Completion queues |
|---|---|
| CDNA1, CDNA2, CDNA3, CDNA4 | `vmcnt`, `lgkmcnt`, `expcnt` |
| RDNA1, RDNA2, RDNA3, RDNA3.5 | `vmcnt`, `vscnt`, `lgkmcnt`, `expcnt` |
| RDNA4 | `loadcnt`, `storecnt`, `dscnt`, `kmcnt`, `expcnt`, `samplecnt`, `bvhcnt` |
| CDNA5 | `loadcnt`, `storecnt`, `dscnt`, `kmcnt`, `expcnt`, `asynccnt`, `tensorcnt` |

Only instructions present in a target's ISA produce entries. Counter names do not imply
that every target has every instruction associated with that family. The GFX10 legacy
`lgkmcnt` field has six bits; the CDNA legacy field has four.

Each ordered counter has monotonically increasing issue and retirement positions. For
`Q` issued entries, a wait with threshold `N` proves completion through `max(0, Q-N)`. A
later, weaker wait cannot undo completion. Instructions without register results occupy
positions too. An instruction contributing to two counters contributes to both
independently. Counter occupancy and the readiness of individual returned lanes are
separate: each register dependency follows the pipeline that writes those lanes.

Scalar memory is unordered: nonzero waits do not prove individual results ready. This
includes cache operations; their bank-level counter increments are not simulated. Routed
SMEM transfers use the decoded `MemoryCounterObligation::counter_increment()` for
both admission and issue accounting (two units for transfers larger than a DWORD),
but readiness still requires zero. Mixed hardware event types sharing a counter retain separate ordered-class
positions: a nonzero wait proves an ordered result complete only when at least that
many younger operations in its own class follow it. Generic FLAT on older CDNA is treated
conservatively as unordered. Zero waits reset this ordering state. FLAT keeps both queue
entries even when all lanes use one segment; register dependencies are restricted using
the resolved routing masks. Mixed global/shared FLAT functional execution still has the
existing first-request-lane routing limitation; the checker does not repair memory
routing.

Legacy VMEM writeback can avoid an overwrite warning only when the pending result
and the incoming producer share an ordered completion class. On legacy RDNA,
ordinary loads, image samples and BVH results share VMcnt but keep separate
classes. A newer result supersedes a fully covered older dependency only within
the same ordered class; other classes and unordered results remain tracked.

Before reading an incoming producer's operands, the checker applies the finite
counter bound to each qualified ordered class in that counter. A capacity of `C`
and an incoming increment of `U` leave at most `C-U` old counter units before
admission, including when the incoming producer itself is unordered. Younger operations from a
different completion class cannot prove readiness. Capacities come from the
target's counter fields (for example, 15 for legacy CDNA LGKM and 63 for VMEM).
SMEM, GDS, exports, messages and legacy CDNA FLAT do not acquire a FIFO guarantee
from sharing a counter; their results require a zero wait, even after a nonzero
partial wait. CDNA5 async load and store completion classes are separate;
async barrier arrive orders with async loads. Proven completion also releases the
associated replay translation prefix.

## FLAT register readiness

FLAT contributes to both VMEM and DS counters even when every active lane selects
one address space. Its global/scratch and LDS portions can complete independently;
RDNA4 ISA section 5.7.1.3 describes complementary lane masks for these portions.
The checker uses the resolved address of each lane, rather than the functional
pipeline's first-lane route, to associate a returned VGPR lane with its counter.
For example, if lane 0 loads from LDS and lane 1 loads from global memory, a zero
DS wait releases lane 0's register dependency while lane 1 still needs a VMEM wait.
A consumer reading both lanes needs both; a consumer reading only lane 0 does not.

This distinction also appears in LLVM's AMDGPU wait insertion:
[`mayAccessVMEMThroughFlat` and `mayAccessLDSThroughFlat`](https://github.com/llvm/llvm-project/blob/1ac93e094347b45208f215c98038606432c87d60/llvm/lib/Target/AMDGPU/SIInstrInfo.cpp#L4769)
exclude the irrelevant address space when it is known from the memory operand.
[`getEventsFor`](https://github.com/llvm/llvm-project/blob/1ac93e094347b45208f215c98038606432c87d60/llvm/lib/Target/AMDGPU/Utils/AMDGPUHWEvents.cpp#L110)
then adds only the relevant completion events. Unknown FLAT pointers require both;
the runtime checker has per-lane routing information unavailable to static codegen.
Requiring both waits for every returned lane would warn about accesses that this
policy accepts. Avoiding such false positives takes priority over diagnosing an
unproven additional whole-instruction requirement. This does not treat a successful
run without a wait as proof of an architectural completion guarantee.

A hardware probe on gfx1100 and gfx1201 used inline assembly to load LDS in even
lanes and global memory in odd lanes, then immediately add to the returned value.
With only the DS wait, all 1,048,576 global-lane observations per device were wrong
and all LDS-lane observations were correct. Waiting for both produced no errors;
omitting both broke both lane groups. VMEM-only waits happened to leave both groups
ready in this probe, which does not establish a guarantee for the LDS lanes.
These observations support keeping the lanes separate; they do not establish
wait requirements on untested architectures or under all schedules.

## Producer accounting

| Producers | Counter accounting and register dependencies |
|---|---|
| Global, scratch, buffer, typed buffer, image loads and returning atomics | Load queue; track returned registers |
| Stores and atomics without return | Store queue, or the shared legacy VMEM queue; no destination register |
| Generic FLAT loads, stores and atomics | VMEM and DS queues; returned lanes depend on DS for the resolved shared aperture and VMEM for global/scratch |
| LDS/GDS, including permutation, swizzle and DS no-op | DS/legacy LGKM queue; returning forms track registers. GDS also contributes to EXP |
| Scalar loads, atomics, cache operations, timestamps, barrier-state and wave-ID queries | KM/legacy LGKM queue; returned scalar registers are tracked |
| Messages, including message returns | KM/legacy LGKM queue. Return forms contribute two units, with the result pending through the return unit |
| Barrier signal with an `isfirst` result | KM queue and pending SCC result |
| LDS direct/parameter loads | EXP queue and returned VGPRs |
| Exports | EXP queue accounting |
| Image sampling, gather and LOD queries | SAMPLE on RDNA4, otherwise the target's VMEM queue; decoded returned registers are tracked |
| Image BVH | BVH on RDNA4, otherwise the target's VMEM queue; decoded returned registers are tracked |
| Global and buffer cache operations that participate in counters | Load or store queue accounting, including counter-only invalidations |
| Direct-to-LDS asynchronous loads/stores | Ordinary load/store queue and ASYNC queue |
| Asynchronous barrier arrive | ASYNC queue |
| Tensor DMA loads/stores | TENSOR queue |
| gfx1250 scalar address operations and multi-group VMEM, including async LDS transfers | X translation queue; protect source dwords against overwrite, plus VMEM EXEC. One X unit per instruction, independently of completion-counter units. Tensor DMA has no X event |

Zero-EXEC vector producers still contribute positions behind pending requests; an empty
queue may skip them. Scalar producers do not depend on EXEC. Prefetches that do not
increment completion counters contribute no entries.
If a zero-EXEC instruction occupies an X position but no completion position, it
has no completion-to-X mapping. Waiting on that empty completion counter cannot
release older replay sources. The X entry remains until an X wait or another
proven translation drain reaches it.

The checker handles explicit, combined load/store-plus-DS and idle waits, plus
RDNA3/3.5/4 interpolation's embedded EXP wait. No-wait sentinel fields leave the queue
unchanged. RDNA4's compatibility `s_waitcnt` ignores its immediate and acts as
`s_wait_idle`, draining all supported counters. Older interpolation encodings have no
embedded EXP wait.

## Limits of the guarantee

Both false negatives and false positives are possible. Eager global/LDS memory
effects can hide memory-ordering dependencies even within one wave. Conversely,
architecturally sufficient instruction spacing can make a dependency safe without
an explicit wait; latency and instruction spacing are not modeled here.
See [diagnostic limitations](memory-wait-diagnostics.md#false-negatives-and-false-positives).

The diagnostic checks pending register results and the qualified XCNT source
lifetimes. It does not compare memory addresses, prove memory visibility, check LDS
or tensor transfer footprints, or detect communication hazards within or between
waves. ASYNC, TENSOR and store-only producers still count without a register result.

The [XCNT policy](memory-wait-diagnostics.md#xcnt-replay-source-diagnostics)
covers SMEM sources and VMEM sources in multi-group replay mode. Single-group VMEM,
atomic replay ordering, and scheduling/source-lifetime fields such as `vm_vsrc`,
`va_vdst` and other dependency fields remain outside this checker. Parity and conditional barrier completion
protocols are not used to prove register readiness. Instruction implementations that are
functional stubs, including some image and LDS-direct operations, still lack functional
qualification: decoded counter and destination coverage does not supply the missing
execution semantics.

The first diagnostic recovers that dependency to avoid cascades. Therefore the checker
is not an exhaustive listing of every later use of the same bad result. Pending state is
not serialized in checkpoints.

Tests combine decoded producer cases and wait encodings across all ten AMD architectures
with real instruction execution for scalar/vector loads, inline DS results, counter-only
producers, zero EXEC, message return units, and FLAT's lane-specific dependencies and
both counter positions. Real-kernel checks supplement these tests; they do not prove
complete recall across all kernel families.
