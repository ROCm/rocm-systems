# Memory wait diagnostic coverage

The core checker accounts for completion-counter producers independently of the
functional memory pipelines. Generated instruction metadata selects producers, then the
shared waitcheck target model determines their counter families. Register values and
memory effects are still computed eagerly.

## Architectures and queues

| Architecture | Completion queues |
|---|---|
| CDNA1, CDNA2, CDNA3 | `vmcnt`, `lgkmcnt`, `expcnt` |
| CDNA4 | `vmcnt`, `lgkmcnt`; the encoded `expcnt` field is unused |
| RDNA1, RDNA2, RDNA3, RDNA3.5 | `vmcnt`, `vscnt`, `lgkmcnt`, `expcnt` |
| RDNA4 | `loadcnt`, `storecnt`, `dscnt`, `kmcnt`, `expcnt`, `samplecnt`, `bvhcnt` |
| CDNA5 | `loadcnt`, `storecnt`, `dscnt`, `kmcnt`, `expcnt`, `asynccnt`, `tensorcnt` |

The [CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf),
sections 3.1 and 4.4, specifies EXPCNT as unused. CDNA4 memory
instructions therefore do not acquire an EXPCNT obligation, and static
waitcheck omits the legacy GDS EXP event for this architecture. Table 57 also
marks `GDS=1` reserved, and the generated CDNA4 instruction set has no
`ds_ordered_count`. Tests using that bit or a synthetic `ds_ordered_count`
therefore exercise defensive classification of unsupported encodings, rather
than an executable CDNA4 GDS contract. LLVM accepts `ds_gws_init` and
`ds_gws_barrier` on gfx950 and classifies them with `GDS_GPR_LOCK`, while the
CDNA4 manual omits those opcodes and calls EXPCNT unused. This source
disagreement remains unresolved; the fallback follows the manual's unused-EXP
policy.

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
conservatively as unordered. Zero waits reset this ordering state. FLAT contributes only
to memory domains used by its resolved requests; register dependencies follow the
resolved routing masks. Mixed global/shared FLAT functional execution separates
the lanes by address space; the diagnostic checker tracks their counter obligations
independently of this functional execution.

GFX12 block transfers use their issue-time M0 DWORD mask for register dependencies.
Disabled load destinations and store-data words do not require completion or replay
waits. Address registers remain replay sources even when they overlap disabled data
words. An empty M0 mask still occupies the instruction's counter positions.
Block data operands also check VGPR ownership per DWORD. Loads write and track
their owned, enabled destinations; stores consume their owned words and use the
wave's VGPR0 for enabled words outside the allocation. Replay dependencies and
incoming register checks cover those actual sources, including the VGPR0 fallback.

Legacy VMEM writeback can avoid an overwrite warning only when the pending result
and the incoming producer share an ordered completion class. On legacy RDNA,
ordinary loads, image samples and BVH results share VMcnt but keep separate
classes. A newer result supersedes a fully covered older dependency only within
the same ordered class; other classes and unordered results remain tracked.

Indexed LDS results are written to registers in issue order on every target, so
an incoming indexed LDS load does not need a wait before overwriting a pending
indexed LDS result in both the runtime and static checkers. RDNA3/3.5 section
12.5, RDNA4 sections 5.7.1.4 and 12.5, and CDNA5 sections 5.7.1.4 and 11.2 state
this directly; CDNA1-4 and RDNA1/2 section 4.4 order LGKM results of the same
type. GDS and SMEM results have no ordered class and still require a wait.

On RDNA3, RDNA3.5, RDNA4 and CDNA5, FLAT's LDS-serviced lanes share that order:
incoming LDS-routed FLAT lanes and incoming indexed LDS loads can overwrite
pending indexed LDS or FLAT LDS results in issue order. This exemption is
specific to the runtime checker and its resolved routing (see
[FLAT register readiness](#flat-register-readiness)); static FLAT checks remain
conservative. Global-routed lanes keep their separate VMEM rules. On RDNA1/2 and
CDNA1-4, overwriting FLAT LDS results still requires a wait. RDNA1/2 FLAT LDS
results still retire with indexed LDS on partial waits, as LLVM treats FLAT
counters as in order from GFX10 on.

RDNA3/3.5's FLAT ordering section places the LDS route in order with indexed LDS.
RDNA4 and CDNA5 sections 5.7.1.3-4 state the same LDS-route completion guarantee
and define LDS read completion as data available in VGPRs. CDNA5 section 10.1.1
also warns that FLAT VGPR writes can be out of order while the counters retire
in order. The checker interprets that warning in the context of independently
completing VMEM and LDS paths, consistently with the specific LDS-route guarantee; it does
not establish a CDNA5-specific exception within the LDS path.

RDNA3, RDNA3.5 and RDNA4 direct/parameter LDS loads share an ordered EXP
writeback class, separate from indexed LDS and exports, in both the runtime and
static checkers. RDNA3/3.5 section 12.2.1.2 tracks both instructions through VGPR
delivery and singles out mixed exports as potentially unordered; section 12.3
uses nonzero EXP waits to pipeline parameter loads. RDNA4's EXPcnt table makes
the same relationship explicit. CDNA5 does not support these direct/parameter
instructions or their EXPcnt class. These policies follow the manuals and have
not been qualified on physical hardware by this change.

Globally routed FLAT stores share completion order with ordinary VMEM stores,
so a sufficient partial store wait releases the corresponding gfx1250 replay
sources. Static FLAT routing is unknown, so a possible FLAT store prevents
nonzero Store waits from proving completion. In expert scheduling mode, its
source lifetime remains pending until both VMEM and LDS completion are proven,
or an explicit `vm_vsrc` wait releases it. FLAT load sources likewise require
both routes. Source reads still require their own waits, even when the destination
is an ordered overwrite.

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

FLAT admission is address-dependent, so the checker applies its memory-counter
capacity constraints after routing, before checking result writeback. The instruction
cannot use an unused domain's full counter to prove an older request complete. Address
and other source operands are read before this inference is available.

## VMEM completion and EXP source protection

The dynamic race detector treats each `MemoryIssueInfo` counter obligation as
a completion requirement. `EventRegistry::allWaitCountersSatisfied` requires
every obligation attached to an event before that event can complete and
release its destination-register or memory dependency. These obligations do
not express a separate source-register lifetime.

The obligations also drive emulated counters and plugin-visible metadata.
`MemoryPipeline::issue_counters` turns each obligation into wave counter tokens,
and execution plugins observe the metadata through `amdgpu_memory_issue_info()`.
Removing VMEM EXP obligations therefore means VMEM stores and atomics no longer
contribute EXPCNT tokens in emulation. This is the emulator's concrete accounting
choice; the hardware source-reuse uncertainty described below remains separate.

Generated VMEM store and atomic metadata therefore omits EXP completion
obligations on all supported CDNA and RDNA targets. Source-data release must
not gate a returning atomic's destination or memory completion. On legacy-counter
targets such as gfx942, gfx950 and gfx1151, a returning GLOBAL atomic followed
by `s_waitcnt vmcnt(0)` releases its result; no wait or an EXP-only wait leaves
that result pending. Generic FLAT retains its separate
VMEM/DS obligations for the resolved request domains. The FLAT-to-LDS tests
wait for LGKM and the target's VMEM/store counters before a workgroup barrier,
making the write visible to other waves without an EXP wait. The counter waits
alone do not replace that barrier.

Removing EXP can also change the whole-event ordering class. On gfx942, gfx950
and gfx1151, `memoryOrderForObligations` now gives returning GLOBAL atomics the
`VMEM` class. Combining VMEM with the former EXP/`UNORDERED` obligation made the
whole event unordered. `WaveRaceState::checkVgprWrite` accepts overlapping destinations for
producers in the same ordered class, independently of completion waits. A
returning GLOBAL atomic followed by a GLOBAL load into the same VGPR can
therefore use a final `vmcnt(0)` without a WAW report. A VALU overwrite before
that wait still reports a VGPR write race. This ordering rule does not make an
atomic's returned value ready for a read before its completion wait.

LLVM's
[`vmemWriteNeedsExpWaitcnt`](https://github.com/ROCm/llvm-project/blob/32fb4582f0be6df5437ee1cdb9a38724215ede80/llvm/lib/Target/AMDGPU/GCNSubtarget.h#L431)
restricts VMEM source-register locks to architectures before Sea Islands,
predating every supported profile. Its
[`getEventsForImpl`](https://github.com/ROCm/llvm-project/blob/32fb4582f0be6df5437ee1cdb9a38724215ede80/llvm/lib/Target/AMDGPU/AMDGPUHWEvents.cpp#L107)
also classifies FLAT/GLOBAL operations without an EXP source-lock event.
However, the CDNA1/CDNA2 and RDNA1/RDNA2 register summaries still describe
EXPCNT as counting VMEM writes whose write-data has not yet been sent to the
cache. For example, see the register-state tables in the
[CDNA1 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi100-cdna1-shader-instruction-set-architecture.pdf)
and [CDNA2 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf).
These summaries and LLVM disagree about VMEM source-data accounting. The
completion policy above does not resolve that hardware question or establish
source-reuse guarantees on every target. The narrower rule is that a
source-only obligation cannot serve as a destination or memory-completion
requirement.

Static waitcheck separately protects GDS source VGPRs and EXEC until an EXP
wait on legacy targets other than CDNA4. Generated GDS metadata retains its
LGKM and EXP obligations on those targets, but the dynamic detector's
whole-event obligations do not implement this source-only protection. GDS data
operations, such as DS loads, stores and atomics with `GDS=1`, are
unimplemented and fail when executed. GWS operations (`ds_gws_*`) do execute
with `GDS=1` on CDNA1–3 and RDNA1–3.5. They publish a zero-payload request
through the local-memory pipeline, so their retained EXP obligation adds an
EXPCNT token in emulation. That accounting is still not source VGPR/EXEC
protection, and decoded-metadata and static-checker coverage do not establish
dynamic GDS source/EXEC protection.

The external
[`race` suite](https://github.com/ROCm/rocjitsu-test-corpus#rocjitsu-race-detector-integration)
runs compiler-generated returning atomics and explicit VM-only, no-wait and
EXP-only sequences on gfx942, gfx950 and gfx1151. It also checks synchronized
FLAT-to-LDS exchanges, missing-barrier controls and the atomic-to-GLOBAL/VALU
ordering pair on all three targets. Paired old/new-metadata validation of that
ordering pair was performed on gfx942. These are emulation tests. Generator tests
cover the VMEM metadata contract across all ten profiles, while decoded C++ tests
retain older-target GDS obligations and check the CDNA4 fallback policy.

## FLAT register readiness

FLAT contributes to VMEM when at least one lane requests global/scratch memory and
to DS when at least one lane requests LDS memory. This conditional-participation
rule is the runtime model across AMD GPU targets; physical validation on every
architecture is not established. Its global/scratch and LDS portions can complete independently;
RDNA4 ISA section 5.7.1.3 describes complementary lane masks for these portions.
The checker uses the resolved address of each lane to associate a returned VGPR
lane with its counter.
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
| Generic FLAT loads, stores and atomics | Only the VMEM/DS queues used by resolved requests; returned lanes depend on DS for the shared aperture and VMEM for global/scratch |
| LDS, including permutation, swizzle and DS no-op | DS/legacy LGKM queue; returning forms track registers |
| GDS data operations (decoded metadata; execution unimplemented) | LGKM and, on legacy targets other than CDNA4, EXP. Static waitcheck separately protects source VGPRs and EXEC |
| GWS (`ds_gws_*`; executes on CDNA1–3 and RDNA1–3.5) | Zero-payload request on the DS/legacy LGKM queue with the GDS completion class, plus EXP; no register results. Static waitcheck separately protects source VGPRs and EXEC |
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

Zero-EXEC vector producers generally still contribute positions behind pending requests;
an empty queue may skip them. FLAT contributes no VMEM/DS position without a requesting
lane. Scalar producers do not depend on EXEC. Prefetches that do not
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
