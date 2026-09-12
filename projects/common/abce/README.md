# ABCE — Accelerated Blit Copy Engine

**Status: draft, circulated for comment.** Nothing in the tree consumes ABCE yet; it is a
standalone header-only library at `projects/common/abce`. The interface is still cheap to
change, so this is the right moment for feedback. See
[Assumptions](#13-assumptions) and [Questions for reviewers](#14-questions-for-reviewers) at the
end.

---

## 1. What this is

ABCE builds SDMA (blit/copy engine) command streams and submits them to SDMA rings. It is the
packet-building and engine-scheduling logic of a blit path, factored out of any particular
runtime: it does not allocate queues, does not create signals, does not own a device, and never
dereferences a user pointer to discover where memory lives.

### 1.1 Motivation

Today this logic lives inside ROCr's `BlitSdma`, entangled with ROCr's queue, agent and signal
objects, and reachable only by asking ROCr to perform a copy. That costs us the following.

- **HIP graphs cannot pre-capture copy work.** The entire point of a graph is to pay the
  dispatch cost once at `hipGraphInstantiate` and leave the launch path with nothing to do but
  publish pre-formed packets. That holds for kernel nodes, whose AQL packets are built at
  instantiate time. It does not hold for memcpy nodes: ROCr builds SDMA packets *inside* its
  submit call and offers no way to obtain the packets without also submitting them, so every
  graph launch rebuilds them and memcpy nodes keep paying the dispatch overhead a graph exists
  to remove. `MapCopy` is designed to be exactly that missing call — it returns fully formed,
  relocatable packets without touching a ring, leaving `Submit` as reserve + `memcpy` +
  doorbell. Verified: the bytes `Submit` places in the ring are identical to the bytes `MapCopy`
  produced, with no ring offset or engine identity patched in, and a mapped frame can be
  retargeted to a different engine before submission. (Holding a plan across launches is the
  intended model — see [A3](#13-assumptions) — but still needs the lifetime work in
  [Q1](#14-questions-for-reviewers).)
- **No reuse from CLR/HIP.** A caller that wants SDMA packets must take a dependency on ROCr's
  internal blit path. There is no library form to link against, so CLR cannot build copy packets
  itself even where it already knows everything required.
- **No way to exercise packet construction without a GPU.** Packet layout bugs — a wrong field
  offset, an off-by-one count, an uninitialized reserved bit — are only observable today by
  running a copy on real hardware and noticing corruption. Packet construction over plain memory
  ought to be verifiable in a unit test on any machine. The same property opens up pre-silicon
  use: a frame is just bytes in host memory, so it can be emitted anywhere and driven into FFM or
  RocJitsu instead of a GPU. That makes ABCE a vehicle for future-hardware support and bring-up —
  SDMA packet support for a new part can be written, emitted and validated against a model before
  silicon is available, rather than waiting for a board. Packet capabilities are already keyed on
  gfx version in one place (`DetectPacketCaps`), so a new part is a capability entry plus whatever
  packet layouts are genuinely new, and `abce_decode.h` gives a model-independent check that the
  emitted stream is well-formed.
- **No device-initiated path.** SDMA packet construction is host-only code, so a kernel cannot
  enqueue a copy without a round trip to the CPU.
- **Engine selection is not reusable.** The measured per-engine bandwidth orderings, the xGMI
  affinity table and the gfx90a RAS reservation are policy that any SDMA consumer needs, but
  they are embedded in ROCr's selector rather than exposed as something a client can query,
  extend or override.

### 1.2 Design principles

Three properties drove the design:

- **Header-only and dependency-free at the core.** The builder, frame composer and ring only
  need `<cstdint>`-level headers. HSA/ROCr types appear in exactly one optional adapter header.
- **Host and device from one source.** The packet builders and the ring reserve/commit protocol
  are `__host__ __device__`, so a future device-initiated path shares the implementation rather
  than reimplementing the packet layouts.
- **Sizing and emission can never disagree.** Every packet-producing routine is paired with a
  byte-count routine derived from the same code path, because a mismatch silently corrupts a
  ring.

## 2. Layering

```
                        client runtime (CLR / HRX-systems / ROCr / test / benchmark)
                                          │
  abce_hsa.h        optional adapter: hsa_queue_t -> RingBuffer, amd_signal_t -> SignalRef
                                          │
  abce_host.h       CopyOrchestrator: validate -> select engines -> decompose -> plan -> submit
                    SdmaEnginePolicy: topology/heatmap-driven engine ranking
                                          │
  abce_frame.h      FrameComposer: prologue / bodies / epilogue, size+emit pairs
                                          │
  abce_builder.h    ABCE: one method per SDMA packet, ISA-aware layouts
  abce_ring_host.h  RingBuffer: reserve -> zero -> write -> commit -> doorbell
  abce_ring_core.h  the lock-free reserve/commit protocol (host + device)
                                          │
  sdma_packets.h    hardware packet structs and field encodings
  abce_types.h      ISA/capability detection            abce_topology.h  KFD topology reader
  abce_decode.h     packet walker (verification/debug)
```

A client normally touches only `abce_host.h` (plus `abce_hsa.h` when submitting to
ROCr-created queues). The lower layers are usable on their own — the builder alone is enough
to hand-write a packet stream, which is how the verification programs for this draft work.

## 3. The host interface

### 3.1 Two phases

`CopyOrchestrator` deliberately splits work into two calls:

| Phase | Call | Touches | Can block |
|---|---|---|---|
| Map | `MapCopy(ops, n, metadata) -> Plan` | nothing external; builds packets into scratch | no |
| Submit | `Submit(plan) -> SubmitResult` | rings, doorbell, completion signal | yes (ring space) |

The split exists so that all the expensive, failure-prone work — validation, engine selection,
decomposition, packet emission — happens before any ring is touched. Once `Submit` starts
reserving ring space it must run to completion, because an abandoned reservation would leave a
permanent unpublished hole in the ring. Keeping the fallible part in `MapCopy` means `Submit`
has almost nothing left to fail on.

The split is also what lets a client inspect or override the mapping decision. A `Plan` exposes
the engine chosen for each frame plus the ranked legal alternatives, so a scheduler that knows
something ABCE does not (queue depth, an in-flight copy on that engine) can rewrite
`frame.engine` before submitting.

Most importantly, it is what makes copy work pre-capturable: `MapCopy` is the "compile" step a
HIP graph can run at instantiate time, and `Submit` is the cheap replay step
(see [§1.1](#11-motivation)).

`Dispatch()` is a convenience wrapper for callers that do not want the seam.

### 3.2 Minimal use

```cpp
#include "abce_host.h"

// One builder per device; cheap, holds only ISA-derived capability flags.
abce::ABCE builder(abce::IsaVersion{9, 4, 2});
abce::CopyOrchestrator orchestrator(builder);

// Register each SDMA ring once. hw_engine_id is the hardware SDMA engine index,
// which is what the policy's heatmap and affinity tables are keyed on.
abce::EngineAffinity affinity{};
affinity.hw_engine_id = 2;
orchestrator.RegisterEngine(/*index=*/0, &ring, affinity);

// Describe the transfer. Endpoints are how the policy tells H2D from D2H from
// P2P without inspecting pointers.
abce::CopyOp op{};
op.kind = abce::OpKind::kLinear;
op.src = host_ptr;
op.dst = device_ptr;
op.size = 1 << 20;
op.src_end = {abce::EndpointKind::kHost, abce::kAnyDevice};
op.dst_end = {abce::EndpointKind::kDevice, /*device_id=*/0};

abce::CopyMetadata metadata{};
metadata.out = abce::SignalRef(signal_value_ptr, /*completion_value=*/0);

abce::Plan plan = orchestrator.MapCopy(&op, 1, metadata);
if (!plan.valid()) return Translate(plan.status, plan.failed_op);
if (orchestrator.Submit(plan).status != abce::SubmitStatus::kSuccess) return kRetry;
```

For ROCr-created SDMA queues, `abce_hsa.h` removes the ring plumbing:

```cpp
#include "abce_hsa.h"

abce::HsaQueueRing queue_ring;
queue_ring.Init(sdma_queue);                       // validates it really is an SDMA queue
abce::RegisterHsaEngine(orchestrator, 0, queue_ring);

metadata.out = abce::HsaSignalRef(amd_signal, /*completion_value=*/0);
```

### 3.3 Input types

**`CopyOp`** (88 bytes) — one logical copy. A flat struct with a per-kind tail; `kind` selects
which fields are live.

| Field | Used by | Meaning |
|---|---|---|
| `kind` | all | which primitive (see [§4.1](#41-operation-kinds)) |
| `src`, `dst` | linear, fill, swap, indirect | endpoints; for swap these are the two exchanged regions |
| `dsts`, `num_dsts` | multicast, broadcast | destination array, borrowed (caller keeps it alive) |
| `size` | all but rect | bytes; for fill a byte count that must be a multiple of 4 |
| `size2` | swap | endpoint B's size when asymmetric; honored only on the fused path |
| `rect` | rect | `CopyRectDesc` geometry, borrowed |
| `indirect_src`, `indirect_dst` | indirect | which side is an address list |
| `src_end`, `dst_end` | all | residence: host or device, plus device id |

**`CopyMetadata`** (96 bytes) — parameters for the batch as a whole, notably one completion
signal for the entire batch rather than per copy.

| Field | Meaning |
|---|---|
| `out` | the single completion signal, plus optional interrupt mailbox |
| `deps`, `num_deps` | signals gating the whole batch; an already-satisfied dep is elided at map time |
| `engine_mask` | restrict to a subset of registered engines (0 = all) |
| `max_engines` | cap participating engines, applied after the mask |
| `linear_batch_mode` | force back-to-back or fan-out instead of the size heuristic |
| `multicast_mode` | force the multicast packet or per-destination fan-out |
| `timestamps` | optional start/end SDMA global-clock slots ([§7](#7-profiling)) |
| `execution_descriptor` | optional 32-bit slot stamped with the engines used and transfer kind ([§7](#7-profiling)) |
| `coherency` | per-batch HDP-flush / GCR triggers, further gated by platform caps |
| `prefer_fused` | use fused wait/signal packets where supported (default true) |

### 3.4 Output types and error surface

`Plan` is move-only and single-use: `Submit` marks it submitted and a second `Submit` is
rejected. `MapStatus` distinguishes the failure modes, and `Plan::failed_op` identifies the
offending index for the per-op ones.

| `MapStatus` | Cause |
|---|---|
| `kInvalidArgument` | null/zero field, host→host copy, missing output signal, bad rect geometry |
| `kNoEngine` | no registered engine survives `engine_mask` |
| `kNoLegalEngine` | policy rejected every candidate for this transfer |
| `kUnsupportedOperation` | op needs hardware this ISA lacks (e.g. `kIndirect` below gfx125+) |
| `kTooManyOperations` | batch exceeds 64 Ki entries, or a frame/signal count overflows |
| `kAllocationFailed` | scratch allocation failed |
| `kMissingCoordinationScratch` | the batch fans out and needs a coordination word, but `SignalRef::coordination_scratch` is null |

`SubmitStatus` is deliberately much narrower — `kInvalidPlan`, `kFrameTooLarge`,
`kRingUnavailable` — reflecting that submission is nearly infallible by construction.

## 4. Capabilities

### 4.1 Operation kinds

| Kind | Description | Hardware |
|---|---|---|
| `kLinear` | contiguous copy, auto-chunked to the per-packet limit | all |
| `kMulticast` | one source to up to 1024 destinations in a single packet | gfx125+ |
| `kBroadcast` | one source to exactly 2 destinations | pre-gfx125+ (also chosen internally) |
| `kSwap` | bidirectional exchange of two buffers, sizes may differ | all; fused on gfx125+ |
| `kIndirect` | gather/scatter where src and/or dst is an address list | gfx125+ only |
| `kCopyRect` | 2D/3D strided sub-window copy | all; never fused |
| `kFill` | constant 32-bit pattern fill | all |

A batch may mix kinds freely; each is decomposed independently and placed on an engine.
Multicast degrades gracefully: on hardware without the multicast packet, ABCE emits
2-destination broadcast packets for small transfers and per-destination linear copies for large
ones, so a client can request `kMulticast` unconditionally.

### 4.2 ISA support

Capabilities are derived from the gfx version rather than configured, using an OSS4/OSS5/OSS7
capability table. Verified output of `DetectPacketCaps` / `DetectDefaultPlatformCaps`:

| ISA | GCR | scope fields | fused wait/signal | max linear copy |
|---|---|---|---|---|
| gfx90a (9.0.10) | no | no | no | `0x3fffffff` |
| gfx942 (9.4.2) | no | no | no | `0x3fffffff` |
| gfx10.3 | yes | no | no | `0x3fffffff` |
| gfx11 | yes | no | no | `0x3fffffff` |
| gfx1250 (12.5.0) | no | yes | yes | `0x3fffffff` |

`PlatformCaps` carries the decisions the ISA *cannot* imply — device atomic support, whether to
emit an HDP flush, whether the driver already owns GCR — because the same IP ships in
configurations that differ on all three. Clients are expected to override the defaults from
link topology; an xGMI host link, for instance, does not want the HDP flush.

### 4.3 Packet-level features

Beyond the copies themselves the composer can emit: dependency polls (32- or 64-bit), HDP
flush, GCR invalidate/writeback, SDMA global-clock timestamps bracketing the copy, the
completion update (atomic decrement, or fence where atomics are unavailable), and an
interrupt fence+trap pair for signals with an event mailbox.

## 5. Engine selection

Engine choice is a policy object, not hardcoded. `MakeSdmaPolicy()` picks one from the ISA at
construction; `SetEnginePolicy()` replaces it with a client callback; with no policy the
orchestrator falls back to round-robin over registered engines.

The base policy classifies each transfer from its endpoints into one of four bands — H2D, D2H,
local D2D (same device), P2P (cross-device) — and asks a per-band hook for a ranked list of
hardware engine ids, which are then resolved to registered indices, deduplicated and filtered
against the batch's candidate mask.

**A client never declares what an engine is for.** Registration supplies only the hardware engine
id. Band membership comes from measurement (the H2D/D2H heatmaps) or from the driver (KFD's
`num_sdma_engines` / `num_sdma_xgmi_engines` split, which `LoadTopologyFromKfd` feeds to
`SetSdmaEngineSplit`); absent both, every band widens to all registered engines and selection
degrades to load balancing. An earlier design had clients tag each engine with an `EngineClass`
of host/xGMI/local; measurement showed that to be actively wrong, because a physical class does
not predict which engines are best for a band.

Note which direction the xGMI split is used in: it *prefers* engines for P2P and never excludes
them from host copies. On MI300X, KFD classes engines 2–15 as xGMI, yet four of them outrank both
non-xGMI engines for H2D.

The distinction between a ranking and a reservation matters, because only a reservation should be
able to make a copy unmappable:

- A **ranking** is a preference, so the hook appends every remaining engine behind it. The
  heatmap is a ranking: a client that registers engines the profile does not mention still gets a
  legal engine, in measured order where the profile has an opinion.
- A **reservation** is a hard exclusion, expressed by returning a short list. gfx90a's RAS rule
  is one — SDMA0 is listed for H2D and appears nowhere else, so D2H fails rather than using it —
  as is the cross-hive check, which returns nothing at all.

Two arch policies exist today:

- **`Gfx94xSdmaPolicy`** (gfx94x/95x) — H2D/D2H order comes from bandwidth heatmaps recorded in
  the source, because per-engine SPX bandwidth is markedly non-uniform (on gfx942/16-SDMA some
  engines reach ~20 GB/s D2H against ~55 GB/s for the best). P2P consults an xGMI
  `physical_id`-keyed affinity table for the optimal first choice, then appends the rest of the
  xGMI band, and requires both GPUs to be in the same hive.
- **`Gfx90aSdmaPolicy`** — encodes the RAS restriction that SDMA0 may only drive H2D, by
  listing SDMA0 for H2D and SDMA1 for D2H and never listing SDMA0 elsewhere.

Verified ranking on gfx942, 8 of 16 SDMA engines registered under hardware ids 0–7, with the
heatmap loaded and KFD's real split for this part (`num_sdma_engines 2`, `num_sdma_xgmi_engines
14`):

| Transfer | Ranked engines |
|---|---|
| H2D, D2H | 2, 4, 6, 5, 1, 7, 0, 3 (heatmap order, every engine offered) |
| P2P (dev0→dev1) | 2–7 (the xGMI band, intersected with what is registered) |
| local D2D (dev0→dev0) | 0–7 |
| P2P with unset device ids | 2–7 (xGMI band) |
| P2P, split unknown | 0–7 (widened, not empty) |

That fourth row is a deliberate choice worth noting: two *unknown* device ids are treated as
cross-device, since assuming same-device would steer a real P2P copy away from the xGMI band.
The last row is the reason the split is optional — a client with no KFD topology gets load
balancing across everything rather than an unmappable P2P copy.

Placement across engines is separate from ranking. A batch either goes **back-to-back** on one
ring or **fans out** across many:

- Back-to-back is chosen for an all-linear batch whose copies **total** ≤ 512 KiB. The
  threshold is on the batch total rather than per copy because one ring serializes the whole
  batch: it pays the full total where E rings each pay about total/E, against a fixed cost for
  coordinating a fan-out at all. Measured on an 8-rank all-gather batch (MI300X, 16 SDMA
  engines), fan-out holds a ~27 µs floor however small the copies get while one ring starts near
  13 µs, so one ring wins by a wide margin when there is little to serialize — 15.6 vs 27.7 µs
  at 64 KiB total — and the curves cross just under 1 MiB: 22.4 vs 28.2 µs at 512 KiB total,
  then 29.6 vs 31.6 µs at 1 MiB in fan-out's favour, and 37.2 vs 92.2 µs by 4 MiB.
- Stating that rule per copy is a trap worth calling out, because it is what an earlier
  revision did. A per-copy window is only correct at the entry count it was tuned on: 64 × 8 KiB
  serializes exactly as many bytes as 8 × 64 KiB, so a 256 KiB/copy window let an 8 × 256 KiB
  batch (2 MiB total) serialize onto one ring at 51.6 µs where fan-out finished in 32.3 µs — the
  one size at which ABCE lost to the stock blit path.
- Fan-out is parallel-first: every legal ring takes a copy before any ring takes a second, and
  only when copies outnumber rings do the extras prefer their affinity ring. Very large copies
  (≥ 1 GiB) instead group `kMaxCopiesPerEngine` copies per ring.

## 6. Fan-out coordination and completion

The hard part of fan-out is that a batch spread over N rings must still produce exactly one
completion signal transition, with no ordering violations. ABCE runs that protocol in a 64-bit
*coordination word* it owns outright, kept out of the completion signal's value so the value only
ever holds legal signal states:

- **low 32 bits** — a fan-in counter. Each participating frame decrements after its last body;
  the coordinator polls for the count to drain to zero.
- **bit 62** — a start gate. Non-coordinator frames poll for it to clear before their first
  body; the coordinator clears it after its prologue.

The word is `SignalRef::coordination_scratch`, which `HsaSignalRef()` points at the signal
object's own trailing reserved words — a naturally aligned 64-bit slot in the same 64-byte
object, so coordination shares a cache line with the value it guards and costs no extra
allocation. The protocol drains the word back to zero, so a completed plan leaves no residue.
A fan-out that needs a coordination word and was not given one is rejected at map time with
`kMissingCoordinationScratch` rather than falling back to the signal value.

`frames[0]` is the coordinator and carries the prologue (dependency polls, timestamp start, HDP
flush, GCR invalidate) and the epilogue (fan-in poll, GCR writeback, timestamp end, completion
update, interrupt). `Submit` arms the coordination word only after every ring reservation has
succeeded, so a failure can never strand a gate or a count that nothing will clear. In the gated
form ABCE does not write the caller's signal at all — the client's own pending value stands,
exactly as for a single-engine copy.

Two optimizations drop parts of this, both verified. Neither needs a coordination word: in both,
the bodies' own signals *are* the completion transitions, so they count the signal value itself
down to the completion value and never put anything illegal in it.

**Direct fused completion** — a single-frame gfx125+ batch with no post-copy work needs no
epilogue at all; the fused copy packet's own SIGNAL reaches the completion value. A 4 KiB H2D
copy is then `HDP_FLUSH` + `COPY_LINEAR_WAITSIGNAL`, 72 bytes total.

**Gate-free fused fan-out** — when every body is a fused packet and the coordinator has no
ordering work, all frames start immediately and every frame signals, including the coordinator.
`Submit` arms the signal to N above the completion value and the N body signals walk it down.
4 × 1 MiB fans out to four identical 48-byte frames of one `COPY_LINEAR_WAITSIGNAL` each, with no
gate, no fan-in poll and no completion atomic — 192 bytes against 376 for the gated form.

Note the precondition on that second one. "No ordering work" includes *no HDP flush*, and HDP
flush is on by default for gfx ≥ 9, so with default settings the gate-free path does not
trigger even on gfx1250. Verified:

| `coherency.emit_hdp_flush` | start gate | epilogue | fan-in | coordinator frame |
|---|---|---|---|---|
| 1 (default) | required | required | 3 | `HDP_FLUSH ATOMIC COPY_LINEAR POLL_MEM_64 ATOMIC` |
| 0 | not required | not required | 4 | `COPY_LINEAR_WAITSIGNAL` |

The gate is genuinely necessary when the flush is present — an HDP flush on the coordinator's
ring orders nothing on the other rings, so the other rings must wait for it. The consequence is
simply that the fast path belongs to clients that own their own HDP coherency and turn the flush
off; it is not a path a default configuration should expect to hit.

## 7. Profiling

ABCE does not profile itself; it exposes what a profiler cannot reconstruct from the outside. A
tool watching a completion signal can see *that* a copy finished, and with timestamps *when* it
ran, but not what executed it, which of a dozen SDMA engines carried it, or what kind of transfer
it was — that knowledge exists only inside the mapping decision. Two slots close that gap, both
caller-placed addresses ABCE writes and never allocates.

**Timing — `CopyMetadata::timestamps`.** Two 64-bit slots. The coordinator frame reads the SDMA
global clock into `timestamps.start` in its prologue and into `timestamps.end` in its epilogue,
immediately before the completion update, so the interval brackets the whole batch rather than one
frame. Both slots are written or neither is; requesting them forces a prologue and epilogue onto
the coordinator even where the batch would otherwise have gone without (which costs the direct
fused completion path described in [§6](#6-fan-out-coordination-and-completion)). For a fan-out
the interval covers the entire batch under one signal, not per-engine spans — ABCE deliberately
reports one interval per completion signal, because that is the unit a client waited on.

**Attribution — `CopyMetadata::execution_descriptor`.** One 32-bit slot, stamped with an
`ExecutionDescriptor`:

| bits | field |
|---|---|
| 0–15 | instance mask, interpreted per engine kind — for `kSdma`, bit *h* means hw engine *h* ran part of the batch |
| 16–18 | `TransferKind`: unknown / H2D / D2H / P2P / local D2D |
| 19 | batch mixed transfer kinds, so the kind field reads `kUnknown` |
| 20–22 | `ExecutionEngineKind`: none / `kSdma` / `kCompute` / `kCpu` |
| 23–31 | reserved, zero |

Five choices in that layout are worth stating, since they are the contract:

*The engine kind is explicit, not assumed.* A completion signal is not the property of any one
engine. The same signal object an SDMA batch completes can instead be completed by a compute blit
kernel or a host memcpy, and nothing in the signal value says which. ABCE only ever writes
`kSdma`, because it builds nothing else; the other values exist so a client whose fallback paths
complete the *same* signal can stamp the same slot. That gives a consumer one field to read
instead of a heuristic — ROCr today infers "SDMA or blit kernel?" by zeroing its SDMA timestamps
before a copy and testing them afterwards (`SharedSignal::CopyPrep` / `GetRawTs`). The two paths
do not even record timestamps in the same place, so the engine kind additionally tells a reader
*which timestamp pair to look at*. Three bits rather than two: the spares cost nothing, since the
reserved tail absorbs them, and a field written by more than one component should have room for
executors that have not come up yet.

*Hardware engine ids, not ABCE's registered indices.* A registered index is a client-local
numbering ABCE assigns at `RegisterEngine`; it means nothing to a tool inspecting the machine.
The mask uses the `EngineAffinity::hw_engine_id` the client registered, which is the id the rest
of the stack and the hardware agree on. The mask is only meaningful relative to the engine kind,
which is why the kind is in the contract: a compute kernel has no equivalent of an SDMA engine id
and may leave the mask zero.

*A mask, not a single id.* A fan-out has no single engine, and a field that could only name one
would have to lie or report nothing for exactly the case a profiler most wants to see. The mask
makes a 4-engine batch self-describing, and `kMaxEngines` is 16, so 16 bits is exact rather than a
truncation.

*Written by `Submit`, not `MapCopy`.* A client is allowed to retarget `PlanFrame::engine` between
the two phases, so only `Submit` knows which engines actually ran. It is a single relaxed-cost
host store issued after every ring reservation succeeds, ordered ahead of the copy's completion by
the ring publish that follows; no packets, no device work, and nothing added to the ring.

*Validity is the engine kind, not a magic number.* `kNone` is the zero value, so an untouched
field reads as "nobody stamped this" and `ExecutionDescriptor::Valid()` is just that test.
Validity deliberately does not hang off the instance mask: that would force every writer to invent
an instance, and would make a legitimately mask-less stamp unreadable.

An engine *class* — xGMI versus PCIe within SDMA — is a different question from engine kind, and
is deliberately not reported, though an earlier draft of the layout spent two bits on it. ABCE no
longer has one to report ([§5](#5-engine-selection)), and it would mislead if it did: on MI300X
the driver classes engines 2–15 as xGMI while the measured H2D ranking prefers four of them over
both non-xGMI engines, so "an xGMI engine ran a host copy" is the normal case rather than an
anomaly worth a bit. Engine kind is the opposite: it separates hardware blocks whose behaviour
genuinely differs, down to where they record their timestamps. A reader that wants to group SDMA
engines has the hardware ids and can apply whatever split it trusts.

Mixed batches are reported as mixed rather than guessed at. A batch may legitimately contain both
H2D and D2H copies under one signal; `Plan::transfer_kinds` is a bitmask of every kind present,
`UniformTransferKind()` returns a single kind only when the batch is homogeneous, and bit 19 tells
the reader the field is genuinely unattributable instead of silently labeling the batch by its
first copy. Classification comes from `ClassifyTransfer()`, which derives the kind from endpoint
kinds and device ids only — it never dereferences a client pointer or queries the driver, so it
costs nothing and works for pointers ABCE cannot inspect.

**Placement is the client's decision.** Both slots are addresses, so ABCE commits to no signal
layout. `HsaExecutionDescriptorSlot()` offers the obvious placement — `amd_signal_t::reserved1`,
the spare 32-bit reserved word, which shares the signal's cache line and needs no allocation — but
ABCE never writes it unless a client assigns it into the metadata, because which bytes of a signal
may be repurposed is an ABI decision that is not ABCE's to make. `amd_signal_t` is a frozen format
that may be shared across processes; a client that cannot spend `reserved1` should point the
descriptor at its own wrapper struct. A client that would rather keep this metadata in its own
records than in a slot ABCE writes can leave `execution_descriptor` null and call
`DescribeExecution(plan)` for the same word. There is precedent for exactly that: ROCr keeps its SDMA
timestamps in `SharedSignal::sdma_start_ts`/`sdma_end_ts` rather than in `amd_signal_t`, because
the alignment SDMA needs conflicts with the frozen layout. Note also that the descriptor cannot go
in the signal's *other* reserved word: `reserved3` is the 64-bit coordination word
([§6](#6-fan-out-coordination-and-completion)), which hardware atomically modifies during the
copy, and which must stay 8-byte aligned because the gfx125+ 64-bit poll encodes only `addr[63:3]`.

What this replaces is inference. ROCr today distinguishes "was this copy done by SDMA or by a blit
kernel?" by zeroing its timestamps before a copy and checking afterwards whether they came back
non-zero. A descriptor answers that question directly, and answers the questions that heuristic
cannot: which engine, how many engines, and which direction across the fabric.

## 8. Ring protocol

The SDMA packet processor consumes a ring strictly in order and does not tolerate the write
index moving out of reservation order. `RingBuffer` is therefore single-consumer,
multi-producer, with the protocol in `abce_ring_core.h` shared verbatim with the (not yet
landed) device ring:

1. **Acquire** — advance a monotonic reserve cursor by one CAS, so producers write disjoint
   regions with no lock. The region is zeroed, which both satisfies the builders'
   zeroed-buffer contract and turns any unused pad tail into NOPs.
2. **Write** — the caller fills the region.
3. **Release** — spin until the commit cursor reaches this reservation's start, then advance the
   hardware write pointer and ring the doorbell, then release the next producer. The doorbell is
   thus monotone even when producers finish writing out of order.

A payload that would straddle the physical ring end is handled by reserving and publishing the
wrap tail as a *separate* NOP region, then retrying the payload at offset zero. Keeping the
padding separate means any payload smaller than the ring can still make progress once the ring
drains, even when padding plus payload exceeds the ring size. Verified on a 256-byte ring: a
200-byte reservation at offset 0, then a 100-byte reservation lands at monotonic 256 (offset 0)
after a 56-byte NOP pad, leaving the doorbell at 356.

Attaching to an already-in-use queue seeds both cursors from the queue's current write pointer,
so the first `Acquire` cannot hand out bytes the engine still owns.

Multi-ring submission acquires and publishes rings in a fixed global engine order rather than
frame order. Without that, two concurrent plans with opposite frame orders could each hold one
unpublished reservation while waiting for the other's ring — a cross-ring commit deadlock.

## 9. What a client must supply

| Requirement | How |
|---|---|
| Ring memory + control words | `RingConfig` with either mapped `write_ptr`/`read_ptr`/`doorbell`, or a `RingQueueOps` callback trio for APIs that keep them opaque |
| Ring size a power of two | asserted at `Init` |
| Completion signal | a device-visible 64-bit value location via `SignalRef` |
| Coordination scratch | a device-visible 64-bit word via `SignalRef::coordination_scratch`, for batches that fan out; `HsaSignalRef()` supplies it from the signal's reserved words |
| Memory residence | `src_end`/`dst_end` per op; ABCE never inspects pointers |
| Platform capabilities | `PlatformCaps` (atomics, HDP flush, GCR ownership) |
| Device ids | ABCE-local indices, assigned by the client at registration |
| Engine hardware ids | `EngineAffinity::hw_engine_id` per registered ring — the only per-engine metadata ABCE asks for, and what the heatmap and xGMI split are keyed on |
| Topology (optional) | `LoadTopologyFromKfd()` on Linux; elsewhere, or when KFD reads are undesirable, `SetTopology(TopologyData)` and/or `InitDeviceProfile(total_sdma, num_non_xgmi_sdma)` |
| Profiling slots (optional) | device-visible addresses for `timestamps` and `execution_descriptor` ([§7](#7-profiling)); ABCE writes them and allocates neither |

Deliberate non-responsibilities: ABCE never allocates or frees rings, queues, signals or device
memory; never calls into a driver; and holds no locks across a submission other than the ring's
own commit ordering.

## 10. Tuning knobs

Read once at first use, decimal or `0x` hex, so thresholds can be swept without a rebuild.

| Variable | Default | Effect |
|---|---|---|
| `ABCE_LINEAR_B2B_MAX_TOTAL` | 512 KiB | batch total at or below which a batch stays back-to-back on one ring |
| `ABCE_BROADCAST_MAX` | 16 KiB | size below which pre-gfx125+ multicast uses broadcast packets |

## 11. Verified behavior

Every claim above was checked against the headers as they stand, by building small programs
against the include directory and walking the emitted frames with `abce_decode.h`. Representative
output:

```
gfx942:  single 4 KiB H2D          1 frame,  84 B : HDP_FLUSH COPY_LINEAR ATOMIC
gfx942:  4x 64 KiB H2D             1 frame, 168 B : HDP_FLUSH COPY_LINEAR x4 ATOMIC
gfx942:  4x 1 MiB H2D              4 frames        : coordinator HDP_FLUSH ATOMIC COPY_LINEAR POLL_REGMEM ATOMIC
                                                     others     POLL_REGMEM COPY_LINEAR ATOMIC
gfx942:  multicast 4 KiB, 4 dsts   1 frame, 128 B : HDP_FLUSH COPY_LINEAR_BROADCAST x2 ATOMIC
gfx1250: single 4 KiB H2D fused    1 frame,  72 B : HDP_FLUSH COPY_LINEAR_WAITSIGNAL
gfx1250: multicast 4 KiB, 4 dsts   1 frame,  96 B : HDP_FLUSH COPY_LINEAR_MULTICAST_WAITSIGNAL
```

The decoder landing exactly on each frame's end byte is itself the check that sizing and
emission agree.

Engine banding was checked against this machine's driver rather than against assumptions. KFD's
`num_sdma_engines`/`num_sdma_xgmi_engines` were read from sysfs (2 and 14 on MI300X), and its
per-link `recommended_sdma_engine_id_mask` was compared to the hardcoded `physical_id` affinity
table over every peer link of an 8-GPU node: 56 links, 56 agreements, 0 mismatches. With that
split applied, host bands still offer every registered engine in measured order, P2P narrows to
the xGMI band, an unknown split or a registration that misses the band widens rather than
emptying, and gfx90a's SDMA0 RAS reservation still refuses to appear for D2H.

Back-to-back placement is checked on the property that matters, that the decision follows the
batch total and not the entry size. A batch of 8 × 64 KiB maps to one frame while 64 × 64 KiB —
identical per copy, eight times the bytes — fans out to eight, which is precisely the distinction
a per-copy window cannot express. 64 × 8 KiB and 8 × 64 KiB agree, since both move 512 KiB. The
cap is confirmed inclusive (512 KiB total stays, 512 KiB + 8 fans out), a single multicast is
counted per destination, `kForceBackToBack`/`kForceFanOut` still override the size rule, and a
lone copy of any size is never treated as a back-to-back batch.

The profiling contract was checked the same way. `ClassifyTransfer()` returns the expected kind for
all seven endpoint combinations (including an unknown device id, which classifies as P2P, and
host-to-host, which is rejected upstream); `Plan::transfer_kinds` accumulates correctly and
`UniformTransferKind()` reports mixed batches as `kUnknown` with bit 19 set. Registering engines
under deliberately non-identity hardware ids (4, 5, 9) confirms the descriptor's instance mask
carries hardware ids rather than registered indices — a single-engine H2D on registered engine 0
stamps `0x0010`, and a 4 × 1 MiB fan-out stamps `0x0230` with exactly one mask bit per
participating frame. Every value the 3-bit engine-kind field can hold survives encoding, ABCE
always stamps `kSdma`, a zero word decodes as `kNone` and reads invalid, and a mask-less
`kCompute` stamp — the case that motivated moving validity off the mask — still reads valid. The
slot is untouched until `Submit`, the reserved bits stay zero at field saturation, and arming the
descriptor leaves the `reserved3` coordination word intact.

## 12. Known gaps

- **No tests in tree.** `CMakeLists.txt` defines only an `INTERFACE` library. Everything
  verified for this draft was throwaway. Given the library is pure logic over plain memory, a
  GPU-free unit suite (emit a frame, walk it with `abce_decode.h`, assert the packet sequence
  and byte count) is both cheap and the main argument for the design; it should land with or
  before any integration.
- **`abce_device.h` does not exist.** `abce_ring_core.h` and `abce_ring_host.h` both describe
  sharing their protocol with a `DeviceRing` in that header. The sharing is real and the core is
  already `__host__ __device__`, but the comments describe a file that is not here yet.
- **Windows builds only under Clang.** The public header chain is no longer POSIX-only: the KFD
  sysfs reader is gated behind `ABCE_HAS_KFD_TOPOLOGY` (Linux-only, since it walks
  `/sys/class/kfd` with `opendir`), host spin-waits use `std::this_thread::yield()` instead of
  `sched_yield()`, and bit counting falls back to a portable loop off GCC/Clang. What remains is
  eight uses of GCC/Clang `__atomic_*` builtins — six in `abce_ring_core.h`, two in
  `abce_host.h` (arming the coordination word and stamping the execution descriptor) — which
  Clang accepts on Windows but MSVC does not. Closing that needs a decision, not a mechanical
  fix — see [Q4](#14-questions-for-reviewers).
- **No integration.** Nothing outside `projects/common/abce` refers to ABCE, so none of this has
  run against real hardware through the orchestrator; the heatmap numbers come from measurements
  taken through ROCr's selector, not through this code.
- **Mixed error strategy.** Most of the API returns status codes, but the rect path throws
  `std::invalid_argument` on out-of-range pitch/slice and `MapCopy` catches it. Throwing from
  headers that are also compiled for device is awkward.

## 13. Assumptions

These are stated as decisions rather than open questions. They are load-bearing, so a reviewer
who disagrees should push back — but the intent is to build to them.

**A1 — a fill is host-source to device-destination, and the caller states it.** A fill has no real
source, but it is still classified as a host→device operation, and `dst_end.kind = kDevice` is a
required field rather than something ABCE infers. This is worth calling out because the default
`CopyOp` leaves both endpoints `kHost`, host→host is rejected, and a caller who populates only
the fill descriptor therefore gets `kInvalidArgument` with nothing obviously wrong in the
request. The contract is being specified, not relaxed: endpoint classification stays mandatory
for every operation kind, so no path can reach engine selection without the client having said
where the memory lives.

**A2 — fan-out coordination lives in a reserved field of the signal, not bit 62 of its value.**
*(Implemented.)* The start gate and fan-in counter used to be packed into the completion signal's
own 64-bit value, so between arm and completion the signal transiently held something that was not
a legal HSA signal state. They now live in `SignalRef::coordination_scratch`, which
`HsaSignalRef()` points at `amd_signal_t::reserved3` — a naturally aligned 64-bit slot at offset
56 of a struct that is exactly one 64-byte cache line (`static_assert`ed), so coordination shares
a cache line with the value it guards and costs neither an allocation nor a line.

Three things fell out of the separation, all verified ([§6](#6-fan-out-coordination-and-completion)):
the fan-in counter drains to a plain zero instead of a value biased by the caller's completion
value; ABCE no longer writes the caller's signal at all for a gated fan-out, so multi-engine and
single-engine plans now make the same assumption about who arms the signal; and because the
protocol subtracts the gate and decrements the count to zero, a completed plan leaves the signal's
reserved words exactly as it found them.

The cost is a new requirement on non-HSA clients: a bare `uint64_t*` completion signal is no
longer sufficient for a batch that fans out, since the scratch must also be device-visible memory
the SDMA engines can poll and atomically update. That case is rejected at map time with
`kMissingCoordinationScratch` rather than silently degrading. It is worth knowing that this is
size-dependent — a client that only ever issues small copies never supplies scratch and never
notices, until a copy is large enough to fan out.

**A3 — a graph stores each node's mapped packets at capture time, baked signal addresses
included.** This is the expected model rather than a problem to design around: capture is exactly
the point at which packets should be built and retained, and the signal and dependency addresses
resolved at that point are part of the captured node's state. A node whose addresses change is
re-captured, which is work a graph already does per node. So the fact that `MapCopy` bakes
addresses into the packet bytes is a property to rely on, not one to abstract away. What remains
is purely a lifetime question — who owns the bytes the graph is holding — which is
[Q1](#14-questions-for-reviewers).

**A4 — the engine heatmap stays compiled in and is not runtime-overridable.** These orderings are
measured properties of a specific part, not deployment policy, and a runtime override would
invite tuning by environment variable and make reported performance unreproducible. New parts
keep meaning a new table plus a policy subclass. The follow-up work is consolidation rather than
configurability: the two near-duplicate selection paths (`SelectEngine` and
`PickBalancedEngine`), and `RecommendedMask`, which is parsed out of KFD and exposed on the
policy but never consumed by ranking. Consuming it would retire the hardcoded xGMI affinity
table, which [§11](#11-verified-behavior) confirms it agrees with on every peer link.

## 14. Questions for reviewers

**Q1 — what must a `Plan` become to be held and replayed?** [A3](#13-assumptions) commits to a
graph holding each node's packets across launches, and the hard part is already done: the packets
are relocatable and `Submit` copies them verbatim. Two mechanical things stand in the way, and I
think they want one answer rather than two.

The bytes are borrowed. `MapCopy` builds into a thread-local scratch buffer and
`PlanFrame::packets` is a non-owning view into it, so a plan is invalidated by the next `MapCopy`
on the same thread. This is documented, but I confirmed it is a use-after-free rather than merely
stale data: mapping a larger plan reallocates the scratch and frees the first plan's bytes.
Map-two-then-submit-two is a natural thing for a batching client to write, never mind holding a
plan until the next launch.

The plan is also single-use: `Submit` marks it submitted and a second `Submit` returns
`kInvalidPlan` (verified). So how should a plan own its bytes — a heap allocation per map, which
the scratch exists to avoid, or a per-orchestrator arena with a generation counter that `Submit`
validates? And should replay be explicit (`Rearm`/`Reset`) rather than silently allowing repeated
`Submit`, so that accidental double-submission stays an error?

**Q2 — engine capacity.** `kMaxEngines` is 16, matching MI300X, but masks are `uint64_t` and
`Plan` is a fixed 1472-byte array of 16 frames. Is 16 the right ceiling to bake in, and should
the mask type shrink to match or the engine count grow to 64?

**Q3 — where should the execution descriptor live, and is 32 bits the right budget?** ABCE takes
an address and stamps it ([§7](#7-profiling)), so the mechanism is placement-agnostic and this is a
question about the default a client should adopt rather than about ABCE's internals.
`amd_signal_t::reserved1` is free today and costs no allocation, but it is part of a frozen format
that may be shared across processes, and ROCr's own precedent went the other way — SDMA timestamps
live in the `SharedSignal` wrapper specifically to avoid the frozen layout. Should the descriptor
follow the timestamps into a wrapper struct instead, which would also lift the 32-bit ceiling?

Within 32 bits the layout spends 16 on the instance mask, which is what makes fan-out
self-describing, and 9 more on the transfer kind, mixed flag and engine kind, leaving 9 reserved.
That is enough headroom for another small field but not for the two a consumer might ask for next:
the *destination* device id for a P2P copy (the transfer kind says it crossed the fabric, but not
to where), and per-engine byte counts. Both would fit comfortably in a 64-bit slot. Is 32 bits in
the signal the right trade, or is the descriptor better off outside it from the start?

**Q4 — how far should Windows support go?** ABCE now compiles for Windows under Clang, but the
ring protocol's `__atomic_*` builtins are not MSVC constructs. The awkward part is that the same
functions compile for device code, where `__hip_atomic_*` is used, so the host branch cannot
simply become `std::atomic` — there is no object to wrap, only a plain `uint64_t*` aliasing
memory the client owns and the GPU also writes. `std::atomic_ref` is the clean answer but is
C++20, whereas ABCE targets C++17. That leaves MSVC `_Interlocked*` intrinsics plus explicit
fences for loads and stores. Is MSVC a target at all, or is Clang-only acceptable for Windows,
and is C++20 on the table?
