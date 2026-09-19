# ABCE — Accelerated Blit Copy Engine

**Status: draft, circulated for comment.** Nothing in the tree consumes ABCE yet; it is a
standalone header-only library at `runtimes/api-headers/include/abce`. The interface is C11 — strict C,
`extern "C"` guarded, so it is includable from C++ as-is — with `abce_cxx.h` as an optional
C++17 convenience layer over it. The interface is still cheap to
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
  to remove. `abce_copy_orchestrator_map_copy()` is designed to be exactly that missing call —
  it returns fully formed, relocatable packets without touching a ring, leaving
  `abce_copy_orchestrator_submit()` as reserve + `memcpy` + doorbell. Verified: the bytes submit
  places in the ring are identical to the bytes the map phase
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
  gfx version in one place (`abce_detect_packet_caps()`), so a new part is a capability entry plus whatever
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
  need `<stdint.h>`/`<stdbool.h>`-level headers. HSA/ROCr types appear in exactly one optional
  adapter header, and the C++ conveniences in exactly one optional C++ header.
- **Host and device from one source.** The packet builders are plain functions over a
  caller-supplied buffer, and the ring reserve/commit protocol is free functions over primitive
  pointers whose atomics and spin backoff resolve to `__hip_atomic_*` / `s_sleep` under device
  compilation, so a future device-initiated path shares the implementation rather than
  reimplementing the packet layouts. (The C conversion dropped the explicit `__host__ __device__`
  annotations the C++ version carried — see [Known gaps](#12-known-gaps).)
- **Sizing and emission can never disagree.** Every packet-producing routine is paired with a
  byte-count routine derived from the same code path, because a mismatch silently corrupts a
  ring.

## 2. Layering

```
                        client runtime (CLR / HRX-systems / ROCr / test / benchmark)
                                          │
  abce_cxx.h        optional C++ layer: default-initialized value types, RAII-ish wrappers
  abce_hsa.h        optional adapter: hsa_queue_t -> abce_ring_t, amd_signal_t -> abce_signal_ref_t
                                          │
  abce_host.h       abce_copy_orchestrator_t: validate -> select engines -> decompose -> plan -> submit
                    abce_sdma_engine_policy_t: topology/heatmap-driven engine ranking
                                          │
  abce_frame.h      abce_frame_composer_t: prologue / bodies / epilogue, size+emit pairs
                                          │
  abce_builder.h    abce_builder_t: one function per SDMA packet, ISA-aware layouts
  abce_ring_host.h  abce_ring_t: reserve -> zero -> write -> commit -> doorbell
  abce_ring_core.h  the lock-free reserve/commit protocol (host + device)
                                          │
  sdma_packets.h    hardware packet structs and field encodings
  abce_types.h      ISA/capability detection            abce_topology.h  KFD topology reader
  abce_config.h     abce_status_t, static-assert / bit helpers, compiler configuration
  abce_decode.h     packet walker (verification/debug)
```

A client normally touches only `abce_host.h` (plus `abce_hsa.h` when submitting to
ROCr-created queues, and `abce_cxx.h` if it is C++ and wants the conveniences). The lower layers
are usable on their own — the builder alone is enough
to hand-write a packet stream, which is how the verification programs for this draft work.
`abce_config.h` is included by everything: it holds the single `abce_status_t`
([§3.4](#34-output-types-and-error-surface)), the static-assert / alignment macros that spell
themselves differently in C11 and C++, and the min/max and bit-count helpers that replaced the
C++ templates.

## 3. The host interface

### 3.1 Two phases

`abce_copy_orchestrator_t` deliberately splits work into two calls:

| Phase | Call | Touches | Can block |
|---|---|---|---|
| Map | `abce_copy_orchestrator_map_copy(orch, ops, n, metadata, &plan)` | nothing external; builds packets into scratch | no |
| Submit | `abce_copy_orchestrator_submit(orch, &plan)` | rings, doorbell, completion signal | yes (ring space) |

Both return `abce_status_t`, and the plan is an out-parameter in caller-provided storage, so
neither phase hands back an allocation the caller has to free.

The split exists so that all the expensive, failure-prone work — validation, engine selection,
decomposition, packet emission — happens before any ring is touched. Once submit starts
reserving ring space it must run to completion, because an abandoned reservation would leave a
permanent unpublished hole in the ring. Keeping the fallible part in the map phase means submit
has almost nothing left to fail on.

The split is also what lets a client inspect or override the mapping decision. An
`abce_plan_t` exposes the engine chosen for each frame plus the ranked legal alternatives, so a
scheduler that knows something ABCE does not (queue depth, an in-flight copy on that engine) can
rewrite `frame.engine` before submitting.

Most importantly, it is what makes copy work pre-capturable: the map phase is the "compile" step a
HIP graph can run at instantiate time, and submit is the cheap replay step
(see [§1.1](#11-motivation)).

`abce_copy_orchestrator_dispatch()` is a convenience wrapper for callers that do not want the
seam.

### 3.2 Minimal use

```c
#include "abce_host.h"

// One builder per device; cheap, holds only ISA-derived capability flags.
const abce_isa_version_t isa = {9, 4, 2};
abce_builder_t builder;
abce_builder_initialize(isa, /*config=*/NULL, &builder);

abce_copy_orchestrator_t orchestrator;
abce_copy_orchestrator_initialize(&builder, &orchestrator);

// Register each SDMA ring once. hw_engine_id is the hardware SDMA engine index,
// which is what the policy's heatmap and affinity tables are keyed on.
abce_engine_affinity_t affinity;
abce_engine_affinity_initialize(&affinity);
affinity.hw_engine_id = 2;
abce_copy_orchestrator_register_engine(&orchestrator, /*index=*/0, &ring, &affinity);

// Describe the transfer. Endpoints are how the policy tells H2D from D2H from
// P2P without inspecting pointers.
abce_copy_op_t op;
abce_copy_op_initialize(&op);
op.kind = ABCE_OP_KIND_LINEAR;
op.src = host_ptr;
op.dst = device_ptr;
op.size = 1 << 20;
op.src_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_HOST, ABCE_ANY_DEVICE);
op.dst_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_DEVICE, /*device_id=*/0);

abce_copy_metadata_t metadata;
abce_copy_metadata_initialize(&metadata);
metadata.out = abce_signal_ref_make(signal_value_ptr, /*completion_value=*/0,
                                    /*event_mailbox=*/NULL, /*event_id=*/0,
                                    /*coordination_scratch=*/coordination_word);

abce_plan_t plan;
abce_status_t status =
    abce_copy_orchestrator_map_copy(&orchestrator, &op, 1, &metadata, &plan);
if (!abce_status_is_ok(status)) return Translate(status, plan.failed_op);
if (!abce_status_is_ok(abce_copy_orchestrator_submit(&orchestrator, &plan))) return kRetry;
```

**Use the initializers, not `= {0}`.** Several input structs have non-zero defaults, so zeroing
them is silently wrong rather than merely conservative. `abce_copy_op_initialize()` leaves both
endpoint device ids at `ABCE_ANY_DEVICE`, where a zeroed struct would claim device 0 — a
classification the
policy acts on. `abce_copy_metadata_initialize()` turns both coherency triggers and
`prefer_fused` on, so a zeroed metadata quietly drops the HDP flush and the fused packets.
`abce_engine_affinity_initialize()` sets `hw_engine_id = ABCE_ANY_ENGINE` ("use the registered
index"), `abce_builder_config_initialize()` sets `use_copy_size_override`, and
`abce_topology_data_initialize()` sets every unknown xGMI physical id to -1 rather than 0. Each
struct's initializer says so in a comment at its declaration. The map phase runs
`abce_plan_initialize()` on its out-parameter itself, so only the inputs are the caller's
problem; the C++ layer exists partly to make this automatic
([§3.5](#35-optional-c-layer)).

For ROCr-created SDMA queues, `abce_hsa.h` removes the ring plumbing:

```c
#include "abce_hsa.h"

abce_hsa_queue_ring_t queue_ring;
// The HSA entry points are a table rather than direct calls, so a client with a
// dynamically loaded ROCr can inject its own instead of linking the runtime.
const abce_hsa_queue_api_t hsa_api = abce_hsa_queue_api_direct();
// Validates that the queue really is an SDMA queue and reads its hardware engine id.
abce_hsa_queue_ring_initialize(&queue_ring, sdma_queue, &hsa_api, /*options=*/NULL);
abce_hsa_register_engine(&orchestrator, /*orchestrator_index=*/0, &queue_ring);

metadata.out = abce_hsa_signal_ref(amd_signal, /*completion_value=*/0);
```

### 3.3 Input types

**`abce_copy_op_t`** (88 bytes) — one logical copy. A flat struct with a per-kind tail; `kind`
selects which fields are live.

| Field | Used by | Meaning |
|---|---|---|
| `kind` | all | which primitive (see [§4.1](#41-operation-kinds)) |
| `src`, `dst` | linear, fill, swap, indirect | endpoints; for swap these are the two exchanged regions |
| `dsts`, `num_dsts` | multicast, broadcast | destination array, borrowed (caller keeps it alive) |
| `size` | all but rect | bytes; for fill a byte count that must be a multiple of 4 |
| `size2` | swap | endpoint B's size when asymmetric; honored only on the fused path |
| `rect` | rect | `abce_copy_rect_desc_t` geometry, borrowed |
| `indirect_src`, `indirect_dst` | indirect | which side is an address list |
| `src_end`, `dst_end` | all | residence: host or device, plus device id |

**`abce_copy_metadata_t`** (120 bytes) — parameters for the batch as a whole, notably one
completion signal for the entire batch rather than per copy.

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

One enum, `abce_status_t` in `abce_config.h`, covers every fallible entry point: the map phase,
submit, the ring and the rect builder. It replaced three separate enums (`MapStatus`,
`SubmitStatus`, `RingStatus`) plus the one place the C++ headers threw, so a caller has exactly
one thing to check — `abce_status_is_ok(status)`, with `abce_status_name(status)` for a log line.

`abce_plan_t` is a plain value in caller-provided storage rather than a move-only handle, and it
carries its own outcome: `valid` is set by a successful map, `submitted` by a successful submit
(so a second submit returns `ABCE_STATUS_INVALID_PLAN`), and `failed_op` names the offending
operation index, or `UINT32_MAX` when the failure was not attributable to one op. The status is
returned rather than stored on the plan.

Which statuses come out of which phase:

| `abce_status_t` | Phase | Cause |
|---|---|---|
| `ABCE_STATUS_INVALID_ARGUMENT` | map | null/zero field, host→host copy, missing output signal, bad rect geometry |
| `ABCE_STATUS_RECT_OUT_OF_RANGE` | map | rect pitch/slice exceeds what the packet can encode |
| `ABCE_STATUS_NO_ENGINE` | map | no registered engine survives `engine_mask` |
| `ABCE_STATUS_NO_LEGAL_ENGINE` | map | policy rejected every candidate for this transfer |
| `ABCE_STATUS_UNIMPLEMENTED` | map | op needs hardware this ISA lacks (e.g. an indirect copy below gfx125+) |
| `ABCE_STATUS_TOO_MANY_OPERATIONS` | map | batch exceeds 64 Ki entries, or a frame/signal count overflows |
| `ABCE_STATUS_RESOURCE_EXHAUSTED` | map | scratch allocation failed |
| `ABCE_STATUS_MISSING_COORDINATION_SCRATCH` | map | the batch fans out and needs a coordination word, but `abce_signal_ref_t::coordination_scratch` is null |
| `ABCE_STATUS_INVALID_PLAN` | submit | plan did not map, or was already submitted |
| `ABCE_STATUS_OUT_OF_RANGE` | submit / ring | a frame is at or above the ring size, so no reservation can ever fit it |
| `ABCE_STATUS_RING_UNAVAILABLE` | submit | a frame's engine is not registered, or its reservation failed |

The submit half of that table is deliberately much narrower than the map half, reflecting that
submission is nearly infallible by construction; merging the enums did not blur that, since the
two phases never shared a failure mode. The tail values are ABCE-specific cases worth telling
apart in a log, which is the only reason they are not folded into
`ABCE_STATUS_INVALID_ARGUMENT`.

`ABCE_STATUS_RECT_OUT_OF_RANGE` is where the rect builder's `std::invalid_argument` went. It is
also now raised earlier: an unencodable pitch or slice used to be discovered as an exception
thrown out of frame sizing and caught in the map call, whereas request validation now asks the
builder for the packet count up front, so the error reports against the operation that caused it
in `failed_op` like every other per-op failure.

### 3.5 Optional C++ layer

`abce_cxx.h` is the only C++ header and nothing requires it: the C headers are `extern "C"`
guarded precisely so a C++ client can include them as-is. It adds nothing to the ABI — every
value type derives from its C struct purely to run the matching initializer, so it is
layout-identical and converts to its C counterpart by plain reference.

`abce::CopyOp`, `abce::CopyMetadata`, `abce::Plan`, `abce::BuilderConfig` and
`abce::EngineAffinity` are the default-initialized wrappers, which is the `= {0}` hazard above
turned into a constructor. `abce::Builder` and `abce::CopyOrchestrator` wrap the initialize/use
lifecycle; the orchestrator is non-copyable and non-movable because registered rings and the
built-in policy are reached through pointers into it, so it has to keep its address.
`abce::IsOk()` and `abce::FormatStatus()` wrap the two status helpers.

```cpp
#include "abce_cxx.h"
#include "abce_hsa.h"

abce::Builder builder(abce_isa_version_t{9, 4, 2});
abce::CopyOrchestrator orchestrator(builder);
orchestrator.RegisterEngine(/*index=*/0, &ring);  // affinity optional: index as hw id

abce::CopyOp op;                    // LINEAR, both endpoints ABCE_ANY_DEVICE
op.src = host_ptr;
op.dst = device_ptr;
op.size = 1 << 20;
op.dst_end = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_DEVICE, /*device_id=*/0);

abce::CopyMetadata metadata;        // coherency triggers and prefer_fused already on
metadata.out = abce_hsa_signal_ref(amd_signal, /*completion_value=*/0);

abce::Plan plan;
const abce_status_t status = orchestrator.MapCopy(&op, 1, metadata, &plan);
if (!abce::IsOk(status)) return Translate(status, plan.failed_op);
if (!abce::IsOk(orchestrator.Submit(plan))) return kRetry;
```

## 4. Capabilities

### 4.1 Operation kinds

| Kind (`abce_op_kind_t`) | Description | Hardware |
|---|---|---|
| `ABCE_OP_KIND_LINEAR` | contiguous copy, auto-chunked to the per-packet limit | all |
| `ABCE_OP_KIND_MULTICAST` | one source to up to 1024 destinations in a single packet | gfx125+ |
| `ABCE_OP_KIND_BROADCAST` | one source to exactly 2 destinations | pre-gfx125+ (also chosen internally) |
| `ABCE_OP_KIND_SWAP` | bidirectional exchange of two buffers, sizes may differ | all; fused on gfx125+ |
| `ABCE_OP_KIND_INDIRECT` | gather/scatter where src and/or dst is an address list | gfx125+ only |
| `ABCE_OP_KIND_COPY_RECT` | 2D/3D strided sub-window copy | all; never fused |
| `ABCE_OP_KIND_FILL` | constant 32-bit pattern fill | all |

A batch may mix kinds freely; each is decomposed independently and placed on an engine.
Multicast degrades gracefully: on hardware without the multicast packet, ABCE emits
2-destination broadcast packets for small transfers and per-destination linear copies for large
ones, so a client can request `ABCE_OP_KIND_MULTICAST` unconditionally.

### 4.2 ISA support

Capabilities are derived from the gfx version rather than configured, using an OSS4/OSS5/OSS7
capability table. Verified output of `abce_detect_packet_caps()` /
`abce_detect_default_platform_caps()`:

| ISA | GCR | scope fields | fused wait/signal | max linear copy |
|---|---|---|---|---|
| gfx90a (9.0.10) | no | no | no | `0x3fffffff` |
| gfx942 (9.4.2) | no | no | no | `0x3fffffff` |
| gfx10.3 | yes | no | no | `0x3fffffff` |
| gfx11 | yes | no | no | `0x3fffffff` |
| gfx1250 (12.5.0) | no | yes | yes | `0x3fffffff` |

`abce_platform_caps_t` carries the decisions the ISA *cannot* imply — device atomic support, whether to
emit an HDP flush, whether the driver already owns GCR — because the same IP ships in
configurations that differ on all three. Clients are expected to override the defaults from
link topology; an xGMI host link, for instance, does not want the HDP flush.

### 4.3 Packet-level features

Beyond the copies themselves the composer can emit: dependency polls (32- or 64-bit), HDP
flush, GCR invalidate/writeback, SDMA global-clock timestamps bracketing the copy, the
completion update (atomic decrement, or fence where atomics are unavailable), and an
interrupt fence+trap pair for signals with an event mailbox.

## 5. Engine selection

Engine choice is a policy, not hardcoded. `abce_copy_orchestrator_initialize()` selects one for
the part through `abce_sdma_policy_initialize_for_isa()` and holds the
`abce_sdma_engine_policy_t` by value, since it is plain data plus a vtable;
`abce_copy_orchestrator_set_engine_policy()` replaces it with a client callback and a `user_data`
context; with no policy the orchestrator falls back to round-robin over registered engines.

The base policy classifies each transfer from its endpoints into one of four bands — H2D, D2H,
local D2D (same device), P2P (cross-device) — and asks the matching entry of its
`abce_sdma_engine_policy_vtable_t` (`rank_h2d`, `rank_d2h`, `rank_p2p`, `rank_local_d2d`) for a
ranked list of hardware engine ids, which are then resolved to registered indices, deduplicated
and filtered against the batch's candidate mask.

**A client never declares what an engine is for.** Registration supplies only the hardware engine
id. Band membership comes from measurement (the H2D/D2H heatmaps) or from the driver (KFD's
`num_sdma_engines` / `num_sdma_xgmi_engines` split, which
`abce_sdma_policy_load_topology_from_kfd()` feeds to `abce_sdma_policy_set_engine_split()`);
absent both, every band widens to all registered engines and selection
degrades to load balancing. An earlier design had clients tag each engine with an engine class
of host/xGMI/local; measurement showed that to be actively wrong, because a physical class does
not predict which engines are best for a band.

Note which direction the xGMI split is used in: it *prefers* engines for P2P and never excludes
them from host copies. On MI300X, KFD classes engines 2–15 as xGMI, yet four of them outrank both
non-xGMI engines for H2D.

The distinction between a ranking and a reservation matters, because only a reservation should be
able to make a copy unmappable:

- A **ranking** is a preference, so the band function appends every remaining engine behind it. The
  heatmap is a ranking: a client that registers engines the profile does not mention still gets a
  legal engine, in measured order where the profile has an opinion.
- A **reservation** is a hard exclusion, expressed by returning a short list. gfx90a's RAS rule
  is one — SDMA0 is listed for H2D and appears nowhere else, so D2H fails rather than using it —
  as is the cross-hive check, which returns nothing at all.

Two arch policies exist today. They are vtables rather than subclasses of a base policy, selected
by `abce_sdma_policy_initialize_for_isa()`; a band a variant does not specialize points straight
at the base implementation (`abce_sdma_policy_base_vtable`), which is what inheriting the base
method used to mean.

- **`abce_sdma_policy_gfx94x_vtable`** (gfx94x/95x) — H2D/D2H order comes from bandwidth heatmaps recorded in
  the source, because per-engine SPX bandwidth is markedly non-uniform (on gfx942/16-SDMA some
  engines reach ~20 GB/s D2H against ~55 GB/s for the best). P2P consults an xGMI
  `physical_id`-keyed affinity table for the optimal first choice, then appends the rest of the
  xGMI band, and requires both GPUs to be in the same hive.
- **`abce_sdma_policy_gfx90a_vtable`** — encodes the RAS restriction that SDMA0 may only drive
  H2D, by listing SDMA0 for H2D and SDMA1 for D2H and never listing SDMA0 elsewhere.

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
  (≥ 1 GiB) instead group `ABCE_MAX_COPIES_PER_ENGINE` copies per ring.

## 6. Fan-out coordination and completion

The hard part of fan-out is that a batch spread over N rings must still produce exactly one
completion signal transition, with no ordering violations. ABCE runs that protocol in a 64-bit
*coordination word* it owns outright, kept out of the completion signal's value so the value only
ever holds legal signal states:

- **low 32 bits** — a fan-in counter. Each participating frame decrements after its last body;
  the coordinator polls for the count to drain to zero.
- **bit 62** — a start gate (`ABCE_FAN_OUT_START_GATE`). Non-coordinator frames poll for it to
  clear before their first body; the coordinator clears it after its prologue.

The word is `abce_signal_ref_t::coordination_scratch`, which `abce_hsa_signal_ref()` points at the
signal object's own trailing reserved words — a naturally aligned 64-bit slot in the same 64-byte
object, so coordination shares a cache line with the value it guards and costs no extra
allocation. The protocol drains the word back to zero, so a completed plan leaves no residue.
A fan-out that needs a coordination word and was not given one is rejected at map time with
`ABCE_STATUS_MISSING_COORDINATION_SCRATCH` rather than falling back to the signal value.

`frames[0]` is the coordinator and carries the prologue (dependency polls, timestamp start, HDP
flush, GCR invalidate) and the epilogue (fan-in poll, GCR writeback, timestamp end, completion
update, interrupt). Submit arms the coordination word only after every ring reservation has
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
Submit arms the signal to N above the completion value and the N body signals walk it down.
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

**Timing — `abce_copy_metadata_t::timestamps`.** Two 64-bit slots. The coordinator frame reads the
SDMA global clock into `timestamps.start_value` in its prologue and into `timestamps.end_value` in
its epilogue,
immediately before the completion update, so the interval brackets the whole batch rather than one
frame. Both slots are written or neither is; requesting them forces a prologue and epilogue onto
the coordinator even where the batch would otherwise have gone without (which costs the direct
fused completion path described in [§6](#6-fan-out-coordination-and-completion)). For a fan-out
the interval covers the entire batch under one signal, not per-engine spans — ABCE deliberately
reports one interval per completion signal, because that is the unit a client waited on.

**Attribution — `abce_copy_metadata_t::execution_descriptor`.** One 32-bit slot, stamped with the
word `abce_execution_descriptor_encode()` builds and
`abce_execution_descriptor_engine_kind()` / `_instance_mask()` / `_transfer_kind()` /
`_mixed_kinds()` read back:

| bits | field |
|---|---|
| 0–15 | instance mask, interpreted per engine kind — for SDMA, bit *h* means hw engine *h* ran part of the batch |
| 16–18 | `abce_transfer_kind_t`: unknown / H2D / D2H / P2P / local D2D |
| 19 | batch mixed transfer kinds, so the kind field reads `ABCE_TRANSFER_KIND_UNKNOWN` |
| 20–22 | `abce_execution_engine_kind_t`: `NONE` / `SDMA` / `COMPUTE` / `CPU`, prefixed `ABCE_EXECUTION_ENGINE_KIND_` |
| 23–31 | reserved, zero |

Five choices in that layout are worth stating, since they are the contract:

*The engine kind is explicit, not assumed.* A completion signal is not the property of any one
engine. The same signal object an SDMA batch completes can instead be completed by a compute blit
kernel or a host memcpy, and nothing in the signal value says which. ABCE only ever writes
`ABCE_EXECUTION_ENGINE_KIND_SDMA`, because it builds nothing else; the other values exist so a
client whose fallback paths
complete the *same* signal can stamp the same slot. That gives a consumer one field to read
instead of a heuristic — ROCr today infers "SDMA or blit kernel?" by zeroing its SDMA timestamps
before a copy and testing them afterwards (`SharedSignal::CopyPrep` / `GetRawTs`). The two paths
do not even record timestamps in the same place, so the engine kind additionally tells a reader
*which timestamp pair to look at*. Three bits rather than two: the spares cost nothing, since the
reserved tail absorbs them, and a field written by more than one component should have room for
executors that have not come up yet.

*Hardware engine ids, not ABCE's registered indices.* A registered index is a client-local
numbering ABCE assigns at engine registration; it means nothing to a tool inspecting the machine.
The mask uses the `abce_engine_affinity_t::hw_engine_id` the client registered, which is the id the rest
of the stack and the hardware agree on. The mask is only meaningful relative to the engine kind,
which is why the kind is in the contract: a compute kernel has no equivalent of an SDMA engine id
and may leave the mask zero.

*A mask, not a single id.* A fan-out has no single engine, and a field that could only name one
would have to lie or report nothing for exactly the case a profiler most wants to see. The mask
makes a 4-engine batch self-describing, and `ABCE_MAX_ENGINES` is 16, so 16 bits is exact rather
than a truncation (a static assertion in `abce_host.h` keeps the two in step).

*Written by submit, not the map phase.* A client is allowed to retarget
`abce_plan_frame_t::engine` between
the two phases, so only submit knows which engines actually ran. It is a single relaxed-cost
host store issued after every ring reservation succeeds, ordered ahead of the copy's completion by
the ring publish that follows; no packets, no device work, and nothing added to the ring.

*Validity is the engine kind, not a magic number.* `ABCE_EXECUTION_ENGINE_KIND_NONE` is the zero
value, so an untouched
field reads as "nobody stamped this" and `abce_execution_descriptor_valid()` is just that test.
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
H2D and D2H copies under one signal; `abce_plan_t::transfer_kinds` is a bitmask of every kind
present, `abce_plan_uniform_transfer_kind()` returns a single kind only when the batch is
homogeneous, and bit 19 tells
the reader the field is genuinely unattributable instead of silently labeling the batch by its
first copy. Classification comes from `abce_classify_transfer()`, which derives the kind from endpoint
kinds and device ids only — it never dereferences a client pointer or queries the driver, so it
costs nothing and works for pointers ABCE cannot inspect.

**Placement is the client's decision.** Both slots are addresses, so ABCE commits to no signal
layout. `abce_hsa_execution_descriptor_slot()` offers the obvious placement — `amd_signal_t::reserved1`,
the spare 32-bit reserved word, which shares the signal's cache line and needs no allocation — but
ABCE never writes it unless a client assigns it into the metadata, because which bytes of a signal
may be repurposed is an ABI decision that is not ABCE's to make. `amd_signal_t` is a frozen format
that may be shared across processes; a client that cannot spend `reserved1` should point the
descriptor at its own wrapper struct. A client that would rather keep this metadata in its own
records than in a slot ABCE writes can leave `execution_descriptor` null and call
`abce_copy_orchestrator_describe_execution()` for the same word. There is precedent for exactly
that: ROCr keeps its SDMA
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
index moving out of reservation order. `abce_ring_t` is therefore single-consumer,
multi-producer, with the protocol in `abce_ring_core.h` shared verbatim with the (not yet
landed) device ring. The core takes its read-index and publish steps as function pointers bundled
with one context pointer (`abce_ring_producer_t` for the reserve side, `abce_ring_publisher_t` for
the commit side) where the C++ version took them as template parameters instantiated on lambdas:

1. **`abce_ring_acquire()`** — advance a monotonic reserve cursor by one CAS, so producers write
   disjoint regions with no lock. The region is zeroed, which both satisfies the builders'
   zeroed-buffer contract and turns any unused pad tail into NOPs.
2. **Write** — the caller fills the region.
3. **`abce_ring_release()`** — spin until the commit cursor reaches this reservation's start, then
   advance the hardware write pointer and ring the doorbell, then release the next producer. The
   doorbell is thus monotone even when producers finish writing out of order.

A payload that would straddle the physical ring end is handled by reserving and publishing the
wrap tail as a *separate* NOP region, then retrying the payload at offset zero. Keeping the
padding separate means any payload smaller than the ring can still make progress once the ring
drains, even when padding plus payload exceeds the ring size. Verified on a 256-byte ring: a
200-byte reservation at offset 0, then a 100-byte reservation lands at monotonic 256 (offset 0)
after a 56-byte NOP pad, leaving the doorbell at 356.

Attaching to an already-in-use queue seeds the cursors from the queue's current write pointer,
so the first acquire cannot hand out bytes the engine still owns.

Multi-ring submission acquires and publishes rings in a fixed global engine order rather than
frame order. Without that, two concurrent plans with opposite frame orders could each hold one
unpublished reservation while waiting for the other's ring — a cross-ring commit deadlock.

## 9. What a client must supply

| Requirement | How |
|---|---|
| Ring memory + control words | `abce_ring_config_t` with either mapped `write_ptr`/`read_ptr`/`doorbell`, or an `abce_ring_queue_ops_t` callback trio (plus its `user_data`) for APIs that keep them opaque |
| Ring size a power of two | asserted in `abce_ring_initialize()` |
| Completion signal | a device-visible 64-bit value location via `abce_signal_ref_t` |
| Coordination scratch | a device-visible 64-bit word via `abce_signal_ref_t::coordination_scratch`, for batches that fan out; `abce_hsa_signal_ref()` supplies it from the signal's reserved words |
| Memory residence | `src_end`/`dst_end` per op; ABCE never inspects pointers |
| Platform capabilities | `abce_platform_caps_t` (atomics, HDP flush, GCR ownership) |
| Device ids | ABCE-local indices, assigned by the client at registration |
| Engine hardware ids | `abce_engine_affinity_t::hw_engine_id` per registered ring — the only per-engine metadata ABCE asks for, and what the heatmap and xGMI split are keyed on |
| Topology (optional) | `abce_sdma_policy_load_topology_from_kfd()` on Linux; elsewhere, or when KFD reads are undesirable, `abce_sdma_policy_set_topology()` with a caller-filled `abce_topology_data_t`, and/or `abce_copy_orchestrator_init_device_profile(total_sdma, num_non_xgmi_sdma)` |
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
against the include directory and walking the emitted frames with `abce_decode.h`. The C
conversion was held to the same bar: the emitted SDMA bytes were compared against the C++
implementation's and are identical across the single, batched, fan-out, back-to-back, dependency,
timestamp, fill and multicast cases, so the frames and byte counts below still describe exactly
what the library emits. Representative output:

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
counted per destination, `ABCE_LINEAR_BATCH_MODE_FORCE_BACK_TO_BACK` /
`ABCE_LINEAR_BATCH_MODE_FORCE_FAN_OUT` still override the size rule, and a
lone copy of any size is never treated as a back-to-back batch.

The profiling contract was checked the same way. `abce_classify_transfer()` returns the expected
kind for
all seven endpoint combinations (including an unknown device id, which classifies as P2P, and
host-to-host, which is rejected upstream); `abce_plan_t::transfer_kinds` accumulates correctly and
`abce_plan_uniform_transfer_kind()` reports mixed batches as `ABCE_TRANSFER_KIND_UNKNOWN` with bit
19 set. Registering engines
under deliberately non-identity hardware ids (4, 5, 9) confirms the descriptor's instance mask
carries hardware ids rather than registered indices — a single-engine H2D on registered engine 0
stamps `0x0010`, and a 4 × 1 MiB fan-out stamps `0x0230` with exactly one mask bit per
participating frame. Every value the 3-bit engine-kind field can hold survives encoding, ABCE
always stamps `ABCE_EXECUTION_ENGINE_KIND_SDMA`, a zero word decodes as
`ABCE_EXECUTION_ENGINE_KIND_NONE` and reads invalid, and a mask-less
`ABCE_EXECUTION_ENGINE_KIND_COMPUTE` stamp — the case that motivated moving validity off the mask
— still reads valid. The
slot is untouched until submit, the reserved bits stay zero at field saturation, and arming the
descriptor leaves the `reserved3` coordination word intact.

## 12. Known gaps

- **No tests in tree.** There is no CMake target either: `runtimes/api-headers` exports none yet,
  so a consumer adds this directory to its include path and builds as C11, or C++17 for
  `abce_cxx.h`. Everything verified for this draft was throwaway. Given the library is pure logic over plain memory, a
  GPU-free unit suite (emit a frame, walk it with `abce_decode.h`, assert the packet sequence
  and byte count) is both cheap and the main argument for the design; it should land with or
  before any integration.
- **`abce_device.h` does not exist, and the device annotations are gone.** The reserve/commit core
  is still written to be shared with a device producer — its atomics and backoff select
  `__hip_atomic_*` / `s_sleep` under device compilation — but the C conversion dropped the
  `ABCE_HD` (`__host__ __device__`) annotations the C++ headers put on the builders and the ring
  core, and the comments no longer name a device header. So the sharing is now a property of how
  the code is written rather than something the headers declare; a device-side producer landing
  later has to put the annotations back.
- **Windows is further away than it was.** Three things in the public header chain are not MSVC
  constructs. The KFD sysfs reader is at least gated behind `ABCE_HAS_KFD_TOPOLOGY` (Linux-only,
  since it walks `/sys/class/kfd` with `opendir`) and bit counting falls back to a portable loop
  off GCC/Clang, but: `abce_host.h` includes `<pthread.h>` unconditionally for the per-thread
  packet scratch's key and destructor; `abce_ring_core.h` includes `<sched.h>` for the host
  spin-wait's `sched_yield()`, where the C++ version used `std::this_thread::yield()`; and there
  are thirteen uses of the GCC/Clang `__atomic_*` builtins — eight in `abce_ring_core.h`, four in
  `abce_host.h` (the cached environment knobs, arming the coordination word, stamping the
  execution descriptor) and one in `abce_ring_host.h` (the release fence before the doorbell) —
  which Clang accepts on Windows but MSVC does not. Closing that needs a decision, not a
  mechanical fix — see [Q4](#14-questions-for-reviewers).
- **No integration.** Nothing outside `runtimes/api-headers/include/abce` refers to ABCE, so none of this has
  run against real hardware through the orchestrator; the heatmap numbers come from measurements
  taken through ROCr's selector, not through this code.
- **Mixed error strategy — closed by the C conversion.** The rect path used to throw
  `std::invalid_argument` on out-of-range pitch/slice, which the map call caught, while the rest of
  the API returned status codes; throwing from headers that are also compiled for device was
  awkward. There is now one `abce_status_t` and the rect range check is part of request validation
  ([§3.4](#34-output-types-and-error-surface)).

## 13. Assumptions

These are stated as decisions rather than open questions. They are load-bearing, so a reviewer
who disagrees should push back — but the intent is to build to them.

**A1 — a fill is host-source to device-destination, and the caller states it.** A fill has no real
source, but it is still classified as a host→device operation, and
`dst_end.kind = ABCE_ENDPOINT_KIND_DEVICE` is a
required field rather than something ABCE infers. This is worth calling out because
`abce_copy_op_initialize()` leaves both endpoints on the host side, host→host is rejected, and a
caller who populates only
the fill descriptor therefore gets `ABCE_STATUS_INVALID_ARGUMENT` with nothing obviously wrong in the
request. The contract is being specified, not relaxed: endpoint classification stays mandatory
for every operation kind, so no path can reach engine selection without the client having said
where the memory lives.

**A2 — fan-out coordination lives in a reserved field of the signal, not bit 62 of its value.**
*(Implemented.)* The start gate and fan-in counter used to be packed into the completion signal's
own 64-bit value, so between arm and completion the signal transiently held something that was not
a legal HSA signal state. They now live in `abce_signal_ref_t::coordination_scratch`, which
`abce_hsa_signal_ref()` points at `amd_signal_t::reserved3` — a 64-bit slot at offset
56 of a struct that is exactly one 64-byte cache line, its size and 64-bit alignment checked by
`ABCE_STATIC_ASSERT` so a change to the frozen layout is a build failure rather than a silently
misaligned poll. Coordination therefore shares
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
`ABCE_STATUS_MISSING_COORDINATION_SCRATCH` rather than silently degrading. It is worth knowing that this is
size-dependent — a client that only ever issues small copies never supplies scratch and never
notices, until a copy is large enough to fan out.

**A3 — a graph stores each node's mapped packets at capture time, baked signal addresses
included.** This is the expected model rather than a problem to design around: capture is exactly
the point at which packets should be built and retained, and the signal and dependency addresses
resolved at that point are part of the captured node's state. A node whose addresses change is
re-captured, which is work a graph already does per node. So the fact that the map phase bakes
addresses into the packet bytes is a property to rely on, not one to abstract away. What remains
is purely a lifetime question — who owns the bytes the graph is holding — which is
[Q1](#14-questions-for-reviewers).

**A4 — the engine heatmap stays compiled in and is not runtime-overridable.** These orderings are
measured properties of a specific part, not deployment policy, and a runtime override would
invite tuning by environment variable and make reported performance unreproducible. New parts
keep meaning a new table plus a new policy vtable. The follow-up work is consolidation rather than
configurability: the two near-duplicate selection paths
(`abce_copy_orchestrator_select_engine()` and `abce_pick_balanced_engine()`), and the per-link
recommended engine mask, which is parsed out of KFD into `abce_topology_data_t` and readable
through `abce_topology_data_recommended_mask()`
but never consumed by ranking. Consuming it would retire the hardcoded xGMI affinity
table, which [§11](#11-verified-behavior) confirms it agrees with on every peer link.

## 14. Questions for reviewers

**Q1 — what must an `abce_plan_t` become to be held and replayed?** [A3](#13-assumptions) commits
to a
graph holding each node's packets across launches, and the hard part is already done: the packets
are relocatable and submit copies them verbatim. Two mechanical things stand in the way, and I
think they want one answer rather than two.

The bytes are borrowed. The map phase builds into a per-thread scratch buffer and
`abce_plan_frame_t::packets` is a non-owning view into it, so a plan is invalidated by the next map
on the same thread. This is documented, but I confirmed it is a use-after-free rather than merely
stale data: mapping a larger plan reallocates the scratch and frees the first plan's bytes.
Map-two-then-submit-two is a natural thing for a batching client to write, never mind holding a
plan until the next launch. (The C conversion changed only the buffer's release point: the scratch
now hangs off a `pthread` key whose destructor frees it at thread exit, where the C++ version had a
`thread_local` `unique_ptr`. The borrowing, and the realloc that invalidates an earlier plan, are
unchanged.)

The plan is also single-use: submit sets `submitted` and a second submit returns
`ABCE_STATUS_INVALID_PLAN` (verified). So how should a plan own its bytes — a heap allocation per
map, which
the scratch exists to avoid, or a per-orchestrator arena with a generation counter that submit
validates? And should replay be explicit (an `abce_plan_rearm()` / `abce_plan_reset()`) rather than
silently allowing repeated submits, so that accidental double-submission stays an error?

**Q2 — engine capacity.** `ABCE_MAX_ENGINES` is 16, matching MI300X, but masks are `uint64_t` and
`abce_plan_t` is a fixed 1464-byte struct built around an array of 16 frames. Is 16 the right
ceiling to bake in, and should
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

**Q4 — how far should Windows support go?** ABCE compiled for Windows under Clang before the C
conversion; the header chain now also pulls in `<pthread.h>` and `<sched.h>`
([§12](#12-known-gaps)), so the question is broader than it was, and two of the three pieces are
mechanical. The `__atomic_*` builtins are the piece that needs a decision: the same ring functions
compile for device code, where `__hip_atomic_*` is used, so the host branch cannot simply wrap an
object — there is no object, only a plain `uint64_t*` aliasing memory the client owns and the GPU
also writes. That ruled out `std::atomic` and left `std::atomic_ref` as the clean C++ answer,
which was unavailable at C++17; now that the interface is C11 the equivalent is
`<stdatomic.h>` operations on a cast `_Atomic uint64_t*`, which has the same aliasing question and
adds MSVC's C11 atomics support to the list of unknowns. The fallback is still MSVC
`_Interlocked*` intrinsics plus explicit fences for loads and stores. Is MSVC a target at all, or
is Clang-only acceptable for Windows — and if it is a target, who owns replacing the pthread key
and `sched_yield()`?
