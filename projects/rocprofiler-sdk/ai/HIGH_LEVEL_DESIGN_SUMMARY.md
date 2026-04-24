# Firmware-Assisted Kernel Dispatch Tracing — Design Summary

> Detail docs: `KNOWN_ISSUES.md`, `FIRMWARE_RING_HYBRID_DESIGN.md`,
> `KFD_DISPATCH_LOG_DESIGN.md` (in this directory).

## What this is

A re-architecture of how `rocprofiler-sdk` collects per-kernel start
and end timestamps. The new path:

* **NO HSA queue interception.** The SDK does not wrap the application's
  HSA queues, does not create intercept-queues, does not register
  packet-write callbacks.
* **NO AQL packet rewriting.** The SDK does not modify, expand, or
  insert any packets into the application's AQL queue. The application's
  packet flow is byte-identical to a no-profiler run.
* **NO completion-signal allocation by the SDK.** The MEC firmware
  writes timestamps directly into a host-visible ring buffer at kernel
  start and end-of-packet. There is no signal pool, no signal swap,
  no async-signal-handler thread on the SDK side.

Everything else flows from those three facts.

## Why this matters

| Today's standard path | The firmware-ring path |
|---|---|
| Per-dispatch CPU overhead from packet wrapping + signal allocation + barrier injection | Effectively zero per-dispatch CPU overhead on the launching thread |
| Late-attach is impossible — must intercept queue at create time | Late-attach works — drainer iterates existing queues at attach |
| Coexistence with applications that own their own AQL queue is fragile | Application's queue is unmodified — no coexistence question |
| Cannot profile applications that bypass HIP (raw HSA tools, custom runtimes) | Works for any application whose dispatches go through an HSA queue |

## What exists today

**On the firmware-onboarding test machine (`gbt350-odcdh5-wbc1-b.png-odc.dcgpu`), end-to-end working:**

| Layer | What ships | Status |
|---|---|---|
| MEC firmware (`f32_mec.uc`) | `SubAqlProfBufWriteRecord` writes 16-byte records `{ts_lo, ts_hi, record_type, dispatch_idx}` to a host-visible ring on every dispatch start and EOP | gfx9_5_0 / MI350 only |
| KFD (`amdgpu`) | Custom branch with `dispatch_record_buffer_addr/size` plumbed through `UPDATE_QUEUE` ioctl into MQD DW43–47 | gfx9 only |
| ROCr / thunk | New HSA APIs `hsa_amd_queue_iterate`, `hsa_amd_profiling_get_dispatch_records`. Allocates and registers the ring. | Posted at `users/bewelton/cpc_tracing` |
| rocprofiler-sdk | Standalone polling drainer reads the ring at 1ms cadence, pairs START/END, emits `KERNEL_DISPATCH_COMPLETE` records | Posted at `users/bewelton/cpc_tracing` |

This baseline produces correct kernel timing on MI350 today.

**Performance impact on the non-profiling path is dependent on the
ratio of kernel launches to device syncs.** The cost lives in MEC
firmware's `AqlConnect` path (the routine that runs when the hardware
scheduler attaches a queue to a compute pipe), which now does two
extra TC reads of the MQD to pick up the dispatch ring buffer config.
That work happens **per queue-reconnect, not per dispatch**.
`hipDeviceSynchronize()` typically forces a queue drain → eviction →
re-connect, so syncing after every launch puts the cost on the per-
dispatch path. Workloads with batched dispatches between syncs
amortize the cost over many launches.

> **All numbers in this section are preliminary and subject to
> change.** They are point-in-time results from a small number of runs
> on a single machine; they have not been characterized across firmware
> revisions, ASIC variants, or extensive workload coverage. Treat as
> directional, not as a steady-state performance contract.

| Workload pattern                                | Preliminary overhead |
|-------------------------------------------------|----------------------|
| 1 launch + 1 sync, microbenchmark (worst case)  | +1.7 µs / dispatch (+10.9%) |
| graphbench (batched, real workload)             | &lt;0.4% (in the noise floor of the run) |
| Typical batched inference / training / Triton   | Expected near-zero by extension of the graphbench result; not yet measured per-workload |

The per-`AqlConnect` overhead is the only added cost on the
non-profiling path; the per-dispatch path itself is unchanged.

## What's broken / missing today

The standalone path has 10 known issues catalogued in
`KNOWN_ISSUES.md`. The five that affect correctness for users:

1. **No correlation IDs.** Kernel records cannot be linked to the HIP
   API call that launched them. HIP-API tracing and kernel-dispatch
   tracing produce two disjoint streams.
2. **No `KERNEL_DISPATCH_ENQUEUE` callbacks.** Tools that hook
   per-launch instrumentation see nothing.
3. **No launching-thread tid** on dispatch records — drainer's tid is
   reported instead.
4. **No workgroup/grid sizes** on dispatch records (zeroed).
5. **External correlation IDs** pushed by the tool before launch are
   not visible on the dispatch record.

Plus structural issues in the prototype that block upstream:

* KFD's record-buffer setup is bolted onto `UPDATE_QUEUE` instead of
  a profiler-scoped ioctl.
* SDK has a hardcoded `mec_dispatch_record_16` C struct duplicating
  what the firmware writes.
* Bunch of "save" commits in the branch history.
* Per-process descriptor / cleanup paths not robust.

## The plan, in three pieces

### Piece 1 — SDK: hybrid firmware-ring + launching-thread doorbell hook

Plan: `FIRMWARE_RING_HYBRID_DESIGN.md`.

Adds a thin wrapper on `hsa_signal_store_relaxed/screlease` (the
doorbell stores) that runs synchronously on the launching thread.
On every doorbell ring it captures the correlation ID and tid into a
per-queue side table, then chains through. The drainer thread looks up
the side table when it processes a firmware-ring END record.

**This recovers correlation IDs, ENQUEUE callbacks, launching tid,
external correlations — without ever invoking `WriteInterceptor` and
without allocating any HSA signals.** Strictly cheaper than the
existing `WriteInterceptor`-based interception.

**Critical dependency:** PR 5219 (`users/bewelton/no-interecept-queue`).
That PR already adds the doorbell-wrap machinery for a different
purpose (inline queue intercept that still calls `WriteInterceptor`).
Our hybrid sits on top of that PR's doorbell-map / wrapper
infrastructure — we **need PR 5219 merged before this hybrid can land**.
The `users/bewelton/cpc_tracing` branch is already rebased onto
PR 5219 to make this dependency explicit.

Phasing (in `FIRMWARE_RING_HYBRID_DESIGN.md` §12):

* Phase 1 (~200 lines): doorbell hook + side table + drainer lookup.
  Recovers correlation, tid, ancestor, external-corr.
* Phase 2 (~50 lines): fire ENQUEUE ENTER+EXIT callbacks.
* Phase 3 (~100 lines): workgroup/grid sizes from AQL packet, bounded
  dedup, profiler-enable cleanup.

### Piece 2 — KFD: clean ioctl + self-describing format

Plan: `KFD_DISPATCH_LOG_DESIGN.md`.

* Revert `UPDATE_QUEUE` to upstream-clean shape. Move ring buffer
  setup into a new sub-op of the existing `AMDKFD_IOC_PROFILER` ioctl.
* Ship the record-format descriptor as a JSON `static const char[]`
  in kernel `.rodata`. Userspace queries it via a new `GET_DESCRIPTOR`
  sub-op. **Any tool** (rocprofiler-sdk, crash dumpers, custom
  profilers) can ask the kernel for the format with no upload, no
  coordination with libhsakmt or ROCr.
* No firmware change required — wire format unchanged.
* Bumps `KFD_IOCTL_MINOR_VERSION` 22→23.

This piece has been adversarially reviewed (gpt-5.3-codex);
review-driven revisions in commit `fa0764e8ee`.

### Piece 3 — Cross-arch firmware port

Today's firmware support is **gfx9_5_0 / MI350 only**. Bringing this
feature to other ASIC families requires firmware changes:

* MEC microcode (`f32_mec.uc` equivalent) for each family must add
  the `SubAqlProfBufWriteRecord` path and the MQD-cached buffer config.
* MQD struct headers (`v10_structs.h`, `v11_structs.h`,
  `v12_structs.h`) must add the same DW43–47 fields that `v9_structs.h`
  already has.
* KFD MQD managers (`kfd_mqd_manager_v10/11/12.c`) must mirror v9's
  write block.

Each ASIC family is roughly the same scope of firmware work as the
original MI350 implementation. **This is the longest-pole item by
far** — the SDK and KFD changes are in the weeks-of-effort range; each
firmware port is months including build/sign/deploy/validate cycles.

## Expertise required to land this

| Area | Expertise needed | Why |
|---|---|---|
| **KFD / kernel maintainer** | Linux kernel UAPI conventions, ioctl design, KFD's queue-manager + MQD-manager internals, locking discipline | Reviewing and merging the `KFD_IOC_PROFILER_DISPATCH_LOG` migration. Validating `_IOC_SIZE` compatibility; signing off on the descriptor blob mechanism. |
| **HSA runtime / ROCr** | `hsa_amd_queue_get_info` attribute conventions, `AqlQueue` internals | Single small addition: a new `HSA_AMD_QUEUE_INFO_KFD_QUEUE_ID` attribute on `hsa_amd_queue_get_info` so rocprofiler-sdk can map an `hsa_queue_t*` to its underlying KFD queue id. ~5 lines. The earlier draft of this plan also had ROCr's `AqlQueue::SetProfiling` calling a new libhsakmt thunk on the SDK's behalf — that whole path was removed in favor of the SDK going direct to KFD. ROCr is no longer in the dispatch-ring SET path. |
| **rocprofiler-sdk** | Existing kernel-dispatch tracing pipeline, correlation-id system, late-attach, queue lifecycle | Implementing the hybrid drainer + doorbell hook (Phase 1 ~200 lines on top of PR 5219). |
| **MEC firmware (CRITICAL)** | F32 microcode for the `SubAqlProfBufWriteRecord` path, MQD layout per ASIC, the build/sign/deploy/test loop | **Required for any ASIC beyond MI350.** Each new family (gfx10/11/12) needs an equivalent of the MI350 firmware work. This is the longest pole and most specialized skill set. Without firmware engagement, this feature stays MI350-only indefinitely. |
| **CI / release engineering** | Cross-repo coordinated rolls (KFD ↔ ROCr ↔ rocprofiler-sdk ↔ firmware-per-ASIC) | The userspace migration is now smaller — only ROCr (one new attribute) and rocprofiler-sdk (direct ioctl) need to ship. libhsakmt is no longer in the path. Sequencing for the kernel UAPI bump is described in `KFD_DISPATCH_LOG_DESIGN.md` §5.1 (`_IOC_SIZE` matrix). |
| **Adversarial / security review** | UAPI hazards, capability model, container/namespace semantics | One round of cross-vendor adversarial review (gpt-5.3-codex) already complete; findings addressed. A second pass (Gemini, or in-house security) before upstream submission is recommended. |

## Status snapshot

| Item | Status |
|---|---|
| Firmware ring + drainer baseline (MI350) | working on `gbt350-odcdh5-wbc1-b` |
| Standalone branch `users/bewelton/cpc_tracing` | pushed, builds clean (verified on remote) |
| Branch rebased on PR 5219 (`users/bewelton/no-interecept-queue`) | done |
| SDK hybrid plan (`FIRMWARE_RING_HYBRID_DESIGN.md`) | written; review pending |
| KFD ioctl + descriptor plan (`KFD_DISPATCH_LOG_DESIGN.md`) | written; **adversarially reviewed and revised** (commit `fa0764e8ee`) |
| Phase 1 implementation | not yet started — gated on PR 5219 merge |
| KFD implementation | not yet started — gated on KFD-team review of the plan |
| Firmware port to non-MI350 | not yet started — **requires firmware-team engagement** |

## Medium-term, in parallel: replacing the HIP/HSA → rocprofiler-sdk callback path

Today, HIP and HSA call back into rocprofiler-sdk via function pointers
populated through `rocprofiler-register`. The runtimes know about
rocprofiler-sdk (via the shared API table contract), and rocprofiler-sdk
receives data **synchronously on the producer's thread**. This couples
the runtimes to a specific consumer and forces always-on producer-side
overhead.

The medium-term goal is to move HIP/HSA to a **generic emit-and-subscribe
transport**: HIP/HSA emit typed events to a tracing channel without
naming a consumer, and rocprofiler-sdk (or any other tool) subscribes
out-of-process. Steady-state cost when nobody is subscribed should be
~one branch per event.

**Hard constraints** (same as the firmware-ring side of this work):
1. **No `LD_PRELOAD`** — must support attaching to a process that did
   not pre-arrange tracing.
2. **No runtime binary modification** — no text-segment patching, JIT
   code rewriting, or in-place instruction overwrite when a consumer
   attaches.

These eliminate uprobes, eBPF uprobes, and (under the strict reading)
USDT — all of which arm probes by having the kernel write `INT3` (0xCC)
into the producer's text segment when a consumer attaches.

**Surviving techniques** (from `TRACING_DELIVERY_RESEARCH.md`):

| Technique | Verdict |
|---|---|
| **LTTng-UST** | Primary recommendation today |
| **Linux `user_events`** (kernel 6.4+) | Better long-term fit; held back by distro reach (RHEL 9 / Ubuntu 22.04 LTS lack the kernel) |
| io_uring channels | Borderline — late-attach discovery is unsolved on shipping kernels |

**Recommended architecture:** abstract HIP/HSA's emit interface so
**LTTng-UST is the today-backend** and **`user_events` is a drop-in
swap** when distro reach allows (RHEL 10 / Ubuntu 24.04+ era). The
hot-path call site stays the same; the backend is a per-event function
pointer set at init.

**Why this is parallel-able with the firmware-ring work:** the two
designs touch entirely different layers. The firmware-ring path is
about kernel-dispatch *timestamps*; the LTTng-UST path is about
*API tracing event delivery*. Neither blocks the other. The
firmware-ring side can ship while the LTTng-UST side is still in
prototype, and vice versa.

**Additional expertise needed for this medium-term track:**

| Area | Why |
|---|---|
| HIP / HSA runtime maintainers | Defining and inserting the typed tracepoint set; build-time `liblttng-ust` dependency |
| LTTng / CTF tooling expertise | Schema design, version evolution, multi-tool subscriber semantics, live-consumption performance |
| Distro / packaging | Verify `liblttng-ust-dev` availability across RHEL 9/10, Ubuntu 22.04/24.04, SLES 15, and AMD's container baselines |
| rocprofiler-sdk consumer side | New code path to subscribe to LTTng (via `liblttng-ctl` + `babeltrace2` or LTTng-live TCP), join with the existing in-process callback model |

**Open questions before committing** (full list in
`TRACING_DELIVERY_RESEARCH.md` §"Open Questions"):

1. Real-world LTTng-UST overhead on the HIP dispatch hot path (the
   path already has 1248 ns burned by `pool::acquire`; ~100 ns for a
   tracepoint should fit, but verify on the actual workload).
2. Multi-tool subscriber semantics — many tools each running their own
   LTTng session vs single rocprofiler-sdk consumer that re-fans-out.
3. Whether to renegotiate the "no binary modification" constraint
   specifically for **USDT**, which patches a deliberately-reserved
   NOP slot (not random executable code). Strict reading rules USDT
   out; relaxed reading reopens it as a strong candidate. Worth a
   conversation.

Full research, comparison matrix, and per-technique deep-dives:
`TRACING_DELIVERY_RESEARCH.md`.


