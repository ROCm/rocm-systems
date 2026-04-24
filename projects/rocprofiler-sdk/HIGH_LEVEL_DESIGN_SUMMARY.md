# Firmware-Assisted Kernel Dispatch Tracing — Design Summary

> Audience: leadership review and cross-team discussion.
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
**Performance regression vs stock** on a pathological per-dispatch-sync
microbenchmark is +1.7 µs / dispatch (+10.9%); on real workloads
(batched inference, training, Triton) the regression is &lt;0.1%.

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

## Critical constraints (please read)

* **NO queue interception.** This is the defining property of the
  whole architecture. Any review comment that says "just intercept
  the queue" is rejecting the design itself.
* **NO packet rewriting.** The SDK never touches the application's AQL
  packets. The 1:1 packet invariant is trivially satisfied because
  there is no rewriting at all.
* **NO completion signal allocation by the SDK.** The firmware ring's
  END record IS the completion event. Any review comment that says
  "use a completion signal for this" is rejecting the design.

These three are not optimizations — they are the entire point of the
architecture. The firmware-ring path is what makes late-attach,
custom-runtime profiling, and zero-overhead dispatch possible.
Reverting any of the three collapses us back to the standard
intercept-and-wrap path with all its limitations.

## Expertise required to land this

| Area | Expertise needed | Why |
|---|---|---|
| **KFD / kernel maintainer** | Linux kernel UAPI conventions, ioctl design, KFD's queue-manager + MQD-manager internals, locking discipline | Reviewing and merging the `KFD_IOC_PROFILER_DISPATCH_LOG` migration. Validating `_IOC_SIZE` compatibility; signing off on the descriptor blob mechanism. |
| **HSA runtime / ROCr** | `AqlQueue::SetProfiling`, libhsakmt thunk plumbing, signal/queue lifecycle | Updating producers to use the new ioctl. Implementing the v22/v23 compat shim in libhsakmt. |
| **rocprofiler-sdk** | Existing kernel-dispatch tracing pipeline, correlation-id system, late-attach, queue lifecycle | Implementing the hybrid drainer + doorbell hook (Phase 1 ~200 lines on top of PR 5219). |
| **MEC firmware (CRITICAL)** | F32 microcode for the `SubAqlProfBufWriteRecord` path, MQD layout per ASIC, the build/sign/deploy/test loop | **Required for any ASIC beyond MI350.** Each new family (gfx10/11/12) needs an equivalent of the MI350 firmware work. This is the longest pole and most specialized skill set. Without firmware engagement, this feature stays MI350-only indefinitely. |
| **CI / release engineering** | Cross-repo coordinated rolls (KFD ↔ libhsakmt ↔ ROCr ↔ rocprofiler-sdk ↔ firmware) | The migration touches 5 layers across 3 repos plus the firmware build tree. Sequencing matters — see `KFD_DISPATCH_LOG_DESIGN.md` §5.1. |
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

## Single-paragraph summary for a slide

> Re-architect rocprofiler-sdk's kernel-dispatch tracing path to use
> MEC firmware-written ring buffer records for timestamps and a thin
> launching-thread doorbell hook for correlation. **No HSA queue
> interception. No AQL packet rewriting. No SDK-allocated completion
> signals.** Enables true late-attach profiling and effectively-zero
> per-dispatch CPU overhead. Working today on MI350; depends on
> PR 5219's doorbell-wrap infrastructure for the correlation hook.
> KFD-side cleanup migrates the buffer-setup ioctl out of
> `UPDATE_QUEUE` and adds a kernel-shipped JSON format descriptor so
> any tool can decode the ring without depending on libhsakmt.
> **Bringing this to ASIC families beyond MI350 requires firmware
> engineering effort per family — this is the long pole.**
