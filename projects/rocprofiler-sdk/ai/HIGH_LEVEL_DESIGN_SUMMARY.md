# Firmware-Assisted Kernel Dispatch Tracing — Design Summary

> Detail docs: `KNOWN_ISSUES.md`, `FIRMWARE_RING_HYBRID_DESIGN.md`,
> `KFD_DISPATCH_LOG_DESIGN.md`, `LTTNG_DLOPEN_DESIGN.md` (in this
> directory).

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

Two userspace paths exist today over the same MEC + KFD substrate. The
**SDK-side polling drainer** on `users/bewelton/cpc_tracing` is the
original reference baseline that this document and the companion docs
were written against. The **HSA-resident drainer + LTTng-UST emission**
on `users/bewelton/lttng-kernel-ts` (PR #5519) is the actively
developed path and the one currently producing measurements on gbt350.

| Layer | What ships | Status |
|---|---|---|
| MEC firmware (`/lib/firmware/amdgpu/gc_9_5_0_mec.bin`) | `SubAqlProfBufWriteRecord` writes 16-byte records `{ts_lo, ts_hi, record_type, dispatch_idx}` to a host-visible ring on every dispatch start (`record_type=2`) and EOP (`record_type=1`). **Post Apr-29 firmware update on this machine, both record types are produced at ~1:1 ratio**; the prior FW build wrote `rt=2` only ~0.17% of the time (a FW bug). The blob mtime did not change — the live FW behavior changed at Apr 29 13:02 GPU resets. | gfx9_5_0 / MI350 only |
| KFD (`amdgpu`) | Installed DKMS at `/usr/src/amdgpu-6.18.4-2286447.22.04/` plumbs `dispatch_record_buffer_addr/size` through extended `AMDKFD_IOC_UPDATE_QUEUE` (KFD `MINOR=22`) into v9 MQD fields 0x2B–0x2F. BO size validation is `count * 40` bytes (per-slot stride; FW writes 16; trailing 24 unused). Upstream `* 16` fix is in user's amdgpu tree at HEAD `03a8b58c3b96` but not yet in installed DKMS — see `KFD_DISPATCH_LOG_DESIGN.md` §1.5. | gfx9 only |
| HSA runtime (ROCr) — SDK-side reference | Original HSA APIs `hsa_amd_queue_iterate`, `hsa_amd_profiling_get_dispatch_records`. Allocates and registers the ring; consumer is the standalone SDK drainer below. Foundation for `FIRMWARE_RING_HYBRID_DESIGN.md`. | Posted at `users/bewelton/cpc_tracing` |
| HSA runtime (ROCr) — HSA-resident drainer (production path) | Per-queue drainer threads in `core/runtime/dispatch_log.cpp` use sentinel-scan (no host-visible FW wptr; `record_type==0` is the empty marker). One worker thread per active queue. `AqlQueue::SetProfiling(true/false)` driven by `QueueProfilingAcquire`/`Release` refcount (spec §4a). Enable/disable hooks in `core/runtime/hsa.cpp::on_queue_create / on_queue_destroy`. | Posted at `users/bewelton/lttng-kernel-ts` (PR #5519, draft) at HEAD `e6abbfc7fa`; 4 implementation commits stage-1 + stage-2 debate-reviewed; spec at `~/ai/specs/2026-04-27-hsa-lttng-kernel-dispatch-tracing-design.md` (8-round adversarial debate, 27 claims accepted) |
| LTTng-UST transport (HIP CLR + ROCr) | Vendored LTTng-UST 2.13.7 + URCU 0.14.0 submodules under `projects/{clr,rocr-runtime}/external/`; ~530 HIP + ~270 HSA wrappers migrated; 73 HIP + 10 HSA curated typed-args APIs; schema v3 (`(vpid, vtid, ts)` join, no in-band correlation IDs). HSA emits `rocm_hsa:kernel_dispatch_record(queue_id, dispatch_idx, gpu_ts, record_type, ...)` into the same CTF stream as `rocm_hip:hip_aql_kernel_dispatch_submit`. | Posted at `users/bewelton/lttng` (PR #5475) + `users/bewelton/lttng-curated-verifier` (PR #5513, stacked) |
| rocprofiler-sdk — SDK-side polling drainer (reference) | Standalone polling drainer reads the ring at 1 ms cadence, pairs START/END heuristically, emits `KERNEL_DISPATCH_COMPLETE` records. Reference baseline only; this is what the `KNOWN_ISSUES.md` 10 numbered items are scoped to. | Posted at `users/bewelton/cpc_tracing` |
| rocprofiler-sdk — LTTng CTF consumer | Translates `rocm_hip:*` + `rocm_hsa:*` LTTng events to existing `rocprofiler_*_record_t` shapes; joins HIP submit + HSA dispatch_record on `(queue_id, dispatch_idx)`. | **Not yet started.** Two integration paths under consideration in `FIRMWARE_RING_HYBRID_DESIGN.md` §13 (Path A: consume LTTng; Path B: bypass HSA, query KFD directly). |

**End-to-end measurement on gbt350 (HSA-resident drainer + LTTng-UST,
graphbench, 5.17M HIP submits per run):** **169.3% combined-record
capture rate (~85% per record_type)**, range 158.7%–184.0% across reps.
Two perf experiments tried this session — 256K ring (regressed to
126.3%) and a batched lock-once translate in the drainer (regressed to
154.6%) — both reverted; 64K ring + per-queue threads is the current
local optimum.

**Performance impact on the non-profiling path is dependent on the
ratio of kernel launches to device syncs.** The cost lives in MEC
firmware's `AqlConnect` path (the routine that runs when the hardware
scheduler attaches a queue to a compute pipe), which now does two
extra TC reads of the MQD to pick up the dispatch ring buffer config.
That work happens **per queue-reconnect, not per dispatch**.
`hipDeviceSynchronize()` typically forces a queue drain → eviction →
re-connect, so syncing after every launch puts the cost on the per-
dispatch path. Workloads with batched dispatches between syncs
amortize the cost over many launches. This cost is identical for both
the SDK-side and HSA-side drainer paths — it is on the FW/MQD side.

> **All numbers in this section are preliminary and subject to
> change.** They are point-in-time results from a small number of runs
> on a single machine; they have not been characterized across firmware
> revisions, ASIC variants, or extensive workload coverage. Treat as
> directional, not as a steady-state performance contract.

| Workload pattern                                | Preliminary overhead | Source |
|-------------------------------------------------|----------------------|--------|
| graphbench, additional LTTng overhead from the 2 new HSA tracepoints (`kernel_dispatch_record`, `kernel_dispatch_drop`) | ~0.3% wall on top of the existing base LTTng instrumentation | gbt350, this session |
| graphbench (banff MI325X), full LTTng curated capture (97 event types, ~10M events) | +0.9% wall vs no-tracing baseline, **0 drops** | LTTng-UST track measurement; see "Parallel track" section below |

The per-`AqlConnect` overhead is the only added cost the FW/MQD side
imposes on the non-profiling path; the per-dispatch path itself is
unchanged.

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

> **Status update — 2026-04-30.** This SDK-side hybrid has been
> overtaken by an alternative implementation that moves the drainer
> **out of rocprofiler-sdk and into the HSA runtime**, and delivers
> the FW kernel-dispatch records via the same LTTng-UST channel that
> already carries HIP/HSA API events. That work lives on
> `users/bewelton/lttng-kernel-ts` (PR #5519, draft) — see the
> firmware-ring track row in the Status snapshot below, and
> `FIRMWARE_RING_HYBRID_DESIGN.md` §13 for the two SDK integration
> paths against the HSA-resident drainer (consume LTTng directly, or
> query KFD ioctl directly). The PR 5219 critical dependency below
> applies only if rocprofiler-sdk pursues the SDK-side hybrid; the
> HSA-side path bypasses PR 5219 entirely.

Adds a thin wrapper on `hsa_signal_store_relaxed/screlease` (the
doorbell stores) that runs synchronously on the launching thread.
On every doorbell ring it captures the correlation ID and tid into a
per-queue side table, then chains through. The drainer thread looks up
the side table when it processes a firmware-ring END record.

**This recovers correlation IDs, ENQUEUE callbacks, launching tid,
external correlations — without ever invoking `WriteInterceptor` and
without allocating any HSA signals.** Strictly cheaper than the
existing `WriteInterceptor`-based interception.

**Critical dependency (SDK-side path only):** PR 5219
(`users/bewelton/no-interecept-queue`). That PR already adds the
doorbell-wrap machinery for a different purpose (inline queue intercept
that still calls `WriteInterceptor`). The SDK-side hybrid sits on top
of that PR's doorbell-map / wrapper infrastructure — we **need PR 5219
merged before the SDK-side hybrid can land**. The
`users/bewelton/cpc_tracing` branch is already rebased onto PR 5219 to
make this dependency explicit. The HSA-side path on
`users/bewelton/lttng-kernel-ts` does not require PR 5219.

Phasing (in `FIRMWARE_RING_HYBRID_DESIGN.md` §12) — applies to the
SDK-side hybrid only:

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

> The LTTng-UST track has its own expertise table in the parallel-track
> section below (§"Parallel track: LTTng-UST emit-and-subscribe transport").
> The two tracks have largely disjoint reviewer sets — firmware-ring needs
> KFD/MEC/HSA-runtime expertise; LTTng-UST needs HIP/HSA-runtime
> instrumentation, CTF tooling, and (eventually) rocprofiler-sdk
> consumer-side expertise.

## Status snapshot

**Firmware-ring track:**

| Item | Status |
|---|---|
| Firmware ring + drainer baseline (MI350) | working on `gbt350-odcdh5-wbc1-b` |
| Standalone branch `users/bewelton/cpc_tracing` (SDK-side drainer reference) | pushed, builds clean (verified on remote) |
| Branch rebased on PR 5219 (`users/bewelton/no-interecept-queue`) | done |
| SDK hybrid plan (`FIRMWARE_RING_HYBRID_DESIGN.md`) | written; **superseded by HSA-resident drainer below — preserved as reference / Path B integration option** (see `FIRMWARE_RING_HYBRID_DESIGN.md` §13) |
| **HSA-resident drainer + LTTng emission (`users/bewelton/lttng-kernel-ts`, PR #5519, draft)** | **HEAD `e6abbfc7fa`; per-queue drainer threads emit `rocm_hsa:kernel_dispatch_record` via LTTng-UST; spec at `~/ai/specs/2026-04-27-hsa-lttng-kernel-dispatch-tracing-design.md` (8-round adversarial debate, 27 claims accepted); 4 implementation commits stage-1+stage-2 debate-reviewed. Measured 169.3% combined-record capture rate on graphbench (~85% per record_type); 158.7%–184.0% range across reps. No PR 5219 dependency. Currently uses KFD MINOR=22 (extended `UPDATE_QUEUE`); will migrate to MINOR=23 when KFD plan lands.** |
| KFD ioctl + descriptor plan (`KFD_DISPATCH_LOG_DESIGN.md`) | written; **adversarially reviewed and revised** (commit `fa0764e8ee`). Today's interface (extended `UPDATE_QUEUE`, MINOR=22) documented as the ABI `lttng-kernel-ts` ships against; proposed migration (profiler ioctl, MINOR=23) decoupled from the userspace path choice |
| Phase 1 implementation (SDK-side hybrid) | not yet started — gated on PR 5219 merge **and** on a decision to pursue the SDK-side path over LTTng consumption (`FIRMWARE_RING_HYBRID_DESIGN.md` §13.5 recommends LTTng) |
| rocprofiler-sdk consumer of HSA `kernel_dispatch_record` LTTng events | **not yet started** — same scope as the SDK consumer for HIP/HSA API events, joins on `(queue_id, dispatch_idx)` against `rocm_hip:hip_aql_kernel_dispatch_submit` |
| KFD implementation | not yet started — gated on KFD-team review of the plan |
| Firmware port to non-MI350 | not yet started — **requires firmware-team engagement** |

**LTTng-UST emit-and-subscribe track (parallel):**

| Item | Status |
|---|---|
| Producer-side instrumentation (HIP CLR + ROCr) | **complete; in PR #5475 (draft)** |
| Vendored LTTng-UST 2.13.7 + URCU 0.14.0 (submodules, flat install) | **complete; in PR #5475** |
| Schema design (v3, no in-band corr_id; vpid+vtid+ts join) | **complete; adversarially reviewed in 6+8 rounds across 3 vendors** |
| Curated typed-args events (73 HIP + 10 HSA APIs) | **complete; in PR #5475** |
| Wrapper migration (~530 HIP + ~270 HSA) via libclang AST rewrite | **complete; coverage gate enforced** |
| libclang YAML↔header drift verifier + test suite | **complete; in PR #5513 (stacked on #5475)** |
| Real-resource coverage harness + payload/invariant tests | **complete; in PR #5513** |
| Performance validated on GraphBench (MI325X) | **+0.9% wall-time at full curated capture, 0 drops** |
| TheRock manylinux build validation | **complete on `bewelton_therock` container** |
| rocprofiler-sdk consumer side (CTF→record translator) | **not yet started — separate planned PR** |
| Cross-distro CI matrix | not yet wired |

## Parallel track: LTTng-UST emit-and-subscribe transport (in flight)

Today, HIP and HSA call back into rocprofiler-sdk via function pointers
populated through `rocprofiler-register`. The runtimes know about
rocprofiler-sdk (via the shared API table contract), and rocprofiler-sdk
receives data **synchronously on the producer's thread**. This couples
the runtimes to a specific consumer and forces always-on producer-side
overhead even when nobody is subscribed.

The parallel goal is to move HIP/HSA to a **generic emit-and-subscribe
transport**: HIP/HSA emit typed events to a tracing channel without
naming a consumer, and rocprofiler-sdk (or any other tool) subscribes
out-of-process. Steady-state cost when nobody is subscribed is ~one
atomic load + branch per event.

**Hard constraints** (same as the firmware-ring side of this work):
1. **No `LD_PRELOAD`** — must support attaching to a process that did
   not pre-arrange tracing.
2. **No runtime binary modification** — no text-segment patching, JIT
   code rewriting, or in-place instruction overwrite when a consumer
   attaches.

These eliminate uprobes, eBPF uprobes, and (under the strict reading)
USDT — all of which arm probes by having the kernel write `INT3` (0xCC)
into the producer's text segment when a consumer attaches. Full survey
of considered techniques and the elimination process: see
`TRACING_DELIVERY_RESEARCH.md`.

**Why this is parallel-able with the firmware-ring work:** the two
designs touch entirely different layers. The firmware-ring path is
about kernel-dispatch *timestamps*; the LTTng-UST path is about
*API-event delivery*. Neither blocks the other. The firmware-ring side
can ship while the LTTng-UST side ships, and vice versa.

> **2026-04-30 update.** The two tracks are now actively converging.
> The HSA-resident drainer on `users/bewelton/lttng-kernel-ts`
> (PR #5519) emits FW kernel-dispatch records as
> `rocm_hsa:kernel_dispatch_record` LTTng events into the same CTF
> stream that already carries `rocm_hip:hip_aql_kernel_dispatch_submit`
> and the curated `<api>_args` events. Schema v3's
> `(queue_id, write_idx)` join key from the LTTng track is the same
> `(queue_id, dispatch_idx)` join key the FW writes into each record,
> so a single LTTng consumer can pair HIP API events with FW
> dispatch-completion timestamps over one transport. This collapses
> the SDK consumer surface from "two transports + two pairing paths"
> to "one CTF stream + one CTF event-pair routine".
> See `FIRMWARE_RING_HYBRID_DESIGN.md` §13.3 for the consumer-side
> mechanism.

### Status: producer-side instrumentation complete; consumer side not started

Two stacked draft PRs are open against `develop`:

| PR | Branch | Contents |
|---|---|---|
| **#5475** | `users/bewelton/lttng` | Producer-side instrumentation: HIP CLR + ROCr LTTng-UST tracepoint providers, vendored LTTng-UST 2.13.7 + URCU 0.14.0 submodules, `~530` HIP + `~270` HSA wrappers migrated, schema v3 (no in-band correlation IDs) |
| **#5513** | `users/bewelton/lttng-curated-verifier` | Stacked on #5475: libclang YAML↔header drift verifier, build-time symbol-coverage gate, DSL parser library, real-resource coverage harness, payload/invariant test suite, CI workflow |

What's been built (~14k LOC across both PRs):

* **Vendored LTTng-UST 2.13.7 + userspace-rcu 0.14.0** as submodules under
  `projects/{clr,rocr-runtime}/external/`. Built once per ROCm build via
  `ExternalProject_Add`, installed flat into `/opt/rocm/lib/` alongside
  `libamdhip64.so` and `libhsa-runtime64.so`. **No system `liblttng-ust-dev`
  required at runtime or build time** — the original distro-portability
  concern is dissolved by vendoring.
* **HIP CLR producer** (`projects/clr/hipamd/src/lttng/`): 6 generic event
  types (`hip_api_enter`, three typed exits, `hip_kernel_dispatch_enqueue`,
  `hip_aql_kernel_dispatch_submit`) plus 73 typed `<api>_args` events for
  curated APIs.
* **ROCr producer** (`projects/rocr-runtime/runtime/hsa-runtime/lttng/`): 8
  generic events (the HIP set plus `hsa_api_exit_{u64,i64}`,
  `hsa_doorbell_ring`, `hsa_intercept_packets`) plus 10 typed `<api>_args`
  events.
* **Curated parameter capture (Option C, QEMU/DPDK pattern):** YAML DSL
  declares typed args for 73 HIP + 10 HSA APIs; Python codegen emits
  tracepoint-event headers and emit helpers that are checked into the tree
  (default build needs no Python or libclang); a libclang-based drift
  verifier runs in CI to catch YAML↔header drift; a build-time
  symbol-coverage gate fails the build if any exported HIP/HSA symbol is
  unmigrated.
* **Schema v3 (no in-band correlation IDs):** the producer carries no
  per-event correlation identifiers. Consumers reconstruct enter/exit
  pairing and parent attribution by walking the per-`(vpid, vtid)` event
  stream sorted by CTF event-header timestamp using a LIFO stack. This is
  strictly more correct for multi-process tracing than a per-process
  counter would be — `(vpid, vtid)` is unambiguous across MPI ranks even
  if both ranks happen to mint identical thread IDs. ~530 LOC and the
  entire `librocprofiler-register/correlation` ABI surface were deleted as
  part of this decision.
* **Cross-runtime correlation join key:** `(queue_id, write_idx)` on
  `hip_aql_kernel_dispatch_submit` is the natural key for joining HIP API
  events with the firmware-ring track's GPU completion records.
* **Default-on with escape valve:** `HIP_ENABLE_LTTNG_UST=ON` and
  `ROCR_ENABLE_LTTNG_UST=ON` by default. Setting either to `OFF` compiles
  out all tracing. Runtime kill switch: `ROCM_LTTNG_UST_DISABLE=1` skips
  provider registration entirely (useful in restricted containers without
  `lttng-sessiond` access).

### Measured overhead (GraphBench, MI325X gfx942, see PR #5475)

12 reps each, `taskset -c 4 chrt -f 50`, /dev/shm 64 MiB, LTTng channel
`--discard --subbuf-size=65536 --num-subbuf=4`.

| Config | wall mean (s) | × lttng_off | drops |
|:---|---:|---:|---:|
| **lttng_off** (build with LTTng-UST linked but no session) | **6.51** | 1.000× | n/a |
| **lttng_on**, generic only (14 event types) | **6.55** | **1.006×** (+0.6%) | **0** |
| **lttng_on**, generic + curated (97 event types, ~10M events) | **6.58** | **1.009×** (+0.9%) | **0** |
| rocprofv3 `--hip-trace` | 7.20 | 1.107× (+10.7%) | n/a |
| rocprofv3 `--hsa-trace` | 11.28 | 1.733× (+73.3%) | n/a |
| rocprofv3 `--hip-trace --hsa-trace` | 11.93 | 1.832× (+83.2%) | n/a |

**Headline:** capturing the full 97 event types — including the 83 typed
`_args` events with per-API parameter capture — adds **0.9% wall-time vs
the no-tracing baseline with zero discarded events**. The same workload
under `rocprofv3 --hsa-trace` at full fidelity costs **+73-83% — roughly
80× more expensive** than the LTTng-UST producer at full curated capture.

The "lttng_off" cost (build-with-LTTng-linked-but-no-session) over a
hypothetical no-LTTng build is ~one atomic load + branch per event — too
small to separate from noise on this workload, consistent with upstream
LTTng-UST microbenchmark numbers (~5 ns when OFF).

Per-question resolution against the original open-questions list lives in
`TRACING_DELIVERY_RESEARCH.md` §"Open Questions" (each item annotated
RESOLVED / DEFERRED / NOT PURSUED / PARTIAL).

### Remaining gaps

* **rocprofiler-sdk consumer side:** the producer ships standalone; any tool
  using `lttng create` + `lttng enable-event` + `babeltrace2` can consume
  today. A first-class rocprofiler-sdk CTF→record translator (so existing
  tools that subscribe via the rocprofiler-sdk callback API see LTTng-sourced
  events transparently) is **not yet started**. This is a separate planned
  PR scoped roughly at "consume CTF live via libbabeltrace2 + LTTng-live
  protocol; emit existing `rocprofiler_*_record_t` shapes".
* **Cross-distro CI:** PRs validate on TheRock manylinux container (gfx942
  MI325X). Per-distro coverage (RHEL 9/10, Ubuntu 22.04/24.04, SLES 15) not
  yet wired into CI. Vendoring removes the runtime risk; build-side risk is
  autotools availability for the bootstrap step.

### Expertise still needed

| Area | Status / why |
|---|---|
| HIP / HSA runtime maintainers | **Engaged via PR review on #5475 and #5513.** Wrapper migration via libclang AST rewrite is automated; new tracepoints require YAML edit + regen via opt-in CMake target. |
| LTTng / CTF tooling | **Internal to this work today.** External CTF expertise becomes relevant for the rocprofiler-sdk consumer translator (live-CTF read perf, schema evolution discipline). |
| Distro / packaging | **Largely dissolved by vendoring.** Build-side: autotools (`autoconf`, `automake`, `libtool`, `libtool-bin`, `pkg-config`, `patchelf`) is a hard build dep for the vendored bootstrap. |
| rocprofiler-sdk consumer side | **Not yet started.** Requires libbabeltrace2 integration in rocprofiler-sdk, design for the CTF→`rocprofiler_*_record_t` translator, and decision on live vs offline consumption. |


