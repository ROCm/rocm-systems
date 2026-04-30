# Known Issues — MEC Firmware-Assisted Dispatch Tracing

This document lists issues, hacks, and missing pieces in the firmware
dispatch ring path for kernel-dispatch tracing. The code path is gated on
`hsa::firmware_dispatch_ring_available()` and is otherwise inactive.

Each item is referenced by `TODO(KNOWN_ISSUES.md item N)` comments at the
relevant `file:line` site.

> **Scope of this document — 2026-04-30 update.** All ten numbered items
> below are scoped to the **SDK-side firmware-ring drainer**
> (`source/lib/rocprofiler-sdk/kernel_dispatch/firmware_ring_drainer.{cpp,hpp}`)
> on `users/bewelton/cpc_tracing`. An alternative implementation now
> lives on `users/bewelton/lttng-kernel-ts` (PR #5519) that moves the
> drainer **into the HSA runtime** and emits FW kernel-dispatch records
> via LTTng-UST. The HSA-side drainer has a materially different
> failure-mode profile — items 1, 2, 5, 6, 9, 10 below are either
> resolved-by-construction or do-not-apply on that path. The "Applies
> to" line on each item below records which path it affects:
>
> * **SDK-side drainer** — the standalone polling drainer in
>   `firmware_ring_drainer.cpp` on `users/bewelton/cpc_tracing`.
> * **HSA-side drainer** — the per-queue drainer threads in
>   `core/runtime/dispatch_log.cpp` on `users/bewelton/lttng-kernel-ts`.
> * **Both** — applies regardless of where the drainer lives.
>
> The HSA-side drainer also has its own set of open questions (capture
> rate plateau at ~85% per record_type, two failed perf experiments
> this session, no FW-side overrun visibility); those are tracked in
> the spec at `~/ai/specs/2026-04-27-hsa-lttng-kernel-dispatch-tracing-design.md`
> and the per-experiment writeups under `~/ai/transcripts/`.

> **See also:**
> - `HIGH_LEVEL_DESIGN_SUMMARY.md` — concise high-level summary of
>   the overall design (now reflects both SDK-side and HSA-side
>   drainer paths).
> - `FIRMWARE_RING_HYBRID_DESIGN.md` — original SDK-side hybrid
>   plan that resolves items 1, 7, 8, 9, 10 by combining the
>   firmware ring with a launching-thread doorbell hook
>   (no SDK-allocated completion signals). §13 of that doc covers
>   how the SDK can integrate against the HSA-side drainer instead.
> - `KFD_DISPATCH_LOG_DESIGN.md` — proposed KFD/UAPI evolution that
>   moves dispatch-log setup from `UPDATE_QUEUE` (today's
>   `lttng-kernel-ts` interface, MINOR=22) into the existing
>   profiler ioctl (proposed MINOR=23), plus a self-describing JSON
>   descriptor so the consumer no longer hardcodes record format.
> - `TRACING_DELIVERY_RESEARCH.md` — research on replacing the
>   HIP/HSA → rocprofiler-sdk callback delivery with a generic
>   emit-and-subscribe transport (LTTng-UST). Producer-side complete
>   in PRs #5475 + #5513; HSA FW kernel-dispatch records also flow
>   through this transport on the `lttng-kernel-ts` branch.

## Scope

The MEC firmware writes 16-byte dispatch records into a host-visible ring
buffer per HSA queue. The host SDK polls the ring, pairs START/END
records, resolves kernel objects from the AQL packet ring, and emits
`ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH` /
`ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH` records.

The path bypasses HSA queue interception. The known issues fall into three
buckets:

* **Correctness gaps** — items 1, 2, 9 produce records that are wrong or
  unlinked to the rest of the trace.
* **Robustness/scalability** — items 3, 4, 5, 6 are fine in short-lived
  test runs but will misbehave at scale or under stress.
* **Feature gaps** — items 7, 8, 10 are missing data fields and missing
  call sites.

---

## 1. Correlation IDs are decoupled from API tracing

**Applies to:** SDK-side drainer.
**On HSA-side drainer:** RESOLVED BY CONSTRUCTION. The HSA-side path
emits `rocm_hsa:kernel_dispatch_record(queue_id, dispatch_idx, ...)`
into the same CTF stream as `rocm_hip:hip_aql_kernel_dispatch_submit`.
Consumers join on `(queue_id, dispatch_idx)` (== schema-v3
`(queue_id, write_idx)`); the HIP submit event already carries the
correlation_id from the HIP API ENTER's TLS stack, the launching tid,
and the external-correlation snapshot. No drainer-side correlation
construction is needed.

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`process_dispatch_record()` (`fallback_cid` block).

The drainer constructs a fresh `correlation_id` per record at drain time.
Standard kernel-dispatch tracing assigns the correlation id at enqueue
time inside `hsa::WriteInterceptor`, where the HIP API wrapper's
correlation id is on the per-thread stack and gets propagated into the
dispatch.

Consequences:

* Internal correlation IDs do not link kernel records to HIP/HSA API
  records.
* Ancestor correlation IDs are always zero.
* External correlation IDs pushed by the tool before
  `hipLaunchKernel` are not seen by the dispatch record.

If `correlation_tracing_service::construct()` returns null during
finalization (the service has begun teardown), the drainer hands back a
`thread_local` zero-initialized `correlation_id` so the record still
flows. Records emitted via this fallback have all-zero correlation IDs
that cannot be distinguished from a genuine zero correlation.

**Real fix:** either intercept HIP API entry points to push a correlation
id even in firmware-ring mode, or implement a flush handshake so the
drainer drains before the correlation service tears down.

## 2. Multi-XCC START/END pairing is heuristic

**Applies to:** SDK-side drainer.
**On HSA-side drainer:** DOES NOT APPLY. The HSA-side path emits one
LTTng event per FW record without pairing — it is intentionally
`record_type`-opaque (HSA emits records as-is; consumer-side decides
START vs END semantics). Multi-XCC pairing, if needed, becomes a
consumer-side join over per-`record_type` events that share the same
`(queue_id, dispatch_idx)`. Per-XCC correctness still requires the FW
to include an XCC/pipe id in the record; that part is unchanged.

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` — `drain_all()`.

START/END records are paired by selecting, for each END, the most recent
pending START with the smallest positive duration. This works for
serialized kernels and reasonably-spaced concurrent kernels. For
concurrent kernels on different XCCs that overlap in time and have
similar durations, the heuristic will pair START and END records from
different XCCs and produce wrong durations.

**Real fix:** the firmware record needs to include an XCC/pipe id (one of
the reserved scratch fields can hold it). Pairing then becomes per-XCC
and trivially correct.

## 3. Shutdown grace period is a fixed 10ms sleep

**Applies to:** SDK-side drainer.
**On HSA-side drainer:** PARTIALLY ADDRESSED. The HSA-side path
performs a synchronous final-drain pass per queue at disable
(`per_queue_drain_loop` calls `drain_one_queue(force_emit=true)` after
observing `should_stop`), and waits for all per-queue worker threads to
join before releasing the per-queue rings (`std::thread::join` on each
worker; the destructor protects against self-join via
`std::thread::id` comparison). There is still no FW quiesce — records
written by FW between the final drain and queue destruction will be
lost. The HSA-side spec §4 disable sequence treats this as an accepted
gap.

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`stop_firmware_dispatch_ring_drainer()`.

`stop_firmware_dispatch_ring_drainer()` sleeps for 10ms before signaling
the drainer to stop, in the hope that the drainer's 1ms polling loop will
catch any records emitted by recently-completed kernels.

**Real fix:** quiesce the firmware (no new dispatches in flight), then
explicitly drain and join.

## 4. Hard-coded 1ms drainer poll cadence

**Applies to:** SDK-side drainer.
**On HSA-side drainer:** DIFFERENT TRADE-OFF. The HSA-side per-queue
worker uses `std::this_thread::yield()` when work was found and
`sleep_for(500us)` when idle — so an idle process costs ~2000
wakeups/sec/queue (vs SDK-side's 1000/sec/process), but a busy queue
spins faster than the SDK-side 1ms cadence. The capture-rate plateau
at ~85% per record_type on graphbench suggests the bottleneck is not
the polling cadence; adaptive backoff or condition-variable wakeups
remain candidates if FW-side telemetry shows the drainer falling
behind during bursts. Two perf experiments tried this session
(256K ring; batched lock-once translate) both regressed and were
reverted.

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`drainer_loop()`.

The drainer thread wakes every 1ms regardless of activity. On idle
processes this is 1000 wakeups/sec.

**Real fix:** adaptive backoff (e.g., exponential, capped at some upper
bound), or condition-variable wakeups driven by an HSA signal that the
firmware can pulse.

## 5. `g_emitted_dispatch_idx` and `last_processed_record_count` grow unboundedly

**Applies to:** SDK-side drainer.
**On HSA-side drainer:** DOES NOT APPLY. The HSA-side path uses a
host-managed monotonic cursor (`next_idx`) per queue plus
slot-zeroing-on-consume (`record_type==0` is the empty-slot
sentinel). No process-global dedup set. Memory cost is fixed at
ring-size per queue (today: 64K records × 16 B FW stride × N queues;
the kernel BO is oversized at `count*40` only for KFD's BO-size
validation — see KFD_DISPATCH_LOG_DESIGN.md §0). 32-bit
`dispatch_idx` wraparound is still a hazard if a single queue exceeds
2³² dispatches without re-creation; the consumer-side join key
(`queue_id, dispatch_idx`) inherits the same wrap.

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`g_emitted_dispatch_idx`, `queue_ring_state_t::last_processed_record_count`.

`g_emitted_dispatch_idx` is a process-global `unordered_set<uint32_t>` used
to deduplicate records across drain passes. It is cleared only at
shutdown.

`last_processed_record_count` is a per-queue monotonic counter used to
short-circuit the rescan when no new records have arrived.

Two failure modes for long-running processes:

1. `g_emitted_dispatch_idx` memory grows linearly with dispatch count.
2. After 2^32 dispatches, the firmware's `dispatch_idx` (which is
   `read_dispatch_id[31:0]`) wraps. `g_emitted_dispatch_idx` will then
   suppress legitimate new records.

**Real fix:** replace with a per-queue ring of recently-seen indices
sized to `queue->size` (or a small multiple), and use the underlying
`read_dispatch_id` 64-bit width if exposed.

## 6. `hsa_amd_profiling_set_profiler_enabled(true)` is sticky

**Applies to:** SDK-side drainer.
**On HSA-side drainer:** RESOLVED BY CONSTRUCTION. The HSA-side path
owns the profiling-bit lifecycle directly via
`AqlQueue::SetProfiling(bool)` driven by a refcount
(`QueueProfilingAcquire`/`QueueProfilingRelease`, spec §4a).
`SetProfiling(false)` runs on the disable edge and unpublishes the
buffer from KFD/MQD before clearing the bit (spec §4 ordering).
Stage-1 and stage-2 debate reviews verified the rollback ordering on
allocation/registration failure (commits `4312e06cc0`, `e6abbfc7fa`).

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`register_or_refresh_queue()`.

We unconditionally enable profiling on every discovered GPU queue and
never disable it. This is an externally-visible HSA state mutation that
persists past `stop_firmware_dispatch_ring_drainer()`.

**Real fix:** track per-queue enabled state, only enable on first
discovery, and disable on `stop_firmware_dispatch_ring_drainer()` for
queues we enabled.

## 7. Late-attach kernel-symbol resolution falls back to raw `kernel_object`

**Applies to:** SDK-side drainer (consumer-side concern on HSA-side
drainer too).
**On HSA-side drainer:** SHIFTED to the consumer. The HSA-side path
emits raw `(queue_id, dispatch_idx, gpu_ts, record_type)` and does
not look up `kernel_object`. The consumer (rocprofiler-sdk's
LTTng-CTF translator, or babeltrace2-based tools) must do the
late-attach kernel-symbol synthesis on its end. The fix described
below — enumerate loaded executables at attach time and synthesize
code-object load callbacks — applies to whichever component owns
the consumer-side `KERNEL_DISPATCH` record translation.

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`emit_kernel_dispatch_tracing()` (kid fallback).

If `code_object::get_kernel_id()` returns zero — typically because the
code-object load callback ran before the tool attached — the drainer
substitutes the raw `kernel_object` GPU VA as a synthetic `kernel_id`.
Tools doing `kernel_id`-keyed lookups against
`ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT` records will see ids they
never registered.

**Real fix:** at attach time, enumerate all loaded executables and
synthesize the corresponding code-object load callbacks before any
dispatch records can be emitted.

## 8. `workgroup_size`, `grid_size`, `private_segment_size`, `group_segment_size` are zeroed

**Applies to:** SDK-side drainer.
**On HSA-side drainer:** RESOLVED BY HIP CLR INSTRUMENTATION. The
LTTng-UST track in PR #5475 already emits `workgroup_size`,
`grid_size`, `private_segment_size`, `group_segment_size` on
`rocm_hip:hip_aql_kernel_dispatch_submit` from the HIP CLR side, where
the AQL packet is built. The consumer joins that submit event with
the HSA-emitted `kernel_dispatch_record` on
`(queue_id, dispatch_idx)` and recovers the dimensions for free.

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`emit_kernel_dispatch_tracing()` (`dispatch_info` setup).

The interception path captures these from the AQL packet at enqueue.
The drainer can read them from the same AQL packet slot it already reads
the `kernel_object` from (`lookup_kernel_object`); we just don't yet.

**Real fix:** extend `lookup_kernel_object` to return the relevant fields
or pass back the whole packet by value.

## 9. `thread_id` is the drainer thread's tid

**Applies to:** SDK-side drainer.
**On HSA-side drainer:** RESOLVED BY CONSTRUCTION. The HSA-side path
does not put a `thread_id` on the `kernel_dispatch_record` event at
all — the launching thread's tid is recovered consumer-side by joining
with `rocm_hip:hip_aql_kernel_dispatch_submit`, whose CTF channel
context's `vtid` is the launching thread (the HIP wrapper runs on the
launcher's thread). The drainer's tid is irrelevant to the consumer.

**Location:** `kernel_dispatch/firmware_ring_drainer.cpp` —
`emit_kernel_dispatch_tracing()` (`thr_id`).

The interception path captures the launching thread's tid at enqueue.
The drainer runs on its own thread and has no way to know who launched
the kernel.

**Real fix:** if HIP API entry points are intercepted (see item 1), the
launching tid can be propagated through the AQL packet's user data
field.

## 10. `KERNEL_DISPATCH_ENQUEUE` callbacks never fire

**Applies to:** SDK-side drainer.
**On HSA-side drainer:** RESOLVED at the SDK consumer layer. The HIP
wrapper's `rocm_hip:hip_aql_kernel_dispatch_submit` is the natural
ENQUEUE event — it fires on the launching thread at the moment the
AQL packet is committed. The SDK's CTF translator can fire
`KERNEL_DISPATCH_ENQUEUE` ENTER/EXIT callbacks at the moment that
event is decoded (caveat: callback runs on the LTTng-live consumer
thread, not the launching thread; see
`FIRMWARE_RING_HYBRID_DESIGN.md` §13.3 cons).

The firmware-ring path bypasses `hsa::WriteInterceptor`, which is the
source of `ROCPROFILER_KERNEL_DISPATCH_ENQUEUE` ENTER/EXIT phase
callbacks. Tools relying on the ENQUEUE phase will silently see nothing.
Only the `COMPLETE` phase fires.

**Real fix:** if/when API interception is added (see item 1), fire the
ENQUEUE callbacks from the API entry point.

---

## Out of scope for this branch

* No tests added — existing kernel-dispatch tests exercise only the
  interception path. A firmware-ring path test would need either a
  firmware-capable target or a faked ROCr that produces synthetic ring
  buffers.
* `cmake/rocprofiler_sdk_options.cmake` does not gate this code on a
  build flag. The path is silently inactive without runtime support.
