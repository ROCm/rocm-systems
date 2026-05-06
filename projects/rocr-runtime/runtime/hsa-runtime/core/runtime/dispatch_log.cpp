////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

// HSA-resident firmware-ring drainer with LTTng emission for kernel-dispatch
// timestamps.
//
// This file implements:
//   - The ts_poller control-plane thread (~5 ms tick, idempotent enable /
//     disable passes per spec §4 lifecycle state machine).
//   - One drainer thread per active queue (spawned at enable, joined at
//     disable). Each per-queue worker independently drains its own ring
//     buffer at the FW's local write rate, so cross-queue serialization
//     present in the prior single-shared-drainer design is eliminated.
//     See spec §7.
//   - Per-queue ENABLE / DISABLE sequences (spec §5 ordering).
//   - The QueueProfilingAcquire / Release refcount API (spec §4a).
//   - The on_queue_create / on_queue_destroy hooks (spec §4 hooks).
//
// SUBSTRATE NOTE: The KFD-side substrate is the existing
// AMDKFD_IOC_UPDATE_QUEUE ioctl extended with dispatch_record_buffer_addr +
// dispatch_record_buffer_size trailing fields (KFD minor version 22+).
// The HSA-side AqlQueue::SetProfiling owns the per-queue buffer
// allocation, the libhsakmt hsaKmtSetQueueProfilingBuffer call, and the
// Suspend/Resume that flushes the MQD via UPDATE_QUEUE. The dispatch_log
// path here just calls QueueProfilingAcquire/Release (which forward to
// SetProfiling) and then queries hsa_amd_profiling_get_dispatch_records
// to fetch the buffer info.
//
// SENTINEL-SCAN DESIGN: The substrate publishes neither a host-readable FW
// write pointer nor any other end-of-records marker. The drainer locates
// fresh records by scanning the ring sequentially from a host-managed
// monotonic cursor (next_idx). A slot whose record_type is 0 is "empty"
// (FW writes record_type ∈ {1,2}, and the buffer is pre-zeroed at alloc
// in AqlQueue::SetProfiling). After consuming a slot the drainer clears
// the record_type sentinel so wraparound rewrites are re-detectable. See drain_one_queue
// below for the canonical implementation.

#include "core/inc/dispatch_log.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <sys/ioctl.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/types.h>
#include <unistd.h>
#include "hsakmt/linux/kfd_ioctl.h"
#endif

#include "core/inc/agent.h"
#include "core/inc/amd_aql_queue.h"
#include "core/inc/amd_gpu_agent.h"
#include "core/inc/queue.h"
#include "lttng/rocm_trace_emit.h"

namespace rocr {
namespace dispatch_log {

namespace {

// ============================================================================
// Per-queue profiling-bit refcount (spec §4a). Each Queue gets one of these
// keyed in g_profiling_refcounts; the mutex serializes the refcount RMW + the bit set/clear
// transition. Phase A uses a side-map keyed on core::Queue* rather than a
// dl_state pointer hung off the Queue itself; the in-Queue void*
// dispatch_log_state is reserved for a future refactor.
// ============================================================================
struct queue_profiling_refcount {
  std::mutex m;
  uint32_t   refcount = 0;
};

// ============================================================================
// queue_drain_state per spec §7. Held via std::shared_ptr from both the
// drainer registry (g_active_drainers) and the per-queue worker thread (which
// holds its own ref by-value via per_queue_drain_loop). Stores only NON-OWNING
// pointers into the buffer owned by AqlQueue (the buffer is alloc'd by
// AqlQueue::SetProfiling(true) and freed by SetProfiling(false) or the
// AqlQueue dtor). Deliberately stores NO Queue* — see spec §4 "Queue /
// drain-state lifetime" for the dangling-Queue rationale.
// ============================================================================
struct queue_drain_state {
  // Full 64-bit hsa_queue_t::id (see hsa.h). Internal ownership / lifetime
  // maps key on this full id to avoid the high-32-bit collision risk
  // flagged in C5. The on-the-wire LTTng tracepoint payload still carries
  // only the low 32 bits per spec §6/§14 — we narrow at the
  // rocm_trace_emit_* call sites, not here.
  uint64_t  queue_id = 0;

  // NON-OWNING ring view. Buffer is owned by AqlQueue::dispatch_record_buffer_
  // (alloc'd by SetProfiling(true), freed by SetProfiling(false) or the
  // AqlQueue dtor). FW writes 16-byte mec_dispatch_record entries at
  // 16-byte stride; the drainer reads at the same stride. The host BO is
  // allocated oversized at `ring_records * 40` to satisfy the deployed
  // KFD's BO-size validation; bytes beyond `ring_records * 16` are unused.
  void*    ring_base    = nullptr;
  uint32_t ring_records = 0;     // power-of-2 slot count
  uint32_t ring_mask    = 0;     // ring_records - 1, for slot indexing

  // Host-managed monotonic record cursor. Mutated only under drain_mu.
  // Slot index is (next_idx & ring_mask).
  //
  // Two drain modes select on whether signal_ptr below is non-null:
  //   - SIGNAL-bound scan (preferred, MINOR>=20): next_idx tracks the
  //     absolute count of records the host has emitted. Each pass reads
  //     *signal_ptr atomically and iterates [next_idx, signal). After
  //     emitting all records, next_idx is set to signal and the same
  //     value is published to *rptr_ptr for FW backpressure visibility
  //     (FW does not currently read it but publishing keeps the door
  //     open for future FW that does).
  //   - Sentinel scan (fallback, older kernel/FW): next_idx still
  //     advances per non-zero slot. record_type==0 marks empty; the
  //     drainer clears the sentinel post-emit so wraparound rewrites
  //     are re-detectable.
  uint64_t next_idx = 0;

  // Phase-2 host-VA pointer set. Populated by
  // enable_dispatch_log_for_queue_locked from
  // AqlQueue::GetDispatchLogPointers when the running kernel supports
  // KFD_IOC_PROFILER_DISPATCH_LOG (MINOR>=20) and the new sub-op
  // succeeded. All three are nullptr in fallback mode.
  //
  // Memory model:
  //   - signal_ptr: FW writes per-record (post-increment record count)
  //     AFTER first writing the same value to wptr_ptr (see f32_mec.uc
  //     @SubAqlProfBufWriteRecord lines 29605-29626 — wptr write THEN
  //     TC-drain THEN signal write THEN TC-drain). Per single FW thread
  //     signal is the LAST write per record and carries the strongest
  //     "record committed" semantics. The drainer uses signal_ptr as
  //     its source of truth (see drain_one_queue Path A for the
  //     multi-XCC race rationale).
  //   - wptr_ptr: FW writes the same per-record value as signal_ptr,
  //     just earlier in the publish sequence. Kept registered so the
  //     KFD ioctl + FW-side state stays symmetric, but the drainer
  //     does not read it (signal is the source of truth).
  //   - rptr_ptr: host writes after consume with release. FW reads on
  //     queue connect (currently informational; backpressure not
  //     enforced).
  volatile uint64_t* wptr_ptr   = nullptr;
  volatile uint64_t* rptr_ptr   = nullptr;
  volatile uint64_t* signal_ptr = nullptr;

  // Per-queue drain mutex. With one worker per queue this is technically
  // redundant in steady state (the worker is the sole drainer for its
  // queue), but it is retained as defense in depth against any future
  // synchronous-final-drain helper from another thread.
  std::mutex drain_mu;

  // Per-queue worker thread shutdown signal + handle. Disable path stores
  // should_stop with release ordering and joins worker before dropping the
  // registry's shared_ptr ref. The worker checks should_stop with acquire
  // ordering each iteration.
  std::atomic<bool> should_stop{false};
  std::thread       worker;

  // Defensive safety net for early-construction failures (review C2,
  // stage-2 code-quality). Supported lifetime model is explicit
  // disable / shutdown stops + joins worker BEFORE dropping the
  // registry's shared_ptr ref. This destructor only catches the narrow
  // case where the shared_ptr's last ref is dropped with the worker
  // still running (e.g. construction-failure rollback inside
  // enable_dispatch_log_for_queue_locked). Self-join is illegal —
  // detect via thread::id and detach() instead.
  ~queue_drain_state() {
    if (worker.joinable()) {
      if (worker.get_id() == std::this_thread::get_id()) {
        worker.detach();
      } else {
        worker.join();
      }
    }
  }
};

// ============================================================================
// Global state.
// ============================================================================

// Spec §4 single source of truth for "should HSA be collecting kernel-dispatch
// timestamps right now?". Folded predicate of !rocm_trace_disabled() &&
// lttng_ust_tracepoint_enabled(rocm_hsa, kernel_dispatch_record) &&
// g_kfd_supports_dispatch_log. Written only by ts_poller; read by
// on_queue_create (relaxed, hot path).
std::atomic<bool> g_dispatch_logging_active{false};

// Set true if the KFD substrate version probe at init() reports a kernel
// minor version >= 22 (the version that introduced the
// dispatch_record_buffer_{addr,size} trailing fields on UPDATE_QUEUE).
// Probed once at init via AMDKFD_IOC_GET_VERSION; never mutated afterwards.
std::atomic<bool> g_kfd_supports_dispatch_log{false};

// Latched true after we log the "substrate absent, tracepoint enable
// is a no-op" message once. Avoids spamming the user log on every poll tick.
std::atomic<bool> g_kfd_unsupported_warned{false};

// Driven by shutdown(); poller and per-queue workers observe and exit.
std::atomic<bool> g_shutdown{false};

// Spec §4 lifecycle mutex. Guards g_active_drainers, g_all_queues, and
// g_no_dispatch_log_agents. Acquired only briefly: enable/disable per-queue
// and snapshot-cloning.
std::mutex g_queue_registry_mu;

// Drainer registry (spec §4 lifetime protocol). Keyed by full 64-bit
// hsa_queue_t::id. One shared_ptr per active queue; the per-queue worker
// thread holds its own ref by-value via per_queue_drain_loop.
std::unordered_map<uint64_t, std::shared_ptr<queue_drain_state>> g_active_drainers;

// Poller-only "live queue" side map (spec §4 enable/disable convergence).
// Distinct from g_active_drainers: tracks every Queue* known to the
// dispatch_log subsystem, regardless of whether dispatch logging is
// currently enabled on it. Iterated by the poller under g_queue_registry_mu.
std::unordered_map<uint64_t, core::Queue*> g_all_queues;

// Side-map for the QueueProfilingAcquire/Release refcount (spec §4a). One
// entry per Queue* that has ever had at least one acquire. Held via
// shared_ptr because Acquire/Release intentionally drop
// g_profiling_refcounts_mu before locking the per-entry mutex.
std::mutex g_profiling_refcounts_mu;
std::unordered_map<core::Queue*, std::shared_ptr<queue_profiling_refcount>> g_profiling_refcounts;

// Per-agent "no-dispatch-log" mark. If GetProfilingDispatchRecords reports
// NOT_INITIALIZED for any queue on a given agent, that agent is added here
// and subsequent enable_dispatch_log_for_queue calls on queues belonging to
// it short-circuit (spec §5/§9 unsupported-agent rows).
std::unordered_set<core::Agent*> g_no_dispatch_log_agents;

// Threads. Per-queue drainer threads (one per active queue) live inside
// each queue_drain_state.worker; only the poller is global.
std::thread g_poller_thread;

// ============================================================================
// Helpers.
// ============================================================================

// Full 64-bit hsa_queue_t::id (see hsa.h:2359-2361). Internal registries
// (g_active_drainers, g_all_queues) and queue_drain_state.queue_id all key
// on this full value to avoid the high-32-bit collision risk flagged in
// C5. The LTTng tracepoint payload narrows to the low 32 bits at the
// emit call sites only.
uint64_t queue_id_of(core::Queue* q) {
  return q->amd_queue_.hsa_queue.id;
}

// Narrow the 64-bit internal queue_id to the 32-bit on-the-wire identifier
// emitted by the rocm_hsa:kernel_dispatch_record / kernel_dispatch_drop
// LTTng tracepoints. Spec §6 / §14 fix the wire format at uint32_t for
// payload size; consumers join with the low 32 bits of the masked AQL
// doorbell write_idx (also 32-bit on-the-wire). This narrowing is
// deliberate and lossy: it must NOT be used for internal lookup.
uint32_t queue_id_to_wire(uint64_t qid) {
  return static_cast<uint32_t>(qid);
}

bool agent_is_gpu(core::Agent* a) {
  return a != nullptr && a->device_type() == core::Agent::kAmdGpuDevice;
}

#if defined(__linux__)
// Probe the KFD substrate version. Returns true iff the running kernel
// reports KFD minor version >= 22 (the minor that introduced the
// dispatch_record_buffer_{addr,size} trailing fields on UPDATE_QUEUE).
//
// This replaces the Phase A placeholder PROBE ioctl. We do NOT keep the
// fd — the per-queue path uses HSA's own KFD handle via the AqlQueue
// SetProfiling -> agent_->driver().SetQueueProfilingBuffer chain.
bool probe_substrate_version() {
  constexpr const char* kKfdDevicePath = "/dev/kfd";
  int fd = ::open(kKfdDevicePath, O_RDWR | O_CLOEXEC);
  if (fd < 0) return false;
  struct kfd_ioctl_get_version_args args = {};
  int rc = ::ioctl(fd, AMDKFD_IOC_GET_VERSION, &args);
  ::close(fd);
  if (rc != 0) return false;
  // The dispatch_record_buffer fields landed in the host kernel ABI at
  // KFD 1.22. Older kernels will silently ignore the trailing UPDATE_QUEUE
  // bytes (or worse, return -ENOTTY because of the _IOWR-encoded size
  // mismatch on related ioctls), so this version gate is mandatory.
  // Accept any KFD whose ABI is at-or-after 1.22. A naive
  // (major>=1 && minor>=22) check would reject a hypothetical KFD 2.0
  // (which has minor==0). Compare lexicographically instead.
  return args.major_version > 1 ||
         (args.major_version == 1 && args.minor_version >= 22);
}
#else
bool probe_substrate_version() { return false; }
#endif

// Iterate g_all_queues under g_queue_registry_mu and invoke fn(queue_id, queue)
// for each entry. C6 fix: replaces the previous "snapshot into vector then
// iterate" pattern. fn must NOT mutate g_all_queues. fn IS allowed to
// mutate g_active_drainers (e.g. enable_dispatch_log_for_queue_locked inserts,
// disable_dispatch_log_for_queue_locked erases).
template <typename F>
void for_each_known_queue_locked(F&& fn) {
  std::lock_guard<std::mutex> lk(g_queue_registry_mu);
  for (auto& kv : g_all_queues) fn(kv.first, kv.second);
}

// ============================================================================
// drain_one_queue. Holds qs.drain_mu for the entire pass.
//
// Sentinel-scan design: the substrate publishes no host-readable FW write
// pointer (KFD `kfd_ioctl_update_queue_args` has only
// dispatch_record_buffer_addr/size — see
// /usr/src/amdgpu-*/include/uapi/linux/kfd_ioctl.h:99-108 on a host with
// the gbt350 KFD). Instead, FW writes record_type ∈ {1, 2} into each slot
// it produces; the host pre-zeroes the buffer at allocation
// (AqlQueue::SetProfiling), so a slot whose record_type is 0 is "empty".
//
// We walk the ring sequentially from a host-managed monotonic cursor
// (qs.next_idx). Each iteration:
//   1. Read record_type for the slot at (next_idx & ring_mask).
//   2. If 0, the ring tail has caught up — break and return.
//   3. Otherwise snapshot the rest of the record, ZERO the slot (so a
//      future wraparound re-write of the same slot can be re-detected),
//      and process the START / END pairing as before.
//
// FW writes 16-byte mec_dispatch_record entries at 16-byte stride. The
// host kernel BO size validation on the gbt350-installed KFD patch is
// `count * 40` (kfd_process_queue_manager.c:633), which over-counts the
// per-record stride — see upstream fix `kfd: validate dispatch record
// buffer using 16-byte record size` (commit `03a8b58c3b96`,
// 2026-04-08), which corrects the kernel side to `count * 16` to
// match the firmware/SDK contract. We allocate `count * 40` so the
// older host kernel's BO validation passes (forward-compatible with
// the fix — bigger is fine), but FW writes records back-to-back at
// 16-byte stride within that buffer. The drainer therefore reads at
// 16-byte stride. The unused tail of the buffer (slots beyond
// `count * 16` bytes) is never written by FW and never read by the
// drainer.
// ============================================================================

// Per-record stride at which FW writes records into the buffer. Matches
// `sizeof(mec_dispatch_record)` and is independent of the host kernel's
// BO-size validation constant (which is `count * 40` today on gbt350,
// `count * 16` after the upstream fix lands).
constexpr size_t kSlotStride = 16;

// Per-pass batch size: drainer accumulates up to this many records per
// LTTng tracepoint call. 256 records × 16 bytes = 4 KiB stack-allocated
// buffer per drain pass. Batching is the primary mechanism for keeping
// up with peak FW write rates of ~1M records/sec on busy queues; per-record
// LTTng tracepoint emit cost (~1-2 us) was the host-side bottleneck before
// batching.
constexpr uint32_t kBatchMax = 256;

bool drain_one_queue(queue_drain_state& qs, bool force_emit) {
  std::lock_guard<std::mutex> lk(qs.drain_mu);

  // Poisoned (disable in flight) or never-populated.
  if (qs.ring_base == nullptr || qs.ring_records == 0) return false;

  // Shared batch buffer for both Path A (signal-bound) and Path B
  // (sentinel-scan). Stack-allocated; per-pass scope.
  alignas(16) uint8_t batch_buf[kBatchMax * kSlotStride];
  uint32_t batch_count = 0;
  auto flush_batch = [&]() {
    if (batch_count == 0) return;
    rocm_trace_emit_hsa_kernel_dispatch_record(
        queue_id_to_wire(qs.queue_id),
        batch_count, batch_buf,
        (size_t)batch_count * kSlotStride, force_emit);
    batch_count = 0;
  };

  // ============================================================================
  // PATH A: SIGNAL-BOUND SCAN (preferred, KFD MINOR>=20 + new FW)
  //
  // FW publishes the post-increment record count to *qs.signal_ptr after
  // first publishing the same value to *qs.wptr_ptr (see f32_mec.uc
  // @SubAqlProfBufWriteRecord lines 29605-29626 — wptr write THEN
  // TC-drain THEN signal write THEN TC-drain). Per single FW thread,
  // signal_addr is the LAST write per record, so any non-zero observation
  // implies the corresponding record body, wptr increment, and host
  // wptr_addr publish all already happened. We use signal as the source
  // of truth.
  //
  // Why not wptr too? KFD programs the same wptr_addr / signal_addr into
  // the per-MQD slots for every XCC (kfd_mqd_manager_v9.c:450-461 — no
  // master/slave gating). FW state machines on multiple XCCs can publish
  // concurrently to the same host VA without atomics, so both wptr and
  // signal can race / oscillate. Empirically wptr can go backward by
  // 300-450 records between polls. Signal races too (it's the same store
  // pattern), but using signal-only:
  //   1. Aligns with FW's intended per-record publish boundary (signal
  //      is written AFTER wptr, semantically "record committed").
  //   2. Simplifies the drainer (one counter, not two).
  //   3. Opens the door to hsa_signal_wait on signal_addr instead of
  //      polling, deferred to a future optimization.
  //
  // Backward-publish protection (still needed because signal races too):
  // qs.next_idx is our host-side monotonic floor. If FW publishes a
  // value < next_idx, treat it as stale and skip the pass; we wait for
  // a future monotonic publish to advance.
  //
  // Overrun handling: if (signal - qs.next_idx) > qs.ring_records the
  // host has fallen behind by more than one ring lap; the older records
  // were silently overwritten (FW does not stall on full ring). We jump
  // qs.next_idx forward by the overrun distance and emit a
  // kernel_dispatch_drop tracepoint so the consumer sees the gap.
  // ============================================================================
  if (qs.signal_ptr != nullptr) {
    const uint64_t observed = __atomic_load_n(qs.signal_ptr, __ATOMIC_ACQUIRE);

    // Backward (or stale) FW publish: ignore this pass entirely.
    // qs.next_idx is our monotonic floor.
    if (observed <= qs.next_idx) return false;

    // Initial-state sync. The FW maintains its dispatch-log wptr counter
    // in per-pipe scratch RAM (LdMecAqlProfBufWptr in f32_mec.uc:29579),
    // and that counter PERSISTS across queue lifecycles. When KFD
    // destroys a queue and creates a new one on the same pipe, the new
    // queue inherits the previous queue's wptr value. At
    // enable_dispatch_log_for_queue time we initialize qs.next_idx from
    // the current FW signal value, but the host signal_addr is still 0
    // at that moment (FW hasn't published since enable). Once FW writes
    // its first record post-enable, signal jumps from 0 to whatever the
    // pipe scratch wptr was + 1, which can be huge (millions).
    //
    // Without this sync the overrun handler below would interpret the
    // huge gap as "we fell behind by ring_records or more" and emit a
    // drop event followed by reads of stale ring storage (most slots
    // empty / pre-zero, only the freshly-written slot has real data).
    // Net effect: hundreds of zero-record batched events per queue
    // immediately after enable.
    //
    // Heuristic: if next_idx is 0 (never advanced past initial enable)
    // AND observed > ring_records (impossibly far ahead for a fresh
    // queue), treat as initial-state sync. Set next_idx = observed - 1
    // so we drain ONE record (the actual fresh dispatch FW just wrote)
    // and start tracking from there. The single record at slot
    // (observed - 1) & ring_mask is the real one FW just wrote at the
    // pre-increment scratch wptr position.
    if (qs.next_idx == 0 && observed > qs.ring_records) {
      qs.next_idx = observed - 1;
    }

    // Overrun detection. If FW lapped us, log a drop event and skip
    // the unrecoverable region. The records we DO read after the skip
    // are still valid (they're the most recent observed - ring_records
    // records, just with a gap before them).
    uint64_t start = qs.next_idx;
    uint64_t end   = observed;
    if ((end - start) > qs.ring_records) {
      const uint64_t lost = (end - start) - qs.ring_records;
      rocm_trace_emit_hsa_kernel_dispatch_drop(
          queue_id_to_wire(qs.queue_id), lost);
      start = end - qs.ring_records;
    }

    // Batched LTTng emit (see batch_buf / flush_batch declared above the
    // path branch). Records accumulate as we walk [start, end); when the
    // batch fills (kBatchMax records) we flush mid-loop, and after the
    // loop we flush whatever's left.
    bool any = false;
    for (uint64_t i = start; i < end; ++i) {
      // Kill-switch pre-check: bail before reading any record. Safe to
      // bail here because qs.next_idx still points at the unread head
      // and the next pass will pick up where we stopped (FW state
      // unchanged — the wptr-bound scan never mutates the slot). Flush
      // any pending batch first so we don't lose records already read.
      if (!force_emit && rocm_trace_disabled()) {
        flush_batch();
        qs.next_idx = i;  // record progress so far
        if (qs.rptr_ptr) __atomic_store_n(qs.rptr_ptr, i, __ATOMIC_RELEASE);
        return any;
      }

      const uint64_t slot = i & qs.ring_mask;
      auto* rec = reinterpret_cast<mec_dispatch_record_16*>(
          static_cast<char*>(qs.ring_base) + slot * kSlotStride);

      // FW publish-ordering contract (per f32_mec.uc Q6): record body
      // is host-visible BEFORE the wptr update we acquire-loaded above.
      // So all four DWs of this record are coherent. We copy the entire
      // 16-byte record into the batch buffer in one shot via memcpy
      // (the source is host-coherent FW-written memory at this point).
      std::memcpy(batch_buf + (size_t)batch_count * kSlotStride,
                  rec, kSlotStride);
      batch_count++;
      any = true;

      // No slot mutation: FW will overwrite this slot when wptr next
      // reaches the same physical slot, and our wptr-based iteration
      // will read it as a new record at that time.

      if (batch_count == kBatchMax) {
        flush_batch();
      }
    }
    flush_batch();

    qs.next_idx = end;
    // Publish our consumer position back to FW (informational; FW does
    // not enforce backpressure today but reads on connect/dequeue).
    if (qs.rptr_ptr) __atomic_store_n(qs.rptr_ptr, end, __ATOMIC_RELEASE);
    return any;
  }

  // ============================================================================
  // PATH B: SENTINEL-SCAN FALLBACK (older kernel/FW without host-VA wptr)
  //
  // FW writes record_type ∈ {1, 2} into each slot it produces; host
  // pre-zeroes the buffer at allocation, so a slot whose record_type
  // is 0 is "empty". Drainer walks slots from qs.next_idx, atomic-stores
  // record_type=0 after consume so wraparound rewrites are re-detectable.
  //
  // KNOWN ISSUE: FW slot reuse can cause the same (dispatch_idx,
  // record_type) pair to be observed twice in a single session under
  // burst patterns (capture rate observed at 250-316% vs expected 200%).
  // The wptr-bound path above is immune to this; this fallback is only
  // used on older kernels that don't ship the new MQD slots.
  // ============================================================================
  const uint64_t pass_budget = qs.ring_records;
  uint64_t pass_consumed = 0;

  bool any = false;
  while (pass_consumed < pass_budget) {
    // Cheap kill-switch pre-check (review C4): tested at the very top of
    // the per-record body, BEFORE any slot mutation. translate_gpu_ts
    // performs a linear interpolation under the agent's clock-cache
    // lock, so doing it 1× per record × kRingSlots in the disable-race
    // window is wasted work the emit helper would discard anyway. The
    // helper's own kill-switch check still runs (defense in depth).
    //
    // Position rationale: this check MUST sit before the record_type
    // load + slot mutation (below) so that bailing here leaves the slot
    // intact and `qs.next_idx` still pointing at it. If we cleared the
    // sentinel first and then bailed, the slot would be permanently
    // destroyed (FW had already written it, and the wptr cursor stays
    // put), and on the next drain pass the sentinel scan would observe
    // rt == 0 at next_idx and stop — stranding any later FW-written
    // records until the ring wraps and FW rewrites the broken slot.
    // Today no runtime API toggles the kill switch (it is set-once at
    // construction in lttng/rocm_trace_init.cpp), so the wedge is not
    // currently reachable, but ordering this check before slot mutation
    // removes the latent footgun for any future runtime-toggle path.
    if (!force_emit && rocm_trace_disabled()) break;

    const uint64_t slot = qs.next_idx & qs.ring_mask;
    auto* rec = reinterpret_cast<mec_dispatch_record_16*>(
        static_cast<char*>(qs.ring_base) + slot * kSlotStride);

    // Sentinel: record_type == 0 means "empty / not yet written by FW".
    //
    // FW publish atomicity contract (per core/inc/mec_dispatch_record.h):
    // each 16-byte mec_dispatch_record is written by a single GFX
    // BUFFER_STORE_DWORDX4 (4 DWords = single TC write transaction). So
    // any non-zero record_type observation implies the entire 16-byte
    // record (ts_lo, ts_hi, record_type, dispatch_idx) is coherent. A
    // partial / torn observation of "non-zero record_type but stale body"
    // is not possible under that FW contract.
    //
    // We still use an acquire on the record_type load so the subsequent
    // body loads are not reordered above it; this
    // turns the type-load into a proper atomic acquire instead of a
    // plain (data-racy) load on memory another agent writes
    // concurrently. The body fields are read with relaxed atomicity for
    // the same reason — to avoid UB from plain loads on FW-written
    // memory — but no further fence is needed because the acquire on
    // record_type already happens-before them.
    uint32_t rt = __atomic_load_n(&rec->record_type, __ATOMIC_ACQUIRE);
    if (rt == 0) break;

    // Sentinel-scan path also batches: copy the FW record (16 bytes,
    // exactly the mec_dispatch_record_16 layout the new tracepoint
    // expects) into the batch buffer, then clear the sentinel.
    std::memcpy(batch_buf + (size_t)batch_count * kSlotStride,
                rec, kSlotStride);
    batch_count++;

    // Zero the record_type field so a wraparound rewrite by FW (next time
    // the ring loops past this position) is re-detected as a fresh
    // non-zero record_type. We only need to clear the sentinel field,
    // not the entire 16-byte slot, since the FW always writes the full
    // 16 bytes atomically.
    __atomic_store_n(&rec->record_type, 0, __ATOMIC_RELEASE);

    qs.next_idx += 1;
    pass_consumed += 1;
    any = true;

    if (batch_count == kBatchMax) {
      flush_batch();
    }
  }
  flush_batch();
  return any;
}

// Per-queue drain loop. One instance per active queue, spawned at enable
// (enable_dispatch_log_for_queue_locked) and joined at disable
// (disable_dispatch_log_for_queue_locked). The thread holds its
// shared_ptr<queue_drain_state> by value, so qs is alive for the full
// thread lifetime regardless of registry mutations elsewhere.
//
// Per-queue ownership eliminates cross-queue serialization: the previous
// shared drainer iterated all queues sequentially under g_queue_registry_mu
// snapshot, with a per-pass budget capped at qs.ring_records, so a
// larger ring slowed the visit cycle to other queues. Each per-queue
// worker now stays hot on its own ring at the FW's local write rate
// without competing with other queues for thread time.
//
// Memory ordering on shutdown:
//   - disable path: store should_stop = true with release, then join.
//   - worker: load should_stop with acquire each iteration.
//   - The release/acquire pair guarantees that the worker observes
//     should_stop=true on its next loop check after the disable thread
//     stores it (within the bounded sleep cadence). join() then waits
//     for the worker's final-drain pass to complete.
void per_queue_drain_loop(std::shared_ptr<queue_drain_state> qs) {
  // TEST HOOK: HSA_DISPATCH_LOG_NO_DRAIN=1 makes the drainer worker idle
  // (sleep-loop until stop) instead of reading FW records. Lets a test
  // harness inspect raw FW output via hsa_amd_dispatch_log_test_get_state
  // without the drainer competing or emitting LTTng events. Used by the
  // dlog_test* programs to isolate FW behavior from drainer behavior.
  static const bool s_no_drain = (std::getenv("HSA_DISPATCH_LOG_NO_DRAIN") != nullptr);
  if (s_no_drain) {
    while (!qs->should_stop.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return;
  }

  // Signal-poll loop with adaptive backoff.
  //
  // The previous implementation busy-spun calling drain_one_queue() and
  // yielding only when it had work, which kept a CPU core saturated even
  // when the queue was idle. With many queues running concurrently this
  // multiplied: N queues × N busy-spin workers = N cores at 100%.
  //
  // New strategy: read the FW signal value cheaply (one acquire load),
  // compare to last_observed. If unchanged, sleep with adaptive backoff;
  // if changed, drain immediately. Backoff schedule:
  //   - 0 idle iterations: drain back-to-back (no sleep)
  //   - 1-3 idle:           yield only
  //   - 4-15 idle:           sleep 10 us
  //   - 16-63 idle:          sleep 100 us
  //   - 64+ idle:            sleep 1 ms (steady-state idle)
  // First sign of work resets the counter to 0 and resumes hot-loop.
  //
  // Path B (sentinel-scan, no signal_ptr): falls back to the old yield-
  // only loop because we have no cheap "no work" probe — we have to
  // actually read the slot's record_type to know.
  uint64_t last_signal = 0;
  uint32_t idle_count = 0;
  while (!qs->should_stop.load(std::memory_order_acquire)) {
    // Path A fast-path: if signal_ptr is registered, read it cheaply and
    // skip the drain call entirely when no new records exist.
    if (qs->signal_ptr != nullptr) {
      const uint64_t cur = __atomic_load_n(qs->signal_ptr, __ATOMIC_ACQUIRE);
      if (cur == last_signal) {
        // No new records — adaptive backoff
        ++idle_count;
        if (idle_count <= 3) {
          std::this_thread::yield();
        } else if (idle_count <= 15) {
          std::this_thread::sleep_for(std::chrono::microseconds(10));
        } else if (idle_count <= 63) {
          std::this_thread::sleep_for(std::chrono::microseconds(100));
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        continue;
      }
      last_signal = cur;
      idle_count = 0;
    }

    const bool any = drain_one_queue(*qs, /* force_emit = */ false);
    if (!any) {
      // Path B (sentinel-scan) fallback or spurious wakeup: yield.
      ++idle_count;
      if (idle_count > 3) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      } else {
        std::this_thread::yield();
      }
    }
  }
  // Final drain after the stop signal so any FW records written between
  // the last steady-state pass and the disable transition are emitted.
  // force_emit=true bypasses the per-tracepoint enabled check inside the
  // emission helper (spec §6) so disable does not lose pending records.
  drain_one_queue(*qs, /* force_emit = */ true);
}

// ============================================================================
// Bounded wait_for_idle. Spec §4 bounds at 100 ms. Polls the queue's read
// index against its write index. Returns when the queue is idle or the
// budget elapses.
// ============================================================================
void wait_for_idle(core::Queue* q) {
  using clock = std::chrono::steady_clock;
  const auto deadline = clock::now() + std::chrono::milliseconds(100);
  while (clock::now() < deadline) {
    const uint64_t r = q->LoadReadIndexAcquire();
    const uint64_t w = q->LoadWriteIndexAcquire();
    if (r >= w) return;
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
}

// ============================================================================
// Profiling-bit refcount (spec §4a).
// ============================================================================

std::shared_ptr<queue_profiling_refcount> get_or_create_owners(core::Queue* q) {
  // Caller holds g_profiling_refcounts_mu.
  auto it = g_profiling_refcounts.find(q);
  if (it != g_profiling_refcounts.end()) return it->second;
  auto inserted = g_profiling_refcounts.emplace(
      q, std::make_shared<queue_profiling_refcount>());
  return inserted.first->second;
}

}  // namespace

hsa_status_t QueueProfilingAcquire(core::Queue* q) {
  if (q == nullptr) return HSA_STATUS_ERROR_INVALID_QUEUE;
  // Pin the entry alive across the per-entry lock acquire by copying the
  // shared_ptr out under g_profiling_refcounts_mu. Even if on_queue_destroy erases the
  // map slot between here and the lock_guard below, this local shared_ptr
  // keeps the queue_profiling_refcount object alive (spec §4a + lifetime
  // contract on g_profiling_refcounts). See C4 fix.
  std::shared_ptr<queue_profiling_refcount> owners;
  {
    std::lock_guard<std::mutex> lk(g_profiling_refcounts_mu);
    owners = get_or_create_owners(q);
  }
  std::lock_guard<std::mutex> lk(owners->m);
  const uint32_t prev = owners->refcount;
  if (prev == 0) {
    // 0->1 transition: enable profiling. AqlQueue::SetProfiling now also
    // allocates the per-queue dispatch-record buffer and pushes it to KFD
    // via hsaKmtSetQueueProfilingBuffer + Suspend/Resume (UPDATE_QUEUE).
    //
    // C3: propagate status. If SetProfiling fails (allocation failure or
    // libhsakmt registration failure on an unsupported kernel), do NOT
    // bump the refcount — the queue is not in the profiling-enabled
    // state and a subsequent release would be unbalanced.
    hsa_status_t s = q->SetProfiling(true);
    if (s != HSA_STATUS_SUCCESS) return s;
  }
  owners->refcount = prev + 1;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t QueueProfilingRelease(core::Queue* q) {
  if (q == nullptr) return HSA_STATUS_ERROR_INVALID_QUEUE;
  // See QueueProfilingAcquire for the shared_ptr lifetime rationale (C4).
  std::shared_ptr<queue_profiling_refcount> owners;
  {
    std::lock_guard<std::mutex> lk(g_profiling_refcounts_mu);
    auto it = g_profiling_refcounts.find(q);
    if (it == g_profiling_refcounts.end()) {
      // Underflow without ever acquiring. Caller bug; spec §4a says return
      // failure without modifying state.
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    owners = it->second;
  }
  std::lock_guard<std::mutex> lk(owners->m);
  if (owners->refcount == 0) {
    // Underflow: caller bug.
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  owners->refcount -= 1;
  if (owners->refcount == 0) {
    // 1->0 transition: AqlQueue::SetProfiling(false) frees the per-queue
    // dispatch-record buffer and clears the KFD profiling-buffer
    // registration via UPDATE_QUEUE.
    //
    // C3: propagate libhsakmt teardown status. The host-side buffer is
    // freed unconditionally (see SetProfiling), so we cannot recover the
    // resources by retrying — a non-success status here is informational
    // for the caller (e.g. test harnesses) and not a state-corruption
    // signal.
    return q->SetProfiling(false);
  }
  return HSA_STATUS_SUCCESS;
}

namespace {

// ============================================================================
// Per-queue ENABLE / DISABLE sequences (spec §5).
//
// Flow:
//   1. Skip non-AQL / non-GPU queues.
//   2. Skip if substrate is absent or agent is on the no-dispatch-log list.
//   3. QueueProfilingAcquire -> AqlQueue::SetProfiling(true) -> alloc
//      buffer + KFD UPDATE_QUEUE with the new trailing fields.
//   4. AqlQueue::GetProfilingDispatchRecords to read back the buffer base
//      and the FW-record-stride capacity in bytes (record_count * 16).
//      The substrate publishes no host-visible FW write pointer; consumers
//      locate fresh records via sentinel scan over per-slot record_type.
//   5. Register a non-owning queue_drain_state in g_active_drainers with
//      next_idx = 0 (the host-managed monotonic record cursor used by
//      the sentinel-scan drainer).
// ============================================================================

void enable_dispatch_log_for_queue_locked(core::Queue* q) {
  if (q == nullptr) return;

  // Idempotence guard: skip if already active.
  if (q->dispatch_log_active.load(std::memory_order_relaxed)) return;

  // Only AqlQueue (hardware AQL queues on GPU agents) have the FW
  // dispatch-record path. Host queues, soft queues, intercept queues
  // wrapping a non-AqlQueue: silently skip.
  if (!agent_is_gpu(q->GetAgent())) return;
  if (!AMD::AqlQueue::IsType(q)) return;
  auto* aql_queue = static_cast<AMD::AqlQueue*>(q);

  // Substrate / per-agent gate.
  if (!g_kfd_supports_dispatch_log.load(std::memory_order_relaxed)) return;
  if (g_no_dispatch_log_agents.count(q->GetAgent()) > 0) return;

  // Step 1: enable profiling on the queue. This is the AqlQueue::SetProfiling
  // call that allocates the dispatch-record buffer, calls
  // hsaKmtSetQueueProfilingBuffer, and Suspends/Resumes the queue to flush
  // the MQD via UPDATE_QUEUE. QueueProfilingAcquire takes g_profiling_refcounts_mu (not
  // g_queue_registry_mu), so it's safe to call while holding g_queue_registry_mu.
  hsa_status_t s = QueueProfilingAcquire(q);
  if (s != HSA_STATUS_SUCCESS) {
    // C5: distinguish substrate-absent from per-queue transient failures.
    //
    // Spec §5 (lines 604-609) only marks the agent as "no-dispatch-log"
    // when the failure indicates the substrate itself does not support
    // a dispatch-record buffer for queues on this agent — modeled as
    // HSA_STATUS_ERROR_NOT_SUPPORTED in the AqlQueue status surface
    // (KfdDriver returns NOT_SUPPORTED when the libhsakmt thunk symbol
    // is absent). For other failures (allocation OOM, transient KFD
    // errors), we just skip THIS queue/THIS tick and let later queues
    // and later ticks retry. Marking the whole agent on a transient
    // failure would permanently disable an otherwise-supported agent
    // from a one-off resource pressure event. The refcount was not
    // bumped (Acquire returns early on the 0->1 SetProfiling failure
    // path), so no Release is needed.
    const bool agent_unsupported = (s == HSA_STATUS_ERROR_NOT_SUPPORTED);
    std::fprintf(stderr,
                 "[hsa-runtime] dispatch_log: queue_id=%llu ENABLE step 1 "
                 "(QueueProfilingAcquire) failed status=%d%s\n",
                 static_cast<unsigned long long>(queue_id_of(q)),
                 static_cast<int>(s),
                 agent_unsupported ? "; marking agent as no-dispatch-log"
                                   : "; per-queue skip (transient)");
    if (agent_unsupported) {
      g_no_dispatch_log_agents.insert(q->GetAgent());
    }
    return;
  }

  // Step 2: read back the per-queue ring buffer info from AqlQueue.
  void* buf = nullptr;
  uint32_t buf_bytes = 0;
  hsa_status_t gs = aql_queue->GetProfilingDispatchRecords(&buf, &buf_bytes);
  if (gs != HSA_STATUS_SUCCESS || buf == nullptr || buf_bytes == 0) {
    std::fprintf(stderr,
                 "[hsa-runtime] dispatch_log: queue_id=%llu ENABLE step 2 "
                 "(GetProfilingDispatchRecords) failed status=%d\n",
                 static_cast<unsigned long long>(queue_id_of(q)),
                 static_cast<int>(gs));
    g_no_dispatch_log_agents.insert(q->GetAgent());
    QueueProfilingRelease(q);
    return;
  }

  // GetProfilingDispatchRecords reports buf_bytes = record_count * 16
  // (the FW record-stride sized capacity). The kernel-side BO is
  // oversized at count * 40 to satisfy host KFD ABI validation, but the
  // FW writes — and the drainer reads — at the 16-byte FW record stride
  // (kSlotStride). See dispatch_log.cpp:390 and the slot addressing at
  // dispatch_log.cpp:400-402.
  const uint32_t record_count = buf_bytes / static_cast<uint32_t>(sizeof(mec_dispatch_record_16));
  if (record_count == 0 || (record_count & (record_count - 1)) != 0) {
    std::fprintf(stderr,
                 "[hsa-runtime] dispatch_log: queue_id=%llu ENABLE step 2 "
                 "GetProfilingDispatchRecords returned non-pow2 record_count=%u\n",
                 static_cast<unsigned long long>(queue_id_of(q)), record_count);
    g_no_dispatch_log_agents.insert(q->GetAgent());
    QueueProfilingRelease(q);
    return;
  }

  // Steps 3-4: post-acquire setup. EVERYTHING that can throw between
  // QueueProfilingAcquire (above, which bumped the profiling refcount)
  // and the final dispatch_log_active.store(true) goes inside one
  // try block, so any throw rolls the profiling refcount back via
  // QueueProfilingRelease(q) and we exit cleanly. The next poller
  // tick will retry this queue.
  //
  // What can throw here:
  //   - std::make_shared<queue_drain_state>() (allocation)
  //   - g_active_drainers[qs->queue_id] = qs (unordered_map insert,
  //     can throw on rehash allocation)
  //   - std::thread(per_queue_drain_loop, qs) (std::system_error on
  //     EAGAIN / RLIMIT_NPROC / address-space exhaustion — review
  //     C1 stage-2 code-quality)
  //   - shared_ptr copies into per_queue_drain_loop's by-value
  //     parameter (allocation; happens inside std::thread ctor)
  //
  // Stack-size note: std::thread does not expose pthread_attr_setstacksize
  // portably, so each worker uses the default pthread stack (typically
  // 8 MiB virtual on glibc, lazily backed). With ~16 typical queues per
  // process this costs ~128 MiB of virtual address space, which is fine
  // for our hosts. Workloads pushing into the thousands-of-queues regime
  // would need a pthread_create-based spawn helper to set a small (e.g.
  // 64 KiB) stack; defer that until measured pressure exists.
  //
  // Order rationale (review C3, stage-2 code-quality): catch ANY
  // exception (not just std::system_error) so that allocation failures
  // before the std::thread spawn — std::make_shared, std::function
  // copy, unordered_map insert — also unwind cleanly. The previous
  // narrower catch left a refcount-leak window from
  // QueueProfilingAcquire success to the std::thread call.
  std::shared_ptr<queue_drain_state> qs;
  try {
    // Step 3: register in the drainer registry. NON-OWNING — buffer is
    // owned by AqlQueue, freed on SetProfiling(false) or AqlQueue dtor.
    qs = std::make_shared<queue_drain_state>();
    qs->queue_id         = queue_id_of(q);
    qs->ring_base        = buf;
    qs->ring_records     = record_count;
    qs->ring_mask        = record_count - 1;

    // Capture the Phase-2 host-VA pointer set if the new
    // KFD_IOC_PROFILER_DISPATCH_LOG path is active for this queue.
    // GetDispatchLogPointers returns NOT_INITIALIZED on older
    // kernels (MINOR<20) or older libhsakmt (no SetDispatchLog
    // thunk) — in that case the pointers stay null and the drainer
    // falls back to sentinel-scan automatically.
    volatile uint64_t* wptr_p   = nullptr;
    volatile uint64_t* rptr_p   = nullptr;
    volatile uint64_t* signal_p = nullptr;
    hsa_status_t dl_status =
        aql_queue->GetDispatchLogPointers(&wptr_p, &rptr_p, &signal_p);
    if (dl_status == HSA_STATUS_SUCCESS) {
      qs->wptr_ptr   = wptr_p;
      qs->rptr_ptr   = rptr_p;
      qs->signal_ptr = signal_p;
    }
    // else: legacy sentinel-scan mode; null pointers tell drain_one_queue
    // to use the fallback scan path.

    // next_idx starts at 0. The drain loop's initial-state sync (see
    // drain_one_queue Path A) detects the FW per-pipe scratch wptr
    // persistence on the FIRST observed signal advance and adjusts
    // next_idx then. We can't read FW's wptr at enable time because
    // FW won't publish to signal_addr until the first new record write
    // POST-enable — at this exact point the host word is still 0
    // (initialized by SetProfiling).
    qs->next_idx = 0;

    g_active_drainers[qs->queue_id] = qs;

    // Step 4: spawn the per-queue drainer worker AFTER registration.
    // The worker holds its own shared_ptr<queue_drain_state> by value
    // (passed into per_queue_drain_loop), so qs is alive for the full
    // thread lifetime even if the registry entry is erased before the
    // worker observes should_stop.
    qs->worker = std::thread(per_queue_drain_loop, qs);
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "[hsa-runtime] dispatch_log: queue_id=%llu ENABLE post-acquire "
                 "setup failed: %s; rolling back enable\n",
                 static_cast<unsigned long long>(queue_id_of(q)),
                 e.what());
    // Erase from registry if we got far enough to insert. erase() is
    // safe on a missing key (returns 0).
    g_active_drainers.erase(queue_id_of(q));
    // Release the profiling ref bumped by QueueProfilingAcquire above.
    // dispatch_log_active was never flipped to true, so no flag
    // rollback is needed. The next poller enable pass will retry.
    QueueProfilingRelease(q);
    return;
  }

  q->dispatch_log_active.store(true, std::memory_order_release);
}

void disable_dispatch_log_for_queue_locked(core::Queue* q) {
  if (q == nullptr) return;
  if (!q->dispatch_log_active.load(std::memory_order_acquire)) return;

  // Spec §4 disable-edge per-queue sequence.
  //
  // Per-queue drainer ordering: with one worker thread per queue (spawned
  // in enable_dispatch_log_for_queue_locked), the disable path must
  // signal stop and join the worker BEFORE freeing the buffer. The
  // worker does its own final drain (force_emit=true) inside
  // per_queue_drain_loop after observing should_stop, so a separate
  // synchronous-final-drain step is no longer needed.
  //
  // Pointer poisoning under drain_mu is retained as defense in depth
  // against any future code path that calls drain_one_queue
  // synchronously from another thread; today the worker is the sole
  // drainer for this queue and is already joined by the time we reach
  // that step, so the poisoning is a no-op-but-cheap belt-and-braces.

  // a. wait_for_idle (bounded ≤ 100 ms).
  wait_for_idle(q);

  // b. Look up the registered drain-state shared_ptr. After this we hold
  //    a local ref so qs survives even after the registry erase below.
  std::shared_ptr<queue_drain_state> qs_ptr;
  {
    auto it = g_active_drainers.find(queue_id_of(q));
    if (it != g_active_drainers.end()) qs_ptr = it->second;
  }

  // c. Stop and join the per-queue worker thread. The worker performs
  //    its own final drain (force_emit=true) after observing
  //    should_stop, so all pending FW records are emitted before
  //    join() returns. should_stop store uses release ordering;
  //    the worker reads it with acquire on each iteration.
  //
  //    join() may throw on a programming error (worker is the current
  //    thread, or the std::thread is not joinable). Both are
  //    impossible here: this function is called from the poller /
  //    on_queue_destroy / shutdown threads (never the worker), and
  //    the worker was assigned to qs->worker in the enable path
  //    immediately before flipping dispatch_log_active to true. If
  //    join() ever does throw the C++ runtime calls std::terminate;
  //    we have no recovery path beyond the diagnostic the runtime
  //    will emit. (No exception handler here — propagating up to
  //    the poller would not improve the situation.)
  if (qs_ptr) {
    qs_ptr->should_stop.store(true, std::memory_order_release);
    if (qs_ptr->worker.joinable()) qs_ptr->worker.join();
  }

  // d. Poison the non-owning pointers under drain_mu (defense in depth;
  //    see function-level comment). The worker is already joined by
  //    this point, so drain_mu is uncontended.
  if (qs_ptr) {
    std::lock_guard<std::mutex> lk(qs_ptr->drain_mu);
    qs_ptr->ring_base    = nullptr;
    qs_ptr->ring_records = 0;
    qs_ptr->ring_mask    = 0;
  }

  // e. Release profiling ref. On the 1->0 edge AqlQueue::SetProfiling(false)
  //    clears the KFD profiling-buffer registration (UPDATE_QUEUE with
  //    addr=0) and frees the per-queue dispatch-record buffer.
  //    QueueProfilingRelease takes g_profiling_refcounts_mu, NOT g_queue_registry_mu.
  //    By this point the worker is joined so no thread can dereference
  //    the buffer.
  //
  //    Order rationale (review C1, stage-1 spec-compliance): release
  //    BEFORE the registry erase. The release is the step that drives
  //    SetProfiling(false) on the underlying AqlQueue, which is the
  //    transition that makes the per-queue ring buffer cease to be
  //    a profiling buffer; once that has returned, the queue is fully
  //    quiesced from the dispatch_log perspective and erasing the
  //    registry slot is the final hand-off.
  QueueProfilingRelease(q);

  // f. Unregister from drainer registry. Drops the registry's shared_ptr
  //    ref. The worker's own shared_ptr (held by-value in
  //    per_queue_drain_loop) is already released because the worker
  //    has returned and joined, so this drops the last ref and the
  //    queue_drain_state destructor fires.
  g_active_drainers.erase(queue_id_of(q));

  q->dispatch_log_active.store(false, std::memory_order_release);
}

// ============================================================================
// ts_poller_loop (spec §4 lifecycle state machine).
// ============================================================================

bool predicate_now() {
#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST
  /* DIAGNOSTIC: log the three predicate inputs once on the first call
   * so we can see why dispatch_log is or isn't enabling. */
  static std::atomic<bool> diag_first{true};
  bool first_call = diag_first.exchange(false);
  bool trace_dis = rocm_trace_disabled();
  bool kfd_ok   = g_kfd_supports_dispatch_log.load(std::memory_order_relaxed);
  int  tp_en    = lttng_ust_tracepoint_enabled(rocm_hsa, kernel_dispatch_record);
  if (first_call) {
    std::fprintf(stderr,
                 "[hsa-runtime] dispatch_log DIAG predicate_now first-call: "
                 "trace_disabled=%d kfd_supports=%d tracepoint_enabled=%d\n",
                 (int)trace_dis, (int)kfd_ok, tp_en);
  }

  if (trace_dis) return false;
  if (!kfd_ok) {
    if (tp_en) {
      bool warned = g_kfd_unsupported_warned.load(std::memory_order_relaxed);
      if (!warned && g_kfd_unsupported_warned.compare_exchange_strong(
                         warned, true)) {
        std::fprintf(stderr,
                     "[hsa-runtime] dispatch_log: tracepoint "
                     "rocm_hsa:kernel_dispatch_record is enabled, but the "
                     "KFD substrate is not available on this kernel "
                     "(KFD minor version < 22). No events will be produced.\n");
      }
    }
    return false;
  }
  return tp_en != 0;
#else
  return false;
#endif
}

void ts_poller_loop() {
  const auto tick_interval = std::chrono::milliseconds(5);

  /* DIAGNOSTIC: confirm the poller actually starts, and log every
   * predicate-result transition so we can see when (or if) it ever
   * flips true. Steady-state silent. */
  std::fprintf(stderr,
               "[hsa-runtime] dispatch_log DIAG ts_poller_loop entered\n");
  bool last_curr = false;
  bool first_iter = true;

  while (!g_shutdown.load(std::memory_order_relaxed)) {
    const bool curr = predicate_now();
    g_dispatch_logging_active.store(curr, std::memory_order_relaxed);

    if (first_iter || curr != last_curr) {
      std::fprintf(stderr,
                   "[hsa-runtime] dispatch_log DIAG ts_poller_loop predicate=%d "
                   "(was %d)\n",
                   (int)curr, (int)last_curr);
      last_curr = curr;
      first_iter = false;
    }

    if (curr) {
      // ENABLE pass (spec §4): for each known live queue Q with
      // dispatch_log_active==false, run the per-queue ENABLE sequence.
      // Each enable spawns a per-queue drainer worker thread inside
      // enable_dispatch_log_for_queue_locked; there is no longer any
      // shared drainer to start.
      for_each_known_queue_locked(
          [](uint64_t /*qid*/, core::Queue* q) {
            enable_dispatch_log_for_queue_locked(q);
          });
    } else {
      // DISABLE pass (spec §4 + §9 ROCM_LTTNG_UST_DISABLE row).
      for_each_known_queue_locked(
          [](uint64_t /*qid*/, core::Queue* q) {
            disable_dispatch_log_for_queue_locked(q);
          });
    }

    std::this_thread::sleep_for(tick_interval);
  }
}

}  // namespace

// ============================================================================
// Public API.
// ============================================================================

void init() {
  // KFD substrate version probe via AMDKFD_IOC_GET_VERSION. We require
  // KFD minor >= 22 (the version that introduced the
  // dispatch_record_buffer_{addr,size} trailing fields on UPDATE_QUEUE).
  const bool present = probe_substrate_version();
  g_kfd_supports_dispatch_log.store(present, std::memory_order_release);

  // Spawn poller unconditionally so the steady-state idle path is exercised
  // even when the substrate is missing. The poller's enable branch
  // short-circuits via predicate_now() -> false in that case.
  g_shutdown.store(false, std::memory_order_relaxed);
  g_poller_thread = std::thread(ts_poller_loop);
}

void shutdown() {
  g_shutdown.store(true, std::memory_order_release);

  // Join the poller first so no new per-queue workers can be spawned (the
  // poller's enable pass is the only spawner) while we are tearing down
  // the active set.
  if (g_poller_thread.joinable()) g_poller_thread.join();

  // Spec §4 process-exit cleanup: run the per-queue DISABLE sequence for
  // every queue still active at process exit. disable_dispatch_log_for_
  // queue_locked stops + joins each queue's worker thread before
  // releasing its profiling buffer, so by the time these calls return
  // every per-queue worker has finished its final drain.
  for_each_known_queue_locked(
      [](uint64_t /*qid*/, core::Queue* q) {
        disable_dispatch_log_for_queue_locked(q);
      });

  // Drop any drain-state shared_ptrs still pinned by g_active_drainers.
  // The destructor on queue_drain_state defensively joins worker if it
  // is still joinable (it should not be — disable above joined it
  // already), so this is safe even if a queue somehow escaped the
  // disable pass above.
  {
    std::lock_guard<std::mutex> lk(g_queue_registry_mu);
    g_all_queues.clear();
    g_active_drainers.clear();
    g_no_dispatch_log_agents.clear();
  }
  {
    std::lock_guard<std::mutex> lk(g_profiling_refcounts_mu);
    g_profiling_refcounts.clear();
  }
}

void on_queue_create(core::Queue* q) {
  if (q == nullptr) return;

  // Register the queue in the poller's live-queue side map BEFORE running
  // the enable sequence. Holding g_queue_registry_mu across both the registry
  // insert and the enable serializes us against the poller (which iterates
  // g_all_queues under the same mutex) and against on_queue_destroy.
  std::lock_guard<std::mutex> lk(g_queue_registry_mu);
  g_all_queues[queue_id_of(q)] = q;
  if (g_dispatch_logging_active.load(std::memory_order_relaxed)) {
    enable_dispatch_log_for_queue_locked(q);
  }
}

void on_queue_destroy(core::Queue* q) {
  if (q == nullptr) return;

  // Remove from g_all_queues BEFORE the disable sequence so the poller
  // can never observe a Queue* whose underlying core::Queue is being torn
  // down. Held across the disable so the poller's iterating thread is
  // forced to wait behind the mutex if it currently holds it.
  {
    std::lock_guard<std::mutex> lk(g_queue_registry_mu);
    g_all_queues.erase(queue_id_of(q));
    if (q->dispatch_log_active.load(std::memory_order_acquire)) {
      disable_dispatch_log_for_queue_locked(q);
    }
  }

  // Drop the per-queue refcount entry. Safe to do unconditionally — if the
  // queue never had an acquire, the lookup is just a miss.
  {
    std::lock_guard<std::mutex> lk(g_profiling_refcounts_mu);
    g_profiling_refcounts.erase(q);
  }
}

}  // namespace dispatch_log
}  // namespace rocr

// =============================================================================
// TEST-ONLY introspection API for standalone dispatch_log validation programs
// (dlog_test1, dlog_test2, ...). NOT part of the public HSA ABI; symbol prefix
// hsa_amd_dispatch_log_test_ to make grep-ability obvious.
//
// Usage from a test:
//   1. set HSA_DISPATCH_LOG_NO_DRAIN=1 in the env so the per-queue worker
//      sleeps instead of draining (preserves raw FW buffer state for the
//      test to inspect).
//   2. register a queue-create callback via
//      hsa_amd_runtime_queue_create_register
//   3. in the callback, call hsa_amd_dispatch_log_test_enable(queue) which
//      invokes AqlQueue::SetProfiling(true), allocating the dispatch record
//      buffer + host-VA wptr/rptr/signal words and calling SetDispatchLog.
//   4. submit kernels normally, sync, sleep briefly to let FW finish writes.
//   5. call hsa_amd_dispatch_log_test_get_state(queue, ...) to retrieve
//      buffer base + record count + signal pointer for direct verification.
// =============================================================================

extern "C" {

// Force-enable dispatch_log on the given queue. Calls AqlQueue::SetProfiling
// (true) which allocates the dispatch record buffer and registers the host-VA
// pointer set via the new KFD_IOC_PROFILER_DISPATCH_LOG sub-op.
__attribute__((visibility("default")))
hsa_status_t hsa_amd_dispatch_log_test_enable(hsa_queue_t* queue) {
  if (queue == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  rocr::core::Queue* q = rocr::core::Queue::Convert(queue);
  if (q == nullptr) return HSA_STATUS_ERROR_INVALID_QUEUE;
  if (!rocr::AMD::AqlQueue::IsType(q)) return HSA_STATUS_ERROR_INVALID_QUEUE;
  auto* aq = static_cast<rocr::AMD::AqlQueue*>(q);
  return aq->SetProfiling(true);
}

// Return the dispatch_log buffer base + record count + host-VA wptr / signal
// pointers for the given queue. Only valid AFTER hsa_amd_dispatch_log_test_enable
// (or any other path that has called SetProfiling(true) with the new path
// active).
__attribute__((visibility("default")))
hsa_status_t hsa_amd_dispatch_log_test_get_state(
    hsa_queue_t* queue,
    void** buffer_base,
    uint32_t* num_records,
    const volatile uint64_t** wptr_ptr,
    const volatile uint64_t** signal_ptr) {
  if (queue == nullptr || buffer_base == nullptr || num_records == nullptr ||
      wptr_ptr == nullptr || signal_ptr == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  rocr::core::Queue* q = rocr::core::Queue::Convert(queue);
  if (q == nullptr) return HSA_STATUS_ERROR_INVALID_QUEUE;
  if (!rocr::AMD::AqlQueue::IsType(q)) return HSA_STATUS_ERROR_INVALID_QUEUE;
  auto* aq = static_cast<rocr::AMD::AqlQueue*>(q);

  uint32_t buf_bytes = 0;
  hsa_status_t s = aq->GetProfilingDispatchRecords(buffer_base, &buf_bytes);
  if (s != HSA_STATUS_SUCCESS) return s;
  // FW writes 16-byte records. GetProfilingDispatchRecords reports total bytes
  // available for FW writes (records * 16); convert back to record count.
  *num_records = buf_bytes / 16u;

  volatile uint64_t* w = nullptr;
  volatile uint64_t* r = nullptr;
  volatile uint64_t* sig = nullptr;
  s = aq->GetDispatchLogPointers(&w, &r, &sig);
  if (s != HSA_STATUS_SUCCESS) return s;
  *wptr_ptr   = w;
  *signal_ptr = sig;
  return HSA_STATUS_SUCCESS;
}

}  // extern "C"

namespace rocr {
namespace dispatch_log {

// Trailing namespace re-open so clang-format doesn't complain about the file
// not ending inside a namespace; the body is intentionally empty.

}  // namespace dispatch_log
}  // namespace rocr
