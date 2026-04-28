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
//   - The ts_drainer data-plane thread (adaptive cadence, snapshot-based
//     drain per spec §7).
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

#include "core/inc/dispatch_log.h"

#include <cerrno>
#include <chrono>
#include <condition_variable>
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
#include <vector>

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
// keyed in g_owners; the mutex serializes the refcount RMW + the bit set/clear
// transition. Phase A uses a side-map keyed on core::Queue* rather than a
// dl_state pointer hung off the Queue itself; the in-Queue void*
// dispatch_log_state is reserved for a future refactor.
// ============================================================================
struct queue_profiling_owners {
  std::mutex m;
  uint32_t   refcount = 0;
};

// ============================================================================
// queue_drain_state per spec §7. Held via std::shared_ptr from both the
// drainer registry (g_active_queues) and per-pass snapshot vectors. Stores
// only NON-OWNING pointers into the buffer owned by AqlQueue (the buffer
// is alloc'd by AqlQueue::SetProfiling(true) and freed by SetProfiling(false)
// or the AqlQueue dtor). Deliberately stores NO Queue* — see spec §4
// "Queue / drain-state lifetime" for the dangling-Queue rationale.
// ============================================================================
struct queue_drain_state {
  // Full 64-bit hsa_queue_t::id (see hsa.h). Internal ownership / lifetime
  // maps key on this full id to avoid the high-32-bit collision risk
  // flagged in C5. The on-the-wire LTTng tracepoint payload still carries
  // only the low 32 bits per spec §6/§14 — we narrow at the
  // rocm_trace_emit_* call sites, not here.
  uint64_t  queue_id = 0;

  // Owned, queue-independent GPU->system time-translation callable.
  // Captured at register time so it remains callable even after the
  // underlying core::Queue (and any queue-scoped state on the agent) has
  // been destroyed. The agent itself outlives any queue created on it
  // (HSA agents have process lifetime), so capturing the agent reference
  // by value inside this callable is safe.
  std::function<uint64_t(uint64_t /* gpu_ts */)> translate_gpu_ts;

  // NON-OWNING ring view. The buffer is owned by AqlQueue::dispatch_record_buffer_;
  // do not free here.
  void*    ring_base    = nullptr;     // host-virtual base of the FW record area
  uint32_t record_count = 0;           // power-of-2 number of 16-byte slots
  uint32_t record_mask  = 0;           // record_count - 1, for slot indexing

  // FW-written 32-bit monotonic record counter. Lives outside the buffer
  // (cpc_tracing infrastructure exposes it as a separate uint32_t* via
  // hsa_amd_profiling_get_dispatch_records). Pointer is owned by AqlQueue.
  volatile uint32_t* fw_wptr_records = nullptr;

  // Mutated only under drain_mu (see below). Host-extended 64-bit version
  // of the FW record cursor; we extend the FW's uint32_t monotonic record
  // counter into a uint64_t so wraparound across the 32-bit boundary doesn't
  // confuse our overrun bookkeeping.
  uint64_t read_record_cursor = 0;
  std::unordered_map<uint32_t, uint64_t> pending_starts;

  // Per-queue drain mutex. Serializes drain_one_queue() (steady state) and
  // ts_drainer_drain_now() (synchronous final drain on the destroying /
  // disable-edge thread). Per-queue scope, so other queues' drains are not
  // blocked.
  std::mutex drain_mu;
};

// ============================================================================
// Global state.
// ============================================================================

// Spec §4 single source of truth for "should HSA be collecting kernel-dispatch
// timestamps right now?". Folded predicate of !rocm_trace_disabled() &&
// lttng_ust_tracepoint_enabled(rocm_hsa, kernel_dispatch_complete) &&
// g_substrate_present. Written only by ts_poller; read by on_queue_create
// (relaxed, hot path) and ts_drainer.
std::atomic<bool> G_tracepoint_enabled{false};

// Set true if the KFD substrate version probe at init() reports a kernel
// minor version >= 22 (the minor version that introduced the
// dispatch_record_buffer_{addr,size} trailing fields on UPDATE_QUEUE).
// Probed once at init via AMDKFD_IOC_GET_VERSION; never mutated afterwards.
std::atomic<bool> g_substrate_present{false};

// Latched true after we log the "substrate absent, tracepoint enable
// is a no-op" message once. Avoids spamming the user log on every poll tick.
std::atomic<bool> g_substrate_absent_warned{false};

// Driven by shutdown(); poller and drainer observe and exit.
std::atomic<bool> g_shutdown{false};

// Spec §4 lifecycle mutex. Guards g_active_queues + g_no_dispatch_log_agents.
// Acquired only briefly: enable/disable per-queue and snapshot-cloning.
std::mutex g_lifecycle_mu;

// Drainer registry (spec §4 lifetime protocol). Keyed by the full 64-bit
// hsa_queue_t::id (the value FW writes into the record dispatch_idx is
// per-queue, so queue_id distinguishes drains). Holds one shared_ptr per
// active queue. See C5 fix: the LTTng tracepoint payload narrows this to
// uint32_t at emit, but internal lifetime keys must stay 64-bit because
// hsa_queue_t::id is uint64_t and unique over the application's lifetime.
std::unordered_map<uint64_t, std::shared_ptr<queue_drain_state>> g_active_queues;

// Poller-only "live queue" side map (spec §4 enable/disable convergence).
//
// Distinct from g_active_queues: this tracks every Queue* known to the
// dispatch_log subsystem (registered by on_queue_create, removed by
// on_queue_destroy), regardless of whether dispatch logging is currently
// enabled on it. The poller iterates this set under g_lifecycle_mu to drive
// the §4 idempotent enable/disable passes.
//
// Keyed by full 64-bit hsa_queue_t::id (see g_active_queues commentary
// above and C5 fix).
std::unordered_map<uint64_t, core::Queue*> g_known_queues;

// Side-map for the QueueProfilingAcquire/Release refcount (spec §4a). One
// entry per Queue* that has ever had at least one acquire. Looked up under
// g_owners_mu.
//
// **Lifetime contract.** Entries are held via std::shared_ptr because
// QueueProfilingAcquire / QueueProfilingRelease intentionally drop
// g_owners_mu before locking the per-entry mutex (the per-entry lock can
// be held across SetProfiling(), which we don't want to do under the
// global map mutex). Without shared ownership, on_queue_destroy could
// erase the map entry between our map-lookup and our per-entry-lock,
// freeing the queue_profiling_owners object out from under us. Copying
// the shared_ptr while still holding g_owners_mu pins the entry alive
// until the API call's local shared_ptr drops, even if on_queue_destroy
// concurrently erases the map slot.
std::mutex g_owners_mu;
std::unordered_map<core::Queue*, std::shared_ptr<queue_profiling_owners>> g_owners;

// Per-agent "no-dispatch-log" mark. If GetProfilingDispatchRecords reports
// NOT_INITIALIZED (or another error) for any queue on a given agent, that
// agent is added here and subsequent enable_dispatch_log_for_queue calls on
// queues belonging to it short-circuit (spec §5 unsupported-agent row,
// §9 unsupported-agent row).
std::unordered_set<core::Agent*> g_no_dispatch_log_agents;

// Threads.
std::thread g_poller_thread;
std::thread g_drainer_thread;
std::atomic<bool> g_drainer_running{false};

// Drainer wakeup CV. Used to long-sleep (50 ms) when G_tracepoint_enabled is
// false and short-sleep (500 us) on idle. Notified on shutdown.
std::mutex g_drainer_mu;
std::condition_variable g_drainer_cv;

// ============================================================================
// Helpers.
// ============================================================================

// Full 64-bit hsa_queue_t::id (see hsa.h:2359-2361). Internal registries
// (g_active_queues, g_known_queues) and queue_drain_state.queue_id all key
// on this full value to avoid the high-32-bit collision risk flagged in
// C5. The LTTng tracepoint payload narrows to the low 32 bits at the
// emit call sites only.
uint64_t queue_id_of(core::Queue* q) {
  // amd_queue_t.hsa_queue.id is set in the queue ctor (see e.g.
  // amd_aql_queue.cpp:298, host_queue.cpp:77).
  return q->amd_queue_.hsa_queue.id;
}

// Narrow the 64-bit internal queue_id to the 32-bit on-the-wire identifier
// emitted by the rocm_hsa:kernel_dispatch_complete / kernel_dispatch_drop
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

// Build the GPU->system translation callable from the Queue's owning agent.
// Captures the agent pointer by value; agents have process lifetime so this
// is safe even after the queue is destroyed (spec §7).
std::function<uint64_t(uint64_t)> make_translate_gpu_ts(core::Queue* q) {
  core::Agent* agent = q->GetAgent();
  if (!agent_is_gpu(agent)) {
    // Non-GPU (CPU / soft) queue. Pass-through; we shouldn't be enabling on
    // these at all, but be defensive.
    return [](uint64_t t) { return t; };
  }
  auto* gpu = static_cast<AMD::GpuAgentInt*>(agent);
  return [gpu](uint64_t gpu_ts) -> uint64_t {
    return gpu->TranslateTime(gpu_ts);
  };
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
  return args.major_version >= 1 && args.minor_version >= 22;
}
#else
bool probe_substrate_version() { return false; }
#endif

// Snapshot: clone shared_ptrs out of g_active_queues under the lifecycle
// mutex, return the vector. Drainer then iterates WITHOUT holding the
// mutex (this is the only place where snapshotting matters: the drainer
// runs on the data-plane hot path and must not block lifecycle ops, and
// shared_ptr ownership keeps the entries alive across the unlocked loop).
std::vector<std::shared_ptr<queue_drain_state>> snapshot_active_queues() {
  std::vector<std::shared_ptr<queue_drain_state>> out;
  std::lock_guard<std::mutex> lk(g_lifecycle_mu);
  out.reserve(g_active_queues.size());
  for (auto& kv : g_active_queues) out.push_back(kv.second);
  return out;
}

// Iterate g_known_queues under g_lifecycle_mu and invoke fn(queue_id, queue)
// for each entry. C6 fix: replaces the previous "snapshot into vector then
// iterate" pattern. fn must NOT mutate g_known_queues. fn IS allowed to
// mutate g_active_queues (e.g. enable_dispatch_log_for_queue_locked inserts,
// disable_dispatch_log_for_queue_locked erases).
template <typename F>
void for_each_known_queue_locked(F&& fn) {
  std::lock_guard<std::mutex> lk(g_lifecycle_mu);
  for (auto& kv : g_known_queues) fn(kv.first, kv.second);
}

// ============================================================================
// drain_one_queue (spec §7). Holds qs.drain_mu for the entire pass.
//
// Reads the FW-written uint32_t record counter, host-extends it to 64 bits
// (so wrap of the FW counter doesn't confuse us across drain calls), then
// walks new records from read_record_cursor up to the host-extended fw cursor.
// Each record is 16 bytes; the slot index is (cursor & record_mask).
// ============================================================================

bool drain_one_queue(queue_drain_state& qs, bool force_emit) {
  std::lock_guard<std::mutex> lk(qs.drain_mu);

  if (qs.fw_wptr_records == nullptr || qs.ring_base == nullptr ||
      qs.record_count == 0) {
    return false;
  }

  // Snapshot the FW's 32-bit monotonic record counter and host-extend it.
  // The high 32 bits we add are taken from our last-known cursor: as long
  // as we drain at least once per (2^32) records, this extension is correct.
  // 65536 records buffered, so we'd lose data well before wrap-extension
  // ambiguity matters.
  const uint32_t fw_lo = __atomic_load_n(qs.fw_wptr_records, __ATOMIC_ACQUIRE);
  uint64_t fw_extended = (qs.read_record_cursor & ~uint64_t(0xFFFFFFFFu)) | fw_lo;
  if (fw_extended < qs.read_record_cursor) {
    // Low 32 bits wrapped past us; bump the high half.
    fw_extended += uint64_t(1) << 32;
  }
  if (fw_extended == qs.read_record_cursor) return false;

  // Wraparound check (spec §6 bytes_lost field definition). With 65536
  // records, an overrun means FW outpaced us by more than the buffer
  // depth.
  const uint64_t records_avail = fw_extended - qs.read_record_cursor;
  if (records_avail > qs.record_count) {
    // Narrow internal 64-bit queue_id to the 32-bit on-the-wire id at the
    // tracepoint boundary (spec §6 / §14, C5 fix). Report the loss in
    // bytes for compatibility with the existing tracepoint payload.
    rocm_trace_emit_hsa_kernel_dispatch_drop(
        queue_id_to_wire(qs.queue_id),
        records_avail * sizeof(mec_dispatch_record_16));
    qs.read_record_cursor = fw_extended - qs.record_count;

    // C7 fix: across an overrun event, ALL previously cached STARTs are
    // suspect. The drainer cannot tell which records the FW just overwrote
    // (spec §7 / drop tracepoint semantics). Conservative: drop them.
    qs.pending_starts.clear();
  }

  while (qs.read_record_cursor < fw_extended) {
    const uint32_t slot = static_cast<uint32_t>(qs.read_record_cursor & qs.record_mask);
    const auto* rec = reinterpret_cast<const mec_dispatch_record_16*>(
        static_cast<const char*>(qs.ring_base) + slot * sizeof(mec_dispatch_record_16));

    const uint64_t gpu_ts =
        (static_cast<uint64_t>(rec->ts_hi) << 32) | rec->ts_lo;

    if (rec->record_type == DISPATCH_LOG_RECORD_START) {
      qs.pending_starts[rec->dispatch_idx] = gpu_ts;
    } else if (rec->record_type == DISPATCH_LOG_RECORD_END) {
      auto it = qs.pending_starts.find(rec->dispatch_idx);
      if (it != qs.pending_starts.end()) {
        const uint64_t start_gpu = it->second;
        const uint32_t didx = rec->dispatch_idx;
        qs.pending_starts.erase(it);

        // Use the captured queue-independent translation callable. Spec §7
        // explicitly forbids reaching through a Queue* here — Queue may be
        // destroyed mid-pass even though the shared_ptr keeps qs alive.
        const uint64_t start_sys = qs.translate_gpu_ts(start_gpu);
        const uint64_t end_sys   = qs.translate_gpu_ts(gpu_ts);

        // Narrow internal 64-bit queue_id at the tracepoint boundary
        // (spec §6 / §14, C5 fix). dispatch_idx is already 32-bit per the
        // FW record contract (mec_dispatch_record_16::dispatch_idx).
        rocm_trace_emit_hsa_kernel_dispatch_complete(
            queue_id_to_wire(qs.queue_id), didx, start_sys, end_sys,
            force_emit);
      }
      // else: orphan END (start was lost to wraparound). Silently skip.
    }

    qs.read_record_cursor += 1;
  }
  return true;
}

// Synchronous final drain on the calling thread. Looks up the queue's
// shared_ptr from the registry and forwards to drain_one_queue with
// force_emit=true. Per spec §4: serialization with the drainer thread
// happens via qs.drain_mu inside drain_one_queue.
//
// _locked variant: caller holds g_lifecycle_mu. Used by the poller and by
// disable_dispatch_log_for_queue_locked, both of which iterate registries
// under the lifecycle mutex.
void ts_drainer_drain_now_locked(core::Queue* q) {
  std::shared_ptr<queue_drain_state> qs_ptr;
  auto it = g_active_queues.find(queue_id_of(q));
  if (it == g_active_queues.end()) return;
  qs_ptr = it->second;
  if (qs_ptr) {
    drain_one_queue(*qs_ptr, /* force_emit = */ true);
  }
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

std::shared_ptr<queue_profiling_owners> get_or_create_owners(core::Queue* q) {
  // Caller holds g_owners_mu.
  auto it = g_owners.find(q);
  if (it != g_owners.end()) return it->second;
  auto inserted = g_owners.emplace(
      q, std::make_shared<queue_profiling_owners>());
  return inserted.first->second;
}

}  // namespace

hsa_status_t QueueProfilingAcquire(core::Queue* q) {
  if (q == nullptr) return HSA_STATUS_ERROR_INVALID_QUEUE;
  // Pin the entry alive across the per-entry lock acquire by copying the
  // shared_ptr out under g_owners_mu. Even if on_queue_destroy erases the
  // map slot between here and the lock_guard below, this local shared_ptr
  // keeps the queue_profiling_owners object alive (spec §4a + lifetime
  // contract on g_owners). See C4 fix.
  std::shared_ptr<queue_profiling_owners> owners;
  {
    std::lock_guard<std::mutex> lk(g_owners_mu);
    owners = get_or_create_owners(q);
  }
  std::lock_guard<std::mutex> lk(owners->m);
  const uint32_t prev = owners->refcount;
  owners->refcount = prev + 1;
  if (prev == 0) {
    // 0->1 transition: enable profiling. AqlQueue::SetProfiling now also
    // allocates the per-queue dispatch-record buffer and pushes it to KFD
    // via hsaKmtSetQueueProfilingBuffer + Suspend/Resume (UPDATE_QUEUE).
    q->SetProfiling(true);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t QueueProfilingRelease(core::Queue* q) {
  if (q == nullptr) return HSA_STATUS_ERROR_INVALID_QUEUE;
  // See QueueProfilingAcquire for the shared_ptr lifetime rationale (C4).
  std::shared_ptr<queue_profiling_owners> owners;
  {
    std::lock_guard<std::mutex> lk(g_owners_mu);
    auto it = g_owners.find(q);
    if (it == g_owners.end()) {
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
    q->SetProfiling(false);
  }
  return HSA_STATUS_SUCCESS;
}

namespace {

// ============================================================================
// Per-queue ENABLE / DISABLE sequences (spec §5).
//
// New flow (post cpc_tracing infrastructure import):
//   1. Skip non-AQL / non-GPU queues.
//   2. Skip if substrate is absent or agent is on the no-dispatch-log list.
//   3. QueueProfilingAcquire -> AqlQueue::SetProfiling(true) -> alloc
//      buffer + KFD UPDATE_QUEUE with the new trailing fields.
//   4. AqlQueue::GetProfilingDispatchRecords to read back the buffer base,
//      total bytes, and FW wptr pointer.
//   5. Register a non-owning queue_drain_state in g_active_queues.
//
// The previous Phase A path's bespoke buffer alloc + placeholder KFD ioctl
// has been removed entirely.
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
  if (!g_substrate_present.load(std::memory_order_relaxed)) return;
  if (g_no_dispatch_log_agents.count(q->GetAgent()) > 0) return;

  // Step 1: enable profiling on the queue. This is the AqlQueue::SetProfiling
  // call that allocates the dispatch-record buffer, calls
  // hsaKmtSetQueueProfilingBuffer, and Suspends/Resumes the queue to flush
  // the MQD via UPDATE_QUEUE. QueueProfilingAcquire takes g_owners_mu (not
  // g_lifecycle_mu), so it's safe to call while holding g_lifecycle_mu.
  hsa_status_t s = QueueProfilingAcquire(q);
  if (s != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr,
                 "[hsa-runtime] dispatch_log: queue_id=%llu ENABLE step 1 "
                 "(QueueProfilingAcquire) failed status=%d\n",
                 static_cast<unsigned long long>(queue_id_of(q)),
                 static_cast<int>(s));
    return;
  }

  // Step 2: read back the per-queue ring buffer info from AqlQueue.
  void* buf = nullptr;
  uint32_t buf_bytes = 0;
  volatile uint32_t* fw_wptr = nullptr;
  hsa_status_t gs = aql_queue->GetProfilingDispatchRecords(&buf, &buf_bytes, &fw_wptr);
  if (gs != HSA_STATUS_SUCCESS || buf == nullptr || buf_bytes == 0 || fw_wptr == nullptr) {
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
  // oversized (count * 40) per the host KFD ABI; the drainer iterates
  // 16-byte records.
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

  // Step 3: register in the drainer registry. NON-OWNING — buffer is
  // owned by AqlQueue, freed on SetProfiling(false) or AqlQueue dtor.
  auto qs = std::make_shared<queue_drain_state>();
  qs->queue_id           = queue_id_of(q);
  qs->translate_gpu_ts   = make_translate_gpu_ts(q);
  qs->ring_base          = buf;
  qs->record_count       = record_count;
  qs->record_mask        = record_count - 1;
  qs->fw_wptr_records    = fw_wptr;
  qs->read_record_cursor = 0;

  g_active_queues[qs->queue_id] = qs;

  q->dispatch_log_active.store(true, std::memory_order_release);
}

void disable_dispatch_log_for_queue_locked(core::Queue* q) {
  if (q == nullptr) return;
  if (!q->dispatch_log_active.load(std::memory_order_acquire)) return;

  // Spec §4 disable-edge per-queue sequence.

  // a. wait_for_idle (bounded ≤ 100 ms).
  wait_for_idle(q);

  // b. Final drain with force_emit=true (spec §6).
  ts_drainer_drain_now_locked(q);

  // c. Release profiling ref. On the 1->0 edge AqlQueue::SetProfiling(false)
  // clears the KFD profiling-buffer registration (UPDATE_QUEUE with addr=0)
  // and frees the per-queue dispatch-record buffer. QueueProfilingRelease
  // takes g_owners_mu, NOT g_lifecycle_mu.
  QueueProfilingRelease(q);

  // d. Unregister from drainer registry. Drops the registry's shared_ptr
  //    ref. The buffer is owned by AqlQueue (we just released it via
  //    SetProfiling(false)); our queue_drain_state holds only non-owning
  //    pointers, so the destructor is a no-op for buffer memory.
  g_active_queues.erase(queue_id_of(q));

  q->dispatch_log_active.store(false, std::memory_order_release);
}

// ============================================================================
// ts_drainer_loop (spec §7).
// ============================================================================

void ts_drainer_loop() {
  while (!g_shutdown.load(std::memory_order_relaxed)) {
    if (!G_tracepoint_enabled.load(std::memory_order_relaxed)) {
      std::unique_lock<std::mutex> lk(g_drainer_mu);
      g_drainer_cv.wait_for(lk, std::chrono::milliseconds(50));
      continue;
    }

    bool any_progress = false;
    auto snapshot = snapshot_active_queues();
    for (auto& qs_ptr : snapshot) {
      any_progress |= drain_one_queue(*qs_ptr, /* force_emit = */ false);
    }

    if (any_progress) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
  }
}

void start_drainer_if_needed() {
  bool expected = false;
  if (!g_drainer_running.compare_exchange_strong(expected, true)) return;
  g_drainer_thread = std::thread(ts_drainer_loop);
}

// ============================================================================
// ts_poller_loop (spec §4 lifecycle state machine).
// ============================================================================

bool predicate_now() {
#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST
  if (rocm_trace_disabled()) return false;
  if (!g_substrate_present.load(std::memory_order_relaxed)) {
    if (lttng_ust_tracepoint_enabled(rocm_hsa, kernel_dispatch_complete)) {
      bool warned = g_substrate_absent_warned.load(std::memory_order_relaxed);
      if (!warned && g_substrate_absent_warned.compare_exchange_strong(
                         warned, true)) {
        std::fprintf(stderr,
                     "[hsa-runtime] dispatch_log: tracepoint "
                     "rocm_hsa:kernel_dispatch_complete is enabled, but the "
                     "KFD substrate is not available on this kernel "
                     "(KFD minor version < 22). No events will be produced.\n");
      }
    }
    return false;
  }
  return lttng_ust_tracepoint_enabled(rocm_hsa, kernel_dispatch_complete) != 0;
#else
  return false;
#endif
}

void ts_poller_loop() {
  const auto tick_interval = std::chrono::milliseconds(5);

  while (!g_shutdown.load(std::memory_order_relaxed)) {
    const bool curr = predicate_now();
    G_tracepoint_enabled.store(curr, std::memory_order_relaxed);

    if (curr) {
      start_drainer_if_needed();

      // ENABLE pass (spec §4): for each known live queue Q with
      // dispatch_log_active==false, run the per-queue ENABLE sequence.
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
  g_substrate_present.store(present, std::memory_order_release);

  // Spawn poller unconditionally so the steady-state idle path is exercised
  // even when the substrate is missing. The poller's enable branch
  // short-circuits via predicate_now() -> false in that case.
  g_shutdown.store(false, std::memory_order_relaxed);
  g_poller_thread = std::thread(ts_poller_loop);
}

void shutdown() {
  g_shutdown.store(true, std::memory_order_release);
  g_drainer_cv.notify_all();

  if (g_poller_thread.joinable()) g_poller_thread.join();
  if (g_drainer_thread.joinable()) g_drainer_thread.join();
  g_drainer_running.store(false, std::memory_order_relaxed);

  // Spec §4 process-exit cleanup: run the per-queue DISABLE sequence
  // (wait_for_idle, final drain, QueueProfilingRelease) for every queue
  // still active at process exit.
  for_each_known_queue_locked(
      [](uint64_t /*qid*/, core::Queue* q) {
        disable_dispatch_log_for_queue_locked(q);
      });
  {
    std::lock_guard<std::mutex> lk(g_lifecycle_mu);
    g_known_queues.clear();
    g_active_queues.clear();
    g_no_dispatch_log_agents.clear();
  }
  {
    std::lock_guard<std::mutex> lk(g_owners_mu);
    g_owners.clear();
  }
}

void on_queue_create(core::Queue* q) {
  if (q == nullptr) return;

  // Register the queue in the poller's live-queue side map BEFORE running
  // the enable sequence. Holding g_lifecycle_mu across both the registry
  // insert and the enable serializes us against the poller (which iterates
  // g_known_queues under the same mutex) and against on_queue_destroy.
  std::lock_guard<std::mutex> lk(g_lifecycle_mu);
  g_known_queues[queue_id_of(q)] = q;
  if (G_tracepoint_enabled.load(std::memory_order_relaxed)) {
    enable_dispatch_log_for_queue_locked(q);
  }
}

void on_queue_destroy(core::Queue* q) {
  if (q == nullptr) return;

  // Remove from g_known_queues BEFORE the disable sequence so the poller
  // can never observe a Queue* whose underlying core::Queue is being torn
  // down. Held across the disable so the poller's iterating thread is
  // forced to wait behind the mutex if it currently holds it.
  {
    std::lock_guard<std::mutex> lk(g_lifecycle_mu);
    g_known_queues.erase(queue_id_of(q));
    if (q->dispatch_log_active.load(std::memory_order_acquire)) {
      disable_dispatch_log_for_queue_locked(q);
    }
  }

  // Drop the per-queue refcount entry. Safe to do unconditionally — if the
  // queue never had an acquire, the lookup is just a miss.
  {
    std::lock_guard<std::mutex> lk(g_owners_mu);
    g_owners.erase(q);
  }
}

}  // namespace dispatch_log
}  // namespace rocr
