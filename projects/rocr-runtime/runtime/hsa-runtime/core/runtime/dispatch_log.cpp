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
//
// SENTINEL-SCAN DESIGN: The substrate publishes neither a host-readable FW
// write pointer nor any other end-of-records marker. The drainer locates
// fresh records by scanning the ring sequentially from a host-managed
// monotonic cursor (next_idx). A slot whose record_type is 0 is "empty"
// (FW writes record_type ∈ {1,2}, and the buffer is pre-zeroed at alloc
// in AqlQueue::SetProfiling). After consuming a slot the drainer zeroes
// it again so wraparound rewrites are re-detectable. See drain_one_queue
// below for the canonical implementation.

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
  // do not free here. The kernel-side BO is allocated oversized at
  // `ring_records * 40` bytes purely to satisfy host KFD BO-size validation
  // (amd/amdkfd/kfd_process_queue_manager.c:633 `buf_byte_size = count * 40`).
  // The active FW/drainer ring uses the 16-byte FW record stride
  // (kSlotStride): FW writes records at 16-byte stride and the drainer
  // reads them at 16-byte stride. The extra allocation tail beyond
  // `ring_records * 16` is unused/reserved and is not part of the ring.
  void*    ring_base    = nullptr;     // host-virtual base of the FW record area
  uint32_t ring_records = 0;           // power-of-2 slot count (e.g. 65536)
  uint32_t ring_mask    = 0;           // ring_records - 1, for slot indexing

  // Host-managed monotonic record cursor. Mutated only under drain_mu.
  // Slot index is (next_idx & ring_mask). Sentinel-scan design: the
  // substrate publishes no FW wptr, so we advance next_idx for every
  // slot whose record_type is non-zero, and zero the slot after consume
  // so a wraparound re-write is re-detectable. See drain_one_queue.
  uint64_t next_idx = 0;
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
  // Accept any KFD whose ABI is at-or-after 1.22. A naive
  // (major>=1 && minor>=22) check would reject a hypothetical KFD 2.0
  // (which has minor==0). Compare lexicographically instead.
  return args.major_version > 1 ||
         (args.major_version == 1 && args.minor_version >= 22);
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

bool drain_one_queue(queue_drain_state& qs, bool force_emit) {
  std::lock_guard<std::mutex> lk(qs.drain_mu);

  // Poisoned (disable in flight) or never-populated.
  if (qs.ring_base == nullptr || qs.ring_records == 0) return false;

  // C9 (stage-2 review): bound each pass to one full ring sweep. The
  // sentinel-scan loop exits only when it observes record_type == 0 for
  // the current slot; if FW keeps the ring continuously non-empty, an
  // unbounded loop here would let one queue monopolize ts_drainer (which
  // walks queues sequentially under g_lifecycle_mu in
  // ts_drainer_thread_main) and starve other queues, plus delay disable
  // / shutdown paths that need drain_mu. Capping at ring_records slots
  // (= one full ring sweep) preserves forward progress: at any instant
  // FW can have at most ring_records unconsumed slots, so a full sweep
  // always drains everything that was visible at pass start. Anything
  // FW writes during the pass becomes the next pass's work — which is
  // also the prior wptr-snapshot design's behaviour.
  const uint64_t pass_budget = qs.ring_records;
  uint64_t pass_consumed = 0;

  bool any = false;
  while (pass_consumed < pass_budget) {
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
    // body loads (and the slot memset) are not reordered above it; this
    // turns the type-load into a proper atomic acquire instead of a
    // plain (data-racy) load on memory another agent writes
    // concurrently. The body fields are read with relaxed atomicity for
    // the same reason — to avoid UB from plain loads on FW-written
    // memory — but no further fence is needed because the acquire on
    // record_type already happens-before them.
    uint32_t rt = __atomic_load_n(&rec->record_type, __ATOMIC_ACQUIRE);
    if (rt == 0) break;

    // Snapshot the rest of the record locally before zeroing the slot.
    // Relaxed atomic loads suffice (see comment above) — the acquire on
    // record_type above already orders these.
    const uint32_t ts_lo        = __atomic_load_n(&rec->ts_lo, __ATOMIC_RELAXED);
    const uint32_t ts_hi        = __atomic_load_n(&rec->ts_hi, __ATOMIC_RELAXED);
    const uint32_t dispatch_idx = __atomic_load_n(&rec->dispatch_idx, __ATOMIC_RELAXED);
    const uint64_t gpu_ts       = (static_cast<uint64_t>(ts_hi) << 32) | ts_lo;

    // Zero the consumed slot so a wraparound rewrite by FW (next time
    // the ring loops past this position) is re-detected as a fresh
    // non-zero record_type. FW writes 16-byte records at 16-byte
    // stride, so zeroing kSlotStride bytes resets the entire next-write
    // target.
    std::memset(rec, 0, kSlotStride);

    if (rt == DISPATCH_LOG_RECORD_START) {
      qs.pending_starts[dispatch_idx] = gpu_ts;
    } else if (rt == DISPATCH_LOG_RECORD_END) {
      auto it = qs.pending_starts.find(dispatch_idx);
      if (it != qs.pending_starts.end()) {
        const uint64_t start_gpu = it->second;
        qs.pending_starts.erase(it);

        // Use the captured queue-independent translation callable —
        // Queue may be destroyed mid-pass even though the shared_ptr
        // keeps qs alive.
        const uint64_t start_sys = qs.translate_gpu_ts(start_gpu);
        const uint64_t end_sys   = qs.translate_gpu_ts(gpu_ts);

        // Narrow internal 64-bit queue_id at the tracepoint boundary
        // (spec §6 / §14, C5 fix). dispatch_idx is 32-bit per FW contract.
        rocm_trace_emit_hsa_kernel_dispatch_complete(
            queue_id_to_wire(qs.queue_id), dispatch_idx, start_sys,
            end_sys, force_emit);
      }
      // else: orphan END (matching START was lost). Silently skip.
    }

    qs.next_idx += 1;
    pass_consumed += 1;
    any = true;
  }
  return any;
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
//   5. Register a non-owning queue_drain_state in g_active_queues with
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
  if (!g_substrate_present.load(std::memory_order_relaxed)) return;
  if (g_no_dispatch_log_agents.count(q->GetAgent()) > 0) return;

  // Step 1: enable profiling on the queue. This is the AqlQueue::SetProfiling
  // call that allocates the dispatch-record buffer, calls
  // hsaKmtSetQueueProfilingBuffer, and Suspends/Resumes the queue to flush
  // the MQD via UPDATE_QUEUE. QueueProfilingAcquire takes g_owners_mu (not
  // g_lifecycle_mu), so it's safe to call while holding g_lifecycle_mu.
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

  // Step 3: register in the drainer registry. NON-OWNING — buffer is
  // owned by AqlQueue, freed on SetProfiling(false) or AqlQueue dtor.
  auto qs = std::make_shared<queue_drain_state>();
  qs->queue_id         = queue_id_of(q);
  qs->translate_gpu_ts = make_translate_gpu_ts(q);
  qs->ring_base        = buf;
  qs->ring_records     = record_count;
  qs->ring_mask        = record_count - 1;
  qs->next_idx         = 0;

  g_active_queues[qs->queue_id] = qs;

  q->dispatch_log_active.store(true, std::memory_order_release);
}

void disable_dispatch_log_for_queue_locked(core::Queue* q) {
  if (q == nullptr) return;
  if (!q->dispatch_log_active.load(std::memory_order_acquire)) return;

  // Spec §4 disable-edge per-queue sequence.
  //
  // C2 fix: ordering between final-drain, drainer-snapshot poisoning,
  // registry erase, and buffer release is critical. queue_drain_state
  // stores NON-OWNING pointers into a buffer owned by AqlQueue; once
  // QueueProfilingRelease triggers SetProfiling(false) the buffer is
  // freed. A snapshot held by the drainer thread (snapshot_active_queues)
  // pins the queue_drain_state alive but does NOT pin the buffer, so
  // the drainer could dereference freed memory in drain_one_queue.
  //
  // We close that window by poisoning the snapshot-visible pointers
  // under qs->drain_mu, then erasing from g_active_queues, and only
  // then calling QueueProfilingRelease (which frees the buffer):
  //
  //   - drain_one_queue takes drain_mu first thing, then early-returns
  //     on ring_base==nullptr || ring_records==0. Any drainer that
  //     hasn't started its pass for this queue yet will exit early.
  //   - Any drainer already inside drain_one_queue holds drain_mu, so
  //     the poisoning step blocks until it completes its current pass.
  //     The buffer is not yet freed at that point (we haven't released
  //     yet), so that pass is safe to finish.

  // a. wait_for_idle (bounded ≤ 100 ms).
  wait_for_idle(q);

  // b. Final drain with force_emit=true (spec §6).
  ts_drainer_drain_now_locked(q);

  // c. Look up the registered drain-state shared_ptr. After this we hold
  //    a local ref so qs survives even after the registry erase below.
  std::shared_ptr<queue_drain_state> qs_ptr;
  {
    auto it = g_active_queues.find(queue_id_of(q));
    if (it != g_active_queues.end()) qs_ptr = it->second;
  }

  // d. Poison the non-owning pointers under drain_mu so any drainer
  //    snapshot still in flight will exit drain_one_queue early. The
  //    drain_mu acquire serializes with any in-progress drain pass.
  if (qs_ptr) {
    std::lock_guard<std::mutex> lk(qs_ptr->drain_mu);
    qs_ptr->ring_base    = nullptr;
    qs_ptr->ring_records = 0;
    qs_ptr->ring_mask    = 0;
  }

  // e. Unregister from drainer registry. Drops the registry's shared_ptr
  //    ref. Any in-flight drainer snapshot still holds its own ref to
  //    qs_ptr but its ring_base is now nullptr.
  g_active_queues.erase(queue_id_of(q));

  // f. Release profiling ref. On the 1->0 edge AqlQueue::SetProfiling(false)
  //    clears the KFD profiling-buffer registration (UPDATE_QUEUE with
  //    addr=0) and frees the per-queue dispatch-record buffer.
  //    QueueProfilingRelease takes g_owners_mu, NOT g_lifecycle_mu.
  //    By this point no drainer can dereference the buffer because
  //    every snapshot's qs->ring_base is nullptr.
  QueueProfilingRelease(q);

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
