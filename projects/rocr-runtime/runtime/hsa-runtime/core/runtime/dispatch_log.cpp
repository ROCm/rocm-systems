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
// timestamps. Phase A scaffolding (spec 2026-04-27).
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
// PHASE A NOTE (KFD substrate): the AMDKFD_IOC_PROFILER_DISPATCH_LOG ioctl
// op number defined below is a *placeholder* — the kernel does not yet
// recognize it (spec §10 dep #2). The init() probe therefore always sets
// g_substrate_present=false on Phase A, which short-circuits the entire
// data plane: the poller still ticks, but its enable pass treats every
// queue as "no-op (substrate absent)" and never spawns the drainer. The
// scaffolding compiles, links, and runs as a no-op against today's KFD.
// When the real ioctl number lands, swap AMDKFD_IOC_PROFILER_DISPATCH_LOG
// for the upstream value and the rest of the data plane is exercised.

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
#include <sys/mman.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/types.h>
#include <unistd.h>
#endif

#include "core/inc/agent.h"
#include "core/inc/amd_gpu_agent.h"
#include "core/inc/queue.h"
#include "lttng/rocm_trace_emit.h"

namespace rocr {
namespace dispatch_log {

namespace {

// ============================================================================
// Phase A KFD ioctl placeholder. See file header note above and spec §10 dep
// #2. The op number is a plausible scratch value; it is not the upstream
// AMDKFD_IOC_PROFILER number. The probe at init() expects ENOTSUP / EINVAL
// from today's kernel and disables the data plane via g_substrate_present.
// ============================================================================
#if defined(__linux__)

#ifndef AMDKFD_IOCTL_BASE
#define AMDKFD_IOCTL_BASE 'K'
#endif

// Plausible scratch op number. Real upstream allocation TBD.
struct kfd_profiler_dispatch_log_args {
  uint32_t op;             // PROBE / ENABLE / DISABLE / QUERY
  uint32_t flags;
  uint32_t gpu_id;
  uint32_t queue_id;
  uint64_t buffer_gpu_addr;
  uint64_t buffer_size;
  uint64_t write_ptr_addr;
  uint64_t read_ptr_addr;
  uint64_t generation;
};

// Op tags carried in kfd_profiler_dispatch_log_args::op.
enum {
  KFD_DISPATCH_LOG_OP_PROBE   = 0,
  KFD_DISPATCH_LOG_OP_ENABLE  = 1,
  KFD_DISPATCH_LOG_OP_DISABLE = 2,
  KFD_DISPATCH_LOG_OP_QUERY   = 3,
};

// Scratch ioctl number 0xFE in the AMDKFD ioctl space. This is a placeholder
// per spec §10 dep #2; the upstream KFD does not recognize it and will return
// ENOTSUP / EINVAL, which is the documented Phase A path.
#define AMDKFD_IOC_PROFILER_DISPATCH_LOG \
  _IOWR(AMDKFD_IOCTL_BASE, 0xFE, struct kfd_profiler_dispatch_log_args)

// Path used to probe KFD support at init().
constexpr const char* kKfdDevicePath = "/dev/kfd";

#endif  // __linux__

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
// drainer registry (g_active_queues) and per-pass snapshot vectors. The
// destructor is what frees the ring buffer; freeing happens exactly once,
// when the last shared_ptr ref is released. Deliberately stores NO Queue*
// — see spec §4 "Queue / drain-state lifetime" for the dangling-Queue
// rationale.
// ============================================================================
struct queue_drain_state {
  // Full 64-bit hsa_queue_t::id (see hsa.h). Internal ownership / lifetime
  // maps key on this full id to avoid the high-32-bit collision risk
  // flagged in C5 (two distinct live queues differing only in the high 32
  // bits would have aliased a uint32_t key). The on-the-wire LTTng
  // tracepoint payload still carries only the low 32 bits per spec §6/§14
  // — we narrow at the rocm_trace_emit_* call sites, not here.
  uint64_t  queue_id = 0;

  // Owned, queue-independent GPU->system time-translation callable.
  // Captured at register time so it remains callable even after the
  // underlying core::Queue (and any queue-scoped state on the agent) has
  // been destroyed. The agent itself outlives any queue created on it
  // (HSA agents have process lifetime), so capturing the agent reference
  // by value inside this callable is safe.
  std::function<uint64_t(uint64_t /* gpu_ts */)> translate_gpu_ts;

  // Buffer ownership.
  void*    buffer_base = nullptr;     // host-virtual base of the full alloc
                                      // (header + data area). Freed by the
                                      // destructor below.
  size_t   buffer_total_bytes = 0;    // bytes to munmap / free()

  void*    ring_base   = nullptr;     // = buffer_base + DISPATCH_LOG_HEADER_BYTES
  uint32_t ring_bytes  = 0;           // power-of-2 size of the data area
  uint32_t ring_mask   = 0;           // ring_bytes - 1

  volatile uint64_t* fw_wptr = nullptr;  // lives in the header at
                                         // buffer_base + 0; FW-updated.

  uint64_t generation = 0;

  // Mutated only under drain_mu (see below).
  uint64_t read_cursor = 0;
  std::unordered_map<uint32_t, uint64_t> pending_starts;

  // Per-queue drain mutex. Serializes drain_one_queue() (steady state) and
  // ts_drainer_drain_now() (synchronous final drain on the destroying /
  // disable-edge thread). Per-queue scope, so other queues' drains are not
  // blocked.
  std::mutex drain_mu;

  ~queue_drain_state() {
    if (buffer_base != nullptr && buffer_total_bytes > 0) {
#if defined(__linux__)
      ::munmap(buffer_base, buffer_total_bytes);
#else
      std::free(buffer_base);
#endif
      buffer_base = nullptr;
    }
  }
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

// Set true once at init() if the KFD substrate probe succeeds. Phase A: this
// stays false because the placeholder ioctl is unknown to today's KFD.
std::atomic<bool> g_substrate_present{false};

// Phase A: latched true after we log the "substrate absent, tracepoint enable
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
// the §4 idempotent enable/disable passes:
//
//   - curr==true tick: for each Q in g_known_queues with
//     dispatch_log_active==false, call enable_dispatch_log_for_queue(Q).
//   - curr==false tick: for each Q in g_known_queues with
//     dispatch_log_active==true, call disable_dispatch_log_for_queue(Q).
//
// **Lifetime contract.** The drainer NEVER reads this map (the
// no-Queue*-in-drainer invariant of queue_drain_state — see spec §4 "Queue /
// drain-state lifetime" — is therefore preserved). The poller and the
// queue-create/destroy hooks all serialize on g_lifecycle_mu when touching
// this map AND when calling enable/disable; on_queue_destroy removes from
// g_known_queues BEFORE running the disable sequence, so the poller can
// never observe a Queue* whose underlying core::Queue is being torn down by
// the destroy thread (the destroy thread is forced to wait behind the
// mutex if the poller is currently iterating).
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

// Per-agent "no-dispatch-log" mark. If the KFD ioctl returns ENOTSUP for any
// queue on a given agent, that agent is added here and subsequent
// enable_dispatch_log_for_queue calls on queues belonging to it short-circuit
// (spec §5 unsupported-agent row, §9 unsupported-agent row).
std::unordered_set<core::Agent*> g_no_dispatch_log_agents;

// KFD fd held open for the lifetime of the runtime. -1 if probe failed.
int g_kfd_fd = -1;

// Threads.
std::thread g_poller_thread;
std::thread g_drainer_thread;
std::atomic<bool> g_drainer_running{false};

// Drainer wakeup CV. Used to long-sleep (50 ms) when G_tracepoint_enabled is
// false and short-sleep (500 us) on idle. Notified on shutdown.
std::mutex g_drainer_mu;
std::condition_variable g_drainer_cv;

// Monotonically incremented for every per-queue ENABLE so a stale FW write
// to a freed buffer is detectable on the host side.
std::atomic<uint64_t> g_generation_counter{1};

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
bool kfd_open_and_probe() {
  if (g_kfd_fd != -1) return true;
  int fd = ::open(kKfdDevicePath, O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    return false;
  }
  // Probe: zero-arg PROBE call. The Phase A ioctl is unknown to today's KFD
  // and will return -1 / ENOTSUP / EINVAL; that is the documented Phase A
  // result. If a future KFD recognizes the op, it should return 0.
  struct kfd_profiler_dispatch_log_args args = {};
  args.op    = KFD_DISPATCH_LOG_OP_PROBE;
  args.flags = 0;
  int rc = ::ioctl(fd, AMDKFD_IOC_PROFILER_DISPATCH_LOG, &args);
  if (rc != 0) {
    // Substrate absent — don't keep the fd around either. We use the runtime's
    // own KFD fd for any future per-queue ioctls, so closing this probe fd is
    // fine.
    ::close(fd);
    g_kfd_fd = -1;
    return false;
  }
  g_kfd_fd = fd;
  return true;
}
#else
bool kfd_open_and_probe() { return false; }
#endif

// Issue an ENABLE / DISABLE ioctl. Returns 0 on success, errno on failure.
int kfd_dispatch_log_op(uint32_t op, core::Queue* q,
                        uint64_t buffer_gpu_addr, uint64_t buffer_size,
                        uint64_t write_ptr_addr, uint64_t generation) {
#if defined(__linux__)
  if (g_kfd_fd < 0) return ENOTSUP;
  struct kfd_profiler_dispatch_log_args args = {};
  args.op              = op;
  args.flags           = 0;
  // gpu_id: derive from agent if a GPU agent. Phase A leaves zero on failure.
  args.gpu_id          = 0;
  // KFD ioctl arg field is uint32_t per the (Phase A placeholder) wire
  // contract. Narrow internally; the struct field, not our internal map,
  // dictates the width here.
  args.queue_id        = queue_id_to_wire(queue_id_of(q));
  args.buffer_gpu_addr = buffer_gpu_addr;
  args.buffer_size     = buffer_size;
  args.write_ptr_addr  = write_ptr_addr;
  args.read_ptr_addr   = 0;  // host-managed locally
  args.generation      = generation;
  if (::ioctl(g_kfd_fd, AMDKFD_IOC_PROFILER_DISPATCH_LOG, &args) != 0) {
    return errno != 0 ? errno : ENOTSUP;
  }
  return 0;
#else
  (void)op; (void)q; (void)buffer_gpu_addr; (void)buffer_size;
  (void)write_ptr_addr; (void)generation;
  return ENOTSUP;
#endif
}

// Allocate the per-queue ring buffer. Phase A: page-aligned anonymous mmap.
// In a future revision this should use the existing pinned-host allocator
// from the agent so the buffer is GPU-visible without an explicit pin
// (mirroring the AQL queue ring allocation pattern). The buffer total size
// is DISPATCH_LOG_HEADER_BYTES + DISPATCH_LOG_RING_BYTES.
struct ring_alloc_t {
  void*  base;
  size_t total_bytes;
};
ring_alloc_t alloc_ring_buffer() {
  const size_t total = DISPATCH_LOG_HEADER_BYTES + DISPATCH_LOG_RING_BYTES;
  ring_alloc_t out{nullptr, 0};
#if defined(__linux__)
  void* p = ::mmap(nullptr, total, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return out;
  std::memset(p, 0, total);
  out.base        = p;
  out.total_bytes = total;
#else
  void* p = std::calloc(1, total);
  if (p == nullptr) return out;
  out.base        = p;
  out.total_bytes = total;
#endif
  return out;
}

void free_ring_buffer(ring_alloc_t& a) {
  if (a.base == nullptr) return;
#if defined(__linux__)
  ::munmap(a.base, a.total_bytes);
#else
  std::free(a.base);
#endif
  a.base = nullptr;
  a.total_bytes = 0;
}

// Snapshot: clone shared_ptrs out of g_active_queues under the lifecycle
// mutex, return the vector. Drainer then iterates without holding the mutex.
std::vector<std::shared_ptr<queue_drain_state>> snapshot_active_queues() {
  std::vector<std::shared_ptr<queue_drain_state>> out;
  std::lock_guard<std::mutex> lk(g_lifecycle_mu);
  out.reserve(g_active_queues.size());
  for (auto& kv : g_active_queues) out.push_back(kv.second);
  return out;
}

// ============================================================================
// drain_one_queue (spec §7). Holds qs.drain_mu for the entire pass.
// ============================================================================

bool drain_one_queue(queue_drain_state& qs, bool force_emit) {
  std::lock_guard<std::mutex> lk(qs.drain_mu);

  if (qs.fw_wptr == nullptr) return false;

  const uint64_t fw_wptr = __atomic_load_n(qs.fw_wptr, __ATOMIC_ACQUIRE);
  if (fw_wptr == qs.read_cursor) return false;

  // Wraparound check (spec §6 bytes_lost field definition).
  if (fw_wptr - qs.read_cursor > qs.ring_bytes) {
    // Narrow internal 64-bit queue_id to the 32-bit on-the-wire id at the
    // tracepoint boundary (spec §6 / §14, C5 fix).
    rocm_trace_emit_hsa_kernel_dispatch_drop(queue_id_to_wire(qs.queue_id),
                                             fw_wptr - qs.read_cursor);
    qs.read_cursor = fw_wptr - qs.ring_bytes;
  }

  while (qs.read_cursor < fw_wptr) {
    const auto* rec = reinterpret_cast<const mec_dispatch_record_16*>(
        static_cast<const char*>(qs.ring_base) +
        (qs.read_cursor & qs.ring_mask));

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

    qs.read_cursor += sizeof(mec_dispatch_record_16);
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
    // 0->1 transition: set the bit via the existing API. SetProfiling()
    // forwards to AMD_HSA_BITS_SET on amd_queue_.queue_properties (the
    // GPU-visible bit) and is the same path hsa_amd_profiling_set_profiler_enabled
    // takes today (see queue.h:446 / amd_aql_queue.cpp:1559).
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
    q->SetProfiling(false);
  }
  return HSA_STATUS_SUCCESS;
}

namespace {

// ============================================================================
// Per-queue ENABLE / DISABLE sequences (spec §5).
//
// enable_dispatch_log_for_queue runs the per-queue all-or-nothing sequence:
//   1. alloc buffer
//   2. QueueProfilingAcquire (sets the bit if 0->1)
//   3. KFD ioctl ENABLE
//   4..6. (queue unmap/remap and MQD propagation are KFD-side; the ioctl
//         encapsulates them)
//   7. register in g_active_queues
//
// Each step has its own unwind on failure. On any failure: free buffer,
// release any acquired profiling ref, leave dispatch_log_active=false, log
// once-per-queue at WARNING.
// ============================================================================

// _locked variant. Caller holds g_lifecycle_mu. Used by the poller's enable
// pass (which iterates g_known_queues under the mutex) and by on_queue_create
// via the public wrapper below.
void enable_dispatch_log_for_queue_locked(core::Queue* q) {
  if (q == nullptr) return;

  // Idempotence guard: skip if already active.
  if (q->dispatch_log_active.load(std::memory_order_relaxed)) return;

  // Only meaningful for GPU agents — host/soft queues have no MEC firmware
  // to write records, so the KFD ioctl would either be rejected or be a
  // no-op. Skip them outright.
  if (!agent_is_gpu(q->GetAgent())) return;

  // Phase A short-circuit: substrate absent.
  if (!g_substrate_present.load(std::memory_order_relaxed)) return;

  // Per-agent "no-dispatch-log" filter.
  if (g_no_dispatch_log_agents.count(q->GetAgent()) > 0) return;

  // Step 1: alloc.
  ring_alloc_t alloc = alloc_ring_buffer();
  if (alloc.base == nullptr) {
    std::fprintf(stderr,
                 "[hsa-runtime] dispatch_log: queue_id=%llu ENABLE step 1 (alloc) "
                 "failed errno=%d\n",
                 static_cast<unsigned long long>(queue_id_of(q)), errno);
    return;
  }

  const uint64_t generation = g_generation_counter.fetch_add(
      1, std::memory_order_relaxed);

  // Initialize the header now (host-side write before KFD publishes the
  // buffer to FW). Layout per spec §5.
  auto* hdr = reinterpret_cast<dispatch_log_header*>(alloc.base);
  hdr->fw_wptr    = 0;
  hdr->generation = generation;
  hdr->version    = DISPATCH_LOG_VERSION;
  std::memset(hdr->reserved, 0, sizeof(hdr->reserved));

  // Step 2: acquire profiling ref FIRST (spec §5 ordering rationale).
  // QueueProfilingAcquire takes g_owners_mu, NOT g_lifecycle_mu, so it's
  // safe to call while holding g_lifecycle_mu.
  hsa_status_t s = QueueProfilingAcquire(q);
  if (s != HSA_STATUS_SUCCESS) {
    std::fprintf(stderr,
                 "[hsa-runtime] dispatch_log: queue_id=%llu ENABLE step 2 "
                 "(QueueProfilingAcquire) failed status=%d\n",
                 static_cast<unsigned long long>(queue_id_of(q)),
                 static_cast<int>(s));
    free_ring_buffer(alloc);
    return;
  }

  // Step 3: KFD ioctl ENABLE. buffer_gpu_addr points at the data area
  // (after the header). Phase A: this currently always returns ENOTSUP
  // since the substrate is absent — but g_substrate_present should have
  // been false above and we'd have early-returned. Defense in depth:
  // if we reach here the probe lied. Treat any failure as a per-queue
  // failure per spec §5.
  //
  // NOTE: this Phase A scaffolding passes the host virtual address as the
  // GPU address. Real implementation must pin the buffer via the existing
  // HSA pinned-host allocator and pass the GPU-visible address. That hookup
  // is left for Phase B once a real substrate exists.
  const uint64_t host_base = reinterpret_cast<uint64_t>(alloc.base);
  const int rc = kfd_dispatch_log_op(
      KFD_DISPATCH_LOG_OP_ENABLE, q,
      host_base + DISPATCH_LOG_HEADER_BYTES,    // FW data area base
      DISPATCH_LOG_RING_BYTES,                  // data area size
      host_base + 0,                            // fw_wptr lives in header
      generation);
  if (rc != 0) {
    std::fprintf(stderr,
                 "[hsa-runtime] dispatch_log: queue_id=%llu ENABLE step 3 "
                 "(KFD ioctl) failed errno=%d\n",
                 static_cast<unsigned long long>(queue_id_of(q)), rc);
    if (rc == ENOTSUP) {
      g_no_dispatch_log_agents.insert(q->GetAgent());
    }
    QueueProfilingRelease(q);
    free_ring_buffer(alloc);
    return;
  }

  // Step 7: register in the drainer registry.
  auto qs = std::make_shared<queue_drain_state>();
  qs->queue_id           = queue_id_of(q);
  qs->translate_gpu_ts   = make_translate_gpu_ts(q);
  qs->buffer_base        = alloc.base;
  qs->buffer_total_bytes = alloc.total_bytes;
  qs->ring_base          = static_cast<char*>(alloc.base) + DISPATCH_LOG_HEADER_BYTES;
  qs->ring_bytes         = static_cast<uint32_t>(DISPATCH_LOG_RING_BYTES);
  qs->ring_mask          = qs->ring_bytes - 1;
  qs->fw_wptr            = &hdr->fw_wptr;
  qs->generation         = generation;
  qs->read_cursor        = 0;

  g_active_queues[qs->queue_id] = qs;

  // Ownership of the buffer now lives in qs (shared_ptr destructor frees it).
  // Suppress the local alloc bookkeeping so a later free_ring_buffer on
  // `alloc` would be a no-op.
  alloc.base        = nullptr;
  alloc.total_bytes = 0;

  q->dispatch_log_active.store(true, std::memory_order_release);
}

// _locked variant. Caller holds g_lifecycle_mu. Used by the poller's
// disable pass, by shutdown(), and by on_queue_destroy via the public
// wrapper below. Note that wait_for_idle and the KFD ioctl run while
// the lifecycle mutex is held; this is acceptable because the lifecycle
// mutex is contended only by other lifecycle events (queue create/destroy,
// poller ticks every 5 ms), not by per-dispatch hot paths.
void disable_dispatch_log_for_queue_locked(core::Queue* q) {
  if (q == nullptr) return;
  if (!q->dispatch_log_active.load(std::memory_order_acquire)) return;

  // Spec §4 disable-edge per-queue sequence.

  // a. wait_for_idle (bounded ≤ 100 ms).
  wait_for_idle(q);

  // b. Final drain with force_emit=true (spec §6).
  ts_drainer_drain_now_locked(q);

  // c. KFD ioctl DISABLE (best-effort).
  (void)kfd_dispatch_log_op(KFD_DISPATCH_LOG_OP_DISABLE, q,
                            /*buffer_gpu_addr=*/0, /*buffer_size=*/0,
                            /*write_ptr_addr=*/0, /*generation=*/0);

  // d. Release profiling ref (clears the bit only if 1->0).
  // QueueProfilingRelease takes g_owners_mu, NOT g_lifecycle_mu.
  QueueProfilingRelease(q);

  // e. Unregister from drainer registry. Drops the registry's shared_ptr
  //    ref. Buffer free is gated by shared_ptr destructor — if a drainer
  //    snapshot still holds a ref the destructor fires when that snapshot
  //    vector is destroyed (spec §4 lifetime protocol).
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
//
// Every 5 ms:
//   1. Compute curr = !rocm_trace_disabled() && lttng_ust_tracepoint_enabled
//                     (rocm_hsa, kernel_dispatch_complete) && g_substrate_present.
//   2. Store G_tracepoint_enabled = curr.
//   3. If curr=true: spawn drainer if needed; iterate registered queues,
//      enable any not-yet-active.
//   4. If curr=false: iterate registered queues, disable any still-active.
//
// Both branches run every tick (idempotent, not edge-triggered) — this is
// the convergence guarantee for the queue-create-vs-disable race
// (spec §4 "Queue-create vs. disable-pass race").
// ============================================================================

bool predicate_now() {
#if defined(HSA_ENABLE_LTTNG_UST) && HSA_ENABLE_LTTNG_UST
  if (rocm_trace_disabled()) return false;
  if (!g_substrate_present.load(std::memory_order_relaxed)) {
    // Phase A: log once that the substrate is missing so the user knows
    // why an enabled tracepoint produces no events.
    if (lttng_ust_tracepoint_enabled(rocm_hsa, kernel_dispatch_complete)) {
      bool warned = g_substrate_absent_warned.load(std::memory_order_relaxed);
      if (!warned && g_substrate_absent_warned.compare_exchange_strong(
                         warned, true)) {
        std::fprintf(stderr,
                     "[hsa-runtime] dispatch_log: tracepoint "
                     "rocm_hsa:kernel_dispatch_complete is enabled, but the "
                     "KFD substrate (AMDKFD_IOC_PROFILER_DISPATCH_LOG) is not "
                     "available on this kernel. No events will be produced. "
                     "(Phase A scaffolding; see spec §10 dep #2.)\n");
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
      // Iterates g_known_queues under the lifecycle mutex; on_queue_create
      // and on_queue_destroy serialize on the same mutex, so the Queue*
      // values we observe here cannot be torn down concurrently.
      // enable_dispatch_log_for_queue_locked is itself a no-op for queues
      // that are already active (idempotent enable).
      std::lock_guard<std::mutex> lk(g_lifecycle_mu);
      // Snapshot first so enable's per-queue work doesn't mutate the map
      // we're iterating (g_active_queues is mutated, but g_known_queues
      // is stable across enable for an existing queue — defensive copy
      // anyway).
      std::vector<core::Queue*> known;
      known.reserve(g_known_queues.size());
      for (auto& kv : g_known_queues) known.push_back(kv.second);
      for (core::Queue* q : known) {
        enable_dispatch_log_for_queue_locked(q);
      }
    } else {
      // DISABLE pass (spec §4 + §9 ROCM_LTTNG_UST_DISABLE row): for each
      // known live queue Q with dispatch_log_active==true, run the
      // per-queue DISABLE sequence (wait_for_idle, final drain, KFD
      // DISABLE, QueueProfilingRelease, mark inactive). Idempotent over
      // already-inactive queues. Required for kill-switch hygiene per
      // spec §1039.
      std::lock_guard<std::mutex> lk(g_lifecycle_mu);
      std::vector<core::Queue*> known;
      known.reserve(g_known_queues.size());
      for (auto& kv : g_known_queues) known.push_back(kv.second);
      for (core::Queue* q : known) {
        disable_dispatch_log_for_queue_locked(q);
      }
    }

    // Sleep until next tick.
    std::this_thread::sleep_for(tick_interval);
  }
}

}  // namespace

// ============================================================================
// Public API.
// ============================================================================

void init() {
  // KFD substrate probe. Phase A: this is expected to fail.
  const bool present = kfd_open_and_probe();
  g_substrate_present.store(present, std::memory_order_release);

  // Spawn poller unconditionally so the steady-state idle path is exercised
  // even on Phase A (substrate absent). The poller's enable branch
  // short-circuits via predicate_now() → false when substrate is missing.
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
  // (wait_for_idle, final drain, KFD DISABLE, QueueProfilingRelease) for
  // every queue still active at process exit. The poller is joined above
  // so no other thread is touching g_known_queues / g_active_queues here;
  // the lock is taken for consistency with the helpers' contract.
  {
    std::lock_guard<std::mutex> lk(g_lifecycle_mu);
    std::vector<core::Queue*> known;
    known.reserve(g_known_queues.size());
    for (auto& kv : g_known_queues) known.push_back(kv.second);
    for (core::Queue* q : known) {
      disable_dispatch_log_for_queue_locked(q);
    }
    g_known_queues.clear();
    // Defensive: drop any remaining drainer refs (the disable sequence
    // above should have removed them, but if a destroy hook racing with
    // shutdown left dangling state, this releases the buffers).
    g_active_queues.clear();
    g_no_dispatch_log_agents.clear();
  }
  {
    std::lock_guard<std::mutex> lk(g_owners_mu);
    g_owners.clear();
  }

#if defined(__linux__)
  if (g_kfd_fd >= 0) {
    ::close(g_kfd_fd);
    g_kfd_fd = -1;
  }
#endif
}

void on_queue_create(core::Queue* q) {
  if (q == nullptr) return;

  // Register the queue in the poller's live-queue side map BEFORE running
  // the enable sequence. Holding g_lifecycle_mu across both the registry
  // insert and the enable serializes us against the poller (which iterates
  // g_known_queues under the same mutex) and against on_queue_destroy.
  // The queue-create-vs-disable race is resolved by the idempotent disable
  // convergence guarantee in spec §4: if curr flips to false between our
  // load and the next poll tick, the disable pass will iterate
  // g_known_queues (which now contains q) and clean up.
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
