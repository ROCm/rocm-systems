// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

// Range recording state for range replay (see experimental/range_replay.h).
//
// A range is thread-scoped: rocprofiler_range_replay_begin() opens one on the calling thread and
// rocprofiler_range_replay_end() closes it. While it is open, WriteInterceptor hands every
// submission from that thread to note_submission(), which either records the dispatch packets (with
// their kernarg bytes copied out, since HIP recycles kernarg blocks as soon as a kernel completes)
// or declines the range.
//
// The eligibility bookkeeping is deliberately GPU-free and lives in range_record_t so it can be
// unit tested without HSA: the executor consumes an already-decided record.
//
// Declining is never fatal. The application's own execution of the range -- pass 0 -- is forwarded
// unmodified in every case; a decline only means the SDK will not re-execute the recording.
//
// A host-replay range (host_range_state_t::enabled) records nothing: the extra passes re-run the
// application's code via siglongjmp, so note_submission only counts its dispatches, and the
// packet-recording declines (queue/agent binding, graphs, copies from the owner thread, budget,
// kernarg layout) do not apply. See host_replay.hpp.

#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/local_context.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"
#include "lib/rocprofiler-sdk/range_replay/digest.hpp"
#include "lib/rocprofiler-sdk/range_replay/replay_callbacks.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
class Queue;
}  // namespace hsa

namespace range_replay
{
// Upper bound on recorded dispatches per range. A range is meant to bracket a phase (an iteration,
// a layer, a collective step), not a whole run: every recorded dispatch holds a packet plus its
// kernarg bytes, and the executor re-submits all of them per pass. Exceeding the budget declines
// the range rather than growing without bound.
constexpr size_t kMaxRecordedDispatches = 4096;

// One recorded dispatch. `packet` is a byte copy of what the application submitted; `kernarg` is a
// copy of the kernarg segment it pointed at, taken at record time. The executor stages the kernarg
// bytes into its own buffer per pass and patches `packet` to point there, because the application's
// kernarg block may be recycled by a later launch in the same range.
struct recorded_dispatch_t
{
    hsa::rocprofiler_packet packet        = {};
    uint64_t                kernel_object = 0;
    uint64_t                kernel_id     = 0;
    std::vector<uint8_t>    kernarg       = {};
};

// GPU-free record of one open range: what it recorded and whether it is still replayable.
class range_record_t
{
public:
    range_record_t() = default;
    explicit range_record_t(uint64_t range_id)
    : m_range_id{range_id}
    {}

    uint64_t range_id() const { return m_range_id; }
    uint64_t queue_key() const { return m_queue_key; }
    uint64_t agent_key() const { return m_agent_key; }
    size_t   dispatch_count() const { return m_dispatches.size(); }

    // Dispatches seen while the range was open and still eligible, up to and including the one
    // whose arrival declined it. This is what the range reports to the tool: for a declined range
    // "how many dispatches were in the range" is still the useful number, while dispatch_count()
    // drops to zero. Recording stops at the decline, so for a range declined partway through this
    // is a lower bound on the dispatches the range actually contained.
    size_t observed_dispatch_count() const { return m_observed; }

    // The recorded decline reason, or ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED when the range has
    // not been declined. "REPLAYED" is the neutral value here: a range that is still eligible when
    // it closes and then runs at least one extra pass reports exactly that status to the tool.
    rocprofiler_range_replay_status_t status() const { return m_status; }
    bool                              eligible() const;

    // Record a decline. First reason wins: later observations cannot mask the original cause, and a
    // declined range stops recording (so a range that grew past its budget does not keep copying
    // kernargs).
    void decline(rocprofiler_range_replay_status_t reason);

    // Bind the range to the queue/agent of its first recorded dispatch. A submission to any other
    // queue declines with MULTI_QUEUE; a different agent declines with MULTI_AGENT, which is
    // checked first because it is the more informative reason.
    bool bind(uint64_t queue_key, uint64_t agent_key);

    // Append a recorded dispatch, declining with PROGRAM_TOO_LARGE at the budget.
    bool add_dispatch(recorded_dispatch_t&& dispatch);

    // Count dispatches without recording them (host replay). No-op once declined.
    void count_dispatches(size_t count);

    // Code objects are numbered in load order. The entry snapshot is taken when the range binds and
    // holds the module variables (__device__ / __constant__ globals) of every code object loaded by
    // then, so a kernel from a code object numbered above the watermark -- loaded lazily by HIP or
    // by a JIT after the range began -- would replay with globals the snapshot cannot restore. Such
    // a dispatch declines with CODE_OBJECT_CHANGED_IN_RANGE. A watermark of 0 means none was set.
    void set_code_object_watermark(uint64_t newest_code_object_id);
    bool admit_code_object(uint64_t code_object_id);

    const std::vector<recorded_dispatch_t>& dispatches() const { return m_dispatches; }
    std::vector<recorded_dispatch_t>&       dispatches() { return m_dispatches; }

private:
    uint64_t                          m_range_id   = 0;
    uint64_t                          m_queue_key  = 0;  // 0 = unbound
    uint64_t                          m_agent_key  = 0;  // 0 = unbound
    rocprofiler_range_replay_status_t m_status     = ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED;
    std::vector<recorded_dispatch_t>  m_dispatches = {};
    size_t                            m_observed   = 0;
    uint64_t                          m_code_object_watermark = 0;
};

// Cross-thread decline channel. Other threads (a foreign dispatch, an async copy) observe that a
// range's isolation assumption is broken, but only the owning thread may touch range_record_t. They
// publish the reason here instead; the owner folds it in when it next records and at close.
// ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED (0) means "nothing published".
struct external_decline_t
{
    std::atomic<uint32_t> reason = {ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED};
    // Device writes from the range's own thread. They are part of the program a host range
    // re-executes, so they are not a decline on their own. finish_pass snapshots this around the
    // tool callbacks: a copy those callbacks submit runs after the pass fence, and the compute
    // queue barrier does not wait for it.
    std::atomic<uint64_t> owning_device_writes = {0};
};

// Device state of one GPU agent captured when a host-replay range opens.
struct host_agent_state_t
{
    hsa_agent_t                                       hsa_agent = {.handle = 0};
    rocprofiler_agent_id_t                            agent_id  = {.handle = 0};
    kernel_replay::memory_snapshot::device_snapshot_t snapshot  = {};
    // Base, size and lifetime of each entry allocation. Reusing a freed address is not unchanged.
    kernel_replay::memory_tracker::tracked_map_t tracked      = {};
    digest::region_digests_t                     exit_digests = {};
};

// Host memory the application registered with rocprofiler_range_replay_host_track_memory.
struct host_region_t
{
    void*                ptr         = nullptr;
    std::vector<uint8_t> saved       = {};
    uint64_t             exit_digest = 0;
};

struct host_range_state_t
{
    bool     enabled = false;
    uint64_t pass    = 0;  // pass the application is executing: 0 is its own run

    bool                            snapshot_taken        = false;
    std::vector<host_agent_state_t> agents                = {};
    std::vector<host_region_t>      regions               = {};
    uint64_t                        code_object_watermark = 0;
    bool                            exit_digests_taken    = false;

    // Carries the PASS callback context from the PHASE_ENTER raised just before a siglongjmp to the
    // PHASE_EXIT raised when that pass reaches end() again.
    pass_context_state_t pass_state = {};

    // The local-context override must outlive the jumps between passes, so it is heap-owned here
    // rather than scoped to one call frame. Released when the range closes.
    std::shared_ptr<kernel_replay::scoped_local_context_control> local_context = {};
};

// The live, HSA-facing half of an open range: the record plus what the executor needs to re-submit
// it (queue, agent, snapshot taken before the first recorded dispatch) and the plan the tool
// supplied at CONFIG.
struct range_context_t
{
    range_record_t         record    = {};
    const hsa::Queue*      queue     = nullptr;
    hsa_agent_t            hsa_agent = {.handle = 0};
    rocprofiler_agent_id_t agent_id  = {.handle = 0};

    kernel_replay::memory_snapshot::device_snapshot_t snapshot       = {};
    bool                                              snapshot_taken = false;
    // Base pointers of the tracked allocations the snapshot covers, sorted. Re-checked at close: a
    // range that allocated or freed device memory while open has regions the snapshot cannot
    // restore, so it is declined rather than replayed against partially-restored inputs.
    std::vector<void*> snapshot_ptrs = {};

    std::shared_ptr<external_decline_t> external = {};

    // What the tool asked for at CONFIG, and the contexts this range's PASS/CLOSE callbacks go to.
    range_plan_t plan = {};

    // Thread and correlation identity captured at begin() so PASS and CLOSE callbacks carry the
    // same provenance the CONFIG callback did.
    uint64_t thread_id        = 0;
    uint64_t internal_corr_id = 0;
    uint64_t ancestor_corr_id = 0;

    host_range_state_t host = {};
};

// Staged thread-local sigjmp_buf behind rocprofiler_range_replay_host_jmpbuf(). host_jmpbuf() arms
// it; take_host_jmp_armed() consumes the arming and promotes the staged target only without an open
// range, so a rejected nested begin cannot replace the active range's target.
void*
host_jmpbuf();

bool
take_host_jmp_armed();

// The buffer itself, without arming. Only a valid siglongjmp target for a range whose begin()
// consumed an arming.
void*
host_jmp_target();

// Host regions registered while no range was open on this thread, consumed by the next begin().
struct pending_host_region_t
{
    void*  ptr  = nullptr;
    size_t size = 0;
};

void
add_pending_host_region(void* ptr, size_t size);

std::vector<pending_host_region_t>
take_pending_host_regions();

// Mark the calling thread's open range as host replay. Its registry entry then matches every agent
// for foreign dispatches and device writes from other threads, while device writes from the owning
// thread are part of the re-executed program and are ignored.
void
set_host_mode(range_context_t& ctx);

// Id of the most recently loaded code object (0 when none is loaded).
uint64_t
newest_code_object_id();

// Publish a decline reason from a thread that does not own the range.
void
publish_external_decline(external_decline_t& channel, rocprofiler_range_replay_status_t reason);

// Fold any externally published decline into the record. Called by the owning thread.
void
fold_external_decline(range_context_t& ctx);

// True while any thread in the process has a range open. Read on the dispatch path before any
// locking, so a run with no open range pays one relaxed atomic load.
bool
any_range_open();

// True while this thread is executing a range's replay passes. The queue path uses it to (a) skip
// re-recording the packets it is itself submitting and (b) skip re-acquiring the per-agent replay
// lock this thread already holds as a writer.
bool
this_thread_replaying();

void
set_this_thread_replaying(bool replaying);

// This thread's open range, or nullptr.
range_context_t*
current_range();

// Open a range on the calling thread. Fails if one is already open.
bool
open_range(uint64_t range_id);

// Detach the thread's range for the executor to consume, unregistering it first so no further
// cross-thread declines can be published, then folding in whatever was published already. Returns
// false when no range was open.
bool
take_range(range_context_t& out);

// Called from the queue path for every submission made by a thread with an open range. Records the
// submission's dispatch packets (copying out their kernarg bytes) or declines the range. Must be
// called before the packets are submitted, while the kernarg blocks they point at still hold this
// launch's arguments.
void
note_submission(const hsa::Queue&              queue,
                const hsa::rocprofiler_packet* packets,
                size_t                         packet_count,
                bool                           graph_launch_active);

// Called from the queue path independently of this thread's own range: if another thread has a
// range open on the same agent, that range's isolation assumption is broken.
void
note_foreign_dispatch(uint64_t agent_key);

// Called from the async-copy path when a copy writes memory owned by a GPU agent. Any open range on
// that agent is declined: the copy's write is not part of the recorded program, so a replayed pass
// would run without it. A host-replay range is declined only by a copy from another thread: its own
// copies re-run with every pass.
void
note_device_write(uint64_t agent_key);

// Called when a code object is about to be unloaded. Every bound packet range and host range is
// declined, because saved packets or executable globals may refer to freed code, and layouts cached
// by kernel object address are dropped, because a later load may reuse those addresses for other
// kernels.
void
note_code_object_unload();
}  // namespace range_replay
}  // namespace rocprofiler
