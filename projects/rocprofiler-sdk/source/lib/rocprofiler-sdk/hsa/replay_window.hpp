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

// Primitives shared by the replay mechanisms that re-execute recorded GPU work: the per-agent
// serialization lock, the drain helpers a replay window needs before it snapshots device memory,
// and the queue submission path used to re-submit recorded packets.
//
// Kernel replay (single dispatch, driven from inside WriteInterceptor) and range replay (a recorded
// sequence, driven from rocprofiler_range_replay_end) must agree on all of these: they share one
// lock per agent, and both must exclude the same concurrent GPU work from their snapshot->restore
// windows.

#include "lib/common/logging.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <fmt/format.h>
#include <hsa/hsa.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <string_view>
#include <utility>

namespace rocprofiler
{
namespace hsa
{
class Queue;
union rocprofiler_packet;

// Per-agent replay serialization (reader/writer). A replay's snapshot->restore window must exclude
// any concurrent GPU work on the agent that could mutate tracked device memory, but ordinary
// dispatches do not conflict with one another. So this is a shared_mutex:
//   * a replay window takes the UNIQUE (writer) lock for the whole drain->snap->passes->restore
//     sequence, excluding all other replays and every non-replay dispatch on the agent;
//   * a non-replay dispatch takes the SHARED (reader) lock across its submit, so many normal
//     dispatches still run concurrently while a pending replay writer waits for in-flight submits
//     to finish and blocks new ones from entering the window.
// Different agents use different mutexes and run concurrently; combined with agent-scoped snapshots
// this keeps multi-GPU replay isolated. The reader lock only bounds *submission*; the async GPU
// tail is drained by the replay window before it snapshots.
std::shared_mutex&
agent_replay_mutex(rocprofiler_agent_id_t agent_id);

// Count of dispatch submissions on the agent, advanced under the reader lock after the packets
// are written. A fence only covers work submitted before it, and a host range releases the agent
// between its fence and its writer lock (the fence needs the queue's ROCr intercept lock, which a
// waiting submitter can hold). Reading this before the fence and again under the writer lock tells
// the range whether any dispatch slipped into that gap.
std::atomic<uint64_t>&
agent_submit_epoch(rocprofiler_agent_id_t agent_id);

// A single blocking wait -- Queue::sync(), or the replay drain barrier -- gives up after one slice
// and reports whether it drained. The replay drain helpers retry for at most max_slices slices and
// report elapsed time in multiples of one slice, so every wait in that loop has to use the same
// slice length; hardcoding it separately would make those elapsed figures fiction as soon as one
// of them changed.
inline constexpr int  seconds_per_slice = 5;
inline constexpr int  max_slices        = 12;
inline constexpr auto drain_slice       = std::chrono::seconds{seconds_per_slice};
inline constexpr int  drain_budget_secs = seconds_per_slice * max_slices;

// How long a non-fatal replay wait may block: `slices` waits of at most `slice` each. The default
// is the drain budget above. Tests pass a shorter one so that a timeout takes milliseconds.
struct replay_wait_budget_t
{
    std::chrono::milliseconds slice  = drain_slice;
    int                       slices = max_slices;
};

// Retry `try_drain_once`, which must block for at most one budget.slice, up to budget.slices
// times, warning after each slice that ends undrained. Returns whether it drained.
template <typename TryDrainFn>
bool
replay_wait(TryDrainFn&& try_drain_once, std::string_view what, replay_wait_budget_t budget = {})
{
    for(int i = 0; i < budget.slices; ++i)
    {
        if(try_drain_once()) return true;
        ROCP_WARNING << fmt::format(
            "replay: still waiting for {} (~{}s elapsed)",
            what,
            std::chrono::duration_cast<std::chrono::seconds>(budget.slice * (i + 1)).count());
    }
    return false;
}

// replay_wait over the full drain budget, aborting if it never drains. A replay that cannot fence
// itself against concurrent GPU work has no safe way to continue: proceeding would snapshot or
// restore memory that another kernel is still writing.
template <typename TryDrainFn>
void
replay_wait_or_fatal(TryDrainFn&& try_drain_once, std::string_view what)
{
    if(replay_wait(std::forward<TryDrainFn>(try_drain_once), what)) return;
    ROCP_FATAL << fmt::format("replay: {} did not drain after ~{}s", what, drain_budget_secs);
}

// When a wait that can decline instead of aborting gives up. A host replay fence makes one per
// fence and passes it to each of its queue barriers and agent drains, so the fence as a whole gives
// up within one budget however many queues and agents it covers.
class replay_deadline_t
{
public:
    using clock = std::chrono::steady_clock;

    explicit replay_deadline_t(clock::duration budget = std::chrono::seconds{drain_budget_secs})
    : m_start{clock::now()}
    , m_budget{budget}
    {}

    clock::duration budget() const { return m_budget; }
    clock::duration elapsed() const { return clock::now() - m_start; }
    bool            expired() const { return elapsed() >= m_budget; }

    // How long the next blocking wait may take: one drain slice, cut short at the deadline.
    clock::duration next_wait() const
    {
        return std::clamp<clock::duration>(
            m_budget - elapsed(), clock::duration::zero(), drain_slice);
    }

private:
    clock::time_point m_start  = {};
    clock::duration   m_budget = {};
};

// Retry `try_wait(timeout)`, which must block for at most `timeout`, until it succeeds or
// `deadline` passes, warning after each drain slice spent waiting. It tries at least once, so work
// that has already finished succeeds even past the deadline. Returns whether it succeeded.
template <typename TryWaitFn>
bool
replay_wait_until(TryWaitFn&& try_wait, std::string_view what, const replay_deadline_t& deadline)
{
    auto warned_at = std::chrono::seconds{0};
    while(!try_wait(deadline.next_wait()))
    {
        if(deadline.expired()) return false;
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(deadline.elapsed());
        if(elapsed - warned_at < drain_slice) continue;
        warned_at = elapsed;
        ROCP_WARNING << fmt::format(
            "replay: still waiting for {} (~{}s elapsed)", what, elapsed.count());
    }
    return true;
}

// Drain a queue's in-flight async completion handler(s) during replay. Unlike Queue::sync()'s
// teardown use (warn once and proceed), a replay pass must NOT proceed while a handler is still
// running: PASS-EXIT, the tool's continue-decision, restore(), and the next submit would race the
// handler that is still emitting records, releasing signals, and dropping correlation-id refs.
// Each sync() call blocks up to one ~5s slice and reports whether the queue drained.
void
replay_drain_or_fatal(const Queue& queue);

// Drain every queue on `agent` before snapshotting, WITHOUT holding the queue-map lock across the
// wait. iterate_queues holds the queue-map read lock for the duration of its callback, so a
// per-sibling blocking drain there would block stream creation/destruction for the whole (up to
// ~60s) drain. Instead poll each queue's in-flight async count under a brief read lock and sleep
// between polls, so the map lock is held only for the microsecond poll -- never across the wait.
// Safe against concurrent queue destruction: a Queue is only dereferenced while the queue-map
// read lock is held. A queue that destroy has already removed from that map stays reachable
// through the retiring set until its completion handlers finish. The live and retiring sets are
// re-read every poll. The per-agent writer lock held by the replay window blocks new dispatches
// on the agent, so in-flight work only decreases and the poll converges. Returns false if
// handlers are still in flight at `deadline`.
bool
replay_drain_agent(hsa_agent_t agent, const replay_deadline_t& deadline = replay_deadline_t{});

// replay_drain_agent's poll over the full drain budget, fatal on a genuinely stuck queue (beta
// feature). Like replay_drain_or_fatal it waits in drain slices and warns after every slice that
// ends undrained, the last one included.
void
replay_drain_agent_or_fatal(hsa_agent_t agent);

// Write `count` packets into `queue`'s ring buffer and ring its doorbell -- the same submission the
// application itself performs, so whichever interception mechanism is installed sees them. Used to
// re-submit recorded packets from outside a WriteInterceptor call (range replay), where no
// interceptor-provided writer is in scope. Blocks, without a deadline, while the ring is full.
// Returns false if the queue's packet buffer is unavailable or its engine does not accept AQL
// packets.
//
// Callers submitting *recorded* packets must clear their completion signals first: re-firing a
// signal the application has already consumed corrupts its synchronization.
bool
replay_ring_submit(const Queue& queue, const rocprofiler_packet* packets, size_t count);

enum class barrier_wait_result_t
{
    completed,      // the barrier retired: all prior work on the queue finished
    not_submitted,  // the barrier could not be written, so nothing was waited on
    ring_full,      // no ring slot freed up before the deadline, so nothing was written
    timed_out,      // the barrier is still in the ring and will decrement `completion` later
};

// Submit a barrier packet carrying `completion` and wait until `deadline` for it, fencing the CPU
// against all GPU work previously submitted to `queue`. Waiting for a free ring slot counts against
// the same deadline; unlike replay_ring_submit, a slot is only claimed once it is free, so giving
// up leaves the ring untouched. After timed_out the packet processor still owns `completion`: the
// caller must neither reuse nor destroy it.
barrier_wait_result_t
replay_barrier_and_wait(const Queue&             queue,
                        hsa_signal_t             completion,
                        const replay_deadline_t& deadline = replay_deadline_t{});

// While set on this thread, WriteInterceptor forwards packets untouched. Range replay's re-submit
// path transforms the recorded packets once (via Queue::invoke_write_interceptor) and then writes
// the result to the ring, which re-enters interception on this same thread; without this flag the
// packets would be transformed twice (two pooled signals and two records per dispatch).
void
set_interceptor_passthrough(bool enabled);

bool
interceptor_passthrough();
}  // namespace hsa
}  // namespace rocprofiler
