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

#include <chrono>
#include <cstddef>
#include <shared_mutex>
#include <string_view>

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

// A single blocking wait -- Queue::sync(), or the replay drain barrier -- gives up after one slice
// and reports whether it drained. The replay drain helpers retry for at most max_slices slices and
// report elapsed time in multiples of one slice, so every wait in that loop has to use the same
// slice length; hardcoding it separately would make those elapsed figures fiction as soon as one
// of them changed.
inline constexpr int  seconds_per_slice = 5;
inline constexpr int  max_slices        = 12;
inline constexpr auto drain_slice       = std::chrono::seconds{seconds_per_slice};
inline constexpr int  drain_budget_secs = seconds_per_slice * max_slices;

// Retry `try_drain_once` in drain_slice-sized waits, warning per slice and aborting after
// drain_budget_secs. A replay that cannot fence itself against concurrent GPU work has no safe
// way to continue: proceeding would snapshot or restore memory that another kernel is still
// writing.
template <typename TryDrainFn>
void
replay_wait_or_fatal(TryDrainFn&& try_drain_once, std::string_view what)
{
    // One try_drain_once() call blocks for at most one slice, so the two together bound the wait
    // at drain_budget_secs.
    for(int i = 0; i < max_slices; ++i)
    {
        if(try_drain_once()) return;
        ROCP_WARNING << fmt::format(
            "replay: still waiting for {} (~{}s elapsed)", what, (i + 1) * seconds_per_slice);
    }
    ROCP_FATAL << fmt::format("replay: {} did not drain after ~{}s", what, drain_budget_secs);
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
// Safe against concurrent queue destruction: a Queue is only dereferenced while the read lock is
// held (destroy_queue erases under the write lock), and the live set is re-read every poll. The
// per-agent writer lock held by the replay window blocks new dispatches on the agent, so in-flight
// work only decreases and the poll converges; fatal on a genuinely stuck queue (beta feature),
// matching replay_drain_or_fatal.
void
replay_drain_agent_or_fatal(hsa_agent_t agent);

// Write `count` packets into `queue`'s ring buffer and ring its doorbell -- the same submission the
// application itself performs, so whichever interception mechanism is installed sees them. Used to
// re-submit recorded packets from outside a WriteInterceptor call (range replay), where no
// interceptor-provided writer is in scope. Blocks while the ring is full. Returns false if the
// queue's packet buffer is unavailable.
//
// Callers submitting *recorded* packets must clear their completion signals first: re-firing a
// signal the application has already consumed corrupts its synchronization.
bool
replay_ring_submit(const Queue& queue, const rocprofiler_packet* packets, size_t count);

// Submit a barrier packet carrying `completion` and wait for it, fencing the CPU against all GPU
// work previously submitted to `queue`. Returns false if the submission itself failed.
bool
replay_barrier_and_wait(const Queue& queue, hsa_signal_t completion);

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
