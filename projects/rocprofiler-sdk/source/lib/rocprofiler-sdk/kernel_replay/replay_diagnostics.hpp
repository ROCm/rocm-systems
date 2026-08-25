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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// Admission control and per-dispatch reporting for the replay window.
//
// Two problems this solves. First, a replay can be *sound but useless* (the snapshot missed the
// application's real data) or *correct but ruinous* (the footprint dwarfs the kernel), and both are
// decidable before any device->host traffic is issued. Second, when a replay is declined or when
// its inputs were not in fact identical across passes, nothing in the output says so today -- a
// user reading counter rows cannot tell a clean replay from one that silently ran on mutated
// memory.
//
// Everything here is a free function over plain structs, per rocprofiler-sdk CONTRIBUTING guidance
// on internal interfaces.

#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace rocprofiler
{
namespace kernel_replay
{
// Why a dispatch that requested replay ran only once. `none` means it was replayed.
enum class decline_reason
{
    none = 0,
    untracked_memory,  // application data lives outside the snapshot (VMM / managed / fine-grained)
    footprint_budget,  // snapshot would not fit the configured host budget
    snapshot_failed,   // snap() could not capture a complete image
    queue_drain_stuck,  // this queue's prior GPU work did not complete within the drain bound
    agent_drain_stuck,  // sibling queues on the agent did not go idle within the drain bound
    pass_drain_stuck,   // a pass's async completion handler did not finish within the drain bound
    reentrant_dispatch  // a dispatch was submitted on the replaying agent from inside the window
};

const char*
to_string(decline_reason reason);

// Policy for a single replay window, resolved once from the environment.
//
// The two "strict" knobs exist because the evidence differs in strength. A live virtual-memory
// mapping is unambiguous -- nothing maps virtual memory unless the application asked for it -- so
// declining by default is right. A GPU-resident pool allocation that failed the coarse-grained test
// can also come from the runtime rather than the application, so that only warns unless the user
// opts in.
struct replay_policy_t
{
    // Decline when live virtual-memory mappings are found on the agent. Default true.
    // ROCPROFILER_KERNEL_REPLAY_DECLINE_ON_VMEM=0 to override.
    bool decline_on_vmem = true;

    // Decline when GPU-resident, non-snapshottable pool allocations are found. Default false.
    // ROCPROFILER_KERNEL_REPLAY_DECLINE_ON_UNTRACKED_POOL=1 to enable.
    bool decline_on_untracked_pool = false;

    // Maximum snapshot size, in bytes. 0 means "derive from MemAvailable" (see resolve_policy).
    // ROCPROFILER_KERNEL_REPLAY_MAX_SNAPSHOT_BYTES.
    size_t max_snapshot_bytes = 0;

    // Warn when the projected host-link traffic for the whole window exceeds this many seconds.
    // ROCPROFILER_KERNEL_REPLAY_WARN_SECONDS, 0 disables.
    double warn_seconds = 10.0;

    // Host-link bandwidth assumed when projecting copy time, in GB/s. The snapshot destination is
    // unpinned host memory, so this is deliberately a low single-digit number rather than the
    // link's pinned-transfer rate. ROCPROFILER_KERNEL_REPLAY_ASSUMED_GBPS.
    double assumed_gbps = 4.0;
};

const replay_policy_t&
resolve_policy();

// What the window observed for one dispatch. Populated incrementally as the window proceeds so a
// decline at any stage still reports the fields established up to that point.
struct replay_outcome_t
{
    rocprofiler_dispatch_id_t dispatch_id = 0;
    decline_reason            reason      = decline_reason::none;

    uint64_t requested_passes = 0;  // what pass_count_cb asked for (0 == indefinite)
    uint64_t executed_passes  = 0;  // how many actually ran

    size_t footprint_bytes   = 0;
    size_t footprint_regions = 0;

    memory_tracker::untracked_summary_t untracked{};

    double snap_seconds          = 0.0;
    double restore_total_seconds = 0.0;

    // Set when a tool callback submitted GPU work on the replaying agent during the window. The
    // counters for such a dispatch are not trustworthy even though the window completed.
    bool reentrancy_observed = false;
};

// Emit one machine-parseable line per replayed or declined dispatch, tagged `[kernel-replay]`, so a
// test or a user can tell from the log whether a dispatch was replayed, how much memory moved, and
// what was missing. Declines log at warning level, successful replays at info.
void
log_replay_outcome(const replay_outcome_t& outcome);

// Format the same fields as `key=value` pairs. Separated from logging so tests can assert on the
// exact text without capturing a logger.
std::string
format_replay_outcome(const replay_outcome_t& outcome);

// Decide whether the untracked-memory summary should block replay under `policy`.
decline_reason
check_untracked(const memory_tracker::untracked_summary_t& summary, const replay_policy_t& policy);

// Decide whether a snapshot of `footprint` over `passes` passes should be admitted. Emits the
// projected-time warning as a side effect when the window is admitted but expensive, because that
// is the only point where both the footprint and the pass count are known.
decline_reason
check_admission(const memory_snapshot::snapshot_footprint_t& footprint,
                uint64_t                                     passes,
                const replay_policy_t&                       policy);

// Thread-scoped marker for "this thread is inside a replay window on this agent".
//
// The per-agent replay lock is a std::shared_mutex, which is not recursive, and every dispatch on
// the agent takes one side of it while a replay service is active. A tool callback (CONFIG, PASS,
// pass_count_cb, replay_continue_cb) that launches GPU work therefore submits a dispatch from the
// one thread that already holds the writer lock. Unlike the drain waits there is no timeout: the
// acquisition never returns and the process is left blocked in a lock, unkillable by signal. Two
// distinct acquisitions can hit this, so the dispatch path consults in_replay_window() at both:
//
//   * the reader side of an ordinary dispatch, which is skipped, letting the dispatch through; and
//   * the writer side of a *nested* replay request, which cannot be skipped -- a window inside a
//     window has no meaning -- so that dispatch is declined with `reentrant_dispatch` and run once.
//
// Both trade a wrong measurement (reported via note_replay_reentrancy on the window's outcome line)
// for liveness. The agent is part of the state because the deadlock is per-agent: a callback that
// launches work on a *different* GPU contends for a different mutex and must keep its lock, or two
// concurrent replays on two agents would stop excluding each other's dispatches.
void
enter_replay_window(rocprofiler_agent_id_t agent);

void
exit_replay_window();

bool
in_replay_window(rocprofiler_agent_id_t agent);

// Record that a dispatch was submitted from inside this thread's replay window. Cleared by
// enter_replay_window so the flag describes one window only.
void
note_replay_reentrancy();

bool
replay_reentrancy_observed();

// True the first time it is called per process, so the (long) explanatory message is logged once
// rather than once per nested dispatch.
bool
should_warn_replay_reentrancy();
}  // namespace kernel_replay
}  // namespace rocprofiler
