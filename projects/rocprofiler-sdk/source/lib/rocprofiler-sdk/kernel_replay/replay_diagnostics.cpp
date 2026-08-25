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

#include "lib/rocprofiler-sdk/kernel_replay/replay_diagnostics.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <atomic>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
namespace
{
constexpr double bytes_per_gb = 1.0e9;

// MemAvailable from /proc/meminfo, in bytes; 0 when it cannot be read. Used only to pick a default
// snapshot budget, so a failure to read it degrades to "no budget" rather than to a wrong answer.
size_t
host_mem_available_bytes()
{
    auto stream = std::ifstream{"/proc/meminfo"};
    if(!stream.is_open()) return 0;

    auto label = std::string{};
    while(stream >> label)
    {
        if(label != "MemAvailable:")
        {
            stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        size_t kib = 0;
        if(!(stream >> kib)) return 0;
        return kib * 1024;
    }
    return 0;
}

replay_policy_t
build_policy()
{
    auto policy = replay_policy_t{};

    policy.decline_on_vmem =
        common::get_env("ROCPROFILER_KERNEL_REPLAY_DECLINE_ON_VMEM", policy.decline_on_vmem);
    policy.decline_on_untracked_pool = common::get_env(
        "ROCPROFILER_KERNEL_REPLAY_DECLINE_ON_UNTRACKED_POOL", policy.decline_on_untracked_pool);
    policy.warn_seconds =
        common::get_env("ROCPROFILER_KERNEL_REPLAY_WARN_SECONDS", policy.warn_seconds);
    policy.assumed_gbps =
        common::get_env("ROCPROFILER_KERNEL_REPLAY_ASSUMED_GBPS", policy.assumed_gbps);
    policy.max_snapshot_bytes =
        common::get_env("ROCPROFILER_KERNEL_REPLAY_MAX_SNAPSHOT_BYTES", size_t{0});

    if(policy.max_snapshot_bytes == 0)
    {
        // Half of what the host currently has available. The snapshot is one full copy of the
        // footprint held for the whole window, and Linux overcommits, so the observable failure of
        // guessing high is the OOM killer rather than the std::bad_alloc that snap() handles
        // gracefully. Leaving headroom keeps the failure inside our own error path.
        const auto avail = host_mem_available_bytes();
        if(avail > 0) policy.max_snapshot_bytes = avail / 2;
    }

    if(!(policy.assumed_gbps > 0.0)) policy.assumed_gbps = replay_policy_t{}.assumed_gbps;

    return policy;
}
}  // namespace

const char*
to_string(decline_reason reason)
{
    switch(reason)
    {
        case decline_reason::none: return "none";
        case decline_reason::untracked_memory: return "untracked_memory";
        case decline_reason::footprint_budget: return "footprint_budget";
        case decline_reason::snapshot_failed: return "snapshot_failed";
        case decline_reason::queue_drain_stuck: return "queue_drain_stuck";
        case decline_reason::agent_drain_stuck: return "agent_drain_stuck";
        case decline_reason::pass_drain_stuck: return "pass_drain_stuck";
        case decline_reason::reentrant_dispatch: return "reentrant_dispatch";
    }
    return "unknown";
}

const replay_policy_t&
resolve_policy()
{
    // Read once. A tool cannot change these mid-run, and re-reading the environment (and
    // /proc/meminfo) per dispatch would put a file open in the replay gate.
    static const auto policy = build_policy();
    return policy;
}

std::string
format_replay_outcome(const replay_outcome_t& outcome)
{
    return fmt::format(
        "dispatch_id={} outcome={} reason={} requested_passes={} executed_passes={} "
        "footprint_bytes={} footprint_regions={} untracked_vmem_bytes={} untracked_vmem_regions={} "
        "untracked_pool_bytes={} untracked_pool_regions={} snap_seconds={:.4f} "
        "restore_seconds={:.4f} reentrancy={}",
        outcome.dispatch_id,
        (outcome.reason == decline_reason::none ? "replayed" : "declined"),
        to_string(outcome.reason),
        outcome.requested_passes,
        outcome.executed_passes,
        outcome.footprint_bytes,
        outcome.footprint_regions,
        outcome.untracked.vmem_bytes,
        outcome.untracked.vmem_regions,
        outcome.untracked.pool_bytes,
        outcome.untracked.pool_regions,
        outcome.snap_seconds,
        outcome.restore_total_seconds,
        (outcome.reentrancy_observed ? 1 : 0));
}

void
log_replay_outcome(const replay_outcome_t& outcome)
{
    const auto line = fmt::format("[kernel-replay] {}", format_replay_outcome(outcome));

    if(outcome.reason != decline_reason::none || outcome.reentrancy_observed)
        ROCP_WARNING << line;
    else
        ROCP_INFO << line;
}

decline_reason
check_untracked(const memory_tracker::untracked_summary_t& summary, const replay_policy_t& policy)
{
    if(summary.vmem_regions > 0)
    {
        // Naming the allocators is the point of this message: the user's next action is to change a
        // build option or an environment variable, and they cannot do that from "replay declined".
        ROCP_WARNING << fmt::format(
            "kernel replay: {} live virtual-memory mapping(s) totalling {} bytes are visible to "
            "this agent. Their contents are not part of the snapshot, so passes after the first "
            "would run on mutated inputs. This is what hipMallocAsync with the default pool, "
            "hipMemAddressReserve/hipMemMap, PyTorch PYTORCH_HIP_ALLOC_CONF=expandable_segments, "
            "and Kokkos built with KOKKOS_ENABLE_IMPL_HIP_MALLOC_ASYNC (the default for HIP < 7.0) "
            "all produce. {}",
            summary.vmem_regions,
            summary.vmem_bytes,
            (policy.decline_on_vmem
                 ? "Declining replay for this dispatch; set "
                   "ROCPROFILER_KERNEL_REPLAY_DECLINE_ON_VMEM=0 to replay anyway and accept that "
                   "the results may be wrong"
                 : "ROCPROFILER_KERNEL_REPLAY_DECLINE_ON_VMEM=0 is set, so replay proceeds and the "
                   "results may be wrong"));

        if(policy.decline_on_vmem) return decline_reason::untracked_memory;
    }

    if(summary.pool_regions > 0)
    {
        ROCP_INFO << fmt::format(
            "kernel replay: {} GPU-resident allocation(s) totalling {} bytes are not snapshottable "
            "(fine-grained, managed, or IPC-imported memory). Runtime-internal allocations also "
            "land here, so this only blocks replay when "
            "ROCPROFILER_KERNEL_REPLAY_DECLINE_ON_UNTRACKED_POOL=1",
            summary.pool_regions,
            summary.pool_bytes);

        if(policy.decline_on_untracked_pool) return decline_reason::untracked_memory;
    }

    return decline_reason::none;
}

decline_reason
check_admission(const memory_snapshot::snapshot_footprint_t& footprint,
                uint64_t                                     passes,
                const replay_policy_t&                       policy)
{
    if(policy.max_snapshot_bytes > 0 && footprint.bytes > policy.max_snapshot_bytes)
    {
        ROCP_WARNING << fmt::format(
            "kernel replay: declining -- a snapshot of this agent needs {} bytes across {} "
            "region(s), over the {} byte budget. Restrict replay with kernel filtering, reduce the "
            "application's resident footprint, or raise "
            "ROCPROFILER_KERNEL_REPLAY_MAX_SNAPSHOT_BYTES",
            footprint.bytes,
            footprint.regions,
            policy.max_snapshot_bytes);
        return decline_reason::footprint_budget;
    }

    // Projected host-link traffic: one capture plus one restore before each subsequent pass. An
    // indefinite loop (passes == 0) is projected as two passes, the minimum that restores at all.
    const auto effective_passes = (passes == 0) ? uint64_t{2} : passes;
    const auto projected_bytes =
        static_cast<double>(footprint.bytes) * static_cast<double>(effective_passes);
    const auto projected_seconds = projected_bytes / (policy.assumed_gbps * bytes_per_gb);

    if(policy.warn_seconds > 0.0 && projected_seconds > policy.warn_seconds)
    {
        ROCP_WARNING << fmt::format(
            "kernel replay: this dispatch will move about {} bytes across the host link ({} bytes "
            "x {} passes), roughly {:.1f}s at {:.1f} GB/s, before the kernel's own time. If that "
            "is not what you intended, restrict replay with kernel filtering or request fewer "
            "counter groups",
            static_cast<size_t>(projected_bytes),
            footprint.bytes,
            effective_passes,
            projected_seconds,
            policy.assumed_gbps);
    }

    return decline_reason::none;
}

namespace
{
struct window_frame_t
{
    rocprofiler_agent_id_t agent{};
    bool                   reentrancy_observed = false;
};

// The replay windows this thread has open, innermost last. Empty when the thread is not inside one.
// A vector rather than a single value so a window opened on another agent from inside a callback
// does not erase this thread's knowledge of the enclosing window (see the header).
std::vector<window_frame_t>&
window_stack()
{
    thread_local auto frames = std::vector<window_frame_t>{};
    return frames;
}

window_frame_t*
find_frame(rocprofiler_agent_id_t agent)
{
    auto& frames = window_stack();
    // Innermost first: with one frame per agent at most, the first match is the only match, and
    // searching from the top is the cheapest order for the common single-window case.
    for(auto itr = frames.rbegin(); itr != frames.rend(); ++itr)
        if(itr->agent.handle == agent.handle) return &(*itr);
    return nullptr;
}

std::atomic<bool> reentrancy_warning_emitted = false;
}  // namespace

void
enter_replay_window(rocprofiler_agent_id_t agent)
{
    window_stack().push_back(window_frame_t{agent, false});
}

void
exit_replay_window(rocprofiler_agent_id_t agent)
{
    auto& frames = window_stack();
    if(frames.empty()) return;

    // Windows are opened and closed by a scope guard, so the frame being closed is the innermost
    // one. Anything else means the guards were not nested, which would leave a stale frame behind
    // and deadlock the next dispatch on that agent, so say so rather than silently unwind.
    ROCP_ERROR_IF(frames.back().agent.handle != agent.handle) << fmt::format(
        "kernel replay: closing the replay window for agent {} but the innermost open window is "
        "for agent {}; replay window guards must nest",
        agent.handle,
        frames.back().agent.handle);

    frames.pop_back();
}

bool
in_replay_window(rocprofiler_agent_id_t agent)
{
    return find_frame(agent) != nullptr;
}

void
note_replay_reentrancy(rocprofiler_agent_id_t agent)
{
    if(auto* frame = find_frame(agent)) frame->reentrancy_observed = true;
}

bool
replay_reentrancy_observed(rocprofiler_agent_id_t agent)
{
    const auto* frame = find_frame(agent);
    return frame != nullptr && frame->reentrancy_observed;
}

bool
should_warn_replay_reentrancy()
{
    return !reentrancy_warning_emitted.exchange(true, std::memory_order_relaxed);
}
}  // namespace kernel_replay
}  // namespace rocprofiler
