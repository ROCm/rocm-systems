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

#include "lib/rocprofiler-sdk/range_replay/host_replay.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/replay_window.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/local_context.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/range_replay/digest.hpp"
#include "lib/rocprofiler-sdk/range_replay/executor.hpp"
#include "lib/rocprofiler-sdk/range_replay/replay_callbacks.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/format.h>
#include <hsa/hsa.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace range_replay
{
namespace host
{
namespace
{
using phase_clock = std::chrono::steady_clock;

double
elapsed_ms(phase_clock::duration elapsed)
{
    return std::chrono::duration<double, std::milli>{elapsed}.count();
}

// Every GPU agent, in agent-id order. The order is also the lock order for the restore.
std::vector<host_agent_state_t>
gpu_agents()
{
    auto out = std::vector<host_agent_state_t>{};
    for(const auto* agent : agent::get_agents())
    {
        if(agent == nullptr || agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;
        const auto hsa_agent = agent::get_hsa_agent(agent);
        if(!hsa_agent) continue;

        auto state      = host_agent_state_t{};
        state.hsa_agent = *hsa_agent;
        state.agent_id  = agent->id;
        out.emplace_back(std::move(state));
    }
    std::sort(out.begin(), out.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.agent_id.handle < rhs.agent_id.handle;
    });
    return out;
}

void
fence_queue(const hsa::Queue& queue)
{
    auto       signal        = hsa_signal_t{.handle = 0};
    auto*      pooled_signal = hsa::Queue::create_signal(0, &signal, /*use_pool=*/true);
    const auto _release      = common::scope_destructor{[&]() {
        if(pooled_signal != nullptr)
            hsa::Queue::release_signal(pooled_signal);
        else if(signal.handle != 0)
            queue.core_api().hsa_signal_destroy_fn(signal);
    }};

    if(signal.handle == 0 || !hsa::replay_barrier_and_wait(queue, signal))
        ROCP_WARNING << "range replay: host replay could not fence a queue with a barrier; "
                        "relying on the application having synchronized the device";
}

// Wait for every GPU command submitted to `agent` and for the completion handlers behind it. The
// queues are collected under the queue-map lock and fenced after it is released: the barrier goes
// through the interceptor, which must not run under that lock.
void
fence_agent(hsa_agent_t agent)
{
    auto queues = std::vector<const hsa::Queue*>{};
    if(auto* controller = hsa::get_queue_controller(); controller != nullptr)
    {
        controller->iterate_queues([&](const hsa::Queue* queue) {
            if(queue != nullptr && queue->get_agent().get_hsa_agent().handle == agent.handle)
                queues.emplace_back(queue);
        });
    }
    for(const auto* queue : queues)
        fence_queue(*queue);
    hsa::replay_drain_agent_or_fatal(agent);
}

void
fence_all(const std::vector<host_agent_state_t>& agents)
{
    for(const auto& agent : agents)
        fence_agent(agent.hsa_agent);
}

// Device digests of every agent followed by the host regions', so two points in time compare
// positionally. Only meaningful while the allocation sets are unchanged, which finish_pass checks
// first.
bool
state_digests(const host_range_state_t& host, std::vector<digest::region_digests_t>& out)
{
    out.clear();
    for(const auto& agent : host.agents)
    {
        const auto snapshot = kernel_replay::memory_snapshot::snap(agent.hsa_agent);
        if(!snapshot.ok) return false;
        out.emplace_back(snapshot_digests(snapshot));
    }
    auto regions = digest::region_digests_t{};
    for(const auto& region : host.regions)
        regions.emplace_back(digest::hash_bytes(region.ptr, region.saved.size()));
    out.emplace_back(std::move(regions));
    return true;
}

void
store_exit_digests(host_range_state_t& host)
{
    auto digests = std::vector<digest::region_digests_t>{};
    if(!state_digests(host, digests)) return;
    for(size_t i = 0; i < host.agents.size(); ++i)
        host.agents[i].exit_digests = std::move(digests[i]);
    for(size_t i = 0; i < host.regions.size(); ++i)
        host.regions[i].exit_digest = digests.back()[i];
    host.exit_digests_taken = true;
}

uint64_t
count_divergence(const host_range_state_t& host)
{
    auto digests = std::vector<digest::region_digests_t>{};
    if(!host.exit_digests_taken || !state_digests(host, digests)) return 0;

    auto divergent = uint64_t{0};
    for(size_t i = 0; i < host.agents.size(); ++i)
        divergent += digest::count_divergent(host.agents[i].exit_digests, digests[i]);
    for(size_t i = 0; i < host.regions.size(); ++i)
        if(host.regions[i].exit_digest != digests.back()[i]) ++divergent;
    return divergent;
}

bool
allocations_unchanged(const host_range_state_t& host)
{
    return std::all_of(host.agents.begin(), host.agents.end(), [](const auto& agent) {
        return tracked_pointers(agent.hsa_agent) == agent.tracked;
    });
}

void
restore_devices(const host_range_state_t& host)
{
    // Writer locks in agent-id order (gpu_agents() sorted them), so a packet replay or kernel
    // replay on any of these agents cannot interleave with the restore.
    auto guards = std::vector<std::unique_lock<std::shared_mutex>>{};
    guards.reserve(host.agents.size());
    for(const auto& agent : host.agents)
        guards.emplace_back(hsa::agent_replay_mutex(agent.agent_id));

    for(const auto& agent : host.agents)
    {
        ROCP_FATAL_IF(!kernel_replay::memory_snapshot::restore(agent.snapshot)) << fmt::format(
            "range replay: host replay could not restore the range-entry snapshot of agent {} "
            "(partial host->device copy); aborting rather than re-running the range over "
            "corrupted device memory",
            agent.agent_id.handle);
    }
}
}  // namespace

host_region_t
capture_region(void* ptr, size_t size)
{
    auto region = host_region_t{};
    region.ptr  = ptr;
    region.saved.resize(size);
    if(size > 0) std::memcpy(region.saved.data(), ptr, size);
    return region;
}

void
restore_regions(const std::vector<host_region_t>& regions)
{
    for(const auto& region : regions)
        if(!region.saved.empty()) std::memcpy(region.ptr, region.saved.data(), region.saved.size());
}

void
enter_range(range_context_t& ctx, std::vector<pending_host_region_t>&& pending_regions)
{
    set_host_mode(ctx);
    auto& host = ctx.host;
    host.pass  = 0;

    // Without extra passes there is nothing to restore later, so the range only reports CLOSE.
    if(!ctx.plan.replay_requested || !ctx.record.eligible()) return;

    const auto start = phase_clock::now();

    // Taken before the snapshot, so every code object at or below it has its module variables in
    // the snapshot.
    host.code_object_watermark = newest_code_object_id();
    host.agents                = gpu_agents();

    fence_all(host.agents);
    const auto fenced_at = phase_clock::now();

    auto device_bytes = size_t{0};
    for(auto& agent : host.agents)
    {
        agent.snapshot = kernel_replay::memory_snapshot::snap(agent.hsa_agent);
        if(!agent.snapshot.ok)
        {
            ROCP_WARNING << fmt::format(
                "range replay: host replay snapshot of agent {} failed (memory pressure or copy "
                "error); range {} will not be replayed",
                agent.agent_id.handle,
                ctx.record.range_id());
            ctx.record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);
            host.agents.clear();
            return;
        }
        agent.tracked = tracked_pointers(agent.hsa_agent);
        device_bytes += snapshot_bytes(agent.snapshot);
    }

    auto host_bytes = size_t{0};
    for(const auto& region : pending_regions)
    {
        host.regions.emplace_back(capture_region(region.ptr, region.size));
        host_bytes += region.size;
    }
    host.snapshot_taken = true;

    ROCP_INFO << fmt::format("range replay: host range {} entry: fence {:.3f} ms, snapshot {:.3f} "
                             "ms ({} device bytes on {} agents, {} host bytes in {} regions)",
                             ctx.record.range_id(),
                             elapsed_ms(fenced_at - start),
                             elapsed_ms(phase_clock::now() - fenced_at),
                             device_bytes,
                             host.agents.size(),
                             host_bytes,
                             host.regions.size());
}

void
track_region(range_context_t& ctx, void* ptr, size_t size)
{
    auto& host = ctx.host;
    if(!host.enabled || host.pass != 0 || !host.snapshot_taken) return;
    host.regions.emplace_back(capture_region(ptr, size));
}

step_result_t
finish_pass(range_context_t& ctx)
{
    auto&      host           = ctx.host;
    const auto pass           = host.pass;
    const auto dispatch_count = static_cast<uint64_t>(ctx.record.observed_dispatch_count());

    // The pass's GPU tail belongs to the pass: its records are emitted before PASS exit, and the
    // checks below see settled memory.
    if(host.snapshot_taken) fence_all(host.agents);

    if(pass > 0)
        execute_pass_phase_exit(ctx.plan, pass, ctx.agent_id, dispatch_count, host.pass_state);

    auto finish = [&](rocprofiler_range_replay_status_t status, uint64_t divergence = 0) {
        host.local_context.reset();
        return step_result_t{0, status, divergence};
    };

    fold_external_decline(ctx);
    if(!ctx.record.eligible()) return finish(ctx.record.status());
    if(!ctx.plan.replay_requested) return finish(ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT);
    if(!host.snapshot_taken) return finish(ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);

    if(pass == 0)
    {
        if(dispatch_count == 0) return finish(ROCPROFILER_RANGE_REPLAY_STATUS_NO_DISPATCH);
        // HIP loads a module on the first launch of one of its kernels, so a range that is the
        // first to use a kernel loads its code object here. Declining would refuse ordinary
        // applications; the cost of accepting is that module variables of such a code object are
        // not rewound, which only matters to kernels that write __device__ globals.
        if(newest_code_object_id() > host.code_object_watermark)
            ROCP_WARNING << fmt::format(
                "range replay: host range {} loaded a code object during its first execution; its "
                "__device__ / __constant__ variables are not rewound between passes (launch each "
                "kernel once before the range to include them)",
                ctx.record.range_id());
    }

    // A restore into an allocation set that differs from the snapshot's would write into memory
    // the application has repurposed, or leave a new allocation un-rewound.
    if(!allocations_unchanged(host))
        return finish(ROCPROFILER_RANGE_REPLAY_STATUS_ALLOCATION_CHANGED_IN_RANGE);

    if(pass == 0 && divergence_check_enabled()) store_exit_digests(host);

    // The application's own run always gets at least one re-execution once the tool asked for
    // replay, matching packet replay, which never consults replay_continue_cb before pass 1.
    const bool is_final = !ctx.plan.indefinite && (pass + 1 >= ctx.plan.total_passes);
    if(pass > 0 && !should_continue_replay(ctx.plan, pass, is_final))
        return finish(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED, count_divergence(host));

    const auto next = pass + 1;
    if(next > static_cast<uint64_t>(INT_MAX))
    {
        ROCP_WARNING << "range replay: host replay pass count exceeds the siglongjmp value range; "
                        "stopping";
        return finish(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED, count_divergence(host));
    }

    const auto restore_start = phase_clock::now();
    restore_devices(host);
    restore_regions(host.regions);
    ROCP_TRACE << fmt::format("range replay: host range {} restore before pass {}: {:.3f} ms",
                              ctx.record.range_id(),
                              next,
                              elapsed_ms(phase_clock::now() - restore_start));

    // Created on the first re-executed pass and kept for the rest of the range, so toggles are
    // sticky across passes exactly as in packet replay. It is installed for the whole host pass,
    // not just around the tool callback, because the dispatches that must honor it are the
    // application's.
    if(!host.local_context)
        host.local_context = std::make_shared<kernel_replay::scoped_local_context_control>(
            context::get_active_contexts());

    host.pass_state = pass_context_state_t{};
    execute_pass_phase_enter(ctx.plan,
                             next,
                             ctx.agent_id,
                             dispatch_count,
                             ctx.thread_id,
                             ctx.internal_corr_id,
                             ctx.ancestor_corr_id,
                             host.pass_state);
    host.pass = next;

    return step_result_t{next, ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED, 0};
}
}  // namespace host
}  // namespace range_replay
}  // namespace rocprofiler
