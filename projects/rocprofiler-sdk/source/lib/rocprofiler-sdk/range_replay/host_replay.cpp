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
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
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
#include <new>
#include <shared_mutex>
#include <stdexcept>
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
    if(queue.get_engine_type() != hsa::Queue::engine_type::compute) return;
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

// Fence compute queues while excluding destruction, but without holding the queue-map lock.
// Copies and non-compute work must already have been synchronized by the application.
void
fence_agent(hsa_agent_t agent)
{
    if(auto* controller = hsa::get_queue_controller(); controller != nullptr)
    {
        controller->iterate_queues_with_lifetime([&](const hsa::Queue* queue) {
            if(queue != nullptr && queue->get_agent().get_hsa_agent().handle == agent.handle)
                fence_queue(*queue);
        });
    }
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
        return kernel_replay::memory_tracker::allocations_match(agent.hsa_agent, agent.tracked);
    });
}

std::vector<std::unique_lock<std::shared_mutex>>
lock_agents(const host_range_state_t& host)
{
    auto guards = std::vector<std::unique_lock<std::shared_mutex>>{};
    guards.reserve(host.agents.size());
    for(const auto& agent : host.agents)
        guards.emplace_back(hsa::agent_replay_mutex(agent.agent_id));
    return guards;
}

std::vector<uint64_t>
submit_epochs(const host_range_state_t& host)
{
    auto epochs = std::vector<uint64_t>{};
    epochs.reserve(host.agents.size());
    for(const auto& agent : host.agents)
        epochs.emplace_back(
            hsa::agent_submit_epoch(agent.agent_id).load(std::memory_order_acquire));
    return epochs;
}

// Fences without re-running anything between: a tool's PASS exit or continue callback, which runs
// after the pass's fence, can itself submit GPU work. Bounded so a thread that never stops
// submitting declines the range instead of starving it.
constexpr int max_refences = 8;

// Writer locks over a fence that no dispatch overtook. `fenced_epochs` were read before that
// fence. A dispatch whose epoch advance is visible under the writer locks wrote its packets after
// the read, so the fence may not cover it: release, fence again, and re-check. Every submit
// advances the epoch under the reader lock, so with the writer locks held nothing is mid-submit.
std::vector<std::unique_lock<std::shared_mutex>>
lock_after_fence(range_context_t& ctx, std::vector<uint64_t> fenced_epochs)
{
    for(int refences = 0;; ++refences)
    {
        auto guards = lock_agents(ctx.host);
        if(submit_epochs(ctx.host) == fenced_epochs) return guards;
        if(refences == max_refences)
        {
            ROCP_WARNING << fmt::format("range replay: host range {} kept seeing new dispatches "
                                        "after {} fences; it will not be replayed",
                                        ctx.record.range_id(),
                                        refences + 1);
            ctx.record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_CONCURRENT_DISPATCH);
            return guards;
        }
        guards.clear();
        fenced_epochs = submit_epochs(ctx.host);
        fence_all(ctx.host.agents);
    }
}

rocprofiler_range_replay_status_t
restore_entry_state(range_context_t& ctx)
{
    const auto& host = ctx.host;
    // The caller holds every agent's writer lock, taken after its final fence, through this
    // restore. Frees do not take those locks. Validate every agent before the first copy and keep
    // the inventory read-locked across all copies so a free/reallocation cannot reuse an entry
    // address.
    return kernel_replay::memory_tracker::inventory().rlock([&](const auto& inventory) {
        fold_external_decline(ctx);
        if(!ctx.record.eligible()) return ctx.record.status();
        for(const auto& agent : host.agents)
            if(!kernel_replay::memory_tracker::allocations_match(
                   agent.hsa_agent, agent.tracked, inventory))
                return ROCPROFILER_RANGE_REPLAY_STATUS_ALLOCATION_CHANGED_IN_RANGE;

        for(const auto& agent : host.agents)
        {
            ROCP_FATAL_IF(!kernel_replay::memory_snapshot::restore(agent.snapshot, inventory))
                << fmt::format(
                       "range replay: host replay could not restore the range-entry snapshot of "
                       "agent {} (partial host->device copy); aborting rather than re-running the "
                       "range over corrupted device memory",
                       agent.agent_id.handle);
        }
        restore_regions(host.regions);
        return ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED;
    });
}
}  // namespace

host_region_t
capture_region(void* ptr, size_t size)
{
    auto        region = host_region_t{};
    const auto* bytes  = static_cast<const uint8_t*>(ptr);
    region.ptr         = ptr;
    if(size > 0) region.saved.assign(bytes, bytes + size);
    return region;
}

// Newest first, so a byte covered by several regions ends with its earliest capture, the one
// closest to range entry.
void
restore_regions(const std::vector<host_region_t>& regions)
{
    for(auto itr = regions.rbegin(); itr != regions.rend(); ++itr)
        if(!itr->saved.empty()) std::memcpy(itr->ptr, itr->saved.data(), itr->saved.size());
}

void
enter_range(range_context_t& ctx, std::vector<pending_host_region_t>&& pending_regions)
try
{
    set_host_mode(ctx);
    auto& host = ctx.host;
    host.pass  = 0;

    // Without extra passes there is nothing to restore later, so the range only reports CLOSE.
    if(!ctx.plan.replay_requested || !ctx.record.eligible()) return;

    const auto start = phase_clock::now();

    host.agents = gpu_agents();

    // Fence before taking the writer locks: the barrier rings the intercept queue's doorbell, which
    // needs that queue's ROCr intercept lock. Another thread can hold that lock while it waits in
    // WriteInterceptor for this agent's reader lock.
    const auto epochs = submit_epochs(host);
    fence_all(host.agents);
    const auto fenced_at = phase_clock::now();

    // A foreign dispatch publishes its decline before it takes the reader lock, so after the
    // writer locks are held, any dispatch that slipped in after the fence is visible here.
    const auto guards = lock_after_fence(ctx, epochs);
    fold_external_decline(ctx);
    if(!ctx.record.eligible())
    {
        host.agents.clear();
        return;
    }

    // Drain callbacks before taking executable lifetime protection: a callback can unload a module.
    const auto executable_guard = std::shared_lock{code_object::executable_lifetime_mutex()};
    host.code_object_watermark  = newest_code_object_id();

    auto device_bytes = size_t{0};
    for(auto& agent : host.agents)
    {
        agent.tracked  = kernel_replay::memory_tracker::snap_allocation_identity(agent.hsa_agent);
        agent.snapshot = kernel_replay::memory_snapshot::snap(agent.hsa_agent);
        if(!agent.snapshot.ok ||
           !kernel_replay::memory_tracker::allocations_match(agent.hsa_agent, agent.tracked))
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
} catch(const std::bad_alloc&)
{
    ctx.record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);
    ctx.host.snapshot_taken = false;
    ctx.host.agents.clear();
    ctx.host.regions.clear();
} catch(const std::length_error&)
{
    ctx.record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);
    ctx.host.snapshot_taken = false;
    ctx.host.agents.clear();
    ctx.host.regions.clear();
}

void
track_region(range_context_t& ctx, void* ptr, size_t size)
{
    auto& host = ctx.host;
    if(!host.enabled || host.pass != 0 || !host.snapshot_taken || !ctx.record.eligible()) return;
    try
    {
        host.regions.emplace_back(capture_region(ptr, size));
    } catch(const std::bad_alloc&)
    {
        ctx.record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);
        host.regions.clear();
    } catch(const std::length_error&)
    {
        ctx.record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);
        host.regions.clear();
    }
}

step_result_t
finish_pass(range_context_t& ctx)
{
    auto&      host           = ctx.host;
    const auto pass           = host.pass;
    const auto dispatch_count = static_cast<uint64_t>(ctx.record.observed_dispatch_count());

    // The pass's GPU tail belongs to the pass: its records are emitted before PASS exit, and the
    // checks below see settled memory. The writer locks are taken just before the restore.
    const auto epochs = submit_epochs(host);
    if(host.snapshot_taken) fence_all(host.agents);
    // Writes the application made during the pass are already in this count. A tool callback that
    // copies afterwards moves it, and that copy is not waited on by the compute-queue fence.
    const auto writes_at_fence =
        ctx.external ? ctx.external->owning_device_writes.load(std::memory_order_relaxed) : 0;

    if(pass > 0)
        execute_pass_phase_exit(ctx.plan, pass, ctx.agent_id, dispatch_count, host.pass_state);

    // Unload publishes declines before retiring metadata under exclusive lifetime ownership.
    // Hold shared ownership from the final decline check through globals capture/restore.
    auto executable_guard = std::shared_lock{code_object::executable_lifetime_mutex()};

    auto finish = [&](rocprofiler_range_replay_status_t status, uint64_t divergence = 0) {
        host.local_context.reset();
        return step_result_t{0, status, divergence};
    };

    auto decline_if_callback_copied = [&]() {
        if(ctx.external != nullptr &&
           ctx.external->owning_device_writes.load(std::memory_order_relaxed) != writes_at_fence)
            ctx.record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_MEMORY_COPY_IN_RANGE);
    };

    // PASS exit and the continuation callback run after the pass fence. Dispatches they submit
    // advance the epoch and are fenced again here, before divergence or the restore observes
    // memory.
    auto settle_callback_gpu_work = [&]() {
        if(executable_guard.owns_lock()) executable_guard.unlock();
        auto guards = lock_after_fence(ctx, epochs);
        guards.clear();
        executable_guard.lock();
        fold_external_decline(ctx);
        decline_if_callback_copied();
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
    if(pass > 0)
    {
        // A continuation callback may unload an executable. Never call it under lifetime
        // protection.
        executable_guard.unlock();
        const bool continue_replay = should_continue_replay(ctx.plan, pass, is_final);
        executable_guard.lock();
        fold_external_decline(ctx);
        if(!ctx.record.eligible()) return finish(ctx.record.status());
        if(!allocations_unchanged(host))
            return finish(ROCPROFILER_RANGE_REPLAY_STATUS_ALLOCATION_CHANGED_IN_RANGE);
        if(!continue_replay)
        {
            settle_callback_gpu_work();
            if(!ctx.record.eligible()) return finish(ctx.record.status());
            return finish(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED, count_divergence(host));
        }
    }

    const auto next = pass + 1;
    if(next > static_cast<uint64_t>(INT_MAX))
    {
        ROCP_WARNING << "range replay: host replay pass count exceeds the siglongjmp value range; "
                        "stopping";
        settle_callback_gpu_work();
        if(!ctx.record.eligible()) return finish(ctx.record.status());
        return finish(ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED, count_divergence(host));
    }

    // Writer locks in gpu_agents() order, taken without the lifetime lock held. They are not held
    // across the fence or the tool callbacks above: the fence barrier needs the queue's ROCr
    // intercept lock, which a foreign submitter can hold while it waits for the reader lock, and a
    // tool callback that submits on this thread would wait for its own writer lock.
    // restore_entry_state re-reads the decline channel under them, and a foreign dispatch publishes
    // its decline before taking the reader lock, so none can be in flight during the restore. GPU
    // work the callbacks submitted after the fence is caught by the epoch check and fenced again.
    if(executable_guard.owns_lock()) executable_guard.unlock();
    auto guards = lock_after_fence(ctx, epochs);
    executable_guard.lock();
    fold_external_decline(ctx);
    decline_if_callback_copied();
    if(!ctx.record.eligible())
    {
        guards.clear();
        return finish(ctx.record.status());
    }

    const auto restore_start  = phase_clock::now();
    const auto restore_status = restore_entry_state(ctx);
    if(restore_status != ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED) return finish(restore_status);
    executable_guard.unlock();
    guards.clear();
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

    // Visible to the PASS enter callback. Leaving it at the previous pass made that callback's
    // dispatches count as pass 0 and made host_track_memory capture regions on a re-executed pass.
    host.pass       = next;
    host.pass_state = pass_context_state_t{};
    execute_pass_phase_enter(ctx.plan,
                             next,
                             ctx.agent_id,
                             dispatch_count,
                             ctx.thread_id,
                             ctx.internal_corr_id,
                             ctx.ancestor_corr_id,
                             host.pass_state);

    return step_result_t{next, ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED, 0};
}

void
abandon_pass(range_context_t& ctx)
{
    auto& host = ctx.host;
    if(host.snapshot_taken) fence_all(host.agents);
    if(host.pass > 0)
        execute_pass_phase_exit(ctx.plan,
                                host.pass,
                                ctx.agent_id,
                                static_cast<uint64_t>(ctx.record.observed_dispatch_count()),
                                host.pass_state);
    host.local_context.reset();
}
}  // namespace host
}  // namespace range_replay
}  // namespace rocprofiler
