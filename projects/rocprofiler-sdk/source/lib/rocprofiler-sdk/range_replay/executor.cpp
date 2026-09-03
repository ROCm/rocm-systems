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

#include "lib/rocprofiler-sdk/range_replay/executor.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/replay_window.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/local_context.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"
#include "lib/rocprofiler-sdk/range_replay/digest.hpp"
#include "lib/rocprofiler-sdk/range_replay/replay_callbacks.hpp"

#include <fmt/format.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace range_replay
{
namespace
{
using snapshot_t = kernel_replay::memory_snapshot::device_snapshot_t;

// Queue used by the ring writer below. The interceptor's writer signature carries no user data, so
// the target queue is passed through thread-local state for the duration of one submission.
thread_local const hsa::Queue* tl_submit_queue = nullptr;

// Interceptor packet writer for replayed passes: puts the *transformed* packets on the ring the
// application submits through. Writing to that ring re-enters interception on this thread, so the
// passthrough flag is held across the write to keep the packets from being transformed twice.
void
ring_writer(const void* packets, uint64_t count)
{
    if(tl_submit_queue == nullptr) return;

    hsa::set_interceptor_passthrough(true);
    const auto _clear = common::scope_destructor{[]() { hsa::set_interceptor_passthrough(false); }};

    hsa::replay_ring_submit(
        *tl_submit_queue, static_cast<const hsa::rocprofiler_packet*>(packets), count);
}

std::vector<void*>
tracked_pointers(hsa_agent_t agent)
{
    auto inventory = kernel_replay::memory_tracker::snap_inventory(agent);
    auto out       = std::vector<void*>{};
    out.reserve(inventory.size());
    for(const auto& [ptr, size] : inventory)
        out.emplace_back(ptr);
    std::sort(out.begin(), out.end());
    return out;
}

// Per-region digests of a snapshot's host copies, ordered by device address so two snapshots of the
// same regions compare positionally (the inventory itself is unordered).
digest::region_digests_t
snapshot_digests(const snapshot_t& snapshot)
{
    auto keyed = std::vector<std::pair<const void*, uint64_t>>{};
    keyed.reserve(snapshot.blocks.size());
    for(const auto& block : snapshot.blocks)
        keyed.emplace_back(block.gpu_addr,
                           digest::hash_bytes(block.host_copy.data(), block.host_copy.size()));

    std::sort(keyed.begin(), keyed.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    auto out = digest::region_digests_t{};
    out.reserve(keyed.size());
    for(const auto& [addr, hash] : keyed)
        out.emplace_back(hash);
    return out;
}

// A kernarg block for one replay pass: every recorded dispatch's arguments, laid out back to back
// with each kernel's alignment honored.
class kernarg_staging
{
public:
    kernarg_staging() = default;
    ~kernarg_staging() { reset(); }

    kernarg_staging(const kernarg_staging&) = delete;
    kernarg_staging& operator=(const kernarg_staging&) = delete;
    kernarg_staging(kernarg_staging&&)                 = delete;
    kernarg_staging& operator=(kernarg_staging&&) = delete;

    // Reserve space for `dispatches` and record each one's offset. Returns false if the kernarg
    // pool allocation fails.
    bool reserve(const hsa::Queue& queue, const std::vector<recorded_dispatch_t>& dispatches)
    {
        constexpr size_t alignment = 256;  // AQL kernarg segments are 256-byte aligned

        m_offsets.reserve(dispatches.size());
        size_t total = 0;
        for(const auto& dispatch : dispatches)
        {
            m_offsets.emplace_back(total);
            total += ((dispatch.kernarg.size() + alignment - 1) / alignment) * alignment;
        }

        if(total == 0) return true;

        const auto& ext  = queue.ext_api();
        const auto  pool = queue.get_agent().kernarg_pool();
        if(pool.handle == 0 || ext.hsa_amd_memory_pool_allocate_fn == nullptr) return false;

        if(ext.hsa_amd_memory_pool_allocate_fn(pool, total, 0, &m_base) != HSA_STATUS_SUCCESS ||
           m_base == nullptr)
            return false;

        m_free_fn  = ext.hsa_amd_memory_pool_free_fn;
        auto agent = queue.get_agent().get_hsa_agent();
        if(ext.hsa_amd_agents_allow_access_fn != nullptr)
            ext.hsa_amd_agents_allow_access_fn(1, &agent, nullptr, m_base);

        return true;
    }

    // Copy the recorded argument bytes into the block and point each packet at its slot. Done once
    // per pass so a pass never observes another pass's kernarg contents.
    void fill(const std::vector<recorded_dispatch_t>& dispatches,
              std::vector<hsa::rocprofiler_packet>&   packets) const
    {
        for(size_t i = 0; i < dispatches.size(); ++i)
        {
            if(dispatches[i].kernarg.empty())
            {
                packets[i].kernel_dispatch.kernarg_address = nullptr;
                continue;
            }

            auto* slot = static_cast<uint8_t*>(m_base) + m_offsets[i];
            std::memcpy(slot, dispatches[i].kernarg.data(), dispatches[i].kernarg.size());
            packets[i].kernel_dispatch.kernarg_address = slot;
        }
    }

    void reset()
    {
        if(m_base != nullptr && m_free_fn != nullptr) m_free_fn(m_base);
        m_base = nullptr;
    }

private:
    void* m_base                     = nullptr;
    hsa_status_t (*m_free_fn)(void*) = nullptr;
    std::vector<size_t> m_offsets    = {};
};

// Build the packet list for a pass: the recorded packets, forced to execute one at a time. The
// barrier bit makes each dispatch wait for the previous one, which is stricter than the application
// (packets may have been free to overlap) and is why replayed passes are not a faithful source of
// concurrency-sensitive timings.
std::vector<hsa::rocprofiler_packet>
build_pass_packets(const std::vector<recorded_dispatch_t>& dispatches)
{
    auto packets = std::vector<hsa::rocprofiler_packet>{};
    packets.reserve(dispatches.size());
    for(const auto& dispatch : dispatches)
    {
        auto packet = dispatch.packet;
        packet.kernel_dispatch.header |= (1U << HSA_PACKET_HEADER_BARRIER);
        packets.emplace_back(packet);
    }
    return packets;
}
}  // namespace

bool
divergence_check_enabled()
{
    static const bool _enabled = common::get_env("ROCPROF_RANGE_REPLAY_VERIFY", false);
    return _enabled;
}

void
ensure_entry_snapshot(range_context_t&                      ctx,
                      const hsa::Queue&                     queue,
                      hsa_amd_queue_intercept_packet_writer writer)
{
    if(ctx.snapshot_taken || !ctx.record.eligible() || ctx.queue == nullptr) return;
    if(!ctx.plan.replay_requested) return;

    // Fence against work already in flight so the snapshot sees settled memory: first this queue
    // (a barrier packet through the interceptor's writer, exactly as the kernel-replay window
    // does), then every other queue on the agent.
    auto drain_signal = hsa_signal_t{.handle = 0};
    hsa::Queue::create_signal(0, &drain_signal, /*use_pool=*/false);
    const auto _destroy_signal = common::scope_destructor{[&]() {
        if(drain_signal.handle != 0) queue.core_api().hsa_signal_destroy_fn(drain_signal);
    }};

    if(writer != nullptr && drain_signal.handle != 0)
    {
        auto barrier   = hsa_barrier_and_packet_t{};
        barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier.header |= (1U << HSA_PACKET_HEADER_BARRIER);
        barrier.completion_signal = drain_signal;

        const auto packet = hsa::rocprofiler_packet{barrier};
        writer(&packet, 1);

        using namespace std::chrono_literals;
        const auto& core = queue.core_api();
        hsa::replay_wait_or_fatal(
            [&]() {
                return core.hsa_signal_wait_scacquire_fn(drain_signal,
                                                         HSA_SIGNAL_CONDITION_EQ,
                                                         0,
                                                         std::chrono::nanoseconds{5s}.count(),
                                                         HSA_WAIT_STATE_BLOCKED) == 0;
            },
            "this queue's prior GPU work");
    }

    hsa::replay_drain_agent_or_fatal(ctx.hsa_agent);

    ctx.snapshot       = kernel_replay::memory_snapshot::snap(ctx.hsa_agent);
    ctx.snapshot_taken = true;

    if(!ctx.snapshot.ok)
    {
        ROCP_WARNING << "range replay: snapshot capture failed (memory pressure or copy error); "
                        "this range will not be replayed";
        ctx.record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);
        ctx.snapshot = snapshot_t{};
        return;
    }

    ctx.snapshot_ptrs = tracked_pointers(ctx.hsa_agent);
}

rocprofiler_range_replay_status_t
execute_range(range_context_t& ctx, uint64_t& divergence_count)
{
    divergence_count = 0;

    if(!ctx.record.eligible()) return ctx.record.status();
    if(!ctx.plan.replay_requested) return ROCPROFILER_RANGE_REPLAY_STATUS_NO_PASS_COUNT;
    if(ctx.record.dispatch_count() == 0 || ctx.queue == nullptr)
        return ROCPROFILER_RANGE_REPLAY_STATUS_NO_DISPATCH;
    if(!ctx.snapshot_taken || !ctx.snapshot.ok)
        return ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED;

    const auto& queue = *ctx.queue;

    // Exclude every other replay and every non-replay dispatch on this agent for the whole window.
    const auto replay_guard =
        std::unique_lock<std::shared_mutex>{hsa::agent_replay_mutex(ctx.agent_id)};

    // The application's own execution of the range is complete as far as the host is concerned, but
    // its GPU tail may not be. Drain before touching device memory.
    hsa::replay_drain_or_fatal(queue);
    hsa::replay_drain_agent_or_fatal(ctx.hsa_agent);

    // Every region the entry snapshot covers must still be the same allocation, or restoring it
    // would write into memory the application has since repurposed.
    if(tracked_pointers(ctx.hsa_agent) != ctx.snapshot_ptrs)
        return ROCPROFILER_RANGE_REPLAY_STATUS_ALLOCATION_CHANGED_IN_RANGE;

    // The state the application must resume with, captured before the first pass overwrites it.
    const auto exit_snapshot = kernel_replay::memory_snapshot::snap(ctx.hsa_agent);
    if(!exit_snapshot.ok) return ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED;

    auto staging = kernarg_staging{};
    if(!staging.reserve(queue, ctx.record.dispatches()))
        return ROCPROFILER_RANGE_REPLAY_STATUS_STAGING_FAILED;

    auto packets = build_pass_packets(ctx.record.dispatches());

    // Localized context control for this range's replay loop (shared with kernel replay): connects
    // the tool's PASS toggles to the services that read them at dispatch, without touching global
    // context state.
    auto local_ctx_tls_guard =
        kernel_replay::scoped_local_context_control{context::get_active_contexts()};

    // Mark the thread as replaying for the duration: the queue path uses it to skip re-recording
    // the packets we submit and to skip the per-agent reader lock we already hold as a writer.
    set_this_thread_replaying(true);
    tl_submit_queue     = &queue;
    const auto _restore = common::scope_destructor{[]() {
        set_this_thread_replaying(false);
        tl_submit_queue = nullptr;
    }};

    const auto agent_id       = ctx.agent_id;
    const auto dispatch_count = static_cast<uint64_t>(ctx.record.dispatch_count());

    // Pass 0 was the application's own execution, so the replayed passes are numbered from 1.
    for(uint64_t pass = 1;; ++pass)
    {
        const bool is_final = !ctx.plan.indefinite && (pass == ctx.plan.total_passes - 1);

        // Rewind to the state the range started from. A failed restore leaves device memory
        // partially written; continuing would submit a pass over corrupted data and hand that
        // corruption back to the application.
        ROCP_FATAL_IF(!kernel_replay::memory_snapshot::restore(ctx.snapshot)) << fmt::format(
            "range replay: restore of the range-entry snapshot failed (partial host->device copy); "
            "aborting rather than continuing with corrupted device memory");

        auto pass_state = pass_context_state_t{};
        execute_pass_phase_enter(ctx.plan,
                                 pass,
                                 agent_id,
                                 dispatch_count,
                                 ctx.thread_id,
                                 ctx.internal_corr_id,
                                 ctx.ancestor_corr_id,
                                 pass_state);

        staging.fill(ctx.record.dispatches(), packets);
        queue.invoke_write_interceptor(packets.data(), packets.size(), ring_writer);

        // Drain this pass's async completion handlers (a separate HSA thread reads counters, emits
        // records, and releases signals) before the continue-decision, the next restore, and the
        // next submit.
        hsa::replay_drain_or_fatal(queue);

        execute_pass_phase_exit(ctx.plan, pass, agent_id, dispatch_count, pass_state);

        if(!should_continue_replay(ctx.plan, pass, is_final)) break;
    }

    if(divergence_check_enabled())
    {
        const auto post_snapshot = kernel_replay::memory_snapshot::snap(ctx.hsa_agent);
        if(post_snapshot.ok)
            divergence_count = digest::count_divergent(snapshot_digests(exit_snapshot),
                                                       snapshot_digests(post_snapshot));
    }

    // Hand the application back the state its own execution produced, whether or not the replay
    // reproduced it.
    ROCP_FATAL_IF(!kernel_replay::memory_snapshot::restore(exit_snapshot)) << fmt::format(
        "range replay: restore of the range-exit snapshot failed (partial host->device copy); "
        "aborting rather than returning corrupted device memory to the application");

    return ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED;
}
}  // namespace range_replay
}  // namespace rocprofiler
