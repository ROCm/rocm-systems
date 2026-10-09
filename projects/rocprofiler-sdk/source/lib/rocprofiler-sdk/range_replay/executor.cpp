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
#include "lib/rocprofiler-sdk/range_replay/retained_kernarg.hpp"

#include <fmt/format.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
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

// A kernarg block for one replay pass: every recorded dispatch's arguments, laid out back to back
// with each kernel's alignment honored. The block is retained for the agent's next range when this
// one finishes (see retained_kernarg.hpp).
class kernarg_staging
{
public:
    kernarg_staging() = default;
    ~kernarg_staging() { reset(); }

    kernarg_staging(const kernarg_staging&) = delete;
    kernarg_staging& operator=(const kernarg_staging&) = delete;
    kernarg_staging(kernarg_staging&&)                 = delete;
    kernarg_staging& operator=(kernarg_staging&&) = delete;

    // Reserve space for `dispatches` and record each one's offset, reusing the block retained from
    // an earlier range on this agent when it fits. Returns false if the kernarg pool allocation
    // fails.
    bool reserve(const hsa::Queue&                       queue,
                 rocprofiler_agent_id_t                  agent_id,
                 const std::vector<recorded_dispatch_t>& dispatches)
    {
        auto placement = plan_kernarg_layout(dispatches);
        m_offsets      = std::move(placement.offsets);
        m_agent_id     = agent_id;

        const auto total = placement.total;
        if(total == 0) return true;

        m_block = take_retained_kernarg_block(agent_id);
        if(kernarg_block_fits(m_block.capacity, total))
        {
            m_reused = true;
            return true;
        }
        free_kernarg_block(m_block);

        const auto& ext  = queue.ext_api();
        const auto  pool = queue.get_agent().kernarg_pool();
        if(pool.handle == 0 || ext.hsa_amd_memory_pool_allocate_fn == nullptr) return false;

        void* base = nullptr;
        if(ext.hsa_amd_memory_pool_allocate_fn(pool, total, 0, &base) != HSA_STATUS_SUCCESS ||
           base == nullptr)
            return false;

        m_block    = kernarg_block_t{base, total, ext.hsa_amd_memory_pool_free_fn};
        auto agent = queue.get_agent().get_hsa_agent();
        if(ext.hsa_amd_agents_allow_access_fn != nullptr)
            ext.hsa_amd_agents_allow_access_fn(1, &agent, nullptr, base);

        return true;
    }

    size_t capacity() const { return m_block.capacity; }

    std::string_view origin() const
    {
        if(m_block.base == nullptr) return "none";
        return m_reused ? "reused" : "allocated";
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

            auto* slot = static_cast<uint8_t*>(m_block.base) + m_offsets[i];
            std::memcpy(slot, dispatches[i].kernarg.data(), dispatches[i].kernarg.size());
            packets[i].kernel_dispatch.kernarg_address = slot;
        }
    }

    // Must not run while a pass submitted from this block can still be executing: the next range
    // on the agent refills it.
    void reset()
    {
        if(m_block.base != nullptr)
            retain_kernarg_block(m_agent_id, std::exchange(m_block, kernarg_block_t{}));
        m_reused = false;
    }

private:
    kernarg_block_t        m_block    = {};
    rocprofiler_agent_id_t m_agent_id = {.handle = 0};
    bool                   m_reused   = false;
    std::vector<size_t>    m_offsets  = {};
};

using phase_clock = std::chrono::steady_clock;

double
elapsed_ms(phase_clock::duration elapsed)
{
    return std::chrono::duration<double, std::milli>{elapsed}.count();
}

size_t
snapshot_bytes(const snapshot_t& snapshot)
{
    auto total = size_t{0};
    for(const auto& block : snapshot.blocks)
        total += block.host_copy.size();
    return total;
}
}  // namespace

kernarg_placement_t
plan_kernarg_layout(const std::vector<recorded_dispatch_t>& dispatches)
{
    auto out = kernarg_placement_t{};
    out.offsets.reserve(dispatches.size());

    for(const auto& dispatch : dispatches)
    {
        out.offsets.emplace_back(out.total);
        out.total += ((dispatch.kernarg.size() + kKernargAlignment - 1) / kKernargAlignment) *
                     kKernargAlignment;
    }

    return out;
}

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

    const auto entry_start = phase_clock::now();

    // Fence against work already in flight so the snapshot sees settled memory: first this queue
    // (a barrier packet through the interceptor's writer, exactly as the kernel-replay window
    // does), then every other queue on the agent. The signal comes from the SDK's pool because
    // creating one costs a KFD event allocation, which a range per iteration would pay every
    // iteration.
    auto       drain_signal        = hsa_signal_t{.handle = 0};
    auto*      pooled_drain_signal = hsa::Queue::create_signal(0, &drain_signal, /*use_pool=*/true);
    const auto _release_signal     = common::scope_destructor{[&]() {
        if(pooled_drain_signal != nullptr)
            hsa::Queue::release_signal(pooled_drain_signal);
        else if(drain_signal.handle != 0)
            queue.core_api().hsa_signal_destroy_fn(drain_signal);
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
    const auto drained_at = phase_clock::now();

    ctx.snapshot          = kernel_replay::memory_snapshot::snap(ctx.hsa_agent);
    ctx.snapshot_taken    = true;
    const auto snapped_at = phase_clock::now();

    if(!ctx.snapshot.ok)
    {
        ROCP_WARNING << "range replay: snapshot capture failed (memory pressure or copy error); "
                        "this range will not be replayed";
        ctx.record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED);
        ctx.snapshot = snapshot_t{};
        return;
    }

    ctx.snapshot_ptrs = tracked_pointers(ctx.hsa_agent);

    ROCP_INFO << fmt::format("range replay: range {} entry: drain {:.3f} ms, snapshot {:.3f} ms "
                             "({} bytes in {} regions)",
                             ctx.record.range_id(),
                             elapsed_ms(drained_at - entry_start),
                             elapsed_ms(snapped_at - drained_at),
                             snapshot_bytes(ctx.snapshot),
                             ctx.snapshot.blocks.size());
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

    // Wall time of each phase of the window, reported at INFO once the window closes, so the lock
    // and drain waits, the exit snapshot, the passes and the restores can be told apart.
    const auto window_start = phase_clock::now();

    // Exclude every other replay and every non-replay dispatch on this agent for the whole window.
    const auto replay_guard =
        std::unique_lock<std::shared_mutex>{hsa::agent_replay_mutex(ctx.agent_id)};

    // The application's own execution of the range is complete as far as the host is concerned, but
    // its GPU tail may not be. Drain before touching device memory.
    hsa::replay_drain_or_fatal(queue);
    hsa::replay_drain_agent_or_fatal(ctx.hsa_agent);
    const auto drained_at = phase_clock::now();

    // Every region the entry snapshot covers must still be the same allocation, or restoring it
    // would write into memory the application has since repurposed.
    if(tracked_pointers(ctx.hsa_agent) != ctx.snapshot_ptrs)
        return ROCPROFILER_RANGE_REPLAY_STATUS_ALLOCATION_CHANGED_IN_RANGE;

    // The state the application must resume with, captured before the first pass overwrites it.
    const auto snap_start    = phase_clock::now();
    const auto exit_snapshot = kernel_replay::memory_snapshot::snap(ctx.hsa_agent);
    const auto snapped_at    = phase_clock::now();
    if(!exit_snapshot.ok) return ROCPROFILER_RANGE_REPLAY_STATUS_SNAPSHOT_FAILED;

    auto staging = kernarg_staging{};
    if(!staging.reserve(queue, ctx.agent_id, ctx.record.dispatches()))
        return ROCPROFILER_RANGE_REPLAY_STATUS_STAGING_FAILED;
    const auto staged_at = phase_clock::now();

    auto packets = build_pass_packets(ctx.record.dispatches());

    // Localized context control for this range's replay loop (shared with kernel replay): connects
    // the tool's PASS toggles to the services that read them at dispatch, without touching global
    // context state.
    auto local_ctx_tls_guard =
        kernel_replay::scoped_local_context_control{context::get_active_contexts()};

    // Mark the thread as replaying for the duration: the queue path uses it to skip re-recording
    // the packets we submit and to skip the per-agent reader lock we already hold as a writer.
    set_this_thread_replaying(true);
    set_this_thread_replaying_agent(ctx.agent_id);
    tl_submit_queue     = &queue;
    const auto _restore = common::scope_destructor{[]() {
        set_this_thread_replaying(false);
        set_this_thread_replaying_agent(std::nullopt);
        tl_submit_queue = nullptr;
    }};

    const auto agent_id       = ctx.agent_id;
    const auto dispatch_count = static_cast<uint64_t>(ctx.record.dispatch_count());

    auto     passes_time   = phase_clock::duration{};
    auto     restores_time = phase_clock::duration{};
    uint64_t passes_run    = 0;
    uint64_t restores_run  = 0;

    // Pass 0 was the application's own execution, so the replayed passes are numbered from 1.
    for(uint64_t pass = 1;; ++pass)
    {
        const bool is_final = !ctx.plan.indefinite && (pass == ctx.plan.total_passes - 1);

        // Rewind to the state the range started from. A failed restore leaves device memory
        // partially written; continuing would submit a pass over corrupted data and hand that
        // corruption back to the application.
        const auto restore_start = phase_clock::now();
        ROCP_FATAL_IF(!kernel_replay::memory_snapshot::restore(ctx.snapshot)) << fmt::format(
            "range replay: restore of the range-entry snapshot failed (partial host->device copy); "
            "aborting rather than continuing with corrupted device memory");
        const auto pass_start = phase_clock::now();
        restores_time += pass_start - restore_start;
        ++restores_run;

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
        passes_time += phase_clock::now() - pass_start;
        ++passes_run;

        if(!should_continue_replay(ctx.plan, pass, is_final)) break;
    }

    const auto verify_start = phase_clock::now();
    if(divergence_check_enabled())
    {
        const auto post_snapshot = kernel_replay::memory_snapshot::snap(ctx.hsa_agent);
        if(post_snapshot.ok)
            divergence_count = digest::count_divergent(snapshot_digests(exit_snapshot),
                                                       snapshot_digests(post_snapshot));
    }
    const auto verified_at = phase_clock::now();

    // Hand the application back the state its own execution produced, whether or not the replay
    // reproduced it.
    ROCP_FATAL_IF(!kernel_replay::memory_snapshot::restore(exit_snapshot)) << fmt::format(
        "range replay: restore of the range-exit snapshot failed (partial host->device copy); "
        "aborting rather than returning corrupted device memory to the application");
    const auto window_end = phase_clock::now();

    ROCP_INFO << fmt::format(
        "range replay: range {} ({} dispatches) phases: lock+drain {:.3f} ms, exit snapshot {:.3f} "
        "ms ({} bytes in {} regions), kernarg staging {:.3f} ms ({} bytes, {}), {} passes {:.3f} "
        "ms, {} restores {:.3f} ms, verify {}, exit restore {:.3f} ms, window {:.3f} ms",
        ctx.record.range_id(),
        dispatch_count,
        elapsed_ms(drained_at - window_start),
        elapsed_ms(snapped_at - snap_start),
        snapshot_bytes(exit_snapshot),
        exit_snapshot.blocks.size(),
        elapsed_ms(staged_at - snapped_at),
        staging.capacity(),
        staging.origin(),
        passes_run,
        elapsed_ms(passes_time),
        restores_run,
        elapsed_ms(restores_time),
        divergence_check_enabled()
            ? fmt::format("{:.3f} ms", elapsed_ms(verified_at - verify_start))
            : std::string{"off"},
        elapsed_ms(window_end - verified_at),
        elapsed_ms(window_end - window_start));

    return ROCPROFILER_RANGE_REPLAY_STATUS_REPLAYED;
}
}  // namespace range_replay
}  // namespace rocprofiler
