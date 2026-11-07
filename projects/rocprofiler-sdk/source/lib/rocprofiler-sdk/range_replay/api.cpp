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

#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_interposition.hpp"
#include "lib/rocprofiler-sdk/range_replay/executor.hpp"
#include "lib/rocprofiler-sdk/range_replay/host_replay.hpp"
#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"
#include "lib/rocprofiler-sdk/range_replay/replay_callbacks.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <csetjmp>
#include <cstdint>

namespace rocprofiler
{
namespace range_replay
{
namespace
{
void
close_range(range_context_t&                  ctx,
            rocprofiler_range_replay_status_t status,
            uint64_t                          divergence_count)
{
    execute_close_callback(ctx.plan,
                           status,
                           ctx.agent_id,
                           ctx.record.observed_dispatch_count(),
                           divergence_count,
                           ctx.thread_id,
                           ctx.internal_corr_id,
                           ctx.ancestor_corr_id);
}

// Everything end() does except the jump, so that every C++ object it creates is destroyed before
// the caller calls siglongjmp. Returns the host pass to jump into, 0 once the range is closed, and
// -1 when no range is open on this thread.
int64_t
end_range()
{
    auto* live = current_range();
    if(live == nullptr) return -1;

    if(live->host.enabled)
    {
        const auto step = host::finish_pass(*live);
        if(step.next_pass > 0) return static_cast<int64_t>(step.next_pass);

        auto ctx = range_context_t{};
        if(!take_range(ctx)) return -1;
        close_range(ctx, step.status, step.divergence_count);
        return 0;
    }

    auto ctx = range_context_t{};
    if(!take_range(ctx)) return -1;

    auto       divergence_count = uint64_t{0};
    const auto status           = execute_range(ctx, divergence_count);
    close_range(ctx, status, divergence_count);
    return 0;
}
}  // namespace
}  // namespace range_replay
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_range_replay_begin(uint64_t range_id)
{
    namespace range_replay = ::rocprofiler::range_replay;

    // Consumed before any early return, so an arming is never carried to a later begin() whose
    // sigsetjmp frame may be gone.
    const bool host_jmp_armed  = range_replay::take_host_jmp_armed();
    auto       pending_regions = range_replay::take_pending_host_regions();

    if(!range_replay::has_registered_range_replay_context())
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    if(!range_replay::has_active_range_replay_contexts())
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED;

    if(range_replay::current_range() != nullptr) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    if(!range_replay::open_range(range_id)) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto* ctx = range_replay::current_range();

    const auto* corr_id = ::rocprofiler::context::get_latest_correlation_id();
    ctx->thread_id = (corr_id != nullptr) ? corr_id->thread_idx : ::rocprofiler::common::get_tid();
    ctx->internal_corr_id = (corr_id != nullptr) ? corr_id->internal : 0;
    ctx->ancestor_corr_id = (corr_id != nullptr) ? corr_id->ancestor : 0;

    ctx->plan = range_replay::execute_config_callback(
        range_id, ctx->thread_id, ctx->internal_corr_id, ctx->ancestor_corr_id);

    // Both replay paths depend on the interception path in hsa/queue.cpp: packet replay to
    // re-submit, host replay to see dispatches from other threads. The write-pointer
    // virtualization path has its own submission handling, which this beta does not cover, so a
    // range opened under it is declined rather than replayed.
    if(::rocprofiler::hsa::queue_interposition::supports_queue_interposition())
    {
        static std::atomic<bool> _warned{false};
        if(!_warned.exchange(true, std::memory_order_relaxed))
            ROCP_WARNING << "range replay: not supported with queue interposition enabled; ranges "
                            "will be recorded and declined";
        ctx->record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_UNSUPPORTED_QUEUE_PATH);
    }

    if(ctx->plan.host_replay)
    {
        if(host_jmp_armed)
            range_replay::host::enter_range(*ctx, std::move(pending_regions));
        else
            ctx->record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_HOST_JMP_NOT_SET);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

void*
rocprofiler_range_replay_host_jmpbuf(void)
{
    return ::rocprofiler::range_replay::host_jmpbuf();
}

rocprofiler_status_t
rocprofiler_range_replay_host_track_memory(void* ptr, uint64_t size)
{
    namespace range_replay = ::rocprofiler::range_replay;

    if(ptr == nullptr || size == 0) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    if(auto* ctx = range_replay::current_range(); ctx != nullptr)
        range_replay::host::track_region(*ctx, ptr, static_cast<size_t>(size));
    else
        range_replay::add_pending_host_region(ptr, static_cast<size_t>(size));
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_range_replay_get_host_pass(uint64_t* pass)
{
    if(pass == nullptr) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    const auto* ctx = ::rocprofiler::range_replay::current_range();
    *pass           = (ctx != nullptr && ctx->host.enabled) ? ctx->host.pass : 0;
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_range_replay_end(void)
{
    const auto next = ::rocprofiler::range_replay::end_range();
    if(next < 0) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    if(next > 0)
    {
        // Only PODs are live in this frame: siglongjmp does not run C++ destructors.
        const int value = static_cast<int>(std::min<int64_t>(next, INT_MAX));
        siglongjmp(  // NOLINT(cert-err52-cpp)
            *static_cast<sigjmp_buf*>(::rocprofiler::range_replay::host_jmp_target()),
            value);
    }
    return ROCPROFILER_STATUS_SUCCESS;
}
}
