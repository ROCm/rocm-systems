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
#include "lib/common/scope_destructor.hpp"
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
#include <limits>

namespace rocprofiler
{
namespace range_replay
{
namespace
{
// How the extra passes of a host-replay range re-run the application's code.
enum class host_rerun_t
{
    jump,  // siglongjmp to the target ROCPROFILER_RANGE_REPLAY_HOST_BEGIN armed
    body,  // rocprofiler_range_replay_host_run calls the body again
};

rocprofiler_status_t
begin_range(uint64_t range_id, host_rerun_t rerun)
{
    // Consumed before any early return, so an arming is never carried to a later begin() whose
    // sigsetjmp frame may be gone.
    const bool host_jmp_armed  = take_host_jmp_armed();
    auto       pending_regions = take_pending_host_regions();

    if(!has_registered_range_replay_context()) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    if(!has_active_range_replay_contexts()) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED;

    if(current_range() != nullptr) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    if(!open_range(range_id)) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto* ctx               = current_range();
    ctx->opened_by_host_run = (rerun == host_rerun_t::body);

    const auto* corr_id   = context::get_latest_correlation_id();
    ctx->thread_id        = (corr_id != nullptr) ? corr_id->thread_idx : common::get_tid();
    ctx->internal_corr_id = (corr_id != nullptr) ? corr_id->internal : 0;
    ctx->ancestor_corr_id = (corr_id != nullptr) ? corr_id->ancestor : 0;

    ctx->plan = execute_config_callback(
        range_id, ctx->thread_id, ctx->internal_corr_id, ctx->ancestor_corr_id);

    // Both replay paths depend on the interception path in hsa/queue.cpp: packet replay to
    // re-submit, host replay to see dispatches from other threads. The write-pointer
    // virtualization path has its own submission handling, which this beta does not cover, so a
    // range opened under it is declined rather than replayed.
    if(hsa::queue_interposition::supports_queue_interposition())
    {
        static std::atomic<bool> _warned{false};
        if(!_warned.exchange(true, std::memory_order_relaxed))
            ROCP_WARNING << "range replay: not supported with queue interposition enabled; ranges "
                            "will be recorded and declined";
        ctx->record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_UNSUPPORTED_QUEUE_PATH);
    }

    if(ctx->plan.host_replay)
    {
        if(rerun == host_rerun_t::body || host_jmp_armed)
            host::enter_range(*ctx, std::move(pending_regions));
        else
            ctx->record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_HOST_JMP_NOT_SET);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

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
        // finish_pass copied its status before this fold. A decline published in that window
        // (foreign dispatch, device write, executable unload) is in the record and must be what
        // CLOSE reports. Statuses finish_pass returns without declining the record, such as
        // NO_PASS_COUNT, stay as step.status while the record is still eligible.
        const auto status = ctx.record.eligible() ? step.status : ctx.record.status();
        close_range(ctx, status, step.divergence_count);
        return 0;
    }

    auto ctx = range_context_t{};
    if(!take_range(ctx)) return -1;

    auto       divergence_count = uint64_t{0};
    const auto status           = execute_range(ctx, divergence_count);
    close_range(ctx, status, divergence_count);
    return 0;
}

// host_run's body exited by an exception: close the range without running the passes it has left.
// The range is taken first, so the thread is released even if a callback below throws.
void
abandon_range()
{
    auto ctx = range_context_t{};
    if(!take_range(ctx)) return;
    ctx.record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_BODY_EXCEPTION);
    if(ctx.host.enabled) host::abandon_pass(ctx);
    close_range(ctx, ctx.record.status(), 0);
}

uint64_t
current_host_pass()
{
    const auto* ctx = current_range();
    return (ctx != nullptr && ctx->host.enabled) ? ctx->host.pass : 0;
}

// The body of a host_run whose range did not open still runs, once, as the pass the thread is
// executing. Regions it registers outside any range would otherwise be pending for the next
// begin(), which may capture them after the memory is gone.
void
run_body_once(rocprofiler_range_replay_body_t body, void* user_data)
{
    const auto _drop_pending = common::scope_destructor{[]() { take_pending_host_regions(); }};
    body(current_host_pass(), user_data);
}
}  // namespace
}  // namespace range_replay
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_range_replay_begin(uint64_t range_id)
{
    namespace range_replay = ::rocprofiler::range_replay;
    return range_replay::begin_range(range_id, range_replay::host_rerun_t::jump);
}

rocprofiler_status_t
rocprofiler_range_replay_host_run(uint64_t                        range_id,
                                  rocprofiler_range_replay_body_t body,
                                  void*                           user_data)
{
    namespace range_replay = ::rocprofiler::range_replay;

    if(body == nullptr) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    const auto status = range_replay::begin_range(range_id, range_replay::host_rerun_t::body);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        range_replay::run_body_once(body, user_data);
        return status;
    }

    for(auto pass = uint64_t{0};;)
    {
        try
        {
            body(pass, user_data);
        } catch(...)
        {
            range_replay::abandon_range();
            throw;
        }

        const auto next = range_replay::end_range();
        if(next < 0) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        if(next == 0) return ROCPROFILER_STATUS_SUCCESS;
        pass = static_cast<uint64_t>(next);
    }
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
    if(size > std::numeric_limits<uintptr_t>::max() - reinterpret_cast<uintptr_t>(ptr))
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

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

    *pass = ::rocprofiler::range_replay::current_host_pass();
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_range_replay_end(void)
{
    namespace range_replay = ::rocprofiler::range_replay;

    if(const auto* live = range_replay::current_range();
       live != nullptr && live->opened_by_host_run)
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    const auto next = range_replay::end_range();
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
