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
#include "lib/rocprofiler-sdk/range_replay/range_state.hpp"
#include "lib/rocprofiler-sdk/range_replay/replay_callbacks.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>

#include <atomic>
#include <cstdint>

extern "C" {
rocprofiler_status_t
rocprofiler_range_replay_begin(uint64_t range_id)
{
    namespace range_replay = ::rocprofiler::range_replay;

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

    // Range replay re-submits recorded packets through the interception path in hsa/queue.cpp. The
    // write-pointer virtualization path has its own submission handling, which this beta does not
    // cover, so a range opened under it is recorded and declined rather than replayed.
    if(::rocprofiler::hsa::queue_interposition::supports_queue_interposition())
    {
        static std::atomic<bool> _warned{false};
        if(!_warned.exchange(true, std::memory_order_relaxed))
            ROCP_WARNING << "range replay: not supported with queue interposition enabled; ranges "
                            "will be recorded and declined";
        ctx->record.decline(ROCPROFILER_RANGE_REPLAY_STATUS_UNSUPPORTED_QUEUE_PATH);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_range_replay_end(void)
{
    namespace range_replay = ::rocprofiler::range_replay;

    auto ctx = range_replay::range_context_t{};
    if(!range_replay::take_range(ctx)) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto       divergence_count = uint64_t{0};
    const auto status           = range_replay::execute_range(ctx, divergence_count);

    range_replay::execute_close_callback(ctx.plan,
                                         status,
                                         ctx.agent_id,
                                         ctx.record.observed_dispatch_count(),
                                         divergence_count,
                                         ctx.thread_id,
                                         ctx.internal_corr_id,
                                         ctx.ancestor_corr_id);

    return ROCPROFILER_STATUS_SUCCESS;
}
}
