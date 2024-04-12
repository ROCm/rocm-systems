// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/kernel_dispatch/tracing.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>

#include <string_view>

#if defined(ROCPROFILER_CI)
#    define ROCP_CI_LOG_IF(NON_CI_LEVEL, ...) LOG_IF(FATAL, __VA_ARGS__)
#    define ROCP_CI_LOG(NON_CI_LEVEL, ...)    ROCP_FATAL
#else
#    define ROCP_CI_LOG_IF(NON_CI_LEVEL, ...) LOG_IF(NON_CI_LEVEL, __VA_ARGS__)
#    define ROCP_CI_LOG(NON_CI_LEVEL, ...)    LOG(NON_CI_LEVEL)
#endif

namespace rocprofiler
{
namespace kernel_dispatch
{
namespace
{
using queue_info_session_t     = hsa::queue_info_session;
using kernel_dispatch_record_t = rocprofiler_buffer_tracing_kernel_dispatch_record_t;
}  // namespace

void
dispatch_complete(queue_info_session_t& session)
{
    // get the contexts that were active when the signal was created
    auto& tracing_data_v = session.tracing_data;
    if(tracing_data_v.callback_contexts.empty() && tracing_data_v.buffered_contexts.empty()) return;

    // we need to decrement this reference count at the end of the functions
    auto* _corr_id = session.correlation_id;

    // only do the following work if there are contexts that require this info
    auto&       callback_record  = session.callback_record;
    const auto& _extern_corr_ids = session.tracing_data.external_correlation_ids;
    const auto* _rocp_agent      = agent::get_agent(callback_record.agent_id);
    auto        _hsa_agent       = agent::get_hsa_agent(_rocp_agent);
    auto        _kern_id         = callback_record.kernel_id;
    auto        _signal          = session.kernel_pkt.kernel_dispatch.completion_signal;
    auto        _tid             = session.tid;

    auto dispatch_time = hsa_amd_profiling_dispatch_time_t{};
    auto dispatch_time_status =
        (_hsa_agent) ? hsa::get_amd_ext_table()->hsa_amd_profiling_get_dispatch_time_fn(
                           *_hsa_agent, _signal, &dispatch_time)
                     : HSA_STATUS_ERROR;

    if(dispatch_time_status == HSA_STATUS_SUCCESS)
    {
        callback_record.start_timestamp = dispatch_time.start;
        callback_record.end_timestamp   = dispatch_time.end;
    }

    // if we encounter this in CI, it will cause test to fail
    ROCP_CI_LOG_IF(
        ERROR,
        dispatch_time_status == HSA_STATUS_SUCCESS && dispatch_time.end < dispatch_time.start)
        << "hsa_amd_profiling_get_dispatch_time for kernel_id=" << _kern_id
        << " on rocprofiler_agent=" << _rocp_agent->id.handle
        << " returned dispatch times where the end time (" << dispatch_time.end
        << ") was less than the start time (" << dispatch_time.start << ")";

    ROCP_CI_LOG_IF(ERROR, dispatch_time_status != HSA_STATUS_SUCCESS)
        << "hsa_amd_profiling_get_dispatch_time for kernel id=" << _kern_id << " returned "
        << dispatch_time_status << " :: " << hsa::get_hsa_status_string(dispatch_time_status);

    auto _internal_corr_id = (_corr_id) ? _corr_id->internal : 0;

    if(dispatch_time_status == HSA_STATUS_SUCCESS)
    {
        if(!tracing_data_v.callback_contexts.empty())
        {
            auto tracer_data = callback_record;
            tracing::execute_phase_none_callbacks(tracing_data_v.callback_contexts,
                                                  _tid,
                                                  _internal_corr_id,
                                                  _extern_corr_ids,
                                                  ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                  ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                                  tracer_data);
        }

        if(!tracing_data_v.buffered_contexts.empty())
        {
            auto record = kernel_dispatch_record_t{sizeof(kernel_dispatch_record_t),
                                                   ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                   ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                                   rocprofiler_correlation_id_t{},
                                                   _tid,
                                                   callback_record.start_timestamp,
                                                   callback_record.end_timestamp,
                                                   callback_record.agent_id,
                                                   callback_record.queue_id,
                                                   callback_record.kernel_id,
                                                   callback_record.dispatch_id,
                                                   callback_record.private_segment_size,
                                                   callback_record.group_segment_size,
                                                   callback_record.workgroup_size,
                                                   callback_record.grid_size};

            tracing::execute_buffer_record_emplace(tracing_data_v.buffered_contexts,
                                                   _tid,
                                                   _internal_corr_id,
                                                   _extern_corr_ids,
                                                   ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                   ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                                   record);
        }
    }
}
}  // namespace kernel_dispatch
}  // namespace rocprofiler
