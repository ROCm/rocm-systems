// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>

#include <string_view>

namespace rocprofiler
{
namespace kernel_dispatch
{
namespace
{
using queue_info_session_t     = hsa::queue_info_session;
using kernel_dispatch_record_t = rocprofiler_buffer_tracing_kernel_dispatch_record_t;
}  // namespace

profiling_time
get_dispatch_time(const hsa::queue_info_session& session)
{
    const auto& callback_record = session.callback_record;
    const auto* _rocp_agent     = agent::get_agent(callback_record.dispatch_info.agent_id);
    auto        _hsa_agent      = agent::get_hsa_agent(_rocp_agent);
    auto        _signal         = session.kernel_pkt.kernel_dispatch.completion_signal;
    auto        _kern_id        = callback_record.dispatch_info.kernel_id;

    return (_hsa_agent) ? get_dispatch_time(*_hsa_agent, _signal, _kern_id, session.enqueue_ts)
                        : profiling_time{.status = HSA_STATUS_ERROR_INVALID_AGENT};
}

void
dispatch_complete(queue_info_session_t& session, profiling_time dispatch_time)
{
    // get the contexts that were active when the signal was created
    auto& tracing_data_v = session.tracing_data;
    if(tracing_data_v.callback_contexts.empty() && tracing_data_v.buffered_contexts.empty()) return;

    // we need to decrement this reference count at the end of the functions
    auto* _corr_id = session.correlation_id;

    // only do the following work if there are contexts that require this info
    auto&       callback_record   = session.callback_record;
    const auto& _extern_corr_ids  = session.tracing_data.external_correlation_ids;
    auto        _tid              = session.tid;
    auto        _internal_corr_id = (_corr_id) ? _corr_id->internal : 0;
    auto        _ancestor_corr_id = (_corr_id) ? _corr_id->ancestor : 0;

    if(dispatch_time.status == HSA_STATUS_SUCCESS)
    {
        callback_record.start_timestamp = dispatch_time.start;
        callback_record.end_timestamp   = dispatch_time.end;

        if(!tracing_data_v.callback_contexts.empty())
        {
            auto tracer_data = callback_record;
            tracing::execute_phase_none_callbacks(tracing_data_v.callback_contexts,
                                                  _tid,
                                                  _internal_corr_id,
                                                  _extern_corr_ids,
                                                  _ancestor_corr_id,
                                                  ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                  ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                                  tracer_data);
        }

        if(!tracing_data_v.buffered_contexts.empty())
        {
            auto record = kernel_dispatch_record_t{sizeof(kernel_dispatch_record_t),
                                                   ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                   ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                                   rocprofiler_async_correlation_id_t{},
                                                   _tid,
                                                   callback_record.start_timestamp,
                                                   callback_record.end_timestamp,
                                                   callback_record.dispatch_info};

            tracing::execute_buffer_record_emplace(tracing_data_v.buffered_contexts,
                                                   _tid,
                                                   _internal_corr_id,
                                                   _extern_corr_ids,
                                                   _ancestor_corr_id,
                                                   ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                                                   ROCPROFILER_KERNEL_DISPATCH_COMPLETE,
                                                   record);
        }
    }
}
}  // namespace kernel_dispatch
}  // namespace rocprofiler
