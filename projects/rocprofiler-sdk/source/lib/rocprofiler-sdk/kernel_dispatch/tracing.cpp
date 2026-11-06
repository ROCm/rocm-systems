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
#include "lib/rocprofiler-sdk/hsa/queue_interposition.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_reader.hpp"
#include "lib/rocprofiler-sdk/kfd/signal_less.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <hsa/hsa.h>

#include <string_view>

namespace rocprofiler
{
namespace kernel_dispatch
{
rocprofiler_status_t
flush_buffer_records(rocprofiler_buffer_id_t buffer_id)
{
    if(internal_threading::fork_stale() || registration::get_fini_status() != 0)
        return ROCPROFILER_STATUS_SUCCESS;

    auto* target = buffer::get_buffer(buffer_id);
    if(!target) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;
    auto                    contexts = context::get_registered_contexts_snapshot();
    const context::context* owner    = nullptr;
    for(const auto* candidate : contexts)
    {
        if(candidate && candidate->context_idx == target->context_id)
        {
            owner = candidate;
            break;
        }
    }
    if(!owner || !owner->buffered_tracer ||
       owner->buffered_tracer->buffer_data.at(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) !=
           buffer_id)
        return ROCPROFILER_STATUS_SUCCESS;

    if(internal_threading::is_task_group_thread() || hsa::is_async_signal_handler_thread())
        return ROCPROFILER_STATUS_ERROR_BUFFER_BUSY;

    const auto reader_ready   = !kfd::signal_less_feature_enabled() || kfd::flush_reader_records();
    const auto inline_ready   = hsa::queue_interposition::interposition_fence_records();
    const auto expanded_ready = hsa::flush_async_signal_records();
    return reader_ready && inline_ready && expanded_ready ? ROCPROFILER_STATUS_SUCCESS
                                                          : ROCPROFILER_STATUS_ERROR_BUFFER_BUSY;
}

profiling_time
get_dispatch_time(const queue_info_session_t& session, packet_data_t& packet_data)
{
    const auto& callback_record = packet_data.callback_record;
    const auto* _rocp_agent     = agent::get_agent(callback_record.dispatch_info.agent_id);
    auto        _hsa_agent      = agent::get_hsa_agent(_rocp_agent);
    auto        _signal         = packet_data.kernel_packet.kernel_dispatch.completion_signal;
    auto        _kern_id        = callback_record.dispatch_info.kernel_id;

    return (_hsa_agent) ? get_dispatch_time(*_hsa_agent, _signal, _kern_id, session.enqueue_ts)
                        : profiling_time{.status = HSA_STATUS_ERROR_INVALID_AGENT};
}

void
emit_kernel_dispatch_record(tracing::tracing_data&                               tracing_data_v,
                            rocprofiler_callback_tracing_kernel_dispatch_data_t& callback_record,
                            context::correlation_id*                             _corr_id,
                            rocprofiler_thread_id_t                              _tid,
                            uint64_t                                             start_timestamp,
                            uint64_t                                             end_timestamp)
{
    using kernel_dispatch_record_t = rocprofiler_buffer_tracing_kernel_dispatch_record_t;

    // get the contexts that were active when the dispatch was traced
    if(tracing_data_v.callback_contexts.empty() && tracing_data_v.buffered_contexts.empty()) return;

    const auto& _extern_corr_ids  = tracing_data_v.external_correlation_ids;
    auto        _internal_corr_id = (_corr_id) ? _corr_id->internal : 0;
    auto        _ancestor_corr_id = (_corr_id) ? _corr_id->ancestor : 0;

    callback_record.start_timestamp = start_timestamp;
    callback_record.end_timestamp   = end_timestamp;

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

void
dispatch_complete(queue_info_session_t& session,
                  packet_data_t&        packet_data,
                  profiling_time        dispatch_time)
{
    if(dispatch_time.status != HSA_STATUS_SUCCESS) return;

    emit_kernel_dispatch_record(packet_data.tracing_data,
                                packet_data.callback_record,
                                session.correlation_id,
                                session.tid,
                                dispatch_time.start,
                                dispatch_time.end);
}
}  // namespace kernel_dispatch
}  // namespace rocprofiler
