// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hip/event.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>

#include <hip/amd_detail/hip_api_trace.hpp>

#include <string_view>

namespace rocprofiler
{
namespace hip
{
namespace event
{
namespace
{
#define ROCPROFILER_HIP_EVENT_INFO(CODE)                                                           \
    template <>                                                                                    \
    struct hip_event_info<ROCPROFILER_##CODE>                                                      \
    {                                                                                              \
        static constexpr auto operation_idx = ROCPROFILER_##CODE;                                  \
        static constexpr auto name          = #CODE;                                               \
    };

template <size_t Idx>
struct hip_event_info;

ROCPROFILER_HIP_EVENT_INFO(HIP_EVENT_NONE)
ROCPROFILER_HIP_EVENT_INFO(HIP_EVENT_RECORD)
ROCPROFILER_HIP_EVENT_INFO(HIP_EVENT_WAIT)

template <size_t Idx, size_t... IdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return hip_event_info<Idx>::name;
    if constexpr(sizeof...(IdxTail) > 0)
        return name_by_id(id, std::index_sequence<IdxTail...>{});
    else
        return nullptr;
}

template <size_t Idx, size_t... IdxTail>
uint32_t
id_by_name(const char* name, std::index_sequence<Idx, IdxTail...>)
{
    if(std::string_view{hip_event_info<Idx>::name} == std::string_view{name})
        return hip_event_info<Idx>::operation_idx;
    if constexpr(sizeof...(IdxTail) > 0)
        return id_by_name(name, std::index_sequence<IdxTail...>{});
    else
        return ROCPROFILER_HIP_EVENT_LAST;
}

template <size_t... Idx>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < static_cast<uint32_t>(ROCPROFILER_HIP_EVENT_LAST)) _vec.emplace_back(_v);
    };

    (_emplace(_id_list, hip_event_info<Idx>::operation_idx), ...);
}

template <size_t... Idx>
void
get_names(std::vector<const char*>& _name_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, const char* _v) {
        if(_v != nullptr && !std::string_view{_v}.empty()) _vec.emplace_back(_v);
    };

    (_emplace(_name_list, hip_event_info<Idx>::name), ...);
}
}  // namespace

const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_HIP_EVENT_LAST>{});
}

uint32_t
id_by_name(const char* name)
{
    return id_by_name(name, std::make_index_sequence<ROCPROFILER_HIP_EVENT_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    auto _data = std::vector<uint32_t>{};
    _data.reserve(ROCPROFILER_HIP_EVENT_LAST);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_HIP_EVENT_LAST>{});
    return _data;
}

std::vector<const char*>
get_names()
{
    auto _data = std::vector<const char*>{};
    _data.reserve(ROCPROFILER_HIP_EVENT_LAST);
    get_names(_data, std::make_index_sequence<ROCPROFILER_HIP_EVENT_LAST>{});
    return _data;
}

void
barrier_complete(tracing::tracing_data&                        tracing_data_v,
                 rocprofiler_thread_id_t                       tid,
                 uint64_t                                      internal_corr_id,
                 uint64_t                                      ancestor_corr_id,
                 profiling_time                                barrier_time,
                 rocprofiler_hip_event_operation_t             operation,
                 rocprofiler_callback_tracing_hip_event_data_t callback_record)
{
    using hip_event_record_t = rocprofiler_buffer_tracing_hip_event_record_t;

    if(tracing_data_v.callback_contexts.empty() && tracing_data_v.buffered_contexts.empty()) return;

    const auto& _extern_corr_ids = tracing_data_v.external_correlation_ids;

    if(barrier_time.status == HSA_STATUS_SUCCESS)
    {
        callback_record.start_timestamp = barrier_time.start;
        callback_record.end_timestamp   = barrier_time.end;

        if(!tracing_data_v.callback_contexts.empty())
        {
            auto tracer_data = callback_record;
            tracing::execute_phase_none_callbacks(tracing_data_v.callback_contexts,
                                                  tid,
                                                  internal_corr_id,
                                                  _extern_corr_ids,
                                                  ancestor_corr_id,
                                                  ROCPROFILER_CALLBACK_TRACING_HIP_EVENT,
                                                  operation,
                                                  tracer_data);
        }

        if(!tracing_data_v.buffered_contexts.empty())
        {
            auto record = hip_event_record_t{sizeof(hip_event_record_t),
                                             ROCPROFILER_BUFFER_TRACING_HIP_EVENT,
                                             operation,
                                             rocprofiler_async_correlation_id_t{},
                                             tid,
                                             callback_record.start_timestamp,
                                             callback_record.end_timestamp,
                                             callback_record.agent_id,
                                             callback_record.queue_id,
                                             callback_record.hip_event_handle,
                                             callback_record.source_queue_id};

            tracing::execute_buffer_record_emplace(tracing_data_v.buffered_contexts,
                                                   tid,
                                                   internal_corr_id,
                                                   _extern_corr_ids,
                                                   ancestor_corr_id,
                                                   ROCPROFILER_BUFFER_TRACING_HIP_EVENT,
                                                   operation,
                                                   record);
        }
    }
}
namespace
{
thread_local active_event_context_t g_active_event_ctx = {};

template <typename ApiTag, typename RetT>
auto event_record_wrapper(RetT (*next)(hipEvent_t, hipStream_t))
{
    static auto next_func = next;
    return +[](hipEvent_t event, hipStream_t stream) -> RetT {
        g_active_event_ctx = {
            ROCPROFILER_HIP_EVENT_RECORD, reinterpret_cast<uint64_t>(event), false};
        auto ret           = next_func(event, stream);
        g_active_event_ctx = {};
        return ret;
    };
}

template <typename ApiTag, typename RetT>
auto event_record_with_flags_wrapper(RetT (*next)(hipEvent_t, hipStream_t, unsigned int))
{
    static auto next_func = next;
    return +[](hipEvent_t event, hipStream_t stream, unsigned int flags) -> RetT {
        g_active_event_ctx = {
            ROCPROFILER_HIP_EVENT_RECORD, reinterpret_cast<uint64_t>(event), false};
        auto ret           = next_func(event, stream, flags);
        g_active_event_ctx = {};
        return ret;
    };
}

template <typename ApiTag, typename RetT>
auto stream_wait_event_wrapper(RetT (*next)(hipStream_t, hipEvent_t, unsigned int))
{
    static auto next_func = next;
    return +[](hipStream_t stream, hipEvent_t event, unsigned int flags) -> RetT {
        g_active_event_ctx = {ROCPROFILER_HIP_EVENT_WAIT, reinterpret_cast<uint64_t>(event), false};
        auto ret           = next_func(stream, event, flags);
        g_active_event_ctx = {};
        return ret;
    };
}
}  // namespace

active_event_context_t*
get_active_event_context()
{
    if(g_active_event_ctx.operation == ROCPROFILER_HIP_EVENT_NONE) return nullptr;
    return &g_active_event_ctx;
}

namespace api
{
struct hipEventRecord;
struct hipEventRecord_spt;
struct hipEventRecordWithFlags;
struct hipStreamWaitEvent;
struct hipStreamWaitEvent_spt;
}  // namespace api

template <>
void
update_table<::HipDispatchTable>(::HipDispatchTable* table)
{
    if(table == nullptr) return;
    if(table->hipEventRecord_fn)
        table->hipEventRecord_fn =
            event_record_wrapper<api::hipEventRecord>(table->hipEventRecord_fn);
    if(table->hipEventRecord_spt_fn)
        table->hipEventRecord_spt_fn =
            event_record_wrapper<api::hipEventRecord_spt>(table->hipEventRecord_spt_fn);
    if(table->hipEventRecordWithFlags_fn)
        table->hipEventRecordWithFlags_fn =
            event_record_with_flags_wrapper<api::hipEventRecordWithFlags>(
                table->hipEventRecordWithFlags_fn);
    if(table->hipStreamWaitEvent_fn)
        table->hipStreamWaitEvent_fn =
            stream_wait_event_wrapper<api::hipStreamWaitEvent>(table->hipStreamWaitEvent_fn);
    if(table->hipStreamWaitEvent_spt_fn)
        table->hipStreamWaitEvent_spt_fn = stream_wait_event_wrapper<api::hipStreamWaitEvent_spt>(
            table->hipStreamWaitEvent_spt_fn);
}

template void
update_table<::HipDispatchTable>(::HipDispatchTable*);

}  // namespace event
}  // namespace hip
}  // namespace rocprofiler
