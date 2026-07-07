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

#include "lib/rocprofiler-sdk/gpu_events/tracing.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/gpu_events/profiling_time.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa.h>

#include <string_view>

namespace rocprofiler
{
namespace gpu_events
{
profiling_time
get_gpu_event_time(const queue_info_session_t& session, packet_data_t& packet_data)
{
    const auto& callback_record = packet_data.event_record;
    const auto* _rocp_agent     = agent::get_agent(callback_record.event_info.agent_id);
    auto        _hsa_agent      = agent::get_hsa_agent(_rocp_agent);

    auto        _signal         = (callback_record.event_info.type_id == ROCPROFILER_GPU_EVENT_WAIT_ENQUEUE) 
                                ? packet_data.kernel_packet.barrier_and.completion_signal 
                                : packet_data.kernel_packet.ext_amd_aql_pm4.completion_signal;

    return (_hsa_agent) ? get_gpu_event_time(*_hsa_agent, _signal, 0, session.enqueue_ts)
                        : profiling_time{.status = HSA_STATUS_ERROR_INVALID_AGENT};
}

void
gpu_event_complete(queue_info_session_t& session,
                  packet_data_t&        packet_data,
                  profiling_time        dispatch_time)
{
    using gpu_event_record_t = rocprofiler_buffer_tracing_gpu_event_record_t;

    // get the contexts that were active when the signal was created
    auto& tracing_data_v = packet_data.tracing_data;
    if(tracing_data_v.callback_contexts.empty() && tracing_data_v.buffered_contexts.empty()) return;

    // we need to decrement this reference count at the end of the functions
    auto* _corr_id = session.correlation_id;

    // only do the following work if there are contexts that require this info
    auto&       callback_record   = packet_data.event_record;
    const auto& _extern_corr_ids  = packet_data.tracing_data.external_correlation_ids;
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
                                                  ROCPROFILER_CALLBACK_TRACING_GPU_EVENTS,
                                                  (callback_record.event_info.type_id == ROCPROFILER_GPU_EVENT_WAIT_ENQUEUE)
                                                  ? ROCPROFILER_GPU_EVENT_WAIT_COMPLETE
                                                  : ROCPROFILER_GPU_EVENT_RECORD_COMPLETE,
                                                  tracer_data);
        }

        if(!tracing_data_v.buffered_contexts.empty())
        {
            auto record = gpu_event_record_t{sizeof(gpu_event_record_t),
                                                   ROCPROFILER_BUFFER_TRACING_GPU_EVENTS,
                                                   (callback_record.event_info.type_id == ROCPROFILER_GPU_EVENT_WAIT_ENQUEUE)
                                                  ? ROCPROFILER_GPU_EVENT_WAIT_COMPLETE
                                                  : ROCPROFILER_GPU_EVENT_RECORD_COMPLETE,
                                                   rocprofiler_async_correlation_id_t{},
                                                   _tid,
                                                   callback_record.start_timestamp,
                                                   callback_record.end_timestamp,
                                                   callback_record.event_info};

            tracing::execute_buffer_record_emplace(tracing_data_v.buffered_contexts,
                                                   _tid,
                                                   _internal_corr_id,
                                                   _extern_corr_ids,
                                                   _ancestor_corr_id,
                                                   ROCPROFILER_BUFFER_TRACING_GPU_EVENTS,
                                                   (callback_record.event_info.type_id == ROCPROFILER_GPU_EVENT_WAIT_ENQUEUE)
                                                  ? ROCPROFILER_GPU_EVENT_WAIT_COMPLETE
                                                  : ROCPROFILER_GPU_EVENT_RECORD_COMPLETE,
                                                   record);
        }
    }
}


 
struct hip_event_api_record_t
{
    rocprofiler_hip_runtime_api_id_t operation;
    hipEvent_t event;
    uint64_t event_id;
};

thread_local hip_event_api_record_t* hip_event_record_tls = nullptr;

void SetHipEventRecordTag(hip_event_api_record_t* record)
{
    hip_event_record_tls = record;
}

struct EventMap
{
    std::map<hipEvent_t, uint64_t> event_ids;
    std::mutex event_mutex;
    uint64_t event_index = 0;
};

static EventMap event_map;

static t_hipEventCreate baseEventCreate = nullptr;
hipError_t 	hipEventCreateWrapper(hipEvent_t *event)
{
    hipError_t err = baseEventCreate(event);

    if (event && *event && err == hipSuccess)
    {
        event_map.event_mutex.lock();
        event_map.event_ids.insert(std::make_pair(*event, ++event_map.event_index));
        event_map.event_mutex.unlock();
    }

    return err;
}

static t_hipEventCreateWithFlags baseEventCreateWithFlags = nullptr;
hipError_t 	hipEventCreateWithFlagsWrapper(hipEvent_t *event, unsigned flags)
{
    hipError_t err = baseEventCreateWithFlags(event, flags);

    if (event && *event && err == hipSuccess)
    {
        event_map.event_mutex.lock();
        event_map.event_ids.insert(std::make_pair(*event, ++event_map.event_index));
        event_map.event_mutex.unlock();
    }

    return err;
}

static t_hipStreamWaitEvent baseStreamWaitEvent = nullptr;
hipError_t hipStreamWaitEventWrapper(hipStream_t stream, hipEvent_t event,
                                           unsigned int flags)
{
    hip_event_api_record_t hip_record;
    hip_record.operation = (rocprofiler_hip_runtime_api_id_t)ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent;
    hip_record.event = event;
    hip_record.event_id = 0;

    event_map.event_mutex.lock();
    auto it = event_map.event_ids.find(hip_record.event);
    if (it != event_map.event_ids.end())
    {
        hip_record.event_id = it->second;
    }
    event_map.event_mutex.unlock();

    SetHipEventRecordTag(&hip_record);

    hipError_t err = baseStreamWaitEvent(stream, event, flags);

    SetHipEventRecordTag(nullptr);

    return err;
}

static t_hipStreamWaitEvent_spt baseStreamWaitEventSpt = nullptr;
hipError_t hipStreamWaitEventSptWrapper(hipStream_t stream, hipEvent_t event,
                                           unsigned int flags)
{
    hip_event_api_record_t hip_record;
    hip_record.operation = (rocprofiler_hip_runtime_api_id_t)ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent_spt;
    hip_record.event = event;
    hip_record.event_id = 0;

    event_map.event_mutex.lock();
    auto it = event_map.event_ids.find(hip_record.event);
    if (it != event_map.event_ids.end())
    {
        hip_record.event_id = it->second;
    }
    event_map.event_mutex.unlock();

    SetHipEventRecordTag(&hip_record);

    hipError_t err = baseStreamWaitEventSpt(stream, event, flags);

    SetHipEventRecordTag(nullptr);

    return err;
}

static t_hipEventRecord baseEventRecord = nullptr;
hipError_t hipEventRecordWrapper(hipEvent_t event, hipStream_t stream)
{
    hip_event_api_record_t hip_record;
    hip_record.operation = (rocprofiler_hip_runtime_api_id_t)ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent_spt;
    hip_record.event = event;
    hip_record.event_id = 0;

    event_map.event_mutex.lock();
    auto it = event_map.event_ids.find(hip_record.event);
    if (it != event_map.event_ids.end())
    {
        hip_record.event_id = it->second;
    }
    event_map.event_mutex.unlock();

    SetHipEventRecordTag(&hip_record);

    hipError_t err = baseEventRecord(event, stream);

    SetHipEventRecordTag(nullptr);

    return err;
}

static t_hipEventRecord_spt baseEventRecordSpt = nullptr;
hipError_t hipEventRecordSptWrapper(hipEvent_t event, hipStream_t stream)
{
    hip_event_api_record_t hip_record;
    hip_record.operation = (rocprofiler_hip_runtime_api_id_t)ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent_spt;
    hip_record.event = event;
    hip_record.event_id = 0;

    event_map.event_mutex.lock();
    auto it = event_map.event_ids.find(hip_record.event);
    if (it != event_map.event_ids.end())
    {
        hip_record.event_id = it->second;
    }
    event_map.event_mutex.unlock();

    SetHipEventRecordTag(&hip_record);

    hipError_t err = baseEventRecordSpt(event, stream);

    SetHipEventRecordTag(nullptr);

    return err;
}

static t_hipEventRecordWithFlags baseEventRecordWithFlags = nullptr;
hipError_t hipEventRecordWithFlagsWrapper(hipEvent_t event, hipStream_t stream,
                                                unsigned int flags)
{
    hip_event_api_record_t hip_record;
    hip_record.operation = (rocprofiler_hip_runtime_api_id_t)ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent_spt;
    hip_record.event = event;
    hip_record.event_id = 0;

    event_map.event_mutex.lock();
    auto it = event_map.event_ids.find(hip_record.event);
    if (it != event_map.event_ids.end())
    {
        hip_record.event_id = it->second;
    }
    event_map.event_mutex.unlock();

    SetHipEventRecordTag(&hip_record);

    hipError_t err = baseEventRecordWithFlags(event, stream, flags);

    SetHipEventRecordTag(nullptr);

    return err;
}

bool gpu_event_tracing(void)
{
    return (hip_event_record_tls != nullptr);
}

uint64_t get_gpu_event_id(void)
{
    return (hip_event_record_tls ? hip_event_record_tls->event_id : 0);
}

}  // namespace gpu_events
}  // namespace rocprofiler

extern "C" {

ROCPROFILER_API void
hip_gpu_event_registration_callback(rocprofiler_intercept_table_t type,
                          uint64_t                      lib_version,
                          uint64_t                      lib_instance,
                          void**                        tables,
                          uint64_t                      num_tables,
                          void*                         user_data)
{

    auto* hip_api_table = static_cast<HipDispatchTable*>(tables[0]);

    rocprofiler::gpu_events::baseEventCreate = hip_api_table->hipEventCreate_fn;
    rocprofiler::gpu_events::baseEventCreateWithFlags = hip_api_table->hipEventCreateWithFlags_fn;
    rocprofiler::gpu_events::baseStreamWaitEvent = hip_api_table->hipStreamWaitEvent_fn;
    rocprofiler::gpu_events::baseStreamWaitEventSpt = hip_api_table->hipStreamWaitEvent_spt_fn;
    rocprofiler::gpu_events::baseEventRecord = hip_api_table->hipEventRecord_fn;
    rocprofiler::gpu_events::baseEventRecordSpt = hip_api_table->hipEventRecord_spt_fn;
    rocprofiler::gpu_events::baseEventRecordWithFlags = hip_api_table->hipEventRecordWithFlags_fn;

    hip_api_table->hipEventCreate_fn = &rocprofiler::gpu_events::hipEventCreateWrapper;
    hip_api_table->hipEventCreateWithFlags_fn = &rocprofiler::gpu_events::hipEventCreateWithFlagsWrapper;
    hip_api_table->hipStreamWaitEvent_fn = &rocprofiler::gpu_events::hipStreamWaitEventWrapper;
    hip_api_table->hipStreamWaitEvent_spt_fn = &rocprofiler::gpu_events::hipStreamWaitEventSptWrapper;
    hip_api_table->hipEventRecord_fn = &rocprofiler::gpu_events::hipEventRecordWrapper;
    hip_api_table->hipEventRecord_spt_fn = &rocprofiler::gpu_events::hipEventRecordSptWrapper;
    hip_api_table->hipEventRecordWithFlags_fn = &rocprofiler::gpu_events::hipEventRecordWithFlagsWrapper;
}

}
