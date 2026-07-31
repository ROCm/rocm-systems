// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/gpu_events/hip_interception.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/hip/hip.hpp"
#include "lib/rocprofiler-sdk/hip/stream.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>

#include <unordered_map>

namespace rocprofiler
{
namespace gpu_events
{
struct hip_event_api_record_t
{
    rocprofiler_hip_runtime_api_id_t operation;
    hipEvent_t                       event;
    uint64_t                         event_id;
    hipStream_t                      stream;
};

thread_local hip_event_api_record_t* hip_event_record_tls = nullptr;

void
set_hip_event_record_tag(hip_event_api_record_t* record)
{
    ROCP_CI_LOG_IF(WARNING, record != nullptr && hip_event_record_tls != nullptr)
        << "gpu_events TLS already set when entering HIP event API wrapper — possible reentrancy";
    hip_event_record_tls = record;
}

using event_id_map_t = std::unordered_map<hipEvent_t, uint64_t>;

static auto*&
get_event_map()
{
    static auto*& _v = common::static_object<common::Synchronized<event_id_map_t>>::construct();
    return _v;
}

static uint64_t event_index = 0;

static t_hipEventCreate base_event_create = nullptr;
hipError_t
hip_event_create_wrapper(hipEvent_t* event)
{
    hipError_t err = base_event_create(event);

    if(event && *event && err == hipSuccess)
    {
        get_event_map()->wlock(
            [&](auto& data) { data.insert(std::make_pair(*event, ++event_index)); });
    }

    return err;
}

static t_hipEventCreateWithFlags base_event_create_with_flags = nullptr;
hipError_t
hip_event_create_with_flags_wrapper(hipEvent_t* event, unsigned flags)
{
    hipError_t err = base_event_create_with_flags(event, flags);

    if(event && *event && err == hipSuccess)
    {
        get_event_map()->wlock(
            [&](auto& data) { data.insert(std::make_pair(*event, ++event_index)); });
    }

    return err;
}

static t_hipEventDestroy base_event_destroy = nullptr;
hipError_t
hip_event_destroy_wrapper(hipEvent_t event)
{
    get_event_map()->wlock([&](auto& data) { data.erase(event); });

    return base_event_destroy(event);
}

uint64_t
lookup_event_id(hipEvent_t event)
{
    return get_event_map()->rlock([&](const auto& data) -> uint64_t {
        auto it = data.find(event);
        return (it != data.end()) ? it->second : 0;
    });
}

static t_hipStreamWaitEvent base_stream_wait_event = nullptr;
hipError_t
hip_stream_wait_event_wrapper(hipStream_t stream, hipEvent_t event, unsigned int flags)
{
    hip_event_api_record_t hip_record{
        ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent, event, lookup_event_id(event), stream};

    set_hip_event_record_tag(&hip_record);
    auto _cleanup = common::scope_destructor{[] { set_hip_event_record_tag(nullptr); }};
    return base_stream_wait_event(stream, event, flags);
}

static t_hipStreamWaitEvent_spt base_stream_wait_event_spt = nullptr;
hipError_t
hip_stream_wait_event_spt_wrapper(hipStream_t stream, hipEvent_t event, unsigned int flags)
{
    hip_event_api_record_t hip_record{ROCPROFILER_HIP_RUNTIME_API_ID_hipStreamWaitEvent_spt,
                                      event,
                                      lookup_event_id(event),
                                      stream};

    set_hip_event_record_tag(&hip_record);
    auto _cleanup = common::scope_destructor{[] { set_hip_event_record_tag(nullptr); }};
    return base_stream_wait_event_spt(stream, event, flags);
}

static t_hipEventRecord base_event_record = nullptr;
hipError_t
hip_event_record_wrapper(hipEvent_t event, hipStream_t stream)
{
    hip_event_api_record_t hip_record{
        ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecord, event, lookup_event_id(event), stream};

    set_hip_event_record_tag(&hip_record);
    auto _cleanup = common::scope_destructor{[] { set_hip_event_record_tag(nullptr); }};
    return base_event_record(event, stream);
}

static t_hipEventRecord_spt base_event_record_spt = nullptr;
hipError_t
hip_event_record_spt_wrapper(hipEvent_t event, hipStream_t stream)
{
    hip_event_api_record_t hip_record{
        ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecord_spt, event, lookup_event_id(event), stream};

    set_hip_event_record_tag(&hip_record);
    auto _cleanup = common::scope_destructor{[] { set_hip_event_record_tag(nullptr); }};
    return base_event_record_spt(event, stream);
}

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 10
static t_hipEventRecordWithFlags base_event_record_with_flags = nullptr;
hipError_t
hip_event_record_with_flags_wrapper(hipEvent_t event, hipStream_t stream, unsigned int flags)
{
    hip_event_api_record_t hip_record{ROCPROFILER_HIP_RUNTIME_API_ID_hipEventRecordWithFlags,
                                      event,
                                      lookup_event_id(event),
                                      stream};

    set_hip_event_record_tag(&hip_record);
    auto _cleanup = common::scope_destructor{[] { set_hip_event_record_tag(nullptr); }};
    return base_event_record_with_flags(event, stream, flags);
}
#endif

bool
gpu_event_tracing()
{
    return (hip_event_record_tls != nullptr);
}

uint64_t
get_gpu_event_id()
{
    return (hip_event_record_tls ? hip_event_record_tls->event_id : 0);
}

rocprofiler_stream_id_t
get_gpu_event_stream_id()
{
    if(hip_event_record_tls && hip_event_record_tls->stream)
        return hip::stream::get_stream_id(hip_event_record_tls->stream);
    return rocprofiler_stream_id_t{.handle = 0};
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
    (void) type;
    (void) lib_version;
    (void) lib_instance;
    (void) num_tables;
    (void) user_data;

    ROCP_FATAL_IF(tables == nullptr || num_tables < 1) << "invalid tables in gpu_events "
                                                          "registration callback";

    auto* hip_api_table = static_cast<HipDispatchTable*>(tables[0]);

    rocprofiler::gpu_events::base_event_create = hip_api_table->hipEventCreate_fn;
    rocprofiler::gpu_events::base_event_create_with_flags =
        hip_api_table->hipEventCreateWithFlags_fn;
    rocprofiler::gpu_events::base_event_destroy         = hip_api_table->hipEventDestroy_fn;
    rocprofiler::gpu_events::base_stream_wait_event     = hip_api_table->hipStreamWaitEvent_fn;
    rocprofiler::gpu_events::base_stream_wait_event_spt = hip_api_table->hipStreamWaitEvent_spt_fn;
    rocprofiler::gpu_events::base_event_record          = hip_api_table->hipEventRecord_fn;
    rocprofiler::gpu_events::base_event_record_spt      = hip_api_table->hipEventRecord_spt_fn;

#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 10
    rocprofiler::gpu_events::base_event_record_with_flags =
        hip_api_table->hipEventRecordWithFlags_fn;
#endif

    hip_api_table->hipEventCreate_fn = &rocprofiler::gpu_events::hip_event_create_wrapper;
    hip_api_table->hipEventCreateWithFlags_fn =
        &rocprofiler::gpu_events::hip_event_create_with_flags_wrapper;
    hip_api_table->hipEventDestroy_fn    = &rocprofiler::gpu_events::hip_event_destroy_wrapper;
    hip_api_table->hipStreamWaitEvent_fn = &rocprofiler::gpu_events::hip_stream_wait_event_wrapper;
    hip_api_table->hipStreamWaitEvent_spt_fn =
        &rocprofiler::gpu_events::hip_stream_wait_event_spt_wrapper;
    hip_api_table->hipEventRecord_fn     = &rocprofiler::gpu_events::hip_event_record_wrapper;
    hip_api_table->hipEventRecord_spt_fn = &rocprofiler::gpu_events::hip_event_record_spt_wrapper;
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 10
    hip_api_table->hipEventRecordWithFlags_fn =
        &rocprofiler::gpu_events::hip_event_record_with_flags_wrapper;
#endif
}
}
