// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "thread_trace_hotspots.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/thread-trace/agent.h>
#include <rocprofiler-sdk/experimental/thread-trace/core.h>
#include <rocprofiler-sdk/experimental/thread-trace/dispatch.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <rocprof_trace_decoder/rocprof_trace_decoder.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    if(auto ec = (result); ec != ROCPROFILER_STATUS_SUCCESS)                                       \
    {                                                                                              \
        std::cerr << "rocprofiler-sdk error at " << __FILE__ << ":" << __LINE__                    \
                  << " :: " << #result << std::endl;                                               \
        std::cerr << "rocprofiler-sdk error code " << ec << ": "                                   \
                  << rocprofiler_get_status_string(ec) << " :: " << msg << std::endl;              \
        abort();                                                                                   \
    }

#define DECODER_CALL(result)                                                                       \
    if(auto ec = (result); ec != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)                  \
    {                                                                                              \
        std::cerr << "Decoder error at " << __FILE__ << ":" << __LINE__ << std::endl;              \
        std::cerr << "Decoder error code " << ec << ": "                                           \
                  << rocprof_trace_decoder_get_status_string(ec) << std::endl;                     \
    }

#define CHECK_NOTNULL(x)                                                                           \
    if(!(x))                                                                                       \
    {                                                                                              \
        abort();                                                                                   \
    };

namespace
{
constexpr uint64_t TARGET_CU   = 1;           // CU (gfx9) or WGP (gfx10+)
constexpr uint64_t SHADER_MASK = 0x1;         // Only enable SE=0
constexpr uint64_t BUFFER_SIZE = 0x10000000;  // 256MB
};                                            // namespace

namespace Decoder
{
rocprof_trace_decoder_handle_t decoder{};

void
tool_codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t* /* user_data */,
                              void* /* userdata */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.operation != ROCPROFILER_CODE_OBJECT_LOAD) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    auto* data = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);

    if(data->storage_type != ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
    {
        auto* memorybase = reinterpret_cast<const void*>(data->memory_base);
        CHECK_NOTNULL(memorybase);
        DECODER_CALL(rocprof_trace_decoder_codeobj_load(decoder,
                                                        data->code_object_id,
                                                        data->load_delta,
                                                        data->load_size,
                                                        memorybase,
                                                        data->memory_size));
    }

    hotspots::register_codeobj_disasm(*data);
}

void
shader_data_callback(rocprofiler_thread_trace_shader_data_t shader_data,
                     rocprofiler_user_data_t /* userdata */)
{
    DECODER_CALL(rocprof_trace_decoder_parse(
        decoder, shader_data.data, shader_data.data_size, &hotspots::accumulate, nullptr));
}

}  // namespace Decoder

namespace ThreadTracer
{
rocprofiler_client_id_t* client_id   = nullptr;
rocprofiler_context_id_t agent_ctx   = {};
rocprofiler_context_id_t tracing_ctx = {};

rocprofiler_status_t
query_available_agents(rocprofiler_agent_version_t /* version */,
                       const void** agents,
                       size_t       num_agents,
                       void*        user_data)
{
    rocprofiler_user_data_t user{};
    user.ptr = user_data;

    for(size_t idx = 0; idx < num_agents; idx++)
    {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
        if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

        auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{};
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, {TARGET_CU}});
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {BUFFER_SIZE}});
        parameters.push_back(
            {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {SHADER_MASK}});

        ROCPROFILER_CALL(
            rocprofiler_configure_device_thread_trace_service(agent_ctx,
                                                              agent->id,
                                                              parameters.data(),
                                                              parameters.size(),
                                                              Decoder::shader_data_callback,
                                                              user),
            "thread trace service configure");
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

void
cntrl_tracing_callback(rocprofiler_callback_tracing_record_t record,
                       rocprofiler_user_data_t* /* user_data */,
                       void* /* cb_data */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API) return;

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER &&
       record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause)
    {
        ROCPROFILER_CALL(rocprofiler_stop_context(agent_ctx), "stopping context");
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
            record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume)
    {
        ROCPROFILER_CALL(rocprofiler_start_context(agent_ctx), "starting context");
    }
}

int
tool_init(rocprofiler_client_finalize_t /* fini_func */, void* /* tool_data */)
{
    DECODER_CALL(rocprof_trace_decoder_create_handle(&Decoder::decoder));

    ROCPROFILER_CALL(rocprofiler_create_context(&tracing_ctx), "context creation");
    ROCPROFILER_CALL(rocprofiler_create_context(&agent_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(tracing_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       Decoder::tool_codeobj_tracing_callback,
                                                       nullptr),
        "code object tracing service configure");

    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         tracing_ctx,
                         ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                         nullptr,
                         0,
                         cntrl_tracing_callback,
                         nullptr),
                     "marker tracing callback service configure");

    ROCPROFILER_CALL(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                                        &query_available_agents,
                                                        sizeof(rocprofiler_agent_t),
                                                        nullptr),
                     "Failed to find GPU agents");

    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(agent_ctx, &valid_ctx), "validity check");
    assert(valid_ctx != 0);
    ROCPROFILER_CALL(rocprofiler_context_is_valid(tracing_ctx, &valid_ctx), "validity check");
    assert(valid_ctx != 0);

    ROCPROFILER_CALL(rocprofiler_start_context(tracing_ctx), "context start");

    // no errors
    return 0;
}

void
tool_fini(void* /* tool_data */)
{
    rocprof_trace_decoder_destroy_handle(Decoder::decoder);
    hotspots::write_top_hotspots("thread_trace.log");
}

}  // namespace ThreadTracer

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /* version */,
                      const char* /* runtime_version */,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // only activate if main tool
    if(priority > 0) return nullptr;

    // set the client name
    id->name = "Thread Trace Sample";

    // store client info
    ThreadTracer::client_id = id;

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &ThreadTracer::tool_init,
                                            &ThreadTracer::tool_fini,
                                            nullptr};

    // return pointer to configure data
    return &cfg;
}
