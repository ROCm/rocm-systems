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

#include </opt/rocm/include/rocprof-trace-decoder/rocprof_trace_decoder/rocprof_trace_decoder.h>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/thread-trace/core.h>
#include <rocprofiler-sdk/experimental/thread-trace/agent.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <future>
#include <thread>

#define ROCPROFILER_CALL(result, msg)                                                              \
    if(auto ec = (result); ec != ROCPROFILER_STATUS_SUCCESS)                                       \
    {                                                                                              \
        std::cerr << "rocprofiler-sdk error at " << __FILE__ << ":" << __LINE__                    \
                  << " :: " << #result << std::endl;                                               \
        std::cerr << "rocprofiler-sdk error code " << ec << ": "                                   \
                  << rocprofiler_get_status_string(ec) << " :: " << msg << std::endl;              \
        abort();                                                                                   \
    }

#define DECODER_CALL(result) if(auto ec = (result); ec != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) abort();

#define CHECK_NOTNULL(x)                                                                           \
    if(!(x))                                                                                       \
    {                                                                                              \
        abort();                                                                                   \
    };

rocprofiler_client_id_t* client_id   = nullptr;
rocprofiler_context_id_t agent_ctx   = {};
rocprofiler_context_id_t tracing_ctx = {};

static std::atomic<bool> started{false};
static std::atomic<bool> ended{false};
std::future<void> thread{};

void wait_and_turn_off_context()
{
    auto* timeout_var = std::getenv("SQTT_TIMEOUT");
    auto timeout = timeout_var ? std::atoi(timeout_var) : 1;

    std::cout << "Waiting for " << timeout << " ms." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(timeout));

    if (!ended.exchange(false)) ROCPROFILER_CALL(rocprofiler_stop_context(agent_ctx), "stopping context");
    std::cout << "FINALIZED!" << std::endl;
}

void
tool_codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t* /* user_data */,
                              void* /* userdata */)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD &&
       record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
    {
        if (started.exchange(true)) return;
        std::cout << "STARTING!" << std::endl;
        rocprofiler_start_context(agent_ctx);
        thread = std::async(std::launch::async, wait_and_turn_off_context);
    }
}

void
shader_data_callback(rocprofiler_thread_trace_shader_data_t shader_data,
                     rocprofiler_user_data_t /* userdata */)
{
    auto name = "agent_0_dispatch_0_shader_engine_" + std::to_string(shader_data.shader_engine_id) +".att";

    std::cout << "Writing: " << name << std::endl;
    std::ofstream file(name, std::ios::out | std::ios::binary);
    if (file.good()) file.write(static_cast<char*>(shader_data.data), shader_data.data_size);
    if (!file.good()) std::cerr << "Failed to write " << name << std::endl;

    auto parse = [](rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                    void*                                          events,
                    uint64_t                                       num_events,
                    void* /* userdata */) { return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS; };

    auto decoder = rocprof_trace_decoder_handle_t{};
    DECODER_CALL(rocprof_trace_decoder_create_handle(&decoder));
    DECODER_CALL(rocprof_trace_decoder_parse(decoder, shader_data.data, shader_data.data_size, parse, nullptr));
}

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
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {uint64_t(1)<<32}});
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {0xFFFF}});
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_NO_DETAIL, {1}});

        ROCPROFILER_CALL(
            rocprofiler_configure_device_thread_trace_service(agent_ctx,
                                                              agent->id,
                                                              parameters.data(),
                                                              parameters.size(),
                                                              shader_data_callback,
                                                              user),
            "thread trace service configure");
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

int
tool_init(rocprofiler_client_finalize_t /* fini_func */, void* /* tool_data */)
{
    ROCPROFILER_CALL(rocprofiler_create_context(&tracing_ctx), "context creation");
    ROCPROFILER_CALL(rocprofiler_create_context(&agent_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(tracing_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_codeobj_tracing_callback,
                                                       nullptr),
        "code object tracing service configure");

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
    std::cout << "DESTRUCTOR!" << std::endl;
    if (thread.valid()) thread.get();
}

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
    client_id = id;

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &tool_init,
                                            &tool_fini,
                                            nullptr};

    // return pointer to configure data
    return &cfg;
}
