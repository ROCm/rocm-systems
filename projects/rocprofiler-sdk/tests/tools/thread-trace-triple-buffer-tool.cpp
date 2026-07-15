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

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    if((result) != ROCPROFILER_STATUS_SUCCESS)                                                     \
    {                                                                                              \
        std::cerr << "Error: " << msg << std::endl;                                                \
        abort();                                                                                   \
    }

namespace ATTTest
{
namespace TripleBuffer
{
constexpr size_t MIN_TRACE_SIZE = 16 << 20;

struct agent_output_buffer_t
{
    agent_output_buffer_t(rocprofiler_agent_id_t _id)
    : id(_id)
    {
        output_buffer.resize(BUFFER_SIZE);
    };
    agent_output_buffer_t(agent_output_buffer_t&& other)
    {
        output_buffer = std::move(other.output_buffer);
        output_size   = other.output_size.load();
        id            = other.id;
    };

    rocprofiler_agent_id_t id{};
    std::vector<char>      output_buffer{};
    std::atomic<size_t>    output_size{0};

    static constexpr size_t BUFFER_SIZE = 256ul << 20;
};

rocprofiler_thread_trace_decoder_id_t decoder{};
rocprofiler_context_id_t              agent_ctx{};
rocprofiler_context_id_t              tracing_ctx{};
std::vector<agent_output_buffer_t>*   agent_buffers{};

void
tool_codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t* /* user_data */,
                              void* /* userdata */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
       record.operation != ROCPROFILER_CODE_OBJECT_LOAD ||
       record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD)
        return;

    auto* data = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);

    if(data->storage_type != ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
    {
        rocprofiler_thread_trace_decoder_codeobj_load(
            decoder,
            data->code_object_id,
            data->load_delta,
            data->load_size,
            reinterpret_cast<const void*>(data->memory_base),
            data->memory_size);
    }
}

void
shader_data_callback(rocprofiler_agent_id_t /* agent */,
                     int64_t /* se_id */,
                     void*  se_data,
                     size_t data_size,
                     rocprofiler_thread_trace_shader_data_flags_t /* flags */,
                     rocprofiler_user_data_t userdata)
{
    static auto* is_slow  = std::getenv("ATT_SLOW_CALLBACK");
    static bool  do_sleep = is_slow ? atoi(is_slow) != 0 : false;

    if(do_sleep) std::this_thread::sleep_for(std::chrono::milliseconds(300));

    auto* agent_output_buffer = static_cast<agent_output_buffer_t*>(userdata.ptr);

    size_t output_buf_size = agent_output_buffer->output_buffer.size();
    size_t location        = agent_output_buffer->output_size.fetch_add(data_size);
    void*  output          = agent_output_buffer->output_buffer.data();

    // Discard
    if(location >= output_buf_size) return;

    data_size = std::min(data_size, output_buf_size - location);

    auto is_ptr_mod8 = [](void* data) { return (reinterpret_cast<std::uintptr_t>(data) % 8) == 0; };
    auto is_int_mod8 = [](size_t data) { return (data % 8) == 0; };

    if(is_int_mod8(location) && is_int_mod8(data_size) && is_ptr_mod8(se_data) &&
       is_ptr_mod8(output))
    {
        for(size_t j = 0; j < data_size / 8; j++)
            static_cast<uint64_t*>(output)[j + location / 8] = static_cast<uint64_t*>(se_data)[j];
    }
    else
    {
        for(size_t j = 0; j < data_size; j++)
            static_cast<char*>(output)[j + location] = static_cast<char*>(se_data)[j];
    }
}

rocprofiler_status_t
query_available_agents(rocprofiler_agent_version_t /* version */,
                       const void** agents,
                       size_t       num_agents,
                       void* /* user_data */)
{
    for(size_t idx = 0; idx < num_agents; idx++)
    {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);

        if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            agent_buffers->emplace_back(ATTTest::TripleBuffer::agent_output_buffer_t{agent->id});
    }

    uint64_t gpu_buffer_size = 64 << 20;

    auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{};
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {1}});
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {gpu_buffer_size}});
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {1}});
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFERING_MODE,
                          ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFERING_MODE_TRIPLE_BUFFER});

    auto* nodetail   = std::getenv("ATT_NODETAIL");
    bool  extra_args = nodetail ? atoi(nodetail) != 0 : false;

    if(extra_args)
    {
        // Dont generate instruction profiling, only occupancy and shaderdata
        parameters.emplace_back(rocprofiler_thread_trace_parameter_t{
            ROCPROFILER_THREAD_TRACE_PARAMETER_NO_DETAIL, {1}});
        gpu_buffer_size = 4 << 20;
    }

    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {gpu_buffer_size}});

    for(auto& agent : *agent_buffers)
    {
        auto userdata = rocprofiler_user_data_t{};
        userdata.ptr  = &agent;
        ROCPROFILER_CALL(rocprofiler_configure_device_thread_trace_service(
                             agent_ctx,
                             agent.id,
                             parameters.data(),
                             parameters.size(),
                             ATTTest::TripleBuffer::shader_data_callback,
                             userdata),
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

        auto parse = [](rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                        void*                                          events,
                        uint64_t                                       num_events,
                        void*                                          userdata) {
            if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO)
            {
                auto* infos = (rocprofiler_thread_trace_decoder_info_t*) events;
                for(size_t i = 0; i < num_events; i++)
                    std::cerr << rocprofiler_thread_trace_decoder_info_string(decoder, infos[i])
                              << std::endl;
            }
            else if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA)
            {
                auto& current_sdata = *static_cast<uint32_t*>(userdata);

                auto* sdata = static_cast<rocprofiler_thread_trace_decoder_shaderdata_t*>(events);
                for(size_t i = 0; i < num_events; i++)
                {
                    if(sdata[i].value < current_sdata)
                    {
                        std::cerr << "Error: Invalid sdata value " << sdata[i].value << std::endl;
                        abort();
                    }
                    current_sdata = sdata[i].value;
                }
            }
        };

        size_t total_size = 0;
        for(auto& output_buffer : *agent_buffers)
        {
            uint32_t current_sdata = 0;
            auto&    buffer        = output_buffer.output_buffer;
            size_t   output_size   = std::min(output_buffer.output_size.exchange(0), buffer.size());
            rocprofiler_trace_decode(decoder, parse, buffer.data(), output_size, &current_sdata);
            total_size += output_size;
        }

        static bool ignore_size = std::getenv("STARTSTOP") ? atoi(std::getenv("STARTSTOP")) : false;
        if(!ignore_size && total_size < MIN_TRACE_SIZE)
            throw std::runtime_error("Trace is too small!");
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
    agent_buffers = new std::vector<agent_output_buffer_t>{};

    rocprofiler_thread_trace_decoder_create(&decoder, "/opt/rocm/lib");

    ROCPROFILER_CALL(rocprofiler_create_context(&agent_ctx), "context creation");
    ROCPROFILER_CALL(rocprofiler_create_context(&tracing_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(tracing_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_codeobj_tracing_callback,
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
    if(valid_ctx == 0) throw std::runtime_error("agent_ctx is not valid!");
    ROCPROFILER_CALL(rocprofiler_context_is_valid(tracing_ctx, &valid_ctx), "validity check");
    if(valid_ctx == 0) throw std::runtime_error("tracing_ctx is not valid!");

    ROCPROFILER_CALL(rocprofiler_start_context(tracing_ctx), "context start");

    // no errors
    return 0;
}

void
tool_fini(void*){};

}  // namespace TripleBuffer
}  // namespace ATTTest

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /* version */,
                      const char* /* runtime_version */,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // only activate if main tool
    if(priority > 0) return nullptr;

    // set the client name
    id->name = "ATT_test_agent";

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &ATTTest::TripleBuffer::tool_init,
                                            &ATTTest::TripleBuffer::tool_fini,
                                            nullptr};

    // return pointer to configure data
    return &cfg;
}
