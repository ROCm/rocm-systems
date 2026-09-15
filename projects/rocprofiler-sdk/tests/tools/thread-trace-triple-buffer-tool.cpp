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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
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
constexpr size_t   MIN_TRACE_SIZE         = 16 << 20;
constexpr uint32_t SHADER_ENGINE_ID_SHIFT = 20;
constexpr uint32_t SHADER_DATA_MASK       = (uint32_t{1} << SHADER_ENGINE_ID_SHIFT) - 1;

struct shader_engine_output_buffer_t
{
    explicit shader_engine_output_buffer_t(size_t buffer_size)
    {
        output_buffer.resize(buffer_size);
    }

    std::vector<char>   output_buffer{};
    std::atomic<size_t> output_size{0};

    // Reordering state. Multi-consumer ATT can deliver chunks out of order;
    // each callback blocks until its chunk_index is the next expected, then
    // writes its data, advances the counter, and wakes the next waiter.
    std::mutex              reorder_mut{};
    std::condition_variable reorder_cv{};
    uint64_t                next_expected_chunk{0};

    std::vector<size_t> end_chunks{};
};

struct agent_output_buffer_t
{
    agent_output_buffer_t(rocprofiler_agent_id_t _id,
                          uint32_t               num_shader_engines,
                          bool                   enable_all_shader_engines)
    : id(_id)
    {
        const auto enabled_shader_engines = enable_all_shader_engines ? num_shader_engines : 1u;

        shader_engines.reserve(enabled_shader_engines);
        for(uint32_t i = 0; i < enabled_shader_engines; ++i)
            shader_engines.emplace_back(
                std::make_unique<shader_engine_output_buffer_t>(BUFFER_SIZE));
    }

    rocprofiler_agent_id_t                                      id{};
    std::vector<std::unique_ptr<shader_engine_output_buffer_t>> shader_engines{};

    static constexpr size_t BUFFER_SIZE = 512ul << 20;
};

struct shader_data_parse_state_t
{
    uint32_t expected_shader_engine_id = 0;
    uint32_t current_dispatch_id       = 0;
    size_t   record_count              = 0;
    bool     has_dispatch              = false;
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
shader_data_callback(rocprofiler_thread_trace_shader_data_t shader_data,
                     rocprofiler_user_data_t                userdata)
{
    auto  chunk_index = shader_data.chunk_index;
    auto* se_data     = shader_data.data;
    auto  data_size   = static_cast<size_t>(shader_data.data_size);

    static auto* is_slow  = std::getenv("ATT_SLOW_CALLBACK");
    static bool  do_sleep = is_slow ? atoi(is_slow) != 0 : false;
    static bool  is_multi_se_test =
        std::getenv("ATT_MULTI_SE") ? atoi(std::getenv("ATT_MULTI_SE")) != 0 : false;

    if(do_sleep) std::this_thread::sleep_for(std::chrono::milliseconds(300));

    constexpr auto buffer_full_flags = ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL |
                                       ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL;
    if(is_multi_se_test && (shader_data.flags & buffer_full_flags))
        throw std::runtime_error("Thread trace buffer filled during multi-SE consistency test");

    auto* agent_output_buffer = static_cast<agent_output_buffer_t*>(userdata.ptr);
    auto  shader_engine_id    = shader_data.shader_engine_id;

    if(shader_engine_id < 0 ||
       static_cast<size_t>(shader_engine_id) >= agent_output_buffer->shader_engines.size())
    {
        std::cerr << "Received data for unexpected shader engine " << shader_engine_id << std::endl;
        abort();
    }

    auto& shader_engine = *agent_output_buffer->shader_engines.at(shader_engine_id);

    // Multi-consumer ATT can deliver chunks out of order. Block until our
    // chunk_index is the next-expected one, then write directly to the
    // output buffer in order.
    {
        auto lk = std::unique_lock{shader_engine.reorder_mut};
        shader_engine.reorder_cv.wait(
            lk, [&] { return shader_engine.next_expected_chunk == chunk_index; });
    }

    size_t output_buf_size = shader_engine.output_buffer.size();
    size_t location        = shader_engine.output_size.fetch_add(data_size);
    void*  output          = shader_engine.output_buffer.data();

    if(is_multi_se_test && (location > output_buf_size || data_size > output_buf_size - location))
    {
        std::cerr << "Host output buffer filled for SE " << shader_engine_id << std::endl;
        abort();
    }

    // Advance and wake the next-in-line consumer regardless of whether we
    // had room to actually write the bytes.
    auto release = [&] {
        {
            auto lk                           = std::unique_lock{shader_engine.reorder_mut};
            shader_engine.next_expected_chunk = chunk_index + 1;

            if(shader_data.flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END)
                shader_engine.end_chunks.push_back(location + data_size);
        }
        shader_engine.reorder_cv.notify_all();
    };

    if(location >= output_buf_size)
    {
        release();
        return;
    }

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

    release();
}

rocprofiler_status_t
query_available_agents(rocprofiler_agent_version_t /* version */,
                       const void** agents,
                       size_t       num_agents,
                       void* /* user_data */)
{
    static const bool enable_all_shader_engines =
        std::getenv("ATT_MULTI_SE") ? atoi(std::getenv("ATT_MULTI_SE")) != 0 : false;

    for(size_t idx = 0; idx < num_agents; idx++)
    {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);

        if(agent->type == ROCPROFILER_AGENT_TYPE_GPU && agent->runtime_visibility.hsa)
            agent_buffers->emplace_back(
                agent->id, agent->num_shader_banks, enable_all_shader_engines);
    }

    auto* nodetail   = std::getenv("ATT_NODETAIL");
    bool  extra_args = nodetail ? atoi(nodetail) != 0 : false;

    for(auto& agent : *agent_buffers)
    {
        const auto num_shader_engines = static_cast<uint64_t>(agent.shader_engines.size());
        const auto num_buffers        = enable_all_shader_engines ? 4 + num_shader_engines : 3;
        const auto shader_engine_mask =
            enable_all_shader_engines ? std::numeric_limits<uint32_t>::max() : 1u;
        const uint64_t gpu_buffer_size =
            (extra_args ? 4ul : 16ul) * (enable_all_shader_engines ? num_shader_engines : 1) << 20;

        auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{};
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {1}});
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {gpu_buffer_size}});
        parameters.push_back(
            {ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {shader_engine_mask}});
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_NUM_BUFFERS, {num_buffers}});

        if(extra_args)
        {
            // Don't generate instruction profiling, only occupancy and shaderdata.
            parameters.emplace_back(rocprofiler_thread_trace_parameter_t{
                ROCPROFILER_THREAD_TRACE_PARAMETER_NO_DETAIL, {1}});
        }

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

        auto parse_multi_se = [](rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                                 void*                                          events,
                                 uint64_t                                       num_events,
                                 void*                                          userdata) {
            if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO)
            {
                auto* infos = static_cast<rocprofiler_thread_trace_decoder_info_t*>(events);
                for(size_t i = 0; i < num_events; i++)
                    std::cerr << rocprofiler_thread_trace_decoder_info_string(decoder, infos[i])
                              << std::endl;
            }
            else if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA)
            {
                auto& state = *static_cast<shader_data_parse_state_t*>(userdata);
                auto* sdata = static_cast<rocprofiler_thread_trace_decoder_shaderdata_t*>(events);
                for(size_t i = 0; i < num_events; i++)
                {
                    const auto shader_engine_id = sdata[i].value >> SHADER_ENGINE_ID_SHIFT;
                    const auto dispatch_id      = sdata[i].value & SHADER_DATA_MASK;

                    if(shader_engine_id != state.expected_shader_engine_id)
                    {
                        std::cerr << i << " Error: Shaderdata tagged for SE " << shader_engine_id
                                  << " appeared in SE " << state.expected_shader_engine_id
                                  << std::endl;
                        abort();
                    }

                    if(!state.has_dispatch)
                    {
                        state.current_dispatch_id = dispatch_id;
                        state.has_dispatch        = true;
                    }
                    else if(dispatch_id < state.current_dispatch_id)
                    {
                        std::cerr << i << " Error: Dispatch " << dispatch_id
                                  << " followed dispatch " << state.current_dispatch_id << " on SE "
                                  << state.expected_shader_engine_id << std::endl;
                        abort();
                    }

                    state.current_dispatch_id = dispatch_id;
                    ++state.record_count;
                }
            }
        };

        auto parse_single_se = [](rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                                  void*                                          events,
                                  uint64_t                                       num_events,
                                  void*                                          userdata) {
            if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO)
            {
                auto* infos = static_cast<rocprofiler_thread_trace_decoder_info_t*>(events);
                for(size_t i = 0; i < num_events; ++i)
                    std::cerr << rocprofiler_thread_trace_decoder_info_string(decoder, infos[i])
                              << std::endl;
            }
            else if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA)
            {
                auto& current_sdata = *static_cast<uint32_t*>(userdata);
                auto* sdata = static_cast<rocprofiler_thread_trace_decoder_shaderdata_t*>(events);
                for(size_t i = 0; i < num_events; ++i)
                {
                    // The final token in a segment may have been cut off when the buffer filled.
                    if(i != num_events - 1 && sdata[i].value < current_sdata)
                    {
                        std::cerr << i << " Error: Invalid sdata value " << sdata[i].value << " vs "
                                  << current_sdata << std::endl;
                        abort();
                    }
                    current_sdata = sdata[i].value;
                }
            }
        };

        static const bool is_multi_se_test =
            std::getenv("ATT_MULTI_SE") ? atoi(std::getenv("ATT_MULTI_SE")) != 0 : false;

        size_t total_size = 0;
        for(auto& agent_output : *agent_buffers)
        {
            for(size_t shader_engine_id = 0; shader_engine_id < agent_output.shader_engines.size();
                ++shader_engine_id)
            {
                auto&      shader_engine = *agent_output.shader_engines.at(shader_engine_id);
                auto       lk            = std::unique_lock{shader_engine.reorder_mut};
                auto&      buffer        = shader_engine.output_buffer;
                const auto emitted_size  = shader_engine.output_size.exchange(0);
                const auto output_size   = std::min(emitted_size, buffer.size());

                if(is_multi_se_test)
                {
                    auto state                      = shader_data_parse_state_t{};
                    state.expected_shader_engine_id = static_cast<uint32_t>(shader_engine_id);
                    ROCPROFILER_CALL(
                        rocprofiler_trace_decode(
                            decoder, parse_multi_se, buffer.data(), output_size, &state),
                        "trace decode");

                    if(state.record_count == 0)
                    {
                        std::cerr << "Incomplete shaderdata for SE " << shader_engine_id
                                  << ": decoded " << state.record_count << " records" << std::endl;
                        abort();
                    }
                }
                else
                {
                    size_t current_byte = 0;
                    for(auto end_byte : shader_engine.end_chunks)
                    {
                        if(current_byte >= output_size) break;

                        const auto bounded_end_byte = std::min(end_byte, output_size);
                        auto       current_sdata    = uint32_t{0};
                        rocprofiler_trace_decode(decoder,
                                                 parse_single_se,
                                                 buffer.data() + current_byte,
                                                 bounded_end_byte - current_byte,
                                                 &current_sdata);
                        current_byte = end_byte;
                    }
                }

                total_size += output_size;
                shader_engine.next_expected_chunk = 0;
                shader_engine.end_chunks.clear();
            }
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
