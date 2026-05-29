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

#include <rocprof_trace_decoder/rocprof_trace_decoder.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
        output_buffer       = std::move(other.output_buffer);
        output_size         = other.output_size.load();
        id                  = other.id;
        next_expected_chunk = other.next_expected_chunk;
    };

    rocprofiler_agent_id_t id{};
    std::vector<char>      output_buffer{};
    std::atomic<size_t>    output_size{0};

    // Reordering state. Multi-consumer ATT can deliver chunks out of order;
    // each callback blocks until its chunk_index is the next expected, then
    // writes its data, advances the counter, and wakes the next waiter.
    std::mutex              reorder_mut{};
    std::condition_variable reorder_cv{};
    uint64_t                next_expected_chunk{0};

    static constexpr size_t BUFFER_SIZE = 256ul << 20;
};

rocprof_trace_decoder_handle_t      decoder{};
rocprofiler_context_id_t            agent_ctx{};
rocprofiler_context_id_t            tracing_ctx{};
std::vector<agent_output_buffer_t>* agent_buffers{};

// Tracks REALTIME records to verify that the gfx9 query_status path emits
// periodic RT_TIMESTAMP_LO32 markers in addition to the Begin/End anchors.
//
// Notes on validation scope:
//   - shader_clock is per-decode-pass (each rocprofiler_trace_decode call has
//     its own time origin), so monotonicity must be checked per-pass only.
//   - realtime_clock is the GPU wall-clock and IS monotonic across passes.
//   - "Periodic emission worked" means at least one pass produced > 2 records
//     (Begin + End anchors plus periodic markers). Cross-pass total isn't
//     enough on its own because N anchor-only passes also exceed 2.
std::atomic<size_t> realtime_total_count{0};
std::atomic<size_t> realtime_max_per_pass{0};
std::atomic<bool>   realtime_monotonicity_violation{false};

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
        rocprof_trace_decoder_codeobj_load(decoder,
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
                     uint64_t chunk_index,
                     void*    se_data,
                     size_t   data_size,
                     rocprofiler_thread_trace_shader_data_flags_t /* flags */,
                     rocprofiler_user_data_t userdata)
{
    static auto* is_slow  = std::getenv("ATT_SLOW_CALLBACK");
    static bool  do_sleep = is_slow ? atoi(is_slow) != 0 : false;

    if(do_sleep) std::this_thread::sleep_for(std::chrono::milliseconds(300));

    auto* agent_output_buffer = static_cast<agent_output_buffer_t*>(userdata.ptr);

    // Multi-consumer ATT can deliver chunks out of order. Block until our
    // chunk_index is the next-expected one, then write directly to the
    // output buffer in order.
    {
        auto lk = std::unique_lock{agent_output_buffer->reorder_mut};
        agent_output_buffer->reorder_cv.wait(
            lk, [&] { return agent_output_buffer->next_expected_chunk == chunk_index; });
    }

    size_t output_buf_size = agent_output_buffer->output_buffer.size();
    size_t location        = agent_output_buffer->output_size.fetch_add(data_size);
    void*  output          = agent_output_buffer->output_buffer.data();

    // Advance and wake the next-in-line consumer regardless of whether we
    // had room to actually write the bytes.
    auto release = [&] {
        {
            auto lk = std::unique_lock{agent_output_buffer->reorder_mut};
            agent_output_buffer->next_expected_chunk = chunk_index + 1;
        }
        agent_output_buffer->reorder_cv.notify_all();
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
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_NUM_BUFFERS, {3}});

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

        struct parse_state_t
        {
            uint32_t current_sdata    = 0;
            size_t   realtime_count   = 0;
            int64_t  prev_shader      = 0;
            uint64_t prev_realtime    = 0;
            bool     monotonicity_bad = false;
        };

        auto parse = [](rocprofiler_thread_trace_decoder_record_type_t record_type_id,
                        void*                                          events,
                        uint64_t                                       num_events,
                        void* userdata) -> rocprofiler_thread_trace_decoder_status_t {
            auto& state = *static_cast<parse_state_t*>(userdata);

            if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO)
            {
                auto* infos = (rocprofiler_thread_trace_decoder_info_t*) events;
                for(size_t i = 0; i < num_events; i++)
                    std::cerr << rocprof_trace_decoder_get_info_string(infos[i]) << std::endl;
            }
            else if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_SHADERDATA)
            {
                auto* sdata = static_cast<rocprofiler_thread_trace_decoder_shaderdata_t*>(events);
                for(size_t i = 0; i < num_events; i++)
                {
                    if(sdata[i].value < state.current_sdata)
                    {
                        std::cerr << "Error: Invalid sdata value " << sdata[i].value << std::endl;
                        abort();
                    }
                    state.current_sdata = sdata[i].value;
                }
            }
            else if(record_type_id == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_REALTIME)
            {
                auto* rts = static_cast<rocprofiler_thread_trace_decoder_realtime_t*>(events);
                for(size_t i = 0; i < num_events; i++)
                {
                    // Per-pass monotonicity: shader_clock has a per-pass time
                    // origin, so this can only be checked within one decode.
                    if(rts[i].shader_clock < state.prev_shader ||
                       rts[i].realtime_clock < state.prev_realtime)
                    {
                        state.monotonicity_bad = true;
                    }
                    state.prev_shader   = rts[i].shader_clock;
                    state.prev_realtime = rts[i].realtime_clock;
                }
                state.realtime_count += num_events;
            }
            return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
        };

        size_t total_size = 0;
        for(auto& output_buffer : *agent_buffers)
        {
            parse_state_t state{};
            auto&         buffer = output_buffer.output_buffer;
            size_t output_size   = std::min(output_buffer.output_size.exchange(0), buffer.size());
            rocprof_trace_decoder_parse(decoder, buffer.data(), output_size, parse, &state);
            total_size += output_size;

            // Reset reorder state so the next start_context cycle (whose
            // chunk_index restarts at 0) lands in the correct slot. The
            // producer/consumer threads have been stopped above, so no
            // callback can be in flight here.
            {
                std::lock_guard<std::mutex> lk(output_buffer.reorder_mut);
                output_buffer.next_expected_chunk = 0;
            }

            realtime_total_count.fetch_add(state.realtime_count);
            // Track the largest single-pass count to prove periodic emission
            // happened within at least one pass (anchor-only passes give 2).
            size_t prev_max = realtime_max_per_pass.load();
            while(state.realtime_count > prev_max &&
                  !realtime_max_per_pass.compare_exchange_weak(prev_max, state.realtime_count))
            {}
            if(state.monotonicity_bad) realtime_monotonicity_violation = true;
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

    rocprof_trace_decoder_create_handle(&decoder);

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
tool_fini(void*)
{
    if(realtime_monotonicity_violation.load())
    {
        std::cerr << "Error: REALTIME records were not monotonic within a decode pass" << std::endl;
        abort();
    }

    // gfx9 query_status path emits periodic RT_TIMESTAMP_LO32 markers in
    // addition to the Begin/End anchors. Gate the strict count check behind an
    // env var so the same tool keeps working on gfx10+ where only the two
    // boundary anchors exist. Use the per-pass max because N anchor-only
    // passes also exceed 2 globally; we want proof that periodic emission
    // happened within at least one decode pass.
    const char* expect_env = std::getenv("EXPECT_PERIODIC_REALTIME");
    if(expect_env && atoi(expect_env) != 0)
    {
        size_t max_pass = realtime_max_per_pass.load();
        if(max_pass <= 2)
        {
            std::cerr << "Error: expected periodic REALTIME records within a single pass (>2), "
                      << "got max-per-pass=" << max_pass << " total=" << realtime_total_count.load()
                      << std::endl;
            abort();
        }
    }

    rocprof_trace_decoder_destroy_handle(decoder);
}

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
