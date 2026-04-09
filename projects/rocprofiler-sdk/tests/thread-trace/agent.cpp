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

#include "trace_callbacks.hpp"

#include <atomic>
#include <mutex>
#include <set>

namespace ATTTest
{
namespace Agent
{
rocprofiler_client_id_t* client_id   = nullptr;
rocprofiler_context_id_t agent_ctx   = {};
rocprofiler_context_id_t tracing_ctx = {};

// Callback state allocated on heap to control destruction order
struct CallbackState
{
    std::atomic<bool> isprofiling{false};
    std::atomic<bool> stop_profiling{false};
    std::mutex        mut{};
    std::set<int>     captured_ids{};
};

CallbackState* callback_state = nullptr;

void
tool_fini(void* tool_data)
{
    // Stop contexts to ensure no more callbacks are dispatched before static destruction
    rocprofiler_stop_context(tracing_ctx);
    rocprofiler_stop_context(agent_ctx);

    // Call the shared finalize logic
    Callbacks::finalize(tool_data);

    // Clean up heap-allocated callback state after finalize
    delete callback_state;
    callback_state = nullptr;
}

void
dispatch_tracing_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t* /* user_data */,
                          void* /* userdata */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH) return;
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) return;

    // Check if callback_state is still valid (may be null during shutdown)
    if(!callback_state) return;

    assert(record.payload);
    auto* rdata = static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload);
    auto  dispatch_id = rdata->dispatch_info.dispatch_id;

    // Choose two dispatches to begin(6) and end(10) the trace
    constexpr uint64_t begin_dispatch = 6;
    constexpr uint64_t end_dispatch   = 10;

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        if(dispatch_id == begin_dispatch)
        {
            ROCPROFILER_CALL(rocprofiler_start_context(agent_ctx), "context start");
            callback_state->isprofiling.store(true);
        }
        if(callback_state->isprofiling && dispatch_id <= end_dispatch)
        {
            std::unique_lock<std::mutex> lk(callback_state->mut);
            callback_state->captured_ids.insert(dispatch_id);
        }
        if(dispatch_id > end_dispatch) callback_state->stop_profiling.store(true);
        return;
    }

    assert(record.phase == ROCPROFILER_CALLBACK_PHASE_NONE);

    if(!callback_state->isprofiling) return;

    std::unique_lock<std::mutex> lk(callback_state->mut);
    callback_state->captured_ids.erase(dispatch_id);
    if(!callback_state->captured_ids.empty() || callback_state->stop_profiling == false) return;

    bool _exp = true;
    if(!callback_state->isprofiling.compare_exchange_strong(_exp, false, std::memory_order_relaxed))
        return;

    ROCPROFILER_CALL(rocprofiler_stop_context(agent_ctx), "context stop");
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

        uint64_t buffer_size_gb = 1;

        // Are we testing for larger buffers?
        if(const char* var = std::getenv("ATT_LARGE_BUFFER_TEST"); var && atoi(var))
        {
            // To fully test this feature, we need >4GB per shader engine (>8GB total).
            // Some RDNA GPUs only have 8GB of VRAM, so we have to use 5GB total = 2.5GB per SE.
            uint64_t total_memory = 0;
            for(uint32_t i = 0; i < agent->mem_banks_count; i++)
                total_memory += agent->mem_banks[i].size_in_bytes;

            // Check we have >11GB VRAM. If so, allocate 10GB.
            if(total_memory > (11ul << 30))
                buffer_size_gb = 10;
            else
                buffer_size_gb = 5;
        }

        uint64_t buffer_size_bytes = buffer_size_gb << 30;
        if(agent->gfx_target_version / 10000 == 11u)
            buffer_size_bytes = 255ul << 20;  // gfx11 limititation

        auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{};
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, {1}});
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, {0xF}});
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {buffer_size_bytes}});
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {0x3}});

        static const bool extra_args =
            std::getenv("ATT_NODETAIL") ? std::stoi(std::getenv("ATT_NODETAIL")) != 0 : false;
        if(extra_args)
        {
            // Dont generate instruction profiling, only occupancy and shaderdata
            parameters.emplace_back(rocprofiler_thread_trace_parameter_t{
                ROCPROFILER_THREAD_TRACE_PARAMETER_NO_DETAIL, {1}});
        }

        ROCPROFILER_CALL(
            rocprofiler_configure_device_thread_trace_service(agent_ctx,
                                                              agent->id,
                                                              parameters.data(),
                                                              parameters.size(),
                                                              Callbacks::shader_data_callback,
                                                              user),
            "thread trace service configure");
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

int
tool_init(rocprofiler_client_finalize_t /* fini_func */, void* /* tool_data */)
{
    Callbacks::init();

    // Allocate callback state on heap for controlled destruction order
    callback_state = new CallbackState{};

    ROCPROFILER_CALL(rocprofiler_create_context(&tracing_ctx), "context creation");
    ROCPROFILER_CALL(rocprofiler_create_context(&agent_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(tracing_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       Callbacks::tool_codeobj_tracing_callback,
                                                       nullptr),
        "code object tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(tracing_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                       nullptr,
                                                       0,
                                                       dispatch_tracing_callback,
                                                       nullptr),
        "dispatch tracing service configure");

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

}  // namespace Agent
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

    // store client info
    ATTTest::Agent::client_id = id;

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &ATTTest::Agent::tool_init,
                                            &ATTTest::Agent::tool_fini,
                                            nullptr};

    // return pointer to configure data
    return &cfg;
}
