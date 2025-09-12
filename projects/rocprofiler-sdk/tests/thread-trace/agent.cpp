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

#include <set>

namespace ATTTest
{
namespace Agent
{
rocprofiler_client_id_t* client_id   = nullptr;
rocprofiler_context_id_t agent_ctx   = {};
rocprofiler_context_id_t tracing_ctx = {};

void
dispatch_tracing_callback(rocprofiler_callback_tracing_record_t record,
                          rocprofiler_user_data_t* /* user_data */,
                          void* /* userdata */)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH) return;
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT) return;

    assert(record.payload);
    auto* rdata = static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload);
    auto  dispatch_id = rdata->dispatch_info.dispatch_id;

    // Choose two dispatches to begin(6) and end(10) the trace
    constexpr uint64_t       begin_dispatch = 6;
    constexpr uint64_t       end_dispatch   = 10;
    static std::atomic<bool> isprofiling{false};
    static std::atomic<bool> stop_profiling{false};

    static std::mutex    mut{};
    static std::set<int> captured_ids{};

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        if(dispatch_id == begin_dispatch)
        {
            ROCPROFILER_CALL(rocprofiler_start_context(agent_ctx), "context start");
            isprofiling.store(true);
        }
        if(isprofiling && dispatch_id <= end_dispatch)
        {
            std::unique_lock<std::mutex> lk(mut);
            captured_ids.insert(dispatch_id);
        }
        if(dispatch_id > end_dispatch) stop_profiling.store(true);
        return;
    }

    assert(record.phase == ROCPROFILER_CALLBACK_PHASE_NONE);

    if(!isprofiling) return;

    std::unique_lock<std::mutex> lk(mut);
    captured_ids.erase(dispatch_id);
    if(!captured_ids.empty() || stop_profiling == false) return;

    bool _exp = true;
    if(!isprofiling.compare_exchange_strong(_exp, false, std::memory_order_relaxed)) return;

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

        // Check if we are testing for large buffers
        static const char* var            = std::getenv("ATT_BUFFER_SIZE_MB");
        static uint64_t    buffer_size_mb = (var ? atoi(var) : 96) * 1024ul * 1024ul;

        std::vector<rocprofiler_thread_trace_parameter_t> parameters;
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, 1});
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, 0xF});
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, buffer_size_mb});
        parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, 0x1});

        static const bool extra_args =
            std::getenv("ATT_NODETAIL") ? std::stoi(std::getenv("ATT_NODETAIL")) != 0 : false;
        if(extra_args)
        {
            // Dont generate instruction profiling, only occupancy and shaderdata
            parameters.emplace_back(rocprofiler_thread_trace_parameter_t{
                ROCPROFILER_THREAD_TRACE_PARAMETER_NO_DETAIL, 1});
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
                                            &Callbacks::finalize,
                                            nullptr};

    // return pointer to configure data
    return &cfg;
}
