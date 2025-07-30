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

constexpr double WAVE_RATIO_TOLERANCE = 0.05;
constexpr size_t NUM_KERNELS          = 5;

namespace ATTTest
{
namespace Multi
{
rocprofiler_client_id_t* client_id = nullptr;

rocprofiler_thread_trace_control_flags_t
dispatch_callback(rocprofiler_agent_id_t /* agent */,
                  rocprofiler_queue_id_t /* queue_id  */,
                  rocprofiler_async_correlation_id_t /* correlation_id  */,
                  rocprofiler_kernel_id_t /* kernel_id */,
                  rocprofiler_dispatch_id_t /* dispatch_id */,
                  void*                    userdata,
                  rocprofiler_user_data_t* dispatch_userdata)
{
    static std::atomic<size_t> count{0};
    if(count.fetch_add(1) > NUM_KERNELS) return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;

    assert(dispatch_userdata && "Dispatch callback passed null!");
    dispatch_userdata->ptr = userdata;

    return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
}

int
tool_init(rocprofiler_client_finalize_t /* fini_func */, void* /* tool_data */)
{
    Callbacks::init();

    static rocprofiler_context_id_t client_ctx = {0};

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       Callbacks::tool_codeobj_tracing_callback,
                                                       nullptr),
        "code object tracing service configure");

    std::vector<rocprofiler_thread_trace_parameter_t> params{};
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SERIALIZE_ALL, 1});

    std::vector<rocprofiler_agent_id_t> agents{};

    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(
            ROCPROFILER_AGENT_INFO_VERSION_0,
            [](rocprofiler_agent_version_t, const void** _agents, size_t _num_agents, void* _data) {
                auto* agent_v = static_cast<std::vector<rocprofiler_agent_id_t>*>(_data);
                for(size_t i = 0; i < _num_agents; ++i)
                {
                    auto* agent = static_cast<const rocprofiler_agent_v0_t*>(_agents[i]);
                    if(agent->type == ROCPROFILER_AGENT_TYPE_GPU) agent_v->emplace_back(agent->id);
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            sizeof(rocprofiler_agent_v0_t),
            &agents),
        "Failed to iterate agents");

    for(auto id : agents)
    {
        ROCPROFILER_CALL(
            rocprofiler_configure_dispatch_thread_trace_service(client_ctx,
                                                                id,
                                                                params.data(),
                                                                params.size(),
                                                                dispatch_callback,
                                                                Callbacks::shader_data_callback,
                                                                nullptr),
            "thread trace service configure");
    }

    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                     "context validity check");
    if(valid_ctx == 0)
    {
        // notify rocprofiler that initialization failed
        // and all the contexts, buffers, etc. created
        // should be ignored
        return -1;
    }

    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "context start");

    // no errors
    return 0;
}

}  // namespace Multi
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
    id->name = "ATT_test_multi_dispatch";

    // store client info
    ATTTest::Multi::client_id = id;

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &ATTTest::Multi::tool_init,
                                            &Callbacks::finalize,
                                            nullptr};

    // return pointer to configure data
    return &cfg;
}
