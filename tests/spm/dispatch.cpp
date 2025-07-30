// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common.hpp"

#define C_API_BEGIN                                                                                \
    try                                                                                            \
    {
#define C_API_END                                                                                  \
    }                                                                                              \
    catch(std::exception & e)                                                                      \
    {                                                                                              \
        std::cerr << "Error in " << __FILE__ << ':' << __LINE__ << ' ' << e.what() << std::endl;   \
    }                                                                                              \
    catch(...) { std::cerr << "Error in " << __FILE__ << ':' << __LINE__ << std::endl; }

namespace SPMTest
{
namespace Agent
{
rocprofiler_client_id_t*                  client_id = nullptr;
rocprofiler_context_id_t                  spm_ctx   = {};
std::unordered_map<uint64_t, std::string> kernel_id_to_kernel_name{};

void
codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                         rocprofiler_user_data_t* /* user_data */,
                         void* /* userdata */)
{
    C_API_BEGIN
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    if(record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data =
            static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(
                record.payload);
        kernel_id_to_kernel_name.emplace(data->kernel_id, data->kernel_name);
    }

    C_API_END
}

int
dispatch_callback(rocprofiler_agent_id_t /* agent_id */,
                  rocprofiler_queue_id_t /* queue_id */,
                  rocprofiler_kernel_id_t kernel_id,
                  rocprofiler_dispatch_id_t /* dispatch_id */,
                  void* /* config_userdata */,
                  rocprofiler_user_data_t* /* dispatch_userdata */
)
{
    if(kernel_id_to_kernel_name.at(kernel_id).find("__amd") == 0) return 0;

    static std::atomic<int> trigger{0};
    if(trigger.fetch_add(1) > 0) return 0;

    return 1;
}

rocprofiler_status_t
query_available_agents(rocprofiler_agent_version_t /* version */,
                       const void** agents,
                       size_t       num_agents,
                       void* /* user_data */)
{
    std::vector<rocprofiler_spm_parameter_t> params{};
    params.push_back({ROCPROFILER_SPM_PARAMETER_SAMPLE_FREQUENCY, 640000});
    params.push_back({ROCPROFILER_SPM_PARAMETER_TIMEOUT_MS, 10});

    for(size_t idx = 0; idx < num_agents; idx++)
    {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
        if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

        auto counters = common::init_counters(agent->id);

        ROCPROFILER_CALL(rocprofiler_configure_spm_dispatch_service(spm_ctx,
                                                                    agent->id,
                                                                    counters.data(),
                                                                    counters.size(),
                                                                    params.data(),
                                                                    params.size(),
                                                                    dispatch_callback,
                                                                    common::spm_data_callback,
                                                                    nullptr),
                         "thread trace service configure");
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

int
tool_init(rocprofiler_client_finalize_t /* fini_func */, void* tool_data)
{
    ROCPROFILER_CALL(rocprofiler_create_context(&spm_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(spm_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       codeobj_tracing_callback,
                                                       nullptr),
        "code object tracing service configure");

    ROCPROFILER_CALL(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                                        &query_available_agents,
                                                        sizeof(rocprofiler_agent_t),
                                                        tool_data),
                     "Failed to find GPU agents");

    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(spm_ctx, &valid_ctx), "validity check");
    assert(valid_ctx != 0);

    ROCPROFILER_CALL(rocprofiler_start_context(spm_ctx), "context start");

    // no errors
    return 0;
}

void
tool_fini(void* /* tool_data */)
{
    common::finalize();
}

}  // namespace Agent
}  // namespace SPMTest

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /* version */,
                      const char* /* runtime_version */,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // only activate if main tool
    if(priority > 0) return nullptr;

    // set the client name
    id->name = "SPM_test_agent";

    // store client info
    SPMTest::Agent::client_id = id;

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &SPMTest::Agent::tool_init,
                                            &SPMTest::Agent::tool_fini,
                                            nullptr};

    // return pointer to configure data
    return &cfg;
}
