// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/amd_detail/thread_trace.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include "common.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#define HIP_API_CALL(CALL) assert((CALL) == hipSuccess)

namespace ATTTest
{
namespace Agent
{
rocprofiler_context_id_t client_ctx = {0};
rocprofiler_client_id_t* client_id  = nullptr;
std::atomic<bool>        valid_data{false};

void
shader_data_callback(int64_t /* se_id */,
                     void*  se_data,
                     size_t data_size,
                     rocprofiler_user_data_t /* userdata */)
{
    if(se_data && data_size) valid_data.store(true);
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
        if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

        ROCPROFILER_CALL(rocprofiler_configure_agent_thread_trace_service(
                             client_ctx, nullptr, 0, agent->id, shader_data_callback, nullptr),
                         "thread trace service configure");

        return ROCPROFILER_STATUS_SUCCESS;
    }
    return ROCPROFILER_STATUS_ERROR;
}

int
tool_init(rocprofiler_client_finalize_t /* fini_func */, void* /* tool_data */)
{
    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");

    ROCPROFILER_CALL(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                                        query_available_agents,
                                                        sizeof(rocprofiler_agent_t),
                                                        nullptr),
                     "");

    int valid = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid), "context validity check");
    return (valid == 0) ? -1 : 0;
}

void
tool_fini(void* /* tool_data */)
{
    assert(valid_data.load());
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
    id->name = "ATT_test_agent_api";

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

void
run(int dev)
{
    constexpr size_t size = 0x1000;
    float*           ptr  = nullptr;

    HIP_API_CALL(hipSetDevice(dev));

    HIP_API_CALL(hipMalloc(&ptr, size * sizeof(float)));
    HIP_API_CALL(hipMemset(ptr, 0x55, size * sizeof(float)));
    HIP_API_CALL(hipFree(ptr));
}

int
main()
{
    int ndev = 0;
    HIP_API_CALL(hipGetDeviceCount(&ndev));

    for(int dev = 0; dev < ndev; dev++)
        run(dev);

    ROCPROFILER_CALL(rocprofiler_start_context(ATTTest::Agent::client_ctx), "context start");

    for(int dev = 0; dev < ndev; dev++)
        run(dev);
    usleep(100);

    ROCPROFILER_CALL(rocprofiler_stop_context(ATTTest::Agent::client_ctx), "context stop");
    return 0;
}
