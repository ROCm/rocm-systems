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
#include <vector>

constexpr double WAVE_RATIO_TOLERANCE = 0.05;

namespace ATTTest
{
namespace Multi
{
rocprofiler_client_id_t* client_id = nullptr;

rocprofiler_att_control_flags_t
dispatch_callback(rocprofiler_queue_id_t /* queue_id  */,
                  const rocprofiler_agent_t* /* agent  */,
                  rocprofiler_correlation_id_t /* correlation_id  */,
                  rocprofiler_kernel_id_t /* kernel_id */,
                  rocprofiler_dispatch_id_t /* dispatch_id */,
                  rocprofiler_user_data_t* dispatch_userdata,
                  void*                    userdata)
{
    C_API_BEGIN

    assert(userdata && "Dispatch callback passed null!");
    dispatch_userdata->ptr = userdata;

    C_API_END
    return ROCPROFILER_ATT_CONTROL_START_AND_STOP;
}

int
tool_init(rocprofiler_client_finalize_t /* fini_func */, void* tool_data)
{
    Callbacks::callbacks_init();
    static rocprofiler_context_id_t client_ctx = {0};

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       Callbacks::tool_codeobj_tracing_callback,
                                                       tool_data),
        "code object tracing service configure");

    std::vector<rocprofiler_att_parameter_t> params{};

    ROCPROFILER_CALL(
        rocprofiler_configure_dispatch_thread_trace_service(client_ctx,
                                                            params.data(),
                                                            params.size(),
                                                            dispatch_callback,
                                                            Callbacks::shader_data_callback,
                                                            tool_data),
        "thread trace service configure");

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

void
tool_fini(void* tool_data)
{
    assert(tool_data && "tool_fini callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(tool_data);

    double wave_started     = (double) tool.waves_started.load();
    double wave_event_ratio = wave_started / (wave_started + (double) tool.waves_ended.load());
    assert(wave_event_ratio > 0.5 - WAVE_RATIO_TOLERANCE);
    assert(wave_event_ratio < 0.5 + WAVE_RATIO_TOLERANCE);

    // Expected: Two kernels in  kernel_run.cpp
    assert(tool.wave_start_locations.size() >= 2);

    // Expected at least one known code object ID
    bool bHasMarkerId = false;
    for(auto& pc : tool.wave_start_locations)
        bHasMarkerId |= pc.marker_id != 0;

    assert(bHasMarkerId);

    Callbacks::callbacks_fini();
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

    auto* data = new ATTTest::ToolData{};

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &ATTTest::Multi::tool_init,
                                            &ATTTest::Multi::tool_fini,
                                            reinterpret_cast<void*>(data)};

    // return pointer to configure data
    return &cfg;
}
