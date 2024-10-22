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

constexpr double WAVE_RATIO_TOLERANCE = 0.05;

namespace ATTTest
{
namespace Single
{
rocprofiler_client_id_t* client_id = nullptr;

rocprofiler_att_control_flags_t
dispatch_callback(rocprofiler_queue_id_t /* queue_id  */,
                  const rocprofiler_agent_t* /* agent  */,
                  rocprofiler_correlation_id_t /* correlation_id  */,
                  rocprofiler_kernel_id_t kernel_id,
                  rocprofiler_dispatch_id_t /* dispatch_id */,
                  rocprofiler_user_data_t* dispatch_userdata,
                  void*                    userdata)
{
    C_API_BEGIN
    assert(userdata && "Dispatch callback passed null!");
    ToolData& tool         = *reinterpret_cast<ToolData*>(userdata);
    dispatch_userdata->ptr = userdata;

    static std::string_view desired_func_name = "branching_kernel";

    try
    {
        auto& kernel_name = tool.kernel_id_to_kernel_name.at(kernel_id);
        if(kernel_name.find(desired_func_name) == std::string::npos)
            return ROCPROFILER_ATT_CONTROL_NONE;

        return ROCPROFILER_ATT_CONTROL_START_AND_STOP;
    } catch(...)
    {
        std::cerr << "Could not find kernel id: " << kernel_id << std::endl;
    }

    C_API_END
    return ROCPROFILER_ATT_CONTROL_NONE;
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

    ROCPROFILER_CALL(
        rocprofiler_configure_dispatch_thread_trace_service(
            client_ctx, nullptr, 0, dispatch_callback, Callbacks::shader_data_callback, tool_data),
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

    std::unique_lock<std::mutex> isa_lk(tool.isa_map_mut);

    // Find largest instruction
    size_t max_inst_size = 0;
    for(auto& [addr, lines] : tool.isa_map)
        max_inst_size = std::max(max_inst_size, lines->inst.size());

    size_t total_hit    = 0;
    size_t total_cycles = 0;

    for(auto& [addr, line] : tool.isa_map)
    {
        total_hit += line->hitcount.load(std::memory_order_relaxed);
        total_cycles += line->latency.load(std::memory_order_relaxed);
    }

    assert(total_cycles > 0);
    assert(total_hit > 0);

    double wave_started     = (double) tool.waves_started.load();
    double wave_event_ratio = wave_started / (wave_started + (double) tool.waves_ended.load());
    assert(wave_event_ratio > 0.5 - WAVE_RATIO_TOLERANCE);
    assert(wave_event_ratio < 0.5 + WAVE_RATIO_TOLERANCE);

    Callbacks::callbacks_fini();
}

}  // namespace Single
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
    id->name = "ATT_test_single_dispatch";

    // store client info
    ATTTest::Single::client_id = id;

    auto* data = new ATTTest::ToolData{};

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &ATTTest::Single::tool_init,
                                            &ATTTest::Single::tool_fini,
                                            reinterpret_cast<void*>(data)};

    // return pointer to configure data
    return &cfg;
}
