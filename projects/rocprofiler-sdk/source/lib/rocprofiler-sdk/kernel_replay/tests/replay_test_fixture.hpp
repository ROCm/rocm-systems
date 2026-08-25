// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

// Shared setup for the GPU-backed kernel-replay snapshot tests: force rocprofiler configuration so
// the HSA allocation interceptors are installed on the live table, bring up HIP, and locate the GPU
// agent that owns the test's allocations.
//
// snap_restore.cpp and snap_bandwidth.cpp predate this header and carry their own copies. New test
// files should use this instead of adding a third.
//
// Note that each TEST() runs as its own ctest process (see gtest_add_tests in CMakeLists.txt), so
// exactly one file's rocprofiler_force_configure runs per process even when several are linked into
// the same executable.

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include <cstdint>

namespace rocprofiler
{
namespace kernel_replay
{
namespace test
{
inline rocprofiler_context_id_t&
context()
{
    static rocprofiler_context_id_t ctx{};
    return ctx;
}

inline void
tracing_noop(rocprofiler_callback_tracing_record_t /*record*/,
             rocprofiler_user_data_t* /*user_data*/,
             void* /*callback_data*/)
{}

inline int
tool_init(rocprofiler_client_finalize_t /*fini*/, void* /*tool_data*/)
{
    if(rocprofiler_create_context(&context()) != ROCPROFILER_STATUS_SUCCESS) return -1;
    // Any callback-tracing service is enough: configuring one installs the HSA table interception
    // that the allocation tracker chains onto. The tracker itself is switched on explicitly by
    // ensure_live_tracking() because these tests drive snap/restore directly rather than through a
    // KERNEL_REPLAY service.
    rocprofiler_configure_callback_tracing_service(context(),
                                                  ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
                                                  nullptr,
                                                  0,
                                                  tracing_noop,
                                                  nullptr);
    rocprofiler_start_context(context());
    return 0;
}

inline void
tool_fini(void* /*tool_data*/)
{}

inline rocprofiler_tool_configure_result_t*
configure(uint32_t /*version*/,
          const char* /*runtime_version*/,
          uint32_t /*priority*/,
          rocprofiler_client_id_t* id)
{
    id->name        = "kernel-replay-test";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}

// True when rocprofiler is configured, a HIP GPU is present, and the allocation tracker is on.
// Tests should GTEST_SKIP() when this is false so the suite still passes on a CPU-only runner.
inline bool
ensure_live_tracking()
{
    static bool ok = [] {
        if(rocprofiler_force_configure(&configure) != ROCPROFILER_STATUS_SUCCESS) return false;
        if(hipInit(0) != hipSuccess) return false;
        int devs = 0;
        if(hipGetDeviceCount(&devs) != hipSuccess || devs == 0) return false;
        return true;
    }();
    if(ok) memory_tracker::set_tracking_enabled(true);
    return ok;
}

// The GPU agent that owns the test's device allocations. Tests run with a single visible device, so
// the first GPU agent is the one hipMalloc allocates on.
inline hsa_agent_t
gpu_agent()
{
    for(const auto* rocp_agent : agent::get_agents())
    {
        if(rocp_agent == nullptr || rocp_agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;
        if(auto hsa = agent::get_hsa_agent(rocp_agent); hsa.has_value()) return *hsa;
    }
    return hsa_agent_t{.handle = 0};
}

inline void
sync_ok()
{
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
}
}  // namespace test
}  // namespace kernel_replay
}  // namespace rocprofiler
