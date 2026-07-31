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

#pragma once

#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstdint>

namespace rocprofiler
{
namespace thread_trace
{
namespace test
{
/// Emitted by a suite that cannot run on the device under test. The thread trace
/// unit tests are registered with a matching SKIP_REGULAR_EXPRESSION so ctest
/// reports the test as skipped rather than as a silent pass.
constexpr auto SKIP_MESSAGE = "Thread trace unavailable";

/// Whether the agent implements the SQTT multi-buffer (ring) path. aqlprofile
/// only programs the additional buffer registers on gfx94x, gfx95x and gfx12xx;
/// everywhere else aqlprofile_att_create_packets() rejects the request with
/// "Not supported".
///
/// This mirrors rocprofiler_sdk_sqtt_triple_buffer_disabled() in
/// cmake/Modules/rocprofiler-sdk-utilities.cmake, but is answered by the GPU
/// running the test. Unit tests are installed and replayed on runners other than
/// the machine that configured the build, where a configure-time answer either
/// describes the wrong device or, on a GPU-less builder, disables the suite
/// everywhere.
inline bool
supports_multi_buffer_sqtt(const rocprofiler_agent_t* agent)
{
    if(agent == nullptr) return false;

    // gfx_target_version packs the target as major * 10000 + minor * 100 + step
    const uint32_t major = (agent->gfx_target_version / 10000) % 100;
    const uint32_t minor = (agent->gfx_target_version / 100) % 100;

    return (major == 9 && (minor == 4 || minor == 5)) || major == 12;
}

/// First HSA-visible GPU agent that can run the multi-buffer suite, or nullptr.
inline const hsa::AgentCache*
find_multi_buffer_agent()
{
    for(const auto& [_, agent] : hsa::get_queue_controller()->get_supported_agents())
    {
        const auto* rocp_agent = agent.get_rocp_agent();

        if(rocp_agent == nullptr) continue;
        if(rocp_agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;
        if(rocp_agent->runtime_visibility.hsa == 0) continue;
        if(!supports_multi_buffer_sqtt(rocp_agent)) continue;

        return &agent;
    }
    return nullptr;
}
}  // namespace test
}  // namespace thread_trace
}  // namespace rocprofiler
