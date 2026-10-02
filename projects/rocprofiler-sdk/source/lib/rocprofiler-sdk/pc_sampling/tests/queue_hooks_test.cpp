// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/pc_sampling/queue_hooks.hpp"

#include <gtest/gtest.h>

namespace
{
// is_configured_on_agent replaces the queue.get_notifiers() signal that the old
// per-queue callback registration used to provide to the HSA write interceptor
// gate. With no PC sampling service configured, any agent id must report as not
// configured. This is always linkable regardless of HSA PC sampling support and
// requires no GPU / HSA runtime.
TEST(pc_sampling_queue_hooks, is_configured_on_agent_unconfigured)
{
    rocprofiler_agent_id_t agent_id;
    agent_id.handle = 123456;
    EXPECT_FALSE(rocprofiler::pc_sampling::is_configured_on_agent(agent_id));
}

TEST(pc_sampling_queue_hooks, signal_completion_hook_null_session_is_noop)
{
    rocprofiler::hsa::rocprofiler_packet kern_pkt{};
    std::shared_ptr<rocprofiler::hsa::queue_info_session_t> null_session;
    rocprofiler::hsa::packet_data_t                         packet{};
    rocprofiler::hsa::inst_pkt_t                            inst_pkt{};

    rocprofiler::pc_sampling::signal_completion_hook(
        *reinterpret_cast<rocprofiler::hsa::Queue*>(nullptr),
        kern_pkt,
        null_session,
        packet,
        inst_pkt,
        rocprofiler::kernel_dispatch::profiling_time{});

    SUCCEED();
}
}  // namespace
