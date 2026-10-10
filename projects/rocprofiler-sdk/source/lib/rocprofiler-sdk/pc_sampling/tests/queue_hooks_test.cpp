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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/pc_sampling/queue_hooks.hpp"

#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
#    include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#endif

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

TEST(pc_sampling_queue_hooks, exit_hook_null_session_is_noop)
{
    rocprofiler::hsa::rocprofiler_packet                    kern_pkt{};
    std::shared_ptr<rocprofiler::hsa::queue_info_session_t> null_session;
    rocprofiler::hsa::packet_data_t                         packet{};
    rocprofiler::hsa::inst_pkt_t                            inst_pkt{};

    rocprofiler::pc_sampling::kernel_dispatch_phase_exit_hook(
        nullptr,
        kern_pkt,
        null_session,
        packet,
        inst_pkt,
        rocprofiler::kernel_dispatch::profiling_time{});

    SUCCEED();
}

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
// The unconfigured case above can only show the gate stays shut. Seed the two pieces of
// state is_pc_sample_service_configured reads -- the HSA-init flag and the global session
// map -- and the gate must open. Configuring a real service would take a GPU, but the map
// lookup is a find(), so the session is never dereferenced and a null entry is enough.
TEST(pc_sampling_queue_hooks, is_configured_on_agent_configured)
{
    rocprofiler_agent_id_t agent_id;
    agent_id.handle = 424242;

    auto&      hsa_ready  = rocprofiler::pc_sampling::is_hsa_initialized();
    const bool prev_ready = hsa_ready.exchange(true);

    rocprofiler::pc_sampling::get_global_pc_sampling_sessions().wlock(
        [agent_id](auto& sessions) { sessions[agent_id] = nullptr; });

    EXPECT_TRUE(rocprofiler::pc_sampling::is_configured_on_agent(agent_id));

    // Drop the session but leave the HSA-init flag set, so this shows the session lookup is
    // what closed the gate again rather than the flag doing all the work.
    rocprofiler::pc_sampling::get_global_pc_sampling_sessions().wlock(
        [agent_id](auto& sessions) { sessions.erase(agent_id); });

    EXPECT_FALSE(rocprofiler::pc_sampling::is_configured_on_agent(agent_id));

    // pcs-test runs every PC sampling suite in one process.
    hsa_ready.store(prev_ready);
}
#endif
}  // namespace
