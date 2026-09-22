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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/spm/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"

#include <gtest/gtest.h>

namespace
{
// is_any_active replaces the queue.get_notifiers() signal that the old per-queue
// callback registration used to provide to the HSA write interceptor gate. With
// no SPM context active, it must report inactive. This requires no GPU / HSA
// runtime.
TEST(spm_queue_hooks, is_any_active_false_when_no_context_active)
{
    EXPECT_FALSE(rocprofiler::spm::is_any_active());
}

// The write interceptor gates on the agent-scoped variant so that an SPM context restricted to
// one GPU does not pull the other GPUs' queues off the fast path. With no context active it must
// report inactive for any agent, including ones that were never registered.
TEST(spm_queue_hooks, is_active_on_agent_false_when_no_context_active)
{
    EXPECT_FALSE(rocprofiler::spm::is_active_on_agent(rocprofiler_agent_id_t{.handle = 0}));
    EXPECT_FALSE(rocprofiler::spm::is_active_on_agent(rocprofiler_agent_id_t{.handle = 1}));
}

TEST(spm_queue_hooks, exit_hook_skips_when_inst_pkt_has_no_spm_client_id)
{
    rocprofiler::hsa::inst_pkt_t inst_pkt;
    inst_pkt.emplace_back(std::make_pair(std::make_unique<rocprofiler::hsa::EmptyAQLPacket>(),
                                         rocprofiler::hsa::queue_hooks::COUNTERS_CLIENT_ID));

    // The hook returns before it dereferences the session, so a null one keeps this test free of
    // any HSA runtime: queue_info_session_t holds a Queue& and cannot be default constructed.
    auto sess   = std::shared_ptr<rocprofiler::hsa::queue_info_session_t>{};
    auto packet = rocprofiler::hsa::packet_data_t{};
    auto fq_pkt = rocprofiler::hsa::rocprofiler_packet{};

    rocprofiler::spm::signal_completion_hook(
        nullptr, fq_pkt, sess, packet, inst_pkt, rocprofiler::kernel_dispatch::profiling_time{});
    SUCCEED();
}

// write_hook with no active SPM context must be a no-op: it must not append to inst_pkt
// and must not change is_serialized. queue is nullptr because the hook returns before
// dereferencing it when no dispatch_spm context is active (same no-HSA pattern as the
// exit-hook test).
TEST(spm_queue_hooks, write_hook_noop_when_no_context_active)
{
    EXPECT_FALSE(rocprofiler::spm::is_any_active());

    rocprofiler::hsa::inst_pkt_t inst_pkt;
    inst_pkt.emplace_back(std::make_pair(std::make_unique<rocprofiler::hsa::EmptyAQLPacket>(),
                                         rocprofiler::hsa::queue_hooks::COUNTERS_CLIENT_ID));
    const auto size_before = inst_pkt.size();

    auto                    fq_pkt    = rocprofiler::hsa::rocprofiler_packet{};
    rocprofiler_user_data_t user_data = {};
    const auto ext_corr_ids  = rocprofiler::hsa::queue_info_session_t::external_corr_id_map_t{};
    bool       is_serialized = true;

    rocprofiler::spm::write_hook(
        nullptr, fq_pkt, 0, 0, &user_data, ext_corr_ids, nullptr, inst_pkt, is_serialized);

    EXPECT_EQ(inst_pkt.size(), size_before);
    EXPECT_TRUE(is_serialized);
}
}  // namespace
