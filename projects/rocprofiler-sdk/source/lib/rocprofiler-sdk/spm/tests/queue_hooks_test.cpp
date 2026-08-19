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

#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"
#include "lib/rocprofiler-sdk/spm/queue_hooks.hpp"

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

TEST(spm_queue_hooks, exit_hook_skips_when_inst_pkt_has_no_spm_client_id)
{
    rocprofiler::hsa::inst_pkt_t inst_pkt;
    inst_pkt.emplace_back(
        std::make_pair(std::make_unique<rocprofiler::hsa::AQLPacket>(),
                       rocprofiler::hsa::queue_hooks::COUNTERS_CLIENT_ID));

    auto sess    = std::make_shared<rocprofiler::hsa::queue_info_session_t>();
    auto packet  = rocprofiler::hsa::packet_data_t{};
    auto fq_pkt  = rocprofiler::hsa::rocprofiler_packet{};

    rocprofiler::spm::signal_completion_hook(
        *reinterpret_cast<rocprofiler::hsa::Queue*>(nullptr),
        fq_pkt,
        sess,
        packet,
        inst_pkt,
        rocprofiler::kernel_dispatch::profiling_time{});
    SUCCEED();
}
}  // namespace
