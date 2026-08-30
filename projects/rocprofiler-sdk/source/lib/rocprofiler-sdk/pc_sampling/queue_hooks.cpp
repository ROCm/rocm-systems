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

#include "lib/rocprofiler-sdk/pc_sampling/queue_hooks.hpp"

#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
#    include "lib/rocprofiler-sdk/pc_sampling/hsa_adapter.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#endif

namespace rocprofiler
{
namespace pc_sampling
{
bool
is_configured_on_agent(rocprofiler_agent_id_t agent_id)
{
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    return is_pc_sample_service_configured(agent_id);
#else
    (void) agent_id;
    return false;
#endif
}

void
signal_completion_hook(const ::rocprofiler::hsa::Queue&                           queue,
                       const ::rocprofiler::hsa::rocprofiler_packet&              kernel_packet,
                       std::shared_ptr<::rocprofiler::hsa::queue_info_session_t>& session,
                       ::rocprofiler::hsa::packet_data_t& /*packet*/,
                       ::rocprofiler::hsa::inst_pkt_t& /*inst_pkt*/,
                       kernel_dispatch::profiling_time /*dispatch_time*/)
{
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    if(!session) return;
    // kernel_completion_cb takes a mutable reference to the kernel packet even
    // though it does not modify it; copy to bind a non-const lvalue.
    auto kern_pkt = kernel_packet;
    hsa::kernel_completion_cb(queue.get_agent().get_rocp_agent(), kern_pkt, *session);
#else
    (void) queue;
    (void) kernel_packet;
    (void) session;
#endif
}
}  // namespace pc_sampling
}  // namespace rocprofiler
