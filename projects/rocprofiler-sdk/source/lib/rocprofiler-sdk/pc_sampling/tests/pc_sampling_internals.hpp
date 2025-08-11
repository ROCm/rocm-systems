// MIT License
//
// Copyright (c) 2023-2025 ROCm Developer Tools
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

#include "lib/rocprofiler-sdk/pc_sampling/hsa_adapter.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"

namespace rocprofiler
{
namespace pc_sampling
{
void
post_hsa_init_start_active_service();

namespace hsa
{
extern void
amd_intercept_marker_handler_callback(const struct amd_aql_intercept_marker_s* packet,
                                      hsa_queue_t*                             queue,
                                      uint64_t                                 packet_id);

extern void
kernel_completion_cb(const std::shared_ptr<rocprofiler::counters::counter_callback_info>& info,
                     const rocprofiler_agent_t*                           rocp_agent,
                     rocprofiler::hsa::ClientID                           client_id,
                     const rocprofiler::hsa::rocprofiler_packet&          kernel_pkt,
                     const rocprofiler::hsa::Queue::queue_info_session_t& session,
                     std::unique_ptr<rocprofiler::hsa::AQLPacket>         pkt);

extern void
data_ready_callback(void*                                client_callback_data,
                    size_t                               data_size,
                    size_t                               lost_sample_count,
                    hsa_ven_amd_pcs_data_copy_callback_t data_copy_callback,
                    void*                                hsa_callback_data);

extern atomic_pc_sampling_service_t&
get_active_pc_sampling_service();

}  // namespace hsa
}  // namespace pc_sampling
}  // namespace rocprofiler
