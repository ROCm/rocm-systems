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

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

namespace rocprofiler
{
namespace counters
{
using ClientID   = int64_t;
using inst_pkt_t = common::container::
    small_vector<std::pair<std::unique_ptr<rocprofiler::hsa::AQLPacket>, ClientID>, 4>;

hsa::Queue::pkt_and_serialize_t
queue_cb(const context::context*                                         ctx,
         const std::shared_ptr<counter_callback_info>&                   info,
         const hsa::Queue&                                               queue,
         const hsa::rocprofiler_packet&                                  pkt,
         rocprofiler_kernel_id_t                                         kernel_id,
         rocprofiler_dispatch_id_t                                       dispatch_id,
         rocprofiler_user_data_t*                                        user_data,
         const hsa::Queue::queue_info_session_t::external_corr_id_map_t& extern_corr_ids,
         const context::correlation_id*                                  correlation_id);

void
completed_cb(const context::context*                            ctx,
             const std::shared_ptr<counter_callback_info>&      info,
             std::shared_ptr<hsa::Queue::queue_info_session_t>& session,
             inst_pkt_t&                                        pkts,
             kernel_dispatch::profiling_time                    dispatch_time);

}  // namespace counters
}  // namespace rocprofiler
