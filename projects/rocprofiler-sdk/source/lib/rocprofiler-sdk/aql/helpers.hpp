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

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <functional>
#include <map>
#include <string>

namespace rocprofiler
{
namespace aql
{
using rocprofiler_profile_pkt_cb = std::function<void(hsa::rocprofiler_packet)>;
// Query HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID from aqlprofile
hsa_ven_amd_aqlprofile_id_query_t
get_query_info(rocprofiler_agent_id_t agent, const counters::Metric& metric);

// Query HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS from aqlprofiler
uint32_t
get_block_counters(rocprofiler_agent_id_t agent, const aqlprofile_pmc_event_t& event);

// Query dimimension ids for counter event. Returns AQLProfiler ID -> extent
rocprofiler_status_t
get_dim_info(rocprofiler_agent_id_t   agent,
             aqlprofile_pmc_event_t   event,
             uint32_t                 sample_id,
             std::map<int, uint64_t>& dims);

// Set dimension ids into id for sample
rocprofiler_status_t
set_dim_id_from_sample(rocprofiler_counter_instance_id_t& id,
                       aqlprofile_agent_handle_t          agent,
                       aqlprofile_pmc_event_t             event,
                       size_t                             sample_id);

rocprofiler_status_t
set_profiler_active_on_queue(hsa_amd_memory_pool_t             pool,
                             hsa_agent_t                       hsa_agent,
                             const rocprofiler_profile_pkt_cb& packet_submit);
}  // namespace aql
}  // namespace rocprofiler
