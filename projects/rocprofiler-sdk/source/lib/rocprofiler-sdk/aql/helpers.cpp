// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/aql/helpers.hpp"

#include <fmt/core.h>
#include <glog/logging.h>

namespace rocprofiler
{
namespace aql
{
hsa_ven_amd_aqlprofile_id_query_t
get_query_info(hsa_agent_t agent, const counters::Metric& metric)
{
    DLOG(WARNING) << fmt::format("Querying HSA for Counter: {}", metric);

    hsa_ven_amd_aqlprofile_profile_t  profile{.agent = agent};
    hsa_ven_amd_aqlprofile_id_query_t query = {metric.block().c_str(), 0, 0};
    if(hsa_ven_amd_aqlprofile_get_info(&profile, HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID, &query) !=
       HSA_STATUS_SUCCESS)
    {
        DLOG(FATAL) << fmt::format("AQL failed to query info for counter {}", metric);
        throw std::runtime_error(fmt::format("AQL failed to query info for counter {}", metric));
    }
    return query;
}

uint32_t
get_block_counters(hsa_agent_t agent, const hsa_ven_amd_aqlprofile_event_t& event)
{
    hsa_ven_amd_aqlprofile_profile_t query              = {.agent       = agent,
                                              .type        = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC,
                                              .events      = &event,
                                              .event_count = 1};
    uint32_t                         max_block_counters = 0;
    if(hsa_ven_amd_aqlprofile_get_info(&query,
                                       HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS,
                                       &max_block_counters) != HSA_STATUS_SUCCESS)
    {
        throw std::runtime_error(fmt::format("AQL failed to max block info for counter {}",
                                             static_cast<int64_t>(event.block_name)));
    }
    return max_block_counters;
}
}  // namespace aql
}  // namespace rocprofiler
