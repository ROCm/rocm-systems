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

#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <fmt/core.h>

namespace rocprofiler
{
namespace aql
{
hsa_ven_amd_aqlprofile_id_query_t
get_query_info(rocprofiler_agent_id_t agent, const counters::Metric& metric)
{
    auto                     aql_agent = *CHECK_NOTNULL(rocprofiler::agent::get_aql_agent(agent));
    aqlprofile_pmc_profile_t profile   = {.agent = aql_agent, .events = nullptr, .event_count = 0};
    hsa_ven_amd_aqlprofile_id_query_t query = {metric.block().c_str(), 0, 0};
    if(aqlprofile_get_pmc_info(&profile, AQLPROFILE_INFO_BLOCK_ID, &query) != HSA_STATUS_SUCCESS)
    {
        ROCP_DFATAL << fmt::format("AQL failed to query info for counter {}", metric);
        throw std::runtime_error(fmt::format("AQL failed to query info for counter {}", metric));
    }
    return query;
}

uint32_t
get_block_counters(rocprofiler_agent_id_t agent, const aqlprofile_pmc_event_t& event)
{
    auto                     aql_agent = *CHECK_NOTNULL(rocprofiler::agent::get_aql_agent(agent));
    aqlprofile_pmc_profile_t query     = {.agent = aql_agent, .events = &event, .event_count = 1};
    uint32_t                 max_block_counters = 0;
    if(aqlprofile_get_pmc_info(&query, AQLPROFILE_INFO_BLOCK_COUNTERS, &max_block_counters) !=
       HSA_STATUS_SUCCESS)
    {
        throw std::runtime_error(fmt::format("AQL failed to max block info for counter {}",
                                             static_cast<int64_t>(event.block_name)));
    }
    return max_block_counters;
}

rocprofiler_status_t
set_dim_id_from_sample(rocprofiler_counter_instance_id_t& id,
                       aqlprofile_agent_handle_t          agent,
                       aqlprofile_pmc_event_t             event,
                       size_t                             sample_id)
{
    auto callback =
        [](int, int sid, int, int coordinate, const char*, void* userdata) -> hsa_status_t {
        if(const auto* rocprof_id =
               rocprofiler::common::get_val(counters::aqlprofile_id_to_rocprof_instance(), sid))
        {
            counters::set_dim_in_rec(*static_cast<rocprofiler_counter_instance_id_t*>(userdata),
                                     *rocprof_id,
                                     static_cast<size_t>(coordinate));
        }
        return HSA_STATUS_SUCCESS;
    };

    if(aqlprofile_iterate_event_coord(agent, event, sample_id, callback, static_cast<void*>(&id)) !=
       HSA_STATUS_SUCCESS)
    {
        return ROCPROFILER_STATUS_ERROR_AQL_NO_EVENT_COORD;
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
get_dim_info(rocprofiler_agent_id_t   agent,
             aqlprofile_pmc_event_t   event,
             uint32_t                 sample_id,
             std::map<int, uint64_t>& dims)
{
    auto callback = [](int, int id, int extent, int, const char*, void* userdata) -> hsa_status_t {
        auto& map = *static_cast<std::map<int, uint64_t>*>(userdata);
        map.emplace(id, extent);
        return HSA_STATUS_SUCCESS;
    };

    auto aql_agent = *CHECK_NOTNULL(rocprofiler::agent::get_aql_agent(agent));

    if(aqlprofile_iterate_event_coord(
           aql_agent, event, sample_id, callback, static_cast<void*>(&dims)) != HSA_STATUS_SUCCESS)
    {
        return ROCPROFILER_STATUS_ERROR_AQL_NO_EVENT_COORD;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
set_profiler_active_on_queue(hsa_amd_memory_pool_t             pool,
                             hsa_agent_t                       hsa_agent,
                             const rocprofiler_profile_pkt_cb& packet_submit)
{
    CHECK(hsa::get_amd_ext_table() != nullptr);
    CHECK(hsa::get_amd_ext_table()->hsa_amd_memory_pool_allocate_fn != nullptr);
    CHECK(hsa::get_amd_ext_table()->hsa_amd_agents_allow_access_fn != nullptr);
    CHECK(hsa::get_amd_ext_table()->hsa_amd_memory_pool_free_fn != nullptr);

    // Inject packet to enable profiling of other process queues on this queue
    hsa_ven_amd_aqlprofile_profile_t profile{};
    profile.agent = hsa_agent;

    // Query for cmd buffer size
    hsa_ven_amd_aqlprofile_info_type_t info_type =
        (hsa_ven_amd_aqlprofile_info_type_t)((int) HSA_VEN_AMD_AQLPROFILE_INFO_ENABLE_CMD);
    if(hsa_ven_amd_aqlprofile_get_info(&profile, info_type, nullptr) != HSA_STATUS_SUCCESS)
    {
        return ROCPROFILER_STATUS_ERROR;
    }
    // Allocate cmd buffer
    const size_t mask = 0x1000 - 1;
    auto         size = (profile.command_buffer.size + mask) & ~mask;

    if(hsa::get_amd_ext_table()->hsa_amd_memory_pool_allocate_fn(
           pool, size, hsa::hsa_amd_memory_pool_executable_flag, &profile.command_buffer.ptr) !=
       HSA_STATUS_SUCCESS)
    {
        ROCP_WARNING << "Failed to allocate memory to enable profile command on agent, some "
                        "counters will be unavailable";
        return ROCPROFILER_STATUS_ERROR;
    }
    if(hsa::get_amd_ext_table()->hsa_amd_agents_allow_access_fn(
           1, &hsa_agent, nullptr, profile.command_buffer.ptr) != HSA_STATUS_SUCCESS)
    {
        ROCP_WARNING << "Agent cannot access memory, some counters will be unavailable";
        return ROCPROFILER_STATUS_ERROR;
    }

    hsa::rocprofiler_packet packet{};
    if(hsa_ven_amd_aqlprofile_get_info(&profile, info_type, &packet.ext_amd_aql_pm4) !=
       HSA_STATUS_SUCCESS)
    {
        ROCP_WARNING << "Failed to generate command packet, some counters will be unavailable";
        return ROCPROFILER_STATUS_ERROR;
    }

    packet_submit(packet);
    hsa::get_amd_ext_table()->hsa_amd_memory_pool_free_fn(profile.command_buffer.ptr);
    return ROCPROFILER_STATUS_SUCCESS;
}

}  // namespace aql
}  // namespace rocprofiler
