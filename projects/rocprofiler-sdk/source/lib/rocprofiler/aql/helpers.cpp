#include "lib/rocprofiler/aql/helpers.hpp"

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
