
#pragma once

#include <functional>

#include <hsa/hsa_ven_amd_aqlprofile.h>

#include "lib/rocprofiler/counters/metrics.hpp"

namespace rocprofiler
{
namespace aql
{
// Query HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID from aqlprofile
hsa_ven_amd_aqlprofile_id_query_t
get_query_info(hsa_agent_t agent, const counters::Metric& metric);

// Query HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS from aqlprofiler
uint32_t
get_block_counters(hsa_agent_t agent, const hsa_ven_amd_aqlprofile_event_t& event);
}  // namespace aql
}  // namespace rocprofiler
