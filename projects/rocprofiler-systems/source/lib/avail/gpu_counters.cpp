// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/gpu_counters.hpp"
#include "avail/devices.hpp"
#include "avail/records.hpp"

#include <exception>
#include <optional>
#include <string>
#include <utility>

#if defined(ROCPROFSYS_AVAIL_HAS_SDK)
#    include "backends/rocprofiler_sdk/wrapper.hpp"
#endif

namespace rocprofsys::avail
{
namespace
{

diagnostic
make_diagnostic(capability_kind capability, source_id source, std::string message)
{
    return diagnostic{
        .capability = capability,
        .source     = source,
        .message    = std::move(message),
    };
}

[[nodiscard]] std::optional<diagnostic>
sdk_agent_query_failure(const gpu_agents_listing_result& agents)
{
    if(!agents.issue.has_value() || !agents.agents.empty())
    {
        return std::nullopt;
    }
    if(agents.issue->source != source_id::rocprofiler_sdk)
    {
        return std::nullopt;
    }
    return diagnostic{
        .capability = capability_kind::gpu_counters,
        .source     = source_id::rocprofiler_sdk,
        .message    = agents.issue->message,
    };
}

}  // namespace

counters_listing_result
query_gpu_counters()
{
#if defined(ROCPROFSYS_AVAIL_HAS_SDK)
    try
    {
        auto                    agents = query_gpu_agent_infos();
        counters_listing_result result;
        if(auto issue = sdk_agent_query_failure(agents); issue.has_value())
        {
            result.issue = std::move(issue);
            return result;
        }
        result.records = inventory::gpu_counters<rocprofiler_sdk::wrapper>(agents.agents);
        return result;
    } catch(const std::exception& err)
    {
        return counters_listing_result{
            .records = {},
            .issue   = make_diagnostic(capability_kind::gpu_counters,
                                       source_id::rocprofiler_sdk, err.what()),
        };
    }
#else
    return counters_listing_result{
        .records = {},
        .issue =
            make_diagnostic(capability_kind::gpu_counters, source_id::rocprofiler_sdk,
                            std::string{ k_sdk_unavailable_message }),
    };
#endif
}

}  // namespace rocprofsys::avail
