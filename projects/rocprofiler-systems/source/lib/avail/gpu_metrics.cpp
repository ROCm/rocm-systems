// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/gpu_metrics.hpp"
#include "avail/devices.hpp"
#include "avail/records.hpp"
#include "avail/smi_session.hpp"

#include <exception>
#include <string>
#include <utility>

namespace rocprofsys::avail
{

metrics_listing_result
query_gpu_metrics()
{
    auto agents = query_gpu_agent_infos(false);
    if(agents.issue)
    {
        agents.issue->capability = capability_kind::gpu_metrics;
        return metrics_listing_result{ .issue = std::move(agents.issue) };
    }
    if(agents.agents.empty())
    {
        return {};
    }

    try
    {
        const smi_session session;
        return inventory::gpu_metrics(agents.agents, session.backend());
    } catch(const std::exception& err)
    {
        return metrics_listing_result{
            .issue =
                diagnostic{
                    .capability = capability_kind::gpu_metrics,
                    .source     = source_id::amd_smi,
                    .message    = std::string{ err.what() },
                },
        };
    }
}

}  // namespace rocprofsys::avail
