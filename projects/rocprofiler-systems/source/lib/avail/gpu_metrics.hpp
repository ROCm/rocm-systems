// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "avail/devices.hpp"
#include "avail/records.hpp"

#include "backends/amd_smi/device.hpp"
#include "backends/amd_smi/metric_tokens.hpp"

#include <algorithm>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::avail
{

struct metrics_listing_result
{
    std::vector<metric_record> records;
    std::optional<diagnostic>  issue;
};

namespace inventory
{

inline void
set_gpu_metrics_issue(metrics_listing_result& result, std::string message)
{
    if(!result.issue)
    {
        result.issue = diagnostic{
            .capability = capability_kind::gpu_metrics,
            .source     = source_id::amd_smi,
            .message    = std::move(message),
        };
    }
}

inline void
append_gpu_metric_records(
    std::vector<metric_record>& records, std::string_view device_id,
    const std::optional<backends::amd_smi::gpu::metric_support>& support)
{
    records.reserve(records.size() + backends::amd_smi::gpu::k_metric_tokens.size());
    for(const auto& token : backends::amd_smi::gpu::k_metric_tokens)
    {
        auto status = availability::unknown;
        if(support)
        {
            status = (*support)[token.group] ? availability::available
                                             : availability::unavailable;
        }
        records.emplace_back(metric_record{
            .name        = std::string{ token.name },
            .description = std::string{ token.description },
            .unit        = std::string{ token.unit },
            .source      = source_id::amd_smi,
            .status      = status,
            .device_id   = std::string{ device_id },
        });
    }
}

[[nodiscard]] inline const gpu_agent_info*
find_gpu_agent(const std::vector<gpu_agent_info>& agents, std::string_view pci_bdf)
{
    const auto itr = std::ranges::find_if(agents, [pci_bdf](const gpu_agent_info& agent) {
        return agent.device.pci_bdf && *agent.device.pci_bdf == pci_bdf;
    });
    return (itr == agents.end()) ? nullptr : &*itr;
}

/**
 * Lists CLI GPU metric tokens for each AMD SMI GPU associated with an SDK agent.
 *
 * @param agents SDK GPU agents that provide stable catalog device identifiers.
 * @param backend Initialized AMD SMI backend session.
 * @return Metric token records and the first AMD SMI diagnostic, if any.
 */
template <typename Backend>
[[nodiscard]] metrics_listing_result
gpu_metrics(const std::vector<gpu_agent_info>& agents, std::shared_ptr<Backend> backend)
{
    metrics_listing_result result;
    for(const auto handle : backend->enumerate_gpu_handles())
    {
        const backends::amd_smi::device<Backend> gpu{ backend, handle };
        const gpu_agent_info*                    agent = nullptr;
        try
        {
            agent = find_gpu_agent(agents, gpu.get_bdf());
        } catch(const std::exception& err)
        {
            set_gpu_metrics_issue(result, err.what());
            continue;
        }
        if(agent == nullptr)
        {
            set_gpu_metrics_issue(result,
                                  "AMD SMI GPU did not match a rocprofiler-sdk agent");
            continue;
        }

        try
        {
            append_gpu_metric_records(result.records, agent->device.id,
                                      gpu.get_metric_support());
        } catch(const std::exception& err)
        {
            append_gpu_metric_records(result.records, agent->device.id, std::nullopt);
            set_gpu_metrics_issue(result, err.what());
        }
    }
    return result;
}

}  // namespace inventory

[[nodiscard]] metrics_listing_result
query_gpu_metrics();

}  // namespace rocprofsys::avail
