// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "avail/records.hpp"

#include "common/pci_bdf.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::avail
{

struct gpu_agent_info
{
    std::uint64_t handle = 0;
    device_record device;
};

struct devices_listing_result
{
    std::vector<device_record> records;
    std::optional<diagnostic>  issue;
};

struct gpu_agents_listing_result
{
    std::vector<gpu_agent_info> agents;
    std::optional<diagnostic>   issue;
};

[[nodiscard]] inline std::string
c_string_or_empty(const char* text)
{
    if(text == nullptr || text[0] == '\0')
    {
        return {};
    }
    return std::string{ text };
}

namespace inventory
{

template <typename SdkWrapper>
using sdk_agent_t = SdkWrapper::agent_t;
template <typename SdkWrapper>
using sdk_agent_version_t = SdkWrapper::agent_version_t;
template <typename SdkWrapper>
using sdk_status_t = SdkWrapper::status_t;

template <typename SdkWrapper>
[[nodiscard]] gpu_agent_info
gpu_agent_from_sdk(const sdk_agent_t<SdkWrapper>& agent)
{
    const auto logical_index = static_cast<std::size_t>(agent.logical_node_type_id);
    const auto vendor        = c_string_or_empty(agent.vendor_name);
    const auto product       = c_string_or_empty(agent.product_name);
    auto       name          = product;
    if(name.empty())
    {
        name = c_string_or_empty(agent.name);
    }

    gpu_agent_info listed;
    listed.handle = agent.id.handle;
    listed.device = device_record{
        .id            = "gpu-" + std::to_string(logical_index),
        .kind          = device_kind::gpu,
        .name          = std::move(name),
        .logical_index = logical_index,
        .vendor  = vendor.empty() ? std::nullopt : std::optional<std::string>{ vendor },
        .product = product.empty() ? std::nullopt : std::optional<std::string>{ product },
        .pci_bdf =
            common::format_pci_bdf_from_location_id(agent.domain, agent.location_id),
        .numa_node = static_cast<std::size_t>(agent.node_id),
    };
    return listed;
}

template <typename SdkWrapper>
void
append_gpu_agent(std::vector<gpu_agent_info>& listed, const void* raw)
{
    const auto* agent = static_cast<const sdk_agent_t<SdkWrapper>*>(raw);
    if(agent == nullptr || agent->type != SdkWrapper::AGENT_TYPE_GPU)
    {
        return;
    }
    listed.push_back(gpu_agent_from_sdk<SdkWrapper>(*agent));
}

template <typename SdkWrapper>
[[nodiscard]] sdk_status_t<SdkWrapper>
collect_gpu_agents(sdk_agent_version_t<SdkWrapper>, const void** agents,
                   std::size_t agent_count, void* user_data)
{
    auto* listed = static_cast<std::vector<gpu_agent_info>*>(user_data);
    for(std::size_t idx = 0; idx < agent_count; ++idx)
    {
        append_gpu_agent<SdkWrapper>(*listed, agents[idx]);
    }
    return SdkWrapper::STATUS_SUCCESS;
}

template <typename SdkWrapper>
void
set_agent_query_issue(gpu_agents_listing_result& result, sdk_status_t<SdkWrapper> status)
{
    const char* message = SdkWrapper::get_status_string(status);
    result.agents.clear();
    result.issue = diagnostic{
        .capability = capability_kind::gpu_devices,
        .source     = source_id::rocprofiler_sdk,
        .message    = c_string_or_empty(message).empty()
                          ? std::string{ "rocprofiler-sdk agent query failed" }
                          : std::string{ message },
    };
}

template <typename SdkWrapper>
[[nodiscard]] gpu_agents_listing_result
gpu_agents()
{
    gpu_agents_listing_result result;
    const auto                status = SdkWrapper::query_available_agents(
        SdkWrapper::AGENT_INFO_VERSION_0, collect_gpu_agents<SdkWrapper>,
        sizeof(sdk_agent_t<SdkWrapper>), &result.agents);
    if(status != SdkWrapper::STATUS_SUCCESS)
    {
        set_agent_query_issue<SdkWrapper>(result, status);
    }
    return result;
}

}  // namespace inventory

[[nodiscard]] devices_listing_result
query_gpu_devices();

[[nodiscard]] gpu_agents_listing_result
query_gpu_agent_infos();

}  // namespace rocprofsys::avail
