// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "avail/devices.hpp"
#include "avail/records.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::avail
{

struct counters_listing_result
{
    std::vector<counter_record> records;
    std::optional<diagnostic>   issue;
};

namespace inventory
{

template <typename SdkWrapper>
using sdk_agent_id_t = SdkWrapper::agent_id;
template <typename SdkWrapper>
using sdk_counter_id_t = SdkWrapper::counter_id;
template <typename SdkWrapper>
using sdk_dimension_info_t = SdkWrapper::dimension_info_t;
template <typename SdkWrapper>
using sdk_counter_info_v0_t = SdkWrapper::counter_info_v0_t;

template <typename SdkWrapper>
[[nodiscard]] sdk_status_t<SdkWrapper>
collect_supported_counter_ids(sdk_agent_id_t<SdkWrapper>,
                              sdk_counter_id_t<SdkWrapper>* counters,
                              std::size_t counter_count, void* user_data)
{
    auto* listed = static_cast<std::vector<sdk_counter_id_t<SdkWrapper>>*>(user_data);
    listed->insert(listed->end(), counters, counters + counter_count);
    return SdkWrapper::STATUS_SUCCESS;
}

template <typename SdkWrapper>
[[nodiscard]] std::vector<sdk_counter_id_t<SdkWrapper>>
supported_counter_ids(sdk_agent_id_t<SdkWrapper> agent)
{
    std::vector<sdk_counter_id_t<SdkWrapper>> ids;
    const auto status = SdkWrapper::iterate_agent_supported_counters(
        agent, collect_supported_counter_ids<SdkWrapper>, &ids);
    if(status != SdkWrapper::STATUS_SUCCESS)
    {
        return {};
    }
    return ids;
}

template <typename SdkWrapper>
[[nodiscard]] sdk_status_t<SdkWrapper>
collect_counter_dimensions(sdk_counter_id_t<SdkWrapper>,
                           const sdk_dimension_info_t<SdkWrapper>* info,
                           std::size_t dim_count, void* user_data)
{
    auto* listed = static_cast<std::vector<counter_dimension>*>(user_data);
    for(std::size_t idx = 0; idx < dim_count; ++idx)
    {
        listed->push_back(counter_dimension{
            .name   = c_string_or_empty(info[idx].name),
            .extent = info[idx].instance_size,
        });
    }
    return SdkWrapper::STATUS_SUCCESS;
}

template <typename SdkWrapper>
[[nodiscard]] std::vector<counter_dimension>
counter_dimensions(sdk_counter_id_t<SdkWrapper> counter)
{
    std::vector<counter_dimension> dimensions;
    (void) SdkWrapper::iterate_counter_dimensions(
        counter, collect_counter_dimensions<SdkWrapper>, &dimensions);
    return dimensions;
}

template <typename SdkWrapper>
[[nodiscard]] std::optional<counter_record>
counter_from_id(sdk_counter_id_t<SdkWrapper> counter, const gpu_agent_info& agent)
{
    sdk_counter_info_v0_t<SdkWrapper> info{};
    const auto                        status = SdkWrapper::query_counter_info(
        counter, SdkWrapper::COUNTER_INFO_VERSION_0, &info);
    if(status != SdkWrapper::STATUS_SUCCESS || info.name == nullptr || info.is_constant)
    {
        return std::nullopt;
    }

    const auto block      = c_string_or_empty(info.block);
    const auto expression = c_string_or_empty(info.expression);

    return counter_record{
        .name        = std::string{ info.name },
        .description = c_string_or_empty(info.description),
        .source      = source_id::rocprofiler_sdk,
        .status      = availability::available,
        .id          = counter.handle,
        .device_id   = agent.device.id,
        .block       = block.empty() ? std::nullopt : std::optional<std::string>{ block },
        .expression =
            expression.empty() ? std::nullopt : std::optional<std::string>{ expression },
        .aliases    = {},
        .dimensions = counter_dimensions<SdkWrapper>(counter),
    };
}

template <typename SdkWrapper>
void
append_agent_counters(std::vector<counter_record>& records, const gpu_agent_info& agent)
{
    const auto agent_id = sdk_agent_id_t<SdkWrapper>{ agent.handle };
    for(const auto& counter : supported_counter_ids<SdkWrapper>(agent_id))
    {
        auto record = counter_from_id<SdkWrapper>(counter, agent);
        if(record.has_value())
        {
            records.push_back(std::move(*record));
        }
    }
}

template <typename SdkWrapper>
[[nodiscard]] std::vector<counter_record>
gpu_counters(const std::vector<gpu_agent_info>& agents)
{
    std::vector<counter_record> records;
    for(const auto& agent : agents)
    {
        append_agent_counters<SdkWrapper>(records, agent);
    }
    return records;
}

}  // namespace inventory

[[nodiscard]] counters_listing_result
query_gpu_counters();

}  // namespace rocprofsys::avail
