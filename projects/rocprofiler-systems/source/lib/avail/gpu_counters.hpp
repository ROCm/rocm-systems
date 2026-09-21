// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "avail/records.hpp"

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <exception>
#include <iterator>
#include <vector>

namespace rocprofsys::avail
{
/// Supplies the counters one GPU agent supports.
///
/// Split into listing and describing because rocprofiler-sdk reports counter
/// ids per agent but counter metadata per counter.
template <typename T>
concept counter_inventory_source =
    requires(T& source, std::uint64_t agent_handle, std::uint64_t counter_id) {
        { source.counter_ids(agent_handle) } -> std::same_as<std::vector<std::uint64_t>>;
        { source.describe(counter_id) } -> std::same_as<std::vector<counter_record>>;
    };

/// Enumerates counters for each device, keeping devices independent: an agent
/// whose counters cannot be listed yields a diagnostic and an empty group
/// rather than discarding the counters of the other devices.
///
/// Counters the source cannot describe are skipped; a partial catalog is more
/// useful than none. Each group is sorted by counter name for stable output.
template <counter_inventory_source Source>
[[nodiscard]] counter_query_result
collect_gpu_counters(Source& source, const std::vector<device_record>& devices)
{
    auto result = counter_query_result{};
    result.devices.reserve(devices.size());

    for(const auto& device : devices)
    {
        auto group         = device_counters{};
        group.agent_handle = device.agent_handle;
        group.index        = device.index;
        group.device_name  = device.name;

        auto ids = std::vector<std::uint64_t>{};
        try
        {
            ids = source.counter_ids(device.agent_handle);
        } catch(const std::exception& e)
        {
            result.diagnostics.push_back({ source_id::rocprofiler_sdk, e.what() });
            result.devices.push_back(std::move(group));
            continue;
        }

        for(auto id : ids)
        {
            auto described = std::vector<counter_record>{};
            try
            {
                described = source.describe(id);
            } catch(const std::exception& e)
            {
                result.diagnostics.push_back({ source_id::rocprofiler_sdk, e.what() });
                continue;
            }
            group.counters.insert(group.counters.end(),
                                  std::make_move_iterator(described.begin()),
                                  std::make_move_iterator(described.end()));
        }

        std::sort(group.counters.begin(), group.counters.end(),
                  [](const counter_record& lhs, const counter_record& rhs) {
                      if(lhs.name != rhs.name) return lhs.name < rhs.name;
                      return lhs.id < rhs.id;
                  });

        result.devices.push_back(std::move(group));
    }

    return result;
}

/// Counter catalog from the live rocprofiler-sdk backend.
[[nodiscard]] counter_query_result
query_gpu_counters(const std::vector<device_record>& devices);
}  // namespace rocprofsys::avail
