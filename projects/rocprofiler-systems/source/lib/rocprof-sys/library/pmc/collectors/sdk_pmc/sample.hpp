// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/sample_type.hpp"

#include <cstdint>
#include <vector>

namespace rocprofsys::pmc::collectors::sdk_pmc
{

/**
 * @brief SDK PMC sample type for trace cache serialization.
 *
 * Stores a snapshot of GPU hardware counter values collected via
 * rocprofiler_sample_device_counting_service(). Values are ordered
 * to match the counter names registered during config() — processors
 * look up names via metadata_registry::get_sdk_pmc_counter_names().
 */
struct sample : trace_cache::cacheable_t
{
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::sdk_pmc_sample
    };

    sample() = default;
    sample(uint32_t dev_id, uint64_t ts, std::vector<double> vals)
    : device_id(dev_id)
    , timestamp(ts)
    , values(std::move(vals))
    {}

    uint32_t            device_id = 0;
    uint64_t            timestamp = 0;
    std::vector<double> values;
};

}  // namespace rocprofsys::pmc::collectors::sdk_pmc

namespace rocprofsys::trace_cache
{

/// @brief SDK PMC sample type alias
using sdk_pmc_sample = pmc::collectors::sdk_pmc::sample;

template <>
inline void
serialize(uint8_t* buffer, const pmc::collectors::sdk_pmc::sample& item)
{
    auto num_counters = static_cast<uint32_t>(item.values.size());
    utility::store_value(buffer, item.device_id, item.timestamp, num_counters);
    for(uint32_t i = 0; i < num_counters; ++i)
    {
        utility::store_value(buffer, item.values[i]);
    }
}

template <>
inline pmc::collectors::sdk_pmc::sample
deserialize(uint8_t*& buffer)
{
    pmc::collectors::sdk_pmc::sample item;
    uint32_t                         num_counters = 0;
    utility::parse_value(buffer, item.device_id, item.timestamp, num_counters);
    item.values.resize(num_counters);
    for(uint32_t i = 0; i < num_counters; ++i)
    {
        utility::parse_value(buffer, item.values[i]);
    }
    return item;
}

template <>
inline size_t
get_size(const pmc::collectors::sdk_pmc::sample& item)
{
    return utility::get_size(item.device_id, item.timestamp,
                             static_cast<uint32_t>(item.values.size())) +
           item.values.size() * sizeof(double);
}

}  // namespace rocprofsys::trace_cache
