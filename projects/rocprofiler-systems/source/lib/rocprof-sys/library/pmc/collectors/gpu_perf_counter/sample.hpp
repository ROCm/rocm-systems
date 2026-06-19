// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/cache_type_traits.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{
/**
 * @brief SDK PMC sample type for trace cache serialization.
 *
 * Stores a snapshot of GPU hardware counter values collected via
 * rocprofiler_sample_device_counting_service(). Each entry carries its
 * own qualified counter name including dimension positions.
 */
struct sample : trace_cache::cacheable_t
{
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::gpu_perf_counter_sample
    };

    sample() = default;
    sample(std::uint32_t dev_id, std::uint64_t time_ns, std::vector<counter_value> ent)
    : device_id(dev_id)
    , timestamp(time_ns)
    , entries(std::move(ent))
    {}

    std::uint32_t              device_id = 0;
    std::uint64_t              timestamp = 0;
    std::vector<counter_value> entries;
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter

namespace rocprofsys::trace_cache
{

/// @brief SDK PMC sample type alias
using gpu_perf_counter_sample = pmc::collectors::gpu_perf_counter::sample;

template <>
inline void
serialize(std::uint8_t* buffer, const pmc::collectors::gpu_perf_counter::sample& item)
{
    size_t     pos         = 0;
    const auto num_entries = static_cast<std::uint32_t>(item.entries.size());
    utility::store_value(item.device_id, buffer, pos);
    utility::store_value(item.timestamp, buffer, pos);
    utility::store_value(num_entries, buffer, pos);
    for(const auto& entry : item.entries)
    {
        utility::store_value(entry.counter_id, buffer, pos);
        utility::store_value(entry.value, buffer, pos);
    }
}

template <>
inline pmc::collectors::gpu_perf_counter::sample
deserialize(std::uint8_t*& buffer)
{
    pmc::collectors::gpu_perf_counter::sample item;
    std::uint32_t                             num_entries = 0;
    utility::parse_value(buffer, item.device_id, item.timestamp, num_entries);
    item.entries.resize(num_entries);
    for(auto& entry : item.entries)
    {
        utility::parse_value(buffer, entry.counter_id, entry.value);
    }
    return item;
}

template <>
inline size_t
get_size(const pmc::collectors::gpu_perf_counter::sample& item)
{
    size_t total_size = utility::get_size(
        item.device_id, item.timestamp, static_cast<std::uint32_t>(item.entries.size()));
    for(const auto& entry : item.entries)
    {
        total_size += utility::get_size(entry.counter_id, entry.value);
    }
    return total_size;
}

}  // namespace rocprofsys::trace_cache
