// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/sample_type.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace rocprofsys::pmc::collectors::sdk_pmc
{

/**
 * @brief A single counter entry in an SDK PMC sample.
 *
 * Uses string_view — the backing data lives in the device's m_instance_map
 * during serialization, and in the trace cache buffer during deserialization.
 */
struct sample_entry
{
    std::string_view name;  ///< Qualified name, e.g. "SQC_ICACHE_HITS[WGP=0,SA=0,SE=0]"
    double           value;
};

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
        trace_cache::type_identifier_t::sdk_pmc_sample
    };

    sample() = default;
    sample(uint32_t dev_id, uint64_t time_ns, std::vector<sample_entry> ent)
    : device_id(dev_id)
    , timestamp(time_ns)
    , entries(std::move(ent))
    {}

    uint32_t                  device_id = 0;
    uint64_t                  timestamp = 0;
    std::vector<sample_entry> entries;
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
    size_t     pos         = 0;
    const auto num_entries = static_cast<uint32_t>(item.entries.size());
    utility::store_value(item.device_id, buffer, pos);
    utility::store_value(item.timestamp, buffer, pos);
    utility::store_value(num_entries, buffer, pos);
    for(uint32_t i = 0; i < num_entries; ++i)
    {
        utility::store_value(std::string_view(item.entries[i].name), buffer, pos);
        utility::store_value(item.entries[i].value, buffer, pos);
    }
}

template <>
inline pmc::collectors::sdk_pmc::sample
deserialize(uint8_t*& buffer)
{
    pmc::collectors::sdk_pmc::sample item;
    uint32_t                         num_entries = 0;
    utility::parse_value(buffer, item.device_id, item.timestamp, num_entries);
    item.entries.resize(num_entries);
    for(uint32_t i = 0; i < num_entries; ++i)
    {
        utility::parse_value(buffer, item.entries[i].name, item.entries[i].value);
    }
    return item;
}

template <>
inline size_t
get_size(const pmc::collectors::sdk_pmc::sample& item)
{
    size_t total_size = utility::get_size(item.device_id, item.timestamp,
                                          static_cast<uint32_t>(item.entries.size()));
    for(const auto& entry : item.entries)
    {
        total_size += utility::get_size(std::string_view(entry.name), entry.value);
    }
    return total_size;
}

}  // namespace rocprofsys::trace_cache
