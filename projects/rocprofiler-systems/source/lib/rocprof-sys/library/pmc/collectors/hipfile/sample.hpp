// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/cache_type_traits.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/hipfile/types.hpp"

#include <cstddef>
#include <cstdint>

namespace rocprofsys::pmc::collectors::hipfile
{

/**
 * @brief One poll of hipFile telemetry for a single GPU.
 *
 * The whole per-GPU snapshot travels as one record, matching the AMD SMI, NIC, and CPU
 * collectors. The alternative - one pmc_event_with_sample per metric - would rebuild the
 * track name and the PMC identifier as heap strings for every metric on every poll, on
 * the sampling thread that all the other collectors share. Here the names are resolved
 * once during post-processing, from METRIC_TABLE, so the sampling path stores a POD.
 */
struct sample : trace_cache::cacheable_t
{
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::hipfile_pmc_sample
    };

    sample() = default;
    sample(enabled_metrics _settings, std::uint32_t _device_id, std::uint64_t _timestamp,
           metrics _metric_values)
    : enabled_metric(_settings)
    , device_id(_device_id)
    , timestamp(_timestamp)
    , metric_values(_metric_values)
    {}

    enabled_metrics enabled_metric{};
    std::uint32_t   device_id = 0;
    std::uint64_t   timestamp = 0;
    metrics         metric_values{};
};

}  // namespace rocprofsys::pmc::collectors::hipfile

namespace rocprofsys::trace_cache
{

// metrics::query_failed is deliberately not carried: cache_policy::store_sample drops a
// failed query before it reaches the buffer, so every stored sample has it false and the
// default-constructed value round-trips correctly.
// Field order after timestamp is metrics member order, excluding query_failed (dropped
// before store). serialize, deserialize, and get_size must list the same fields.
template <>
inline void
serialize(std::uint8_t* buffer, const pmc::collectors::hipfile::sample& item)
{
    utility::store_value(
        buffer, static_cast<std::uint32_t>(item.enabled_metric.value), item.device_id,
        item.timestamp, item.metric_values.read_bytes, item.metric_values.write_bytes,
        item.metric_values.read_ops, item.metric_values.write_ops,
        item.metric_values.fastpath_reads, item.metric_values.fastpath_writes,
        item.metric_values.fallback_reads, item.metric_values.fallback_writes,
        item.metric_values.unaligned_reads, item.metric_values.unaligned_writes,
        item.metric_values.read_errors, item.metric_values.write_errors,
        item.metric_values.read_bandwidth, item.metric_values.write_bandwidth);
}

template <>
inline pmc::collectors::hipfile::sample
deserialize(std::uint8_t*& buffer)
{
    pmc::collectors::hipfile::sample item;
    utility::parse_value(
        buffer, item.enabled_metric.value, item.device_id, item.timestamp,
        item.metric_values.read_bytes, item.metric_values.write_bytes,
        item.metric_values.read_ops, item.metric_values.write_ops,
        item.metric_values.fastpath_reads, item.metric_values.fastpath_writes,
        item.metric_values.fallback_reads, item.metric_values.fallback_writes,
        item.metric_values.unaligned_reads, item.metric_values.unaligned_writes,
        item.metric_values.read_errors, item.metric_values.write_errors,
        item.metric_values.read_bandwidth, item.metric_values.write_bandwidth);
    return item;
}

template <>
inline size_t
get_size(const pmc::collectors::hipfile::sample& item)
{
    return utility::get_size(
        item.enabled_metric.value, item.device_id, item.timestamp,
        item.metric_values.read_bytes, item.metric_values.write_bytes,
        item.metric_values.read_ops, item.metric_values.write_ops,
        item.metric_values.fastpath_reads, item.metric_values.fastpath_writes,
        item.metric_values.fallback_reads, item.metric_values.fallback_writes,
        item.metric_values.unaligned_reads, item.metric_values.unaligned_writes,
        item.metric_values.read_errors, item.metric_values.write_errors,
        item.metric_values.read_bandwidth, item.metric_values.write_bandwidth);
}

/// @brief hipFile PMC sample type alias
using hipfile_pmc_sample = pmc::collectors::hipfile::sample;

}  // namespace rocprofsys::trace_cache
