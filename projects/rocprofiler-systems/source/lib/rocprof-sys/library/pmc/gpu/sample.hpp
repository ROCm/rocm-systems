// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/gpu/types.hpp"

#include <cstdint>

namespace rocprofsys
{
namespace pmc
{
namespace gpu
{

/**
 * @brief GPU PMC sample type.
 *
 * This struct represents a sample of GPU performance metrics collected by the PMC.
 */
struct sample : trace_cache::cacheable_t
{
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::gpu_pmc_sample
    };

    sample() = default;
    sample(enabled_metrics _settings, uint32_t _device_id, size_t _timestamp,
           metrics _metric_values)
    : enabled_metric(_settings)
    , device_id(_device_id)
    , timestamp(_timestamp)
    , metric_values(_metric_values)
    {}

    enabled_metrics enabled_metric;
    uint32_t        device_id;
    uint64_t        timestamp;
    metrics         metric_values;
};

}  // namespace gpu
}  // namespace pmc

namespace trace_cache
{

template <>
inline void
serialize(uint8_t* buffer, const pmc::gpu::sample& item)
{
    utility::store_value(
        buffer, static_cast<uint32_t>(item.enabled_metric.value()), item.device_id,
        item.timestamp, item.metric_values.average_socket_power,
        item.metric_values.current_socket_power, item.metric_values.memory_usage,
        item.metric_values.hotspot_temperature, item.metric_values.edge_temperature,
        item.metric_values.gfx_activity, item.metric_values.umc_activity,
        item.metric_values.mm_activity, item.metric_values.xcp_stats,
        item.metric_values.vcn_activity, item.metric_values.jpeg_activity,
        item.metric_values.xgmi.link.width, item.metric_values.xgmi.link.speed,
        item.metric_values.xgmi.data_acc.read, item.metric_values.xgmi.data_acc.write,
        item.metric_values.pcie.link.width, item.metric_values.pcie.link.speed,
        item.metric_values.pcie.bandwidth.acc, item.metric_values.pcie.bandwidth.inst);
}

template <>
inline pmc::gpu::sample
deserialize(uint8_t*& buffer)
{
    pmc::gpu::sample item;
    utility::parse_value(
        buffer, item.enabled_metric.value_ref(), item.device_id, item.timestamp,
        item.metric_values.average_socket_power, item.metric_values.current_socket_power,
        item.metric_values.memory_usage, item.metric_values.hotspot_temperature,
        item.metric_values.edge_temperature, item.metric_values.gfx_activity,
        item.metric_values.umc_activity, item.metric_values.mm_activity,
        item.metric_values.xcp_stats, item.metric_values.vcn_activity,
        item.metric_values.jpeg_activity, item.metric_values.xgmi.link.width,
        item.metric_values.xgmi.link.speed, item.metric_values.xgmi.data_acc.read,
        item.metric_values.xgmi.data_acc.write, item.metric_values.pcie.link.width,
        item.metric_values.pcie.link.speed, item.metric_values.pcie.bandwidth.acc,
        item.metric_values.pcie.bandwidth.inst);
    return item;
}

template <>
inline size_t
get_size(const pmc::gpu::sample& item)
{
    return utility::get_size(
        item.enabled_metric.value(), item.device_id, item.timestamp,
        item.metric_values.average_socket_power, item.metric_values.current_socket_power,
        item.metric_values.memory_usage, item.metric_values.hotspot_temperature,
        item.metric_values.edge_temperature, item.metric_values.gfx_activity,
        item.metric_values.umc_activity, item.metric_values.mm_activity,
        item.metric_values.xcp_stats, item.metric_values.vcn_activity,
        item.metric_values.jpeg_activity, item.metric_values.xgmi.link.width,
        item.metric_values.xgmi.link.speed, item.metric_values.xgmi.data_acc.read,
        item.metric_values.xgmi.data_acc.write, item.metric_values.pcie.link.width,
        item.metric_values.pcie.link.speed, item.metric_values.pcie.bandwidth.acc,
        item.metric_values.pcie.bandwidth.inst);
}

}  // namespace trace_cache
}  // namespace rocprofsys
