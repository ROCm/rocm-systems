// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/gpu/types.hpp"

#include <cstdint>

namespace rocprofsys::pmc::collectors::gpu
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
    sample(enabled_metrics _settings, std::uint32_t _device_id, size_t _timestamp,
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

    template <class Archive>
    ROCPROFSYS_INLINE void serialize(Archive& ar)
    {
        // Field order MUST match the legacy serialize free function exactly.
        // xcp_stats is a std::array<xcp_metrics, 8> and is written as raw
        // bytes (sizeof(std::array<xcp_metrics, 8>)) by the legacy path -
        // matched here via archive's std::array branch (memcpy whole when the
        // element is trivially copyable). vcn_activity / jpeg_activity /
        // xgmi.data_acc.read / write are similarly handled.
        auto enabled_value = static_cast<std::uint32_t>(enabled_metric.value);
        ar(enabled_value, device_id, timestamp, metric_values.average_socket_power,
           metric_values.current_socket_power, metric_values.memory_usage,
           metric_values.hotspot_temperature, metric_values.edge_temperature,
           metric_values.gfx_activity, metric_values.umc_activity,
           metric_values.mm_activity, metric_values.xcp_stats, metric_values.vcn_activity,
           metric_values.jpeg_activity, metric_values.xgmi.link.width,
           metric_values.xgmi.link.speed, metric_values.xgmi.data_acc.read,
           metric_values.xgmi.data_acc.write, metric_values.pcie.link.width,
           metric_values.pcie.link.speed, metric_values.pcie.bandwidth.acc,
           metric_values.pcie.bandwidth.inst, metric_values.sdma_usage,
           metric_values.gfx_clock_mhz, metric_values.mem_clock_mhz);
        if constexpr(std::is_same_v<Archive, trace_cache::input_archive>)
        {
            enabled_metric.value = enabled_value;
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu

namespace rocprofsys::trace_cache
{

template <>
inline void
serialize(std::uint8_t* buffer, const pmc::collectors::gpu::sample& item)
{
    utility::store_value(
        buffer, static_cast<std::uint32_t>(item.enabled_metric.value), item.device_id,
        item.timestamp, item.metric_values.average_socket_power,
        item.metric_values.current_socket_power, item.metric_values.memory_usage,
        item.metric_values.hotspot_temperature, item.metric_values.edge_temperature,
        item.metric_values.gfx_activity, item.metric_values.umc_activity,
        item.metric_values.mm_activity, item.metric_values.xcp_stats,
        item.metric_values.vcn_activity, item.metric_values.jpeg_activity,
        item.metric_values.xgmi.link.width, item.metric_values.xgmi.link.speed,
        item.metric_values.xgmi.data_acc.read, item.metric_values.xgmi.data_acc.write,
        item.metric_values.pcie.link.width, item.metric_values.pcie.link.speed,
        item.metric_values.pcie.bandwidth.acc, item.metric_values.pcie.bandwidth.inst,
        item.metric_values.sdma_usage, item.metric_values.gfx_clock_mhz,
        item.metric_values.mem_clock_mhz);
}

template <>
inline pmc::collectors::gpu::sample
deserialize(std::uint8_t*& buffer)
{
    pmc::collectors::gpu::sample item;
    utility::parse_value(
        buffer, item.enabled_metric.value, item.device_id, item.timestamp,
        item.metric_values.average_socket_power, item.metric_values.current_socket_power,
        item.metric_values.memory_usage, item.metric_values.hotspot_temperature,
        item.metric_values.edge_temperature, item.metric_values.gfx_activity,
        item.metric_values.umc_activity, item.metric_values.mm_activity,
        item.metric_values.xcp_stats, item.metric_values.vcn_activity,
        item.metric_values.jpeg_activity, item.metric_values.xgmi.link.width,
        item.metric_values.xgmi.link.speed, item.metric_values.xgmi.data_acc.read,
        item.metric_values.xgmi.data_acc.write, item.metric_values.pcie.link.width,
        item.metric_values.pcie.link.speed, item.metric_values.pcie.bandwidth.acc,
        item.metric_values.pcie.bandwidth.inst, item.metric_values.sdma_usage,
        item.metric_values.gfx_clock_mhz, item.metric_values.mem_clock_mhz);
    return item;
}

template <>
inline size_t
get_size(const pmc::collectors::gpu::sample& item)
{
    return utility::get_size(
        item.enabled_metric.value, item.device_id, item.timestamp,
        item.metric_values.average_socket_power, item.metric_values.current_socket_power,
        item.metric_values.memory_usage, item.metric_values.hotspot_temperature,
        item.metric_values.edge_temperature, item.metric_values.gfx_activity,
        item.metric_values.umc_activity, item.metric_values.mm_activity,
        item.metric_values.xcp_stats, item.metric_values.vcn_activity,
        item.metric_values.jpeg_activity, item.metric_values.xgmi.link.width,
        item.metric_values.xgmi.link.speed, item.metric_values.xgmi.data_acc.read,
        item.metric_values.xgmi.data_acc.write, item.metric_values.pcie.link.width,
        item.metric_values.pcie.link.speed, item.metric_values.pcie.bandwidth.acc,
        item.metric_values.pcie.bandwidth.inst, item.metric_values.sdma_usage,
        item.metric_values.gfx_clock_mhz, item.metric_values.mem_clock_mhz);
}

/// @brief GPU PMC sample type alias
using gpu_pmc_sample = pmc::collectors::gpu::sample;

namespace type_traits
{
template <>
inline constexpr bool use_custom<pmc::collectors::gpu::sample> = true;
}  // namespace type_traits

}  // namespace rocprofsys::trace_cache
