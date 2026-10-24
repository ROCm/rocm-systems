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

    template <typename Archive>
    void serialize(Archive& archive)
    {
        // The byte layout of the cache: gpu::metrics is a backend type and does not
        // serialize itself, so its fields are listed here in byte order.
        archive(enabled_metric, device_id, timestamp, metric_values.average_socket_power,
                metric_values.current_socket_power, metric_values.memory_usage,
                metric_values.hotspot_temperature, metric_values.edge_temperature,
                metric_values.gfx_activity, metric_values.umc_activity,
                metric_values.mm_activity, metric_values.xcp_stats,
                metric_values.vcn_activity, metric_values.jpeg_activity,
                metric_values.xgmi.link.width, metric_values.xgmi.link.speed,
                metric_values.xgmi.data_acc.read, metric_values.xgmi.data_acc.write,
                metric_values.pcie.link.width, metric_values.pcie.link.speed,
                metric_values.pcie.bandwidth.acc, metric_values.pcie.bandwidth.inst,
                metric_values.sdma_usage, metric_values.gfx_clock_mhz,
                metric_values.mem_clock_mhz);
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu

namespace rocprofsys::trace_cache
{

/// @brief GPU PMC sample type alias
using gpu_pmc_sample = pmc::collectors::gpu::sample;

}  // namespace rocprofsys::trace_cache
