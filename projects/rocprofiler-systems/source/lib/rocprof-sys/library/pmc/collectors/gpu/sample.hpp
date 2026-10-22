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
        archive(enabled_metric, device_id, timestamp, metric_values);
    }
};

}  // namespace rocprofsys::pmc::collectors::gpu

namespace rocprofsys::trace_cache
{

/// @brief GPU PMC sample type alias
using gpu_pmc_sample = pmc::collectors::gpu::sample;

}  // namespace rocprofsys::trace_cache
