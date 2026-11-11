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
struct sample
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

    template <typename Archive>
    void serialize(Archive& archive)
    {
        archive(enabled_metric, device_id, timestamp, metric_values);
    }
};

}  // namespace rocprofsys::pmc::collectors::hipfile

namespace rocprofsys::trace_cache
{

/// @brief hipFile PMC sample type alias
using hipfile_pmc_sample = pmc::collectors::hipfile::sample;

}  // namespace rocprofsys::trace_cache
