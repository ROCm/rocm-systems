// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/trace_cache/sample_type.hpp"
#include "library/pmc/collectors/nic/types.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace rocprofsys::pmc::collectors::nic
{

/**
 * @brief NIC RDMA sample type for trace cache.
 *
 * This struct represents a sample of NIC RDMA metrics collected by the PMC.
 * It inherits from cacheable_t to support serialization to the trace cache.
 */
struct sample : trace_cache::cacheable_t
{
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::ainic_pmc_sample
    };

    sample() = default;
    sample(enabled_metrics _settings, std::uint32_t _device_id, std::string _device_name,
           std::uint64_t _timestamp, metrics _metric_values)
    : enabled_metric(_settings)
    , device_id(_device_id)
    , device_name(std::move(_device_name))
    , timestamp(_timestamp)
    , metric_values(_metric_values)
    {}

    enabled_metrics enabled_metric;
    std::uint32_t   device_id;
    std::string     device_name;
    std::uint64_t   timestamp;
    metrics         metric_values;

    template <typename Archive>
    void serialize(Archive& archive)
    {
        std::string_view name_view = device_name;
        archive(enabled_metric, device_id, name_view, timestamp, metric_values);
        if constexpr(Archive::k_is_loading)
        {
            device_name = std::string(name_view);
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::nic

namespace rocprofsys::trace_cache
{

/// @brief AINIC PMC sample type alias
using ainic_pmc_sample = pmc::collectors::nic::sample;

}  // namespace rocprofsys::trace_cache
