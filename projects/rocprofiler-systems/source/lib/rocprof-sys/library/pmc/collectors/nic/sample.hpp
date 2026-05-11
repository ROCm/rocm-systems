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

    template <class Archive>
    ROCPROFSYS_INLINE void serialize(Archive& ar)
    {
        // enabled_metric is a union (enabled_metrics); the legacy code passes
        // its `value` (uint32) on the wire. Stage through a local so input
        // archives write back to .value.
        auto enabled_value = static_cast<std::uint32_t>(enabled_metric.value);
        ar(enabled_value, device_id, device_name, timestamp,
           metric_values.rx_rdma_ucast_bytes, metric_values.tx_rdma_ucast_bytes,
           metric_values.rx_rdma_ucast_pkts, metric_values.tx_rdma_ucast_pkts,
           metric_values.rx_rdma_cnp_pkts, metric_values.tx_rdma_cnp_pkts);
        if constexpr(std::is_same_v<Archive, trace_cache::input_archive>)
        {
            enabled_metric.value = enabled_value;
        }
    }
};

}  // namespace rocprofsys::pmc::collectors::nic

namespace rocprofsys
{
namespace trace_cache
{

/// @brief AINIC PMC sample type alias
using ainic_pmc_sample = pmc::collectors::nic::sample;

}  // namespace trace_cache
}  // namespace rocprofsys
