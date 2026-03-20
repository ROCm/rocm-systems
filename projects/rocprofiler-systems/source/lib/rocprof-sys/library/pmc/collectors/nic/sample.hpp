// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

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
    sample(enabled_metrics _settings, uint32_t _device_id, std::string _device_name,
           uint64_t _timestamp, metrics _metric_values)
    : enabled_metric(_settings)
    , device_id(_device_id)
    , device_name(std::move(_device_name))
    , timestamp(_timestamp)
    , metric_values(_metric_values)
    {}

    enabled_metrics enabled_metric;
    uint32_t        device_id;
    std::string     device_name;
    uint64_t        timestamp;
    metrics         metric_values;
};

}  // namespace rocprofsys::pmc::collectors::nic

namespace rocprofsys
{
namespace trace_cache
{

/// @brief AINIC PMC sample type alias
using ainic_pmc_sample = pmc::collectors::nic::sample;

template <>
inline void
serialize(uint8_t* buffer, const pmc::collectors::nic::sample& item)
{
    utility::store_value(
        buffer, static_cast<uint32_t>(item.enabled_metric.value), item.device_id,
        std::string_view(item.device_name), item.timestamp,
        item.metric_values.rx_rdma_ucast_bytes, item.metric_values.tx_rdma_ucast_bytes,
        item.metric_values.rx_rdma_ucast_pkts, item.metric_values.tx_rdma_ucast_pkts,
        item.metric_values.rx_rdma_cnp_pkts, item.metric_values.tx_rdma_cnp_pkts);
}

template <>
inline pmc::collectors::nic::sample
deserialize(uint8_t*& buffer)
{
    pmc::collectors::nic::sample item;
    std::string_view             device_name_view;
    utility::parse_value(
        buffer, item.enabled_metric.value, item.device_id, device_name_view,
        item.timestamp, item.metric_values.rx_rdma_ucast_bytes,
        item.metric_values.tx_rdma_ucast_bytes, item.metric_values.rx_rdma_ucast_pkts,
        item.metric_values.tx_rdma_ucast_pkts, item.metric_values.rx_rdma_cnp_pkts,
        item.metric_values.tx_rdma_cnp_pkts);
    item.device_name = std::string(device_name_view);
    return item;
}

template <>
inline size_t
get_size(const pmc::collectors::nic::sample& item)
{
    return utility::get_size(
        item.enabled_metric.value, item.device_id, std::string_view(item.device_name),
        item.timestamp, item.metric_values.rx_rdma_ucast_bytes,
        item.metric_values.tx_rdma_ucast_bytes, item.metric_values.rx_rdma_ucast_pkts,
        item.metric_values.tx_rdma_ucast_pkts, item.metric_values.rx_rdma_cnp_pkts,
        item.metric_values.tx_rdma_cnp_pkts);
}

}  // namespace trace_cache
}  // namespace rocprofsys
