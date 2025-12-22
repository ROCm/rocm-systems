#pragma once

#include "core/trace_cache/sample_type.hpp"
#include "library/amd_smi/common.hpp"
#include <cstdint>

namespace rocprofsys
{

namespace trace_cache
{

struct amd_smi_sample : cacheable_t
{
    static constexpr trace_cache::type_identifier_t type_identifier{
        trace_cache::type_identifier_t::amd_smi_sample
    };

    amd_smi_sample() = default;
    amd_smi_sample(amd_smi::enabled_metric _settings, uint32_t _device_id,
                   size_t _timestamp, amd_smi::smi_metrics _metrics)
    : enabled_metric(_settings)
    , device_id(_device_id)
    , timestamp(_timestamp)
    , metrics(_metrics)
    {}

    amd_smi::enabled_metric enabled_metric;
    size_t                  device_id;
    uint64_t                timestamp;
    amd_smi::smi_metrics    metrics;
};

template <>
inline void
serialize(uint8_t* buffer, const amd_smi_sample& item)
{
    utility::store_value(
        buffer, static_cast<uint32_t>(item.enabled_metric.value), item.device_id,
        item.timestamp, item.metrics.average_socket_power,
        item.metrics.current_socket_power, item.metrics.memory_usage,
        item.metrics.hotspot_temperature, item.metrics.edge_temperature,
        item.metrics.gfx_activity, item.metrics.umc_activity, item.metrics.mm_activity,
        item.metrics.xcp_stats, item.metrics.xgmi_link_width,
        item.metrics.xgmi_link_speed, item.metrics.xgmi_read_data_acc,
        item.metrics.xgmi_write_data_acc, item.metrics.pcie_link_width,
        item.metrics.pcie_link_speed, item.metrics.pcie_bandwidth_acc,
        item.metrics.pcie_bandwidth_inst);
}

template <>
inline amd_smi_sample
deserialize(uint8_t*& buffer)
{
    amd_smi_sample item;
    utility::parse_value(
        buffer, item.enabled_metric.value, item.device_id, item.timestamp,
        item.metrics.average_socket_power, item.metrics.current_socket_power,
        item.metrics.memory_usage, item.metrics.hotspot_temperature,
        item.metrics.edge_temperature, item.metrics.gfx_activity,
        item.metrics.umc_activity, item.metrics.mm_activity, item.metrics.xcp_stats,
        item.metrics.xgmi_link_width, item.metrics.xgmi_link_speed,
        item.metrics.xgmi_read_data_acc, item.metrics.xgmi_write_data_acc,
        item.metrics.pcie_link_width, item.metrics.pcie_link_speed,
        item.metrics.pcie_bandwidth_acc, item.metrics.pcie_bandwidth_inst);
    return item;
}

template <>
inline size_t
get_size(const amd_smi_sample& item)
{
    return utility::get_size(
        item.enabled_metric.value, item.device_id, item.timestamp,
        item.metrics.average_socket_power, item.metrics.current_socket_power,
        item.metrics.memory_usage, item.metrics.hotspot_temperature,
        item.metrics.edge_temperature, item.metrics.gfx_activity,
        item.metrics.umc_activity, item.metrics.mm_activity, item.metrics.xcp_stats,
        item.metrics.xgmi_link_width, item.metrics.xgmi_link_speed,
        item.metrics.xgmi_read_data_acc, item.metrics.xgmi_write_data_acc,
        item.metrics.pcie_link_width, item.metrics.pcie_link_speed,
        item.metrics.pcie_bandwidth_acc, item.metrics.pcie_bandwidth_inst);
}

}  // namespace trace_cache
}  // namespace rocprofsys
