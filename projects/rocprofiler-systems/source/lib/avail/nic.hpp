// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "avail/records.hpp"

#include "backends/amd_smi/device.hpp"
#include "backends/amd_smi/nic_types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofsys::avail
{

inline constexpr std::string_view k_ainic_unavailable_message =
    "AMD SMI AINIC support is not available in this build";

struct nic_listing_result
{
    std::vector<device_record> devices;
    std::vector<metric_record> metrics;
    std::optional<diagnostic>  issue;
};

#if defined(ROCPROFSYS_BUILD_AINIC) && ROCPROFSYS_BUILD_AINIC == 1
namespace inventory
{

struct nic_metric_description
{
    std::string_view name;
    std::string_view description;
    std::string_view unit;
};

inline constexpr auto k_nic_metric_descriptions = std::array{
    nic_metric_description{ .name        = "rx_rdma_ucast_bytes",
                            .description = "RDMA unicast bytes received",
                            .unit        = "bytes" },
    nic_metric_description{ .name        = "tx_rdma_ucast_bytes",
                            .description = "RDMA unicast bytes transmitted",
                            .unit        = "bytes" },
    nic_metric_description{ .name        = "rx_rdma_ucast_pkts",
                            .description = "RDMA unicast packets received",
                            .unit        = "packets" },
    nic_metric_description{ .name        = "tx_rdma_ucast_pkts",
                            .description = "RDMA unicast packets transmitted",
                            .unit        = "packets" },
    nic_metric_description{ .name = "rx_rdma_cnp_pkts",
                            .description =
                                "RDMA congestion notification packets received",
                            .unit = "packets" },
    nic_metric_description{ .name = "tx_rdma_cnp_pkts",
                            .description =
                                "RDMA congestion notification packets transmitted",
                            .unit = "packets" },
    nic_metric_description{ .name        = "tx_rdma_ack_timeout",
                            .description = "RDMA local acknowledgement timeout errors",
                            .unit        = "count" },
    nic_metric_description{ .name        = "resp_tx_pkt_seq_err",
                            .description = "RDMA responder packet sequence errors",
                            .unit        = "count" },
    nic_metric_description{ .name        = "req_rx_pkt_seq_err",
                            .description = "RDMA requester packet sequence errors",
                            .unit        = "count" },
    nic_metric_description{ .name        = "req_rx_impl_nak_seq_err",
                            .description = "RDMA requester implicit NAK sequence errors",
                            .unit        = "count" },
};

inline void
set_nic_issue(nic_listing_result& result, capability_kind capability, std::string message)
{
    if(!result.issue)
    {
        result.issue = diagnostic{
            .capability = capability,
            .source     = source_id::amd_smi,
            .message    = std::move(message),
        };
    }
}

[[nodiscard]] inline const nic_metric_description*
find_nic_metric_description(std::string_view name)
{
    const auto itr =
        std::ranges::find_if(k_nic_metric_descriptions,
                             [name](const auto& entry) { return entry.name == name; });
    return (itr == k_nic_metric_descriptions.end()) ? nullptr : &*itr;
}

inline void
append_nic_metric(std::vector<metric_record>& records, std::string name,
                  std::string device_id)
{
    const auto* metadata = find_nic_metric_description(name);
    records.emplace_back(metric_record{
        .name        = std::move(name),
        .description = metadata ? std::string{ metadata->description }
                                : std::string{ "AMD SMI RDMA counter" },
        .unit        = metadata ? std::string{ metadata->unit } : std::string{ "count" },
        .source      = source_id::amd_smi,
        .status      = availability::available,
        .device_id   = std::move(device_id),
    });
}

template <typename Backend>
[[nodiscard]] backends::amd_smi::nic::ports
query_nic_ports(nic_listing_result& result, const backends::amd_smi::device<Backend>& nic,
                bool emit_devices)
{
    try
    {
        return nic.get_nic_ports();
    } catch(const std::exception& err)
    {
        set_nic_issue(result,
                      emit_devices ? capability_kind::nic_devices
                                   : capability_kind::nic_metrics,
                      err.what());
        return {};
    }
}

template <typename Backend>
[[nodiscard]] backends::amd_smi::nic::rdma_ports
query_nic_rdma_ports(nic_listing_result&                       result,
                     const backends::amd_smi::device<Backend>& nic)
{
    try
    {
        return nic.get_nic_rdma_ports();
    } catch(const std::exception& err)
    {
        set_nic_issue(result, capability_kind::nic_metrics, err.what());
        return {};
    }
}

template <typename Backend>
[[nodiscard]] std::vector<backends::amd_smi::nic::stat_entry>
query_nic_statistics(nic_listing_result&                       result,
                     const backends::amd_smi::device<Backend>& nic,
                     std::uint8_t                              query_index)
{
    try
    {
        return nic.get_nic_rdma_port_statistics(query_index);
    } catch(const std::exception& err)
    {
        set_nic_issue(result, capability_kind::nic_metrics, err.what());
        return {};
    }
}

template <typename Backend>
void
append_nic_metrics(nic_listing_result&                                 result,
                   const backends::amd_smi::device<Backend>&           nic,
                   const std::unordered_map<std::string, std::string>& port_ids)
{
    for(const auto& rdma_port : query_nic_rdma_ports(result, nic))
    {
        const auto port = port_ids.find(rdma_port.device_name);
        if(port == port_ids.end())
        {
            set_nic_issue(result, capability_kind::nic_metrics,
                          "AMD SMI RDMA port did not match a NIC port");
            continue;
        }
        for(auto& stat : query_nic_statistics(result, nic, rdma_port.query_index))
        {
            append_nic_metric(result.metrics, std::move(stat.name), port->second);
        }
    }
}

inline std::unordered_map<std::string, std::string>
append_nic_ports(nic_listing_result& result, const backends::amd_smi::nic::ports& ports,
                 const std::string& parent_id, bool emit_devices)
{
    std::unordered_map<std::string, std::string> port_ids;
    for(const auto& port : ports)
    {
        const auto port_id = parent_id + "-port-" + std::to_string(port.number);
        if(emit_devices)
        {
            result.devices.emplace_back(device_record{
                .id        = port_id,
                .kind      = device_kind::nic_port,
                .name      = port.device_name.empty() ? port_id : port.device_name,
                .parent_id = parent_id,
            });
        }
        if(!port.device_name.empty())
        {
            port_ids.emplace(port.device_name, port_id);
        }
    }
    return port_ids;
}

template <typename Backend>
[[nodiscard]] device_record
make_nic_parent(nic_listing_result& result, const backends::amd_smi::device<Backend>& nic,
                std::size_t index)
{
    const auto    parent_id = "nic-" + std::to_string(index);
    device_record parent{
        .id            = parent_id,
        .kind          = device_kind::nic,
        .name          = parent_id,
        .logical_index = index,
    };
    try
    {
        const auto asic = nic.get_nic_asic_info();
        parent.name     = asic.product_name.empty() ? parent_id : asic.product_name;
        parent.vendor   = asic.vendor_name.empty()
                              ? std::nullopt
                              : std::optional<std::string>{ asic.vendor_name };
        parent.product  = asic.product_name.empty()
                              ? std::nullopt
                              : std::optional<std::string>{ asic.product_name };
    } catch(const std::exception& err)
    {
        set_nic_issue(result, capability_kind::nic_devices, err.what());
    }
    try
    {
        parent.pci_bdf = nic.get_nic_bdf();
    } catch(const std::exception& err)
    {
        set_nic_issue(result, capability_kind::nic_devices, err.what());
    }
    return parent;
}

/**
 * Lists AMD SMI NIC parent/port devices and RDMA metric capabilities.
 *
 * @param backend Initialized AMD SMI backend session.
 * @param include_devices Whether device records were requested.
 * @param include_metrics Whether metric records were requested.
 * @return Requested NIC records and the first AMD SMI diagnostic, if any.
 */
template <typename Backend>
[[nodiscard]] nic_listing_result
nic_inventory(std::shared_ptr<Backend> backend, bool include_devices,
              bool include_metrics)
{
    nic_listing_result result;
    const auto         handles = backend->enumerate_nic_handles();
    for(std::size_t idx = 0; idx < handles.size(); ++idx)
    {
        const backends::amd_smi::device<Backend> nic{ backend, handles[idx] };
        const auto                               parent_id = "nic-" + std::to_string(idx);
        if(include_devices)
        {
            result.devices.push_back(make_nic_parent(result, nic, idx));
        }
        const auto ports    = query_nic_ports(result, nic, include_devices);
        const auto port_ids = append_nic_ports(result, ports, parent_id, include_devices);
        if(include_metrics)
        {
            append_nic_metrics(result, nic, port_ids);
        }
    }
    return result;
}

}  // namespace inventory
#endif

[[nodiscard]] nic_listing_result
query_nic_inventory(bool include_devices, bool include_metrics);

}  // namespace rocprofsys::avail
