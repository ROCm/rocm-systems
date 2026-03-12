// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/collectors/nic/types.hpp"
#include "logger/debug.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace nic
{

#if ROCPROFSYS_USE_ROCM > 0

using ::rocprofsys::pmc::collectors::nic::enabled_metrics;
using ::rocprofsys::pmc::collectors::nic::metrics;

/**
 * @brief NIC device wrapper for collecting RDMA statistics.
 *
 * Wraps an AMD SMI NIC processor handle and provides methods to
 * query RDMA port statistics including bytes, packets, and CNP metrics.
 *
 * @tparam Driver The AMD SMI driver type (allows mock injection for testing)
 */
template <typename Driver>
class device
{
public:
    /**
     * @brief Construct a NIC device wrapper.
     *
     * @param driver Shared pointer to the driver instance
     * @param handle AMD SMI processor handle for this NIC
     * @param processor_type Type of processor (should be AMD_NIC)
     * @param logical_index Device index for identification
     */
    device(std::shared_ptr<Driver> driver, amdsmi_processor_handle handle,
           processor_type_t processor_type, size_t logical_index)
    : m_driver_api{ std::move(driver) }
    , m_device_handle{ handle }
    , m_device_type{ processor_type }
    , m_index{ logical_index }
    {
        m_is_supported = initialize_device_info();
    }

    [[nodiscard]] bool is_supported() const noexcept { return m_is_supported; }

    [[nodiscard]] enabled_metrics get_supported_metrics() const noexcept
    {
        return m_supported_metrics;
    }

    [[nodiscard]] processor_type_t get_device_type() const noexcept
    {
        return m_device_type;
    }

    [[nodiscard]] size_t get_index() const noexcept { return m_index; }

    [[nodiscard]] const std::string& get_name() const noexcept { return m_device_name; }

    [[nodiscard]] const std::string& get_product_name() const noexcept
    {
        return m_product_name;
    }

    [[nodiscard]] const std::string& get_vendor_name() const noexcept
    {
        return m_vendor_name;
    }

    [[nodiscard]] amdsmi_processor_handle get_handle() const noexcept
    {
        return m_device_handle;
    }

    /**
     * @brief Collect current NIC RDMA metrics.
     *
     * Queries the first RDMA port for statistics and extracts the
     * 6 RDMA metrics: rx/tx bytes, rx/tx packets, and rx/tx CNP packets.
     *
     * @return Collected metrics (zeros if query fails)
     */
    [[nodiscard]] metrics get_nic_metrics() const
    {
        metrics nic_metrics{};

        if(m_rdma_port_count == 0)
        {
            return nic_metrics;
        }

        // Query statistics for the first RDMA port
        uint32_t num_stats = 0;
        if(m_driver_api->get_nic_rdma_port_statistics(m_device_handle, 0, &num_stats,
                                                      nullptr) != AMDSMI_STATUS_SUCCESS)
        {
            return nic_metrics;
        }

        if(num_stats == 0)
        {
            return nic_metrics;
        }

        std::vector<amdsmi_nic_stat_t> stats(num_stats);
        if(m_driver_api->get_nic_rdma_port_statistics(
               m_device_handle, 0, &num_stats, stats.data()) != AMDSMI_STATUS_SUCCESS)
        {
            return nic_metrics;
        }

        // Extract the 6 metrics we care about
        for(const auto& stat : stats)
        {
            if(std::strcmp(stat.name, "rx_rdma_ucast_bytes") == 0)
            {
                nic_metrics.rx_rdma_ucast_bytes = stat.value;
            }
            else if(std::strcmp(stat.name, "tx_rdma_ucast_bytes") == 0)
            {
                nic_metrics.tx_rdma_ucast_bytes = stat.value;
            }
            else if(std::strcmp(stat.name, "rx_rdma_ucast_pkts") == 0)
            {
                nic_metrics.rx_rdma_ucast_pkts = stat.value;
            }
            else if(std::strcmp(stat.name, "tx_rdma_ucast_pkts") == 0)
            {
                nic_metrics.tx_rdma_ucast_pkts = stat.value;
            }
            else if(std::strcmp(stat.name, "rx_rdma_cnp_pkts") == 0)
            {
                nic_metrics.rx_rdma_cnp_pkts = stat.value;
            }
            else if(std::strcmp(stat.name, "tx_rdma_cnp_pkts") == 0)
            {
                nic_metrics.tx_rdma_cnp_pkts = stat.value;
            }
        }

        return nic_metrics;
    }

private:
    /**
     * @brief Initialize device info and determine supported metrics.
     *
     * Queries port and RDMA device information to determine what
     * statistics are available from this NIC.
     *
     * @return true if the device supports at least some metrics
     */
    bool initialize_device_info()
    {
        // Get ASIC info for vendor and product names
        amdsmi_nic_asic_info_t asic_info{};
        if(m_driver_api->get_nic_asic_info(m_device_handle, &asic_info) ==
           AMDSMI_STATUS_SUCCESS)
        {
            m_product_name = asic_info.product_name;
            m_vendor_name  = asic_info.vendor_name;
        }

        // Get port info to determine the device name
        amdsmi_nic_port_info_t port_info{};
        if(m_driver_api->get_nic_port_info(m_device_handle, &port_info) ==
           AMDSMI_STATUS_SUCCESS)
        {
            if(port_info.num_ports > 0)
            {
                m_device_name = port_info.ports[0].netdev;
            }
        }

        // Get RDMA device info
        amdsmi_nic_rdma_devices_info_t rdma_info{};
        if(m_driver_api->get_nic_rdma_dev_info(m_device_handle, &rdma_info) !=
           AMDSMI_STATUS_SUCCESS)
        {
            LOG_DEBUG("NIC device [{}] does not support RDMA queries", m_index);
            return false;
        }

        if(rdma_info.num_rdma_dev == 0)
        {
            LOG_DEBUG("NIC device [{}] has no RDMA devices", m_index);
            return false;
        }

        // Use the first RDMA device's first port
        m_rdma_port_count = rdma_info.rdma_dev_info[0].num_rdma_ports;
        if(m_rdma_port_count == 0)
        {
            LOG_DEBUG("NIC device [{}] has no RDMA ports", m_index);
            return false;
        }

        // Try to get statistics to verify support
        uint32_t num_stats = 0;
        if(m_driver_api->get_nic_rdma_port_statistics(m_device_handle, 0, &num_stats,
                                                      nullptr) != AMDSMI_STATUS_SUCCESS)
        {
            LOG_DEBUG("NIC device [{}] failed to query statistics count", m_index);
            return false;
        }

        if(num_stats == 0)
        {
            LOG_DEBUG("NIC device [{}] has no RDMA statistics available", m_index);
            return false;
        }

        // All 6 metrics are assumed supported if we can query stats
        m_supported_metrics.bits.rx_rdma_ucast_bytes = 1;
        m_supported_metrics.bits.tx_rdma_ucast_bytes = 1;
        m_supported_metrics.bits.rx_rdma_ucast_pkts  = 1;
        m_supported_metrics.bits.tx_rdma_ucast_pkts  = 1;
        m_supported_metrics.bits.rx_rdma_cnp_pkts    = 1;
        m_supported_metrics.bits.tx_rdma_cnp_pkts    = 1;

        LOG_DEBUG("NIC device [{}] ({}) initialized with {} RDMA port(s)", m_index,
                  m_device_name, m_rdma_port_count);

        return m_supported_metrics.value != 0;
    }

    std::shared_ptr<Driver> m_driver_api;
    amdsmi_processor_handle m_device_handle;
    processor_type_t        m_device_type;
    enabled_metrics         m_supported_metrics;
    size_t                  m_index;
    std::string             m_device_name;
    std::string             m_product_name;
    std::string             m_vendor_name;
    uint8_t                 m_rdma_port_count = 0;
    bool                    m_is_supported    = false;
};

#endif  // ROCPROFSYS_USE_ROCM > 0

}  // namespace nic
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
