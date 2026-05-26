// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/nic/types.hpp"
#include "logger/debug.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofsys::pmc::collectors::nic
{

/**
 * @brief NIC device wrapper for collecting RDMA statistics.
 *
 * Wraps a NIC driver and provides methods to query RDMA port statistics
 * including bytes, packets, and CNP metrics. Has zero AMD SMI dependency.
 *
 * @tparam Driver The NIC driver type (nic_driver or mock_nic_driver)
 */
template <typename Driver>
class device
{
public:
    /**
     * @brief Construct a NIC device wrapper.
     *
     * @param driver Shared pointer to the driver instance (owns the handle)
     * @param logical_index Device index for identification
     */
    device(std::shared_ptr<Driver> driver, size_t logical_index)
    : m_driver{ std::move(driver) }
    , m_index{ logical_index }
    {
        m_is_supported = initialize_device_info();
    }

    [[nodiscard]] bool is_supported() const noexcept { return m_is_supported; }

    [[nodiscard]] enabled_metrics get_supported_metrics() const noexcept
    {
        return m_supported_metrics;
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

    /**
     * @brief Collect current NIC RDMA metrics.
     *
     * Queries the first RDMA port for statistics and extracts
     * all metrics enumerated by METRIC_MAP.
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

        try
        {
            auto stats = m_driver->get_nic_rdma_port_statistics(0);

            static const std::unordered_map<std::string_view, std::uint64_t metrics::*>
                METRIC_MAP = { { "rx_rdma_ucast_bytes", &metrics::rx_rdma_ucast_bytes },
                               { "tx_rdma_ucast_bytes", &metrics::tx_rdma_ucast_bytes },
                               { "rx_rdma_ucast_pkts", &metrics::rx_rdma_ucast_pkts },
                               { "tx_rdma_ucast_pkts", &metrics::tx_rdma_ucast_pkts },
                               { "rx_rdma_cnp_pkts", &metrics::rx_rdma_cnp_pkts },
                               { "tx_rdma_cnp_pkts", &metrics::tx_rdma_cnp_pkts },
                               { "tx_rdma_ack_timeout", &metrics::tx_rdma_ack_timeout },
                               { "resp_tx_pkt_seq_err", &metrics::resp_tx_pkt_seq_err },
                               { "req_rx_pkt_seq_err", &metrics::req_rx_pkt_seq_err },
                               { "req_rx_impl_nak_seq_err",
                                 &metrics::req_rx_impl_nak_seq_err } };

            for(const auto& stat : stats)
            {
                auto it = METRIC_MAP.find(stat.name);
                if(it != METRIC_MAP.end())
                {
                    nic_metrics.*(it->second) = stat.value;
                }
            }
        } catch(const std::runtime_error& e)
        {
            LOG_DEBUG("NIC device [{}] metrics query failed: {}", m_index, e.what());
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
        try
        {
            auto asic      = m_driver->get_nic_asic_info();
            m_product_name = asic.product_name;
            m_vendor_name  = asic.vendor_name;
        } catch(const std::runtime_error& e)
        {
            LOG_DEBUG("NIC device [{}]: {}", m_index, e.what());
        }

        try
        {
            auto port     = m_driver->get_nic_port_info();
            m_device_name = port.device_name;
        } catch(const std::runtime_error& e)
        {
            LOG_DEBUG("NIC device [{}]: {}", m_index, e.what());
        }

        try
        {
            auto rdma         = m_driver->get_nic_rdma_info();
            m_rdma_port_count = rdma.port_count;
        } catch(const std::runtime_error& e)
        {
            LOG_DEBUG("NIC device [{}]: {}", m_index, e.what());
            return false;
        }

        if(m_rdma_port_count == 0)
        {
            LOG_DEBUG("NIC device [{}] has no RDMA ports", m_index);
            return false;
        }

        try
        {
            auto stats = m_driver->get_nic_rdma_port_statistics(0);
            if(stats.empty())
            {
                LOG_DEBUG("NIC device [{}] has no RDMA statistics available", m_index);
                return false;
            }
        } catch(const std::runtime_error& e)
        {
            LOG_DEBUG("NIC device [{}]: {}", m_index, e.what());
            return false;
        }

        m_supported_metrics.value = ALL_NIC_METRICS;

        LOG_DEBUG("NIC device [{}] ({}) initialized with {} RDMA port(s)", m_index,
                  m_device_name, m_rdma_port_count);

        return true;
    }

    std::shared_ptr<Driver> m_driver;
    enabled_metrics         m_supported_metrics;
    const size_t            m_index;
    std::string             m_device_name;
    std::string             m_product_name;
    std::string             m_vendor_name;
    std::uint8_t            m_rdma_port_count = 0;
    bool                    m_is_supported    = false;
};

}  // namespace rocprofsys::pmc::collectors::nic
