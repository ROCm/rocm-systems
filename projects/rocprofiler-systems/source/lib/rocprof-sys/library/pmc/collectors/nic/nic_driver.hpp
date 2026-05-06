// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/nic/types.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <amd_smi/amdsmi.h>

namespace rocprofsys::pmc::collectors::nic
{

/**
 * @brief NIC driver that owns a processor handle and abstracts AMD SMI.
 *
 * Each instance wraps a single NIC processor handle. All AMD SMI details
 * (handle, status codes, raw structs) are encapsulated here so that
 * device.hpp has zero AMD SMI dependency.
 *
 * Methods throw std::runtime_error on AMD SMI failures.
 */
class nic_driver
{
public:
    explicit nic_driver(amdsmi_processor_handle handle) noexcept
    : m_handle{ handle }
    {}

    /**
     * @brief Get NIC ASIC information.
     * @return ASIC info with vendor and product names.
     * @throws std::runtime_error If AMD SMI query fails.
     */
    [[nodiscard]] asic_info get_nic_asic_info() const
    {
        amdsmi_nic_asic_info_t raw{};
        check(amdsmi_get_nic_asic_info(m_handle, &raw), "get_nic_asic_info");
        return { raw.product_name, raw.vendor_name };
    }

    /**
     * @brief Get NIC port information.
     * @return Port info with device name (empty if no ports).
     * @throws std::runtime_error If AMD SMI query fails.
     */
    [[nodiscard]] port_info get_nic_port_info() const
    {
        amdsmi_nic_port_info_t raw{};
        check(amdsmi_get_nic_port_info(m_handle, &raw), "get_nic_port_info");
        if(raw.num_ports == 0)
        {
            return {};
        }
        return { raw.ports[0].netdev };
    }

    /**
     * @brief Get NIC RDMA device information.
     * @return RDMA info with port count (zero if no RDMA devices).
     * @throws std::runtime_error If AMD SMI query fails.
     */
    [[nodiscard]] rdma_info get_nic_rdma_info() const
    {
        amdsmi_nic_rdma_devices_info_t raw{};
        check(amdsmi_get_nic_rdma_dev_info(m_handle, &raw), "get_nic_rdma_dev_info");
        if(raw.num_rdma_dev == 0)
        {
            return { 0 };
        }
        return { raw.rdma_dev_info[0].num_rdma_ports };
    }

    /**
     * @brief Get NIC RDMA port statistics.
     * @param rdma_port_idx RDMA port index.
     * @return Vector of stat entries (empty if no statistics available).
     * @throws std::runtime_error If AMD SMI query fails.
     */
    [[nodiscard]] std::vector<stat_entry> get_nic_rdma_port_statistics(
        std::uint8_t rdma_port_idx) const
    {
        std::uint32_t num_stats = 0;
        check(amdsmi_get_nic_rdma_port_statistics(m_handle, rdma_port_idx, &num_stats,
                                                  nullptr),
              "get_nic_rdma_port_statistics (count)");

        if(num_stats == 0)
        {
            return {};
        }

        std::vector<amdsmi_nic_stat_t> raw(num_stats);
        check(amdsmi_get_nic_rdma_port_statistics(m_handle, rdma_port_idx, &num_stats,
                                                  raw.data()),
              "get_nic_rdma_port_statistics (data)");

        std::vector<stat_entry> result;
        result.reserve(num_stats);
        for(const auto& stat : raw)
        {
            result.emplace_back(stat_entry{ stat.name, stat.value });
        }
        return result;
    }

private:
    static void check(amdsmi_status_t status, std::string_view func)
    {
        if(status != AMDSMI_STATUS_SUCCESS)
        {
            const char* status_msg = nullptr;
            if(amdsmi_status_code_to_string(status, &status_msg) ==
                   AMDSMI_STATUS_SUCCESS &&
               status_msg != nullptr)
            {
                throw std::runtime_error(std::string(func) + " failed: " + status_msg);
            }
            throw std::runtime_error(std::string(func) + " failed with status " +
                                     std::to_string(static_cast<int>(status)));
        }
    }

    const amdsmi_processor_handle m_handle;
};

}  // namespace rocprofsys::pmc::collectors::nic
