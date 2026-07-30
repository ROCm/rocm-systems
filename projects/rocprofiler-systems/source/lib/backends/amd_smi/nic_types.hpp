// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace rocprofsys::backends::amd_smi::nic
{

/**
 * @brief NIC ASIC identification info.
 */
struct asic_info
{
    std::string product_name;
    std::string vendor_name;
};

/**
 * @brief NIC port identification info.
 */
struct port_info
{
    std::string device_name;
};

/**
 * @brief NIC RDMA device info.
 */
struct rdma_info
{
    std::uint8_t port_count = 0;
};

/**
 * @brief Single RDMA port statistic entry.
 */
struct stat_entry
{
    std::string   name;
    std::uint64_t value = 0;
};

}  // namespace rocprofsys::backends::amd_smi::nic
