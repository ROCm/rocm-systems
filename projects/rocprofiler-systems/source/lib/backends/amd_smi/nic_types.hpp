// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

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

struct port
{
    std::uint32_t number = 0;
    std::string   device_name;
};

/**
 * @brief NIC RDMA device info.
 */
struct rdma_info
{
    std::uint8_t port_count = 0;
};

struct rdma_port
{
    std::uint8_t query_index = 0;
    std::uint8_t number      = 0;
    std::string  device_name;
};

using ports      = std::vector<port>;
using rdma_ports = std::vector<rdma_port>;

/**
 * @brief Single RDMA port statistic entry.
 */
struct stat_entry
{
    std::string   name;
    std::uint64_t value = 0;
};

}  // namespace rocprofsys::backends::amd_smi::nic
