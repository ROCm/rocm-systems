// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/device_backend.hpp"

#include <gmock/gmock.h>

#include <cstdint>
#include <vector>

namespace rocprofsys::backends::amd_smi::testing
{

/**
 * @brief Mock NIC backend for unit testing device.hpp without AMD SMI.
 *
 * Provides the same interface as nic_backend but with zero AMD SMI dependency.
 * Methods throw std::runtime_error to simulate AMD SMI failures.
 */
struct mock_nic_backend
{
    MOCK_METHOD(nic::asic_info, get_nic_asic_info, (), (const));
    MOCK_METHOD(nic::port_info, get_nic_port_info, (), (const));
    MOCK_METHOD(nic::rdma_info, get_nic_rdma_info, (), (const));
    MOCK_METHOD(std::vector<nic::stat_entry>, get_nic_rdma_port_statistics,
                (std::uint8_t rdma_port_idx), (const));
};

}  // namespace rocprofsys::backends::amd_smi::testing
