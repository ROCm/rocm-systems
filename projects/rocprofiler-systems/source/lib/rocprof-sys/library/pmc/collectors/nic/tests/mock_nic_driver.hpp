// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/nic/types.hpp"

#include <gmock/gmock.h>

#include <cstdint>
#include <vector>

namespace rocprofsys::pmc::collectors::nic::testing
{

/**
 * @brief Mock NIC driver for unit testing device.hpp without AMD SMI.
 *
 * Provides the same interface as nic_driver but with zero AMD SMI dependency.
 * Methods throw std::runtime_error to simulate AMD SMI failures.
 */
struct mock_nic_driver
{
    MOCK_METHOD(asic_info, get_nic_asic_info, (), (const));
    MOCK_METHOD(port_info, get_nic_port_info, (), (const));
    MOCK_METHOD(rdma_info, get_nic_rdma_info, (), (const));
    MOCK_METHOD(std::vector<stat_entry>, get_nic_rdma_port_statistics,
                (std::uint8_t rdma_port_idx), (const));
};

}  // namespace rocprofsys::pmc::collectors::nic::testing
