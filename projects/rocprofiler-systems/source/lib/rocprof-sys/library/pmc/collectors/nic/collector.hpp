// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "library/pmc/collectors/base/collector.hpp"
#include "library/pmc/collectors/nic/nic_traits.hpp"

namespace rocprofsys::pmc::collectors::nic
{

/**
 * @brief NIC RDMA metrics collector for performance monitoring.
 *
 * This collector specializes the base::collector template for NIC devices
 * using AMD SMI. All NIC-specific behavior is defined in nic_traits.
 *
 * @tparam DeviceProvider Type providing NIC device enumeration and management
 * @tparam Config Configuration policy providing settings and output policies
 */
template <typename DeviceProvider, typename Config>
using collector = base::collector<nic_traits<DeviceProvider>, DeviceProvider, Config>;

}  // namespace rocprofsys::pmc::collectors::nic
