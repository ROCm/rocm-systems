// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_NIC_DRIVER_STATS_H_
#define AMDSMI_UNIFIED_NIC_DRIVER_STATS_H_

#include <linux/ethtool.h>

#include <cstddef>
#include <cstdint>
#include <functional>

#include "smi_nic_transport.h"

namespace amd::smi::nic::transport::ioctl_internal {

/**
 * The three ethtool requests behind `ethtool -S`, injectable so a test can model a driver whose
 * statistics count changes between them. `capacity` is the number of entries the buffer holds.
 */
struct DriverStatsIoctls_t {
  std::function<Result<uint32_t>()> get_count;
  std::function<int(ethtool_gstrings*, size_t capacity)> get_strings;
  std::function<int(ethtool_stats*, size_t capacity)> get_stats;
};

Result<VendorStatistics> read_driver_statistics(const DriverStatsIoctls_t& ioctls);

}  // namespace amd::smi::nic::transport::ioctl_internal

#endif  // AMDSMI_UNIFIED_NIC_DRIVER_STATS_H_
