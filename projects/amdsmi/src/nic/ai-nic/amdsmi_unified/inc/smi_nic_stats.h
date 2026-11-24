// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_SMI_NIC_STATS_H_
#define AMDSMI_UNIFIED_SMI_NIC_STATS_H_

#include <vector>

// Which output tier a vendor counter belongs to. Extended is a superset: every
// Default-tier entry is also returned when the caller asks for Extended.
enum class StatTier_t { Default, Extended };

// One vendor counter name and the tier it belongs to. name must match the
// driver's own `ethtool -S` (or netlink FEC stat) name exactly; no cross-driver
// normalization happens here.
struct StatEntry_t {
  const char* name;
  StatTier_t tier;
};

// A vendor's full counter allowlist, owned by its plugin (pensando_stats.cpp,
// broadcom_stats.cpp). A name absent from a port's collected counters is
// omitted, never reported as a placeholder value.
using StatTable_t = std::vector<StatEntry_t>;

#endif  // AMDSMI_UNIFIED_SMI_NIC_STATS_H_
