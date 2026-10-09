// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_AMD_SMI_NIC_STATS_H_
#define AMD_SMI_INCLUDE_AMD_SMI_NIC_STATS_H_

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "amd_smi/amdsmi.h"

// Included bare: the header opens its own extern "C".
#include "amd_smi/impl/nic/amdsmi_unified/interface/smi_nic_interface.h"

namespace amd::smi {

/**
 * Maps the public statistics scope onto the NIC library's. False for a value outside the enum, so
 * a mistyped scope is rejected instead of silently reading the default set. Takes an int because
 * a C caller can pass any value, and holding one in the enum type is undefined behavior.
 */
inline auto was_nic_stat_scope_resolved(int scope, smi_nic_stat_scope_t* nic_scope) -> bool {
  if (nic_scope == nullptr) {
    return false;
  }
  switch (scope) {
    case AMDSMI_NIC_STAT_SCOPE_DEFAULT:
      *nic_scope = SMI_NIC_STAT_SCOPE_DEFAULT;
      return true;
    case AMDSMI_NIC_STAT_SCOPE_EXTENDED:
      *nic_scope = SMI_NIC_STAT_SCOPE_EXTENDED;
      return true;
  }
  return false;
}

/**
 * Copies the entries that fit into the caller's array and sets *num_stats to the number available.
 * INSUFFICIENT_SIZE tells the caller the array was too small, so it can resize and call again.
 */
inline auto copy_nic_stats(const smi_nic_stat_info_t& info, uint32_t* num_stats,
                           amdsmi_nic_stat_t* stats) -> amdsmi_status_t {
  const auto num_copied = std::min(info.count, *num_stats);
  for (auto i = uint32_t{0}; i < num_copied; ++i) {
    std::snprintf(stats[i].name, sizeof(stats[i].name), "%s", info.stats[i].name);
    stats[i].value = info.stats[i].value;
  }
  const auto is_truncated = (info.count > *num_stats);
  *num_stats = info.count;
  return is_truncated ? AMDSMI_STATUS_INSUFFICIENT_SIZE : AMDSMI_STATUS_SUCCESS;
}

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_NIC_STATS_H_
