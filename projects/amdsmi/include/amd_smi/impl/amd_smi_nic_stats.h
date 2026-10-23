// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_AMD_SMI_NIC_STATS_H_
#define AMD_SMI_INCLUDE_AMD_SMI_NIC_STATS_H_

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>

#include "amd_smi/amdsmi.h"

// Included bare: the header opens its own extern "C".
#include "amd_smi/impl/nic/amd_smi_nic_interface.h"
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

/** The public link type for how the NIC library says a NIC connects to a device. */
inline auto to_amdsmi_link_type(smi_nic_link_type_t link) -> amdsmi_link_type_t {
  switch (link) {
    case SMI_NIC_LINK_TYPE_PCIE:
      return AMDSMI_LINK_TYPE_PCIE;
    case SMI_NIC_LINK_TYPE_NUMA:
      return AMDSMI_LINK_TYPE_NUMA;
    case SMI_NIC_LINK_TYPE_XNUMA:
      return AMDSMI_LINK_TYPE_XNUMA;
    case SMI_NIC_LINK_TYPE_UNKNOWN:
      break;
  }
  return AMDSMI_LINK_TYPE_UNKNOWN;
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

/**
 * Copies the hwmon records that fit into the caller's array and sets *num_sensors to the number
 * available. With no array only the count is reported. INSUFFICIENT_SIZE tells the caller the
 * array was too small.
 */
inline auto copy_nic_sensors(const smi_nic_sensor_info_t& info, uint32_t* num_sensors,
                             amdsmi_nic_sensor_t* sensors) -> amdsmi_status_t {
  const auto available = info.count;
  if (sensors == nullptr) {
    *num_sensors = available;
    return AMDSMI_STATUS_SUCCESS;
  }

  const auto num_copied = std::min(available, *num_sensors);
  for (auto i = uint32_t{0}; i < num_copied; ++i) {
    const auto& record = info.sensors[i];
    sensors[i] = {};
    sensors[i].category = static_cast<amdsmi_metric_category_t>(record.category);
    sensors[i].name = static_cast<amdsmi_nic_sensor_name_t>(record.attribute);
    sensors[i].unit = static_cast<amdsmi_metric_unit_t>(record.unit);
    sensors[i].channel = record.channel;
    sensors[i].instance = record.instance;
    std::snprintf(sensors[i].label, sizeof(sensors[i].label), "%s", record.label);
    sensors[i].value = record.value;
  }

  const auto is_truncated = (available > *num_sensors);
  *num_sensors = available;
  return is_truncated ? AMDSMI_STATUS_INSUFFICIENT_SIZE : AMDSMI_STATUS_SUCCESS;
}

/**
 * The two-call pattern behind the statistics getters. With no array the count answers; with an
 * array only the list is read, so the device is queried once and not twice. Both callables return
 * the amdsmi status.
 */
template <typename CountFn, typename ListFn>
inline auto get_nic_stats(CountFn&& count_stats, ListFn&& list_stats, uint32_t* num_stats,
                          amdsmi_nic_stat_t* stats) -> amdsmi_status_t {
  if (stats == nullptr) {
    auto available = uint32_t{0};
    const auto count_status = count_stats(&available);
    if (count_status != AMDSMI_STATUS_SUCCESS) {
      return count_status;
    }
    *num_stats = available;
    return AMDSMI_STATUS_SUCCESS;
  }

  auto info = std::make_unique<smi_nic_stat_info_t>();
  const auto list_status = list_stats(info.get());
  if (list_status != AMDSMI_STATUS_SUCCESS) {
    return list_status;
  }
  return copy_nic_stats(*info, num_stats, stats);
}

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_AMD_SMI_NIC_STATS_H_
