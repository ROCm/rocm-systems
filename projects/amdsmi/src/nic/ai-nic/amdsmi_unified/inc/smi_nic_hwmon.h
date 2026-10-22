// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDSMI_UNIFIED_NIC_HWMON_H_
#define AMDSMI_UNIFIED_NIC_HWMON_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace amd::smi::nic::hwmon {

/**
 * The numeric values equal the public amdsmi_metric_category_t, amdsmi_metric_unit_t and
 * amdsmi_nic_sensor_name_t enumerators; amd_smi.cc static_asserts each match, so this header
 * does not need amdsmi.h.
 */
enum class Category_t : uint32_t { Temperature = 3, Power = 4, Voltage = 14 };
enum class Unit_t : uint32_t { MilliVolt = 5, MilliDegreeCelsius = 15, MicroWatt = 16 };
enum class Attribute_t : uint32_t {
  Input = 1,
  Min = 2,
  Max = 3,
  LCrit = 4,
  Crit = 5,
  Emergency = 6,
  Shutdown = 7,
  Cap = 8,
};

/** One readable hwmon attribute file; `value` is the raw integer in `unit`. */
struct SensorRecord_t {
  Category_t category;
  Attribute_t attribute;
  Unit_t unit;
  uint32_t channel;   // the N in tempN_, inN_, powerN_
  uint32_t instance;  // the N in hwmon<N>
  std::string label;  // contents of <kind>N_label, empty when the driver gives none
  int64_t value;
};

/** Which temperature channel find_temperature_input() looks for. */
enum class TempRole_t : uint8_t { Asic, Board };

/**
 * Every recognised attribute of every hwmon<N> directory under `hwmon_root`, ordered by
 * instance, category, channel, attribute. Empty when the directory is absent or unreadable.
 */
auto scan_sensors(const std::string& hwmon_root) -> std::vector<SensorRecord_t>;

/**
 * Path of the tempN_input that plays `role`. Asic is the channel labelled "die", or temp1 for a
 * driver that labels none of its channels; Board is the channel labelled "board". Returns
 * nullopt when no channel matches.
 */
auto find_temperature_input(const std::string& hwmon_root, TempRole_t role)
    -> std::optional<std::string>;

}  // namespace amd::smi::nic::hwmon

#endif  // AMDSMI_UNIFIED_NIC_HWMON_H_
