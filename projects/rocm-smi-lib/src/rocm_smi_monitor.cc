/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017-2023, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <dirent.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>  // NOLINT
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi_monitor.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace amd {
namespace smi {

struct MonitorNameEntry {
    MonitorTypes type;
    const char *name;
};


static const char *kMonTempFName = "temp#_input";
static const char *kMonFanSpeedFName = "pwm#";
static const char *kMonMaxFanSpeedFName = "pwm#_max";
static const char *kMonFanRPMsName = "fan#_input";
static const char *kMonFanControlEnableName = "pwm#_enable";
static const char *kMonNameFName = "name";
static const char *kMonPowerCapDefaultName = "power#_cap_default";
static const char *kMonPowerCapName = "power#_cap";
static const char *kMonPowerCapMaxName = "power#_cap_max";
static const char *kMonPowerCapMinName = "power#_cap_min";
static const char *kMonPowerAveName = "power#_average";
static const char *kMonPowerInputName = "power#_input";
static const char *kMonPowerLabelName = "power#_label";
static const char *kMonTempMaxName = "temp#_max";
static const char *kMonTempMinName = "temp#_min";
static const char *kMonTempMaxHystName = "temp#_max_hyst";
static const char *kMonTempMinHystName = "temp#_min_hyst";
static const char *kMonTempCriticalName = "temp#_crit";
static const char *kMonTempCriticalHystName = "temp#_crit_hyst";
static const char *kMonTempEmergencyName = "temp#_emergency";
static const char *kMonTempEmergencyHystName = "temp#_emergency_hyst";
static const char *kMonTempCritMinName = "temp#_lcrit";
static const char *kMonTempCritMinHystName = "temp#_lcrit_hyst";
static const char *kMonTempOffsetName = "temp#_offset";
static const char *kMonTempLowestName = "temp#_lowest";
static const char *kMonTempHighestName = "temp#_highest";
static const char *kMonTempLabelName = "temp#_label";
static const char *kMonVoltFName = "in#_input";
static const char *kMonVoltMinName = "in#_min";
static const char *kMonVoltMinCritName = "in#_lcrit";
static const char *kMonVoltMaxName = "in#_max";
static const char *kMonVoltMaxCritName = "in#_crit";
static const char *kMonVoltAverageName = "in#_average";
static const char *kMonVoltLowestName = "in#_lowest";
static const char *kMonVoltHighestName = "in#_highest";
static const char *kMonVoltLabelName = "in#_label";

static const char *kTempSensorTypeMemoryName = "mem";
static const char *kTempSensorTypeJunctionName = "junction";
static const char *kTempSensorTypeEdgeName = "edge";

static const char *kTempSensorTypeVddgfxName = "vddgfx";


static const std::map<std::string, rsmi_temperature_type_t>
                                                        kTempSensorNameMap = {
    {kTempSensorTypeMemoryName, RSMI_TEMP_TYPE_MEMORY},
    {kTempSensorTypeJunctionName, RSMI_TEMP_TYPE_JUNCTION},
    {kTempSensorTypeEdgeName, RSMI_TEMP_TYPE_EDGE},
};

static const std::map<std::string, rsmi_voltage_type_t>
                                                        kVoltSensorNameMap = {
    {kTempSensorTypeVddgfxName, RSMI_VOLT_TYPE_VDDGFX},
};

static const std::map<MonitorTypes, const char *> kMonitorNameMap = {
    {kMonName, kMonNameFName},
    {kMonTemp, kMonTempFName},
    {kMonFanSpeed, kMonFanSpeedFName},
    {kMonFanCntrlEnable, kMonFanControlEnableName},
    {kMonMaxFanSpeed, kMonMaxFanSpeedFName},
    {kMonFanRPMs, kMonFanRPMsName},
    {kMonPowerCap, kMonPowerCapName},
    {kMonPowerCapDefault, kMonPowerCapDefaultName},
    {kMonPowerCapMax, kMonPowerCapMaxName},
    {kMonPowerCapMin, kMonPowerCapMinName},
    {kMonPowerAve, kMonPowerAveName},
    {kMonPowerInput, kMonPowerInputName},
    {kMonPowerLabel, kMonPowerLabelName},
    {kMonTempMax, kMonTempMaxName},
    {kMonTempMin, kMonTempMinName},
    {kMonTempMaxHyst, kMonTempMaxHystName},
    {kMonTempMinHyst, kMonTempMinHystName},
    {kMonTempCritical, kMonTempCriticalName},
    {kMonTempCriticalHyst, kMonTempCriticalHystName},
    {kMonTempEmergency, kMonTempEmergencyName},
    {kMonTempEmergencyHyst, kMonTempEmergencyHystName},
    {kMonTempCritMin, kMonTempCritMinName},
    {kMonTempCritMinHyst, kMonTempCritMinHystName},
    {kMonTempOffset, kMonTempOffsetName},
    {kMonTempLowest, kMonTempLowestName},
    {kMonTempHighest, kMonTempHighestName},
    {kMonTempLabel, kMonTempLabelName},
    {kMonVolt, kMonVoltFName},
    {kMonVoltMin, kMonVoltMinName},
    {kMonVoltMinCrit, kMonVoltMinCritName},
    {kMonVoltMax, kMonVoltMaxName},
    {kMonVoltMaxCrit, kMonVoltMaxCritName},
    {kMonVoltAverage, kMonVoltAverageName},
    {kMonVoltLowest, kMonVoltLowestName},
    {kMonVoltHighest, kMonVoltHighestName},
    {kMonVoltLabel, kMonVoltLabelName},
};

static  std::map<MonitorTypes, uint64_t> kMonInfoVarTypeToRSMIVariant = {
    // rsmi_temperature_metric_t
    {kMonTemp, RSMI_TEMP_CURRENT},
    {kMonTempMax, RSMI_TEMP_MAX},
    {kMonTempMin, RSMI_TEMP_MIN},
    {kMonTempMaxHyst, RSMI_TEMP_MAX_HYST},
    {kMonTempMinHyst, RSMI_TEMP_MIN_HYST},
    {kMonTempCritical, RSMI_TEMP_CRITICAL},
    {kMonTempCriticalHyst, RSMI_TEMP_CRITICAL_HYST},
    {kMonTempEmergency, RSMI_TEMP_EMERGENCY},
    {kMonTempEmergencyHyst, RSMI_TEMP_EMERGENCY_HYST},
    {kMonTempCritMin, RSMI_TEMP_CRIT_MIN},
    {kMonTempCritMinHyst, RSMI_TEMP_CRIT_MIN_HYST},
    {kMonTempOffset, RSMI_TEMP_OFFSET},
    {kMonTempLowest, RSMI_TEMP_LOWEST},
    {kMonTempHighest, RSMI_TEMP_HIGHEST},
    {kMonInvalid, RSMI_DEFAULT_VARIANT},
    // rsmi_voltage_metric_t
    {kMonVolt, RSMI_VOLT_CURRENT},
    {kMonVoltMin, RSMI_VOLT_MIN},
    {kMonVoltMinCrit, RSMI_VOLT_MIN_CRIT},
    {kMonVoltMax, RSMI_VOLT_MAX},
    {kMonVoltMaxCrit, RSMI_VOLT_MAX_CRIT},
    {kMonVoltAverage, RSMI_VOLT_AVERAGE},
    {kMonVoltLowest, RSMI_VOLT_LOWEST},
    {kMonVoltHighest, RSMI_VOLT_HIGHEST},
};

typedef struct {
    std::vector<const char *> mandatory_depends;
    std::vector<MonitorTypes> variants;
} monitor_depends_t;

static const std::map<const char *, monitor_depends_t> kMonFuncDependsMap = {
  {"rsmi_dev_power_ave_get",        { .mandatory_depends = {kMonPowerAveName},
                                      .variants = {kMonInvalid},
                                    }
  },
  {"rsmi_dev_power_cap_get",        { .mandatory_depends = {kMonPowerCapName},
                                      .variants = {kMonInvalid},
                                    }
  },
  {"rsmi_dev_power_cap_default_get", { .mandatory_depends =
                                      {kMonPowerCapDefaultName},
                                      .variants = {kMonInvalid},
                                    }
  },
  {"rsmi_dev_power_cap_range_get",  { .mandatory_depends =
                                                        {kMonPowerCapMaxName,
                                                         kMonPowerCapMinName},
                                      .variants = {kMonInvalid},
                                    }
  },
  {"rsmi_dev_power_cap_set",        { .mandatory_depends =
                                                         {kMonPowerCapMaxName,
                                                          kMonPowerCapMinName,
                                                          kMonPowerCapName},
                                      .variants = {kMonInvalid},
                                    }
  },

  {"rsmi_dev_fan_rpms_get",         { .mandatory_depends = {kMonFanRPMsName},
                                      .variants = {kMonInvalid},
                                    }
  },
  {"rsmi_dev_fan_speed_get",        { .mandatory_depends = {kMonFanSpeedFName},
                                      .variants = {kMonInvalid},
                                    }
  },
  {"rsmi_dev_fan_speed_max_get",    { .mandatory_depends =
                                                       {kMonMaxFanSpeedFName},
                                      .variants = {kMonInvalid},
                                    }
  },
  {"rsmi_dev_temp_metric_get",      { .mandatory_depends =
                                                          {kMonTempLabelName},
                                      .variants = {kMonTemp,
                                                   kMonTempMax,
                                                   kMonTempMin,
                                                   kMonTempMaxHyst,
                                                   kMonTempMinHyst,
                                                   kMonTempCritical,
                                                   kMonTempCriticalHyst,
                                                   kMonTempEmergency,
                                                   kMonTempEmergencyHyst,
                                                   kMonTempCritMin,
                                                   kMonTempCritMinHyst,
                                                   kMonTempOffset,
                                                   kMonTempLowest,
                                                   kMonTempHighest,
                                        },
                                    }
  },
  {"rsmi_dev_fan_reset",            { .mandatory_depends =
                                                   {kMonFanControlEnableName},
                                      .variants = {kMonInvalid},
                                    }
  },
  {"rsmi_dev_fan_speed_set",        { .mandatory_depends =
                                                    {kMonMaxFanSpeedFName,
                                                     kMonFanControlEnableName,
                                                     kMonFanSpeedFName},
                                      .variants = {kMonInvalid},
                                    }
  },
  {"rsmi_dev_volt_metric_get",      { .mandatory_depends =
                                                          {kMonVoltLabelName},
                                      .variants = {kMonVolt,
                                                   kMonVoltMin,
                                                   kMonVoltMinCrit,
                                                   kMonVoltMax,
                                                   kMonVoltMaxCrit,
                                                   kMonVoltAverage,
                                                   kMonVoltLowest,
                                                   kMonVoltHighest,
                                                  },
                                      }
  },
};

  Monitor::Monitor(std::string path, RocmSMI_env_vars const *e) :
                                                        path_(path), env_(e) {
#ifndef DEBUG
    env_ = nullptr;
#endif
}
Monitor::~Monitor(void) = default;

std::string
Monitor::MakeMonitorPath(MonitorTypes type, uint32_t sensor_id) {
  std::string tempPath = path_;
  std::string fn = kMonitorNameMap.at(type);

  std::replace(fn.begin(), fn.end(), '#', static_cast<char>('0' + sensor_id));

  tempPath += "/";
  tempPath += fn;

  return tempPath;
}

int Monitor::writeMonitor(MonitorTypes type, uint32_t sensor_id,
                                                            std::string val) {
  std::string sysfs_path = MakeMonitorPath(type, sensor_id);

  DBG_FILE_ERROR(sysfs_path, &val)
  return WriteSysfsStr(sysfs_path, val);
}

// This string version should work for all valid monitor types
int Monitor::readMonitor(MonitorTypes type, uint32_t sensor_id,
                                                           std::string *val) {
  std::ostringstream ss;
  assert(val != nullptr);

  std::string temp_str;
  std::string sysfs_path = MakeMonitorPath(type, sensor_id);

  DBG_FILE_ERROR(sysfs_path, (std::string *)nullptr)
  int ret = ReadSysfsStr(sysfs_path, val);
  ss << __PRETTY_FUNCTION__
     << " | Success | Read hwmon file: " << sysfs_path
     << " | Type: " << monitorTypesToString.at(type)
     << " | Sensor id: " << std::to_string(sensor_id)
     << " | Data: " << *val
     << " | Returning: " << std::to_string(ret) << " |";
  LOG_INFO(ss);
  return ret;
}

int32_t
Monitor::setTempSensorLabelMap(void) {
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);
  std::string type_str;
  int ret;

  if (!temp_type_index_map_.empty()) {
    return 0;  // We've already filled in the map
  }
  auto add_temp_sensor_entry = [&](uint32_t file_index) {
    ret = readMonitor(kMonTempLabel, file_index, &type_str);
    rsmi_temperature_type_t t_type;

    // If readMonitor fails, there is no label file for the file_index.
    // In that case, map the type to file index 0, which is not supported
    // and will fail appropriately later when we check for support.
    if (ret) {
      index_temp_type_map_.insert({file_index, RSMI_TEMP_TYPE_INVALID});
    } else {
      t_type = kTempSensorNameMap.at(type_str);
      temp_type_index_map_[t_type] = file_index;
      index_temp_type_map_.insert({file_index, t_type});
    }
    return 0;
  };

  for (uint32_t t = RSMI_TEMP_TYPE_FIRST; t <= RSMI_TEMP_TYPE_LAST; ++t) {
    temp_type_index_map_.insert(
       {static_cast<rsmi_temperature_type_t>(t), RSMI_TEMP_TYPE_INVALID});
  }
  for (uint32_t i = 1; i <= RSMI_TEMP_TYPE_LAST + 1; ++i) {
    ret = add_temp_sensor_entry(i);
    if (ret) {
      return ret;
    }
  }
  return 0;
}

int32_t
Monitor::setVoltSensorLabelMap(void) {
  std::string type_str;
  int ret;

  if (!volt_type_index_map_.empty()) {
    return 0;  // We've already filled in the map
  }
  auto add_volt_sensor_entry = [&](uint32_t file_index) {
    ret = readMonitor(kMonVoltLabel, file_index, &type_str);
    rsmi_voltage_type_t t_type;

    // If readMonitor fails, there is no label file for the file_index.
    // In that case, map the type to file index 0, which is not supported
    // and will fail appropriately later when we check for support.
    if (ret) {
      index_volt_type_map_.insert({file_index, RSMI_VOLT_TYPE_INVALID});
    } else {
      t_type = kVoltSensorNameMap.at(type_str);
      volt_type_index_map_[t_type] = file_index;
      index_volt_type_map_.insert({file_index, t_type});
    }
    return 0;
  };

  for (uint32_t i = 0; i < RSMI_VOLT_TYPE_LAST + 1; ++i) {
    ret = add_volt_sensor_entry(i);
    if (ret) {
      return ret;
    }
  }
  return 0;
}

static int get_supported_sensors(std::string dir_path, std::string fn_reg_ex,
                                              std::vector<uint64_t> *sensors) {
  auto hwmon_dir = opendir(dir_path.c_str());
  assert(hwmon_dir != nullptr);
  assert(sensors != nullptr);

  sensors->clear();

  std::string::size_type pos = fn_reg_ex.find('#');

  if (pos == std::string::npos) {
    closedir(hwmon_dir);
    return -1;
  }
  fn_reg_ex.erase(pos, 1);
  fn_reg_ex.insert(pos, "([0-9]+)");
  fn_reg_ex = "\\b" + fn_reg_ex + "\\b";

  auto dentry = readdir(hwmon_dir);
  std::smatch match;
  uint64_t mon_val;

  char *endptr;
  try {
    std::regex re(fn_reg_ex);
    std::string fn;

    while (dentry != nullptr) {
      fn = dentry->d_name;
      if (std::regex_search(fn, match, re)) {
        assert(match.size() == 2);  // 1 for whole match + 1 for sub-match
        errno = 0;
        std::string val_str(match.str(1));
        mon_val = strtoul(val_str.c_str(), &endptr, 10);
        assert(errno == 0);
        assert(*endptr == '\0');
        if (errno) {
          closedir(hwmon_dir);
          return -2;
        }
        sensors->push_back(mon_val);
      }
      dentry = readdir(hwmon_dir);
    }
    if (closedir(hwmon_dir)) {
      return errno;
    }
  } catch (std::regex_error& e) {
    std::cout << "Regular expression error:" << std::endl;
    std::cout << e.what() << std::endl;
    std::cout << "Regex error code: " << e.code() << std::endl;
    return -3;
  }
  return 0;
}

uint32_t
Monitor::getTempSensorIndex(rsmi_temperature_type_t type) {
  return temp_type_index_map_.at(type);
}

rsmi_temperature_type_t
Monitor::getTempSensorEnum(uint64_t ind) {
  return index_temp_type_map_.at(ind);
}

uint32_t
Monitor::getVoltSensorIndex(rsmi_voltage_type_t type) {
  return volt_type_index_map_.at(type);
}

rsmi_voltage_type_t
Monitor::getVoltSensorEnum(uint64_t ind) {
  return index_volt_type_map_.at(ind);
}

static std::vector<uint64_t> get_intersection(std::vector<uint64_t> *v1,
                                                   std::vector<uint64_t> *v2) {
  assert(v1 != nullptr);
  assert(v2 != nullptr);
  std::vector<uint64_t> intersect;

  std::sort(v1->begin(), v1->end());
  std::sort(v2->begin(), v2->end());

  std::set_intersection(v1->begin(), v1->end(), v2->begin(), v2->end(),
                                               std::back_inserter(intersect));
  return intersect;
}

// Use this enum to encode the monitor type into the monitor ID.
// We can later use this to convert to rsmi-api sensor types; for exampple,
// rsmi_temperature_type_t, which is what the caller will expect. Add
// new types as needed.

typedef enum {
  eDefaultMonitor = 0,
  eTempMonitor,
  eVoltMonitor,
} monitor_types;

static monitor_types getFuncType(std::string f_name) {
  monitor_types ret = eDefaultMonitor;

  if (f_name == "rsmi_dev_temp_metric_get") {
    ret = eTempMonitor;
  }
  if (f_name == "rsmi_dev_volt_metric_get") {
    ret = eVoltMonitor;
  }
  return ret;
}

void Monitor::fillSupportedFuncs(SupportedFuncMap *supported_funcs) {
  std::map<const char *, monitor_depends_t>::const_iterator it =
                                                   kMonFuncDependsMap.begin();
  std::string mon_root = path_;
  bool mand_depends_met;
  std::shared_ptr<VariantMap> supported_variants;
  std::vector<uint64_t> sensors_i;
  std::vector<uint64_t> intersect;
  int ret;
  monitor_types m_type;

  assert(supported_funcs != nullptr);

  while (it != kMonFuncDependsMap.end()) {
    // First, see if all the mandatory dependencies are there
    std::vector<const char *>::const_iterator dep =
                                         it->second.mandatory_depends.begin();

    m_type = getFuncType(it->first);
    mand_depends_met = true;

    // Initialize "intersect". A monitor is considered supported if all of its
    // dependency monitors with the same sensor index are present. So we
    // initialize "intersect" with the set of sensors that exist for the first
    // mandatory monitor, and take intersection of that with the subsequent
    // dependency monitors. The main assumption here is that
    // variant_<sensor_i>'s sensor-based dependencies have the same index i;
    // in other words, variant_i is not dependent on a sensor j, j != i

    // Initialize intersect with the available monitors for the first
    // mandatory dependency.
    ret = get_supported_sensors(mon_root + "/", *dep, &intersect);
    std::string dep_path;
    if (ret == -1) {
      // In this case, the dependency is not sensor-specific, so just
      // see if the file exists.
      dep_path = mon_root + "/" + *dep;
      if (!FileExists(dep_path.c_str())) {
        mand_depends_met = false;
      }
    } else if (ret <= -2) {
      throw amd::smi::rsmi_exception(RSMI_STATUS_INTERNAL_EXCEPTION,
                        "Failed to parse monitor file name: " + dep_path);
    }
    dep++;

    while (mand_depends_met && dep != it->second.mandatory_depends.end()) {
      ret = get_supported_sensors(mon_root + "/", *dep, &sensors_i);

      if (ret == 0) {
        intersect = get_intersection(&sensors_i, &intersect);
      } else if (ret == -1) {
        // In this case, the dependency is not sensor-specific, so just
        // see if the file exists.
        std::string dep_path = mon_root + "/" + *dep;
        if (!FileExists(dep_path.c_str())) {
          mand_depends_met = false;
          break;
        }
      } else if (ret <= -2) {
        throw amd::smi::rsmi_exception(RSMI_STATUS_INTERNAL_EXCEPTION,
                          "Failed to parse monitor file name: " + dep_path);
      }

      dep++;
    }

    if (!mand_depends_met) {
      it++;
      continue;
    }

    // "intersect" holds the set of sensors for the mandatory dependencies
    // that exist.

    std::vector<MonitorTypes>::const_iterator var =
                                                  it->second.variants.begin();
    supported_variants = std::make_shared<VariantMap>();

    std::vector<uint64_t> supported_monitors;

    for (; var != it->second.variants.end(); var++) {
      if (*var != kMonInvalid) {
        ret = get_supported_sensors(mon_root + "/",
                                        kMonitorNameMap.at(*var), &sensors_i);

        if (ret == 0) {
          supported_monitors = get_intersection(&sensors_i, &intersect);
        } else if (ret <= -2) {
          throw amd::smi::rsmi_exception(RSMI_STATUS_INTERNAL_EXCEPTION,
                            "Failed to parse monitor file name: " + dep_path);
        }
      } else {
        supported_monitors = intersect;
      }
      if (!supported_monitors.empty()) {
        for (uint64_t &supported_monitor : supported_monitors) {
          if (m_type == eDefaultMonitor) {
            assert(supported_monitor > 0);
            supported_monitor |=
                    (supported_monitor - 1) << MONITOR_TYPE_BIT_POSITION;
          } else if (m_type == eTempMonitor) {
            // Temp sensor file names are 1-based
            assert(supported_monitor > 0);
            supported_monitor |=
                 static_cast<uint64_t>(getTempSensorEnum(supported_monitor))
                                                << MONITOR_TYPE_BIT_POSITION;
          } else if (m_type == eVoltMonitor) {
            // Voltage sensor file names are 0-based
            supported_monitor |=
                 static_cast<uint64_t>(getVoltSensorEnum(supported_monitor))
                                                << MONITOR_TYPE_BIT_POSITION;
          } else {
            assert(false);  // Unexpected monitor type
          }
        }
      (*supported_variants)[kMonInfoVarTypeToRSMIVariant.at(*var)] =
                             std::make_shared<SubVariant>(supported_monitors);
      }
    }

    if (it->second.variants.empty()) {
      (*supported_funcs)[it->first] = nullptr;
      supported_variants = nullptr;  // Invoke destructor
    } else if (!(*supported_variants).empty()) {
      (*supported_funcs)[it->first] = supported_variants;
    }

    it++;
  }
}

}  // namespace smi
}  // namespace amd
