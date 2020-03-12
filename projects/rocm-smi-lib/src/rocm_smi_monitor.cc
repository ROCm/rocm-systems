/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
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

#include <assert.h>
#include <dirent.h>

#include <fstream>
#include <string>
#include <cstdint>
#include <map>
#include <iostream>
#include <algorithm>
#include <regex>  // NOLINT
#include <vector>

#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_monitor.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_exception.h"

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
static const char *kMonPowerCapName = "power#_cap";
static const char *kMonPowerCapMaxName = "power#_cap_max";
static const char *kMonPowerCapMinName = "power#_cap_min";
static const char *kMonPowerAveName = "power#_average";
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

static const char *kTempSensorTypeMemoryName = "mem";
static const char *kTempSensorTypeJunctionName = "junction";
static const char *kTempSensorTypeEdgeName = "edge";

static const std::map<std::string, rsmi_temperature_type_t>
                                                        kTempSensorNameMap = {
    {kTempSensorTypeMemoryName, RSMI_TEMP_TYPE_MEMORY},
    {kTempSensorTypeJunctionName, RSMI_TEMP_TYPE_JUNCTION},
    {kTempSensorTypeEdgeName, RSMI_TEMP_TYPE_EDGE},
};

static const std::map<MonitorTypes, const char *> kMonitorNameMap = {
    {kMonName, kMonNameFName},
    {kMonTemp, kMonTempFName},
    {kMonFanSpeed, kMonFanSpeedFName},
    {kMonFanCntrlEnable, kMonFanControlEnableName},
    {kMonMaxFanSpeed, kMonMaxFanSpeedFName},
    {kMonFanRPMs, kMonFanRPMsName},
    {kMonPowerCap, kMonPowerCapName},
    {kMonPowerCapMax, kMonPowerCapMaxName},
    {kMonPowerCapMin, kMonPowerCapMinName},
    {kMonPowerAve, kMonPowerAveName},
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
};

  Monitor::Monitor(std::string path, RocmSMI_env_vars const *e) :
                                                        path_(path), env_(e) {
#ifdef NDEBUG
    env_ = nullptr;
#endif
}
Monitor::~Monitor(void) {
}

std::string
Monitor::MakeMonitorPath(MonitorTypes type, int32_t sensor_id) {
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
  assert(val != nullptr);

  std::string temp_str;
  std::string sysfs_path = MakeMonitorPath(type, sensor_id);

  DBG_FILE_ERROR(sysfs_path, (std::string *)nullptr)
  return ReadSysfsStr(sysfs_path, val);
}

uint32_t
Monitor::setSensorLabelMap(void) {
  std::string type_str;
  int ret;

  if (temp_type_index_map_.size() > 0) {
    return 0;  // We've already filled in the map
  }
  auto add_temp_sensor_entry = [&](uint32_t file_index) {
    ret = readMonitor(kMonTempLabel, file_index, &type_str);
    if (ret) {
      return ret;
    }

    rsmi_temperature_type_t t_type = kTempSensorNameMap.at(type_str);
    temp_type_index_map_.insert({t_type, file_index});
    return 0;
  };

  for (uint32_t i = 1; i <= 3; ++i) {
    ret = add_temp_sensor_entry(i);
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
  int64_t mon_val;

  char *endptr;
  std::regex re(fn_reg_ex);
  std::string fn;

  while (dentry != nullptr) {
    fn = dentry->d_name;
    if (std::regex_search(fn, match, re)) {
      assert(match.size() == 2);  // 1 for whole match + 1 for sub-match
      errno = 0;
      mon_val = strtol(match.str(1).c_str(), &endptr, 10);
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
  return 0;
}

uint32_t
Monitor::getSensorIndex(rsmi_temperature_type_t type) {
  return temp_type_index_map_.at(type);
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

void Monitor::fillSupportedFuncs(SupportedFuncMap *supported_funcs) {
  std::map<const char *, monitor_depends_t>::const_iterator it =
                                                   kMonFuncDependsMap.begin();
  std::string mon_root = path_;
  bool mand_depends_met;
  std::shared_ptr<VariantMap> supported_variants;
  std::vector<uint64_t> sensors_i;
  std::vector<uint64_t> intersect;
  int ret;

  assert(supported_funcs != nullptr);

  while (it != kMonFuncDependsMap.end()) {
    // First, see if all the mandatory dependencies are there
    std::vector<const char *>::const_iterator dep =
                                         it->second.mandatory_depends.begin();

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
    } else if (ret == -2) {
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
      } else if (ret == -2) {
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
        }
      } else {
        supported_monitors = intersect;
      }
      if (supported_monitors.size() > 0) {
        (*supported_variants)[kMonInfoVarTypeToRSMIVariant.at(*var)] =
                             std::make_shared<SubVariant>(supported_monitors);
      }
    }

    if (it->second.variants.size() == 0) {
      (*supported_funcs)[it->first] = nullptr;
      supported_variants = nullptr;  // Invoke destructor
    } else if ((*supported_variants).size() > 0) {
      (*supported_funcs)[it->first] = supported_variants;
    }

    it++;
  }
}

}  // namespace smi
}  // namespace amd
