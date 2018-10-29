/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
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
#include <errno.h>

#include <sstream>
#include <algorithm>
#include <cerrno>
#include <bitset>
#include <cstdint>
#include <unordered_map>
#include <map>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_utils.h"

static const uint32_t kMaxOverdriveLevel = 20;

static rsmi_status_t handleException() {
  try {
    throw;
  } catch (const std::bad_alloc& e) {
    debug_print("RSMI exception: BadAlloc\n");
    return RSMI_STATUS_OUT_OF_RESOURCES;
  } catch (const std::exception& e) {
    debug_print("Unhandled exception: %s\n", e.what());
    assert(false && "Unhandled exception.");
    return RSMI_STATUS_INTERNAL_EXCEPTION;
  } catch (const std::nested_exception& e) {
    debug_print("Callback threw, forwarding.\n");
    e.rethrow_nested();
    return RSMI_STATUS_INTERNAL_EXCEPTION;
  } catch (...) {
    assert(false && "Unhandled exception.");
    abort();
    return RSMI_STATUS_INTERNAL_EXCEPTION;
  }
}

#define TRY try {
#define CATCH } catch (...) {return handleException();}
#define GET_DEV_FROM_INDX  \
    amd::smi::RocmSMI smi = amd::smi::RocmSMI::getInstance(); \
    if (dv_ind >= smi.monitor_devices().size()) { \
      return RSMI_STATUS_INVALID_ARGS; \
    } \
  std::shared_ptr<amd::smi::Device> dev = smi.monitor_devices()[dv_ind]; \
  assert(dev != nullptr);

static rsmi_status_t errno_to_rsmi_status(uint32_t err) {
  switch (err) {
    case 0:      return RSMI_STATUS_SUCCESS;
    case EACCES: return RSMI_STATUS_PERMISSION;
    case EPERM:  return RSMI_STATUS_NOT_SUPPORTED;
    case ENOENT: return RSMI_STATUS_FILE_ERROR;
    default:     return RSMI_STATUS_UNKNOWN_ERROR;
  }
}
/**
 * Parse a string of the form "<int index>:  <int freq><freq. unit string> <|*>"
 */
static uint32_t freq_string_to_int(std::string freq_line, bool *is_curr) {
  assert(is_curr != nullptr);

  std::istringstream fs(freq_line);

  uint32_t ind;
  uint32_t freq;
  std::string junk;
  std::string units_str;
  std::string star_str;

  fs >> ind;
  fs >> junk;  // colon
  fs >> freq;
  fs >> units_str;
  fs >> star_str;

  if (is_curr != nullptr) {
    if (freq_line.find("*") != std::string::npos) {
      *is_curr = true;
    } else {
      *is_curr = false;
    }
  }
  uint32_t multiplier = 0;

  if (units_str == "Mhz") {
    multiplier = 1000000;
  } else if (units_str == "Ghz") {
    multiplier = 1000000000;
  } else if (units_str == "Khz") {
    multiplier = 1000;
  } else if (units_str == "Hz") {
    multiplier = 1;
  } else {
    assert(!"Unexpected units for frequency");
  }

  return freq*multiplier;
}


/**
 * Parse a string of the form "<int index> <mode name string> <|*>"
 */
static rsmi_power_profile_preset_masks
power_prof_string_to_int(std::string pow_prof_line, bool *is_curr) {
  std::istringstream fs(pow_prof_line);
  uint32_t ind;
  std::string mode;
  size_t tmp;

  rsmi_power_profile_preset_masks ret = RSMI_PWR_PROF_PRST_INVALID;

  fs >> ind;
  fs >> mode;

  while (1) {
    tmp = mode.find_last_of("* :");
    if (tmp == std::string::npos) {
      break;
    }
    mode = mode.substr(0, tmp);
  }

  if (is_curr != nullptr) {
    if (pow_prof_line.find("*") != std::string::npos) {
      *is_curr = true;
    } else {
      *is_curr = false;
    }
  }

  const std::unordered_map<std::string, std::function<void()>> mode_map {
    {"3D_FULL_SCREEN",   [&](){ ret = RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK; }},
    {"POWER_SAVING",   [&](){ ret = RSMI_PWR_PROF_PRST_POWER_SAVING_MASK; }},
    {"VIDEO", [&](){ ret = RSMI_PWR_PROF_PRST_VIDEO_MASK; }},
    {"VR", [&](){ ret = RSMI_PWR_PROF_PRST_VR_MASK; }},
    {"COMPUTE", [&](){ ret = RSMI_PWR_PROF_PRST_COMPUTE_MASK; }},
    {"CUSTOM", [&](){ ret = RSMI_PWR_PROF_PRST_CUSTOM_MASK; }},
  };
  auto mode_iter = mode_map.find(mode);

  if (mode_iter != mode_map.end()) {
    mode_iter->second();
  }
  return ret;
}

static rsmi_status_t get_dev_value_str(amd::smi::DevInfoTypes type,
                                      uint32_t dv_ind, std::string *val_str) {
  GET_DEV_FROM_INDX
  int ret = dev->readDevInfo(type, val_str);

  return errno_to_rsmi_status(ret);
}

static rsmi_status_t set_dev_value(amd::smi::DevInfoTypes type,
                                              uint32_t dv_ind, uint64_t val) {
  GET_DEV_FROM_INDX

  int ret = dev->writeDevInfo(type, val);
  return errno_to_rsmi_status(ret);
}

static rsmi_status_t get_dev_mon_value(amd::smi::MonitorTypes type,
                         uint32_t dv_ind, uint32_t sensor_ind, int64_t *val) {
  GET_DEV_FROM_INDX

  assert(dev->monitor() != nullptr);

  std::string val_str;

  int ret = dev->monitor()->readMonitor(type, sensor_ind, &val_str);
  if (ret) {
    return errno_to_rsmi_status(ret);
  }

  *val = std::stoi(val_str);

  return RSMI_STATUS_SUCCESS;
}

static rsmi_status_t get_dev_mon_value(amd::smi::MonitorTypes type,
                        uint32_t dv_ind, uint32_t sensor_ind, uint64_t *val) {
  GET_DEV_FROM_INDX

  assert(dev->monitor() != nullptr);

  std::string val_str;

  int ret = dev->monitor()->readMonitor(type, sensor_ind, &val_str);
  if (ret) {
    return errno_to_rsmi_status(ret);
  }

  *val = std::stoul(val_str);

  return RSMI_STATUS_SUCCESS;
}

template <typename T>
static rsmi_status_t set_dev_mon_value(amd::smi::MonitorTypes type,
                                 uint32_t dv_ind, int32_t sensor_ind, T val) {
  GET_DEV_FROM_INDX

  assert(dev->monitor() != nullptr);

  int ret =  dev->monitor()->writeMonitor(type, sensor_ind,
                                                         std::to_string(val));

  return errno_to_rsmi_status(ret);
}

static rsmi_status_t get_power_mon_value(amd::smi::PowerMonTypes type,
                                      uint32_t dv_ind, uint64_t *val) {
  amd::smi::RocmSMI smi = amd::smi::RocmSMI::getInstance();

  if (dv_ind >= smi.monitor_devices().size() || val == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  uint32_t ret = smi.DiscoverAMDPowerMonitors();
  if (ret == EACCES) {
    return RSMI_STATUS_PERMISSION;
  } else if (ret != 0) {
    return RSMI_STATUS_FILE_ERROR;
  }

  std::shared_ptr<amd::smi::Device> dev = smi.monitor_devices()[dv_ind];
  assert(dev != nullptr);
  assert(dev->monitor() != nullptr);

  ret = dev->power_monitor()->readPowerValue(type, val);

  return errno_to_rsmi_status(ret);
}

static rsmi_status_t get_dev_mon_value_str(amd::smi::MonitorTypes type,
                 uint32_t dv_ind, int32_t sensor_ind, std::string *val_str) {
  GET_DEV_FROM_INDX

  assert(dev->monitor() != nullptr);

  int ret = dev->monitor()->readMonitor(type, sensor_ind, val_str);
  return errno_to_rsmi_status(ret);
}


static rsmi_status_t get_dev_value_vec(amd::smi::DevInfoTypes type,
                         uint32_t dv_ind, std::vector<std::string> *val_vec) {
  GET_DEV_FROM_INDX

  int ret = dev->readDevInfo(type, val_vec);
  return errno_to_rsmi_status(ret);
}

// A call to rsmi_init is not technically necessary at this time, but may be
// in the future.
rsmi_status_t
rsmi_init(uint64_t init_flags) {
  TRY
  (void)init_flags;  // unused for now; for future use

  amd::smi::RocmSMI smi = amd::smi::RocmSMI::getInstance();
  return RSMI_STATUS_SUCCESS;
  CATCH
}

// A call to rsmi_shut_down is not technically necessary at this time,
// but may be in the future.
rsmi_status_t
rsmi_shut_down(void) {
  TRY
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_num_monitor_devices(uint32_t *num_devices) {
  TRY
  if (num_devices == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  amd::smi::RocmSMI smi = amd::smi::RocmSMI::getInstance();

  *num_devices = smi.monitor_devices().size();
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_id_get(uint32_t dv_ind, uint64_t *id) {
  TRY
  std::string val_str;
  rsmi_status_t ret = get_dev_value_str(amd::smi::kDevDevID, dv_ind, &val_str);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  errno = 0;
  *id = strtoul(val_str.c_str(), nullptr, 16);
  assert(errno == 0);

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_perf_level_get(uint32_t dv_ind, rsmi_dev_perf_level *perf) {
  TRY
  std::string val_str;
  rsmi_status_t ret = get_dev_value_str(amd::smi::kDevPerfLevel, dv_ind,
                                                                    &val_str);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  *perf = amd::smi::Device::perfLvlStrToEnum(val_str);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_overdrive_level_get(uint32_t dv_ind, uint32_t *od) {
  TRY
  std::string val_str;
  rsmi_status_t ret = get_dev_value_str(amd::smi::kDevOverDriveLevel, dv_ind,
                                                                    &val_str);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  errno = 0;
  *od = strtoul(val_str.c_str(), nullptr, 10);
  assert(errno == 0);

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_overdrive_level_set(int32_t dv_ind, uint32_t od) {
  TRY
  if (od > kMaxOverdriveLevel) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  return set_dev_value(amd::smi::kDevOverDriveLevel, dv_ind, od);
  CATCH
}

rsmi_status_t
rsmi_dev_perf_level_set(int32_t dv_ind, rsmi_dev_perf_level perf_level) {
  TRY
  if (perf_level > RSMI_DEV_PERF_LEVEL_LAST) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  return set_dev_value(amd::smi::kDevPerfLevel, dv_ind, perf_level);
  CATCH
}

static rsmi_status_t get_frequencies(amd::smi::DevInfoTypes type,
                                       uint32_t dv_ind, rsmi_frequencies *f) {
  TRY
  std::vector<std::string> val_vec;
  rsmi_status_t ret;

  if (f == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  ret = get_dev_value_vec(type, dv_ind, &val_vec);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }
  assert(val_vec.size() <= RSMI_MAX_NUM_FREQUENCIES);

  f->num_supported = val_vec.size();
  bool current = false;
  f->current = RSMI_MAX_NUM_FREQUENCIES + 1;  // init to an invalid value

  for (uint32_t i = 0; i < f->num_supported; ++i) {
    f->frequency[i] = freq_string_to_int(val_vec[i], &current);

    // Our assumption is that frequencies are read in from lowest to highest.
    // Check that that is true.
    if (i > 0) {
      assert(f->frequency[i-1] <= f->frequency[i]);
    }
    if (current) {
      // Should only be 1 current frequency
      assert(f->current == RSMI_MAX_NUM_FREQUENCIES + 1);
      f->current = i;
    }
  }

  assert(f->current < f->num_supported);
  return RSMI_STATUS_SUCCESS;
  CATCH
}

static rsmi_status_t get_power_profiles(uint32_t dv_ind,
                                        rsmi_power_profile_status *p,
               std::map<rsmi_power_profile_preset_masks, uint32_t> *ind_map) {
  TRY
  std::vector<std::string> val_vec;
  rsmi_status_t ret;

  if (p == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  ret = get_dev_value_vec(amd::smi::kDevPowerProfileMode, dv_ind, &val_vec);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }
  assert(val_vec.size() <= RSMI_MAX_NUM_POWER_PROFILES);

  p->num_profiles = val_vec.size() - 1;  // -1 for the header line
  bool current = false;
  p->current = RSMI_PWR_PROF_PRST_INVALID;  // init to an invalid value
  p->available_profiles = 0;

  rsmi_power_profile_preset_masks prof;

  for (uint32_t i = 1; i < val_vec.size(); ++i) {
    prof = power_prof_string_to_int(val_vec[i], &current);

    if (prof == RSMI_PWR_PROF_PRST_INVALID) {
      return RSMI_STATUS_NOT_SUPPORTED;
    }

    if (ind_map != nullptr) {
      (*ind_map)[prof] = i-1;
    }

    p->available_profiles |= prof;
    if (current) {
      // Should only be 1 current profile
      assert(p->current == RSMI_PWR_PROF_PRST_INVALID);
      p->current = prof;
    }
  }

  assert(p->current != RSMI_PWR_PROF_PRST_INVALID);
  return RSMI_STATUS_SUCCESS;
  CATCH
}

static bool is_power_of_2(uint64_t n) {
      return n && !(n & (n - 1));
}
static rsmi_status_t set_power_profile(uint32_t dv_ind,
                                    rsmi_power_profile_preset_masks profile) {
  TRY

  rsmi_status_t ret;
  rsmi_power_profile_status avail_profiles = {0, RSMI_PWR_PROF_PRST_INVALID, 0};

  // Determine if the provided profile is valid
  if (!is_power_of_2(profile)) {
    return RSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }

  std::map<rsmi_power_profile_preset_masks, uint32_t> ind_map;
  ret = get_power_profiles(dv_ind, &avail_profiles, &ind_map);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  if (!(profile & avail_profiles.available_profiles)) {
    return RSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }
  assert(ind_map.find(profile) != ind_map.end());

  // Set perf. level to manual so that we can then set the power profile
  ret = rsmi_dev_perf_level_set(dv_ind, RSMI_DEV_PERF_LEVEL_MANUAL);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  // Write the new profile
  ret = set_dev_value(amd::smi::kDevPowerProfileMode, dv_ind,
                                                            ind_map[profile]);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_gpu_clk_freq_get(uint32_t dv_ind, rsmi_clk_type clk_type,
                                                        rsmi_frequencies *f) {
  TRY
  switch (clk_type) {
    case RSMI_CLK_TYPE_SYS:
      return get_frequencies(amd::smi::kDevGPUSClk, dv_ind, f);
      break;
    case RSMI_CLK_TYPE_MEM:
      return get_frequencies(amd::smi::kDevGPUMClk, dv_ind, f);
      break;
    default:
      return RSMI_STATUS_INVALID_ARGS;
  }
  CATCH
}

static std::string bitfield_to_freq_string(uint64_t bitf,
                                                     uint32_t num_supported) {
  std::string bf_str("");
  std::bitset<RSMI_MAX_NUM_FREQUENCIES> bs(bitf);

  for (uint32_t i = 0; i < num_supported; ++i) {
    if (bs[i]) {
      bf_str += std::to_string(i);
      bf_str += " ";
    }
  }
  return bf_str;
}

rsmi_status_t
rsmi_dev_gpu_clk_freq_set(uint32_t dv_ind,
                              rsmi_clk_type clk_type, uint64_t freq_bitmask) {
  rsmi_status_t ret;
  rsmi_frequencies freqs;

  TRY
  ret = rsmi_dev_gpu_clk_freq_get(dv_ind, clk_type, &freqs);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  assert(freqs.num_supported <= RSMI_MAX_NUM_FREQUENCIES);

  amd::smi::RocmSMI smi = amd::smi::RocmSMI::getInstance();

  // Above call to rsmi_dev_get_gpu_clk_freq should have emitted an error if
  // assert below is not true
  assert(dv_ind < smi.monitor_devices().size());

  std::string freq_enable_str =
                   bitfield_to_freq_string(freq_bitmask, freqs.num_supported);


  std::shared_ptr<amd::smi::Device> dev = smi.monitor_devices()[dv_ind];
  assert(dev != nullptr);

  ret = rsmi_dev_perf_level_set(dv_ind, RSMI_DEV_PERF_LEVEL_MANUAL);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  int ret_i;
  switch (clk_type) {
    case RSMI_CLK_TYPE_SYS:
      ret_i = dev->writeDevInfo(amd::smi::kDevGPUSClk, freq_enable_str);
      return errno_to_rsmi_status(ret_i);
      break;

    case RSMI_CLK_TYPE_MEM:
      ret_i = dev->writeDevInfo(amd::smi::kDevGPUMClk, freq_enable_str);
      return errno_to_rsmi_status(ret_i);
      break;

    default:
      return RSMI_STATUS_INVALID_ARGS;
  }

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_name_get(uint32_t dv_ind, char *name, size_t len) {
  TRY
  if (name == nullptr || len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::string val_str;
  rsmi_status_t ret;

  ret = get_dev_mon_value_str(amd::smi::kMonName, dv_ind, -1, &val_str);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  size_t ln = val_str.copy(name, len);

  name[std::min(len - 1, ln)] = '\0';
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_temp_metric_get(uint32_t dv_ind, uint32_t sensor_ind,
                       rsmi_temperature_metric metric, int64_t *temperature) {
  TRY

  if (temperature == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  rsmi_status_t ret;
  amd::smi::MonitorTypes mon_type;


  // Make any adjustments to sensor_ind here, if index is not a 0 based. For
  // rocm_smi we are using a 0-based index. However, most of the Linux sysfs
  // monitor files are 1-based, so we will increment by 1 and make adjustments
  // for exceptions later.
  // See https://www.kernel.org/doc/Documentation/hwmon/sysfs-interface
  ++sensor_ind;

  switch (metric) {
    case RSMI_TEMP_CURRENT:
      mon_type = amd::smi::kMonTemp;
      break;
    case RSMI_TEMP_MAX:
      mon_type = amd::smi::kMonTempMax;
      break;
    case RSMI_TEMP_MIN:
      mon_type = amd::smi::kMonTempMin;
      break;
    case RSMI_TEMP_MAX_HYST:
      mon_type = amd::smi::kMonTempMaxHyst;
      break;
    case RSMI_TEMP_MIN_HYST:
      mon_type = amd::smi::kMonTempMinHyst;
      break;
    case RSMI_TEMP_CRITICAL:
      mon_type = amd::smi::kMonTempCritical;
      break;
    case RSMI_TEMP_CRITICAL_HYST:
      mon_type = amd::smi::kMonTempCriticalHyst;
      break;
    case RSMI_TEMP_EMERGENCY:
      mon_type = amd::smi::kMonTempEmergency;
      break;
    case RSMI_TEMP_EMERGENCY_HYST:
      mon_type = amd::smi::kMonTempEmergencyHyst;
      break;
    case RSMI_TEMP_CRIT_MIN:
      mon_type = amd::smi::kMonTempCritMin;
      break;
    case RSMI_TEMP_CRIT_MIN_HYST:
      mon_type = amd::smi::kMonTempCritMinHyst;
      break;
    case RSMI_TEMP_OFFSET:
      mon_type = amd::smi::kMonTempOffset;
      break;
    case RSMI_TEMP_LOWEST:
      mon_type = amd::smi::kMonTempLowest;
      break;
    case RSMI_TEMP_HIGHEST:
      mon_type = amd::smi::kMonTempHighest;
      break;
    default:
      mon_type = amd::smi::kMonInvalid;
  }

  ret = get_dev_mon_value(mon_type, dv_ind, sensor_ind, temperature);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_fan_speed_get(uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed) {
  TRY

  if (speed == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  rsmi_status_t ret;

  ++sensor_ind;  // fan sysfs files have 1-based indices

  ret = get_dev_mon_value(amd::smi::kMonFanSpeed, dv_ind, sensor_ind, speed);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_fan_rpms_get(uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed) {
  TRY

  if (speed == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  ++sensor_ind;  // fan sysfs files have 1-based indices

  rsmi_status_t ret;

  ret = get_dev_mon_value(amd::smi::kMonFanRPMs, dv_ind, sensor_ind, speed);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_fan_reset(uint32_t dv_ind, uint32_t sensor_ind) {
  TRY

  rsmi_status_t ret;

  ++sensor_ind;  // fan sysfs files have 1-based indices

  ret = set_dev_mon_value<uint64_t>(amd::smi::kMonFanCntrlEnable,
                                                       dv_ind, sensor_ind, 2);

  return ret;

  CATCH
}

rsmi_status_t
rsmi_dev_fan_speed_set(uint32_t dv_ind, uint32_t sensor_ind, uint64_t speed) {
  TRY

  rsmi_status_t ret;
  uint64_t max_speed;


  ret = rsmi_dev_fan_speed_max_get(dv_ind, sensor_ind, &max_speed);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  if (speed > max_speed) {
    return RSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }

  ++sensor_ind;  // fan sysfs files have 1-based indices

  // First need to set fan mode (pwm1_enable) to 1 (aka, "manual")
  ret = set_dev_mon_value<uint64_t>(amd::smi::kMonFanCntrlEnable, dv_ind,
                                                               sensor_ind, 1);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  ret = set_dev_mon_value<uint64_t>(amd::smi::kMonFanSpeed, dv_ind,
                                                           sensor_ind, speed);

  return ret;

  CATCH
}

rsmi_status_t
rsmi_dev_fan_speed_max_get(uint32_t dv_ind, uint32_t sensor_ind,
                                                        uint64_t *max_speed) {
  TRY

  if (max_speed == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  ++sensor_ind;  // fan sysfs files have 1-based indices

  rsmi_status_t ret;

  ret = get_dev_mon_value(amd::smi::kMonMaxFanSpeed, dv_ind, sensor_ind,
                                      reinterpret_cast<int64_t *>(max_speed));

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_max_get(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power) {
  TRY

  if (power == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  (void)sensor_ind;  // Not used yet
  // ++sensor_ind;  // power sysfs files have 1-based indices

  rsmi_status_t ret;
  ret = get_power_mon_value(amd::smi::kPowerMaxGPUPower, dv_ind, power);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_ave_get(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power) {
  TRY

  if (power == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  ++sensor_ind;  // power sysfs files have 1-based indices

  rsmi_status_t ret;
  ret = get_dev_mon_value(amd::smi::kMonPowerAve, dv_ind, sensor_ind, power);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_cap_get(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *cap) {
  TRY

  if (cap == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  ++sensor_ind;  // power sysfs files have 1-based indices

  rsmi_status_t ret;
  ret = get_dev_mon_value(amd::smi::kMonPowerCap, dv_ind, sensor_ind, cap);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_cap_range_get(uint32_t dv_ind, uint32_t sensor_ind,
                                               uint64_t *max, uint64_t *min) {
  TRY

  if (max == nullptr || min == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  ++sensor_ind;  // power sysfs files have 1-based indices

  rsmi_status_t ret;
  ret = get_dev_mon_value(amd::smi::kMonPowerCapMax, dv_ind, sensor_ind, max);

  if (ret == RSMI_STATUS_SUCCESS) {
    ret = get_dev_mon_value(amd::smi::kMonPowerCapMin, dv_ind,
                                                             sensor_ind, min);
  }

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_cap_set(uint32_t dv_ind, uint32_t sensor_ind, uint64_t cap) {
  TRY
  rsmi_status_t ret;
  uint64_t min, max;


  ret = rsmi_dev_power_cap_range_get(dv_ind, sensor_ind, &max, &min);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  // All rsmi_* calls that use sensor_ind should use the 0-based value,
  // so increment this after the call above.
  ++sensor_ind;  // power sysfs files have 1-based indices

  if (cap > max || cap < min) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  ret = set_dev_mon_value<uint64_t>(amd::smi::kMonPowerCap, dv_ind,
                                                             sensor_ind, cap);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_profile_presets_get(uint32_t dv_ind, uint32_t sensor_ind,
                                           rsmi_power_profile_status *status) {
  TRY

  ++sensor_ind;  // power sysfs files have 1-based indices

  rsmi_status_t ret = get_power_profiles(dv_ind, status, nullptr);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_profile_set(uint32_t dv_ind, uint32_t sensor_ind,
                                     rsmi_power_profile_preset_masks profile) {
  TRY
  ++sensor_ind;  // power sysfs files have 1-based indices

  rsmi_status_t ret = set_power_profile(dv_ind, profile);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_status_string(rsmi_status_t status, const char **status_string) {
  TRY
  if (status_string == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  const size_t status_u = static_cast<size_t>(status);
  switch (status_u) {
    case RSMI_STATUS_SUCCESS:
      *status_string = "RSMI_STATUS_SUCCESS: The function has been executed"
                          " successfully.";
      break;

    case RSMI_STATUS_INVALID_ARGS:
      *status_string =
          "RSMI_STATUS_INVALID_ARGS: The provided arguments do not"
                " meet the preconditions required for calling this function.";
      break;

    case RSMI_STATUS_NOT_SUPPORTED:
      *status_string = "RSMI_STATUS_NOT_SUPPORTED: This function is not"
                                     " supported in the current environment.";
      break;

    case RSMI_STATUS_FILE_ERROR:
      *status_string =
        "RSMI_STATUS_FILE_ERROR: There was an error in finding or"
        " opening a file or directory. The operation may not be supported by "
        "this Linux kernel version.";
      break;

    case RSMI_STATUS_PERMISSION:
      *status_string = "RSMI_STATUS_PERMISSION: The user ID of the calling"
        " process does not have sufficient permission to execute a command."
        "  Often this is fixed by running as root (sudo).";
      break;

    case RSMI_STATUS_OUT_OF_RESOURCES:
      *status_string = "Unable to acquire memory or other resource";
      break;

    case RSMI_STATUS_INTERNAL_EXCEPTION:
      *status_string = "An internal exception was caught";
      break;

    case RSMI_STATUS_INPUT_OUT_OF_BOUNDS:
      *status_string = "The provided input is out of allowable or safe range";
      break;

    default:
      *status_string = "An unknown error occurred";
      return RSMI_STATUS_UNKNOWN_ERROR;
  }
  return RSMI_STATUS_SUCCESS;
  CATCH
}
