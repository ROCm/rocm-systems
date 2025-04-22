/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <cstddef>
#include <string>
#include <sys/ioctl.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <inttypes.h>

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "rocm_smi/rocm_smi_common.h"  // Should go before rocm_smi.h
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_gpu_metrics.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_counters.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi_io_link.h"
#include "rocm_smi/rocm_smi64Config.h"
#include "rocm_smi/rocm_smi_logger.h"

using amd::smi::monitorTypesToString;
using amd::smi::getRSMIStatusString;
using amd::smi::AMDGpuMetricsUnitType_t;
using amd::smi::AMDGpuMetricTypeId_t;
auto &devInfoTypesStrings = amd::smi::Device::devInfoTypesStrings;

static const uint32_t kMaxOverdriveLevel = 20;
static const float kEnergyCounterResolution = 15.3F;

static const std::map<rsmi_clk_type_t, std::string> kClkStateMap = {
  { RSMI_CLK_TYPE_SYS, "SCLK" },
  { RSMI_CLK_TYPE_DF, "DFCLK" },
  { RSMI_CLK_TYPE_DCEF, "DCEFCLK" },
  { RSMI_CLK_TYPE_SOC, "SOCCLK" },
  { RSMI_CLK_TYPE_MEM, "MCLK" },
  { RSMI_CLK_TYPE_PCIE, "PCIECLK" },
};

static const std::map<rsmi_clk_type_t, amd::smi::DevInfoTypes> kClkTypeMap = {
  { RSMI_CLK_TYPE_SYS, amd::smi::kDevGPUSClk },
  { RSMI_CLK_TYPE_MEM, amd::smi::kDevGPUMClk },
  { RSMI_CLK_TYPE_DF, amd::smi::kDevFClk },
  { RSMI_CLK_TYPE_DCEF, amd::smi::kDevDCEFClk },
  { RSMI_CLK_TYPE_SOC, amd::smi::kDevSOCClk },
};

#define TRY try {
#define CATCH } catch (...) {return amd::smi::handleException();}

// declare pm metrics and register table function
namespace amd::smi {
int present_pmmetrics(const char* sysfs_file_name,
        rsmi_name_value_t **kv, uint32_t *kvnum);
int present_reg_state(const char* fname, rsmi_reg_type_t reg_type,
        rsmi_name_value_t **kv, uint32_t *kvnum);
}

static uint64_t get_multiplier_from_char(char units_char) {
  uint32_t multiplier = 0;

  switch (units_char) {
    case 'G':   // GT or GHz
      multiplier = 1000000000;
      break;

    case 'M':   // MT or MHz
      multiplier = 1000000;
      break;

    case 'K':   // KT or KHz
    case 'V':   // default unit for voltage is mV
      multiplier = 1000;
      break;

    case 'T':   // Transactions
    case 'H':   // Hertz
    case 'm':   // mV (we will make mV the default unit for voltage)
      multiplier = 1;
      break;

    default:
      assert(false);  // Unexpected units for frequency
      throw amd::smi::rsmi_exception(RSMI_STATUS_UNEXPECTED_DATA, __FUNCTION__);
  }
  return multiplier;
}

/**
 * Parse a string of the form:
 *        "<int index>:  <int freq><freq. unit string> <|*>"
 */
static uint64_t freq_string_to_int(const std::vector<std::string> &freq_lines,
                                bool *is_curr, uint32_t lanes[], uint32_t i) {
  assert(i < freq_lines.size());
  if (i >= freq_lines.size()) {
    throw amd::smi::rsmi_exception(RSMI_STATUS_INPUT_OUT_OF_BOUNDS,
                                                                __FUNCTION__);
  }

  std::istringstream fs(freq_lines[i]);

  char junk_ch;
  int ind;
  float freq;
  std::string junk_str;
  std::string units_str;
  std::string star_str;

  if (fs.peek() == 'S') {
    // Deep Sleep frequency is only supported by some GPUs
    fs >> junk_ch;
  } else {
    // All other frequency indices are numbers
    fs >> ind;
  }
  fs >> junk_str;  // colon
  fs >> freq;
  fs >> units_str;
  fs >> star_str;

  if (freq < 0) {
    throw amd::smi::rsmi_exception(RSMI_STATUS_UNEXPECTED_SIZE, __FUNCTION__);
  }

  if (is_curr != nullptr) {
    if (freq_lines[i].find('*') != std::string::npos) {
      *is_curr = true;
    } else {
      *is_curr = false;
    }
  }
  long double multiplier = get_multiplier_from_char(units_str[0]);

  if (star_str[0] == 'x') {
    assert(lanes != nullptr && "Lanes are provided but null lanes pointer");
    if (lanes) {
      if (star_str.substr(1).empty()) {
        throw amd::smi::rsmi_exception(RSMI_STATUS_NO_DATA, __FUNCTION__);
      }

      lanes[i] =
                static_cast<uint32_t>(std::stoi(star_str.substr(1), nullptr));
    }
  }
  return static_cast<uint64_t>(freq*multiplier);
}

static void od_value_pair_str_to_range(std::string in_line, rsmi_range_t *rg) {
  std::istringstream fs_rng(in_line);

  assert(rg != nullptr);
  THROW_IF_NULLPTR_DEREF(rg)

  std::string clk;
  float lo;
  float hi;
  std::string lo_units_str;
  std::string hi_units_str;

  fs_rng >> clk;  // This is clk + colon; e.g., "SCLK:"
  fs_rng >> lo;
  fs_rng >> lo_units_str;
  fs_rng >> hi;
  fs_rng >> hi_units_str;

  long double multiplier = get_multiplier_from_char(lo_units_str[0]);

  rg->lower_bound = static_cast<uint64_t>(lo*multiplier);

  multiplier = get_multiplier_from_char(hi_units_str[0]);
  rg->upper_bound = static_cast<uint64_t>(hi*multiplier);
}

/**
 * Parse a string of the form "<int index> <mode name string> <|*>"
 */
static rsmi_power_profile_preset_masks
power_prof_string_to_int(std::string pow_prof_line, bool *is_curr,
                                                          uint32_t *prof_ind) {
  std::istringstream fs(pow_prof_line);
  std::string mode;
  size_t tmp;

  THROW_IF_NULLPTR_DEREF(prof_ind)

  rsmi_power_profile_preset_masks_t ret = RSMI_PWR_PROF_PRST_INVALID;

  fs >> *prof_ind;
  fs >> mode;

  while (true) {
    tmp = mode.find_last_of("* :");
    if (tmp == std::string::npos) {
      break;
    }
    mode = mode.substr(0, tmp);
  }

  if (is_curr != nullptr) {
    if (pow_prof_line.find('*') != std::string::npos) {
      *is_curr = true;
    } else {
      *is_curr = false;
    }
  }

  const std::unordered_map<std::string, std::function<void()>> mode_map {
    {"BOOTUP_DEFAULT",   [&](){ ret = RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT; }},
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
  assert(val_str != nullptr);
  if (val_str == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  GET_DEV_FROM_INDX
  int ret = dev->readDevInfo(type, val_str);

  return amd::smi::ErrnoToRsmiStatus(ret);
}
static rsmi_status_t get_dev_value_int(amd::smi::DevInfoTypes type,
                                         uint32_t dv_ind, uint64_t *val_int) {
  assert(val_int != nullptr);
  if (val_int == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  GET_DEV_FROM_INDX
  int ret = dev->readDevInfo(type, val_int);

  return amd::smi::ErrnoToRsmiStatus(ret);
}

static rsmi_status_t get_dev_value_line(amd::smi::DevInfoTypes type,
                                      uint32_t dv_ind, std::string *val_str) {
  assert(val_str != nullptr);
  if (val_str == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  GET_DEV_FROM_INDX
  int ret = dev->readDevInfoLine(type, val_str);

  return amd::smi::ErrnoToRsmiStatus(ret);
}

static rsmi_status_t set_dev_value(amd::smi::DevInfoTypes type,
                                              uint32_t dv_ind, uint64_t val) {
  GET_DEV_FROM_INDX

  int ret = dev->writeDevInfo(type, val);
  return amd::smi::ErrnoToRsmiStatus(ret);
}

static rsmi_status_t get_dev_mon_value(amd::smi::MonitorTypes type,
                         uint32_t dv_ind, uint32_t sensor_ind, int64_t *val) {
  assert(val != nullptr);
  if (val == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  GET_DEV_FROM_INDX

  if (dev->monitor() == nullptr) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  std::string val_str;

  int ret = dev->monitor()->readMonitor(type, sensor_ind, &val_str);
  if (ret) {
    return amd::smi::ErrnoToRsmiStatus(ret);
  }

  if (!amd::smi::IsInteger(val_str)) {
    std::ostringstream ss;
    ss << "Expected integer value from monitor, but got \"" << val_str << "\"";
    LOG_ERROR(ss);
    return RSMI_STATUS_UNEXPECTED_DATA;
  }

  *val = std::stoi(val_str);

  return RSMI_STATUS_SUCCESS;
}

static rsmi_status_t get_dev_mon_value(amd::smi::MonitorTypes type,
                        uint32_t dv_ind, uint32_t sensor_ind, uint64_t *val) {
  assert(val != nullptr);
  if (val == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  GET_DEV_FROM_INDX
  if (dev->monitor() == nullptr) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  std::string val_str;

  int ret = dev->monitor()->readMonitor(type, sensor_ind, &val_str);
  if (ret) {
    return amd::smi::ErrnoToRsmiStatus(ret);
  }

  if (!amd::smi::IsInteger(val_str)) {
    std::ostringstream ss;
    ss << "Expected integer value from monitor, but got \"" << val_str << "\"";
    LOG_ERROR(ss);
    return RSMI_STATUS_UNEXPECTED_DATA;
  }

  *val = std::stoul(val_str);

  return RSMI_STATUS_SUCCESS;
}

template <typename T>
static rsmi_status_t set_dev_mon_value(amd::smi::MonitorTypes type,
                                uint32_t dv_ind, uint32_t sensor_ind, T val) {
  GET_DEV_FROM_INDX

  if (dev->monitor() == nullptr) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  int ret = dev->monitor()->writeMonitor(type, sensor_ind,
                                                         std::to_string(val));
  /// If the sysfs file doesn't exist, it is not supported.
  if (ret == ENOENT) {
    return rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED;
  }
  return amd::smi::ErrnoToRsmiStatus(ret);
}

static rsmi_status_t get_power_mon_value(amd::smi::PowerMonTypes type,
                                      uint32_t dv_ind, uint64_t *val) {
  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  if (dv_ind >= smi.devices().size() || val == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  int ret = smi.DiscoverAMDPowerMonitors();
  if (ret != 0) {
    return amd::smi::ErrnoToRsmiStatus(ret);
  }

  std::shared_ptr<amd::smi::Device> dev = smi.devices()[dv_ind];
  if (dev == nullptr || dev->monitor() == nullptr) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  ret = dev->power_monitor()->readPowerValue(type, val);

  return amd::smi::ErrnoToRsmiStatus(ret);
}

static bool is_power_of_2(uint64_t n) {
      return n && !(n & (n - 1));
}

rsmi_status_t
rsmi_init(uint64_t flags) {
  TRY
  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
  std::lock_guard<std::mutex> guard(*smi.bootstrap_mutex());

  if (smi.ref_count() == INT32_MAX) {
    return RSMI_STATUS_REFCOUNT_OVERFLOW;
  }

  (void)smi.ref_count_inc();

  // If smi.Initialize() throws, we should clean up and dec. ref_count_.
  // Otherwise, if no issues, the Dismiss() will prevent the ref_count_
  // decrement.
  MAKE_NAMED_SCOPE_GUARD(refGuard, [&]() { (void)smi.ref_count_dec(); });

  if (smi.ref_count() == 1) {
    try {
      smi.Initialize(flags);
    } catch(...) {
      smi.Cleanup();
      throw amd::smi::rsmi_exception(RSMI_STATUS_INIT_ERROR, __FUNCTION__);
    }
  }
  refGuard.Dismiss();

  return RSMI_STATUS_SUCCESS;
  CATCH
}

// A call to rsmi_shut_down is not technically necessary at this time,
// but may be in the future.
rsmi_status_t
rsmi_shut_down(void) {
  TRY

  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
  std::lock_guard<std::mutex> guard(*smi.bootstrap_mutex());

  if (smi.ref_count() == 0) {
    return RSMI_STATUS_INIT_ERROR;
  }

  // Release any device mutexes that are being held
#if DEBUG
  int ret = 0;
#endif
  for (uint32_t i = 0; i < smi.devices().size(); ++i) {
#if DEBUG
    ret = pthread_mutex_unlock(smi.devices()[i]->mutex());
    if (ret != EPERM) {  // We expect to get EPERM if the lock has already
                         // been released
      if (ret == 0) {
        std::cout << "WARNING: Unlocked monitor_devices lock; " <<
                    "it should have already been unlocked." << std::endl;
      } else {
      std::cout << "WARNING: pthread_mutex_unlock() returned " << ret <<
                   " for device " << i << " in rsmi_shut_down()" << std::endl;
      }
    }
#else
    (void)pthread_mutex_unlock(smi.devices()[i]->mutex());
#endif
  }

  (void)smi.ref_count_dec();

  if (smi.ref_count() == 0) {
    smi.Cleanup();
  }
  return RSMI_STATUS_SUCCESS;
  CATCH
}


rsmi_status_t rsmi_driver_status(rsmi_driver_state_t* state) {
  TRY
  if (state == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  // live, coming, going
  static const char *kDevInitStateID = "/sys/module/amdgpu/initstate";
  std::ifstream infile(kDevInitStateID);
  if (!infile) {
    *state = RSMI_DRIVER_NOT_FOUND;
    return RSMI_STATUS_SUCCESS;
  }

  std::string stat_str;
  infile >> stat_str;

  *state = RSMI_DRIVER_MODULE_STATE_UNKNOWN;
  if (stat_str == "live") *state = RSMI_DRIVER_MODULE_STATE_LIVE;
  if (stat_str == "coming") *state = RSMI_DRIVER_MODULE_STATE_LOADING;
  if (stat_str == "going") *state = RSMI_DRIVER_MODULE_STATE_UNLOADING;

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_num_monitor_devices(uint32_t *num_devices) {
  TRY
  assert(num_devices != nullptr);
  if (num_devices == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  *num_devices = static_cast<uint32_t>(smi.devices().size());
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t rsmi_dev_ecc_enabled_get(uint32_t dv_ind,
                                                    uint64_t *enabled_blks) {
  TRY
  rsmi_status_t ret;
  std::string feature_line;
  std::string tmp_str;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  CHK_SUPPORT_NAME_ONLY(enabled_blks)

  DEVICE_MUTEX

  ret = get_dev_value_line(amd::smi::kDevErrCntFeatures, dv_ind, &feature_line);
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", returning get_dev_value_line() response = "
       << amd::smi::getRSMIStatusString(ret);
    LOG_ERROR(ss);
    return ret;
  }

  std::istringstream fs1(feature_line);

  fs1 >> tmp_str;  // ignore
  assert(tmp_str == "feature");
  fs1 >> tmp_str;  // ignore
  assert(tmp_str == "mask:");
  fs1 >> tmp_str;

  errno = 0;
  *enabled_blks = strtoul(tmp_str.c_str(), nullptr, 16);
  assert(errno == 0);

  ss << __PRETTY_FUNCTION__ << " | ======= end ======="
     << ", returning strtoul() response = "
     << amd::smi::getRSMIStatusString(amd::smi::ErrnoToRsmiStatus(errno));
  LOG_TRACE(ss);

  return amd::smi::ErrnoToRsmiStatus(errno);
  CATCH
}

static const std::map<std::string, rsmi_ras_err_state_t> kRocmSMIStateMap = {
    {"none", RSMI_RAS_ERR_STATE_NONE},
    {"disabled", RSMI_RAS_ERR_STATE_DISABLED},
    {"parity", RSMI_RAS_ERR_STATE_PARITY},
    {"single_correctable", RSMI_RAS_ERR_STATE_SING_C},
    {"multi_uncorrectable", RSMI_RAS_ERR_STATE_MULT_UC},
    {"poison", RSMI_RAS_ERR_STATE_POISON},
    {"off", RSMI_RAS_ERR_STATE_DISABLED},
    {"on", RSMI_RAS_ERR_STATE_ENABLED},
};
static_assert(RSMI_RAS_ERR_STATE_LAST == RSMI_RAS_ERR_STATE_ENABLED,
                 "rsmi_gpu_block_t and/or above name map need to be updated"
                                                     " and then this assert");

rsmi_status_t rsmi_dev_ecc_status_get(uint32_t dv_ind, rsmi_gpu_block_t block,
                                                 rsmi_ras_err_state_t *state) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  CHK_SUPPORT_NAME_ONLY(state)

  if (!is_power_of_2(block)) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", ret was not power of 2 "
       << "-> reporting RSMI_STATUS_INVALID_ARGS";
    LOG_ERROR(ss);
    return RSMI_STATUS_INVALID_ARGS;
  }
  rsmi_status_t ret;
  uint64_t features_mask;

  DEVICE_MUTEX

  ret = rsmi_dev_ecc_enabled_get(dv_ind, &features_mask);

  if (ret == RSMI_STATUS_FILE_ERROR) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", rsmi_dev_ecc_enabled_get() ret was RSMI_STATUS_FILE_ERROR "
       << "-> reporting RSMI_STATUS_NOT_SUPPORTED";
    LOG_ERROR(ss);
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", returning rsmi_dev_ecc_enabled_get() response = "
       << amd::smi::getRSMIStatusString(ret);
    LOG_ERROR(ss);
    return ret;
  }

  *state = (features_mask & block) ?
                     RSMI_RAS_ERR_STATE_ENABLED : RSMI_RAS_ERR_STATE_DISABLED;

  ss << __PRETTY_FUNCTION__ << " | ======= end ======="
     << ", reporting RSMI_STATUS_SUCCESS";
  LOG_TRACE(ss);
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_ecc_count_get(uint32_t dv_ind, rsmi_gpu_block_t block,
                                                     rsmi_error_count_t *ec) {
  std::vector<std::string> val_vec;
  rsmi_status_t ret(RSMI_STATUS_NOT_SUPPORTED);
  std::ostringstream ss;

  TRY
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  amd::smi::DevInfoTypes type;
  switch (block) {
    case RSMI_GPU_BLOCK_UMC:
      type = amd::smi::kDevErrCntUMC;
      break;

    case RSMI_GPU_BLOCK_SDMA:
      type = amd::smi::kDevErrCntSDMA;
      break;

    case RSMI_GPU_BLOCK_GFX:
      type = amd::smi::kDevErrCntGFX;
      break;

    case RSMI_GPU_BLOCK_MMHUB:
      type = amd::smi::kDevErrCntMMHUB;
      break;

    case RSMI_GPU_BLOCK_PCIE_BIF:
      type = amd::smi::kDevErrCntPCIEBIF;
      break;

    case RSMI_GPU_BLOCK_HDP:
      type = amd::smi::kDevErrCntHDP;
      break;

    case RSMI_GPU_BLOCK_XGMI_WAFL:
      type = amd::smi::kDevErrCntXGMIWAFL;
      break;

    default:
      ss << __PRETTY_FUNCTION__ << " | ======= end ======="
         << ", default case -> reporting "
         << amd::smi::getRSMIStatusString(RSMI_STATUS_NOT_SUPPORTED);
      LOG_ERROR(ss);
      return RSMI_STATUS_NOT_SUPPORTED;
  }

  DEVICE_MUTEX

  ret = GetDevValueVec(type, dv_ind, &val_vec);
  if (val_vec.size() < 2 ) ret = RSMI_STATUS_FILE_ERROR;

  if (ret == RSMI_STATUS_FILE_ERROR) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", GetDevValueVec() ret was RSMI_STATUS_FILE_ERROR "
       << "-> reporting RSMI_STATUS_NOT_SUPPORTED";
    LOG_ERROR(ss);
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", GetDevValueVec() ret was not RSMI_STATUS_SUCCESS"
       << " -> reporting " << amd::smi::getRSMIStatusString(ret);
    LOG_ERROR(ss);
    return ret;
  }

  if (ec == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::string junk;
  std::istringstream fs1(val_vec[0]);

  fs1 >> junk;
  assert(junk == "ue:");
  fs1 >> ec->uncorrectable_err;

  std::istringstream fs2(val_vec[1]);

  fs2 >> junk;
  assert(junk == "ce:");
  fs2 >> ec->correctable_err;

  ec->deferred_err = 0;
  if (val_vec.size() > 2) {
    std::istringstream fs3(val_vec[2]);
    fs3 >> junk;
    if (junk == "de:") {
      fs3 >> ec->deferred_err;
    } else {
      ss << __PRETTY_FUNCTION__
       << "Trying to get the de count, but got " << junk
       << " ignore the defer count";
    LOG_ERROR(ss);
    }
  }

  ss << __PRETTY_FUNCTION__ << " | ======= end ======="
     << ", reporting " << amd::smi::getRSMIStatusString(ret);
  LOG_TRACE(ss);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_pci_id_get(uint32_t dv_ind, uint64_t *bdfid) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  GET_DEV_AND_KFDNODE_FROM_INDX
  CHK_API_SUPPORT_ONLY(bdfid, RSMI_DEFAULT_VARIANT, RSMI_DEFAULT_VARIANT)
  DEVICE_MUTEX

  *bdfid = dev->bdfid();

  uint64_t domain = 0;

  kfd_node->get_property_value("domain", &domain);

  /**
   * Add domain to full pci_id:
   * BDFID = ((DOMAIN & 0xFFFFFFFF) << 32) | ((PARTITION_ID & 0xF) << 28) |
   * ((BUS & 0xFF) << 8) | ((DEVICE & 0x1F) <<3 ) | (FUNCTION & 0x7)
   *
   * bits [63:32] = domain
   * bits [31:28] or bits [2:0] = partition id
   * bits [27:16] = reserved
   * bits [15:8]  = Bus
   * bits [7:3] = Device
   * bits [2:0] = Function (partition id maybe in bits [2:0]) <-- Fallback for non SPX modes
   */
  assert((domain & 0xFFFFFFFF00000000) == 0);
  (*bdfid) &= 0xFFFFFFFF;  // keep bottom 32 bits of pci_id
  *bdfid |= (domain & 0xFFFFFFFF) << 32;  // Add domain to top of pci_id
  uint64_t pci_id = *bdfid;
  uint32_t node = UINT32_MAX;
  rsmi_dev_node_id_get(dv_ind, &node);
  ss << __PRETTY_FUNCTION__ << " | kfd node = "
  << std::to_string(node) << "\n"
  << " returning pci_id = "
  << std::to_string(pci_id) << " ("
  << amd::smi::print_int_as_hex(pci_id) << ")";
  LOG_INFO(ss);

  ss << __PRETTY_FUNCTION__ << " | ======= end ======="
     << ", reporting RSMI_STATUS_SUCCESS";
  LOG_TRACE(ss);
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_topo_numa_affinity_get(uint32_t dv_ind, int32_t *numa_node) {
  TRY
  rsmi_status_t ret;

  CHK_SUPPORT_NAME_ONLY(numa_node)

  DEVICE_MUTEX
  std::string str_val;
  ret = get_dev_value_str(amd::smi::kDevNumaNode, dv_ind, &str_val);
  *numa_node = std::stoi(str_val, nullptr);

  return ret;
  CATCH
}

static rsmi_status_t
get_id(uint32_t dv_ind, amd::smi::DevInfoTypes typ, uint16_t *id) {
  TRY
  std::string val_str;
  uint64_t val_u64;

  assert(id != nullptr);
  if (id == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX

  rsmi_status_t ret = get_dev_value_str(typ, dv_ind, &val_str);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  errno = 0;
  val_u64 = strtoul(val_str.c_str(), nullptr, 16);
  assert(errno == 0);
  if (errno != 0) {
    return amd::smi::ErrnoToRsmiStatus(errno);
  }
  if (val_u64 > 0xFFFF) {
    return RSMI_STATUS_UNEXPECTED_SIZE;
  }
  *id = static_cast<uint16_t>(val_u64);

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t rsmi_ras_feature_info_get(
  uint32_t dv_ind, rsmi_ras_feature_info_t *ras_feature) {
  TRY
  rsmi_status_t ret;
  std::string feature_line;
  std::string tmp_str;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  CHK_SUPPORT_NAME_ONLY(ras_feature)

  DEVICE_MUTEX

  ret = get_dev_value_line(amd::smi::kDevErrTableVersion,
                dv_ind, &feature_line);
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", returning get_dev_value_line() response = "
       << amd::smi::getRSMIStatusString(ret);
    LOG_ERROR(ss);
    return ret;
  }

  // table version: 0x10000
  const char* version_key = "table version: ";
  if (feature_line.rfind(version_key, 0) == 0) {
    errno = 0;
    auto eeprom_version = strtoul(
      feature_line.substr(strlen(version_key)).c_str(), nullptr, 16);
    if (errno == 0) {
      ras_feature->ras_eeprom_version = static_cast<uint32_t>(eeprom_version);
    } else {
      return RSMI_STATUS_NOT_SUPPORTED;
    }
  } else {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  ret = get_dev_value_line(amd::smi::kDevErrRASSchema,
                dv_ind, &feature_line);
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", returning get_dev_value_line() response = "
       << amd::smi::getRSMIStatusString(ret);
    LOG_ERROR(ss);
    return ret;
  }
  // schema: 0xf
  const char* schema_key = "schema: ";
  if (feature_line.rfind(schema_key, 0) == 0) {
    errno = 0;
    auto schema = strtoul(
      feature_line.substr(strlen(schema_key)).c_str(), nullptr, 16);
    if (errno == 0) {
      ras_feature->ecc_correction_schema_flag = static_cast<uint32_t>(schema);
    } else {
      return RSMI_STATUS_NOT_SUPPORTED;
    }
  } else {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_id_get(uint32_t dv_ind, uint16_t *id) {
  std::ostringstream ss;
  rsmi_status_t ret;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(id)

  ret = get_id(dv_ind, amd::smi::kDevDevID, id);
  ss << __PRETTY_FUNCTION__ << " | ======= end ======="
     << ", reporting " << amd::smi::getRSMIStatusString(ret);
  LOG_TRACE(ss);
  return ret;
}

rsmi_status_t
rsmi_dev_xgmi_physical_id_get(uint32_t dv_ind, uint16_t *id) {
  std::ostringstream ss;
  rsmi_status_t ret;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(id)
  *id = std::numeric_limits<uint16_t>::max();

  ret = get_id(dv_ind, amd::smi::kDevXGMIPhysicalID, id);
  ss << __PRETTY_FUNCTION__ << " | ======= end ======="
     << ", reporting " << amd::smi::getRSMIStatusString(ret);
  LOG_TRACE(ss);
  return ret;
}

rsmi_status_t
rsmi_dev_revision_get(uint32_t dv_ind, uint16_t *revision) {
  std::ostringstream outss;
  rsmi_status_t ret;
  outss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(outss);
  CHK_SUPPORT_NAME_ONLY(revision)

  ret = get_id(dv_ind, amd::smi::kDevDevRevID, revision);
  outss << __PRETTY_FUNCTION__ << " | ======= end ======="
     << ", reporting " << amd::smi::getRSMIStatusString(ret);
  LOG_TRACE(outss);
  return ret;
}

rsmi_status_t
rsmi_dev_sku_get(uint32_t dv_ind, uint16_t *id) {
  TRY
  std::ostringstream ss;
  rsmi_status_t ret;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(id)
  ret = get_id(dv_ind, amd::smi::kDevDevProdNum, id);
  ss << __PRETTY_FUNCTION__ << " | ======= end ======="
     << ", reporting " << amd::smi::getRSMIStatusString(ret);
  LOG_TRACE(ss);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_subsystem_id_get(uint32_t dv_ind, uint16_t *id) {
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(id)
  return get_id(dv_ind, amd::smi::kDevSubSysDevID, id);
}

rsmi_status_t
rsmi_dev_vendor_id_get(uint32_t dv_ind, uint16_t *id) {
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(id)
  return get_id(dv_ind, amd::smi::kDevVendorID, id);
}

rsmi_status_t
rsmi_dev_pcie_slot_type_get(uint32_t dv_ind, rsmi_pcie_slot_type_t* type) {
  TRY

  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(type)
  DEVICE_MUTEX

  std::string value;
  int ret = dev->readDevInfo(amd::smi::kDevBoardInfo, "type", value);
  if (ret != 0) return RSMI_STATUS_NOT_SUPPORTED;

  *type = RSMI_PCIE_SLOT_PCIE;
  if (value.compare("oam") == 0) *type=RSMI_PCIE_SLOT_OAM;
  else if (value.compare("cem") == 0 ) *type=RSMI_PCIE_SLOT_CEM;
  else if (value.compare("unknown") == 0 ) *type=RSMI_PCIE_SLOT_UNKNOWN;
  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_subsystem_vendor_id_get(uint32_t dv_ind, uint16_t *id) {
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(id)
  return get_id(dv_ind, amd::smi::kDevSubSysVendorID, id);
}

rsmi_status_t
rsmi_dev_perf_level_get(uint32_t dv_ind, rsmi_dev_perf_level_t *perf) {
  TRY
  std::string val_str;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  CHK_SUPPORT_NAME_ONLY(perf)
  DEVICE_MUTEX

  rsmi_status_t ret = get_dev_value_str(amd::smi::kDevPerfLevel, dv_ind,
                                                                    &val_str);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  *perf = amd::smi::Device::perfLvlStrToEnum(val_str);

  return ret;
  CATCH
}

static rsmi_status_t
set_dev_range(uint32_t dv_ind, std::string range) {
  GET_DEV_FROM_INDX

  int ret = dev->writeDevInfo(amd::smi::kDevPowerODVoltage, range);
  return amd::smi::ErrnoToRsmiStatus(ret);
}

rsmi_status_t
rsmi_perf_determinism_mode_set(uint32_t dv_ind, uint64_t clkvalue) {
  TRY
  DEVICE_MUTEX
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  // Set perf. level to performance determinism so that we can then set the power profile
  rsmi_status_t ret = rsmi_dev_perf_level_set_v1(dv_ind,
                                          RSMI_DEV_PERF_LEVEL_DETERMINISM);
  if (ret != RSMI_STATUS_SUCCESS) {
      return ret;
  }

  // For clock frequency setting, enter a new value by writing a string that
  // contains "s index clock" to the file. The index should be 1 to set maximum
  // clock. E.g., "s 1 500" will update maximum sclk to be 500 MHz.

  std::string sysvalue = "s";
  sysvalue += ' ' + std::to_string(RSMI_FREQ_IND_MAX);
  sysvalue += ' ' + std::to_string(clkvalue);
  sysvalue += '\n';
  ret = set_dev_range(dv_ind, sysvalue);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }
  ret = set_dev_range(dv_ind, "c");
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}


rsmi_status_t
rsmi_dev_overdrive_level_get(uint32_t dv_ind, uint32_t *od) {
  TRY
  std::string val_str;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(od)
  DEVICE_MUTEX

  // Bare Metal only feature
  if (amd::smi::is_vm_guest()) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  rsmi_status_t ret = get_dev_value_str(amd::smi::kDevOverDriveLevel, dv_ind,
                                                                    &val_str);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  errno = 0;
  uint64_t val_ul = strtoul(val_str.c_str(), nullptr, 10);

  if (val_ul > 0xFFFFFFFF) {
    return RSMI_STATUS_UNEXPECTED_SIZE;
  }

  *od = static_cast<uint32_t>(val_ul);
  assert(errno == 0);

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_mem_overdrive_level_get(uint32_t dv_ind, uint32_t *od) {
  TRY
  std::string val_str;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(od)
  DEVICE_MUTEX

  rsmi_status_t ret = get_dev_value_str(amd::smi::kDevMemOverDriveLevel, dv_ind,
                                                                    &val_str);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  errno = 0;
  uint64_t val_ul = strtoul(val_str.c_str(), nullptr, 10);

  if (val_ul > 0xFFFFFFFF) {
    return RSMI_STATUS_UNEXPECTED_SIZE;
  }

  *od = static_cast<uint32_t>(val_ul);
  assert(errno == 0);

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_overdrive_level_set(uint32_t dv_ind, uint32_t od) {
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  return rsmi_dev_overdrive_level_set_v1(static_cast<uint32_t>(dv_ind), od);
}

rsmi_status_t
rsmi_dev_overdrive_level_set_v1(uint32_t dv_ind, uint32_t od) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS

  if (od > kMaxOverdriveLevel) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  // Bare Metal only feature
  if (amd::smi::is_vm_guest()) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  DEVICE_MUTEX
  return set_dev_value(amd::smi::kDevOverDriveLevel, dv_ind, od);
  CATCH
}

rsmi_status_t
rsmi_dev_perf_level_set(uint32_t dv_ind, rsmi_dev_perf_level_t perf_level) {
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  return rsmi_dev_perf_level_set_v1(dv_ind, perf_level);
}

rsmi_status_t
rsmi_dev_perf_level_set_v1(uint32_t dv_ind, rsmi_dev_perf_level_t perf_level) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS

  if (perf_level > RSMI_DEV_PERF_LEVEL_LAST) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX
  return set_dev_value(amd::smi::kDevPerfLevel, dv_ind, perf_level);
  CATCH
}


static rsmi_status_t get_frequencies(amd::smi::DevInfoTypes type, rsmi_clk_type_t clk_type,
            uint32_t dv_ind, rsmi_frequencies_t *f, uint32_t *lanes = nullptr) {
  TRY
  std::vector<std::string> val_vec;
  rsmi_status_t ret;

  if (f == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  memset(f, 0, sizeof(rsmi_frequencies_t));
  f->current = 0;

  ret = GetDevValueVec(type, dv_ind, &val_vec);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }
  assert(val_vec.size() <= RSMI_MAX_NUM_FREQUENCIES);

  if (val_vec.empty()) {
    return RSMI_STATUS_NOT_YET_IMPLEMENTED;
  }

  f->num_supported = static_cast<uint32_t>(val_vec.size());
  f->current = RSMI_MAX_NUM_FREQUENCIES + 1;  // init to an invalid value

  // Deep Sleep frequency is only supported by some GPUs
  // It is indicated by letter 'S' instead of the index number
  f->has_deep_sleep = (val_vec[0][0] == 'S');

  bool current = false;

  for (uint32_t i = 0; i < f->num_supported; ++i) {
    f->frequency[i] = freq_string_to_int(val_vec, &current, lanes, i);

    // Our assumption is that frequencies are read in from lowest to highest.
    // Check that that is true.
    if (i > 0) {
      if (f->frequency[i] < f->frequency[i-1]) {
        std::string sysvalue;
        sysvalue += kClkStateMap.find(clk_type)->second;
        sysvalue += " Current Value";
        sysvalue += ' ' + std::to_string(f->frequency[i]);
        sysvalue += " Previous Value";
        sysvalue += ' ' + std::to_string(f->frequency[i-1]);
        DEBUG_LOG("Frequencies are not read from lowest to highest. ", sysvalue);
      }
    }
    if (current) {
      // set the current frequency
      if (f->current != RSMI_MAX_NUM_FREQUENCIES + 1) {
        std::string sysvalue;
        sysvalue += kClkStateMap.find(clk_type)->second;
        sysvalue += " Current Value";
        sysvalue += ' ' + std::to_string(f->frequency[i]);
        sysvalue += " Previous Value";
        sysvalue += ' ' + std::to_string(f->frequency[f->current]);
        DEBUG_LOG("More than one current clock. ", sysvalue);
      } else {
          f->current = i;
      }
    }
  }

  // Some older drivers will not have the current frequency set
  // assert(f->current < f->num_supported);
  if (f->current >= f->num_supported) {
      f->current = -1;
      return RSMI_STATUS_UNEXPECTED_DATA;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}

static rsmi_status_t get_power_profiles(uint32_t dv_ind,
                                        rsmi_power_profile_status_t *p,
               std::map<rsmi_power_profile_preset_masks_t, uint32_t> *ind_map) {
  TRY
  std::vector<std::string> val_vec;
  rsmi_status_t ret;

  if (p == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  ret = GetDevValueVec(amd::smi::kDevPowerProfileMode, dv_ind, &val_vec);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }
  assert(val_vec.size() <= RSMI_MAX_NUM_POWER_PROFILES);
  if (val_vec.size() > RSMI_MAX_NUM_POWER_PROFILES + 1 || val_vec.empty()) {
    // Guest may not have power related information.
    if (amd::smi::is_vm_guest()) {
      return RSMI_STATUS_NOT_SUPPORTED;
    }
    return RSMI_STATUS_UNEXPECTED_SIZE;
  }
  // -1 for the header line, below
  p->num_profiles = static_cast<uint32_t>(val_vec.size() - 1);
  bool current = false;
  p->current = RSMI_PWR_PROF_PRST_INVALID;  // init to an invalid value
  p->available_profiles = 0;

  rsmi_power_profile_preset_masks_t prof;
  uint32_t prof_ind;

  for (uint32_t i = 1; i < val_vec.size(); ++i) {
    prof = power_prof_string_to_int(val_vec[i], &current, &prof_ind);

    if (prof == RSMI_PWR_PROF_PRST_INVALID) {
      continue;
    }

    if (ind_map != nullptr) {
      (*ind_map)[prof] = prof_ind;
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

/* We expect the pp_od_clk_voltage file to look like either of the two
formats shown below. Some of the newer ASICs will most likely have the
new format.

Old Format:
OD_SCLK:
0:        872Mhz
1:       1837Mhz
OD_MCLK:
1:       1000Mhz
OD_VDDC_CURVE:
0:        872Mhz        736mV
1:       1354Mhz        860mV
2:       1837Mhz       1186mV
OD_RANGE:
SCLK:     872Mhz       1900Mhz
MCLK:     168Mhz       1200Mhz
VDDC_CURVE_SCLK[0]:     872Mhz       1900Mhz
VDDC_CURVE_VOLT[0]:     737mV        1137mV
VDDC_CURVE_SCLK[1]:     872Mhz       1900Mhz
VDDC_CURVE_VOLT[1]:     737mV        1137mV
VDDC_CURVE_SCLK[2]:     872Mhz       1900Mhz
VDDC_CURVE_VOLT[2]:     737mV        1137mV

New Format:
GFXCLK:
0: 500Mhz
1: 800Mhz *
2: 1275Mhz
MCLK:
0: 400Mhz
1: 700Mhz
2: 1200Mhz
3: 1600Mhz *

For the new format, GFXCLK field will show min and max values(0/1). If the current
frequency in neither min/max but lies within the range, this is indicated by
an additional value followed by * at index 1 and max value at index 2.
*/
constexpr uint32_t kMIN_VALID_LINES = 2;

static rsmi_status_t get_od_clk_volt_info(uint32_t dv_ind,
                                                  rsmi_od_volt_freq_data_t *p) {
  TRY
  std::vector<std::string> val_vec;
  rsmi_status_t ret;

  assert(p != nullptr);
  if (p == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  ret = GetDevValueVec(amd::smi::kDevPowerODVoltage, dv_ind, &val_vec);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  // This is a work-around to handle systems where kDevPowerODVoltage is not
  // fully supported yet.
  if (val_vec.size() < kMIN_VALID_LINES) {
    return RSMI_STATUS_NOT_YET_IMPLEMENTED;
  }

  // Tags expected in this file
  const std::string kTAG_OD_SCLK{"OD_SCLK:"};
  const std::string KTAG_OD_MCLK{"OD_MCLK:"};
  const std::string kTAG_GFXCLK{"GFXCLK:"};
  const std::string KTAG_MCLK{"MCLK:"};
  const std::string KTAG_SCLK{"SCLK:"};
  const std::string KTAG_OD_RANGE{"OD_RANGE:"};
  const std::string KTAG_OD_VDDGFX_OFFSET{"OD_VDDGFX_OFFSET:"};
  const std::string KTAG_FIRST_FREQ_IDX{"0:"};

  amd::smi::TextFileTagContents_t txt_power_dev_od_voltage(val_vec);
  txt_power_dev_od_voltage
    .set_title_terminator(":", amd::smi::TagSplitterPositional_t::kLAST)
    .set_key_data_splitter(":", amd::smi::TagSplitterPositional_t::kBETWEEN)
    .structure_content();

  //
  // Note:  We must have minimum of 'GFXCLK:' && 'MCLK:' OR:
  //        'OD_SCLK:' && 'OD_MCLK:' tags.
  if (txt_power_dev_od_voltage.get_title_size() < kMIN_VALID_LINES)  {
      return rsmi_status_t::RSMI_STATUS_NO_DATA;
  }

  // Note:  For debug builds/purposes only.
  assert(txt_power_dev_od_voltage.contains_title_key(kTAG_GFXCLK) ||
         txt_power_dev_od_voltage.contains_title_key(kTAG_OD_SCLK));
  // Note:  For release builds/purposes.
  if (!txt_power_dev_od_voltage.contains_title_key(kTAG_GFXCLK) &&
      !txt_power_dev_od_voltage.contains_title_key(kTAG_OD_SCLK)) {
      return rsmi_status_t::RSMI_STATUS_UNEXPECTED_DATA;
  }

  // Note: Quick helpers for getting 1st and last elements found
  auto build_lower_bound = [&](const std::string& prim_key) {
      auto lower_bound_data = txt_power_dev_od_voltage.get_structured_data_subkey_first(prim_key);
      return std::vector<std::string>{lower_bound_data};
  };

  auto build_upper_bound = [&](const std::string& prim_key) {
      auto upper_bound_data = txt_power_dev_od_voltage.get_structured_data_subkey_last(prim_key);
      return std::vector<std::string>{upper_bound_data};
  };

  // Validates 'OD_SCLK' is in the structure
  if (txt_power_dev_od_voltage.contains_structured_key(kTAG_OD_SCLK,
                                                       KTAG_FIRST_FREQ_IDX)) {
      p->curr_sclk_range.lower_bound = freq_string_to_int(build_lower_bound(kTAG_OD_SCLK), nullptr, nullptr, 0);
      p->curr_sclk_range.upper_bound = freq_string_to_int(build_upper_bound(kTAG_OD_SCLK), nullptr, nullptr, 0);

      // Validates 'OD_MCLK' is in the structure
      if (txt_power_dev_od_voltage.contains_structured_key(KTAG_OD_MCLK,
                                                           KTAG_FIRST_FREQ_IDX)) {
          p->curr_mclk_range.lower_bound = freq_string_to_int(build_lower_bound(KTAG_OD_MCLK), nullptr, nullptr, 0);
          p->curr_mclk_range.upper_bound = freq_string_to_int(build_upper_bound(KTAG_OD_MCLK), nullptr, nullptr, 0);
      }

      // Validates 'OD_RANGE' is in the structure
      if (txt_power_dev_od_voltage.contains_structured_key(KTAG_OD_RANGE,
                                                           KTAG_SCLK)) {
          od_value_pair_str_to_range(txt_power_dev_od_voltage
                                        .get_structured_value_by_keys(KTAG_OD_RANGE, KTAG_SCLK),
                                     &p->sclk_freq_limits);
      }
      if (txt_power_dev_od_voltage.contains_structured_key(KTAG_OD_RANGE,
                                                           KTAG_MCLK)) {
          od_value_pair_str_to_range(txt_power_dev_od_voltage
                                        .get_structured_value_by_keys(KTAG_OD_RANGE, KTAG_MCLK),
                                     &p->mclk_freq_limits);
      }
  }
  // Validates 'GFXCLK' is in the structure
  else if (txt_power_dev_od_voltage.contains_structured_key(kTAG_GFXCLK,
                                                            KTAG_FIRST_FREQ_IDX)) {
      p->curr_sclk_range.lower_bound = freq_string_to_int(build_lower_bound(kTAG_GFXCLK), nullptr, nullptr, 0);
      p->curr_sclk_range.upper_bound = freq_string_to_int(build_upper_bound(kTAG_GFXCLK), nullptr, nullptr, 0);

      // Validates 'MCLK' is in the structure
      if (txt_power_dev_od_voltage.contains_structured_key(KTAG_MCLK,
                                                           KTAG_FIRST_FREQ_IDX)) {
          p->curr_mclk_range.lower_bound = freq_string_to_int(build_lower_bound(KTAG_MCLK), nullptr, nullptr, 0);
          p->curr_mclk_range.upper_bound = freq_string_to_int(build_upper_bound(KTAG_MCLK), nullptr, nullptr, 0);
      }
  }
  else {
      return RSMI_STATUS_NOT_YET_IMPLEMENTED;
  }

  // Note: No curve entries.
  p->num_regions = 0;

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t rsmi_dev_clk_extremum_set(uint32_t dv_ind, rsmi_freq_ind_t level,
                                        uint64_t clkvalue,
                                        rsmi_clk_type_t clkType) {
 TRY
  rsmi_status_t ret;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  if (clkType != RSMI_CLK_TYPE_SYS && clkType != RSMI_CLK_TYPE_MEM) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  if (level != RSMI_FREQ_IND_MIN && level != RSMI_FREQ_IND_MAX) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::map<rsmi_clk_type_t, std::string> clk_char_map = {
    {RSMI_CLK_TYPE_SYS, "s"},
    {RSMI_CLK_TYPE_MEM, "m"},
  };
  DEVICE_MUTEX

  // Set perf. level to manual so that we can then set the power profile
  ret = rsmi_dev_perf_level_set_v1(dv_ind, RSMI_DEV_PERF_LEVEL_MANUAL);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  // For clock frequency setting, enter a new value by writing a string that
  // contains "s/m index clock" to the file. The index should be 0 if to set
  // minimum clock. And 1 if to set maximum clock. E.g., "s 0 500" will update
  // minimum sclk to be 500 MHz. "m 1 800" will update maximum mclk to 800Mhz.

  std::string sysvalue = clk_char_map[clkType];
  sysvalue += ' ' + std::to_string(level);
  sysvalue += ' ' + std::to_string(clkvalue);
  sysvalue += '\n';

  ret = set_dev_range(dv_ind, sysvalue);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  ret = set_dev_range(dv_ind, "c");
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t rsmi_dev_clk_range_set(uint32_t dv_ind, uint64_t minclkvalue,
                                        uint64_t maxclkvalue,
                                        rsmi_clk_type_t clkType) {
  TRY
  rsmi_status_t ret;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  if (minclkvalue >= maxclkvalue) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  // Bare Metal only feature
  if (amd::smi::is_vm_guest()) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  // Can only set the clock type for sys and mem type
  if (clkType != RSMI_CLK_TYPE_SYS && clkType != RSMI_CLK_TYPE_MEM) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  std::string min_sysvalue;
  std::string max_sysvalue;
  std::map<rsmi_clk_type_t, std::string> clk_char_map = {
    {RSMI_CLK_TYPE_SYS, "s"},
    {RSMI_CLK_TYPE_MEM, "m"},
  };
  DEVICE_MUTEX
  assert(clkType == RSMI_CLK_TYPE_SYS || clkType == RSMI_CLK_TYPE_MEM);

  // Set perf. level to manual so that we can then set the power profile
  ret = rsmi_dev_perf_level_set_v1(dv_ind, RSMI_DEV_PERF_LEVEL_MANUAL);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  // For clock frequency setting, enter a new value by writing a string that
  // contains "s/m index clock" to the file. The index should be 0 if to set
  // minimum clock. And 1 if to set maximum clock. E.g., "s 0 500" will update
  // minimum sclk to be 500 MHz. "m 1 800" will update maximum mclk to 800Mhz.

  min_sysvalue = clk_char_map[clkType];
  min_sysvalue += ' ' + std::to_string(RSMI_FREQ_IND_MIN);
  min_sysvalue += ' ' + std::to_string(minclkvalue);
  min_sysvalue += '\n';
  max_sysvalue = clk_char_map[clkType];
  max_sysvalue += ' ' + std::to_string(RSMI_FREQ_IND_MAX);
  max_sysvalue += ' ' + std::to_string(maxclkvalue);
  max_sysvalue += '\n';

  ret = set_dev_range(dv_ind, min_sysvalue);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }
  ret = set_dev_range(dv_ind, max_sysvalue);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }
  ret = set_dev_range(dv_ind, "c");
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t rsmi_dev_od_clk_info_set(uint32_t dv_ind, rsmi_freq_ind_t level,
                                        uint64_t clkvalue,
                                        rsmi_clk_type_t clkType) {
  TRY
  rsmi_status_t ret;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  std::string sysvalue;
  std::map<rsmi_clk_type_t, std::string> clk_char_map = {
    {RSMI_CLK_TYPE_SYS, "s"},
    {RSMI_CLK_TYPE_MEM, "m"},
  };
  DEVICE_MUTEX

  // Set perf. level to manual so that we can then set the power profile
  ret = rsmi_dev_perf_level_set_v1(dv_ind, RSMI_DEV_PERF_LEVEL_MANUAL);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  // For clock frequency setting, enter a new value by writing a string that
  // contains "s/m index clock" to the file. The index should be 0 if to set
  // minimum clock. And 1 if to set maximum clock. E.g., "s 0 500" will update
  // minimum sclk to be 500 MHz. "m 1 800" will update maximum mclk to 800Mhz.

  switch (clkType) {
    case RSMI_CLK_TYPE_SYS:
    case RSMI_CLK_TYPE_MEM:
      sysvalue = clk_char_map[clkType];
      sysvalue += ' ' + std::to_string(level);
      sysvalue += ' ' + std::to_string(clkvalue);
      sysvalue += '\n';
      break;

    default:
      return RSMI_STATUS_INVALID_ARGS;
  }
  ret = set_dev_range(dv_ind, sysvalue);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }
  ret = set_dev_range(dv_ind, "c");
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}


rsmi_status_t rsmi_dev_od_volt_info_set(uint32_t dv_ind, uint32_t vpoint,
                                      uint64_t clkvalue, uint64_t voltvalue) {
  TRY
  rsmi_status_t ret;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  DEVICE_MUTEX

  // Set perf. level to manual so that we can then set the power profile
  ret = rsmi_dev_perf_level_set_v1(dv_ind, RSMI_DEV_PERF_LEVEL_MANUAL);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  // For sclk voltage curve, enter the new values by writing a string that
  // contains "vc point clock voltage" to the file. The points are indexed
  // by 0, 1 and 2. E.g., "vc 0 300 600" will update point1 with clock set
  // as 300Mhz and voltage as 600mV. "vc 2 1000 1000" will update point3
  // with clock set as 1000Mhz and voltage 1000mV.

  std::string sysvalue = "vc";
  sysvalue += ' ' + std::to_string(vpoint);
  sysvalue += ' ' + std::to_string(clkvalue);
  sysvalue += ' ' + std::to_string(voltvalue);
  sysvalue += '\n';
  ret = set_dev_range(dv_ind, sysvalue);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }
  ret = set_dev_range(dv_ind, "c");
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}


static void get_vc_region(const std::vector<std::string>& val_vec, rsmi_freq_volt_region_t& p)
{
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  //
  amd::smi::TextFileTagContents_t txt_power_dev_od_voltage(val_vec);
  txt_power_dev_od_voltage
    .set_title_terminator(":", amd::smi::TagSplitterPositional_t::kLAST)
    .set_key_data_splitter(":", amd::smi::TagSplitterPositional_t::kBETWEEN)
    .structure_content();

  const std::string KTAG_OD_RANGE{"OD_RANGE:"};
  const std::string KTAG_MCLK{"MCLK:"};
  const std::string KTAG_SCLK{"SCLK:"};
  if (txt_power_dev_od_voltage.contains_structured_key(KTAG_OD_RANGE,
                                                       KTAG_SCLK)) {
      od_value_pair_str_to_range(txt_power_dev_od_voltage
                                    .get_structured_value_by_keys(KTAG_OD_RANGE, KTAG_SCLK),
                                 &p.freq_range);
  }
  if (txt_power_dev_od_voltage.contains_structured_key(KTAG_OD_RANGE,
                                                       KTAG_MCLK)) {
      od_value_pair_str_to_range(txt_power_dev_od_voltage
                                    .get_structured_value_by_keys(KTAG_OD_RANGE, KTAG_MCLK),
                                 &p.volt_range);
  }
}

/*
 * num_regions [inout] on calling, the number of regions requested to be read
 * in. At completion, the number of regions actually read in
 *
 * p [inout] point to pre-allocated memory where function will write region
 * values. Caller must make sure there is enough space for at least
 * *num_regions regions.
 */
static rsmi_status_t get_od_clk_volt_curve_regions(uint32_t dv_ind,
                            uint32_t *num_regions, rsmi_freq_volt_region_t *p) {
  TRY
  std::vector<std::string> val_vec;
  rsmi_status_t ret;
  std::ostringstream ss;

  assert(num_regions != nullptr);
  assert(p != nullptr);
  THROW_IF_NULLPTR_DEREF(p)
  THROW_IF_NULLPTR_DEREF(num_regions)

  ret = GetDevValueVec(amd::smi::kDevPowerODVoltage, dv_ind, &val_vec);
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | Issue: could not retreive kDevPowerODVoltage" << "; returning "
       << getRSMIStatusString(ret);
    LOG_ERROR(ss);
    return ret;
  }

  // This is a work-around to handle systems where kDevPowerODVoltage is not
  // fully supported yet.
  if (val_vec.size() < kMIN_VALID_LINES) {
    ss << __PRETTY_FUNCTION__
       << " | Issue: val_vec.size() < " << kMIN_VALID_LINES << "; returning "
       << getRSMIStatusString(RSMI_STATUS_NOT_YET_IMPLEMENTED);
    LOG_ERROR(ss);
    return RSMI_STATUS_NOT_YET_IMPLEMENTED;
  }

  uint32_t val_vec_size = static_cast<uint32_t>(val_vec.size());
  ss << __PRETTY_FUNCTION__
     << " | val_vec_size = " << std::dec
     << val_vec_size;
  LOG_DEBUG(ss);

  // Note: No curve entries.
  *num_regions = 0;
  // Get OD ranges.
  get_vc_region(val_vec, *p);

  return RSMI_STATUS_SUCCESS;
  CATCH
}

static rsmi_status_t set_power_profile(uint32_t dv_ind,
                                    rsmi_power_profile_preset_masks_t profile) {
  TRY

  rsmi_status_t ret;
  rsmi_power_profile_status_t avail_profiles =
                                           {0, RSMI_PWR_PROF_PRST_INVALID, 0};

  // Determine if the provided profile is valid
  if (!is_power_of_2(profile)) {
    return RSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }

  std::map<rsmi_power_profile_preset_masks_t, uint32_t> ind_map;
  ret = get_power_profiles(dv_ind, &avail_profiles, &ind_map);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  if (!(profile & avail_profiles.available_profiles)) {
    return RSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }
  assert(ind_map.find(profile) != ind_map.end());

  // Set perf. level to manual so that we can then set the power profile
  ret = rsmi_dev_perf_level_set_v1(dv_ind, RSMI_DEV_PERF_LEVEL_MANUAL);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  // Write the new profile
  ret = set_dev_value(amd::smi::kDevPowerProfileMode, dv_ind,
                                                            ind_map[profile]);

  return ret;
  CATCH
}

static rsmi_status_t topo_get_numa_node_number(uint32_t dv_ind,
                     uint32_t *numa_node_number) {
  TRY

  GET_DEV_AND_KFDNODE_FROM_INDX

  *numa_node_number = kfd_node->numa_node_number();

  return RSMI_STATUS_SUCCESS;
  CATCH
}

static rsmi_status_t topo_get_numa_node_weight(uint32_t dv_ind,
                     uint64_t *weight) {
  TRY

  GET_DEV_AND_KFDNODE_FROM_INDX

  *weight = kfd_node->numa_node_weight();

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_gpu_clk_freq_get(uint32_t dv_ind, rsmi_clk_type_t clk_type,
                                                        rsmi_frequencies_t *f) {
  TRY
  amd::smi::DevInfoTypes dev_type;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  CHK_SUPPORT_VAR(f, clk_type)

  const auto & clk_type_it = kClkTypeMap.find(clk_type);
  if (clk_type_it != kClkTypeMap.end()) {
    dev_type = clk_type_it->second;
  } else {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX

  return get_frequencies(dev_type, clk_type, dv_ind, f);

  CATCH
}

rsmi_status_t
rsmi_dev_firmware_version_get(uint32_t dv_ind, rsmi_fw_block_t block,
                                                       uint64_t *fw_version) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_VAR(fw_version, block)

  std::string val_str;
  amd::smi::DevInfoTypes dev_type;

  static const std::map<rsmi_fw_block_t, amd::smi::DevInfoTypes> kFWBlockTypeMap = {
    { RSMI_FW_BLOCK_ASD, amd::smi::kDevFwVersionAsd },
    { RSMI_FW_BLOCK_CE, amd::smi::kDevFwVersionCe },
    { RSMI_FW_BLOCK_DMCU, amd::smi::kDevFwVersionDmcu },
    { RSMI_FW_BLOCK_MC, amd::smi::kDevFwVersionMc },
    { RSMI_FW_BLOCK_ME, amd::smi::kDevFwVersionMe },
    { RSMI_FW_BLOCK_MEC, amd::smi::kDevFwVersionMec },
    { RSMI_FW_BLOCK_MEC2, amd::smi::kDevFwVersionMec2 },
    { RSMI_FW_BLOCK_MES, amd::smi::kDevFwVersionMes },
    { RSMI_FW_BLOCK_MES_KIQ, amd::smi::kDevFwVersionMesKiq },
    { RSMI_FW_BLOCK_PFP, amd::smi::kDevFwVersionPfp },
    { RSMI_FW_BLOCK_RLC, amd::smi::kDevFwVersionRlc },
    { RSMI_FW_BLOCK_RLC_SRLC, amd::smi::kDevFwVersionRlcSrlc },
    { RSMI_FW_BLOCK_RLC_SRLG, amd::smi::kDevFwVersionRlcSrlg },
    { RSMI_FW_BLOCK_RLC_SRLS, amd::smi::kDevFwVersionRlcSrls },
    { RSMI_FW_BLOCK_SDMA, amd::smi::kDevFwVersionSdma },
    { RSMI_FW_BLOCK_SDMA2, amd::smi::kDevFwVersionSdma2 },
    { RSMI_FW_BLOCK_SMC, amd::smi::kDevFwVersionSmc },
    { RSMI_FW_BLOCK_SOS, amd::smi::kDevFwVersionSos },
    { RSMI_FW_BLOCK_TA_RAS, amd::smi::kDevFwVersionTaRas },
    { RSMI_FW_BLOCK_TA_XGMI, amd::smi::kDevFwVersionTaXgmi },
    { RSMI_FW_BLOCK_UVD, amd::smi::kDevFwVersionUvd },
    { RSMI_FW_BLOCK_VCE, amd::smi::kDevFwVersionVce },
    { RSMI_FW_BLOCK_VCN, amd::smi::kDevFwVersionVcn },
  };

  const auto & dev_type_it = kFWBlockTypeMap.find(block);
  if (dev_type_it != kFWBlockTypeMap.end()) {
    dev_type = dev_type_it->second;
  } else {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX
  return get_dev_value_int(dev_type, dv_ind, fw_version);
  CATCH
}

static std::string bitfield_to_freq_string(uint64_t bitf,
                                                     uint32_t num_supported) {
  std::string bf_str;
  std::bitset<RSMI_MAX_NUM_FREQUENCIES> bs(bitf);

  if (num_supported > RSMI_MAX_NUM_FREQUENCIES) {
    throw amd::smi::rsmi_exception(RSMI_STATUS_INVALID_ARGS, __FUNCTION__);
  }

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
                              rsmi_clk_type_t clk_type, uint64_t freq_bitmask) {
  rsmi_status_t ret;
  rsmi_frequencies_t freqs;

  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX

  if (clk_type > RSMI_CLK_TYPE_LAST) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  // Bare Metal only feature
  if (amd::smi::is_vm_guest()) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  ret = rsmi_dev_gpu_clk_freq_get(dv_ind, clk_type, &freqs);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  assert(freqs.num_supported <= RSMI_MAX_NUM_FREQUENCIES);
  if (freqs.num_supported > RSMI_MAX_NUM_FREQUENCIES) {
    return RSMI_STATUS_UNEXPECTED_SIZE;
  }

  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  // Above call to rsmi_dev_get_gpu_clk_freq should have emitted an error if
  // assert below is not true
  assert(dv_ind < smi.devices().size());

  std::string freq_enable_str =
                   bitfield_to_freq_string(freq_bitmask, freqs.num_supported);


  std::shared_ptr<amd::smi::Device> dev = smi.devices()[dv_ind];
  assert(dev != nullptr);

  ret = rsmi_dev_perf_level_set_v1(dv_ind, RSMI_DEV_PERF_LEVEL_MANUAL);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  rsmi_status_t status;
  amd::smi::DevInfoTypes dev_type;

  const auto & clk_type_it = kClkTypeMap.find(clk_type);
  if (clk_type_it != kClkTypeMap.end()) {
    dev_type = clk_type_it->second;
  } else {
    return RSMI_STATUS_INVALID_ARGS;
  }

  status =  amd::smi::ErrnoToRsmiStatus(dev->writeDevInfo(dev_type, freq_enable_str));

  // If an operation is not supported, the dev file, ie /sys/class/drm/card1/device/pp_dpm_pcie
  // will have read-only perms, and the OS will deny access, before the request hits the driver level
  if (status == RSMI_STATUS_PERMISSION){
    bool read_only = false;
    amd::smi::isReadOnlyForAll(dev->path(), &read_only);
    if(read_only){
      return RSMI_STATUS_NOT_SUPPORTED;
    }
  }

  return status;

  CATCH
}


rsmi_status_t rsmi_dev_process_isolation_get(uint32_t dv_ind,
                             uint32_t* pisolate) {
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start ======= dev_ind:"
    << dv_ind;
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(pisolate)

  // the enforce_isolation sysfs is in this format <partition_id, enable_flag>
  // Get the partition_id. For SPX, the partition_id will be 0.
  uint32_t partition_id = 0;
  rsmi_dev_partition_id_get(dv_ind, &partition_id);

  DEVICE_MUTEX

  std::string str_val;
  rsmi_status_t ret = get_dev_value_line(amd::smi::kDevProcessIsolation, dv_ind, &str_val);
  if (ret == RSMI_STATUS_FILE_ERROR) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", get_dev_value_str() ret was RSMI_STATUS_FILE_ERROR "
       << "-> reporting RSMI_STATUS_NOT_SUPPORTED";
    LOG_ERROR(ss);
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", get_dev_value_str() ret was not RSMI_STATUS_SUCCESS"
       << " -> reporting " << amd::smi::getRSMIStatusString(ret);
    LOG_ERROR(ss);
    return ret;
  }

  /*
  for 4 partition: enforce isolation is enabled on partition 2 and
  disabled on partitions 0, 1, 3.
  $ cat /sys/class/drm/cardX/device/enforce_isolation
   0 0 1 0
  */
  std::stringstream iss(str_val);
  int number;
  std::vector<int> partition_status;
  while ( iss >> number )
    partition_status.push_back(number);
  if (partition_status.size() <= partition_id) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
              << ", the sysfs line " << str_val
              << " does not have the partition_id "
              << partition_id;
              LOG_ERROR(ss);
              return RSMI_STATUS_UNEXPECTED_DATA;
  }
  *pisolate = partition_status[partition_id];
  return RSMI_STATUS_SUCCESS;
}

rsmi_status_t rsmi_dev_process_isolation_set(uint32_t dv_ind,
                             uint32_t pisolate) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX
  GET_DEV_FROM_INDX

  // To set the values,need to specify the setting for all of the partitions
  // For two partition
  // echo "1 0"  | sudo tee  /sys/class/drm/cardX/device/enforce_isolation
  uint32_t partition_id = 0;
  rsmi_dev_partition_id_get(dv_ind, &partition_id);
  std::string str_val;
  rsmi_status_t ret = get_dev_value_line(amd::smi::kDevProcessIsolation, dv_ind, &str_val);
  if (ret == RSMI_STATUS_FILE_ERROR) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", get_dev_value_str() ret was RSMI_STATUS_FILE_ERROR "
       << "-> reporting RSMI_STATUS_NOT_SUPPORTED";
    LOG_ERROR(ss);
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", get_dev_value_str() ret was not RSMI_STATUS_SUCCESS"
       << " -> reporting " << amd::smi::getRSMIStatusString(ret);
    LOG_ERROR(ss);
    return ret;
  }

  // craft the string need to be writeen.
  // (1) parse the read enforce_isolation data into a vector
  std::stringstream iss(str_val);
  int number;
  std::vector<int> partition_status;
  while ( iss >> number ) {
    partition_status.push_back(number);
  }

  // (2) Validate the data
  if (partition_status.size() <= partition_id) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
              << ", the sysfs line " << str_val
              << " does not have the partition_id "
              << partition_id;
    LOG_ERROR(ss);
    return RSMI_STATUS_UNEXPECTED_DATA;
  }

  // (3) Create the complete list with the update
  partition_status[partition_id] = pisolate;
  std::stringstream result;
  std::copy(partition_status.begin(), partition_status.end(),
        std::ostream_iterator<int>(result, " "));

  std::string value = result.str().c_str();
  int write_ret = dev->writeDevInfo(amd::smi::kDevProcessIsolation , value);
  return amd::smi::ErrnoToRsmiStatus(write_ret);

  CATCH
}

rsmi_status_t rsmi_dev_gpu_run_cleaner_shader(uint32_t dv_ind) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX
  GET_DEV_FROM_INDX

  // To reset you need to provide the partition id
  // echo "0" | sudo tee  /sys/class/drm/cardX/device/run_cleaner_shader
  uint32_t partition_id = 0;
  rsmi_dev_partition_id_get(dv_ind, &partition_id);
  std::string value = std::to_string(partition_id);
  int ret = dev->writeDevInfo(amd::smi::kDevShaderClean , value);
  return amd::smi::ErrnoToRsmiStatus(ret);

  CATCH
}

rsmi_status_t
rsmi_dev_xgmi_plpd_get(uint32_t dv_ind,
                      rsmi_dpm_policy_t* policy) {
  rsmi_status_t ret;
  std::vector<std::string> val_vec;

  if (policy == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  *policy = {};

  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);
  DEVICE_MUTEX

  ret = GetDevValueVec(amd::smi::kDevXgmiPlpd, dv_ind, &val_vec);
  if (ret == RSMI_STATUS_FILE_ERROR) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", GetDevValueVec() ret was RSMI_STATUS_FILE_ERROR "
       << "-> reporting RSMI_STATUS_NOT_SUPPORTED";
    LOG_ERROR(ss);
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", GetDevValueVec() ret was not RSMI_STATUS_SUCCESS"
       << " -> reporting " << amd::smi::getRSMIStatusString(ret);
    LOG_ERROR(ss);
    return ret;
  }
  /*
    It will reply on the number but no string as it may vary from soc to soc.
    The current xmgi plpd marked with *
      0 : plpd_disallow
      1 : plpd_default*
      2 : plpd_optimized
  */
  bool see_current = false;
  policy->num_supported = 0;
  for (uint32_t i = 0; i < val_vec.size(); ++i) {
    auto current_line = amd::smi::trim(val_vec[i]);

    // Get tokens: <integer> : <string *>
    std::vector<std::string> tokens;
    std::istringstream f(current_line);
    std::string s;
    while (getline(f, s, ':')) {
          tokens.push_back(s);
    }

    int value = 0;
    // At the end
    if (tokens.size() < 2 || !amd::smi::stringToInteger(tokens[0], value)) {
      break;
    }

    if (value < 0 || policy->num_supported >= RSMI_MAX_NUM_PM_POLICIES) {
      ss << __PRETTY_FUNCTION__ << " | ======= end ======="
          << ", Unexpected pstat data: the id is negative or too many plpd policies.";
          LOG_ERROR(ss);
          return RSMI_STATUS_UNEXPECTED_DATA;
    }

    policy->policies[policy->num_supported].policy_id = value;
    std::string description = amd::smi::trim(tokens[1]);
    if (current_line.back() == '*') {  // current policy
        description.pop_back();  // remove last *
        description = amd::smi::trim(description);
        policy->current = policy->num_supported;
        see_current = true;
    }
    strncpy(policy->policies[policy->num_supported].policy_description,
          description.c_str(),
          RSMI_MAX_POLICY_NAME-1);
    policy->num_supported++;
  }  //  end for

  if (!see_current) {
      ss << __PRETTY_FUNCTION__ << " | ======= end ======="
          << ", Unexpected pstat data: cannot find the current xgmi_plpd policy.";
          LOG_ERROR(ss);
          return RSMI_STATUS_UNEXPECTED_DATA;
  }

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_xgmi_plpd_set(uint32_t dv_ind,
                      uint32_t plpd_id) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX
  GET_DEV_FROM_INDX

  // Need to add new line character
  std::string value = std::to_string(plpd_id) + "\n";
  int ret = dev->writeDevInfo(amd::smi::kDevXgmiPlpd , value);
  return amd::smi::ErrnoToRsmiStatus(ret);

  CATCH
}

rsmi_status_t
rsmi_dev_soc_pstate_get(uint32_t dv_ind,
                      rsmi_dpm_policy_t* policy) {
  rsmi_status_t ret;
  std::vector<std::string> val_vec;

  if (policy == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  *policy = {};

  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);
  DEVICE_MUTEX

  ret = GetDevValueVec(amd::smi::kDevSocPstate, dv_ind, &val_vec);
  if (ret == RSMI_STATUS_FILE_ERROR) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", GetDevValueVec() ret was RSMI_STATUS_FILE_ERROR "
       << "-> reporting RSMI_STATUS_NOT_SUPPORTED";
    LOG_ERROR(ss);
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__ << " | ======= end ======="
       << ", GetDevValueVec() ret was not RSMI_STATUS_SUCCESS"
       << " -> reporting " << amd::smi::getRSMIStatusString(ret);
    LOG_ERROR(ss);
    return ret;
  }
  /*
    It will reply on the number but no string as it may vary from soc to soc.
    The current pstate marked with *
    0 : soc_pstate_default
    1 : soc_pstate_0
    2 : soc_pstate_1*
    3 : soc_pstate_2
  */
  bool see_current = false;
  policy->num_supported = 0;
  for (uint32_t i = 0; i < val_vec.size(); ++i) {
    auto current_line = amd::smi::trim(val_vec[i]);

    // Get tokens: <integer> : <string *>
    std::vector<std::string> tokens;
    std::istringstream f(current_line);
    std::string s;
    while (getline(f, s, ':')) {
          tokens.push_back(s);
    }

    int value = 0;
    // At the end
    if (tokens.size() < 2 || !amd::smi::stringToInteger(tokens[0], value)) {
      break;
    }

    if (value < 0 || policy->num_supported >= RSMI_MAX_NUM_PM_POLICIES) {
      ss << __PRETTY_FUNCTION__ << " | ======= end ======="
          << ", Unexpected pstat data: the id is negative or too many policies.";
          LOG_ERROR(ss);
          return RSMI_STATUS_UNEXPECTED_DATA;
    }

    policy->policies[policy->num_supported].policy_id = value;
    std::string description = amd::smi::trim(tokens[1]);
    if (current_line.back() == '*') {  // current policy
        description.pop_back();  // remove last *
        description = amd::smi::trim(description);
        policy->current = policy->num_supported;
        see_current = true;
    }
    strncpy(policy->policies[policy->num_supported].policy_description,
          description.c_str(),
          RSMI_MAX_POLICY_NAME-1);
    policy->num_supported++;
  }  //  end for

  if (!see_current) {
      ss << __PRETTY_FUNCTION__ << " | ======= end ======="
          << ", Unexpected pstat data: cannot find the current policy.";
          LOG_ERROR(ss);
          return RSMI_STATUS_UNEXPECTED_DATA;
  }
  // Cannot find it
  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_soc_pstate_set(uint32_t dv_ind,
                      uint32_t policy_id) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX
  GET_DEV_FROM_INDX

  // need to add new line character
  std::string value = std::to_string(policy_id) + "\n";
  int ret = dev->writeDevInfo(amd::smi::kDevSocPstate , value);
  return amd::smi::ErrnoToRsmiStatus(ret);

  CATCH
}

static std::vector<std::string> pci_name_files = {
  "/usr/share/misc/pci.ids",
  "/usr/share/hwdata/pci.ids",
  "/usr/share/pci.ids",
  "/var/lib/pciutils/pci.ids"
};

enum eNameStrType {
  NAME_STR_VENDOR = 0,
  NAME_STR_DEVICE,
  NAME_STR_SUBSYS
};

static const std::map<std::string, rsmi_compute_partition_type_t>
mapStringToRSMIComputePartitionTypes {
  {"CPX", RSMI_COMPUTE_PARTITION_CPX},
  {"SPX", RSMI_COMPUTE_PARTITION_SPX},
  {"DPX", RSMI_COMPUTE_PARTITION_DPX},
  {"TPX", RSMI_COMPUTE_PARTITION_TPX},
  {"QPX", RSMI_COMPUTE_PARTITION_QPX}
};

static const std::map<rsmi_compute_partition_type_t, std::string>
mapRSMIToStringComputePartitionTypes {
  {RSMI_COMPUTE_PARTITION_INVALID, "UNKNOWN"},
  {RSMI_COMPUTE_PARTITION_CPX, "CPX"},
  {RSMI_COMPUTE_PARTITION_SPX, "SPX"},
  {RSMI_COMPUTE_PARTITION_DPX, "DPX"},
  {RSMI_COMPUTE_PARTITION_TPX, "TPX"},
  {RSMI_COMPUTE_PARTITION_QPX, "QPX"}
};

static const std::map<rsmi_memory_partition_type_t, std::string>
mapRSMIToStringMemoryPartitionTypes {
  {RSMI_MEMORY_PARTITION_UNKNOWN, "UNKNOWN"},
  {RSMI_MEMORY_PARTITION_NPS1, "NPS1"},
  {RSMI_MEMORY_PARTITION_NPS2, "NPS2"},
  {RSMI_MEMORY_PARTITION_NPS4, "NPS4"},
  {RSMI_MEMORY_PARTITION_NPS8, "NPS8"}
};

static const std::map<std::string, rsmi_memory_partition_type_t>
mapStringToMemoryPartitionTypes {
  {"NPS1", RSMI_MEMORY_PARTITION_NPS1},
  {"NPS2", RSMI_MEMORY_PARTITION_NPS2},
  {"NPS4", RSMI_MEMORY_PARTITION_NPS4},
  {"NPS8", RSMI_MEMORY_PARTITION_NPS8}
};

static std::string
get_id_name_str_from_line(uint64_t id, std::string ln,
                                                 std::istringstream *ln_str) {
  std::string token1;
  std::string ret_str;

  assert(ln_str != nullptr);
  THROW_IF_NULLPTR_DEREF(ln_str)

  *ln_str >> token1;

  if (token1.empty()) {
    throw amd::smi::rsmi_exception(RSMI_STATUS_NO_DATA, __FUNCTION__);
  }

  if (std::stoul(token1, nullptr, 16) == id) {
    int64_t pos = ln_str->tellg();

    assert(pos >= 0);
    if (pos < 0) {
      throw amd::smi::rsmi_exception(
          RSMI_STATUS_UNEXPECTED_DATA, __FUNCTION__);
    }
    size_t s_pos = ln.find_first_not_of("\t ", static_cast<size_t>(pos));
    ret_str = ln.substr(static_cast<uint32_t>(s_pos));
  }
  return ret_str;
}

static rsmi_status_t get_backup_name(uint16_t id, char *name, size_t len) {
  std::string name_str;

  assert(name != nullptr);
  if (name == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  name_str += "0x";

  std::stringstream strm;
  strm << std::hex << id;
  name_str += strm.str();

  name[0] = '\0';
  size_t ct = name_str.copy(name, len);

  name[std::min(len - 1, ct)] = '\0';

  if (len < (name_str.size() + 1)) {
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  return RSMI_STATUS_SUCCESS;
}

static rsmi_status_t get_dev_name_from_file(uint32_t dv_ind, char *name,
                                               size_t len) {
  std::string val_str;
  rsmi_status_t ret =
              get_dev_value_line(amd::smi::kDevDevProdName, dv_ind, &val_str);

  if (ret != 0) {
    return amd::smi::ErrnoToRsmiStatus(ret);
  }
  size_t ct = val_str.copy(name, len);

  name[std::min(len - 1, ct)] = '\0';

  if (len < (val_str.size() + 1)) {
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }

  return RSMI_STATUS_SUCCESS;
}

// return empty if cannot find it in pci.ids
static std::string get_vendor_name_from_id(uint16_t vendor_id) {
  for (auto fl : pci_name_files) {
    std::ifstream id_file_strm(fl);
    std::string ln;

    while (std::getline(id_file_strm, ln)) {
      // parse line
      if (ln == ""  || ln[0] == '#' || ln[0] == '\t') {
        continue;
      }
      std::istringstream ln_str(ln);
      std::string val_str = get_id_name_str_from_line(vendor_id, ln, &ln_str);

      if (val_str != "") {
        return val_str;
      }
    }
  }
  return "";
}

// Parse pci.ids files. Comment lines have # in first column. Otherwise,
// Syntax:
// vendor  vendor_name
//       device  device_name                             <-- single tab
//               subvendor subdevice  subsystem_name     <-- two tabs
static rsmi_status_t get_dev_name_from_id(uint32_t dv_ind, char *name,
                                               size_t len, eNameStrType typ) {
  std::string ln;
  std::string token1;
  rsmi_status_t ret;
  uint16_t device_id;
  uint16_t vendor_id;
  uint16_t subsys_vend_id;
  uint16_t subsys_id;
  bool found_device_vendor = false;
  // to match subsystem, it must match the device id at previous line
  bool found_device_id_for_subsys = false;
  std::string val_str;

  assert(name != nullptr);
  assert(len > 0);

  if (name == nullptr || len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  name[0] = '\0';

  ret = rsmi_dev_vendor_id_get(dv_ind, &vendor_id);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  if (typ != NAME_STR_VENDOR) {
    ret = rsmi_dev_id_get(dv_ind, &device_id);
    if (ret != RSMI_STATUS_SUCCESS) {
      return ret;
    }
    if (typ != NAME_STR_DEVICE) {
      ret = rsmi_dev_subsystem_vendor_id_get(dv_ind, &subsys_vend_id);
      if (ret != RSMI_STATUS_SUCCESS) {
        return ret;
      }
      ret = rsmi_dev_subsystem_id_get(dv_ind, &subsys_id);
      if (ret != RSMI_STATUS_SUCCESS) {
        return ret;
      }
    }
  }

  for (const auto& fl : pci_name_files) {
    std::ifstream id_file_strm(fl);

    while (std::getline(id_file_strm, ln)) {
      std::istringstream ln_str(ln);
      // parse line
      if (ln[0] == '#' || ln.empty()) {
        continue;
      }

      if (ln[0] == '\t') {
        if (found_device_vendor) {
          if (ln[1] == '\t') {
            // The subsystem line, ignore a line if the device id not match
            if (typ == NAME_STR_SUBSYS && found_device_id_for_subsys) {
              val_str = get_id_name_str_from_line(subsys_vend_id, ln, &ln_str);

              if (!val_str.empty()) {
                // We've chopped the subsys_vend ID, now we need to get the
                // subsys description
                val_str = get_id_name_str_from_line(subsys_id, ln, &ln_str);

                if (!val_str.empty()) {
                  break;
                }
                val_str.clear();
              }
            }
          } else if (typ == NAME_STR_DEVICE) {  // ln[1] != '\t'
            // This is a device line
            val_str = get_id_name_str_from_line(device_id, ln, &ln_str);

            if (!val_str.empty()) {
              break;
            }
          } else if (typ == NAME_STR_SUBSYS) {
            // match the device id line
            val_str = get_id_name_str_from_line(device_id, ln, &ln_str);
            if (!val_str.empty()) {
              found_device_id_for_subsys = true;
            }
          }
        }
      } else {  // ln[0] != '\t'; Vendor line
        if (found_device_vendor) {
          assert(typ != NAME_STR_VENDOR);
          // We already found the vendor but didn't find the device or
          // subsystem we were looking for, so bail out.
          val_str.clear();

          return get_backup_name(typ == NAME_STR_DEVICE ?
                                            device_id : subsys_id, name, len);
        }

        val_str = get_id_name_str_from_line(vendor_id, ln, &ln_str);

        if (!val_str.empty()) {
          if (typ == NAME_STR_VENDOR) {
            break;
          }
          val_str.clear();
          found_device_vendor = true;
        }
      }
    }
    if (!val_str.empty()) {
      break;
    }
  }

  if (val_str.empty()) {
    return get_backup_name(vendor_id, name, len);
  }
  size_t ct = val_str.copy(name, len);

  name[std::min(len - 1, ct)] = '\0';

  if (len < (val_str.size() + 1)) {
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }

  return RSMI_STATUS_SUCCESS;
}

static rsmi_status_t
get_dev_drm_render_minor(uint32_t dv_ind, uint32_t *minor) {
  GET_DEV_FROM_INDX

  assert(minor != nullptr);
  if (minor == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  *minor = dev->drm_render_minor();
  if (*minor)
    return RSMI_STATUS_SUCCESS;

  return RSMI_STATUS_INIT_ERROR;
}

rsmi_status_t
rsmi_dev_name_get(uint32_t dv_ind, char *name, size_t len) {
  rsmi_status_t ret;

  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(name)

  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX

  ret = get_dev_name_from_file(dv_ind, name, len);

  if (ret || name[0] == '\0' || !isprint(name[0]) ) {
    ret = get_dev_name_from_id(dv_ind, name, len, NAME_STR_DEVICE);
  }

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_pcie_vendor_name_get(uint32_t dv_ind, char *name, size_t len) {
  rsmi_status_t ret;

  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(name)

  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX
  uint16_t id = 0;
  ret = get_id(dv_ind, amd::smi::kDevPCieVendorID, &id);
  if (ret != RSMI_STATUS_SUCCESS) return ret;
  std::string vendor_name = get_vendor_name_from_id(id);

  if (vendor_name == "") {
    return RSMI_STATUS_NOT_FOUND;
  }
  memset(name, 0, len);
  strncpy(name, vendor_name.c_str(), len-1);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_brand_get(uint32_t dv_ind, char *brand, uint32_t len) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(brand)
  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  DEVICE_MUTEX

  std::map<std::string, std::string> brand_names = {
    {"D05121", "mi25"},
    {"D05131", "mi25"},
    {"D05133", "mi25"},
    {"D05151", "mi25"},
    {"D16304", "mi50"},
    {"D16302", "mi60"}
  };
  std::map<std::string, std::string>::iterator it;
  std::string vbios_value;
  std::string sku_value;
  // Retrieve vbios and store in vbios_value string
  int ret = dev->readDevInfo(amd::smi::kDevVBiosVer, &vbios_value);
  if (ret != 0) {
    return amd::smi::ErrnoToRsmiStatus(ret);
  }
  if (vbios_value.length() == 16) {
    sku_value = vbios_value.substr(4, 6);
    // Find the brand name using sku_value
    it = brand_names.find(sku_value);
    if (it != brand_names.end()) {
      uint32_t ln = static_cast<uint32_t>(it->second.copy(brand, len));
      brand[std::min(len - 1, ln)] = '\0';

      if (len < (it->second.size() + 1)) {
        return RSMI_STATUS_INSUFFICIENT_SIZE;
      }

      return RSMI_STATUS_SUCCESS;
    }
  }
  // If there is no SKU match, return marketing name instead
  rsmi_dev_name_get(dv_ind, brand, len);
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_vram_vendor_get(uint32_t dv_ind, char *brand, uint32_t len) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(brand)

  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  std::string val_str;
  DEVICE_MUTEX
  int ret = dev->readDevInfo(amd::smi::kDevVramVendor, &val_str);

  if (ret != 0) {
    return amd::smi::ErrnoToRsmiStatus(ret);
  }

  uint32_t ln = static_cast<uint32_t>(val_str.copy(brand, len));

  brand[std::min(len - 1, ln)] = '\0';

  if (len < (val_str.size() + 1)) {
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_subsystem_name_get(uint32_t dv_ind, char *name, size_t len) {
  rsmi_status_t ret;

  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(name)

  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX

  ret = get_dev_name_from_id(dv_ind, name, len, NAME_STR_SUBSYS);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_drm_render_minor_get(uint32_t dv_ind, uint32_t *minor) {
  rsmi_status_t ret;

  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(minor)

  DEVICE_MUTEX
  ret = get_dev_drm_render_minor(dv_ind, minor);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_vendor_name_get(uint32_t dv_ind, char *name, size_t len) {
  rsmi_status_t ret;

  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(name)

  assert(len > 0);
  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX
  ret = get_dev_name_from_id(dv_ind, name, len, NAME_STR_VENDOR);
  return ret;
  CATCH
}

rsmi_status_t rsmi_dev_pm_metrics_info_get(uint32_t dv_ind,
                      rsmi_name_value_t** pm_metrics,
                      uint32_t *num_of_metrics) {
  TRY
  DEVICE_MUTEX
  CHK_SUPPORT_NAME_ONLY(num_of_metrics)
  std::string file_path = dev->
          get_sys_file_path_by_type(amd::smi::kDevPmMetrics);

  int ret = amd::smi::present_pmmetrics(
          file_path.c_str(), pm_metrics, num_of_metrics);
  if (ret == 0) return RSMI_STATUS_SUCCESS;
  return RSMI_STATUS_NOT_SUPPORTED;

  CATCH
}

rsmi_status_t rsmi_dev_reg_table_info_get(uint32_t dv_ind,
                      rsmi_reg_type_t reg_type,
                      rsmi_name_value_t** reg_metrics,
                      uint32_t *num_of_metrics) {
  TRY
  DEVICE_MUTEX
  CHK_SUPPORT_NAME_ONLY(num_of_metrics)
  std::string file_path = dev->
          get_sys_file_path_by_type(amd::smi::kDevRegMetrics);

  int ret = amd::smi::present_reg_state(
          file_path.c_str(), reg_type, reg_metrics, num_of_metrics);
  if (ret == 0) return RSMI_STATUS_SUCCESS;
  return RSMI_STATUS_NOT_SUPPORTED;

  CATCH
}

rsmi_status_t
rsmi_dev_pci_bandwidth_get(uint32_t dv_ind, rsmi_pcie_bandwidth_t *b) {
  rsmi_status_t ret;
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  GET_DEV_AND_KFDNODE_FROM_INDX
  CHK_API_SUPPORT_ONLY((b), RSMI_DEFAULT_VARIANT, RSMI_DEFAULT_VARIANT)
  DEVICE_MUTEX
  ret = get_frequencies(amd::smi::kDevPCIEClk, RSMI_CLK_TYPE_PCIE, dv_ind,
                                        &b->transfer_rate, b->lanes);
  if (ret == RSMI_STATUS_SUCCESS) {
    return ret;
  }

  // Only fallback to gpu_metric if connecting via PCIe
  if (kfd_node->numa_node_type() != amd::smi::IOLINK_TYPE_PCIEXPRESS) {
    return ret;
  }

  rsmi_gpu_metrics_t gpu_metrics;
  ret = rsmi_dev_gpu_metrics_info_get(dv_ind, &gpu_metrics);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  // Hardcode based on PCIe specification: search PCI_Express on wikipedia
  const uint32_t link_width[] = {1, 2, 4, 8, 12, 16};
  const uint32_t link_speed[] = {25, 50, 80, 160};  // 0.1 Ghz
  const uint32_t WIDTH_DATA_LENGTH = sizeof(link_width)/sizeof(uint32_t);
  const uint32_t SPEED_DATA_LENGTH = sizeof(link_speed)/sizeof(uint32_t);

  // Calculate the index
  int32_t width_index = -1;
  int32_t speed_index = -1;
  uint32_t cur_index = 0;
  for (cur_index = 0; cur_index < WIDTH_DATA_LENGTH; cur_index++) {
    if (link_width[cur_index] == gpu_metrics.pcie_link_width) {
      width_index = static_cast<int32_t>(cur_index);
      break;
    }
  }
  for (cur_index = 0; cur_index < SPEED_DATA_LENGTH; cur_index++) {
    if (link_speed[cur_index] == gpu_metrics.pcie_link_speed) {
      speed_index = static_cast<int32_t>(cur_index);
      break;
    }
  }
  if (width_index == -1 || speed_index == -1) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  // Set possible lanes and frequencies
  b->transfer_rate.num_supported = WIDTH_DATA_LENGTH * SPEED_DATA_LENGTH;
  b->transfer_rate.current = static_cast<uint32_t>(speed_index)*WIDTH_DATA_LENGTH + static_cast<uint32_t>(width_index);
  for (cur_index = 0; cur_index < WIDTH_DATA_LENGTH * SPEED_DATA_LENGTH; cur_index++) {
    b->transfer_rate.frequency[cur_index] =
      static_cast<long>(link_speed[cur_index/WIDTH_DATA_LENGTH]) * 100 * 1000000L;
    b->lanes[cur_index] = link_width[cur_index % WIDTH_DATA_LENGTH];
  }
  /*
  frequency = {2500, 2500, 2500, 2500, 2500, 2500,
              5000, 5000, 5000, 5000, 5000, 5000,
              8000, 8000, 8000, 8000, 8000, 8000,
              16000, 16000, 16000, 16000, 16000, 16000};  // Mhz
  lanes = {1, 2, 4, 8, 12, 16,
              1, 2, 4, 8, 12, 16,
              1, 2, 4, 8, 12, 16,
              1, 2, 4, 8, 12, 16 };  // For each frequency
  */

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_pci_bandwidth_set(uint32_t dv_ind, uint64_t bw_bitmask) {
  rsmi_status_t ret;
  rsmi_pcie_bandwidth_t bws;

  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX
  // Bare Metal only feature
  if (amd::smi::is_vm_guest()) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  ret = rsmi_dev_pci_bandwidth_get(dv_ind, &bws);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  assert(bws.transfer_rate.num_supported <= RSMI_MAX_NUM_FREQUENCIES);

  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  // Above call to rsmi_dev_pci_bandwidth_get() should have emitted an error
  // if assert below is not true
  assert(dv_ind < smi.devices().size());

  std::string freq_enable_str =
         bitfield_to_freq_string(bw_bitmask, bws.transfer_rate.num_supported);

  std::shared_ptr<amd::smi::Device> dev = smi.devices()[dv_ind];
  assert(dev != nullptr);

  ret = rsmi_dev_perf_level_set_v1(dv_ind, RSMI_DEV_PERF_LEVEL_MANUAL);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  int32_t ret_i;
  ret_i = dev->writeDevInfo(amd::smi::kDevPCIEClk, freq_enable_str);
  //
  // NOTE:  kDevPCIEClk sysfs file maybe not exist for all cases.
  //        If it doesn't exist (pp_dpm_pcie), it shouldn't be an error
  //        and will get translated to RSMI_STATUS_NOT_SUPPORTED.
  return amd::smi::ErrnoToRsmiStatus(ret_i);

  CATCH
}

rsmi_status_t
rsmi_dev_pci_throughput_get(uint32_t dv_ind, uint64_t *sent,
                                   uint64_t *received, uint64_t *max_pkt_sz) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  rsmi_status_t ret;
  std::string val_str;

  // We don't do CHK_SUPPORT_NAME_ONLY in this case as the user may
  // choose to have any of the inout parameters as 0. Let the return code from
  // get_dev_value_line() tell if this function is supported or not.
  // CHK_SUPPORT_NAME_ONLY(...)

  DEVICE_MUTEX

  ret = get_dev_value_line(amd::smi::kDevPCIEThruPut, dv_ind, &val_str);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  std::istringstream fs_rng(val_str);

  if (sent) {
    fs_rng >> *sent;
  }
  if (received) {
    fs_rng >> *received;
  }
  if (max_pkt_sz) {
    fs_rng >> *max_pkt_sz;
  }

  if ((sent && *sent == UINT64_MAX) || (received && *received == UINT64_MAX)){
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_temp_metric_get(uint32_t dv_ind, uint32_t sensor_type,
                       rsmi_temperature_metric_t metric, int64_t *temperature) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  rsmi_status_t ret;
  amd::smi::MonitorTypes mon_type = amd::smi::kMonInvalid;
  uint16_t val_ui16;

  static const std::map<rsmi_temperature_metric_t, amd::smi::MonitorTypes>
    kMetricTypeMap = {
      { RSMI_TEMP_CURRENT, amd::smi::kMonTemp },
      { RSMI_TEMP_MAX, amd::smi::kMonTempMax },
      { RSMI_TEMP_MIN, amd::smi::kMonTempMin },
      { RSMI_TEMP_MAX_HYST, amd::smi::kMonTempMaxHyst },
      { RSMI_TEMP_MIN_HYST, amd::smi::kMonTempMinHyst },
      { RSMI_TEMP_CRITICAL, amd::smi::kMonTempCritical },
      { RSMI_TEMP_CRITICAL_HYST, amd::smi::kMonTempCriticalHyst },
      { RSMI_TEMP_EMERGENCY, amd::smi::kMonTempEmergency },
      { RSMI_TEMP_EMERGENCY_HYST, amd::smi::kMonTempEmergencyHyst },
      { RSMI_TEMP_CRIT_MIN, amd::smi::kMonTempCritMin },
      { RSMI_TEMP_CRIT_MIN_HYST, amd::smi::kMonTempCritMinHyst },
      { RSMI_TEMP_OFFSET, amd::smi::kMonTempOffset },
      { RSMI_TEMP_LOWEST, amd::smi::kMonTempLowest },
      { RSMI_TEMP_HIGHEST, amd::smi::kMonTempHighest },
  };

  const auto mon_type_it = kMetricTypeMap.find(metric);
  if (mon_type_it != kMetricTypeMap.end()) {
    mon_type = mon_type_it->second;
  }

  if (temperature == nullptr) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: " << monitorTypesToString.at(mon_type)
       << " | Cause: temperature was a null ptr reference"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS) << " |";
    LOG_ERROR(ss);
    return RSMI_STATUS_INVALID_ARGS;
  }

  // The HBM temperature is retrieved from the gpu_metrics
  if (sensor_type == RSMI_TEMP_TYPE_HBM_0 ||
      sensor_type == RSMI_TEMP_TYPE_HBM_1 ||
      sensor_type == RSMI_TEMP_TYPE_HBM_2 ||
      sensor_type == RSMI_TEMP_TYPE_HBM_3) {
    if (metric != RSMI_TEMP_CURRENT) {   // only support RSMI_TEMP_CURRENT
      ss << __PRETTY_FUNCTION__
         << " | ======= end ======= "
         << " | Fail "
         << " | Device #: " << dv_ind
         << " | Type: " << monitorTypesToString.at(mon_type)
         << " | Cause: To retrieve HBM temp, we only support metric = "
         << "RSMI_TEMP_CURRENT"
         << " | Returning = "
         << getRSMIStatusString(RSMI_STATUS_NOT_SUPPORTED) << " |";
      LOG_ERROR(ss);
      return RSMI_STATUS_NOT_SUPPORTED;
    }

    rsmi_gpu_metrics_t gpu_metrics;
    ret = rsmi_dev_gpu_metrics_info_get(dv_ind, &gpu_metrics);
    if (ret != RSMI_STATUS_SUCCESS) {
      ss << __PRETTY_FUNCTION__
         << " | ======= end ======= "
         << " | Fail "
         << " | Device #: " << dv_ind
         << " | Type: " << monitorTypesToString.at(mon_type)
         << " | Cause: rsmi_dev_gpu_metrics_info_get returned "
         << getRSMIStatusString(ret)
         << " | Returning = "
         << getRSMIStatusString(ret) << " |";
      LOG_ERROR(ss);
      return ret;
    }

    switch (sensor_type) {
      case RSMI_TEMP_TYPE_HBM_0:
        val_ui16 = gpu_metrics.temperature_hbm[0];
        break;
      case RSMI_TEMP_TYPE_HBM_1:
        val_ui16 = gpu_metrics.temperature_hbm[1];
        break;
      case RSMI_TEMP_TYPE_HBM_2:
        val_ui16 = gpu_metrics.temperature_hbm[2];
        break;
      case RSMI_TEMP_TYPE_HBM_3:
        val_ui16 = gpu_metrics.temperature_hbm[3];
        break;
      default:
        return RSMI_STATUS_INVALID_ARGS;
    }
    if (val_ui16 == UINT16_MAX) {
      ss << __PRETTY_FUNCTION__
         << " | ======= end ======= "
         << " | Fail "
         << " | Device #: " << dv_ind
         << " | Type: " << monitorTypesToString.at(mon_type)
         << " | Cause: Reached UINT16 max value, overflow"
         << " | Returning = "
         << getRSMIStatusString(RSMI_STATUS_NOT_SUPPORTED) << " |";
      LOG_ERROR(ss);
      return RSMI_STATUS_NOT_SUPPORTED;
    }

    *temperature =
      static_cast<int64_t>(val_ui16) * CENTRIGRADE_TO_MILLI_CENTIGRADE;

    ss << __PRETTY_FUNCTION__ << " | ======= end ======= "
       << " | Success "
       << " | Device #: " << dv_ind
       << " | Type: " << monitorTypesToString.at(mon_type)
       << " | Data: " << *temperature
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " | ";
    LOG_INFO(ss);
    return RSMI_STATUS_SUCCESS;
  }  // end HBM temperature

  DEVICE_MUTEX

  GET_DEV_FROM_INDX

  if (dev->monitor() == nullptr) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: " << monitorTypesToString.at(mon_type)
       << " | Cause: monitor returned nullptr"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_NOT_SUPPORTED) << " |";
    LOG_ERROR(ss);
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  std::shared_ptr<amd::smi::Monitor> m = dev->monitor();

  // getTempSensorIndex will throw an out of range exception if sensor_type is
  // not found
  uint32_t sensor_index =
     m->getTempSensorIndex(static_cast<rsmi_temperature_type_t>(sensor_type));


  CHK_API_SUPPORT_ONLY(temperature, metric, sensor_index)

  ret = get_dev_mon_value(mon_type, dv_ind, sensor_index, temperature);
  ss << __PRETTY_FUNCTION__ << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Sensor_index: " << sensor_index
     << " | Type: " << monitorTypesToString.at(mon_type)
     << " | Data: " << *temperature
     << " | Returning = "
     << getRSMIStatusString(ret) << " | ";
  LOG_INFO(ss);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_volt_metric_get(uint32_t dv_ind, rsmi_voltage_type_t sensor_type,
                       rsmi_voltage_metric_t metric, int64_t *voltage) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  rsmi_status_t ret;
  amd::smi::MonitorTypes mon_type;

  switch (metric) {
    case RSMI_VOLT_CURRENT:
      mon_type = amd::smi::kMonVolt;
      break;
    case RSMI_VOLT_MIN:
      mon_type = amd::smi::kMonVoltMin;
      break;
    case RSMI_VOLT_MIN_CRIT:
      mon_type = amd::smi::kMonVoltMinCrit;
      break;
    case RSMI_VOLT_MAX:
      mon_type = amd::smi::kMonVoltMax;
      break;
    case RSMI_VOLT_MAX_CRIT:
      mon_type = amd::smi::kMonVoltMaxCrit;
      break;
    case RSMI_VOLT_AVERAGE:
      mon_type = amd::smi::kMonVoltAverage;
      break;
    case RSMI_VOLT_LOWEST:
      mon_type = amd::smi::kMonVoltLowest;
      break;
    case RSMI_VOLT_HIGHEST:
      mon_type = amd::smi::kMonVoltHighest;
      break;
    default:
      mon_type = amd::smi::kMonInvalid;
  }

  DEVICE_MUTEX

  GET_DEV_FROM_INDX

  if (dev->monitor() == nullptr) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  std::shared_ptr<amd::smi::Monitor> m = dev->monitor();

  // getVoltSensorIndex will throw an out of range exception if sensor_type is
  // not found
  uint32_t sensor_index;
  try {
    sensor_index =
      m->getVoltSensorIndex(sensor_type);
  } catch (...) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  CHK_API_SUPPORT_ONLY(voltage, metric, sensor_index)

  ret = get_dev_mon_value(mon_type, dv_ind, sensor_index, voltage);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_fan_speed_get(uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  rsmi_status_t ret;

  ++sensor_ind;  // fan sysfs files have 1-based indices

  CHK_SUPPORT_SUBVAR_ONLY(speed, sensor_ind)

  DEVICE_MUTEX

  ret = get_dev_mon_value(amd::smi::kMonFanSpeed, dv_ind, sensor_ind, speed);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_fan_rpms_get(uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  ++sensor_ind;  // fan sysfs files have 1-based indices

  CHK_SUPPORT_SUBVAR_ONLY(speed, sensor_ind)

  rsmi_status_t ret;

  DEVICE_MUTEX

  ret = get_dev_mon_value(amd::smi::kMonFanRPMs, dv_ind, sensor_ind, speed);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_fan_reset(uint32_t dv_ind, uint32_t sensor_ind) {
  TRY
  rsmi_status_t ret;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  ++sensor_ind;  // fan sysfs files have 1-based indices
  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX
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
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX

  // Bare Metal only feature
  if (amd::smi::is_vm_guest()) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

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
  rsmi_status_t ret;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  ++sensor_ind;  // fan sysfs files have 1-based indices
  CHK_SUPPORT_SUBVAR_ONLY(max_speed, sensor_ind)
  DEVICE_MUTEX

  ret = get_dev_mon_value(amd::smi::kMonMaxFanSpeed, dv_ind, sensor_ind,
                                      reinterpret_cast<int64_t *>(max_speed));

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_od_volt_info_get(uint32_t dv_ind, rsmi_od_volt_freq_data_t *odv) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  DEVICE_MUTEX
  CHK_SUPPORT_NAME_ONLY(odv)
  rsmi_status_t ret = get_od_clk_volt_info(dv_ind, odv);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_gpu_reset(uint32_t dv_ind) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS
  // No longer using DEVICE_MUTEX as it blocks long running processes
  // DEVICE_MUTEX

  rsmi_status_t ret;
  uint64_t status_code = 0;

  // Read amdgpu_gpu_recover to reset it
  ret = get_dev_value_int(amd::smi::kDevGpuReset, dv_ind, &status_code);
  return ret;

  CATCH
}

rsmi_status_t rsmi_dev_od_volt_curve_regions_get(uint32_t dv_ind,
                     uint32_t *num_regions, rsmi_freq_volt_region_t *buffer) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======";
  LOG_TRACE(ss);

  CHK_SUPPORT_NAME_ONLY((num_regions == nullptr || buffer == nullptr) ?
                                                        nullptr : num_regions)
  if (*num_regions == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX
  rsmi_status_t ret = get_od_clk_volt_curve_regions(dv_ind, num_regions,
                                                                      buffer);
  if (*num_regions == 0) {
    ret = RSMI_STATUS_NOT_SUPPORTED;
  }
  ss << __PRETTY_FUNCTION__ << " | ======= end ======= | returning "
     << getRSMIStatusString(ret);
  LOG_TRACE(ss);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_max_get(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  (void)sensor_ind;  // Not used yet
  // ++sensor_ind;  // power sysfs files have 1-based indices
  CHK_SUPPORT_NAME_ONLY(power)

  rsmi_status_t ret;

  DEVICE_MUTEX
  ret = get_power_mon_value(amd::smi::kPowerMaxGPUPower, dv_ind, power);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_ave_get(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  ++sensor_ind;  // power sysfs files have 1-based indices

  CHK_SUPPORT_SUBVAR_ONLY(power, sensor_ind)
  rsmi_status_t ret;

  DEVICE_MUTEX
  ret = get_dev_mon_value(amd::smi::kMonPowerAve, dv_ind, sensor_ind, power);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_current_socket_power_get(uint32_t dv_ind, uint64_t *socket_power) {
  TRY
  std::ostringstream ss;
  rsmi_status_t rsmiReturn = RSMI_STATUS_NOT_SUPPORTED;
  std::string val_str;
  uint32_t sensor_ind = 1;  // socket_power sysfs files have 1-based indices
  amd::smi::MonitorTypes mon_type = amd::smi::kMonPowerInput;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, dv_ind="
     << std::to_string(dv_ind);
  LOG_TRACE(ss);
  if (socket_power == nullptr) {
    rsmiReturn = RSMI_STATUS_INVALID_ARGS;
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: " << monitorTypesToString.at(mon_type)
       << " | Cause: socket_power was a null ptr reference"
       << " | Returning = "
       << getRSMIStatusString(rsmiReturn) << " |";
    LOG_ERROR(ss);
    return RSMI_STATUS_INVALID_ARGS;
  }
  CHK_SUPPORT_SUBVAR_ONLY(socket_power, sensor_ind)
  DEVICE_MUTEX

  if (dev->monitor() == nullptr) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: " << monitorTypesToString.at(mon_type)
       << " | Cause: hwmon monitor was a null ptr reference"
       << " | Returning = "
       << getRSMIStatusString(rsmiReturn) << " |";
    LOG_ERROR(ss);
    return rsmiReturn;
  }

  int ret = dev->monitor()->readMonitor(amd::smi::kMonPowerLabel,
                                        sensor_ind, &val_str);
  if (ret || val_str != "PPT" || val_str.size() != 3) {
    if (ret != 0) {
      rsmiReturn = amd::smi::ErrnoToRsmiStatus(ret);
    }
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: " << monitorTypesToString.at(mon_type)
       << " | Cause: readMonitor() returned an error status"
       << " or Socket Power label did not show PPT or size of label data was"
       << " unexpected"
       << " | Returning = "
       << getRSMIStatusString(rsmiReturn) << " |";
    LOG_ERROR(ss);
    return rsmiReturn;
  }
  rsmiReturn = get_dev_mon_value(mon_type, dv_ind, sensor_ind,
                                 socket_power);
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: " << monitorTypesToString.at(mon_type)
     << " | Data: " << *socket_power
     << " | Returning = "
     << getRSMIStatusString(rsmiReturn) << " |";
  LOG_TRACE(ss);
  return rsmiReturn;
  CATCH
}

rsmi_status_t rsmi_dev_power_get(uint32_t dv_ind, uint64_t *power,
                                 RSMI_POWER_TYPE *type) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, dv_ind="
     << std::to_string(dv_ind);
  LOG_TRACE(ss);
  rsmi_status_t ret = RSMI_STATUS_NOT_SUPPORTED;
  RSMI_POWER_TYPE temp_power_type = RSMI_INVALID_POWER;
  uint64_t temp_power = 0;

  if (type == nullptr || power == nullptr) {
    ret = RSMI_STATUS_INVALID_ARGS;
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: " << amd::smi::power_type_string(temp_power_type)
       << " | Cause: power or monitor type was a null ptr reference"
       << " | Returning = "
       << getRSMIStatusString(ret) << " |";
    LOG_ERROR(ss);
    return ret;
  }

  // only change return value on success, invalid otherwise
  rsmi_status_t temp_ret = rsmi_dev_current_socket_power_get(dv_ind, &temp_power);
  if (temp_ret == RSMI_STATUS_SUCCESS) {
    temp_power_type = RSMI_CURRENT_POWER;
    ret = temp_ret;
  } else {
    temp_ret = rsmi_dev_power_ave_get(dv_ind, 0, &temp_power);
    if (temp_ret == RSMI_STATUS_SUCCESS) {
      temp_power_type = RSMI_AVERAGE_POWER;
      ret = temp_ret;
    }
  }
  *power = temp_power;
  *type = temp_power_type;
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: " << amd::smi::power_type_string(temp_power_type)
     << " | Data: " << *power
     << " | Returning = "
     << getRSMIStatusString(ret) << " |";
  LOG_TRACE(ss);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_energy_count_get(uint32_t dv_ind, uint64_t *power,
                          float *counter_resolution, uint64_t *timestamp) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  if (power == nullptr ||
      timestamp == nullptr) {
      return RSMI_STATUS_INVALID_ARGS;
  }

  rsmi_status_t ret;
  rsmi_gpu_metrics_t gpu_metrics;
  ret = rsmi_dev_gpu_metrics_info_get(dv_ind, &gpu_metrics);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  *power = gpu_metrics.energy_accumulator;
  *timestamp = gpu_metrics.system_clock_counter;
  // hard-coded for now since all ASICs have same resolution. If it ASIC
  // dependent then this information should come from Kernel
  if (counter_resolution)
    *counter_resolution = kEnergyCounterResolution;

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_cap_default_get(uint32_t dv_ind, uint64_t *default_cap) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  uint32_t sensor_ind = 1; // power sysfs files have 1-based indices
  CHK_SUPPORT_SUBVAR_ONLY(default_cap, sensor_ind)

  rsmi_status_t ret;

  DEVICE_MUTEX
  ret = get_dev_mon_value(amd::smi::kMonPowerCapDefault, dv_ind, sensor_ind, default_cap);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_cap_get(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *cap) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  ++sensor_ind;  // power sysfs files have 1-based indices
  CHK_SUPPORT_SUBVAR_ONLY(cap, sensor_ind)

  rsmi_status_t ret;

  DEVICE_MUTEX
  ret = get_dev_mon_value(amd::smi::kMonPowerCap, dv_ind, sensor_ind, cap);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_cap_range_get(uint32_t dv_ind, uint32_t sensor_ind,
                                               uint64_t *max, uint64_t *min) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  ++sensor_ind;  // power sysfs files have 1-based indices
  CHK_SUPPORT_SUBVAR_ONLY((min == nullptr || max == nullptr ?nullptr : min),
                                                                   sensor_ind)
  rsmi_status_t ret;

  DEVICE_MUTEX
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
  uint64_t min;
  uint64_t max;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX

  // Bare Metal only feature
  if (amd::smi::is_vm_guest()) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

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
rsmi_dev_power_profile_presets_get(uint32_t dv_ind, uint32_t reserved,
                                        rsmi_power_profile_status_t *status) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  (void)reserved;
  CHK_SUPPORT_NAME_ONLY(status)

  DEVICE_MUTEX
  rsmi_status_t ret = get_power_profiles(dv_ind, status, nullptr);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_profile_set(uint32_t dv_ind, uint32_t dummy,
                                  rsmi_power_profile_preset_masks_t profile) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS

  (void)dummy;
  DEVICE_MUTEX
  // Bare Metal only feature
  if (amd::smi::is_vm_guest()) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  rsmi_status_t ret = set_power_profile(dv_ind, profile);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_memory_total_get(uint32_t dv_ind, rsmi_memory_type_t mem_type,
                                                            uint64_t *total) {
  TRY
  rsmi_status_t ret;
  amd::smi::DevInfoTypes mem_type_file;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  CHK_SUPPORT_VAR(total, mem_type)

  switch (mem_type) {
    case RSMI_MEM_TYPE_GTT:
      mem_type_file = amd::smi::kDevMemTotGTT;
      break;

    case RSMI_MEM_TYPE_VIS_VRAM:
      mem_type_file = amd::smi::kDevMemTotVisVRAM;
      break;

    case RSMI_MEM_TYPE_VRAM:
      mem_type_file = amd::smi::kDevMemTotVRAM;
      break;

    default:
      assert(false);  // Unexpected memory type
      return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX
  ret = get_dev_value_int(mem_type_file, dv_ind, total);

  // Fallback to KFD reported memory if VRAM total is 0
  if (mem_type == RSMI_MEM_TYPE_VRAM && *total == 0) {
    GET_DEV_AND_KFDNODE_FROM_INDX
    if (kfd_node->get_total_memory(total) == 0 && *total > 0) {
      ss << __PRETTY_FUNCTION__
         << " | inside success fallback... "
         << " | Device #: " << std::to_string(dv_ind)
         << " | Type = " << amd::smi::Device::get_type_string(mem_type_file)
         << " | Data: total = " << std::to_string(*total)
         << " | ret = " << getRSMIStatusString(RSMI_STATUS_SUCCESS);
      LOG_DEBUG(ss);
      return RSMI_STATUS_SUCCESS;
    }
  }

  ss << __PRETTY_FUNCTION__
     << " | after fallback... "
     << " | Device #: " << std::to_string(dv_ind)
     << " | Type = " << amd::smi::Device::get_type_string(mem_type_file)
     << " | Data: total = " << std::to_string(*total)
     << " | ret = " << getRSMIStatusString(ret);
  LOG_DEBUG(ss);
  return ret;
  CATCH
}


rsmi_status_t rsmi_dev_cache_info_get(
      uint32_t dv_ind, rsmi_gpu_cache_info_t *info) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  if (info == nullptr) return RSMI_STATUS_INVALID_ARGS;

  GET_DEV_AND_KFDNODE_FROM_INDX

  if (kfd_node->get_cache_info(info) == 0) return RSMI_STATUS_SUCCESS;

  return RSMI_STATUS_NOT_SUPPORTED;
  CATCH
}

rsmi_status_t
rsmi_dev_memory_usage_get(uint32_t dv_ind, rsmi_memory_type_t mem_type,
                                                              uint64_t *used) {
  TRY
  rsmi_status_t ret;
  amd::smi::DevInfoTypes mem_type_file;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  CHK_SUPPORT_VAR(used, mem_type)

  switch (mem_type) {
    case RSMI_MEM_TYPE_GTT:
      mem_type_file = amd::smi::kDevMemUsedGTT;
      break;

    case RSMI_MEM_TYPE_VIS_VRAM:
      mem_type_file = amd::smi::kDevMemUsedVisVRAM;
      break;

    case RSMI_MEM_TYPE_VRAM:
      mem_type_file = amd::smi::kDevMemUsedVRAM;
      break;

    default:
      assert(false);  // Unexpected memory type
      return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX
  ret = get_dev_value_int(mem_type_file, dv_ind, used);

  // Fallback to KFD reported memory if no VRAM
  if (mem_type == RSMI_MEM_TYPE_VRAM && *used == 0) {
    GET_DEV_AND_KFDNODE_FROM_INDX
    uint64_t total = 0;
    ret = get_dev_value_int(amd::smi::kDevMemTotVRAM, dv_ind, &total);
    if (total != 0) {
      ss << __PRETTY_FUNCTION__
         << " no fallback needed! - "
         << " | Device #: " << std::to_string(dv_ind)
         << " | Type = " << amd::smi::Device::get_type_string(mem_type_file)
         << " | Data: Used = " << std::to_string(*used)
         << " | Data: total = " << std::to_string(total)
         << " | ret = " << getRSMIStatusString(ret);
    LOG_DEBUG(ss);
      return ret;  // do not need to fallback
    }
    if ( kfd_node->get_used_memory(used) == 0 ) {
      ss << __PRETTY_FUNCTION__
         << " | in fallback == success ..."
         << " | Device #: " << std::to_string(dv_ind)
         << " | Type = " << amd::smi::Device::get_type_string(mem_type_file)
         << " | Data: Used = " << std::to_string(*used)
         << " | Data: total = " << std::to_string(total)
         << " | ret = " << getRSMIStatusString(RSMI_STATUS_SUCCESS);
      LOG_DEBUG(ss);
      return RSMI_STATUS_SUCCESS;
    }
  }
  ss << __PRETTY_FUNCTION__
     << " | at end!!!! after fallback ..."
     << " | Device #: " << std::to_string(dv_ind)
     << " | Type = " << amd::smi::Device::get_type_string(mem_type_file)
     << " | Data: Used = " << std::to_string(*used)
     << " | ret = " << getRSMIStatusString(ret);
  LOG_DEBUG(ss);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_memory_busy_percent_get(uint32_t dv_ind, uint32_t *busy_percent) {
  TRY
  rsmi_status_t ret;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  CHK_SUPPORT_NAME_ONLY(busy_percent)

  uint64_t tmp_util = 0;

  DEVICE_MUTEX
  ret = get_dev_value_int(amd::smi::kDevMemBusyPercent, dv_ind, &tmp_util);

  if (tmp_util > 100) {
    return RSMI_STATUS_UNEXPECTED_DATA;
  }
  *busy_percent = static_cast<uint32_t>(tmp_util);
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
      *status_string = "RSMI_STATUS_OUT_OF_RESOURCES: Unable to acquire memory "
                       "or other resource";
      break;

    case RSMI_STATUS_INTERNAL_EXCEPTION:
      *status_string = "RSMI_STATUS_INTERNAL_EXCEPTION: An internal exception "
                       "was caught";
      break;

    case RSMI_STATUS_INPUT_OUT_OF_BOUNDS:
      *status_string = "RSMI_STATUS_INPUT_OUT_OF_BOUNDS: The provided input is "
                       "out of allowable or safe range";
      break;

    case RSMI_STATUS_INIT_ERROR:
      *status_string = "RSMI_STATUS_INIT_ERROR: An error occurred during "
                       "initialization, during monitor discovery or when when "
                       "initializing internal data structures";
      break;

    case RSMI_STATUS_NOT_YET_IMPLEMENTED:
      *status_string = "RSMI_STATUS_NOT_YET_IMPLEMENTED: The called function "
                        "has not been implemented in this system for this "
                        "device type";
      break;

    case RSMI_STATUS_NOT_FOUND:
      *status_string = "RSMI_STATUS_NOT_FOUND: An item required to complete "
                       "the call was not found";
      break;

    case RSMI_STATUS_INSUFFICIENT_SIZE:
      *status_string = "RSMI_STATUS_INSUFFICIENT_SIZE: Not enough resources "
                       "were available to fully execute the call";
      break;

    case RSMI_STATUS_INTERRUPT:
      *status_string = "RSMI_STATUS_INTERRUPT: An interrupt occurred while "
                       "executing the function";
      break;

    case RSMI_STATUS_UNEXPECTED_SIZE:
      *status_string = "RSMI_STATUS_UNEXPECTED_SIZE: Data (usually from reading"
                       " a file) was out of range from what was expected";
      break;

    case RSMI_STATUS_NO_DATA:
      *status_string = "RSMI_STATUS_NO_DATA: No data was found (usually from "
                       "reading a file) where data was expected";
      break;

    case RSMI_STATUS_UNEXPECTED_DATA:
      *status_string = "RSMI_STATUS_UNEXPECTED_DATA: Data read (usually from "
                       "a file) or provided to function is "
                       "not what was expected";
      break;

    case RSMI_STATUS_BUSY:
      *status_string = "RSMI_STATUS_BUSY: A resource or mutex could not be "
                        "acquired because it is already being used";
    break;

    case RSMI_STATUS_REFCOUNT_OVERFLOW:
      *status_string = "RSMI_STATUS_REFCOUNT_OVERFLOW: An internal reference "
                       "counter exceeded INT32_MAX";
      break;

    case RSMI_STATUS_SETTING_UNAVAILABLE:
      *status_string = "RSMI_STATUS_SETTING_UNAVAILABLE: Requested setting is "
                        "unavailable for the current device";
      break;

    case RSMI_STATUS_AMDGPU_RESTART_ERR:
      *status_string = "RSMI_STATUS_AMDGPU_RESTART_ERR: Could not successfully "
                        "restart the amdgpu driver";
      break;

    case RSMI_STATUS_UNKNOWN_ERROR:
      *status_string = "RSMI_STATUS_UNKNOWN_ERROR: An unknown error prevented "
                       "the call from completing successfully";
      break;

    default:
      *status_string = "RSMI_STATUS_UNKNOWN_ERROR: An unknown error occurred";
      return RSMI_STATUS_UNKNOWN_ERROR;
  }
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_busy_percent_get(uint32_t dv_ind, uint32_t *busy_percent) {
  TRY
  std::string val_str;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  CHK_SUPPORT_NAME_ONLY(busy_percent)

  DEVICE_MUTEX
  rsmi_status_t ret = get_dev_value_str(amd::smi::kDevUsage, dv_ind,
                                                                    &val_str);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  errno = 0;
  *busy_percent = static_cast<uint32_t>(strtoul(val_str.c_str(), nullptr, 10));

  if (*busy_percent > 100) {
    return RSMI_STATUS_UNEXPECTED_DATA;
  }
  assert(errno == 0);

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_utilization_count_get(uint32_t dv_ind,
                rsmi_utilization_counter_t utilization_counters[],
                uint32_t count,
                uint64_t *timestamp) {
  TRY

  if (timestamp == nullptr ||
      utilization_counters == nullptr) {
      return RSMI_STATUS_INVALID_ARGS;
  }

  rsmi_status_t ret;
  rsmi_gpu_metrics_t gpu_metrics;
  uint16_t val_counter(0);

  ret = rsmi_dev_gpu_metrics_info_get(dv_ind, &gpu_metrics);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  for (uint32_t index = 0 ; index < count; index++) {
    switch (utilization_counters[index].type) {

      case RSMI_COARSE_GRAIN_GFX_ACTIVITY:
      case RSMI_FINE_GRAIN_GFX_ACTIVITY:
        val_counter = 1;
        utilization_counters[index].value = gpu_metrics.gfx_activity_acc;
        utilization_counters[index].fine_value[0] =
          (gpu_metrics.gfx_activity_acc != std::numeric_limits<decltype(gpu_metrics.gfx_activity_acc)>::max())
            ? gpu_metrics.gfx_activity_acc : std::numeric_limits<uint64_t>::max();
        utilization_counters[index].fine_value_count =
            (gpu_metrics.gfx_activity_acc == std::numeric_limits<decltype(gpu_metrics.gfx_activity_acc)>::max())
            ? 0 : val_counter;
        break;

      case RSMI_COARSE_GRAIN_MEM_ACTIVITY:
      case RSMI_FINE_GRAIN_MEM_ACTIVITY:
        val_counter = 1;
        utilization_counters[index].value = gpu_metrics.mem_activity_acc;
        utilization_counters[index].fine_value[0] =
          (gpu_metrics.mem_activity_acc != std::numeric_limits<decltype(gpu_metrics.mem_activity_acc)>::max())
            ? gpu_metrics.mem_activity_acc : std::numeric_limits<uint64_t>::max();
        utilization_counters[index].fine_value_count =
            (gpu_metrics.mem_activity_acc == std::numeric_limits<decltype(gpu_metrics.mem_activity_acc)>::max())
            ? 0 : val_counter;
        break;

      case RSMI_COARSE_DECODER_ACTIVITY:
      case RSMI_FINE_DECODER_ACTIVITY:
        {
          auto value_count = uint16_t(0);
          auto value_accum = uint64_t(0);
          for (const auto& elem : gpu_metrics.vcn_activity) {
            if (elem != std::numeric_limits<uint16_t>::max()) {
              ++value_count;
              value_accum += elem;
            }

            if (utilization_counters[index].type == RSMI_UTILIZATION_COUNTER_TYPE::RSMI_FINE_DECODER_ACTIVITY) {
              utilization_counters[index].fine_value[value_count] = elem;
            }
          }

          utilization_counters[index].value = 0;
          utilization_counters[index].fine_value_count = value_count;
          if (utilization_counters[index].type == RSMI_UTILIZATION_COUNTER_TYPE::RSMI_COARSE_DECODER_ACTIVITY) {
            if (value_count > 0) {
              utilization_counters[index].value = (value_accum / value_count);
            }
          }
        }
        break;

      default:
        return RSMI_STATUS_INVALID_ARGS;
    }
  }
  *timestamp = gpu_metrics.system_clock_counter;

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_activity_metric_get(uint32_t dv_ind,
                             rsmi_activity_metric_t activity_metric_type,
                             rsmi_activity_metric_counter_t* activity_metric_counter) {

  TRY
  std::ostringstream ostrstream;
  ostrstream << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ostrstream);

  if (!activity_metric_counter) {
    ostrstream << __PRETTY_FUNCTION__
               << " | ======= end ======= "
               << " | Fail "
               << " | Device #: " << dv_ind
               << " | Metric Type: " << activity_metric_type
               << " | Cause: rsmi_activity_metric_counter_t was a null ptr reference"
               << " | Returning = "
               << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS) << " |";
    LOG_ERROR(ostrstream);
    return rsmi_status_t::RSMI_STATUS_INVALID_ARGS;
  }

  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  rsmi_gpu_metrics_t gpu_metrics;
  status_code = rsmi_dev_gpu_metrics_info_get(dv_ind, &gpu_metrics);
  if (status_code != rsmi_status_t::RSMI_STATUS_SUCCESS) {
    ostrstream << __PRETTY_FUNCTION__
               << " | ======= end ======= "
               << " | Fail "
               << " | Device #: " << dv_ind
               << " | Metric Type: " << activity_metric_type
               << " | Cause: rsmi_dev_gpu_metrics_info_get returned "
               << getRSMIStatusString(status_code)
               << " | Returning = "
               << status_code << " |";
    LOG_ERROR(ostrstream);
    return status_code;
  }

  if (activity_metric_type & rsmi_activity_metric_t::RSMI_ACTIVITY_GFX) {
    activity_metric_counter->average_gfx_activity = gpu_metrics.average_gfx_activity;
    ostrstream << __PRETTY_FUNCTION__
               << " | For GFX: " << activity_metric_counter->average_gfx_activity;
    LOG_INFO(ostrstream);
  }
  if (activity_metric_type & rsmi_activity_metric_t::RSMI_ACTIVITY_UMC) {
    activity_metric_counter->average_umc_activity = gpu_metrics.average_umc_activity;
    ostrstream << __PRETTY_FUNCTION__
               << " | For UMC: " << activity_metric_counter->average_umc_activity;
    LOG_INFO(ostrstream);
  }
  if (activity_metric_type & rsmi_activity_metric_t::RSMI_ACTIVITY_MM) {
    activity_metric_counter->average_mm_activity  = gpu_metrics.average_mm_activity;
    ostrstream << __PRETTY_FUNCTION__
               << " | For MM: " << activity_metric_counter->average_mm_activity;
    LOG_INFO(ostrstream);
  }

  ostrstream << __PRETTY_FUNCTION__
             << " | ======= end ======= "
             << " | Success "
             << " | Device #: " << dv_ind
             << " | Metric Type: " << activity_metric_type
             << " | Returning = "
             << getRSMIStatusString(status_code) << " |";
  LOG_INFO(ostrstream);

  return status_code;
  CATCH
}

rsmi_status_t
rsmi_dev_activity_avg_mm_get(uint32_t dv_ind, uint16_t* avg_activity) {

  TRY
  std::ostringstream ostrstream;
  ostrstream << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ostrstream);

  if (!avg_activity) {
    ostrstream << __PRETTY_FUNCTION__
               << " | ======= end ======= "
               << " | Fail "
               << " | Device #: " << dv_ind
               << " | Metric Type: " << rsmi_activity_metric_t::RSMI_ACTIVITY_MM
               << " | Cause: avg_activity was a null ptr reference"
               << " | Returning = "
               << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS) << " |";
    LOG_ERROR(ostrstream);
    return rsmi_status_t::RSMI_STATUS_INVALID_ARGS;
  }

  auto status_code(rsmi_status_t::RSMI_STATUS_SUCCESS);
  rsmi_activity_metric_counter_t activity_metric_counter;
  status_code = rsmi_dev_activity_metric_get(dv_ind, rsmi_activity_metric_t::RSMI_ACTIVITY_MM, &activity_metric_counter);
  avg_activity = &activity_metric_counter.average_mm_activity;

  ostrstream << __PRETTY_FUNCTION__
             << " | ======= end ======= "
             << " | Success "
             << " | Device #: " << dv_ind
             << " | Metric Type: " << rsmi_activity_metric_t::RSMI_ACTIVITY_MM
             << " | Returning = "
             << getRSMIStatusString(status_code) << " |";
  LOG_INFO(ostrstream);

  return status_code;
  CATCH
}

rsmi_status_t
rsmi_dev_vbios_version_get(uint32_t dv_ind, char *vbios, uint32_t len) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(vbios)

  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::string val_str;

  DEVICE_MUTEX
  int ret = dev->readDevInfo(amd::smi::kDevVBiosVer, &val_str);

  if (ret != 0) {
    return amd::smi::ErrnoToRsmiStatus(ret);
  }

  uint32_t ln = static_cast<uint32_t>(val_str.copy(vbios, len));

  vbios[std::min(len - 1, ln)] = '\0';

  if (len < (val_str.size() + 1)) {
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_version_get(rsmi_version_t *version) {
  TRY

  if (version == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  version->major = rocm_smi_VERSION_MAJOR;
  version->minor = rocm_smi_VERSION_MINOR;
  version->patch = rocm_smi_VERSION_PATCH;
  version->build = rocm_smi_VERSION_BUILD;

  return RSMI_STATUS_SUCCESS;

  CATCH
}

static const char *kROCmDriverVersionPath = "/sys/module/amdgpu/version";

rsmi_status_t
rsmi_version_str_get(rsmi_sw_component_t component, char *ver_str,
                                                               uint32_t len) {
  if (ver_str == nullptr || len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  TRY

  int err;
  std::string val_str;
  std::string ver_path;

  switch (component) {
    case RSMI_SW_COMP_DRIVER:
      ver_path = kROCmDriverVersionPath;
      break;

    default:
      assert(false);  // Unexpected component type provided
      return RSMI_STATUS_INVALID_ARGS;
  }

  err = amd::smi::ReadSysfsStr(ver_path, &val_str);

  if (err != 0) {
    struct utsname buf;
    err = uname(&buf);

    if (err != 0) {
      return amd::smi::ErrnoToRsmiStatus(err);
    }

    val_str = buf.release;
  }

  uint32_t ln = static_cast<uint32_t>(val_str.copy(ver_str, len));

  ver_str[std::min(len - 1, ln)] = '\0';

  if (len < (val_str.size() + 1)) {
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t rsmi_dev_serial_number_get(uint32_t dv_ind,
                                             char *serial_num, uint32_t len) {
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(serial_num)
  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  TRY
  DEVICE_MUTEX

  std::string val_str;
  rsmi_status_t ret = get_dev_value_str(amd::smi::kDevSerialNumber,
                                                            dv_ind, &val_str);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  uint32_t ln = static_cast<uint32_t>(val_str.copy(serial_num, len));

  serial_num[std::min(len - 1, ln)] = '\0';

  if (len < (val_str.size() + 1)) {
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_pci_replay_counter_get(uint32_t dv_ind, uint64_t *counter) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(counter)

  rsmi_status_t ret;

  DEVICE_MUTEX
  ret = get_dev_value_int(amd::smi::kDevPCIEReplayCount, dv_ind, counter);
  return ret;

  CATCH
}

rsmi_status_t
rsmi_dev_unique_id_get(uint32_t dv_ind, uint64_t *unique_id) {
  TRY
  rsmi_status_t ret;
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  CHK_SUPPORT_NAME_ONLY(unique_id)

  DEVICE_MUTEX
  if (unique_id == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  *unique_id = std::numeric_limits<uint64_t>::max();
  ret = get_dev_value_int(amd::smi::kDevUniqueId, dv_ind, unique_id);

  ss << __PRETTY_FUNCTION__
     << (ret == RSMI_STATUS_SUCCESS ?
      " | No fall back needed retrieved from KGD" : " | fall back needed")
     << " | Device #: " << std::to_string(dv_ind)
     << " | Data: unique_id = " << std::to_string(*unique_id)
     << " | ret = " << getRSMIStatusString(ret, false);
  LOG_DEBUG(ss);
  // If the unique ID is not supported, use KFD's unique ID
  if (ret != RSMI_STATUS_SUCCESS) {
    GET_DEV_AND_KFDNODE_FROM_INDX
    uint32_t node_id;
    uint64_t kfd_unique_id;
    int ret_kfd = kfd_node->get_node_id(&node_id);
    ret_kfd = amd::smi::read_node_properties(node_id, "unique_id", &kfd_unique_id);
    if (ret_kfd == 0) {
      *unique_id = kfd_unique_id;
      ret = RSMI_STATUS_SUCCESS;
    } else {
      *unique_id = std::numeric_limits<uint64_t>::max();
      ret = RSMI_STATUS_NOT_SUPPORTED;
    }
    ss << __PRETTY_FUNCTION__
       << " | Issue: Could not read unique_id from sysfs, falling back to KFD" << "\n"
       << " ; Device #: " << std::to_string(dv_ind) << "\n"
       << " ; ret_kfd: " << std::to_string(ret_kfd) << "\n"
       << " ; node: " << std::to_string(node_id) << "\n"
       << " ; Data: unique_id (from KFD)= " << std::to_string(*unique_id) << "\n"
       << " ; ret = " << getRSMIStatusString(ret, false);
    LOG_DEBUG(ss);
  }
  return ret;

  CATCH
}

rsmi_status_t
rsmi_dev_counter_create(uint32_t dv_ind, rsmi_event_type_t type,
                                           rsmi_event_handle_t *evnt_handle) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS

  // Note we don't need to pass in the variant to CHK_SUPPORT_VAR because
  // the success of this call doesn't depend on a sysfs file existing.
  CHK_SUPPORT_NAME_ONLY(evnt_handle)
  DEVICE_MUTEX
  *evnt_handle = reinterpret_cast<uintptr_t>(
                                      new amd::smi::evt::Event(type, dv_ind));

  if (evnt_handle == nullptr) {
    return RSMI_STATUS_OUT_OF_RESOURCES;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_counter_destroy(rsmi_event_handle_t evnt_handle) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  if (evnt_handle == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  int ret = 0;
  amd::smi::evt::Event *evt =
                        reinterpret_cast<amd::smi::evt::Event *>(evnt_handle);
  uint32_t dv_ind = evt->dev_ind();
  DEVICE_MUTEX
  REQUIRE_ROOT_ACCESS

  ret = evt->stopCounter();

  delete evt;
  return amd::smi::ErrnoToRsmiStatus(ret);;
  CATCH
}

rsmi_status_t
rsmi_counter_control(rsmi_event_handle_t evt_handle,
                                 rsmi_counter_command_t cmd, void * /*unused*/) {
  TRY

  amd::smi::evt::Event *evt =
                         reinterpret_cast<amd::smi::evt::Event *>(evt_handle);
  amd::smi::pthread_wrap _pw(*amd::smi::GetMutex(evt->dev_ind()));
  amd::smi::ScopedPthread _lock(_pw);

  REQUIRE_ROOT_ACCESS

  int ret = 0;

  if (evt_handle == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  switch (cmd) {
    case RSMI_CNTR_CMD_START:
      ret = evt->startCounter();
      break;

    case RSMI_CNTR_CMD_STOP:
      ret = evt->stopCounter();
      break;

    default:
      assert(false);  // Unexpected perf counter command
      return RSMI_STATUS_INVALID_ARGS;
  }
  return amd::smi::ErrnoToRsmiStatus(ret);

  CATCH
}

rsmi_status_t
rsmi_counter_read(rsmi_event_handle_t evt_handle,
                                                rsmi_counter_value_t *value) {
  TRY

  if (value == nullptr || evt_handle == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  amd::smi::evt::Event *evt =
                         reinterpret_cast<amd::smi::evt::Event *>(evt_handle);

  uint32_t dv_ind = evt->dev_ind();
  DEVICE_MUTEX
  REQUIRE_ROOT_ACCESS

  uint32_t ret;

  ret = evt->getValue(value);

  // If value > 2^48, then an overflow has occurred. We need to discard this
  // value and re-read:
  if (ret == 0 && value->value > 0xFFFFFFFFFFFF) {
    ret = evt->getValue(value);
  }
  if (ret == 0) {
    return RSMI_STATUS_SUCCESS;
  }

  return RSMI_STATUS_UNEXPECTED_SIZE;
  CATCH
}

rsmi_status_t
rsmi_counter_available_counters_get(uint32_t dv_ind,
                                rsmi_event_group_t grp, uint32_t *available) {
  rsmi_status_t ret;

  TRY
  CHK_SUPPORT_VAR(available, grp)
  DEVICE_MUTEX
  uint64_t val = 0;

  switch (grp) {
    case RSMI_EVNT_GRP_XGMI:
    case RSMI_EVNT_GRP_XGMI_DATA_OUT:

      ret = get_dev_value_int(amd::smi::kDevDFCountersAvailable, dv_ind, &val);
      if (ret != RSMI_STATUS_SUCCESS)
        return ret;
      if (val == UINT32_MAX)
        return RSMI_STATUS_NOT_SUPPORTED;
      *available = static_cast<uint32_t>(val);
      break;

    default:
      return RSMI_STATUS_INVALID_ARGS;
  }
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_counter_group_supported(uint32_t dv_ind, rsmi_event_group_t group) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  DEVICE_MUTEX
  GET_DEV_FROM_INDX

  amd::smi::evt::dev_evt_grp_set_t *grp = dev->supported_event_groups();

  if (grp->find(group) == grp->end()) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_compute_process_info_get(rsmi_process_info_t *procs,
                                                        uint32_t *num_items) {
  TRY

  if (num_items == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  uint32_t procs_found = 0;

  int err = amd::smi::GetProcessInfo(procs, *num_items, &procs_found);

  if (err) {
    return amd::smi::ErrnoToRsmiStatus(err);
  }

  if (procs && *num_items < procs_found) {
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  if (procs == nullptr || *num_items > procs_found) {
    *num_items = procs_found;
  }

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_compute_process_gpus_get(uint32_t pid, uint32_t *dv_indices,
                                                      uint32_t *num_devices) {
  TRY

  if (num_devices == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::unordered_set<uint64_t> gpu_set;
  int err = amd::smi::GetProcessGPUs(pid, &gpu_set);

  if (err) {
    return amd::smi::ErrnoToRsmiStatus(err);
  }

  uint32_t i = 0;
  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  // filter out the devices not visible to container
  auto& nodes = smi.kfd_node_map();
  for (auto nit = gpu_set.begin(); nit != gpu_set.end();) {
    uint64_t gpu_id_val = (*nit);
    auto kfd_ite = nodes.find(gpu_id_val);
    if (kfd_ite == nodes.end()) {
      nit = gpu_set.erase(nit);
    } else {
      nit++;
    }
  }

  if (dv_indices != nullptr) {
    for (auto it = gpu_set.begin(); i < *num_devices && it != gpu_set.end();
                                                                  ++it, ++i) {
      uint64_t gpu_id_val = (*it);
      dv_indices[i] = nodes[gpu_id_val]->amdgpu_dev_index();
    }
  }

  if (dv_indices && *num_devices < gpu_set.size()) {
    // In this case, *num_devices should already hold the number of items
    // written to dv_devices. We just have to let the caller know there's more.
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }

  *num_devices = static_cast<uint32_t>(gpu_set.size());
  if (gpu_set.size() > smi.devices().size()) {
    return RSMI_STATUS_UNEXPECTED_SIZE;
  }

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_memory_reserved_pages_get(uint32_t dv_ind, uint32_t *num_pages,
                                          rsmi_retired_page_record_t *records) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  rsmi_status_t ret;
  CHK_SUPPORT_NAME_ONLY(num_pages)

  std::vector<std::string> val_vec;

  ret = GetDevValueVec(amd::smi::kDevMemPageBad, dv_ind, &val_vec);

  // file is empty, which is valid for no errors
  if (ret == RSMI_STATUS_UNEXPECTED_DATA) {
    ret = RSMI_STATUS_SUCCESS;
  }
  if (ret == RSMI_STATUS_FILE_ERROR) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  if (records == nullptr || *num_pages > val_vec.size()) {
    *num_pages = static_cast<uint32_t>(val_vec.size());
  }
  if (records == nullptr) {
    return RSMI_STATUS_SUCCESS;
  }

  // Fill in records
  char status_code;
  rsmi_memory_page_status_t tmp_stat;
  std::string junk;

  for (uint32_t i = 0; i < *num_pages; ++i) {
    std::istringstream fs1(val_vec[i]);

    fs1 >> std::hex >> records[i].page_address;
    fs1 >> junk;
    assert(junk == ":");
    fs1 >> std::hex >> records[i].page_size;
    fs1 >> junk;
    assert(junk == ":");
    fs1 >> status_code;

    switch (status_code) {
      case 'P':
        tmp_stat = RSMI_MEM_PAGE_STATUS_PENDING;
        break;

      case 'F':
        tmp_stat = RSMI_MEM_PAGE_STATUS_UNRESERVABLE;
        break;

      case 'R':
        tmp_stat = RSMI_MEM_PAGE_STATUS_RESERVED;
        break;
      default:
        assert(false);  // Unexpected retired memory page status code read
        return RSMI_STATUS_UNKNOWN_ERROR;
    }
    records[i].status = tmp_stat;
  }
  if (*num_pages < val_vec.size()) {
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_compute_process_info_by_pid_get(uint32_t pid,
                                                  rsmi_process_info_t *proc) {
  TRY

  if (proc == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::unordered_set<uint64_t> gpu_set;
  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
  auto it = smi.kfd_node_map().begin();

  while (it != smi.kfd_node_map().end()) {
    uint64_t gpu_id = it->first;
    gpu_set.insert(gpu_id);
    it++;
  }

  int err = amd::smi::GetProcessInfoForPID(pid, proc, &gpu_set);

  if (err) {
    return amd::smi::ErrnoToRsmiStatus(err);
  }

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_compute_process_info_by_device_get(uint32_t pid, uint32_t dv_ind,
                                          rsmi_process_info_t *proc) {
  TRY
  if (proc == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  // Check the device and kfdnode exist
  GET_DEV_AND_KFDNODE_FROM_INDX

  std::unordered_set<uint64_t> gpu_set;
  gpu_set.insert(dev->kfd_gpu_id());
  int err = amd::smi::GetProcessInfoForPID(pid, proc, &gpu_set);

  if (err) {
    return amd::smi::ErrnoToRsmiStatus(err);
  }

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_xgmi_error_status(uint32_t dv_ind, rsmi_xgmi_status_t *status) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(status)

  rsmi_status_t ret;
  uint64_t status_code;

  DEVICE_MUTEX
  ret = get_dev_value_int(amd::smi::kDevXGMIError, dv_ind, &status_code);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  switch (status_code) {
    case 0:
      *status = RSMI_XGMI_STATUS_NO_ERRORS;
      break;

    case 1:
      *status = RSMI_XGMI_STATUS_ERROR;
      break;

    case 2:
      *status = RSMI_XGMI_STATUS_MULTIPLE_ERRORS;
      break;

    default:
      assert(false);  // Unexpected XGMI error status read
      return RSMI_STATUS_UNKNOWN_ERROR;
  }
  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_xgmi_error_reset(uint32_t dv_ind) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  DEVICE_MUTEX

  rsmi_status_t ret;
  uint64_t status_code;

  // Reading xgmi_error resets it
  ret = get_dev_value_int(amd::smi::kDevXGMIError, dv_ind, &status_code);
  return ret;

  CATCH
}

rsmi_status_t
rsmi_dev_xgmi_hive_id_get(uint32_t dv_ind, uint64_t *hive_id) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  if (hive_id == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  GET_DEV_AND_KFDNODE_FROM_INDX

  *hive_id = kfd_node->xgmi_hive_id();

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_topo_get_numa_node_number(uint32_t dv_ind, uint32_t *numa_node) {
  TRY

  return topo_get_numa_node_number(dv_ind, numa_node);
  CATCH
}

rsmi_status_t
rsmi_topo_get_link_weight(uint32_t dv_ind_src, uint32_t dv_ind_dst,
                          uint64_t *weight) {
  TRY

  uint32_t dv_ind = dv_ind_src;
  GET_DEV_AND_KFDNODE_FROM_INDX
  DEVICE_MUTEX

  if (weight == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  rsmi_status_t status;
  uint32_t node_ind_dst;
  int ret = smi.get_node_index(dv_ind_dst, &node_ind_dst);

  if (ret == 0) {
    amd::smi::IO_LINK_TYPE type;
    ret = kfd_node->get_io_link_type(node_ind_dst, &type);
    if (ret == 0) {
      if (type == amd::smi::IOLINK_TYPE_XGMI) {
        ret = kfd_node->get_io_link_weight(node_ind_dst, weight);
        if (ret == 0)
          status = RSMI_STATUS_SUCCESS;
        else
          status = RSMI_STATUS_INIT_ERROR;
      } else {
        assert(false);  // Unexpected IO Link type read
        status = RSMI_STATUS_NOT_SUPPORTED;
      }
    } else if (kfd_node->numa_node_type() == amd::smi::IOLINK_TYPE_PCIEXPRESS) {
      *weight = kfd_node->numa_node_weight();  // from src GPU to it's CPU node
      uint64_t numa_weight_dst = 0;
      status = topo_get_numa_node_weight(dv_ind_dst, &numa_weight_dst);
      // from dst GPU to it's CPU node
      if (status == RSMI_STATUS_SUCCESS) {
        *weight = *weight + numa_weight_dst;
        uint32_t numa_number_src = kfd_node->numa_node_number();
        uint32_t numa_number_dst;
        status = topo_get_numa_node_number(dv_ind_dst, &numa_number_dst);
        if (status == RSMI_STATUS_SUCCESS) {
          if (numa_number_src != numa_number_dst) {
            uint64_t io_link_weight;
            ret = smi.get_io_link_weight(numa_number_src, numa_number_dst,
                                         &io_link_weight);
            if (ret == 0) {
              *weight = *weight + io_link_weight;
              // from src numa CPU node to dst numa CPU node
            } else {
              *weight = *weight + 10;
              // More than one CPU hops, hard coded 10
            }
          }
          status = RSMI_STATUS_SUCCESS;
        } else {
          assert(false);  // Error to read numa node number
          status = RSMI_STATUS_INIT_ERROR;
        }
      } else {
        assert(false);  // Error to read numa node weight
        status = RSMI_STATUS_INIT_ERROR;
      }
    } else {
      status = RSMI_STATUS_NOT_SUPPORTED;
    }
  } else {
    status = RSMI_STATUS_INVALID_ARGS;
  }

  return status;
  CATCH
}

 rsmi_status_t
 rsmi_minmax_bandwidth_get(uint32_t dv_ind_src, uint32_t dv_ind_dst,
                           uint64_t *min_bandwidth, uint64_t *max_bandwidth) {
  TRY

  uint32_t dv_ind = dv_ind_src;
  GET_DEV_AND_KFDNODE_FROM_INDX
  DEVICE_MUTEX

  if (min_bandwidth == nullptr || max_bandwidth == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  if (dv_ind_src == dv_ind_dst) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  rsmi_status_t status;
  uint32_t node_ind_dst;
  int ret = smi.get_node_index(dv_ind_dst, &node_ind_dst);

  if (ret != 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }


  amd::smi::IO_LINK_TYPE type;
  ret = kfd_node->get_io_link_type(node_ind_dst, &type);
  if ( ret == 0 && type == amd::smi::IOLINK_TYPE_XGMI) {
      ret = kfd_node->get_io_link_bandwidth(node_ind_dst,max_bandwidth,
                                                               min_bandwidth);
      if (ret == 0)
        status = RSMI_STATUS_SUCCESS;
      else
        status = RSMI_STATUS_INIT_ERROR;
  } else {  // from src GPU to it's CPU node, or type not XGMI
    status = RSMI_STATUS_NOT_SUPPORTED;
  }

  return status;
  CATCH
}

rsmi_status_t
rsmi_topo_get_link_type(uint32_t dv_ind_src, uint32_t dv_ind_dst,
                        uint64_t *hops, RSMI_IO_LINK_TYPE *type) {
  TRY

  uint32_t dv_ind = dv_ind_src;
  GET_DEV_AND_KFDNODE_FROM_INDX

  if (hops == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  if (type == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  rsmi_status_t status;
  uint32_t node_ind_dst;

  // handle the link type for CPU
  if (dv_ind_dst == CPU_NODE_INDEX) {
    // No CPU connected
    if (kfd_node->numa_node_weight() == 0) {
      return RSMI_STATUS_NOT_SUPPORTED;
    }
    amd::smi::IO_LINK_TYPE io_link_type =
              kfd_node->numa_node_type();
    switch (io_link_type) {
      case amd::smi::IOLINK_TYPE_XGMI:
        *type = RSMI_IOLINK_TYPE_XGMI;
        *hops = 1;
        return RSMI_STATUS_SUCCESS;
      case amd::smi::IOLINK_TYPE_PCIEXPRESS:
        *type = RSMI_IOLINK_TYPE_PCIEXPRESS;
        // always be the same CPU node
        *hops = 2;
        return RSMI_STATUS_SUCCESS;
      default:
        return RSMI_STATUS_NOT_SUPPORTED;
    }
  }

  int ret = smi.get_node_index(dv_ind_dst, &node_ind_dst);

  if (ret == 0) {
    amd::smi::IO_LINK_TYPE io_link_type;
    ret = kfd_node->get_io_link_type(node_ind_dst, &io_link_type);
    if (ret == 0) {
      if (io_link_type == amd::smi::IOLINK_TYPE_XGMI) {
        *type = RSMI_IOLINK_TYPE_XGMI;
        *hops = 1;
        status = RSMI_STATUS_SUCCESS;
      } else {
        assert(false);  // Unexpected IO Link type read
        status = RSMI_STATUS_NOT_SUPPORTED;
      }
    } else if (kfd_node->numa_node_type() == amd::smi::IOLINK_TYPE_PCIEXPRESS) {
      uint32_t numa_number_dst;
      status = topo_get_numa_node_number(dv_ind_dst, &numa_number_dst);
      if (status == RSMI_STATUS_SUCCESS) {
        uint32_t numa_number_src = kfd_node->numa_node_number();
        if (numa_number_src == numa_number_dst) {
          *hops = 2;  // same CPU node
        } else {
          uint64_t io_link_weight;
          ret = smi.get_io_link_weight(numa_number_src, numa_number_dst,
                                       &io_link_weight);
          if (ret == 0)
            *hops = 3;  // from src CPU node to dst CPU node
          else
            *hops = 4;  // More than one CPU hops, hard coded as 4
        }
        *type = RSMI_IOLINK_TYPE_PCIEXPRESS;
        status = RSMI_STATUS_SUCCESS;
      } else {
        assert(false);  // Error to get numa node number
        status = RSMI_STATUS_INIT_ERROR;
      }
    } else {
      status = RSMI_STATUS_NOT_SUPPORTED;
    }
  } else {
    status = RSMI_STATUS_INVALID_ARGS;
  }

  return status;
  CATCH
}

rsmi_status_t
rsmi_is_P2P_accessible(uint32_t dv_ind_src, uint32_t dv_ind_dst,
                       bool *accessible) {
  TRY

  uint32_t dv_ind = dv_ind_src;
  GET_DEV_AND_KFDNODE_FROM_INDX

  if (accessible == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  uint32_t node_ind_src;
  uint32_t node_ind_dst;
  // Fetch the source and destination GPU node index
  if (smi.get_node_index(dv_ind_src, &node_ind_src) ||
      smi.get_node_index(dv_ind_dst, &node_ind_dst)) {
    *accessible = false;
    return RSMI_STATUS_INVALID_ARGS;
  }
  // If source device is same as destination, return true
  if (dv_ind_src == dv_ind_dst) {
    *accessible = true;
    return RSMI_STATUS_SUCCESS;
  }
  std::map<uint32_t, std::shared_ptr<amd::smi::IOLink>> io_link_map_tmp;
  std::map<uint32_t, std::shared_ptr<amd::smi::IOLink>>::iterator it;
  // Iterate over P2P links
  if (DiscoverP2PLinksPerNode(node_ind_src, &io_link_map_tmp) == 0) {
    for (it = io_link_map_tmp.begin(); it != io_link_map_tmp.end(); it++) {
      if(it->first == node_ind_dst) {
        *accessible = true;
        return RSMI_STATUS_SUCCESS;
      }
    }
    io_link_map_tmp.clear();
  } else {
    *accessible = false;
    return RSMI_STATUS_FILE_ERROR;
  }
  // Iterate over IO links
  if (DiscoverIOLinksPerNode(node_ind_src, &io_link_map_tmp) == 0) {
    for (it = io_link_map_tmp.begin(); it != io_link_map_tmp.end(); it++) {
      if(it->first == node_ind_dst) {
        *accessible = true;
        return RSMI_STATUS_SUCCESS;
      }
    }
  } else {
    *accessible = false;
    return RSMI_STATUS_FILE_ERROR;
  }
  *accessible = false;
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_topo_get_p2p_status(uint32_t dv_ind_src, uint32_t dv_ind_dst,
                         RSMI_IO_LINK_TYPE *type, rsmi_p2p_capability_t *cap) {
  TRY

  uint32_t dv_ind = dv_ind_src;
  GET_DEV_AND_KFDNODE_FROM_INDX
  DEVICE_MUTEX

  if (type == nullptr || cap == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  // If source device is same as destination, return invalid args
  if (dv_ind_src == dv_ind_dst) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  uint32_t node_ind_src, node_ind_dst;
  // Fetch the source and destination node index
  if (smi.get_node_index(dv_ind_src, &node_ind_src) ||
      smi.get_node_index(dv_ind_dst, &node_ind_dst)) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  bool node_is_find = false;
  std::map<uint32_t, std::shared_ptr<amd::smi::IOLink>> io_link_map_tmp;
  std::map<uint32_t, std::shared_ptr<amd::smi::IOLink>>::iterator it;
  // Iterate over P2P links
  if (DiscoverP2PLinksPerNode(node_ind_src, &io_link_map_tmp) == 0) {
    for (it = io_link_map_tmp.begin(); it != io_link_map_tmp.end(); it++) {
      if (it->first == node_ind_dst) {
        node_is_find = true;
        break;
      }
    }
    io_link_map_tmp.clear();
  } else {
    return RSMI_STATUS_FILE_ERROR;
  }

  if (!node_is_find) {
    // Iterate over IO links
    if (DiscoverIOLinksPerNode(node_ind_src, &io_link_map_tmp) == 0) {
      for (it = io_link_map_tmp.begin(); it != io_link_map_tmp.end(); it++) {
        if (it->first == node_ind_dst) {
          node_is_find = true;
          break;
        }
      }
      io_link_map_tmp.clear();
    } else {
      return RSMI_STATUS_FILE_ERROR;
    }
  }

  if (node_is_find) {
    amd::smi::IO_LINK_TYPE io_link_type = it->second->type();
    if (io_link_type == amd::smi::IOLINK_TYPE_PCIEXPRESS) {
      *type = RSMI_IOLINK_TYPE_PCIEXPRESS;
    } else if (io_link_type == amd::smi::IOLINK_TYPE_XGMI) {
      *type = RSMI_IOLINK_TYPE_XGMI;
    } else {
      // Unexpected IO Link type read
      return RSMI_STATUS_NOT_SUPPORTED;
    }

    /*
     *  Note: Adjust tmp_capability for the returned capabilities.
     *  Todo: We need to fix it directy as part of the KFD Nodes 'KFDNode::Initialize(void)'
     *        However, it involves a more complex change, so we will discuss it and fix in in the future.
     *          Ideally, due to the fact we would need to check every IO link (for each KFD node), and
     *          considering the topology could change (ie; new GPUs added, partioning changed, etc), we
     *          are looking into O(N^2) time complexity, no to mention the fact that the IO links then
     *          need to be changed/updated too. I have some ideas about how to do this, but it will take
     *          some time to implement and test it, should we consider it is *really necessary*.
     *
     */
    auto tmp_capability = it->second->get_link_capability();
    if (auto link_direction_result = amd::smi::DiscoverIOLinkPerNodeDirection(node_ind_src, node_ind_dst);
        link_direction_result == amd::smi::IOLinkDirectionType_t::kBiDirectional) {
        // 1 = true, 0 = false
        tmp_capability.is_iolink_bi_directional = 1;
    }
    *cap = tmp_capability;
    return RSMI_STATUS_SUCCESS;
  }

  return RSMI_STATUS_NOT_SUPPORTED;

  CATCH
}

static rsmi_status_t get_compute_partition(uint32_t dv_ind,
                                          std::string &compute_partition) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(compute_partition.c_str())
  std::string compute_partition_str;

  DEVICE_MUTEX
  rsmi_status_t ret = get_dev_value_str(amd::smi::kDevComputePartition,
                                        dv_ind, &compute_partition_str);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  switch (mapStringToRSMIComputePartitionTypes.at(compute_partition_str)) {
    case RSMI_COMPUTE_PARTITION_CPX:
    case RSMI_COMPUTE_PARTITION_SPX:
    case RSMI_COMPUTE_PARTITION_DPX:
    case RSMI_COMPUTE_PARTITION_TPX:
    case RSMI_COMPUTE_PARTITION_QPX:
      break;
    case RSMI_COMPUTE_PARTITION_INVALID:
    default:
      // Retrieved an unknown compute partition
      return RSMI_STATUS_UNEXPECTED_DATA;
  }
  compute_partition = compute_partition_str;
  ss << __PRETTY_FUNCTION__ << " | ======= END =======, " << dv_ind;
  LOG_TRACE(ss);
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_compute_partition_get(uint32_t dv_ind, char *compute_partition,
                               uint32_t len) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, dv_ind = "
     << dv_ind;
  LOG_TRACE(ss);
  if ((len == 0) || (compute_partition == nullptr)) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevComputePartition)
       << " | Cause: len was 0 or compute_partition variable was null"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS) << " |";
    LOG_ERROR(ss);
    return RSMI_STATUS_INVALID_ARGS;
  }
  CHK_SUPPORT_NAME_ONLY(compute_partition)

  std::string returning_compute_partition;
  rsmi_status_t ret = get_compute_partition(dv_ind,
                               returning_compute_partition);

  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevComputePartition)
       << " | Cause: could not retrieve current compute partition"
       << " | Returning = "
       << getRSMIStatusString(ret) << " |";
    LOG_ERROR(ss);
    return ret;
  }

  std::size_t length = returning_compute_partition.copy(compute_partition, len-1);
  compute_partition[length]='\0';

  if (len < (returning_compute_partition.size() + 1)) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevComputePartition)
       << " | Cause: requested size was insufficient"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INSUFFICIENT_SIZE) << " |";
    LOG_ERROR(ss);
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevComputePartition)
     << " | Data: " << compute_partition
     << " | Returning = "
     << getRSMIStatusString(ret) << " |";
  LOG_TRACE(ss);
  return ret;
  CATCH
}

static rsmi_status_t
is_available_compute_partition(uint32_t dv_ind,
                               std::string new_compute_partition) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  DEVICE_MUTEX
  std::string availableComputePartitions;
  rsmi_status_t ret =
      get_dev_value_line(amd::smi::kDevAvailableComputePartition,
                         dv_ind, &availableComputePartitions);
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | FAIL "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevAvailableComputePartition)
       << " | Data: could not retrieve requested data"
       << " | Returning = "
       << getRSMIStatusString(ret) << " |";
    LOG_ERROR(ss);
    return ret;
  }

  bool isComputePartitionAvailable =
      amd::smi::containsString(availableComputePartitions,
                               new_compute_partition);

  ret = ((isComputePartitionAvailable) ? RSMI_STATUS_SUCCESS :
                                         RSMI_STATUS_SETTING_UNAVAILABLE);
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevAvailableComputePartition)
     << " | Data: available_partitions = " << availableComputePartitions
     << " | Data: isComputePartitionAvailable = "
     << (isComputePartitionAvailable ? "True" : "False")
     << " | Returning = "
     << getRSMIStatusString(ret) << " |";
  LOG_INFO(ss);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_compute_partition_set(uint32_t dv_ind,
                              rsmi_compute_partition_type_t compute_partition) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS
  if (!amd::smi::is_sudo_user()) {
    return RSMI_STATUS_PERMISSION;
  }
  std::string currentComputePartition = "";
  std::string newComputePartitionStr = "";

  switch (compute_partition) {
    case RSMI_COMPUTE_PARTITION_CPX:
    case RSMI_COMPUTE_PARTITION_SPX:
    case RSMI_COMPUTE_PARTITION_DPX:
    case RSMI_COMPUTE_PARTITION_TPX:
    case RSMI_COMPUTE_PARTITION_QPX:
      newComputePartitionStr =
        mapRSMIToStringComputePartitionTypes.at(compute_partition);
      break;
    case RSMI_COMPUTE_PARTITION_INVALID:
    default:
      newComputePartitionStr =
        mapRSMIToStringComputePartitionTypes.at(RSMI_COMPUTE_PARTITION_INVALID);
      ss << __PRETTY_FUNCTION__
         << " | ======= end ======= "
         << " | Fail "
         << " | Device #: " << dv_ind
         << " | Type: "
         << amd::smi::Device::get_type_string(amd::smi::kDevComputePartition)
         << " | Data: " << newComputePartitionStr
         << " | Cause: requested setting was invalid"
         << " | Returning = "
         << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS) << " |";
      LOG_ERROR(ss);
      return RSMI_STATUS_INVALID_ARGS;
  }

  // Confirm what we are trying to set is available, otherwise provide
  // RSMI_STATUS_SETTING_UNAVAILABLE
  rsmi_status_t available_ret =
      is_available_compute_partition(dv_ind, newComputePartitionStr);
  if (available_ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevComputePartition)
       << " | Data: " << newComputePartitionStr
       << " | Cause: not an available compute partition setting"
       << " | Returning = "
       << getRSMIStatusString(available_ret) << " |";
    LOG_ERROR(ss);
    return available_ret;
  }

  // do nothing if compute_partition is the current compute partition
  rsmi_status_t ret_get =
                        get_compute_partition(dv_ind, currentComputePartition);
  // we can try to set, even if we get unexpected data
  if (ret_get != RSMI_STATUS_SUCCESS
      && ret_get != RSMI_STATUS_UNEXPECTED_DATA) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevComputePartition)
       << " | Cause: could retrieve current compute partition or retrieved"
       << " unexpected data"
       << " | Returning = "
       << getRSMIStatusString(ret_get) << " |";
    LOG_ERROR(ss);
    return ret_get;
  }
  rsmi_compute_partition_type_t currRSMIComputePartition
    = mapStringToRSMIComputePartitionTypes.at(currentComputePartition);
  if (currRSMIComputePartition == compute_partition) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Success - compute partition was already set at requested value"
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevComputePartition)
       << " | Data: " << newComputePartitionStr
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " |";
    LOG_TRACE(ss);
    return RSMI_STATUS_SUCCESS;
  }

  ss <<  __PRETTY_FUNCTION__ << " | about to try writing |"
     << newComputePartitionStr
     << "| size of string = " << newComputePartitionStr.size()
     << "| size of c-string = "<< std::dec
     << sizeof(newComputePartitionStr.c_str())/sizeof(newComputePartitionStr[0])
     << "| sizeof string = " << std::dec
     << sizeof(newComputePartitionStr);
  LOG_DEBUG(ss);
  GET_DEV_FROM_INDX
  DEVICE_MUTEX
  int ret = dev->writeDevInfo(amd::smi::kDevComputePartition,
                              newComputePartitionStr);
  rsmi_status_t returnResponse = amd::smi::ErrnoToRsmiStatus(ret);
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevComputePartition)
     << " | Data: " << newComputePartitionStr
     << " | Returning = "
     << getRSMIStatusString(returnResponse) << " |";
  LOG_TRACE(ss);

  return returnResponse;
  CATCH
}

rsmi_status_t rsmi_dev_compute_partition_capabilities_get(
      uint32_t dv_ind, char *compute_partition_caps, uint32_t len) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  DEVICE_MUTEX
  std::string availableComputePartitions;
  rsmi_status_t ret =
      get_dev_value_line(amd::smi::kDevAvailableComputePartition,
                         dv_ind, &availableComputePartitions);
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | FAIL "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevAvailableComputePartition)
       << " | Data: could not retrieve requested data"
       << " | Returning = "
       << getRSMIStatusString(ret) << " |";
    LOG_ERROR(ss);
    return ret;
  }

  std::size_t length = availableComputePartitions.copy(compute_partition_caps, len-1);
  compute_partition_caps[length]='\0';

  if (len < (availableComputePartitions.size() + 1)) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevAvailableComputePartition)
       << " | Cause: requested size was insufficient"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INSUFFICIENT_SIZE) << " |";
    LOG_ERROR(ss);
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevAvailableComputePartition)
     << " | Data: " << compute_partition_caps
     << " | Returning = "
     << getRSMIStatusString(ret) << " |";
  LOG_TRACE(ss);
  return ret;
  CATCH
}

rsmi_status_t rsmi_dev_compute_partition_supported_xcp_configs_get(uint32_t dv_ind,
                    char *supported_configs, uint32_t len) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  DEVICE_MUTEX
  std::string supported_xcp_configs;
  rsmi_status_t ret =
      get_dev_value_line(amd::smi::kDevSupportedXcpConfigs,
                         dv_ind, &supported_xcp_configs);
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | FAIL "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevSupportedXcpConfigs)
       << " | Data: could not retrieve requested data"
       << " | Returning = "
       << getRSMIStatusString(ret) << " |";
    LOG_ERROR(ss);
    return ret;
  }

  std::size_t length = supported_xcp_configs.copy(supported_configs, len-1);
  supported_configs[length]='\0';

  if (len < (supported_xcp_configs.size() + 1)) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevSupportedXcpConfigs)
       << " | Cause: requested size was insufficient"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INSUFFICIENT_SIZE) << " |";
    LOG_ERROR(ss);
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevSupportedXcpConfigs)
     << " | Data: " << supported_configs
     << " | Returning = "
     << getRSMIStatusString(ret) << " |";
  LOG_TRACE(ss);
  return ret;
  CATCH
}

rsmi_status_t rsmi_dev_compute_partition_supported_nps_configs_get(uint32_t dv_ind,
                    char *supported_configs, uint32_t len) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  DEVICE_MUTEX
  std::string supported_nps_configs;
  rsmi_status_t ret =
      get_dev_value_line(amd::smi::kDevSupportedNpsConfigs,
                         dv_ind, &supported_nps_configs);
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | FAIL "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevSupportedNpsConfigs)
       << " | Data: could not retrieve requested data"
       << " | Returning = "
       << getRSMIStatusString(ret) << " |";
    LOG_ERROR(ss);
    return ret;
  }

  std::size_t length = supported_nps_configs.copy(supported_configs, len-1);
  supported_configs[length]='\0';

  if (len < (supported_nps_configs.size() + 1)) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevSupportedNpsConfigs)
       << " | Cause: requested size was insufficient"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INSUFFICIENT_SIZE) << " |";
    LOG_ERROR(ss);
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevSupportedNpsConfigs)
     << " | Data: " << supported_configs
     << " | Returning = "
     << getRSMIStatusString(ret) << " |";
  LOG_TRACE(ss);
  return ret;
  CATCH
}

rsmi_status_t rsmi_dev_current_compute_xcp_config_get(
      uint32_t dv_ind, char *current_xcp_config, uint32_t len) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  DEVICE_MUTEX
  std::string currentXcpConfigStr;
  rsmi_status_t ret =
      get_dev_value_line(amd::smi::kDevXcpConfig,
                         dv_ind, &currentXcpConfigStr);
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | FAIL "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevXcpConfig)
       << " | Data: could not retrieve requested data"
       << " | Returning = "
       << getRSMIStatusString(ret) << " |";
    LOG_ERROR(ss);
    return ret;
  }

  std::size_t length = currentXcpConfigStr.copy(current_xcp_config, len-1);
  current_xcp_config[length]='\0';

  if (len < (currentXcpConfigStr.size() + 1)) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevXcpConfig)
       << " | Cause: requested size was insufficient"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INSUFFICIENT_SIZE) << " |";
    LOG_ERROR(ss);
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevXcpConfig)
     << " | Data: " << currentXcpConfigStr
     << " | Returning = "
     << getRSMIStatusString(ret) << " |";
  LOG_TRACE(ss);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_compute_partition_xcp_config_set(uint32_t dv_ind,
                                          rsmi_compute_partition_type_t xcp_config) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS
  if (!amd::smi::is_sudo_user()) {
    return RSMI_STATUS_PERMISSION;
  }
  std::string currentXcpConfig = "";
  std::string newXcpConfigStr = "";
  std::string availableXcpConfigsStr = "";
  const int kLen30 = 30;
  char available_xcp_configs[kLen30];
  available_xcp_configs[0] = '\0';

  switch (xcp_config) {
    case RSMI_COMPUTE_PARTITION_CPX:
    case RSMI_COMPUTE_PARTITION_SPX:
    case RSMI_COMPUTE_PARTITION_DPX:
    case RSMI_COMPUTE_PARTITION_TPX:
    case RSMI_COMPUTE_PARTITION_QPX:
      newXcpConfigStr =
        mapRSMIToStringComputePartitionTypes.at(xcp_config);
      break;
    case RSMI_COMPUTE_PARTITION_INVALID:
    default:
      newXcpConfigStr =
        mapRSMIToStringComputePartitionTypes.at(RSMI_COMPUTE_PARTITION_INVALID);
      ss << __PRETTY_FUNCTION__
         << " | ======= end ======= "
         << " | Fail "
         << " | Device #: " << dv_ind
         << " | Type: "
         << amd::smi::Device::get_type_string(amd::smi::kDevXcpConfig)
         << " | Data: " << newXcpConfigStr
         << " | Cause: requested setting was invalid"
         << " | Returning = "
         << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS) << " |";
      LOG_ERROR(ss);
      return RSMI_STATUS_INVALID_ARGS;
  }

  // Confirm what we are trying to set is available, otherwise provide
  // RSMI_STATUS_INVALID_ARGS
  rsmi_status_t available_ret =
      rsmi_dev_compute_partition_supported_xcp_configs_get(dv_ind, available_xcp_configs, kLen30);
  if (available_ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevSupportedXcpConfigs)
       << " | Data: " << newXcpConfigStr
       << " | Cause: could not find an available xcp configs file"
       << " | Returning = "
       << getRSMIStatusString(available_ret) << " |";
    LOG_ERROR(ss);
    return available_ret;
  } else {
    availableXcpConfigsStr = available_xcp_configs;
  }

  bool isXcpConfigAvailable =
      amd::smi::containsString(availableXcpConfigsStr,
                               newXcpConfigStr);
  if (!isXcpConfigAvailable) {
    ss << __PRETTY_FUNCTION__
       << " | Fail  - Detected that the requested xcp config is not available"
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevXcpConfig)
       << " | Data (newXcpConfigStr): " << newXcpConfigStr
       << " | Data (availableXcpConfigsStr): " << availableXcpConfigsStr;
    LOG_ERROR(ss);
    // We do not return RSMI_STATUS_INVALID_ARGS
    // Instead we try setting anyways as requested
    // write will provide the correct error code
  }

  ss <<  __PRETTY_FUNCTION__ << " | about to try writing |"
     << newXcpConfigStr
     << "| size of string = " << newXcpConfigStr.size()
     << "| size of c-string = "<< std::dec
     << sizeof(newXcpConfigStr.c_str())/sizeof(newXcpConfigStr[0])
     << "| sizeof string = " << std::dec
     << sizeof(newXcpConfigStr);
  LOG_DEBUG(ss);
  GET_DEV_FROM_INDX
  DEVICE_MUTEX
  int ret = dev->writeDevInfo(amd::smi::kDevXcpConfig,
                              newXcpConfigStr);
  rsmi_status_t returnResponse = amd::smi::ErrnoToRsmiStatus(ret);
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevXcpConfig)
     << " | Data: " << newXcpConfigStr
     << " | Returning = "
     << getRSMIStatusString(returnResponse) << " |";
  LOG_TRACE(ss);

  return returnResponse;
  CATCH
}

rsmi_status_t rsmi_dev_compute_partition_resource_profile_get(uint32_t dv_ind,
                                  rsmi_accelerator_partition_resource_type_t *type,
                                  rsmi_accelerator_partition_resource_profile_t *profile) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  if (type == nullptr) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevXcpConfig)
       << " | Cause: user sent invalid arguments, type was a null ptr"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS, false);
    LOG_ERROR(ss);
    return RSMI_STATUS_INVALID_ARGS;
  }
  // initialize the profile
  profile->partition_resource = std::numeric_limits<uint32_t>::max();
  profile->num_partitions_share_resource = std::numeric_limits<uint32_t>::max();

  DEVICE_MUTEX
  rsmi_status_t ret = RSMI_STATUS_NOT_SUPPORTED;
  // check if user provided supported resource types
  // Note: RSMI_ACCELERATOR_MAX is == largest enum value
  bool isAcceleratorTypeValid = false;
  for (int i = 0; i <= RSMI_ACCELERATOR_MAX; i++) {
    if (*type == i) {
      isAcceleratorTypeValid = true;
      break;
    }
  }
  if (isAcceleratorTypeValid == false) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevXcpConfig)
       << " | Cause: user sent invalid arguments, type was out of range"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS, false);
    LOG_ERROR(ss);
    return RSMI_STATUS_INVALID_ARGS;
  }
  amd::smi::DevInfoTypes dev_info_type_inst;
  amd::smi::DevInfoTypes dev_info_type_shared;
  if (*type == RSMI_ACCELERATOR_XCC) {
    profile->resource_type = RSMI_ACCELERATOR_XCC;
    dev_info_type_inst = amd::smi::kDevXccInst;
    dev_info_type_shared = amd::smi::kDevXccShared;
    std::string val_str;
    ret = get_dev_value_str(amd::smi::kDevXccInst, dv_ind, &val_str);
    if (ret == RSMI_STATUS_SUCCESS) {
      uint64_t val_ul = strtoul(val_str.c_str(), nullptr, 10);
      if (val_ul <= std::numeric_limits<uint32_t>::max()) {
        profile->partition_resource = static_cast<uint32_t>(val_ul);
      }
    }

    val_str.clear();
    ret = get_dev_value_str(amd::smi::kDevXccShared, dv_ind, &val_str);
    if (ret == RSMI_STATUS_SUCCESS) {
      uint64_t val_ul = strtoul(val_str.c_str(), nullptr, 10);
      if (val_ul <= std::numeric_limits<uint32_t>::max()) {
        profile->num_partitions_share_resource = static_cast<uint32_t>(val_ul);
      }
    }
  }

  if (*type == RSMI_ACCELERATOR_ENCODER) {
    profile->resource_type = RSMI_ACCELERATOR_ENCODER;
    dev_info_type_inst = amd::smi::kDevEncoderInst;
    dev_info_type_shared = amd::smi::kDevEncoderShared;
    std::string val_str;
    ret = get_dev_value_str(amd::smi::kDevEncoderInst, dv_ind, &val_str);
    if (ret == RSMI_STATUS_SUCCESS) {
      uint64_t val_ul = strtoul(val_str.c_str(), nullptr, 10);
      if (val_ul <= std::numeric_limits<uint32_t>::max()) {
        profile->partition_resource = static_cast<uint32_t>(val_ul);
      }
    }

    val_str.clear();
    ret = get_dev_value_str(amd::smi::kDevEncoderShared, dv_ind, &val_str);
    if (ret == RSMI_STATUS_SUCCESS) {
      uint64_t val_ul = strtoul(val_str.c_str(), nullptr, 10);
      if (val_ul <= std::numeric_limits<uint32_t>::max()) {
        profile->num_partitions_share_resource = static_cast<uint32_t>(val_ul);
      }
    }
  }

  if (*type == RSMI_ACCELERATOR_DECODER) {
    profile->resource_type = RSMI_ACCELERATOR_DECODER;
    dev_info_type_inst = amd::smi::kDevDecoderInst;
    dev_info_type_shared = amd::smi::kDevDecoderShared;
    std::string val_str;
    ret = get_dev_value_str(amd::smi::kDevDecoderInst, dv_ind, &val_str);
    if (ret == RSMI_STATUS_SUCCESS) {
      uint64_t val_ul = strtoul(val_str.c_str(), nullptr, 10);
      if (val_ul <= std::numeric_limits<uint32_t>::max()) {
        profile->partition_resource = static_cast<uint32_t>(val_ul);
      }
    }

    val_str.clear();
    ret = get_dev_value_str(amd::smi::kDevDecoderShared, dv_ind, &val_str);
    if (ret == RSMI_STATUS_SUCCESS) {
      uint64_t val_ul = strtoul(val_str.c_str(), nullptr, 10);
      if (val_ul <= std::numeric_limits<uint32_t>::max()) {
        profile->num_partitions_share_resource = static_cast<uint32_t>(val_ul);
      }
    }
  }

  if (*type == RSMI_ACCELERATOR_DMA) {
    profile->resource_type = RSMI_ACCELERATOR_DMA;
    dev_info_type_inst = amd::smi::kDevDmaInst;
    dev_info_type_shared = amd::smi::kDevDmaShared;
    std::string val_str;
    ret = get_dev_value_str(amd::smi::kDevDmaInst, dv_ind, &val_str);
    if (ret == RSMI_STATUS_SUCCESS) {
      uint64_t val_ul = strtoul(val_str.c_str(), nullptr, 10);
      if (val_ul <= std::numeric_limits<uint32_t>::max()) {
        profile->partition_resource = static_cast<uint32_t>(val_ul);
      }
    }

    val_str.clear();
    ret = get_dev_value_str(amd::smi::kDevDmaShared, dv_ind, &val_str);
    if (ret == RSMI_STATUS_SUCCESS) {
      uint64_t val_ul = strtoul(val_str.c_str(), nullptr, 10);
      if (val_ul <= std::numeric_limits<uint32_t>::max()) {
        profile->num_partitions_share_resource = static_cast<uint32_t>(val_ul);
      }
    }
  }

  // RSMI_ACCELERATOR_MAX == RSMI_ACCELERATOR_JPEG
  if (*type == RSMI_ACCELERATOR_JPEG) {
    profile->resource_type = RSMI_ACCELERATOR_JPEG;
    dev_info_type_inst = amd::smi::kDevJpegInst;
    dev_info_type_shared = amd::smi::kDevJpegShared;
    std::string val_str;
    ret = get_dev_value_str(amd::smi::kDevJpegInst, dv_ind, &val_str);
    if (ret == RSMI_STATUS_SUCCESS) {
      uint64_t val_ul = strtoul(val_str.c_str(), nullptr, 10);
      if (val_ul <= std::numeric_limits<uint32_t>::max()) {
        profile->partition_resource = static_cast<uint32_t>(val_ul);
      }
    }

    val_str.clear();
    ret = get_dev_value_str(amd::smi::kDevJpegShared, dv_ind, &val_str);
    if (ret == RSMI_STATUS_SUCCESS) {
      uint64_t val_ul = strtoul(val_str.c_str(), nullptr, 10);
      if (val_ul <= std::numeric_limits<uint32_t>::max()) {
        profile->num_partitions_share_resource = static_cast<uint32_t>(val_ul);
      }
    }
  }

  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type (partition_resource): "
     << amd::smi::Device::get_type_string(dev_info_type_inst)
     << " | Data: " << profile->partition_resource
     << " | Type (num_partitions_share_resource): "
     << amd::smi::Device::get_type_string(dev_info_type_shared)
     << " | Data: " << profile->num_partitions_share_resource
     << " | Returning = "
     << getRSMIStatusString(ret, false) << " |";
  LOG_TRACE(ss);

  return ret;
  CATCH
}

static rsmi_status_t get_memory_partition(uint32_t dv_ind,
                                          std::string &memory_partition) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  CHK_SUPPORT_NAME_ONLY(memory_partition.c_str())
  std::string val_str;

  DEVICE_MUTEX
  rsmi_status_t ret = get_dev_value_str(amd::smi::kDevMemoryPartition,
                                        dv_ind, &val_str);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  switch (mapStringToMemoryPartitionTypes.at(val_str)) {
    case RSMI_MEMORY_PARTITION_NPS1:
    case RSMI_MEMORY_PARTITION_NPS2:
    case RSMI_MEMORY_PARTITION_NPS4:
    case RSMI_MEMORY_PARTITION_NPS8:
      break;
    case RSMI_MEMORY_PARTITION_UNKNOWN:
    default:
      // Retrieved an unknown memory partition
      return RSMI_STATUS_UNEXPECTED_DATA;
  }
  memory_partition = val_str;
  ss << __PRETTY_FUNCTION__ << " | ======= END =======, " << dv_ind;
  LOG_TRACE(ss);
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_memory_partition_set(uint32_t dv_ind,
                              rsmi_memory_partition_type_t memory_partition) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX
  const int k1000_MS_WAIT = 1000;

  const uint32_t kMaxMemoryCapabilitiesSize = 30;
  char available_memory_capabilities[kMaxMemoryCapabilitiesSize];
  available_memory_capabilities[0] = '\0';

  const uint32_t kMaxCurrentMemoryMode = 5;
  char current_memory_mode[kMaxCurrentMemoryMode];
  current_memory_mode[0] = '\0';


  // Is the current mode already what user requested?
  switch (memory_partition) {
    case RSMI_MEMORY_PARTITION_NPS1:
    case RSMI_MEMORY_PARTITION_NPS2:
    case RSMI_MEMORY_PARTITION_NPS4:
    case RSMI_MEMORY_PARTITION_NPS8:
      break;
    case RSMI_MEMORY_PARTITION_UNKNOWN:
    default:
      ss << __PRETTY_FUNCTION__
         << " | ======= end ======= "
         << " | Fail "
         << " | Device #: " << dv_ind
         << " | Type: "
         << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
         << " | Cause: requested setting was invalid"
         << " | Returning = "
         << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS, false);
      LOG_ERROR(ss);
      return RSMI_STATUS_INVALID_ARGS;
  }
  std::string newMemoryPartition
              = mapRSMIToStringMemoryPartitionTypes.at(memory_partition);
  std::string currentMemoryPartition;

  // do nothing if memory_partition is the current mode
  rsmi_status_t ret_get = get_memory_partition(dv_ind, currentMemoryPartition);
  // we can try to set, even if we get unexpected data
  if (ret_get != RSMI_STATUS_SUCCESS
      && ret_get != RSMI_STATUS_UNEXPECTED_DATA) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
       << " | Cause: could retrieve current memory partition or retrieved"
       << " unexpected data"
       << " | Returning = "
       << getRSMIStatusString(ret_get, false);
    LOG_ERROR(ss);
    return ret_get;
  }
  rsmi_memory_partition_type_t currRSMIMemoryPartition
    = mapStringToMemoryPartitionTypes.at(currentMemoryPartition);
  if (currRSMIMemoryPartition == memory_partition) {
    ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success - no change, current memory partition was already requested"
     << " setting"
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
     << " | Data: " << newMemoryPartition
     << " | Returning = "
     << getRSMIStatusString(RSMI_STATUS_SUCCESS, false);
    LOG_TRACE(ss);
    return RSMI_STATUS_SUCCESS;
  }

  // is this an available mode to set to?
  std::string memory_capabilities_str = "unknown";
  std::string user_requested_memory_partition = newMemoryPartition;
  std::transform(user_requested_memory_partition.begin(), user_requested_memory_partition.end(),
                  user_requested_memory_partition.begin(), ::toupper);
  rsmi_status_t caps_ret = rsmi_dev_memory_partition_capabilities_get(dv_ind,
                              available_memory_capabilities, kMaxMemoryCapabilitiesSize);
  memory_capabilities_str = available_memory_capabilities;
  std::transform(memory_capabilities_str.begin(), memory_capabilities_str.end(),
                  memory_capabilities_str.begin(), ::toupper);
  ss << __PRETTY_FUNCTION__ << " | user_requested_memory_partition: "
     << user_requested_memory_partition
     << "; memory_capabilities_str: " << memory_capabilities_str
     << "; rsmi_dev_memory_partition_capabilities_get(" << dv_ind
     << ", " << user_requested_memory_partition << "): return = "
     << amd::smi::getRSMIStatusString(caps_ret, false);
  LOG_DEBUG(ss);
  if ((caps_ret == RSMI_STATUS_SUCCESS)
      && (!memory_capabilities_str.empty())
      && (!user_requested_memory_partition.empty())) {
    bool is_available_mode = amd::smi::containsString(memory_capabilities_str,
                                user_requested_memory_partition, true);
    ss << __PRETTY_FUNCTION__
       << " | is_available_mode: " << (is_available_mode ? "True": "False");
    LOG_DEBUG(ss);
    if (is_available_mode == false) {  // report RSMI_STATUS_INVALID_ARGS
      ss << __PRETTY_FUNCTION__
         << " | ======= Check if available mode ======= "
         << " | WARNING: detected invalid mode to set to, will try to set anyways"
         << " | Device #: " << dv_ind
         << " | Type: "
         << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
         << " | Data (user requested mode): " << user_requested_memory_partition
         << " | Available Memory Partition Modes: " << memory_capabilities_str
         << " | Cause: requested setting was not an available mode"
         << " | Returning = "
         << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS, false);
      LOG_INFO(ss);
    }
  }

  GET_DEV_FROM_INDX
  int ret = dev->writeDevInfo(amd::smi::kDevMemoryPartition,
                              newMemoryPartition);

  if (amd::smi::ErrnoToRsmiStatus(ret) != RSMI_STATUS_SUCCESS) {
    rsmi_status_t err = amd::smi::ErrnoToRsmiStatus(ret);
    if (ret == EACCES) {
      err = RSMI_STATUS_NOT_SUPPORTED;  // already verified permissions
    }
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
       << " | Cause: issue writing reqested setting of " + newMemoryPartition
       << " | Returning = "
       << getRSMIStatusString(err, false);
    LOG_ERROR(ss);
    return err;
  }

  rsmi_status_t restartRet = dev->restartAMDGpuDriver();
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success - if restart completed successfully"
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
     << " | Data: " << newMemoryPartition
     << " | Returning = "
     << getRSMIStatusString(restartRet, false);
  LOG_TRACE(ss);

  if (restartRet != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail - restart AMD GPU detected"
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
       << " | Cause: issue writing reqested setting of " + newMemoryPartition
       << " | Returning = "
       << getRSMIStatusString(restartRet, false);
    LOG_ERROR(ss);
    return restartRet;
  }

  std::string current_memory_mode_str = "unknown";
  rsmi_status_t can_read_sysfs_again = RSMI_STATUS_AMDGPU_RESTART_ERR;
  int maxWaitSeconds = 10;
  // wait until we can read SYSFS again
  if (restartRet == RSMI_STATUS_SUCCESS) {
    while ((current_memory_mode_str != user_requested_memory_partition)
          && maxWaitSeconds > 0) {
      maxWaitSeconds -= 1;
      can_read_sysfs_again =
        rsmi_dev_memory_partition_get(dv_ind, current_memory_mode, kMaxCurrentMemoryMode);
      if (can_read_sysfs_again == RSMI_STATUS_SUCCESS) {
        current_memory_mode_str.clear();
        current_memory_mode_str = current_memory_mode;
        ss << __PRETTY_FUNCTION__
           << " | ======= rsmi_dev_memory_partition_get ======= "
           << " | Success - can read SYSFS"
           << " | Device #: " << dv_ind
           << " | Type: "
           << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
           << " | Data (user requested mode): " << user_requested_memory_partition
           << " | Current Memory Partition Mode: " << current_memory_mode_str
           << " | Available Memory Partition Modes: " << memory_capabilities_str
           << " | maxWaitSeconds: " << maxWaitSeconds
           << " | total wait time (sec): " << (10 - maxWaitSeconds)
           << " | Returning = "
           << getRSMIStatusString(can_read_sysfs_again, false);
        LOG_TRACE(ss);
        if (!current_memory_mode_str.empty()
            && (current_memory_mode_str == user_requested_memory_partition)) {
          break;
        }
      }
      amd::smi::system_wait(k1000_MS_WAIT);
    }
  }

  if (current_memory_mode_str == user_requested_memory_partition) {
    restartRet = RSMI_STATUS_SUCCESS;
  } else {
    restartRet = RSMI_STATUS_AMDGPU_RESTART_ERR;
  }

  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success - completed driver restart and all SYSFS are active"
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
     << " | Data: " << user_requested_memory_partition
     << " | Current Memory Partition Mode: " << current_memory_mode_str
     << " | Available Memory Partition Modes: " << memory_capabilities_str
     << " | Returning = "
     << getRSMIStatusString(restartRet, false);
  LOG_TRACE(ss);

  return restartRet;
  CATCH
}

rsmi_status_t
rsmi_dev_memory_partition_get(uint32_t dv_ind, char *memory_partition,
                               uint32_t len) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  if ((len == 0) || (memory_partition == nullptr)) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
       << " | Cause: user sent invalid arguments, len = 0 or memory partition"
       << " was a null ptr"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS) << " |";
    LOG_ERROR(ss);
    return RSMI_STATUS_INVALID_ARGS;
  }
  CHK_SUPPORT_NAME_ONLY(memory_partition)

  std::string returning_memory_partition;
  rsmi_status_t ret = get_memory_partition(dv_ind,
                               returning_memory_partition);

  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
       << " | Cause: could not successfully retrieve current memory partition "
       << " | Returning = "
       << getRSMIStatusString(ret) << " |";
    LOG_ERROR(ss);
    return ret;
  }

  std::size_t buff_size =
      returning_memory_partition.copy(memory_partition, len);
  memory_partition[buff_size] = '\0';

  if (len < (returning_memory_partition.size() + 1)) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
       << " | Cause: could not successfully retrieve current memory partition "
       << " | Returning = "
       << getRSMIStatusString(ret) << " |";
    LOG_ERROR(ss);
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevMemoryPartition)
     << " | Data: " << memory_partition
     << " | Returning = "
     << getRSMIStatusString(ret) << " |";
  LOG_TRACE(ss);
  return ret;
  CATCH
}

rsmi_status_t rsmi_dev_memory_partition_capabilities_get(
      uint32_t dv_ind, char *memory_partition_caps, uint32_t len) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << " | ======= start =======, " << dv_ind;
  LOG_TRACE(ss);

  if ((len == 0) || (memory_partition_caps == nullptr)) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevAvailableMemoryPartition)
       << " | Cause: user sent invalid arguments, len = 0 or memory_partition_caps"
       << " was a null ptr"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS, false);
    LOG_ERROR(ss);
    return RSMI_STATUS_INVALID_ARGS;
  }
  CHK_SUPPORT_NAME_ONLY(memory_partition_caps)
  DEVICE_MUTEX

  std::string availableMemoryPartitions;
  rsmi_status_t ret =
      get_dev_value_line(amd::smi::kDevAvailableMemoryPartition,
                         dv_ind, &availableMemoryPartitions);
  if (ret != RSMI_STATUS_SUCCESS) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | FAIL "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevAvailableMemoryPartition)
       << " | Data: could not retrieve requested data"
       << " | Returning = "
       << getRSMIStatusString(ret, false);
    LOG_ERROR(ss);
    return ret;
  }

  std::size_t length = availableMemoryPartitions.copy(memory_partition_caps, len-1);
  memory_partition_caps[length]='\0';

  if (len < (availableMemoryPartitions.size() + 1)) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Fail "
       << " | Device #: " << dv_ind
       << " | Type: "
       << amd::smi::Device::get_type_string(amd::smi::kDevAvailableMemoryPartition)
       << " | Cause: requested size was insufficient"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INSUFFICIENT_SIZE, false);
    LOG_ERROR(ss);
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success "
     << " | Device #: " << dv_ind
     << " | Type: "
     << amd::smi::Device::get_type_string(amd::smi::kDevAvailableMemoryPartition)
     << " | Data: " << memory_partition_caps
     << " | Returning = "
     << getRSMIStatusString(ret, false);
  LOG_TRACE(ss);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_partition_id_get(uint32_t dv_ind, uint32_t *partition_id) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======, " << dv_ind;
  LOG_TRACE(ss);
  if (partition_id == nullptr) {
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | FAIL"
       << " | Device #: " << dv_ind
       << " | Type: partition_id"
       << " | Data: nullptr"
       << " | Returning = "
       << getRSMIStatusString(RSMI_STATUS_INVALID_ARGS) << " |";
    LOG_ERROR(ss);
    return RSMI_STATUS_INVALID_ARGS;
  }
  DEVICE_MUTEX
  std::string strCompPartition = "UNKNOWN";
  const uint32_t PARTITION_LEN = 10;
  char compute_partition[PARTITION_LEN];
  compute_partition[0] = '\0';
  rsmi_status_t ret = rsmi_dev_compute_partition_get(dv_ind, compute_partition, PARTITION_LEN);
  if (ret == RSMI_STATUS_SUCCESS) {
    strCompPartition.clear();
    strCompPartition = compute_partition;
  }
  uint64_t pci_id = UINT64_MAX;
  *partition_id = UINT32_MAX;
  ret = rsmi_dev_pci_id_get(dv_ind, &pci_id);
  if (ret == RSMI_STATUS_SUCCESS) {
    *partition_id = static_cast<uint32_t>((pci_id >> 28) & 0xf);
  }
  std::ostringstream bdf_sstream;
  bdf_sstream << std::hex << std::setfill('0') << std::setw(4)
  << ((pci_id >> 32) & 0xFFFFFFFF) << ":";
  bdf_sstream << std::hex << std::setfill('0') << std::setw(2) << ((pci_id >> 8) & 0xFF) << ":";
  bdf_sstream << std::hex << std::setfill('0') << std::setw(2) << ((pci_id >> 3) & 0x1F) << ".";
  bdf_sstream << std::hex << std::setfill('0') << +(pci_id & 0x7);
  bdf_sstream << "\n[Option 1] Partition ID ((pci_id >> 28) & 0xf): " << std::dec
  << static_cast<int>((pci_id >> 28) & 0xf);
  bdf_sstream << "\n[Option 2] Partition ID (pci_id & 0x7): " << std::dec
  << static_cast<int>(pci_id & 0x7);
  // std::cout << __PRETTY_FUNCTION__ << " BDF: " << bdf_sstream.str() << std::endl;

  /**
   * Fall back is required due to driver changes within KFD.
   * Some devices may report bits [31:28] or [2:0].
   * With the newly added rsmi_dev_partition_id_get(..),
   * we provided this fallback to properly retrieve the partition ID. We
   * plan to eventually remove partition ID from the function portion of the
   * BDF (Bus Device Function). See below for PCI ID description.
   *
   * bits [63:32] = domain
   * bits [31:28] or bits [2:0] = partition id
   * bits [27:16] = reserved
   * bits [15:8]  = Bus
   * bits [7:3] = Device
   * bits [2:0] = Function (partition id maybe in bits [2:0]) <-- Fallback for non SPX modes
   */

  // If the partition_id is still not set (bits [31:28]), we will use the fallback
  // in function bits. We will use bits [2:0] as the partition ID.
  if (*partition_id != UINT32_MAX && *partition_id == 0 &&
     static_cast<uint32_t>(pci_id & 0x7) != 0) {
    *partition_id = static_cast<uint32_t>(pci_id & 0x7);
  }
  ss << __PRETTY_FUNCTION__
     << " | ======= end ======= "
     << " | Success"
     << " | Device #: " << dv_ind
     << " | Compute Partition: " << strCompPartition
     << " | Type: partition_id"
     << " | Data: " << static_cast<int>(*partition_id)
     << " | Returning = "
     << getRSMIStatusString(RSMI_STATUS_SUCCESS) << " |"
     << "\n BDF: " << bdf_sstream.str() << std::endl;
  // std::cout << ss.str() << std::endl;
  LOG_INFO(ss);
  return ret;
  CATCH
}

rsmi_status_t rsmi_dev_target_graphics_version_get(uint32_t dv_ind,
                                            uint64_t *gfx_version) {
    TRY
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | ======= start ======="
       << " | Device #: " << dv_ind;
    LOG_TRACE(ss);
    rsmi_status_t ret = RSMI_STATUS_NOT_SUPPORTED;
    std::string version = "";
    const uint64_t undefined_gfx_version = std::numeric_limits<uint64_t>::max();
    if (gfx_version == nullptr) {
      ret = RSMI_STATUS_INVALID_ARGS;
    } else {
      *gfx_version = undefined_gfx_version;
      ret = amd::smi::rsmi_get_gfx_target_version(dv_ind , &version);
    }
    if (ret == RSMI_STATUS_SUCCESS) {
      version = amd::smi::removeString(version, "gfx");
      *gfx_version = uint64_t(std::stoull(version, nullptr, 16));
    }
    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Returning: " << getRSMIStatusString(ret, false)
       << " | Device #: " << dv_ind
       << " | Type: Target_graphics_version"
       << " | Data: "
       << ((gfx_version == nullptr) ? "nullptr" :
           amd::smi::print_unsigned_hex_and_int(*gfx_version));
    LOG_TRACE(ss);
    return ret;
    CATCH
}

rsmi_status_t rsmi_dev_guid_get(uint32_t dv_ind, uint64_t *guid) {
    TRY
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | ======= start ======="
       << " | Device #: " << dv_ind;
    LOG_TRACE(ss);
    GET_DEV_AND_KFDNODE_FROM_INDX
    uint64_t kgd_gpu_id = 0;
    rsmi_status_t resp = RSMI_STATUS_NOT_SUPPORTED;
    int ret = kfd_node->KFDNode::get_gpu_id(&kgd_gpu_id);
    resp = amd::smi::ErrnoToRsmiStatus(ret);

    if (guid == nullptr) {
      resp = RSMI_STATUS_INVALID_ARGS;
    } else {
      *guid = kgd_gpu_id;
    }

    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Returning: " << getRSMIStatusString(resp, false)
       << " | Device #: " << dv_ind
       << " | Type: GUID (gpu_id)"
       << " | Data: " << ((guid == nullptr) ? "nullptr" :
          amd::smi::print_unsigned_hex_and_int(*guid));
    LOG_INFO(ss);
    return resp;
    CATCH
}

rsmi_status_t rsmi_dev_node_id_get(uint32_t dv_ind, uint32_t *node_id) {
    TRY
     std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << " | ======= start ======="
       << " | Device #: " << dv_ind;
    LOG_TRACE(ss);
    GET_DEV_AND_KFDNODE_FROM_INDX
    uint32_t kfd_node_id = std::numeric_limits<uint32_t>::max();
    rsmi_status_t resp = RSMI_STATUS_NOT_SUPPORTED;
    int ret = kfd_node->KFDNode::get_node_id(&kfd_node_id);
    resp = amd::smi::ErrnoToRsmiStatus(ret);

    if (node_id == nullptr) {
      resp = RSMI_STATUS_INVALID_ARGS;
    } else {
      *node_id = kfd_node_id;
      if (kfd_node_id == std::numeric_limits<uint32_t>::max()) {
        resp = RSMI_STATUS_NOT_SUPPORTED;
      }
    }

    ss << __PRETTY_FUNCTION__
       << " | ======= end ======= "
       << " | Returning: " << getRSMIStatusString(resp, false)
       << " | Device #: " << dv_ind
       << " | Type: node_id"
       << " | Data: " << ((node_id == nullptr) ? "nullptr" :
          amd::smi::print_unsigned_hex_and_int(*node_id));
    LOG_INFO(ss);
    return resp;
    CATCH
}

enum iterator_handle_type {
  FUNC_ITER = 0,
  VARIANT_ITER,
  SUBVARIANT_ITER,
};

rsmi_status_t
rsmi_dev_supported_func_iterator_open(uint32_t dv_ind,
                                         rsmi_func_id_iter_handle_t *handle) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  GET_DEV_FROM_INDX

  if (handle == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  dev->fillSupportedFuncs();

  *handle = new rsmi_func_id_iter_handle;

  if (*handle == nullptr) {
    return RSMI_STATUS_OUT_OF_RESOURCES;
  }

  (*handle)->id_type = FUNC_ITER;

  if (dev->supported_funcs()->begin() == dev->supported_funcs()->end()) {
    delete *handle;
    return RSMI_STATUS_NO_DATA;
  }

  SupportedFuncMapIt *supp_func_iter = new SupportedFuncMapIt;

  if (supp_func_iter == nullptr) {
    return RSMI_STATUS_OUT_OF_RESOURCES;
  }
  *supp_func_iter = dev->supported_funcs()->begin();

  (*handle)->func_id_iter = reinterpret_cast<uintptr_t>(supp_func_iter);
  (*handle)->container_ptr =
                        reinterpret_cast<uintptr_t>(dev->supported_funcs());

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_supported_variant_iterator_open(
                                 rsmi_func_id_iter_handle_t parent_iter,
                                       rsmi_func_id_iter_handle_t *var_iter) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  if (var_iter == nullptr || parent_iter->id_type == SUBVARIANT_ITER) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  if (parent_iter->func_id_iter == 0) {
    return RSMI_STATUS_NO_DATA;
  }

  *var_iter = new rsmi_func_id_iter_handle;

  if (*var_iter == nullptr) {
    return RSMI_STATUS_OUT_OF_RESOURCES;
  }

  VariantMapIt *variant_itr = nullptr;
  SubVariantIt *sub_var_itr = nullptr;
  SupportedFuncMapIt *func_iter;
  std::shared_ptr<VariantMap> var_map_container;
  std::shared_ptr<SubVariant> sub_var_map_container;

  switch (parent_iter->id_type) {
    case FUNC_ITER:
      func_iter =
            reinterpret_cast<SupportedFuncMapIt *>(parent_iter->func_id_iter);
      var_map_container = (*func_iter)->second;

      if (var_map_container == nullptr) {
        delete *var_iter;
        return RSMI_STATUS_NO_DATA;
      }

      variant_itr = new VariantMapIt;
      *variant_itr = var_map_container->begin();
      (*var_iter)->func_id_iter = reinterpret_cast<uintptr_t>(variant_itr);
      (*var_iter)->container_ptr =
                         reinterpret_cast<uintptr_t>(var_map_container.get());
      (*var_iter)->id_type = VARIANT_ITER;
      break;

    case VARIANT_ITER:
      variant_itr =
                  reinterpret_cast<VariantMapIt *>(parent_iter->func_id_iter);
      sub_var_map_container = (*variant_itr)->second;

      if (sub_var_map_container == nullptr) {
        delete *var_iter;
        return RSMI_STATUS_NO_DATA;
      }

      sub_var_itr = new SubVariantIt;
      *sub_var_itr = sub_var_map_container->begin();
      (*var_iter)->func_id_iter = reinterpret_cast<uintptr_t>(sub_var_itr);
      (*var_iter)->container_ptr =
                     reinterpret_cast<uintptr_t>(sub_var_map_container.get());
      (*var_iter)->id_type = SUBVARIANT_ITER;
      break;

    default:
      assert(false);  // Unexpected iterator type
      return RSMI_STATUS_INVALID_ARGS;
  }
  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_supported_func_iterator_close(rsmi_func_id_iter_handle_t *handle) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);

  if (handle == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  if ((*handle)->id_type == FUNC_ITER) {
    SupportedFuncMapIt *supp_func_iter =
              reinterpret_cast<SupportedFuncMapIt *>((*handle)->func_id_iter);
    delete supp_func_iter;
  } else if ((*handle)->id_type == VARIANT_ITER) {
    VariantMapIt *var_iter =
                    reinterpret_cast<VariantMapIt *>((*handle)->func_id_iter);
    delete var_iter;
  }  else if ((*handle)->id_type == SUBVARIANT_ITER) {
    SubVariantIt *subvar_iter =
                      reinterpret_cast<SubVariantIt *>((*handle)->func_id_iter);
    delete subvar_iter;
  } else {
    return RSMI_STATUS_INVALID_ARGS;
  }

  delete *handle;

  *handle = nullptr;

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_func_iter_value_get(rsmi_func_id_iter_handle_t handle,
                                                rsmi_func_id_value_t *value) {
  TRY
  if (value == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  if (handle->func_id_iter == 0) {
    return RSMI_STATUS_NO_DATA;
  }

  SupportedFuncMapIt *func_itr = nullptr;
  VariantMapIt *variant_itr = nullptr;
  SubVariantIt *sub_var_itr = nullptr;

  switch (handle->id_type) {
    case FUNC_ITER:
      func_itr = reinterpret_cast<SupportedFuncMapIt *>(handle->func_id_iter);
      value->name = (*func_itr)->first.c_str();
      break;

    case VARIANT_ITER:
      variant_itr = reinterpret_cast<VariantMapIt *>(handle->func_id_iter);
      value->id = (*variant_itr)->first;
      break;

    case SUBVARIANT_ITER:
      sub_var_itr = reinterpret_cast<SubVariantIt *>(handle->func_id_iter);

      // The hwmon file index that is appropriate for the rsmi user is stored
      // at bit position MONITOR_TYPE_BIT_POSITION.
      value->id = *(*sub_var_itr) >> MONITOR_TYPE_BIT_POSITION;
      break;

    default:
      return RSMI_STATUS_INVALID_ARGS;
  }
  CATCH
  return RSMI_STATUS_SUCCESS;
}

rsmi_status_t
rsmi_func_iter_next(rsmi_func_id_iter_handle_t handle) {
  TRY
  if (handle->func_id_iter == 0) {
    return RSMI_STATUS_NO_DATA;
  }

  SupportedFuncMapIt *func_iter;
  VariantMapIt *var_iter;
  SubVariantIt *sub_var_iter;

  switch (handle->id_type) {
    case FUNC_ITER:
      func_iter = reinterpret_cast<SupportedFuncMapIt *>(handle->func_id_iter);
      (*func_iter)++;

      if (*func_iter ==
         reinterpret_cast<SupportedFuncMap *>(handle->container_ptr)->end()) {
        return RSMI_STATUS_NO_DATA;
      }
      break;

    case VARIANT_ITER:
      var_iter = reinterpret_cast<VariantMapIt *>(handle->func_id_iter);
      (*var_iter)++;
      if (*var_iter ==
               reinterpret_cast<VariantMap *>(handle->container_ptr)->end()) {
        return RSMI_STATUS_NO_DATA;
      }
      break;

    case SUBVARIANT_ITER:
      sub_var_iter = reinterpret_cast<SubVariantIt *>(handle->func_id_iter);
      (*sub_var_iter)++;
      if (*sub_var_iter ==
               reinterpret_cast<SubVariant *>(handle->container_ptr)->end()) {
        return RSMI_STATUS_NO_DATA;
      }
      break;

    default:
      return RSMI_STATUS_INVALID_ARGS;
  }

  return RSMI_STATUS_SUCCESS;

  CATCH
}


static bool check_evt_notif_support(int kfd_fd) {
  struct kfd_ioctl_get_version_args args = {0, 0};

  if (ioctl(kfd_fd, AMDKFD_IOC_GET_VERSION, &args) == -1) {
    return false;
  }

  if (args.minor_version < 3) {
    return false;
  }
  return true;
}

static const char *kPathKFDIoctl = "/dev/kfd";

rsmi_status_t
rsmi_event_notification_init(uint32_t dv_ind) {
  TRY
  GET_DEV_FROM_INDX
  DEVICE_MUTEX

  std::lock_guard<std::mutex> guard(*smi.kfd_notif_evt_fh_mutex());
  if (smi.kfd_notif_evt_fh() == -1) {
    assert(smi.kfd_notif_evt_fh_refcnt() == 0);
    int kfd_fd = open(kPathKFDIoctl, O_RDWR | O_CLOEXEC);

    if (kfd_fd <= 0) {
      return RSMI_STATUS_FILE_ERROR;
    }

    if (!check_evt_notif_support(kfd_fd)) {
      close(kfd_fd);
      return RSMI_STATUS_NOT_SUPPORTED;
    }

    smi.set_kfd_notif_evt_fh(kfd_fd);
  }
  (void)smi.kfd_notif_evt_fh_refcnt_inc();
  struct kfd_ioctl_smi_events_args args;

  assert(dev->kfd_gpu_id() <= UINT32_MAX);
  args.gpuid = static_cast<uint32_t>(dev->kfd_gpu_id());

  int ret = ioctl(smi.kfd_notif_evt_fh(), AMDKFD_IOC_SMI_EVENTS, &args);
  if (ret < 0) {
    return amd::smi::ErrnoToRsmiStatus(errno);
  }
  if (args.anon_fd < 1) {
    return RSMI_STATUS_NO_DATA;
  }

  dev->set_evt_notif_anon_fd(args.anon_fd);
  FILE *anon_file_ptr = fdopen(static_cast<int>(args.anon_fd), "r");
  if (anon_file_ptr == nullptr) {
    close(dev->evt_notif_anon_fd());
    return amd::smi::ErrnoToRsmiStatus(errno);
  }
  dev->set_evt_notif_anon_file_ptr(anon_file_ptr);

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_event_notification_mask_set(uint32_t dv_ind, uint64_t mask) {
  TRY
  GET_DEV_FROM_INDX
  DEVICE_MUTEX

  if (dev->evt_notif_anon_fd() == -1) {
    return RSMI_INITIALIZATION_ERROR;
  }
  ssize_t ret = write(dev->evt_notif_anon_fd(), &mask, sizeof(uint64_t));

  if (ret == -1) {
    return amd::smi::ErrnoToRsmiStatus(errno);
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_event_notification_get(int timeout_ms,
                     uint32_t *num_elem, rsmi_evt_notification_data_t *data) {
  TRY

  if (num_elem == nullptr || data == nullptr || *num_elem == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  uint32_t buffer_size = *num_elem;
  *num_elem = 0;

  rsmi_evt_notification_data_t *data_item;

  //  struct pollfd {
  //      int   fd;         /* file descriptor */
  //      short events;     /* requested events */
  //      short revents;    /* returned events */
  //  };
  std::vector<struct pollfd> fds;
  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
  std::vector<uint32_t> fd_indx_to_dev_id;

  for (uint32_t i = 0; i < smi.devices().size(); ++i) {
    if (smi.devices()[i]->evt_notif_anon_fd() == -1) {
      continue;
    }
    fds.push_back({smi.devices()[i]->evt_notif_anon_fd(),
                                                     POLLIN | POLLRDNORM, 0});
    fd_indx_to_dev_id.push_back(i);
  }

  auto fill_data_buffer = [&](bool did_poll) {
    for (uint32_t i = 0; i < fds.size(); ++i) {
      if (did_poll) {
        if (!(fds[i].revents & (POLLIN | POLLRDNORM))) {
          continue;
        }
      }

      if (*num_elem >= buffer_size) {
        return;
      }

      FILE *anon_fp =
         smi.devices()[fd_indx_to_dev_id[i]]->evt_notif_anon_file_ptr();
      data_item =
           reinterpret_cast<rsmi_evt_notification_data_t *>(&data[*num_elem]);

      uint32_t event;
      char event_in[MAX_EVENT_NOTIFICATION_MSG_SIZE];
      memcpy(reinterpret_cast<char *>(event_in), "\0", MAX_EVENT_NOTIFICATION_MSG_SIZE);
      while (fgets(event_in, MAX_EVENT_NOTIFICATION_MSG_SIZE, anon_fp)) {
        /* Output is in format as "event_number message_information\n"
         * Both event are expressed in hex.
         * information is a string
         */
        char message[MAX_EVENT_NOTIFICATION_MSG_SIZE];
        // parse the line here for event_number and rest of message_information
        sscanf(event_in, "%x %[^\n]\n", &event, message);

        // parse message based on event received
        switch (event){
          case RSMI_EVT_NOTIF_NONE:
            strcpy(reinterpret_cast<char *>(&data_item->message), "Event type None received");
            break;
          case RSMI_EVT_NOTIF_VMFAULT:
          {
            uint32_t pid;
            char task_name[MAX_EVENT_NOTIFICATION_MSG_SIZE];
            memcpy(reinterpret_cast<char *>(task_name), "\0", MAX_EVENT_NOTIFICATION_MSG_SIZE);

            sscanf(message, "%x:%s\n", &pid, task_name);
            std::stringstream final_message;
            final_message << "PID: " << std::to_string(pid).c_str()
                          << "  task name: " << task_name;

            strcpy(reinterpret_cast<char *>(&data_item->message), final_message.str().c_str());
          }
          break;
          case RSMI_EVT_NOTIF_THERMAL_THROTTLE:
          {
            uint64_t bitmask;
            uint64_t counter;

            sscanf(message, "%" PRIx64 ":%" PRIx64 "\n", &bitmask, &counter);
            std::stringstream final_message;
            final_message << "bitmask: 0x" << std::hex << bitmask
                          << "  counter: 0x" << std::hex << counter;

            strcpy(reinterpret_cast<char *>(&data_item->message), final_message.str().c_str());
          }
          break;
          case RSMI_EVT_NOTIF_GPU_PRE_RESET:
          {
            uint32_t reset_seq_num;
            char reset_cause[MAX_EVENT_NOTIFICATION_MSG_SIZE];
            memcpy(reinterpret_cast<char *>(reset_cause), "\0", MAX_EVENT_NOTIFICATION_MSG_SIZE);

            sscanf(message, "%x %[^\n]\n", &reset_seq_num, reset_cause);
            std::stringstream final_message;
            final_message << "reset sequence number: " << std::to_string(reset_seq_num).c_str()
                          << "  reset cause: " << reset_cause;

            strcpy(reinterpret_cast<char *>(&data_item->message), final_message.str().c_str());
          }
          break;
          case RSMI_EVT_NOTIF_GPU_POST_RESET:
          {
            uint32_t reset_seq_num;

            char tmp[MAX_EVENT_NOTIFICATION_MSG_SIZE];
            sscanf(message, "%x %[^\n]\n", &reset_seq_num, tmp);
            std::stringstream final_message;
            final_message << "reset sequence number: " << std::to_string(reset_seq_num).c_str();

            strcpy(reinterpret_cast<char *>(&data_item->message), final_message.str().c_str());
          }
          break;
          case RSMI_EVT_NOTIF_EVENT_MIGRATE_START:
          {
            int64_t ns;
            int32_t pid;
            uint32_t start;
            uint32_t size;
            uint32_t from;
            uint32_t to;
            uint32_t prefetch_loc;
            uint32_t preferred_loc;
            int32_t migrate_trigger;

            sscanf(message, "%" PRId64 " -%d @%" PRIu32 "(%" PRIu32 ") %x->%x %x:%x %d\n", &ns, &pid, &start, &size, &from, &to, &prefetch_loc, &preferred_loc, &migrate_trigger);
            std::stringstream final_message;
            final_message << "nd: " << std::to_string(ns).c_str() 
                          << "  pid: " << std::to_string(pid).c_str()
                          << "  start: 0x" << std::hex << start
                          << "  size: 0x" << std::hex << size
                          << "  from: 0x" << std::hex << from
                          << "  to: 0x" << std::hex << to
                          << "  prefetch_loc: 0x" << std::hex << prefetch_loc
                          << "  preferred_loc: 0x" << std::hex << preferred_loc
                          << "  migrate_trigger: " << std::to_string(migrate_trigger).c_str();

            strcpy(reinterpret_cast<char *>(&data_item->message), final_message.str().c_str());
          }
          break;
          case RSMI_EVT_NOTIF_EVENT_MIGRATE_END:
          {
            int64_t ns;
            int32_t pid;
            uint32_t start;
            uint32_t size;
            uint32_t from;
            uint32_t to;
            uint32_t migrate_trigger;
            uint32_t error_code;

            sscanf(message, "%" PRId64 " -%d @%" PRIu32 "(%" PRIu32 ") %x->%x %d %d\n", &ns, &pid, &start, &size, &from, &to, &migrate_trigger, &error_code);
            std::stringstream final_message;
            final_message << "nd: " << std::to_string(ns).c_str() 
                          << "  pid: " << std::to_string(pid).c_str()
                          << "  start: 0x" << std::hex << start
                          << "  size: 0x" << std::hex << size
                          << "  from: 0x" << std::hex << from
                          << "  to: 0x" << std::hex << to
                          << "  migrate_trigger: " << std::to_string(migrate_trigger).c_str()
                          << "  error_code: " << std::to_string(error_code).c_str();

            strcpy(reinterpret_cast<char *>(&data_item->message), final_message.str().c_str());
          }
          break;
          case RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_START:
          {
            int64_t ns;
            int32_t pid;
            uint32_t addr;
            uint32_t node;
            char *rw = "\0";

            sscanf(message, "%" PRId64 " -%d @%" PRIx32 "(%x) %c\n", &ns, &pid, &addr, &node, rw);
            std::stringstream final_message;
            final_message << "ns: " << std::to_string(ns).c_str()
                          << "  pid: " << std::to_string(pid).c_str()
                          << "  addr: 0x" << std::hex << addr
                          << "  node: 0x" << std::hex << node
                          << "  rw: " << rw;

            strcpy(reinterpret_cast<char *>(&data_item->message), final_message.str().c_str());
          }
          break;
          case RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_END:
          {
            int64_t ns;
            int32_t pid;
            uint32_t addr;
            uint32_t node;
            char *migrate_update = "\0";

            sscanf(message, "%" PRId64 " -%d @%" PRIx32 "(%x) %c\n", &ns, &pid, &addr, &node, migrate_update);
            std::stringstream final_message;
            final_message << "ns: " << std::to_string(ns).c_str()
                          << "  pid: " << std::to_string(pid).c_str()
                          << "  addr: 0x" << std::hex << addr
                          << "  node: 0x" << std::hex << node
                          << "  migrate_udpate: " << migrate_update;

            strcpy(reinterpret_cast<char *>(&data_item->message), final_message.str().c_str());
          }
          break;
          case RSMI_EVT_NOTIF_EVENT_QUEUE_EVICTION:
          {
            int64_t ns;
            int32_t pid;
            uint32_t node;
            uint32_t evict_trigger;

            sscanf(message, "%" PRId64 "-%d %x %d\n", &ns, &pid, &node, &evict_trigger);
            std::stringstream final_message;
            final_message << "ns: " << std::to_string(ns).c_str()
                          << "  pid: " << std::to_string(pid).c_str()
                          << "  node: 0x" << std::hex << node
                          << "  evict_trigger: " << std::to_string(evict_trigger).c_str();

            strcpy(reinterpret_cast<char *>(&data_item->message), final_message.str().c_str());
          }
          break;
          case RSMI_EVT_NOTIF_EVENT_QUEUE_RESTORE:
          {
            int64_t ns;
            int32_t pid;
            uint32_t node;
            char *rescheduled = "\0";

            sscanf(message, "%" PRId64 "-%d %x %c\n", &ns, &pid, &node, rescheduled);
            std::stringstream final_message;
            final_message << "ns: " << std::to_string(ns).c_str()
                          << "  pid: " << std::to_string(pid).c_str()
                          << "  node: 0x" << std::hex << node
                          << "  rescheduled: " << rescheduled;

            strcpy(reinterpret_cast<char *>(&data_item->message), final_message.str().c_str());
          }
          break;
          case RSMI_EVT_NOTIF_EVENT_UNMAP_FROM_GPU:
          {
            int64_t ns;
            int32_t pid;
            uint32_t addr;
            uint32_t size;
            uint32_t node;
            uint32_t unmap_trigger;

            sscanf(message, "%" PRId64 " -%d @%" PRIx32 "(%" PRIx32 ") %x %d\n", &ns, &pid, &addr, &size, &node, &unmap_trigger);
            std::stringstream final_message;
            final_message << "ns: " << std::to_string(ns).c_str()
                          << "  pid: " << std::to_string(pid).c_str()
                          << "  addr: 0x" <<std::hex << addr
                          << "  size: 0x" <<std::hex << size
                          << "  node: 0x" << std::hex << node
                          << "  unmap_trigger: " << std::to_string(unmap_trigger).c_str();

            strcpy(reinterpret_cast<char *>(&data_item->message), final_message.str().c_str());
          }
          break;
          default:
            strcpy(reinterpret_cast<char *>(&data_item->message), "Unknown event received");
            break;
        }
        data_item->event = (rsmi_evt_notification_type_t)event;
        data_item->dv_ind = fd_indx_to_dev_id[i];
        ++(*num_elem);

        // zero out event_in after each use
        memcpy(reinterpret_cast<char *>(event_in), "\0", MAX_EVENT_NOTIFICATION_MSG_SIZE);

        if (*num_elem >= buffer_size) {
          break;
        }
        data_item =
             reinterpret_cast<rsmi_evt_notification_data_t *>(&data[*num_elem]);
      }
    }
  };

  // Collect any left-over events from a poll in a previous call to
  // rsmi_event_notification_get()
  fill_data_buffer(false);

  if (*num_elem < buffer_size && errno != EAGAIN) {
    return amd::smi::ErrnoToRsmiStatus(errno);
  }
  if (*num_elem >= buffer_size) {
    return RSMI_STATUS_SUCCESS;
  }

  // We still have buffer left, see if there are any new events
  int p_ret = poll(fds.data(), fds.size(), timeout_ms);
  if (p_ret > 0) {
    fill_data_buffer(true);
  } else if (p_ret < 0) {
    return amd::smi::ErrnoToRsmiStatus(errno);
  }
  if (*num_elem == 0) {
    return RSMI_STATUS_NO_DATA;
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t rsmi_event_notification_stop(uint32_t dv_ind) {
  TRY
  GET_DEV_FROM_INDX
  DEVICE_MUTEX

  std::lock_guard<std::mutex> guard(*smi.kfd_notif_evt_fh_mutex());

  if (dev->evt_notif_anon_fd() == -1) {
    return RSMI_STATUS_INVALID_ARGS;
  }
//  close(dev->evt_notif_anon_fd());
  FILE *anon_fp = smi.devices()[dv_ind]->evt_notif_anon_file_ptr();
  fclose(anon_fp);
  assert(errno == 0 || errno == EAGAIN);
  dev->set_evt_notif_anon_file_ptr(nullptr);
  dev->set_evt_notif_anon_fd(-1);

  if (smi.kfd_notif_evt_fh_refcnt_dec() == 0) {
    int ret = close(smi.kfd_notif_evt_fh());
    smi.set_kfd_notif_evt_fh(-1);
    if (ret < 0) {
      return amd::smi::ErrnoToRsmiStatus(errno);
    }
  }

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_metrics_header_info_get(uint32_t dv_ind, metrics_table_header_t* header_value)
{
  TRY
  std::ostringstream ostrstream;
  ostrstream << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ostrstream);

  assert(header_value != nullptr);
  if (header_value == nullptr) {
    return rsmi_status_t::RSMI_STATUS_INVALID_ARGS;
  }

  auto status_code = rsmi_dev_gpu_metrics_header_info_get(dv_ind, *header_value);
  ostrstream << __PRETTY_FUNCTION__
             << " | ======= end ======= "
             << " | End Result "
             << " | Device #:  " << dv_ind
             << " | Format Revision: " << header_value->format_revision
             << " | Content Revision: " << header_value->content_revision
             << " | Header Size: " << header_value->structure_size
             << " | Returning = " << status_code << " " << getRSMIStatusString(status_code) << " |";
  LOG_INFO(ostrstream);

  return status_code;
  CATCH
}


rsmi_status_t
rsmi_dev_metrics_xcd_counter_get(uint32_t dv_ind, uint16_t* xcd_counter_value)
{
  TRY
  std::ostringstream ostrstream;
  ostrstream << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ostrstream);

  assert(xcd_counter_value != nullptr);
  if (xcd_counter_value == nullptr) {
    return rsmi_status_t::RSMI_STATUS_INVALID_ARGS;
  }

  auto xcd_counter = uint16_t(0);
  rsmi_gpu_metrics_t gpu_metrics;
  auto status_code = rsmi_dev_gpu_metrics_info_get(dv_ind, &gpu_metrics);
  if (status_code == rsmi_status_t::RSMI_STATUS_SUCCESS) {
    for (const auto& gfxclk : gpu_metrics.current_gfxclks) {
      if (gfxclk == UINT16_MAX) {
        break;
      }
      if ((gfxclk != 0) && (gfxclk != UINT16_MAX)) {
        xcd_counter++;
      }
    }
  }

  *xcd_counter_value = xcd_counter;
  ostrstream << __PRETTY_FUNCTION__
             << " | ======= end ======= "
             << " | End Result "
             << " | Device #:  " << dv_ind
             << " | XCDs counter: " << xcd_counter
             << " | Returning = " << status_code << " " << getRSMIStatusString(status_code) << " |";
  LOG_INFO(ostrstream);

  return status_code;
  CATCH
}

rsmi_status_t
rsmi_dev_metrics_log_get(uint32_t dv_ind)
{
  TRY
  std::ostringstream ostrstream;
  ostrstream << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ostrstream);

  GET_DEV_FROM_INDX
  auto status_code = dev->dev_log_gpu_metrics(ostrstream);
  ostrstream << __PRETTY_FUNCTION__
             << " | ======= end ======= "
             << " | End Result "
             << " | Device #:  " << dv_ind
             << " | Metric Type: " << "All GPU Metrics..."
             << " | Returning = " << status_code << " " << getRSMIStatusString(status_code) << " |";
  LOG_INFO(ostrstream);

  return status_code;
  CATCH
}

rsmi_status_t rsmi_dev_device_identifiers_get(uint32_t dv_ind,
                  rsmi_device_identifiers_t *smi_device_identifiers) {
  TRY
  std::ostringstream ss;
  ss << __PRETTY_FUNCTION__ << "| ======= start =======";
  LOG_TRACE(ss);
  GET_DEV_FROM_INDX
  if (smi_device_identifiers == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  rsmi_status_t ret = RSMI_STATUS_NOT_SUPPORTED;
  return ret = dev->get_smi_device_identifiers(dv_ind, smi_device_identifiers);
  CATCH
}


// UNDOCUMENTED FUNCTIONS
// This functions are not declared in rocm_smi.h. They are either not fully
// supported, or to be used for test purposes.

// This function acquires a mutex and waits for a number of seconds
rsmi_status_t
rsmi_test_sleep(uint32_t dv_ind, uint32_t seconds) {
//  DEVICE_MUTEX
  amd::smi::pthread_wrap _pw(*amd::smi::GetMutex(dv_ind));
  amd::smi::RocmSMI& smi_ = amd::smi::RocmSMI::getInstance();
  bool blocking_ = !(smi_.init_options() &
                      static_cast<uint64_t>(RSMI_INIT_FLAG_RESRV_TEST1));
  amd::smi::ScopedPthread _lock(_pw, blocking_);
  if (!blocking_ && _lock.mutex_not_acquired()) {
    return RSMI_STATUS_BUSY;
  }

  sleep(seconds);
  return RSMI_STATUS_SUCCESS;
}

int32_t
rsmi_test_refcount(uint64_t refcnt_type) {
  (void)refcnt_type;

  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
  std::lock_guard<std::mutex> guard(*smi.bootstrap_mutex());

  if (smi.ref_count() == 0 && !smi.devices().empty()) {
    return -1;
  }

  return static_cast<int32_t>(smi.ref_count());
}

