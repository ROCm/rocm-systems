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
#include <errno.h>
#include <sys/utsname.h>
#include <pthread.h>
#include <string.h>

#include <sstream>
#include <algorithm>
#include <cerrno>
#include <bitset>
#include <cstdint>
#include <unordered_map>
#include <map>
#include <fstream>
#include <iostream>

#include "rocm_smi/rocm_smi_common.h"  // Should go before rocm_smi.h
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_exception.h"
#include "rocm_smi/rocm_smi_counters.h"
#include "rocm_smi/rocm_smi_kfd.h"

#include "rocm_smi/rocm_smi64Config.h"

static const uint32_t kMaxOverdriveLevel = 20;

static rsmi_status_t handleException() {
  try {
    throw;
  } catch (const std::bad_alloc& e) {
    debug_print("RSMI exception: BadAlloc\n");
    return RSMI_STATUS_OUT_OF_RESOURCES;
  } catch (const amd::smi::rsmi_exception& e) {
    debug_print("Exception caught: %s.\n", e.what());
    return e.error_code();
    return RSMI_STATUS_INTERNAL_EXCEPTION;
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

#define CHECK_DV_IND_RANGE \
    amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance(); \
    if (dv_ind >= smi.monitor_devices().size()) { \
      return RSMI_STATUS_INVALID_ARGS; \
    } \

#define GET_DEV_FROM_INDX  \
  CHECK_DV_IND_RANGE \
  std::shared_ptr<amd::smi::Device> dev = smi.monitor_devices()[dv_ind]; \
  assert(dev != nullptr);

#define REQUIRE_ROOT_ACCESS \
    if (amd::smi::RocmSMI::getInstance().euid()) { \
      return RSMI_STATUS_PERMISSION; \
    }

#define DEVICE_MUTEX \
    amd::smi::pthread_wrap _pw(*get_mutex(dv_ind)); \
    amd::smi::ScopedPthread _lock(_pw);

/* This group of macros is used to facilitate checking of support for rsmi_dev*
 * "getter" functions. When the return buffer is set to nullptr, the macro will
 * check the previously gathered device support data to see if the function,
 * with possible variants (e.g., memory types, firware types,...) and
 * subvariants (e.g. monitors/sensors) are supported.
 */
// This macro assumes dev already available
#define CHK_API_SUPPORT_ONLY(RT_PTR, VR, SUB_VR) \
    if ((RT_PTR) == nullptr) { \
      if (!dev->DeviceAPISupported(__FUNCTION__, (VR), (SUB_VR))) { \
        return RSMI_STATUS_NOT_SUPPORTED; \
      }  \
      return RSMI_STATUS_INVALID_ARGS; \
    } \

#define CHK_SUPPORT(RT_PTR, VR, SUB_VR)  \
    GET_DEV_FROM_INDX \
    CHK_API_SUPPORT_ONLY(RT_PTR, VR, SUB_VR)

#define CHK_SUPPORT_NAME_ONLY(RT_PTR) \
    CHK_SUPPORT((RT_PTR), RSMI_DEFAULT_VARIANT, RSMI_DEFAULT_VARIANT) \

#define CHK_SUPPORT_VAR(RT_PTR, VR) \
    CHK_SUPPORT((RT_PTR), (VR), RSMI_DEFAULT_VARIANT) \

#define CHK_SUPPORT_SUBVAR_ONLY(RT_PTR, SUB_VR) \
    CHK_SUPPORT((RT_PTR), RSMI_DEFAULT_VARIANT, (SUB_VR)) \

static pthread_mutex_t *get_mutex(uint32_t dv_ind) {
  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  if (dv_ind >= smi.monitor_devices().size()) {
    return nullptr;
  }
  std::shared_ptr<amd::smi::Device> dev = smi.monitor_devices()[dv_ind];
  assert(dev != nullptr);

  return dev->mutex();
}

static rsmi_status_t errno_to_rsmi_status(uint32_t err) {
  switch (err) {
    case 0:      return RSMI_STATUS_SUCCESS;
    case ESRCH:  return RSMI_STATUS_NOT_FOUND;
    case EACCES: return RSMI_STATUS_PERMISSION;
    case EPERM:
    case ENOENT: return RSMI_STATUS_NOT_SUPPORTED;
    case EBADF:
    case EISDIR: return RSMI_STATUS_FILE_ERROR;
    case EINTR:  return RSMI_STATUS_INTERRUPT;
    case EIO:    return RSMI_STATUS_UNEXPECTED_SIZE;
    default:     return RSMI_STATUS_UNKNOWN_ERROR;
  }
}

static uint64_t get_multiplier_from_str(char units_char) {
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
      assert(!"Unexpected units for frequency");
  }
  return multiplier;
}

/**
 * Parse a string of the form:
 *        "<int index>:  <int freq><freq. unit string> <|*>"
 */
static uint64_t freq_string_to_int(const std::vector<std::string> &freq_lines,
                                     bool *is_curr, uint32_t lanes[], int i) {
  std::istringstream fs(freq_lines[i]);

  uint32_t ind;
  long double freq;
  std::string junk;
  std::string units_str;
  std::string star_str;

  fs >> ind;
  fs >> junk;  // colon
  fs >> freq;
  fs >> units_str;
  fs >> star_str;

  if (is_curr != nullptr) {
    if (freq_lines[i].find("*") != std::string::npos) {
      *is_curr = true;
    } else {
      *is_curr = false;
    }
  }
  uint32_t multiplier = get_multiplier_from_str(units_str[0]);

  if (star_str[0] == 'x') {
    assert(lanes != nullptr && "Lanes are provided but null lanes pointer");
    if (lanes) {
      lanes[i] = std::stoi(star_str.substr(1), nullptr);
    }
  }
  return freq*multiplier;
}

static void freq_volt_string_to_point(std::string in_line,
                                                     rsmi_od_vddc_point_t *pt) {
  std::istringstream fs_vlt(in_line);

  assert(pt != nullptr);

  uint32_t ind;
  long double freq;
  long double volts;
  std::string junk;
  std::string freq_units_str;
  std::string volts_units_str;

  fs_vlt >> ind;
  fs_vlt >> junk;  // colon
  fs_vlt >> freq;
  fs_vlt >> freq_units_str;
  fs_vlt >> volts;
  fs_vlt >> volts_units_str;

  uint32_t multiplier = get_multiplier_from_str(freq_units_str[0]);

  pt->frequency = freq*multiplier;

  multiplier = get_multiplier_from_str(volts_units_str[0]);
  pt->voltage = volts*multiplier;

  return;
}

static void od_value_pair_str_to_range(std::string in_line, rsmi_range_t *rg) {
  std::istringstream fs_rng(in_line);

  assert(rg != nullptr);

  std::string clk;
  uint64_t lo;
  uint64_t hi;
  std::string lo_units_str;
  std::string hi_units_str;

  fs_rng >> clk;  // This is clk + colon; e.g., "SCLK:"
  fs_rng >> lo;
  fs_rng >> lo_units_str;
  fs_rng >> hi;
  fs_rng >> hi_units_str;

  uint32_t multiplier = get_multiplier_from_str(lo_units_str[0]);

  rg->lower_bound = lo*multiplier;

  multiplier = get_multiplier_from_str(hi_units_str[0]);
  rg->upper_bound = hi*multiplier;

  return;
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

  rsmi_power_profile_preset_masks_t ret = RSMI_PWR_PROF_PRST_INVALID;

  fs >> *prof_ind;
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
  GET_DEV_FROM_INDX
  int ret = dev->readDevInfo(type, val_str);

  return errno_to_rsmi_status(ret);
}
static rsmi_status_t get_dev_value_int(amd::smi::DevInfoTypes type,
                                         uint32_t dv_ind, uint64_t *val_int) {
  GET_DEV_FROM_INDX
  int ret = dev->readDevInfo(type, val_int);

  return errno_to_rsmi_status(ret);
}

static rsmi_status_t get_dev_value_line(amd::smi::DevInfoTypes type,
                                      uint32_t dv_ind, std::string *val_str) {
  GET_DEV_FROM_INDX
  int ret = dev->readDevInfoLine(type, val_str);

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

  int ret = dev->monitor()->writeMonitor(type, sensor_ind,
                                                         std::to_string(val));

  return errno_to_rsmi_status(ret);
}

static rsmi_status_t get_power_mon_value(amd::smi::PowerMonTypes type,
                                      uint32_t dv_ind, uint64_t *val) {
  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  if (dv_ind >= smi.monitor_devices().size() || val == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  uint32_t ret = smi.DiscoverAMDPowerMonitors();
  if (ret != 0) {
    return errno_to_rsmi_status(ret);
  }

  std::shared_ptr<amd::smi::Device> dev = smi.monitor_devices()[dv_ind];
  assert(dev != nullptr);
  assert(dev->monitor() != nullptr);

  ret = dev->power_monitor()->readPowerValue(type, val);

  return errno_to_rsmi_status(ret);
}

static rsmi_status_t get_dev_value_vec(amd::smi::DevInfoTypes type,
                         uint32_t dv_ind, std::vector<std::string> *val_vec) {
  GET_DEV_FROM_INDX

  int ret = dev->readDevInfo(type, val_vec);
  return errno_to_rsmi_status(ret);
}
static bool is_power_of_2(uint64_t n) {
      return n && !(n & (n - 1));
}

rsmi_status_t
rsmi_init(uint64_t flags) {
  TRY

  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();
  smi.Initialize(flags);

  return RSMI_STATUS_SUCCESS;
  CATCH
}

// A call to rsmi_shut_down is not technically necessary at this time,
// but may be in the future.
rsmi_status_t
rsmi_shut_down(void) {
  TRY

  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  smi.Cleanup();

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_num_monitor_devices(uint32_t *num_devices) {
  TRY
  if (num_devices == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  *num_devices = smi.monitor_devices().size();
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t rsmi_dev_ecc_enabled_get(uint32_t dv_ind,
                                                    uint64_t *enabled_blks) {
  TRY
  rsmi_status_t ret;
  std::string feature_line;
  std::string tmp_str;

  CHK_SUPPORT_NAME_ONLY(enabled_blks)

  DEVICE_MUTEX

  ret = get_dev_value_line(amd::smi::kDevErrCntFeatures, dv_ind, &feature_line);
  if (ret != RSMI_STATUS_SUCCESS) {
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

  return errno_to_rsmi_status(errno);
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

  CHK_SUPPORT_NAME_ONLY(state)

  if (!is_power_of_2(block)) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  rsmi_status_t ret;
  uint64_t features_mask;

  DEVICE_MUTEX

  ret = rsmi_dev_ecc_enabled_get(dv_ind, &features_mask);

  if (ret == RSMI_STATUS_FILE_ERROR) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  *state = (features_mask & block) ?
                     RSMI_RAS_ERR_STATE_ENABLED : RSMI_RAS_ERR_STATE_DISABLED;

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_ecc_count_get(uint32_t dv_ind, rsmi_gpu_block_t block,
                                                     rsmi_error_count_t *ec) {
  std::vector<std::string> val_vec;
  rsmi_status_t ret;

  TRY
  CHK_SUPPORT_VAR(ec, block)


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

    default:
      return RSMI_STATUS_NOT_SUPPORTED;
  }

  DEVICE_MUTEX

  ret = get_dev_value_vec(type, dv_ind, &val_vec);

  if (ret == RSMI_STATUS_FILE_ERROR) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  assert(val_vec.size() == 2);

  std::string junk;
  std::istringstream fs1(val_vec[0]);

  fs1 >> junk;
  assert(junk == "ue:");
  fs1 >> ec->uncorrectable_err;

  std::istringstream fs2(val_vec[1]);

  fs2 >> junk;
  assert(junk == "ce:");
  fs2 >> ec->correctable_err;

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_pci_id_get(uint32_t dv_ind, uint64_t *bdfid) {
  TRY

  CHK_SUPPORT_NAME_ONLY(bdfid)
  DEVICE_MUTEX

  *bdfid = dev->bdfid();

  int32_t ret = dev->populateKFDNodeProperties();

  if (ret) {
    return errno_to_rsmi_status(errno);
  }

  uint64_t domain = 0;

  ret = dev->getKFDNodeProperty(amd::smi::kDevKFDNodePropDomain, &domain);

  if (ret == EINVAL) {
    // "domain" is not found in properties file; just go with the 16 bit
    // domain already found
    return RSMI_STATUS_SUCCESS;
  }

  // Replace the 16 bit domain originally set like this:
  // BDFID = ((<DOMAIN> & 0xffff) << 32) | ((<BUS> & 0xff) << 8) |
  //                        ((device& 0x1f) <<3 ) | (function & 0x7)
  // with this:
  // BDFID = ((<DOMAIN> & 0xffffffff) << 32) | ((<BUS> & 0xff) << 8) |
  //                        ((device& 0x1f) <<3 ) | (function & 0x7)

  assert((domain & 0xFFFFFFFF00000000) == 0);
  (*bdfid) &= 0xFFFF;  // Clear out the old 16 bit domain
  *bdfid |= (domain & 0xFFFFFFFF) << 32;

  return RSMI_STATUS_SUCCESS;
  CATCH
}

static rsmi_status_t
get_id(uint32_t dv_ind, amd::smi::DevInfoTypes typ, uint16_t *id) {
  TRY
  std::string val_str;

  DEVICE_MUTEX

  rsmi_status_t ret = get_dev_value_str(typ, dv_ind, &val_str);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  errno = 0;
  *id = strtoul(val_str.c_str(), nullptr, 16);
  assert(errno == 0);

  return errno_to_rsmi_status(errno);
  CATCH
}

rsmi_status_t
rsmi_dev_id_get(uint32_t dv_ind, uint16_t *id) {
  CHK_SUPPORT_NAME_ONLY(id)
  DEVICE_MUTEX
  return get_id(dv_ind, amd::smi::kDevDevID, id);
}

rsmi_status_t
rsmi_dev_subsystem_id_get(uint32_t dv_ind, uint16_t *id) {
  CHK_SUPPORT_NAME_ONLY(id)
  DEVICE_MUTEX
  return get_id(dv_ind, amd::smi::kDevSubSysDevID, id);
}

rsmi_status_t
rsmi_dev_vendor_id_get(uint32_t dv_ind, uint16_t *id) {
  CHK_SUPPORT_NAME_ONLY(id)
  DEVICE_MUTEX
  return get_id(dv_ind, amd::smi::kDevVendorID, id);
}

rsmi_status_t
rsmi_dev_subsystem_vendor_id_get(uint32_t dv_ind, uint16_t *id) {
  CHK_SUPPORT_NAME_ONLY(id)
  DEVICE_MUTEX
  return get_id(dv_ind, amd::smi::kDevSubSysVendorID, id);
}

rsmi_status_t
rsmi_dev_perf_level_get(uint32_t dv_ind, rsmi_dev_perf_level_t *perf) {
  TRY
  std::string val_str;

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

rsmi_status_t
rsmi_dev_overdrive_level_get(uint32_t dv_ind, uint32_t *od) {
  TRY
  std::string val_str;
  CHK_SUPPORT_NAME_ONLY(od)
  DEVICE_MUTEX

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
  REQUIRE_ROOT_ACCESS

  if (od > kMaxOverdriveLevel) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  DEVICE_MUTEX
  return set_dev_value(amd::smi::kDevOverDriveLevel, dv_ind, od);
  CATCH
}

rsmi_status_t
rsmi_dev_perf_level_set(int32_t dv_ind, rsmi_dev_perf_level_t perf_level) {
  TRY
  REQUIRE_ROOT_ACCESS

  if (perf_level > RSMI_DEV_PERF_LEVEL_LAST) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX
  return set_dev_value(amd::smi::kDevPerfLevel, dv_ind, perf_level);
  CATCH
}

static rsmi_status_t get_frequencies(amd::smi::DevInfoTypes type,
            uint32_t dv_ind, rsmi_frequencies_t *f, uint32_t *lanes = nullptr) {
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

  if (val_vec.size() == 0) {
    return RSMI_STATUS_NOT_YET_IMPLEMENTED;
  }

  f->num_supported = val_vec.size();
  bool current = false;
  f->current = RSMI_MAX_NUM_FREQUENCIES + 1;  // init to an invalid value

  for (uint32_t i = 0; i < f->num_supported; ++i) {
    f->frequency[i] = freq_string_to_int(val_vec, &current, lanes, i);

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

  // Some older drivers will not have the current frequency set
  // assert(f->current < f->num_supported);
  if (f->current >= f->num_supported) {
    return RSMI_STATUS_NOT_SUPPORTED;
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

  ret = get_dev_value_vec(amd::smi::kDevPowerProfileMode, dv_ind, &val_vec);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }
  assert(val_vec.size() <= RSMI_MAX_NUM_POWER_PROFILES);

  p->num_profiles = val_vec.size() - 1;  // -1 for the header line
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

/* We expect the format of the the pp_od_clk_voltage file to look like this:
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
*/
static const uint32_t kOD_SCLK_label_array_index = 0;
static const uint32_t kOD_MCLK_label_array_index =
                                               kOD_SCLK_label_array_index + 3;
static const uint32_t kOD_VDDC_CURVE_label_array_index =
                                               kOD_MCLK_label_array_index + 2;
static const uint32_t kOD_OD_RANGE_label_array_index =
                                         kOD_VDDC_CURVE_label_array_index + 4;
static const uint32_t kOD_VDDC_CURVE_start_index =
                                           kOD_OD_RANGE_label_array_index + 3;
static const uint32_t kOD_VDDC_CURVE_num_lines =
                                               kOD_VDDC_CURVE_start_index + 4;

static rsmi_status_t get_od_clk_volt_info(uint32_t dv_ind,
                                                  rsmi_od_volt_freq_data_t *p) {
  TRY
  std::vector<std::string> val_vec;
  rsmi_status_t ret;

  assert(p != nullptr);

  ret = get_dev_value_vec(amd::smi::kDevPowerODVoltage, dv_ind, &val_vec);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  // This is a work-around to handle systems where kDevPowerODVoltage is not
  // fully supported yet.
  if (val_vec.size() < 2) {
    return RSMI_STATUS_NOT_YET_IMPLEMENTED;
  }

  assert(val_vec[kOD_SCLK_label_array_index] == "OD_SCLK:");

  p->curr_sclk_range.lower_bound = freq_string_to_int(val_vec, nullptr,
                                     nullptr, kOD_SCLK_label_array_index + 1);
  p->curr_sclk_range.upper_bound = freq_string_to_int(val_vec, nullptr,
                                     nullptr, kOD_SCLK_label_array_index + 2);


  // The condition below indicates old style format, which is not supported
  if (val_vec[kOD_MCLK_label_array_index] != "OD_MCLK:") {
    return RSMI_STATUS_NOT_YET_IMPLEMENTED;
  }
  p->curr_mclk_range.lower_bound = 0;

  p->curr_mclk_range.upper_bound = freq_string_to_int(val_vec, nullptr,
                                     nullptr, kOD_MCLK_label_array_index + 1);

  assert(val_vec[kOD_VDDC_CURVE_label_array_index] == "OD_VDDC_CURVE:");

  uint32_t tmp = kOD_VDDC_CURVE_label_array_index + 1;
  for (uint32_t i = 0; i < RSMI_NUM_VOLTAGE_CURVE_POINTS; ++i) {
    freq_volt_string_to_point(val_vec[tmp + i], &(p->curve.vc_points[i]));
  }

  assert(val_vec[kOD_OD_RANGE_label_array_index] == "OD_RANGE:");
  od_value_pair_str_to_range(val_vec[kOD_OD_RANGE_label_array_index + 1],
                                                      &(p->sclk_freq_limits));
  od_value_pair_str_to_range(val_vec[kOD_OD_RANGE_label_array_index + 2],
                                                      &(p->mclk_freq_limits));

  assert((val_vec.size() - kOD_VDDC_CURVE_start_index)%2 == 0);
  p->num_regions = (val_vec.size() - kOD_VDDC_CURVE_start_index) / 2;

  return RSMI_STATUS_SUCCESS;
  CATCH
}

static void get_vc_region(uint32_t start_ind,
                std::vector<std::string> *val_vec, rsmi_freq_volt_region_t *p) {
  assert(p != nullptr);
  assert(val_vec != nullptr);
  // There must be at least 1 region to read in
  assert(val_vec->size() >= kOD_OD_RANGE_label_array_index + 2);
  assert((*val_vec)[kOD_OD_RANGE_label_array_index] == "OD_RANGE:");

  od_value_pair_str_to_range((*val_vec)[start_ind], &p->freq_range);
  od_value_pair_str_to_range((*val_vec)[start_ind + 1], &p->volt_range);
  return;
}

/*
 * num_regions [inout] on calling, the number of regions requested to be read
 * in. At completion, the number of regions actually read in
 * 
 * p [inout] point to pre-allocated memory where function will write region
 * values. Caller must make sure there is enough space for at least
 * *num_regions regions. On 
 */
static rsmi_status_t get_od_clk_volt_curve_regions(uint32_t dv_ind,
                            uint32_t *num_regions, rsmi_freq_volt_region_t *p) {
  TRY
  std::vector<std::string> val_vec;
  rsmi_status_t ret;

  assert(num_regions != nullptr);
  assert(p != nullptr);

  ret = get_dev_value_vec(amd::smi::kDevPowerODVoltage, dv_ind, &val_vec);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  // This is a work-around to handle systems where kDevPowerODVoltage is not
  // fully supported yet.
  if (val_vec.size() < 2) {
    return RSMI_STATUS_NOT_YET_IMPLEMENTED;
  }

  uint32_t val_vec_size = val_vec.size();
  assert((val_vec_size - kOD_VDDC_CURVE_start_index) > 0);
  assert((val_vec_size - kOD_VDDC_CURVE_start_index)%2 == 0);
  *num_regions = std::min((val_vec_size - kOD_VDDC_CURVE_start_index) / 2,
                                                                *num_regions);

  for (uint32_t i=0; i < *num_regions; ++i) {
    get_vc_region(kOD_VDDC_CURVE_start_index + i*2, &val_vec, p + i);
  }

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
rsmi_dev_gpu_clk_freq_get(uint32_t dv_ind, rsmi_clk_type_t clk_type,
                                                        rsmi_frequencies_t *f) {
  TRY
  amd::smi::DevInfoTypes dev_type;

  CHK_SUPPORT_VAR(f, clk_type)

  switch (clk_type) {
    case RSMI_CLK_TYPE_SYS:
      dev_type = amd::smi::kDevGPUSClk;
      break;
    case RSMI_CLK_TYPE_MEM:
      dev_type = amd::smi::kDevGPUMClk;
      break;
    case RSMI_CLK_TYPE_DF:
      dev_type = amd::smi::kDevFClk;
      break;
    case RSMI_CLK_TYPE_DCEF:
      dev_type = amd::smi::kDevDCEFClk;
      break;
    case RSMI_CLK_TYPE_SOC:
      dev_type = amd::smi::kDevSOCClk;
      break;
    default:
      return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX

  return get_frequencies(dev_type, dv_ind, f);

  CATCH
}

rsmi_status_t
rsmi_dev_firmware_version_get(uint32_t dv_ind, rsmi_fw_block_t block,
                                                       uint64_t *fw_version) {
  rsmi_status_t ret;

  TRY
  CHK_SUPPORT_VAR(fw_version, block)

  std::string val_str;
  amd::smi::DevInfoTypes dev_type;

  switch (block) {
    case RSMI_FW_BLOCK_ASD:
      dev_type = amd::smi::kDevFwVersionAsd;
      break;
    case RSMI_FW_BLOCK_CE:
      dev_type = amd::smi::kDevFwVersionCe;
      break;
    case RSMI_FW_BLOCK_DMCU:
      dev_type = amd::smi::kDevFwVersionDmcu;
      break;
    case RSMI_FW_BLOCK_MC:
      dev_type = amd::smi::kDevFwVersionMc;
      break;
    case RSMI_FW_BLOCK_ME:
      dev_type = amd::smi::kDevFwVersionMe;
      break;
    case RSMI_FW_BLOCK_MEC:
      dev_type = amd::smi::kDevFwVersionMec;
      break;
    case RSMI_FW_BLOCK_MEC2:
      dev_type = amd::smi::kDevFwVersionMec2;
      break;
    case RSMI_FW_BLOCK_PFP:
      dev_type = amd::smi::kDevFwVersionPfp;
      break;
    case RSMI_FW_BLOCK_RLC:
      dev_type = amd::smi::kDevFwVersionRlc;
      break;
    case RSMI_FW_BLOCK_RLC_SRLC:
      dev_type = amd::smi::kDevFwVersionRlcSrlc;
      break;
    case RSMI_FW_BLOCK_RLC_SRLG:
      dev_type = amd::smi::kDevFwVersionRlcSrlg;
      break;
    case RSMI_FW_BLOCK_RLC_SRLS:
      dev_type = amd::smi::kDevFwVersionRlcSrls;
      break;
    case RSMI_FW_BLOCK_SDMA:
      dev_type = amd::smi::kDevFwVersionSdma;
      break;
    case RSMI_FW_BLOCK_SDMA2:
      dev_type = amd::smi::kDevFwVersionSdma2;
      break;
    case RSMI_FW_BLOCK_SMC:
      dev_type = amd::smi::kDevFwVersionSmc;
      break;
    case RSMI_FW_BLOCK_SOS:
      dev_type = amd::smi::kDevFwVersionSos;
      break;
    case RSMI_FW_BLOCK_TA_RAS:
      dev_type = amd::smi::kDevFwVersionTaRas;
      break;
    case RSMI_FW_BLOCK_TA_XGMI:
      dev_type = amd::smi::kDevFwVersionTaXgmi;
      break;
    case RSMI_FW_BLOCK_UVD:
      dev_type = amd::smi::kDevFwVersionUvd;
      break;
    case RSMI_FW_BLOCK_VCE:
      dev_type = amd::smi::kDevFwVersionVce;
      break;
    case RSMI_FW_BLOCK_VCN:
      dev_type = amd::smi::kDevFwVersionVcn;
      break;
  }
  ret = get_dev_value_int(dev_type, dv_ind, fw_version);
  if (ret != 0) {
    return errno_to_rsmi_status(ret);
  }

  return RSMI_STATUS_SUCCESS;
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
                              rsmi_clk_type_t clk_type, uint64_t freq_bitmask) {
  rsmi_status_t ret;
  rsmi_frequencies_t freqs;

  TRY
  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX

  ret = rsmi_dev_gpu_clk_freq_get(dv_ind, clk_type, &freqs);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  assert(freqs.num_supported <= RSMI_MAX_NUM_FREQUENCIES);

  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

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
  amd::smi::DevInfoTypes dev_type;

  switch (clk_type) {
    case RSMI_CLK_TYPE_SYS:
      dev_type = amd::smi::kDevGPUSClk;
      break;
    case RSMI_CLK_TYPE_MEM:
      dev_type = amd::smi::kDevGPUMClk;
      break;
    case RSMI_CLK_TYPE_DF:
      dev_type = amd::smi::kDevFClk;
      break;
    case RSMI_CLK_TYPE_SOC:
      dev_type = amd::smi::kDevSOCClk;
      break;
    case RSMI_CLK_TYPE_DCEF:
      dev_type = amd::smi::kDevDCEFClk;
      break;
    default:
      return RSMI_STATUS_INVALID_ARGS;
  }

  ret_i = dev->writeDevInfo(dev_type, freq_enable_str);
  return errno_to_rsmi_status(ret_i);

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

static std::string
get_id_name_str_from_line(uint64_t id, std::string ln,
                                                 std::istringstream *ln_str) {
  std::string token1;
  std::string ret_str;

  assert(ln_str != nullptr);

  *ln_str >> token1;
  if (std::stoul(token1, nullptr, 16) == id) {
    int64_t pos = ln_str->tellg();

    pos = ln.find_first_not_of("\t ", pos);
    ret_str = ln.substr(pos);
  }
  return ret_str;
}

static rsmi_status_t get_backup_name(uint16_t id, char *name, size_t len) {
  std::string name_str;

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
  std::string val_str;

  assert(name != nullptr);
  assert(len > 0);

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

  for (auto fl : pci_name_files) {
    std::ifstream id_file_strm(fl);

    while (std::getline(id_file_strm, ln)) {
      std::istringstream ln_str(ln);
      // parse line
      if (ln[0] == '#' || ln.size() == 0) {
        continue;
      }

      if (ln[0] == '\t') {
        if (found_device_vendor) {
          if (ln[1] == '\t') {
            // This is a subsystem line
            if (typ == NAME_STR_SUBSYS) {
              val_str = get_id_name_str_from_line(subsys_vend_id, ln, &ln_str);

              if (val_str.size() > 0) {
                // We've chopped the subsys_vend ID, now we need to get the
                // subsys description
                val_str = get_id_name_str_from_line(subsys_id, ln, &ln_str);

                if (val_str.size() > 0) {
                  break;
                } else {
                  val_str.clear();
                }
              }
            }
          } else if (typ == NAME_STR_DEVICE) {  // ln[1] != '\t'
            // This is a device line
            val_str = get_id_name_str_from_line(device_id, ln, &ln_str);

            if (val_str.size() > 0) {
              break;
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

        if (val_str.size() > 0) {
          if (typ == NAME_STR_VENDOR) {
            break;
          } else {
            val_str.clear();
            found_device_vendor = true;
          }
        }
      }
    }
    if (val_str.size() > 0) {
      break;
    }
  }

  if (val_str.size() == 0) {
    // We should have already returned if we were looking for
    // device or subdivce
    assert(typ == NAME_STR_VENDOR);
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

  *minor = dev->drm_render_minor();
  if (*minor)
    return RSMI_STATUS_SUCCESS;

  return RSMI_STATUS_INIT_ERROR;
}

rsmi_status_t
rsmi_dev_name_get(uint32_t dv_ind, char *name, size_t len) {
  rsmi_status_t ret;

  TRY
  CHK_SUPPORT_NAME_ONLY(name)

  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX

  ret = get_dev_name_from_id(dv_ind, name, len, NAME_STR_DEVICE);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_brand_get(uint32_t dv_ind, char *brand, uint32_t len) {
  CHK_SUPPORT_NAME_ONLY(brand)
  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }
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
    return errno_to_rsmi_status(ret);
  }
  if (vbios_value.length() == 16) {
    sku_value = vbios_value.substr(4, 6);
    // Find the brand name using sku_value
    it = brand_names.find(sku_value);
    if (it != brand_names.end()) {
      uint32_t ln = it->second.copy(brand, len);
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
}

rsmi_status_t
rsmi_dev_vram_vendor_get(uint32_t dv_ind, char *brand, uint32_t len) {
  TRY
  CHK_SUPPORT_NAME_ONLY(brand)

  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  std::string val_str;
  DEVICE_MUTEX
  int ret = dev->readDevInfo(amd::smi::kDevVramVendor, &val_str);

  if (ret != 0) {
    return errno_to_rsmi_status(ret);
  }

  uint32_t ln = val_str.copy(brand, len);

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
  CHK_SUPPORT_NAME_ONLY(name)

  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX
  ret = get_dev_name_from_id(dv_ind, name, len, NAME_STR_VENDOR);
  return ret;
  CATCH
}


rsmi_status_t
rsmi_dev_pci_bandwidth_get(uint32_t dv_ind, rsmi_pcie_bandwidth_t *b) {
  TRY
  CHK_SUPPORT_NAME_ONLY(b)

  DEVICE_MUTEX

  return get_frequencies(amd::smi::kDevPCIEClk, dv_ind,
                                        &b->transfer_rate, b->lanes);

  CATCH
}

rsmi_status_t
rsmi_dev_pci_bandwidth_set(uint32_t dv_ind, uint64_t bw_bitmask) {
  rsmi_status_t ret;
  rsmi_pcie_bandwidth_t bws;

  TRY
  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX
  ret = rsmi_dev_pci_bandwidth_get(dv_ind, &bws);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  assert(bws.transfer_rate.num_supported <= RSMI_MAX_NUM_FREQUENCIES);

  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance();

  // Above call to rsmi_dev_pci_bandwidth_get() should have emitted an error
  // if assert below is not true
  assert(dv_ind < smi.monitor_devices().size());

  std::string freq_enable_str =
         bitfield_to_freq_string(bw_bitmask, bws.transfer_rate.num_supported);

  std::shared_ptr<amd::smi::Device> dev = smi.monitor_devices()[dv_ind];
  assert(dev != nullptr);

  ret = rsmi_dev_perf_level_set(dv_ind, RSMI_DEV_PERF_LEVEL_MANUAL);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  uint32_t ret_i;
  ret_i = dev->writeDevInfo(amd::smi::kDevPCIEClk, freq_enable_str);

  return errno_to_rsmi_status(ret_i);

  CATCH
}

rsmi_status_t
rsmi_dev_pci_throughput_get(uint32_t dv_ind, uint64_t *sent,
                                   uint64_t *received, uint64_t *max_pkt_sz) {
  TRY
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

  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_temp_metric_get(uint32_t dv_ind, uint32_t sensor_type,
                       rsmi_temperature_metric_t metric, int64_t *temperature) {
  TRY

  rsmi_status_t ret;
  amd::smi::MonitorTypes mon_type;

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

  DEVICE_MUTEX

  GET_DEV_FROM_INDX

  assert(dev->monitor() != nullptr);
  std::shared_ptr<amd::smi::Monitor> m = dev->monitor();

  uint32_t err  = m->setSensorLabelMap();
  if (err) {
     return errno_to_rsmi_status(err);
  }

  uint32_t sensor_index =
         m->getSensorIndex(static_cast<rsmi_temperature_type_t>(sensor_type));

  CHK_API_SUPPORT_ONLY(temperature, metric, sensor_index)

  ret = get_dev_mon_value(mon_type, dv_ind, sensor_index, temperature);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_fan_speed_get(uint32_t dv_ind, uint32_t sensor_ind, int64_t *speed) {
  TRY

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

  ++sensor_ind;  // fan sysfs files have 1-based indices

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

  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX

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
  DEVICE_MUTEX
  CHK_SUPPORT_NAME_ONLY(odv)
  rsmi_status_t ret = get_od_clk_volt_info(dv_ind, odv);

  return ret;
  CATCH
}

rsmi_status_t rsmi_dev_od_volt_curve_regions_get(uint32_t dv_ind,
                     uint32_t *num_regions, rsmi_freq_volt_region_t *buffer) {
  TRY

  CHK_SUPPORT_NAME_ONLY((num_regions == nullptr || buffer == nullptr) ?
                                                        nullptr : num_regions)
  if (*num_regions == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX
  rsmi_status_t ret = get_od_clk_volt_curve_regions(dv_ind, num_regions,
                                                                      buffer);
  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_max_get(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *power) {
  TRY

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

  ++sensor_ind;  // power sysfs files have 1-based indices

  CHK_SUPPORT_SUBVAR_ONLY(power, sensor_ind)
  rsmi_status_t ret;

  DEVICE_MUTEX
  ret = get_dev_mon_value(amd::smi::kMonPowerAve, dv_ind, sensor_ind, power);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_power_cap_get(uint32_t dv_ind, uint32_t sensor_ind, uint64_t *cap) {
  TRY

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
  uint64_t min, max;

  REQUIRE_ROOT_ACCESS
  DEVICE_MUTEX

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
  REQUIRE_ROOT_ACCESS

  (void)dummy;
  DEVICE_MUTEX
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
      assert(!"Unexpected memory type");
      return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX
  ret = get_dev_value_int(mem_type_file, dv_ind, total);

  return ret;
  CATCH
}
rsmi_status_t
rsmi_dev_memory_usage_get(uint32_t dv_ind, rsmi_memory_type_t mem_type,
                                                              uint64_t *used) {
  TRY
  rsmi_status_t ret;
  amd::smi::DevInfoTypes mem_type_file;

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
      assert(!"Unexpected memory type");
      return RSMI_STATUS_INVALID_ARGS;
  }

  DEVICE_MUTEX
  ret = get_dev_value_int(mem_type_file, dv_ind, used);

  return ret;
  CATCH
}

rsmi_status_t
rsmi_dev_memory_busy_percent_get(uint32_t dv_ind, uint32_t *busy_percent) {
  TRY
  rsmi_status_t ret;

  if (busy_percent == nullptr) {
    return RSMI_STATUS_INVALID_ARGS;
  }
  CHK_SUPPORT_NAME_ONLY(busy_percent)

  uint64_t tmp_util = 0;

  DEVICE_MUTEX
  ret = get_dev_value_int(amd::smi::kDevMemBusyPercent, dv_ind, &tmp_util);

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
      *status_string = "Unable to acquire memory or other resource";
      break;

    case RSMI_STATUS_INTERNAL_EXCEPTION:
      *status_string = "An internal exception was caught";
      break;

    case RSMI_STATUS_INPUT_OUT_OF_BOUNDS:
      *status_string = "The provided input is out of allowable or safe range";
      break;

    case RSMI_STATUS_INIT_ERROR:
      *status_string = "An error occurred during initialization, during "
       "monitor discovery or when when initializing internal data structures";
      break;

    case RSMI_STATUS_NOT_YET_IMPLEMENTED:
      *status_string = "The called function has not been implemented in this "
        "system for this device type";
      break;

    case RSMI_STATUS_NOT_FOUND:
      *status_string = "An item required to complete the call was not found";
      break;

    case RSMI_STATUS_INSUFFICIENT_SIZE:
      *status_string = "Not enough resources were available to fully execute"
                             " the call";
      break;

    case RSMI_STATUS_UNKNOWN_ERROR:
      *status_string = "An unknown error prevented the call from completing"
                          " successfully";
      break;

    case RSMI_STATUS_INTERRUPT:
      *status_string = "An interrupt occurred while executing the function";
      break;

    default:
      *status_string = "An unknown error occurred";
      return RSMI_STATUS_UNKNOWN_ERROR;
  }
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_dev_busy_percent_get(uint32_t dv_ind, uint32_t *busy_percent) {
  TRY
  std::string val_str;

  CHK_SUPPORT_NAME_ONLY(busy_percent)

  DEVICE_MUTEX
  rsmi_status_t ret = get_dev_value_str(amd::smi::kDevUsage, dv_ind,
                                                                    &val_str);
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  errno = 0;
  *busy_percent = strtoul(val_str.c_str(), nullptr, 10);
  assert(errno == 0);

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_vbios_version_get(uint32_t dv_ind, char *vbios, uint32_t len) {
  TRY
  CHK_SUPPORT_NAME_ONLY(vbios)

  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  std::string val_str;

  DEVICE_MUTEX
  int ret = dev->readDevInfo(amd::smi::kDevVBiosVer, &val_str);

  if (ret != 0) {
    return errno_to_rsmi_status(ret);
  }

  uint32_t ln = val_str.copy(vbios, len);

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
      assert(!"Unexpected component type provided");
      return RSMI_STATUS_INVALID_ARGS;
  }

  err = amd::smi::ReadSysfsStr(ver_path, &val_str);

  if (err != 0) {
    struct utsname buf;
    err = uname(&buf);

    if (err != 0) {
      return errno_to_rsmi_status(err);
    }

    val_str = buf.release;
  }

  uint32_t ln = val_str.copy(ver_str, len);

  ver_str[std::min(len - 1, ln)] = '\0';

  if (len < (val_str.size() + 1)) {
    return RSMI_STATUS_INSUFFICIENT_SIZE;
  }
  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t rsmi_dev_serial_number_get(uint32_t dv_ind,
                                             char *serial_num, uint32_t len) {
  CHK_SUPPORT_NAME_ONLY(serial_num)
  if (len == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  TRY

  std::string val_str;
  rsmi_status_t ret = get_dev_value_str(amd::smi::kDevSerialNumber,
                                                            dv_ind, &val_str);

  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  uint32_t ln = val_str.copy(serial_num, len);

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
  CHK_SUPPORT_NAME_ONLY(counter)

  DEVICE_MUTEX
  rsmi_status_t ret;

  ret = get_dev_value_int(amd::smi::kDevPCIEReplayCount, dv_ind, counter);
  return ret;

  CATCH
}

rsmi_status_t
rsmi_dev_unique_id_get(uint32_t dv_ind, uint64_t *unique_id) {
  TRY
  DEVICE_MUTEX
  rsmi_status_t ret;

  CHK_SUPPORT_NAME_ONLY(unique_id)

  ret = get_dev_value_int(amd::smi::kDevUniqueId, dv_ind, unique_id);
  return ret;

  CATCH
}
rsmi_status_t
rsmi_dev_counter_create(uint32_t dv_ind, rsmi_event_type_t type,
                                           rsmi_event_handle_t *evnt_handle) {
  TRY
  DEVICE_MUTEX
  REQUIRE_ROOT_ACCESS

  // Note we don't need to pass in the variant to CHK_SUPPORT_VAR because
  // the success of this call doesn't depend on a sysfs file existing.
  CHK_SUPPORT_NAME_ONLY(evnt_handle)

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

  if (evnt_handle == 0) {
    return RSMI_STATUS_INVALID_ARGS;
  }

  amd::smi::evt::Event *evt =
                        reinterpret_cast<amd::smi::evt::Event *>(evnt_handle);
  uint32_t dv_ind = evt->dev_ind();
  DEVICE_MUTEX
  REQUIRE_ROOT_ACCESS

  delete evt;
  return RSMI_STATUS_SUCCESS;
  CATCH
}

rsmi_status_t
rsmi_counter_control(rsmi_event_handle_t evt_handle,
                                 rsmi_counter_command_t cmd, void *cmd_args) {
  TRY

  amd::smi::evt::Event *evt =
                         reinterpret_cast<amd::smi::evt::Event *>(evt_handle);
  amd::smi::pthread_wrap _pw(*get_mutex(evt->dev_ind()));
  amd::smi::ScopedPthread _lock(_pw);

  REQUIRE_ROOT_ACCESS

  uint32_t ret;

  // This is for future command args. This would work in conjunction with a
  // new function to set perf attributes.
  (void) cmd_args;

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
      assert(!"Unexpected perf counter command");
  }
  return errno_to_rsmi_status(ret);

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

  return errno_to_rsmi_status(ret);
  CATCH
}

rsmi_status_t
rsmi_counter_available_counters_get(uint32_t dv_ind,
                                rsmi_event_group_t grp, uint32_t *available) {
  rsmi_status_t ret;

  TRY
  CHK_SUPPORT_VAR(available, grp)
  DEVICE_MUTEX
  uint64_t val;

  switch (grp) {
    case RSMI_EVNT_GRP_XGMI:
      ret = get_dev_value_int(amd::smi::kDevDFCountersAvailable, dv_ind, &val);
      assert(val < UINT32_MAX);
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
  DEVICE_MUTEX
  GET_DEV_FROM_INDX

  amd::smi::evt::dev_evt_grp_set_t *grp = dev->supported_event_groups();

  if (grp->find(group) == grp->end()) {
    return RSMI_STATUS_NOT_SUPPORTED;
  } else {
    return RSMI_STATUS_SUCCESS;
  }
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
    return errno_to_rsmi_status(err);
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
rsmi_dev_memory_reserved_pages_get(uint32_t dv_ind, uint32_t *num_pages,
                                          rsmi_retired_page_record_t *records) {
  TRY

  rsmi_status_t ret;
  CHK_SUPPORT_NAME_ONLY(num_pages)

  std::vector<std::string> val_vec;

  ret = get_dev_value_vec(amd::smi::kDevMemPageBad, dv_ind, &val_vec);

  if (ret == RSMI_STATUS_FILE_ERROR) {
    return RSMI_STATUS_NOT_SUPPORTED;
  }
  if (ret != RSMI_STATUS_SUCCESS) {
    return ret;
  }

  if (records == nullptr || *num_pages > val_vec.size()) {
    *num_pages = val_vec.size();
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
        assert(!"Unexpected retired memory page status code read");
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

  int err = amd::smi::GetProcessInfoForPID(pid, proc);

  if (err) {
    return errno_to_rsmi_status(err);
  }

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_xgmi_error_status(uint32_t dv_ind, rsmi_xgmi_status_t *status) {
  TRY
  DEVICE_MUTEX

  CHK_SUPPORT_NAME_ONLY(status)

  rsmi_status_t ret;
  uint64_t status_code;

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
      assert(!"Unexpected XGMI error status read");
      return RSMI_STATUS_UNKNOWN_ERROR;
  }
  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_xgmi_error_reset(uint32_t dv_ind) {
  TRY
  DEVICE_MUTEX

  rsmi_status_t ret;
  uint64_t status_code;

  // Reading xgmi_error resets it
  ret = get_dev_value_int(amd::smi::kDevXGMIError, dv_ind, &status_code);
  return ret;

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
    return RSMI_STATUS_NO_DATA;
  } else {
    SupportedFuncMapIt *supp_func_iter = new SupportedFuncMapIt;

    if (supp_func_iter == nullptr) {
      return RSMI_STATUS_OUT_OF_RESOURCES;
    }
    *supp_func_iter = dev->supported_funcs()->begin();

    (*handle)->func_id_iter = reinterpret_cast<uintptr_t>(supp_func_iter);
    (*handle)->container_ptr =
                          reinterpret_cast<uintptr_t>(dev->supported_funcs());
  }

  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_supported_variant_iterator_open(
                                 rsmi_func_id_iter_handle_t parent_iter,
                                       rsmi_func_id_iter_handle_t *var_iter) {
  TRY

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
      assert(!"Unexpected iterator type");
      return RSMI_STATUS_INVALID_ARGS;
  }
  return RSMI_STATUS_SUCCESS;

  CATCH
}

rsmi_status_t
rsmi_dev_supported_func_iterator_close(rsmi_func_id_iter_handle_t *handle) {
  TRY

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
    SubVariant *subvar_iter =
                      reinterpret_cast<SubVariant *>((*handle)->func_id_iter);
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
      value->id = *(*sub_var_itr);
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
        handle->func_id_iter = 0;
        return RSMI_STATUS_NO_DATA;
      }
      break;

    case VARIANT_ITER:
      var_iter = reinterpret_cast<VariantMapIt *>(handle->func_id_iter);
      (*var_iter)++;
      if (*var_iter ==
               reinterpret_cast<VariantMap *>(handle->container_ptr)->end()) {
        handle->func_id_iter = 0;
        return RSMI_STATUS_NO_DATA;
      }
      break;

    case SUBVARIANT_ITER:
      sub_var_iter = reinterpret_cast<SubVariantIt *>(handle->func_id_iter);
      (*sub_var_iter)++;
      if (*sub_var_iter ==
               reinterpret_cast<SubVariant *>(handle->container_ptr)->end()) {
        handle->func_id_iter = 0;
        return RSMI_STATUS_NO_DATA;
      }
      break;

    default:
      return RSMI_STATUS_INVALID_ARGS;
  }

  return RSMI_STATUS_SUCCESS;

  CATCH
}
