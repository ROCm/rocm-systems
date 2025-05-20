/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017-2025, Advanced Micro Devices, Inc.
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
#include <stdint.h>
#include <unistd.h>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <map>
#include <vector>
#include <type_traits>
#include <cstring>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_utils.h"

#define PRINT_RSMI_ERR(RET) { \
  if (RET != RSMI_STATUS_SUCCESS) { \
    std::cout << "[ERROR] RSMI call returned " << (RET) \
      << " at line " << __LINE__ << "\n"; \
      std::cout << amd::smi::getRSMIStatusString(RET) << "\n"; \
  } \
}

#define CHK_RSMI_RET(RET) { \
  PRINT_RSMI_ERR(RET) \
  if (RET != RSMI_STATUS_SUCCESS) { \
    return (RET); \
  } \
}

#define CHK_AND_PRINT_RSMI_ERR_RET(RET) { \
  PRINT_RSMI_ERR(RET) \
  CHK_RSMI_RET(RET) \
}

#define CHK_RSMI_RET_I(RET) { \
  PRINT_RSMI_ERR(RET) \
  if (RET != RSMI_STATUS_SUCCESS) { \
    return static_cast<int>(RET); \
  } \
}

#define CHK_FILE_PERMISSIONS(RET) { \
    if ((RET) == RSMI_STATUS_PERMISSION) { \
      if (isFileWritable(RET)) { \
        CHK_RSMI_RET(RET) \
      } \
    } else { \
      CHK_RSMI_RET(RET) \
    } \
}

#define CHK_FILE_PERMISSIONS_AND_NOT_SUPPORTED_OR_UNIMPLEMENTED(RET) { \
    if ((RET) == RSMI_STATUS_PERMISSION) { \
      if (isFileWritable(RET)) { \
        CHK_RSMI_RET(RET) \
      } \
    } else if ((RET) == RSMI_STATUS_NOT_SUPPORTED) { \
      std::cout << "Not Supported." \
      << "\n"; \
    } else if ((RET) == RSMI_STATUS_NOT_YET_IMPLEMENTED) { \
      std::cout << "Not Yet Implemented." \
      << "\n"; \
    } else { \
      CHK_RSMI_RET(RET) \
    } \
}

#define CHK_RSMI_NOT_SUPPORTED_RET(RET) { \
    if ((RET) == RSMI_STATUS_NOT_SUPPORTED) { \
      std::cout << "Not Supported." \
      << "\n"; \
    } else { \
      CHK_RSMI_RET(RET) \
    } \
}

#define CHK_RSMI_NOT_SUPPORTED_OR_UNEXPECTED_DATA_RET(RET) { \
    if ((RET) == RSMI_STATUS_NOT_SUPPORTED) { \
      std::cout << "Not Supported." \
      << "\n"; \
    } else if ((RET) == RSMI_STATUS_UNEXPECTED_DATA) { \
      std::cout << "[ERROR] RSMI_STATUS_UNEXPECTED_DATA retrieved." \
      << "\n"; \
    } else { \
      CHK_RSMI_RET(RET) \
    } \
}

#define CHK_RSMI_NOT_SUPPORTED_OR_SETTING_UNAVAILABLE_RET(RET) {\
    if ((RET) == RSMI_STATUS_NOT_SUPPORTED) { \
      std::cout << "Not Supported."\
      << "\n"; \
    } else if ((RET) == RSMI_STATUS_SETTING_UNAVAILABLE) { \
      std::cout << "[WARN] RSMI_STATUS_SETTING_UNAVAILABLE retrieved." \
      << "\n"; \
    } else { \
      CHK_RSMI_RET(RET) \
    } \
}

#define CHK_NOT_SUPPORTED_OR_UNEXPECTED_DATA_OR_INSUFFICIENT_SIZE_RET(RET) { \
    if ((RET) == RSMI_STATUS_NOT_SUPPORTED) { \
      std::cout << "Not Supported." \
      << "\n"; \
    } else if ((RET) == RSMI_STATUS_UNEXPECTED_DATA) { \
      std::cout << "[WARN] RSMI_STATUS_UNEXPECTED_DATA retrieved." \
      << "\n"; \
    } else if ((RET) == RSMI_STATUS_INSUFFICIENT_SIZE) { \
      std::cout << "[WARN] RSMI_STATUS_INSUFFICIENT_SIZE retrieved." \
      << "\n"; \
    } else { \
      CHK_RSMI_RET(RET) \
    } \
}

void print_function_header_with_rsmi_ret(
  rsmi_status_t myReturn, std::string header = "") {
  std::cout << "\t** ";
  if (!header.empty()) {
    std::cout << header << ": ";
  }
  std::cout << amd::smi::getRSMIStatusString(myReturn, false) << "\n";
}

static void print_test_header(const char *str, uint32_t dv_ind) {
  std::cout << "******************************************" << "\n";
  std::cout << "*** " << str << "\n";
  std::cout << "******************************************" << "\n";
  std::cout << "Device index: " << dv_ind << "\n";
}

static void print_mini_header(const char *str) {
  std::cout << "\n>> " << str << " <<" << "\n";
}

static const char *
power_profile_string(rsmi_power_profile_preset_masks_t profile) {
  switch (profile) {
    case RSMI_PWR_PROF_PRST_CUSTOM_MASK:
      return "CUSTOM";
    case RSMI_PWR_PROF_PRST_VIDEO_MASK:
      return "VIDEO";
    case RSMI_PWR_PROF_PRST_POWER_SAVING_MASK:
      return "POWER SAVING";
    case RSMI_PWR_PROF_PRST_COMPUTE_MASK:
      return "COMPUTE";
    case RSMI_PWR_PROF_PRST_VR_MASK:
      return "VR";
    case RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK:
      return "3D FULL SCREEN";
    default:
      return "UNKNOWN";
  }
}

static const std::string
compute_partition_string(rsmi_compute_partition_type_t partition) {
  switch (partition) {
    case RSMI_COMPUTE_PARTITION_CPX:
      return "CPX";
    case RSMI_COMPUTE_PARTITION_SPX:
      return "SPX";
    case RSMI_COMPUTE_PARTITION_DPX:
      return "DPX";
    case RSMI_COMPUTE_PARTITION_TPX:
      return "TPX";
    case RSMI_COMPUTE_PARTITION_QPX:
      return "QPX";
    default:
      return "UNKNOWN";
  }
}

static std::map<std::string, rsmi_compute_partition_type_t>
mapStringToRSMIComputePartitionTypes {
  {"CPX", RSMI_COMPUTE_PARTITION_CPX},
  {"SPX", RSMI_COMPUTE_PARTITION_SPX},
  {"DPX", RSMI_COMPUTE_PARTITION_DPX},
  {"TPX", RSMI_COMPUTE_PARTITION_TPX},
  {"QPX", RSMI_COMPUTE_PARTITION_QPX}
};

static const std::string
memory_partition_string(rsmi_memory_partition_type_t partition) {
  switch (partition) {
    case RSMI_MEMORY_PARTITION_NPS1:
      return "NPS1";
    case RSMI_MEMORY_PARTITION_NPS2:
      return "NPS2";
    case RSMI_MEMORY_PARTITION_NPS4:
      return "NPS4";
    case RSMI_MEMORY_PARTITION_NPS8:
      return "NPS8";
    default:
      return "UNKNOWN";
  }
}

static std::map<std::string, rsmi_memory_partition_type_t>
mapStringToRSMIMemoryPartitionTypes {
  {"NPS1", RSMI_MEMORY_PARTITION_NPS1},
  {"NPS2", RSMI_MEMORY_PARTITION_NPS2},
  {"NPS4", RSMI_MEMORY_PARTITION_NPS4},
  {"NPS8", RSMI_MEMORY_PARTITION_NPS8}
};

static const char *
perf_level_string(rsmi_dev_perf_level_t perf_lvl) {
  switch (perf_lvl) {
    case RSMI_DEV_PERF_LEVEL_AUTO:
      return "AUTO";
    case RSMI_DEV_PERF_LEVEL_LOW:
      return "LOW";
    case RSMI_DEV_PERF_LEVEL_HIGH:
      return "HIGH";
    case RSMI_DEV_PERF_LEVEL_MANUAL:
      return "MANUAL";
    default:
      return "UNKNOWN";
  }
}

static const std::string
clock_type_string(rsmi_clk_type_t clk) {
  switch (clk) {
    case RSMI_CLK_TYPE_SYS:
      return "RSMI_CLK_TYPE_SYS";
    case RSMI_CLK_TYPE_DF:
      return "RSMI_CLK_TYPE_DF";
    case RSMI_CLK_TYPE_DCEF:
      return "RSMI_CLK_TYPE_DCEF";
    case RSMI_CLK_TYPE_SOC:
      return "RSMI_CLK_TYPE_SOC";
    case RSMI_CLK_TYPE_MEM:
      return "RSMI_CLK_TYPE_MEM";
    case RSMI_CLK_TYPE_PCIE:
      return "RSMI_CLK_TYPE_PCIE";
    default:
      return "RSMI_CLK_INVALID";
  }
}

static bool isFileWritable(rsmi_status_t response) {
  // Clock files may not be writable, causing sets to
  // return RSMI_STATUS_PERMISSION. If running as sudo,
  // this means file is not writable.
  // isFileWritable(ret) - intends to capture this
  // response situation.
  bool fileWritable = true;
  if (amd::smi::is_sudo_user() && (response == RSMI_STATUS_PERMISSION)) {
      std::cout << "[WARN] User is running with sudo "
                << "permissions, file is not writable." << "\n";
      fileWritable = false;
  } else {
      CHK_AND_PRINT_RSMI_ERR_RET(response)
  }
  return fileWritable;
}

static rsmi_status_t test_power_profile(uint32_t dv_ind) {
  rsmi_status_t ret;
  rsmi_power_profile_status_t status;

  print_test_header("Power Profile", dv_ind);

  std::cout << "The available power profiles are: ";
  ret = rsmi_dev_power_profile_presets_get(dv_ind, 0, &status);
  CHK_RSMI_NOT_SUPPORTED_RET(ret)
  if (ret != RSMI_STATUS_SUCCESS) {
    std::cout << "***Skipping Power Profile test." << "\n";
    return RSMI_STATUS_SUCCESS;
  }
  CHK_RSMI_RET(ret)

  std::cout << "The available power profiles are:" << "\n";
  uint64_t tmp = 1;

  while (tmp <= RSMI_PWR_PROF_PRST_LAST) {
    if ((tmp & status.available_profiles) == tmp) {
      std::cout << "\t" <<
      power_profile_string((rsmi_power_profile_preset_masks_t)tmp) << "\n";
    }
    tmp = tmp << 1;
  }
  std::cout << "The current power profile is: " <<
                            power_profile_string(status.current) << "\n";

  // Try setting the profile to a different power profile
  rsmi_bit_field_t diff_profiles;
  rsmi_power_profile_preset_masks_t new_prof;
  diff_profiles = status.available_profiles & (~status.current);

  if (diff_profiles & RSMI_PWR_PROF_PRST_COMPUTE_MASK) {
    new_prof = RSMI_PWR_PROF_PRST_COMPUTE_MASK;
  } else if (diff_profiles & RSMI_PWR_PROF_PRST_VIDEO_MASK) {
    new_prof = RSMI_PWR_PROF_PRST_VIDEO_MASK;
  } else if (diff_profiles & RSMI_PWR_PROF_PRST_VR_MASK) {
    new_prof = RSMI_PWR_PROF_PRST_VR_MASK;
  } else if (diff_profiles & RSMI_PWR_PROF_PRST_POWER_SAVING_MASK) {
    new_prof = RSMI_PWR_PROF_PRST_POWER_SAVING_MASK;
  } else if (diff_profiles & RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK) {
    new_prof = RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK;
  } else {
    std::cout << "No other non-custom power profiles to set to" << "\n";
    return ret;
  }

  std::cout << "Setting power profile to " << power_profile_string(new_prof)
                                                        << "..." << "\n";
  ret = rsmi_dev_power_profile_set(dv_ind, 0, new_prof);
  CHK_RSMI_RET(ret)
  std::cout << "Done." << "\n";
  rsmi_dev_perf_level_t pfl;
  ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
  CHK_RSMI_RET(ret)
  std::cout << "Performance Level is now " <<
                                          perf_level_string(pfl) << "\n";

  ret = rsmi_dev_power_profile_presets_get(dv_ind, 0, &status);
  CHK_RSMI_RET(ret)
  std::cout << "The current power profile is: " <<
                            power_profile_string(status.current) << "\n";
  std::cout << "Resetting perf level to auto..." << "\n";

  ret = rsmi_dev_perf_level_set_v1(dv_ind, RSMI_DEV_PERF_LEVEL_AUTO);
  CHK_RSMI_RET(ret)
  std::cout << "Done." << "\n";

  ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
  CHK_RSMI_RET(ret)
  std::cout << "Performance Level is now " <<
                                          perf_level_string(pfl) << "\n";

  ret = rsmi_dev_power_profile_presets_get(dv_ind, 0, &status);
  CHK_RSMI_RET(ret)
  std::cout << "The current power profile is: " <<
                            power_profile_string(status.current) << "\n";

  return ret;
}

static rsmi_status_t test_power_cap(uint32_t dv_ind) {
  rsmi_status_t ret;
  uint64_t orig, min, max, new_cap;

  print_test_header("Power Control", dv_ind);

  ret = rsmi_dev_power_cap_range_get(dv_ind, 0, &max, &min);
  CHK_RSMI_RET(ret)

  ret = rsmi_dev_power_cap_get(dv_ind, 0, &orig);
  CHK_RSMI_RET(ret)

  std::cout << "Original Power Cap: " << orig << " uW" << "\n";
  std::cout << "Power Cap Range: " << max << " uW to " << min <<
                                                           " uW" << "\n";
  new_cap = (max + min)/2;

  std::cout << "Setting new cap to " << new_cap << "..." << "\n";

  ret = rsmi_dev_power_cap_set(dv_ind, 0, new_cap);
  CHK_RSMI_RET(ret)

  ret = rsmi_dev_power_cap_get(dv_ind, 0, &new_cap);
  CHK_RSMI_RET(ret)

  std::cout << "New Power Cap: " << new_cap << " uW" << "\n";
  std::cout << "Resetting cap to " << orig << "..." << "\n";

  ret = rsmi_dev_power_cap_set(dv_ind, 0, orig);
  CHK_RSMI_RET(ret)

  ret = rsmi_dev_power_cap_get(dv_ind, 0, &new_cap);
  CHK_RSMI_RET(ret)
  std::cout << "Current Power Cap: " << new_cap << " uW" << "\n";

  return ret;
}

static rsmi_status_t test_set_overdrive(uint32_t dv_ind) {
  rsmi_status_t ret;
  uint32_t val;

  print_test_header("Overdrive Control", dv_ind);
  std::cout << "Set Overdrive level to 0%..." << "\n";
  ret = rsmi_dev_overdrive_level_set_v1(dv_ind, 0);
  CHK_RSMI_RET(ret)
  std::cout << "Set Overdrive level to 10%..." << "\n";
  ret = rsmi_dev_overdrive_level_set_v1(dv_ind, 10);
  CHK_RSMI_RET(ret)
  ret = rsmi_dev_overdrive_level_get(dv_ind, &val);
  CHK_RSMI_RET(ret)
  std::cout << "\t**New OverDrive Level:" << std::dec << val << "\n";
  std::cout << "Reset Overdrive level to 0%..." << "\n";
  ret = rsmi_dev_overdrive_level_set_v1(dv_ind, 0);
  CHK_RSMI_RET(ret)
  ret = rsmi_dev_overdrive_level_get(dv_ind, &val);
  CHK_RSMI_RET(ret)
  std::cout << "\t**New OverDrive Level:" << std::dec << val << "\n";

  return ret;
}

static rsmi_status_t test_set_fan_speed(uint32_t dv_ind) {
  rsmi_status_t ret;
  int64_t orig_speed;
  double new_speed;
  int64_t cur_spd;

  print_test_header("Fan Speed Control", dv_ind);

  std::cout << "Original fan speed: ";
  ret = rsmi_dev_fan_speed_get(dv_ind, 0, &orig_speed);
  if (ret == RSMI_STATUS_SUCCESS) {
    std::cout << orig_speed << "\n";
  } else {
    CHK_RSMI_NOT_SUPPORTED_RET(ret)
    std::cout << "***Skipping Fan Speed Control test." << "\n";
    return RSMI_STATUS_SUCCESS;
  }

  if (orig_speed == 0) {
    std::cout << "***System fan speed value is 0. Skip fan test." << "\n";
    return RSMI_STATUS_SUCCESS;
  }

  new_speed = 1.1 * static_cast<double>(orig_speed);

  std::cout << "Setting fan speed to " << new_speed << "\n";

  ret = rsmi_dev_fan_speed_set(dv_ind, 0, static_cast<uint64_t>(new_speed));
  CHK_RSMI_RET(ret)

  sleep(4);

  ret = rsmi_dev_fan_speed_get(dv_ind, 0, &cur_spd);
  CHK_RSMI_RET(ret)

  std::cout << "New fan speed: " << cur_spd << "\n";

  assert(
      (cur_spd > static_cast<int64_t>(0.95 * static_cast<double>(new_speed)) &&
       cur_spd < static_cast<int64_t>(1.1 * static_cast<double>(new_speed))) ||
       (cur_spd >
       static_cast<int64_t>(0.95 * static_cast<double>(RSMI_MAX_FAN_SPEED))));

  std::cout << "Resetting fan control to auto..." << "\n";

  ret = rsmi_dev_fan_reset(dv_ind, 0);
  CHK_RSMI_RET(ret)

  sleep(3);

  ret = rsmi_dev_fan_speed_get(dv_ind, 0, &cur_spd);
  CHK_RSMI_RET(ret)

  std::cout << "End fan speed: " << cur_spd << "\n";

  return ret;
}

static rsmi_status_t test_set_perf_level(uint32_t dv_ind) {
  rsmi_status_t ret;

  rsmi_dev_perf_level_t pfl, orig_pfl;

  print_test_header("Performance Level Control", dv_ind);

  ret = rsmi_dev_perf_level_get(dv_ind, &orig_pfl);
  CHK_RSMI_RET(ret)
  std::cout << "\t**Original Perf Level:" << perf_level_string(orig_pfl) <<
                                                                    "\n";

  pfl =
     (rsmi_dev_perf_level_t)((orig_pfl + 1) % (RSMI_DEV_PERF_LEVEL_LAST + 1));

  std::cout << "Set Performance Level to " << (uint32_t)pfl << " ..." <<
                                                                   "\n";
  ret = rsmi_dev_perf_level_set_v1(dv_ind, pfl);
  if (ret != RSMI_STATUS_SUCCESS) {
    CHK_RSMI_NOT_SUPPORTED_RET(ret)
    std::cout << "***Skipping Performance Level Control test." << "\n";
    return RSMI_STATUS_SUCCESS;
  }
  CHK_RSMI_RET(ret)
  ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
  CHK_RSMI_RET(ret)
  std::cout << "\t**New Perf Level:" << perf_level_string(pfl) << "\n";
  std::cout << "Reset Perf level to " << orig_pfl << " ..." << "\n";
  ret = rsmi_dev_perf_level_set_v1(dv_ind, orig_pfl);
  CHK_RSMI_RET(ret)
  ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
  CHK_RSMI_RET(ret)
  std::cout << "\t**New Perf Level:" << perf_level_string(pfl) << "\n";
  return ret;
}

static rsmi_status_t test_set_freq(uint32_t dv_ind) {
  rsmi_status_t ret;
  rsmi_frequencies_t f;
  uint32_t freq_bitmask;
  rsmi_clk_type rsmi_clk;

  // Clock files may not be writable, causing sets to
  // return RSMI_STATUS_PERMISSION even if running with
  // sudo. See isFileWritable() for more info.

  print_test_header("Clock Frequency Control", dv_ind);
  for (uint32_t clk = (uint32_t)RSMI_CLK_TYPE_FIRST;
                                           clk <= RSMI_CLK_TYPE_LAST; ++clk) {
    std::string miniHeader = "Testing clock" + std::to_string(clk);
    print_mini_header(miniHeader.c_str());
    rsmi_clk = (rsmi_clk_type)clk;

    ret = rsmi_dev_gpu_clk_freq_get(dv_ind, rsmi_clk, &f);
    CHK_FILE_PERMISSIONS_AND_NOT_SUPPORTED_OR_UNIMPLEMENTED(ret)

    std::cout << "Initial frequency for clock" << rsmi_clk << " is " <<
                                                      f.current << "\n";

    // Set clocks to something other than the usual default of the lowest
    // frequency.
    freq_bitmask = 0b01100;  // Try the 3rd and 4th clocks

    std::string freq_bm_str =
               std::bitset<RSMI_MAX_NUM_FREQUENCIES>(freq_bitmask).to_string();

    freq_bm_str.erase(0, std::min(freq_bm_str.find_first_not_of('0'),
                                                       freq_bm_str.size()-1));

    std::cout << "Setting frequency mask for clock " << rsmi_clk <<
        " to 0b" << freq_bm_str << " ..." << "\n";

    ret = rsmi_dev_gpu_clk_freq_set(dv_ind, rsmi_clk, freq_bitmask);
    CHK_FILE_PERMISSIONS_AND_NOT_SUPPORTED_OR_UNIMPLEMENTED(ret)

    ret = rsmi_dev_gpu_clk_freq_get(dv_ind, rsmi_clk, &f);
    CHK_FILE_PERMISSIONS_AND_NOT_SUPPORTED_OR_UNIMPLEMENTED(ret)

    std::cout << "Frequency is now index " << f.current << "\n";
    std::cout << "Resetting mask to all frequencies." << "\n";
    ret = rsmi_dev_gpu_clk_freq_set(dv_ind, rsmi_clk, 0xFFFFFFFF);
    CHK_FILE_PERMISSIONS_AND_NOT_SUPPORTED_OR_UNIMPLEMENTED(ret)

    ret = rsmi_dev_perf_level_set_v1(dv_ind, RSMI_DEV_PERF_LEVEL_AUTO);
    CHK_FILE_PERMISSIONS(ret)
  }
  std::cout << "\n";
  return RSMI_STATUS_SUCCESS;
}

static void print_frequencies(rsmi_frequencies_t *f) {
  bool hasDeepSleep = false;
  if (f == nullptr) {
    std::cout << "Freq was nullptr\n";
    return;
  }
  for (uint32_t j = 0; j < f->num_supported; ++j) {
    if (f->has_deep_sleep && j == 0) {
      std::cout << "\t**  S: " << std::to_string(f->frequency[j]);
      hasDeepSleep = true;
    } else {
      std::cout << "\t**  " << (hasDeepSleep ? j-1 : j)
                << ": " << std::to_string(f->frequency[j]);
    }
    if (j == f->current) {
      std::cout << " *";
    }
    std::cout << "\n";
  }
}

static rsmi_status_t test_set_compute_partitioning(uint32_t dv_ind) {
  rsmi_status_t ret;
  const uint32_t kLength = 10;
  char originalComputePartition[kLength];
  originalComputePartition[0] = '\0';
  print_test_header("Compute Partitioning Control", dv_ind);

  ret = rsmi_dev_compute_partition_get(dv_ind, originalComputePartition,
                                       kLength);
  CHK_RSMI_NOT_SUPPORTED_OR_UNEXPECTED_DATA_RET(ret)
  if (ret == RSMI_STATUS_NOT_SUPPORTED) {
    return RSMI_STATUS_SUCCESS;
  }

  std::cout << "Original Compute Partition: "
            << (((originalComputePartition == nullptr)
                || ((originalComputePartition != nullptr)
                && (originalComputePartition[0] == '\0')))
                ? "UNKNOWN" : originalComputePartition)
            << "\n" << "\n";

  for (int newComputePartition = RSMI_COMPUTE_PARTITION_CPX;
       newComputePartition <= RSMI_COMPUTE_PARTITION_QPX;
       newComputePartition++) {
    rsmi_compute_partition_type_t newPartition
      = static_cast<rsmi_compute_partition_type_t>(newComputePartition);
    std::cout << "Attempting to set compute partition to "
              << compute_partition_string(newPartition) << "..."
              << "\n";
    ret = rsmi_dev_compute_partition_set(dv_ind, newPartition);
    CHK_RSMI_NOT_SUPPORTED_OR_SETTING_UNAVAILABLE_RET(ret)
    std::cout << "Done setting compute partition to "
              << compute_partition_string(newPartition) << "." << "\n";
    std::cout << "\n" << "\n";
  }

  std::string myComputePartition = originalComputePartition;
  if (myComputePartition.empty() == false) {
    std::cout << "Resetting back to original compute partition to "
              << originalComputePartition << "... " << "\n";
    rsmi_compute_partition_type origComputePartitionType
      = mapStringToRSMIComputePartitionTypes[originalComputePartition];
    ret = rsmi_dev_compute_partition_set(dv_ind, origComputePartitionType);
    CHK_RSMI_NOT_SUPPORTED_OR_SETTING_UNAVAILABLE_RET(ret)
    std::cout << "Done" << "\n";
  }
  return RSMI_STATUS_SUCCESS;
}

static rsmi_status_t test_set_memory_partition(uint32_t dv_ind) {
  rsmi_status_t ret;
  const uint32_t kLength = 10;
  char originalMemoryPartition[kLength];
  originalMemoryPartition[0] = '\0';
  print_test_header("Memory Partition Control", dv_ind);

  ret = rsmi_dev_memory_partition_get(dv_ind, originalMemoryPartition, kLength);
  CHK_RSMI_NOT_SUPPORTED_OR_UNEXPECTED_DATA_RET(ret)
  if (ret == RSMI_STATUS_NOT_SUPPORTED) {
    return RSMI_STATUS_SUCCESS;
  }

  std::cout << "Original Memory Partition: "
            << (((originalMemoryPartition == nullptr)
                || ((originalMemoryPartition != nullptr)
                && (originalMemoryPartition[0] == '\0')))
                ? "UNKNOWN" : originalMemoryPartition)
            << "\n\n";

  for (int newMemPartition = RSMI_MEMORY_PARTITION_NPS1;
       newMemPartition <= RSMI_MEMORY_PARTITION_NPS8;
       newMemPartition++) {
    rsmi_memory_partition_type_t newMemoryPartition
      = static_cast<rsmi_memory_partition_type_t>(newMemPartition);
    std::cout << "Attempting to set memory partition to "
              << memory_partition_string(newMemoryPartition) << "..."
              << "\n";
    ret = rsmi_dev_memory_partition_set(dv_ind, newMemoryPartition);
    CHK_RSMI_NOT_SUPPORTED_RET(ret)
    if (ret == RSMI_STATUS_NOT_SUPPORTED) {
      // do not continue attempting to set, device does not support setting
      return RSMI_STATUS_SUCCESS;
    }
    std::cout << "Done setting memory partition to "
              << memory_partition_string(newMemoryPartition)
              << "." << "\n\n\n";
  }

  std::string myMemPart = originalMemoryPartition;
  if (myMemPart.empty() == false) {
    std::cout << "Resetting memory partition to " << originalMemoryPartition
              << "...\n";
    rsmi_memory_partition_type_t origMemoryPartitionType
      = mapStringToRSMIMemoryPartitionTypes[originalMemoryPartition];
    ret = rsmi_dev_memory_partition_set(dv_ind, origMemoryPartitionType);
    CHK_RSMI_NOT_SUPPORTED_RET(ret)
    std::cout << "Done\n";
  }
  return RSMI_STATUS_SUCCESS;
}

template<typename T> constexpr float convert_mw_to_w(T mw) {
    return static_cast<float>(mw / 1000.0);
}


template <typename T>
std::string print_unsigned_int(T value) {
  std::stringstream ss;
  ss << static_cast<uint64_t>(value | 0);

  return ss.str();
}


int main() {
  rsmi_status_t ret;

  ret = rsmi_init(0);
  CHK_RSMI_RET_I(ret)

  std::vector<std::string> val_vec;
  uint64_t val_ui64, val2_ui64;
  int64_t val_i64;
  uint32_t val_ui32;
  uint16_t val_ui16;
  rsmi_dev_perf_level_t pfl;
  rsmi_frequencies_t f;
  uint32_t num_monitor_devs = 0;
  rsmi_gpu_metrics_t gpu_metrics;
  std::string val_str;

  RSMI_POWER_TYPE power_type = RSMI_INVALID_POWER;

  rsmi_num_monitor_devices(&num_monitor_devs);
  for (uint32_t i = 0; i < num_monitor_devs; ++i) {
    std::cout << "\t**Device #: " << std::dec << i << "\n";
    ret = rsmi_dev_id_get(i, &val_ui16);
    CHK_RSMI_RET_I(ret)
    std::cout << "\t**Device ID: 0x" << std::hex << val_ui16 << "\n";
    ret = rsmi_dev_revision_get(i, &val_ui16);
    CHK_RSMI_RET_I(ret)
    std::cout << "\t**Dev.Rev.ID: 0x" << std::hex << val_ui16 << "\n";
    ret = rsmi_dev_target_graphics_version_get(i, &val_ui64);
    std::cout << "\t**Target Graphics Version: " << std::dec
    << static_cast<uint64_t>(val_ui64) << "\n";
    ret = rsmi_dev_guid_get(i, &val_ui64);
    std::cout << "\t**GUID: " << std::dec
    << static_cast<uint64_t>(val_ui64) << "\n";
    ret = rsmi_dev_node_id_get(i, &val_ui32);
    std::cout << "\t**Node ID: " << std::dec
    << static_cast<uint32_t>(val_ui32) << "\n";
    char vbios_version[256];
    ret = rsmi_dev_vbios_version_get(i, vbios_version, 256);
    if (ret == RSMI_STATUS_SUCCESS) {
      std::cout << "\t**VBIOS Version: " << vbios_version << "\n";
    } else {
      std::cout << "\t**VBIOS Version: "
      << amd::smi::getRSMIStatusString(ret, false) << "\n";
    }

    char current_compute_partition[256];
    current_compute_partition[0] = '\0';
    ret = rsmi_dev_compute_partition_get(i, current_compute_partition, 256);
    std::cout << "\t**Current Compute Partition: "
              << (((current_compute_partition == nullptr)
                  || ((current_compute_partition != nullptr)
                  && (current_compute_partition[0] == '\0')))
                  ? "UNKNOWN" : current_compute_partition);
    if (ret != RSMI_STATUS_SUCCESS) {
      std::cout << ", RSMI_STATUS = ";
    } else {
      std::cout << "\n";
    }
    CHK_RSMI_NOT_SUPPORTED_OR_UNEXPECTED_DATA_RET(ret)

    const uint32_t kLength = 5;
    char memory_partition[kLength];
    memory_partition[0] = '\0';
    ret = rsmi_dev_memory_partition_get(i, memory_partition, kLength);
    std::cout << "\t**Current Memory Partition: "
              << (((memory_partition == nullptr)
                  || ((memory_partition != nullptr)
                  && (memory_partition[0] == '\0')))
                  ? "UNKNOWN" : memory_partition);
    if (ret != RSMI_STATUS_SUCCESS) {
      std::cout << ", RSMI_STATUS = ";
    } else {
      std::cout << "\n";
    }
    CHK_NOT_SUPPORTED_OR_UNEXPECTED_DATA_OR_INSUFFICIENT_SIZE_RET(ret)

    std::cout << "\t**rsmi_minmax_bandwidth_get(0, " << i << ", ...): ";
    ret = rsmi_dev_pci_id_get(0, &val_ui64);
    ret = rsmi_dev_pci_id_get(i, &val2_ui64);
    if (i > 0 && val_ui64 != val2_ui64) {
      uint64_t min_bandwidth = 0;
      uint64_t max_bandwidth = 0;
      ret = rsmi_minmax_bandwidth_get(0, i, &min_bandwidth, &max_bandwidth);
      CHK_RSMI_NOT_SUPPORTED_OR_UNEXPECTED_DATA_RET(ret)
      std::cout << "\n\t**\tMinimum Bandwidth: " << std::dec << min_bandwidth
                << "\n\t**\tMaximum Bandwidth: " << std::dec
                << max_bandwidth << "\n";
    } else {
      std::cout << "Not Supported\n";
    }

    //
    std::cout << "\n";
    print_test_header("GPU METRICS: Using static struct (Backwards Compatibility) ", i);
    ret = rsmi_dev_gpu_metrics_info_get(i, &gpu_metrics);
    print_function_header_with_rsmi_ret(ret, "rsmi_dev_gpu_metrics_info_get("
      + std::to_string(i) + ", &gpu_metrics)");

    std::cout << "\t**.common_header.format_revision : "
              << print_unsigned_int(gpu_metrics.common_header.format_revision) << "\n";
    std::cout << "\t**.common_header.content_revision : "
              << print_unsigned_int(gpu_metrics.common_header.content_revision) << "\n";

    std::cout << "\t**.temperature_edge : " << std::dec
              << gpu_metrics.temperature_edge << "\n";
    std::cout << "\t**.temperature_hotspot : " << std::dec
              << gpu_metrics.temperature_hotspot << "\n";
    std::cout << "\t**.temperature_mem : " << std::dec
              << gpu_metrics.temperature_mem << "\n";
    std::cout << "\t**.temperature_vrgfx : " << std::dec
              << gpu_metrics.temperature_vrgfx << "\n";
    std::cout << "\t**.temperature_vrsoc : " << std::dec
              << gpu_metrics.temperature_vrsoc << "\n";
    std::cout << "\t**.temperature_vrmem : " << std::dec
              << gpu_metrics.temperature_vrmem << "\n";
    std::cout << "\t**.average_gfx_activity : " << std::dec
              << gpu_metrics.average_gfx_activity << "\n";
    std::cout << "\t**.average_umc_activity : " << std::dec
              << gpu_metrics.average_umc_activity << "\n";
    std::cout << "\t**.average_mm_activity : " << std::dec
              << gpu_metrics.average_mm_activity << "\n";
    std::cout << "\t**.average_socket_power : " << std::dec
              << gpu_metrics.average_socket_power << "\n";
    std::cout << "\t**.energy_accumulator : " << std::dec
              << gpu_metrics.energy_accumulator << "\n";
    std::cout << "\t**.system_clock_counter : " << std::dec
              << gpu_metrics.system_clock_counter << "\n";
    std::cout << "\t**.average_gfxclk_frequency : " << std::dec
              << gpu_metrics.average_gfxclk_frequency << "\n";
    std::cout << "\t**.average_socclk_frequency : " << std::dec
              << gpu_metrics.average_socclk_frequency << "\n";
    std::cout << "\t**.average_uclk_frequency : " << std::dec
              << gpu_metrics.average_uclk_frequency << "\n";
    std::cout << "\t**.average_vclk0_frequency : " << std::dec
              << gpu_metrics.average_vclk0_frequency<< "\n";
    std::cout << "\t**.average_dclk0_frequency : " << std::dec
              << gpu_metrics.average_dclk0_frequency << "\n";
    std::cout << "\t**.average_vclk1_frequency : " << std::dec
              << gpu_metrics.average_vclk1_frequency << "\n";
    std::cout << "\t**.average_dclk1_frequency : " << std::dec
              << gpu_metrics.average_dclk1_frequency << "\n";
    std::cout << "\t**.current_gfxclk : " << std::dec
              << gpu_metrics.current_gfxclk << "\n";
    std::cout << "\t**.current_socclk : " << std::dec
              << gpu_metrics.current_socclk << "\n";
    std::cout << "\t**.current_uclk : " << std::dec
              << gpu_metrics.current_uclk << "\n";
    std::cout << "\t**.current_vclk0 : " << std::dec
              << gpu_metrics.current_vclk0 << "\n";
    std::cout << "\t**.current_dclk0 : " << std::dec
              << gpu_metrics.current_dclk0 << "\n";
    std::cout << "\t**.current_vclk1 : " << std::dec
              << gpu_metrics.current_vclk1 << "\n";
    std::cout << "\t**.current_dclk1 : " << std::dec
              << gpu_metrics.current_dclk1 << "\n";
    std::cout << "\t**.throttle_status : " << std::dec
              << gpu_metrics.throttle_status << "\n";
    std::cout << "\t**.current_fan_speed : " << std::dec
              << gpu_metrics.current_fan_speed << "\n";
    std::cout << "\t**.pcie_link_width : " << std::dec
              << gpu_metrics.pcie_link_width << "\n";
    std::cout << "\t**.pcie_link_speed : " << std::dec
              << gpu_metrics.pcie_link_speed << "\n";
    std::cout << "\t**.gfx_activity_acc : " << std::dec
              << gpu_metrics.gfx_activity_acc << "\n";
    std::cout << "\t**.mem_activity_acc : " << std::dec
              << gpu_metrics.mem_activity_acc << "\n";
    std::cout << "\t**.firmware_timestamp : " << std::dec
              << gpu_metrics.firmware_timestamp << "\n";
    std::cout << "\t**.voltage_soc : " << std::dec
              << gpu_metrics.voltage_soc << "\n";
    std::cout << "\t**.voltage_gfx : " << std::dec
              << gpu_metrics.voltage_gfx << "\n";
    std::cout << "\t**.voltage_mem : " << std::dec
              << gpu_metrics.voltage_mem << "\n";
    std::cout << "\t**.indep_throttle_status : " << std::dec
              << gpu_metrics.indep_throttle_status << "\n";
    std::cout << "\t**.current_socket_power : " << std::dec
              << gpu_metrics.current_socket_power << "\n";
    std::cout << "\t**.gfxclk_lock_status : " << std::dec
              << gpu_metrics.gfxclk_lock_status << "\n";
    std::cout << "\t**.xgmi_link_width : " << std::dec
              << gpu_metrics.xgmi_link_width << "\n";
    std::cout << "\t**.xgmi_link_speed : " << std::dec
              << gpu_metrics.xgmi_link_speed << "\n";
    std::cout << "\t**.pcie_bandwidth_acc : " << std::dec
              << gpu_metrics.pcie_bandwidth_acc << "\n";
    std::cout << "\t**.pcie_bandwidth_inst : " << std::dec
              << gpu_metrics.pcie_bandwidth_inst << "\n";
    std::cout << "\t**.vram_max_bandwidth=" << std::dec
              << gpu_metrics.vram_max_bandwidth << "\n";
    std::cout << "\t**.pcie_l0_to_recov_count_acc : " << std::dec
              << gpu_metrics.pcie_l0_to_recov_count_acc << "\n";
    std::cout << "\t**.pcie_replay_count_acc : " << std::dec
              << gpu_metrics.pcie_replay_count_acc << "\n";
    std::cout << "\t**.pcie_replay_rover_count_acc : " << std::dec
              << gpu_metrics.pcie_replay_rover_count_acc << "\n";
    std::cout << "\t**.accumulation_counter : " << std::dec
              << gpu_metrics.accumulation_counter << "\n";
    std::cout << "\t**.prochot_residency_acc : " << std::dec
              << gpu_metrics.prochot_residency_acc << "\n";
    std::cout << "\t**.ppt_residency_acc : " << std::dec
              << gpu_metrics.ppt_residency_acc << "\n";
    std::cout << "\t**.socket_thm_residency_acc : " << std::dec
              << gpu_metrics.socket_thm_residency_acc << "\n";
    std::cout << "\t**.vr_thm_residency_acc : " << std::dec
              << gpu_metrics.vr_thm_residency_acc << "\n";
    std::cout << "\t**.hbm_thm_residency_acc : " << std::dec
              << gpu_metrics.hbm_thm_residency_acc << "\n";
    std::cout  << "\t**.num_partition: " << std::dec
               << gpu_metrics.num_partition << "\n";
    std::cout << "\t**.pcie_lc_perf_other_end_recovery: "
              << gpu_metrics.pcie_lc_perf_other_end_recovery << "\n";

    std::cout << "\t**.temperature_hbm[] : " << std::dec << "\n";
    for (const auto& temp : gpu_metrics.temperature_hbm) {
      std::cout << "\t  -> " << std::dec << temp << "\n";
    }

    std::cout << "\t**.vcn_activity[] : " << std::dec << "\n";
    for (const auto& vcn : gpu_metrics.vcn_activity) {
      std::cout << "\t  -> " << std::dec << vcn << "\n";
    }

    std::cout << "\t**.xgmi_read_data_acc[] : " << std::dec << "\n";
    for (const auto& read_data : gpu_metrics.xgmi_read_data_acc) {
      std::cout << "\t  -> " << std::dec << read_data << "\n";
    }

    std::cout << "\t**.xgmi_write_data_acc[] : " << std::dec << "\n";
    for (const auto& write_data : gpu_metrics.xgmi_write_data_acc) {
      std::cout << "\t  -> " << std::dec << write_data << "\n";
    }

    std::cout << "\t**.xgmi_link_status[] : " << std::dec << "\n";
    for (const auto& write_data : gpu_metrics.xgmi_link_status) {
      std::cout << "\t  -> " << std::dec << write_data << "\n";
    }

    std::cout << "\t**.current_gfxclks[] : " << std::dec << "\n";
    for (const auto& gfxclk : gpu_metrics.current_gfxclks) {
      std::cout << "\t  -> " << std::dec << gfxclk << "\n";
    }

    std::cout << "\t**.current_socclks[] : " << std::dec << "\n";
    for (const auto& socclk : gpu_metrics.current_socclks) {
      std::cout << "\t  -> " << std::dec << socclk << "\n";
    }

    std::cout << "\t**.current_vclk0s[] : " << std::dec << "\n";
    for (const auto& vclk : gpu_metrics.current_vclk0s) {
      std::cout << "\t  -> " << std::dec << vclk << "\n";
    }

    std::cout << "\t**.current_dclk0s[] : " << std::dec << "\n";
    for (const auto& dclk : gpu_metrics.current_dclk0s) {
      std::cout << "\t  -> " << std::dec << dclk << "\n";
    }

    std::cout << "\t**.jpeg_activity[] : " << std::dec << "\n";
    for (const auto& jpeg : gpu_metrics.jpeg_activity) {
      std::cout << "\t  -> " << std::dec << jpeg << "\n";
    }

    std::cout << std::dec << "xcp_stats.gfx_busy_inst = \n";
    auto xcp = 0;
    for (auto& row : gpu_metrics.xcp_stats) {
      std::cout << "XCP[" << xcp << "] = " << "[ ";
      std::copy(std::begin(row.gfx_busy_inst),
                std::end(row.gfx_busy_inst),
                amd::smi::make_ostream_joiner(&std::cout, ", "));
      std::cout << " ]\n";
      xcp++;
    }

    xcp = 0;
    std::cout << std::dec << "xcp_stats.jpeg_busy = \n";
    for (auto& row : gpu_metrics.xcp_stats) {
      std::cout << "XCP[" << xcp << "] = " << "[ ";
      std::copy(std::begin(row.jpeg_busy),
                std::end(row.jpeg_busy),
                amd::smi::make_ostream_joiner(&std::cout, ", "));
      std::cout << " ]\n";
      xcp++;
    }

    xcp = 0;
    std::cout << std::dec << "xcp_stats.vcn_busy = \n";
    for (auto& row : gpu_metrics.xcp_stats) {
      std::cout << "XCP[" << xcp << "] = " << "[ ";
      std::copy(std::begin(row.vcn_busy),
                std::end(row.vcn_busy),
                amd::smi::make_ostream_joiner(&std::cout, ", "));
      std::cout << " ]\n";
      xcp++;
    }

    xcp = 0;
    std::cout << std::dec << "xcp_stats.gfx_busy_acc = \n";
    for (auto& row : gpu_metrics.xcp_stats) {
      std::cout << "XCP[" << xcp << "] = " << "[ ";
        std::copy(std::begin(row.gfx_busy_acc),
                  std::end(row.gfx_busy_acc),
                  amd::smi::make_ostream_joiner(&std::cout, ", "));
        std::cout << " ]\n";
        xcp++;
    }

    xcp = 0;
    std::cout << std::dec << "xcp_stats.gfx_below_host_limit_acc = \n";  // new for 1.7
    for (auto& row : gpu_metrics.xcp_stats) {
      std::cout << "XCP[" << xcp << "] = " << "[ ";
      std::copy(std::begin(row.gfx_below_host_limit_acc),
              std::end(row.gfx_below_host_limit_acc),
              amd::smi::make_ostream_joiner(&std::cout, ", "));
      std::cout << " ]\n";
      xcp++;
    }

    xcp = 0;
    std::cout << std::dec << "xcp_stats.gfx_below_host_limit_ppt_acc = \n";  // new for 1.8
    for (auto& row : gpu_metrics.xcp_stats) {
      std::cout << "XCP[" << xcp << "] = " << "[ ";
      std::copy(std::begin(row.gfx_below_host_limit_ppt_acc),
              std::end(row.gfx_below_host_limit_ppt_acc),
              amd::smi::make_ostream_joiner(&std::cout, ", "));
      std::cout << " ]\n";
      xcp++;
    }

    xcp = 0;
    std::cout << std::dec << "xcp_stats.gfx_below_host_limit_thm_acc = \n";  // new for 1.8
    for (auto& row : gpu_metrics.xcp_stats) {
      std::cout << "XCP[" << xcp << "] = " << "[ ";
      std::copy(std::begin(row.gfx_below_host_limit_thm_acc),
              std::end(row.gfx_below_host_limit_thm_acc),
              amd::smi::make_ostream_joiner(&std::cout, ", "));
      std::cout << " ]\n";
      xcp++;
    }

    xcp = 0;
    std::cout << std::dec << "xcp_stats.gfx_low_utilization_acc = \n";
    for (auto& row : gpu_metrics.xcp_stats) {
      std::cout << "XCP[" << xcp << "] = " << "[ ";
      std::copy(std::begin(row.gfx_low_utilization_acc),
              std::end(row.gfx_low_utilization_acc),
              amd::smi::make_ostream_joiner(&std::cout, ", "));
      std::cout << " ]\n";
      xcp++;
    }

    xcp = 0;
    std::cout << std::dec << "xcp_stats.gfx_below_host_limit_total_acc = \n";
    for (auto& row : gpu_metrics.xcp_stats) {
      std::cout << "XCP[" << xcp << "] = " << "[ ";
      std::copy(std::begin(row.gfx_below_host_limit_total_acc),
              std::end(row.gfx_below_host_limit_total_acc),
              amd::smi::make_ostream_joiner(&std::cout, ", "));
      std::cout << " ]\n";
      xcp++;
    }

    std::cout << "\n";
    std::cout << "\t ** -> Checking metrics with constant changes ** " << "\n";
    constexpr uint16_t kMAX_ITER_TEST = 10;
    rsmi_gpu_metrics_t gpu_metrics_check;
    for (auto idx = uint16_t(1); idx <= kMAX_ITER_TEST; ++idx) {
        rsmi_dev_gpu_metrics_info_get(i, &gpu_metrics_check);
        std::cout << "\t\t -> firmware_timestamp [" << idx
        << "/" << kMAX_ITER_TEST << "]: " << gpu_metrics_check.firmware_timestamp << "\n";
    }

    std::cout << "\n";
    for (auto idx = uint16_t(1); idx <= kMAX_ITER_TEST; ++idx) {
        rsmi_dev_gpu_metrics_info_get(i, &gpu_metrics_check);
        std::cout << "\t\t -> system_clock_counter [" << idx
        << "/" << kMAX_ITER_TEST << "]: " << gpu_metrics_check.system_clock_counter << "\n";
    }

    std::cout << "\n\n";
    std::cout << " ** Note: Values MAX'ed out "
    "(UINTX MAX are unsupported for the version in question) ** " << "\n";


    std::cout << "\n\n";
    print_test_header("GPU METRICS: Using direct APIs (newer)", i);
    metrics_table_header_t header_values;

    ret = rsmi_dev_metrics_header_info_get(i, &header_values);
    std::cout << "\t[Metrics Header]" << "\n";
    std::cout << "\t  -> format_revision  : "
    << print_unsigned_int(header_values.format_revision) << "\n";
    std::cout << "\t  -> content_revision : "
    << print_unsigned_int(header_values.content_revision) << "\n";
    std::cout << "\t--------------------" << "\n";

    std::cout << "\n";
    std::cout << "\t[XCD CounterVoltage]" << "\n";
    ret = rsmi_dev_metrics_xcd_counter_get(i, &val_ui16);
    std::cout << "\t  -> xcd_counter(): " << val_ui16;
    std::cout << "\n\n";

    ret = rsmi_dev_perf_level_get(i, &pfl);
    CHK_AND_PRINT_RSMI_ERR_RET(ret)
    std::cout << "\t**Performance Level:" <<
                                          perf_level_string(pfl) << "\n";
    ret = rsmi_dev_overdrive_level_get(i, &val_ui32);
    std::cout << "\t**OverDrive Level: ";
    if (ret == RSMI_STATUS_SUCCESS) {
      std::cout << val_ui32 << "\n";
    } else {
      CHK_RSMI_NOT_SUPPORTED_OR_UNEXPECTED_DATA_RET(ret)
    }

    print_test_header("GPU Clocks", i);
    for (int clkType = static_cast<int>(RSMI_CLK_TYPE_SYS);
       clkType <= static_cast<int>(RSMI_CLK_TYPE_PCIE);
       clkType++) {
      rsmi_clk_type_t type = static_cast<rsmi_clk_type_t>(clkType);
      ret = rsmi_dev_gpu_clk_freq_get(i, type, &f);
      print_function_header_with_rsmi_ret(ret,
        "rsmi_dev_gpu_clk_freq_get(" + std::to_string(i) +
        ", " + clock_type_string(type) + ", &f)");
      if (ret != RSMI_STATUS_SUCCESS) {
        continue;
      }
      std::cout << "\t** " << clock_type_string(type)
                << " - Supported # of freqs: ";
      std::cout << f.num_supported << "\n";
      std::cout << "\t** " << clock_type_string(type) << " f.current: "
                << f.current << "\n";
      print_frequencies(&f);
    }

    std::cout << "\t**Monitor name: ";
    char name[128];
    ret = rsmi_dev_name_get(i, name, 128);
    CHK_AND_PRINT_RSMI_ERR_RET(ret)
    std::cout << name << "\n";

    std::cout << "\t**Temperature (edge): ";
    ret = rsmi_dev_temp_metric_get(i, RSMI_TEMP_TYPE_EDGE,
      rsmi_temperature_metric_t::RSMI_TEMP_CURRENT, &val_i64);
    if (ret == RSMI_STATUS_SUCCESS) {
      std::cout << std::dec << val_i64/1000 << " C" << "\n";
    }
    CHK_RSMI_NOT_SUPPORTED_RET(ret)

    std::cout << "\t**Temperature (junction): ";
    ret = rsmi_dev_temp_metric_get(i, RSMI_TEMP_TYPE_JUNCTION,
      rsmi_temperature_metric_t::RSMI_TEMP_CURRENT, &val_i64);
    if (ret == RSMI_STATUS_SUCCESS) {
      std::cout << std::dec << (val_i64 / 1000) << " C" << "\n";
    }
    CHK_RSMI_NOT_SUPPORTED_RET(ret)

    std::cout << "\t**Voltage: ";
    ret = rsmi_dev_volt_metric_get(i, RSMI_VOLT_TYPE_VDDGFX,
                                               RSMI_VOLT_CURRENT, &val_i64);
    if (ret == RSMI_STATUS_SUCCESS) {
      std::cout << val_i64 << "mV" << "\n";
    }
    CHK_RSMI_NOT_SUPPORTED_RET(ret)

    std::cout << "\t**Current Fan Speed: ";
    ret = rsmi_dev_fan_speed_get(i, 0, &val_i64);
    if (ret == RSMI_STATUS_SUCCESS) {
      ret = rsmi_dev_fan_speed_max_get(i, 0, &val_ui64);
      CHK_AND_PRINT_RSMI_ERR_RET(ret)
      std::cout << (static_cast<float>(val_i64)/val_ui64) * 100;
      std::cout << "% (" << std::dec << val_i64 << "/"
                << std::dec << val_ui64 << ")" << "\n";
    }
    CHK_RSMI_NOT_SUPPORTED_RET(ret)

    std::cout << "\t**Current fan RPMs: ";
    ret = rsmi_dev_fan_rpms_get(i, 0, &val_i64);
    if (ret == RSMI_STATUS_SUCCESS) {
      std::cout << std::dec << val_i64 << "\n";
    }
    CHK_RSMI_NOT_SUPPORTED_RET(ret)

    std::cout << "\t**Current Power Cap: ";
    ret = rsmi_dev_power_cap_get(i, 0, &val_ui64);
    if (ret == RSMI_STATUS_SUCCESS) {
      std::cout << std::dec << val_ui64 << "uW" <<"\n";
    }
    CHK_RSMI_NOT_SUPPORTED_RET(ret)

    std::cout << "\t**Power Cap Range: ";
    ret = rsmi_dev_power_cap_range_get(i, 0, &val_ui64, &val2_ui64);
    if (ret == RSMI_STATUS_SUCCESS) {
      std::cout << std::dec << val2_ui64 << " to "
                << std::dec << val_ui64 << " uW" << "\n";
    }
    CHK_RSMI_NOT_SUPPORTED_RET(ret)

    std::cout << "\t**Average Power Usage: ";
    ret = rsmi_dev_power_ave_get(i, 0, &val_ui64);
    if (ret == RSMI_STATUS_SUCCESS) {
      std::cout << convert_mw_to_w(val_ui64) << " W" << "\n";
    }
    CHK_RSMI_NOT_SUPPORTED_RET(ret)

    std::cout << "\t**Current Socket Power Usage: ";
    ret = rsmi_dev_current_socket_power_get(i, &val_ui64);
    if (ret == RSMI_STATUS_SUCCESS) {
      std::cout << convert_mw_to_w(val_ui64) << " W" << "\n";
    }
    CHK_RSMI_NOT_SUPPORTED_RET(ret)

    std::cout << "\t**Generic Power Usage: ";
    ret = rsmi_dev_power_get(i, &val_ui64, &power_type);
    if (ret == RSMI_STATUS_SUCCESS) {
      std::cout << "[" << amd::smi::power_type_string(power_type) << "] "
                << convert_mw_to_w(val_ui64) << " W" << "\n";
    }
    CHK_RSMI_NOT_SUPPORTED_RET(ret)
    std::cout << "\t=======" << "\n";
  }

  std::cout << "***** Testing write api's" << "\n";
  if (amd::smi::is_sudo_user() == false) {
    std::cout << "Write APIs require users to execute with sudo. "
              << "Cannot proceed." << "\n";
    return 0;
  }

  for (uint32_t i = 0; i < num_monitor_devs; ++i) {
    ret = test_set_perf_level(i);
    CHK_AND_PRINT_RSMI_ERR_RET(ret)

    ret = test_set_fan_speed(i);
    CHK_AND_PRINT_RSMI_ERR_RET(ret)

    ret = test_power_cap(i);
    CHK_AND_PRINT_RSMI_ERR_RET(ret)

    ret = test_power_profile(i);
    CHK_AND_PRINT_RSMI_ERR_RET(ret)

    ret = test_set_compute_partitioning(i);
    CHK_AND_PRINT_RSMI_ERR_RET(ret)

    ret = test_set_freq(i);
    CHK_AND_PRINT_RSMI_ERR_RET(ret)

    ret = test_set_memory_partition(i);
    CHK_AND_PRINT_RSMI_ERR_RET(ret)

    ret = test_set_overdrive(i);
    CHK_RSMI_NOT_SUPPORTED_RET(ret)
  }

  return 0;
}
