/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018, Advanced Micro Devices, Inc.
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

#include <iostream>
#include <bitset>
#include <string>
#include <algorithm>
#include <vector>
#include <map>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/rsmi_sanity.h"
#include "gtest/gtest.h"

static const uint32_t kNumBufferElements = 256;
static uint32_t gVerbosity = 3;

#define DISPLAY_RSMI_ERR(RET) { \
  if (RET != RSMI_STATUS_SUCCESS) { \
    const char *err_str; \
    std::cout << "RSMI call returned " << (RET); \
    rsmi_status_string((RET), &err_str); \
    std::cout << " (" << err_str << ")" << std::endl; \
    std::cout << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
  } \
}

#define CHK_ERR_RET(RET) { \
  DISPLAY_RSMI_ERR(RET) \
  if ((RET) != RSMI_STATUS_SUCCESS) { \
    return (RET); \
  } \
}

#define CHK_ERR_ASRT(RET) { \
    DISPLAY_RSMI_ERR(RET) \
    ASSERT_EQ((RET), RSMI_STATUS_SUCCESS); \
}

#define CHK_RSMI_PERM_ERR(RET) { \
    if (RET == RSMI_STATUS_PERMISSION) { \
      std::cout << "This command requires root access." << std::endl; \
    } else { \
      DISPLAY_RSMI_ERR(RET) \
    } \
}

#define ALT_ASSERT_EQ(A, B) { \
    if ((A) != (B)) { \
      std::cout << "ASSERT failure: Expected " << #A << " == " << #B << \
        ", but got " << #A << " = " << (A) << ", and " << #B << " = " << \
                                                           (B) << std::endl; \
      std::cout << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      return RSMI_STATUS_UNKNOWN_ERROR; \
    } \
}

#define IF_VERB(VB) if (gVerbosity && gVerbosity >= (TestBase::VERBOSE_##VB))

static void print_test_header(const char *str, uint32_t dv_ind) {
  IF_VERB(STANDARD) {
    std::cout << "********************************" << std::endl;
    std::cout << "*** " << str << std::endl;
    std::cout << "********************************" << std::endl;
    std::cout << "Device index: " << dv_ind << std::endl;
  }
}

static const char *
power_profile_string(rsmi_power_profile_preset_masks profile) {
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
static rsmi_status_t test_power_profile(uint32_t dv_ind) {
  rsmi_status_t ret;
  rsmi_power_profile_status status;

  print_test_header("Power Profile", dv_ind);

  ret = rsmi_dev_power_profile_presets_get(dv_ind, 0, &status);
  CHK_ERR_RET(ret)

  IF_VERB(STANDARD) {
    std::cout << "The available power profiles are:" << std::endl;
    uint64_t tmp = 1;
    while (tmp <= RSMI_PWR_PROF_PRST_LAST) {
      if ((tmp & status.available_profiles) == tmp) {
        std::cout << "\t" <<
            power_profile_string((rsmi_power_profile_preset_masks)tmp) <<
                                                                    std::endl;
      }
      tmp = tmp << 1;
    }
    std::cout << "The current power profile is: " <<
                            power_profile_string(status.current) << std::endl;
  }

  rsmi_power_profile_preset_masks orig_profile = status.current;

  // Try setting the profile to a different power profile
  rsmi_bit_field diff_profiles;
  rsmi_power_profile_preset_masks new_prof;
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
    std::cout << "No other non-custom power profiles to set to" << std::endl;
    return ret;
  }

  ret = rsmi_dev_power_profile_set(dv_ind, 0, new_prof);
  CHK_ERR_RET(ret)

  rsmi_dev_perf_level pfl;
  ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
  CHK_ERR_RET(ret)
  ALT_ASSERT_EQ(pfl, RSMI_DEV_PERF_LEVEL_MANUAL);

  ret = rsmi_dev_power_profile_presets_get(dv_ind, 0, &status);
  CHK_ERR_RET(ret)

  ALT_ASSERT_EQ(status.current, new_prof);

  ret = rsmi_dev_perf_level_set(dv_ind, RSMI_DEV_PERF_LEVEL_AUTO);
  CHK_ERR_RET(ret)

  ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
  CHK_ERR_RET(ret)
  ALT_ASSERT_EQ(pfl, RSMI_DEV_PERF_LEVEL_AUTO);

  ret = rsmi_dev_power_profile_presets_get(dv_ind, 0, &status);
  CHK_ERR_RET(ret)

  ALT_ASSERT_EQ(status.current, orig_profile);

  return ret;
}

static rsmi_status_t test_power_cap(uint32_t dv_ind) {
  rsmi_status_t ret;
  uint64_t orig, min, max, new_cap;

  print_test_header("Power Control", dv_ind);

  ret = rsmi_dev_power_cap_range_get(dv_ind, 0, &max, &min);
  CHK_ERR_RET(ret)

  ret = rsmi_dev_power_cap_get(dv_ind, 0, &orig);
  CHK_ERR_RET(ret)

  new_cap = (max + min)/2;

  IF_VERB(STANDARD) {
    std::cout << "Original Power Cap: " << orig << " uW" << std::endl;
    std::cout << "Power Cap Range: " << max << " uW to " << min <<
                                                           " uW" << std::endl;
    std::cout << "Setting new cap to " << new_cap << "..." << std::endl;
  }
  ret = rsmi_dev_power_cap_set(dv_ind, 0, new_cap);
  CHK_ERR_RET(ret)

  ret = rsmi_dev_power_cap_get(dv_ind, 0, &new_cap);
  CHK_ERR_RET(ret)

  // TODO(cfreehil) add some kind of assertion to verify new_cap is correct
  //       (or within a range)
  IF_VERB(STANDARD) {
    std::cout << "New Power Cap: " << new_cap << " uW" << std::endl;
    std::cout << "Resetting cap to " << orig << "..." << std::endl;
  }

  ret = rsmi_dev_power_cap_set(dv_ind, 0, orig);
  CHK_ERR_RET(ret)

  ret = rsmi_dev_power_cap_get(dv_ind, 0, &new_cap);
  CHK_ERR_RET(ret)

  IF_VERB(STANDARD) {
    std::cout << "Current Power Cap: " << new_cap << " uW" << std::endl;
  }
  return ret;
}

static rsmi_status_t test_set_overdrive(uint32_t dv_ind) {
  rsmi_status_t ret;
  uint32_t val;

  print_test_header("Overdrive Control", dv_ind);
  IF_VERB(STANDARD) {
    std::cout << "Set Overdrive level to 0%..." << std::endl;
  }
  ret = rsmi_dev_overdrive_level_set(dv_ind, 0);
  CHK_ERR_RET(ret)
  IF_VERB(STANDARD) {
    std::cout << "Set Overdrive level to 10%..." << std::endl;
  }
  ret = rsmi_dev_overdrive_level_set(dv_ind, 10);
  CHK_ERR_RET(ret)
  ret = rsmi_dev_overdrive_level_get(dv_ind, &val);
  CHK_ERR_RET(ret)
  IF_VERB(STANDARD) {
    std::cout << "\t**New OverDrive Level:" << val << std::endl;
    std::cout << "Reset Overdrive level to 0%..." << std::endl;
  }
  ret = rsmi_dev_overdrive_level_set(dv_ind, 0);
  CHK_ERR_RET(ret)
  ret = rsmi_dev_overdrive_level_get(dv_ind, &val);
  CHK_ERR_RET(ret)
  IF_VERB(STANDARD) {
    std::cout << "\t**New OverDrive Level:" << val << std::endl;
  }
  return ret;
}

static rsmi_status_t test_set_fan_speed(uint32_t dv_ind) {
  rsmi_status_t ret;
  int64_t orig_speed;
  int64_t new_speed;
  int64_t cur_speed;

  print_test_header("Fan Speed Control", dv_ind);

  ret = rsmi_dev_fan_speed_get(dv_ind, 0, &orig_speed);
  CHK_ERR_RET(ret)
  IF_VERB(STANDARD) {
    std::cout << "Original fan speed: " << orig_speed << std::endl;
  }

  if (orig_speed == 0) {
    std::cout << "***System fan speed value is 0. Skip fan test." << std::endl;
    return RSMI_STATUS_SUCCESS;
  }

  new_speed = 1.1 * orig_speed;

  IF_VERB(STANDARD) {
    std::cout << "Setting fan speed to " << new_speed << std::endl;
  }

  ret = rsmi_dev_fan_speed_set(dv_ind, 0, new_speed);
  CHK_ERR_RET(ret)

  sleep(4);

  ret = rsmi_dev_fan_speed_get(dv_ind, 0, &cur_speed);
  CHK_ERR_RET(ret)

  IF_VERB(STANDARD) {
    std::cout << "New fan speed: " << cur_speed << std::endl;
  }

  EXPECT_TRUE((cur_speed > 0.95 * new_speed && cur_speed < 1.1 * new_speed) ||
      cur_speed > 0.95 * RSMI_MAX_FAN_SPEED);

  IF_VERB(STANDARD) {
    std::cout << "Resetting fan control to auto..." << std::endl;
  }

  ret = rsmi_dev_fan_reset(dv_ind, 0);
  CHK_ERR_RET(ret)

  sleep(3);

  ret = rsmi_dev_fan_speed_get(dv_ind, 0, &cur_speed);
  CHK_ERR_RET(ret)

  IF_VERB(STANDARD) {
    std::cout << "End fan speed: " << cur_speed << std::endl;
  }
  return ret;
}

static const std::map<rsmi_dev_perf_level, const char *> kDevPerfLvlNameMap = {
    {RSMI_DEV_PERF_LEVEL_AUTO, "RSMI_DEV_PERF_LEVEL_AUTO"},
    {RSMI_DEV_PERF_LEVEL_LOW, "RSMI_DEV_PERF_LEVEL_LOW"},
    {RSMI_DEV_PERF_LEVEL_HIGH, "RSMI_DEV_PERF_LEVEL_HIGH"},
    {RSMI_DEV_PERF_LEVEL_MANUAL, "RSMI_DEV_PERF_LEVEL_MANUAL"},
    {RSMI_DEV_PERF_LEVEL_STABLE_STD, "RSMI_DEV_PERF_LEVEL_STABLE_STD"},
    {RSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK,
                                       "RSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK"},
    {RSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK,
                                       "RSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK"},
    {RSMI_DEV_PERF_LEVEL_STABLE_PEAK, "RSMI_DEV_PERF_LEVEL_STABLE_PEAK"},

    {RSMI_DEV_PERF_LEVEL_UNKNOWN, "RSMI_DEV_PERF_LEVEL_UNKNOWN"},
};

static rsmi_status_t test_set_perf_level(uint32_t dv_ind) {
  rsmi_status_t ret;

  rsmi_dev_perf_level pfl, orig_pfl;

  print_test_header("Performance Level Control", dv_ind);

  ret = rsmi_dev_perf_level_get(dv_ind, &orig_pfl);
  CHK_ERR_RET(ret)

  IF_VERB(STANDARD) {
    std::cout << "\t**Original Perf Level:" <<
                                 kDevPerfLvlNameMap.at(orig_pfl) << std::endl;
  }

  uint32_t pfl_i = static_cast<uint32_t>(RSMI_DEV_PERF_LEVEL_FIRST);
  for (; pfl_i <=  static_cast<uint32_t>(RSMI_DEV_PERF_LEVEL_LAST); pfl_i++) {
    if (pfl_i == static_cast<uint32_t>(orig_pfl)) {
      continue;
    }

    IF_VERB(STANDARD) {
      std::cout << "Set Performance Level to " <<
          kDevPerfLvlNameMap.at(static_cast<rsmi_dev_perf_level>(pfl_i)) <<
                                                          " ..." << std::endl;
    }
    ret = rsmi_dev_perf_level_set(dv_ind,
                                     static_cast<rsmi_dev_perf_level>(pfl_i));
    CHK_ERR_RET(ret)
    ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
    CHK_ERR_RET(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**New Perf Level:" << kDevPerfLvlNameMap.at(pfl) <<
                                                                    std::endl;
    }
  }
  std::cout << "Reset Perf level to " << kDevPerfLvlNameMap.at(orig_pfl) <<
                                                          " ..." << std::endl;
  ret = rsmi_dev_perf_level_set(dv_ind, orig_pfl);
  CHK_ERR_RET(ret)
  ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
  CHK_ERR_RET(ret)

  IF_VERB(STANDARD) {
    std::cout << "\t**New Perf Level:" << kDevPerfLvlNameMap.at(pfl) <<
                                                                    std::endl;
  }
  return ret;
}

static rsmi_status_t test_set_freq(uint32_t dv_ind) {
  rsmi_status_t ret;
  rsmi_frequencies f;
  uint32_t freq_bitmask;
  rsmi_clk_type rsmi_clk;

  print_test_header("Clock Frequency Control", dv_ind);
  for (uint32_t clk = (uint32_t)RSMI_CLK_TYPE_FIRST;
                                           clk <= RSMI_CLK_TYPE_LAST; ++clk) {
    rsmi_clk = (rsmi_clk_type)clk;

    ret = rsmi_dev_gpu_clk_freq_get(dv_ind, rsmi_clk, &f);
    CHK_ERR_RET(ret)

    IF_VERB(STANDARD) {
      std::cout << "Initial frequency for clock" << rsmi_clk << " is " <<
                                                      f.current << std::endl;
    }
    // Set clocks to something other than the usual default of the lowest
    // frequency.
    freq_bitmask = 0b01100;  // Try the 3rd and 4th clocks

    std::string freq_bm_str =
               std::bitset<RSMI_MAX_NUM_FREQUENCIES>(freq_bitmask).to_string();

    freq_bm_str.erase(0, std::min(freq_bm_str.find_first_not_of('0'),
                                                       freq_bm_str.size()-1));

    IF_VERB(STANDARD) {
    std::cout << "Setting frequency mask for clock " << rsmi_clk <<
        " to 0b" << freq_bm_str << " ..." << std::endl;
    }
    ret = rsmi_dev_gpu_clk_freq_set(dv_ind, rsmi_clk, freq_bitmask);
    CHK_ERR_RET(ret)

    ret = rsmi_dev_gpu_clk_freq_get(dv_ind, rsmi_clk, &f);
    CHK_ERR_RET(ret)

    IF_VERB(STANDARD) {
      std::cout << "Frequency is now index " << f.current << std::endl;
      std::cout << "Resetting mask to all frequencies." << std::endl;
    }
    ret = rsmi_dev_gpu_clk_freq_set(dv_ind, rsmi_clk, 0xFFFFFFFF);
    CHK_ERR_RET(ret)

    ret = rsmi_dev_perf_level_set(dv_ind, RSMI_DEV_PERF_LEVEL_AUTO);
    CHK_ERR_RET(ret)
  }
  return RSMI_STATUS_SUCCESS;
}

static void print_frequencies(rsmi_frequencies *f) {
  assert(f != nullptr);
  for (uint32_t j = 0; j < f->num_supported; ++j) {
    std::cout << "\t**  " << j << ": " << f->frequency[j];
    if (j == f->current) {
      std::cout << " *";
    }
    std::cout << std::endl;
  }
}

TestSanity::TestSanity(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("rocm_smi_lib Sanity Test");
  set_description("This test runs through most of the rocm_smi_lib calls and "
      "verifies that they execute as expected");
}

TestSanity::~TestSanity(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void TestSanity::SetUp(void) {
  rsmi_status_t err;

  TestBase::SetUp();

  gVerbosity = verbosity();
  err = rsmi_init(0);
  ASSERT_EQ(err, RSMI_STATUS_SUCCESS);

  return;
}

void TestSanity::Run(void) {
  TestBase::Run();

  rsmi_status_t err;
  std::string val_str;
  std::vector<std::string> val_vec;
  uint64_t val_ui64, val2_ui64;
  int64_t val_i64;
  uint32_t val_ui32;
  rsmi_dev_perf_level pfl;
  rsmi_frequencies f;

  uint32_t num_monitor_devs = 0;

  for (uint32_t i = 0; i < num_iteration(); i++) {
    IF_VERB(PROGRESS) {
      std::cout << "Iteration: " << i << std::endl;
      fflush(stdout);
    }

    err = rsmi_num_monitor_devices(&num_monitor_devs);
    ASSERT_EQ(err, RSMI_STATUS_SUCCESS);

    if (num_monitor_devs == 0) {
      std::cout << "No monitor devices found on this machine." << std::endl;
      std::cout << "No ROCm SMI tests can be run." << std::endl;
      break;
    }

    for (uint32_t i = 0; i < num_monitor_devs; ++i) {
      err = rsmi_dev_id_get(i, &val_ui64);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Device ID: 0x" << std::hex << val_ui64 << std::endl;
      }
      err = rsmi_dev_perf_level_get(i, &pfl);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Performance Level:" << std::dec << (uint32_t)pfl <<
                                                                      std::endl;
      }
      err = rsmi_dev_overdrive_level_get(i, &val_ui32);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
      std::cout << "\t**OverDrive Level:" << val_ui32 << std::endl;
      }
      err = rsmi_dev_gpu_clk_freq_get(i, RSMI_CLK_TYPE_MEM, &f);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Supported GPU Memory clock frequencies: ";
        std::cout << f.num_supported << std::endl;
        print_frequencies(&f);
      }
      err = rsmi_dev_gpu_clk_freq_get(i, RSMI_CLK_TYPE_SYS, &f);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Supported GPU clock frequencies: ";
        std::cout << f.num_supported << std::endl;
        print_frequencies(&f);
      }
      char name[20];
      err = rsmi_dev_name_get(i, name, 20);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Monitor name: " << name << std::endl;
      }

      err = rsmi_dev_pci_id_get(i, &val_ui64);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**PCI ID (BDFID): 0x" << std::hex << val_ui64;
        std::cout << " (" << std::dec << val_ui64 << ")" << std::endl;
      }

      auto print_temp_metric = [&](rsmi_temperature_metric met,
                                                          std::string label) {
        err = rsmi_dev_temp_metric_get(i, 0, met, &val_i64);

        if (err != RSMI_STATUS_SUCCESS) {
          if (err == RSMI_STATUS_FILE_ERROR) {
            IF_VERB(STANDARD) {
              std::cout << "\t**" << label << ": " <<
                                 "Not supported on this machine" << std::endl;
              return;
            }
          } else {
            CHK_ERR_ASRT(err)
          }
        }

        IF_VERB(STANDARD) {
          std::cout << "\t**" << label << ": " << val_i64/1000 <<
                                                             "C" << std::endl;
        }
      };
      print_temp_metric(RSMI_TEMP_CURRENT, "Current Temp.");
      print_temp_metric(RSMI_TEMP_MAX, "Temperature max value");
      print_temp_metric(RSMI_TEMP_MIN, "Temperature min value");
      print_temp_metric(RSMI_TEMP_MAX_HYST,
                                "Temperature hysteresis value for max limit");
      print_temp_metric(RSMI_TEMP_MIN_HYST,
                                "Temperature hysteresis value for min limit");
      print_temp_metric(RSMI_TEMP_CRITICAL, "Temperature critical max value");
      print_temp_metric(RSMI_TEMP_CRITICAL_HYST,
                           "Temperature hysteresis value for critical limit");
      print_temp_metric(RSMI_TEMP_EMERGENCY,
                                           "Temperature emergency max value");
      print_temp_metric(RSMI_TEMP_EMERGENCY_HYST,
                          "Temperature hysteresis value for emergency limit");
      print_temp_metric(RSMI_TEMP_CRIT_MIN, "Temperature critical min value");
      print_temp_metric(RSMI_TEMP_CRIT_MIN_HYST,
                       "Temperature hysteresis value for critical min value");
      print_temp_metric(RSMI_TEMP_OFFSET, "Temperature offset");
      print_temp_metric(RSMI_TEMP_LOWEST, "Historical minimum temperature");
      print_temp_metric(RSMI_TEMP_HIGHEST, "Historical maximum temperature");

      err = rsmi_dev_fan_speed_get(i, 0, &val_i64);
      CHK_ERR_ASRT(err)
      err = rsmi_dev_fan_speed_max_get(i, 0, &val_ui64);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Current Fan Speed: ";
        std::cout << val_i64/static_cast<float>(val_ui64)*100;
        std::cout << "% ("<< val_i64 << "/" << val_ui64 << ")" << std::endl;
      }
      err = rsmi_dev_fan_rpms_get(i, 0, &val_i64);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Current fan RPMs: " << val_i64 << std::endl;
      }
#if 0   // Apparently recent change in debug fs causes crash here.
        // Disable able this for now until we debug error path.
      err = rsmi_dev_power_max_get(i, &val_ui64);
      if (err == RSMI_STATUS_NOT_SUPPORTED) {
        const char *s_str;
        err = rsmi_status_string(RSMI_STATUS_NOT_SUPPORTED, &s_str);
        CHK_ERR_ASRT(err)
        std::cout << "\t**rsmi_dev_power_max_get(): " << s_str << std::endl;
      } else {
        IF_VERB(STANDARD) {
          std::cout << "\t**Max Power Usage: ";
          CHK_RSMI_PERM_ERR(err)
          if (err == RSMI_STATUS_SUCCESS) {
            std::cout << static_cast<float>(val_ui64)/1000 << " W" << std::endl;
          }
        }
      }
#endif
      err = rsmi_dev_power_cap_get(i, 0, &val_ui64);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Current Power Cap: " << val_ui64 << "uW" <<std::endl;
      }
      err = rsmi_dev_power_cap_range_get(i, 0, &val_ui64, &val2_ui64);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Power Cap Range: " << val2_ui64 << " to " <<
                                                 val_ui64 << " uW" << std::endl;
      }

      err = rsmi_dev_power_ave_get(i, 0, &val_ui64);
      IF_VERB(STANDARD) {
        std::cout << "\t**Averge Power Usage: ";
        CHK_RSMI_PERM_ERR(err)
        if (err == RSMI_STATUS_SUCCESS) {
          std::cout << static_cast<float>(val_ui64)/1000 << " W" << std::endl;
        }
        std::cout << "\t=======" << std::endl;
      }
    }
    IF_VERB(STANDARD) {
      std::cout << "***** Testing write api's" << std::endl;
    }
    for (uint32_t i = 0; i < num_monitor_devs; ++i) {
      err = test_set_overdrive(i);
      CHK_RSMI_PERM_ERR(err)

      err = test_set_perf_level(i);
      CHK_RSMI_PERM_ERR(err)

      err = test_set_freq(i);
      CHK_RSMI_PERM_ERR(err)

      err = test_set_fan_speed(i);
      CHK_RSMI_PERM_ERR(err)

      err = test_power_cap(i);
      CHK_RSMI_PERM_ERR(err)

      err = test_power_profile(i);
      CHK_RSMI_PERM_ERR(err)
    }
  }
}

void TestSanity::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestSanity::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestSanity::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

