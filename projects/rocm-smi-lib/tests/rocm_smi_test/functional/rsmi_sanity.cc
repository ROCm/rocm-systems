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

#if 0  // reorg test 1

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


static void print_frequencies(rsmi_frequencies *f, uint32_t *l = nullptr) {
  assert(f != nullptr);
  for (uint32_t j = 0; j < f->num_supported; ++j) {
    std::cout << "\t**  " << j << ": " << f->frequency[j];
    if (l != nullptr) {
      std::cout << "T/s; x" << l[j];
    } else {
      std::cout << "Hz";
    }

    if (j == f->current) {
      std::cout << " *";
    }
    std::cout << std::endl;
  }
}


#endif  // reorg test 1

TestSanity::TestSanity(void) :
    TestBase() {
  set_title("rocm_smi_lib Sanity Test");
  set_description("This test runs through most of the rocm_smi_lib calls and "
      "verifies that they execute as expected");
}

TestSanity::~TestSanity(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void TestSanity::SetUp(void) {
//  rsmi_status_t err;

  TestBase::SetUp();

  return;
}

void TestSanity::Run(void) {
  TestBase::Run();

#if 0  // reorg test 2
  rsmi_status_t err;
  std::string val_str;
  std::vector<std::string> val_vec;
  uint64_t val_ui64, val2_ui64;
  int64_t val_i64;
  uint32_t val_ui32;
  rsmi_frequencies f;
  rsmi_pcie_bandwidth b;
  rsmi_version ver = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, nullptr};
  uint32_t num_monitor_devs = 0;
  rsmi_od_volt_freq_data odv;

  err = rsmi_version_get(&ver);
  CHK_ERR_ASRT(err)

  ASSERT_TRUE(ver.major != 0xFFFFFFFF && ver.minor != 0xFFFFFFFF &&
                             ver.patch != 0xFFFFFFFF && ver.build != nullptr);

  IF_VERB(STANDARD) {
    std::cout << "\t**RocM SMI Library version: " << ver.major << "." <<
       ver.minor << "." << ver.patch << " (" << ver.build << ")" << std::endl;
  }

  for (uint32_t i = 0; i < num_iteration(); i++) {
    err = rsmi_num_monitor_devices(&num_monitor_devs);
    ASSERT_EQ(err, RSMI_STATUS_SUCCESS);

    if (num_monitor_devs == 0) {
      std::cout << "No monitor devices found on this machine." << std::endl;
      std::cout << "No ROCm SMI tests can be run." << std::endl;
      break;
    }

    for (uint32_t i = 0; i < num_monitor_devs; ++i) {
      err = test_power_cap(i);
      CHK_RSMI_PERM_ERR(err)

    }
  }
#endif  // reorg test 2
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

