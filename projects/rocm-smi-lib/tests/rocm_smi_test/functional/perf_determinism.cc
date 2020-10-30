/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2020, Advanced Micro Devices, Inc.
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

#include <stdint.h>
#include <stddef.h>

#include <iostream>
#include <string>
#include <map>
#include <bitset>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/perf_determinism.h"
#include "rocm_smi_test/test_common.h"

static const std::map<rsmi_dev_perf_level_t, const char *>
   kDevPerfLvlNameMap = {
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

TestPerfDeterminism::TestPerfDeterminism() : TestBase() {
  set_title("RSMI Performance Determinism Test");
  set_description("The Performance Determinism tests verifies Enabling/Disabling "
                   "performance determinism mode.");
}

TestPerfDeterminism::~TestPerfDeterminism(void) {
}

void TestPerfDeterminism::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPerfDeterminism::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestPerfDeterminism::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPerfDeterminism::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestPerfDeterminism::Run(void) {
  rsmi_status_t err;
  rsmi_dev_perf_level_t pfl;
  rsmi_frequencies_t f;
  rsmi_status_t ret;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }
  // Set clocks to something other than the usual default of the lowest
  // frequency.
  uint64_t freq_bitmask = 0b01100;  // Try the 3rd and 4th clocks
  std::string freq_bm_str =
          std::bitset<RSMI_MAX_NUM_FREQUENCIES>(freq_bitmask).to_string();

  freq_bm_str.erase(0, std::min(freq_bm_str.find_first_not_of('0'),
                                                   freq_bm_str.size()-1));

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(i);

    err = rsmi_perf_determinism_mode_set(i, freq_bitmask);

    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t** Not supported on this machine" << std::endl;
      }
      return;
    } else {
      ret = rsmi_dev_gpu_clk_freq_get(i, RSMI_CLK_TYPE_SYS, &f);
          if (ret != RSMI_STATUS_SUCCESS) {
              return;
          }

      IF_VERB(STANDARD) {
          std::cout << "\tFrequency is now index " << f.current << std::endl;
      }
      ret = rsmi_dev_perf_level_get(i, &pfl);
      CHK_ERR_ASRT(ret)
      IF_VERB(STANDARD) {
          std::cout << "\t**New Perf Level:" << kDevPerfLvlNameMap.at(pfl) <<
                                                                  std::endl;
      }
      std::cout << "\tResetting performance determinism" << std::endl;
      err = rsmi_dev_perf_level_set(i, RSMI_DEV_PERF_LEVEL_AUTO);;
      CHK_ERR_ASRT(err)
      ret = rsmi_dev_perf_level_get(i, &pfl);
      CHK_ERR_ASRT(ret)
      IF_VERB(STANDARD) {
          std::cout << "\t**New Perf Level:" << kDevPerfLvlNameMap.at(pfl) <<
                                                                  std::endl;
      }
      return;
    }
  }
}
