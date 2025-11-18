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

#include <cstdint>

#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "perf_determinism.h"
#include "../test_common.h"


TestPerfDeterminism::TestPerfDeterminism() : TestBase() {
  set_title("AMDSMI Performance Determinism Test");
  set_description("The Performance Determinism tests verifies "
                  "Enabling/Disabling performance determinism mode.");
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
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestPerfDeterminism::Run(void) {
  amdsmi_status_t err;
  amdsmi_dev_perf_level_t pfl;
  amdsmi_od_volt_freq_data_t odv{};
  amdsmi_status_t ret;
  uint64_t clkvalue(0);
  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);
    err =  amdsmi_get_gpu_od_volt_info(processor_handles_[i], &odv);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t** Not supported on this machine\n";
      }
      return;
    }  else if (err == AMDSMI_STATUS_SUCCESS) {
      clkvalue = (odv.curr_sclk_range.lower_bound/1000000) + 50;
    } else {
      IF_VERB(STANDARD) {
        std::cout << "\t** Unable to retrieve lower bound sclk, continue.. \n";
      }
      continue;
    }
    std::cout << "About to rsmi_perf_determinism_mode_set() -->\n";

    err = amdsmi_set_gpu_perf_determinism_mode(processor_handles_[i], clkvalue);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**Not supported on this machine" << std::endl;
      }
      continue;
    } else {
      ret = amdsmi_get_gpu_perf_level(processor_handles_[i], &pfl);
      CHK_ERR_ASRT(ret)
      IF_VERB(STANDARD) {
          std::cout << "\t**New Perf Level:" <<  GetPerfLevelStr(pfl) <<
                                                                  std::endl;
          std::cout << "\t**SCLK is now set to " << clkvalue << std::endl;
      }

      std::cout << "\t**Resetting performance determinism" << std::endl;
      err =  amdsmi_set_gpu_perf_level(processor_handles_[i], AMDSMI_DEV_PERF_LEVEL_AUTO);;
      CHK_ERR_ASRT(err)
      ret = amdsmi_get_gpu_perf_level(processor_handles_[i], &pfl);
      CHK_ERR_ASRT(ret)
      IF_VERB(STANDARD) {
          std::cout << "\t**New Perf Level:" <<  GetPerfLevelStr(pfl) <<
                                                                  std::endl;
      }
    }  // END - SET SUPPORTED
  }  // END - DEVICE LOOP
}
