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

#include <stdint.h>
#include <stddef.h>
#include <gtest/gtest.h>

#include <iostream>
#include <map>

#include "amd_smi/amdsmi.h"
#include "perf_level_read_write.h"
#include "../test_common.h"


TestPerfLevelReadWrite::TestPerfLevelReadWrite() : TestBase() {
  set_title("AMDSMI Performance Level Read/Write Test");
  set_description("The Performance Level tests verify that the performance "
                       "level settings can be read and controlled properly.");
}

TestPerfLevelReadWrite::~TestPerfLevelReadWrite(void) {
}

void TestPerfLevelReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPerfLevelReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestPerfLevelReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPerfLevelReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestPerfLevelReadWrite::Run(void) {
  amdsmi_status_t ret;
  amdsmi_dev_perf_level_t pfl, orig_pfl;

  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) {
      std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    }
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    ret = amdsmi_get_gpu_perf_level(processor_handles_[dv_ind], &orig_pfl);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_perf_level(): Not supported on this machine" << std::endl;
      }
      ASSERT_EQ(ret, AMDSMI_STATUS_NOT_SUPPORTED);
      continue;
    }

    IF_VERB(STANDARD) {
      std::cout << "\t**Original Perf Level:"
                << GetPerfLevelStr(orig_pfl) << std::endl;
    }

    uint32_t pfl_i = static_cast<uint32_t>(AMDSMI_DEV_PERF_LEVEL_FIRST);
    for (; pfl_i <=  static_cast<uint32_t>(AMDSMI_DEV_PERF_LEVEL_LAST); pfl_i++) {
      if (pfl_i == static_cast<uint32_t>(orig_pfl)) {
        continue;
      }

      IF_VERB(STANDARD) {
        std::cout << "Set Performance Level to " <<
            GetPerfLevelStr(static_cast<amdsmi_dev_perf_level_t>(pfl_i)) <<
                                                            " ..." << std::endl;
      }
      ret =  amdsmi_set_gpu_perf_level(processor_handles_[dv_ind],
                                     static_cast<amdsmi_dev_perf_level_t>(pfl_i));
      if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
          std::cout << "\t**" << GetPerfLevelStr(static_cast<amdsmi_dev_perf_level_t>(pfl_i))
                  << " returned AMDSMI_STATUS_NOT_SUPPORTED"  << std::endl;
      } else {
          CHK_ERR_ASRT(ret)
          ret = amdsmi_get_gpu_perf_level(processor_handles_[dv_ind], &pfl);
          CHK_ERR_ASRT(ret)
          IF_VERB(STANDARD) {
              std::cout << "\t**New Perf Level:" << GetPerfLevelStr(pfl) <<
                                                                    std::endl;
        }
      }
    }
    IF_VERB(STANDARD) {
      std::cout << "Reset Perf level to " << GetPerfLevelStr(orig_pfl) <<
                                                            " ..." << std::endl;
    }
    ret =  amdsmi_set_gpu_perf_level(processor_handles_[dv_ind], orig_pfl);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t** Not supported on this machine" << std::endl;
      }
      continue;
    }
    CHK_ERR_ASRT(ret)
    ret = amdsmi_get_gpu_perf_level(processor_handles_[dv_ind], &pfl);
    CHK_ERR_ASRT(ret)

    IF_VERB(STANDARD) {
      std::cout << "\t**New Perf Level:" << GetPerfLevelStr(pfl) <<
                                                                      std::endl;
    }
  }
}
