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

#include <iostream>
#include <bitset>
#include <string>
#include <algorithm>

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "power_cap_read_write.h"
#include "../test_common.h"


TestPowerCapReadWrite::TestPowerCapReadWrite() : TestBase() {
  set_title("AMDSMI Power Cap Read/Write Test");
  set_description("The Power Cap tests verify that the power profile "
                             "settings can be read and written properly.");
}

TestPowerCapReadWrite::~TestPowerCapReadWrite(void) {
}

void TestPowerCapReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPowerCapReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestPowerCapReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPowerCapReadWrite::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestPowerCapReadWrite::Run(void) {
  amdsmi_status_t ret;
  uint64_t default_cap, min, max, new_cap, curr_cap;
  clock_t start, end;
  double cpu_time_used;
  const uint64_t MICRO_CONVERSION = 1000000;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    amdsmi_power_cap_info_t info;
    ret = amdsmi_get_power_cap_info(processor_handles_[dv_ind], 0, &info);
    CHK_ERR_ASRT(ret)
    // Verify api support checking functionality is working
    ret = amdsmi_get_power_cap_info(processor_handles_[dv_ind], 0, nullptr);
    ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);
    min = info.min_power_cap;
    max = info.max_power_cap;
    default_cap = info.default_power_cap;
    curr_cap = info.power_cap;

    new_cap = (max + min)/2;
    // Check if power cap is within the range
    // skip the test otherwise
    if (new_cap < min || new_cap > max) {
      std::cout << "Power cap requested (" << new_cap
                << " uW) is not within the range. Skipping test for " << dv_ind << std::endl;
      continue;
    }

    IF_VERB(STANDARD) {
      std::cout << "[Before Set]  Default Power Cap: " << default_cap << " uW" << std::endl;
      std::cout << "[Before Set]  Current Power Cap: " << curr_cap << " uW" << std::endl;
      std::cout << "[Before Set]  Power Cap Range [max to min]: " << max << " uW to " << min <<
                                                             " uW" << std::endl;
      std::cout << "[Before Set]  Setting new cap to " << new_cap << "..." << std::endl;
    }
    start = clock();
    ret =  amdsmi_set_power_cap(processor_handles_[dv_ind], 0, new_cap);
    end = clock();
    cpu_time_used = ((double) (end - start)) * 1000000UL / CLOCKS_PER_SEC;

    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t** Not supported on this machine" << std::endl;
      }
      continue;
    }
    CHK_ERR_ASRT(ret)

    ret = amdsmi_get_power_cap_info(processor_handles_[dv_ind], 0, &info);
    CHK_ERR_ASRT(ret)
    curr_cap = info.power_cap;

    IF_VERB(STANDARD) {
      std::cout << "[After Set]   Time spent: " << cpu_time_used << " uS" << std::endl;
      std::cout << "[After Set]   Current Power Cap: " << curr_cap << " uW" << std::endl;
      std::cout << "[After Set]   Requested Power Cap: " << new_cap << " uW" << std::endl;
      std::cout << "[After Set]   Power Cap Range [max to min]: " << max << " uW to "
                                                                  << min << " uW" << std::endl;
      std::cout << "[After Set]   Resetting cap to " << default_cap << "..." << std::endl;
    }
    // Confirm in watts the values are equal
    ASSERT_EQ(curr_cap/MICRO_CONVERSION, new_cap/MICRO_CONVERSION);

    // Reset to default power cap
    ret =  amdsmi_set_power_cap(processor_handles_[dv_ind], 0, default_cap);
    CHK_ERR_ASRT(ret)

    ret = amdsmi_get_power_cap_info(processor_handles_[dv_ind], 0, &info);
    CHK_ERR_ASRT(ret)
    curr_cap = info.power_cap;

    IF_VERB(STANDARD) {
      std::cout << "[After Reset] Current Power Cap: " << curr_cap << " uW" << std::endl;
      std::cout << "[After Reset] Requested Power Cap (default): " << default_cap << " uW"
                                                                   << std::endl;
      std::cout << "[After Reset] Power Cap Range [max to min]: " << max << " uW to "
                                                                  << min << " uW" << std::endl;
    }
    // Confirm in watts the values are equal
    ASSERT_EQ(curr_cap/MICRO_CONVERSION, default_cap/MICRO_CONVERSION);
  }
}
