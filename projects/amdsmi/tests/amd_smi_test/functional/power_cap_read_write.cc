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

const uint64_t MICRO_CONVERSION = 1000000;


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

void TestPowerCapReadWrite::SetCheckPowerCap(std::string msg, uint32_t dv_ind, uint64_t &curr_cap,
                                             uint64_t &new_cap, amdsmi_status_t &ret) {
    amdsmi_status_t ret_expected;
    amdsmi_power_cap_info_t info;
    clock_t start, end;
    double cpu_time_used;

    ret_expected = ret;

    IF_VERB(STANDARD) {
      std::cout << msg << std::endl;
      std::cout << "[Before Set]  Current Power Cap: " << curr_cap << " uW" << std::endl;
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
      return;
    }
    ASSERT_EQ(ret, ret_expected);
    if (ret == AMDSMI_STATUS_INVAL) {
        new_cap = curr_cap;
        std::cout << "\t** Expected invalid result" << std::endl;
        return;
    }

    ret = amdsmi_get_power_cap_info(processor_handles_[dv_ind], 0, &info);
    CHK_ERR_ASRT(ret)

    curr_cap = info.power_cap;
    // Confirm in watts the values are equal
    ASSERT_EQ(curr_cap/MICRO_CONVERSION, new_cap/MICRO_CONVERSION);

    if (ret_expected == AMDSMI_STATUS_INVAL) {
      new_cap = curr_cap;
    }

    IF_VERB(STANDARD) {
      std::cout << "[After Set]   Time spent: " << cpu_time_used << " uS" << std::endl;
      std::cout << "[After Set]   Current Power Cap: " << curr_cap << " uW" << std::endl;
      if (ret_expected != AMDSMI_STATUS_INVAL) {
        std::cout << "[After Set]   Requested Power Cap: " << new_cap << " uW" << std::endl;
      }
    }

    return;
}

void TestPowerCapReadWrite::Run(void) {
  amdsmi_status_t ret;
  uint64_t default_cap, min_cap, max_cap, new_cap, curr_cap;

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
    min_cap = info.min_power_cap;
    max_cap = info.max_power_cap;
    default_cap = info.default_power_cap;
    curr_cap = info.power_cap;

    new_cap = (max_cap + min_cap)/2;
    IF_VERB(STANDARD) {
      std::cout << "[Before Set]  Default Power Cap: " << default_cap << " uW" << std::endl;
      std::cout << "[Before Set]  Current Power Cap: " << curr_cap << " uW" << std::endl;
      std::cout << "[Before Set]  Power Cap Range [max to min]: " << max_cap << " uW to " << min_cap <<
                                                            " uW" << std::endl;
      std::cout << "[Before Set]  Setting new cap to " << new_cap << "..." << std::endl;
    }

    // Check if power cap is within the range
    // skip the test otherwise
    if (new_cap < min_cap || new_cap > max_cap) {
      std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for " << dv_ind << std::endl;
      continue;
    }
    ret = AMDSMI_STATUS_SUCCESS;
    SetCheckPowerCap("Setting to Average Power Cap", dv_ind, curr_cap, new_cap, ret);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }
    IF_VERB(STANDARD) {
      if (!new_cap)
        std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for " << dv_ind << std::endl;
    }

    if (min_cap > 0)
    {
      new_cap = min_cap;
      ret = AMDSMI_STATUS_SUCCESS;
      SetCheckPowerCap("Setting to Min Power Cap", dv_ind, curr_cap, new_cap, ret);
      IF_VERB(STANDARD) {
        if (!new_cap)
          std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for " << dv_ind << std::endl;
      }

      new_cap = uint64_t(min_cap - 1);
      ret = AMDSMI_STATUS_INVAL;
      SetCheckPowerCap("Setting to Min Power Cap - 1", dv_ind, curr_cap, new_cap, ret);
      if (ret != AMDSMI_STATUS_INVAL) {
        IF_VERB(STANDARD) {
          if (!new_cap)
            std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for " << dv_ind << std::endl;
        }
      }

      new_cap = uint64_t(static_cast<float>(min_cap) * 0.10F);
      ret = AMDSMI_STATUS_INVAL;
      SetCheckPowerCap("Setting to Min Power Cap * 0.10", dv_ind, curr_cap, new_cap, ret);
      if (ret != AMDSMI_STATUS_INVAL) {
        IF_VERB(STANDARD) {
          if (!new_cap)
            std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for " << dv_ind << std::endl;
        }
      }
    }
    else
    {
      std::cout << "\tPower cap requested is less than or equal to 0, skipping test for " << dv_ind << std::endl;
    }

    new_cap = max_cap;
    ret = AMDSMI_STATUS_SUCCESS;
    SetCheckPowerCap("Setting to Max Power Cap", dv_ind, curr_cap, new_cap, ret);
    IF_VERB(STANDARD) {
      if (!new_cap)
        std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for " << dv_ind << std::endl;
    }

    new_cap = uint64_t(max_cap + 1);
    ret = AMDSMI_STATUS_INVAL;
    SetCheckPowerCap("Setting to Max Power Cap + 1", dv_ind, curr_cap, new_cap, ret);
    if (ret != AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        if (!new_cap)
          std::cout << "\t** Power cap requested (" << new_cap << " uW) failed to set for " << dv_ind << std::endl;
      }
    }

    new_cap = uint64_t(max_cap * 10);
    ret = AMDSMI_STATUS_INVAL;
    SetCheckPowerCap("Setting to Max Power Cap * 10", dv_ind, curr_cap, new_cap, ret);
    if (ret != AMDSMI_STATUS_INVAL) {
      IF_VERB(STANDARD) {
        if (!new_cap)
          std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for " << dv_ind << std::endl;
      }
    }

    // Reset to default power cap
    IF_VERB(STANDARD) {
      std::cout << "Resetting Power Cap" << std::endl;
      std::cout << "[Before Reset] Current Power Cap: " << curr_cap << " uW" << std::endl;
      std::cout << "[Before Reset] Resetting cap to default " << default_cap << "..." << std::endl;
    }
    ret =  amdsmi_set_power_cap(processor_handles_[dv_ind], 0, default_cap);
    CHK_ERR_ASRT(ret)

    ret = amdsmi_get_power_cap_info(processor_handles_[dv_ind], 0, &info);
    CHK_ERR_ASRT(ret)
    curr_cap = info.power_cap;

    IF_VERB(STANDARD) {
      std::cout << "[After Reset] Current Power Cap: " << curr_cap << " uW" << std::endl;
      std::cout << "[After Reset] Requested Power Cap (default): " << default_cap << " uW"
                                                                   << std::endl;
      std::cout << "[After Reset] Power Cap Range [max to min]: " << max_cap << " uW to "
                                                                  << min_cap << " uW" << std::endl;
    }

    // Confirm in watts the values are equal
    ASSERT_EQ(curr_cap/MICRO_CONVERSION, default_cap/MICRO_CONVERSION);
  }
}
