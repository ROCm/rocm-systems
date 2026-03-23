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
#include "power_cap_read_write.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <string>

#include "../test_common.h"
#include "amd_smi/amdsmi.h"

const uint64_t MICRO_CONVERSION = 1000000;

TestPowerCapReadWrite::TestPowerCapReadWrite() : TestBase() {
  set_title("AMDSMI Power Cap Read/Write Test");
  set_description(
      "The Power Cap tests verify that the power profile "
      "settings can be read and written properly.");
}

TestPowerCapReadWrite::~TestPowerCapReadWrite(void) {}

void TestPowerCapReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPowerCapReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestPowerCapReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPowerCapReadWrite::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestPowerCapReadWrite::SetCheckPowerCap(std::string msg, uint32_t dv_ind, uint32_t sensor_ind,
                                             uint64_t& curr_cap, uint64_t& new_cap,
                                             amdsmi_status_t& ret) {
  amdsmi_status_t ret_expected;
  amdsmi_power_cap_info_t info;
  clock_t start, end;
  double cpu_time_used;

  ret_expected = ret;

  IF_VERB(STANDARD) {
    std::cout << "\t" << msg << std::endl;
    std::cout << "\t[Before Set]  Current Power Cap: " << curr_cap << " uW" << std::endl;
    std::cout << "\t[Before Set]  Setting new cap to " << new_cap << "..." << std::endl;
  }
  DISPLAY_AMDSMI_API("amdsmi_set_power_cap",
                     "gpu=" + std::to_string(dv_ind) + ", cap= " + std::to_string(new_cap),
                     VERB(STANDARD));
  start = clock();
  ret = amdsmi_set_power_cap(processor_handles_[dv_ind], sensor_ind, new_cap);
  end = clock();
  cpu_time_used = (static_cast<double>(end - start)) * 1000000UL / CLOCKS_PER_SEC;
  DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, ret_expected);

  if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
    return;
  }
  ASSERT_EQ(ret, ret_expected);
  if (ret == AMDSMI_STATUS_INVAL) {
    new_cap = curr_cap;
    return;
  }

  DISPLAY_AMDSMI_API("amdsmi_get_power_cap_info", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
  ret = amdsmi_get_power_cap_info(processor_handles_[dv_ind], sensor_ind, &info);
  DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
  CHK_ERR_ASRT(ret)

  curr_cap = info.power_cap;
  // Confirm in watts the values are equal
  ASSERT_EQ(curr_cap / MICRO_CONVERSION, new_cap / MICRO_CONVERSION);

  if (ret_expected == AMDSMI_STATUS_INVAL) {
    new_cap = curr_cap;
  }

  IF_VERB(STANDARD) {
    std::cout << "\t[After Set]   Time spent: " << cpu_time_used << " uS" << std::endl;
    std::cout << "\t[After Set]   Current Power Cap: " << curr_cap << " uW" << std::endl;
    if (ret_expected != AMDSMI_STATUS_INVAL) {
      std::cout << "\t[After Set]   Requested Power Cap: " << new_cap << " uW" << std::endl;
    }
  }

  return;
}

void TestPowerCapReadWrite::Run(void) {
  amdsmi_status_t ret;
  uint64_t orig_cap, default_cap, min_cap, max_cap, new_cap, curr_cap;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    // verify amdsmi_get_supported_power_cap_info() works
    uint32_t sensor_count = 0;
    uint32_t sensor_inds[2];
    amdsmi_power_cap_type_t sensor_types[2];
    DISPLAY_AMDSMI_API("amdsmi_get_supported_power_cap", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_supported_power_cap(processor_handles_[dv_ind], &sensor_count, sensor_inds,
                                         nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

    DISPLAY_AMDSMI_API("amdsmi_get_supported_power_cap", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_supported_power_cap(processor_handles_[dv_ind], &sensor_count, sensor_inds,
                                         sensor_types);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      ASSERT_EQ(ret, AMDSMI_STATUS_NOT_SUPPORTED);
      continue;
    }

    for (uint32_t i = 0; i < sensor_count; ++i) {
      std::cout << "\tPower Cap Sensor Index: " << sensor_inds[i] << ", Type: ppt"
                << (sensor_types[i]) << std::endl;

      amdsmi_power_cap_info_t info;
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_power_cap_info", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_get_power_cap_info(processor_handles_[dv_ind], sensor_inds[i], nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

      DISPLAY_AMDSMI_API("amdsmi_get_power_cap_info", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_get_power_cap_info(processor_handles_[dv_ind], sensor_inds[i], &info);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
        ASSERT_EQ(ret, AMDSMI_STATUS_NOT_SUPPORTED);
        continue;
      }
      min_cap = info.min_power_cap;
      max_cap = info.max_power_cap;
      default_cap = info.default_power_cap;
      curr_cap = info.power_cap;
      orig_cap = curr_cap;

      new_cap = (max_cap + min_cap) / 2;
      IF_VERB(STANDARD) {
        std::cout << "\t[Before Set]  Default Power Cap: " << default_cap << " uW" << std::endl;
        std::cout << "\t[Before Set]  Current Power Cap: " << curr_cap << " uW" << std::endl;
        std::cout << "\t[Before Set]  Power Cap Range [max to min]: " << max_cap << " uW to "
                  << min_cap << " uW" << std::endl;
        std::cout << "\t[Before Set]  Setting new cap to " << new_cap << "..." << std::endl;
      }

      // Check if power cap is within the range
      // skip the test otherwise
      if (new_cap < min_cap || new_cap > max_cap || curr_cap == 0) {
        std::cout << "\t** Requested Power cap (" << new_cap
                  << " uW) cannot be changed for device #" << dv_ind << "."
                  << "\nCurrent Power Cap: " << curr_cap << " uW, Min Power Cap: " << min_cap
                  << " uW, Max Power Cap: " << max_cap
                  << " uW.\n[WARN] If current power cap is 0 uW, this means we cannot change"
                  << " this device's current power cap. Skipping test for this device."
                  << std::endl;
        continue;
      }
      ret = AMDSMI_STATUS_SUCCESS;
      SetCheckPowerCap("Setting to Average Power Cap", dv_ind, sensor_inds[i], curr_cap, new_cap,
                       ret);
      if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
        continue;
      }
      IF_VERB(STANDARD) {
        if (!new_cap)
          std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for "
                    << dv_ind << std::endl;
      }

      if (min_cap > 0) {
        new_cap = min_cap;
        ret = AMDSMI_STATUS_SUCCESS;
        SetCheckPowerCap("Setting to Min Power Cap", dv_ind, sensor_inds[i], curr_cap, new_cap,
                         ret);
        IF_VERB(STANDARD) {
          if (!new_cap)
            std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for "
                      << dv_ind << std::endl;
        }

        new_cap = uint64_t(min_cap - 1);
        ret = AMDSMI_STATUS_INVAL;
        SetCheckPowerCap("Setting to Min Power Cap - 1", dv_ind, sensor_inds[i], curr_cap, new_cap,
                         ret);
        if (ret != AMDSMI_STATUS_INVAL) {
          IF_VERB(STANDARD) {
            if (!new_cap)
              std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for "
                        << dv_ind << std::endl;
          }
        }

        new_cap = uint64_t(static_cast<float>(min_cap) * 0.10F);
        ret = AMDSMI_STATUS_INVAL;
        SetCheckPowerCap("Setting to Min Power Cap * 0.10", dv_ind, sensor_inds[i], curr_cap,
                         new_cap, ret);
        if (ret != AMDSMI_STATUS_INVAL) {
          IF_VERB(STANDARD) {
            if (!new_cap)
              std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for "
                        << dv_ind << std::endl;
          }
        }
      } else {
        std::cout << "\tPower cap requested is less than or equal to 0, skipping test for device #"
                  << dv_ind << std::endl;
      }

      new_cap = max_cap;
      ret = AMDSMI_STATUS_SUCCESS;
      SetCheckPowerCap("Setting to Max Power Cap", dv_ind, sensor_inds[i], curr_cap, new_cap, ret);
      IF_VERB(STANDARD) {
        if (!new_cap)
          std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for "
                    << dv_ind << std::endl;
      }

      new_cap = uint64_t(max_cap + 1);
      ret = AMDSMI_STATUS_INVAL;
      SetCheckPowerCap("Setting to Max Power Cap + 1", dv_ind, sensor_inds[i], curr_cap, new_cap,
                       ret);
      if (ret != AMDSMI_STATUS_INVAL) {
        IF_VERB(STANDARD) {
          if (!new_cap)
            std::cout << "\t** Power cap requested (" << new_cap << " uW) failed to set for "
                      << dv_ind << std::endl;
        }
      }

      new_cap = uint64_t(max_cap * 10);
      ret = AMDSMI_STATUS_INVAL;
      SetCheckPowerCap("Setting to Max Power Cap * 10", dv_ind, sensor_inds[i], curr_cap, new_cap,
                       ret);
      if (ret != AMDSMI_STATUS_INVAL) {
        IF_VERB(STANDARD) {
          if (!new_cap)
            std::cout << "\t** Power cap requested (" << new_cap << " uW) is failed to set for "
                      << dv_ind << std::endl;
        }
      }

      // Reset to default power cap -> which is typically the same as the max power cap
      IF_VERB(STANDARD) {
        std::cout << "\tSetting to default power Cap" << std::endl;
        std::cout << "\t[Before Set] Current Power Cap: " << curr_cap << " uW" << std::endl;
        std::cout << "\t[Before Set] Default Power Cap (default_cap): " << default_cap << "..."
                  << std::endl;
      }
      DISPLAY_AMDSMI_API("amdsmi_set_power_cap", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
      ret = amdsmi_set_power_cap(processor_handles_[dv_ind], sensor_inds[i], default_cap);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)

      DISPLAY_AMDSMI_API("amdsmi_get_power_cap_info", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_get_power_cap_info(processor_handles_[dv_ind], sensor_inds[i], &info);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)
      curr_cap = info.power_cap;

      IF_VERB(STANDARD) {
        std::cout << "\t[After Set] Current Power Cap: " << curr_cap << " uW" << std::endl;
        std::cout << "\t[After Set] Requested Power Cap (default_cap): " << default_cap << " uW"
                  << std::endl;
        std::cout << "\t[After Set] Power Cap Range [max to min]: " << max_cap << " uW to "
                  << min_cap << " uW" << std::endl;
      }
      // Confirm in watts the values are equal
      ASSERT_EQ(default_cap / MICRO_CONVERSION, curr_cap / MICRO_CONVERSION);

      // Reset to system's original power cap before the test started
      IF_VERB(STANDARD) {
        std::cout << "\tResetting Power Cap to original power cap" << std::endl;
        std::cout << "\t[Before Reset] Current Power Cap: " << curr_cap << " uW" << std::endl;
        std::cout << "\t[Before Reset] Original Power Cap (orig_cap): " << orig_cap << "..."
                  << std::endl;
      }
      DISPLAY_AMDSMI_API("amdsmi_set_power_cap", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
      ret = amdsmi_set_power_cap(processor_handles_[dv_ind], sensor_inds[i], orig_cap);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)

      DISPLAY_AMDSMI_API("amdsmi_get_power_cap_info", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_get_power_cap_info(processor_handles_[dv_ind], sensor_inds[i], &info);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)
      curr_cap = info.power_cap;

      IF_VERB(STANDARD) {
        std::cout << "\t[After Reset] Current Power Cap: " << curr_cap << " uW" << std::endl;
        std::cout << "\t[After Reset] Requested Power Cap (orig_cap): " << orig_cap << " uW"
                  << std::endl;
        std::cout << "\t[After Reset] Power Cap Range [max to min]: " << max_cap << " uW to "
                  << min_cap << " uW" << std::endl;
      }

      // Confirm in watts the values are equal
      ASSERT_EQ(orig_cap / MICRO_CONVERSION, curr_cap / MICRO_CONVERSION);
    }
  }
}
