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
#include <string>
#include <map>

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "temp_read.h"
#include "../test_common.h"


static const std::map<uint32_t, std::string> kTempSensorNameMap = {
    {AMDSMI_TEMPERATURE_TYPE_VRAM, "Memory"},
    {AMDSMI_TEMPERATURE_TYPE_HOTSPOT, "Hotspot"},
    {AMDSMI_TEMPERATURE_TYPE_JUNCTION, "Junction"},
    {AMDSMI_TEMPERATURE_TYPE_EDGE, "Edge"},
    {AMDSMI_TEMPERATURE_TYPE_HBM_0, "HBM_0"},
    {AMDSMI_TEMPERATURE_TYPE_HBM_1, "HBM_1"},
    {AMDSMI_TEMPERATURE_TYPE_HBM_2, "HBM_2"},
    {AMDSMI_TEMPERATURE_TYPE_HBM_3, "HBM_3"},
    {AMDSMI_TEMPERATURE_TYPE_PLX, "PLX"}
};
TestTempRead::TestTempRead() : TestBase() {
  set_title("AMDSMI Temp Read Test");
  set_description("The Temperature Read tests verifies that the temperature "
                   "monitors can be read properly.");
}

TestTempRead::~TestTempRead(void) {
}

void TestTempRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestTempRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestTempRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestTempRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestTempRead::Run(void) {
  amdsmi_status_t err;
  int64_t val_i64;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  uint32_t type(0);
  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(processor_handles_[i]);

      auto print_temp_metric = [&](amdsmi_temperature_metric_t met,
                                                          std::string label) {
        err =  amdsmi_get_temp_metric(processor_handles_[i], static_cast<amdsmi_temperature_type_t>(type), met, &val_i64);

        if (err != AMDSMI_STATUS_SUCCESS) {
          if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
            IF_VERB(STANDARD) {
              std::cout << "\t**" << label << ": " <<
                                 "Not supported on this machine" << std::endl;
            }

            // Verify api support checking functionality is working
            err =  amdsmi_get_temp_metric(processor_handles_[i],  static_cast<amdsmi_temperature_type_t>(type), met, nullptr);
            ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
            return;
          } else {
            CHK_ERR_ASRT(err)
          }
        }
        // Verify api support checking functionality is working
        err =  amdsmi_get_temp_metric(processor_handles_[i],  static_cast<amdsmi_temperature_type_t>(type), met, nullptr);
        ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

        IF_VERB(STANDARD) {
          std::cout << "\t**" << label << ": " << val_i64 << "C" << std::endl;
        }
      };
      for (type = AMDSMI_TEMPERATURE_TYPE_FIRST; type <= AMDSMI_TEMPERATURE_TYPE__MAX; ++type) {
        IF_VERB(STANDARD) {
          std::cout << "\t** **********" << kTempSensorNameMap.at(type) <<
                                        " Temperatures **********" << std::endl;
        }
        print_temp_metric(AMDSMI_TEMP_CURRENT, "Current Temp.");
        print_temp_metric(AMDSMI_TEMP_MAX, "Temperature max value");
        print_temp_metric(AMDSMI_TEMP_MIN, "Temperature min value");
        print_temp_metric(AMDSMI_TEMP_MAX_HYST,
                                  "Temperature hysteresis value for max limit");
        print_temp_metric(AMDSMI_TEMP_MIN_HYST,
                                  "Temperature hysteresis value for min limit");
        print_temp_metric(AMDSMI_TEMP_CRITICAL, "Temperature critical max value");
        print_temp_metric(AMDSMI_TEMP_CRITICAL_HYST,
                             "Temperature hysteresis value for critical limit");
        print_temp_metric(AMDSMI_TEMP_EMERGENCY,
                                             "Temperature emergency max value");
        print_temp_metric(AMDSMI_TEMP_EMERGENCY_HYST,
                            "Temperature hysteresis value for emergency limit");
        print_temp_metric(AMDSMI_TEMP_CRIT_MIN, "Temperature critical min value");
        print_temp_metric(AMDSMI_TEMP_CRIT_MIN_HYST,
                         "Temperature hysteresis value for critical min value");
        print_temp_metric(AMDSMI_TEMP_OFFSET, "Temperature offset");
        print_temp_metric(AMDSMI_TEMP_LOWEST, "Historical minimum temperature");
        print_temp_metric(AMDSMI_TEMP_HIGHEST, "Historical maximum temperature");
      }
    }
  }  // x
}
