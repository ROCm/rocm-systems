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
#include "volt_read.h"
#include "../test_common.h"


TestVoltRead::TestVoltRead() : TestBase() {
  set_title("AMDSMI Volt Read Test");
  set_description("The Voltage Read tests verifies that the voltage "
                   "monitors can be read properly.");
}

TestVoltRead::~TestVoltRead(void) {
}

void TestVoltRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestVoltRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestVoltRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestVoltRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestVoltRead::Run(void) {
  amdsmi_status_t err;
  int64_t val_i64;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  amdsmi_voltage_type_t type = AMDSMI_VOLT_TYPE_VDDGFX;

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    auto print_volt_metric = [&](amdsmi_voltage_metric_t met,
                                                        std::string label) {
      err =  amdsmi_get_gpu_volt_metric(processor_handles_[i], type, met, &val_i64);

      if (err != AMDSMI_STATUS_SUCCESS) {
        if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
          IF_VERB(STANDARD) {
            std::cout << "\t**" << label << ": " <<
                               "Not supported on this machine" << std::endl;

            // Verify api support checking functionality is working
            err =  amdsmi_get_gpu_volt_metric(processor_handles_[i], type, met, nullptr);
            ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
            return;
          }
        } else {
          CHK_ERR_ASRT(err)
        }
      }
      // Verify api support checking functionality is working
      err =  amdsmi_get_gpu_volt_metric(processor_handles_[i], type, met, nullptr);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

      IF_VERB(STANDARD) {
        std::cout << "\t**" << label << ": " << val_i64 <<
                                                           "mV" << std::endl;
      }
    };
    for (uint32_t i = AMDSMI_VOLT_TYPE_FIRST; i <= AMDSMI_VOLT_TYPE_LAST; ++i) {
      IF_VERB(STANDARD) {
        std::cout << "\t** **********" <<
          GetVoltSensorNameStr(static_cast<amdsmi_voltage_type_t>(i)) <<
                                         " Voltage **********" << std::endl;
      }
      print_volt_metric(AMDSMI_VOLT_CURRENT, "Current Voltage");
      print_volt_metric(AMDSMI_VOLT_MAX, "Voltage max value");
      print_volt_metric(AMDSMI_VOLT_MIN, "Voltage min value");
      print_volt_metric(AMDSMI_VOLT_MAX_CRIT,
                                "Voltage critical max value");
      print_volt_metric(AMDSMI_VOLT_MIN_CRIT,
                                "Voltage critical min value");
      print_volt_metric(AMDSMI_VOLT_AVERAGE, "Voltage critical max value");
      print_volt_metric(AMDSMI_VOLT_LOWEST, "Historical minimum temperature");
      print_volt_metric(AMDSMI_VOLT_HIGHEST, "Historical maximum temperature");
    }
  }
}
