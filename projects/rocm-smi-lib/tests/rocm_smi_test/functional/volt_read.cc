/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
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

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/volt_read.h"
#include "rocm_smi_test/test_common.h"


TestVoltRead::TestVoltRead() : TestBase() {
  set_title("RSMI Volt Read Test");
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
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestVoltRead::Run(void) {
  rsmi_status_t err;
  int64_t val_i64;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  rsmi_voltage_type_t type = RSMI_VOLT_TYPE_VDDGFX;

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    IF_VERB(STANDARD) {
      if (i != 0) {
        std::cout << "\n" << std::endl;
      }
    }
    PrintDeviceHeader(i);

    auto print_volt_metric = [&](rsmi_voltage_metric_t met,
                                                        std::string label) {
      err = rsmi_dev_volt_metric_get(i, type, met, &val_i64);

      if (err != RSMI_STATUS_SUCCESS) {
        if (err == RSMI_STATUS_NOT_SUPPORTED) {
          IF_VERB(STANDARD) {
            std::cout << "\t**" << label << ": " <<
                               "Not supported on this machine" << std::endl;
          }

            // Verify api support checking functionality is working
            err = rsmi_dev_volt_metric_get(i, type, met, nullptr);
            ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
            return;
        } else {
          CHK_ERR_ASRT(err)
        }
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_volt_metric_get(i, type, met, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

      IF_VERB(STANDARD) {
        std::cout << "\t**" << label << ": " << val_i64 <<
                                                           "mV" << std::endl;
      }
    };
    for (uint32_t i = RSMI_VOLT_TYPE_FIRST; i <= RSMI_VOLT_TYPE_LAST; ++i) {
      IF_VERB(STANDARD) {
        std::cout << "\t** **********" <<
          GetVoltSensorNameStr(static_cast<rsmi_voltage_type_t>(i)) <<
                                         " Voltage **********" << std::endl;
      }
      print_volt_metric(RSMI_VOLT_CURRENT, "Current Voltage");
      print_volt_metric(RSMI_VOLT_MAX, "Voltage max value");
      print_volt_metric(RSMI_VOLT_MIN, "Voltage min value");
      print_volt_metric(RSMI_VOLT_MAX_CRIT,
                                "Voltage critical max value");
      print_volt_metric(RSMI_VOLT_MIN_CRIT,
                                "Voltage critical min value");
      print_volt_metric(RSMI_VOLT_AVERAGE, "Voltage critical max value");
      print_volt_metric(RSMI_VOLT_LOWEST, "Historical minimum voltage");
      print_volt_metric(RSMI_VOLT_HIGHEST, "Historical maximum voltage");
    }
  }
}
