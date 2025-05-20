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
#include "rocm_smi_test/functional/temp_read.h"
#include "rocm_smi_test/test_common.h"


static const std::map<uint32_t, std::string> kTempSensorNameMap = {
    {RSMI_TEMP_TYPE_MEMORY, "Memory"},
    {RSMI_TEMP_TYPE_JUNCTION, "Junction"},
    {RSMI_TEMP_TYPE_EDGE, "Edge"},
    {RSMI_TEMP_TYPE_HBM_0, "HBM_0"},
    {RSMI_TEMP_TYPE_HBM_1, "HBM_1"},
    {RSMI_TEMP_TYPE_HBM_2, "HBM_2"},
    {RSMI_TEMP_TYPE_HBM_3, "HBM_3"},
};
TestTempRead::TestTempRead() : TestBase() {
  set_title("RSMI Temp Read Test");
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
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestTempRead::Run(void) {
  rsmi_status_t err;
  int64_t val_i64;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  uint32_t type(0);
  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(i);

      auto print_temp_metric = [&](rsmi_temperature_metric_t met,
                                                          std::string label) {
        err = rsmi_dev_temp_metric_get(i, type, met, &val_i64);

        if (err != RSMI_STATUS_SUCCESS) {
          if (err == RSMI_STATUS_NOT_SUPPORTED) {
            IF_VERB(STANDARD) {
              std::cout << "\t**" << label << ": " <<
                                 "Not supported on this machine" << std::endl;
            }

            // Verify api support checking functionality is working
            err = rsmi_dev_temp_metric_get(i, type, met, nullptr);
            ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
            return;
          } else {
            CHK_ERR_ASRT(err)
          }
        }
        // Verify api support checking functionality is working
        err = rsmi_dev_temp_metric_get(i, type, met, nullptr);
        ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

        IF_VERB(STANDARD) {
          std::cout << "\t**" << label << ": " << val_i64/1000 <<
                                                             "C" << std::endl;
        }
      };
      for (type = RSMI_TEMP_TYPE_FIRST; type <= RSMI_TEMP_TYPE_LAST; ++type) {
        IF_VERB(STANDARD) {
          std::cout << "\t** **********" << kTempSensorNameMap.at(type) <<
                                        " Temperatures **********" << std::endl;
        }
        print_temp_metric(RSMI_TEMP_CURRENT, "Current Temp.");
        print_temp_metric(RSMI_TEMP_MAX, "Temperature max value");
        print_temp_metric(RSMI_TEMP_MIN, "Temperature min value");
        print_temp_metric(RSMI_TEMP_MAX_HYST,
                                  "Temperature hysteresis value for max limit");
        print_temp_metric(RSMI_TEMP_MIN_HYST,
                                  "Temperature hysteresis value for min limit");
        print_temp_metric(RSMI_TEMP_CRITICAL, "Temperature critical max value");
        print_temp_metric(RSMI_TEMP_CRITICAL_HYST,
                             "Temperature hysteresis value for critical limit");
        print_temp_metric(RSMI_TEMP_EMERGENCY,
                                             "Temperature emergency max value");
        print_temp_metric(RSMI_TEMP_EMERGENCY_HYST,
                            "Temperature hysteresis value for emergency limit");
        print_temp_metric(RSMI_TEMP_CRIT_MIN, "Temperature critical min value");
        print_temp_metric(RSMI_TEMP_CRIT_MIN_HYST,
                         "Temperature hysteresis value for critical min value");
        print_temp_metric(RSMI_TEMP_OFFSET, "Temperature offset");
        print_temp_metric(RSMI_TEMP_LOWEST, "Historical minimum temperature");
        print_temp_metric(RSMI_TEMP_HIGHEST, "Historical maximum temperature");
      }
    }
  }  // x
}
