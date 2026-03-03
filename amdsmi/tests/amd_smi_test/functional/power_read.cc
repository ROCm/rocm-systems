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
#include <string>

#include "amd_smi/amdsmi.h"
#include "power_read.h"

TestPowerRead::TestPowerRead() : TestBase() {
  set_title("AMDSMI Power Read Test");
  set_description("The Power Read tests verifies that "
                                "power related values can be read properly.");
}

TestPowerRead::~TestPowerRead(void) {
}

void TestPowerRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPowerRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestPowerRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPowerRead::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestPowerRead::Run(void) {
  amdsmi_status_t err;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(processor_handles_[i]);

      amdsmi_power_cap_info_t info;
      err = amdsmi_get_power_cap_info(processor_handles_[i], 0, &info);
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        std::cout << "\t**Power Cap not supported on this device." << std::endl;
        ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
        continue;
      }
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Current Power Cap: " << info.power_cap << "uW" <<std::endl;
      }

      IF_VERB(STANDARD) {
        std::cout << "\t**Default Power Cap: " << info.default_power_cap << "uW" <<std::endl;
        std::cout << "\t**Power Cap Range: " << info.min_power_cap << " to " <<
                                                 info.max_power_cap << " uW" << std::endl;
      }
      // TODO(amdsmi_team): Add current_socket_power tests
    }
  }
}
