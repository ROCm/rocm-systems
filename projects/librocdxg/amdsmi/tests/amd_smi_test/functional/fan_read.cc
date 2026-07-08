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

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "fan_read.h"

TestFanRead::TestFanRead() : TestBase() {
  set_title("AMDSMI Fan Read Test");
  set_description("The Fan Read tests verifies that the fan monitors can be "
                  "read properly.");
}

TestFanRead::~TestFanRead(void) {
}

void TestFanRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestFanRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestFanRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestFanRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestFanRead::Run(void) {
  uint64_t val_ui64;
  amdsmi_status_t err;
  int64_t val_i64;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }
  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(processor_handles_[i]);

      IF_VERB(STANDARD) {
        std::cout << "\t**Current Fan Speed: ";
      }
      err = amdsmi_get_gpu_fan_speed(processor_handles_[i], 0, &val_i64);
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
          IF_VERB(STANDARD) {
            std::cout << "\t**" <<  ": " <<
                               "Not supported on this machine" << std::endl;
          }
          // Verify api support checking functionality is working
          err = amdsmi_get_gpu_fan_speed(processor_handles_[i], 0, nullptr);
          ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
          return;
      } else {
        CHK_ERR_ASRT(err)
      }


      // Verify api support checking functionality is working
      err = amdsmi_get_gpu_fan_speed(processor_handles_[i], 0, nullptr);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

      err = amdsmi_get_gpu_fan_speed_max(processor_handles_[i], 0, &val_ui64);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << static_cast<float>(val_i64)/static_cast<float>(val_ui64)*100;
        std::cout << "% ("<< val_i64 << "/" << val_ui64 << ")" << std::endl;
      }
      // Verify api support checking functionality is working
      err = amdsmi_get_gpu_fan_speed_max(processor_handles_[i], 0, nullptr);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

      IF_VERB(STANDARD) {
        std::cout << "\t**Current fan RPMs: ";
      }
      err = amdsmi_get_gpu_fan_rpms(processor_handles_[i], 0, &val_i64);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << val_i64 << std::endl;
      }

      // Verify api support checking functionality is working
      err = amdsmi_get_gpu_fan_rpms(processor_handles_[i], 0, nullptr);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
    }
  }
}
