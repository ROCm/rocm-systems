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
#include "gpu_busy_read.h"
#include "../test_common.h"

TestGPUBusyRead::TestGPUBusyRead() : TestBase() {
  set_title("AMDSMI GPU Busy Read Test");
  set_description("The GPU Busy Read tests verifies that the gpu busy "
                   "percentage can be read properly.");
}

TestGPUBusyRead::~TestGPUBusyRead(void) {
}

void TestGPUBusyRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestGPUBusyRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestGPUBusyRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestGPUBusyRead::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestGPUBusyRead::Run(void) {
  amdsmi_status_t err;
  uint32_t val_ui32;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(processor_handles_[i]);

      err = amdsmi_get_gpu_busy_percent(processor_handles_[i], &val_ui32);
      if (err != AMDSMI_STATUS_SUCCESS) {
        if (err == AMDSMI_STATUS_FILE_ERROR || err == AMDSMI_STATUS_NOT_SUPPORTED) {
          IF_VERB(STANDARD) {
            std::cout << "\t**GPU Busy Percent: Not supported on this machine"
                                                                 << std::endl;
          }
          ASSERT_TRUE(err == AMDSMI_STATUS_FILE_ERROR || err == AMDSMI_STATUS_NOT_SUPPORTED);
        } else {
          CHK_ERR_ASRT(err)
        }
      } else {
        IF_VERB(STANDARD) {
          std::cout << "\t**GPU Busy Percent (Percent Idle):" << std::dec <<
                       val_ui32 << " (" << 100 - val_ui32 << ")" << std::endl;
        }
      }
    }
  }
}
