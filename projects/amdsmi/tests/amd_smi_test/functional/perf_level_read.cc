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
#include "perf_level_read.h"

TestPerfLevelRead::TestPerfLevelRead() : TestBase() {
  set_title("AMDSMI Performance Level Read Test");
  set_description("The Performance Level Read tests verifies that the "
                          "performance level monitors can be read properly.");
}

TestPerfLevelRead::~TestPerfLevelRead(void) {
}

void TestPerfLevelRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPerfLevelRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestPerfLevelRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPerfLevelRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestPerfLevelRead::Run(void) {
  amdsmi_status_t err;
  amdsmi_dev_perf_level_t pfl;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    err = amdsmi_get_gpu_perf_level(processor_handles_[i], &pfl);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Performance Level: Not Supported" << std::endl;
      ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Performance Level:" << std::dec << (uint32_t)pfl
                  << std::endl;
      }
    }
    // Verify api support checking functionality is working
    err = amdsmi_get_gpu_perf_level(processor_handles_[i], nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
  }
}
