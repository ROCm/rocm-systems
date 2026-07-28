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
#include "device_cuid_read.h"

#include <gtest/gtest.h>

#include <iostream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestDeviceCuidRead::TestDeviceCuidRead() : TestBase() {
  set_title("AMDSMI Device CUID Read Test");
  set_description(
      "This test verifies that the device CUID can be read via amdsmi_get_gpu_device_cuid().");
}

TestDeviceCuidRead::~TestDeviceCuidRead(void) {}

void TestDeviceCuidRead::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestDeviceCuidRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestDeviceCuidRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestDeviceCuidRead::Close() { TestBase::Close(); }

void TestDeviceCuidRead::Run(void) {
  amdsmi_status_t err;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    IF_VERB(STANDARD) {
      std::cout << "\t*************************" << std::endl;
      std::cout << "\t**Device index: " << i << std::endl;
    }

    // Verify null cuid_length is rejected.
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_cuid",
                       "gpu=" + std::to_string(i) + ", cuid_length=null", VERB(STANDARD));
    char cuid_buf[AMDSMI_GPU_CUID_SIZE];
    err = amdsmi_get_gpu_device_cuid(processor_handles_[i], nullptr, cuid_buf);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    // Verify null cuid buffer is rejected.
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_cuid", "gpu=" + std::to_string(i) + ", cuid=null",
                       VERB(STANDARD));
    unsigned int len = AMDSMI_GPU_CUID_SIZE;
    err = amdsmi_get_gpu_device_cuid(processor_handles_[i], &len, nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    // Verify undersized buffer is rejected.
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_cuid",
                       "gpu=" + std::to_string(i) + ", cuid_length too small", VERB(STANDARD));
    unsigned int short_len = AMDSMI_GPU_CUID_SIZE - 1;
    err = amdsmi_get_gpu_device_cuid(processor_handles_[i], &short_len, cuid_buf);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    // Happy path: either a valid CUID is returned, or the feature is unsupported.
    len = AMDSMI_GPU_CUID_SIZE;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_cuid", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_device_cuid(processor_handles_[i], &len, cuid_buf);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) { std::cout << "\t**CUID not supported on this device." << std::endl; }
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Device CUID: " << std::string(cuid_buf, len) << std::endl;
      }
      ASSERT_GT(len, 0u);
      ASSERT_LT(len, static_cast<unsigned int>(AMDSMI_GPU_CUID_SIZE));
    }
  }
}
