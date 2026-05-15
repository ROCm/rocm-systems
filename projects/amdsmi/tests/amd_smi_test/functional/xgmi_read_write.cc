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
#include "xgmi_read_write.h"

TestXGMIReadWrite::TestXGMIReadWrite() : TestBase() {
  set_title("AMDSMI XGMI Read/Write Test");
  set_description("This test verifies that XGMI error counts can be read"
                               " properly, and that the count can be reset.");
}

TestXGMIReadWrite::~TestXGMIReadWrite(void) {
}

void TestXGMIReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestXGMIReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestXGMIReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestXGMIReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestXGMIReadWrite::Run(void) {
  GTEST_SKIP_("Temporarily disabled");
  amdsmi_status_t err;
  amdsmi_xgmi_status_t err_stat;

  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) {
      std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    }
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    auto device = processor_handles_[dv_ind];
    PrintDeviceHeader(device);

    amdsmi_xgmi_info_t info;
    err = amdsmi_get_xgmi_info(device, &info);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        std::cout <<
            "\t**amdsmi_dev_xgmi_hive_id_get() is not supported"
            " on this machine" << std::endl;
        continue;
    } else {
        CHK_ERR_ASRT(err)
        IF_VERB(STANDARD) {
            std::cout << "\t**XGMI Hive ID : " << std::hex <<
            info.xgmi_hive_id << std::endl;
        }
    }

    err = amdsmi_gpu_xgmi_error_status(device, &err_stat);

    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**XGMI Error Status: Not supported on this machine"
                                                               << std::endl;
      }
      // Verify api support checking functionality is working
      err = amdsmi_gpu_xgmi_error_status(device, nullptr);
      ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);

      continue;
    }
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**XGMI Error Status: " <<
                               static_cast<uint32_t>(err_stat) << std::endl;
    }
    // Verify api support checking functionality is working
    err = amdsmi_gpu_xgmi_error_status(device, nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    // TODO(cfree) We need to find a way to generate xgmi errors so this
    // test won't be meaningless
    err = amdsmi_reset_gpu_xgmi_error(device);
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**Successfully reset XGMI Error Status: " << std::endl;
    }
  }
}
