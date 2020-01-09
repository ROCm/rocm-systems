/*
Copyright (c) 2020 - Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdint.h>
#include <stddef.h>

#include <iostream>

#include "gtest/gtest.h"
#include "rdc/rdc_client.h"
#include "rdc_tests/functional/fan_read.h"
#include "rdc_tests/test_common.h"

TestFanRead::TestFanRead() : TestBase() {
  set_title("RDC Fan Read Test");
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
  TestBase::Close();
}


void TestFanRead::Run(void) {
  uint64_t val_ui64;
  rdc_status_t err;
  int64_t val_i64;

  TestBase::Run();

  err = AllocateRDCChannel();
  ASSERT_EQ(err, RDC_STATUS_SUCCESS);

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(i);

    IF_VERB(STANDARD) {
      std::cout << "\t**Current Fan Speed: ";
    }
    err = rdc_dev_fan_speed_get(rdc_channel(), i, 0, &val_i64);
    CHK_ERR_ASRT(err)

    // Verify api support checking functionality is working
    err = rdc_dev_fan_speed_get(rdc_channel(), i, 0, nullptr);
    ASSERT_EQ(err, RDC_RSMI_STATUS_INVALID_ARGS);

    err = rdc_dev_fan_speed_max_get(rdc_channel(), i, 0, &val_ui64);
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << val_i64/static_cast<float>(val_ui64)*100;
      std::cout << "% ("<< val_i64 << "/" << val_ui64 << ")" << std::endl;
    }
    // Verify api support checking functionality is working
    err = rdc_dev_fan_speed_max_get(rdc_channel(), i, 0, nullptr);
    ASSERT_EQ(err, RDC_RSMI_STATUS_INVALID_ARGS);

    IF_VERB(STANDARD) {
      std::cout << "\t**Current fan RPMs: ";
    }
    err = rdc_dev_fan_rpms_get(rdc_channel(), i, 0, &val_i64);
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << val_i64 << std::endl;
    }

    // Verify api support checking functionality is working
    err = rdc_dev_fan_rpms_get(rdc_channel(), i, 0, nullptr);
    ASSERT_EQ(err, RDC_RSMI_STATUS_INVALID_ARGS);
  }
}
