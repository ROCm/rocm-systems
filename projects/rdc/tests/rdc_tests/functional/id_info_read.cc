/*
Copyright (c) 2019 - Advanced Micro Devices, Inc. All rights reserved.

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
#include <string>

#include "gtest/gtest.h"
#include "rdc/rdc_client.h"
#include "rocm_smi/rocm_smi.h"
#include "rdc_tests/functional/id_info_read.h"
#include "rdc_tests/test_common.h"

TestIdInfoRead::TestIdInfoRead() : TestBase() {
  set_title("RDC ID Info Read Test");
  set_description("This test verifies that ID information such as the "
             "device, subsystem and vendor IDs can be read properly.");
}

TestIdInfoRead::~TestIdInfoRead(void) {
}

void TestIdInfoRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestIdInfoRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestIdInfoRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestIdInfoRead::Close() {
  // This will close handles opened within RDC utility calls
  TestBase::Close();
}

static const uint32_t kBufferLen = 80;

void TestIdInfoRead::Run(void) {
  rdc_status_t err;
//  uint16_t id;
//  uint64_t val_ui64;
//  uint32_t drm_render_minor;
//  char buffer[kBufferLen];

  TestBase::Run();
  err = AllocateRDCChannel();
  ASSERT_EQ(err, RDC_STATUS_SUCCESS);

  // This test doesn't do much at all yet. Add RDC ID type tests below as we
  // add those capabilities. The following commented block is left as a
  // template.

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
//    IF_VERB(STANDARD) {
//      std::cout << "\t**Device index: " << id << std::endl;
//    }


//    // Get the device ID, name, vendor ID and vendor name for the device
//    err = rsmi_dev_id_get(i, &id);
//    if (err == RSMI_STATUS_NOT_SUPPORTED) {
//      rdc_status_t ret;
//      // Verify api support checking functionality is working
//      ret = rsmi_dev_id_get(i, nullptr);
//      ASSERT_EQ(ret, RSMI_STATUS_NOT_SUPPORTED);
//    } else {
//      CHK_ERR_ASRT(err)
//
//      IF_VERB(STANDARD) {
//        std::cout << "\t**Device ID: 0x" << std::hex << id << std::endl;
//      }
//      // Verify api support checking functionality is working
//      err = rsmi_dev_id_get(i, nullptr);
//      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
//    }
  }
}
