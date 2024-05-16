/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2019, Advanced Micro Devices, Inc.
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

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/fan_read.h"
#include "rocm_smi_test/test_common.h"

TestFanRead::TestFanRead() : TestBase() {
  set_title("RSMI Fan Read Test");
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
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestFanRead::Run(void) {
  uint64_t val_ui64;
  rsmi_status_t err;
  int64_t val_i64;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }
  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(i);

      IF_VERB(STANDARD) {
        std::cout << "\t**Current Fan Speed: ";
      }
      err = rsmi_dev_fan_speed_get(i, 0, &val_i64);
      if (err == RSMI_STATUS_NOT_SUPPORTED) {
          IF_VERB(STANDARD) {
            std::cout << "\t**" <<  ": " <<
                               "Not supported on this machine" << std::endl;
          }
          return;
      } else {
        CHK_ERR_ASRT(err)
      }


      // Verify api support checking functionality is working
      err = rsmi_dev_fan_speed_get(i, 0, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

      err = rsmi_dev_fan_speed_max_get(i, 0, &val_ui64);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << val_i64/static_cast<float>(val_ui64)*100;
        std::cout << "% ("<< val_i64 << "/" << val_ui64 << ")" << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_fan_speed_max_get(i, 0, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

      IF_VERB(STANDARD) {
        std::cout << "\t**Current fan RPMs: ";
      }
      err = rsmi_dev_fan_rpms_get(i, 0, &val_i64);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << val_i64 << std::endl;
      }

      // Verify api support checking functionality is working
      err = rsmi_dev_fan_rpms_get(i, 0, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }
  }
}
