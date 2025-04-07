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
#include <string>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/overdrive_read.h"
#include "rocm_smi_test/test_common.h"

TestOverdriveRead::TestOverdriveRead() : TestBase() {
  set_title("RSMI Overdrive Read Test");
  set_description("The Overdrive Read tests verifies that the "
                             "current overdrive level can be read properly.");
}

TestOverdriveRead::~TestOverdriveRead(void) {
}

void TestOverdriveRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestOverdriveRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestOverdriveRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestOverdriveRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestOverdriveRead::Run(void) {
  GTEST_SKIP_("Temporarily disabled due to kernel issue");
  rsmi_status_t err;
  uint32_t val_ui32;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(i);

    err = rsmi_dev_overdrive_level_get(i, &val_ui32);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout <<
          "\t**Overdrive Level get is not supported on this machine" << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_overdrive_level_get(i, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
      continue;
    }
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
    std::cout << "\t**OverDrive Level:" << val_ui32 << std::endl;
    // Verify api support checking functionality is working
    err = rsmi_dev_overdrive_level_get(i, nullptr);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }
  }
}
