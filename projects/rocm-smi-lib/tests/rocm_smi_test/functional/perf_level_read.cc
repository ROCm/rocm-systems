/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
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
#include "rocm_smi_test/functional/perf_level_read.h"
#include "rocm_smi_test/test_common.h"

TestPerfLevelRead::TestPerfLevelRead() : TestBase() {
  set_title("RSMI Performance Level Read Test");
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
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestPerfLevelRead::Run(void) {
  rsmi_status_t err;
  rsmi_dev_perf_level_t pfl;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(i);

    err = rsmi_dev_perf_level_get(i, &pfl);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Performance Level: Not Supported" << std::endl;
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Performance Level:" << std::dec << (uint32_t)pfl
                  << std::endl;
      }
    }
    // Verify api support checking functionality is working
    err = rsmi_dev_perf_level_get(i, nullptr);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
  }
}
