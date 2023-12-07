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
#include "rocm_smi_test/functional/overdrive_read_write.h"
#include "rocm_smi_test/test_common.h"

TestOverdriveReadWrite::TestOverdriveReadWrite() : TestBase() {
  set_title("RSMI Overdrive Read/Write Test");
  set_description("The Fan Read tests verifies that the overdrive settings "
                                      "can be read and controlled properly.");
}

TestOverdriveReadWrite::~TestOverdriveReadWrite(void) {
}

void TestOverdriveReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestOverdriveReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestOverdriveReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestOverdriveReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestOverdriveReadWrite::Run(void) {
  GTEST_SKIP_("Temporarily disabled due to kernel issue");
  rsmi_status_t ret;
  uint32_t val;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(dv_ind);

    IF_VERB(STANDARD) {
      std::cout << "Set Overdrive level to 0%..." << std::endl;
    }
    ret = rsmi_dev_overdrive_level_set(dv_ind, 0);
    if (ret == RSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout <<
          "\t**Overdrive Level set is not supported on this machine" << std::endl;
      }
      continue;
    }
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "Set Overdrive level to 10%..." << std::endl;
    }
    ret = rsmi_dev_overdrive_level_set(dv_ind, 10);
    CHK_ERR_ASRT(ret)
    // this won't be reachable if set doesn't work
    // and is checked by overdrive_read.cc test
    ret = rsmi_dev_overdrive_level_get(dv_ind, &val);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**New OverDrive Level:" << val << std::endl;
      std::cout << "Reset Overdrive level to 0%..." << std::endl;
    }
    ret = rsmi_dev_overdrive_level_set(dv_ind, 0);
    CHK_ERR_ASRT(ret)
    ret = rsmi_dev_overdrive_level_get(dv_ind, &val);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**New OverDrive Level:" << val << std::endl;
    }
  }
}
