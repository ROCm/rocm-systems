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

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/xgmi_read_write.h"
#include "rocm_smi_test/test_common.h"

TestXGMIReadWrite::TestXGMIReadWrite() : TestBase() {
  set_title("RSMI XGMI Read/Write Test");
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
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestXGMIReadWrite::Run(void) {
  GTEST_SKIP_("Temporarily disabled");
  rsmi_status_t err;
  rsmi_xgmi_status_t err_stat;
  uint64_t hive_id;

  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) {
      std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    }
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(dv_ind);

    err = rsmi_dev_xgmi_hive_id_get(dv_ind, &hive_id);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
        std::cout <<
            "\t**rsmi_dev_xgmi_hive_id_get() is not supported"
            " on this machine" << std::endl;
        // Verify api support checking functionality is working
        err = rsmi_dev_xgmi_hive_id_get(dv_ind, nullptr);
        ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);

        continue;
    } else {
        CHK_ERR_ASRT(err)
        IF_VERB(STANDARD) {
            std::cout << "\t**XGMI Hive ID : " << std::hex << hive_id <<
            std::endl;
        }
        // Verify api support checking functionality is working
        err = rsmi_dev_xgmi_hive_id_get(dv_ind, nullptr);
        ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }

    err = rsmi_dev_xgmi_error_status(dv_ind, &err_stat);

    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**XGMI Error Status: Not supported on this machine"
                                                               << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_xgmi_error_status(dv_ind, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);

      continue;
    }
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**XGMI Error Status: " <<
                               static_cast<uint32_t>(err_stat) << std::endl;
    }
    // Verify api support checking functionality is working
    err = rsmi_dev_xgmi_error_status(dv_ind, nullptr);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

    // TODO(cfree) We need to find a way to generate xgmi errors so this
    // test won't be meaningless
    err = rsmi_dev_xgmi_error_reset(dv_ind);
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**Successfully reset XGMI Error Status: " << std::endl;
    }
  }
}
