/*
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
#include "rocm_smi_test/functional/err_cnt_read.h"
#include "rocm_smi_test/test_common.h"

TestErrCntRead::TestErrCntRead() : TestBase() {
  set_title("RSMI Error Count Read Test");
  set_description("The Error Count Read tests verifies that error counts"
                                                 " can be read properly.");
}

TestErrCntRead::~TestErrCntRead(void) {
}

void TestErrCntRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestErrCntRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestErrCntRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestErrCntRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestErrCntRead::Run(void) {
  rsmi_status_t err;
  rsmi_error_count_t ec;
  uint64_t enabled_mask;
  rsmi_ras_err_state_t err_state;

  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) {
      std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    }
    return;
  }

  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(i);

      err = rsmi_dev_ecc_enabled_get(i, &enabled_mask);
      if (err == RSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout <<
            "\t**Error Count Enabled Mask get is not supported on this machine"
                                                                   << std::endl;
        }
        // Verify api support checking functionality is working
        err = rsmi_dev_ecc_enabled_get(i, nullptr);
        ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);

        continue;
      } else {
        CHK_ERR_ASRT(err)

        // Verify api support checking functionality is working
        err = rsmi_dev_ecc_enabled_get(i, nullptr);
        ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

        IF_VERB(STANDARD) {
          std::cout << "Block Error Mask: 0x" << std::hex << enabled_mask <<
                                                                    std::endl;
        }
      }
      for (uint32_t b = RSMI_GPU_BLOCK_FIRST;
                                            b <= RSMI_GPU_BLOCK_LAST; b = b*2) {
        err = rsmi_dev_ecc_status_get(i, static_cast<rsmi_gpu_block_t>(b),
                                                                    &err_state);
        CHK_ERR_ASRT(err)
        IF_VERB(STANDARD) {
          std::cout << "\t**Error Count status for " <<
            GetBlockNameStr(static_cast<rsmi_gpu_block_t>(b)) <<
                   " block: " << GetErrStateNameStr(err_state) << std::endl;
        }
        // Verify api support checking functionality is working
        err = rsmi_dev_ecc_status_get(i, static_cast<rsmi_gpu_block_t>(b),
                                                                       nullptr);
        ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

        err = rsmi_dev_ecc_count_get(i, static_cast<rsmi_gpu_block_t>(b), &ec);

        if (err == RSMI_STATUS_NOT_SUPPORTED) {
          IF_VERB(STANDARD) {
            std::cout << "\t**Error Count for " <<
                        GetBlockNameStr(static_cast<rsmi_gpu_block_t>(b)) <<
                                 ": Not supported for this device" << std::endl;
          }
          // Verify api support checking functionality is working
          err = rsmi_dev_ecc_count_get(i, static_cast<rsmi_gpu_block_t>(b),
                                                                       nullptr);
          ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);

        } else {
            CHK_ERR_ASRT(err)
            IF_VERB(STANDARD) {
              std::cout << "\t**Error counts for " <<
                 GetBlockNameStr(static_cast<rsmi_gpu_block_t>(b)) << " block: "
                                                                   << std::endl;
              std::cout << "\t\tCorrectable errors: " << ec.correctable_err
                                                                   << std::endl;
              std::cout << "\t\tUncorrectable errors: " << ec.uncorrectable_err
                                                                   << std::endl;
            }
            // Verify api support checking functionality is working
            err = rsmi_dev_ecc_count_get(i, static_cast<rsmi_gpu_block_t>(b),
                                                                       nullptr);
            ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
        }
      }
    }
  }
}
