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
#include "err_cnt_read.h"
#include "../test_common.h"

TestErrCntRead::TestErrCntRead() : TestBase() {
  set_title("AMDSMI Error Count Read Test");
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
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestErrCntRead::Run(void) {
  amdsmi_status_t err;
  amdsmi_error_count_t ec;
  uint64_t enabled_mask;
  amdsmi_ras_err_state_t err_state;

  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) {
      std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    }
    return;
  }

  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(processor_handles_[i]);

      err =  amdsmi_get_gpu_ecc_enabled(processor_handles_[i], &enabled_mask);
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout <<
            "\t**Error Count Enabled Mask get is not supported on this machine"
                                                                   << std::endl;
        }
        // Verify api support checking functionality is working
        err =  amdsmi_get_gpu_ecc_enabled(processor_handles_[i], nullptr);
        ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);

        continue;
      } else {
        CHK_ERR_ASRT(err)

        // Verify api support checking functionality is working
        err =  amdsmi_get_gpu_ecc_enabled(processor_handles_[i], nullptr);
        ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

        IF_VERB(STANDARD) {
          std::cout << "Block Error Mask: 0x" << std::hex << enabled_mask <<
                                                                    std::endl;
        }
      }
      for (uint32_t b = AMDSMI_GPU_BLOCK_FIRST;
                                            b <= AMDSMI_GPU_BLOCK_LAST; b = b*2) {
        err =  amdsmi_get_gpu_ecc_status(processor_handles_[i], static_cast<amdsmi_gpu_block_t>(b),
                                                                    &err_state);
        CHK_ERR_ASRT(err)
        IF_VERB(STANDARD) {
          std::cout << "\t**Error Count status for " <<
            GetBlockNameStr(static_cast<amdsmi_gpu_block_t>(b)) <<
                   " block: " << GetErrStateNameStr(err_state) << std::endl;
        }
        // Verify api support checking functionality is working
        err =  amdsmi_get_gpu_ecc_status(processor_handles_[i], static_cast<amdsmi_gpu_block_t>(b),
                                                                       nullptr);
        ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

        err =  amdsmi_get_gpu_ecc_count(processor_handles_[i], static_cast<amdsmi_gpu_block_t>(b), &ec);

        if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
          IF_VERB(STANDARD) {
            std::cout << "\t**Error Count for " <<
                        GetBlockNameStr(static_cast<amdsmi_gpu_block_t>(b)) <<
                          ": Not supported for this device or error accessing file" << std::endl;
          }
          // Verify api support checking functionality is working
          err =  amdsmi_get_gpu_ecc_count(processor_handles_[i], static_cast<amdsmi_gpu_block_t>(b),
                                                                       nullptr);
          ASSERT_TRUE(err == AMDSMI_STATUS_NOT_SUPPORTED);

        } else {
            CHK_ERR_ASRT(err)
            IF_VERB(STANDARD) {
              std::cout << "\t**Error counts for " <<
                 GetBlockNameStr(static_cast<amdsmi_gpu_block_t>(b)) << " block: "
                                                                   << std::endl;
              std::cout << "\t\tCorrectable errors: " << ec.correctable_count
                                                                   << std::endl;
              std::cout << "\t\tUncorrectable errors: " << ec.uncorrectable_count
                                                                   << std::endl;
            }
            // Verify api support checking functionality is working
            err =  amdsmi_get_gpu_ecc_count(processor_handles_[i], static_cast<amdsmi_gpu_block_t>(b),
                                                                       nullptr);
            ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
        }
      }
    }
  }
}
