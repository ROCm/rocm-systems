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

#include "process_list_read_test.h"

#include <gtest/gtest.h>
#include <stdint.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "test_common.h"
#include "amd_smi/amdsmi.h"

TestProcessListRead::TestProcessListRead() : TestBase() {
  set_title("AMDSMI Process List Read Test");
  set_description(
      "This test verifies that amdsmi_get_gpu_process_list reports the "
      "processes running on each GPU through the two-call (count, then fetch) "
      "protocol.");
}

TestProcessListRead::~TestProcessListRead(void) {}

void TestProcessListRead::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestProcessListRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestProcessListRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestProcessListRead::Close() { TestBase::Close(); }

void TestProcessListRead::Run(void) {
  amdsmi_status_t err;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    // A null count pointer must be rejected.
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_process_list(processor_handles_[i], nullptr, nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    // First call with count 0 reports how many processes are on this GPU.
    uint32_t num_procs = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_process_list(processor_handles_[i], &num_procs, nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**GPU process list: Not Supported" << std::endl;
      continue;
    }
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**Processes on GPU: " << std::dec << num_procs << std::endl;
    }

    if (num_procs == 0) {
      continue;
    }

    // Second call fetches the list itself.
    uint32_t count = num_procs;
    std::vector<amdsmi_proc_info_t> procs(count);
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_process_list(processor_handles_[i], &count, procs.data());
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(err)

    // Each returned process must carry a valid (non-zero) PID.
    uint32_t num_read = std::min(count, num_procs);
    for (uint32_t j = 0; j < num_read; ++j) {
      EXPECT_NE(procs[j].pid, 0u);
      IF_VERB(STANDARD) {
        std::cout << "\t** ProcessID: " << std::dec << procs[j].pid << std::endl;
      }
    }
  }
}
