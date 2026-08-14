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

#include "process_list_read.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/fdinfo.h"
#include "test_common.h"

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

  // Per-GPU checks use EXPECT_* (never ASSERT_*/CHK_ERR_ASRT) so that a
  // transient failure on one GPU does not abort the whole sweep and leave the
  // remaining GPUs untested.
  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    // A null count pointer must be rejected.
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_process_list(processor_handles_[i], nullptr, nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    EXPECT_EQ(err, AMDSMI_STATUS_INVAL);

    // First call with count 0 reports how many processes are on this GPU.
    uint32_t num_procs = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_process_list", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_process_list(processor_handles_[i], &num_procs, nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**GPU process list: Not Supported" << std::endl;
      continue;
    }
    EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
    if (err != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
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
    EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
    if (err != AMDSMI_STATUS_SUCCESS) {
      continue;
    }

    // Every reported process must carry a valid, unique PID. PID 0 is the
    // kernel idle task and PID 1 is init; neither owns a GPU compute context,
    // so a value <= 1 or a duplicate PID signals a stale, wrong-GPU, or
    // otherwise corrupted entry.
    uint32_t num_read = std::min(count, num_procs);
    std::set<uint32_t> seen_pids;
    for (uint32_t j = 0; j < num_read; ++j) {
      EXPECT_GT(procs[j].pid, 1u);
      EXPECT_TRUE(seen_pids.insert(procs[j].pid).second)
          << "Duplicate PID " << procs[j].pid << " reported on GPU " << i;
      IF_VERB(STANDARD) {
        std::cout << "\t** ProcessID: " << std::dec << procs[j].pid << std::endl;
      }
    }

    if (num_read == 0) {
      continue;
    }

    // Exercise the KFD-gpu-id fallback that this change added to
    // gpu_is_in_kfd_pid(). The fast path (used by amdsmi_get_gpu_process_list
    // above) passes this device's cached KFD gpu id, while a sentinel of
    // UINT64_MAX or 0 forces the original topology-discovery path that
    // re-derives the id from the BDF. A PID the fast path just reported for
    // this GPU must also be resolved by the discovery fallback; if it is not,
    // the BDF->KFD-id translation has regressed.
    amdsmi_bdf_t bdf = {};
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_bdf", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_device_bdf(processor_handles_[i], &bdf);
    EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
    if (err != AMDSMI_STATUS_SUCCESS) {
      continue;
    }

    const long sample_pid = static_cast<long>(procs[0].pid);
    amdsmi_status_t fallback_max = gpu_is_in_kfd_pid(bdf, sample_pid, UINT64_MAX);

    // Reading KFD process info needs permission; without it the fallback cannot
    // be validated, so skip rather than report a spurious failure.
    if (fallback_max == AMDSMI_STATUS_NO_PERM || fallback_max == AMDSMI_STATUS_API_FAILED) {
      IF_VERB(STANDARD) {
        std::cout << "\t** Skipping KFD fallback check for GPU " << i
                  << " (insufficient permissions for KFD topology)" << std::endl;
      }
      continue;
    }

    EXPECT_EQ(fallback_max, AMDSMI_STATUS_SUCCESS);
    // Both sentinels select the same discovery path and must agree.
    EXPECT_EQ(fallback_max, gpu_is_in_kfd_pid(bdf, sample_pid, 0));
  }
}
