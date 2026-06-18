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
#include "computepartition_memallocmode_read_write.h"

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "../test_base.h"
#include "../test_common.h"
#include "amd_smi/amdsmi.h"

TestComputePartitionMemAllocModeReadWrite::TestComputePartitionMemAllocModeReadWrite()
    : TestBase() {
  set_title("AMDSMI Compute Partition Memory Allocation Mode Read/Write Test");
  set_description(
      "The Compute Partition Memory Allocation Mode tests verify that the "
      "compute partition memory allocation mode can be read and updated properly.");
}

TestComputePartitionMemAllocModeReadWrite::~TestComputePartitionMemAllocModeReadWrite(void) {}

void TestComputePartitionMemAllocModeReadWrite::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestComputePartitionMemAllocModeReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestComputePartitionMemAllocModeReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestComputePartitionMemAllocModeReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const std::string memAllocModeString(amdsmi_compute_partition_mem_alloc_mode_t mode) {
  switch (mode) {
    case AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_CAPPING:
      return "CAPPING";
    case AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_ALL:
      return "ALL";
    default:
      return "INVALID";
  }
}

void TestComputePartitionMemAllocModeReadWrite::Run(void) {
  amdsmi_status_t ret;
  amdsmi_compute_partition_mem_alloc_mode_t original_mode;
  amdsmi_compute_partition_mem_alloc_mode_t current_mode;

  if (num_monitor_devs() == 0) {
    return;
  }

  // Invalid input tests (device-independent)
  ret = amdsmi_get_gpu_compute_partition_mem_alloc_mode(processor_handles_[0], nullptr);
  EXPECT_EQ(ret, AMDSMI_STATUS_INVAL);

  ret = amdsmi_set_gpu_compute_partition_mem_alloc_mode(processor_handles_[0],
                                                        AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_INVALID);
  EXPECT_EQ(ret, AMDSMI_STATUS_INVAL);

  ret = amdsmi_set_gpu_compute_partition_mem_alloc_mode(
      processor_handles_[0], static_cast<amdsmi_compute_partition_mem_alloc_mode_t>(99));
  EXPECT_EQ(ret, AMDSMI_STATUS_INVAL);

  const bool isVerbose = (verbosity() > 0);
  const std::vector<amdsmi_compute_partition_mem_alloc_mode_t> modes_to_test = {
      AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_CAPPING,
      AMDSMI_COMPUTE_PARTITION_MEM_ALLOC_ALL,
  };

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); dv_ind++) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    // Get original mode
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition_mem_alloc_mode",
                       "gpu=" + std::to_string(dv_ind), isVerbose);
    ret =
        amdsmi_get_gpu_compute_partition_mem_alloc_mode(processor_handles_[dv_ind], &original_mode);
    DISPLAY_AMDSMI_STATUS(isVerbose, __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);

    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**Device " << dv_ind
                  << ": compute_partition_mem_alloc_mode not supported; skipping.\n";
      }
      continue;
    }

    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      continue;
    }

    IF_VERB(STANDARD) {
      std::cout << "\t**Device " << dv_ind << ": original compute_partition_mem_alloc_mode = "
                << memAllocModeString(original_mode) << "\n";
    }

    // Test setting each mode
    for (auto mode : modes_to_test) {
      IF_VERB(STANDARD) {
        std::cout << "\t**Device " << dv_ind << ": setting mode to " << memAllocModeString(mode)
                  << "\n";
      }

      DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition_mem_alloc_mode",
                         "gpu=" + std::to_string(dv_ind) + ", mode=" + memAllocModeString(mode),
                         isVerbose);
      ret = amdsmi_set_gpu_compute_partition_mem_alloc_mode(processor_handles_[dv_ind], mode);
      DISPLAY_AMDSMI_STATUS(isVerbose, __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);

      if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**Device " << dv_ind << ": set not supported for mode "
                    << memAllocModeString(mode) << "; skipping.\n";
        }
        continue;
      }

      EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
      if (ret != AMDSMI_STATUS_SUCCESS) {
        continue;
      }

      // Verify the mode was applied
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_compute_partition_mem_alloc_mode",
                         "gpu=" + std::to_string(dv_ind) + " (verify)", isVerbose);
      ret = amdsmi_get_gpu_compute_partition_mem_alloc_mode(processor_handles_[dv_ind],
                                                            &current_mode);
      DISPLAY_AMDSMI_STATUS(isVerbose, __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
      if (ret == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(current_mode, mode);
        IF_VERB(STANDARD) {
          std::cout << "\t**Device " << dv_ind
                    << ": verified mode = " << memAllocModeString(current_mode) << "\n";
        }
      }
    }

    // Restore original mode
    IF_VERB(STANDARD) {
      std::cout << "\t**Device " << dv_ind
                << ": restoring original mode = " << memAllocModeString(original_mode) << "\n";
    }
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_compute_partition_mem_alloc_mode",
                       "gpu=" + std::to_string(dv_ind) +
                           ", mode=" + memAllocModeString(original_mode) + " (restore)",
                       isVerbose);
    ret =
        amdsmi_set_gpu_compute_partition_mem_alloc_mode(processor_handles_[dv_ind], original_mode);
    DISPLAY_AMDSMI_STATUS(isVerbose, __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
  }
}
