// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "computepartition_memallocmode_read_write.h"

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "test_base.h"
#include "test_common.h"

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

static std::string memAllocModeString(amdsmi_accelerator_partition_mem_alloc_mode_t mode) {
  switch (mode) {
    case AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_CAPPING:
      return "CAPPING";
    case AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_ALL:
      return "ALL";
    default:
      return "INVALID";
  }
}

void TestComputePartitionMemAllocModeReadWrite::Run(void) {
  amdsmi_status_t ret;
  amdsmi_accelerator_partition_mem_alloc_mode_t original_mode;
  amdsmi_accelerator_partition_mem_alloc_mode_t current_mode;

  if (num_monitor_devs() == 0) {
    return;
  }

  // Invalid argument tests (run only when at least one device exists;
  // the handle is validated by SetUp).
  ret = amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(processor_handles_[0], nullptr);
  EXPECT_EQ(ret, AMDSMI_STATUS_INVAL);

  ret = amdsmi_set_gpu_accelerator_partition_mem_alloc_mode(
      processor_handles_[0], AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_INVALID);
  EXPECT_EQ(ret, AMDSMI_STATUS_INVAL);

  ret = amdsmi_set_gpu_accelerator_partition_mem_alloc_mode(
      processor_handles_[0], static_cast<amdsmi_accelerator_partition_mem_alloc_mode_t>(99));
  EXPECT_EQ(ret, AMDSMI_STATUS_INVAL);

  const bool isVerbose = (verbosity() > 0);
  const std::vector<amdsmi_accelerator_partition_mem_alloc_mode_t> modes_to_test = {
      AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_CAPPING,
      AMDSMI_ACCELERATOR_PARTITION_MEM_ALLOC_ALL,
  };

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); dv_ind++) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    // Get original mode
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_mem_alloc_mode",
                       "gpu=" + std::to_string(dv_ind), isVerbose);
    ret = amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(processor_handles_[dv_ind],
                                                              &original_mode);
    DISPLAY_AMDSMI_STATUS(isVerbose, __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);

    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**Device " << dv_ind
                  << ": accelerator_partition_mem_alloc_mode not supported; skipping.\n";
      }
      continue;
    }

    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      continue;
    }

    IF_VERB(STANDARD) {
      std::cout << "\t**Device " << dv_ind << ": original accelerator_partition_mem_alloc_mode = "
                << memAllocModeString(original_mode) << "\n";
    }

    // Test setting each mode
    for (auto mode : modes_to_test) {
      IF_VERB(STANDARD) {
        std::cout << "\t**Device " << dv_ind << ": setting mode to " << memAllocModeString(mode)
                  << "\n";
      }

      DISPLAY_AMDSMI_API("amdsmi_set_gpu_accelerator_partition_mem_alloc_mode",
                         "gpu=" + std::to_string(dv_ind) + ", mode=" + memAllocModeString(mode),
                         isVerbose);
      ret = amdsmi_set_gpu_accelerator_partition_mem_alloc_mode(processor_handles_[dv_ind], mode);
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
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_accelerator_partition_mem_alloc_mode",
                         "gpu=" + std::to_string(dv_ind) + " (verify)", isVerbose);
      ret = amdsmi_get_gpu_accelerator_partition_mem_alloc_mode(processor_handles_[dv_ind],
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
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_accelerator_partition_mem_alloc_mode",
                       "gpu=" + std::to_string(dv_ind) +
                           ", mode=" + memAllocModeString(original_mode) + " (restore)",
                       isVerbose);
    ret = amdsmi_set_gpu_accelerator_partition_mem_alloc_mode(processor_handles_[dv_ind],
                                                              original_mode);
    DISPLAY_AMDSMI_STATUS(isVerbose, __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      std::cerr << "\t**WARNING: failed to restore device " << dv_ind
                << " to original accelerator_partition_mem_alloc_mode = "
                << memAllocModeString(original_mode) << "; device left in modified state.\n";
    }
  }
}
