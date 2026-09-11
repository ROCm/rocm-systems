// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "perf_level_read.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <iostream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestPerfLevelRead::TestPerfLevelRead() : TestBase() {
  set_title("AMDSMI Performance Level Read Test");
  set_description(
      "The Performance Level Read tests verifies that the "
      "performance level monitors can be read properly.");
}

TestPerfLevelRead::~TestPerfLevelRead(void) {}

void TestPerfLevelRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPerfLevelRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestPerfLevelRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPerfLevelRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestPerfLevelRead::Run(void) {
  amdsmi_status_t err;
  amdsmi_dev_perf_level_t pfl;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_perf_level(processor_handles_[i], &pfl);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Performance Level: Not Supported" << std::endl;
      ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Performance Level:" << std::dec << (uint32_t)pfl << std::endl;
      }
    }
    // Verify api support checking functionality is working
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_perf_level(processor_handles_[i], nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
  }
}
