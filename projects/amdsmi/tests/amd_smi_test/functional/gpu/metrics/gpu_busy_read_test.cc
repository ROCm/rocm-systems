// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_busy_read.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <iostream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestGPUBusyRead::TestGPUBusyRead() : TestBase() {
  set_title("AMDSMI GPU Busy Read Test");
  set_description(
      "The GPU Busy Read tests verifies that the gpu busy "
      "percentage can be read properly.");
}

TestGPUBusyRead::~TestGPUBusyRead(void) {}

void TestGPUBusyRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestGPUBusyRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestGPUBusyRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestGPUBusyRead::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestGPUBusyRead::Run(void) {
  amdsmi_status_t err;
  uint32_t val_ui32;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(processor_handles_[i]);

      DISPLAY_AMDSMI_API("amdsmi_get_gpu_busy_percent", "gpu=" + std::to_string(i), VERB(STANDARD));
      err = amdsmi_get_gpu_busy_percent(processor_handles_[i], &val_ui32);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
      if (err != AMDSMI_STATUS_SUCCESS) {
        if (err == AMDSMI_STATUS_FILE_ERROR || err == AMDSMI_STATUS_NOT_SUPPORTED) {
          IF_VERB(STANDARD) {
            std::cout << "\t**GPU Busy Percent: Not supported on this machine" << std::endl;
          }
          ASSERT_TRUE(err == AMDSMI_STATUS_FILE_ERROR || err == AMDSMI_STATUS_NOT_SUPPORTED);
        } else {
          CHK_ERR_ASRT(err)
        }
      } else {
        IF_VERB(STANDARD) {
          std::cout << "\t**GPU Busy Percent (Percent Idle):" << std::dec << val_ui32 << " ("
                    << 100 - val_ui32 << ")" << std::endl;
        }
      }
    }
  }
}
