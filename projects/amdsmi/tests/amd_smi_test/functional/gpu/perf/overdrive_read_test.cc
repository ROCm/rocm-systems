// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "overdrive_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestOverdriveRead::TestOverdriveRead() : TestBase() {
  set_title("AMDSMI Overdrive Read Test");
  set_description(
      "The Overdrive Read tests verifies that the "
      "current overdrive level can be read properly.");
}

TestOverdriveRead::~TestOverdriveRead(void) {}

void TestOverdriveRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestOverdriveRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestOverdriveRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestOverdriveRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestOverdriveRead::Run(void) {
  amdsmi_status_t err;
  uint32_t val_ui32;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "gpu=" + std::to_string(i),
                       VERB(STANDARD));
    err = amdsmi_get_gpu_overdrive_level(processor_handles_[i], &val_ui32);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**OverDrive Level:" << val_ui32 << std::endl;
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_overdrive_level(processor_handles_[i], nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
    }
  }
}
