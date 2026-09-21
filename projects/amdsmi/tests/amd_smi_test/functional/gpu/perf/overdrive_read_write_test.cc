// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "overdrive_read_write.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestOverdriveReadWrite::TestOverdriveReadWrite() : TestBase() {
  set_title("AMDSMI Overdrive Read/Write Test");
  set_description(
      "The Fan Read tests verifies that the overdrive settings "
      "can be read and controlled properly.");
}

TestOverdriveReadWrite::~TestOverdriveReadWrite(void) {}

void TestOverdriveReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestOverdriveReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestOverdriveReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestOverdriveReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestOverdriveReadWrite::Run(void) {
  amdsmi_status_t ret;
  uint32_t val;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    IF_VERB(STANDARD) { std::cout << "Set Overdrive level to 0%..." << std::endl; }
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_overdrive_level", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_overdrive_level(processor_handles_[dv_ind], 0);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) { std::cout << "Set Overdrive level to 10%..." << std::endl; }
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_overdrive_level", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_overdrive_level(processor_handles_[dv_ind], 10);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret)
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_overdrive_level(processor_handles_[dv_ind], &val);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) {
      std::cout << "\t**New OverDrive Level:" << val << std::endl;
      std::cout << "Reset Overdrive level to 0%..." << std::endl;
    }
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_overdrive_level", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_overdrive_level(processor_handles_[dv_ind], 0);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret)
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_overdrive_level(processor_handles_[dv_ind], &val);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) { std::cout << "\t**New OverDrive Level:" << val << std::endl; }
  }
}
