// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "xgmi_read_write.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestXGMIReadWrite::TestXGMIReadWrite() : TestBase() {
  set_title("AMDSMI XGMI Read/Write Test");
  set_description(
      "This test verifies that XGMI error counts can be read"
      " properly, and that the count can be reset.");
}

TestXGMIReadWrite::~TestXGMIReadWrite(void) {}

void TestXGMIReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestXGMIReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestXGMIReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestXGMIReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestXGMIReadWrite::Run(void) {
  GTEST_SKIP_("Temporarily disabled");
  amdsmi_status_t err;
  amdsmi_xgmi_status_t err_stat;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    IF_VERB(STANDARD) { std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl; }
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    auto device = processor_handles_[dv_ind];
    PrintDeviceHeader(device);

    amdsmi_xgmi_info_t info;
    DISPLAY_AMDSMI_API("amdsmi_get_xgmi_info", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
    err = amdsmi_get_xgmi_info(device, &info);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      continue;
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**XGMI Hive ID : " << std::hex << info.xgmi_hive_id << std::endl;
      }
    }

    DISPLAY_AMDSMI_API("amdsmi_gpu_xgmi_error_status", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = amdsmi_gpu_xgmi_error_status(device, &err_stat);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);

    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_gpu_xgmi_error_status", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      err = amdsmi_gpu_xgmi_error_status(device, nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_NOT_SUPPORTED);
      ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
      continue;
    }
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**XGMI Error Status: " << static_cast<uint32_t>(err_stat) << std::endl;
    }
    // Verify api support checking functionality is working
    DISPLAY_AMDSMI_API("amdsmi_gpu_xgmi_error_status", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = amdsmi_gpu_xgmi_error_status(device, nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    // TODO(cfree) We need to find a way to generate xgmi errors so this
    // test won't be meaningless
    DISPLAY_AMDSMI_API("amdsmi_reset_gpu_xgmi_error", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    err = amdsmi_reset_gpu_xgmi_error(device);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) { std::cout << "\t**Successfully reset XGMI Error Status: " << std::endl; }
  }
}
