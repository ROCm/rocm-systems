// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "power_read.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <iostream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestPowerRead::TestPowerRead() : TestBase() {
  set_title("AMDSMI Power Read Test");
  set_description(
      "The Power Read tests verifies that "
      "power related values can be read properly.");
}

TestPowerRead::~TestPowerRead(void) {}

void TestPowerRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPowerRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestPowerRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPowerRead::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestPowerRead::Run(void) {
  amdsmi_status_t err;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(processor_handles_[i]);

      amdsmi_power_cap_info_t info;
      DISPLAY_AMDSMI_API("amdsmi_get_power_cap_info", "gpu=" + std::to_string(i), VERB(STANDARD));
      err = amdsmi_get_power_cap_info(processor_handles_[i], 0, &info);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
        continue;
      }
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Current Power Cap: " << info.power_cap << "uW" << std::endl;
      }

      IF_VERB(STANDARD) {
        std::cout << "\t**Default Power Cap: " << info.default_power_cap << "uW" << std::endl;
        std::cout << "\t**Power Cap Range: " << info.min_power_cap << " to " << info.max_power_cap
                  << " uW" << std::endl;
      }
      // TODO(amdsmi_team): Add current_socket_power tests
    }
  }
}
