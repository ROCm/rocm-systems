// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdlib>

#include "amd_smi/amdsmi.h"

// A process may exit without calling amdsmi_shut_down(). The death-test child re-runs this
// test and exits right after its second amdsmi_init(); under LeakSanitizer a leak fails it.
TEST(SystemFunctionalReadOnly, ExitWithoutShutDownIsClean) {
  amdsmi_status_t status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  if (status == AMDSMI_STATUS_DRIVER_NOT_LOADED) GTEST_SKIP() << "amdgpu driver not loaded";
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);

  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  EXPECT_EXIT(
      {
        if (amdsmi_init(AMDSMI_INIT_AMD_GPUS) != AMDSMI_STATUS_SUCCESS) std::exit(2);
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}
