// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Argument rejection on the public fabric surface. The functional fabric suites cover this
// logic only behind a hardware-and-root gate, so it never runs in CI.

#include <gtest/gtest.h>

#include "amd_smi/amdsmi.h"

TEST(GpuUnit, FabricPublicApiGetRejectsNullArguments) {
  auto info = amdsmi_fabric_info_t{};
  info.fabric_version = AMDSMI_FABRIC_INFO_VERSION_2;

  EXPECT_EQ(amdsmi_get_gpu_fabric_info(nullptr, &info), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_gpu_fabric_info(nullptr, nullptr), AMDSMI_STATUS_INVAL);
}
