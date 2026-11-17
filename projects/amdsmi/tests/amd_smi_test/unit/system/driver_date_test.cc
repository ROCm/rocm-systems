// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Validates driver date formatting. No GPU required.

#include <gtest/gtest.h>

#include <string>

#include "amd_smi/impl/amd_smi_utils.h"

TEST(SystemUnit, DriverDateFormatsDrmDate) {
  EXPECT_EQ(smi_amdgpu_format_driver_date("20150101"), "2015/01/01 00:00");
}

TEST(SystemUnit, DriverDateEmptyWhenKernelReportsNone) {
  // Kernels that no longer use the DRM driver date report "0".
  EXPECT_EQ(smi_amdgpu_format_driver_date("0"), "");
  EXPECT_EQ(smi_amdgpu_format_driver_date(""), "");
}

TEST(SystemUnit, DriverDatePassesThroughUnexpectedFormats) {
  EXPECT_EQ(smi_amdgpu_format_driver_date("2015-01-01"), "2015-01-01");
}
