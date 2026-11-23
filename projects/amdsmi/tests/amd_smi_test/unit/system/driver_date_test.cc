// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Validates driver date formatting. No GPU required.

#include <gtest/gtest.h>

#include <string>

#include "amd_smi/impl/amd_smi_utils.h"
#include "rocm_smi/rocm_smi_utils.h"

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

TEST(SystemUnit, FileCreationDateReturnsEmptyForInvalidPath) {
  EXPECT_EQ(amd::smi::getFileCreationDate("/dev/null/not-a-directory"), "");
}

TEST(SystemUnit, FileCreationDateFormatsExistingFile) {
  const std::string date = amd::smi::getFileCreationDate("/dev/null");
  EXPECT_FALSE(date.empty());
  EXPECT_EQ(date.find('\n'), std::string::npos);
}
