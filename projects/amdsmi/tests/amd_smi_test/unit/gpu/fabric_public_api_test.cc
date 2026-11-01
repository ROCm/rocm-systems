// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Argument rejection on the public fabric surface. The functional fabric suites cover this
// logic only behind a hardware-and-root gate, so it never runs in CI.

#include <gtest/gtest.h>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TEST(GpuUnit, FabricPublicApiGetRejectsNullArguments) {
  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;
  amdsmi_status_t err;
  auto info = amdsmi_fabric_info_t{};

  DISPLAY_AMDSMI_API("amdsmi_get_gpu_fabric_info",
                     "FabricPublicApiGetRejectsNullArguments, processor_handle=nullptr, "
                     "info=&info",
                     verbose);
  err = amdsmi_get_gpu_fabric_info(nullptr, &info);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);

  DISPLAY_AMDSMI_API("amdsmi_get_gpu_fabric_info",
                     "FabricPublicApiGetRejectsNullArguments, processor_handle=nullptr, "
                     "info=nullptr",
                     verbose);
  err = amdsmi_get_gpu_fabric_info(nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}
