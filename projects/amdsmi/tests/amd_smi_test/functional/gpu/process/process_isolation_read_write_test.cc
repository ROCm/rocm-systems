// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// amdsmi_get_gpu_process_isolation / amdsmi_set_gpu_process_isolation.
TEST_F(GpuFunctionalReadWrite, ProcessIsolation_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_gpu_process_isolation");
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t initial = 0;
    if (amdsmi_get_gpu_process_isolation(gpus()[i], &initial) != AMDSMI_STATUS_SUCCESS) continue;

    uint32_t target = initial ? 0u : 1u;
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_process_isolation",
                       "gpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_process_isolation(gpus()[i], target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record("gpu=" + std::to_string(i), err,
               ::amdsmi::test::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

    if (err == AMDSMI_STATUS_SUCCESS) {
      uint32_t readback = 0;
      if (amdsmi_get_gpu_process_isolation(gpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target) << "gpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_gpu_process_isolation(gpus()[i], initial);
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_process_isolation",
                         "gpu=" + std::to_string(i) + " restore=" + std::to_string(initial),
                         kVerbose);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS)
          << "gpu=" << i << " failed to restore process isolation";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_gpu_process_isolation(gpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "gpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}
