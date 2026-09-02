// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>
#include <vector>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// amdsmi_get_gpu_fan_speed / amdsmi_set_gpu_fan_speed (sensor 0).
TEST_F(GpuFunctionalReadWrite, FanSpeed_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_gpu_fan_speed");
  // Only the GPUs this test actually switched to manual control get reset, so a
  // GPU it skipped keeps whatever fan mode the operator had configured.
  std::vector<size_t> switched_to_manual;
  for (size_t i = 0; i < gpus().size(); ++i) {
    int64_t initial = 0;
    if (amdsmi_get_gpu_fan_speed(gpus()[i], 0, &initial) != AMDSMI_STATUS_SUCCESS) continue;

    uint64_t max_speed = 0;
    if (amdsmi_get_gpu_fan_speed_max(gpus()[i], 0, &max_speed) != AMDSMI_STATUS_SUCCESS) continue;
    if (max_speed == 0) continue;

    uint64_t target = (static_cast<uint64_t>(initial) == 0) ? (max_speed / 2) : 0;
    if (target == static_cast<uint64_t>(initial)) continue;

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_fan_speed",
                       "gpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_fan_speed(gpus()[i], 0, target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record("gpu=" + std::to_string(i), err,
               ::amdsmi::test::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

    if (err == AMDSMI_STATUS_SUCCESS) {
      switched_to_manual.push_back(i);
      int64_t readback = 0;
      if (amdsmi_get_gpu_fan_speed(gpus()[i], 0, &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<uint64_t>(readback), target)
            << "gpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_gpu_fan_speed(gpus()[i], 0, static_cast<uint64_t>(initial));
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_fan_speed",
                         "gpu=" + std::to_string(i) + " restore=" + std::to_string(initial),
                         kVerbose);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "gpu=" << i << " failed to restore fan speed";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_gpu_fan_speed(gpus()[i], 0, &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "gpu=" << i << " restore did not take effect";
      }
    }
  }
  // amdsmi_set_gpu_fan_speed leaves the fan under manual control and AMD SMI
  // exposes no getter for the prior control mode, so this returns the fans this
  // test switched -- and only those -- to automatic.
  for (size_t i : switched_to_manual) {
    DISPLAY_AMDSMI_API("amdsmi_reset_gpu_fan", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t rerr = amdsmi_reset_gpu_fan(gpus()[i], 0);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
  }
  col.ExpectNoFailures();
}
