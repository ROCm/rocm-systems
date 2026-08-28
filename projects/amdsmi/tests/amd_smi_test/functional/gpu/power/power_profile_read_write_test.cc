// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// amdsmi_get_power_cap_info / amdsmi_set_power_cap.
TEST_F(GpuFunctionalReadWrite, PowerCap_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_power_cap");
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_power_cap_info_t info;
    memset(&info, 0, sizeof(info));
    if (amdsmi_get_power_cap_info(gpus()[i], 0, &info) != AMDSMI_STATUS_SUCCESS) continue;
    if (info.max_power_cap <= info.min_power_cap) continue;

    uint64_t initial = info.power_cap;
    uint64_t target = info.min_power_cap + (info.max_power_cap - info.min_power_cap) / 2;
    if (target == initial) target = info.min_power_cap;
    if (target == initial) continue;

    DISPLAY_AMDSMI_API("amdsmi_set_power_cap",
                       "gpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_power_cap(gpus()[i], 0, target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record("gpu=" + std::to_string(i), err,
               ::amdsmi::test::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

    if (err == AMDSMI_STATUS_SUCCESS) {
      amdsmi_power_cap_info_t readback;
      memset(&readback, 0, sizeof(readback));
      if (amdsmi_get_power_cap_info(gpus()[i], 0, &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback.power_cap, target) << "gpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_power_cap(gpus()[i], 0, initial);
      DISPLAY_AMDSMI_API("amdsmi_set_power_cap",
                         "gpu=" + std::to_string(i) + " restore=" + std::to_string(initial),
                         kVerbose);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "gpu=" << i << " failed to restore power cap";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_power_cap_info(gpus()[i], 0, &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback.power_cap, initial) << "gpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}

// amdsmi_get_gpu_power_profile_presets / amdsmi_set_gpu_power_profile.
TEST_F(GpuFunctionalReadWrite, PowerProfile_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_gpu_power_profile");
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_power_profile_status_t status;
    memset(&status, 0, sizeof(status));
    if (amdsmi_get_gpu_power_profile_presets(gpus()[i], 0, &status) != AMDSMI_STATUS_SUCCESS)
      continue;

    // current is INVALID when no preset is active; only an available preset can be set back.
    const uint64_t initial = static_cast<uint64_t>(status.current);
    if (initial == 0 || (status.available_profiles & initial) != initial) continue;

    // Pick a different available preset from the bitmask.
    amdsmi_power_profile_preset_masks_t target = AMDSMI_PWR_PROF_PRST_INVALID;
    for (uint64_t bit = 0x1; bit <= AMDSMI_PWR_PROF_PRST_BOOTUP_DEFAULT; bit <<= 1) {
      if ((status.available_profiles & bit) && bit != initial) {
        target = static_cast<amdsmi_power_profile_preset_masks_t>(bit);
        break;
      }
    }
    if (target == AMDSMI_PWR_PROF_PRST_INVALID) continue;

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile",
                       "gpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_power_profile(gpus()[i], 0, target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record("gpu=" + std::to_string(i), err,
               ::amdsmi::test::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

    if (err == AMDSMI_STATUS_SUCCESS) {
      amdsmi_power_profile_status_t readback;
      memset(&readback, 0, sizeof(readback));
      if (amdsmi_get_gpu_power_profile_presets(gpus()[i], 0, &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback.current, target) << "gpu=" << i << " set did not take effect";
      }

      amdsmi_power_profile_preset_masks_t restore = status.current;
      amdsmi_status_t rerr = amdsmi_set_gpu_power_profile(gpus()[i], 0, restore);
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_power_profile",
                         "gpu=" + std::to_string(i) + " restore=" + std::to_string(restore),
                         kVerbose);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "gpu=" << i << " failed to restore power profile";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_gpu_power_profile_presets(gpus()[i], 0, &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback.current, restore) << "gpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}
