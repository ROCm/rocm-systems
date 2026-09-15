// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "amd_smi/impl/amd_smi_utils.h"

TEST(GpuUnit, ApuIdentificationUsesFusionBitOnly) {
  for (uint64_t flags : {0u, 1u, 2u, 3u, 24u, 25u}) {
    const bool expected = (flags & 1u) != 0;
    bool is_apu = !expected;
    EXPECT_EQ(smi_amdgpu_get_apu_status(flags, &is_apu), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(is_apu, expected) << "ids_flags=" << flags;
  }
}

TEST(GpuUnit, ApuIdentificationUnavailablePreservesOutput) {
  for (bool initial : {false, true}) {
    bool is_apu = initial;
    EXPECT_EQ(smi_amdgpu_get_apu_status(std::numeric_limits<uint64_t>::max(), &is_apu),
              AMDSMI_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(is_apu, initial);
  }
}

TEST(GpuUnit, ApuIdentificationRejectsNullOutput) {
  EXPECT_EQ(smi_amdgpu_get_apu_status(1u, nullptr), AMDSMI_STATUS_INVAL);
}
