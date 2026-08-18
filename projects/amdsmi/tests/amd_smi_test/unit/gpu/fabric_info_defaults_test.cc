// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// std::numeric_limits is not specialized for enumeration types, so max() on one
// yields 0 rather than the intended sentinel. Pin the enum defaults so a field
// that reverts to that form reports a plausible state instead of "unknown".
// No GPU required.

#include <gtest/gtest.h>

#include <cstring>
#include <limits>

#include "amd_smi/impl/amd_smi_utils.h"

namespace {

amdsmi_fabric_info_t DefaultsFromGarbage() {
  amdsmi_fabric_info_t info;
  std::memset(&info, 0x5A, sizeof(info));
  init_fabric_info_defaults(&info);
  return info;
}

TEST(GpuUnit, FabricInfoDefaultsMarkEnumsUnknown) {
  const amdsmi_fabric_info_t info = DefaultsFromGarbage();

  EXPECT_EQ(info.fabric_info.v1.fabric_type, AMDSMI_FABRIC_TYPE_UNKNOWN);
  EXPECT_EQ(info.fabric_info.v1.addr_mode, AMDSMI_FABRIC_NPA_ADDRESS_MODE_UNKNOWN);
  EXPECT_EQ(info.fabric_info.v1.accel_state, AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNKNOWN);
}

TEST(GpuUnit, FabricInfoDefaultsMarkScalarsNotSupported) {
  const amdsmi_fabric_info_t info = DefaultsFromGarbage();
  const auto u32_max = std::numeric_limits<uint32_t>::max();

  EXPECT_EQ(info.fabric_version, u32_max);
  EXPECT_EQ(info.fabric_info.v1.accelerator_id, u32_max);
  EXPECT_EQ(info.fabric_info.v1.bandwidth, u32_max);
  EXPECT_EQ(info.fabric_info.v1.latency, u32_max);
  EXPECT_EQ(info.fabric_info.v1.ppod_size, u32_max);
  EXPECT_EQ(info.fabric_info.v1.vpod_id, u32_max);
  EXPECT_EQ(info.fabric_info.v1.vpod_size, u32_max);

  for (const auto byte : info.fabric_info.v1.ppod_id) {
    EXPECT_EQ(byte, 0x99);
  }
}

}  // namespace
