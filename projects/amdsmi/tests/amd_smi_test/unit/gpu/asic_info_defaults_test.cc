// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Every backend relies on these defaults to report N/A for what it cannot
// supply, so a field dropped from the initializer would surface as real data.
// No GPU required.

#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <string>

#include "amd_smi/impl/amd_smi_utils.h"
#include "unit_fixtures.h"

namespace {

amdsmi_asic_info_t DefaultsFromGarbage() {
  amdsmi_asic_info_t info;
  std::memset(&info, 0x5A, sizeof(info));
  init_asic_info_defaults(&info);
  return info;
}

TEST_F(GpuUnit, AsicInfoDefaultsMarkScalarsNotSupported) {
  const amdsmi_asic_info_t info = DefaultsFromGarbage();
  const auto u32_max = std::numeric_limits<uint32_t>::max();

  EXPECT_EQ(info.vendor_id, u32_max);
  EXPECT_EQ(info.subvendor_id, u32_max);
  EXPECT_EQ(info.rev_id, u32_max);
  EXPECT_EQ(info.chip_rev_id, u32_max);
  EXPECT_EQ(info.external_rev_id, u32_max);
  EXPECT_EQ(info.oam_id, u32_max);
  EXPECT_EQ(info.physical_acc_id, u32_max);
  EXPECT_EQ(info.num_of_compute_units, u32_max);
  EXPECT_EQ(info.subsystem_id, u32_max);
  EXPECT_EQ(info.device_id, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(info.target_graphics_version, std::numeric_limits<uint64_t>::max());
}

TEST_F(GpuUnit, AsicInfoDefaultsClearStringsAndReserved) {
  const amdsmi_asic_info_t info = DefaultsFromGarbage();

  EXPECT_STREQ(info.market_name, "");
  EXPECT_STREQ(info.vendor_name, "");
  EXPECT_STREQ(info.asic_serial, "ffffffffffffffff");
  EXPECT_EQ(info.flags, 0u);
  for (uint32_t word : info.reserved) {
    EXPECT_EQ(word, 0u);
  }
}

}  // namespace
