// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * The vendor statistics tier is per-handle state. amdsmi_get_nic_vendor_statistics() keeps its
 * released signature, so the tier is chosen with amdsmi_set_nic_stat_scope() and held on the NIC
 * device until the next set. A handle nobody configured must read the default tier, as a caller of
 * the released API expects.
 */

#include <gtest/gtest.h>

#include <thread>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/nic/amd_smi_ainic_device.h"
#include "nic_unit_fixture.h"

namespace {

using amd::smi::AMDSmiAINICDevice;

constexpr auto kNumToggles = 20000;

TEST_F(NicUnit, StatScopeDefaultsToTheDefaultTier) {
  const AMDSmiAINICDevice device(info_);

  EXPECT_EQ(device.stat_scope(), AMDSMI_NIC_STAT_SCOPE_DEFAULT);
}

// Setting back to the default must stick; a one-way latch to EXTENDED would pass the first set.
TEST_F(NicUnit, StatScopeKeepsTheLastTierSet) {
  AMDSmiAINICDevice device(info_);

  device.set_stat_scope(AMDSMI_NIC_STAT_SCOPE_EXTENDED);
  EXPECT_EQ(device.stat_scope(), AMDSMI_NIC_STAT_SCOPE_EXTENDED);

  device.set_stat_scope(AMDSMI_NIC_STAT_SCOPE_DEFAULT);
  EXPECT_EQ(device.stat_scope(), AMDSMI_NIC_STAT_SCOPE_DEFAULT);
}

TEST_F(NicUnit, StatScopeIsKeptPerDevice) {
  AMDSmiAINICDevice first(info_);
  const AMDSmiAINICDevice second(info_);

  first.set_stat_scope(AMDSMI_NIC_STAT_SCOPE_EXTENDED);

  EXPECT_EQ(second.stat_scope(), AMDSMI_NIC_STAT_SCOPE_DEFAULT);
}

// Two threads share one handle; under TSan a non-atomic tier reports a data race here.
TEST_F(NicUnit, StatScopeSurvivesConcurrentSetAndGet) {
  AMDSmiAINICDevice device(info_);
  auto is_scope_seen_valid = true;

  std::thread writer([&device] {
    for (auto i = 0; i < kNumToggles; ++i) {
      device.set_stat_scope(((i % 2) == 0) ? AMDSMI_NIC_STAT_SCOPE_EXTENDED
                                           : AMDSMI_NIC_STAT_SCOPE_DEFAULT);
    }
  });
  for (auto i = 0; i < kNumToggles; ++i) {
    const auto scope = device.stat_scope();
    if ((scope != AMDSMI_NIC_STAT_SCOPE_DEFAULT) && (scope != AMDSMI_NIC_STAT_SCOPE_EXTENDED)) {
      is_scope_seen_valid = false;
    }
  }
  writer.join();

  EXPECT_TRUE(is_scope_seen_valid);
}

}  // namespace
