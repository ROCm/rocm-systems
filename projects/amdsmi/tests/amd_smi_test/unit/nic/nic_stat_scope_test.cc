// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * The vendor statistics tier is an argument of amdsmi_get_nic_vendor_statistics_with_scope(), so no
 * state is left on a handle. amdsmi_get_nic_vendor_statistics() keeps its released signature and
 * reads the default tier.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_nic_stats.h"
#include "nic_unit_fixture.h"

namespace {

TEST_F(NicUnit, StatScopeEnumValuesAreStable) {
  EXPECT_EQ(static_cast<int>(AMDSMI_NIC_STAT_SCOPE_DEFAULT), 0);
  EXPECT_EQ(static_cast<int>(AMDSMI_NIC_STAT_SCOPE_EXTENDED), 1);
}

TEST_F(NicUnit, StatScopeIsAnArgumentOfTheScopedGetter) {
  using ScopedSignature_t = amdsmi_status_t (*)(
      amdsmi_processor_handle, uint32_t, amdsmi_nic_stat_scope_t, uint32_t*, amdsmi_nic_stat_t*);

  EXPECT_TRUE(
      (std::is_same_v<decltype(&amdsmi_get_nic_vendor_statistics_with_scope), ScopedSignature_t>));
}

// A value outside the enum is rejected rather than read as the default tier.
TEST_F(NicUnit, StatScopeOutsideTheEnumIsNotResolved) {
  auto nic_scope = SMI_NIC_STAT_SCOPE_DEFAULT;

  EXPECT_FALSE(amd::smi::was_nic_stat_scope_resolved(2, &nic_scope));
  EXPECT_FALSE(amd::smi::was_nic_stat_scope_resolved(-1, &nic_scope));
}

TEST_F(NicUnit, StatScopeMapsEachPublicTierToTheLibraryTier) {
  auto nic_scope = SMI_NIC_STAT_SCOPE_DEFAULT;

  ASSERT_TRUE(amd::smi::was_nic_stat_scope_resolved(AMDSMI_NIC_STAT_SCOPE_EXTENDED, &nic_scope));
  EXPECT_EQ(nic_scope, SMI_NIC_STAT_SCOPE_EXTENDED);
  ASSERT_TRUE(amd::smi::was_nic_stat_scope_resolved(AMDSMI_NIC_STAT_SCOPE_DEFAULT, &nic_scope));
  EXPECT_EQ(nic_scope, SMI_NIC_STAT_SCOPE_DEFAULT);
}

}  // namespace
