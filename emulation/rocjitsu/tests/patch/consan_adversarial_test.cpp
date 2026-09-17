// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_dispatch_preload.h"
#include <array>
#include <cstdint>
#include <gtest/gtest.h>

namespace rocjitsu::consan {
namespace {

TEST(ConSanAdversarial, DispatchIdSgprCheckpointRestoresEveryShiftedGuestRegister) {
  for (const auto [user_count, system_count, prefix] :
       std::array<std::array<uint16_t, 3>, 4>{{{8, 4, 0}, {12, 2, 4}, {14, 0, 10}, {10, 6, 8}}}) {
    SCOPED_TRACE(user_count);
    const auto plan = plan_dispatch_id_preload(user_count, system_count, prefix,
                                               /*dispatch_id_already_enabled=*/false);
    ASSERT_TRUE(plan.supported());
    ASSERT_TRUE(plan.descriptor_change_required());
    EXPECT_EQ(plan.dispatch_id_sgpr, prefix);
    for (uint16_t destination = 0; destination < plan.required_sgpr_count - 2u; ++destination) {
      const auto source = dispatch_id_restore_source(plan, destination);
      if (destination < prefix) {
        EXPECT_FALSE(source.has_value());
      } else {
        ASSERT_TRUE(source.has_value());
        EXPECT_EQ(*source, destination + 2u);
      }
    }
  }
  EXPECT_EQ(plan_dispatch_id_preload(15, 0, 4, false).support,
            DispatchIdPreloadSupport::UserSgprInitializationLimit);
  EXPECT_EQ(plan_dispatch_id_preload(14, 4, 4, false, /*sgpr_limit=*/19).support,
            DispatchIdPreloadSupport::SgprAllocationLimit);
}

} // namespace
} // namespace rocjitsu::consan
