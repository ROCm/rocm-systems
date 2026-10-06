// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops_internal.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_profiles.h"

#include <gtest/gtest.h>

#include <array>

namespace rocjitsu::consan {

#include "rocjitsu/code/patch/consan/targets/shared/cdna3_cdna4/consan_cdna3_cdna4_target_profile.h.inc"

#include "rocjitsu/code/patch/consan/targets/cdna3/consan_gfx942_target_profile.h.inc"

namespace {

// The operation table is defined in another translation unit. GCC with UBSan
// must still validate its registration during constant evaluation.
constexpr std::array kRegisteredProfiles = {kGfx942TargetProfile};
constexpr auto kUnregisteredProfiles = [] {
  auto profiles = kRegisteredProfiles;
  profiles[0].program_analysis = {};
  return profiles;
}();

static_assert(target_profiles_are_valid(kRegisteredProfiles));
static_assert(!target_profiles_are_valid(kUnregisteredProfiles));

TEST(ConSanCapabilityContract, TargetProfileRegistrationIsRequiredForEveryTarget) {
  for (std::size_t index = 0; index < kTargetProfiles.size(); ++index) {
    SCOPED_TRACE(index);
    auto profiles = kTargetProfiles;
    ASSERT_TRUE(target_profiles_are_valid(profiles));
    profiles[index].program_analysis = {};
    EXPECT_FALSE(target_profiles_are_valid(profiles));
  }
}

} // namespace
} // namespace rocjitsu::consan
