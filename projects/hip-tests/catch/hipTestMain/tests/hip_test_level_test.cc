/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <catch2/catch_test_macros.hpp>
#include <hip_test_level.hh>

#include <string>

using HipTestLevel::LevelSource;

TEST_CASE("LevelResolve_DefaultsWhenUnset") {
  const auto resolution = HipTestLevel::resolveLevel(nullptr);
  REQUIRE(resolution.level == HipTestLevel::kDefaultLevel);
  REQUIRE(resolution.source == LevelSource::kDefault);
  REQUIRE_FALSE(resolution.rejected());
}

TEST_CASE("LevelResolve_EmptyEnvironmentIsTreatedAsUnset") {
  // CI scripts commonly export an empty variable; that must not abort the run.
  const auto resolution = HipTestLevel::resolveLevel("");
  REQUIRE(resolution.level == HipTestLevel::kDefaultLevel);
  REQUIRE(resolution.source == LevelSource::kDefault);
  REQUIRE_FALSE(resolution.rejected());
}

TEST_CASE("LevelResolve_EnvironmentSetsEverySupportedLevel") {
  for (const char* supported : HipTestLevel::kSupportedLevels) {
    const auto resolution = HipTestLevel::resolveLevel(supported);
    REQUIRE(resolution.level == supported);
    REQUIRE(resolution.source == LevelSource::kEnvironment);
    REQUIRE_FALSE(resolution.rejected());
  }
}

TEST_CASE("LevelResolve_ListIsRejected") {
  // A list used to select every level while the highest supplied parameters.
  // HIP_TEST_LEVEL no longer selects, so a list has no meaning.
  REQUIRE(HipTestLevel::resolveLevel("level_0,level_2").rejected());
}

TEST_CASE("LevelResolve_ExclusionIsRejected") {
  REQUIRE(HipTestLevel::resolveLevel("~level_4").rejected());
}

TEST_CASE("LevelResolve_BracketedTagIsRejected") {
  REQUIRE(HipTestLevel::resolveLevel("[level_0]").rejected());
}

TEST_CASE("LevelResolve_UnsupportedLevelIsRejected") {
  REQUIRE(HipTestLevel::resolveLevel("level_5").rejected());
}

TEST_CASE("LevelResolve_GarbageIsRejected") {
  REQUIRE(HipTestLevel::resolveLevel("not_a_level").rejected());
  REQUIRE(HipTestLevel::resolveLevel("level_2x").rejected());
  REQUIRE(HipTestLevel::resolveLevel(" level_2").rejected());
  REQUIRE(HipTestLevel::resolveLevel("LEVEL_2").rejected());
}

TEST_CASE("LevelResolve_RejectionKeepsTheDefaultAndNamesTheValue") {
  // The level stays usable so nothing downstream sees an empty or bogus level,
  // but the parameter listener aborts the run on rejected() before any test runs.
  const auto resolution = HipTestLevel::resolveLevel("level_0,level_2");
  REQUIRE(resolution.level == HipTestLevel::kDefaultLevel);
  REQUIRE(resolution.source == LevelSource::kDefault);
  REQUIRE(resolution.error ==
          "HIP_TEST_LEVEL='level_0,level_2' is not a supported level; expected exactly one of "
          "level_0, level_1, level_2, level_3, level_4");
}

TEST_CASE("LevelDescribe_ReportsSourceAndLevel") {
  REQUIRE(HipTestLevel::resolveLevel("level_1").describe() ==
          "[Test Level] Parameters from HIP_TEST_LEVEL: level_1");
  REQUIRE(HipTestLevel::resolveLevel(nullptr).describe() ==
          "[Test Level] Parameters from default: level_2");
}

TEST_CASE("LevelSupport_KnownAndUnknown") {
  REQUIRE(HipTestLevel::isSupportedLevel("level_0"));
  REQUIRE(HipTestLevel::isSupportedLevel("level_4"));
  REQUIRE_FALSE(HipTestLevel::isSupportedLevel("level_5"));
  REQUIRE_FALSE(HipTestLevel::isSupportedLevel(""));
}
