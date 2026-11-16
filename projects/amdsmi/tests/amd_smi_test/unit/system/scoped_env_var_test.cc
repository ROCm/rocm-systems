// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for ScopedEnvVar (test_fixture_utils.h): a test's environment change must be visible
// inside its scope and undone after it, so it cannot reach other tests. The variable below is used
// by these tests only.

#include <gtest/gtest.h>

#include <cstdlib>

#include "test_fixture_utils.h"

namespace {

using amd::smi::testing::ScopedEnvVar;

constexpr const char* kVar = "AMDSMI_TEST_SCOPED_ENV_VAR";

TEST(SystemUnit, SCOPED_ENV_VAR__UnsetsAfterScopeWhenPreviouslyUnset) {
  ASSERT_EQ(unsetenv(kVar), 0);

  {
    ScopedEnvVar guard(kVar, "1");
    EXPECT_STREQ(std::getenv(kVar), "1");
  }

  EXPECT_EQ(std::getenv(kVar), nullptr);
}

TEST(SystemUnit, SCOPED_ENV_VAR__RestoresPreviousValue) {
  ASSERT_EQ(setenv(kVar, "original", 1), 0);

  {
    ScopedEnvVar guard(kVar, "changed");
    EXPECT_STREQ(std::getenv(kVar), "changed");
  }

  EXPECT_STREQ(std::getenv(kVar), "original");
  unsetenv(kVar);
}

// A variable that is set but empty is not the same as an unset one.
TEST(SystemUnit, SCOPED_ENV_VAR__RestoresEmptyValue) {
  ASSERT_EQ(setenv(kVar, "", 1), 0);

  {
    ScopedEnvVar guard(kVar, "changed");
    EXPECT_STREQ(std::getenv(kVar), "changed");
  }

  ASSERT_NE(std::getenv(kVar), nullptr);
  EXPECT_STREQ(std::getenv(kVar), "");
  unsetenv(kVar);
}

}  // namespace
