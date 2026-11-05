/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include <cstdlib>

#include "EnvIsolationListener.hpp"
#include "ScopedEnvVar.hpp"

namespace RcclUnitTesting
{
  namespace
  {
    ::testing::TestInfo const& CurrentTest()
    {
      return *::testing::UnitTest::GetInstance()->current_test_info();
    }
  }

  TEST(EnvIsolation, RestoresAndReportsAddedVariable)
  {
    EnvIsolationListener listener;
    listener.OnTestStart(CurrentTest());
    setenv("RCCL_UT_ENV_PROBE", "1", 1);

    EXPECT_NONFATAL_FAILURE(listener.OnTestEnd(CurrentTest()), "RCCL_UT_ENV_PROBE (set)");
    EXPECT_EQ(std::getenv("RCCL_UT_ENV_PROBE"), nullptr);
  }

  TEST(EnvIsolation, RestoresAndReportsChangedVariable)
  {
    ScopedEnvVar probe("RCCL_UT_ENV_PROBE", "before");

    EnvIsolationListener listener;
    listener.OnTestStart(CurrentTest());
    setenv("RCCL_UT_ENV_PROBE", "after", 1);

    EXPECT_NONFATAL_FAILURE(listener.OnTestEnd(CurrentTest()), "RCCL_UT_ENV_PROBE (changed)");
    EXPECT_STREQ(std::getenv("RCCL_UT_ENV_PROBE"), "before");
  }

  TEST(EnvIsolation, RestoresAndReportsRemovedVariable)
  {
    ScopedEnvVar probe("RCCL_UT_ENV_PROBE", "kept");

    EnvIsolationListener listener;
    listener.OnTestStart(CurrentTest());
    unsetenv("RCCL_UT_ENV_PROBE");

    EXPECT_NONFATAL_FAILURE(listener.OnTestEnd(CurrentTest()), "RCCL_UT_ENV_PROBE (unset)");
    EXPECT_STREQ(std::getenv("RCCL_UT_ENV_PROBE"), "kept");
  }

  TEST(EnvIsolation, ScopedEnvVarLeavesNothingToReport)
  {
    EnvIsolationListener listener;
    listener.OnTestStart(CurrentTest());
    {
      ScopedEnvVar added("RCCL_UT_ENV_PROBE", "1");
      ScopedEnvVar removedPath("PATH", nullptr);
    }

    listener.OnTestEnd(CurrentTest());
    EXPECT_FALSE(::testing::Test::HasNonfatalFailure());
  }
}
