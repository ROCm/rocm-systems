/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/ras_param.cc.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>

#include "fakes/env_fakes.h"

// rasTimeoutFactorSec collides with fakes/ras_param_fakes.cc and rasTimeoutFactorNs with ras-test.cc;
// the parameter accessor rename keeps this included implementation consistently isolated.
#define ncclParamRasTimeoutFactor RasParamTestNcclParamRasTimeoutFactor
#define rasTimeoutFactorNs RasParamTestRasTimeoutFactorNs
#define rasTimeoutFactorSec RasParamTestRasTimeoutFactorSec

#include RAS_PARAM_CC_PATH

namespace {

class RasParamMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { SetMicroEnvAbsent("NCCL_RAS_TIMEOUT_FACTOR"); }
  void TearDown() override { ResetEnvFakes(); }
};

TEST_F(RasParamMicrotest, LoadTimeoutFactorMissingOrEmptyUsesDefault) {
  EXPECT_FLOAT_EQ(1.0f, rasLoadTimeoutFactor());

  SetMicroEnv("NCCL_RAS_TIMEOUT_FACTOR", "");
  EXPECT_FLOAT_EQ(1.0f, rasLoadTimeoutFactor());
}

TEST_F(RasParamMicrotest, LoadTimeoutFactorAcceptsPositiveFiniteValues) {
  const struct {
    const char* value;
    float expected;
  } cases[] = {{"0.25", 0.25f}, {"3", 3.0f}, {"2.5", 2.5f}, {"1e3", 1000.0f},
               {" 2.5", 2.5f}, {"+2.5", 2.5f}};
  for (const auto& testCase : cases) {
    SetMicroEnv("NCCL_RAS_TIMEOUT_FACTOR", testCase.value);
    EXPECT_FLOAT_EQ(testCase.expected, rasLoadTimeoutFactor()) << testCase.value;
  }
}

TEST_F(RasParamMicrotest, LoadTimeoutFactorRejectsInvalidValues) {
  for (const char* value : {"abc", "1x", "0", "-1", "nan", "inf", "1e9999", "1e-9999", "1e-310"}) {
    SetMicroEnv("NCCL_RAS_TIMEOUT_FACTOR", value);
    EXPECT_FLOAT_EQ(1.0f, rasLoadTimeoutFactor()) << value;
  }
}

TEST_F(RasParamMicrotest, LoadTimeoutFactorRejectsTrailingWhitespace) {
  SetMicroEnv("NCCL_RAS_TIMEOUT_FACTOR", "2.5 ");
  EXPECT_FLOAT_EQ(1.0f, rasLoadTimeoutFactor());
}

TEST_F(RasParamMicrotest, PublicAccessorsScaleAndCacheTheFactor) {
  // ncclParamRasTimeoutFactor() owns a function-local static cache with no reset hook. This test
  // must remain the only accessor test; under --gtest_repeat, only its first iteration reloads the
  // environment and later iterations intentionally observe the cached 2.5f value.
  SetMicroEnv("NCCL_RAS_TIMEOUT_FACTOR", "2.5");
  EXPECT_FLOAT_EQ(2.5f, ncclParamRasTimeoutFactor());
  EXPECT_EQ(7500000000LL, rasTimeoutFactorNs(3));
  EXPECT_DOUBLE_EQ(10.0, rasTimeoutFactorSec(4));

  SetMicroEnv("NCCL_RAS_TIMEOUT_FACTOR", "9");
  EXPECT_FLOAT_EQ(2.5f, ncclParamRasTimeoutFactor());
}

}  // namespace
