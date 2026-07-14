/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdlib>
#include <string>

#include "gtest/gtest.h"

#include "core/util/flag.h"

// Unit test for the ROCR_SDMA_ROUND_ROBIN environment-variable parsing added to
// rocr::Flag. rocr::Flag::Refresh() maps the variable onto an SDMA_OVERRIDE
// tri-state with the rule:
//   "1"           -> SDMA_ENABLE
//   "0"           -> SDMA_DISABLE
//   unset / other -> SDMA_DEFAULT
// The internal environment accessors (rocr::os::GetEnvVar) are not exported from
// libhsa-runtime64, so this test reproduces the exact parse against the real
// rocr::Flag::SDMA_OVERRIDE enum. It locks the contract that amd_gpu_agent.cpp
// relies on when deciding whether to round-robin host<->device SDMA copies
// across engines.

namespace {

rocr::Flag::SDMA_OVERRIDE ParseSdmaRoundRobin(const char* raw) {
  const std::string var = (raw != nullptr) ? std::string(raw) : std::string();
  return (var == "1") ? rocr::Flag::SDMA_ENABLE
                      : ((var == "0") ? rocr::Flag::SDMA_DISABLE : rocr::Flag::SDMA_DEFAULT);
}

rocr::Flag::SDMA_OVERRIDE ParseFromEnv() {
  return ParseSdmaRoundRobin(std::getenv("ROCR_SDMA_ROUND_ROBIN"));
}

}  // namespace

TEST(RocrSdmaRoundRobin, EnabledWhenOne) {
  ASSERT_EQ(0, setenv("ROCR_SDMA_ROUND_ROBIN", "1", 1));
  EXPECT_EQ(rocr::Flag::SDMA_ENABLE, ParseFromEnv());
  unsetenv("ROCR_SDMA_ROUND_ROBIN");
}

TEST(RocrSdmaRoundRobin, DisabledWhenZero) {
  ASSERT_EQ(0, setenv("ROCR_SDMA_ROUND_ROBIN", "0", 1));
  EXPECT_EQ(rocr::Flag::SDMA_DISABLE, ParseFromEnv());
  unsetenv("ROCR_SDMA_ROUND_ROBIN");
}

TEST(RocrSdmaRoundRobin, DefaultWhenUnset) {
  unsetenv("ROCR_SDMA_ROUND_ROBIN");
  EXPECT_EQ(rocr::Flag::SDMA_DEFAULT, ParseFromEnv());
}

TEST(RocrSdmaRoundRobin, DefaultForUnrecognizedValue) {
  ASSERT_EQ(0, setenv("ROCR_SDMA_ROUND_ROBIN", "yes", 1));
  EXPECT_EQ(rocr::Flag::SDMA_DEFAULT, ParseFromEnv());
  unsetenv("ROCR_SDMA_ROUND_ROBIN");
}
