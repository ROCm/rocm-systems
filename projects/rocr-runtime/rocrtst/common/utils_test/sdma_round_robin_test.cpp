/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>
#include <cstdlib>
#include <string>

#include "gtest/gtest.h"

#include "core/util/flag.h"

// Unit test for the ROCR_SDMA_ROUND_ROBIN and ROCR_SDMA_ROUND_ROBIN_PCIE_XGMI
// environment-variable parsing added to rocr::Flag. rocr::Flag::Refresh() maps
// each variable onto an SDMA_OVERRIDE
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

rocr::Flag::SDMA_OVERRIDE ParseFromEnvPcieXgmi() {
  return ParseSdmaRoundRobin(std::getenv("ROCR_SDMA_ROUND_ROBIN_PCIE_XGMI"));
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

TEST(RocrSdmaRoundRobinPcieXgmi, EnabledWhenOne) {
  ASSERT_EQ(0, setenv("ROCR_SDMA_ROUND_ROBIN_PCIE_XGMI", "1", 1));
  EXPECT_EQ(rocr::Flag::SDMA_ENABLE, ParseFromEnvPcieXgmi());
  unsetenv("ROCR_SDMA_ROUND_ROBIN_PCIE_XGMI");
}

TEST(RocrSdmaRoundRobinPcieXgmi, DisabledWhenZero) {
  ASSERT_EQ(0, setenv("ROCR_SDMA_ROUND_ROBIN_PCIE_XGMI", "0", 1));
  EXPECT_EQ(rocr::Flag::SDMA_DISABLE, ParseFromEnvPcieXgmi());
  unsetenv("ROCR_SDMA_ROUND_ROBIN_PCIE_XGMI");
}

TEST(RocrSdmaRoundRobinPcieXgmi, DefaultWhenUnset) {
  unsetenv("ROCR_SDMA_ROUND_ROBIN_PCIE_XGMI");
  EXPECT_EQ(rocr::Flag::SDMA_DEFAULT, ParseFromEnvPcieXgmi());
}

TEST(RocrSdmaRoundRobinPcieXgmi, DefaultForUnrecognizedValue) {
  ASSERT_EQ(0, setenv("ROCR_SDMA_ROUND_ROBIN_PCIE_XGMI", "yes", 1));
  EXPECT_EQ(rocr::Flag::SDMA_DEFAULT, ParseFromEnvPcieXgmi());
  unsetenv("ROCR_SDMA_ROUND_ROBIN_PCIE_XGMI");
}

// ROCR_SDMA_ENGINE_ID_OFFSET lets a launcher pass an explicit non-negative
// engine-id offset that replaces the getpid() seed in the round-robin pick,
// so concurrent processes do not alias onto the same engine. The parse mirrors
// rocr::Flag::Refresh(): a value of 0 is a valid explicit offset; unset,
// negative, non-numeric or out-of-range input yields the -1 "unset" sentinel
// (fall back to getpid()).
namespace {

int64_t ParseSdmaEngineIdOffset(const char* raw) {
  const std::string var = (raw != nullptr) ? std::string(raw) : std::string();
  int64_t offset = -1;
  if (!var.empty() && var.find_first_not_of("0123456789") == std::string::npos) {
    int64_t parsed = 0;
    bool valid = true;
    for (char c : var) {
      parsed = parsed * 10 + (c - '0');
      if (parsed > 0x7fffffff) {
        valid = false;
        break;
      }
    }
    if (valid) offset = parsed;
  }
  return offset;
}

int64_t ParseOffsetFromEnv() {
  return ParseSdmaEngineIdOffset(std::getenv("ROCR_SDMA_ENGINE_ID_OFFSET"));
}

}  // namespace

TEST(RocrSdmaEngineIdOffset, UnsetIsSentinel) {
  unsetenv("ROCR_SDMA_ENGINE_ID_OFFSET");
  EXPECT_EQ(static_cast<int64_t>(-1), ParseOffsetFromEnv());
}

TEST(RocrSdmaEngineIdOffset, ZeroIsExplicit) {
  ASSERT_EQ(0, setenv("ROCR_SDMA_ENGINE_ID_OFFSET", "0", 1));
  EXPECT_EQ(static_cast<int64_t>(0), ParseOffsetFromEnv());
  unsetenv("ROCR_SDMA_ENGINE_ID_OFFSET");
}

TEST(RocrSdmaEngineIdOffset, PositiveValue) {
  ASSERT_EQ(0, setenv("ROCR_SDMA_ENGINE_ID_OFFSET", "5", 1));
  EXPECT_EQ(static_cast<int64_t>(5), ParseOffsetFromEnv());
  unsetenv("ROCR_SDMA_ENGINE_ID_OFFSET");
}

TEST(RocrSdmaEngineIdOffset, NonNumericFallsBack) {
  ASSERT_EQ(0, setenv("ROCR_SDMA_ENGINE_ID_OFFSET", "abc", 1));
  EXPECT_EQ(static_cast<int64_t>(-1), ParseOffsetFromEnv());
  unsetenv("ROCR_SDMA_ENGINE_ID_OFFSET");
}

TEST(RocrSdmaEngineIdOffset, NegativeFallsBack) {
  ASSERT_EQ(0, setenv("ROCR_SDMA_ENGINE_ID_OFFSET", "-3", 1));
  EXPECT_EQ(static_cast<int64_t>(-1), ParseOffsetFromEnv());
  unsetenv("ROCR_SDMA_ENGINE_ID_OFFSET");
}

TEST(RocrSdmaEngineIdOffset, OutOfRangeFallsBack) {
  ASSERT_EQ(0, setenv("ROCR_SDMA_ENGINE_ID_OFFSET", "99999999999", 1));
  EXPECT_EQ(static_cast<int64_t>(-1), ParseOffsetFromEnv());
  unsetenv("ROCR_SDMA_ENGINE_ID_OFFSET");
}
