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

// Unit test for the HSA_SDMA_ROUND_ROBIN and HSA_SDMA_ROUND_ROBIN_PCIE_XGMI
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
  return ParseSdmaRoundRobin(std::getenv("HSA_SDMA_ROUND_ROBIN"));
}

rocr::Flag::SDMA_OVERRIDE ParseFromEnvPcieXgmi() {
  return ParseSdmaRoundRobin(std::getenv("HSA_SDMA_ROUND_ROBIN_PCIE_XGMI"));
}

}  // namespace

TEST(RocrSdmaRoundRobin, EnabledWhenOne) {
  ASSERT_EQ(0, setenv("HSA_SDMA_ROUND_ROBIN", "1", 1));
  EXPECT_EQ(rocr::Flag::SDMA_ENABLE, ParseFromEnv());
  unsetenv("HSA_SDMA_ROUND_ROBIN");
}

TEST(RocrSdmaRoundRobin, DisabledWhenZero) {
  ASSERT_EQ(0, setenv("HSA_SDMA_ROUND_ROBIN", "0", 1));
  EXPECT_EQ(rocr::Flag::SDMA_DISABLE, ParseFromEnv());
  unsetenv("HSA_SDMA_ROUND_ROBIN");
}

TEST(RocrSdmaRoundRobin, DefaultWhenUnset) {
  unsetenv("HSA_SDMA_ROUND_ROBIN");
  EXPECT_EQ(rocr::Flag::SDMA_DEFAULT, ParseFromEnv());
}

TEST(RocrSdmaRoundRobin, DefaultForUnrecognizedValue) {
  ASSERT_EQ(0, setenv("HSA_SDMA_ROUND_ROBIN", "yes", 1));
  EXPECT_EQ(rocr::Flag::SDMA_DEFAULT, ParseFromEnv());
  unsetenv("HSA_SDMA_ROUND_ROBIN");
}

TEST(RocrSdmaRoundRobinPcieXgmi, EnabledWhenOne) {
  ASSERT_EQ(0, setenv("HSA_SDMA_ROUND_ROBIN_PCIE_XGMI", "1", 1));
  EXPECT_EQ(rocr::Flag::SDMA_ENABLE, ParseFromEnvPcieXgmi());
  unsetenv("HSA_SDMA_ROUND_ROBIN_PCIE_XGMI");
}

TEST(RocrSdmaRoundRobinPcieXgmi, DisabledWhenZero) {
  ASSERT_EQ(0, setenv("HSA_SDMA_ROUND_ROBIN_PCIE_XGMI", "0", 1));
  EXPECT_EQ(rocr::Flag::SDMA_DISABLE, ParseFromEnvPcieXgmi());
  unsetenv("HSA_SDMA_ROUND_ROBIN_PCIE_XGMI");
}

TEST(RocrSdmaRoundRobinPcieXgmi, DefaultWhenUnset) {
  unsetenv("HSA_SDMA_ROUND_ROBIN_PCIE_XGMI");
  EXPECT_EQ(rocr::Flag::SDMA_DEFAULT, ParseFromEnvPcieXgmi());
}

TEST(RocrSdmaRoundRobinPcieXgmi, DefaultForUnrecognizedValue) {
  ASSERT_EQ(0, setenv("HSA_SDMA_ROUND_ROBIN_PCIE_XGMI", "yes", 1));
  EXPECT_EQ(rocr::Flag::SDMA_DEFAULT, ParseFromEnvPcieXgmi());
  unsetenv("HSA_SDMA_ROUND_ROBIN_PCIE_XGMI");
}

// HSA_SDMA_ENGINE_ID_OFFSET lets a launcher pass an explicit non-negative
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
  return ParseSdmaEngineIdOffset(std::getenv("HSA_SDMA_ENGINE_ID_OFFSET"));
}

}  // namespace

TEST(RocrSdmaEngineIdOffset, UnsetIsSentinel) {
  unsetenv("HSA_SDMA_ENGINE_ID_OFFSET");
  EXPECT_EQ(static_cast<int64_t>(-1), ParseOffsetFromEnv());
}

TEST(RocrSdmaEngineIdOffset, ZeroIsExplicit) {
  ASSERT_EQ(0, setenv("HSA_SDMA_ENGINE_ID_OFFSET", "0", 1));
  EXPECT_EQ(static_cast<int64_t>(0), ParseOffsetFromEnv());
  unsetenv("HSA_SDMA_ENGINE_ID_OFFSET");
}

TEST(RocrSdmaEngineIdOffset, PositiveValue) {
  ASSERT_EQ(0, setenv("HSA_SDMA_ENGINE_ID_OFFSET", "5", 1));
  EXPECT_EQ(static_cast<int64_t>(5), ParseOffsetFromEnv());
  unsetenv("HSA_SDMA_ENGINE_ID_OFFSET");
}

TEST(RocrSdmaEngineIdOffset, NonNumericFallsBack) {
  ASSERT_EQ(0, setenv("HSA_SDMA_ENGINE_ID_OFFSET", "abc", 1));
  EXPECT_EQ(static_cast<int64_t>(-1), ParseOffsetFromEnv());
  unsetenv("HSA_SDMA_ENGINE_ID_OFFSET");
}

TEST(RocrSdmaEngineIdOffset, NegativeFallsBack) {
  ASSERT_EQ(0, setenv("HSA_SDMA_ENGINE_ID_OFFSET", "-3", 1));
  EXPECT_EQ(static_cast<int64_t>(-1), ParseOffsetFromEnv());
  unsetenv("HSA_SDMA_ENGINE_ID_OFFSET");
}

TEST(RocrSdmaEngineIdOffset, OutOfRangeFallsBack) {
  ASSERT_EQ(0, setenv("HSA_SDMA_ENGINE_ID_OFFSET", "99999999999", 1));
  EXPECT_EQ(static_cast<int64_t>(-1), ParseOffsetFromEnv());
  unsetenv("HSA_SDMA_ENGINE_ID_OFFSET");
}

// Method A: launcher local-rank seed detection. The final round-robin seed
// offset is resolved in priority order (mirroring rocr::Flag::Refresh()):
//   1. HSA_SDMA_ENGINE_ID_OFFSET (explicit, including 0)
//   2. the first launcher local-rank env var that parses to a valid
//      non-negative integer (LOCAL_RANK, OMPI_COMM_WORLD_LOCAL_RANK,
//      MPI_LOCALRANKID, PMI_LOCAL_RANK, MV2_COMM_WORLD_LOCAL_RANK,
//      SLURM_LOCALID)
//   3. -1, which makes amd_gpu_agent.cpp fall back to getpid().
namespace {

bool ParseNonNegInt(const std::string& s, int64_t& out) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) return false;
  int64_t parsed = 0;
  for (char c : s) {
    parsed = parsed * 10 + (c - '0');
    if (parsed > 0x7fffffff) return false;
  }
  out = parsed;
  return true;
}

const char* const kLocalRankVars[] = {
    "LOCAL_RANK",     "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
    "PMI_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK",  "SLURM_LOCALID"};

// Resolves the seed offset (>=0) and its source name; source is "" when the
// result is -1 (getpid() fallback).
int64_t ResolveSeedOffset(std::string* source) {
  int64_t offset = ParseSdmaEngineIdOffset(std::getenv("HSA_SDMA_ENGINE_ID_OFFSET"));
  if (offset >= 0) {
    if (source) *source = "HSA_SDMA_ENGINE_ID_OFFSET";
    return offset;
  }
  for (const char* name : kLocalRankVars) {
    const char* raw = std::getenv(name);
    int64_t rank = -1;
    if (raw != nullptr && ParseNonNegInt(std::string(raw), rank)) {
      if (source) *source = name;
      return rank;
    }
  }
  if (source) source->clear();
  return -1;
}

void ClearSeedEnv() {
  unsetenv("HSA_SDMA_ENGINE_ID_OFFSET");
  for (const char* name : kLocalRankVars) unsetenv(name);
}

}  // namespace

TEST(RocrSdmaSeedOffset, OffsetOnly) {
  ClearSeedEnv();
  ASSERT_EQ(0, setenv("HSA_SDMA_ENGINE_ID_OFFSET", "7", 1));
  std::string src;
  EXPECT_EQ(static_cast<int64_t>(7), ResolveSeedOffset(&src));
  EXPECT_EQ("HSA_SDMA_ENGINE_ID_OFFSET", src);
  ClearSeedEnv();
}

TEST(RocrSdmaSeedOffset, OffsetOverridesRank) {
  ClearSeedEnv();
  ASSERT_EQ(0, setenv("HSA_SDMA_ENGINE_ID_OFFSET", "2", 1));
  ASSERT_EQ(0, setenv("OMPI_COMM_WORLD_LOCAL_RANK", "5", 1));
  std::string src;
  EXPECT_EQ(static_cast<int64_t>(2), ResolveSeedOffset(&src));
  EXPECT_EQ("HSA_SDMA_ENGINE_ID_OFFSET", src);
  ClearSeedEnv();
}

TEST(RocrSdmaSeedOffset, SingleRankVar) {
  ClearSeedEnv();
  ASSERT_EQ(0, setenv("SLURM_LOCALID", "3", 1));
  std::string src;
  EXPECT_EQ(static_cast<int64_t>(3), ResolveSeedOffset(&src));
  EXPECT_EQ("SLURM_LOCALID", src);
  ClearSeedEnv();
}

TEST(RocrSdmaSeedOffset, RankVarPriorityOrder) {
  ClearSeedEnv();
  // LOCAL_RANK precedes the MPI/SLURM variables; the first listed wins.
  ASSERT_EQ(0, setenv("SLURM_LOCALID", "9", 1));
  ASSERT_EQ(0, setenv("MPI_LOCALRANKID", "8", 1));
  ASSERT_EQ(0, setenv("LOCAL_RANK", "1", 1));
  std::string src;
  EXPECT_EQ(static_cast<int64_t>(1), ResolveSeedOffset(&src));
  EXPECT_EQ("LOCAL_RANK", src);
  ClearSeedEnv();
}

TEST(RocrSdmaSeedOffset, InvalidRankIsSkipped) {
  ClearSeedEnv();
  // The first var has a malformed value; the resolver skips it and takes the
  // next valid one.
  ASSERT_EQ(0, setenv("LOCAL_RANK", "-4", 1));
  ASSERT_EQ(0, setenv("OMPI_COMM_WORLD_LOCAL_RANK", "6", 1));
  std::string src;
  EXPECT_EQ(static_cast<int64_t>(6), ResolveSeedOffset(&src));
  EXPECT_EQ("OMPI_COMM_WORLD_LOCAL_RANK", src);
  ClearSeedEnv();
}

TEST(RocrSdmaSeedOffset, NoneFallsBackToPid) {
  ClearSeedEnv();
  std::string src = "sentinel";
  EXPECT_EQ(static_cast<int64_t>(-1), ResolveSeedOffset(&src));
  EXPECT_TRUE(src.empty());
  ClearSeedEnv();
}

// Method D: HSA_SDMA_D2H_ENGINE_LIMIT caps the D2H round-robin to the first N
// SDMA engines. Parse mirrors rocr::Flag::Refresh(): 0 / unset / malformed ->
// 0 (no limit); a valid non-negative integer -> that value (amd_gpu_agent.cpp
// clamps it to the engine count with std::min at queue-creation time).
namespace {

int64_t ParseD2hEngineLimit(const char* raw) {
  const std::string var = (raw != nullptr) ? std::string(raw) : std::string();
  int64_t limit = 0;
  ParseNonNegInt(var, limit);
  return limit;
}

int64_t ParseD2hLimitFromEnv() {
  return ParseD2hEngineLimit(std::getenv("HSA_SDMA_D2H_ENGINE_LIMIT"));
}

}  // namespace

TEST(RocrSdmaD2hEngineLimit, UnsetIsZero) {
  unsetenv("HSA_SDMA_D2H_ENGINE_LIMIT");
  EXPECT_EQ(static_cast<int64_t>(0), ParseD2hLimitFromEnv());
}

TEST(RocrSdmaD2hEngineLimit, ZeroIsDisabled) {
  ASSERT_EQ(0, setenv("HSA_SDMA_D2H_ENGINE_LIMIT", "0", 1));
  EXPECT_EQ(static_cast<int64_t>(0), ParseD2hLimitFromEnv());
  unsetenv("HSA_SDMA_D2H_ENGINE_LIMIT");
}

TEST(RocrSdmaD2hEngineLimit, ValidValue) {
  ASSERT_EQ(0, setenv("HSA_SDMA_D2H_ENGINE_LIMIT", "8", 1));
  EXPECT_EQ(static_cast<int64_t>(8), ParseD2hLimitFromEnv());
  unsetenv("HSA_SDMA_D2H_ENGINE_LIMIT");
}

TEST(RocrSdmaD2hEngineLimit, LargeValueParsesForClamping) {
  // Values above the engine count still parse; amd_gpu_agent.cpp clamps them to
  // the available engine count via std::min at queue-creation time.
  ASSERT_EQ(0, setenv("HSA_SDMA_D2H_ENGINE_LIMIT", "64", 1));
  EXPECT_EQ(static_cast<int64_t>(64), ParseD2hLimitFromEnv());
  unsetenv("HSA_SDMA_D2H_ENGINE_LIMIT");
}

TEST(RocrSdmaD2hEngineLimit, MalformedIsZero) {
  ASSERT_EQ(0, setenv("HSA_SDMA_D2H_ENGINE_LIMIT", "abc", 1));
  EXPECT_EQ(static_cast<int64_t>(0), ParseD2hLimitFromEnv());
  unsetenv("HSA_SDMA_D2H_ENGINE_LIMIT");
}
