// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Pins the sysfs cpulist parsing: a CPU past the end of the bitmask must not be
// written, and an over-large range must still set the CPUs that do fit.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "unit_fixtures.h"

namespace {

std::vector<uint64_t> Parse(const std::string& cpulist, size_t words) {
  std::vector<uint64_t> bitmask(words, 0);
  amd::smi::AMDSmiGPUDevice::parse_cpulist(cpulist, bitmask);
  return bitmask;
}

TEST_F(GpuUnit, CpuListSingleValue) { EXPECT_EQ(Parse("5", 1)[0], 1ULL << 5); }

TEST_F(GpuUnit, CpuListRange) { EXPECT_EQ(Parse("0-3", 1)[0], 0xFULL); }

TEST_F(GpuUnit, CpuListCommaSeparated) { EXPECT_EQ(Parse("0-1,4", 1)[0], 0x13ULL); }

TEST_F(GpuUnit, CpuListSpansWords) {
  const std::vector<uint64_t> bitmask = Parse("63-64", 2);
  EXPECT_EQ(bitmask[0], 1ULL << 63);
  EXPECT_EQ(bitmask[1], 1ULL);
}

// The bitmask holds 64 CPUs; the range names more than that.
TEST_F(GpuUnit, CpuListRangeBeyondBitmaskKeepsWhatFits) { EXPECT_EQ(Parse("0-255", 1)[0], ~0ULL); }

TEST_F(GpuUnit, CpuListValueBeyondBitmaskIsDropped) { EXPECT_EQ(Parse("64", 1)[0], 0ULL); }

TEST_F(GpuUnit, CpuListNegativeIsDropped) { EXPECT_EQ(Parse("-1", 1)[0], 0ULL); }

TEST_F(GpuUnit, CpuListMalformedIsSkipped) {
  EXPECT_EQ(Parse("not-a-cpu", 1)[0], 0ULL);
  EXPECT_EQ(Parse("", 1)[0], 0ULL);
}

// std::stol throws out_of_range here; an escaping exception would cross the C API.
TEST_F(GpuUnit, CpuListOutOfRangeValueIsSkipped) {
  EXPECT_EQ(Parse("99999999999999999999", 1)[0], 0ULL);
}

TEST_F(GpuUnit, CpuListMalformedEntryDoesNotDropValidOnes) {
  EXPECT_EQ(Parse("1,bogus,3", 1)[0], 0xAULL);
}

TEST_F(GpuUnit, CpuListEmptyBitmaskIsSafe) {
  std::vector<uint64_t> bitmask;
  amd::smi::AMDSmiGPUDevice::parse_cpulist("0-7", bitmask);
  EXPECT_TRUE(bitmask.empty());
}

}  // namespace
