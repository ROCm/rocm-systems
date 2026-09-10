// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "scaling_thread_counts.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace {

using rocjitsu::test::parse_thread_counts;

constexpr uint32_t kMaxThreads = 8;

std::optional<std::vector<uint32_t>> parse(std::initializer_list<const char *> args) {
  const std::vector<const char *> argv(args);
  return parse_thread_counts(std::span<const char *const>(argv), kMaxThreads);
}

TEST(ScalingThreadCountsTest, NoArgumentsSweepsEveryThreadCount) {
  const auto counts = parse({});
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(*counts, (std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(ScalingThreadCountsTest, KeepsRequestedCountsInOrderAndAllowsRepeats) {
  const auto counts = parse({"4", "1", "8", "4"});
  ASSERT_TRUE(counts.has_value());
  EXPECT_EQ(*counts, (std::vector<uint32_t>{4, 1, 8, 4}));
}

// strtoul() wrapped a negative into a huge unsigned, which the uint32_t cast
// then turned into an absurd thread count instead of an error.
TEST(ScalingThreadCountsTest, RejectsNegativeCounts) {
  EXPECT_FALSE(parse({"-1"}).has_value());
  EXPECT_FALSE(parse({"-4294967295"}).has_value());
  EXPECT_FALSE(parse({"4", "-2"}).has_value());
}

TEST(ScalingThreadCountsTest, RejectsCountsOutsideTheSupportedRange) {
  EXPECT_FALSE(parse({"0"}).has_value());
  EXPECT_FALSE(parse({"9"}).has_value());
  EXPECT_FALSE(parse({"4294967296"}).has_value()) << "must not wrap to a valid count";
  EXPECT_FALSE(parse({"18446744073709551617"}).has_value());
}

// A typo used to parse as zero, get dropped, and leave an empty list -- which
// then ran the whole sweep as if no argument had been given at all.
TEST(ScalingThreadCountsTest, RejectsNonNumericArgumentsRatherThanSweeping) {
  EXPECT_FALSE(parse({"all"}).has_value());
  EXPECT_FALSE(parse({""}).has_value());
  EXPECT_FALSE(parse({"--threads=4"}).has_value());
}

// Partial parses are rejected too: "4x" is a typo, not a request for 4.
TEST(ScalingThreadCountsTest, RejectsTrailingTextAfterANumber) {
  EXPECT_FALSE(parse({"4x"}).has_value());
  EXPECT_FALSE(parse({"4 8"}).has_value());
  EXPECT_FALSE(parse({"2."}).has_value());
  EXPECT_FALSE(parse({" 2"}).has_value()) << "leading space is not a number";
}

TEST(ScalingThreadCountsTest, RejectsAlternateBasesAndSignedForms) {
  EXPECT_FALSE(parse({"+4"}).has_value());
  EXPECT_FALSE(parse({"0x4"}).has_value());
}

} // namespace
