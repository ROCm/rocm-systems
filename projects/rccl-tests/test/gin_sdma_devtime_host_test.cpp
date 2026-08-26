/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for rccl-tests device-timing helpers in gin_sdma_devtime_host.h
// (BenchTime zero-time guards, gin_sdma_devtime.h stamp-reduction math, and
// skipped-row stdout/JSON layout contract shared with util.cu).
// No GPU or MPI required.

#include <gtest/gtest.h>

#include <string>

#include "gin_sdma_devtime_host.h"

using rccl_tests_devtime::applyCudaGraphLaunchesScale;
using rccl_tests_devtime::emitSkippedOutOfPlaceColumnToStdout;
using rccl_tests_devtime::goldenSkippedRowStdoutFragment;
using rccl_tests_devtime::gridBusyWindowPerIterUs;
using rccl_tests_devtime::kSkippedOutOfPlaceColumnFiller;
using rccl_tests_devtime::kSkippedOutOfPlaceColumnFillerLen;
using rccl_tests_devtime::normalizeElapsedPerIter;
using rccl_tests_devtime::reduceMaxEnd;
using rccl_tests_devtime::reduceMinStart;
using rccl_tests_devtime::shouldComputeIterStats;
using rccl_tests_devtime::shouldSkipBenchTimeRow;
using rccl_tests_devtime::skippedOutOfPlaceJsonNullFragment;

TEST(GinSdmaDevtimeHost, NormalizeElapsedPerIter) {
  EXPECT_DOUBLE_EQ(normalizeElapsedPerIter(2.0, 20, 1), 0.1);
  EXPECT_DOUBLE_EQ(normalizeElapsedPerIter(1.5, 10, 2), 0.075);
  EXPECT_DOUBLE_EQ(normalizeElapsedPerIter(1.0, 0, 1), 0.0);
  EXPECT_DOUBLE_EQ(normalizeElapsedPerIter(1.0, 1, 0), 0.0);
}

TEST(GinSdmaDevtimeHost, ApplyCudaGraphLaunchesScale) {
  EXPECT_DOUBLE_EQ(applyCudaGraphLaunchesScale(0.4, 0), 0.4);
  EXPECT_DOUBLE_EQ(applyCudaGraphLaunchesScale(0.4, 4), 0.1);
  EXPECT_DOUBLE_EQ(applyCudaGraphLaunchesScale(0.4, -1), 0.4);
}

TEST(GinSdmaDevtimeHost, ShouldSkipBenchTimeRow) {
  EXPECT_TRUE(shouldSkipBenchTimeRow(0.0));
  EXPECT_TRUE(shouldSkipBenchTimeRow(-1e-12));
  EXPECT_FALSE(shouldSkipBenchTimeRow(1e-9));
}

TEST(GinSdmaDevtimeHost, ShouldComputeIterStats) {
  EXPECT_TRUE(shouldComputeIterStats(20, 0));
  EXPECT_TRUE(shouldComputeIterStats(20, 19));
  EXPECT_FALSE(shouldComputeIterStats(20, 20));
}

TEST(GinSdmaDevtimeHost, GridBusyWindowReduction) {
  const long long starts[] = {100, 50, 80};
  const long long ends[] = {250, 300, 200};
  EXPECT_EQ(reduceMinStart(starts, 3), 50);
  EXPECT_EQ(reduceMaxEnd(ends, 3), 300);
  // (300-50)/100 kHz * 1e3 = 2500 us total; loop=10 -> 250 us/iter
  EXPECT_DOUBLE_EQ(gridBusyWindowPerIterUs(50, 300, 100, 10), 250.0);
  EXPECT_DOUBLE_EQ(gridBusyWindowPerIterUs(50, 300, 100, 0), 2500.0);
}

TEST(GinSdmaDevtimeHost, SkippedOutOfPlaceStdoutGolden) {
  testing::internal::CaptureStdout();
  emitSkippedOutOfPlaceColumnToStdout();
  const std::string out = testing::internal::GetCapturedStdout();
  EXPECT_EQ(out, kSkippedOutOfPlaceColumnFiller);
  EXPECT_EQ(out.size(), kSkippedOutOfPlaceColumnFillerLen);
  EXPECT_EQ(kSkippedOutOfPlaceColumnFillerLen, 32u);
}

TEST(GinSdmaDevtimeHost, SkippedOutOfPlaceRowLayoutGolden) {
  // Simulates preamble + skipped OOP metric + in-place body on the same row.
  // BenchTime must emit the filler (not return early) so in-place metrics stay
  // in the in-place column and JSON retains out_of_place:null.
  const char* preamble =
      "        4096          1024   float     sum       0";
  const char* inPlaceBody = "   12.34   1.23   2.46    N/A";
  const std::string row = goldenSkippedRowStdoutFragment(preamble, inPlaceBody);

  const size_t fillerPos = row.find(kSkippedOutOfPlaceColumnFiller);
  const size_t inPlacePos = row.find(inPlaceBody);
  ASSERT_NE(fillerPos, std::string::npos);
  ASSERT_NE(inPlacePos, std::string::npos);
  EXPECT_LT(fillerPos, inPlacePos);
  EXPECT_EQ(row.substr(fillerPos, kSkippedOutOfPlaceColumnFillerLen),
            kSkippedOutOfPlaceColumnFiller);

  const std::string json = std::string("{\"size\":4096,") +
                           skippedOutOfPlaceJsonNullFragment() + ",\"in_place\":{";
  const size_t oopJsonPos = json.find(skippedOutOfPlaceJsonNullFragment());
  const size_t ipJsonPos = json.find("\"in_place\"");
  ASSERT_NE(oopJsonPos, std::string::npos);
  ASSERT_NE(ipJsonPos, std::string::npos);
  EXPECT_LT(oopJsonPos, ipJsonPos);
}
