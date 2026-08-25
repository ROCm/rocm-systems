/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for rccl-tests device-timing helpers in gin_sdma_devtime_host.h
// (BenchTime zero-time guards and gin_sdma_devtime.h stamp-reduction math).
// No GPU or MPI required.

#include <gtest/gtest.h>

#include "gin_sdma_devtime_host.h"

using rccl_tests_devtime::applyCudaGraphLaunchesScale;
using rccl_tests_devtime::gridBusyWindowPerIterUs;
using rccl_tests_devtime::normalizeElapsedPerIter;
using rccl_tests_devtime::reduceMaxEnd;
using rccl_tests_devtime::reduceMinStart;
using rccl_tests_devtime::shouldComputeIterStats;
using rccl_tests_devtime::shouldSkipBenchTimeRow;

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
  EXPECT_FALSE(shouldComputeIterStats(5, 10));
}

TEST(GinSdmaDevtimeHost, GridBusyWindowReduction) {
  const long long starts[] = {100, 50, 80};
  const long long ends[] = {250, 300, 200};
  EXPECT_EQ(reduceMinStart(starts, 3), 50);
  EXPECT_EQ(reduceMaxEnd(ends, 3), 300);
  // (300-50)/100 kHz * 1e3 = 2500 us total; loop=10 -> 250 us/iter
  EXPECT_DOUBLE_EQ(gridBusyWindowPerIterUs(50, 300, 100, 10), 250.0);
  EXPECT_DOUBLE_EQ(gridBusyWindowPerIterUs(50, 300, 100, 0), 2500.0);
  EXPECT_DOUBLE_EQ(gridBusyWindowPerIterUs(50, 300, 0, 10), 0.0);
}

TEST(GinSdmaDevtimeHost, ZeroNormalizedDeltaWouldBeSkipped) {
  const double deltaSec = applyCudaGraphLaunchesScale(
      normalizeElapsedPerIter(0.0, 20, 1), 1);
  EXPECT_TRUE(shouldSkipBenchTimeRow(deltaSec));
}
