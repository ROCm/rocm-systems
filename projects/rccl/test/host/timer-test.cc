/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/include/timer.h built with ENABLE_TIMER: the rdtsc calibration and
// the TIME_* accumulators. That arm needs <x86intrin.h> and __rdtsc, so the suite is x86-64 only.

#if defined(__x86_64__)

#define ENABLE_TIMER 1

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#include "../common/LogCapture.hpp"
#include "timer.h"

namespace {

using RcclUnitTesting::CaptureStdout;

class TimerMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::memset(counts, 0, sizeof(counts));
    std::memset(times, 0, sizeof(times));
  }
};

TEST_F(TimerMicrotest, GetTime_CalibratesOnceAndCountsMicroseconds) {
  double t0 = gettime();
  double freq0 = freq;
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  double t1 = gettime();

  EXPECT_GT(freq0, 0);
  EXPECT_EQ(freq, freq0);
  EXPECT_GE(t1 - t0, 15000);
  EXPECT_LT(t1 - t0, 2000000);
}

TEST_F(TimerMicrotest, StartStop_AccumulateCountsAndTimes) {
  TIME_START(2);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  TIME_STOP(2);
  TIME_START(2);
  TIME_STOP(2);
  TIME_START(3);
  TIME_CANCEL(3);

  EXPECT_EQ(counts[2], 2u);
  EXPECT_GE(times[2], 4000);
  EXPECT_EQ(counts[3], 0u);
}

// TIME_PRINT resets the counts but not the accumulated times.
TEST_F(TimerMicrotest, Print_ListsActiveSlotsAndResetsTheirCounts) {
  counts[1] = 2;
  times[1] = 10;
  counts[5] = 1;
  times[5] = 3;
  times[6] = 7;  // no count: not printed

  std::string out = CaptureStdout([] { TIME_PRINT("proxy"); });

  EXPECT_EQ(out, "proxy stats [1] 10/2 = 5 [5] 3/1 = 3\n");
  EXPECT_EQ(counts[1], 0u);
  EXPECT_EQ(counts[5], 0u);
  EXPECT_EQ(times[1], 10);
}

}  // namespace

#endif  // defined(__x86_64__)
