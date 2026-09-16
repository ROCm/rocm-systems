/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "rdc_lib/impl/SmiUtils.h"

namespace {
// firmware_timestamp is in 10ns units, so one firmware millisecond is 100000
// units. Tests express windows as a whole number of firmware milliseconds.
constexpr uint64_t kMsUnits = 100000;
}  // namespace

// Without a previous sample the derived value must equal the instantaneous
// UMC-activity reading (the pre-existing behavior on the first fetch).
TEST(MemCurBandwidthTest, FallsBackToUmcWithoutPreviousSample) {
  EXPECT_DOUBLE_EQ(20.0,
                   amd::rdc::derive_mem_activity_percent(20.0, false, 0, 0, 12345, 100 * kMsUnits));
}

// The DMA case: the instantaneous reading is zero but the accumulator advances,
// so a non-zero activity is derived from the accumulator delta over firmware
// time. 1000 counts over 1000 ms == 1.0%.
TEST(MemCurBandwidthTest, AccumulatorCapturesTrafficWhenUmcIsZero) {
  EXPECT_NEAR(1.0, amd::rdc::derive_mem_activity_percent(0.0, true, 0, 0, 1000, 1000 * kMsUnits),
              1e-9);
}

// Under a steady compute workload the accumulator delta matches the
// instantaneous reading. 40000 counts over 2000 ms == 20.0%.
TEST(MemCurBandwidthTest, AccumulatorMatchesInstantaneousUnderSteadyLoad) {
  EXPECT_NEAR(20.0,
              amd::rdc::derive_mem_activity_percent(20.0, true, 100000, 500 * kMsUnits, 140000,
                                                    2500 * kMsUnits),
              1e-9);
}

// A tiny accumulator delta must never lower the reported value below the
// instantaneous reading.
TEST(MemCurBandwidthTest, NeverBelowInstantaneous) {
  EXPECT_DOUBLE_EQ(50.0,
                   amd::rdc::derive_mem_activity_percent(50.0, true, 0, 0, 100, 1000 * kMsUnits));
}

// A non-advancing firmware clock is ignored (no stale delta, no divide by zero).
TEST(MemCurBandwidthTest, IgnoresStalledFirmwareClock) {
  EXPECT_DOUBLE_EQ(
      7.0, amd::rdc::derive_mem_activity_percent(7.0, true, 0, 5 * kMsUnits, 9999, 5 * kMsUnits));
}

// A decreasing accumulator (counter reset or wrap) is ignored.
TEST(MemCurBandwidthTest, IgnoresAccumulatorReset) {
  EXPECT_DOUBLE_EQ(3.0,
                   amd::rdc::derive_mem_activity_percent(3.0, true, 5000, 0, 100, 1000 * kMsUnits));
}

// The gpu_metrics "not supported" sentinel (max uint64) must never be treated as
// real accumulator or timestamp data; the helper falls back to the instantaneous
// reading instead of a bogus clamped-to-100% value.
TEST(MemCurBandwidthTest, RejectsUnsupportedSentinel) {
  constexpr uint64_t kNotSupported = std::numeric_limits<uint64_t>::max();
  // Sentinel accumulator after a valid cached sample.
  EXPECT_DOUBLE_EQ(5.0, amd::rdc::derive_mem_activity_percent(5.0, true, 100, 1 * kMsUnits,
                                                              kNotSupported, 1000 * kMsUnits));
  // Sentinel firmware timestamp.
  EXPECT_DOUBLE_EQ(
      5.0, amd::rdc::derive_mem_activity_percent(5.0, true, 100, 1 * kMsUnits, 200, kNotSupported));
  // Sentinel in the cached (previous) sample.
  EXPECT_DOUBLE_EQ(5.0, amd::rdc::derive_mem_activity_percent(5.0, true, kNotSupported, 0, 200,
                                                              1000 * kMsUnits));
}

// A sample window that exceeds the staleness bound (for example a watcher that
// paused for a long time) must be ignored so a recent burst is not diluted
// across an unbounded gap; the helper falls back to the instantaneous reading.
TEST(MemCurBandwidthTest, IgnoresStaleWindowBeyondCap) {
  // ~1 hour window with a large accumulator delta that would otherwise clamp to
  // 100%; the staleness cap rejects it and the instantaneous reading is returned.
  const uint64_t one_hour_ms = 3600ULL * 1000ULL;
  EXPECT_DOUBLE_EQ(5.0, amd::rdc::derive_mem_activity_percent(5.0, true, 0, 0, 999999999,
                                                              one_hour_ms * kMsUnits));
}

// The derived percentage is clamped to 100.
TEST(MemCurBandwidthTest, ClampsToOneHundredPercent) {
  EXPECT_DOUBLE_EQ(100.0,
                   amd::rdc::derive_mem_activity_percent(0.0, true, 0, 0, 999999, 1000 * kMsUnits));
}
