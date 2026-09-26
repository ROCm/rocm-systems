/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "rocm_smi/rocm_smi_gpu_metrics.h"

namespace {

using amd::smi::widen_keep_sentinel;

constexpr auto kU32Max = std::numeric_limits<uint32_t>::max();
constexpr auto kU64Max = std::numeric_limits<uint64_t>::max();

// The regression: an "unset" 32-bit field copied into a 64-bit public field
// used to zero-extend to 0x00000000FFFFFFFF, which callers report as a real
// reading of 4294967295 instead of "N/A".
TEST(GpuUnit, MetricsSentinelWidensTheUnsetSentinel) {
  EXPECT_EQ(widen_keep_sentinel<uint64_t>(kU32Max), kU64Max);
  EXPECT_EQ(widen_keep_sentinel<uint64_t>(std::numeric_limits<uint16_t>::max()), kU64Max);
  EXPECT_EQ(widen_keep_sentinel<uint32_t>(std::numeric_limits<uint8_t>::max()), kU32Max);
}

TEST(GpuUnit, MetricsSentinelPreservesOrdinaryValuesWhenWidening) {
  EXPECT_EQ(widen_keep_sentinel<uint64_t>(uint32_t{0}), 0u);
  EXPECT_EQ(widen_keep_sentinel<uint64_t>(uint32_t{42}), 42u);
  EXPECT_EQ(widen_keep_sentinel<uint64_t>(kU32Max - 1), uint64_t{kU32Max} - 1);
}

// Only a genuine width increase remaps the sentinel; a same-width copy must
// still pass an all-ones reading through untouched.
TEST(GpuUnit, MetricsSentinelSameWidthCopyIsUnchanged) {
  EXPECT_EQ(widen_keep_sentinel<uint64_t>(kU64Max), kU64Max);
  EXPECT_EQ(widen_keep_sentinel<uint32_t>(kU32Max), kU32Max);
  EXPECT_EQ(widen_keep_sentinel<uint32_t>(uint32_t{7}), 7u);
}

TEST(GpuUnit, MetricsSentinelNarrowingIsAPlainCast) {
  EXPECT_EQ(widen_keep_sentinel<uint32_t>(kU64Max), kU32Max);
  EXPECT_EQ(widen_keep_sentinel<uint8_t>(uint32_t{0x1FF}), uint8_t{0xFF});
}

// Usable in constant expressions, so a mis-widened sentinel is a compile error
// rather than a runtime surprise.
TEST(GpuUnit, MetricsSentinelIsConstexpr) {
  static_assert(widen_keep_sentinel<uint64_t>(kU32Max) == kU64Max);
  static_assert(widen_keep_sentinel<uint64_t>(uint32_t{1}) == 1u);
  SUCCEED();
}

}  // namespace
