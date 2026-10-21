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

// GpuPartitions() (common/rdc_utils.h) gives each GPU index the socket and partition that
// RDC's partition IDs (gS.P) name; rdci discovery -i prints them.

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "common/rdc_utils.h"

namespace {

// GpuPartitions() takes, for each GPU index, the partition count rdc_get_num_partition()
// reports: its socket's count, 1 for a partitionable GPU in SPX mode, UINT16_MAX for a GPU that
// cannot be partitioned.
using Slots = std::vector<std::pair<uint32_t, int32_t>>;  // (socket, partition or -1)
constexpr uint16_t kNotPartitionable = UINT16_MAX;

Slots slots(const std::vector<uint16_t>& counts) {
  Slots out;
  for (const amd::rdc::GpuPartition& p : amd::rdc::GpuPartitions(counts)) {
    out.emplace_back(p.socket, p.partition);
  }
  return out;
}

TEST(GpuPartitions, OneCpxSocketThenSpxSockets) {
  // The MI350X of F27's run: socket 0 in CPX holds GPU indexes 0-7, sockets 1-7 one each.
  std::vector<uint16_t> counts(8, 8);
  counts.insert(counts.end(), 7, 1);
  Slots want;
  for (int32_t k = 0; k < 8; ++k) {
    want.emplace_back(0, k);
  }
  for (uint32_t s = 1; s <= 7; ++s) {
    want.emplace_back(s, -1);
  }
  EXPECT_EQ(slots(counts), want);
}

TEST(GpuPartitions, SpxSocketThenPartitionedSockets) {
  EXPECT_EQ(slots({1, 2, 2, 4, 4, 4, 4}),
            (Slots{{0, -1}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {2, 2}, {2, 3}}));
}

TEST(GpuPartitions, GpusWithoutPartitions) {
  EXPECT_EQ(slots({kNotPartitionable, kNotPartitionable, 0, 1}),
            (Slots{{0, -1}, {1, -1}, {2, -1}, {3, -1}}));
}

TEST(GpuPartitions, ACountPastTheListsEnd) { EXPECT_EQ(slots({4, 4}), (Slots{{0, 0}, {0, 1}})); }

TEST(GpuPartitions, AnotherCountEndsASocket) {
  EXPECT_EQ(slots({2, 4, 4, 4, 4, 2, 2}),
            (Slots{{0, 0}, {1, 0}, {1, 1}, {1, 2}, {1, 3}, {2, 0}, {2, 1}}));
}

TEST(GpuPartitions, NoGpus) { EXPECT_TRUE(slots({}).empty()); }

}  // namespace
