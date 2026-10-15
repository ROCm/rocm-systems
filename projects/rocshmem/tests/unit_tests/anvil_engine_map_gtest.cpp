/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "gtest/gtest.h"
#include "../src/sdma/anvil_engine_map.hpp"

using namespace sdma_anvil;

namespace {

// MI300X engine counts as ROCr reports them.
constexpr uint32_t kSpxXgmi = 14;
constexpr uint32_t kSpxTotal = 16;
constexpr uint32_t kCpxXgmi = 0;
constexpr uint32_t kCpxTotal = 2;

}  // namespace

TEST(AnvilEngineMap, PartitionIsNoXgmiEnginesWithKnownTotal) {
  EXPECT_TRUE(isSdmaPartition(kCpxXgmi, kCpxTotal));
  EXPECT_FALSE(isSdmaPartition(kSpxXgmi, kSpxTotal));
  // A failed engine-count query reports a total of 0 and must not be read as a partition.
  EXPECT_FALSE(isSdmaPartition(0, 0));
}

TEST(AnvilEngineMap, PartitionFoldsEvenWhenDoubledIdIsInRange) {
  // Same-device CPX peers share an OAM id and hit the map diagonal, doubled id 0. Without the
  // partition fold they would all be pinned to engine 0.
  EXPECT_TRUE(oamMapEngineNeedsFold(kCpxXgmi, kCpxTotal, 0));
  EXPECT_TRUE(oamMapEngineNeedsFold(kCpxXgmi, kCpxTotal, 1));
}

TEST(AnvilEngineMap, SpxUsesInRangeIdsAsIs) {
  for (int id = 0; id < static_cast<int>(kSpxTotal); id += 2) {
    EXPECT_FALSE(oamMapEngineNeedsFold(kSpxXgmi, kSpxTotal, id)) << "id=" << id;
  }
}

TEST(AnvilEngineMap, OutOfRangeIdFolds) {
  EXPECT_TRUE(oamMapEngineNeedsFold(kSpxXgmi, kSpxTotal, static_cast<int>(kSpxTotal)));
  EXPECT_TRUE(oamMapEngineNeedsFold(2, 4, 12));
}

TEST(AnvilEngineMap, UnknownTotalNeverFolds) {
  EXPECT_FALSE(oamMapEngineNeedsFold(0, 0, 0));
  EXPECT_FALSE(oamMapEngineNeedsFold(0, 0, 12));
}

TEST(AnvilEngineMap, FoldStaysInRangeForEveryInput) {
  for (int oam = 0; oam < 8; ++oam) {
    for (int src = -1; src < 8; ++src) {
      for (int dst = -1; dst < 8; ++dst) {
        const int e = foldOamMapEngine(oam, src, dst, kCpxTotal);
        EXPECT_GE(e, 0);
        EXPECT_LT(e, static_cast<int>(kCpxTotal));
      }
    }
  }
}

TEST(AnvilEngineMap, FoldSeparatesPartitionsOfOneDevice) {
  // Diagonal (oam 0) pairs from function 0 to functions 0 and 1 land on different engines.
  EXPECT_NE(foldOamMapEngine(0, 0, 0, kCpxTotal), foldOamMapEngine(0, 0, 1, kCpxTotal));
}

TEST(AnvilEngineMap, FoldTreatsUnreadableFunctionAsZero) {
  // One unreadable side: without the clamp, (0 + -1 + 0) % 2 would be -1.
  EXPECT_EQ(foldOamMapEngine(0, -1, 0, kCpxTotal), foldOamMapEngine(0, 0, 0, kCpxTotal));
  EXPECT_EQ(foldOamMapEngine(0, 0, -1, kCpxTotal), foldOamMapEngine(0, 0, 0, kCpxTotal));
  EXPECT_EQ(foldOamMapEngine(3, -1, -1, kCpxTotal), foldOamMapEngine(3, 0, 0, kCpxTotal));
}

TEST(AnvilEngineMap, PciFunctionBusRewritesPartitionToFunctionZero) {
  const PciFunctionBus loc = pciFunctionBus("0000:05:00.1");
  EXPECT_EQ(loc.function, 1);
  EXPECT_EQ(loc.busId, "0000:05:00.1");
  EXPECT_EQ(loc.physBusId, "0000:05:00.0");
}

TEST(AnvilEngineMap, PciFunctionBusAcceptsOnlyThreeBitFunctions) {
  for (char c = '0'; c <= '7'; ++c) {
    EXPECT_EQ(pciFunctionBus(std::string("0000:05:00.") + c).function, c - '0');
  }
  for (const char* bad : {"0000:05:00.8", "0000:05:00.9", "0000:05:00.a"}) {
    const PciFunctionBus loc = pciFunctionBus(bad);
    EXPECT_EQ(loc.function, -1) << bad;
    EXPECT_EQ(loc.physBusId, loc.busId) << bad;
  }
}

TEST(AnvilEngineMap, GpuDoorbellMisroutedMatchesMi300xProbe) {
  // Every queue slot of every SPX engine, measured with a GPU-written doorbell on MI300X: the
  // first 16 bytes of each odd engine's range fail, everything else is serviced.
  for (uint32_t engine = 0; engine < kSpxTotal; ++engine) {
    const uint32_t base = kSdmaDoorbellBase + engine * kSdmaDoorbellStride;
    for (uint32_t q = 0; q < 8; ++q) {
      const uint64_t doorbell = base + (q & 1) * kDoorbellHalfPage + (q >> 1) * 8;
      EXPECT_EQ(gpuDoorbellMisrouted(doorbell), engine % 2 == 1 && q < 4)
          << "engine=" << engine << " q=" << q;
    }
  }
}

TEST(AnvilEngineMap, GpuDoorbellMisroutedUsesThePageOffsetOnly) {
  EXPECT_TRUE(gpuDoorbellMisrouted(0x7f3f3f428850ull));
  EXPECT_TRUE(gpuDoorbellMisrouted(0x7f3f3f429858ull));
  EXPECT_FALSE(gpuDoorbellMisrouted(0x7f3f3f428860ull));
  EXPECT_FALSE(gpuDoorbellMisrouted(0x7f3f3f428810ull));
  // Below the SDMA range: not an SDMA doorbell, never refused.
  EXPECT_FALSE(gpuDoorbellMisrouted(0x7f3f3f428010ull));
}

TEST(AnvilEngineMap, OddEnginesKeepHalfTheirSlots) {
  EXPECT_EQ(gpuReachableQueueSlots(0, 8), 8u);
  EXPECT_EQ(gpuReachableQueueSlots(1, 8), 4u);
  EXPECT_EQ(gpuReachableQueueSlots(14, 8), 8u);
  EXPECT_EQ(gpuReachableQueueSlots(15, 8), 4u);
  EXPECT_EQ(gpuReachableQueueSlots(1, 2), 0u);
}

TEST(AnvilEngineMap, QueueBudgetCountsReachableSlots) {
  EXPECT_EQ(sdmaQueueBudget(kSpxTotal, 8), 96u);
  EXPECT_EQ(sdmaQueueBudget(8, 8), 48u);  // DPX
  EXPECT_EQ(sdmaQueueBudget(4, 8), 24u);  // QPX
  EXPECT_EQ(sdmaQueueBudget(kCpxTotal, 8), 12u);
  EXPECT_EQ(sdmaQueueBudget(0, 8), 0u);
  EXPECT_EQ(sdmaQueueBudget(kCpxTotal, 0), 0u);
}

TEST(AnvilEngineMap, QueueBudgetRefusesOverSubscription) {
  // CPX: engine 0's 8 slots and engine 1's 4 reachable ones.
  const uint32_t cpx = sdmaQueueBudget(kCpxTotal, 8);
  EXPECT_FALSE(queueBudgetExceeded(5, 7, cpx));
  EXPECT_TRUE(queueBudgetExceeded(6, 7, cpx));
  EXPECT_TRUE(queueBudgetExceeded(0, 13, cpx));
}

TEST(AnvilEngineMap, QueueBudgetUnknownNeverRefuses) {
  EXPECT_FALSE(queueBudgetExceeded(100, 8, 0));
}

TEST(AnvilEngineMap, QueueBudgetIgnoresEmptyRequest) {
  EXPECT_FALSE(queueBudgetExceeded(16, 0, 12));
  EXPECT_FALSE(queueBudgetExceeded(16, -1, 12));
}

TEST(AnvilEngineMap, PciFunctionBusHandlesEmptyInput) {
  const PciFunctionBus loc = pciFunctionBus("");
  EXPECT_EQ(loc.function, -1);
  EXPECT_TRUE(loc.physBusId.empty());
}
