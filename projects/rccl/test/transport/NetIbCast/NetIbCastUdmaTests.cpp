/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Host-only unit tests for the dynamic UDMA count discovery and
 * round-robin QP-to-UDMA pinning logic in net_ib_cast/connect.cc.
 *
 * These tests exercise the pure-logic round-robin distribution via
 * the ncclIbCastTestUdmaRoundRobin() inspect API — no MPI, no RDMA
 * verbs, no NIC hardware required.
 */

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

#include "net_ib_cast_inspect.h"

namespace {

// =====================================================================
// 1. Basic round-robin with udmaCount=2 (legacy behavior)
// =====================================================================

TEST(NetIbCastUdma, RoundRobin2Engines) {
  const uint8_t udmaCount = 2;
  const int nChannels = 8;
  uint8_t masks[8] = {};

  ASSERT_EQ(ncclIbCastTestUdmaRoundRobin(udmaCount, nChannels, masks), ncclSuccess);

  // Expect alternating: 0x01, 0x02, 0x01, 0x02, ...
  for (int i = 0; i < nChannels; i++) {
    uint8_t expected = 1u << (i % 2);
    EXPECT_EQ(masks[i], expected) << "channel " << i;
  }
}

// =====================================================================
// 2. Round-robin with udmaCount=4
// =====================================================================

TEST(NetIbCastUdma, RoundRobin4Engines) {
  const uint8_t udmaCount = 4;
  const int nChannels = 12;
  uint8_t masks[12] = {};

  ASSERT_EQ(ncclIbCastTestUdmaRoundRobin(udmaCount, nChannels, masks), ncclSuccess);

  // Expect cycling: 0x01, 0x02, 0x04, 0x08, 0x01, 0x02, ...
  for (int i = 0; i < nChannels; i++) {
    uint8_t expected = 1u << (i % 4);
    EXPECT_EQ(masks[i], expected) << "channel " << i;
  }
}

// =====================================================================
// 3. Round-robin with udmaCount=1 (single engine — all QPs same mask)
// =====================================================================

TEST(NetIbCastUdma, RoundRobin1Engine) {
  const uint8_t udmaCount = 1;
  const int nChannels = 5;
  uint8_t masks[5] = {};

  ASSERT_EQ(ncclIbCastTestUdmaRoundRobin(udmaCount, nChannels, masks), ncclSuccess);

  // All channels should get mask=0x01 (the only engine)
  for (int i = 0; i < nChannels; i++) {
    EXPECT_EQ(masks[i], 0x01u) << "channel " << i;
  }
}

// =====================================================================
// 4. Round-robin with udmaCount=8 (maximum practical engines)
// =====================================================================

TEST(NetIbCastUdma, RoundRobin8Engines) {
  const uint8_t udmaCount = 8;
  const int nChannels = 16;
  uint8_t masks[16] = {};

  ASSERT_EQ(ncclIbCastTestUdmaRoundRobin(udmaCount, nChannels, masks), ncclSuccess);

  for (int i = 0; i < nChannels; i++) {
    uint8_t expected = 1u << (i % 8);
    EXPECT_EQ(masks[i], expected) << "channel " << i;
  }
}

// =====================================================================
// 5. Every UDMA engine is used at least once (balance check)
// =====================================================================

TEST(NetIbCastUdma, AllEnginesUsed) {
  for (uint8_t udmaCount = 1; udmaCount <= 8; udmaCount++) {
    int nChannels = udmaCount * 3;  // enough to cycle through at least 3x
    std::vector<uint8_t> masks(nChannels, 0);

    ASSERT_EQ(ncclIbCastTestUdmaRoundRobin(udmaCount, nChannels, masks.data()), ncclSuccess)
      << "udmaCount=" << (int)udmaCount;

    // Count how many times each engine appears
    std::vector<int> counts(udmaCount, 0);
    for (int i = 0; i < nChannels; i++) {
      // Verify mask is a power of two
      ASSERT_NE(masks[i], 0u);
      ASSERT_EQ(masks[i] & (masks[i] - 1), 0u) << "mask not power of two at ch " << i;

      // Find which engine it is
      uint8_t engine = 0;
      uint8_t m = masks[i];
      while (m >>= 1) engine++;
      ASSERT_LT(engine, udmaCount);
      counts[engine]++;
    }

    // With nChannels = udmaCount*3, each engine should appear exactly 3 times
    for (uint8_t e = 0; e < udmaCount; e++) {
      EXPECT_EQ(counts[e], 3) << "udmaCount=" << (int)udmaCount << " engine=" << (int)e;
    }
  }
}


// =====================================================================
// 7. Error handling: null pointer, zero udmaCount, zero channels
// =====================================================================

TEST(NetIbCastUdma, InvalidArgs) {
  uint8_t masks[4] = {};

  // null outMasks
  EXPECT_EQ(ncclIbCastTestUdmaRoundRobin(2, 4, nullptr), ncclInvalidArgument);
  // zero udmaCount
  EXPECT_EQ(ncclIbCastTestUdmaRoundRobin(0, 4, masks), ncclInvalidArgument);
  // zero or negative channels
  EXPECT_EQ(ncclIbCastTestUdmaRoundRobin(2, 0, masks), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastTestUdmaRoundRobin(2, -1, masks), ncclInvalidArgument);
}

// =====================================================================
// 8. Backward compatibility: 2-engine distribution matches legacy
//    IONIC_UDMA_MASK_LOW (0x01) / IONIC_UDMA_MASK_HIGH (0x02) pattern
// =====================================================================

TEST(NetIbCastUdma, BackwardCompatWith2Engines) {
  const uint8_t IONIC_UDMA_MASK_LOW = 0x01;
  const uint8_t IONIC_UDMA_MASK_HIGH = 0x02;
  const int nChannels = 6;
  uint8_t masks[6] = {};

  ASSERT_EQ(ncclIbCastTestUdmaRoundRobin(2, nChannels, masks), ncclSuccess);

  // Verify the legacy alternating pattern
  for (int i = 0; i < nChannels; i++) {
    uint8_t expected = (i % 2 == 0) ? IONIC_UDMA_MASK_LOW : IONIC_UDMA_MASK_HIGH;
    EXPECT_EQ(masks[i], expected) << "channel " << i;
  }
}

// =====================================================================
// 9. Normalize helper: zero from driver falls back to 2
// =====================================================================

TEST(NetIbCastUdma, NormalizeUdmaCountZeroFallback) {
  // Zero (driver unavailable or too old) → fallback to 2
  EXPECT_EQ(ncclIbCastNormalizeUdmaCount(0), 2);
  // Non-zero values within range pass through unchanged
  EXPECT_EQ(ncclIbCastNormalizeUdmaCount(1), 1);
  EXPECT_EQ(ncclIbCastNormalizeUdmaCount(2), 2);
  EXPECT_EQ(ncclIbCastNormalizeUdmaCount(4), 4);
  EXPECT_EQ(ncclIbCastNormalizeUdmaCount(8), 8);
  // Values exceeding uint8_t mask width are clamped to 8
  EXPECT_EQ(ncclIbCastNormalizeUdmaCount(9), 8);
  EXPECT_EQ(ncclIbCastNormalizeUdmaCount(255), 8);
}

// =====================================================================
// 10. Shared helper produces identical masks to production formula
// =====================================================================

TEST(NetIbCastUdma, SharedHelperMatchesProductionFormula) {
  // Verify that ncclIbCastUdmaRoundRobinMask produces the same result
  // as the inline formula 1u << (cursor % udmaCount) that was used
  // before the shared helper was extracted.
  for (uint8_t udmaCount : {1, 2, 3, 4, 8}) {
    uint8_t cursor = 0;
    for (int i = 0; i < 32; i++) {
      uint8_t expected = 1u << (i % udmaCount);
      uint8_t actual = ncclIbCastUdmaRoundRobinMask(udmaCount, &cursor);
      EXPECT_EQ(actual, expected)
        << "udmaCount=" << (int)udmaCount << " iteration=" << i;
    }
    // cursor should have wrapped around and be at 32 % udmaCount
    EXPECT_EQ(cursor, 32 % udmaCount)
      << "udmaCount=" << (int)udmaCount;
  }
}

// =====================================================================
// 11. Cross-channel round-robin: consecutive channels get distinct
//     engines, cycling through all udmaCount engines.
// =====================================================================

TEST(NetIbCastUdma, CrossChannelDistribution) {
  // Simulate what the production non-sharing path does: each new channel
  // gets the next engine in the round-robin sequence.  Verify that after
  // allocating udmaCount channels, all engines are used exactly once,
  // and the pattern repeats.
  for (uint8_t udmaCount : {1, 2, 4, 8}) {
    int nChannels = udmaCount * 3;
    std::vector<uint8_t> masks(nChannels, 0);
    ASSERT_EQ(ncclIbCastTestUdmaRoundRobin(udmaCount, nChannels, masks.data()),
              ncclSuccess);

    // Verify cross-channel distribution: within each full cycle of
    // udmaCount channels, every engine appears exactly once.
    for (int cycle = 0; cycle < 3; cycle++) {
      std::vector<bool> seen(udmaCount, false);
      for (uint8_t e = 0; e < udmaCount; e++) {
        uint8_t mask = masks[cycle * udmaCount + e];
        uint8_t engine = 0;
        for (uint8_t m = mask; m >>= 1; ) engine++;
        ASSERT_LT(engine, udmaCount);
        EXPECT_FALSE(seen[engine])
          << "udmaCount=" << (int)udmaCount
          << " cycle=" << cycle << " engine=" << (int)engine
          << " appeared twice";
        seen[engine] = true;
      }
      for (uint8_t e = 0; e < udmaCount; e++) {
        EXPECT_TRUE(seen[e])
          << "udmaCount=" << (int)udmaCount
          << " cycle=" << cycle << " engine=" << (int)e
          << " never used";
      }
    }
  }
}

}  // namespace
