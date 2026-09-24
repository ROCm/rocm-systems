/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only tests for rcclGfx1250SendRecvLl128MaxBytes(), the gfx1250 SendRecv
// auto-window table used by enqueue.cc and init.cc. No comm and no GPU needed.

#include "gtest/gtest.h"
#include "rccl_common.h"

namespace RcclUnitTesting
{

static constexpr int kGfx1250 = 1250;
static constexpr int kGfx950 = 950;

TEST(Gfx1250SendRecvLl128WindowTests, NonGfx1250HasNoWindow)
{
  EXPECT_EQ(rcclGfx1250SendRecvLl128MaxBytes(kGfx950, 1, 4), 0);
  EXPECT_EQ(rcclGfx1250SendRecvLl128MaxBytes(0, 1, 8), 0);
}

TEST(Gfx1250SendRecvLl128WindowTests, RankCountsMapToCaps)
{
  EXPECT_EQ(rcclGfx1250SendRecvLl128MaxBytes(kGfx1250, 1, 4), 16 << 10);
  EXPECT_EQ(rcclGfx1250SendRecvLl128MaxBytes(kGfx1250, 1, 8), 256 << 10);
  EXPECT_EQ(rcclGfx1250SendRecvLl128MaxBytes(kGfx1250, 1, 16), 128 << 10);
}

TEST(Gfx1250SendRecvLl128WindowTests, OtherRankCountsHaveNoWindow)
{
  for (int nRanks : {2, 6, 12, 32}) {
    EXPECT_EQ(rcclGfx1250SendRecvLl128MaxBytes(kGfx1250, 1, nRanks), 0) << "nRanks=" << nRanks;
  }
}

TEST(Gfx1250SendRecvLl128WindowTests, nNodesArgumentDoesNotChangeCaps)
{
  // MNNVL reports nNodes=1 for multi-host gfx1250; the table is keyed on nRanks.
  EXPECT_EQ(rcclGfx1250SendRecvLl128MaxBytes(kGfx1250, 1, 8),
            rcclGfx1250SendRecvLl128MaxBytes(kGfx1250, 2, 8));
  EXPECT_EQ(rcclGfx1250SendRecvLl128MaxBytes(kGfx1250, 1, 16),
            rcclGfx1250SendRecvLl128MaxBytes(kGfx1250, 4, 16));
}

TEST(Gfx1250SendRecvLl128WindowTests, MinBytesIs4KiB)
{
  EXPECT_EQ(rcclGfx1250SendRecvLl128MinBytes, 4 << 10);
}

} // namespace RcclUnitTesting
