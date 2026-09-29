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
  EXPECT_EQ(rcclGfx1250SendRecvLl128MaxBytes(kGfx1250, 1, 4), 1 << 20);
  EXPECT_EQ(rcclGfx1250SendRecvLl128MaxBytes(kGfx1250, 1, 8), 512 << 10);
  EXPECT_EQ(rcclGfx1250SendRecvLl128MaxBytes(kGfx1250, 1, 16), 256 << 10);
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

TEST(Gfx1250SendRecvLl128WindowTests, AutoProtocolBySize)
{
  constexpr ssize_t hi4 = 1 << 20;
  EXPECT_EQ(rcclGfx1250SendRecvAutoProtocol(2048, hi4, true, true), NCCL_PROTO_LL);
  EXPECT_EQ(rcclGfx1250SendRecvAutoProtocol(4095, hi4, true, true), NCCL_PROTO_LL);
  EXPECT_EQ(rcclGfx1250SendRecvAutoProtocol(4096, hi4, true, true), NCCL_PROTO_LL128);
  EXPECT_EQ(rcclGfx1250SendRecvAutoProtocol(8192, hi4, true, true), NCCL_PROTO_LL128);
  EXPECT_EQ(rcclGfx1250SendRecvAutoProtocol(hi4, hi4, true, true), NCCL_PROTO_LL128);
  EXPECT_EQ(rcclGfx1250SendRecvAutoProtocol(hi4 + 1, hi4, true, true), NCCL_PROTO_SIMPLE);
}

TEST(Gfx1250SendRecvLl128WindowTests, MissingLl128StagingFallsToSimpleNotLl)
{
  constexpr ssize_t hi4 = 1 << 20;
  EXPECT_EQ(rcclGfx1250SendRecvAutoProtocol(8192, hi4, true, false), NCCL_PROTO_SIMPLE);
}

TEST(Gfx1250SendRecvLl128WindowTests, MixedRoundSizesAreDifferentLlFamilies)
{
  constexpr ssize_t hi4 = 1 << 20;
  int const recv2k = rcclGfx1250SendRecvAutoProtocol(2048, hi4, true, true);
  int const send8k = rcclGfx1250SendRecvAutoProtocol(8192, hi4, true, true);
  EXPECT_EQ(recv2k, NCCL_PROTO_LL);
  EXPECT_EQ(send8k, NCCL_PROTO_LL128);
  EXPECT_TRUE(rcclP2pLlFamilyMix(recv2k, send8k));
  EXPECT_FALSE(rcclP2pLlFamilyMix(send8k, send8k));
  EXPECT_FALSE(rcclP2pLlFamilyMix(NCCL_PROTO_SIMPLE, NCCL_PROTO_LL128));
  EXPECT_FALSE(rcclP2pLlFamilyMix(NCCL_PROTO_LL, NCCL_PROTO_SIMPLE));
}

TEST(Gfx1250SendRecvLl128WindowTests, AllocP2pNetLlBuffersFormula)
{
  EXPECT_EQ(rcclAllocP2pNetLLBuffers(1250, 1, 4, -1, 0), 0);
  EXPECT_EQ(rcclAllocP2pNetLLBuffers(1250, 1, 8, -1, 0), 1);
  EXPECT_EQ(rcclAllocP2pNetLLBuffers(1250, 1, 16, -1, 0), 1);
  EXPECT_EQ(rcclAllocP2pNetLLBuffers(1250, 1, 2, -1, 0), 0);
  EXPECT_EQ(rcclAllocP2pNetLLBuffers(1250, 1, 8, 0, 0), 0);
  EXPECT_EQ(rcclAllocP2pNetLLBuffers(1250, 1, 2, 1, 0), 1);
  EXPECT_EQ(rcclAllocP2pNetLLBuffers(1250, 1, 4, 0, 1), 1);
  EXPECT_EQ(rcclAllocP2pNetLLBuffers(950, 1, 8, -1, 0), 0);
}

} // namespace RcclUnitTesting
