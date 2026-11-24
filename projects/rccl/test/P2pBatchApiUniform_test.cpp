/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include "channel.h"

// P2P batching fuses rounds only for collectives whose per-peer sizes are the same on every
// rank, so all ranks make the same fusion decision and derive the same proxy opCounts.
TEST(P2pBatchApiUniformTests, RankUniformCollectivesFuse)
{
    EXPECT_TRUE(rcclP2pBatchApiUniform(ncclFuncAlltoAll));
    EXPECT_TRUE(rcclP2pBatchApiUniform(ncclFuncAllGather));
    EXPECT_TRUE(rcclP2pBatchApiUniform(ncclFuncReduceScatter));
}

TEST(P2pBatchApiUniformTests, SendRecvAndVariableSizeCollectivesDoNotFuse)
{
    // Send/recv (and alltoallv built on them) and the variable-size collectives can differ per rank.
    EXPECT_FALSE(rcclP2pBatchApiUniform(ncclFuncSend));
    EXPECT_FALSE(rcclP2pBatchApiUniform(ncclFuncRecv));
    EXPECT_FALSE(rcclP2pBatchApiUniform(ncclFuncSendRecv));
    EXPECT_FALSE(rcclP2pBatchApiUniform(ncclFuncAlltoAllv));
    EXPECT_FALSE(rcclP2pBatchApiUniform(ncclFuncAllGatherV));
    EXPECT_FALSE(rcclP2pBatchApiUniform(ncclFuncScatter));
    EXPECT_FALSE(rcclP2pBatchApiUniform(ncclFuncGather));
}
