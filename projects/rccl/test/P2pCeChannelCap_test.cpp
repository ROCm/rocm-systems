/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include "graph.h"

TEST(P2pCeChannelCapTests, Gfx1250CeDefaultsToSafePerPeerCount)
{
    EXPECT_EQ(rcclP2pCeChannelsPerPeerCap(/*current=*/64, /*pool=*/128, true, true, /*configuredCap=*/16), 16);
}

TEST(P2pCeChannelCapTests, OtherArchitecturesAndKernelP2pAreUnchanged)
{
    EXPECT_EQ(rcclP2pCeChannelsPerPeerCap(/*current=*/64, /*pool=*/128, false, true, /*configuredCap=*/2), 64);
    EXPECT_EQ(rcclP2pCeChannelsPerPeerCap(/*current=*/64, /*pool=*/128, true, false, /*configuredCap=*/2), 64);
}

TEST(P2pCeChannelCapTests, NonPositiveConfigurationDisablesTheCap)
{
    EXPECT_EQ(rcclP2pCeChannelsPerPeerCap(/*current=*/64, /*pool=*/128, true, true, /*configuredCap=*/0), 64);
}

TEST(P2pCeChannelCapTests, OverrideIsPowerOfTwoAndBoundedByPool)
{
    EXPECT_EQ(rcclP2pCeChannelsPerPeerCap(/*current=*/64, /*pool=*/128, true, true, /*configuredCap=*/7), 4);
    EXPECT_EQ(rcclP2pCeChannelsPerPeerCap(/*current=*/64, /*pool=*/2, true, true, /*configuredCap=*/8), 2);
    EXPECT_EQ(rcclP2pCeChannelsPerPeerCap(/*current=*/8, /*pool=*/128, true, true, /*configuredCap=*/16), 8);
    EXPECT_EQ(rcclP2pCeChannelsPerPeerCap(/*current=*/64, /*pool=*/128, true, true,
                                         /*configuredCap=*/int64_t{1} << 40),
              64);
}

TEST(P2pCeChannelCapTests, OnlySendRecvConnectionsUseTheCopyEngine)
{
    EXPECT_TRUE(rcclP2pUseCeForConnection(/*useMemcpy=*/true, NCCL_CONN_IDX_P2P_CE));
    EXPECT_FALSE(rcclP2pUseCeForConnection(/*useMemcpy=*/true, /*kernel P2P=*/1));
    EXPECT_FALSE(rcclP2pUseCeForConnection(/*useMemcpy=*/false, NCCL_CONN_IDX_P2P_CE));
}

TEST(P2pCeChannelCapTests, Gfx1250CeStepOnlyGrowsTheDefaultStep)
{
    EXPECT_EQ(rcclP2pCeStepSizePolicy(/*cudaArch=*/1250, true, 4 * 1024 * 1024, 512 * 1024), 4 * 1024 * 1024);
    EXPECT_EQ(rcclP2pCeStepSizePolicy(/*cudaArch=*/1250, true, 0, 512 * 1024), 512 * 1024);
    EXPECT_EQ(rcclP2pCeStepSizePolicy(/*cudaArch=*/1250, true, 256 * 1024, 512 * 1024), 512 * 1024);
    EXPECT_EQ(rcclP2pCeStepSizePolicy(/*cudaArch=*/1200, true, 4 * 1024 * 1024, 512 * 1024), 512 * 1024);
    EXPECT_EQ(rcclP2pCeStepSizePolicy(/*cudaArch=*/1250, false, 4 * 1024 * 1024, 512 * 1024), 512 * 1024);
}

TEST(P2pCeChannelCapTests, Gfx1250CeStepIsCappedAt32MiB)
{
    EXPECT_EQ(rcclP2pCeStepSizePolicy(/*cudaArch=*/1250, true, 32 * 1024 * 1024, 512 * 1024), 32 * 1024 * 1024);
    EXPECT_EQ(rcclP2pCeStepSizePolicy(/*cudaArch=*/1250, true, 512 * 1024 * 1024, 512 * 1024), 32 * 1024 * 1024);
    EXPECT_EQ(rcclP2pCeStepSizePolicy(/*cudaArch=*/1250, true, int64_t(1) << 40, 512 * 1024), 32 * 1024 * 1024);
    EXPECT_EQ(rcclP2pCeStepSizePolicy(/*cudaArch=*/1250, true, 4 * 1024 * 1024, 128 * 1024 * 1024),
              32 * 1024 * 1024);
}

TEST(P2pCeChannelCapTests, Gfx1250CeConnectionsShareTheProxyStream)
{
    EXPECT_TRUE(rcclP2pCeSharesProxyStream(/*cudaArch=*/1250));
    EXPECT_FALSE(rcclP2pCeSharesProxyStream(/*cudaArch=*/1200));
}

TEST(P2pCeChannelCapTests, SimpleChunkFollowsCopyEngineConnectorSteps)
{
    const int chunk = 512 * 1024;
    EXPECT_EQ(rcclP2pConnectorStepSize(chunk, /*useCe=*/true, /*connStepSize=*/4 * 1024 * 1024),
              4 * 1024 * 1024);
    EXPECT_EQ(rcclP2pConnectorStepSize(chunk, /*useCe=*/true, /*connStepSize=*/chunk), chunk);
    EXPECT_EQ(rcclP2pConnectorStepSize(/*p2pChunkSize=*/256 * 1024, /*useCe=*/false,
                                      /*connStepSize=*/4 * 1024 * 1024), 256 * 1024);
    EXPECT_EQ(rcclP2pConnectorStepSize(chunk, /*useCe=*/true, /*connStepSize=*/0), chunk);
}
