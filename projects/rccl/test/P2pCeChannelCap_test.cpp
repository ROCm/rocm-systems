/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include "graph.h"

TEST(P2pCeChannelCapTests, Gfx1250CeDefaultsToSafePerPeerCount)
{
    EXPECT_EQ(rcclP2pCeChannelsPerPeerCap(/*current=*/64, /*pool=*/128, true, true, /*configuredCap=*/2), 2);
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
}
