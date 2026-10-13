/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <memory>

#include "comm.h"
#include "device.h"
#include "enqueue.h"
#include "nccl.h"

TEST(P2pDirectMaxChannelsTests, DirectApisUseRequestedPoolWhenEnabled)
{
    auto comm                    = std::make_unique<ncclComm>();
    comm->p2pnChannels           = 64;
    comm->p2pnChannelsPerPeer    = 16;
    comm->workArgsBytes          = sizeof(ncclDevKernelArgs) + 64 * sizeof(ncclDevWorkBatch);

    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSend, 128), 64);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncRecv, 128), 64);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, 128), 64);
}

TEST(P2pDirectMaxChannelsTests, DisabledAndP2pBackedCollectivesKeepPeerSplit)
{
    auto comm                    = std::make_unique<ncclComm>();
    comm->p2pnChannels           = 64;
    comm->p2pnChannelsPerPeer    = 16;
    comm->workArgsBytes          = sizeof(ncclDevKernelArgs) + 64 * sizeof(ncclDevWorkBatch);

    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSend, 0), 16);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncAlltoAll, 128), 16);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncGather, 128), 16);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, 128, /*useMemcpy=*/true), 16);
}

TEST(P2pDirectMaxChannelsTests, RequestedPoolRespectsKernelArgsBudget)
{
    auto comm                    = std::make_unique<ncclComm>();
    comm->p2pnChannels           = 64;
    comm->p2pnChannelsPerPeer    = 16;
    comm->workArgsBytes          = sizeof(ncclDevKernelArgs) + 32 * sizeof(ncclDevWorkBatch);

    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, 128), 32);
}

TEST(P2pDirectMaxChannelsTests, RequestedPoolIsCappedAtValidatedMaximum)
{
    auto comm                    = std::make_unique<ncclComm>();
    comm->p2pnChannels           = 256;
    comm->p2pnChannelsPerPeer    = 64;
    comm->workArgsBytes          = sizeof(ncclDevKernelArgs) + 256 * sizeof(ncclDevWorkBatch);

    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, 64), 64);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, 256), 128);
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
TEST(P2pDirectMaxChannelsTests, DefaultKernelArgsBudgetFits256ChannelBatches)
{
    EXPECT_GE(ncclDevMaxChannelsForArgsBytes(ncclMaxKernelArgsSize(1250)), 256);
}
#endif
