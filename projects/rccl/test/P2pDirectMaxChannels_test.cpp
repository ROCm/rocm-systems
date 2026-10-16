/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "comm.h"
#include "device.h"
#include "enqueue.h"
#include "nccl.h"

namespace {
constexpr int kPeer = 1;

struct TwoRankComm {
    std::unique_ptr<ncclComm> comm = std::make_unique<ncclComm>();
    std::vector<ncclPeerInfo> peerInfo = std::vector<ncclPeerInfo>(2);

    TwoRankComm(int p2pnChannels, int p2pnChannelsPerPeer, int argsBatches)
    {
        comm->rank                = 0;
        comm->nRanks              = 2;
        comm->p2pnChannels        = p2pnChannels;
        comm->p2pnChannelsPerPeer = p2pnChannelsPerPeer;
        comm->workArgsBytes       = sizeof(ncclDevKernelArgs) + argsBatches * sizeof(ncclDevWorkBatch);
        peerInfo[0].hostHash      = 0x1234;
        peerInfo[1].hostHash      = 0x1234;
        comm->peerInfo            = peerInfo.data();
    }
    ncclComm* get() { return comm.get(); }
};
} // namespace

TEST(P2pDirectMaxChannelsTests, DirectApisUseRequestedPoolWhenEnabled)
{
    TwoRankComm comm(64, 16, 80);

    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSend, kPeer, 128), 64);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncRecv, kPeer, 128), 64);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, kPeer, 128), 64);
}

TEST(P2pDirectMaxChannelsTests, DisabledAndP2pBackedCollectivesKeepPeerSplit)
{
    TwoRankComm comm(64, 16, 64);

    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSend, kPeer, 0), 16);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncAlltoAll, kPeer, 128), 16);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncGather, kPeer, 128), 16);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, kPeer, 128, /*useMemcpy=*/true), 16);
}

TEST(P2pDirectMaxChannelsTests, OnlyPeersOnTheSameHostOrMnnvlCliqueAreWidened)
{
    TwoRankComm comm(64, 16, 80);
    comm.peerInfo[kPeer].hostHash = 0x5678;

    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSend, kPeer, 128), 16);

    int cliqueRanks[]        = {0, kPeer};
    comm.get()->MNNVL        = 1;
    comm.get()->clique.size  = 2;
    comm.get()->clique.ranks = cliqueRanks;
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSend, kPeer, 128), 64);

    comm.get()->clique.size = 1;
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSend, kPeer, 128), 16);
    comm.get()->clique.ranks = nullptr;
}

TEST(P2pDirectMaxChannelsTests, SelfTransfersKeepThePerPeerTile)
{
    TwoRankComm comm(256, 64, 256);

    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSend, /*peer=*/0, 128), 64);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncRecv, /*peer=*/0, 128), 64);
}

TEST(P2pDirectMaxChannelsTests, RequestedPoolLeavesRoomForASplitRound)
{
    // A round split into a widened and a per-peer work needs both tiles in one empty plan.
    TwoRankComm comm(64, 16, 48);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, kPeer, 128), 32);

    TwoRankComm tight(64, 16, 32);
    EXPECT_EQ(rcclP2pChannelStrideForApi(tight.get(), ncclFuncSendRecv, kPeer, 128), 16);
}

TEST(P2pDirectMaxChannelsTests, ShiftedMultiNodeMappingCapsTheWidenedTile)
{
    // With more than two nodes and a shift size, only p2pnChannels >> shift parts map to distinct
    // channels.
    TwoRankComm comm(128, 16, 256);
    comm.get()->nNodes              = 4;
    comm.get()->p2pChannelShiftSize = 1;
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, kPeer, 128), 64);

    comm.get()->p2pChannelShiftSize = -1;  // bit-reversed mapping
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, kPeer, 128), 128);
}

TEST(P2pDirectMaxChannelsTests, RequestedPoolIsCappedAtValidatedMaximum)
{
    TwoRankComm comm(256, 64, 256);

    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, kPeer, 64), 64);
    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, kPeer, 256), 128);
}

TEST(P2pDirectMaxChannelsTests, RequestedPoolNeverNarrowsThePeerTile)
{
    TwoRankComm comm(256, 64, 256);

    EXPECT_EQ(rcclP2pChannelStrideForApi(comm.get(), ncclFuncSendRecv, kPeer, 32), 64);
}

TEST(P2pDirectMaxChannelsTests, WidenedTransfersUseThePeerTileOrTheWholeWidenedTile)
{
    // Counts up to the per-peer tile keep their channels; counts between the two tiles fall back
    // to the per-peer tile, so the peer never needs channels outside the two tiles it connected.
    EXPECT_EQ(rcclP2pDirectTileChannels(16, 32, 128), 16);
    EXPECT_EQ(rcclP2pDirectTileChannels(32, 32, 128), 32);
    EXPECT_EQ(rcclP2pDirectTileChannels(64, 32, 128), 32);
    EXPECT_EQ(rcclP2pDirectTileChannels(127, 32, 128), 32);
    EXPECT_EQ(rcclP2pDirectTileChannels(128, 32, 128), 128);
    // Without widening the count is untouched.
    EXPECT_EQ(rcclP2pDirectTileChannels(48, 64, 64), 48);
}

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
TEST(P2pDirectMaxChannelsTests, DefaultKernelArgsBudgetFits256ChannelBatches)
{
    EXPECT_GE(ncclDevMaxChannelsForArgsBytes(ncclMaxKernelArgsSize(1250)), 256);
}
#endif
