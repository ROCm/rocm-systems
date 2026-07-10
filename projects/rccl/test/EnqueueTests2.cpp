/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Tests for enqueue.cc that don't depend on GPUs
//

#include "nccl.h"
#include <gtest/gtest.h>
#include <hip/amd_detail/amd_hip_runtime.h>

#include ENQUEUE_CC_PATH

namespace RcclUnitTesting
{
namespace
{
    // Distinct min/max channel counts so the nChStart choice is observable.
    constexpr int kNChannelsMin = 4;
    constexpr int kNChannelsMax = 8;

    class AddP2pToPlanTest : public ::testing::Test
    {
    protected:
        // Value-initialised (zeroed) heap storage for the large POD structs,
        // released automatically at end of test.
        std::unique_ptr<ncclComm> commStorage;
        std::unique_ptr<ncclKernelPlan> planStorage;
        std::unique_ptr<ncclTopoSystem> topoStorage;

        ncclComm* comm = nullptr;
        ncclKernelPlan* plan = nullptr;

        void MakeComm(int nNodes)
        {
            commStorage = std::make_unique<ncclComm>();
            planStorage = std::make_unique<ncclKernelPlan>();
            topoStorage = std::make_unique<ncclTopoSystem>();
            comm = commStorage.get();
            plan = planStorage.get();
            ncclTopoSystem* topo = topoStorage.get();

            // Bump-allocator used by ncclMemoryStackAlloc* inside the function.
            ncclMemoryStackConstruct(&comm->memScoped);

            comm->rank = 0;
            comm->nNodes = nNodes;
            comm->maxLocalRanks = 1;
            comm->p2pnChannels = 8;         // power of two, >= nChannelsMax
            comm->p2pnChannelsPerPeer = 8;
            comm->p2pChannelShiftSize = 0;
            comm->p2pChunkSize = 1 << 16;   // 64 KiB
            comm->p2pNet = 0;
            for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
                comm->buffSizes[p] = NCCL_STEPS * comm->p2pChunkSize;

            // One GPU node so the multi-node arch checks have a gcn string.
            topo->nodes[GPU].count = 1;
            snprintf(topo->nodes[GPU].nodes[0].gpu.gcn, GCN_ARCH_NAME_LEN, "gfx942");
            comm->topo = topo;

            plan->comm = comm;

            // ncclDevFuncId_P2p() looks up the send/recv function id in this map;
            // an empty map yields -1 and an early ncclInvalidUsage. Register the
            // single key computed for (ncclFuncSendRecv, -1, -1, UNDEF, UNDEF).
            uint64_t p2pKey = (uint64_t)(ncclFuncSendRecv & RCCL_FUNC_ID_MASK) << RCCL_COLL_SHIFT;
            ncclDevFuncNameToId[p2pKey] = 0;
        }
    };

    // Parameters for one P2P channel-scaling scenario. bytes are per direction
    // where index 0 = recv, 1 = send; -1 marks a no-op (inactive) direction.
    struct P2pScalingCase
    {
        const char* name;
        int nNodes;
        int planTotalTasks[2];
        ssize_t bytes[2];         // {recvBytes, sendBytes}
        int expectedNChannels;    // expected nChannels on each active direction
    };

    class AddP2pToPlanScaling : public AddP2pToPlanTest,
                                public ::testing::WithParamInterface<P2pScalingCase> {};

    TEST_P(AddP2pToPlanScaling, SelectsExpectedChannelCount)
    {
        const P2pScalingCase& tc = GetParam();
        MakeComm(tc.nNodes);

        // Non-null task per active direction so we can read back the count.
        ncclTaskP2p recvTask = {};
        ncclTaskP2p sendTask = {};
        ncclTaskP2p* p2pTasks[2] = {
            tc.bytes[0] != -1 ? &recvTask : nullptr,
            tc.bytes[1] != -1 ? &sendTask : nullptr,
        };

        int planTotalTasks[2] = {tc.planTotalTasks[0], tc.planTotalTasks[1]};

        const int rank = comm->rank;
        ncclResult_t result = addP2pToPlan(
            comm, plan,
            /*nChannelsMin*/ kNChannelsMin, /*nChannelsMax*/ kNChannelsMax, /*p2pRound*/ 0,
            /*sendRank*/ rank, /*sendAddr*/ nullptr, /*sendBytes*/ tc.bytes[1],
            /*recvRank*/ rank, /*recvAddr*/ nullptr, /*recvBytes*/ tc.bytes[0],
            /*sendOpCount*/ 0, /*recvOpCount*/ 0,
            planTotalTasks, p2pTasks);

        EXPECT_EQ(result, ncclSuccess);

        if (p2pTasks[0]) EXPECT_EQ(recvTask.nChannels, tc.expectedNChannels);
        if (p2pTasks[1]) EXPECT_EQ(sendTask.nChannels, tc.expectedNChannels);
    }

    // The byte sizes below are chosen so the std::min + partSize doubling loop
    // lands on the intended value:
    //   * Single-node minPartSize = p2pChunkSize/8 = 8KiB, maxPartSize = 2MiB.
    //       4MiB   -> asymmetric picks nChannelsMax (8); old min-start gave 4.
    //       32KiB  -> divUp(32KiB,8KiB) = 4, clamps below nChannelsMax even with
    //                 an nChannelsMax start (small-message boundary).
    //   * Multi-node minPartSize = p2pChunkSize/2 = 32KiB, maxPartSize = 64KiB.
    //       256KiB -> min-start yields 4, an nChannelsMax start would give 8, so
    //                 this size distinguishes the guarded from the boosted result
    //                 (at 4MiB the doubling loop promotes both to 8).
    const P2pScalingCase kCases[] = {
        // Single-node asymmetric gather ({N,0}): recv-heavy -> nChannelsMax.
        {"Gather_RecvHeavy_UsesNChannelsMax",
         1, {kNChannelsMax, 0}, {4 * 1024 * 1024, -1}, kNChannelsMax},

        // Single-node asymmetric scatter ({0,N}): send-heavy -> nChannelsMax.
        {"Scatter_SendHeavy_UsesNChannelsMax",
         1, {0, kNChannelsMax}, {-1, 4 * 1024 * 1024}, kNChannelsMax},

        // Single-node symmetric alltoall ({N,N}) -> nChannelsMin.
        {"AllToAll_Symmetric_UsesNChannelsMin",
         1, {kNChannelsMax, kNChannelsMax}, {4 * 1024 * 1024, 4 * 1024 * 1024}, kNChannelsMin},

        // 2-rank symmetric exchange ({1,1}): == 0 vs <= 1 regression guard
        // (commit 8ce40df957). Must be classified symmetric -> nChannelsMin.
        {"TwoRankExchange_UsesNChannelsMin",
         1, {1, 1}, {4 * 1024 * 1024, 4 * 1024 * 1024}, kNChannelsMin},

        // Multi-node asymmetric gather: single-node guard (nNodes <= 1) must
        // suppress the boost even though the traffic is asymmetric.
        {"MultiNode_Asymmetric_UsesNChannelsMin",
         2, {kNChannelsMax, 0}, {256 * 1024, -1}, kNChannelsMin},

        // Single-node asymmetric but small message: std::min clamps the count
        // below nChannelsMax (choosing nChannelsMax as start must not over-
        // allocate channels for small transfers).
        {"SmallMessage_ClampsBelowNChannelsMax",
         1, {kNChannelsMax, 0}, {32 * 1024, -1}, kNChannelsMin},
    };

    INSTANTIATE_TEST_SUITE_P(
        addP2pToPlan, AddP2pToPlanScaling, ::testing::ValuesIn(kCases),
        [](const ::testing::TestParamInfo<P2pScalingCase>& info) {
            return std::string(info.param.name);
        });

    // Smoke test: self-send of zero bytes never reaches the changed nChannels
    // computation (nChannels[dir] short-circuits to 1 when bytes == 0). Confirms
    // the harness/stubs link and that a work item is enqueued.
    TEST_F(AddP2pToPlanTest, SelfSendZeroBytes)
    {
        MakeComm(/*nNodes*/ 1);

        int planTotalTasks[2] = {1, 1};
        ncclTaskP2p* p2pTasks[2] = {nullptr, nullptr};

        const int rank = comm->rank;
        ncclResult_t result = addP2pToPlan(
            comm, plan,
            /*nChannelsMin*/ 1, /*nChannelsMax*/ 1, /*p2pRound*/ 0,
            /*sendRank*/ rank, /*sendAddr*/ nullptr, /*sendBytes*/ 0,
            /*recvRank*/ rank, /*recvAddr*/ nullptr, /*recvBytes*/ 0,
            /*sendOpCount*/ 0, /*recvOpCount*/ 0,
            planTotalTasks, p2pTasks);

        EXPECT_EQ(result, ncclSuccess);

        // A single work item should have been enqueued into the plan.
        EXPECT_FALSE(ncclIntruQueueEmpty(&plan->workQueue));
        EXPECT_EQ(plan->workBytes, sizeof(struct ncclDevWorkP2p));
    }
}
}
