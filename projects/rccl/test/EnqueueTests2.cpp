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
    TEST(addP2pToPlan, SelfSendZeroBytes)
    {
        struct ncclComm* comm = static_cast<struct ncclComm*>(calloc(1, sizeof(*comm)));
        struct ncclKernelPlan* plan = static_cast<struct ncclKernelPlan*>(calloc(1, sizeof(*plan)));

        // Bump-allocator used by ncclMemoryStackAlloc* inside the function.
        ncclMemoryStackConstruct(&comm->memScoped);

        // Minimal single-node, single-channel topology.
        comm->rank = 0;
        comm->nNodes = 1;
        comm->maxLocalRanks = 1;
        comm->p2pnChannels = 1;
        comm->p2pnChannelsPerPeer = 1;
        comm->p2pChannelShiftSize = 0;
        comm->p2pChunkSize = 1 << 16;
        comm->p2pNet = 0;
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
            comm->buffSizes[p] = NCCL_STEPS * comm->p2pChunkSize;

        plan->comm = comm;

        // ncclDevFuncId_P2p() looks up the send/recv function id in this map;
        // an empty map yields -1 and an early ncclInvalidUsage. Register the
        // single key computed for (ncclFuncSendRecv, -1, -1, UNDEF, UNDEF).
        uint64_t p2pKey = (uint64_t)(ncclFuncSendRecv & RCCL_FUNC_ID_MASK) << RCCL_COLL_SHIFT;
        ncclDevFuncNameToId[p2pKey] = 0;

        // Self-send of zero bytes: recv and send are the local rank.
        const int rank = comm->rank;
        int planTotalTasks[2] = {1, 1};
        struct ncclTaskP2p* p2pTasks[2] = {nullptr, nullptr};

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

        free(plan);
        free(comm);
    }

    // Gather_RecvHeavy_UsesNChannelsMax
    //
    // Exercises the ROCM-26926 fix: single-node asymmetric P2P traffic
    // (planTotalTasks has a zero direction, as in gather where the root only
    // receives) must start channel selection from nChannelsMax instead of
    // nChannelsMin.
    //
    // We drive the recv direction (dir 0) with a large positive byte count and
    // leave the send direction as a no-op (-1). planTotalTasks = {N, 0} makes
    // asymmetric == true. With nNodes == 1 the fix selects nChStart ==
    // nChannelsMax; a large enough message keeps the std::min from clamping
    // below it. The resulting count is read back via p2pTasks[0]->nChannels.
    //
    // A self-send (sendRank == recvRank == comm->rank) is used so the function
    // skips the connector-graph walk and proxy-op logic, keeping the host-only
    // setup minimal while still executing the changed nChannels computation.
    TEST(addP2pToPlan, Gather_RecvHeavy_UsesNChannelsMax)
    {
        struct ncclComm* comm = static_cast<struct ncclComm*>(calloc(1, sizeof(*comm)));
        struct ncclKernelPlan* plan = static_cast<struct ncclKernelPlan*>(calloc(1, sizeof(*plan)));

        ncclMemoryStackConstruct(&comm->memScoped);

        // Single-node topology with distinct min/max channel counts so the
        // nChStart choice is observable.
        const int kNChannelsMin = 4;
        const int kNChannelsMax = 8;
        comm->rank = 0;
        comm->nNodes = 1;
        comm->maxLocalRanks = 1;
        comm->p2pnChannels = 8;         // power of two, >= nChannelsMax
        comm->p2pnChannelsPerPeer = 8;
        comm->p2pChannelShiftSize = 0;
        comm->p2pChunkSize = 1 << 16;   // 64 KiB
        comm->p2pNet = 0;
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
            comm->buffSizes[p] = NCCL_STEPS * comm->p2pChunkSize;

        plan->comm = comm;

        uint64_t p2pKey = (uint64_t)(ncclFuncSendRecv & RCCL_FUNC_ID_MASK) << RCCL_COLL_SHIFT;
        ncclDevFuncNameToId[p2pKey] = 0;

        // Recv message sized so the fix is observable: large enough that
        // divUp(bytes, p2pChunkSize/8) >= nChannelsMax (std::min does not clamp
        // below nChStart), but small enough that the subsequent partSize
        // doubling loop would NOT have promoted the old nChannelsMin start up
        // to nChannelsMax. With p2pChunkSize=64KiB (minPartSize=8KiB,
        // maxPartSize=2MiB), 4MiB yields nChannelsMax=8 under the fix versus
        // nChannelsMin=4 under the old behaviour.
        const ssize_t kRecvBytes = 4 * 1024 * 1024;  // 4 MiB

        // Non-null recv task so we can read back the selected channel count.
        struct ncclTaskP2p recvTask = {};
        struct ncclTaskP2p* p2pTasks[2] = {&recvTask, nullptr};

        // Gather at the root: recv total is nonzero, send total is zero.
        int planTotalTasks[2] = {kNChannelsMax, 0};

        const int rank = comm->rank;
        ncclResult_t result = addP2pToPlan(
            comm, plan,
            /*nChannelsMin*/ kNChannelsMin, /*nChannelsMax*/ kNChannelsMax, /*p2pRound*/ 0,
            /*sendRank*/ rank, /*sendAddr*/ nullptr, /*sendBytes*/ -1,
            /*recvRank*/ rank, /*recvAddr*/ nullptr, /*recvBytes*/ kRecvBytes,
            /*sendOpCount*/ 0, /*recvOpCount*/ 0,
            planTotalTasks, p2pTasks);

        EXPECT_EQ(result, ncclSuccess);

        // The asymmetric path must start from nChannelsMax (not nChannelsMin).
        EXPECT_EQ(recvTask.nChannels, kNChannelsMax);
        EXPECT_GT(recvTask.nChannels, kNChannelsMin);

        free(plan);
        free(comm);
    }

    // Scatter_SendHeavy_UsesNChannelsMax
    //
    // Mirror of Gather_RecvHeavy on the send direction (dir 1). Scatter fans
    // out from the root, so the send total is nonzero while the recv total is
    // zero: planTotalTasks = {0, N}, asymmetric == true. With nNodes == 1 the
    // fix selects nChStart == nChannelsMax for the send direction. Read back
    // via p2pTasks[1]->nChannels.
    TEST(addP2pToPlan, Scatter_SendHeavy_UsesNChannelsMax)
    {
        struct ncclComm* comm = static_cast<struct ncclComm*>(calloc(1, sizeof(*comm)));
        struct ncclKernelPlan* plan = static_cast<struct ncclKernelPlan*>(calloc(1, sizeof(*plan)));

        ncclMemoryStackConstruct(&comm->memScoped);

        const int kNChannelsMin = 4;
        const int kNChannelsMax = 8;
        comm->rank = 0;
        comm->nNodes = 1;
        comm->maxLocalRanks = 1;
        comm->p2pnChannels = 8;
        comm->p2pnChannelsPerPeer = 8;
        comm->p2pChannelShiftSize = 0;
        comm->p2pChunkSize = 1 << 16;   // 64 KiB
        comm->p2pNet = 0;
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
            comm->buffSizes[p] = NCCL_STEPS * comm->p2pChunkSize;

        plan->comm = comm;

        uint64_t p2pKey = (uint64_t)(ncclFuncSendRecv & RCCL_FUNC_ID_MASK) << RCCL_COLL_SHIFT;
        ncclDevFuncNameToId[p2pKey] = 0;

        // 4 MiB: fix -> nChannelsMax=8, old behaviour -> nChannelsMin=4 (see
        // Gather_RecvHeavy for the sizing rationale).
        const ssize_t kSendBytes = 4 * 1024 * 1024;

        // Non-null send task so we can read back the selected channel count.
        struct ncclTaskP2p sendTask = {};
        struct ncclTaskP2p* p2pTasks[2] = {nullptr, &sendTask};

        // Scatter at the root: send total nonzero, recv total zero.
        int planTotalTasks[2] = {0, kNChannelsMax};

        const int rank = comm->rank;
        ncclResult_t result = addP2pToPlan(
            comm, plan,
            /*nChannelsMin*/ kNChannelsMin, /*nChannelsMax*/ kNChannelsMax, /*p2pRound*/ 0,
            /*sendRank*/ rank, /*sendAddr*/ nullptr, /*sendBytes*/ kSendBytes,
            /*recvRank*/ rank, /*recvAddr*/ nullptr, /*recvBytes*/ -1,
            /*sendOpCount*/ 0, /*recvOpCount*/ 0,
            planTotalTasks, p2pTasks);

        EXPECT_EQ(result, ncclSuccess);

        // The asymmetric path must start from nChannelsMax (not nChannelsMin).
        EXPECT_EQ(sendTask.nChannels, kNChannelsMax);
        EXPECT_GT(sendTask.nChannels, kNChannelsMin);

        free(plan);
        free(comm);
    }

    // AllToAll_Symmetric_UsesNChannelsMin
    //
    // Symmetric traffic: all ranks both send and receive, so neither direction
    // of planTotalTasks is zero (alltoall = {N, N}). asymmetric == false, so
    // the fix must keep nChStart == nChannelsMin to avoid XGMI link contention
    // when all ranks transmit simultaneously.
    //
    // Both directions carry a positive 4 MiB payload (the size at which the
    // asymmetric path would have selected nChannelsMax). Both tasks are
    // non-null so we can confirm each direction stays at nChannelsMin.
    TEST(addP2pToPlan, AllToAll_Symmetric_UsesNChannelsMin)
    {
        struct ncclComm* comm = static_cast<struct ncclComm*>(calloc(1, sizeof(*comm)));
        struct ncclKernelPlan* plan = static_cast<struct ncclKernelPlan*>(calloc(1, sizeof(*plan)));

        ncclMemoryStackConstruct(&comm->memScoped);

        const int kNChannelsMin = 4;
        const int kNChannelsMax = 8;
        comm->rank = 0;
        comm->nNodes = 1;
        comm->maxLocalRanks = 1;
        comm->p2pnChannels = 8;
        comm->p2pnChannelsPerPeer = 8;
        comm->p2pChannelShiftSize = 0;
        comm->p2pChunkSize = 1 << 16;   // 64 KiB
        comm->p2pNet = 0;
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
            comm->buffSizes[p] = NCCL_STEPS * comm->p2pChunkSize;

        plan->comm = comm;

        uint64_t p2pKey = (uint64_t)(ncclFuncSendRecv & RCCL_FUNC_ID_MASK) << RCCL_COLL_SHIFT;
        ncclDevFuncNameToId[p2pKey] = 0;

        // 4 MiB in both directions: large enough that the asymmetric path would
        // pick nChannelsMax, so a symmetric result of nChannelsMin proves the
        // predicate correctly classified this as symmetric.
        const ssize_t kBytes = 4 * 1024 * 1024;

        struct ncclTaskP2p recvTask = {};
        struct ncclTaskP2p sendTask = {};
        struct ncclTaskP2p* p2pTasks[2] = {&recvTask, &sendTask};

        // Alltoall: both directions have nonzero task totals.
        int planTotalTasks[2] = {kNChannelsMax, kNChannelsMax};

        const int rank = comm->rank;
        ncclResult_t result = addP2pToPlan(
            comm, plan,
            /*nChannelsMin*/ kNChannelsMin, /*nChannelsMax*/ kNChannelsMax, /*p2pRound*/ 0,
            /*sendRank*/ rank, /*sendAddr*/ nullptr, /*sendBytes*/ kBytes,
            /*recvRank*/ rank, /*recvAddr*/ nullptr, /*recvBytes*/ kBytes,
            /*sendOpCount*/ 0, /*recvOpCount*/ 0,
            planTotalTasks, p2pTasks);

        EXPECT_EQ(result, ncclSuccess);

        // Symmetric traffic must stay at nChannelsMin, not be boosted to max.
        EXPECT_EQ(recvTask.nChannels, kNChannelsMin);
        EXPECT_EQ(sendTask.nChannels, kNChannelsMin);
        EXPECT_LT(recvTask.nChannels, kNChannelsMax);

        free(plan);
        free(comm);
    }

    // TwoRankExchange_UsesNChannelsMin
    //
    // Regression guard for the == 0 vs <= 1 tightening (commit 8ce40df957).
    // The original fix used `planTotalTasks[dir] <= 1`, which misclassified a
    // 2-rank symmetric exchange (planTotalTasks = {1, 1}) as asymmetric and
    // wrongly boosted it to nChannelsMax. The corrected predicate uses == 0, so
    // {1, 1} is symmetric and must stay at nChannelsMin.
    //
    // Both directions carry a positive 4 MiB payload (the size at which an
    // asymmetric classification would select nChannelsMax), so a result of
    // nChannelsMin proves {1, 1} is treated as symmetric.
    TEST(addP2pToPlan, TwoRankExchange_UsesNChannelsMin)
    {
        struct ncclComm* comm = static_cast<struct ncclComm*>(calloc(1, sizeof(*comm)));
        struct ncclKernelPlan* plan = static_cast<struct ncclKernelPlan*>(calloc(1, sizeof(*plan)));

        ncclMemoryStackConstruct(&comm->memScoped);

        const int kNChannelsMin = 4;
        const int kNChannelsMax = 8;
        comm->rank = 0;
        comm->nNodes = 1;
        comm->maxLocalRanks = 1;
        comm->p2pnChannels = 8;
        comm->p2pnChannelsPerPeer = 8;
        comm->p2pChannelShiftSize = 0;
        comm->p2pChunkSize = 1 << 16;   // 64 KiB
        comm->p2pNet = 0;
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
            comm->buffSizes[p] = NCCL_STEPS * comm->p2pChunkSize;

        plan->comm = comm;

        uint64_t p2pKey = (uint64_t)(ncclFuncSendRecv & RCCL_FUNC_ID_MASK) << RCCL_COLL_SHIFT;
        ncclDevFuncNameToId[p2pKey] = 0;

        const ssize_t kBytes = 4 * 1024 * 1024;

        struct ncclTaskP2p recvTask = {};
        struct ncclTaskP2p sendTask = {};
        struct ncclTaskP2p* p2pTasks[2] = {&recvTask, &sendTask};

        // 2-rank symmetric exchange: exactly one task in each direction.
        int planTotalTasks[2] = {1, 1};

        const int rank = comm->rank;
        ncclResult_t result = addP2pToPlan(
            comm, plan,
            /*nChannelsMin*/ kNChannelsMin, /*nChannelsMax*/ kNChannelsMax, /*p2pRound*/ 0,
            /*sendRank*/ rank, /*sendAddr*/ nullptr, /*sendBytes*/ kBytes,
            /*recvRank*/ rank, /*recvAddr*/ nullptr, /*recvBytes*/ kBytes,
            /*sendOpCount*/ 0, /*recvOpCount*/ 0,
            planTotalTasks, p2pTasks);

        EXPECT_EQ(result, ncclSuccess);

        // {1, 1} is symmetric under the == 0 predicate: must stay at min.
        EXPECT_EQ(recvTask.nChannels, kNChannelsMin);
        EXPECT_EQ(sendTask.nChannels, kNChannelsMin);
        EXPECT_LT(recvTask.nChannels, kNChannelsMax);

        free(plan);
        free(comm);
    }

    // MultiNode_Asymmetric_UsesNChannelsMin
    //
    // The nChannelsMax boost is gated on the single-node guard
    // (comm->nNodes <= 1 && asymmetric). Multi-node collectives use inter-node
    // channel scheduling and must not receive the single-node boost even when
    // the traffic is asymmetric. Here planTotalTasks = {N, 0} (asymmetric) but
    // nNodes > 1, so nChStart must remain nChannelsMin.
    //
    // With nNodes > 1 the function reads comm->topo (rcclEffectiveP2pBatchEnable
    // and the batchP2P arch check), so a minimal topo with one GPU node is
    // allocated. minPartSize/maxPartSize also change for the multi-node case
    // (stepSize/2 .. stepSize), which this test exercises.
    TEST(addP2pToPlan, MultiNode_Asymmetric_UsesNChannelsMin)
    {
        struct ncclComm* comm = static_cast<struct ncclComm*>(calloc(1, sizeof(*comm)));
        struct ncclKernelPlan* plan = static_cast<struct ncclKernelPlan*>(calloc(1, sizeof(*plan)));
        struct ncclTopoSystem* topo = static_cast<struct ncclTopoSystem*>(calloc(1, sizeof(*topo)));

        ncclMemoryStackConstruct(&comm->memScoped);

        const int kNChannelsMin = 4;
        const int kNChannelsMax = 8;
        comm->rank = 0;
        comm->nNodes = 2;               // multi-node: single-node guard is false
        comm->maxLocalRanks = 1;
        comm->p2pnChannels = 8;
        comm->p2pnChannelsPerPeer = 8;
        comm->p2pChannelShiftSize = 0;
        comm->p2pChunkSize = 1 << 16;   // 64 KiB
        comm->p2pNet = 0;
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
            comm->buffSizes[p] = NCCL_STEPS * comm->p2pChunkSize;

        // Minimal topology: one GPU node whose gcn arch string is read by
        // rcclEffectiveP2pBatchEnable() and the batchP2P arch check when
        // nNodes > 1.
        topo->nodes[GPU].count = 1;
        snprintf(topo->nodes[GPU].nodes[0].gpu.gcn, GCN_ARCH_NAME_LEN, "gfx942");
        comm->topo = topo;

        plan->comm = comm;

        uint64_t p2pKey = (uint64_t)(ncclFuncSendRecv & RCCL_FUNC_ID_MASK) << RCCL_COLL_SHIFT;
        ncclDevFuncNameToId[p2pKey] = 0;

        // Multi-node minPartSize/maxPartSize are stepSize/2 .. stepSize
        // (32KiB..64KiB here). At 256KiB the old nChannelsMin start yields 4
        // (partSize 64KiB, no doubling) while an nChannelsMax start would yield
        // 8 -- so this size distinguishes the guarded from the boosted result.
        // (At larger sizes the partSize doubling loop promotes both to 8 and
        // the assertion would not prove the guard.)
        const ssize_t kRecvBytes = 256 * 1024;

        struct ncclTaskP2p recvTask = {};
        struct ncclTaskP2p* p2pTasks[2] = {&recvTask, nullptr};

        // Asymmetric (gather) traffic, but multi-node.
        int planTotalTasks[2] = {kNChannelsMax, 0};

        const int rank = comm->rank;
        ncclResult_t result = addP2pToPlan(
            comm, plan,
            /*nChannelsMin*/ kNChannelsMin, /*nChannelsMax*/ kNChannelsMax, /*p2pRound*/ 0,
            /*sendRank*/ rank, /*sendAddr*/ nullptr, /*sendBytes*/ -1,
            /*recvRank*/ rank, /*recvAddr*/ nullptr, /*recvBytes*/ kRecvBytes,
            /*sendOpCount*/ 0, /*recvOpCount*/ 0,
            planTotalTasks, p2pTasks);

        EXPECT_EQ(result, ncclSuccess);

        // Multi-node must not take the single-node nChannelsMax boost.
        EXPECT_EQ(recvTask.nChannels, kNChannelsMin);
        EXPECT_LT(recvTask.nChannels, kNChannelsMax);

        free(topo);
        free(plan);
        free(comm);
    }

    // SmallMessage_ClampsBelowNChannelsMax
    //
    // The fix picks nChStart = nChannelsMax for single-node asymmetric traffic,
    // but the actual channel count is std::min(nChStart, divUp(bytes,
    // minPartSize)). For a small message the divUp term is the binding
    // constraint, so selecting nChannelsMax as the start must NOT force a high
    // channel count -- small transfers stay proportional to their size.
    //
    // Single-node minPartSize = p2pChunkSize/8 = 8KiB. A 32KiB message gives
    // divUp(32KiB, 8KiB) = 4, so nChannels clamps to 4 even though nChStart is
    // nChannelsMax (8), and the partSize doubling loop does not promote it
    // (partSize 8KiB <= maxPartSize 2MiB).
    TEST(addP2pToPlan, SmallMessage_ClampsBelowNChannelsMax)
    {
        struct ncclComm* comm = static_cast<struct ncclComm*>(calloc(1, sizeof(*comm)));
        struct ncclKernelPlan* plan = static_cast<struct ncclKernelPlan*>(calloc(1, sizeof(*plan)));

        ncclMemoryStackConstruct(&comm->memScoped);

        const int kNChannelsMin = 4;
        const int kNChannelsMax = 8;
        comm->rank = 0;
        comm->nNodes = 1;
        comm->maxLocalRanks = 1;
        comm->p2pnChannels = 8;
        comm->p2pnChannelsPerPeer = 8;
        comm->p2pChannelShiftSize = 0;
        comm->p2pChunkSize = 1 << 16;   // 64 KiB -> minPartSize 8 KiB
        comm->p2pNet = 0;
        for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++)
            comm->buffSizes[p] = NCCL_STEPS * comm->p2pChunkSize;

        plan->comm = comm;

        uint64_t p2pKey = (uint64_t)(ncclFuncSendRecv & RCCL_FUNC_ID_MASK) << RCCL_COLL_SHIFT;
        ncclDevFuncNameToId[p2pKey] = 0;

        // 32 KiB: divUp(32KiB, 8KiB) = 4 channels, below nChannelsMax.
        const ssize_t kRecvBytes = 32 * 1024;
        const int kExpectedChannels = 4;

        struct ncclTaskP2p recvTask = {};
        struct ncclTaskP2p* p2pTasks[2] = {&recvTask, nullptr};

        // Asymmetric (gather) traffic -> nChStart = nChannelsMax, but the
        // message is too small to use them all.
        int planTotalTasks[2] = {kNChannelsMax, 0};

        const int rank = comm->rank;
        ncclResult_t result = addP2pToPlan(
            comm, plan,
            /*nChannelsMin*/ kNChannelsMin, /*nChannelsMax*/ kNChannelsMax, /*p2pRound*/ 0,
            /*sendRank*/ rank, /*sendAddr*/ nullptr, /*sendBytes*/ -1,
            /*recvRank*/ rank, /*recvAddr*/ nullptr, /*recvBytes*/ kRecvBytes,
            /*sendOpCount*/ 0, /*recvOpCount*/ 0,
            planTotalTasks, p2pTasks);

        EXPECT_EQ(result, ncclSuccess);

        // std::min clamps to the message-derived count, below nChannelsMax.
        EXPECT_EQ(recvTask.nChannels, kExpectedChannels);
        EXPECT_LT(recvTask.nChannels, kNChannelsMax);

        free(plan);
        free(comm);
    }
}
}
