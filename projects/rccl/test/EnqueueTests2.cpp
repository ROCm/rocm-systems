/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Tests for enqueue.cc that don't depend on GPUs
//
// ---------------------------------------------------------------------------
// Test list
// ---------------------------------------------------------------------------
// These tests target the single-node asymmetric P2P channel-scaling fix in
// addP2pToPlan (ROCM-26926). The changed code is:
//
//     bool asymmetric = planTotalTasks[0] == 0 || planTotalTasks[1] == 0;
//     int nChStart = (comm->nNodes <= 1 && asymmetric) ? nChannelsMax
//                                                       : nChannelsMin;
//     nChannels[dir] = std::min<int>(nChStart, divUp(bytes[dir], minPartSize));
//
// To reach that else-branch a direction must have bytes[dir] > 0 (bytes == -1
// or 0 short-circuit earlier). To make the nChStart choice observable we set
// nChannelsMin < nChannelsMax and a large enough message that the std::min
// does not clamp back below nChStart. The resulting channel count is read back
// via p2pTasks[dir]->nChannels, which the function writes.
//
//   [existing]
//   * SelfSendZeroBytes
//       Minimal smoke test. selfSend + zero bytes: never reaches the changed
//       code (nChannels[dir] = 1 short-circuit). Confirms the harness/stubs
//       link and the function returns ncclSuccess.
//
//   [asymmetric -> expect nChannelsMax]
//   * Gather_RecvHeavy_UsesNChannelsMax
//       planTotalTasks = {N, 0} (root receives only). nNodes == 1.
//       Expect nChStart == nChannelsMax on the recv direction.
//   * Scatter_SendHeavy_UsesNChannelsMax
//       planTotalTasks = {0, N} (root sends only). nNodes == 1.
//       Expect nChStart == nChannelsMax on the send direction.
//
//   [symmetric -> expect nChannelsMin]
//   * AllToAll_Symmetric_UsesNChannelsMin
//       planTotalTasks = {N, N} (all ranks send and receive). nNodes == 1.
//       Expect nChStart == nChannelsMin (asymmetric == false).
//   * TwoRankExchange_UsesNChannelsMin
//       planTotalTasks = {1, 1}. Regression guard for the == 0 vs <= 1
//       tightening: a 2-rank symmetric exchange must NOT be treated as
//       asymmetric. Expect nChannelsMin.
//
//   [multi-node -> always nChannelsMin]
//   * MultiNode_Asymmetric_UsesNChannelsMin
//       planTotalTasks = {N, 0} but nNodes > 1. The single-node guard
//       (comm->nNodes <= 1) must keep nChStart == nChannelsMin even though
//       the traffic is asymmetric.
//
//   [boundary]
//   * SmallMessage_ClampsBelowNChannelsMax
//       Asymmetric + single-node, but bytes small enough that
//       divUp(bytes, minPartSize) < nChannelsMax, so std::min clamps the
//       result below nChannelsMax. Confirms nChStart alone does not force a
//       high channel count for small messages.
// ---------------------------------------------------------------------------
#include "nccl.h"
#include <gtest/gtest.h>
#include <hip/amd_detail/amd_hip_runtime.h>

#include ENQUEUE_CC_PATH

namespace RcclUnitTesting
{
namespace
{
    // addP2pToPlan is a file-static function, so it is only reachable because
    // this translation unit #includes the whole hipified enqueue.cc above.
    //
    // The function walks a populated communicator and kernel plan. To exercise
    // it on a CPU host with no GPU, we build the minimal state needed for the
    // simplest path: a self-send (sendRank == recvRank == comm->rank) of zero
    // bytes, which sets nProxyOps == 0 and skips all connector, registration,
    // and proxy-op logic.
    TEST(addP2pToPlan, SelfSendZeroBytes)
    {
        // Heap-allocate the large structs zero-initialised.
        struct ncclComm* comm = static_cast<struct ncclComm*>(calloc(1, sizeof(*comm)));
        struct ncclKernelPlan* plan = static_cast<struct ncclKernelPlan*>(calloc(1, sizeof(*plan)));
        ASSERT_NE(comm, nullptr);
        ASSERT_NE(plan, nullptr);

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
        ASSERT_NE(comm, nullptr);
        ASSERT_NE(plan, nullptr);

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
        ASSERT_NE(comm, nullptr);
        ASSERT_NE(plan, nullptr);

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
}
}
