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
}
}
