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
