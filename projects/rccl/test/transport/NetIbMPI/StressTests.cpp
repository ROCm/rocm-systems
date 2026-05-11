/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Stress and branch-coverage unit tests for the base net-ib transport
// (src/transport/net_ib.cc).  These target paths not exercised by the
// happy-path functional tests in GeneralTests.cpp: resource exhaustion,
// FIFO backpressure, multi-QP striping, adaptive routing thresholds,
// multi-rank fan-in / fan-out / all-to-all, connection lifecycle stress,
// and endurance.

#include "NetIbMPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED

// =====================================================================
//  Group E: Branch-coverage (2-rank)
// =====================================================================

// E0.  InvalidRecvCount — calls ncclIbIrecv with n > NCCL_NET_IB_MAX_RECVS (8).
//      Covers the early-return branch at net_ib.cc:2731.
//      Requires a live recvComm (ready==1); the call must not crash.
TEST_F(NetIbMPITest, InvalidRecvCount) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(/*dev=*/0, cp, guard);

    if (rank == 0) {
        // n=9 > NCCL_NET_IB_MAX_RECVS (8) — must return ncclInternalError
        static constexpr int kOverLimit = 9;
        void*  data[kOverLimit]     = {};
        size_t sizes[kOverLimit]    = {};
        int    tags[kOverLimit]     = {};
        void*  mhandles[kOverLimit] = {};
        void*  req                  = nullptr;
        ncclResult_t r = PostRecv(cp.recvComm, kOverLimit, data, sizes, tags, mhandles, &req);
        EXPECT_EQ(r, ncclInternalError);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// E1.  MrCacheRefCount — registers the same host buffer twice on the same comm.
//      On the second RegisterMemory call the MR cache finds the range and increments
//      refs to 2 (ncclIbRegMrDmaBufInternal L2276). The first DeregMr decrements
//      refs to 1 without freeing (ncclIbDeregMrInternal refs>0 branch). The second
//      DeregMr decrements to 0 and actually calls wrap_ibv_dereg_mr.
//      Covers ncclIbDeregMrInternal L2326 "refs > 0" path.

#endif /* MPI_TESTS_ENABLED */
