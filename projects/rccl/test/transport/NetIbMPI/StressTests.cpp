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
TEST_F(NetIbMPITest, MrCacheRefCount) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(/*dev=*/0, cp, guard);

    void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;

    const size_t sz = 4096;
    auto buf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(buf.get(), nullptr);

    // First registration — inserts into cache with refs=1
    void* mh1 = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &mh1), ncclSuccess);
    ASSERT_NE(mh1, nullptr);

    // Second registration of the same buffer — hits cache, bumps refs to 2
    void* mh2 = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &mh2), ncclSuccess);
    ASSERT_NE(mh2, nullptr);

    // First deregMr — refs drops to 1; underlying MR stays alive
    ASSERT_EQ(DeregisterMemory(comm, mh1), ncclSuccess);

    // Second deregMr — refs drops to 0; underlying MR is freed
    ASSERT_EQ(DeregisterMemory(comm, mh2), ncclSuccess);

    MPI_Barrier(MPI_COMM_WORLD);
}

// E2.  SendSizeClamping — sender posts a buffer larger than the receiver's posted size.
//      Covers the ncclIbIsend L2543 branch: if (size > slots[r].size) size = slots[r].size.
//      The transfer should complete cleanly; only recv_size bytes are transferred.
TEST_F(NetIbMPITest, SendSizeClamping) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(/*dev=*/0, cp, guard);

    static constexpr size_t kRecvSize = 4096;
    static constexpr size_t kSendSize = 65536;  // larger than kRecvSize — will be clamped
    static constexpr int    kTag      = 9900;

    auto sendBuf = makeHostBufferAutoGuard(malloc(kSendSize));
    auto recvBuf = makeHostBufferAutoGuard(malloc(kRecvSize));
    ASSERT_NE(sendBuf.get(), nullptr);
    ASSERT_NE(recvBuf.get(), nullptr);

    void* recvMh = nullptr;
    void* sendMh = nullptr;

    if (rank == 0) {
        // Register only kRecvSize — this sets slots[r].size = kRecvSize in the FIFO
        ASSERT_EQ(RegisterMemory(cp.recvComm, recvBuf.get(), kRecvSize, NCCL_PTR_HOST, &recvMh), ncclSuccess);
        ASSERT_NE(recvMh, nullptr);
    } else {
        // Sender registers larger buffer
        ASSERT_EQ(RegisterMemory(cp.sendComm, sendBuf.get(), kSendSize, NCCL_PTR_HOST, &sendMh), ncclSuccess);
        ASSERT_NE(sendMh, nullptr);
        fillHostBufferWithPattern<uint8_t>(sendBuf.get(), kSendSize, makeBytePattern(42));
    }

    void* req = nullptr;
    if (rank == 0) {
        PostSingleRecv(cp.recvComm, recvBuf.get(), kRecvSize, kTag, recvMh, &req);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        // Post send with kSendSize > kRecvSize — ncclIbIsend will clamp to kRecvSize
        PostSendWithRetry(cp.sendComm, sendBuf.get(), kSendSize, kTag, sendMh, &req);
        ASSERT_NE(req, nullptr);
    }

    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(req, sizes, kLargeTransferTimeoutMs), ncclSuccess)
        << "WaitForCompletion failed on rank " << rank;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        // Verify: recv reports kRecvSize bytes (clamped), not kSendSize
        EXPECT_EQ(sizes[0], static_cast<int>(kRecvSize))
            << "Expected recv size to be clamped to " << kRecvSize;
        // Verify first kRecvSize bytes match the sender's pattern
        bool ok = verifyHostBufferData<uint8_t>(recvBuf.get(), kRecvSize, makeBytePattern(42));
        EXPECT_TRUE(ok) << "Data mismatch in clamped receive";
    }

    if (rank == 0 && recvMh) DeregisterMemory(cp.recvComm, recvMh);
    if (rank == 1 && sendMh) DeregisterMemory(cp.sendComm, sendMh);
}

// E3.  NullCommClose — closes a NULL send/recv comm pointer.
//      Covers the ncclIbCloseSend/ncclIbCloseRecv null-guard branch
//      (net_ib.cc:3063 and 3083: if (comm) {...}).
//      The API must return ncclSuccess without crashing.
TEST_F(NetIbMPITest, NullCommClose) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    AssertInitAndGetDevices(nullptr);

    EXPECT_EQ(CloseSendComm(nullptr), ncclSuccess);
    EXPECT_EQ(CloseRecvComm(nullptr), ncclSuccess);
    EXPECT_EQ(CloseListenComm(nullptr), ncclSuccess);
    MPI_Barrier(MPI_COMM_WORLD);
}

// E4.  TagZeroReuse — 50 messages all sent with tag=0.
//      Verifies FIFO ordering: messages arrive in send order because
//      the FIFO is a strict ring (slot = fifoHead % MAX_REQUESTS).
TEST_F(NetIbMPITest, TagZeroReuse) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(/*dev=*/0, cp, guard);

    const size_t sz = kSmallBufferSize;
    auto buf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(buf.get(), nullptr);

    void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;
    void* mh   = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &mh), ncclSuccess);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    static constexpr int kIters = 50;
    for (int i = 0; i < kIters; i++) {
        DoSendRecv(cp.sendComm, cp.recvComm,
                   buf.get(), buf.get(), sz,
                   /*tag=*/0, mh, mh,
                   /*patternSeed=*/i);
    }
}

// E6.  AdaptiveRoutingThresholdBoundary — sizes around AR_THRESHOLD (8192).
//      When AR is enabled and size > threshold, ncclIbMultiSend adds a
//      0-byte RDMA_WRITE_WITH_IMM work request (net_ib.cc:2396).
TEST_F(NetIbMPITest, AdaptiveRoutingThresholdBoundary) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, cp, guard);

    const size_t maxSz = 16384;
    auto buf = makeHostBufferAutoGuard(malloc(maxSz));
    ASSERT_NE(buf.get(), nullptr);

    void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;
    void* mh = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.get(), maxSz, NCCL_PTR_HOST, &mh), ncclSuccess);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    // Sizes that straddle the default AR threshold of 8192
    const size_t sizes[] = {8190, 8191, 8192, 8193, 8194};
    static constexpr int kRepeats = 10;

    for (size_t sz : sizes) {
        for (int r = 0; r < kRepeats; r++) {
            int tag = static_cast<int>(sz) + r;
            DoSendRecv(cp.sendComm, cp.recvComm,
                       buf.get(), buf.get(), sz,
                       tag, mh, mh,
                       /*patternSeed=*/tag);
        }
    }
}

// E7.  InlineSendBoundary — CTS inline path (NCCL_IB_USE_INLINE=1).
//      Exercises IBV_SEND_INLINE for the FIFO CTS write.

#endif /* MPI_TESTS_ENABLED */
