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

// E0.  InvalidRecvCount — ncclIbIrecv with n > NCCL_NET_IB_MAX_RECVS (8)
//      must reject the request without crashing on a live recvComm.
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

// E1.  MrCacheRefCount — registers the same host buffer twice on the same
//      comm: cache hit bumps refs to 2, two Dereg calls bring it back to 0
//      (covers the refs>0 branch in ncclIbDeregMrInternal).
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

// E2.  SendSizeClamping — sender posts a buffer larger than the receiver's
//      posted size; isend must clamp to recv_size and complete cleanly.
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
        // Post send with kSendSize > kRecvSize — ncclIbIsend will clamp to kRecvSize.
        // PostSendWithRetry already calls FAIL() internally if it exhausts retries;
        // a redundant ASSERT_NE here would exit rank 1 before MPI_Barrier, leaving
        // rank 0 hung on WaitForCompletion.
        PostSendWithRetry(cp.sendComm, sendBuf.get(), kSendSize, kTag, sendMh, &req);
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

// E3.  NullCommClose — ncclIbCloseSend/Recv on NULL must return
//      ncclSuccess without crashing (null-guard branch).
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

// E5.  AdaptiveRoutingThresholdBoundary — sizes around AR_THRESHOLD (8192).
//      When AR is enabled and size > threshold, ncclIbMultiSend appends a
//      0-byte RDMA_WRITE_WITH_IMM work request.
TEST_F(NetIbMPITest, AdaptiveRoutingThresholdBoundary) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    const char* arEnv = getenv("NCCL_IB_ADAPTIVE_ROUTING");
    if (!arEnv || atoi(arEnv) == 0) {
        GTEST_SKIP() << "Set NCCL_IB_ADAPTIVE_ROUTING=1 to run this test";
    }
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

// E6.  InlineSendBoundary — CTS inline path (NCCL_IB_USE_INLINE=1).
//      Exercises IBV_SEND_INLINE for the FIFO CTS write.
TEST_F(NetIbMPITest, InlineSendBoundary) {
    const char* env = getenv("NCCL_IB_USE_INLINE");
    if (!env || strcmp(env, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_USE_INLINE=1";
    }

    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, cp, guard);

    const size_t maxSz = 1024 * 1024; // 1 MB
    auto buf = makeHostBufferAutoGuard(malloc(maxSz));
    ASSERT_NE(buf.get(), nullptr);

    void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;
    void* mh = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.get(), maxSz, NCCL_PTR_HOST, &mh), ncclSuccess);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    const size_t sizes[] = {1, 64, 128, 4096, maxSz};
    int tag = 0;
    for (size_t sz : sizes) {
        DoSendRecv(cp.sendComm, cp.recvComm,
                   buf.get(), buf.get(), sz,
                   tag++, mh, mh, static_cast<int>(sz));
    }
}

// E7.  MixedSizeBarrage — 25 sizes from 1B to 64MB on a single connection.
//      Exercises alignment boundaries, AR threshold, inline paths, and
//      large-MR registration.
TEST_F(NetIbMPITest, MixedSizeBarrage) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, cp, guard);

    const size_t maxSz = 64 * 1024 * 1024; // 64 MB
    auto buf = makeHostBufferAutoGuard(malloc(maxSz));
    ASSERT_NE(buf.get(), nullptr);

    void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;
    void* mh = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.get(), maxSz, NCCL_PTR_HOST, &mh), ncclSuccess);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    const size_t sizes[] = {
        1, 7, 13, 64, 127, 128, 129, 255, 256, 512,
        1023, 1024, 4095, 4096, 8191, 8192, 8193,
        16384, 32768, 65536, 131072,
        1048576,   // 1 MB
        4194304,   // 4 MB
        16777216,  // 16 MB
        67108864   // 64 MB
    };

    int tag = 0;
    for (size_t sz : sizes) {
        int timeout = (sz > 1024 * 1024) ? kLargeTransferTimeoutMs : kDefaultTimeoutMs;
        DoSendRecv(cp.sendComm, cp.recvComm,
                   buf.get(), buf.get(), sz,
                   tag, mh, mh,
                   /*patternSeed=*/tag, timeout);
        tag++;
    }
}

// =====================================================================
//  Group A: Resource exhaustion (2-rank)
// =====================================================================

// A1.  FifoPressureSenderFast — receiver deliberately slow, sender in
//      tight loop.  Verifies that isend returns *request==NULL when the
//      FIFO slot is not ready (backpressure).
TEST_F(NetIbMPITest, FifoPressureSenderFast) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, cp, guard);

    const size_t sz = kSmallBufferSize;
    auto buf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(buf.get(), nullptr);

    void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;
    void* mh = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &mh), ncclSuccess);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    static constexpr int kMsgs = 50;
    static constexpr int kRecvDelayUs = 200000; // 200ms

    // Both ranks operate in independent loops with only one MPI_Barrier at the end.
    // ASSERT_ inside these loops would cause one rank to exit before the final barrier,
    // leaving the other rank hung (deadlock) or waiting up to kStressTimeoutMs per
    // unmatched recv (long hang). Use EXPECT_ throughout so both ranks reach the barrier.
    if (rank == 0) {
        // Receiver: deliberately slow
        for (int i = 0; i < kMsgs; i++) {
            if (i > 0) usleep(kRecvDelayUs);
            void* req = nullptr;
            PostSingleRecv(cp.recvComm, buf.get(), sz, /*tag=*/i, mh, &req);
            int rsz = 0;
            EXPECT_EQ(WaitForCompletion(req, &rsz, kStressTimeoutMs), ncclSuccess);
        }
    } else {
        // Sender: tight loop, count NULL-request returns
        int nullCount = 0;
        for (int i = 0; i < kMsgs; i++) {
            fillHostBufferWithPattern<uint8_t>(buf.get(), sz, makeBytePattern(i));

            void* req = nullptr;
            int attempts = 0;
            do {
                ncclResult_t r = PostSend(cp.sendComm, buf.get(), sz, /*tag=*/i, mh, &req);
                EXPECT_EQ(r, ncclSuccess);
                if (r != ncclSuccess) break;
                if (!req) {
                    nullCount++;
                    usleep(1000); // 1ms
                }
                if (++attempts > 100000) {
                    ADD_FAILURE() << "PostSend stuck at msg " << i;
                    break;
                }
            } while (!req);

            if (req) {
                int rsz = 0;
                EXPECT_EQ(WaitForCompletion(req, &rsz, kStressTimeoutMs), ncclSuccess);
            }
        }
        // Backpressure must have been observed at least once
        EXPECT_GT(nullCount, 0) << "Expected at least one NULL-request (FIFO backpressure)";
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// A2.  RequestSlotExhaustion — post NCCL_NET_MAX_REQUESTS (32) irecvs,
//      then drain all via matching sends, then verify slots are recycled.
TEST_F(NetIbMPITest, RequestSlotExhaustion) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, cp, guard);

    static constexpr int kMaxReqs = 32; // NCCL_NET_MAX_REQUESTS
    const size_t sz = 256;
    auto buf = makeHostBufferAutoGuard(malloc(sz * kMaxReqs));
    ASSERT_NE(buf.get(), nullptr);

    void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;
    void* mh = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.get(), sz * kMaxReqs, NCCL_PTR_HOST, &mh), ncclSuccess);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    if (rank == 0) {
        // Post all 32 irecvs
        std::vector<void*> reqs(kMaxReqs, nullptr);
        for (int i = 0; i < kMaxReqs; i++) {
            char* p = static_cast<char*>(buf.get()) + i * sz;
            PostSingleRecv(cp.recvComm, p, sz, /*tag=*/i, mh, &reqs[i]);
        }
        // Signal rank 1 that all recvs are posted
        MPI_Barrier(MPI_COMM_WORLD);

        // Wait for all completions
        for (int i = 0; i < kMaxReqs; i++) {
            int rsz = 0;
            ASSERT_EQ(WaitForCompletion(reqs[i], &rsz, kStressTimeoutMs), ncclSuccess)
                << "Completion failed for recv " << i;
        }
    } else {
        // Wait for rank 0 to post all recvs
        MPI_Barrier(MPI_COMM_WORLD);

        // Send all 32
        for (int i = 0; i < kMaxReqs; i++) {
            char* p = static_cast<char*>(buf.get()) + i * sz;
            fillHostBufferWithPattern<uint8_t>(p, sz, makeBytePattern(i));
            void* req = nullptr;
            PostSendWithRetry(cp.sendComm, p, sz, /*tag=*/i, mh, &req);
            int rsz = 0;
            ASSERT_EQ(WaitForCompletion(req, &rsz, kStressTimeoutMs), ncclSuccess)
                << "Completion failed for send " << i;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Verify recycling: post and drain one more round
    for (int i = 0; i < 4; i++) {
        DoSendRecv(cp.sendComm, cp.recvComm,
                   buf.get(), buf.get(), sz,
                   /*tag=*/100 + i, mh, mh, 100 + i);
    }
}

// A3.  MemoryRegistrationStorm — register/deregister 512 unique buffers
//      in various patterns to stress the MR cache.
TEST_F(NetIbMPITest, MemoryRegistrationStorm) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, cp, guard);

    void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;

    static constexpr int kNumBufs = 512;
    static constexpr size_t kBufSz = 4096;

    // Allocate all buffers
    std::vector<void*> bufs(kNumBufs);
    for (int i = 0; i < kNumBufs; i++) {
        bufs[i] = malloc(kBufSz);
        ASSERT_NE(bufs[i], nullptr) << "malloc failed at buffer " << i;
    }
    auto bufCleanup = makeScopeGuard([&]() {
        for (auto* p : bufs) free(p);
    });

    // Phase 1: register all forward
    std::vector<void*> handles(kNumBufs, nullptr);
    for (int i = 0; i < kNumBufs; i++) {
        ASSERT_EQ(RegisterMemory(comm, bufs[i], kBufSz, NCCL_PTR_HOST, &handles[i]), ncclSuccess)
            << "regMr failed at " << i;
    }

    // Phase 2: deregister in reverse
    for (int i = kNumBufs - 1; i >= 0; i--) {
        ASSERT_EQ(DeregisterMemory(comm, handles[i]), ncclSuccess)
            << "deregMr (reverse) failed at " << i;
        handles[i] = nullptr;
    }

    // Phase 3: re-register all forward
    for (int i = 0; i < kNumBufs; i++) {
        ASSERT_EQ(RegisterMemory(comm, bufs[i], kBufSz, NCCL_PTR_HOST, &handles[i]), ncclSuccess)
            << "re-regMr failed at " << i;
    }

    // Phase 4: deregister in shuffled order (deterministic)
    std::vector<int> order(kNumBufs);
    for (int i = 0; i < kNumBufs; i++) order[i] = i;
    // Simple deterministic shuffle (seed=42)
    for (int i = kNumBufs - 1; i > 0; i--) {
        int j = (i * 42 + 17) % (i + 1);
        std::swap(order[i], order[j]);
    }
    for (int idx : order) {
        ASSERT_EQ(DeregisterMemory(comm, handles[idx]), ncclSuccess)
            << "deregMr (shuffled) failed at idx " << idx;
        handles[idx] = nullptr;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 5: verify connection is still alive with one transfer
    auto tbuf = makeHostBufferAutoGuard(malloc(kBufSz));
    ASSERT_NE(tbuf.get(), nullptr);
    void* mh = nullptr;
    ASSERT_EQ(RegisterMemory(comm, tbuf.get(), kBufSz, NCCL_PTR_HOST, &mh), ncclSuccess);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    DoSendRecv(cp.sendComm, cp.recvComm,
               tbuf.get(), tbuf.get(), kBufSz,
               /*tag=*/0, mh, mh, /*patternSeed=*/999);
}

// =====================================================================
//  Group C: Connection concurrency (2-rank)
// =====================================================================

// C1.  ConcurrentMultiConnectionStress — 4 connections on same device,
//      concurrent I/O with independent tag spaces.
TEST_F(NetIbMPITest, ConcurrentMultiConnectionStress) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    static constexpr int kNumConns = 4;
    static constexpr int kIters = 10;
    const size_t sz = kSmallBufferSize;

    struct ConnInfo {
        ConnectionPair pair;
        void* buf = nullptr;
        void* mh  = nullptr;
    };
    std::vector<ConnInfo> conns(kNumConns);
    // Track per-connection setup success so the transfer loop can skip broken
    // connections without rank skew on MPI_Barrier counts.
    std::vector<bool> conn_ok(kNumConns, false);

    // Setup all connections with unique MPI tags.
    // Use EXPECT_ (non-fatal) throughout so both ranks always reach MPI_Barrier.
    // On rank 0 failure we still call MPI_Send (with a zeroed handle) to unblock
    // rank 1 which is waiting on MPI_Recv — ASSERT_ before MPI_Send would leave
    // rank 1 hung indefinitely.
    for (int c = 0; c < kNumConns; c++) {
        bool ok = true;
        if (rank == 0) {
            ncclResult_t r = CreateListenComm(0, &conns[c].pair.handle, &conns[c].pair.listenComm);
            EXPECT_EQ(r, ncclSuccess) << "CreateListenComm failed for conn " << c;
            ok = (r == ncclSuccess && conns[c].pair.listenComm != nullptr);
            // Always send handle to unblock rank 1; zeroed handle causes ConnectToRemote
            // to fail on rank 1 so both ranks end up with ok=false consistently.
            if (!ok) memset(&conns[c].pair.handle, 0, sizeof(conns[c].pair.handle));
            MPI_Send(&conns[c].pair.handle, sizeof(ncclNetHandle_t), MPI_BYTE, 1, 500 + c, MPI_COMM_WORLD);
            if (ok) {
                for (int a = 0; a < kMaxRetryAttempts && !conns[c].pair.recvComm; a++) {
                    AcceptConnection(conns[c].pair.listenComm, &conns[c].pair.recvComm);
                    if (!conns[c].pair.recvComm) usleep(kPollIntervalUs);
                }
                EXPECT_NE(conns[c].pair.recvComm, nullptr) << "Accept failed for conn " << c;
                ok = (conns[c].pair.recvComm != nullptr);
            }
        } else {
            MPI_Recv(&conns[c].pair.handle, sizeof(ncclNetHandle_t), MPI_BYTE, 0, 500 + c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int a = 0; a < kMaxRetryAttempts && !conns[c].pair.sendComm; a++) {
                ConnectToRemote(0, &conns[c].pair.handle, &conns[c].pair.sendComm);
                if (!conns[c].pair.sendComm) usleep(kPollIntervalUs);
            }
            EXPECT_NE(conns[c].pair.sendComm, nullptr) << "Connect failed for conn " << c;
            ok = (conns[c].pair.sendComm != nullptr);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (ok) {
            conns[c].buf = malloc(sz);
            EXPECT_NE(conns[c].buf, nullptr) << "malloc failed for conn " << c;
            ok = (conns[c].buf != nullptr);
        }
        if (ok) {
            void* comm = (rank == 0) ? conns[c].pair.recvComm : conns[c].pair.sendComm;
            EXPECT_EQ(RegisterMemory(comm, conns[c].buf, sz, NCCL_PTR_HOST, &conns[c].mh), ncclSuccess)
                << "RegisterMemory failed for conn " << c;
            ok = (conns[c].mh != nullptr);
        }
        conn_ok[c] = ok;
    }

    auto cleanup = makeScopeGuard([&]() {
        for (auto& ci : conns) {
            void* comm = (rank == 0) ? ci.pair.recvComm : ci.pair.sendComm;
            if (ci.mh) DeregisterMemory(comm, ci.mh);
            free(ci.buf);
            if (rank == 0) {
                if (ci.pair.recvComm)   CloseRecvComm(ci.pair.recvComm);
                if (ci.pair.listenComm) CloseListenComm(ci.pair.listenComm);
            } else {
                if (ci.pair.sendComm) CloseSendComm(ci.pair.sendComm);
            }
        }
    });

    // Transfer on all 4 connections per iteration.
    // Skip connections that failed during setup; both ranks must take the same
    // branch to keep MPI_Barrier counts consistent (DoSendRecv has one barrier,
    // the explicit MPI_Barrier below substitutes it when skipping).
    for (int iter = 0; iter < kIters; iter++) {
        for (int c = 0; c < kNumConns; c++) {
            int tag  = iter * kNumConns + c;
            int seed = tag + 1000;
            if (!conn_ok[c]) {
                MPI_Barrier(MPI_COMM_WORLD);  // matches the barrier inside DoSendRecv
                continue;
            }
            DoSendRecv(conns[c].pair.sendComm, conns[c].pair.recvComm,
                       conns[c].buf, conns[c].buf, sz,
                       tag, conns[c].mh, conns[c].mh, seed);
        }
    }
}

// C2.  ConnectionChurnUnderLoad — 1000 connect→transfer→close cycles,
//      with RDMA resource leak detection.
TEST_F(NetIbMPITest, ConnectionChurnUnderLoad) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    static constexpr int kIters = 1000;
    const size_t sz = 1024;

    auto buf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(buf.get(), nullptr);

    // Run one warmup connection+close before capturing the baseline.
    // The mlx5 driver lazily allocates internal CQs (e.g. for async-event
    // bookkeeping) the first time a user-space CQ is created under a given
    // ibv_context.  Those driver-internal objects survive until ibv_close_device
    // and therefore show up as a stable "+N" in rdma resource show after the
    // first connection.  By warming up first we let those objects settle so that
    // the baseline reflects steady-state, not pre-first-connection state.
    {
        static constexpr int kWarmupTag = 599;
        ConnectionPair wcp;
        if (rank == 0) {
            ncclResult_t r = CreateListenComm(0, &wcp.handle, &wcp.listenComm);
            MPI_Send(&wcp.handle, sizeof(ncclNetHandle_t), MPI_BYTE, 1, kWarmupTag, MPI_COMM_WORLD);
            ASSERT_EQ(r, ncclSuccess) << "Warmup CreateListenComm failed";
            for (int a = 0; a < kMaxRetryAttempts && !wcp.recvComm; a++) {
                AcceptConnection(wcp.listenComm, &wcp.recvComm);
                if (!wcp.recvComm) usleep(kPollIntervalUs);
            }
            ASSERT_NE(wcp.recvComm, nullptr) << "Warmup accept failed";
        } else {
            MPI_Recv(&wcp.handle, sizeof(ncclNetHandle_t), MPI_BYTE, 0, kWarmupTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int a = 0; a < kMaxRetryAttempts && !wcp.sendComm; a++) {
                ConnectToRemote(0, &wcp.handle, &wcp.sendComm);
                if (!wcp.sendComm) usleep(kPollIntervalUs);
            }
            ASSERT_NE(wcp.sendComm, nullptr) << "Warmup connect failed";
        }
        void* wcomm = (rank == 0) ? wcp.recvComm : wcp.sendComm;
        void* wmh   = nullptr;
        ASSERT_EQ(RegisterMemory(wcomm, buf.get(), sz, NCCL_PTR_HOST, &wmh), ncclSuccess);
        DoSendRecv(wcp.sendComm, wcp.recvComm, buf.get(), buf.get(), sz, kWarmupTag, wmh, wmh, /*seed=*/0);
        ASSERT_EQ(DeregisterMemory(wcomm, wmh), ncclSuccess);
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(wcp.recvComm), ncclSuccess);
            ASSERT_EQ(CloseListenComm(wcp.listenComm), ncclSuccess);
        } else {
            ASSERT_EQ(CloseSendComm(wcp.sendComm), ncclSuccess);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    RdmaResourceCounts before = CaptureRdmaResources();
    MPI_Barrier(MPI_COMM_WORLD);

    for (int iter = 0; iter < kIters; iter++) {
        // Setup connection
        ConnectionPair cp;
        if (rank == 0) {
            ncclResult_t r = CreateListenComm(0, &cp.handle, &cp.listenComm);
            MPI_Send(&cp.handle, sizeof(ncclNetHandle_t), MPI_BYTE, 1, 600 + (iter % 1000), MPI_COMM_WORLD);
            ASSERT_EQ(r, ncclSuccess) << "CreateListenComm failed at iter " << iter;
            for (int a = 0; a < kMaxRetryAttempts && !cp.recvComm; a++) {
                AcceptConnection(cp.listenComm, &cp.recvComm);
                if (!cp.recvComm) usleep(kPollIntervalUs);
            }
            ASSERT_NE(cp.recvComm, nullptr) << "Accept failed at iter " << iter;
        } else {
            MPI_Recv(&cp.handle, sizeof(ncclNetHandle_t), MPI_BYTE, 0, 600 + (iter % 1000), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int a = 0; a < kMaxRetryAttempts && !cp.sendComm; a++) {
                ConnectToRemote(0, &cp.handle, &cp.sendComm);
                if (!cp.sendComm) usleep(kPollIntervalUs);
            }
            ASSERT_NE(cp.sendComm, nullptr) << "Connect failed at iter " << iter;
        }

        // Register, transfer, verify
        void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;
        void* mh = nullptr;
        ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &mh), ncclSuccess);

        DoSendRecv(cp.sendComm, cp.recvComm,
                   buf.get(), buf.get(), sz,
                   /*tag=*/iter % 1000, mh, mh, iter);

        ASSERT_EQ(DeregisterMemory(comm, mh), ncclSuccess);

        // Close
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(cp.recvComm), ncclSuccess);
            ASSERT_EQ(CloseListenComm(cp.listenComm), ncclSuccess);
        } else {
            ASSERT_EQ(CloseSendComm(cp.sendComm), ncclSuccess);
        }

        // RDMA checkpoint at midpoint
        if (iter == 499) {
            MPI_Barrier(MPI_COMM_WORLD);
            RdmaResourceCounts mid = CaptureRdmaResources();
            AssertNoRdmaLeaks(before, mid, "midpoint");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    RdmaResourceCounts after = CaptureRdmaResources();
    AssertNoRdmaLeaks(before, after, "final");
}

// C3.  ConnectionBatchCreateDestroy — create 10 connections, transfer on
//      all, close all, repeat 100 times (1000 lifecycles total).
//      RDMA checkpoints at batches 25, 50, 75, 100.
TEST_F(NetIbMPITest, ConnectionBatchCreateDestroy) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    static constexpr int kBatches  = 100;
    static constexpr int kPerBatch = 10;
    const size_t sz = 512;

    auto buf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(buf.get(), nullptr);

    // Warmup: one connect+transfer+close to let the driver settle its
    // internal CQs before we capture the baseline (same reason as in
    // ConnectionChurnUnderLoad — see comment there).
    {
        static constexpr int kWarmupTag = 699;
        ConnectionPair wcp;
        if (rank == 0) {
            ncclResult_t r = CreateListenComm(0, &wcp.handle, &wcp.listenComm);
            MPI_Send(&wcp.handle, sizeof(ncclNetHandle_t), MPI_BYTE, 1, kWarmupTag, MPI_COMM_WORLD);
            ASSERT_EQ(r, ncclSuccess) << "Warmup CreateListenComm failed";
            for (int a = 0; a < kMaxRetryAttempts && !wcp.recvComm; a++) {
                AcceptConnection(wcp.listenComm, &wcp.recvComm);
                if (!wcp.recvComm) usleep(kPollIntervalUs);
            }
            ASSERT_NE(wcp.recvComm, nullptr) << "Warmup accept failed";
        } else {
            MPI_Recv(&wcp.handle, sizeof(ncclNetHandle_t), MPI_BYTE, 0, kWarmupTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int a = 0; a < kMaxRetryAttempts && !wcp.sendComm; a++) {
                ConnectToRemote(0, &wcp.handle, &wcp.sendComm);
                if (!wcp.sendComm) usleep(kPollIntervalUs);
            }
            ASSERT_NE(wcp.sendComm, nullptr) << "Warmup connect failed";
        }
        void* wcomm = (rank == 0) ? wcp.recvComm : wcp.sendComm;
        void* wmh   = nullptr;
        ASSERT_EQ(RegisterMemory(wcomm, buf.get(), sz, NCCL_PTR_HOST, &wmh), ncclSuccess);
        DoSendRecv(wcp.sendComm, wcp.recvComm, buf.get(), buf.get(), sz, kWarmupTag, wmh, wmh, /*seed=*/0);
        ASSERT_EQ(DeregisterMemory(wcomm, wmh), ncclSuccess);
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(wcp.recvComm), ncclSuccess);
            ASSERT_EQ(CloseListenComm(wcp.listenComm), ncclSuccess);
        } else {
            ASSERT_EQ(CloseSendComm(wcp.sendComm), ncclSuccess);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    RdmaResourceCounts before = CaptureRdmaResources();
    MPI_Barrier(MPI_COMM_WORLD);

    for (int batch = 0; batch < kBatches; batch++) {
        // Create kPerBatch connections
        struct BatchConn {
            ConnectionPair cp;
            void* mh = nullptr;
        };
        std::vector<BatchConn> conns(kPerBatch);

        for (int c = 0; c < kPerBatch; c++) {
            int mpiTag = 700 + batch * kPerBatch + c;
            if (rank == 0) {
                ncclResult_t r = CreateListenComm(0, &conns[c].cp.handle, &conns[c].cp.listenComm);
                MPI_Send(&conns[c].cp.handle, sizeof(ncclNetHandle_t), MPI_BYTE, 1, mpiTag, MPI_COMM_WORLD);
                ASSERT_EQ(r, ncclSuccess) << "CreateListenComm failed at batch " << batch << " conn " << c;
                for (int a = 0; a < kMaxRetryAttempts && !conns[c].cp.recvComm; a++) {
                    AcceptConnection(conns[c].cp.listenComm, &conns[c].cp.recvComm);
                    if (!conns[c].cp.recvComm) usleep(kPollIntervalUs);
                }
                ASSERT_NE(conns[c].cp.recvComm, nullptr);
            } else {
                MPI_Recv(&conns[c].cp.handle, sizeof(ncclNetHandle_t), MPI_BYTE, 0, mpiTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int a = 0; a < kMaxRetryAttempts && !conns[c].cp.sendComm; a++) {
                    ConnectToRemote(0, &conns[c].cp.handle, &conns[c].cp.sendComm);
                    if (!conns[c].cp.sendComm) usleep(kPollIntervalUs);
                }
                ASSERT_NE(conns[c].cp.sendComm, nullptr);
            }
            MPI_Barrier(MPI_COMM_WORLD);

            void* comm = (rank == 0) ? conns[c].cp.recvComm : conns[c].cp.sendComm;
            ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &conns[c].mh), ncclSuccess);
        }

        // Transfer on each connection
        for (int c = 0; c < kPerBatch; c++) {
            int seed = batch * kPerBatch + c;
            DoSendRecv(conns[c].cp.sendComm, conns[c].cp.recvComm,
                       buf.get(), buf.get(), sz,
                       /*tag=*/c, conns[c].mh, conns[c].mh, seed);
        }

        // Close all in this batch
        for (int c = 0; c < kPerBatch; c++) {
            void* comm = (rank == 0) ? conns[c].cp.recvComm : conns[c].cp.sendComm;
            ASSERT_EQ(DeregisterMemory(comm, conns[c].mh), ncclSuccess);
            if (rank == 0) {
                ASSERT_EQ(CloseRecvComm(conns[c].cp.recvComm), ncclSuccess);
                ASSERT_EQ(CloseListenComm(conns[c].cp.listenComm), ncclSuccess);
            } else {
                ASSERT_EQ(CloseSendComm(conns[c].cp.sendComm), ncclSuccess);
            }
        }

        // RDMA checkpoints
        if ((batch + 1) % 25 == 0) {
            MPI_Barrier(MPI_COMM_WORLD);
            RdmaResourceCounts cp = CaptureRdmaResources();
            char label[64];
            snprintf(label, sizeof(label), "batch %d", batch + 1);
            AssertNoRdmaLeaks(before, cp, label);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    RdmaResourceCounts after = CaptureRdmaResources();
    AssertNoRdmaLeaks(before, after, "final");
}

// =====================================================================
//  Group D: Multi-QP outside CAST (2-rank)
// =====================================================================

// D1.  MultiQpSplitDataStress — QPS=8, SPLIT=1, alignment boundary sizes.
//      Exercises the per-QP split logic.
TEST_F(NetIbMPITest, MultiQpSplitDataStress) {
    const char* qps  = getenv("NCCL_IB_QPS_PER_CONNECTION");
    const char* split = getenv("NCCL_IB_SPLIT_DATA_ON_QPS");
    if (!qps || atoi(qps) < 2 || !split || strcmp(split, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_QPS_PER_CONNECTION>=2, NCCL_IB_SPLIT_DATA_ON_QPS=1";
    }

    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, cp, guard);

    const size_t maxSz = 1024 * 1024; // 1 MB
    auto buf = makeHostBufferAutoGuard(malloc(maxSz));
    ASSERT_NE(buf.get(), nullptr);

    void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;
    void* mh = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.get(), maxSz, NCCL_PTR_HOST, &mh), ncclSuccess);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    // Alignment boundary sizes (128B is the split threshold)
    const size_t sizes[] = {
        127, 128, 129, 255, 256, 257,
        1023, 1024, 1025, 4095, 4096, 4097,
        65535, 65536, 65537
    };

    static constexpr int kRepeats = 10;
    int tag = 0;
    for (size_t sz : sizes) {
        for (int r = 0; r < kRepeats; r++) {
            DoSendRecv(cp.sendComm, cp.recvComm,
                       buf.get(), buf.get(), sz,
                       tag, mh, mh, tag);
            tag++;
        }
    }
}

// D2.  MultiQpNoSplitStress — QPS=4, SPLIT=0; exercises the nDataQps path.
TEST_F(NetIbMPITest, MultiQpNoSplitStress) {
    const char* qps  = getenv("NCCL_IB_QPS_PER_CONNECTION");
    const char* split = getenv("NCCL_IB_SPLIT_DATA_ON_QPS");
    if (!qps || atoi(qps) < 2 || !split || strcmp(split, "0") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_QPS_PER_CONNECTION>=2, NCCL_IB_SPLIT_DATA_ON_QPS=0";
    }

    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, cp, guard);

    const size_t maxSz = 1024 * 1024;
    auto buf = makeHostBufferAutoGuard(malloc(maxSz));
    ASSERT_NE(buf.get(), nullptr);

    void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;
    void* mh = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.get(), maxSz, NCCL_PTR_HOST, &mh), ncclSuccess);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    const size_t sizes[] = {
        127, 128, 129, 255, 256, 257,
        1023, 1024, 1025, 4095, 4096, 4097,
        65535, 65536, 65537
    };

    static constexpr int kRepeats = 10;
    int tag = 0;
    for (size_t sz : sizes) {
        for (int r = 0; r < kRepeats; r++) {
            DoSendRecv(cp.sendComm, cp.recvComm,
                       buf.get(), buf.get(), sz,
                       tag, mh, mh, tag);
            tag++;
        }
    }
}

// =====================================================================
//  Group B: Multi-rank patterns (4 ranks)
//  B1 FanInStress, B2 FanOutStress, B3 AllToAllStress,
//  B4 MultiQpFanIn, B5 BidirectionalMultiRank, B6 LongRunningMultiRank
// =====================================================================

// B1.  FanInStress — ranks 1,2,3 each send to rank 0, 100 iterations.
TEST_F(NetIbMPITest, FanInStress) {
    ASSERT_TRUE(validateTestPrerequisites(kMinFourProcesses, kMinFourProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    std::vector<DirectedConnection> conns;
    SetupFanIn(/*dev=*/0, /*receiverRank=*/0, {1, 2, 3}, conns);

    const size_t sz = kSmallBufferSize;
    auto buf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(buf.get(), nullptr);

    // Register buffer for each connection this rank participates in
    std::vector<void*> mhandles;
    for (auto& c : conns) {
        void* comm = nullptr;
        if (rank == c.senderRank)   comm = c.sendComm;
        if (rank == c.receiverRank) comm = c.recvComm;
        void* mh = nullptr;
        if (comm) {
            ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &mh), ncclSuccess);
        }
        mhandles.push_back(mh);
    }
    auto mrCleanup = makeScopeGuard([&]() {
        for (size_t i = 0; i < conns.size(); i++) {
            if (!mhandles[i]) continue;
            void* comm = nullptr;
            if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
            if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
            if (comm) DeregisterMemory(comm, mhandles[i]);
        }
    });

    static constexpr int kIters = 100;
    for (int iter = 0; iter < kIters; iter++) {
        for (size_t c = 0; c < conns.size(); c++) {
            int seed = iter * 10 + conns[c].senderRank;
            DoDirectedSendRecv(conns[c], buf.get(), buf.get(), sz,
                               /*tag=*/iter, mhandles[c], mhandles[c], seed);
        }
    }

    // Deregister memory before closing connections (comm must be valid for DeregMr).
    for (size_t i = 0; i < conns.size(); i++) {
        if (!mhandles[i]) continue;
        void* comm = nullptr;
        if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
        if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
        if (comm) DeregisterMemory(comm, mhandles[i]);
        mhandles[i] = nullptr;
    }
    mrCleanup.dismiss();

    for (auto& c : conns) CloseDirectedConnection(c);
}

// B2.  FanOutStress — rank 0 sends to ranks 1,2,3, 100 iterations.
TEST_F(NetIbMPITest, FanOutStress) {
    ASSERT_TRUE(validateTestPrerequisites(kMinFourProcesses, kMinFourProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    std::vector<DirectedConnection> conns;
    SetupFanOut(/*dev=*/0, /*senderRank=*/0, {1, 2, 3}, conns);

    const size_t sz = kSmallBufferSize;
    auto buf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(buf.get(), nullptr);

    std::vector<void*> mhandles;
    for (auto& c : conns) {
        void* comm = nullptr;
        if (rank == c.senderRank)   comm = c.sendComm;
        if (rank == c.receiverRank) comm = c.recvComm;
        void* mh = nullptr;
        if (comm) {
            ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &mh), ncclSuccess);
        }
        mhandles.push_back(mh);
    }
    auto mrCleanup = makeScopeGuard([&]() {
        for (size_t i = 0; i < conns.size(); i++) {
            if (!mhandles[i]) continue;
            void* comm = nullptr;
            if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
            if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
            if (comm) DeregisterMemory(comm, mhandles[i]);
        }
    });

    static constexpr int kIters = 100;
    for (int iter = 0; iter < kIters; iter++) {
        for (size_t c = 0; c < conns.size(); c++) {
            int seed = iter * 10 + conns[c].receiverRank;
            DoDirectedSendRecv(conns[c], buf.get(), buf.get(), sz,
                               /*tag=*/iter, mhandles[c], mhandles[c], seed);
        }
    }

    // Deregister memory before closing connections (comm must be valid for DeregMr).
    for (size_t i = 0; i < conns.size(); i++) {
        if (!mhandles[i]) continue;
        void* comm = nullptr;
        if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
        if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
        if (comm) DeregisterMemory(comm, mhandles[i]);
        mhandles[i] = nullptr;
    }
    mrCleanup.dismiss();

    for (auto& c : conns) CloseDirectedConnection(c);
}

// B3.  AllToAllStress — full mesh (12 connections for 4 ranks), 50 iters.
TEST_F(NetIbMPITest, AllToAllStress) {
    ASSERT_TRUE(validateTestPrerequisites(kMinFourProcesses, kMinFourProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    int nranks = MPIEnvironment::world_size;
    AssertInitAndGetDevices(nullptr);

    std::vector<DirectedConnection> conns;
    SetupAllToAll(/*dev=*/0, nranks, conns);

    const size_t sz = kSmallBufferSize;
    auto buf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(buf.get(), nullptr);

    std::vector<void*> mhandles;
    for (auto& c : conns) {
        void* comm = nullptr;
        if (rank == c.senderRank)   comm = c.sendComm;
        if (rank == c.receiverRank) comm = c.recvComm;
        void* mh = nullptr;
        if (comm) {
            ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &mh), ncclSuccess);
        }
        mhandles.push_back(mh);
    }
    auto mrCleanup = makeScopeGuard([&]() {
        for (size_t i = 0; i < conns.size(); i++) {
            if (!mhandles[i]) continue;
            void* comm = nullptr;
            if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
            if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
            if (comm) DeregisterMemory(comm, mhandles[i]);
        }
    });

    static constexpr int kIters = 50;
    for (int iter = 0; iter < kIters; iter++) {
        for (size_t c = 0; c < conns.size(); c++) {
            int seed = iter * 100 + conns[c].senderRank * 10 + conns[c].receiverRank;
            DoDirectedSendRecv(conns[c], buf.get(), buf.get(), sz,
                               /*tag=*/iter, mhandles[c], mhandles[c], seed);
        }
    }

    // Deregister memory before closing connections (comm must be valid for DeregMr).
    for (size_t i = 0; i < conns.size(); i++) {
        if (!mhandles[i]) continue;
        void* comm = nullptr;
        if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
        if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
        if (comm) DeregisterMemory(comm, mhandles[i]);
        mhandles[i] = nullptr;
    }
    mrCleanup.dismiss();

    for (auto& c : conns) CloseDirectedConnection(c);
}

// B4.  MultiQpFanIn — fan-in 3→1 with QPS=4, SPLIT=1.
TEST_F(NetIbMPITest, MultiQpFanIn) {
    const char* qps  = getenv("NCCL_IB_QPS_PER_CONNECTION");
    const char* split = getenv("NCCL_IB_SPLIT_DATA_ON_QPS");
    if (!qps || atoi(qps) < 2 || !split || strcmp(split, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_QPS_PER_CONNECTION>=2, NCCL_IB_SPLIT_DATA_ON_QPS=1";
    }

    ASSERT_TRUE(validateTestPrerequisites(kMinFourProcesses, kMinFourProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    std::vector<DirectedConnection> conns;
    SetupFanIn(0, 0, {1, 2, 3}, conns);

    const size_t sz = 16384;
    auto buf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(buf.get(), nullptr);

    std::vector<void*> mhandles;
    for (auto& c : conns) {
        void* comm = nullptr;
        if (rank == c.senderRank)   comm = c.sendComm;
        if (rank == c.receiverRank) comm = c.recvComm;
        void* mh = nullptr;
        if (comm) {
            ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &mh), ncclSuccess);
        }
        mhandles.push_back(mh);
    }
    auto mrCleanup = makeScopeGuard([&]() {
        for (size_t i = 0; i < conns.size(); i++) {
            if (!mhandles[i]) continue;
            void* comm = nullptr;
            if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
            if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
            if (comm) DeregisterMemory(comm, mhandles[i]);
        }
    });

    static constexpr int kIters = 100;
    for (int iter = 0; iter < kIters; iter++) {
        for (size_t c = 0; c < conns.size(); c++) {
            int seed = iter * 10 + conns[c].senderRank;
            DoDirectedSendRecv(conns[c], buf.get(), buf.get(), sz,
                               /*tag=*/iter, mhandles[c], mhandles[c], seed);
        }
    }

    // Deregister memory before closing connections (comm must be valid for DeregMr).
    for (size_t i = 0; i < conns.size(); i++) {
        if (!mhandles[i]) continue;
        void* comm = nullptr;
        if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
        if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
        if (comm) DeregisterMemory(comm, mhandles[i]);
        mhandles[i] = nullptr;
    }
    mrCleanup.dismiss();

    for (auto& c : conns) CloseDirectedConnection(c);
}

// B5.  BidirectionalMultiRank — all pairs bidirectional, 4 ranks, 50 iters.
TEST_F(NetIbMPITest, BidirectionalMultiRank) {
    ASSERT_TRUE(validateTestPrerequisites(kMinFourProcesses, kMinFourProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    int nranks = MPIEnvironment::world_size;
    AssertInitAndGetDevices(nullptr);

    // All-to-all gives us both directions for every pair
    std::vector<DirectedConnection> conns;
    SetupAllToAll(0, nranks, conns);

    const size_t sz = kSmallBufferSize;
    auto buf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(buf.get(), nullptr);

    std::vector<void*> mhandles;
    for (auto& c : conns) {
        void* comm = nullptr;
        if (rank == c.senderRank)   comm = c.sendComm;
        if (rank == c.receiverRank) comm = c.recvComm;
        void* mh = nullptr;
        if (comm) {
            ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &mh), ncclSuccess);
        }
        mhandles.push_back(mh);
    }
    auto mrCleanup = makeScopeGuard([&]() {
        for (size_t i = 0; i < conns.size(); i++) {
            if (!mhandles[i]) continue;
            void* comm = nullptr;
            if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
            if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
            if (comm) DeregisterMemory(comm, mhandles[i]);
        }
    });

    static constexpr int kIters = 50;
    for (int iter = 0; iter < kIters; iter++) {
        for (size_t c = 0; c < conns.size(); c++) {
            int seed = iter * 100 + conns[c].senderRank * 10 + conns[c].receiverRank;
            DoDirectedSendRecv(conns[c], buf.get(), buf.get(), sz,
                               /*tag=*/iter, mhandles[c], mhandles[c], seed);
        }
    }

    // Deregister memory before closing connections (comm must be valid for DeregMr).
    for (size_t i = 0; i < conns.size(); i++) {
        if (!mhandles[i]) continue;
        void* comm = nullptr;
        if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
        if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
        if (comm) DeregisterMemory(comm, mhandles[i]);
        mhandles[i] = nullptr;
    }
    mrCleanup.dismiss();

    for (auto& c : conns) CloseDirectedConnection(c);
}

// B6.  LongRunningMultiRank — fan-in 3→1, 1000 iterations, RDMA checkpoints.
TEST_F(NetIbMPITest, LongRunningMultiRank) {
    ASSERT_TRUE(validateTestPrerequisites(kMinFourProcesses, kMinFourProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    std::vector<DirectedConnection> conns;
    SetupFanIn(0, 0, {1, 2, 3}, conns);

    const size_t sz = kSmallBufferSize;
    auto buf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(buf.get(), nullptr);

    std::vector<void*> mhandles;
    for (auto& c : conns) {
        void* comm = nullptr;
        if (rank == c.senderRank)   comm = c.sendComm;
        if (rank == c.receiverRank) comm = c.recvComm;
        void* mh = nullptr;
        if (comm) {
            ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &mh), ncclSuccess);
        }
        mhandles.push_back(mh);
    }
    auto mrCleanup = makeScopeGuard([&]() {
        for (size_t i = 0; i < conns.size(); i++) {
            if (!mhandles[i]) continue;
            void* comm = nullptr;
            if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
            if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
            if (comm) DeregisterMemory(comm, mhandles[i]);
        }
    });

    RdmaResourceCounts before = CaptureRdmaResources();

    static constexpr int kIters = 1000;
    for (int iter = 0; iter < kIters; iter++) {
        for (size_t c = 0; c < conns.size(); c++) {
            int seed = iter * 10 + conns[c].senderRank;
            DoDirectedSendRecv(conns[c], buf.get(), buf.get(), sz,
                               /*tag=*/iter % 1000, mhandles[c], mhandles[c], seed);
        }
        if ((iter + 1) % 250 == 0) {
            MPI_Barrier(MPI_COMM_WORLD);
            RdmaResourceCounts cp = CaptureRdmaResources();
            char label[64];
            snprintf(label, sizeof(label), "iter %d", iter + 1);
            AssertNoRdmaLeaks(before, cp, label);
        }
    }

    // Deregister memory before closing connections (comm must be valid for DeregMr).
    for (size_t i = 0; i < conns.size(); i++) {
        if (!mhandles[i]) continue;
        void* comm = nullptr;
        if (rank == conns[i].senderRank)   comm = conns[i].sendComm;
        if (rank == conns[i].receiverRank) comm = conns[i].recvComm;
        if (comm) DeregisterMemory(comm, mhandles[i]);
        mhandles[i] = nullptr;
    }
    mrCleanup.dismiss();

    for (auto& c : conns) CloseDirectedConnection(c);
}

// =====================================================================
//  Group G: Bidirectional (2-rank)
// =====================================================================

// G1.  BidirectionalSaturation — two opposing connections, full-duplex,
//      50 iterations.
TEST_F(NetIbMPITest, BidirectionalSaturation) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    // Forward: rank 1 → rank 0
    DirectedConnection fwd;
    SetupDirectedConnection(0, fwd, /*sender=*/1, /*receiver=*/0, /*mpiTag=*/800);

    // Backward: rank 0 → rank 1
    DirectedConnection bwd;
    SetupDirectedConnection(0, bwd, /*sender=*/0, /*receiver=*/1, /*mpiTag=*/801);

    const size_t sz = kSmallBufferSize;

    // Separate buffers per connection — MR handles are not cross-comm portable
    auto fwdBuf = makeHostBufferAutoGuard(malloc(sz));
    auto bwdBuf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(fwdBuf.get(), nullptr);
    ASSERT_NE(bwdBuf.get(), nullptr);

    // Register for each connection this rank uses
    void* fwdMh = nullptr;
    void* bwdMh = nullptr;
    {
        void* fwdComm = (rank == fwd.senderRank) ? fwd.sendComm : fwd.recvComm;
        void* bwdComm = (rank == bwd.senderRank) ? bwd.sendComm : bwd.recvComm;
        if (fwdComm) ASSERT_EQ(RegisterMemory(fwdComm, fwdBuf.get(), sz, NCCL_PTR_HOST, &fwdMh), ncclSuccess);
        if (bwdComm) ASSERT_EQ(RegisterMemory(bwdComm, bwdBuf.get(), sz, NCCL_PTR_HOST, &bwdMh), ncclSuccess);
    }
    static constexpr int kIters = 50;
    for (int iter = 0; iter < kIters; iter++) {
        // Forward direction
        DoDirectedSendRecv(fwd, fwdBuf.get(), fwdBuf.get(), sz,
                           /*tag=*/iter, fwdMh, fwdMh, iter * 2);
        // Backward direction
        DoDirectedSendRecv(bwd, bwdBuf.get(), bwdBuf.get(), sz,
                           /*tag=*/iter, bwdMh, bwdMh, iter * 2 + 1);
    }

    // Deregister MR before closing connections
    if (fwdMh) {
        void* c = (rank == fwd.senderRank) ? fwd.sendComm : fwd.recvComm;
        DeregisterMemory(c, fwdMh);
    }
    if (bwdMh) {
        void* c = (rank == bwd.senderRank) ? bwd.sendComm : bwd.recvComm;
        DeregisterMemory(c, bwdMh);
    }

    CloseDirectedConnection(fwd);
    CloseDirectedConnection(bwd);
}

// =====================================================================
//  Group F: Endurance (2-rank)
// =====================================================================

// F1.  LongRunningEndurance — 10000 transfers on a single connection,
//      tag cycling, RDMA checkpoints.
TEST_F(NetIbMPITest, LongRunningEndurance) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, cp, guard);

    const size_t sz = kSmallBufferSize;
    auto buf = makeHostBufferAutoGuard(malloc(sz));
    ASSERT_NE(buf.get(), nullptr);

    void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;
    void* mh = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.get(), sz, NCCL_PTR_HOST, &mh), ncclSuccess);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    RdmaResourceCounts before = CaptureRdmaResources();

    static constexpr int kIters = 10000;
    for (int i = 0; i < kIters; i++) {
        DoSendRecv(cp.sendComm, cp.recvComm,
                   buf.get(), buf.get(), sz,
                   /*tag=*/i % 1000, mh, mh, i);

        if ((i + 1) % 2500 == 0) {
            RdmaResourceCounts cp2 = CaptureRdmaResources();
            char label[64];
            snprintf(label, sizeof(label), "iter %d", i + 1);
            AssertNoRdmaLeaks(before, cp2, label);
        }
    }
}

// =====================================================================
//  Group H: GPU/GDR (2-rank)
// =====================================================================

// H1.  GpuMemoryTransferStress — 5 alloc/transfer/flush/verify/free cycles.
TEST_F(NetIbMPITest, GpuMemoryTransferStress) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    // Check GDR support
    ncclNetProperties_t props;
    ASSERT_EQ(GetDeviceProperties(0, &props), ncclSuccess);
    if (!(props.ptrSupport & NCCL_PTR_CUDA)) {
        GTEST_SKIP() << "No GPU Direct RDMA support";
    }

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, cp, guard);

    static constexpr int kCycles = 5;
    const size_t sizes[] = {512 * 1024, 1024 * 1024, 2 * 1024 * 1024, 4 * 1024 * 1024, 1024 * 1024};

    for (int cycle = 0; cycle < kCycles; cycle++) {
        size_t sz = sizes[cycle];
        void* devBuf = nullptr;
        ASSERT_EQ(hipMalloc(&devBuf, sz), hipSuccess);
        auto devGuard = makeDeviceBufferAutoGuard(devBuf);

        void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;
        void* mh = nullptr;
        ASSERT_EQ(RegisterMemory(comm, devBuf, sz, NCCL_PTR_CUDA, &mh), ncclSuccess);
        NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

        if (rank == 1) {
            ASSERT_EQ(initializeBufferWithPattern<uint8_t>(devBuf, sz, makeBytePattern(cycle)), hipSuccess);
        }

        void* req = nullptr;
        if (rank == 0) {
            PostSingleRecv(cp.recvComm, devBuf, sz, /*tag=*/cycle, mh, &req);
        } else {
            PostSendWithRetry(cp.sendComm, devBuf, sz, /*tag=*/cycle, mh, &req);
        }

        // Use EXPECT_ (non-fatal) so both ranks always reach MPI_Barrier.
        // ASSERT_ here would exit the failing rank before the barrier, leaving
        // the other rank hung indefinitely.
        int rsz = 0;
        EXPECT_EQ(WaitForCompletion(req, &rsz, kLargeTransferTimeoutMs), ncclSuccess);
        MPI_Barrier(MPI_COMM_WORLD);

        // Flush on receiver
        if (rank == 0) {
            void* flushReq = nullptr;
            int flushSz = static_cast<int>(sz);
            ncclResult_t fr = FlushRecv(cp.recvComm, 1, &devBuf, &flushSz, &mh, &flushReq);
            if (fr == ncclSuccess && flushReq) {
                int fsz = 0;
                EXPECT_EQ(WaitForCompletion(flushReq, &fsz, kLargeTransferTimeoutMs), ncclSuccess);
            }
            // Verify
            bool ok = verifyBufferData<uint8_t>(devBuf, sz, makeBytePattern(cycle));
            EXPECT_TRUE(ok) << "GPU data mismatch at cycle " << cycle;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// =====================================================================
//  Group J: Rapid recycling (2-rank)
// =====================================================================

// J1.  RapidRecvPostDrain — 100 cycles of post-32-recv → send-32 → drain.
TEST_F(NetIbMPITest, RapidRecvPostDrain) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    int rank = MPIEnvironment::world_rank;
    AssertInitAndGetDevices(nullptr);

    ConnectionPair cp;
    NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, cp, guard);

    static constexpr int kCycles = 100;
    static constexpr int kBatch  = 32;
    const size_t sz = 256;

    auto buf = makeHostBufferAutoGuard(malloc(sz * kBatch));
    ASSERT_NE(buf.get(), nullptr);

    void* comm = (rank == 0) ? cp.recvComm : cp.sendComm;
    void* mh = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf.get(), sz * kBatch, NCCL_PTR_HOST, &mh), ncclSuccess);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    for (int cycle = 0; cycle < kCycles; cycle++) {
        if (rank == 0) {
            // Post all recvs
            std::vector<void*> reqs(kBatch, nullptr);
            for (int i = 0; i < kBatch; i++) {
                char* p = static_cast<char*>(buf.get()) + i * sz;
                PostSingleRecv(cp.recvComm, p, sz, /*tag=*/i, mh, &reqs[i]);
            }
            // Signal sender
            MPI_Barrier(MPI_COMM_WORLD);
            // Drain all
            for (int i = 0; i < kBatch; i++) {
                int rsz = 0;
                ASSERT_EQ(WaitForCompletion(reqs[i], &rsz, kStressTimeoutMs), ncclSuccess)
                    << "Drain failed cycle=" << cycle << " msg=" << i;
            }
        } else {
            // Wait for recvs to be posted
            MPI_Barrier(MPI_COMM_WORLD);
            // Send all
            for (int i = 0; i < kBatch; i++) {
                char* p = static_cast<char*>(buf.get()) + i * sz;
                fillHostBufferWithPattern<uint8_t>(p, sz, makeBytePattern(cycle * kBatch + i));
                void* req = nullptr;
                PostSendWithRetry(cp.sendComm, p, sz, /*tag=*/i, mh, &req);
                int rsz = 0;
                ASSERT_EQ(WaitForCompletion(req, &rsz, kStressTimeoutMs), ncclSuccess)
                    << "Send failed cycle=" << cycle << " msg=" << i;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// E8.  SetNetAttrNoOp — calls ncclIbSetNetAttr (the last entry in ncclNet_t).
//      The function is a pure no-op (two (void) casts + return ncclSuccess).
//      All it needs is a call to reach its body; FNDA shows 0 hits without this test.
//      Placed after Group J (not adjacent to E0-E7) because it was added in a later
//      coverage iteration; the test number is kept to preserve git blame continuity.
TEST_F(NetIbMPITest, SetNetAttrNoOp) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit));
    AssertInitAndGetDevices(nullptr);
    ASSERT_NE(net_->setNetAttr, nullptr);
    EXPECT_EQ(net_->setNetAttr(initCtx_, nullptr), ncclSuccess);
    MPI_Barrier(MPI_COMM_WORLD);
}


#endif /* MPI_TESTS_ENABLED */
