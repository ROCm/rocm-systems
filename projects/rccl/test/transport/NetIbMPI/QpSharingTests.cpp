/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Baseline MPI test suite for QP sharing (RCCL_IB_COMM_NGROUPS).  2-rank,
// single-IB-device (dev=0) only for this first pass — fan-in/fan-out/
// all-to-all coverage is deferred to a follow-up.  CAST scheduler,
// resiliency (PORT_FAILOVER/PORT_RECOVERY), and NIC Fusion are mutually
// exclusive with sharing at the source level (see QPSHARE_ENV_CHECK_OR_SKIP)
// and are out of scope here.
//
// RCCL_IB_COMM_NGROUPS and RCCL_IB_QP_DEPTH_MULTIPLIER are RCCL_PARAM-backed
// and therefore cached for the lifetime of the process (src/include/param.h)
// — every test below assumes a fixed value of both for its entire run, and
// the CI wiring (net_ib_transport.json) gives each distinct NGROUPS/depth
// combination its own preset/process rather than varying env vars between
// test_filter entries in one preset.

#include "NetIbMPITestBase.hpp"
#include "NetIbQpSharingInspect.hpp"

#ifdef MPI_TESTS_ENABLED

// =============================================================================
// Test: QpShareNGroupsOne
//
// RCCL_IB_COMM_NGROUPS=1: every connection to the same peer/dev/direction
// lands in group 0. First connection is PRIMARY, the rest are SECONDARY on
// the same physical QP. Verifies exact roles/refcount via the inspect API,
// and data integrity across all 8 connections.
// =============================================================================
TEST_F(NetIbMPITest, QpShareNGroupsOne) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    QPSHARE_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    constexpr int kNConns = 8;
    std::vector<void*> listenComms(kNConns, nullptr);
    std::vector<void*> sendComms(kNConns, nullptr);
    std::vector<void*> recvComms(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++)
        SetupCastConnection(/*dev=*/0, &listenComms[c], &sendComms[c], &recvComms[c]);

    constexpr size_t kMsgSz = 256;
    std::vector<std::vector<char>> sendBufs(kNConns, std::vector<char>(kMsgSz));
    std::vector<std::vector<char>> recvBufs(kNConns, std::vector<char>(kMsgSz));
    std::vector<void*> mhandles(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++) {
        void* comm   = (rank == 0) ? recvComms[c] : sendComms[c];
        char* regBuf = (rank == 0) ? recvBufs[c].data() : sendBufs[c].data();
        ASSERT_EQ(RegisterMemory(comm, regBuf, kMsgSz, NCCL_PTR_HOST, &mhandles[c]), ncclSuccess);
    }

    // Role/refcount are fixed at connect()/accept() time — no warmup traffic needed.
    constexpr int kTagBase = 500;
    uint32_t sharedQpn = 0;
    for (int c = 0; c < kNConns; c++) {
        void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
        struct ncclIbQpSharingState st = GetActualQpSharingState(comm);
        EXPECT_EQ(st.sharedGroupIdx, 0) << "conn " << c << " not in group 0";
        EXPECT_EQ(st.refcount, kNConns) << "conn " << c << " refcount mismatch";
        EXPECT_EQ(st.isSharedQpPrimary, c == 0) << "conn " << c << " role mismatch";
        if (c == 0) {
            sharedQpn = st.qpn[0];
        } else {
            EXPECT_EQ(st.qpn[0], sharedQpn) << "conn " << c << " not on the shared QP";
        }

        DoSendRecv(sendComms[c], recvComms[c],
                   (rank == 0) ? nullptr : sendBufs[c].data(),
                   (rank == 0) ? recvBufs[c].data() : nullptr,
                   kMsgSz, kTagBase + c, mhandles[c], mhandles[c], /*patternSeed=*/c);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int c = 0; c < kNConns; c++) {
        void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
        ASSERT_EQ(DeregisterMemory(comm, mhandles[c]), ncclSuccess);
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(recvComms[c]), ncclSuccess);
            ASSERT_EQ(CloseListenComm(listenComms[c]), ncclSuccess);
        } else {
            ASSERT_EQ(CloseSendComm(sendComms[c]), ncclSuccess);
        }
    }
}

// =============================================================================
// Test: QpShareNGroupsExceedsConns
//
// RCCL_IB_COMM_NGROUPS=16 with only 4 connections: groupIdx = totalRefs %
// ngroups gives each connection its own distinct group (0..3) — sharing
// degenerates to the pre-sharing baseline. Every comm must be its own
// PRIMARY with refcount==1.
// =============================================================================
TEST_F(NetIbMPITest, QpShareNGroupsExceedsConns) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    QPSHARE_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    constexpr int kNConns = 4;
    std::vector<void*> listenComms(kNConns, nullptr);
    std::vector<void*> sendComms(kNConns, nullptr);
    std::vector<void*> recvComms(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++)
        SetupCastConnection(/*dev=*/0, &listenComms[c], &sendComms[c], &recvComms[c]);

    constexpr size_t kMsgSz = 256;
    std::vector<std::vector<char>> sendBufs(kNConns, std::vector<char>(kMsgSz));
    std::vector<std::vector<char>> recvBufs(kNConns, std::vector<char>(kMsgSz));
    std::vector<void*> mhandles(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++) {
        void* comm   = (rank == 0) ? recvComms[c] : sendComms[c];
        char* regBuf = (rank == 0) ? recvBufs[c].data() : sendBufs[c].data();
        ASSERT_EQ(RegisterMemory(comm, regBuf, kMsgSz, NCCL_PTR_HOST, &mhandles[c]), ncclSuccess);
    }

    constexpr int kTagBase = 510;
    std::vector<uint32_t> qpns(kNConns, 0);
    for (int c = 0; c < kNConns; c++) {
        void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
        struct ncclIbQpSharingState st = GetActualQpSharingState(comm);
        EXPECT_EQ(st.sharedGroupIdx, c) << "conn " << c << " groupIdx mismatch";
        EXPECT_EQ(st.refcount, 1) << "conn " << c << " expected to be unshared";
        EXPECT_TRUE(st.isSharedQpPrimary) << "conn " << c << " must be its own PRIMARY";
        qpns[c] = st.qpn[0];

        DoSendRecv(sendComms[c], recvComms[c],
                   (rank == 0) ? nullptr : sendBufs[c].data(),
                   (rank == 0) ? recvBufs[c].data() : nullptr,
                   kMsgSz, kTagBase + c, mhandles[c], mhandles[c], /*patternSeed=*/c + 100);
    }
    for (int c = 0; c < kNConns; c++)
        for (int o = c + 1; o < kNConns; o++)
            EXPECT_NE(qpns[c], qpns[o]) << "conn " << c << " and " << o << " share a QP unexpectedly";

    MPI_Barrier(MPI_COMM_WORLD);
    for (int c = 0; c < kNConns; c++) {
        void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
        ASSERT_EQ(DeregisterMemory(comm, mhandles[c]), ncclSuccess);
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(recvComms[c]), ncclSuccess);
            ASSERT_EQ(CloseListenComm(listenComms[c]), ncclSuccess);
        } else {
            ASSERT_EQ(CloseSendComm(sendComms[c]), ncclSuccess);
        }
    }
}

// =============================================================================
// Test: QpSharePrimarySecondaryMix
//
// RCCL_IB_COMM_NGROUPS=2 with 3 connections: totalRefs 0,1,2 % 2 gives
// group assignment {0, 1, 0} — group 0 ends up with a PRIMARY+SECONDARY
// pair (refcount==2), group 1 has a single PRIMARY (refcount==1), both
// present in the same run.
// =============================================================================
TEST_F(NetIbMPITest, QpSharePrimarySecondaryMix) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    QPSHARE_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    constexpr int kNConns = 3;
    std::vector<void*> listenComms(kNConns, nullptr);
    std::vector<void*> sendComms(kNConns, nullptr);
    std::vector<void*> recvComms(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++)
        SetupCastConnection(/*dev=*/0, &listenComms[c], &sendComms[c], &recvComms[c]);

    constexpr size_t kMsgSz = 256;
    std::vector<std::vector<char>> sendBufs(kNConns, std::vector<char>(kMsgSz));
    std::vector<std::vector<char>> recvBufs(kNConns, std::vector<char>(kMsgSz));
    std::vector<void*> mhandles(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++) {
        void* comm   = (rank == 0) ? recvComms[c] : sendComms[c];
        char* regBuf = (rank == 0) ? recvBufs[c].data() : sendBufs[c].data();
        ASSERT_EQ(RegisterMemory(comm, regBuf, kMsgSz, NCCL_PTR_HOST, &mhandles[c]), ncclSuccess);
    }

    std::vector<struct ncclIbQpSharingState> states(kNConns);
    for (int c = 0; c < kNConns; c++) {
        void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
        states[c] = GetActualQpSharingState(comm);
    }

    EXPECT_EQ(states[0].sharedGroupIdx, 0);
    EXPECT_TRUE(states[0].isSharedQpPrimary);
    EXPECT_EQ(states[0].refcount, 2);

    EXPECT_EQ(states[1].sharedGroupIdx, 1);
    EXPECT_TRUE(states[1].isSharedQpPrimary);
    EXPECT_EQ(states[1].refcount, 1);

    EXPECT_EQ(states[2].sharedGroupIdx, 0);
    EXPECT_FALSE(states[2].isSharedQpPrimary);
    EXPECT_EQ(states[2].refcount, 2);
    EXPECT_EQ(states[2].qpn[0], states[0].qpn[0]) << "conn 2 must share conn 0's QP";
    EXPECT_NE(states[1].qpn[0], states[0].qpn[0]) << "conn 1 must not share group 0's QP";

    constexpr int kTagBase = 520;
    for (int c = 0; c < kNConns; c++) {
        DoSendRecv(sendComms[c], recvComms[c],
                   (rank == 0) ? nullptr : sendBufs[c].data(),
                   (rank == 0) ? recvBufs[c].data() : nullptr,
                   kMsgSz, kTagBase + c, mhandles[c], mhandles[c], /*patternSeed=*/c + 200);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int c = 0; c < kNConns; c++) {
        void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
        ASSERT_EQ(DeregisterMemory(comm, mhandles[c]), ncclSuccess);
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(recvComms[c]), ncclSuccess);
            ASSERT_EQ(CloseListenComm(listenComms[c]), ncclSuccess);
        } else {
            ASSERT_EQ(CloseSendComm(sendComms[c]), ncclSuccess);
        }
    }
}

// =============================================================================
// Test: QpShareSendRecvMultipleSizes
//
// RCCL_IB_COMM_NGROUPS=2, a single connection. Size sweep through the
// forced-prepost data path (common.cc:90-98) at every notable boundary:
// zero-length, single-byte, and powers-of-two +/-1 up to 1MB.
// =============================================================================
TEST_F(NetIbMPITest, QpShareSendRecvMultipleSizes) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    QPSHARE_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

    const std::vector<size_t> kSizes = {0, 1, 127, 128, 4095, 4096, 65535, 65536, 1 << 20};
    const size_t kMaxSz = kSizes.back();

    std::vector<char> sendBuf(kMaxSz);
    std::vector<char> recvBuf(kMaxSz);
    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf.data()) : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMaxSz, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    constexpr int kTagBase = 530;
    for (size_t i = 0; i < kSizes.size(); i++) {
        memset(recvBuf.data(), 0, kMaxSz);
        DoSendRecv(sendComm, recvComm,
                   sendBuf.data(), recvBuf.data(),
                   kSizes[i], kTagBase + static_cast<int>(i), mhandle, mhandle,
                   /*patternSeed=*/static_cast<int>(i) + 300);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: QpShareDepthMultiplierDefault
//
// RCCL_IB_COMM_NGROUPS=1, RCCL_IB_QP_DEPTH_MULTIPLIER left at its default
// (1). Documents a known, current limitation rather than testing the happy
// path: every comm that joins a shared QP (PRIMARY or SECONDARY) unconditionally
// preposts a full NET_IB_MAX_REQUESTS (256) receive WQEs
// (IbCastReceiverPrePostReceiveWorkRequests), but the shared QP's max_recv_wr
// is sized once, at PRIMARY creation, to NET_IB_MAX_REQUESTS * depthMultiplier.
// At the default multiplier of 1, a 2nd comm joining any group always
// overflows the RQ (ibv_post_recv() ENOMEM) — this is a hard connect-time
// failure, not a subtle completion/perf issue. RCCL_IB_QP_DEPTH_MULTIPLIER
// must currently be set manually to at least the number of comms expected to
// share a group (see QpShareNGroupsOne / QpShareDepthMultiplierRaised, which
// do so explicitly).
//
// This test does NOT use SetupCastConnection for the 2nd connection, since
// that helper's ASSERT_EQ/ASSERT_NE are written for the happy path and would
// mark the (expected) failure as a fatal test error. Instead it drives
// accept()/connect() directly and asserts the failure is surfaced cleanly as
// a non-success return — not a crash — and that it does not corrupt the
// unrelated first connection.
// =============================================================================
TEST_F(NetIbMPITest, QpShareDepthMultiplierDefault) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;
    const int peer = 1 - rank;

    QPSHARE_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    // First connection: sole member of group 0. Must succeed even at
    // default depth — nothing to share yet.
    void* listenComm0 = nullptr;
    void* sendComm0   = nullptr;
    void* recvComm0   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm0, &sendComm0, &recvComm0);

    {
        void* comm0 = (rank == 0) ? recvComm0 : sendComm0;
        struct ncclIbQpSharingState st = GetActualQpSharingState(comm0);
        ASSERT_EQ(st.sharedGroupIdx, 0);
        ASSERT_TRUE(st.isSharedQpPrimary);
        ASSERT_EQ(st.refcount, 1);
    }

    // Second connection joins group 0 as SECONDARY. Expected to fail during
    // accept()/connect() at the default depth multiplier.
    ncclNetHandle_t handle;
    memset(&handle, 0, sizeof(handle));
    void* listenComm1 = nullptr;
    void* sendComm1   = nullptr;
    void* recvComm1   = nullptr;
    bool secondConnFailed = false;

    if (rank == 0) {
        ASSERT_EQ(CreateListenComm(/*dev=*/0, &handle, &listenComm1), ncclSuccess);
        MPI_Send(&handle, sizeof(handle), MPI_BYTE, peer, /*tag=*/1, MPI_COMM_WORLD);

        for (int i = 0; i < kMaxRetryAttempts && recvComm1 == nullptr && !secondConnFailed; i++) {
            ncclResult_t r = AcceptConnection(listenComm1, &recvComm1);
            if (r != ncclSuccess) {
                secondConnFailed = true;
                break;
            }
            if (recvComm1 == nullptr) usleep(kPollIntervalUs);
        }
        if (recvComm1 == nullptr) secondConnFailed = true;
    } else {
        MPI_Recv(&handle, sizeof(handle), MPI_BYTE, peer, /*tag=*/1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < kMaxRetryAttempts && sendComm1 == nullptr && !secondConnFailed; i++) {
            ncclResult_t r = ConnectToRemote(/*dev=*/0, &handle, &sendComm1);
            if (r != ncclSuccess) {
                secondConnFailed = true;
                break;
            }
            if (sendComm1 == nullptr) usleep(kPollIntervalUs);
        }
        if (sendComm1 == nullptr) secondConnFailed = true;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    EXPECT_TRUE(secondConnFailed)
        << "Expected the 2nd comm sharing group 0 to fail at default "
           "RCCL_IB_QP_DEPTH_MULTIPLIER=1 (known limitation — depth "
           "multiplier must be set >= comms-per-group). It unexpectedly "
           "succeeded: either this limitation has been fixed (update this "
           "test to the happy-path shape) or group/refcount accounting "
           "changed underneath this test.";

    // The first (unshared) connection must remain healthy — an RQ overflow
    // on conn1 must not corrupt or hang conn0.
    constexpr size_t kMsgSz = 256;
    std::vector<char> sendBuf0(kMsgSz), recvBuf0(kMsgSz);
    void* comm0    = (rank == 0) ? recvComm0 : sendComm0;
    char* regBuf0  = (rank == 0) ? recvBuf0.data() : sendBuf0.data();
    void* mhandle0 = nullptr;
    ASSERT_EQ(RegisterMemory(comm0, regBuf0, kMsgSz, NCCL_PTR_HOST, &mhandle0), ncclSuccess);
    DoSendRecv(sendComm0, recvComm0,
               (rank == 0) ? nullptr : sendBuf0.data(),
               (rank == 0) ? recvBuf0.data() : nullptr,
               kMsgSz, /*tag=*/541, mhandle0, mhandle0, /*patternSeed=*/1);

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm0, listenComm0, sendComm0, mhandle0);

    // Best-effort cleanup of whatever the failed 2nd attempt left behind.
    if (rank == 0) {
        if (recvComm1) CloseRecvComm(recvComm1);
        if (listenComm1) CloseListenComm(listenComm1);
    } else {
        if (sendComm1) CloseSendComm(sendComm1);
    }
}

// =============================================================================
// Test: QpShareDepthMultiplierRaised
//
// Same 8-comms-in-one-group shape as QpShareNGroupsOne (and runs in the same
// preset/process, since both now require RCCL_IB_QP_DEPTH_MULTIPLIER=8 —
// see QpShareDepthMultiplierDefault for why the multiplier must be sized to
// the group's comm count). Correctness boundary, not perf: asserts clean
// completion under concurrent post-all-then-wait-all load from all 8 comms
// sharing one QP now that depth is sized correctly.
// =============================================================================
TEST_F(NetIbMPITest, QpShareDepthMultiplierRaised) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    QPSHARE_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    constexpr int kNConns  = 8;
    constexpr int kTagBase = 545;
    std::vector<void*> listenComms(kNConns, nullptr);
    std::vector<void*> sendComms(kNConns, nullptr);
    std::vector<void*> recvComms(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++)
        SetupCastConnection(/*dev=*/0, &listenComms[c], &sendComms[c], &recvComms[c]);

    constexpr size_t kMsgSz = 256;
    std::vector<std::vector<char>> sendBufs(kNConns, std::vector<char>(kMsgSz));
    std::vector<std::vector<char>> recvBufs(kNConns, std::vector<char>(kMsgSz));
    std::vector<void*> mhandles(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++) {
        void* comm   = (rank == 0) ? recvComms[c] : sendComms[c];
        char* regBuf = (rank == 0) ? recvBufs[c].data() : sendBufs[c].data();
        ASSERT_EQ(RegisterMemory(comm, regBuf, kMsgSz, NCCL_PTR_HOST, &mhandles[c]), ncclSuccess);
    }

    for (int c = 0; c < kNConns; c++) {
        void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
        struct ncclIbQpSharingState st = GetActualQpSharingState(comm);
        ASSERT_EQ(st.sharedGroupIdx, 0) << "conn " << c << " not concentrated into group 0";
        ASSERT_EQ(st.refcount, kNConns) << "conn " << c << " refcount mismatch";
    }

    // Concurrent post-all-then-wait-all across every connection sharing the QP.
    std::vector<void*> reqs(kNConns, nullptr);
    if (rank == 0) {
        for (int c = 0; c < kNConns; c++) {
            void*  bufs[1]    = {recvBufs[c].data()};
            size_t sizes[1]   = {kMsgSz};
            int    tags[1]    = {kTagBase + c};
            void*  handles[1] = {mhandles[c]};
            ASSERT_EQ(PostRecv(recvComms[c], 1, bufs, sizes, tags, handles, &reqs[c]), ncclSuccess);
            ASSERT_NE(reqs[c], nullptr);
        }
        for (int c = 0; c < kNConns; c++) {
            int sz = 0;
            EXPECT_EQ(WaitForCompletion(reqs[c], &sz, kStressTimeoutMs), ncclSuccess)
                << "conn " << c << " did not complete under concurrent load";
        }
    } else {
        for (int c = 0; c < kNConns; c++) {
            fillHostBufferWithPattern<uint8_t>(sendBufs[c].data(), kMsgSz, makeBytePattern(c + 400));
            PostSendWithRetry(sendComms[c], sendBufs[c].data(), kMsgSz, kTagBase + c, mhandles[c], &reqs[c]);
        }
        for (int c = 0; c < kNConns; c++) {
            int sz = 0;
            EXPECT_EQ(WaitForCompletion(reqs[c], &sz, kStressTimeoutMs), ncclSuccess)
                << "conn " << c << " did not complete under concurrent load";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        for (int c = 0; c < kNConns; c++) {
            size_t errIdx; uint8_t errExp, errGot;
            EXPECT_TRUE(verifyHostBufferData<uint8_t>(recvBufs[c].data(), kMsgSz, makeBytePattern(c + 400),
                                                       0, 0.0, &errIdx, &errExp, &errGot))
                << "data mismatch conn " << c << " at byte " << errIdx;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int c = 0; c < kNConns; c++) {
        void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
        ASSERT_EQ(DeregisterMemory(comm, mhandles[c]), ncclSuccess);
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(recvComms[c]), ncclSuccess);
            ASSERT_EQ(CloseListenComm(listenComms[c]), ncclSuccess);
        } else {
            ASSERT_EQ(CloseSendComm(sendComms[c]), ncclSuccess);
        }
    }
}

// =============================================================================
// Test: QpShareManyConnsStress
//
// Flagship regression guard: 100 connections under RCCL_IB_COMM_NGROUPS=2
// (~50 SECONDARY comms per shared QP), sustained batched concurrent traffic
// on every connection. Models CastStressMultiRoundTwoConns.
// =============================================================================
TEST_F(NetIbMPITest, QpShareManyConnsStress) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    QPSHARE_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    constexpr int kNConns = 100;
    std::vector<void*> listenComms(kNConns, nullptr);
    std::vector<void*> sendComms(kNConns, nullptr);
    std::vector<void*> recvComms(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++)
        SetupCastConnection(/*dev=*/0, &listenComms[c], &sendComms[c], &recvComms[c]);

    // Sanity: confirm sharing is actually happening before spending the stress budget.
    {
        void* comm0 = (rank == 0) ? recvComms[0] : sendComms[0];
        struct ncclIbQpSharingState st0 = GetActualQpSharingState(comm0);
        EXPECT_GT(st0.refcount, 1) << "conn 0 is not shared — RCCL_IB_COMM_NGROUPS not in effect?";
    }

    constexpr int    kNMsgs = 20;
    constexpr size_t kMsgSz = 4096;
    constexpr size_t kBufSz = kNMsgs * kMsgSz;

    std::vector<std::vector<char>> sendBufs(kNConns, std::vector<char>(kBufSz));
    std::vector<std::vector<char>> recvBufs(kNConns, std::vector<char>(kBufSz));
    for (int c = 0; c < kNConns; c++) {
        for (size_t i = 0; i < kBufSz; i++) sendBufs[c][i] = static_cast<char>(((i + c) * 5 + 11) & 0xFF);
        memset(recvBufs[c].data(), 0, kBufSz);
    }

    std::vector<void*> mhandles(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++) {
        void* comm   = (rank == 0) ? recvComms[c] : sendComms[c];
        char* regBuf = (rank == 0) ? recvBufs[c].data() : sendBufs[c].data();
        ASSERT_EQ(RegisterMemory(comm, regBuf, kBufSz, NCCL_PTR_HOST, &mhandles[c]), ncclSuccess);
    }

    constexpr int kTagBase = 6000;
    for (int c = 0; c < kNConns; c++) {
        CastDoBatchSendRecv(rank, sendComms[c], recvComms[c],
                            sendBufs[c].data(), recvBufs[c].data(),
                            kMsgSz, kNMsgs, kTagBase + c * kNMsgs, mhandles[c]);
    }

    if (rank == 0) {
        for (int c = 0; c < kNConns; c++) {
            EXPECT_EQ(memcmp(recvBufs[c].data(), sendBufs[c].data(), kBufSz), 0)
                << "data corruption on conn " << c;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int c = 0; c < kNConns; c++) {
        void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
        ASSERT_EQ(DeregisterMemory(comm, mhandles[c]), ncclSuccess);
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(recvComms[c]), ncclSuccess);
            ASSERT_EQ(CloseListenComm(listenComms[c]), ncclSuccess);
        } else {
            ASSERT_EQ(CloseSendComm(sendComms[c]), ncclSuccess);
        }
    }
}

// =============================================================================
// Test: QpShareSharedRqSaturation
//
// Deliberately tries to starve/overflow the shared receive queue: posts one
// message from every connection in a group simultaneously (post-all before
// any wait), repeated over several rounds. Expect a clean pass — prepost
// (mandatory under sharing, see common.cc:90-98) keeps the RQ full
// regardless of which comm's traffic actually lands; failure here is a real
// regression, not an expected limitation.
// =============================================================================
TEST_F(NetIbMPITest, QpShareSharedRqSaturation) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    QPSHARE_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    constexpr int kNConns = 50;
    std::vector<void*> listenComms(kNConns, nullptr);
    std::vector<void*> sendComms(kNConns, nullptr);
    std::vector<void*> recvComms(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++)
        SetupCastConnection(/*dev=*/0, &listenComms[c], &sendComms[c], &recvComms[c]);

    constexpr size_t kMsgSz = 512;
    std::vector<std::vector<char>> sendBufs(kNConns, std::vector<char>(kMsgSz));
    std::vector<std::vector<char>> recvBufs(kNConns, std::vector<char>(kMsgSz));
    std::vector<void*> mhandles(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++) {
        void* comm   = (rank == 0) ? recvComms[c] : sendComms[c];
        char* regBuf = (rank == 0) ? recvBufs[c].data() : sendBufs[c].data();
        ASSERT_EQ(RegisterMemory(comm, regBuf, kMsgSz, NCCL_PTR_HOST, &mhandles[c]), ncclSuccess);
    }

    constexpr int kRounds  = 10;
    constexpr int kTagBase = 7000;
    for (int round = 0; round < kRounds; round++) {
        std::vector<void*> reqs(kNConns, nullptr);
        const int tagRoundBase = kTagBase + round * kNConns;
        if (rank == 0) {
            for (int c = 0; c < kNConns; c++) {
                void*  bufs[1]    = {recvBufs[c].data()};
                size_t sizes[1]   = {kMsgSz};
                int    tags[1]    = {tagRoundBase + c};
                void*  handles[1] = {mhandles[c]};
                ASSERT_EQ(PostRecv(recvComms[c], 1, bufs, sizes, tags, handles, &reqs[c]), ncclSuccess);
                ASSERT_NE(reqs[c], nullptr);
            }
            for (int c = 0; c < kNConns; c++) {
                int sz = 0;
                ASSERT_EQ(WaitForCompletion(reqs[c], &sz, kStressTimeoutMs), ncclSuccess)
                    << "round " << round << " conn " << c;
            }
        } else {
            for (int c = 0; c < kNConns; c++) {
                fillHostBufferWithPattern<uint8_t>(sendBufs[c].data(), kMsgSz,
                                                   makeBytePattern(round * kNConns + c));
                PostSendWithRetry(sendComms[c], sendBufs[c].data(), kMsgSz,
                                  tagRoundBase + c, mhandles[c], &reqs[c]);
            }
            for (int c = 0; c < kNConns; c++) {
                int sz = 0;
                ASSERT_EQ(WaitForCompletion(reqs[c], &sz, kStressTimeoutMs), ncclSuccess)
                    << "round " << round << " conn " << c;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            for (int c = 0; c < kNConns; c++) {
                size_t errIdx; uint8_t errExp, errGot;
                EXPECT_TRUE(verifyHostBufferData<uint8_t>(recvBufs[c].data(), kMsgSz,
                                                          makeBytePattern(round * kNConns + c),
                                                          0, 0.0, &errIdx, &errExp, &errGot))
                    << "round " << round << " conn " << c << " data mismatch at byte " << errIdx;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int c = 0; c < kNConns; c++) {
        void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
        ASSERT_EQ(DeregisterMemory(comm, mhandles[c]), ncclSuccess);
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(recvComms[c]), ncclSuccess);
            ASSERT_EQ(CloseListenComm(listenComms[c]), ncclSuccess);
        } else {
            ASSERT_EQ(CloseSendComm(sendComms[c]), ncclSuccess);
        }
    }
}

// =============================================================================
// Test: QpShareConnectionChurn
//
// 2048 connect->transfer->close cycles under RCCL_IB_COMM_NGROUPS=2, sized
// to deliberately exceed IBCAST_MAX_SHARED_QPS (1024). The shared-QP pool
// (g_IbCastSharedQpPool, qp_sharing.cc) only ever appends — freed slots are
// marked used=false but never reused/compacted — so enough PRIMARY
// create/destroy churn can exhaust it. When exhausted,
// IbCastRegisterSharedQp returns NULL and the caller (connect.cc:1090-1098)
// does not propagate a failure: the QP still works, but silently drops out
// of the shared-tracking pool, so a later comm in the same group cannot
// find it and refcount bookkeeping goes stale. Past the boundary this test
// asserts refcount stays > 0 at each checkpoint — a failure here is that
// exact latent bug, not a false positive.
// =============================================================================
TEST_F(NetIbMPITest, QpShareConnectionChurn) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    QPSHARE_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    constexpr int kIters = 2048;  // > IBCAST_MAX_SHARED_QPS (1024)
    const std::vector<int> kCheckpoints = {512, 1024, 1536, kIters};
    const size_t sz = 1024;

    std::vector<char> buf(sz);

    // Warmup connect+transfer+close so driver-internal CQs settle before baselining.
    {
        void* wListen = nullptr; void* wSend = nullptr; void* wRecv = nullptr;
        SetupCastConnection(/*dev=*/0, &wListen, &wSend, &wRecv);
        // Guard against pool exhaustion carried over from an earlier QP-sharing
        // test in this same process (the shared-QP pool is a process-global
        // that is never reset between TEST_F cases).
        if (HasFatalFailure()) return;
        void* wComm = (rank == 0) ? wRecv : wSend;
        void* wMh   = nullptr;
        ASSERT_EQ(RegisterMemory(wComm, buf.data(), sz, NCCL_PTR_HOST, &wMh), ncclSuccess);
        DoSendRecv(wSend, wRecv, buf.data(), buf.data(), sz, /*tag=*/599, wMh, wMh, /*seed=*/0);
        MPI_Barrier(MPI_COMM_WORLD);
        TeardownConnection(wRecv, wListen, wSend, wMh);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    RdmaResourceCounts before = CaptureRdmaResources();
    MPI_Barrier(MPI_COMM_WORLD);

    int checkpointIdx = 0;
    for (int iter = 0; iter < kIters; iter++) {
        void* listenComm = nullptr; void* sendComm = nullptr; void* recvComm = nullptr;
        SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);
        // ASSERT_* inside SetupCastConnection is fatal only within that function
        // (gtest semantics) — without this check, a connect failure past
        // IBCAST_MAX_SHARED_QPS would leave comm/recvComm null and this loop
        // would plow ahead into RegisterMemory() with a null comm and segfault,
        // masking the real (and expected) pool-exhaustion failure below.
        if (HasFatalFailure()) return;

        void* comm = (rank == 0) ? recvComm : sendComm;
        void* mh   = nullptr;
        ASSERT_EQ(RegisterMemory(comm, buf.data(), sz, NCCL_PTR_HOST, &mh), ncclSuccess);

        DoSendRecv(sendComm, recvComm, buf.data(), buf.data(), sz,
                   /*tag=*/iter % 1000, mh, mh, iter);

        if (checkpointIdx < static_cast<int>(kCheckpoints.size()) &&
            iter == kCheckpoints[checkpointIdx] - 1) {
            struct ncclIbQpSharingState st = GetActualQpSharingState(comm);
            EXPECT_GT(st.refcount, 0)
                << "refcount dropped to 0 at iter " << iter
                << " — likely IBCAST_MAX_SHARED_QPS pool exhaustion";
        }

        MPI_Barrier(MPI_COMM_WORLD);
        TeardownConnection(recvComm, listenComm, sendComm, mh);

        if (checkpointIdx < static_cast<int>(kCheckpoints.size()) &&
            iter == kCheckpoints[checkpointIdx] - 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            RdmaResourceCounts checkpoint = CaptureRdmaResources();
            AssertNoRdmaLeaks(before, checkpoint,
                              ("churn@" + std::to_string(kCheckpoints[checkpointIdx])).c_str());
            checkpointIdx++;
        }
    }
}

// =============================================================================
// Test: QpShareBatchCreateDestroy
//
// Same pool-exhaustion angle as QpShareConnectionChurn, at bursty-batch
// cadence: 110 batches of 10 connections (1100 total, crossing
// IBCAST_MAX_SHARED_QPS between batch 100 and the final batch). RDMA and
// refcount checkpoints every 25 batches.
// =============================================================================
TEST_F(NetIbMPITest, QpShareBatchCreateDestroy) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    QPSHARE_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    constexpr int kBatches  = 110;
    constexpr int kPerBatch = 10;  // 1100 total connections, crosses 1024.
    const size_t sz = 512;

    std::vector<char> buf(sz);

    // Warmup connect+transfer+close so driver-internal CQs settle before baselining.
    {
        void* wListen = nullptr; void* wSend = nullptr; void* wRecv = nullptr;
        SetupCastConnection(/*dev=*/0, &wListen, &wSend, &wRecv);
        // Guard against pool exhaustion carried over from an earlier QP-sharing
        // test in this same process (the shared-QP pool is a process-global
        // that is never reset between TEST_F cases) — see QpShareConnectionChurn.
        if (HasFatalFailure()) return;
        void* wComm = (rank == 0) ? wRecv : wSend;
        void* wMh   = nullptr;
        ASSERT_EQ(RegisterMemory(wComm, buf.data(), sz, NCCL_PTR_HOST, &wMh), ncclSuccess);
        DoSendRecv(wSend, wRecv, buf.data(), buf.data(), sz, /*tag=*/699, wMh, wMh, /*seed=*/0);
        MPI_Barrier(MPI_COMM_WORLD);
        TeardownConnection(wRecv, wListen, wSend, wMh);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    RdmaResourceCounts before = CaptureRdmaResources();
    MPI_Barrier(MPI_COMM_WORLD);

    for (int batch = 0; batch < kBatches; batch++) {
        std::vector<void*> listenComms(kPerBatch, nullptr);
        std::vector<void*> sendComms(kPerBatch, nullptr);
        std::vector<void*> recvComms(kPerBatch, nullptr);
        std::vector<void*> mhandles(kPerBatch, nullptr);

        for (int c = 0; c < kPerBatch; c++) {
            SetupCastConnection(/*dev=*/0, &listenComms[c], &sendComms[c], &recvComms[c]);
            // See QpShareConnectionChurn: SetupCastConnection's ASSERT_* is only
            // fatal within itself, so a pool-exhaustion failure here would leave
            // this comm null and RegisterMemory() would segfault instead of the
            // failure surfacing cleanly.
            if (HasFatalFailure()) return;
            void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
            ASSERT_EQ(RegisterMemory(comm, buf.data(), sz, NCCL_PTR_HOST, &mhandles[c]), ncclSuccess);
        }

        for (int c = 0; c < kPerBatch; c++) {
            int seed = batch * kPerBatch + c;
            DoSendRecv(sendComms[c], recvComms[c], buf.data(), buf.data(), sz,
                       /*tag=*/c, mhandles[c], mhandles[c], seed);
        }

        if ((batch + 1) % 25 == 0 || batch == kBatches - 1) {
            void* comm0 = (rank == 0) ? recvComms[0] : sendComms[0];
            struct ncclIbQpSharingState st = GetActualQpSharingState(comm0);
            EXPECT_GT(st.refcount, 0)
                << "refcount dropped to 0 at batch " << batch
                << " — likely IBCAST_MAX_SHARED_QPS pool exhaustion";
        }

        for (int c = 0; c < kPerBatch; c++) {
            void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
            ASSERT_EQ(DeregisterMemory(comm, mhandles[c]), ncclSuccess);
            if (rank == 0) {
                ASSERT_EQ(CloseRecvComm(recvComms[c]), ncclSuccess);
                ASSERT_EQ(CloseListenComm(listenComms[c]), ncclSuccess);
            } else {
                ASSERT_EQ(CloseSendComm(sendComms[c]), ncclSuccess);
            }
        }

        if ((batch + 1) % 25 == 0 || batch == kBatches - 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            RdmaResourceCounts checkpoint = CaptureRdmaResources();
            AssertNoRdmaLeaks(before, checkpoint,
                              ("batch@" + std::to_string(batch + 1)).c_str());
        }
    }
}

#endif /* MPI_TESTS_ENABLED */
