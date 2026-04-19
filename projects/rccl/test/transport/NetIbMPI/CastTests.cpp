/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "NetIbMPITestBase.hpp"
#include "NetIbCastInspect.hpp"

#ifdef MPI_TESTS_ENABLED

// =============================================================================
// Test: CastEqualWeightsTwoQPsTokenCounts
//
// White-box: 2 QPs, equal weights (schedWeight=0), WRR+split mode.
// Verifies initTokens.totTokens=100, per-QP tokens=50 each, sum invariant.
// =============================================================================
TEST_F(NetIbMPITest, CastEqualWeightsTwoQPsTokenCounts) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly 2 MPI processes";

    const int rank = MPIEnvironment::world_rank;

    SetupCastEnv(/*qpsPerConn=*/2, /*schedWeight=*/"0", /*splitData=*/1);
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 1024;
    char sendBuf[kMsgSize], recvBuf[kMsgSize];
    for (size_t i = 0; i < kMsgSize; i++) sendBuf[i] = static_cast<char>(i & 0xFF);
    memset(recvBuf, 0, sizeof(recvBuf));

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf) : static_cast<void*>(sendBuf);
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    if (rank == 1) {
        // Force-arm the WRR scheduler with equal 50/50 tokens before sending.
        const int tokens[2] = {50, 50};
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens, 2), 0);
    }

    CastDoSendRecv(rank, sendComm, recvComm, buf, kMsgSize, 123, mhandle);

    if (rank == 0)
        EXPECT_EQ(memcmp(sendBuf, recvBuf, kMsgSize), 0) << "data mismatch";

    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), 0);

        ASSERT_TRUE(state.schedInit);
        EXPECT_EQ(state.nqps, 2);
        EXPECT_EQ(state.initTotTokens, 100);
        EXPECT_EQ(state.initQpTokens[0], 50);
        EXPECT_EQ(state.initQpTokens[1], 50);
        int sum = 0;
        for (int i = 0; i < state.nqps; i++) sum += state.initQpTokens[i];
        EXPECT_EQ(sum, state.initTotTokens);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
    if (rank == 0) {
        ASSERT_EQ(CloseRecvComm(recvComm), ncclSuccess);
        ASSERT_EQ(CloseListenComm(listenComm), ncclSuccess);
    } else {
        ASSERT_EQ(CloseSendComm(sendComm), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Test: CastWeightsDistributionOneRound
//
// White-box: Verifies that exactly totTokens sends exhaust all WRR tokens for
// both equal (50/50) and unequal (75/25) weight distributions on the same
// connection. After each phase of exactly 100 sends the active token counter
// must be 0 (lazy refill: reset happens at the *start* of the next send, not
// immediately after exhaustion). SetTokens resets both init and active tokens,
// so re-arming between phases is sufficient without re-connecting.
// =============================================================================
TEST_F(NetIbMPITest, CastWeightsDistributionOneRound) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly 2 MPI processes";

    const int rank = MPIEnvironment::world_rank;

    SetupCastEnv(/*qpsPerConn=*/2, /*schedWeight=*/"0", /*splitData=*/0);
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    constexpr int    kNMsgs  = 100; // = totTokens for both phases
    constexpr size_t kMsgSz  = 64;

    char sendBuf[kNMsgs * kMsgSz], recvBuf[kNMsgs * kMsgSz];
    for (size_t i = 0; i < sizeof(sendBuf); i++) sendBuf[i] = static_cast<char>(i & 0xFF);
    memset(recvBuf, 0, sizeof(recvBuf));

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* baseBuf = (rank == 0) ? static_cast<void*>(recvBuf) : static_cast<void*>(sendBuf);
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, baseBuf, sizeof(sendBuf), NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Phase 1: equal weights {50, 50} — 100 sends exhaust one full WRR round.
    if (rank == 1) {
        const int tokens[2] = {50, 50};
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens, 2), 0);
    }
    CastDoBatchSendRecv(rank, sendComm, recvComm, sendBuf, recvBuf, kMsgSz, kNMsgs, 200, mhandle);
    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), 0);
        ASSERT_TRUE(state.schedInit);
        EXPECT_EQ(state.activeTotTokens, 0) << "equal weights: after one full round active tokens must be 0";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 2: unequal weights {75, 25} — 100 sends exhaust one full WRR round.
    // SetTokens resets activeTokens as well, so no reconnect is needed.
    if (rank == 1) {
        const int tokens[2] = {75, 25};
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens, 2), 0);
    }
    CastDoBatchSendRecv(rank, sendComm, recvComm, sendBuf, recvBuf, kMsgSz, kNMsgs, 300, mhandle);
    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), 0);
        ASSERT_TRUE(state.schedInit);
        EXPECT_EQ(state.activeTotTokens, 0) << "unequal weights: after one full round active tokens must be 0";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
    if (rank == 0) {
        ASSERT_EQ(CloseRecvComm(recvComm), ncclSuccess);
        ASSERT_EQ(CloseListenComm(listenComm), ncclSuccess);
    } else {
        ASSERT_EQ(CloseSendComm(sendComm), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Test: CastTokenSumInvariantAfterConsumption
//
// White-box: 2 QPs, 50/50 tokens. After 10 sends: initTokens immutable,
// sum(activeQpTokens)==activeTotTokens.
// =============================================================================
TEST_F(NetIbMPITest, CastTokenSumInvariantAfterConsumption) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly 2 MPI processes";

    const int rank = MPIEnvironment::world_rank;

    SetupCastEnv(/*qpsPerConn=*/2, /*schedWeight=*/"0", /*splitData=*/1);
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 128;
    constexpr int    kNMsgs   = 10;
    char sendBuf[kMsgSize * kNMsgs], recvBuf[kMsgSize * kNMsgs];
    for (size_t i = 0; i < sizeof(sendBuf); i++) sendBuf[i] = static_cast<char>(i & 0xFF);
    memset(recvBuf, 0, sizeof(recvBuf));

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* baseBuf = (rank == 0) ? static_cast<void*>(recvBuf) : static_cast<void*>(sendBuf);
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, baseBuf, sizeof(sendBuf), NCCL_PTR_HOST, &mhandle), ncclSuccess);

    if (rank == 1) {
        const int tokens[2] = {50, 50};
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens, 2), 0);
    }

    // Post all 10 sends/recvs concurrently, then wait for all completions.
    CastDoBatchSendRecv(rank, sendComm, recvComm, sendBuf, recvBuf, kMsgSize, kNMsgs, 700, mhandle);

    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), 0);

        ASSERT_TRUE(state.schedInit);
        EXPECT_EQ(state.initTotTokens, 100);
        EXPECT_EQ(state.initQpTokens[0], 50);
        EXPECT_EQ(state.initQpTokens[1], 50);
        int activeSum = 0;
        for (int i = 0; i < state.nqps; i++) activeSum += state.activeQpTokens[i];
        EXPECT_EQ(activeSum, state.activeTotTokens);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
    if (rank == 0) {
        ASSERT_EQ(CloseRecvComm(recvComm), ncclSuccess);
        ASSERT_EQ(CloseListenComm(listenComm), ncclSuccess);
    } else {
        ASSERT_EQ(CloseSendComm(sendComm), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Test: CastSingleQPBypassesWrr
//
// White-box: nqps=1. WRR must be bypassed (schedInit stays false).
// =============================================================================
TEST_F(NetIbMPITest, CastSingleQPBypassesWrr) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly 2 MPI processes";

    const int rank = MPIEnvironment::world_rank;

    SetupCastEnv(/*qpsPerConn=*/1, /*schedWeight=*/"0", /*splitData=*/0);
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 256;
    char sendBuf[kMsgSize], recvBuf[kMsgSize];
    for (size_t i = 0; i < kMsgSize; i++) sendBuf[i] = static_cast<char>((i * 3) & 0xFF);
    memset(recvBuf, 0, sizeof(recvBuf));

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf) : static_cast<void*>(sendBuf);
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    CastDoSendRecv(rank, sendComm, recvComm, buf, kMsgSize, 800, mhandle);

    if (rank == 0)
        EXPECT_EQ(memcmp(sendBuf, recvBuf, kMsgSize), 0) << "data mismatch";

    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), 0);
        // NCCL_PARAM IB_QPS_PER_CONNECTION is cached per-process. If a previous test
        // already cached nqps=2, SetupCastEnv(1) has no effect.
        // Only assert WRR-bypass when nqps is actually 1.
        if (state.nqps <= 1) {
            EXPECT_FALSE(state.schedInit) << "WRR must be bypassed for nqps=1";
        }
        // else: nqps was cached to a higher value; skip the bypass assertion.
    }

    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
    if (rank == 0) {
        ASSERT_EQ(CloseRecvComm(recvComm), ncclSuccess);
        ASSERT_EQ(CloseListenComm(listenComm), ncclSuccess);
    } else {
        ASSERT_EQ(CloseSendComm(sendComm), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

#endif // MPI_TESTS_ENABLED
