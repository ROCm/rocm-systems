/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "NetIbMPITestBase.hpp"
#include "NetIbCastInspect.hpp"
#include <initializer_list>

#ifdef MPI_TESTS_ENABLED

// =============================================================================
// Test: CastEqualWeightsTwoQPsTokenCounts
//
// White-box: request 2 QPs, equal weights (schedWeight=0), WRR+split mode.
// Actual nqps determined at runtime (ionic caps at 1, mlx5 supports 2+).
// Verifies: initTokens.totTokens=100, per-QP tokens equal, sum invariant.
// =============================================================================
TEST_F(NetIbMPITest, CastEqualWeightsTwoQPsTokenCounts) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
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

    // Warmup: learn real nqps (NCCL_PARAM cache may cap below requested 2).
    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kMsgSize, 122, mhandle);
    ASSERT_GT(actualNqps, 0);

    // Arm equal-weight tokens.
    if (rank == 1) {
        const std::vector<int> tokens = EqualTokens(actualNqps);
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }

    CastDoSendRecv(rank, sendComm, recvComm, buf, kMsgSize, 123, mhandle);

    if (rank == 0)
        EXPECT_EQ(memcmp(sendBuf, recvBuf, kMsgSize), 0) << "data mismatch";

    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), ncclSuccess);
        ExpectEqualWeightInitTokens(state, actualNqps);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastWeightsDistributionOneRound
//
// White-box: Verifies that exactly totTokens sends exhaust all WRR tokens.
// Two phases on the same connection:
//   Phase 1: equal weights (100/nqps each) — totTokens sends → activeTokens == 0
//   Phase 2: unequal weights (triangular: higher-indexed QPs get fewer tokens)
//            — totTokens sends → activeTokens == 0
// SetTokens resets both init and active tokens, so no reconnect needed.
// =============================================================================
TEST_F(NetIbMPITest, CastWeightsDistributionOneRound) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    CAST_REQUIRE_UPDATE_INTERVAL_OR_SKIP(10000000);
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    constexpr int    kTotTokens = 100;
    constexpr size_t kMsgSz     = 64;

    // Buffer sized for kTotTokens messages.
    char sendBuf[kTotTokens * kMsgSz], recvBuf[kTotTokens * kMsgSz];
    for (size_t i = 0; i < sizeof(sendBuf); i++) sendBuf[i] = static_cast<char>(i & 0xFF);
    memset(recvBuf, 0, sizeof(recvBuf));

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* baseBuf = (rank == 0) ? static_cast<void*>(recvBuf) : static_cast<void*>(sendBuf);
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, baseBuf, sizeof(sendBuf), NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup: learn real nqps before arming tokens.
    const int actualNqps = GetActualNqps(sendComm, recvComm, baseBuf, kMsgSz, 199, mhandle);
    ASSERT_GT(actualNqps, 0);

    // Phase 1: equal weights, kTotTokens sends exhaust one full WRR round.
    if (rank == 1) {
        const std::vector<int> tokens = EqualTokens(actualNqps, kTotTokens);
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }
    CastDoBatchSendRecv(rank, sendComm, recvComm, sendBuf, recvBuf, kMsgSz, kTotTokens, 200, mhandle);
    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), ncclSuccess);
        ASSERT_TRUE(state.schedInit);
        EXPECT_EQ(state.activeTotTokens, 0) << "equal weights: after one full round active tokens must be 0";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 2: unequal weights (triangular: token[i] ∝ nqps-i), kTotTokens sends.
    if (rank == 1) {
        const int weightSum = actualNqps * (actualNqps + 1) / 2;
        std::vector<int> tokens(actualNqps);
        int allocated = 0;
        for (int i = 0; i < actualNqps - 1; i++) {
            tokens[i] = kTotTokens * (actualNqps - i) / weightSum;
            allocated += tokens[i];
        }
        tokens[actualNqps - 1] = kTotTokens - allocated;
        // For nqps=1 the sequence is trivially {100}; strictly-decreasing check only for nqps>1.
        if (actualNqps > 1) {
            for (int i = 0; i < actualNqps - 1; i++)
                ASSERT_GT(tokens[i], tokens[i + 1])
                    << "triangular sequence not strictly decreasing for nqps=" << actualNqps;
        }
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }
    CastDoBatchSendRecv(rank, sendComm, recvComm, sendBuf, recvBuf, kMsgSz, kTotTokens, 300, mhandle);
    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), ncclSuccess);
        ASSERT_TRUE(state.schedInit);
        EXPECT_EQ(state.activeTotTokens, 0) << "unequal weights: after one full round active tokens must be 0";
    }

    // Forward rank-1 gtest failures to rank 0 so the test's exit code reflects them.
    // (rank 1 has its default gtest listeners detached by main_mpi.cpp.)
    {
        int total_failed = 0; std::string msg;
        if (rank == 1) {
            auto* r = ::testing::UnitTest::GetInstance()->current_test_info()->result();
            for (int i = 0; i < r->total_part_count(); i++) {
                const auto& p = r->GetTestPartResult(i);
                if (p.failed()) { total_failed++; msg += std::string("  - ") + (p.summary() ? p.summary() : "") + "\n"; }
            }
            int len = (int)msg.size();
            MPI_Send(&total_failed, 1, MPI_INT, 0, 9620, MPI_COMM_WORLD);
            MPI_Send(&len,          1, MPI_INT, 0, 9621, MPI_COMM_WORLD);
            if (len > 0) MPI_Send(msg.data(), len, MPI_CHAR, 0, 9622, MPI_COMM_WORLD);
        } else if (rank == 0) {
            int len = 0;
            MPI_Recv(&total_failed, 1, MPI_INT, 1, 9620, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&len,          1, MPI_INT, 1, 9621, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (len > 0) {
                msg.assign(len, '\0');
                MPI_Recv(msg.data(), len, MPI_CHAR, 1, 9622, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                ADD_FAILURE() << "rank 1 reported " << total_failed << " failure(s):\n" << msg;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastTokenSumInvariantAfterConsumption
//
// White-box: Equal tokens. After 10 sends: initTokens immutable,
// sum(activeQpTokens)==activeTotTokens.
// =============================================================================
TEST_F(NetIbMPITest, CastTokenSumInvariantAfterConsumption) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
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

    // Warmup: learn real nqps.
    const int actualNqps = GetActualNqps(sendComm, recvComm, baseBuf, kMsgSize, 699, mhandle);
    ASSERT_GT(actualNqps, 0);

    if (rank == 1) {
        const std::vector<int> tokens = EqualTokens(actualNqps);
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }

    CastDoBatchSendRecv(rank, sendComm, recvComm, sendBuf, recvBuf, kMsgSize, kNMsgs, 700, mhandle);

    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), ncclSuccess);
        ExpectEqualWeightInitTokens(state, actualNqps);
        ExpectActiveTokenSumInvariant(state);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastSingleQPBypassesWrr
//
// White-box: nqps=1. WRR must be bypassed (schedInit stays false).
// =============================================================================
TEST_F(NetIbMPITest, CastSingleQPBypassesWrr) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
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
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), ncclSuccess);
        if (state.nqps <= 1) {
            EXPECT_FALSE(state.schedInit) << "WRR must be bypassed for nqps=1";
        }
        // else: nqps was cached to a higher value; skip the bypass assertion.
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastSchedParmsReflectEnvVars
//
// White-box: schedParms inside sendComm must match env vars:
// enable=true, doWrr=true, splitData=false, splitDataMin from env.
// =============================================================================
TEST_F(NetIbMPITest, CastSchedParmsReflectEnvVars) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    if (GetSplitDataMin() == 0) {
        GTEST_SKIP() << "RCCL_IB_QP_SCHED_SPLIT_DATA_MIN=0: library ignores zero, "
                        "state.splitDataMin would not match";
    }
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 64;
    char sendBuf[kMsgSize], recvBuf[kMsgSize];
    memset(sendBuf, 0xAB, sizeof(sendBuf));
    memset(recvBuf, 0, sizeof(recvBuf));

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf) : static_cast<void*>(sendBuf);
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    CastDoSendRecv(rank, sendComm, recvComm, buf, kMsgSize, 900, mhandle);

    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), ncclSuccess);
        EXPECT_TRUE(state.schedEnable);
        EXPECT_TRUE(state.doWrr);
        EXPECT_EQ(state.splitDataMin, GetSplitDataMin());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastCursorWrapsAtNqpsBoundary
//
// White-box: actual nqps determined at runtime. Tokens={0,...,0,1} (only last QP).
// The WRR while(1) loop must skip QP0..nqps-2 and select QP[nqps-1].
// After selection cursor advances to 0.
// =============================================================================
TEST_F(NetIbMPITest, CastCursorWrapsAtNqpsBoundary) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 128;
    char sendBuf[kMsgSize], recvBuf[kMsgSize];
    memset(sendBuf, 0xCC, kMsgSize);
    memset(recvBuf, 0,    kMsgSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf) : static_cast<void*>(sendBuf);
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup: learn real nqps.
    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kMsgSize, 999, mhandle);
    ASSERT_GT(actualNqps, 0);

    // Set single token on the last QP; all others get 0.
    if (rank == 1) {
        std::vector<int> tokens(actualNqps, 0);
        tokens[actualNqps - 1] = 1;
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }

    // The send must land on QP[nqps-1] and wrap cursor to 0.
    CastDoSendRecv(rank, sendComm, recvComm, buf, kMsgSize, 1000, mhandle);

    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), ncclSuccess);

        ASSERT_TRUE(state.schedInit);
        EXPECT_EQ(state.activeTotTokens, 0);
        EXPECT_EQ(state.qpIndex, 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastMaxQPCount128
//
// Run exactly actualNqps sends — one full WRR round.
// Verify every QP was selected exactly once: activeQpTokens[i]==0 for all i,
// activeTotTokens==0, and qpIndex==0 (cursor wrapped back to start).
// =============================================================================
TEST_F(NetIbMPITest, CastMaxQPCount128) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSz  = 32;
    constexpr size_t kBufSz  = (NCCL_IB_MAX_QPS + 1)  * kMsgSz;
    constexpr int    kBaseTag = 1300;

    std::vector<char> sendBuf(kBufSz, 0);
    std::vector<char> recvBuf(kBufSz, 0);
    for (size_t i = 0; i < kBufSz; i++) sendBuf[i] = static_cast<char>(i & 0xFF);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* baseBuf = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, baseBuf, kBufSz, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup: learn real nqps (driver caps 128 at hardware limit).
    const int actualNqps = GetActualNqps(sendComm, recvComm, baseBuf, kMsgSz, 999, mhandle);
    ASSERT_GT(actualNqps, 0);

    // Set 1 token per QP — each QP selected exactly once across kNMsgs sends.
    if (rank == 1) {
        std::vector<int> tokens(actualNqps, 1);
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }

    // Post all actualNqps sends/recvs concurrently (skip slot 0 used by warmup).
    CastDoBatchSendRecv(rank, sendComm, recvComm,
                        sendBuf.data() + kMsgSz, recvBuf.data() + kMsgSz,
                        kMsgSz, actualNqps, kBaseTag, mhandle);

    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), ncclSuccess);

        ASSERT_TRUE(state.schedInit);
        for (int i = 0; i < state.nqps; i++)
            EXPECT_EQ(state.initQpTokens[i], 1) << "QP " << i << " initToken must be 1";
        EXPECT_EQ(state.activeTotTokens, 0);
        for (int i = 0; i < state.nqps; i++)
            EXPECT_EQ(state.activeQpTokens[i], 0) << "QP " << i << " activeToken must be 0";
        EXPECT_EQ(state.qpIndex, 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastFourQPsMonotonicOrder
//
// White-box: request 4 QPs; actual nqps determined at runtime.
// Assign tokens as a strictly-decreasing sequence summing to 100 using
// triangular proportions: token[i] = round(100 * (nqps-i) / weightSum),
// where weightSum = nqps*(nqps+1)/2; the last token absorbs rounding residual.
// For nqps=1: sequence is trivially {100}; monotonic check is skipped.
// Run exactly 100 sends (= totTokens). Verify:
//   - initQpTokens are strictly decreasing (nqps>1)
//   - activeTotTokens == 0 after exactly totTokens sends
//   - activeQpTokens[i] == 0 for all i
// =============================================================================
TEST_F(NetIbMPITest, CastFourQPsMonotonicOrder) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    constexpr int    kNMsgs  = 100;
    constexpr size_t kMsgSz  = 32;
    constexpr size_t kBufSz  = (kNMsgs + 1) * kMsgSz;
    constexpr int    kBaseTag = 1500;

    std::vector<char> sendBuf(kBufSz, 0);
    std::vector<char> recvBuf(kBufSz, 0);
    for (size_t i = 0; i < kBufSz; i++) sendBuf[i] = static_cast<char>(i & 0xFF);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* baseBuf = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, baseBuf, kBufSz, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup: learn real nqps.
    const int actualNqps = GetActualNqps(sendComm, recvComm, baseBuf, kMsgSz, 998, mhandle);
    ASSERT_GT(actualNqps, 0);

    // Build strictly-decreasing token sequence summing to kNMsgs.
    if (rank == 1) {
        const int weightSum = actualNqps * (actualNqps + 1) / 2;
        std::vector<int> tokens(actualNqps);
        int allocated = 0;
        for (int i = 0; i < actualNqps - 1; i++) {
            tokens[i] = kNMsgs * (actualNqps - i) / weightSum;
            allocated += tokens[i];
        }
        tokens[actualNqps - 1] = kNMsgs - allocated;

        if (actualNqps > 1) {
            bool isDecreasing = true;
            for (int i = 0; i < actualNqps - 1; i++)
                if (tokens[i] <= tokens[i + 1]) { isDecreasing = false; break; }
            ASSERT_TRUE(isDecreasing)
                << "Generated token sequence is not strictly decreasing for nqps=" << actualNqps;
        }
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }

    // Verify initTokens immediately after SetTokens (before timer can overwrite).
    if (rank == 1) {
        struct ncclIbCastSchedState probe = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &probe), ncclSuccess);
        ASSERT_TRUE(probe.schedInit);
        if (actualNqps > 1) {
            for (int i = 0; i < probe.nqps - 1; i++)
                EXPECT_GT(probe.initQpTokens[i], probe.initQpTokens[i + 1])
                    << "initTokens not strictly decreasing at index " << i;
        }
    }

    // Phase 1: exactly kNMsgs sends = one full WRR round.
    CastDoBatchSendRecv(rank, sendComm, recvComm,
                        sendBuf.data() + kMsgSz, recvBuf.data() + kMsgSz,
                        kMsgSz, kNMsgs, kBaseTag, mhandle);

    if (rank == 1) {
        struct ncclIbCastSchedState state = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &state), ncclSuccess);
        ASSERT_TRUE(state.schedInit);
        EXPECT_EQ(state.activeTotTokens, 0);
        for (int i = 0; i < state.nqps; i++)
            EXPECT_EQ(state.activeQpTokens[i], 0)
                << "QP " << i << " activeToken must be 0 after full round";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastSplitDataThresholdBoundary
//
// White-box: splitDataMin from env. Per-QP threshold = splitDataMin * nqps.
//   size >= threshold → split path → 0 WRR tokens consumed
//   size  < threshold → WRR  path → 1 WRR token consumed
// Actual nqps determines threshold dynamically.
// =============================================================================
TEST_F(NetIbMPITest, CastSplitDataThresholdBoundary) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    {
        const char* v = getenv("NCCL_IB_SPLIT_DATA_ON_QPS");
        if (!v || strcmp(v, "1") != 0)
            GTEST_SKIP() << "CastSplitDataThresholdBoundary requires NCCL_IB_SPLIT_DATA_ON_QPS=1";
    }
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    void* comm = (rank == 0) ? recvComm : sendComm;

    // Phase 0: small warmup buffer to learn actualNqps before allocating the real buffer.
    constexpr size_t kWarmupSz = 128;
    std::vector<char> warmupBuf(kWarmupSz, 0);
    void* warmupBase = warmupBuf.data();
    void* warmupHandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, warmupBase, kWarmupSz, NCCL_PTR_HOST, &warmupHandle), ncclSuccess);
    const int actualNqps = GetActualNqps(sendComm, recvComm, warmupBase, 64, 1599, warmupHandle);
    ASSERT_GT(actualNqps, 0);
    ASSERT_EQ(DeregisterMemory(comm, warmupHandle), ncclSuccess);

    // threshold = splitDataMin * nqps (from net_ib_cast.cc: dataPerQp = size*nreqs/nqps)
    const size_t kSplitDataMin = GetSplitDataMin();
    if (kSplitDataMin == 0) {
        // With splitDataMin=0 every message goes through the split path regardless of size.
        // There is no WRR boundary to test, so skip rather than crash on zero-size buffer.
        GTEST_SKIP() << "RCCL_IB_QP_SCHED_SPLIT_DATA_MIN=0: no WRR/split boundary exists";
    }
    const size_t kThreshold    = kSplitDataMin * static_cast<size_t>(actualNqps);
    const size_t kSplitSz      = kThreshold;       // dataPerQp == splitDataMin → split
    const size_t kWrrSz        = kThreshold - 1;   // dataPerQp <  splitDataMin → WRR

    // Allocate and register the real buffer sized for the actual threshold.
    const size_t kBufSz = kThreshold;
    std::vector<char> sendBuf(kBufSz, 0x5A);
    std::vector<char> recvBuf(kBufSz, 0x00);
    void* baseBuf = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, baseBuf, kBufSz, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    if (rank == 1) {
        const std::vector<int> tokens = EqualTokens(actualNqps);
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }

    // Large send: split path — activeTotTokens must NOT change.
    int activeTotBeforeSplit = 0;
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        activeTotBeforeSplit = st.activeTotTokens;
    }
    CastDoSendRecv(rank, sendComm, recvComm, baseBuf, kSplitSz, 1600, mhandle);

    int activeTotAfterSplit = 0;
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        ASSERT_TRUE(st.schedInit);
        EXPECT_EQ(st.activeTotTokens, activeTotBeforeSplit)
            << "split path must not consume WRR tokens";
        activeTotAfterSplit = st.activeTotTokens;
    }

    // Small send: WRR path — activeTotTokens must decrease by exactly 1.
    CastDoSendRecv(rank, sendComm, recvComm, baseBuf, kWrrSz, 1601, mhandle);

    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        ASSERT_TRUE(st.schedInit);
        int delta = activeTotAfterSplit - st.activeTotTokens;
        EXPECT_EQ(delta, 1) << "WRR path must consume exactly one token";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastAlternatingWrrNonWrr
//
// White-box: Toggle doWrr mid-test on an established connection.
// Phase 1 (doWrr=true):  10 sends → 10 WRR tokens consumed
// Phase 2 (doWrr=false): 10 sends → 0 WRR tokens consumed
// Phase 3 (doWrr=true):  10 sends → 10 WRR tokens consumed
// =============================================================================
TEST_F(NetIbMPITest, CastAlternatingWrrNonWrr) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    CAST_REQUIRE_UPDATE_INTERVAL_OR_SKIP(10000000);
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    constexpr int    kPhase  = 10;
    constexpr size_t kMsgSz  = 64;
    constexpr int    kBaseTag = 1700;

    char sendBuf[kPhase * 3 * kMsgSz], recvBuf[kPhase * 3 * kMsgSz];
    memset(sendBuf, 0xAA, sizeof(sendBuf));
    memset(recvBuf, 0,    sizeof(recvBuf));

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* baseBuf = (rank == 0) ? static_cast<void*>(recvBuf) : static_cast<void*>(sendBuf);
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, baseBuf, sizeof(sendBuf), NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup: learn real nqps.
    const int actualNqps = GetActualNqps(sendComm, recvComm, baseBuf, kMsgSz, 1699, mhandle);
    ASSERT_GT(actualNqps, 0);

    if (rank == 1) {
        const std::vector<int> tokens = EqualTokens(actualNqps);
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }

    auto doPhase = [&](int phaseBase) {
        CastDoBatchSendRecv(rank, sendComm, recvComm,
                            sendBuf + phaseBase * kMsgSz, recvBuf + phaseBase * kMsgSz,
                            kMsgSz, kPhase, kBaseTag + phaseBase, mhandle);
    };

    // Phase 1: WRR on
    int activeTotBefore1 = 0;
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        activeTotBefore1 = st.activeTotTokens;
    }
    doPhase(0);
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        int consumed = activeTotBefore1 - st.activeTotTokens;
        ASSERT_TRUE(st.schedInit);
        EXPECT_EQ(consumed, kPhase) << "phase 1: expected " << kPhase << " WRR selections";

        ASSERT_EQ(ncclIbCastSetSchedParms(sendComm, true, false, false, GetSplitDataMin()), ncclSuccess);
    }

    int activeTotBefore2 = 0;
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        activeTotBefore2 = st.activeTotTokens;
    }
    doPhase(kPhase);
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        int consumed = activeTotBefore2 - st.activeTotTokens;
        EXPECT_EQ(consumed, 0) << "phase 2: doWrr=false must not consume WRR tokens";

        ASSERT_EQ(ncclIbCastSetSchedParms(sendComm, true, true, false, GetSplitDataMin()), ncclSuccess);
    }

    int activeTotBefore3 = 0;
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        activeTotBefore3 = st.activeTotTokens;
    }
    doPhase(kPhase * 2);
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        int consumed = activeTotBefore3 - st.activeTotTokens;
        EXPECT_EQ(consumed, kPhase) << "phase 3: expected " << kPhase << " WRR selections";
    }

    // Forward rank-1 gtest failures to rank 0 (default listeners detached on non-zero ranks).
    {
        int total_failed = 0; std::string msg;
        if (rank == 1) {
            auto* r = ::testing::UnitTest::GetInstance()->current_test_info()->result();
            for (int i = 0; i < r->total_part_count(); i++) {
                const auto& p = r->GetTestPartResult(i);
                if (p.failed()) { total_failed++; msg += std::string("  - ") + (p.summary() ? p.summary() : "") + "\n"; }
            }
            int len = (int)msg.size();
            MPI_Send(&total_failed, 1, MPI_INT, 0, 9630, MPI_COMM_WORLD);
            MPI_Send(&len,          1, MPI_INT, 0, 9631, MPI_COMM_WORLD);
            if (len > 0) MPI_Send(msg.data(), len, MPI_CHAR, 0, 9632, MPI_COMM_WORLD);
        } else if (rank == 0) {
            int len = 0;
            MPI_Recv(&total_failed, 1, MPI_INT, 1, 9630, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&len,          1, MPI_INT, 1, 9631, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (len > 0) {
                msg.assign(len, '\0');
                MPI_Recv(msg.data(), len, MPI_CHAR, 1, 9632, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                ADD_FAILURE() << "rank 1 reported " << total_failed << " failure(s):\n" << msg;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastEnableDisableSplitData
//
// White-box: Toggle splitData on an established connection.
// Message size = splitDataMin * nqps (at the per-QP threshold boundary).
// Phase 1 (splitData=false): oneQp WRR → 1 token consumed
// Phase 2 (splitData=true):  split path  → 0 tokens consumed
// Phase 3 (splitData=false): oneQp WRR → 1 token consumed
// =============================================================================
TEST_F(NetIbMPITest, CastEnableDisableSplitData) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    CAST_REQUIRE_UPDATE_INTERVAL_OR_SKIP(10000000);
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    void* comm = (rank == 0) ? recvComm : sendComm;

    // Phase 0: small warmup buffer to learn actualNqps before allocating the real buffer.
    constexpr size_t kWarmupSz = 128;
    std::vector<char> warmupBuf(kWarmupSz, 0);
    void* warmupBase = warmupBuf.data();
    void* warmupHandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, warmupBase, kWarmupSz, NCCL_PTR_HOST, &warmupHandle), ncclSuccess);
    const int actualNqps = GetActualNqps(sendComm, recvComm, warmupBase, 64, 1799, warmupHandle);
    ASSERT_GT(actualNqps, 0);
    ASSERT_EQ(DeregisterMemory(comm, warmupHandle), ncclSuccess);

    // Message at the split threshold: dataPerQp = splitDataMin → split when splitData=true.
    const size_t kSplitDataMin = GetSplitDataMin();
    if (kSplitDataMin == 0) {
        GTEST_SKIP() << "RCCL_IB_QP_SCHED_SPLIT_DATA_MIN=0: no WRR/split boundary exists";
    }
    const size_t kMsgSz        = kSplitDataMin * static_cast<size_t>(actualNqps);

    // Allocate and register the real buffer sized for the actual threshold.
    std::vector<char> sendBuf(kMsgSz, 0xBC);
    std::vector<char> recvBuf(kMsgSz, 0x00);
    void* baseBuf = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, baseBuf, kMsgSz, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    if (rank == 1) {
        const std::vector<int> tokens = EqualTokens(actualNqps);
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
        ASSERT_EQ(ncclIbCastSetSchedParms(sendComm, true, true, false, kSplitDataMin), ncclSuccess);
    }

    // Phase 1: splitData=false → oneQp WRR → 1 token consumed
    int activeTotBefore1ed = 0;
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        activeTotBefore1ed = st.activeTotTokens;
    }
    CastDoSendRecv(rank, sendComm, recvComm, baseBuf, kMsgSz, 1800, mhandle);
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        ASSERT_TRUE(st.schedInit);
        EXPECT_EQ(activeTotBefore1ed - st.activeTotTokens, 1);
        ASSERT_EQ(ncclIbCastSetSchedParms(sendComm, true, true, true, kSplitDataMin), ncclSuccess);
    }

    // Phase 2: splitData=true → split path → 0 tokens consumed
    int activeTotBefore2ed = 0;
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        activeTotBefore2ed = st.activeTotTokens;
    }
    CastDoSendRecv(rank, sendComm, recvComm, baseBuf, kMsgSz, 1801, mhandle);
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        EXPECT_EQ(activeTotBefore2ed - st.activeTotTokens, 0)
            << "split path must not consume tokens";
        ASSERT_EQ(ncclIbCastSetSchedParms(sendComm, true, true, false, kSplitDataMin), ncclSuccess);
    }

    // Phase 3: splitData=false → oneQp WRR → 1 token consumed
    int activeTotBefore3ed = 0;
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        activeTotBefore3ed = st.activeTotTokens;
    }
    CastDoSendRecv(rank, sendComm, recvComm, baseBuf, kMsgSz, 1802, mhandle);
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        EXPECT_EQ(activeTotBefore3ed - st.activeTotTokens, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastEnableDisableSched
//
// White-box: Toggle schedEnable on an established connection.
// Message is small (dataPerQp < splitDataMin) with splitData=true.
//   enable=true  → WRR oneQp path → 1 token consumed
//   enable=false → scheduler disabled, all-QP path → 0 tokens consumed
//   enable=true  → WRR resumes → 1 token consumed
// =============================================================================
TEST_F(NetIbMPITest, CastEnableDisableSched) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    CAST_REQUIRE_UPDATE_INTERVAL_OR_SKIP(10000000);
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    // Must be strictly below splitDataMin so dataPerQp < splitDataMin for any nqps,
    // ensuring messages take the WRR path (not the split path).
    if (GetSplitDataMin() == 0) {
        GTEST_SKIP() << "RCCL_IB_QP_SCHED_SPLIT_DATA_MIN=0: no WRR/split boundary exists";
    }
    const size_t kMsgSz = std::max<size_t>(64, GetSplitDataMin() - 1);
    std::vector<char> sendBuf(kMsgSz, 0xCD);
    std::vector<char> recvBuf(kMsgSz, 0);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSz, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup: learn real nqps.
    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kMsgSz, 1899, mhandle);
    ASSERT_GT(actualNqps, 0);

    if (rank == 1) {
        const std::vector<int> tokens = EqualTokens(actualNqps);
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }

    // Phase 1: enable=true → WRR path
    int activeTotBefore1es = 0;
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        activeTotBefore1es = st.activeTotTokens;
    }
    CastDoSendRecv(rank, sendComm, recvComm, buf, kMsgSz, 1900, mhandle);
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        ASSERT_TRUE(st.schedInit);
        EXPECT_EQ(activeTotBefore1es - st.activeTotTokens, 1);
        ASSERT_EQ(ncclIbCastSetSchedParms(sendComm, false, true, true, GetSplitDataMin()), ncclSuccess);
    }

    // Phase 2: enable=false → bypass (all-QP path)
    int activeTotBefore2es = 0;
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        activeTotBefore2es = st.activeTotTokens;
    }
    CastDoSendRecv(rank, sendComm, recvComm, buf, kMsgSz, 1901, mhandle);
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        EXPECT_EQ(activeTotBefore2es - st.activeTotTokens, 0)
            << "enable=false must not consume WRR tokens";
        ASSERT_EQ(ncclIbCastSetSchedParms(sendComm, true, true, true, GetSplitDataMin()), ncclSuccess);
    }

    // Phase 3: enable=true → WRR resumes
    int activeTotBefore3es = 0;
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        activeTotBefore3es = st.activeTotTokens;
    }
    CastDoSendRecv(rank, sendComm, recvComm, buf, kMsgSz, 1902, mhandle);
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        EXPECT_EQ(activeTotBefore3es - st.activeTotTokens, 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastSendRecvMultipleSizes
//
// splitData=true, splitDataMin from env. Threshold = splitDataMin * nqps.
// WRR  sizes (< threshold): each consumes 1 WRR token.
// Split sizes (>= threshold): 0 WRR tokens consumed.
// Data integrity verified for all sizes.
// =============================================================================
TEST_F(NetIbMPITest, CastSendRecvMultipleSizes) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    {
        const char* v = getenv("NCCL_IB_SPLIT_DATA_ON_QPS");
        if (!v || strcmp(v, "1") != 0)
            GTEST_SKIP() << "CastSendRecvMultipleSizes requires NCCL_IB_SPLIT_DATA_ON_QPS=1";
    }
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    constexpr size_t kBufSz = 524288; // 512 KB — fits all test sizes
    std::vector<char> sendBuf(kBufSz);
    std::vector<char> recvBuf(kBufSz);
    for (size_t i = 0; i < kBufSz; i++) sendBuf[i] = static_cast<char>(i & 0xFF);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* baseBuf = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, baseBuf, kBufSz, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup: learn real nqps to compute split threshold.
    const int actualNqps = GetActualNqps(sendComm, recvComm, baseBuf, 64, 1999, mhandle);
    ASSERT_GT(actualNqps, 0);

    const size_t kSplitDataMin = GetSplitDataMin();
    if (kSplitDataMin == 0) {
        GTEST_SKIP() << "RCCL_IB_QP_SCHED_SPLIT_DATA_MIN=0: no WRR/split boundary exists";
    }
    const size_t kThreshold    = kSplitDataMin * static_cast<size_t>(actualNqps);
    ASSERT_LE(kThreshold, kBufSz) << "buffer too small for threshold=" << kThreshold;

    if (rank == 1) {
        const std::vector<int> tokens = EqualTokens(actualNqps);
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }

    // Sizes below threshold → WRR path (1 token consumed each).
    // All entries must be strictly below kThreshold; filter out any that aren't
    // (e.g. 512 and 4096 may exceed kThreshold when splitDataMin is small).
    std::vector<size_t> kWrrSizes;
    for (size_t sz : std::initializer_list<size_t>{512, 4096, kSplitDataMin, kThreshold - 1}) {
        if (sz > 0 && sz < kThreshold && sz <= kBufSz)
            kWrrSizes.push_back(sz);
    }
    // Deduplicate (kSplitDataMin or 512 might equal kThreshold-1).
    kWrrSizes.erase(std::unique(kWrrSizes.begin(), kWrrSizes.end()), kWrrSizes.end());
    // Sizes at/above threshold → split path (0 tokens consumed).
    const std::vector<size_t> kSplitSizes = {kThreshold, kThreshold * 2};
    int baseTag = 2000;

    for (size_t sz : kWrrSizes) {
        if (sz > kBufSz) continue; // skip if buffer too small (shouldn't happen)
        int prevActiveTot = 0;
        if (rank == 1) {
            struct ncclIbCastSchedState st = {};
            ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
            prevActiveTot = st.activeTotTokens;
        }
        CastDoSendRecv(rank, sendComm, recvComm, baseBuf, sz, baseTag++, mhandle);
        if (rank == 0)
            EXPECT_EQ(memcmp(sendBuf.data(), recvBuf.data(), sz), 0) << "data mismatch at size " << sz;
        if (rank == 1) {
            struct ncclIbCastSchedState st = {};
            ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
            EXPECT_EQ(prevActiveTot - st.activeTotTokens, 1)
                << "WRR path must consume exactly 1 token for size=" << sz;
        }
    }

    for (size_t sz : kSplitSizes) {
        if (sz > kBufSz) continue;
        int prevActiveTot = 0;
        if (rank == 1) {
            struct ncclIbCastSchedState st = {};
            ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
            prevActiveTot = st.activeTotTokens;
        }
        CastDoSendRecv(rank, sendComm, recvComm, baseBuf, sz, baseTag++, mhandle);
        if (rank == 0)
            EXPECT_EQ(memcmp(sendBuf.data(), recvBuf.data(), sz), 0) << "data mismatch at size " << sz;
        if (rank == 1) {
            struct ncclIbCastSchedState st = {};
            ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
            EXPECT_EQ(prevActiveTot - st.activeTotTokens, 0)
                << "split path must not consume WRR tokens for size=" << sz;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastLargeTransfer
//
// Black/white-box: 16 MB transfer with splitData=true.
// dataPerQp = 16MB / nqps >> splitDataMin → split path taken.
// Verify: data integrity + 0 WRR tokens consumed.
// =============================================================================
TEST_F(NetIbMPITest, CastLargeTransfer) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    {
        const char* v = getenv("NCCL_IB_SPLIT_DATA_ON_QPS");
        if (!v || strcmp(v, "1") != 0)
            GTEST_SKIP() << "CastLargeTransfer requires NCCL_IB_SPLIT_DATA_ON_QPS=1";
    }
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSz = kLargeBufferSize; // 16 MB
    std::vector<char> sendBuf(kMsgSz);
    std::vector<char> recvBuf(kMsgSz, 0);
    for (size_t i = 0; i < kMsgSz; i++) sendBuf[i] = static_cast<char>(i & 0xFF);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* baseBuf = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, baseBuf, kMsgSz, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup: learn real nqps.
    const int actualNqps = GetActualNqps(sendComm, recvComm, baseBuf, 64, 2099, mhandle);
    ASSERT_GT(actualNqps, 0);

    if (rank == 1) {
        const std::vector<int> tokens = EqualTokens(actualNqps);
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }

    int activeTotBeforeLarge = 0;
    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        activeTotBeforeLarge = st.activeTotTokens;
    }
    CastDoSendRecv(rank, sendComm, recvComm, baseBuf, kMsgSz, 2100, mhandle);

    if (rank == 0)
        EXPECT_EQ(memcmp(sendBuf.data(), recvBuf.data(), kMsgSz), 0) << "data mismatch";

    if (rank == 1) {
        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        ASSERT_TRUE(st.schedInit);
        EXPECT_EQ(activeTotBeforeLarge - st.activeTotTokens, 0)
            << "16 MB transfer must use split path, not WRR";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastSendRecvZeroSize
//
// White-box: Zero-byte send with splitData=true.
// dataPerQp = 0 < splitDataMin → oneQp WRR path (regardless of nqps).
// Verify: send/recv complete with ncclSuccess, received size=0, 1 WRR token consumed.
// =============================================================================
TEST_F(NetIbMPITest, CastSendRecvZeroSize) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(0, &listenComm, &sendComm, &recvComm);

    constexpr size_t kRegSz = 64;
    char buf[kRegSz];
    memset(buf, 0, kRegSz);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kRegSz, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup: learn real nqps.
    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kRegSz, 2199, mhandle);
    ASSERT_GT(actualNqps, 0);

    if (rank == 1) {
        const std::vector<int> tokens = EqualTokens(actualNqps);
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
    }

    void* req = nullptr;
    int   sz  = -1;
    if (rank == 0) {
        void*  bufs[1]    = {buf};
        size_t sizes[1]   = {0};
        int    tags[1]    = {2200};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &req), ncclSuccess);
        ASSERT_NE(req, nullptr);
        ASSERT_EQ(WaitForCompletion(req, &sz, 10000), ncclSuccess);
        EXPECT_EQ(sz, 0) << "received size must be 0";
    } else {
        PostSendWithRetry(sendComm, buf, 0, 2200, mhandle, &req);
        ASSERT_EQ(WaitForCompletion(req, &sz, 10000), ncclSuccess);

        struct ncclIbCastSchedState st = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        ASSERT_TRUE(st.schedInit);
        EXPECT_EQ(st.initTotTokens - st.activeTotTokens, 1)
            << "zero-size send must still trigger WRR (dataPerQp=0 < splitDataMin)";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: CastStressMultiRoundTwoConns
//
// Stress test: 100 independent connections running concurrently, 5000 sends each
// (WRR small-message phase), followed by a ramping-size phase that exercises the
// WRR/split boundary across {splitDataMin/4, /2, *1, *4, *16} with 20 rounds each.
// Actual nqps determined at runtime. Works with nqps=1 (no WRR timer check) or
// nqps>1 (RTT timer fires and rewrites initTokens from asymmetric initial values).
//
// Invariants asserted:
//   - schedInit == true on conn[0].
//   - sum(activeQpTokens[i]) == activeTotTokens for conn[0].
//   - 0 <= activeTotTokens <= initTotTokens for conn[0].
//   - If nqps > 1: initQpTokens changed from asymmetric initial values (timer fired).
// =============================================================================
TEST_F(NetIbMPITest, CastStressMultiRoundTwoConns) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    constexpr int kNConns = 100;
    std::vector<void*> listenComms(kNConns, nullptr);
    std::vector<void*> sendComms(kNConns, nullptr);
    std::vector<void*> recvComms(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++)
        SetupCastConnection(/*dev=*/0, &listenComms[c], &sendComms[c], &recvComms[c]);

    // Scale msgs per connection inversely with connection count so total work stays constant.
    constexpr int kNMsgsTotal = 10000;
    constexpr int kNMsgs      = kNMsgsTotal / kNConns;  // 100 msgs × 100 conns = 10000 total
    if (GetSplitDataMin() == 0) {
        GTEST_SKIP() << "RCCL_IB_QP_SCHED_SPLIT_DATA_MIN=0: no WRR/split boundary exists";
    }
    const size_t kMsgSz = std::max<size_t>(64, GetSplitDataMin() - 1);
    const size_t kBufSz = static_cast<size_t>(kNMsgs) * kMsgSz;

    // One send/recv buffer pair per connection.
    std::vector<std::vector<char>> sendBufs(kNConns, std::vector<char>(kBufSz));
    std::vector<std::vector<char>> recvBufs(kNConns, std::vector<char>(kBufSz));
    for (int c = 0; c < kNConns; c++) {
        for (size_t i = 0; i < kBufSz; i++) {
            sendBufs[c][i] = static_cast<char>(((i + c) * 3 + 7)  & 0xFF);
        }
        memset(recvBufs[c].data(), 0, kBufSz);
    }

    std::vector<void*> mhandles(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++) {
        void* comm   = (rank == 0) ? recvComms[c] : sendComms[c];
        char* regBuf = (rank == 0) ? recvBufs[c].data() : sendBufs[c].data();
        ASSERT_EQ(RegisterMemory(comm, regBuf, kBufSz, NCCL_PTR_HOST, &mhandles[c]), ncclSuccess);
    }

    // Warmup on conn[0]: learn real nqps. All conns share same NCCL_PARAM → same nqps.
    const int actualNqps = GetActualNqps(sendComms[0], recvComms[0],
                                         (rank == 0) ? recvBufs[0].data() : sendBufs[0].data(),
                                         kMsgSz, 3999, mhandles[0]);
    ASSERT_GT(actualNqps, 0);

    // Arm tokens asymmetrically on conn[0] only (timer-fired check uses conn[0]).
    if (rank == 1) {
        std::vector<int> tokens(actualNqps);
        if (actualNqps == 1) {
            tokens[0] = 100;
        } else {
            tokens[0] = 70;
            tokens[1] = 30;
            for (int i = 2; i < actualNqps; i++) tokens[i] = 50;
        }
        ASSERT_EQ(ncclIbCastSetTokens(sendComms[0], tokens.data(), actualNqps), ncclSuccess);
    }

    constexpr int kBatch   = 16;
    constexpr int kTagBase = 10000;   // tag = kTagBase + c * kNMsgs + i

    // ── Phase 1: small-message WRR stress ────────────────────────────────────
    if (rank == 0) {
        for (int base = 0; base < kNMsgs; base += kBatch) {
            const int end = std::min(base + kBatch, kNMsgs);
            const int bsz = end - base;
            std::vector<std::vector<void*>> reqs(kNConns, std::vector<void*>(bsz, nullptr));
            for (int c = 0; c < kNConns; c++) {
                for (int i = base; i < end; i++) {
                    void*  bufs[1]    = {recvBufs[c].data() + i * kMsgSz};
                    size_t sizes[1]   = {kMsgSz};
                    int    tags[1]    = {kTagBase + c * kNMsgs + i};
                    void*  handles[1] = {mhandles[c]};
                    ASSERT_EQ(PostRecv(recvComms[c], 1, bufs, sizes, tags, handles,
                                       &reqs[c][i - base]), ncclSuccess);
                    ASSERT_NE(reqs[c][i - base], nullptr);
                }
            }
            for (int c = 0; c < kNConns; c++) {
                for (int i = 0; i < bsz; i++) {
                    int sz = 0;
                    ASSERT_EQ(WaitForCompletion(reqs[c][i], &sz, 10000), ncclSuccess);
                }
            }
        }
        for (int c = 0; c < kNConns; c++) {
            EXPECT_EQ(memcmp(recvBufs[c].data(), sendBufs[c].data(), kBufSz), 0)
                << "phase1 data corruption on conn " << c;
        }
    } else {
        for (int base = 0; base < kNMsgs; base += kBatch) {
            const int end = std::min(base + kBatch, kNMsgs);
            const int bsz = end - base;
            std::vector<std::vector<void*>> reqs(kNConns, std::vector<void*>(bsz, nullptr));
            for (int c = 0; c < kNConns; c++) {
                for (int i = base; i < end; i++)
                    PostSendWithRetry(sendComms[c], sendBufs[c].data() + i * kMsgSz, kMsgSz,
                                      kTagBase + c * kNMsgs + i, mhandles[c], &reqs[c][i - base]);
            }
            for (int c = 0; c < kNConns; c++) {
                for (int i = 0; i < bsz; i++) {
                    int sz = 0;
                    ASSERT_EQ(WaitForCompletion(reqs[c][i], &sz, 10000), ncclSuccess);
                }
            }
        }
    }

    // Collect sched state from conn[0] after phase 1.
    struct ncclIbCastSchedState stAfterPhase1 = {};
    if (rank == 0) {
        MPI_Recv(&stAfterPhase1, sizeof(stAfterPhase1), MPI_BYTE, 1, 9870, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        int nqps = 0;
        MPI_Recv(&nqps, 1, MPI_INT, 1, 9872, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        EXPECT_TRUE(stAfterPhase1.schedInit) << "conn[0]: schedInit must be true after phase 1";
        {
            int s = 0;
            for (int q = 0; q < stAfterPhase1.nqps; q++) s += stAfterPhase1.activeQpTokens[q];
            EXPECT_EQ(s, stAfterPhase1.activeTotTokens) << "conn[0]: token sum mismatch";
        }
        EXPECT_GE(stAfterPhase1.activeTotTokens, 0);
        EXPECT_LE(stAfterPhase1.activeTotTokens, stAfterPhase1.initTotTokens);
        if (nqps > 1) {
            EXPECT_NE(stAfterPhase1.initQpTokens[0], 70) << "conn[0]: RTT timer did not fire";
            EXPECT_NE(stAfterPhase1.initQpTokens[1], 30) << "conn[0]: RTT timer did not fire";
        }
    } else {
        struct ncclIbCastSchedState st0 = {};
        ASSERT_EQ(ncclIbCastGetSchedState(sendComms[0], &st0), ncclSuccess);
        MPI_Send(&st0, sizeof(st0), MPI_BYTE, 0, 9870, MPI_COMM_WORLD);
        MPI_Send(&actualNqps, 1, MPI_INT, 0, 9872, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ── Phase 2: ramping-size stress ─────────────────────────────────────────
    // Sizes span WRR and split paths: /4, /2, *1, *4, *16 relative to splitDataMin.
    const uint32_t sdm = GetSplitDataMin();
    const std::vector<size_t> kRampSizes = {
        std::max<size_t>(64, sdm / 4),
        std::max<size_t>(64, sdm / 2),
        static_cast<size_t>(sdm),
        static_cast<size_t>(sdm) * 4,
        static_cast<size_t>(sdm) * 16,
    };
    constexpr int kLargeRounds = 20;
    const size_t  kRampBufSz   = kRampSizes.back();   // one MR per conn, sized for largest

    std::vector<std::vector<char>> rampSend(kNConns, std::vector<char>(kRampBufSz));
    std::vector<std::vector<char>> rampRecv(kNConns, std::vector<char>(kRampBufSz));
    for (int c = 0; c < kNConns; c++) {
        for (size_t i = 0; i < kRampBufSz; i++)
            rampSend[c][i] = static_cast<char>(((i + c) * 7 + 3) & 0xFF);
        memset(rampRecv[c].data(), 0, kRampBufSz);
    }

    std::vector<void*> rampHandles(kNConns, nullptr);
    for (int c = 0; c < kNConns; c++) {
        void* comm   = (rank == 0) ? recvComms[c] : sendComms[c];
        char* regBuf = (rank == 0) ? rampRecv[c].data() : rampSend[c].data();
        ASSERT_EQ(RegisterMemory(comm, regBuf, kRampBufSz, NCCL_PTR_HOST, &rampHandles[c]),
                  ncclSuccess);
    }

    // Tag space: 20000 + sizeIdx*kNConns*kLargeRounds + c*kLargeRounds + round
    constexpr int kRampTagBase = 20000;
    for (int si = 0; si < static_cast<int>(kRampSizes.size()); si++) {
        const size_t sz = kRampSizes[si];
        for (int round = 0; round < kLargeRounds; round++) {
            for (int c = 0; c < kNConns; c++) {
                const int tag = kRampTagBase + si * kNConns * kLargeRounds
                                + c * kLargeRounds + round;
                CastDoSendRecv(rank, sendComms[c], recvComms[c],
                               (rank == 0) ? rampRecv[c].data() : rampSend[c].data(),
                               sz, tag, rampHandles[c]);
            }
        }
    }

    if (rank == 0) {
        for (int c = 0; c < kNConns; c++) {
            EXPECT_EQ(memcmp(rampRecv[c].data(), rampSend[c].data(), kRampBufSz), 0)
                << "phase2 data corruption on conn " << c;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ── Teardown ─────────────────────────────────────────────────────────────
    for (int c = 0; c < kNConns; c++) {
        void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
        ASSERT_EQ(DeregisterMemory(comm, rampHandles[c]), ncclSuccess);
        ASSERT_EQ(DeregisterMemory(comm, mhandles[c]), ncclSuccess);
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(recvComms[c]), ncclSuccess);
            ASSERT_EQ(CloseListenComm(listenComms[c]), ncclSuccess);
        } else {
            ASSERT_EQ(CloseSendComm(sendComms[c]), ncclSuccess);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

#endif // MPI_TESTS_ENABLED
