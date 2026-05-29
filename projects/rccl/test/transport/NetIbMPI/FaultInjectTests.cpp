/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "NetIbMPITestBase.hpp"
#include "NetIbCastInspect.hpp"
#include "NetIbFaultInject.hpp"

#if defined(MPI_TESTS_ENABLED) && defined(ENABLE_FAULT_INJECTION)

// IB WC status codes used by FailoverErrorCodeWhitelist test.
// Defined here as constants to avoid depending on infiniband/verbs.h in tests.
static constexpr int kWcWrFlushErr    = 5;  // IBV_WC_WR_FLUSH_ERR
static constexpr int kWcRetryExcErr   = 12; // IBV_WC_RETRY_EXC_ERR
static constexpr int kWcRemAccessErr  = 10; // IBV_WC_REM_ACCESS_ERR
static constexpr int kWcGeneralErr    = 22; // IBV_WC_GENERAL_ERR

// Resiliency device state constants (from ncclIbResiliencyDevState enum).
static constexpr int kDevStateOk              = 0;  // ncclIbResiliencyDevStateOk
static constexpr int kDevStateRecoveryInProgress = 2;  // ncclIbResiliencyDevStateRecoveryInProgress
static constexpr int kDevStateRecoveryFailed   = 3;  // ncclIbResiliencyDevStateRecoveryFailed
static constexpr int kDevStateRecovered        = 4;  // ncclIbResiliencyDevStateRecovered
static constexpr int kDevStateErrorPermanent   = 5;  // ncclIbResiliencyDevStateErrorPermanent

// Helper: Create a NIC Fusion merged device from physical devices 0 and 1.
// Returns the merged device index, or -1 if fewer than 2 devices available.
// Both ranks must call this; result is broadcast from rank 0.
static int CreateMergedDeviceForFailover(ncclNet_t* net, int totalDevs) {
    int mergedDev = -1;
    if (totalDevs >= 2) {
        ncclNetVDeviceProps_t vProps = {};
        vProps.ndevs = 2;
        vProps.devs[0] = 0;
        vProps.devs[1] = 1;
        net->makeVDevice(&mergedDev, &vProps);
    }
    MPI_Bcast(&mergedDev, 1, MPI_INT, /*root=*/0, MPI_COMM_WORLD);
    return mergedDev;
}

// MPI tags used for inter-rank result forwarding (rank 1 → rank 0).
// Keep these distinct from NCCL-level recv tags (per-message ints passed to irecv).
static constexpr int kSchedStateMpiTag  = 9880;  // ncclIbCastSchedState from FaultInjCastSlowQpRebalances
static constexpr int kFaultResultMpiTag = 9881;  // FaultInjectResult from FaultInjCastQpErrorIsFatal

// Carries rank 1's observable state after the fault injection attempt to rank 0
// so all assertions run on rank 0 (the only rank with GTest listeners).
struct FaultInjectResult {
    int  sendRet;       // ncclResult_t from PostSend (cast to int)
    int  fatalCount;    // ncclIbCastFaultGetFatalCount result
    int  clearRet;      // ncclResult_t from ncclIbCastFaultClear (cast to int)
    int  setErrRet;     // first non-Success from ncclIbCastFaultSetQpError, or ncclSuccess
    int  actualNqps;    // number of QPs armed with the error fault
};

// =============================================================================
// Test: FaultInjCastQpErrorIsFatal
//
// CAST path. Arm error injection on QP 0. The fault hook fires inside
// IbCastMultiSend before wrap_ibv_post_send and returns ncclSystemError;
// NCCLCHECK propagates it to IbCastIsend → net_->isend caller.
//
// Verifies:
//   - isend returns an error OR fatalErrorCount > 0
// =============================================================================
TEST_F(NetIbMPITest, FaultInjCastQpErrorIsFatal) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    CAST_ENV_CHECK_OR_SKIP();

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 1024;
    std::vector<char> sendBuf(kMsgSize), recvBuf(kMsgSize);
    for (size_t i = 0; i < kMsgSize; i++) sendBuf[i] = static_cast<char>(i & 0xFF);
    memset(recvBuf.data(), 0, kMsgSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup: initialise WRR scheduler state before arming the fault.
    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kMsgSize, /*tag=*/300, mhandle);
    ASSERT_GT(actualNqps, 0);

    // rank 1 arms the fault on every active QP so it fires regardless of which
    // one the WRR scheduler selects for the next send.
    FaultInjectResult r1 = {};
    r1.actualNqps = actualNqps;
    if (rank == 1) {
        r1.setErrRet = static_cast<int>(ncclSuccess);
        for (int q = 0; q < actualNqps; ++q) {
            ncclResult_t ret = ncclIbCastFaultSetQpError(sendComm, q, /*inject=*/true);
            if (ret != ncclSuccess && r1.setErrRet == static_cast<int>(ncclSuccess))
                r1.setErrRet = static_cast<int>(ret);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Track whether the recv request completed so we can drain before CloseRecvComm.
    bool recvDone = false;
    void* recvReq = nullptr;

    if (rank == 0) {
        void*  bufs[1]    = {buf};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {301};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq), ncclSuccess);
        for (int poll = 0; poll < 100; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq, &done, &sz) != ncclSuccess) break;
            if (done) { recvDone = true; break; }
            usleep(kPollIntervalUs);
        }
    } else {
        // IbCastIsend sets *request before IbCastMultiSend is called, so sendReq
        // may be non-null even when sendRet != ncclSuccess. Break on either condition.
        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm, buf, kMsgSize, 301, mhandle, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }

        int fatalCount = 0;
        ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);

        if (sendRet == ncclSuccess && sendReq != nullptr) {
            for (int poll = 0; poll < 200; poll++) {
                int done = 0, sz = 0;
                TestRequest(sendReq, &done, &sz);
                ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);
                if (done || fatalCount > 0) break;
                usleep(kPollIntervalUs);
            }
        }

        r1.sendRet   = static_cast<int>(sendRet);
        r1.fatalCount = fatalCount;
        r1.clearRet  = static_cast<int>(ncclIbCastFaultClear(sendComm));

        // Ship results to rank 0 for assertions (rank 1 has no GTest listeners).
        MPI_Send(&r1, sizeof(r1), MPI_BYTE, 0, kFaultResultMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Recv(&r1, sizeof(r1), MPI_BYTE, 1, kFaultResultMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        EXPECT_EQ(r1.setErrRet, static_cast<int>(ncclSuccess))
            << "rank 1: ncclIbCastFaultSetQpError failed with " << r1.setErrRet
            << " (armed " << r1.actualNqps << " QPs)";

        bool isendFailed = (r1.sendRet != static_cast<int>(ncclSuccess));
        EXPECT_TRUE(isendFailed || r1.fatalCount > 0)
            << "rank 1: expected isend to fail OR fatalErrorCount > 0 after arming all "
            << r1.actualNqps << " CAST QPs with error injection; "
            << "isend returned " << r1.sendRet << ", fatalCount=" << r1.fatalCount;

        EXPECT_EQ(r1.clearRet, static_cast<int>(ncclSuccess))
            << "rank 1: ncclIbCastFaultClear failed with " << r1.clearRet;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !recvDone)
        DrainRecvRequest(recvReq);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: FaultInjCastSlowQpRebalances
//
// CAST path with WRR enabled. Arm a 10 ms delay on QP 0 and run 500 sends.
// The RTT timer fires every 50 ms (several times during the run) and updates
// initQpTokens based on measured per-QP latency.
//
// The test follows the pattern of CastStressMultiRoundTwoConns:
//   - rank 1 reads ncclIbCastSchedState and sends it to rank 0 via MPI
//   - rank 0 performs all assertions (scheduler state + data integrity)
//
// Verifies:
//   - initQpTokens[0] differs from the initial equal share (RTT timer fired)
//   - Data arrives intact on the receiver
//   - Requires >= 2 actual QPs; skipped otherwise.
//
// Message size is kept below GetSplitDataMin() so messages always take the
// WRR token path (same rationale as CastStressMultiRoundTwoConns).
// =============================================================================
TEST_F(NetIbMPITest, FaultInjCastSlowQpRebalances) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const int rank = MPIEnvironment::world_rank;

    CAST_ENV_CHECK_OR_SKIP();
    if (GetSplitDataMin() == 0)
        GTEST_SKIP() << "RCCL_IB_QP_SCHED_SPLIT_DATA_MIN=0: kMsgSz = GetSplitDataMin()-1 would underflow";
    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

    constexpr int kNMsgs = 500;
    // Keep below splitDataMin so messages take the WRR token path.
    const size_t kMsgSz = std::max<size_t>(64, GetSplitDataMin() - 1);
    const size_t kBufSz = static_cast<size_t>(kNMsgs) * kMsgSz;
    constexpr int kBaseTag = 4000;

    std::vector<char> sendBuf(kBufSz), recvBuf(kBufSz);
    for (size_t i = 0; i < kBufSz; i++) sendBuf[i] = static_cast<char>((i * 3 + 7) & 0xFF);
    memset(recvBuf.data(), 0, kBufSz);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    char* regBuf  = (rank == 0) ? recvBuf.data() : sendBuf.data();
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, regBuf, kBufSz, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup: initialise WRR state and learn actual nqps.
    const int actualNqps = GetActualNqps(sendComm, recvComm, regBuf, kMsgSz, kBaseTag - 1, mhandle);
    ASSERT_GT(actualNqps, 0);

    // Broadcast skip decision from rank 1 (the only rank that owns sendComm
    // and can read the real QP count after the warmup send).  root=1 is
    // intentional: rank 1 sets doSkip, rank 0 receives it.
    int doSkip = (actualNqps < 2) ? 1 : 0;
    MPI_Bcast(&doSkip, 1, MPI_INT, /*root=*/1, MPI_COMM_WORLD);
    if (doSkip) {
        GTEST_SKIP() << "Need >= 2 actual QPs for rebalancing test (got " << actualNqps << ")";
    }

    if (rank == 1) {
        // Arm equal-weight tokens as baseline for the RTT comparison.
        const std::vector<int> tokens = EqualTokens(actualNqps);
        ASSERT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
        // Arm 10 ms delay on QP 0 — makes QP 0 observably slower to the RTT timer.
        ASSERT_EQ(ncclIbCastFaultSetQpDelay(sendComm, /*qpIdx=*/0, /*delayUs=*/10000), ncclSuccess);
    }

    // 500 sends in batches of 16 (mirrors CastStressMultiRoundTwoConns).
    constexpr int kBatch = 16;
    if (rank == 0) {
        for (int base = 0; base < kNMsgs; base += kBatch) {
            const int end = std::min(base + kBatch, kNMsgs);
            std::vector<void*> reqs(end - base, nullptr);
            for (int i = base; i < end; i++) {
                void*  bufs[1]    = {recvBuf.data() + i * kMsgSz};
                size_t sizes[1]   = {kMsgSz};
                int    tags[1]    = {kBaseTag + i};
                void*  handles[1] = {mhandle};
                ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &reqs[i - base]),
                          ncclSuccess);
                ASSERT_NE(reqs[i - base], nullptr);
            }
            for (int i = 0; i < end - base; i++) {
                int sz = 0;
                ASSERT_EQ(WaitForCompletion(reqs[i], &sz, 10000), ncclSuccess);
            }
        }
    } else {
        for (int base = 0; base < kNMsgs; base += kBatch) {
            const int end = std::min(base + kBatch, kNMsgs);
            std::vector<void*> reqs(end - base, nullptr);
            for (int i = base; i < end; i++)
                PostSendWithRetry(sendComm, sendBuf.data() + i * kMsgSz, kMsgSz,
                                  kBaseTag + i, mhandle, &reqs[i - base]);
            for (int i = 0; i < end - base; i++) {
                int sz = 0;
                ASSERT_EQ(WaitForCompletion(reqs[i], &sz, 10000), ncclSuccess);
            }
        }
    }

    // rank 1 reads scheduler state and ships it to rank 0 for assertions.
    // (State lives in sendComm which only rank 1 owns.)
    struct ncclIbCastSchedState st = {};
    if (rank == 1) {
        ASSERT_EQ(ncclIbCastGetSchedState(sendComm, &st), ncclSuccess);
        MPI_Send(&st, sizeof(st), MPI_BYTE, 0, kSchedStateMpiTag, MPI_COMM_WORLD);
        ASSERT_EQ(ncclIbCastFaultClear(sendComm), ncclSuccess);
    } else {
        MPI_Recv(&st, sizeof(st), MPI_BYTE, 1, kSchedStateMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        // Data integrity: recvBuf must match what rank 1 sent.
        EXPECT_EQ(memcmp(recvBuf.data(), sendBuf.data(), kBufSz), 0)
            << "data corruption with 10 ms QP 0 delay";

        // RTT timer check: initQpTokens[0] must have changed from the initial
        // equal share.  Use EqualTokens() — not integer division — to get the
        // exact baseline value that was set by rank 1 before the sends, so the
        // comparison is consistent even when 100 % actualNqps != 0.
        const std::vector<int> baseTokens = EqualTokens(actualNqps);
        EXPECT_LT(st.initQpTokens[0], baseTokens[0])
            << "RTT timer did not reduce QP 0 tokens below equal share (initQpTokens[0]=" << st.initQpTokens[0]
            << ", equalShare=" << baseTokens[0] << "); "
            << "was 10 ms delay insufficient for the RTT timer interval?";

        // WRR invariants: token accounting must be consistent.
        EXPECT_TRUE(st.schedInit) << "schedInit must be true after 500 sends";
        {
            int s = 0;
            for (int q = 0; q < st.nqps; q++) s += st.activeQpTokens[q];
            EXPECT_EQ(s, st.activeTotTokens) << "activeQpTokens sum mismatch";
        }
        EXPECT_GE(st.activeTotTokens, 0);
        EXPECT_LE(st.activeTotTokens, st.initTotTokens);
    }

    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: FaultInjCastDelayDataIntegrity
//
// CAST path. Inject 2 ms delay on QP 0, run 50 sends.
// Verifies: data integrity is preserved despite the artificial per-QP delay.
//
// Unlike FaultInjCastSlowQpRebalances, this test does not aim to trigger
// the RTT timer; it only confirms that usleep in the send path does not
// corrupt or drop data.
// =============================================================================
TEST_F(NetIbMPITest, FaultInjCastDelayDataIntegrity) {
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

    constexpr int    kNMsgs  = 50;
    constexpr size_t kMsgSz  = 8192;
    const size_t     kBufSz  = static_cast<size_t>(kNMsgs) * kMsgSz;
    constexpr int    kBaseTag = 5000;

    std::vector<char> sendBuf(kBufSz), recvBuf(kBufSz);
    for (size_t i = 0; i < kBufSz; i++) sendBuf[i] = static_cast<char>((i * 7 + 3) & 0xFF);
    memset(recvBuf.data(), 0, kBufSz);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    char* regBuf  = (rank == 0) ? recvBuf.data() : sendBuf.data();
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, regBuf, kBufSz, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    const int actualNqps = GetActualNqps(sendComm, recvComm, regBuf, kMsgSz, kBaseTag - 1, mhandle);
    ASSERT_GT(actualNqps, 0);

    if (rank == 1) {
        ASSERT_EQ(ncclIbCastFaultSetQpDelay(sendComm, /*qpIdx=*/0, /*delayUs=*/2000), ncclSuccess);
    }

    CastDoBatchSendRecv(rank, sendComm, recvComm, sendBuf.data(), recvBuf.data(),
                        kMsgSz, kNMsgs, kBaseTag, mhandle);

    if (rank == 1) {
        ASSERT_EQ(ncclIbCastFaultClear(sendComm), ncclSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        // recvBuf must match the pattern rank 1 sent.
        // Both ranks initialise sendBuf with the same deterministic pattern,
        // so rank 0's sendBuf is a valid reference without extra MPI traffic.
        EXPECT_EQ(memcmp(recvBuf.data(), sendBuf.data(), kBufSz), 0)
            << "data corruption with 2 ms QP 0 delay (50 sends)";
    }

    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: FaultInjCastSingleQpErrorIsFatal
//
// CAST path with WRR enabled. Arm error injection on exactly one QP —
// QP 0 — using ncclIbCastSetTokens to steer all WRR tokens there before the
// fault send. This models the realistic "one link failed" scenario where only
// a specific physical path is broken.
//
// Verifies:
//   - isend returns an error OR fatalErrorCount > 0
//   - Skipped when actualNqps < 2 (single-QP: no WRR, cursor is always 0
//     and FaultInjCastQpErrorIsFatal already covers that case)
// =============================================================================
TEST_F(NetIbMPITest, FaultInjCastSingleQpErrorIsFatal) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    CAST_ENV_CHECK_OR_SKIP();

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 1024;  // below splitDataMin → single-QP WRR path
    std::vector<char> sendBuf(kMsgSize), recvBuf(kMsgSize);
    for (size_t i = 0; i < kMsgSize; i++) sendBuf[i] = static_cast<char>(i & 0xFF);
    memset(recvBuf.data(), 0, kMsgSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup: initialise WRR state and determine actual QP count.
    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kMsgSize, /*tag=*/400, mhandle);
    ASSERT_GT(actualNqps, 0);

    // Broadcast skip decision from rank 1 (owns sendComm and the WRR cursor).
    int doSkip = (actualNqps < 2) ? 1 : 0;
    MPI_Bcast(&doSkip, 1, MPI_INT, /*root=*/1, MPI_COMM_WORLD);
    if (doSkip) {
        GTEST_SKIP() << "Need >= 2 actual QPs for single-QP fault test (got " << actualNqps
                     << "); FaultInjCastQpErrorIsFatal covers the single-QP case";
    }

    // rank 1: steer WRR onto QP 0 by setting all tokens there, then arm the
    // fault on QP 0 only. Using SetTokens makes the chosen QP deterministic
    // regardless of where the WRR cursor landed after warmup.
    struct FaultInjectResult r1 = {};
    r1.actualNqps = actualNqps;
    const int targetQp = 0;
    if (rank == 1) {
        // Concentrate all tokens on QP 0: tokens[0]=1, tokens[1..n-1]=0.
        std::vector<int> tokens(actualNqps, 0);
        tokens[0] = 1;
        EXPECT_EQ(ncclIbCastSetTokens(sendComm, tokens.data(), actualNqps), ncclSuccess);
        r1.setErrRet = static_cast<int>(ncclIbCastFaultSetQpError(sendComm, targetQp, /*inject=*/true));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    bool recvDone = false;
    void* recvReq = nullptr;

    if (rank == 0) {
        void*  bufs[1]    = {buf};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {401};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq), ncclSuccess);
        for (int poll = 0; poll < 100; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq, &done, &sz) != ncclSuccess) break;
            if (done) { recvDone = true; break; }
            usleep(kPollIntervalUs);
        }
    } else {
        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm, buf, kMsgSize, 401, mhandle, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }

        int fatalCount = 0;
        ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);

        if (sendRet == ncclSuccess && sendReq != nullptr) {
            for (int poll = 0; poll < 200; poll++) {
                int done = 0, sz = 0;
                TestRequest(sendReq, &done, &sz);
                ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);
                if (done || fatalCount > 0) break;
                usleep(kPollIntervalUs);
            }
        }

        r1.sendRet    = static_cast<int>(sendRet);
        r1.fatalCount = fatalCount;
        r1.clearRet   = static_cast<int>(ncclIbCastFaultClear(sendComm));

        MPI_Send(&r1, sizeof(r1), MPI_BYTE, 0, kFaultResultMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Recv(&r1, sizeof(r1), MPI_BYTE, 1, kFaultResultMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        EXPECT_EQ(r1.setErrRet, static_cast<int>(ncclSuccess))
            << "rank 1: ncclIbCastFaultSetQpError failed for QP " << targetQp;

        bool isendFailed = (r1.sendRet != static_cast<int>(ncclSuccess));
        EXPECT_TRUE(isendFailed || r1.fatalCount > 0)
            << "rank 1: expected isend to fail OR fatalErrorCount > 0 after single-QP "
            << targetQp << " error injection (of " << r1.actualNqps << " total); "
            << "isend returned " << r1.sendRet << ", fatalCount=" << r1.fatalCount;

        EXPECT_EQ(r1.clearRet, static_cast<int>(ncclSuccess))
            << "rank 1: ncclIbCastFaultClear failed";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !recvDone)
        DrainRecvRequest(recvReq);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: FaultInjCastQpErrorClearRecovers
//
// CAST path. Arm error injection on all QPs, trigger one failed send, then
// call ncclIbCastFaultClear and open a new connection. Verifies that after
// clearing fault state a fresh connection sends and receives N messages
// without errors and with full data integrity.
//
// A new connection is used for the recovery phase because a faulted
// connection increments fatalErrorCount and future IbCastIsend calls check
// that counter before proceeding (NCCL_IB_RETURN_ASYNC_EVENTS=1). The test
// therefore validates that fault injection does not leave persistent state
// that contaminates subsequent connections.
//
// Verifies:
//   - After fault injection and FaultClear, a new CAST connection transfers
//     data correctly
//   - fatalErrorCount on the new connection remains 0
// =============================================================================
TEST_F(NetIbMPITest, FaultInjCastQpErrorClearRecovers) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    CAST_ENV_CHECK_OR_SKIP();

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    // ── Phase 1: inject fault on first connection ─────────────────────────
    void* listenComm1 = nullptr;
    void* sendComm1   = nullptr;
    void* recvComm1   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm1, &sendComm1, &recvComm1);

    constexpr size_t kMsgSize = 1024;
    std::vector<char> sendBuf(kMsgSize), recvBuf(kMsgSize);
    for (size_t i = 0; i < kMsgSize; i++) sendBuf[i] = static_cast<char>(i & 0xFF);
    memset(recvBuf.data(), 0, kMsgSize);

    void* comm1   = (rank == 0) ? recvComm1 : sendComm1;
    void* buf1    = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle1 = nullptr;
    ASSERT_EQ(RegisterMemory(comm1, buf1, kMsgSize, NCCL_PTR_HOST, &mhandle1), ncclSuccess);

    const int actualNqps = GetActualNqps(sendComm1, recvComm1, buf1, kMsgSize, /*tag=*/500, mhandle1);
    ASSERT_GT(actualNqps, 0);

    if (rank == 1) {
        for (int q = 0; q < actualNqps; ++q)
            ASSERT_EQ(ncclIbCastFaultSetQpError(sendComm1, q, /*inject=*/true), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Trigger the fault (rank 0 posts recv, rank 1 posts send that will fail).
    // rank 1 forwards its fault result to rank 0 so we can assert the fault
    // was actually observed before testing recovery in phase 2.
    static constexpr int kPhase1MpiTag = 9882;
    FaultInjectResult p1 = {};
    void* recvReq1 = nullptr;
    bool  recvDone1 = false;
    if (rank == 0) {
        void*  bufs[1]    = {buf1};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {501};
        void*  handles[1] = {mhandle1};
        ASSERT_EQ(PostRecv(recvComm1, 1, bufs, sizes, tags, handles, &recvReq1), ncclSuccess);
        for (int poll = 0; poll < 100; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq1, &done, &sz) != ncclSuccess) break;
            if (done) { recvDone1 = true; break; }
            usleep(kPollIntervalUs);
        }
    } else {
        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm1, buf1, kMsgSize, 501, mhandle1, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }
        int fatalCount = 0;
        if (sendRet == ncclSuccess && sendReq != nullptr) {
            for (int poll = 0; poll < 200; poll++) {
                int done = 0, sz = 0;
                TestRequest(sendReq, &done, &sz);
                ncclIbCastFaultGetFatalCount(sendComm1, &fatalCount);
                if (done || fatalCount > 0) break;
                usleep(kPollIntervalUs);
            }
        } else {
            ncclIbCastFaultGetFatalCount(sendComm1, &fatalCount);
        }
        p1.sendRet    = static_cast<int>(sendRet);
        p1.fatalCount = fatalCount;
        EXPECT_EQ(ncclIbCastFaultClear(sendComm1), ncclSuccess);
        MPI_Send(&p1, sizeof(p1), MPI_BYTE, 0, kPhase1MpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Recv(&p1, sizeof(p1), MPI_BYTE, 1, kPhase1MpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        bool isendFailed = (p1.sendRet != static_cast<int>(ncclSuccess));
        EXPECT_TRUE(isendFailed || p1.fatalCount > 0)
            << "Phase 1: fault injection did not trigger — isend returned "
            << p1.sendRet << ", fatalCount=" << p1.fatalCount
            << "; recovery test is meaningless without a confirmed fault";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !recvDone1)
        DrainRecvRequest(recvReq1);
    TeardownConnection(recvComm1, listenComm1, sendComm1, mhandle1);

    // ── Phase 2: open fresh connection, verify it works cleanly ──────────
    constexpr int    kNMsgs   = 20;
    constexpr size_t kMsgSz2  = 4096;
    const size_t     kBufSz2  = static_cast<size_t>(kNMsgs) * kMsgSz2;
    constexpr int    kBaseTag = 510;

    std::vector<char> sendBuf2(kBufSz2), recvBuf2(kBufSz2);
    for (size_t i = 0; i < kBufSz2; i++) sendBuf2[i] = static_cast<char>((i * 5 + 11) & 0xFF);
    memset(recvBuf2.data(), 0, kBufSz2);

    void* listenComm2 = nullptr;
    void* sendComm2   = nullptr;
    void* recvComm2   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm2, &sendComm2, &recvComm2);

    void* comm2   = (rank == 0) ? recvComm2 : sendComm2;
    char* regBuf2 = (rank == 0) ? recvBuf2.data() : sendBuf2.data();
    void* mhandle2 = nullptr;
    ASSERT_EQ(RegisterMemory(comm2, regBuf2, kBufSz2, NCCL_PTR_HOST, &mhandle2), ncclSuccess);

    CastDoBatchSendRecv(rank, sendComm2, recvComm2,
                        sendBuf2.data(), recvBuf2.data(),
                        kMsgSz2, kNMsgs, kBaseTag, mhandle2);

    // Verify fatalCount on the new connection is still 0 (forwarded rank 1 → 0).
    int newFatalCount = 0;
    if (rank == 1) {
        ncclIbCastFaultGetFatalCount(sendComm2, &newFatalCount);
    }
    MPI_Bcast(&newFatalCount, 1, MPI_INT, /*root=*/1, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        EXPECT_EQ(memcmp(recvBuf2.data(), sendBuf2.data(), kBufSz2), 0)
            << "data corruption in post-fault recovery connection";
        EXPECT_EQ(newFatalCount, 0)
            << "new connection has non-zero fatalErrorCount after FaultClear on previous connection";
    }

    TeardownConnection(recvComm2, listenComm2, sendComm2, mhandle2);
}

// =============================================================================
// Test: FailoverErrorCodeWhitelist
//
// Validates IbCastResiliencyCheckErrorNotFatal via the ncclIbCastFaultCheckErrorFatal
// wrapper. WR_FLUSH_ERR and RETRY_EXC_ERR must be non-fatal (eligible for
// failover); other error codes must be fatal.
//
// Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1 so the resiliency context exists.
// Does NOT require ndevs >= 2 — only tests the classification function.
// =============================================================================
TEST_F(NetIbMPITest, FailoverErrorCodeWhitelist) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const char* failoverEnv = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    }

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

    struct WhitelistResult {
        int hasResiliency;
        int flushIsFatal;
        int retryIsFatal;
        int remAccessIsFatal;
        int generalErrIsFatal;
    };

    static constexpr int kWhitelistMpiTag = 9885;
    WhitelistResult r = {};

    if (rank == 1) {
        // Check resiliency context exists
        struct ncclIbCastResiliencyState resState = {};
        r.hasResiliency = (ncclIbCastGetResiliencyState(sendComm, &resState) == ncclSuccess) ? 1 : 0;

        if (r.hasResiliency) {
            bool isFatal = false;

            ncclIbCastFaultCheckErrorFatal(sendComm, kWcWrFlushErr, &isFatal);
            r.flushIsFatal = isFatal ? 1 : 0;

            ncclIbCastFaultCheckErrorFatal(sendComm, kWcRetryExcErr, &isFatal);
            r.retryIsFatal = isFatal ? 1 : 0;

            ncclIbCastFaultCheckErrorFatal(sendComm, kWcRemAccessErr, &isFatal);
            r.remAccessIsFatal = isFatal ? 1 : 0;

            ncclIbCastFaultCheckErrorFatal(sendComm, kWcGeneralErr, &isFatal);
            r.generalErrIsFatal = isFatal ? 1 : 0;
        }

        MPI_Send(&r, sizeof(r), MPI_BYTE, 0, kWhitelistMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Recv(&r, sizeof(r), MPI_BYTE, 1, kWhitelistMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        ASSERT_EQ(r.hasResiliency, 1)
            << "Resiliency context not created — is NCCL_IB_RESILIENCY_PORT_FAILOVER=1?";

        EXPECT_EQ(r.flushIsFatal, 0)
            << "WR_FLUSH_ERR should be non-fatal (eligible for failover)";
        EXPECT_EQ(r.retryIsFatal, 0)
            << "RETRY_EXC_ERR should be non-fatal (eligible for failover)";
        EXPECT_EQ(r.remAccessIsFatal, 1)
            << "REM_ACCESS_ERR should be fatal (not in whitelist)";
        EXPECT_EQ(r.generalErrIsFatal, 1)
            << "GENERAL_ERR should be fatal (not in whitelist)";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, nullptr);
}

// =============================================================================
// Test: FailoverCqeErrorRecovered
//
// Core failover test. Requires NIC Fusion (ndevs >= 2) so there is a
// surviving device. Drives one QP to IBV_QPS_ERR via
// ncclIbCastFaultDriveQpToError, producing real WR_FLUSH_ERR CQEs.
// The resiliency state machine should: detect the error, replace the QP,
// probe the receiver, and complete the request via the surviving device.
//
// Requires:
//   - NCCL_IB_RESILIENCY_PORT_FAILOVER=1
//   - NCCL_IB_MERGE_NICS=1 + NCCL_NET_FORCE_MERGE (or ndevs >= 2 by topology)
//   - sentData fix (C2) — without it, probe result is ignored
// =============================================================================
TEST_F(NetIbMPITest, FailoverCqeErrorRecovered) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const char* failoverEnv = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    }

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    int totalDevs = 0;
    AssertInitAndGetDevices(&totalDevs);

    int mergedDev = CreateMergedDeviceForFailover(net_, totalDevs);
    if (mergedDev < 0) {
        GTEST_SKIP() << "Failover requires NIC Fusion (ndevs >= 2). "
                     << "Found " << totalDevs << " physical devices.";
    }

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/mergedDev, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 8192;
    std::vector<char> sendBuf(kMsgSize), recvBuf(kMsgSize);
    for (size_t i = 0; i < kMsgSize; i++) sendBuf[i] = static_cast<char>((i * 13 + 7) & 0xFF);
    memset(recvBuf.data(), 0, kMsgSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kMsgSize, /*tag=*/600, mhandle);
    ASSERT_GT(actualNqps, 0);

    struct ncclIbCastResiliencyState resState = {};

    struct FailoverResult {
        int sendRet;
        int fatalCount;
        int devState0;
        int inProgress;
        int repostCount;
    };
    static constexpr int kFailoverMpiTag = 9886;
    FailoverResult fr = {};

    bool recvDone = false;
    void* recvReq = nullptr;

    if (rank == 0) {
        void*  bufs[1]    = {buf};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {601};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Drive sender QP 0 to ERR. The receiver should handle this via
    // negative events accounting (BY_ID matching + resiliency).
    if (rank == 1) {
        ASSERT_EQ(ncclIbCastFaultDriveQpToError(sendComm, 0), ncclSuccess);

        // Post the send — WRs on QP 0 will flush, surviving QPs handle the data
        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm, buf, kMsgSize, 601, mhandle, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }

        if (sendRet == ncclSuccess && sendReq != nullptr) {
            // Poll IbCastTest until request completes or timeout
            for (int poll = 0; poll < 500; poll++) {
                int done = 0, sz = 0;
                ncclResult_t testRet = TestRequest(sendReq, &done, &sz);
                if (testRet != ncclSuccess) { sendRet = testRet; break; }
                if (done) break;
                usleep(kPollIntervalUs);
            }
        }

        int fatalCount = 0;
        ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);

        ncclIbCastGetResiliencyState(sendComm, &resState);
        int repostCount = 0;
        ncclIbCastGetRepostCount(sendComm, &repostCount);

        fr.sendRet     = static_cast<int>(sendRet);
        fr.fatalCount  = fatalCount;
        fr.devState0   = resState.devState[0];
        fr.inProgress  = resState.inProgress ? 1 : 0;
        fr.repostCount = repostCount;

        MPI_Send(&fr, sizeof(fr), MPI_BYTE, 0, kFailoverMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Receiver MUST poll IbCastTest concurrently with sender's failover.
        // The sender's probe RDMA-reads receiver's completions[] which is only
        // set when the receiver calls IbCastTest → IbCastCompletionEventProcess.
        // Poll long enough for sender's failover (~10-15 seconds).
        for (int poll = 0; poll < 1500; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq, &done, &sz) != ncclSuccess) break;
            if (done) { recvDone = true; break; }
            usleep(kPollIntervalUs);
        }

        MPI_Recv(&fr, sizeof(fr), MPI_BYTE, 1, kFailoverMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        EXPECT_EQ(fr.sendRet, static_cast<int>(ncclSuccess))
            << "Send should complete via surviving device after failover";
        EXPECT_EQ(fr.fatalCount, 0)
            << "No fatal error expected — failover should handle the QP error";
        // With PORT_RECOVERY enabled, inProgress stays true (recovery ongoing).
        // Without recovery, failover completes and inProgress becomes false.

        // devState[0] should not be Ok — Error(1) without recovery,
        // or RecoveryInProgress(2) if recovery thread picked it up.
        EXPECT_NE(fr.devState0, 0)
            << "Device 0 should not be Ok after failover (got " << fr.devState0 << ")";

        if (fr.sendRet == static_cast<int>(ncclSuccess)) {
            EXPECT_TRUE(recvDone)
                << "Send succeeded but receiver never got the data — data loss";
        }
        if (recvDone) {
            EXPECT_EQ(memcmp(recvBuf.data(), sendBuf.data(), kMsgSize), 0)
                << "Data corruption after failover — receiver got wrong data";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !recvDone)
        DrainRecvRequest(recvReq);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: FailoverSingleDeviceTopology
//
// Single NIC with PORT_FAILOVER=1. Drive QP to ERR. With only one device,
// there is no surviving device — failover should degrade to fatal error.
// Verifies graceful degradation (no crash, no hang).
// =============================================================================
TEST_F(NetIbMPITest, FailoverSingleDeviceTopology) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const char* failoverEnv = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    }

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    // Use device 0 (single NIC, ndevs=1) — no NIC Fusion
    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 4096;
    std::vector<char> sendBuf(kMsgSize), recvBuf(kMsgSize);
    for (size_t i = 0; i < kMsgSize; i++) sendBuf[i] = static_cast<char>((i * 11 + 3) & 0xFF);
    memset(recvBuf.data(), 0, kMsgSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup
    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kMsgSize, /*tag=*/700, mhandle);
    ASSERT_GT(actualNqps, 0);

    struct SingleDevResult {
        int sendRet;
        int fatalCount;
    };
    static constexpr int kSingleDevMpiTag = 9887;
    SingleDevResult sr = {};

    bool recvDone = false;
    void* recvReq = nullptr;

    if (rank == 0) {
        void*  bufs[1]    = {buf};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {701};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        // Drive QP 0 to ERR before sending — with single device,
        // all QPs are on device 0, so failover has no surviving device
        ASSERT_EQ(ncclIbCastFaultDriveQpToError(sendComm, 0), ncclSuccess);

        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm, buf, kMsgSize, 701, mhandle, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }

        if (sendRet == ncclSuccess && sendReq != nullptr) {
            for (int poll = 0; poll < 200; poll++) {
                int done = 0, sz = 0;
                ncclResult_t testRet = TestRequest(sendReq, &done, &sz);
                if (testRet != ncclSuccess) { sendRet = testRet; break; }
                if (done) break;
                usleep(kPollIntervalUs);
            }
        }

        int fatalCount = 0;
        ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);

        sr.sendRet    = static_cast<int>(sendRet);
        sr.fatalCount = fatalCount;

        MPI_Send(&sr, sizeof(sr), MPI_BYTE, 0, kSingleDevMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int poll = 0; poll < 200; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq, &done, &sz) != ncclSuccess) break;
            if (done) { recvDone = true; break; }
            usleep(kPollIntervalUs);
        }

        MPI_Recv(&sr, sizeof(sr), MPI_BYTE, 1, kSingleDevMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        // With single device, failover should detect no surviving device
        // and return a fatal error (ncclRemoteError)
        bool isendFailed = (sr.sendRet != static_cast<int>(ncclSuccess));
        EXPECT_TRUE(isendFailed || sr.fatalCount > 0)
            << "Single-device failover should produce a fatal error or failed send; "
            << "sendRet=" << sr.sendRet << ", fatalCount=" << sr.fatalCount;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !recvDone)
        DrainRecvRequest(recvReq);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: FailoverAllDevicesFailed
//
// NIC Fusion (ndevs >= 2) with PORT_FAILOVER=1. Drive ALL QPs on both
// devices to ERR simultaneously. With no surviving device, failover
// should detect the total failure and return a fatal error, no hang.
// =============================================================================
TEST_F(NetIbMPITest, FailoverAllDevicesFailed) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const char* failoverEnv = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    }

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    int totalDevs = 0;
    AssertInitAndGetDevices(&totalDevs);

    int mergedDev = CreateMergedDeviceForFailover(net_, totalDevs);
    if (mergedDev < 0) {
        GTEST_SKIP() << "Requires NIC Fusion (ndevs >= 2). Need at least 2 IB devices.";
    }

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/mergedDev, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 4096;
    std::vector<char> sendBuf(kMsgSize), recvBuf(kMsgSize);
    for (size_t i = 0; i < kMsgSize; i++) sendBuf[i] = static_cast<char>((i * 17 + 5) & 0xFF);
    memset(recvBuf.data(), 0, kMsgSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kMsgSize, /*tag=*/800, mhandle);
    ASSERT_GT(actualNqps, 0);

    struct MaxAttemptResult {
        int sendRet;
        int fatalCount;
    };
    static constexpr int kMaxAttemptMpiTag = 9888;
    MaxAttemptResult mr = {};

    bool recvDone = false;
    void* recvReq = nullptr;

    if (rank == 0) {
        void*  bufs[1]    = {buf};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {801};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        // Drive ALL QPs to ERR — no surviving device
        for (int q = 0; q < actualNqps; q++) {
            ASSERT_EQ(ncclIbCastFaultDriveQpToError(sendComm, q), ncclSuccess);
        }

        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm, buf, kMsgSize, 801, mhandle, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }

        if (sendRet == ncclSuccess && sendReq != nullptr) {
            for (int poll = 0; poll < 300; poll++) {
                int done = 0, sz = 0;
                ncclResult_t testRet = TestRequest(sendReq, &done, &sz);
                if (testRet != ncclSuccess) { sendRet = testRet; break; }
                if (done) break;
                usleep(kPollIntervalUs);
            }
        }

        int fatalCount = 0;
        ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);

        mr.sendRet    = static_cast<int>(sendRet);
        mr.fatalCount = fatalCount;

        MPI_Send(&mr, sizeof(mr), MPI_BYTE, 0, kMaxAttemptMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int poll = 0; poll < 300; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq, &done, &sz) != ncclSuccess) break;
            if (done) { recvDone = true; break; }
            usleep(kPollIntervalUs);
        }

        MPI_Recv(&mr, sizeof(mr), MPI_BYTE, 1, kMaxAttemptMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        bool isendFailed = (mr.sendRet != static_cast<int>(ncclSuccess));
        EXPECT_TRUE(isendFailed || mr.fatalCount > 0)
            << "All QPs failed — expected fatal error or failed send; "
            << "sendRet=" << mr.sendRet << ", fatalCount=" << mr.fatalCount;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !recvDone)
        DrainRecvRequest(recvReq);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: FailoverLargeMessageDataIntegrity
//
// NIC Fusion (ndevs >= 2). Drive one QP to ERR before sending a large
// message (64 KB). Verifies failover handles multi-chunk data correctly:
// the surviving QPs deliver the data, and the receiver gets the complete
// message with no corruption.
//
// Note: repostCount may be 0 because fault is injected before the send,
// so QP 0 never posts data. Selective retransmit (repostCount > 0) only
// triggers when data is partially sent before the fault — requires
// mid-flight injection which is non-deterministic on fast HW.
// =============================================================================
TEST_F(NetIbMPITest, FailoverLargeMessageDataIntegrity) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const char* failoverEnv = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    }

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    int totalDevs = 0;
    AssertInitAndGetDevices(&totalDevs);

    int mergedDev = CreateMergedDeviceForFailover(net_, totalDevs);
    if (mergedDev < 0) {
        GTEST_SKIP() << "Requires NIC Fusion (ndevs >= 2). Need at least 2 IB devices.";
    }

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/mergedDev, &listenComm, &sendComm, &recvComm);

    // Large message to ensure data spans multiple QPs
    constexpr size_t kMsgSize = 65536;
    std::vector<char> sendBuf(kMsgSize), recvBuf(kMsgSize);
    for (size_t i = 0; i < kMsgSize; i++) sendBuf[i] = static_cast<char>((i * 31 + 17) & 0xFF);
    memset(recvBuf.data(), 0, kMsgSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kMsgSize, /*tag=*/900, mhandle);
    ASSERT_GT(actualNqps, 0);

    struct RetransmitResult {
        int sendRet;
        int fatalCount;
        int repostCount;
        int devState0;
    };
    static constexpr int kRetransmitMpiTag = 9889;
    RetransmitResult rr = {};

    bool recvDone = false;
    void* recvReq = nullptr;

    if (rank == 0) {
        void*  bufs[1]    = {buf};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {901};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        ASSERT_EQ(ncclIbCastFaultDriveQpToError(sendComm, 0), ncclSuccess);

        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm, buf, kMsgSize, 901, mhandle, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }

        if (sendRet == ncclSuccess && sendReq != nullptr) {
            for (int poll = 0; poll < 500; poll++) {
                int done = 0, sz = 0;
                ncclResult_t testRet = TestRequest(sendReq, &done, &sz);
                if (testRet != ncclSuccess) { sendRet = testRet; break; }
                if (done) break;
                usleep(kPollIntervalUs);
            }
        }

        int fatalCount = 0;
        ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);

        struct ncclIbCastResiliencyState resState = {};
        ncclIbCastGetResiliencyState(sendComm, &resState);
        int repostCount = 0;
        ncclIbCastGetRepostCount(sendComm, &repostCount);

        rr.sendRet     = static_cast<int>(sendRet);
        rr.fatalCount  = fatalCount;
        rr.repostCount = repostCount;
        rr.devState0   = resState.devState[0];

        MPI_Send(&rr, sizeof(rr), MPI_BYTE, 0, kRetransmitMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Receiver must poll concurrently with sender's failover
        for (int poll = 0; poll < 1500; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq, &done, &sz) != ncclSuccess) break;
            if (done) { recvDone = true; break; }
            usleep(kPollIntervalUs);
        }

        MPI_Recv(&rr, sizeof(rr), MPI_BYTE, 1, kRetransmitMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        EXPECT_EQ(rr.sendRet, static_cast<int>(ncclSuccess))
            << "Send should complete via surviving device after failover";
        EXPECT_EQ(rr.fatalCount, 0)
            << "No fatal error expected";
        EXPECT_NE(rr.devState0, 0)
            << "Device 0 should not be Ok after failover (got " << rr.devState0 << ")";
        if (rr.sendRet == static_cast<int>(ncclSuccess)) {
            EXPECT_TRUE(recvDone)
                << "Send succeeded but receiver never got the data — data loss";
        }
        if (recvDone) {
            EXPECT_EQ(memcmp(recvBuf.data(), sendBuf.data(), kMsgSize), 0)
                << "Data corruption after failover on large message";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !recvDone)
        DrainRecvRequest(recvReq);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: FailoverDeviceOneFailure
//
// Same as FailoverCqeErrorRecovered but fails device 1 (QP index 1)
// instead of device 0 (QP index 0). Validates that failover works
// regardless of which device fails.
// =============================================================================
TEST_F(NetIbMPITest, FailoverDeviceOneFailure) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const char* failoverEnv = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    }

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    int totalDevs = 0;
    AssertInitAndGetDevices(&totalDevs);

    int mergedDev = CreateMergedDeviceForFailover(net_, totalDevs);
    if (mergedDev < 0) {
        GTEST_SKIP() << "Requires NIC Fusion (ndevs >= 2).";
    }

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/mergedDev, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 8192;
    std::vector<char> sendBuf(kMsgSize), recvBuf(kMsgSize);
    for (size_t i = 0; i < kMsgSize; i++) sendBuf[i] = static_cast<char>((i * 19 + 11) & 0xFF);
    memset(recvBuf.data(), 0, kMsgSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kMsgSize, /*tag=*/1000, mhandle);
    ASSERT_GT(actualNqps, 1);

    struct {
        int sendRet;
        int fatalCount;
        int devState1;
        int inProgress;
    } fr = {};
    static constexpr int kDev1MpiTag = 9890;
    bool recvDone = false;
    void* recvReq = nullptr;

    if (rank == 0) {
        void*  bufs[1]    = {buf};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {1001};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        // Fail QP index 1 = device 1 (QPs interleaved: 0=dev0, 1=dev1, 2=dev0...)
        ASSERT_EQ(ncclIbCastFaultDriveQpToError(sendComm, 1), ncclSuccess);

        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm, buf, kMsgSize, 1001, mhandle, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }

        if (sendRet == ncclSuccess && sendReq != nullptr) {
            for (int poll = 0; poll < 500; poll++) {
                int done = 0, sz = 0;
                ncclResult_t testRet = TestRequest(sendReq, &done, &sz);
                if (testRet != ncclSuccess) { sendRet = testRet; break; }
                if (done) break;
                usleep(kPollIntervalUs);
            }
        }

        int fatalCount = 0;
        ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);
        struct ncclIbCastResiliencyState resState = {};
        ncclIbCastGetResiliencyState(sendComm, &resState);

        fr.sendRet    = static_cast<int>(sendRet);
        fr.fatalCount = fatalCount;
        fr.devState1  = resState.devState[1];
        fr.inProgress = resState.inProgress ? 1 : 0;

        MPI_Send(&fr, sizeof(fr), MPI_BYTE, 0, kDev1MpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int poll = 0; poll < 1500; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq, &done, &sz) != ncclSuccess) break;
            if (done) { recvDone = true; break; }
            usleep(kPollIntervalUs);
        }

        MPI_Recv(&fr, sizeof(fr), MPI_BYTE, 1, kDev1MpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        EXPECT_EQ(fr.sendRet, static_cast<int>(ncclSuccess))
            << "Send should complete via device 0 after device 1 failover";
        EXPECT_EQ(fr.fatalCount, 0);
        EXPECT_NE(fr.devState1, 0) << "Device 1 should not be Ok after failover";
        if (fr.sendRet == static_cast<int>(ncclSuccess)) {
            EXPECT_TRUE(recvDone) << "Send succeeded but receiver lost data";
        }
        if (recvDone) {
            EXPECT_EQ(memcmp(recvBuf.data(), sendBuf.data(), kMsgSize), 0)
                << "Data corruption after device 1 failover";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !recvDone)
        DrainRecvRequest(recvReq);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: FailoverMultiRequestInFlight
//
// Post 4 send requests, then fault QP 0. All 4 requests should
// complete via the surviving device. Validates that the resiliency
// state machine handles multiple concurrent failed requests.
// =============================================================================
TEST_F(NetIbMPITest, FailoverMultiRequestInFlight) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const char* failoverEnv = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    }

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    int totalDevs = 0;
    AssertInitAndGetDevices(&totalDevs);

    int mergedDev = CreateMergedDeviceForFailover(net_, totalDevs);
    if (mergedDev < 0) {
        GTEST_SKIP() << "Requires NIC Fusion (ndevs >= 2).";
    }

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/mergedDev, &listenComm, &sendComm, &recvComm);

    constexpr int    kNumReqs = 4;
    constexpr size_t kMsgSize = 4096;
    const size_t     kBufSize = kMsgSize * (kNumReqs + 1);
    std::vector<char> sendBuf(kBufSize), recvBuf(kBufSize);
    for (size_t i = 0; i < kBufSize; i++) sendBuf[i] = static_cast<char>((i * 29 + 7) & 0xFF);
    memset(recvBuf.data(), 0, kBufSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    char* regBuf  = (rank == 0) ? recvBuf.data() : sendBuf.data();
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, regBuf, kBufSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Warmup
    const int actualNqps = GetActualNqps(sendComm, recvComm, regBuf, kMsgSize, /*tag=*/1200, mhandle);
    ASSERT_GT(actualNqps, 0);

    static constexpr int kMultiMpiTag = 9892;
    constexpr int kBaseTag = 1210;

    // Post all recvs first
    void* recvReqs[kNumReqs] = {};
    if (rank == 0) {
        for (int i = 0; i < kNumReqs; i++) {
            char* buf = regBuf + (i + 1) * kMsgSize;
            void*  bufs[1]    = {buf};
            size_t sizes[1]   = {kMsgSize};
            int    tags[1]    = {kBaseTag + i};
            void*  handles[1] = {mhandle};
            ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReqs[i]), ncclSuccess);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    struct MultiReqResult {
        int sendRet[kNumReqs];
        int fatalCount;
    };
    MultiReqResult mr = {};

    if (rank == 1) {
        // Fault QP 0 before posting any sends
        ASSERT_EQ(ncclIbCastFaultDriveQpToError(sendComm, 0), ncclSuccess);

        // Post all 4 sends
        void* sendReqs[kNumReqs] = {};
        for (int i = 0; i < kNumReqs; i++) {
            char* buf = regBuf + (i + 1) * kMsgSize;
            ncclResult_t sendRet = ncclSuccess;
            for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
                sendRet = PostSend(sendComm, buf, kMsgSize, kBaseTag + i, mhandle, &sendReqs[i]);
                if (sendRet != ncclSuccess || sendReqs[i] != nullptr) break;
                usleep(kPollIntervalUs);
            }
            mr.sendRet[i] = static_cast<int>(sendRet);
        }

        // Poll all requests to completion
        for (int i = 0; i < kNumReqs; i++) {
            if (sendReqs[i] == nullptr) continue;
            for (int poll = 0; poll < 500; poll++) {
                int done = 0, sz = 0;
                ncclResult_t testRet = TestRequest(sendReqs[i], &done, &sz);
                if (testRet != ncclSuccess) { mr.sendRet[i] = static_cast<int>(testRet); break; }
                if (done) break;
                usleep(kPollIntervalUs);
            }
        }

        ncclIbCastFaultGetFatalCount(sendComm, &mr.fatalCount);
        MPI_Send(&mr, sizeof(mr), MPI_BYTE, 0, kMultiMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Poll all recvs concurrently
        bool allDone = false;
        for (int poll = 0; poll < 1500 && !allDone; poll++) {
            allDone = true;
            for (int i = 0; i < kNumReqs; i++) {
                if (recvReqs[i] == nullptr) continue;
                int done = 0, sz = 0;
                TestRequest(recvReqs[i], &done, &sz);
                if (done) { recvReqs[i] = nullptr; }
                else { allDone = false; }
            }
            if (!allDone) usleep(kPollIntervalUs);
        }

        MPI_Recv(&mr, sizeof(mr), MPI_BYTE, 1, kMultiMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        for (int i = 0; i < kNumReqs; i++) {
            EXPECT_EQ(mr.sendRet[i], static_cast<int>(ncclSuccess))
                << "Multi-request " << i << " send failed";
        }
        EXPECT_EQ(mr.fatalCount, 0) << "Fatal error during multi-request failover";
        EXPECT_TRUE(allDone) << "Not all receiver requests completed";

        if (allDone) {
            for (int i = 0; i < kNumReqs; i++) {
                size_t offset = (i + 1) * kMsgSize;
                EXPECT_EQ(memcmp(recvBuf.data() + offset, sendBuf.data() + offset, kMsgSize), 0)
                    << "Data corruption in multi-request " << i;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 0; i < kNumReqs; i++) {
            if (recvReqs[i]) DrainRecvRequest(recvReqs[i]);
        }
    }
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: RecoveryThreadStartedOnlyWithParam
//
// Verifies that the PORT_RECOVERY background thread is started only when
// NCCL_IB_RESILIENCY_PORT_RECOVERY=1 is set, and not otherwise.
//
// Two sub-cases:
//   1. With PORT_RECOVERY=1 + PORT_FAILOVER=1: recoveryEnabled must be true.
//   2. Without PORT_RECOVERY (env unset or 0): recoveryEnabled must be false.
//
// Does not require NIC Fusion — we only check the resiliency state struct.
// =============================================================================
TEST_F(NetIbMPITest, RecoveryThreadStartedOnlyWithParam) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const char* failoverEnv  = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    const char* recoveryEnv  = getenv("NCCL_IB_RESILIENCY_PORT_RECOVERY");

    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    }

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    AssertInitAndGetDevices(nullptr);

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/0, &listenComm, &sendComm, &recvComm);

    struct RecoveryEnabledResult {
        int recoveryEnabled;
        int getStateRet;
    };
    static constexpr int kRecoveryEnabledMpiTag = 9893;
    RecoveryEnabledResult r = {};

    if (rank == 1) {
        struct ncclIbCastResiliencyState resState = {};
        r.getStateRet = static_cast<int>(ncclIbCastGetResiliencyState(sendComm, &resState));
        r.recoveryEnabled = resState.recoveryEnabled ? 1 : 0;
        MPI_Send(&r, sizeof(r), MPI_BYTE, 0, kRecoveryEnabledMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Recv(&r, sizeof(r), MPI_BYTE, 1, kRecoveryEnabledMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        ASSERT_EQ(r.getStateRet, static_cast<int>(ncclSuccess))
            << "ncclIbCastGetResiliencyState failed — resiliency context not created; "
            << "is NCCL_IB_RESILIENCY_PORT_FAILOVER=1?";

        bool recoveryParamSet = (recoveryEnv && strcmp(recoveryEnv, "1") == 0);
        if (recoveryParamSet) {
            EXPECT_EQ(r.recoveryEnabled, 1)
                << "recoveryEnabled should be true when NCCL_IB_RESILIENCY_PORT_RECOVERY=1";
        } else {
            EXPECT_EQ(r.recoveryEnabled, 0)
                << "recoveryEnabled should be false when NCCL_IB_RESILIENCY_PORT_RECOVERY is not set";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    TeardownConnection(recvComm, listenComm, sendComm, nullptr);
}

// =============================================================================
// Test: RecoverySuccessRestoresTraffic
//
// Core PORT_RECOVERY test. Requires NIC Fusion (ndevs >= 2) and both
// PORT_FAILOVER=1 and PORT_RECOVERY=1.
//
// Sequence:
//   1. Drive both sender and receiver data QP 0 to IBV_QPS_ERR — models a
//      link failure. Both sides detect WR_FLUSH_ERR, enter failover, then
//      recovery. The recovery handshake (alive messages + ACK) requires
//      both sides to have recovery contexts.
//   2. Poll ncclIbCastGetResiliencyState on sender until devState[0]
//      returns to Ok/Recovered (recovery restored the data QP to RTS).
//   3. Send 20 messages and verify data integrity — confirms restored QPs
//      carry sustained live traffic.
// =============================================================================
TEST_F(NetIbMPITest, RecoverySuccessRestoresTraffic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const char* failoverEnv  = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    const char* recoveryEnv  = getenv("NCCL_IB_RESILIENCY_PORT_RECOVERY");

    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    }
    if (!recoveryEnv || strcmp(recoveryEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_RECOVERY=1";
    }

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    int totalDevs = 0;
    AssertInitAndGetDevices(&totalDevs);

    int mergedDev = CreateMergedDeviceForFailover(net_, totalDevs);
    if (mergedDev < 0) {
        GTEST_SKIP() << "Requires NIC Fusion (ndevs >= 2). Found " << totalDevs << " physical devices.";
    }

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/mergedDev, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 8192;
    constexpr int    kPostRecoveryMsgs = 20;
    const size_t     kBufSize = kMsgSize * (kPostRecoveryMsgs + 1);
    std::vector<char> sendBuf(kBufSize), recvBuf(kBufSize);
    for (size_t i = 0; i < kBufSize; i++) sendBuf[i] = static_cast<char>((i * 37 + 19) & 0xFF);
    memset(recvBuf.data(), 0, kBufSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    char* regBuf  = (rank == 0) ? recvBuf.data() : sendBuf.data();
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, regBuf, kBufSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    const int actualNqps = GetActualNqps(sendComm, recvComm, regBuf, kMsgSize, /*tag=*/1300, mhandle);
    ASSERT_GT(actualNqps, 0);

    // ── Phase 1: drive both sender and receiver QP 0 to ERR ──────────────
    // Both sides must detect the failure and enter recovery for the
    // handshake to complete (sender=requestor, receiver=responder).
    static constexpr int kRecoveryMpiTag     = 9894;
    static constexpr int kPostRecoveryMpiTag = 9895;
    static constexpr int kRecoveryPollIters  = 4000;  // 4000 * 10ms = 40s

    struct FailoverPhaseResult {
        int sendRet;
        int fatalCount;
        int devState0AfterFailover;
    };
    struct RecoveryPhaseResult {
        int recoveryCount0;
    };

    FailoverPhaseResult fp = {};
    RecoveryPhaseResult rp = {};

    // Post a recv + send that will trigger CQE errors on both sides after
    // we drive their QPs to error.
    bool phase1RecvDone = false;
    void* recvReq1 = nullptr;

    if (rank == 0) {
        void*  bufs[1]    = {regBuf};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {1301};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq1), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Drive both sender and receiver data QP 0 to ERR — models a link failure
    // that breaks QPs on both sides. Both must detect the CQE errors and enter
    // recovery for the alive-message handshake to complete.
    if (rank == 0) {
        ASSERT_EQ(ncclIbCastFaultDriveRecvQpToError(recvComm, 0), ncclSuccess);
    } else {
        ASSERT_EQ(ncclIbCastFaultDriveQpToError(sendComm, 0), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm, regBuf, kMsgSize, 1301, mhandle, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }

        if (sendRet == ncclSuccess && sendReq != nullptr) {
            for (int poll = 0; poll < 500; poll++) {
                int done = 0, sz = 0;
                ncclResult_t testRet = TestRequest(sendReq, &done, &sz);
                if (testRet != ncclSuccess) { sendRet = testRet; break; }
                if (done) break;
                usleep(kPollIntervalUs);
            }
        }

        int fatalCount = 0;
        ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);

        struct ncclIbCastResiliencyState resState = {};
        ncclIbCastGetResiliencyState(sendComm, &resState);

        fp.sendRet              = static_cast<int>(sendRet);
        fp.fatalCount           = fatalCount;
        fp.devState0AfterFailover = resState.devState[0];

        MPI_Send(&fp, sizeof(fp), MPI_BYTE, 0, kRecoveryMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Poll receiver — the recv will see WR_FLUSH_ERR on QP 0 and enter
        // failover+recovery. Data may or may not arrive via surviving device.
        for (int poll = 0; poll < 1500; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq1, &done, &sz) != ncclSuccess) break;
            if (done) { phase1RecvDone = true; break; }
            usleep(kPollIntervalUs);
        }
        if (!phase1RecvDone) DrainRecvRequest(recvReq1);

        MPI_Recv(&fp, sizeof(fp), MPI_BYTE, 1, kRecoveryMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        EXPECT_EQ(fp.sendRet, static_cast<int>(ncclSuccess))
            << "Phase 1: send should complete via surviving device after failover";
        EXPECT_EQ(fp.fatalCount, 0)
            << "Phase 1: no fatal error expected";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Phase 2: wait for recovery to restore devState[0] ──
    // Poll devState (set directly by recovery thread) rather than recoveryCount
    // (which requires IbCastResiliencyProgress to be called from an active request).
    if (rank == 1) {
        struct ncclIbCastResiliencyState resState = {};
        for (int poll = 0; poll < kRecoveryPollIters; poll++) {
            ncclIbCastGetResiliencyState(sendComm, &resState);
            if (resState.devState[0] == kDevStateOk || resState.devState[0] == kDevStateRecovered) break;
            usleep(10000);  // 10 ms
        }
        rp.recoveryCount0 = resState.devState[0];
        MPI_Send(&rp, sizeof(rp), MPI_BYTE, 0, kPostRecoveryMpiTag, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&rp, sizeof(rp), MPI_BYTE, 1, kPostRecoveryMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Phase 3: send 20 messages to verify sustained traffic on restored QPs ──
    if (rank == 0) {
        EXPECT_TRUE(rp.recoveryCount0 == kDevStateOk || rp.recoveryCount0 == kDevStateRecovered)
            << "Recovery did not restore device 0 within timeout; "
            << "devState[0]=" << rp.recoveryCount0;
    }

    if (rp.recoveryCount0 < 1) {
        if (rank == 0) {
            ADD_FAILURE() << "Recovery did not succeed — skipping post-recovery traffic";
        }
        TeardownConnection(recvComm, listenComm, sendComm, mhandle);
        return;
    }

    constexpr int    kBaseTag = 1310;
    for (int m = 0; m < kPostRecoveryMsgs; m++) {
        char* msgSendBuf = sendBuf.data() + (m + 1) * kMsgSize;
        char* msgRecvBuf = recvBuf.data() + (m + 1) * kMsgSize;
        void* req = nullptr;

        if (rank == 0) {
            void*  bufs[1]    = {msgRecvBuf};
            size_t sizes[1]   = {kMsgSize};
            int    tags[1]    = {kBaseTag + m};
            void*  handles[1] = {mhandle};
            ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &req), ncclSuccess);
            int sz = 0;
            ASSERT_EQ(WaitForCompletion(req, &sz, 10000), ncclSuccess)
                << "Post-recovery message " << m << " recv failed";
        } else {
            PostSendWithRetry(sendComm, msgSendBuf, kMsgSize, kBaseTag + m, mhandle, &req);
            int sz = 0;
            ASSERT_EQ(WaitForCompletion(req, &sz, 10000), ncclSuccess)
                << "Post-recovery message " << m << " send failed";
        }
    }

    int fatalCount = 0;
    if (rank == 1) {
        ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);
    }
    MPI_Bcast(&fatalCount, 1, MPI_INT, /*root=*/1, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        for (int m = 0; m < kPostRecoveryMsgs; m++) {
            size_t offset = (m + 1) * kMsgSize;
            EXPECT_EQ(memcmp(recvBuf.data() + offset, sendBuf.data() + offset, kMsgSize), 0)
                << "Data corruption in post-recovery message " << m;
        }
        EXPECT_EQ(fatalCount, 0)
            << "Fatal error during sustained post-recovery traffic";
    }

    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: RecoveryPendingWhileLinkDown
//
// Only the sender's data QP 0 is driven to ERR — simulates a link-down
// where the receiver has not yet detected the failure. The sender enters
// recovery and starts sending alive messages, but the receiver never posts
// recv WRs on its recovery QP (no recovery context), so the sender's
// alive messages hang in RNR retry indefinitely.
//
// This is by design: the recovery thread waits indefinitely for the link
// to be restored (the customer may take an arbitrary amount of time to
// fix the physical link). There is no permanent-failure timeout.
//
// Validates: recovery stays in RecoveryInProgress(2), data still flows
// on the surviving device, no crash, no unexpected state transitions.
// =============================================================================
TEST_F(NetIbMPITest, RecoveryPendingWhileLinkDown) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const char* failoverEnv  = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    const char* recoveryEnv  = getenv("NCCL_IB_RESILIENCY_PORT_RECOVERY");

    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    }
    if (!recoveryEnv || strcmp(recoveryEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_RECOVERY=1";
    }

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    int totalDevs = 0;
    AssertInitAndGetDevices(&totalDevs);

    int mergedDev = CreateMergedDeviceForFailover(net_, totalDevs);
    if (mergedDev < 0) {
        GTEST_SKIP() << "Requires NIC Fusion (ndevs >= 2). Found " << totalDevs << " physical devices.";
    }

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/mergedDev, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 8192;
    std::vector<char> sendBuf(kMsgSize), recvBuf(kMsgSize);
    for (size_t i = 0; i < kMsgSize; i++) sendBuf[i] = static_cast<char>((i * 41 + 23) & 0xFF);
    memset(recvBuf.data(), 0, kMsgSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kMsgSize, /*tag=*/1400, mhandle);
    ASSERT_GT(actualNqps, 0);

    static constexpr int kPendingMpiTag = 9897;

    bool phase1RecvDone = false;
    void* recvReq = nullptr;

    if (rank == 0) {
        void*  bufs[1]    = {buf};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {1401};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Only sender QP — receiver stays healthy (link-down simulation)
    if (rank == 1) {
        ASSERT_EQ(ncclIbCastFaultDriveQpToError(sendComm, 0), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    struct PendingResult {
        int sendRet;
        int fatalCount;
        int devState0;
        int outstandingRecovery;
    };
    PendingResult pr = {};

    if (rank == 1) {
        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm, buf, kMsgSize, 1401, mhandle, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }

        if (sendRet == ncclSuccess && sendReq != nullptr) {
            for (int poll = 0; poll < 500; poll++) {
                int done = 0, sz = 0;
                ncclResult_t testRet = TestRequest(sendReq, &done, &sz);
                if (testRet != ncclSuccess) { sendRet = testRet; break; }
                if (done) break;
                usleep(kPollIntervalUs);
            }
        }

        // Poll until recovery thread starts (devState transitions to
        // RecoveryInProgress). Much faster than a fixed sleep on fast HW.
        struct ncclIbCastResiliencyState resState = {};
        for (int poll = 0; poll < 500; poll++) {
            ncclIbCastGetResiliencyState(sendComm, &resState);
            if (resState.devState[0] == kDevStateRecoveryInProgress) break;
            usleep(10000);  // 10ms
        }

        int fatalCount = 0;
        ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);
        ncclIbCastGetResiliencyState(sendComm, &resState);

        pr.sendRet             = static_cast<int>(sendRet);
        pr.fatalCount          = fatalCount;
        pr.devState0           = resState.devState[0];
        pr.outstandingRecovery = resState.outstandingRecovery;

        MPI_Send(&pr, sizeof(pr), MPI_BYTE, 0, kPendingMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int poll = 0; poll < 1500; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq, &done, &sz) != ncclSuccess) break;
            if (done) { phase1RecvDone = true; break; }
            usleep(kPollIntervalUs);
        }

        MPI_Recv(&pr, sizeof(pr), MPI_BYTE, 1, kPendingMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        // Send should succeed via surviving device (failover works)
        EXPECT_EQ(pr.sendRet, static_cast<int>(ncclSuccess))
            << "Send should complete via surviving device";
        EXPECT_EQ(pr.fatalCount, 0)
            << "No fatal error expected — failover handles the QP error";

        // Recovery should be stuck in RecoveryInProgress(2) — by design,
        // the recovery thread waits indefinitely for the link to come back.
        EXPECT_EQ(pr.devState0, kDevStateRecoveryInProgress)
            << "Device 0 should be in RecoveryInProgress(2) while link is down; "
            << "got " << pr.devState0;
        EXPECT_EQ(pr.outstandingRecovery, 1)
            << "Should have 1 outstanding recovery operation";

        if (pr.sendRet == static_cast<int>(ncclSuccess)) {
            EXPECT_TRUE(phase1RecvDone)
                << "Send succeeded but receiver never got the data";
        }
        if (phase1RecvDone) {
            EXPECT_EQ(memcmp(recvBuf.data(), sendBuf.data(), kMsgSize), 0)
                << "Data corruption on surviving device after failover";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !phase1RecvDone)
        DrainRecvRequest(recvReq);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: RecoveryDeviceOneFailure
//
// Same as RecoverySuccessRestoresTraffic but faults device 1 (QP index 1)
// instead of device 0. QP-to-device mapping is interleaved (QP 0=dev 0,
// QP 1=dev 1, QP 2=dev 0...), so this catches index-arithmetic bugs in
// IbCastPortRecoveryQpsToError, IbCastPortRecoveryQpsRestore, and
// IbCastPortRecoveryContextInit that would only surface with device 1.
// =============================================================================
TEST_F(NetIbMPITest, RecoveryDeviceOneFailure) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const char* failoverEnv  = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    const char* recoveryEnv  = getenv("NCCL_IB_RESILIENCY_PORT_RECOVERY");

    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    }
    if (!recoveryEnv || strcmp(recoveryEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_RECOVERY=1";
    }

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    int totalDevs = 0;
    AssertInitAndGetDevices(&totalDevs);

    int mergedDev = CreateMergedDeviceForFailover(net_, totalDevs);
    if (mergedDev < 0) {
        GTEST_SKIP() << "Requires NIC Fusion (ndevs >= 2). Found " << totalDevs << " physical devices.";
    }

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/mergedDev, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 8192;
    const size_t     kBufSize = kMsgSize * 2;
    std::vector<char> sendBuf(kBufSize), recvBuf(kBufSize);
    for (size_t i = 0; i < kBufSize; i++) sendBuf[i] = static_cast<char>((i * 47 + 31) & 0xFF);
    memset(recvBuf.data(), 0, kBufSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    char* regBuf  = (rank == 0) ? recvBuf.data() : sendBuf.data();
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, regBuf, kBufSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    const int actualNqps = GetActualNqps(sendComm, recvComm, regBuf, kMsgSize, /*tag=*/1600, mhandle);
    ASSERT_GT(actualNqps, 1);

    // ── Phase 1: drive both sender and receiver data QP 1 to ERR ─────────
    static constexpr int kDev1RecoveryMpiTag     = 9900;
    static constexpr int kDev1PostRecoveryMpiTag = 9901;
    static constexpr int kRecoveryPollIters      = 4000;  // 40s

    struct FailoverPhaseResult {
        int sendRet;
        int fatalCount;
        int devState1AfterFailover;
    };

    FailoverPhaseResult fp = {};
    int devState1AfterRecovery = -1;

    bool phase1RecvDone = false;
    void* recvReq1 = nullptr;

    if (rank == 0) {
        void*  bufs[1]    = {regBuf};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {1601};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq1), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Fault QP index 1 = device 1 on both sides
    if (rank == 0) {
        ASSERT_EQ(ncclIbCastFaultDriveRecvQpToError(recvComm, 1), ncclSuccess);
    } else {
        ASSERT_EQ(ncclIbCastFaultDriveQpToError(sendComm, 1), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm, regBuf, kMsgSize, 1601, mhandle, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }

        if (sendRet == ncclSuccess && sendReq != nullptr) {
            for (int poll = 0; poll < 500; poll++) {
                int done = 0, sz = 0;
                ncclResult_t testRet = TestRequest(sendReq, &done, &sz);
                if (testRet != ncclSuccess) { sendRet = testRet; break; }
                if (done) break;
                usleep(kPollIntervalUs);
            }
        }

        int fatalCount = 0;
        ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);

        struct ncclIbCastResiliencyState resState = {};
        ncclIbCastGetResiliencyState(sendComm, &resState);

        fp.sendRet              = static_cast<int>(sendRet);
        fp.fatalCount           = fatalCount;
        fp.devState1AfterFailover = resState.devState[1];

        MPI_Send(&fp, sizeof(fp), MPI_BYTE, 0, kDev1RecoveryMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int poll = 0; poll < 1500; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq1, &done, &sz) != ncclSuccess) break;
            if (done) { phase1RecvDone = true; break; }
            usleep(kPollIntervalUs);
        }
        if (!phase1RecvDone) DrainRecvRequest(recvReq1);

        MPI_Recv(&fp, sizeof(fp), MPI_BYTE, 1, kDev1RecoveryMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        EXPECT_EQ(fp.sendRet, static_cast<int>(ncclSuccess))
            << "Phase 1: send should complete via device 0 after device 1 failover";
        EXPECT_EQ(fp.fatalCount, 0)
            << "Phase 1: no fatal error expected";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Phase 2: wait for recovery on device 1 ──────────────────────────
    if (rank == 1) {
        struct ncclIbCastResiliencyState resState = {};
        for (int poll = 0; poll < kRecoveryPollIters; poll++) {
            ncclIbCastGetResiliencyState(sendComm, &resState);
            if (resState.devState[1] == kDevStateOk || resState.devState[1] == kDevStateRecovered) break;
            usleep(10000);
        }
        devState1AfterRecovery = resState.devState[1];
        MPI_Send(&devState1AfterRecovery, 1, MPI_INT, 0, kDev1PostRecoveryMpiTag, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&devState1AfterRecovery, 1, MPI_INT, 1, kDev1PostRecoveryMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Phase 3: post-recovery send ──────────────────────────────────────
    bool phase3RecvDone = false;
    void* recvReq3 = nullptr;

    if (rank == 0) {
        void*  bufs[1]    = {regBuf + kMsgSize};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {1602};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq3), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    struct PostRecoveryResult {
        int sendRet;
        int fatalCount;
    };
    static constexpr int kDev1Phase3MpiTag = 9902;
    PostRecoveryResult pr = {};

    if (rank == 1) {
        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm, regBuf + kMsgSize, kMsgSize, 1602, mhandle, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }

        if (sendRet == ncclSuccess && sendReq != nullptr) {
            for (int poll = 0; poll < 500; poll++) {
                int done = 0, sz = 0;
                ncclResult_t testRet = TestRequest(sendReq, &done, &sz);
                if (testRet != ncclSuccess) { sendRet = testRet; break; }
                if (done) break;
                usleep(kPollIntervalUs);
            }
        }

        int fatalCount = 0;
        ncclIbCastFaultGetFatalCount(sendComm, &fatalCount);

        pr.sendRet   = static_cast<int>(sendRet);
        pr.fatalCount = fatalCount;

        MPI_Send(&pr, sizeof(pr), MPI_BYTE, 0, kDev1Phase3MpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int poll = 0; poll < 500; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq3, &done, &sz) != ncclSuccess) break;
            if (done) { phase3RecvDone = true; break; }
            usleep(kPollIntervalUs);
        }

        MPI_Recv(&pr, sizeof(pr), MPI_BYTE, 1, kDev1Phase3MpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        EXPECT_TRUE(devState1AfterRecovery == kDevStateOk || devState1AfterRecovery == kDevStateRecovered)
            << "Recovery did not restore device 1 within timeout; "
            << "devState[1]=" << devState1AfterRecovery;

        EXPECT_EQ(pr.sendRet, static_cast<int>(ncclSuccess))
            << "Post-recovery send failed (sendRet=" << pr.sendRet << ")";
        EXPECT_EQ(pr.fatalCount, 0)
            << "Fatal error during post-recovery traffic";

        if (pr.sendRet == static_cast<int>(ncclSuccess)) {
            EXPECT_TRUE(phase3RecvDone)
                << "Post-recovery send succeeded but receiver did not get data";
        }
        if (phase3RecvDone) {
            size_t offset = kMsgSize;
            EXPECT_EQ(memcmp(recvBuf.data() + offset, sendBuf.data() + offset, kMsgSize), 0)
                << "Data corruption in post-recovery message on device 1";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !phase3RecvDone)
        DrainRecvRequest(recvReq3);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

// =============================================================================
// Test: RecoveryUdTimeoutExhaustsAttempts
//
// Validates the UD recovery QP retry/timeout cycle. Only the sender's data
// QP is driven to ERR — the receiver never enters recovery. With UD recovery
// QPs, alive messages complete locally (fire-and-forget) but the receiver
// never ACKs. The sender cycles through:
//   AliveMessages → Ack (timeout) → retry → ... → RecoveryFailed
//
// This path was unreachable with RC recovery QPs (alive messages hung in
// RNR retry forever). UD makes it testable.
//
// Default timing: 200ms start + 5 * (500ms batch + 5s ack timeout) ≈ 28s
// =============================================================================
TEST_F(NetIbMPITest, RecoveryUdTimeoutExhaustsAttempts) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    const char* failoverEnv  = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    const char* recoveryEnv  = getenv("NCCL_IB_RESILIENCY_PORT_RECOVERY");

    if (!failoverEnv || strcmp(failoverEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_FAILOVER=1";
    }
    if (!recoveryEnv || strcmp(recoveryEnv, "1") != 0) {
        GTEST_SKIP() << "Requires NCCL_IB_RESILIENCY_PORT_RECOVERY=1";
    }

    const int rank = MPIEnvironment::world_rank;

    net_ = &netIbCast;
    int totalDevs = 0;
    AssertInitAndGetDevices(&totalDevs);

    int mergedDev = CreateMergedDeviceForFailover(net_, totalDevs);
    if (mergedDev < 0) {
        GTEST_SKIP() << "Requires NIC Fusion (ndevs >= 2). Found " << totalDevs << " physical devices.";
    }

    void* listenComm = nullptr;
    void* sendComm   = nullptr;
    void* recvComm   = nullptr;
    SetupCastConnection(/*dev=*/mergedDev, &listenComm, &sendComm, &recvComm);

    constexpr size_t kMsgSize = 8192;
    std::vector<char> sendBuf(kMsgSize), recvBuf(kMsgSize);
    for (size_t i = 0; i < kMsgSize; i++) sendBuf[i] = static_cast<char>((i * 53 + 37) & 0xFF);
    memset(recvBuf.data(), 0, kMsgSize);

    void* comm    = (rank == 0) ? recvComm : sendComm;
    void* buf     = (rank == 0) ? static_cast<void*>(recvBuf.data())
                                : static_cast<void*>(sendBuf.data());
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buf, kMsgSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    const int actualNqps = GetActualNqps(sendComm, recvComm, buf, kMsgSize, /*tag=*/1700, mhandle);
    ASSERT_GT(actualNqps, 0);

    static constexpr int kTimeoutMpiTag = 9903;
    // 200ms start + 5 * (500ms + 5s) ≈ 28s. Poll for 45s to be safe.
    static constexpr int kTimeoutPollIters = 4500;

    bool phase1RecvDone = false;
    void* recvReq = nullptr;

    if (rank == 0) {
        void*  bufs[1]    = {buf};
        size_t sizes[1]   = {kMsgSize};
        int    tags[1]    = {1701};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &recvReq), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        ASSERT_EQ(ncclIbCastFaultDriveQpToError(sendComm, 0), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    struct TimeoutResult {
        int sendRet;
        int devState0;
        int outstandingRecovery;
    };
    TimeoutResult tr = {};

    if (rank == 1) {
        void* sendReq = nullptr;
        ncclResult_t sendRet = ncclSuccess;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            sendRet = PostSend(sendComm, buf, kMsgSize, 1701, mhandle, &sendReq);
            if (sendRet != ncclSuccess || sendReq != nullptr) break;
            usleep(kPollIntervalUs);
        }

        if (sendRet == ncclSuccess && sendReq != nullptr) {
            for (int poll = 0; poll < 500; poll++) {
                int done = 0, sz = 0;
                ncclResult_t testRet = TestRequest(sendReq, &done, &sz);
                if (testRet != ncclSuccess) { sendRet = testRet; break; }
                if (done) break;
                usleep(kPollIntervalUs);
            }
        }

        // Wait for recovery to exhaust all attempts
        struct ncclIbCastResiliencyState resState = {};
        for (int poll = 0; poll < kTimeoutPollIters; poll++) {
            ncclIbCastGetResiliencyState(sendComm, &resState);
            if (resState.devState[0] == kDevStateRecoveryFailed ||
                resState.devState[0] == kDevStateErrorPermanent) break;
            usleep(10000);  // 10ms
        }

        tr.sendRet             = static_cast<int>(sendRet);
        tr.devState0           = resState.devState[0];
        tr.outstandingRecovery = resState.outstandingRecovery;

        MPI_Send(&tr, sizeof(tr), MPI_BYTE, 0, kTimeoutMpiTag, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int poll = 0; poll < 1500; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq, &done, &sz) != ncclSuccess) break;
            if (done) { phase1RecvDone = true; break; }
            usleep(kPollIntervalUs);
        }

        MPI_Recv(&tr, sizeof(tr), MPI_BYTE, 1, kTimeoutMpiTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        EXPECT_EQ(tr.sendRet, static_cast<int>(ncclSuccess))
            << "Send should complete via surviving device";
        // Recovery should have failed after exhausting all attempts
        EXPECT_TRUE(tr.devState0 == kDevStateRecoveryFailed ||
                    tr.devState0 == kDevStateErrorPermanent)
            << "Expected RecoveryFailed(3) or ErrorPermanent(5) after UD timeout; "
            << "got devState[0]=" << tr.devState0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && !phase1RecvDone)
        DrainRecvRequest(recvReq);
    TeardownConnection(recvComm, listenComm, sendComm, mhandle);
}

#endif /* MPI_TESTS_ENABLED && ENABLE_FAULT_INJECTION */
