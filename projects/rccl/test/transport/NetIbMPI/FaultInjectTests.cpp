/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "NetIbMPITestBase.hpp"
#include "NetIbCastInspect.hpp"
#include "NetIbFaultInject.hpp"

#if defined(MPI_TESTS_ENABLED) && defined(ENABLE_FAULT_INJECTION)

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

#endif /* MPI_TESTS_ENABLED && ENABLE_FAULT_INJECTION */
