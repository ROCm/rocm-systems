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

    // Drain any outstanding recv request before closing the comm.
    // Note: recv may not complete if sender faulted — that is expected here.
    if (rank == 0 && !recvDone && recvReq != nullptr) {
        for (int poll = 0; poll < 500; poll++) {
            int done = 0, sz = 0;
            if (TestRequest(recvReq, &done, &sz) != ncclSuccess) break;
            if (done) break;
            usleep(kPollIntervalUs);
        }
    }

    ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
    if (rank == 0) {
        ASSERT_EQ(CloseRecvComm(recvComm), ncclSuccess);
        ASSERT_EQ(CloseListenComm(listenComm), ncclSuccess);
    } else {
        ASSERT_EQ(CloseSendComm(sendComm), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

#endif /* MPI_TESTS_ENABLED && ENABLE_FAULT_INJECTION */
