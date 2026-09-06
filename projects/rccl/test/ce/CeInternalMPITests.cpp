/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// White-box tests for CE internal functions (ncclCeInit, ncclPrepUCSync,
// ncclCeLaunchBatchOps, ncclCeImplemented). Requires Debug build.

#include "CeTestHelpers.hpp"
#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "SymmetricBufferHelpers.hpp"
#include "rccl/rccl.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <comm.h>
#include <ce_coll.h>

#include <cstdint>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;

// Internal CE helpers not in ce_coll.h; accessible in Debug builds (-fvisibility=hidden not applied).
ncclResult_t ncclPrepUCSync(struct ncclComm* comm, bool isComplete,
                            hipStreamBatchMemOpParams* batchParams,
                            size_t* opIdx, hipStream_t stream);

ncclResult_t ncclCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int nRanks);
void         ncclCeFreeBatchOpsParams(struct ncclCeBatchOpsParams* params);

// Fixture: skip if no CE driver; create comm; warmup AllGather → ncclCeInit; TearDown destroys comm.
class CeInternalMPITest : public MPITestBase
{
protected:
    ncclComm* ceComm = nullptr;

    // RAII guards: set NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS=ALL before communicator
    // creation so that CE init/dispatch log lines are always available for diagnostics,
    // regardless of the external environment.  Mirrors the pattern in CeMPITest.
    std::unique_ptr<MPIHelpers::MpiEnvGuard> debugGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard> debugSubsysGuard_;

    void SetUp() override
    {
        MPITestBase::SetUp();

        if(!isCeDispatchConfigured())
            GTEST_SKIP() << "CE not configured: need CE_BATCH_ASYNC_SUPPORTED build + "
                            "NCCL_CTA_POLICY=2, NCCL_LOCAL_REGISTER=0, NCCL_CUMEM_ENABLE=1";

        // Set debug env vars before ncclCommInitRank so the communicator picks them up.
        debugGuard_       = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG",        "INFO");
        debugSubsysGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG_SUBSYS", "ALL");

        ASSERT_EQ(createTestCommunicator(), ncclSuccess)
            << "ncclCommInitRank failed";

        ceComm = static_cast<ncclComm*>(getActiveCommunicator());
        ASSERT_NE(ceComm, nullptr);

        // Warmup AllGather triggers ncclCeInit via group.cc (NCCL_CTA_POLICY=2).
        // SymBuf (VMM-backed) satisfies ncclDevrFindWindow for ceCollTaskAppend.
        constexpr size_t kElem = 4;
        RCCLTestHelpers::SymBuf sendSym, recvSym;
        ASSERT_EQ(RCCLTestHelpers::ncclSymBufAlloc(
                      getActiveCommunicator(), kElem * sizeof(float), sendSym),
                  ncclSuccess) << "ncclSymBufAlloc for sendBuf failed (cuMem enabled?)";
        ASSERT_EQ(RCCLTestHelpers::ncclSymBufAlloc(
                      getActiveCommunicator(), kElem * ceComm->nRanks * sizeof(float), recvSym),
                  ncclSuccess) << "ncclSymBufAlloc for recvBuf failed";

        ASSERT_EQ(ncclAllGather(sendSym.ptr, recvSym.ptr, kElem, ncclFloat32,
                                getActiveCommunicator(), getActiveStream()),
                  ncclSuccess);
        ASSERT_EQ(hipStreamSynchronize(getActiveStream()), hipSuccess);

        // White-box check: baseUCSymReadyPtr is ncclCeInit's primary output — it is
        // the definition of "CE initialized".
        if(ceComm->ceColl.baseUCSymReadyPtr == nullptr)
            GTEST_SKIP() << "CE not available on this system (ncclCeInit was not triggered)";
    }

    void TearDown() override
    {
        ceComm = nullptr;
        // Destroy communicator before releasing the debug guards so any final
        // NCCL log lines from comm teardown still go to the debug output.
        MPITestBase::TearDown();
        debugSubsysGuard_.reset();
        debugGuard_.reset();
    }

    // Batch buffer for ncclPrepUCSync: capacity = NCCL_CE_SYNC_OPS_PER_RANK_UC * nRanks.
    std::vector<hipStreamBatchMemOpParams> makePrepSyncBatch() const
    {
        size_t batchSize = NCCL_CE_SYNC_OPS_PER_RANK_UC * ceComm->nRanks;
        return std::vector<hipStreamBatchMemOpParams>(batchSize);
    }

    // Skip if fewer than n ranks.
    void requireMinRanks(int n)
    {
        if(ceComm->nRanks < n)
            GTEST_SKIP() << "Need >= " << n << " MPI ranks";
    }

    // Allocate a batch, call ncclPrepUCSync once, and return both for inspection.
    struct PrepSyncResult
    {
        std::vector<hipStreamBatchMemOpParams> batch;
        size_t opIdx = 0;
    };

    PrepSyncResult callPrepUCSync(bool isComplete = false)
    {
        PrepSyncResult res;
        res.batch = makePrepSyncBatch();
        EXPECT_EQ(ncclPrepUCSync(ceComm, isComplete, res.batch.data(), &res.opIdx,
                                 getActiveStream()),
                  ncclSuccess);
        return res;
    }
};

// ===========================================================================
// CeInternal_Init – CE init / finalize lifecycle
// ===========================================================================

// INIT-01: After ncclCeInit, the sync-window pointers are non-null.
TEST_F(CeInternalMPITest, InitSetsWindowPointers)
{
    EXPECT_NE(ceComm->ceColl.baseUCSymReadyPtr, nullptr);
    EXPECT_NE(ceComm->ceColl.baseUCSymComplPtr, nullptr);
    EXPECT_NE(ceComm->ceColl.ceSyncWin, nullptr);
}

// INIT-01b: ncclCeInit allocates ceSeqNumDev and seeds GRAPH_SYNC_VALUE in slot [1].
TEST_F(CeInternalMPITest, InitAllocatesCeSeqNumDev)
{
    ASSERT_NE(ceComm->ceColl.ceSeqNumDev, nullptr);
    uint32_t graphSlot = 0;
    ASSERT_EQ(hipMemcpy(&graphSlot, ceComm->ceColl.ceSeqNumDev + 1, sizeof(uint32_t),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(graphSlot, kCeGraphSyncValue);
}

// INIT-02: ncclCeFinalize sets sync-window pointers back to null.
TEST_F(CeInternalMPITest, FiniNullsPointers)
{
    ASSERT_NE(ceComm->ceColl.baseUCSymReadyPtr, nullptr);
    ASSERT_EQ(ncclCeFinalize(ceComm), ncclSuccess);
    EXPECT_EQ(ceComm->ceColl.baseUCSymReadyPtr, nullptr);
    EXPECT_EQ(ceComm->ceColl.baseUCSymComplPtr, nullptr);
    EXPECT_EQ(ceComm->ceColl.ceSyncWin, nullptr);
    EXPECT_EQ(ceComm->ceColl.ceSeqNumDev, nullptr);
}

// INIT-03: Calling ncclCeFinalize twice is safe (idempotent).
TEST_F(CeInternalMPITest, DoubleFiniIsSafe)
{
    EXPECT_EQ(ncclCeFinalize(ceComm), ncclSuccess);
    EXPECT_EQ(ncclCeFinalize(ceComm), ncclSuccess);
    EXPECT_EQ(ceComm->ceColl.baseUCSymReadyPtr, nullptr);
    EXPECT_EQ(ceComm->ceColl.ceSeqNumDev, nullptr);
}

// INIT-03b: Double finalize must not double-free ceSeqNumDev.
TEST_F(CeInternalMPITest, InitFiniIdempotentCeSeqNumDev)
{
    ASSERT_NE(ceComm->ceColl.ceSeqNumDev, nullptr);
    EXPECT_EQ(ncclCeFinalize(ceComm), ncclSuccess);
    EXPECT_EQ(ceComm->ceColl.ceSeqNumDev, nullptr);
    EXPECT_EQ(ncclCeFinalize(ceComm), ncclSuccess);
    EXPECT_EQ(ceComm->ceColl.ceSeqNumDev, nullptr);
}

// INIT-04: Two comms have independent CE state; advancing one must not affect the other.
TEST_F(CeInternalMPITest, IndependentCeStatePerComm)
{
    requireMinRanks(2);
    const int nRanks = ceComm->nRanks;

    // Create a second communicator spanning the same ranks.
    ncclUniqueId uid{};
    if(ceComm->rank == 0)
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&uid));
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm2 = nullptr;
    ASSERT_EQ(ncclSuccess,
              ncclCommInitRank(&comm2, nRanks, uid, ceComm->rank));
    RCCLTestGuards::NcclCommAutoGuard comm2Guard(comm2);

    auto* ceComm2 = static_cast<ncclComm*>(comm2);

    // Trigger CE init on comm2 via a warmup AllGather.
    constexpr size_t kElem = 4;
    RCCLTestHelpers::SymBuf s2, r2;
    ASSERT_EQ(ncclSuccess,
              RCCLTestHelpers::ncclSymBufAlloc(comm2, kElem * sizeof(float), s2));
    ASSERT_EQ(ncclSuccess,
              RCCLTestHelpers::ncclSymBufAlloc(
                  comm2, kElem * static_cast<size_t>(nRanks) * sizeof(float), r2));
    ASSERT_EQ(ncclSuccess,
              ncclAllGather(s2.ptr, r2.ptr, kElem, ncclFloat32, comm2,
                            getActiveStream()));
    // hipStreamSynchronize has no timeout; a CE hang here means CE dispatch failed.
    // The test runner's --timeout option provides the external deadline.
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    // White-box check: baseUCSymReadyPtr is ncclCeInit's primary output — it is
    // the definition of "CE initialized on this comm".
    if(ceComm2->ceColl.baseUCSymReadyPtr == nullptr)
        GTEST_SKIP() << "CE not available on comm2";

    EXPECT_NE(ceComm->ceColl.baseUCSymReadyPtr,
              ceComm2->ceColl.baseUCSymReadyPtr)
        << "Both comms share the same ready-pointer allocation (CE state not isolated)";
    EXPECT_NE(ceComm->ceColl.ceSyncWin, ceComm2->ceColl.ceSyncWin)
        << "Both comms share the same sync window";

    const uint32_t seqBefore = ceComm->ceColl.ceSeqNum;

    // Run 3 collectives on comm2 to advance its CE sequence number by several steps;
    // any count > 1 is sufficient to detect a ceSeqNum cross-contamination into comm1.
    constexpr int kAdvanceIters = 3;
    for(int i = 0; i < kAdvanceIters; ++i)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(s2.ptr, r2.ptr, kElem, ncclFloat32, comm2,
                                getActiveStream()));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    }

    EXPECT_EQ(seqBefore, ceComm->ceColl.ceSeqNum)
        << "Advancing comm2's CE state leaked into comm1";

    RCCLTestHelpers::SymBuf s1, r1;
    ASSERT_EQ(ncclSuccess,
              RCCLTestHelpers::ncclSymBufAlloc(
                  getActiveCommunicator(), kElem * sizeof(float), s1));
    ASSERT_EQ(ncclSuccess,
              RCCLTestHelpers::ncclSymBufAlloc(
                  getActiveCommunicator(),
                  kElem * static_cast<size_t>(nRanks) * sizeof(float), r1));

    const int rank = ceComm->rank;
    ASSERT_EQ(hipSuccess, ceFillRankScalarFloat(s1.ptr, kElem, rank));

    ASSERT_EQ(ncclSuccess,
              ncclAllGather(s1.ptr, r1.ptr, kElem, ncclFloat32,
                            getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    size_t errIdx; float errExp, errAct;
    EXPECT_TRUE(RCCLTestHelpers::verifyBufferData<float>(
                    r1.ptr, kElem * nRanks,
                    [kElem](size_t i) { return static_cast<float>(i / kElem + 1); },
                    0, 1e-5, &errIdx, &errExp, &errAct))
        << "Data corruption in comm1 AllGather after comm2 state was advanced"
        << " at idx=" << errIdx << " expected=" << errExp << " got=" << errAct;
    // comm2Guard auto-destroys comm2 on scope exit.
}

// INIT-05: ncclCeFinalize with no further collectives after SetUp warmup must succeed and zero pointers.
TEST_F(CeInternalMPITest, FiniAfterZeroCollectives)
{
    ASSERT_NE(ceComm->ceColl.baseUCSymReadyPtr, nullptr)
        << "Precondition: CE must be initialized by fixture SetUp";

    EXPECT_EQ(ncclCeFinalize(ceComm), ncclSuccess)
        << "ncclCeFinalize should succeed with no collectives outstanding";

    EXPECT_EQ(ceComm->ceColl.baseUCSymReadyPtr, nullptr);
    EXPECT_EQ(ceComm->ceColl.baseUCSymComplPtr, nullptr);
    EXPECT_EQ(ceComm->ceColl.ceSyncWin,         nullptr);
}

// ===========================================================================
// CeInternal_GraphLatch – graph-captured AllReduce immediately followed by an
// eager AllReduce on the same comm. Regression for a latch bug where CE
// 2-shot AllReduce could be selected right after a graph capture, deadlocking
// if ranks disagreed on whether CE was allowed for a given call.
// ===========================================================================

// LATCH-01: with the captured plan still live, an eager AllReduce at a
// CE-eligible size on the same comm must not hang and must be correct.
// hipStreamSynchronize has no timeout; a regression here either corrupts
// data (latch silently cleared) or hangs, caught by the test runner's
// external --timeout (same pattern as the CE dispatch tests above).
TEST_F(CeInternalMPITest, GraphThenEagerAllReduceDoesNotHang)
{
    // CE AllReduce range is [NCCL_CE_AR_MIN_MSG_BYTES, NCCL_CE_AR_MAX_MSG_BYTES];
    // 8MiB of float32 sits comfortably inside it.
    constexpr size_t kElem = (8ull * 1024 * 1024) / sizeof(float);
    const int nRanks = ceComm->nRanks;
    const int rank = ceComm->rank;

    RCCLTestHelpers::SymBuf sendSym, recvSym;
    ASSERT_EQ(RCCLTestHelpers::ncclSymBufAlloc(getActiveCommunicator(), kElem * sizeof(float), sendSym),
              ncclSuccess) << "ncclSymBufAlloc for sendBuf failed";
    ASSERT_EQ(RCCLTestHelpers::ncclSymBufAlloc(getActiveCommunicator(), kElem * sizeof(float), recvSym),
              ncclSuccess) << "ncclSymBufAlloc for recvBuf failed";
    ASSERT_EQ(hipSuccess, ceFillRankScalarFloat(sendSym.ptr, kElem, rank));

    const float expectedSum = static_cast<float>(nRanks * (nRanks + 1) / 2); // sum(1..nRanks)

    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;

    ASSERT_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal));
    ncclResult_t ncclErr = ncclAllReduce(sendSym.ptr, recvSym.ptr, kElem, ncclFloat32, ncclSum,
                                          getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, ncclErr);
    ASSERT_EQ(hipSuccess, hipStreamEndCapture(getActiveStream(), &graph));
    ASSERT_NE(nullptr, graph);
    ASSERT_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    SCOPE_EXIT(if(graphExec) (void)hipGraphExecDestroy(graphExec); if(graph) (void)hipGraphDestroy(graph));

    ASSERT_EQ(hipSuccess, hipMemset(recvSym.ptr, 0, kElem * sizeof(float)));
    ASSERT_EQ(hipSuccess, hipGraphLaunch(graphExec, getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    size_t errIdx; float errExp, errAct;
    ASSERT_TRUE(RCCLTestHelpers::verifyBufferData<float>(
                    recvSym.ptr, kElem,
                    [expectedSum](size_t) { return expectedSum; },
                    0, 1e-3, &errIdx, &errExp, &errAct))
        << "Captured AllReduce result mismatch at idx=" << errIdx
        << " expected=" << errExp << " got=" << errAct;

    // White-box: graphExec is still live, so the CE latch must still be
    // engaged -- otherwise the eager call below could pick CE while another
    // rank is still mid-teardown of its own captured plan.
    EXPECT_TRUE(ceComm->ceColl.graphModeSeen)
        << "CE AllReduce graph latch should still be set right after launch";

    // Eager AllReduce, same comm/stream, CE-eligible size, issued immediately.
    ASSERT_EQ(hipSuccess, hipMemset(recvSym.ptr, 0, kElem * sizeof(float)));
    ncclErr = ncclAllReduce(sendSym.ptr, recvSym.ptr, kElem, ncclFloat32, ncclSum,
                             getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, ncclErr);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    ASSERT_TRUE(RCCLTestHelpers::verifyBufferData<float>(
                    recvSym.ptr, kElem,
                    [expectedSum](size_t) { return expectedSum; },
                    0, 1e-3, &errIdx, &errExp, &errAct))
        << "Eager AllReduce immediately after graph capture produced wrong data"
        << " at idx=" << errIdx << " expected=" << errExp << " got=" << errAct;
    // SCOPE_EXIT above destroys graphExec/graph on return.
}

// ===========================================================================
// CeInternal_Sync – ncclPrepUCSync op-count and self-target checks
// ===========================================================================

// SYNC-01: ncclPrepUCSync increments ceSeqNum on each call.
TEST_F(CeInternalMPITest, PrepUCSyncIncrementsCeSeqNum)
{
    constexpr int kCallCount = 10; // number of ncclPrepUCSync calls to verify monotonic increment
    auto batch = makePrepSyncBatch();
    uint32_t initialSeq = ceComm->ceColl.ceSeqNum;
    for(int n = 1; n <= kCallCount; ++n)
    {
        size_t opIdx = 0;
        ASSERT_EQ(ncclPrepUCSync(ceComm, false, batch.data(), &opIdx, getActiveStream()),
                  ncclSuccess)
            << "call " << n;
        EXPECT_EQ(ceComm->ceColl.ceSeqNum, initialSeq + static_cast<uint32_t>(n))
            << "ceSeqNum should increment by 1 per call, call " << n << "/" << kCallCount;
    }
}

// SYNC-02: ncclPrepUCSync produces exactly 2*(nRanks-1) ops (one WRITE +
//          one WAIT per remote rank).
TEST_F(CeInternalMPITest, PrepUCSyncOpCount)
{
    requireMinRanks(2);
    const int nRanks = ceComm->nRanks;
    auto [batch, opIdx] = callPrepUCSync();
    EXPECT_EQ(opIdx, static_cast<size_t>(2 * (nRanks - 1)))
        << "expected 2*(nRanks-1) ops for nRanks=" << nRanks;
}

// SYNC-03: No WRITE_VALUE op targets the local rank's own ready slot.
TEST_F(CeInternalMPITest, PrepUCSyncNoSelfTargetedOp)
{
    requireMinRanks(2);
    const int rank = ceComm->rank;

    auto [batch, opIdx] = callPrepUCSync();

    // baseUCSymReadyPtr points into a VMM-backed symmetric (host-accessible)
    // allocation created by ncclCeInit via ncclSymBufAlloc.  Address arithmetic
    // on the host-side pointer is valid for comparison with batch op addresses,
    // which are also expressed as host-visible hipDeviceptr_t values.
    uint32_t*      readyPtrs = reinterpret_cast<uint32_t*>(ceComm->ceColl.baseUCSymReadyPtr);
    hipDeviceptr_t selfReady = reinterpret_cast<hipDeviceptr_t>(&readyPtrs[rank]);
    int selfTargets = 0;
    for(size_t i = 0; i < opIdx; ++i)
    {
        if(batch[i].operation == hipStreamMemOpWriteValue32 &&
           batch[i].writeValue.address == selfReady)
            ++selfTargets;
    }
    EXPECT_EQ(selfTargets, 0)
        << "A WRITE_VALUE op targeted rank " << rank << "'s own ready slot";
}

// SYNC-04: ceSeqNum wraps correctly from UINT32_MAX → 0.
TEST_F(CeInternalMPITest, SeqNumWrap)
{
    requireMinRanks(2);
    ceComm->ceColl.ceSeqNum = UINT32_MAX - 1;

    callPrepUCSync();
    EXPECT_EQ(ceComm->ceColl.ceSeqNum, UINT32_MAX);
    callPrepUCSync();
    EXPECT_EQ(ceComm->ceColl.ceSeqNum, 0u) << "wrap from UINT32_MAX to 0";
}

// SYNC-05: ceSeqNum wraps correctly through the full AllGather dispatch path (not only PrepUCSync).
TEST_F(CeInternalMPITest, SeqNumWrapAroundCollective)
{
    requireMinRanks(2);
    const int nRanks = ceComm->nRanks;

    constexpr size_t kCount = 256;

    RCCLTestHelpers::SymBuf sendSym, recvSym;
    ASSERT_EQ(ncclSuccess,
              RCCLTestHelpers::ncclSymBufAlloc(
                  getActiveCommunicator(), kCount * sizeof(float), sendSym));
    ASSERT_EQ(ncclSuccess,
              RCCLTestHelpers::ncclSymBufAlloc(
                  getActiveCommunicator(),
                  kCount * static_cast<size_t>(nRanks) * sizeof(float), recvSym));

    const int rank = ceComm->rank;
    ASSERT_EQ(hipSuccess, ceFillRankScalarFloat(sendSym.ptr, kCount, rank));

    ceComm->ceColl.ceSeqNum = UINT32_MAX - 1;

    constexpr int kWrapIters = 2; // iter 0: seqNum wraps to UINT32_MAX; iter 1: wraps to 0
    for(int iter = 0; iter < kWrapIters; ++iter)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(sendSym.ptr, recvSym.ptr, kCount, ncclFloat32,
                                getActiveCommunicator(), getActiveStream()))
            << "ncclAllGather failed at wrap iteration " << iter;
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()))
            << "hipStreamSynchronize failed at wrap iteration " << iter;

        // Verify block pattern: recvbuf[r*kCount + i] == float(r+1).
        size_t errIdx; float errExp, errAct;
        if(!RCCLTestHelpers::verifyBufferData<float>(
               recvSym.ptr, kCount * nRanks,
               [kCount](size_t i) { return static_cast<float>(i / kCount + 1); },
               0, 1e-5, &errIdx, &errExp, &errAct))
        {
            ADD_FAILURE() << "Data corruption at wrap iter=" << iter
                          << " idx=" << errIdx
                          << " expected=" << errExp << " got=" << errAct;
        }
    }
}

// SYNC-06: Under simulated graph capture, batchParams holds waits only (no fused writes).
TEST_F(CeInternalMPITest, PrepUCSyncCaptureOpCount)
{
    if(!ceSimulatedCaptureSupported())
        GTEST_SKIP() << "Simulated capture requires ROCm >= 6.1 graph support";
    ceSkipIfNvls(ceComm);
    requireMinRanks(2);

    const int nRanks = ceComm->nRanks;
    CeSimulatedCaptureGuard capture(ceComm);
    auto [batch, opIdx] = callPrepUCSync();

    EXPECT_EQ(opIdx, static_cast<size_t>(nRanks - 1))
        << "capture path should place only peer waits in batchParams";

    int writeOps = 0;
    int waitOps  = 0;
    for(size_t i = 0; i < opIdx; ++i)
    {
        if(batch[i].operation == hipStreamMemOpWriteValue32)
            ++writeOps;
        if(batch[i].operation == hipStreamMemOpWaitValue32)
        {
            ++waitOps;
            EXPECT_EQ(batch[i].waitValue.value, kCeGraphSyncValue)
                << "capture waits must use GRAPH_SYNC_VALUE";
        }
    }
    EXPECT_EQ(writeOps, 0) << "peer writes must be issued outside batchParams during capture";
    EXPECT_EQ(waitOps, nRanks - 1);
}

// SYNC-07: Non-capture path keeps fused WRITE+WAIT in one batch (latency regression guard).
TEST_F(CeInternalMPITest, PrepUCSyncNonCaptureStillFused)
{
    ceSkipIfNvls(ceComm);
    requireMinRanks(2);

    const int    nRanks     = ceComm->nRanks;
    const uint32_t seqBefore = ceComm->ceColl.ceSeqNum + 1;
    auto [batch, opIdx]      = callPrepUCSync();

    EXPECT_EQ(opIdx, static_cast<size_t>(2 * (nRanks - 1)));

    int writeOps = 0;
    int waitOps  = 0;
    for(size_t i = 0; i < opIdx; ++i)
    {
        if(batch[i].operation == hipStreamMemOpWriteValue32)
        {
            ++writeOps;
            EXPECT_EQ(batch[i].writeValue.value, seqBefore);
        }
        if(batch[i].operation == hipStreamMemOpWaitValue32)
        {
            ++waitOps;
            EXPECT_EQ(batch[i].waitValue.value, seqBefore);
        }
    }
    EXPECT_EQ(writeOps, nRanks - 1);
    EXPECT_EQ(waitOps, nRanks - 1);
}

// ===========================================================================
// CeInternal_MemOpSync – capture-time wait/reset batch split
// ===========================================================================

// MEMOP-01: Capture-time ncclMemOpSync zeros the active ready array after the barrier.
TEST_F(CeInternalMPITest, MemOpSyncCaptureResetsFlags)
{
    if(!ceSimulatedCaptureSupported())
        GTEST_SKIP() << "Simulated capture requires ROCm >= 6.1 graph support";
    ceSkipIfNvls(ceComm);
    requireMinRanks(2);

    const int nRanks = ceComm->nRanks;
    // Warmup AllGather may leave useCompletePtr true; reset targets the active array.
    ceComm->ceColl.useCompletePtr = false;
    ASSERT_EQ(ceResetUcSyncWindowOnStream(ceComm, getActiveStream()), hipSuccess);
    ASSERT_EQ(hipMemsetAsync(ceComm->ceColl.baseUCSymReadyPtr, 0xAB,
                             static_cast<size_t>(nRanks) * sizeof(uint32_t), getActiveStream()),
              hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(getActiveStream()), hipSuccess);
    MPI_Barrier(MPI_COMM_WORLD);

    CeSimulatedCaptureGuard capture(ceComm);
    const bool useCompleteBefore = ceComm->ceColl.useCompletePtr;
    ASSERT_EQ(ncclMemOpSync(ceComm, getActiveStream(), nullptr), ncclSuccess);
    ASSERT_EQ(hipStreamSynchronize(getActiveStream()), hipSuccess);

    const auto ready = ceReadUcReadyFlags(ceComm);
    ASSERT_EQ(ready.size(), static_cast<size_t>(nRanks));
    for(int i = 0; i < nRanks; ++i)
        EXPECT_EQ(ready[static_cast<size_t>(i)], 0u) << "ready slot " << i << " not reset";

    EXPECT_EQ(ceComm->ceColl.useCompletePtr, !useCompleteBefore);
}

// MEMOP-02: Capture reset covers all nRanks slots (not lsaSize only).
TEST_F(CeInternalMPITest, MemOpSyncCaptureResetCoversAllRanks)
{
    if(!ceSimulatedCaptureSupported())
        GTEST_SKIP() << "Simulated capture requires ROCm >= 6.1 graph support";
    ceSkipIfNvls(ceComm);
    requireMinRanks(2);

    const int nRanks = ceComm->nRanks;
    ceComm->ceColl.useCompletePtr = false;
    ASSERT_EQ(ceWriteUcReadyPatternsOnStream(ceComm, getActiveStream()), hipSuccess);
    MPI_Barrier(MPI_COMM_WORLD);

    CeSimulatedCaptureGuard capture(ceComm);
    ASSERT_EQ(ncclMemOpSync(ceComm, getActiveStream(), nullptr), ncclSuccess);
    ASSERT_EQ(hipStreamSynchronize(getActiveStream()), hipSuccess);

    const auto ready = ceReadUcReadyFlags(ceComm);
    for(int i = 0; i < nRanks; ++i)
        EXPECT_EQ(ready[static_cast<size_t>(i)], 0u) << "rank slot " << i;
}

// ===========================================================================
// CeInternal_DevRuntime – graph-safe window registration during capture
// ===========================================================================

// DEVR-01: Symmetric window registration during HIP capture completes successfully.
TEST_F(CeInternalMPITest, SymBufAllocDuringCaptureRestoresCaptureMode)
{
    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal), hipSuccess);

    RCCLTestHelpers::SymBuf extraSym;
    ASSERT_EQ(RCCLTestHelpers::ncclSymBufAlloc(getActiveCommunicator(), 4096, extraSym),
              ncclSuccess)
        << "window registration during capture should succeed";

    ASSERT_EQ(hipStreamEndCapture(getActiveStream(), &graph), hipSuccess);
    ASSERT_NE(graph, nullptr);
    SCOPE_EXIT((void)hipGraphDestroy(graph));

    hipGraphExec_t graphExec = nullptr;
    ASSERT_EQ(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), hipSuccess);
    SCOPE_EXIT(if(graphExec) (void)hipGraphExecDestroy(graphExec));

    ASSERT_EQ(hipGraphLaunch(graphExec, getActiveStream()), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(getActiveStream()), hipSuccess);
}

// DEVR-02: Comm teardown after capture with CE init must not hang (devr finalize path).
TEST_F(CeInternalMPITest, CommTeardownAfterCaptureWithCeInit)
{
    hipGraph_t graph = nullptr;
    ASSERT_EQ(hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal), hipSuccess);

    RCCLTestHelpers::SymBuf extraSym;
    ASSERT_EQ(RCCLTestHelpers::ncclSymBufAlloc(getActiveCommunicator(), 4096, extraSym),
              ncclSuccess);

    ASSERT_EQ(hipStreamEndCapture(getActiveStream(), &graph), hipSuccess);
    SCOPE_EXIT((void)hipGraphDestroy(graph));

    ASSERT_EQ(ncclCeFinalize(ceComm), ncclSuccess);
    EXPECT_EQ(ceComm->ceColl.baseUCSymReadyPtr, nullptr);
    EXPECT_EQ(ceComm->ceColl.ceSeqNumDev, nullptr);
}

// ===========================================================================
// CeInternal_Launch – ncclCeLaunchBatchOps with zero and real ops
// ===========================================================================

// LAUNCH-01: Launching an empty batch (numOps == 0) succeeds.
TEST_F(CeInternalMPITest, LaunchEmptyBatchSucceeds)
{
    ncclCeBatchOpsParams params{};
    ncclCeCollArgs collArgs{};
    EXPECT_EQ(ncclCeLaunchBatchOps(ceComm, &params, getActiveStream(), &collArgs), ncclSuccess);
    EXPECT_EQ(hipStreamSynchronize(getActiveStream()), hipSuccess);
}

// LAUNCH-02: Launching 4 device-to-device copy ops via ncclCeLaunchBatchOps
//            succeeds and the data is correctly transferred.
TEST_F(CeInternalMPITest, LaunchFourOpsSucceeds)
{
    using namespace RCCLTestGuards;
    constexpr int    kOps   = 4;
    constexpr size_t kBytes = kOps * sizeof(float); // allocation size per src/dst buffer

    // DeviceBufferAutoGuard ensures hipFree is called on all paths, including early ASSERT exits.
    std::vector<DeviceBufferAutoGuard> srcGuards(kOps), dstGuards(kOps);
    for(int i = 0; i < kOps; ++i)
    {
        void* p = nullptr;
        ASSERT_EQ(hipMalloc(&p, kBytes), hipSuccess);
        srcGuards[i].set(p);
        p = nullptr;
        ASSERT_EQ(hipMalloc(&p, kBytes), hipSuccess);
        dstGuards[i].set(p);

        float pattern = static_cast<float>(i + 1);
        ASSERT_EQ(hipMemset(srcGuards[i].get(), 0, kBytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(srcGuards[i].get(), &pattern, sizeof(float), hipMemcpyHostToDevice),
                  hipSuccess);
    }

    ncclCeBatchOpsParams params{};
    ASSERT_EQ(ncclCeInitBatchOpsParams(&params, kOps), ncclSuccess);
    SCOPE_EXIT(ncclCeFreeBatchOpsParams(&params));

    for(int i = 0; i < kOps; ++i)
    {
        params.srcs[i]  = static_cast<float*>(srcGuards[i].get());
        params.dsts[i]  = static_cast<float*>(dstGuards[i].get());
        params.sizes[i] = sizeof(float);
    }
    params.numOps = kOps;

    ncclCeCollArgs collArgs{};
    EXPECT_EQ(ncclCeLaunchBatchOps(ceComm, &params, getActiveStream(), &collArgs), ncclSuccess);
    ASSERT_EQ(hipStreamSynchronize(getActiveStream()), hipSuccess);

    for(int i = 0; i < kOps; ++i)
    {
        float result = 0.0f;
        ASSERT_EQ(hipMemcpy(&result, dstGuards[i].get(), sizeof(float), hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_FLOAT_EQ(result, static_cast<float>(i + 1)) << "op " << i;
    }
    // srcGuards, dstGuards, and params freed automatically on scope exit.
}

// LAUNCH-03: ncclCeLaunchBatchOps on the legacy null/default stream (stream == 0).
//
// HIP-behaviour test (not a defect-fix validation): pins down that CE batch
// dispatch on the legacy null/default stream is functionally correct on AMD.
// Upstream NCCL 2.29.7 added an isLegacyStream fallback to per-op cudaMemcpyAsync
// because cudaMemcpyBatchAsync rejects the legacy null stream on CUDA; on AMD/HIP
// hipMemcpyBatchAsync accepts the null stream, so this passes with or without the
// fallback. Kept as a regression guard for the null-stream contract on HIP.
TEST_F(CeInternalMPITest, LaunchFourOpsNullStreamSucceeds)
{
    using namespace RCCLTestGuards;
    constexpr int    kOps   = 4;
    constexpr size_t kBytes = kOps * sizeof(float); // allocation size per src/dst buffer

    // The legacy null/default stream is the exact trigger for the upstream defect.
    hipStream_t nullStream = static_cast<hipStream_t>(0);

    // DeviceBufferAutoGuard ensures hipFree is called on all paths, including early ASSERT exits.
    std::vector<DeviceBufferAutoGuard> srcGuards(kOps), dstGuards(kOps);
    for(int i = 0; i < kOps; ++i)
    {
        void* p = nullptr;
        ASSERT_EQ(hipMalloc(&p, kBytes), hipSuccess);
        srcGuards[i].set(p);
        p = nullptr;
        ASSERT_EQ(hipMalloc(&p, kBytes), hipSuccess);
        dstGuards[i].set(p);

        float pattern = static_cast<float>(i + 1);
        ASSERT_EQ(hipMemset(srcGuards[i].get(), 0, kBytes), hipSuccess);
        ASSERT_EQ(hipMemcpy(srcGuards[i].get(), &pattern, sizeof(float), hipMemcpyHostToDevice),
                  hipSuccess);
    }

    ncclCeBatchOpsParams params{};
    ASSERT_EQ(ncclCeInitBatchOpsParams(&params, kOps), ncclSuccess);
    SCOPE_EXIT(ncclCeFreeBatchOpsParams(&params));

    for(int i = 0; i < kOps; ++i)
    {
        params.srcs[i]  = static_cast<float*>(srcGuards[i].get());
        params.dsts[i]  = static_cast<float*>(dstGuards[i].get());
        params.sizes[i] = sizeof(float);
    }
    params.numOps = kOps;

    ncclCeCollArgs collArgs{};
    EXPECT_EQ(ncclCeLaunchBatchOps(ceComm, &params, nullStream, &collArgs), ncclSuccess)
        << "ncclCeLaunchBatchOps must succeed on the legacy null stream";
    ASSERT_EQ(hipStreamSynchronize(nullStream), hipSuccess);

    for(int i = 0; i < kOps; ++i)
    {
        float result = 0.0f;
        ASSERT_EQ(hipMemcpy(&result, dstGuards[i].get(), sizeof(float), hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_FLOAT_EQ(result, static_cast<float>(i + 1))
            << "op " << i << " produced wrong data on the null stream";
    }
    // srcGuards, dstGuards, and params freed automatically on scope exit.
}

// LAUNCH-04: Odd intra-batch sync count under capture appends an extra ncclMemOpSync so
//            useCompletePtr returns to its pre-launch value (graph replay safety).
TEST_F(CeInternalMPITest, LaunchBatchOpsEvenSyncCountRestoresUseCompletePtr)
{
    if(!ceSimulatedCaptureSupported())
        GTEST_SKIP() << "Simulated capture requires ROCm >= 6.1 graph support";
    ceSkipIfNvls(ceComm);
    requireMinRanks(2);

    using namespace RCCLTestGuards;
    constexpr int    kOps   = 16; // one mid-loop sync + even-count padding → restored toggle
    constexpr size_t kBytes = sizeof(float);

    std::vector<DeviceBufferAutoGuard> srcGuards(kOps), dstGuards(kOps);
    for(int i = 0; i < kOps; ++i)
    {
        void* p = nullptr;
        ASSERT_EQ(hipMalloc(&p, kBytes), hipSuccess);
        srcGuards[i].set(p);
        p = nullptr;
        ASSERT_EQ(hipMalloc(&p, kBytes), hipSuccess);
        dstGuards[i].set(p);

        const float pattern = static_cast<float>(i + 1);
        ASSERT_EQ(hipMemcpy(srcGuards[i].get(), &pattern, sizeof(float), hipMemcpyHostToDevice),
                  hipSuccess);
    }

    ncclCeBatchOpsParams params{};
    ASSERT_EQ(ncclCeInitBatchOpsParams(&params, kOps), ncclSuccess);
    SCOPE_EXIT(ncclCeFreeBatchOpsParams(&params));

    for(int i = 0; i < kOps; ++i)
    {
        params.srcs[i]  = static_cast<float*>(srcGuards[i].get());
        params.dsts[i]  = static_cast<float*>(dstGuards[i].get());
        params.sizes[i] = kBytes;
    }
    params.numOps         = kOps;
    params.intraBatchSync = true;

    const bool useCompleteBefore = ceComm->ceColl.useCompletePtr;
    CeSimulatedCaptureGuard capture(ceComm);
    ncclCeCollArgs          collArgs{};
    ASSERT_EQ(ncclCeLaunchBatchOps(ceComm, &params, getActiveStream(), &collArgs), ncclSuccess);
    ASSERT_EQ(hipStreamSynchronize(getActiveStream()), hipSuccess);
    EXPECT_EQ(ceComm->ceColl.useCompletePtr, useCompleteBefore)
        << "even sync count (with padding) must restore useCompletePtr for replay";
}

// LAUNCH-05: When padding is not needed the internal sync count is already even.
TEST_F(CeInternalMPITest, LaunchBatchOpsEvenInternalSyncCountRestoresUseCompletePtr)
{
    if(!ceSimulatedCaptureSupported())
        GTEST_SKIP() << "Simulated capture requires ROCm >= 6.1 graph support";
    ceSkipIfNvls(ceComm);
    requireMinRanks(2);

    using namespace RCCLTestGuards;
    constexpr int    kOps   = 24; // two mid-loop syncs, no padding
    constexpr size_t kBytes = sizeof(float);

    std::vector<DeviceBufferAutoGuard> srcGuards(kOps), dstGuards(kOps);
    for(int i = 0; i < kOps; ++i)
    {
        void* p = nullptr;
        ASSERT_EQ(hipMalloc(&p, kBytes), hipSuccess);
        srcGuards[i].set(p);
        p = nullptr;
        ASSERT_EQ(hipMalloc(&p, kBytes), hipSuccess);
        dstGuards[i].set(p);
        const float pattern = static_cast<float>(i + 1);
        ASSERT_EQ(hipMemcpy(srcGuards[i].get(), &pattern, sizeof(float), hipMemcpyHostToDevice),
                  hipSuccess);
    }

    ncclCeBatchOpsParams params{};
    ASSERT_EQ(ncclCeInitBatchOpsParams(&params, kOps), ncclSuccess);
    SCOPE_EXIT(ncclCeFreeBatchOpsParams(&params));

    for(int i = 0; i < kOps; ++i)
    {
        params.srcs[i]  = static_cast<float*>(srcGuards[i].get());
        params.dsts[i]  = static_cast<float*>(dstGuards[i].get());
        params.sizes[i] = kBytes;
    }
    params.numOps         = kOps;
    params.intraBatchSync = true;

    const bool useCompleteBefore = ceComm->ceColl.useCompletePtr;
    CeSimulatedCaptureGuard capture(ceComm);
    ncclCeCollArgs          collArgs{};
    ASSERT_EQ(ncclCeLaunchBatchOps(ceComm, &params, getActiveStream(), &collArgs), ncclSuccess);
    ASSERT_EQ(hipStreamSynchronize(getActiveStream()), hipSuccess);
    EXPECT_EQ(ceComm->ceColl.useCompletePtr, useCompleteBefore);
}

// LAUNCH-06: One mid-loop sync (odd) still gets a padding ncclMemOpSync under capture,
//            restoring useCompletePtr.  numOps=9 → sync at op 8 + padding sync.
TEST_F(CeInternalMPITest, LaunchBatchOpsOddMidLoopSyncCountGetsPaddingRestoresUseCompletePtr)
{
    if(!ceSimulatedCaptureSupported())
        GTEST_SKIP() << "Simulated capture requires ROCm >= 6.1 graph support";
    ceSkipIfNvls(ceComm);
    requireMinRanks(2);

    using namespace RCCLTestGuards;
    constexpr int    kOps   = 9; // one mid-loop sync; padding makes total sync count even
    constexpr size_t kBytes = sizeof(float);

    std::vector<DeviceBufferAutoGuard> srcGuards(kOps), dstGuards(kOps);
    for(int i = 0; i < kOps; ++i)
    {
        void* p = nullptr;
        ASSERT_EQ(hipMalloc(&p, kBytes), hipSuccess);
        srcGuards[i].set(p);
        p = nullptr;
        ASSERT_EQ(hipMalloc(&p, kBytes), hipSuccess);
        dstGuards[i].set(p);
        const float pattern = static_cast<float>(i + 1);
        ASSERT_EQ(hipMemcpy(srcGuards[i].get(), &pattern, sizeof(float), hipMemcpyHostToDevice),
                  hipSuccess);
    }

    ncclCeBatchOpsParams params{};
    ASSERT_EQ(ncclCeInitBatchOpsParams(&params, kOps), ncclSuccess);
    SCOPE_EXIT(ncclCeFreeBatchOpsParams(&params));

    for(int i = 0; i < kOps; ++i)
    {
        params.srcs[i]  = static_cast<float*>(srcGuards[i].get());
        params.dsts[i]  = static_cast<float*>(dstGuards[i].get());
        params.sizes[i] = kBytes;
    }
    params.numOps         = kOps;
    params.intraBatchSync = true;

    const bool useCompleteBefore = ceComm->ceColl.useCompletePtr;
    CeSimulatedCaptureGuard capture(ceComm);
    ncclCeCollArgs          collArgs{};
    ASSERT_EQ(ncclCeLaunchBatchOps(ceComm, &params, getActiveStream(), &collArgs), ncclSuccess);
    ASSERT_EQ(hipStreamSynchronize(getActiveStream()), hipSuccess);
    EXPECT_EQ(ceComm->ceColl.useCompletePtr, useCompleteBefore)
        << "odd mid-loop sync count must be padded to restore useCompletePtr for replay";
}

// ===========================================================================
// CeInternal_Neg – ncclCeImplemented logic (no CE hardware required)
// ===========================================================================

// NEG-01: ncclCeImplemented returns false for collectives CE does not handle.
//
// NOTE: ncclFuncAllReduce was moved out of this negative list in PR #8443
// ("Users/galaband/rccl ce ar no pipeline"), which added CE AllReduce support
// (ce_coll.cc's ncclCeImplemented() switch now returns true for
// ncclFuncAllReduce). That PR updated the switch statement but never updated
// this test, leaving a stale assertion that CE_Internal_Neg has failed on
// ever since (verified identical on rccl/gfx1250:f43c171 and PR #8945 tip).
// AllReduce's positive expectation now lives in NEG-02 below, alongside the
// driver-gated checks for the other CE-supported collectives.
TEST(CeInternalNeg, CeImplementedReturnsFalseForUnsupported)
{
    EXPECT_FALSE(ncclCeImplemented(ncclFuncBroadcast,    ncclDevSum, ncclFloat32));
    EXPECT_FALSE(ncclCeImplemented(ncclFuncReduceScatter, ncclDevSum, ncclFloat32));
}

// NEG-02: On ROCm 7.12+ or 7.0.2.x, ncclCeImplemented returns true for
//         the five supported CE collectives.
TEST(CeInternalNeg, CeImplementedReturnsTrueOnSupportedDriver)
{
    if(!isCeDriverSupported())
        GTEST_SKIP() << "CE not available (binary not compiled with CE_BATCH_ASYNC_SUPPORTED)";
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE not supported on this runtime driver "
                        "(need ROCm >= 7.12 or 7.0.2.x backport [70051831, 70060000))";

    EXPECT_TRUE(ncclCeImplemented(ncclFuncAllGather, ncclDevSum, ncclFloat32));
    EXPECT_TRUE(ncclCeImplemented(ncclFuncAlltoAll,  ncclDevSum, ncclFloat32));
    EXPECT_TRUE(ncclCeImplemented(ncclFuncAlltoAllv, ncclDevSum, ncclFloat32));
    EXPECT_TRUE(ncclCeImplemented(ncclFuncScatter,   ncclDevSum, ncclFloat32));
    EXPECT_TRUE(ncclCeImplemented(ncclFuncGather,    ncclDevSum, ncclFloat32));
    EXPECT_TRUE(ncclCeImplemented(ncclFuncAllReduce, ncclDevSum, ncclFloat32));
}

// NEG-03: ncclCeAlltoAllv rejects missing size metadata before launching transfers.
TEST_F(CeInternalMPITest, AlltoAllvMissingSizesReturnsInvalidUsage)
{
    ncclCeCollArgs args{};
    args.sizes = nullptr;
    EXPECT_EQ(ncclCeAlltoAllv(ceComm, &args, getActiveStream()), ncclInvalidUsage);
}

// ===========================================================================
// CeInternal_FaultInj – fault injection (ENABLE_FAULT_INJECTION only)
// Build: cmake -DENABLE_FAULT_INJECTION=ON; Run: --gtest_filter=CeInternalFaultInj*
// ===========================================================================

#ifdef ENABLE_FAULT_INJECTION
#include "ce_fault_inject.h"

// Inherits CeInternalMPITest; clears all CE faults in TearDown.
class CeFaultInjTest : public CeInternalMPITest
{
protected:
    void TearDown() override
    {
        if(ceComm != nullptr)
            ncclCeFaultClear(ceComm);
        CeInternalMPITest::TearDown();
    }
};

// FAULT-01: CE_FAULT_SYNC_PREP makes ncclPrepUCSync return ncclSystemError.
//           After clearing, the same call must succeed.
TEST_F(CeFaultInjTest, SyncPrepErrorPropagates)
{
    requireMinRanks(2);
    auto   batch = makePrepSyncBatch();
    size_t opIdx = 0;

    ASSERT_EQ(ncclCeFaultSet(ceComm, CE_FAULT_SYNC_PREP), ncclSuccess);
    EXPECT_EQ(ncclPrepUCSync(ceComm, false, batch.data(), &opIdx, getActiveStream()),
              ncclSystemError)
        << "Expected ncclSystemError when CE_FAULT_SYNC_PREP is armed";

    ASSERT_EQ(ncclCeFaultClear(ceComm), ncclSuccess);
    opIdx = 0;
    EXPECT_EQ(ncclPrepUCSync(ceComm, false, batch.data(), &opIdx, getActiveStream()),
              ncclSuccess)
        << "Expected ncclSuccess after fault cleared";
}

// FAULT-02: CE_FAULT_LAUNCH_OP makes ncclCeLaunchBatchOps return
//           ncclSystemError even for a non-empty batch.
//           After clearing, the same call must succeed.
TEST_F(CeFaultInjTest, LaunchBatchOpsErrorPropagates)
{
    using namespace RCCLTestGuards;
    constexpr int    kOps  = 2;
    constexpr size_t kBytes = sizeof(float);

    // DeviceBufferAutoGuard ensures hipFree is called on all paths, including early ASSERT exits.
    void* srcPtr = nullptr;
    ASSERT_EQ(hipMalloc(&srcPtr, kBytes), hipSuccess);
    DeviceBufferAutoGuard srcGuard(srcPtr);

    void* dstPtr = nullptr;
    ASSERT_EQ(hipMalloc(&dstPtr, kBytes), hipSuccess);
    DeviceBufferAutoGuard dstGuard(dstPtr);

    ncclCeBatchOpsParams params{};
    ASSERT_EQ(ncclCeInitBatchOpsParams(&params, kOps), ncclSuccess);
    SCOPE_EXIT(ncclCeFreeBatchOpsParams(&params));

    for(int i = 0; i < kOps; ++i)
    {
        params.srcs[i]  = static_cast<float*>(srcPtr);
        params.dsts[i]  = static_cast<float*>(dstPtr);
        params.sizes[i] = kBytes;
    }
    params.numOps = kOps;

    ncclCeCollArgs collArgs{};
    ASSERT_EQ(ncclCeFaultSet(ceComm, CE_FAULT_LAUNCH_OP), ncclSuccess);
    EXPECT_EQ(ncclCeLaunchBatchOps(ceComm, &params, getActiveStream(), &collArgs),
              ncclSystemError)
        << "Expected ncclSystemError when CE_FAULT_LAUNCH_OP is armed";

    ASSERT_EQ(ncclCeFaultClear(ceComm), ncclSuccess);
    EXPECT_EQ(ncclCeLaunchBatchOps(ceComm, &params, getActiveStream(), &collArgs),
              ncclSuccess)
        << "Expected ncclSuccess after fault cleared";
    EXPECT_EQ(hipStreamSynchronize(getActiveStream()), hipSuccess);
    // srcGuard, dstGuard, and params freed automatically on scope exit.
}

// FAULT-03: CE_FAULT_INIT makes ncclCeInit return ncclSystemError and leaves
//           CE un-initialized (baseUCSymReadyPtr remains null after the call).
TEST_F(CeFaultInjTest, InitErrorPreventsSetup)
{
    ASSERT_EQ(ncclCeFinalize(ceComm), ncclSuccess);
    ASSERT_EQ(ceComm->ceColl.baseUCSymReadyPtr, nullptr);

    ASSERT_EQ(ncclCeFaultSet(ceComm, CE_FAULT_INIT), ncclSuccess);
    EXPECT_EQ(ncclCeInit(ceComm), ncclSystemError)
        << "Expected ncclSystemError when CE_FAULT_INIT is armed";
    EXPECT_EQ(ceComm->ceColl.baseUCSymReadyPtr, nullptr)
        << "Failed init must not set baseUCSymReadyPtr";

    ASSERT_EQ(ncclCeFaultClear(ceComm), ncclSuccess);
    EXPECT_EQ(ncclCeInit(ceComm), ncclSuccess)
        << "Expected ncclSuccess after fault cleared";
    EXPECT_NE(ceComm->ceColl.baseUCSymReadyPtr, nullptr)
        << "Successful re-init should set baseUCSymReadyPtr";
}

// FAULT-04: Multiple faults can be armed simultaneously; each targeted
//           function returns ncclSystemError; ncclCeFaultClear restores all.
TEST_F(CeFaultInjTest, MultipleFaultsArmedAndCleared)
{
    requireMinRanks(2);

    ASSERT_EQ(ncclCeFaultSet(ceComm, CE_FAULT_SYNC_PREP | CE_FAULT_LAUNCH_OP),
              ncclSuccess);
    EXPECT_EQ(ncclCeFaultGet(ceComm), CE_FAULT_SYNC_PREP | CE_FAULT_LAUNCH_OP)
        << "Both fault bits should be visible via ncclCeFaultGet";

    auto   batch = makePrepSyncBatch();
    size_t opIdx = 0;
    EXPECT_EQ(ncclPrepUCSync(ceComm, false, batch.data(), &opIdx, getActiveStream()),
              ncclSystemError);

    ncclCeBatchOpsParams emptyParams{};
    emptyParams.numOps = 1; // non-zero to bypass the early-out
    ncclCeCollArgs collArgs{};
    EXPECT_EQ(ncclCeLaunchBatchOps(ceComm, &emptyParams, getActiveStream(), &collArgs),
              ncclSystemError);

    ASSERT_EQ(ncclCeFaultClear(ceComm), ncclSuccess);
    EXPECT_EQ(ncclCeFaultGet(ceComm), 0u) << "All faults should be cleared";

    opIdx = 0;
    EXPECT_EQ(ncclPrepUCSync(ceComm, false, batch.data(), &opIdx, getActiveStream()),
              ncclSuccess);
}

#endif // ENABLE_FAULT_INJECTION

#endif // MPI_TESTS_ENABLED
