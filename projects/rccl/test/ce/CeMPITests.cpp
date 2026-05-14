/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// End-to-end MPI tests for CE collectives via the public RCCL API (AllGather, AlltoAll, Scatter, Gather, Fallback, Stress).

#include "CeTestHelpers.hpp"
#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "SymmetricBufferHelpers.hpp"
#include "TestChecks.hpp"
#include "rccl/rccl.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <string>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestHelpers;

namespace CeMPITestConstants
{
constexpr size_t kSmallCount      = 4096;   // elements per rank for fast tests
constexpr size_t kMediumCount     = 65536;  // elements per rank for larger tests
constexpr int    kMinRanks2       = 2;
constexpr int    kMinRanks4       = 4;
constexpr int    kMinRanks8       = 8;
constexpr int    kStressIters     = 20;     // back-to-back iterations for stress test
constexpr int    kInterleavedIters = 10;    // iterations for CE+SM interleaved stress test
} // namespace CeMPITestConstants

using namespace CeMPITestConstants;

// Base fixture for all CE MPI tests; captures NCCL INFO per rank for log-marker assertions.
// Tests always run: on CE-capable systems they assert the CE path; on others they assert
// the SM fallback path.  isCeExpected() is the single gate for which assertion to make.
class CeMPITest : public MPITestBase
{
protected:
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugSubsysGuard_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;

    void SetUp() override
    {
        MPITestBase::SetUp();
        // Log capture is always enabled so that assertCEPathTaken / assertCEPathNotTaken
        // can inspect the NCCL debug output regardless of whether CE is active.
        debugGuard_       = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG",        "INFO");
        debugSubsysGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG_SUBSYS", "ALL");
        logCtx_ = std::make_unique<MPIHelpers::TestLogAssertionContext>(
            MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));
    }

    void TearDown() override
    {
        // Destroy communicator before logCtx_ so its log lines flush to the debug file first.
        MPITestBase::TearDown();

        logCtx_.reset();
        debugSubsysGuard_.reset();
        debugGuard_.reset();
    }

    // Returns true when all CE prerequisites are met (driver + env vars).
    // Delegates to isCeDispatchConfigured() in CeTestHelpers.hpp — the single
    // source of truth shared with CeInternalMPITests.cpp.
    bool isCeExpected() const { return isCeDispatchConfigured(); }

    // CE log markers: "Init CE" = CE initialised; "CE: rank" = CE path taken for a collective.
    enum class CeLogStatus { Taken, InitOnlyNoOp, NotInitialized };

    static CeLogStatus checkCeLog(const std::string& log)
    {
        if(log.find("Init CE") == std::string::npos)
            return CeLogStatus::NotInitialized;
        if(log.find("CE: rank") != std::string::npos)
            return CeLogStatus::Taken;
        return CeLogStatus::InitOnlyNoOp;
    }

    // Merge debug file + per-rank stderr (covers ncclDebugInit race on first test).
    std::string readAllLogs() const
    {
        return logCtx_->readNcclDebugLog() + logCtx_->readPerRankStderrLog();
    }

    // Assert the CE dispatch path was taken, or — when CE is not expected on this
    // system/configuration — assert the SM fallback path and print an info line.
    void assertCEPathTaken(const char* context)
    {
        const std::string log    = readAllLogs();
        CeLogStatus       status = checkCeLog(log);

        if(isCeExpected())
        {
            EXPECT_EQ(status, CeLogStatus::Taken)
                << context
                << ": \"CE: rank\" absent from NCCL log"
                   " (NCCL_CTA_POLICY=2 set but CE dispatch path was not taken)";
            if(status == CeLogStatus::Taken)
                TEST_INFO("%s: assertion passed — CE dispatch path taken", context);
        }
        else
        {
            EXPECT_EQ(log.find("CE: rank"), std::string::npos)
                << context
                << ": \"CE: rank\" found in NCCL log but CE was not expected"
                   " (driver or env-var prerequisites not met)";
            TEST_INFO("%s: assertion passed — SM fallback path (CE not available/configured)",
                      context);
        }
    }

    // Root-only variant: only root emits "CE: rank" (non-root has numOps=0 in Scatter/Gather).
    void assertCEPathTakenOnRoot(int rootRank, int myRank, const char* context)
    {
        const std::string log    = readAllLogs();
        CeLogStatus       status = checkCeLog(log);

        if(isCeExpected())
        {
            if(myRank == rootRank)
            {
                EXPECT_EQ(status, CeLogStatus::Taken)
                    << context
                    << ": \"CE: rank\" absent from NCCL log on root rank " << myRank
                    << " (NCCL_CTA_POLICY=2 set but CE dispatch path was not taken)";
                if(status == CeLogStatus::Taken)
                    TEST_INFO("%s: assertion passed — root rank %d CE dispatch path taken",
                              context, myRank);
            }
            else
            {
                TEST_INFO("%s: rank %d non-root — no CE ops expected (correct)", context, myRank);
            }
        }
        else
        {
            EXPECT_EQ(log.find("CE: rank"), std::string::npos)
                << context
                << ": \"CE: rank\" found in NCCL log but CE was not expected";
            TEST_INFO("%s: rank %d assertion passed — SM fallback path (CE not available/configured)",
                      context, myRank);
        }
    }

    // Assert CE was explicitly NOT taken.  Prints an info line in all cases.
    void assertCEPathNotTaken(const char* context)
    {
        const std::string log = readAllLogs();
        EXPECT_EQ(log.find("CE: rank"), std::string::npos)
            << context << ": \"CE: rank\" found in NCCL log but was not expected";
        TEST_INFO("%s: assertion passed — CE path not taken (expected for this test)", context);
    }

    // Assert the specific CE batch path sub-variant that was (or was not) taken.
    //
    // When CE is expected:
    //   withIntraBatchSync=true  → asserts "Batch path with intraBatchSync"
    //   withIntraBatchSync=false → asserts "Batch path without intraBatchSync"
    // When CE is not expected the check is skipped (SM fallback emits neither line).
    //
    // Call after assertCEPathTaken() so the "CE: rank" assertion already passed.
    void assertCEBatchPath(bool withIntraBatchSync, const char* context)
    {
        if(!isCeExpected())
            return;

        const std::string needle = withIntraBatchSync
                                       ? "Batch path with intraBatchSync"
                                       : "Batch path without intraBatchSync";
        const std::string log = readAllLogs();
        EXPECT_NE(log.find(needle), std::string::npos)
            << context << ": expected \"" << needle << "\" not found in NCCL log";
        TEST_INFO("%s: batch path assertion passed — %s", context, needle.c_str());
    }

    // Root-only variant: only the root rank emits the batch path log line
    // (non-root ranks have numOps=0 for Scatter/Gather and emit nothing).
    void assertCEBatchPathOnRoot(bool withIntraBatchSync,
                                 int rootRank, int myRank,
                                 const char* context)
    {
        if(!isCeExpected() || myRank != rootRank)
            return;
        assertCEBatchPath(withIntraBatchSync, context);
    }

    // Data helpers: fill/verify with float(rank+1) per-rank or block-index pattern.
    // Thin fixture wrappers; the underlying logic lives in CeTestHelpers.hpp so that
    // CeInternalMPITests.cpp can call the same helpers without going through this fixture.

    void fillRankScalar(void* buf, size_t nElem, int rank)
    {
        ASSERT_EQ(hipSuccess, ceFillRankScalarFloat(buf, nElem, rank));
    }

    void fillBlockPattern(void* buf, size_t totalElem, size_t elemsPerBlock)
    {
        ASSERT_EQ(
            hipSuccess,
            initializeBufferWithPattern<float>(buf, totalElem, [elemsPerBlock](size_t i) {
                return static_cast<float>(i / elemsPerBlock + 1);
            }));
    }

    bool verifyBlockPattern(const void* buf, size_t totalElem, size_t elemsPerBlock)
    {
        return ceVerifyBlockPatternFloat(buf, totalElem, elemsPerBlock);
    }

    bool verifyRankScalar(const void* buf, size_t nElem, int rank)
    {
        return verifyBufferData<float>(buf, nElem,
                                       [rank](size_t) { return static_cast<float>(rank + 1); });
    }

    // Symmetric buffer wrapper bound to the active communicator; call after createTestCommunicator().
    using SymBuf = RCCLTestHelpers::SymBuf;

    ncclResult_t allocSymBuf(size_t bytes, SymBuf& sb)
    {
        return RCCLTestHelpers::ncclSymBufAlloc(getActiveCommunicator(), bytes, sb);
    }
};

// ===========================================================================
// CeMPI_AllGather – ncclAllGather CE correctness + log verification
// ===========================================================================

class CeMPI_AllGather : public CeMPITest
{
protected:
    void runAllGather(int minRanks, size_t count, const char* testId)
    {
        if(!validateTestPrerequisites(minRanks))
            GTEST_SKIP() << "Need >= " << minRanks << " MPI ranks";

        ASSERT_EQ(ncclSuccess, createTestCommunicator());

        int rank{}, nRanks{};
        ncclCommUserRank(getActiveCommunicator(), &rank);
        ncclCommCount(getActiveCommunicator(), &nRanks);

        SymBuf sendSym, recvSym;
        ASSERT_EQ(ncclSuccess, allocSymBuf(count * sizeof(float), sendSym));
        ASSERT_EQ(ncclSuccess, allocSymBuf(count * nRanks * sizeof(float), recvSym));

        fillRankScalar(sendSym.ptr, count, rank);

        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(sendSym.ptr, recvSym.ptr, count, ncclFloat32,
                                getActiveCommunicator(), getActiveStream()));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        ASSERT_TRUE(verifyBlockPattern(recvSym.ptr, count * nRanks, count))
            << "Rank " << rank << ": AllGather data verification failed";

        assertCEPathTaken(testId);
        // numOps = nRanks (one copy per rank including self); chunkBytes = count * sizeof(float)
        assertCEBatchPath(ceExpectIntraBatchSync(nRanks, count * sizeof(float)), testId);
    }
};

// CE-MPI-AG-01: 2 ranks, small buffers.
TEST_F(CeMPI_AllGather, TwoRanks)    { runAllGather(kMinRanks2, kSmallCount,  "CeMPI_AllGather/TwoRanks"); }
// CE-MPI-AG-02: 4 ranks, medium buffers.
TEST_F(CeMPI_AllGather, FourRanks)   { runAllGather(kMinRanks4, kMediumCount, "CeMPI_AllGather/FourRanks"); }
// CE-MPI-AG-03: 8 ranks (default production topology), medium buffers.
TEST_F(CeMPI_AllGather, EightRanks)  { runAllGather(kMinRanks8, kMediumCount, "CeMPI_AllGather/EightRanks"); }
// CE-MPI-AG-04: Edge case — single element per rank.
TEST_F(CeMPI_AllGather, SingleElement) { runAllGather(kMinRanks2, 1,          "CeMPI_AllGather/SingleElement"); }

// ===========================================================================
// CeMPI_AlltoAll – ncclAlltoAll CE correctness + log verification
// ===========================================================================

class CeMPI_AlltoAll : public CeMPITest
{
protected:
    // count is per-rank-per-destination; total send/recv buffer = count * nRanks.
    void runAlltoAll(int minRanks, size_t count, const char* testId)
    {
        if(!validateTestPrerequisites(minRanks))
            GTEST_SKIP() << "Need >= " << minRanks << " MPI ranks";

        ASSERT_EQ(ncclSuccess, createTestCommunicator());

        int rank{}, nRanks{};
        ncclCommUserRank(getActiveCommunicator(), &rank);
        ncclCommCount(getActiveCommunicator(), &nRanks);

        const size_t totalElem = count * static_cast<size_t>(nRanks);

        SymBuf sendSym, recvSym;
        ASSERT_EQ(ncclSuccess, allocSymBuf(totalElem * sizeof(float), sendSym));
        ASSERT_EQ(ncclSuccess, allocSymBuf(totalElem * sizeof(float), recvSym));

        fillRankScalar(sendSym.ptr, totalElem, rank);

        ASSERT_EQ(ncclSuccess,
                  ncclAlltoAll(sendSym.ptr, recvSym.ptr, count, ncclFloat32,
                               getActiveCommunicator(), getActiveStream()));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        ASSERT_TRUE(verifyBlockPattern(recvSym.ptr, totalElem, count))
            << "Rank " << rank << ": AlltoAll data verification failed";

        assertCEPathTaken(testId);
        // numOps = nRanks (one per destination rank); chunkBytes = count * sizeof(float)
        assertCEBatchPath(ceExpectIntraBatchSync(nRanks, count * sizeof(float)), testId);
    }
};

// CE-MPI-A2A-01: 2 ranks, small buffers.
TEST_F(CeMPI_AlltoAll, TwoRanks)   { runAlltoAll(kMinRanks2, kSmallCount,  "CeMPI_AlltoAll/TwoRanks"); }
// CE-MPI-A2A-02: 4 ranks, medium buffers.
TEST_F(CeMPI_AlltoAll, FourRanks)  { runAlltoAll(kMinRanks4, kMediumCount, "CeMPI_AlltoAll/FourRanks"); }
// CE-MPI-A2A-03: 8 ranks (default production topology), medium buffers.
TEST_F(CeMPI_AlltoAll, EightRanks) { runAlltoAll(kMinRanks8, kMediumCount, "CeMPI_AlltoAll/EightRanks"); }
// CE-MPI-A2A-04: Odd rank count (3) — non-power-of-two op layout vs 2/4/8 ranks.
TEST_F(CeMPI_AlltoAll, ThreeRanks) { runAlltoAll(3,          kSmallCount,  "CeMPI_AlltoAll/ThreeRanks"); }

// ===========================================================================
// CeMPI_Scatter – ncclScatter CE correctness + log verification
// ===========================================================================

class CeMPI_Scatter : public CeMPITest
{
protected:
    // Root sends block r to rank r; non-root send bufs are registered but ignored.
    void runScatter(int minRanks, size_t count, int root, const char* testId)
    {
        if(!validateTestPrerequisites(minRanks))
            GTEST_SKIP() << "Need >= " << minRanks << " MPI ranks";

        ASSERT_EQ(ncclSuccess, createTestCommunicator());

        int rank{}, nRanks{};
        ncclCommUserRank(getActiveCommunicator(), &rank);
        ncclCommCount(getActiveCommunicator(), &nRanks);

        SymBuf sendSym, recvSym;
        ASSERT_EQ(ncclSuccess,
                  allocSymBuf(count * static_cast<size_t>(nRanks) * sizeof(float), sendSym));
        ASSERT_EQ(ncclSuccess, allocSymBuf(count * sizeof(float), recvSym));

        if(rank == root)
            fillBlockPattern(sendSym.ptr, count * static_cast<size_t>(nRanks), count);

        ASSERT_EQ(ncclSuccess,
                  ncclScatter(sendSym.ptr, recvSym.ptr, count, ncclFloat32, root,
                              getActiveCommunicator(), getActiveStream()));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        ASSERT_TRUE(verifyRankScalar(recvSym.ptr, count, rank))
            << "Rank " << rank << ": Scatter data verification failed";

        assertCEPathTakenOnRoot(root, rank, testId);
        // Root sends nRanks chunks; chunkBytes = count * sizeof(float)
        assertCEBatchPathOnRoot(ceExpectIntraBatchSync(nRanks, count * sizeof(float)),
                                root, rank, testId);
    }
};

// CE-MPI-SCT-01: 4 ranks, root = 0.
TEST_F(CeMPI_Scatter, FourRanksRoot0)  { runScatter(kMinRanks4, kSmallCount, 0, "CeMPI_Scatter/FourRanksRoot0"); }
// CE-MPI-SCT-02: 4 ranks, non-zero root (root = 1).
TEST_F(CeMPI_Scatter, FourRanksRoot1)  { runScatter(kMinRanks4, kSmallCount, 1, "CeMPI_Scatter/FourRanksRoot1"); }
// CE-MPI-SCT-03: 8 ranks (default production topology), root = 0.
TEST_F(CeMPI_Scatter, EightRanksRoot0) { runScatter(kMinRanks8, kSmallCount, 0, "CeMPI_Scatter/EightRanksRoot0"); }

// ===========================================================================
// CeMPI_Gather – ncclGather CE correctness + log verification
// ===========================================================================

class CeMPI_Gather : public CeMPITest
{
protected:
    // All ranks send; root gathers into block-pattern recvbuf; non-root recv bufs registered.
    void runGather(int minRanks, size_t count, int root, const char* testId)
    {
        if(!validateTestPrerequisites(minRanks))
            GTEST_SKIP() << "Need >= " << minRanks << " MPI ranks";

        ASSERT_EQ(ncclSuccess, createTestCommunicator());

        int rank{}, nRanks{};
        ncclCommUserRank(getActiveCommunicator(), &rank);
        ncclCommCount(getActiveCommunicator(), &nRanks);

        SymBuf sendSym, recvSym;
        ASSERT_EQ(ncclSuccess, allocSymBuf(count * sizeof(float), sendSym));
        ASSERT_EQ(ncclSuccess,
                  allocSymBuf(count * static_cast<size_t>(nRanks) * sizeof(float), recvSym));

        fillRankScalar(sendSym.ptr, count, rank);

        ASSERT_EQ(ncclSuccess,
                  ncclGather(sendSym.ptr, recvSym.ptr, count, ncclFloat32, root,
                             getActiveCommunicator(), getActiveStream()));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        if(rank == root)
        {
            ASSERT_TRUE(verifyBlockPattern(recvSym.ptr, count * static_cast<size_t>(nRanks), count))
                << "Rank " << rank << " (root): Gather data verification failed";
        }

        assertCEPathTakenOnRoot(root, rank, testId);
        // Root receives nRanks chunks; chunkBytes = count * sizeof(float)
        assertCEBatchPathOnRoot(ceExpectIntraBatchSync(nRanks, count * sizeof(float)),
                                root, rank, testId);
    }
};

// CE-MPI-GTH-01: 4 ranks, root = 0.
TEST_F(CeMPI_Gather, FourRanksRoot0)  { runGather(kMinRanks4, kSmallCount, 0, "CeMPI_Gather/FourRanksRoot0"); }
// CE-MPI-GTH-02: 4 ranks, non-zero root (root = 1).
TEST_F(CeMPI_Gather, FourRanksRoot1)  { runGather(kMinRanks4, kSmallCount, 1, "CeMPI_Gather/FourRanksRoot1"); }
// CE-MPI-GTH-03: 8 ranks (default production topology), root = 0.
TEST_F(CeMPI_Gather, EightRanksRoot0) { runGather(kMinRanks8, kSmallCount, 0, "CeMPI_Gather/EightRanksRoot0"); }

// ===========================================================================
// CeMPI_Fallback – CE not taken for AllReduce; SM path when CE env is off
// ===========================================================================

class CeMPI_Fallback : public CeMPITest
{};

// CE-MPI-FALLBACK-01: AllReduce never uses CE (not a CE collective).
// Plain hipMalloc avoids the symmetric-SM kernel path (absent with GENERATE_SYM_KERNELS=OFF).
TEST_F(CeMPI_Fallback, AllReduceNeverUsesCE)
{
    using namespace RCCLTestGuards;
    if(!validateTestPrerequisites(kMinRanks2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t count = kSmallCount;
    const size_t bytes = count * sizeof(float);

    // Plain hipMalloc (not symmetric) is intentional: avoids the sym-SM kernel path.
    // DeviceBufferAutoGuard ensures hipFree on all exit paths including early ASSERTs.
    void* sendBuf = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
    DeviceBufferAutoGuard sendGuard(sendBuf);

    void* recvBuf = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
    DeviceBufferAutoGuard recvGuard(recvBuf);

    fillRankScalar(sendBuf, count, rank);

    ASSERT_EQ(ncclSuccess,
              ncclAllReduce(sendBuf, recvBuf, count, ncclFloat32, ncclSum,
                            getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    const float expected = static_cast<float>(nRanks * (nRanks + 1) / 2);
    ASSERT_TRUE(verifyBufferData<float>(recvBuf, count,
                                        [expected](size_t) { return expected; }))
        << "Rank " << rank << ": AllReduce data verification failed";

    assertCEPathNotTaken("CeMPI_Fallback/AllReduceNeverUsesCE");
}

// CE-MPI-FALLBACK-02: AllGather with plain hipMalloc (no symmetric window) falls back to SM.
// NCCL_CTA_POLICY is process-cached; bypassing CE via unregistered buffers is the correct approach.
TEST_F(CeMPI_Fallback, AllGatherFallbackToSMWhenCEDisabled)
{
    using namespace RCCLTestGuards;
    if(!validateTestPrerequisites(kMinRanks2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t count = kSmallCount;
    const size_t bytes = count * sizeof(float);

    // Plain hipMalloc (not symmetric) is intentional: ncclDevrFindWindow returns NULL
    // → CE dispatch condition false.  Guards ensure cleanup on all exit paths.
    void* sendBuf = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
    DeviceBufferAutoGuard sendGuard(sendBuf);

    void* recvBuf = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf, bytes * static_cast<size_t>(nRanks)));
    DeviceBufferAutoGuard recvGuard(recvBuf);

    fillRankScalar(sendBuf, count, rank);

    ASSERT_EQ(ncclSuccess,
              ncclAllGather(sendBuf, recvBuf, count, ncclFloat32,
                            getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    ASSERT_TRUE(verifyBlockPattern(recvBuf, count * static_cast<size_t>(nRanks), count))
        << "Rank " << rank << ": AllGather (SM path) data verification failed";

    assertCEPathNotTaken("CeMPI_Fallback/AllGatherFallbackToSMWhenCEDisabled");
}

// CE-MPI-FALLBACK-03: Only send registered; recvWin == NULL → CE dispatch skipped, SM ring used.
TEST_F(CeMPI_Fallback, AllGatherRecvNotRegisteredFallsBackToSM)
{
    using namespace RCCLTestGuards;
    if(!validateTestPrerequisites(kMinRanks2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t count = kSmallCount;
    const size_t bytes = count * sizeof(float);

    // Send is a symmetric window; recv is plain hipMalloc (intentional) → CE condition fails.
    SymBuf sendSym;
    ASSERT_EQ(ncclSuccess, allocSymBuf(bytes, sendSym));

    void* recvBuf = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf, bytes * static_cast<size_t>(nRanks)));
    DeviceBufferAutoGuard recvGuard(recvBuf);

    fillRankScalar(sendSym.ptr, count, rank);

    ASSERT_EQ(ncclSuccess,
              ncclAllGather(sendSym.ptr, recvBuf, count, ncclFloat32,
                            getActiveCommunicator(), getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    ASSERT_TRUE(verifyBlockPattern(recvBuf, count * static_cast<size_t>(nRanks), count))
        << "Rank " << rank << ": AllGather (partial-reg SM fallback) data verification failed";

    assertCEPathNotTaken("CeMPI_Fallback/AllGatherRecvNotRegisteredFallsBackToSM");
}

// ===========================================================================
// CeMPI_Stress – back-to-back repetitions stress test
// ===========================================================================

class CeMPI_Stress : public CeMPITest
{};

// CE-MPI-STRESS-01: 20 consecutive AllGather calls; validates ceSeqNum never desyncs.
TEST_F(CeMPI_Stress, AllGatherBackToBack20x)
{
    if(!validateTestPrerequisites(kMinRanks2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t count = kSmallCount;

    SymBuf sendSym, recvSym;
    ASSERT_EQ(ncclSuccess, allocSymBuf(count * sizeof(float), sendSym));
    ASSERT_EQ(ncclSuccess, allocSymBuf(count * nRanks * sizeof(float), recvSym));
    void* sendBuf = sendSym.ptr;
    void* recvBuf = recvSym.ptr;

    fillRankScalar(sendBuf, count, rank);

    for(int iter = 0; iter < kStressIters; ++iter)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(sendBuf, recvBuf, count, ncclFloat32,
                                getActiveCommunicator(), getActiveStream()))
            << "Iteration " << iter << " failed";
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()))
            << "Stream sync failed at iteration " << iter;

        ASSERT_TRUE(verifyBlockPattern(recvBuf, count * nRanks, count))
            << "Rank " << rank << ": AllGather data wrong at iteration " << iter;
    }

    assertCEPathTaken("CeMPI_Stress/AllGatherBackToBack20x");
}

// CE-MPI-STRESS-02: Interleave CE AllGather and SM AllReduce on same comm.
// Validates that CE DMA work and SM kernel work ordered on the same comm do
// not corrupt each other's state.
TEST_F(CeMPI_Stress, InterleavedCEAllGatherAndSMAllReduce)
{
    if(!validateTestPrerequisites(kMinRanks2))
        GTEST_SKIP() << "Need >= 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t count = kSmallCount;

    SymBuf agSendSym, agRecvSym;
    ASSERT_EQ(ncclSuccess, allocSymBuf(count * sizeof(float), agSendSym));
    ASSERT_EQ(ncclSuccess, allocSymBuf(count * nRanks * sizeof(float), agRecvSym));
    void* agSend = agSendSym.ptr;
    void* agRecv = agRecvSym.ptr;

    // AllReduce uses plain hipMalloc (intentional): symmetric buffers would route to a
    // sym-SM kernel absent with GENERATE_SYM_KERNELS=OFF → ncclUnhandledCudaError.
    // DeviceBufferAutoGuard ensures cleanup on all exit paths.
    void* arSend = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&arSend, count * sizeof(float)));
    RCCLTestGuards::DeviceBufferAutoGuard arSendGuard(arSend);

    void* arRecv = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&arRecv, count * sizeof(float)));
    RCCLTestGuards::DeviceBufferAutoGuard arRecvGuard(arRecv);

    fillRankScalar(agSend, count, rank);
    fillRankScalar(arSend, count, rank);

    const float arExpected = static_cast<float>(nRanks * (nRanks + 1) / 2);

    for(int iter = 0; iter < kInterleavedIters; ++iter)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclAllGather(agSend, agRecv, count, ncclFloat32,
                                getActiveCommunicator(), getActiveStream()))
            << "AllGather iter " << iter;
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        ASSERT_TRUE(verifyBlockPattern(agRecv, count * nRanks, count))
            << "AllGather data wrong at iter " << iter;

        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(arSend, arRecv, count, ncclFloat32, ncclSum,
                                getActiveCommunicator(), getActiveStream()))
            << "AllReduce iter " << iter;
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        ASSERT_TRUE(verifyBufferData<float>(arRecv, count,
                                            [arExpected](size_t) { return arExpected; }))
            << "AllReduce data wrong at iter " << iter;
    }

    assertCEPathTaken("CeMPI_Stress/InterleavedCEAllGatherAndSMAllReduce");
}

#endif // MPI_TESTS_ENABLED
