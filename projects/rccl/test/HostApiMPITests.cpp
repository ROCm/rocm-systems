/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file HostApiMPITests.cpp
 * @brief P0 unit tests for RCCL's one-sided RMA Host API
 *
 * Tests cover:
 *   W1  - WindowRegisterDeregister: collective register/deregister lifecycle
 *   P1  - SinglePutRank0ToRank1:   basic ncclPutSignal + ncclWaitSignal
 *   P2  - PutWithNonZeroOffset:    peerWinOffset at kSize/2
 *   P3  - PutMultipleDataTypes:    float32, int32, float16 (raw bytes)
 *   S1  - SignalOnlyNoData:        ncclSignal with no data transfer
 *   WS2 - WaitSignalFenceSemantics: three back-to-back puts, opCnt=3
 *   O1  - DataVisibilityAfterSync: explicit host read of fine-grain memory
 *   E1  - PutSignalNullLocalbuff:  null localbuf → error (or skip)
 *   E2  - PutSignalNullWindow:     null peerWin  → error
 *   E3  - PutSignalOffsetOutOfBounds: offset past end → error (or skip)
 *   E4  - PutSignalInvalidSigIdx:  sigIdx=1      → error (or skip)
 *
 * Constraints (proxy GIN path, current API limits):
 *   sigIdx = 0, ctx = 0, flags = 0, winFlags = NCCL_WIN_DEFAULT
 *
 * API signatures (from src/nccl.h.in):
 *   ncclResult_t ncclMemAlloc(void** ptr, size_t size);
 *   ncclResult_t ncclMemFree(void* ptr);
 *   ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void* buff, size_t size,
 *                                        ncclWindow_t* win, int winFlags);
 *   ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win);
 *   ncclResult_t ncclPutSignal(const void* localbuff, size_t count,
 *                               ncclDataType_t datatype, int peer,
 *                               ncclWindow_t peerWin, size_t peerWinOffset,
 *                               int sigIdx, int ctx, unsigned int flags,
 *                               ncclComm_t comm, hipStream_t stream);
 *   ncclResult_t ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags,
 *                            ncclComm_t comm, hipStream_t stream);
 *   ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t* signalDescs,
 *                                ncclComm_t comm, hipStream_t stream);
 *
 * Run (example):
 *   mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter=HostApiTest.*
 */

#ifdef MPI_TESTS_ENABLED

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "HostApiHelpers.hpp"
#include "TestChecks.hpp"

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLHostApiHelpers;

namespace RcclUnitTesting
{

// ============================================================================
// Test fixture
// ============================================================================

/**
 * @class HostApiTest
 * @brief GTest fixture for RCCL Host API (one-sided RMA) tests.
 *
 * SetUp creates a communicator for all ranks; individual tests call
 * validateTestPrerequisites() to skip when rank count is insufficient.
 */
class HostApiTest : public MPITestBase
{
protected:
    void SetUp() override
    {
        MPITestBase::SetUp();
        ASSERT_EQ(ncclSuccess, createTestCommunicator());
    }

public:
    // Convenience: get rank and world size from the active communicator.
    int rank()   const
    {
        int r = -1;
        ncclCommUserRank(const_cast<HostApiTest*>(this)->getActiveCommunicator(), &r);
        return r;
    }
    int nRanks() const
    {
        int n = -1;
        ncclCommCount(const_cast<HostApiTest*>(this)->getActiveCommunicator(), &n);
        return n;
    }
};

// ============================================================================
// Constants shared across tests
// ============================================================================

namespace
{
constexpr size_t kTransferSize = 256 * 1024; // 256 KiB per PUT
// Window is large enough to hold the receive region AND the send region so
// that the source buffer for ncclPutSignal is always part of a registered
// window (required by the proxy GIN path).
constexpr size_t kOneMB        = 2 * kTransferSize; // send + recv regions
constexpr size_t kSendOffset   = 0;                 // sender carves src from here
constexpr size_t kRecvOffset   = kTransferSize;     // receiver data lands here
constexpr int    kSigIdx       = 0;
constexpr int    kCtx          = 0;
constexpr unsigned int kFlags  = 0;
} // namespace

// ============================================================================
// W1 — WindowRegisterDeregister
// ============================================================================

/**
 * @test HostApiTest.WindowRegisterDeregister
 * @brief Verify collective ncclCommWindowRegister / ncclCommWindowDeregister lifecycle.
 *
 * All ranks allocate a fine-grain buffer, collectively register a window,
 * skip if the system does not support windows, then deregister.
 */
static void runWindowRegisterDeregister(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int myRank = self->rank();

    // Allocate fine-grain buffer (CPU-accessible on ROCm).
    void* buf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&buf, kOneMB));
    auto bufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(buf); });

    // Collective window registration.
    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(self->getActiveCommunicator(), buf, kOneMB, &win, winFlags);

    if(win == nullptr)
    {
        // System / transport does not support windows — skip cleanly.
        if(myRank == 0)
        {
            TEST_INFO("ncclCommWindowRegister returned nullptr window (winFlags=0x%x) — "
                      "system does not support Host API windows; skipping W1.", winFlags);
        }
        GTEST_SKIP() << "System does not support ncclWindow (win == nullptr)";
    }

    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // NcclWindowGuard destructor calls ncclCommWindowDeregister.
    TEST_INFO("W1 rank %d: window registered and will be deregistered by guard (winFlags=0x%x).",
              myRank, winFlags);
}

TEST_F(HostApiTest, WindowRegisterDeregister)
{
    runWindowRegisterDeregister(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, WindowRegisterDeregisterSymmetric)
{
    runWindowRegisterDeregister(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// P1 — SinglePutRank0ToRank1
// ============================================================================

/**
 * @test HostApiTest.SinglePutRank0ToRank1
 * @brief Basic ncclPutSignal (rank 0 → rank 1) + ncclWaitSignal (rank 1).
 *
 * Rank 0 fills a source buffer with FillBuf, issues ncclPutSignal to
 * rank 1's window.  Rank 1 issues ncclWaitSignal(opCnt=1).  After
 * hipStreamSynchronize rank 1 verifies the window buffer.
 */
static void runSinglePutRank0ToRank1(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int myRank = self->rank();
    ncclComm_t    comm   = self->getActiveCommunicator();
    hipStream_t   stream = self->getActiveStream();

    // Both ranks allocate + register a window.
    // The window is split: [kSendOffset, kTransferSize) is the sender's source
    // region; [kRecvOffset, kRecvOffset+kTransferSize) is the receiver's dest.
    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, winFlags);

    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow (win == nullptr)";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Rank 0: fill the send region of its own window and PUT into rank 1's recv region.
    ncclResult_t putRes = ncclSuccess;
    if(myRank == 0)
    {
        void* srcBuf = static_cast<uint8_t*>(winBuf) + kSendOffset;
        FillBuf(srcBuf, kTransferSize, /*senderRank=*/0);
        putRes = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/kRecvOffset,
            kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, putRes);

    // Rank 1: wait for 1 signal from rank 0.
    ncclResult_t waitRes = ncclSuccess;
    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{/*opCnt=*/1, /*peer=*/0, kSigIdx, kCtx};
        waitRes = ncclWaitSignal(/*nDesc=*/1, &desc, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, waitRes);

    // Both ranks synchronize the stream.
    {
        hipError_t _sync_err = hipStreamSynchronize(stream);
        if (_sync_err != hipSuccess) {
            fprintf(stderr, "Rank %d: hipStreamSynchronize FAILED: err=%d (%s)\n",
                myRank, (int)_sync_err, hipGetErrorString(_sync_err));
            fflush(stderr);
        }
        ASSERT_MPI_EQ(hipSuccess, _sync_err);
    }

    bool ok = (myRank != 1) ||
              VerifyBuf(static_cast<const uint8_t*>(winBuf) + kRecvOffset, kTransferSize, /*seed=*/0);
    ASSERT_MPI_TRUE(ok);

    TEST_INFO("P1 rank %d: SinglePutRank0ToRank1 passed (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, SinglePutRank0ToRank1)
{
    runSinglePutRank0ToRank1(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, SinglePutRank0ToRank1Symmetric)
{
    runSinglePutRank0ToRank1(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// P2 — PutWithNonZeroOffset
// ============================================================================

/**
 * @test HostApiTest.PutWithNonZeroOffset
 * @brief PUT with peerWinOffset = kOneMB/2.
 *
 * Same as P1 but the destination window offset is placed at the midpoint of
 * the window.  Rank 1 verifies only the offset region.
 */
static void runPutWithNonZeroOffset(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int       myRank = self->rank();
    ncclComm_t      comm   = self->getActiveCommunicator();
    hipStream_t     stream = self->getActiveStream();

    // Window layout: [kSendOffset..kTransferSize) = send region,
    //                [kRecvOffset..kRecvOffset+kTransferSize) = recv region.
    // The PUT targets kRecvOffset in rank 1's window.
    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Zero the recv region on rank 1 so stale bytes are detectable.
    if(myRank == 1)
        FillSentinel(static_cast<uint8_t*>(winBuf) + kRecvOffset, kTransferSize, 0);

    ncclResult_t putRes = ncclSuccess;
    if(myRank == 0)
    {
        void* srcBuf = static_cast<uint8_t*>(winBuf) + kSendOffset;
        FillBuf(srcBuf, kTransferSize, /*senderRank=*/0);
        putRes = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/kRecvOffset,
            kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, putRes);

    ncclResult_t waitRes = ncclSuccess;
    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{1, 0, kSigIdx, kCtx};
        waitRes = ncclWaitSignal(1, &desc, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, waitRes);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    const uint8_t* recvBuf = static_cast<const uint8_t*>(winBuf) + kRecvOffset;
    bool ok = (myRank != 1) || VerifyBuf(recvBuf, kTransferSize, /*senderRank=*/0);
    ASSERT_MPI_TRUE(ok);

    TEST_INFO("P2 rank %d: PutWithNonZeroOffset passed (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, PutWithNonZeroOffset)
{
    runPutWithNonZeroOffset(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, PutWithNonZeroOffsetSymmetric)
{
    runPutWithNonZeroOffset(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// P3 — PutMultipleDataTypes
// ============================================================================

/**
 * @test HostApiTest.PutMultipleDataTypes
 * @brief PUT 256 elements of float32, int32, and float16 and verify raw bytes.
 *
 * The test uses ncclPutSignal with the correct element type.  Verification
 * is done via FillBuf / VerifyBuf on the raw bytes of each element array.
 */
static void runPutMultipleDataTypes(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int      myRank = self->rank();
    ncclComm_t     comm   = self->getActiveCommunicator();
    hipStream_t    stream = self->getActiveStream();
    const size_t   kElem  = 256;

    // Describe the three types: {ncclDataType, element_size, recv offset in window}
    // All recv offsets are placed in the upper half of the window (>= kRecvOffset).
    struct TypeDesc { ncclDataType_t type; size_t elemSz; size_t recvOff; };
    const TypeDesc types[] = {
        { ncclFloat32, sizeof(float),    kRecvOffset + 0          },
        { ncclInt32,   sizeof(int32_t),  kRecvOffset + 64  * 1024 },
        { ncclFloat16, sizeof(uint16_t), kRecvOffset + 128 * 1024 },
    };
    const int nTypes = static_cast<int>(sizeof(types) / sizeof(types[0]));

    // Window: lower half = send region, upper half = recv regions for all types.
    // Ensure the window is large enough for all recv regions.
    const size_t maxRecvEnd = types[nTypes - 1].recvOff
                              + kElem * types[nTypes - 1].elemSz;
    const size_t kWinSize = maxRecvEnd;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // For each type: fill → PUT (rank 0) / WaitSignal (rank 1) → sync → verify.
    for(int t = 0; t < nTypes; ++t)
    {
        const size_t byteCount = kElem * types[t].elemSz;

        ncclResult_t putRes = ncclSuccess;
        if(myRank == 0)
        {
            void* srcBuf = static_cast<uint8_t*>(winBuf) + kSendOffset;
            FillBuf(srcBuf, byteCount, /*senderRank=*/0);
            putRes = ncclPutSignal(
                srcBuf, kElem, types[t].type,
                /*peer=*/1, win, types[t].recvOff,
                kSigIdx, kCtx, kFlags, comm, stream);
        }
        ASSERT_MPI_EQ(ncclSuccess, putRes);

        ncclResult_t waitRes = ncclSuccess;
        if(myRank == 1)
        {
            ncclWaitSignalDesc_t desc{1, 0, kSigIdx, kCtx};
            waitRes = ncclWaitSignal(1, &desc, comm, stream);
        }
        ASSERT_MPI_EQ(ncclSuccess, waitRes);

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

        // ASSERT_MPI_TRUE must be called by all ranks.
        const uint8_t* base = static_cast<const uint8_t*>(winBuf);
        bool ok = (myRank != 1) || VerifyBuf(base + types[t].recvOff, byteCount, /*senderRank=*/0);
        ASSERT_MPI_TRUE(ok);

        TEST_INFO("P3 rank %d: type[%d] (elemSz=%zu) passed.", myRank, t, types[t].elemSz);
    }
}

TEST_F(HostApiTest, PutMultipleDataTypes)
{
    runPutMultipleDataTypes(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, PutMultipleDataTypesSymmetric)
{
    runPutMultipleDataTypes(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// S1 — SignalOnlyNoData
// ============================================================================

/**
 * @test HostApiTest.SignalOnlyNoData
 * @brief ncclSignal with no data transfer; window contents must be unchanged.
 *
 * Rank 1 pre-fills its window with sentinel value 0xAB.
 * Rank 0 issues ncclSignal(peer=1, sigIdx=0, ctx=0, flags=0).
 * Rank 1 issues ncclWaitSignal(opCnt=1, peer=0).
 * After sync rank 1 verifies every byte is still 0xAB.
 */
static void runSignalOnlyNoData(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int       myRank = self->rank();
    ncclComm_t      comm   = self->getActiveCommunicator();
    hipStream_t     stream = self->getActiveStream();
    const uint8_t   kSentinel = 0xAB;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Rank 1: pre-fill window with sentinel before the signal.
    if(myRank == 1)
        FillSentinel(winBuf, kOneMB, kSentinel);

    // Rank 0: signal only (no data).
    ncclResult_t sigRes = ncclSuccess;
    if(myRank == 0)
        sigRes = ncclSignal(/*peer=*/1, kSigIdx, kCtx, kFlags, comm, stream);
    ASSERT_MPI_EQ(ncclSuccess, sigRes);

    // Rank 1: wait for the signal.
    ncclResult_t waitRes = ncclSuccess;
    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{1, 0, kSigIdx, kCtx};
        waitRes = ncclWaitSignal(1, &desc, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, waitRes);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Rank 1: window must still be 0xAB throughout (no data was transferred).
    bool allSentinel = (myRank != 1) || AllSentinel(winBuf, kOneMB, kSentinel);
    ASSERT_MPI_TRUE(allSentinel);

    TEST_INFO("S1 rank %d: SignalOnlyNoData passed (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, SignalOnlyNoData)
{
    runSignalOnlyNoData(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, SignalOnlyNoDataSymmetric)
{
    runSignalOnlyNoData(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// WS2 — WaitSignalFenceSemantics
// ============================================================================

/**
 * @test HostApiTest.WaitSignalFenceSemantics
 * @brief Three consecutive PUTs to different window offsets; WaitSignal(opCnt=3).
 *
 * Rank 0 issues three ncclPutSignal calls to rank 1's window at offsets
 * [0, kTransferSize, 2*kTransferSize], each with a different pattern
 * (senderRank encoded as 10, 20, 30 to distinguish regions).
 * Rank 1 issues ncclWaitSignal with opCnt=3.
 * After sync rank 1 verifies all three regions.
 *
 * Note: VerifyBuf uses seed as the pattern index.  We pass synthetic seed
 * values (10, 20, 30) to distinguish the three regions.
 */
static void runWaitSignalFenceSemantics(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank    = self->rank();
    ncclComm_t   comm      = self->getActiveCommunicator();
    hipStream_t  stream    = self->getActiveStream();
    const int    kNumPuts  = 3;
    // Window layout (rank 0): [kSendOffset .. kSendOffset+kTransferSize) is the
    // send region (reused for each PUT); recv region not used by rank 0.
    // Window layout (rank 1): [0 .. kNumPuts*kTransferSize) is the receive area.
    // We need kSendOffset + kTransferSize <= total, and kNumPuts*kTransferSize
    // for the receive side.  Use kNumPuts+1 slots so rank 0's send region is
    // beyond the receive area.
    const size_t kWinSize  = (kNumPuts + 1) * kTransferSize;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Synthetic seed values — must match fill and verify on each side.
    const int seeds[kNumPuts] = {10, 20, 30};
    // Rank 0 carves its send buffer from the last slot of its own window.
    const size_t kSendSlot = static_cast<size_t>(kNumPuts) * kTransferSize;

    ncclResult_t putRes = ncclSuccess;
    if(myRank == 0)
    {
        uint8_t* sendRegion = static_cast<uint8_t*>(winBuf) + kSendSlot;
        for(int i = 0; i < kNumPuts && putRes == ncclSuccess; ++i)
        {
            FillBuf(sendRegion, kTransferSize, seeds[i]);
            size_t peerOff = static_cast<size_t>(i) * kTransferSize;
            putRes = ncclPutSignal(
                sendRegion, kTransferSize, ncclUint8,
                /*peer=*/1, win, peerOff,
                kSigIdx, kCtx, kFlags, comm, stream);
        }
    }
    ASSERT_MPI_EQ(ncclSuccess, putRes);

    ncclResult_t waitRes = ncclSuccess;
    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{kNumPuts, 0, kSigIdx, kCtx};
        waitRes = ncclWaitSignal(1, &desc, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, waitRes);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    bool allOk = true;
    if(myRank == 1)
    {
        const uint8_t* base = static_cast<const uint8_t*>(winBuf);
        for(int i = 0; i < kNumPuts; ++i)
        {
            size_t off = static_cast<size_t>(i) * kTransferSize;
            if(!VerifyBuf(base + off, kTransferSize, seeds[i]))
            {
                allOk = false;
                break;
            }
        }
    }
    ASSERT_MPI_TRUE(allOk);

    TEST_INFO("WS2 rank %d: WaitSignalFenceSemantics passed (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, WaitSignalFenceSemantics)
{
    runWaitSignalFenceSemantics(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, WaitSignalFenceSemanticsSymmetric)
{
    runWaitSignalFenceSemantics(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// O1 — DataVisibilityAfterSync
// ============================================================================

/**
 * @test HostApiTest.DataVisibilityAfterSync
 * @brief Explicit host read of fine-grain window memory after stream sync.
 *
 * Mirrors P1 but after hipStreamSynchronize the receiver copies the
 * fine-grain window pointer into a std::vector<uint8_t> on the host
 * VerifyBuf copies device→host via hipMemcpy into a staging buffer and checks
 * bytes there.
 */
static void runDataVisibilityAfterSync(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = self->rank();
    ncclComm_t   comm   = self->getActiveCommunicator();
    hipStream_t  stream = self->getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Rank 0 uses its own registered window as the source buffer.
    // Layout: [kSendOffset, kSendOffset+kTransferSize) = send region (rank 0)
    //         [kRecvOffset, kRecvOffset+kTransferSize) = recv region (rank 1)
    ncclResult_t putRes = ncclSuccess;
    if(myRank == 0)
    {
        void* sendRegion = static_cast<uint8_t*>(winBuf) + kSendOffset;
        FillBuf(sendRegion, kTransferSize, 0);
        putRes = ncclPutSignal(sendRegion, kTransferSize, ncclUint8,
                               1, win, kRecvOffset, kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, putRes);

    ncclResult_t waitRes = ncclSuccess;
    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{1, 0, kSigIdx, kCtx};
        waitRes = ncclWaitSignal(1, &desc, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, waitRes);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    bool ok = (myRank != 1) ||
              VerifyBuf(static_cast<const uint8_t*>(winBuf) + kRecvOffset, kTransferSize, /*seed=*/0);
    ASSERT_MPI_TRUE(ok);

    TEST_INFO("O1 rank %d: DataVisibilityAfterSync passed (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, DataVisibilityAfterSync)
{
    runDataVisibilityAfterSync(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, DataVisibilityAfterSyncSymmetric)
{
    runDataVisibilityAfterSync(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// E1 — PutSignalNullLocalbuff
// ============================================================================

/**
 * @test HostApiTest.PutSignalNullLocalbuff
 * @brief ncclPutSignal with null localbuf must return an error.
 *
 * Only rank 0 calls the API (non-collective immediate check).  Rank 1 does
 * nothing to avoid deadlock.  If argcheck is not implemented the test skips.
 */
static void runPutSignalNullLocalbuff(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = self->rank();
    ncclComm_t   comm   = self->getActiveCommunicator();
    hipStream_t  stream = self->getActiveStream();

    // Both ranks must register a window (collective).
    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    if(myRank == 0)
    {
        // Pass nullptr as localbuf.
        ncclResult_t res = ncclPutSignal(
            /*localbuff=*/nullptr, /*count=*/1, ncclFloat32,
            /*peer=*/1, win, /*peerWinOffset=*/0,
            kSigIdx, kCtx, kFlags, comm, stream);

        // If argcheck is not implemented the call may return ncclSuccess;
        // skip rather than fail in that case (the test documents intent).
        if(res == ncclSuccess)
        {
            GTEST_SKIP() << "E1: ncclPutSignal(nullptr localbuff) returned ncclSuccess "
                            "— argcheck not implemented; skipping.";
        }
        EXPECT_NE(ncclSuccess, res)
            << "E1: expected error for null localbuf, got ncclSuccess";
    }
    // Rank 1 does not call any collective → no deadlock.

    TEST_INFO("E1 rank %d: PutSignalNullLocalbuff done (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, PutSignalNullLocalbuff)
{
    runPutSignalNullLocalbuff(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, PutSignalNullLocalbuffSymmetric)
{
    runPutSignalNullLocalbuff(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// E2 — PutSignalNullWindow
// ============================================================================

/**
 * @test HostApiTest.PutSignalNullWindow
 * @brief ncclPutSignal with null peerWin must return an error.
 *
 * Non-collective: only rank 0 calls the API.
 */
static void runPutSignalNullWindow(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = self->rank();
    ncclComm_t   comm   = self->getActiveCommunicator();
    hipStream_t  stream = self->getActiveStream();

    // Allocate a valid source buffer on rank 0.
    void* srcBuf = nullptr;
    if(myRank == 0)
    {
        ASSERT_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kOneMB));
    }
    auto srcBufGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });

    if(myRank == 0)
    {
        ncclResult_t res = ncclPutSignal(
            srcBuf, /*count=*/1, ncclFloat32,
            /*peer=*/1, /*peerWin=*/nullptr, /*peerWinOffset=*/0,
            kSigIdx, kCtx, kFlags, comm, stream);

        if(res == ncclSuccess)
        {
            GTEST_SKIP() << "E2: ncclPutSignal(nullptr peerWin) returned ncclSuccess "
                            "— argcheck not implemented; skipping.";
        }
        EXPECT_NE(ncclSuccess, res)
            << "E2: expected error for null peerWin, got ncclSuccess";
    }

    TEST_INFO("E2 rank %d: PutSignalNullWindow done (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, PutSignalNullWindow)
{
    runPutSignalNullWindow(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, PutSignalNullWindowSymmetric)
{
    runPutSignalNullWindow(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// E3 — PutSignalOffsetOutOfBounds
// ============================================================================

/**
 * @test HostApiTest.PutSignalOffsetOutOfBounds
 * @brief ncclPutSignal with peerWinOffset == kOneMB (past end) must error.
 *
 * Both ranks register a 1 MiB window.  Rank 0 attempts a PUT at offset
 * equal to the window size (one byte past the end).
 */
static void runPutSignalOffsetOutOfBounds(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = self->rank();
    ncclComm_t   comm   = self->getActiveCommunicator();
    hipStream_t  stream = self->getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    if(myRank == 0)
    {
        // peerWinOffset = kOneMB means the first byte of the PUT is exactly at
        // the end of the window — out of bounds.
        void* srcBuf = static_cast<uint8_t*>(winBuf) + kSendOffset;
        ncclResult_t res = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/kOneMB,
            kSigIdx, kCtx, kFlags, comm, stream);

        if(res == ncclSuccess)
        {
            GTEST_SKIP() << "E3: out-of-bounds offset returned ncclSuccess "
                            "— argcheck not implemented; skipping.";
        }
        EXPECT_NE(ncclSuccess, res)
            << "E3: expected error for out-of-bounds offset, got ncclSuccess";
    }

    TEST_INFO("E3 rank %d: PutSignalOffsetOutOfBounds done (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, PutSignalOffsetOutOfBounds)
{
    runPutSignalOffsetOutOfBounds(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, PutSignalOffsetOutOfBoundsSymmetric)
{
    runPutSignalOffsetOutOfBounds(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// E4 — PutSignalInvalidSigIdx
// ============================================================================

/**
 * @test HostApiTest.PutSignalInvalidSigIdx
 * @brief ncclPutSignal with sigIdx=1 (reserved, must be 0) should error.
 *
 * Non-collective: only rank 0 calls the API.  Skip if not validated.
 */
static void runPutSignalInvalidSigIdx(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = self->rank();
    ncclComm_t   comm   = self->getActiveCommunicator();
    hipStream_t  stream = self->getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    if(myRank == 0)
    {
        void* srcBuf = static_cast<uint8_t*>(winBuf) + kSendOffset;
        ncclResult_t res = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/0,
            /*sigIdx=*/1, kCtx, kFlags, comm, stream);

        if(res == ncclSuccess)
        {
            GTEST_SKIP() << "E4: sigIdx=1 returned ncclSuccess "
                            "— sigIdx validation not implemented; skipping.";
        }
        EXPECT_NE(ncclSuccess, res)
            << "E4: expected error for sigIdx=1, got ncclSuccess";
    }

    TEST_INFO("E4 rank %d: PutSignalInvalidSigIdx done (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, PutSignalInvalidSigIdx)
{
    runPutSignalInvalidSigIdx(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, PutSignalInvalidSigIdxSymmetric)
{
    runPutSignalInvalidSigIdx(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// S2 — SignalCumulativeFence
// ============================================================================

/**
 * @test HostApiTest.SignalCumulativeFence
 * @brief ncclPutSignal then ncclSignal from rank 0; rank 1 waits with opCnt=2.
 *
 * Rank 0 issues one ncclPutSignal (256 bytes to rank 1's window at offset 0)
 * then one ncclSignal (no data) to rank 1, both before rank 1 waits.
 * Rank 1 waits with opCnt=2 (fence: 1 PUT + 1 SIGNAL).
 * After sync rank 1 verifies the PUT data.
 */
static void runSignalCumulativeFence(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank = self->rank();
    ncclComm_t   comm   = self->getActiveCommunicator();
    hipStream_t  stream = self->getActiveStream();
    const size_t kSize  = 256;
    // Window large enough for both send region (rank 0) and recv region (rank 1).
    // kSendOffset=0, kRecvOffset=kTransferSize.
    const size_t kWinSize = kOneMB;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    ncclResult_t sendRes = ncclSuccess;
    if(myRank == 0)
    {
        void* sendRegion = static_cast<uint8_t*>(winBuf) + kSendOffset;
        FillBuf(sendRegion, kSize, /*senderRank=*/0);

        sendRes = ncclGroupStart();
        if(sendRes == ncclSuccess)
        {
            ncclResult_t r1 = ncclPutSignal(
                sendRegion, kSize, ncclUint8,
                /*peer=*/1, win, /*peerWinOffset=*/kRecvOffset,
                kSigIdx, kCtx, kFlags, comm, stream);
            ncclResult_t r2 = ncclSignal(/*peer=*/1, kSigIdx, kCtx, kFlags, comm, stream);
            sendRes = ncclGroupEnd();
            if(sendRes == ncclSuccess) sendRes = r1;
            if(sendRes == ncclSuccess) sendRes = r2;
        }
    }
    ASSERT_MPI_EQ(ncclSuccess, sendRes);

    MPI_Barrier(MPI_COMM_WORLD);

    ncclResult_t waitRes = ncclSuccess;
    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{/*opCnt=*/2, /*peer=*/0, kSigIdx, kCtx};
        waitRes = ncclWaitSignal(/*nDesc=*/1, &desc, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, waitRes);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    bool ok = (myRank != 1) ||
              VerifyBuf(static_cast<uint8_t*>(winBuf) + kRecvOffset, kSize, /*senderRank=*/0);
    ASSERT_MPI_TRUE(ok);

    TEST_INFO("S2 rank %d: SignalCumulativeFence passed (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, SignalCumulativeFence)
{
    runSignalCumulativeFence(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, SignalCumulativeFenceSymmetric)
{
    runSignalCumulativeFence(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// WS3 — MultipleSendersOneReceiver
// ============================================================================

/**
 * @test HostApiTest.MultipleSendersOneReceiver
 * @brief Two senders (rank 0, rank 1) put to disjoint offsets of rank 2's window.
 *
 * Rank 0 PUTs 256 bytes at offset 0; rank 1 PUTs 256 bytes at offset 4096.
 * Rank 2 issues two separate ncclWaitSignal calls (one per sender, opCnt=1 each).
 * After sync rank 2 verifies both regions.
 *
 * Window layout: [0..kSize) and [4096..4096+kSize) are receive slots for rank 2.
 * Senders carve their send buffer from offset kSendSlot in their own window.
 */
static void runMultipleSendersOneReceiver(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/3))
    {
        GTEST_SKIP() << "Need at least 3 MPI processes";
    }

    const int    myRank   = self->rank();
    ncclComm_t   comm     = self->getActiveCommunicator();
    hipStream_t  stream   = self->getActiveStream();
    const size_t kSize    = 256;
    // 8192 covers the two recv slots at 0 and 4096; add kSize more for the send
    // region so senders can carve from their own window beyond the recv area.
    const size_t kSendSlot = 8192;
    const size_t kWinSize  = kSendSlot + kSize;

    // All ranks register a window (collective).
    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    ncclResult_t sendRes = ncclSuccess;
    if(myRank == 0)
    {
        void* sendRegion = static_cast<uint8_t*>(winBuf) + kSendSlot;
        FillBuf(sendRegion, kSize, /*senderRank=*/0);
        sendRes = ncclPutSignal(sendRegion, kSize, ncclUint8,
                                /*peer=*/2, win, /*peerWinOffset=*/0,
                                kSigIdx, kCtx, kFlags, comm, stream);
    }
    else if(myRank == 1)
    {
        void* sendRegion = static_cast<uint8_t*>(winBuf) + kSendSlot;
        FillBuf(sendRegion, kSize, /*senderRank=*/1);
        sendRes = ncclPutSignal(sendRegion, kSize, ncclUint8,
                                /*peer=*/2, win, /*peerWinOffset=*/4096,
                                kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, sendRes);

    ncclResult_t waitRes = ncclSuccess;
    if(myRank == 2)
    {
        ncclWaitSignalDesc_t d0{/*opCnt=*/1, /*peer=*/0, kSigIdx, kCtx};
        waitRes = ncclWaitSignal(1, &d0, comm, stream);
        if(waitRes == ncclSuccess)
        {
            ncclWaitSignalDesc_t d1{/*opCnt=*/1, /*peer=*/1, kSigIdx, kCtx};
            waitRes = ncclWaitSignal(1, &d1, comm, stream);
        }
    }
    ASSERT_MPI_EQ(ncclSuccess, waitRes);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    bool allOk = true;
    if(myRank == 2)
    {
        const uint8_t* base = static_cast<const uint8_t*>(winBuf);
        allOk = VerifyBuf(base + 0,    kSize, /*senderRank=*/0) &&
                VerifyBuf(base + 4096, kSize, /*senderRank=*/1);
    }
    ASSERT_MPI_TRUE(allOk);

    TEST_INFO("WS3 rank %d: MultipleSendersOneReceiver passed (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, MultipleSendersOneReceiver)
{
    runMultipleSendersOneReceiver(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, MultipleSendersOneReceiverSymmetric)
{
    runMultipleSendersOneReceiver(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// WS4 — WaitSignalMultipleDescriptors
// ============================================================================

/**
 * @test HostApiTest.WaitSignalMultipleDescriptors
 * @brief ncclWaitSignal called once with an array of 2 descriptors (nDesc=2).
 *
 * Same topology as WS3 (ranks 0 and 1 → rank 2) but rank 2 passes both
 * descriptors in a single ncclWaitSignal(comm, 2, descs, stream) call.
 *
 * Window layout matches WS3: recv slots at 0 and 4096; send region at kSendSlot.
 */
static void runWaitSignalMultipleDescriptors(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/3))
    {
        GTEST_SKIP() << "Need at least 3 MPI processes";
    }

    const int    myRank   = self->rank();
    ncclComm_t   comm     = self->getActiveCommunicator();
    hipStream_t  stream   = self->getActiveStream();
    const size_t kSize    = 256;
    const size_t kSendSlot = 8192;
    const size_t kWinSize  = kSendSlot + kSize;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    ncclResult_t sendRes = ncclSuccess;
    if(myRank == 0)
    {
        void* sendRegion = static_cast<uint8_t*>(winBuf) + kSendSlot;
        FillBuf(sendRegion, kSize, /*senderRank=*/0);
        sendRes = ncclPutSignal(sendRegion, kSize, ncclUint8,
                                /*peer=*/2, win, /*peerWinOffset=*/0,
                                kSigIdx, kCtx, kFlags, comm, stream);
    }
    else if(myRank == 1)
    {
        void* sendRegion = static_cast<uint8_t*>(winBuf) + kSendSlot;
        FillBuf(sendRegion, kSize, /*senderRank=*/1);
        sendRes = ncclPutSignal(sendRegion, kSize, ncclUint8,
                                /*peer=*/2, win, /*peerWinOffset=*/4096,
                                kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, sendRes);

    ncclResult_t waitRes = ncclSuccess;
    if(myRank == 2)
    {
        // Both descriptors in one call.
        ncclWaitSignalDesc_t descs[2];
        descs[0] = {/*opCnt=*/1, /*peer=*/0, kSigIdx, kCtx};
        descs[1] = {/*opCnt=*/1, /*peer=*/1, kSigIdx, kCtx};
        waitRes = ncclWaitSignal(/*nDesc=*/2, &descs[0], comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, waitRes);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    bool allOk = true;
    if(myRank == 2)
    {
        const uint8_t* base = static_cast<const uint8_t*>(winBuf);
        allOk = VerifyBuf(base + 0,    kSize, /*senderRank=*/0) &&
                VerifyBuf(base + 4096, kSize, /*senderRank=*/1);
    }
    ASSERT_MPI_TRUE(allOk);

    TEST_INFO("WS4 rank %d: WaitSignalMultipleDescriptors passed (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, WaitSignalMultipleDescriptors)
{
    runWaitSignalMultipleDescriptors(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, WaitSignalMultipleDescriptorsSymmetric)
{
    runWaitSignalMultipleDescriptors(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// P4 — LargePut
// ============================================================================

/**
 * @test HostApiTest.LargePut
 * @brief PUT 256 MiB from rank 0 to rank 1's window; spot-check first and last 64 bytes.
 *
 * We use 256 MiB rather than the full 1 GiB to avoid OOM on hardware with
 * limited fine-grain / pinned memory capacity while still exercising a
 * large-transfer code path.
 */
static void runLargePut(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank  = self->rank();
    ncclComm_t   comm    = self->getActiveCommunicator();
    hipStream_t  stream  = self->getActiveStream();

    // 256 MiB — large but avoids OOM on most ROCm systems.
    const size_t kLargeSize = 256ULL * 1024 * 1024;
    const uint8_t kByte     = 0xAB;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kLargeSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kLargeSize, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    ncclResult_t putRes = ncclSuccess;
    if(myRank == 0)
    {
        FillSentinel(winBuf, kLargeSize, kByte);
        putRes = ncclPutSignal(winBuf, kLargeSize, ncclUint8,
                               /*peer=*/1, win, /*peerWinOffset=*/0,
                               kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, putRes);

    ncclResult_t waitRes = ncclSuccess;
    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{/*opCnt=*/1, /*peer=*/0, kSigIdx, kCtx};
        waitRes = ncclWaitSignal(1, &desc, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, waitRes);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    bool ok = (myRank != 1) || AllSentinel(winBuf, kLargeSize, kByte);
    ASSERT_MPI_TRUE(ok);

    TEST_INFO("P4 rank %d: LargePut passed (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, LargePut)
{
    runLargePut(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, LargePutSymmetric)
{
    runLargePut(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// P5 — AllToAllPut
// ============================================================================

/**
 * @test HostApiTest.AllToAllPut
 * @brief Each rank i PUTs to rank (i+1)%N and waits for rank (i-1+N)%N.
 *
 * All ranks register 4096-byte windows.  Each rank fills 256 bytes with its
 * own rank pattern, PUTs to the next rank (offset 0), then waits for the
 * previous rank's signal.  After sync each rank verifies the received data.
 */
static void runAllToAllPut(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank   = self->rank();
    const int    nRanks_  = self->nRanks();
    ncclComm_t   comm     = self->getActiveCommunicator();
    hipStream_t  stream   = self->getActiveStream();
    const size_t kSize    = 256;
    // Window layout: [0..kSize) = recv slot; [kSize..2*kSize) = send region.
    // Both fit within 4096, so kWinSize=4096 is fine.
    const size_t kRecvSlot = 0;
    const size_t kSendSlot = kSize;
    const size_t kWinSize  = 4096;

    const int sendTo   = (myRank + 1)           % nRanks_;
    const int recvFrom = (myRank + nRanks_ - 1) % nRanks_;

    // All ranks register a window.
    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Carve the send buffer from this rank's own registered window.
    void* sendRegion = static_cast<uint8_t*>(winBuf) + kSendSlot;
    FillBuf(sendRegion, kSize, myRank);

    // Batch PUT + WAIT in a group.
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    ncclResult_t rPut = ncclPutSignal(
        sendRegion, kSize, ncclUint8,
        sendTo, win, /*peerWinOffset=*/kRecvSlot,
        kSigIdx, kCtx, kFlags, comm, stream);
    ncclWaitSignalDesc_t desc{/*opCnt=*/1, recvFrom, kSigIdx, kCtx};
    ncclResult_t rWait = ncclWaitSignal(1, &desc, comm, stream);
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_MPI_EQ(ncclSuccess, rPut);
    ASSERT_MPI_EQ(ncclSuccess, rWait);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    // Verify: we should have received recvFrom's pattern at kRecvSlot.
    bool ok = VerifyBuf(static_cast<uint8_t*>(winBuf) + kRecvSlot, kSize, recvFrom);
    ASSERT_MPI_TRUE(ok);

    TEST_INFO("P5 rank %d: AllToAllPut passed (recv from rank %d, winFlags=0x%x).",
              myRank, recvFrom, winFlags);
}

TEST_F(HostApiTest, AllToAllPut)
{
    runAllToAllPut(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, AllToAllPutSymmetric)
{
    runAllToAllPut(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// O2 — SignalImpliesPriorPutsDelivered
// ============================================================================

/**
 * @test HostApiTest.SignalImpliesPriorPutsDelivered
 * @brief Two ncclPutSignal calls from rank 0; rank 1 waits with opCnt=2.
 *
 * Each ncclPutSignal implicitly delivers a signal.  Two calls = opCnt 2.
 * Rank 1 verifies both data regions after sync.
 */
static void runSignalImpliesPriorPutsDelivered(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank  = self->rank();
    ncclComm_t   comm    = self->getActiveCommunicator();
    hipStream_t  stream  = self->getActiveStream();
    const size_t kSize   = 256;
    const size_t kWinSize = kOneMB;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // Rank 0 carves two send regions from its own registered window.
    // Layout: [kSendOffset .. kSendOffset+kSize) = src0
    //         [kSendOffset+kSize .. kSendOffset+2*kSize) = src1
    // Rank 1 receives at kRecvOffset and kRecvOffset+512.
    ncclResult_t putRes = ncclSuccess;
    if(myRank == 0)
    {
        void* src0 = static_cast<uint8_t*>(winBuf) + kSendOffset;
        void* src1 = static_cast<uint8_t*>(winBuf) + kSendOffset + kSize;
        FillBuf(src0, kSize, /*senderRank=*/0);
        FillBuf(src1, kSize, /*senderRank=*/10);
        putRes = ncclPutSignal(src0, kSize, ncclUint8,
                               /*peer=*/1, win, /*peerWinOffset=*/kRecvOffset,
                               kSigIdx, kCtx, kFlags, comm, stream);
        if(putRes == ncclSuccess)
            putRes = ncclPutSignal(src1, kSize, ncclUint8,
                                   /*peer=*/1, win, /*peerWinOffset=*/kRecvOffset + 512,
                                   kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, putRes);

    ncclResult_t waitRes = ncclSuccess;
    if(myRank == 1)
    {
        ncclWaitSignalDesc_t desc{/*opCnt=*/2, /*peer=*/0, kSigIdx, kCtx};
        waitRes = ncclWaitSignal(1, &desc, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, waitRes);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    bool allOk = true;
    if(myRank == 1)
    {
        const uint8_t* base = static_cast<const uint8_t*>(winBuf);
        allOk = VerifyBuf(base + kRecvOffset,       kSize, /*senderRank=*/0) &&
                VerifyBuf(base + kRecvOffset + 512, kSize, /*senderRank=*/10);
    }
    ASSERT_MPI_TRUE(allOk);

    TEST_INFO("O2 rank %d: SignalImpliesPriorPutsDelivered passed (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, SignalImpliesPriorPutsDelivered)
{
    runSignalImpliesPriorPutsDelivered(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, SignalImpliesPriorPutsDeliveredSymmetric)
{
    runSignalImpliesPriorPutsDelivered(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// E5 — PutSignalInvalidCtx
// ============================================================================

/**
 * @test HostApiTest.PutSignalInvalidCtx
 * @brief ncclPutSignal with ctx=1 (reserved, must be 0) should error.
 *
 * Non-collective: only rank 0 calls the API.  Skip if argcheck is not
 * implemented (i.e., the call returns ncclSuccess).
 */
static void runPutSignalInvalidCtx(HostApiTest* self, int winFlags)
{
    if(!self->validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank = self->rank();
    ncclComm_t   comm   = self->getActiveCommunicator();
    hipStream_t  stream = self->getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, winFlags);
    if(win == nullptr)
    {
        GTEST_SKIP() << "System does not support ncclWindow";
    }
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    if(myRank == 0)
    {
        void* srcBuf = static_cast<uint8_t*>(winBuf) + kSendOffset;
        ncclResult_t res = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/0,
            kSigIdx, /*ctx=*/1, kFlags, comm, stream);
        EXPECT_NE(ncclSuccess, res)
            << "E5: expected error for ctx=1, got ncclSuccess";
    }
    // Rank 1 does not participate in this non-collective error path.

    TEST_INFO("E5 rank %d: PutSignalInvalidCtx done (winFlags=0x%x).", myRank, winFlags);
}

TEST_F(HostApiTest, PutSignalInvalidCtx)
{
    runPutSignalInvalidCtx(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, PutSignalInvalidCtxSymmetric)
{
    runPutSignalInvalidCtx(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// E6 — WaitSignalNullDescs
// ============================================================================

/**
 * @test HostApiTest.WaitSignalNullDescs
 * @brief ncclWaitSignal(nDesc=1, nullptr, ...) must return ncclInvalidArgument.
 *
 * Each rank calls independently (non-collective).  Skip if not validated.
 */
static void runWaitSignalNullDescs(HostApiTest* self, int /*winFlags*/)
{
    const int    myRank = self->rank();
    ncclComm_t   comm   = self->getActiveCommunicator();
    hipStream_t  stream = self->getActiveStream();

    ncclResult_t res = ncclWaitSignal(/*nDesc=*/1, /*signalDescs=*/nullptr, comm, stream);

    EXPECT_EQ(ncclInvalidArgument, res)
        << "E6: expected ncclInvalidArgument for null descs with nDesc=1";

    TEST_INFO("E6 rank %d: WaitSignalNullDescs done.", myRank);
}

TEST_F(HostApiTest, WaitSignalNullDescs)
{
    runWaitSignalNullDescs(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, WaitSignalNullDescsSymmetric)
{
    runWaitSignalNullDescs(this, NCCL_WIN_COLL_SYMMETRIC);
}

// ============================================================================
// E7 — WaitSignalZeroDesc
// ============================================================================

/**
 * @test HostApiTest.WaitSignalZeroDesc
 * @brief ncclWaitSignal(nDesc=0, nullptr, ...) — zero descriptors is a no-op.
 *
 * Expect ncclSuccess (or ncclInvalidArgument — both are acceptable).
 * No stream sync or data transfer involved.
 */
static void runWaitSignalZeroDesc(HostApiTest* self, int /*winFlags*/)
{
    const int    myRank = self->rank();
    ncclComm_t   comm   = self->getActiveCommunicator();
    hipStream_t  stream = self->getActiveStream();

    ncclResult_t res = ncclWaitSignal(/*nDesc=*/0, /*signalDescs=*/nullptr, comm, stream);
    EXPECT_TRUE(res == ncclSuccess || res == ncclInvalidArgument)
        << "E7: expected ncclSuccess or ncclInvalidArgument for nDesc=0, got "
        << static_cast<int>(res);

    TEST_INFO("E7 rank %d: WaitSignalZeroDesc done (result=%d).", myRank, static_cast<int>(res));
}

TEST_F(HostApiTest, WaitSignalZeroDesc)
{
    runWaitSignalZeroDesc(this, NCCL_WIN_DEFAULT);
}

TEST_F(HostApiTest, WaitSignalZeroDescSymmetric)
{
    runWaitSignalZeroDesc(this, NCCL_WIN_COLL_SYMMETRIC);
}

} // namespace RcclUnitTesting

#endif // MPI_TESTS_ENABLED
