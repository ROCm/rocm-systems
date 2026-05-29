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
 *   M1  - TwoCommunicatorsIndependentWindows: two comms, independent windows
 *   M2  - StressManySmallPuts:     100×64-byte PUTs, opCnt=100 wait
 *   P6  - PutToSelf:              PUT to own window (peer=self loopback)
 *   W3  - DoubleDeregister:       ncclCommWindowDeregister twice on same handle
 *
 * Constraints (proxy GIN path, current API limits):
 *   sigIdx = 0, ctx = 0, flags = 0, winFlags = kWinMode (NCCL_WIN_DEFAULT)
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

#if defined(MPI_TESTS_ENABLED) && defined(RCCL_ENABLE_HOST_API_TESTS)

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
constexpr int    kWinMode      = NCCL_WIN_DEFAULT;
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
TEST_F(HostApiTest, WindowRegisterDeregister)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int myRank = rank();

    // Allocate fine-grain buffer (CPU-accessible on ROCm).
    void* buf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&buf, kOneMB));
    auto bufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(buf); });

    // Collective window registration.
    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(getActiveCommunicator(), buf, kOneMB, &win, kWinMode);

    ASSERT_MPI_NE(win, nullptr);

    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    // NcclWindowGuard destructor calls ncclCommWindowDeregister.
    TEST_INFO("W1 rank %d: window registered and will be deregistered by guard.", myRank);
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
TEST_F(HostApiTest, SinglePutRank0ToRank1)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int myRank = rank();
    ncclComm_t    comm   = getActiveCommunicator();
    hipStream_t   stream = getActiveStream();

    // Both ranks allocate + register a window.
    // The window is split: [kSendOffset, kTransferSize) is the sender's source
    // region; [kRecvOffset, kRecvOffset+kTransferSize) is the receiver's dest.
    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, kWinMode);

    ASSERT_MPI_NE(win, nullptr);
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

    TEST_INFO("P1 rank %d: SinglePutRank0ToRank1 passed.", myRank);
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
TEST_F(HostApiTest, PutWithNonZeroOffset)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int       myRank = rank();
    ncclComm_t      comm   = getActiveCommunicator();
    hipStream_t     stream = getActiveStream();

    // Window layout: [kSendOffset..kTransferSize) = send region,
    //                [kRecvOffset..kRecvOffset+kTransferSize) = recv region.
    // The PUT targets kRecvOffset in rank 1's window.
    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
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

    TEST_INFO("P2 rank %d: PutWithNonZeroOffset passed.", myRank);
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
TEST_F(HostApiTest, PutMultipleDataTypes)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int      myRank = rank();
    ncclComm_t     comm   = getActiveCommunicator();
    hipStream_t    stream = getActiveStream();
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
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
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
TEST_F(HostApiTest, SignalOnlyNoData)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int       myRank = rank();
    ncclComm_t      comm   = getActiveCommunicator();
    hipStream_t     stream = getActiveStream();
    const uint8_t   kSentinel = 0xAB;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
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

    TEST_INFO("S1 rank %d: SignalOnlyNoData passed.", myRank);
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
TEST_F(HostApiTest, WaitSignalFenceSemantics)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank    = rank();
    ncclComm_t   comm      = getActiveCommunicator();
    hipStream_t  stream    = getActiveStream();
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
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
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

    TEST_INFO("WS2 rank %d: WaitSignalFenceSemantics passed.", myRank);
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
TEST_F(HostApiTest, DataVisibilityAfterSync)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
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

    TEST_INFO("O1 rank %d: DataVisibilityAfterSync passed.", myRank);
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
TEST_F(HostApiTest, PutSignalNullLocalbuff)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    // Both ranks must register a window (collective).
    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, kWinMode);

    ASSERT_MPI_NE(win, nullptr);
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    ncclResult_t res = ncclInvalidArgument;
    if(myRank == 0)
    {
        // Pass nullptr as localbuf.
        res = ncclPutSignal(
            /*localbuff=*/nullptr, /*count=*/1, ncclFloat32,
            /*peer=*/1, win, /*peerWinOffset=*/0,
            kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_NE(ncclSuccess, res);
    TEST_INFO("E1 rank %d: PutSignalNullLocalbuff done.", myRank);
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
TEST_F(HostApiTest, PutSignalNullWindow)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    // Allocate a valid source buffer on rank 0.
    void* srcBuf = nullptr;
    if(myRank == 0)
    {
        ASSERT_EQ(ncclSuccess, allocFineGrainBuffer(&srcBuf, kOneMB));
    }
    auto srcBufGuard = makeScopeGuard([&]() { if(srcBuf) freeFineGrainBuffer(srcBuf); });
    
    ncclResult_t res = ncclInvalidArgument;
    if(myRank == 0)
    {
        res = ncclPutSignal(
            srcBuf, /*count=*/1, ncclFloat32,
            /*peer=*/1, /*peerWin=*/nullptr, /*peerWinOffset=*/0,
            kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_NE(ncclSuccess, res);
    

    TEST_INFO("E2 rank %d: PutSignalNullWindow done.", myRank);
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
TEST_F(HostApiTest, PutSignalOffsetOutOfBounds)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    ncclResult_t res = ncclInvalidArgument;
    if(myRank == 0)
    {
        // peerWinOffset = kOneMB means the first byte of the PUT is exactly at
        // the end of the window — out of bounds.
        void* srcBuf = static_cast<uint8_t*>(winBuf) + kSendOffset;
        res = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/kOneMB,
            kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_NE(ncclSuccess, res);

    TEST_INFO("E3 rank %d: PutSignalOffsetOutOfBounds done.", myRank);
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
TEST_F(HostApiTest, PutSignalInvalidSigIdx)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    ncclResult_t res = ncclInvalidArgument;
    if(myRank == 0)
    {
        void* srcBuf = static_cast<uint8_t*>(winBuf) + kSendOffset;
        res = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/0,
            /*sigIdx=*/1, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_NE(ncclSuccess, res);
    TEST_INFO("E4 rank %d: PutSignalInvalidSigIdx done.", myRank);
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
TEST_F(HostApiTest, SignalCumulativeFence)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();
    const size_t kSize  = 256;
    // Window large enough for both send region (rank 0) and recv region (rank 1).
    // kSendOffset=0, kRecvOffset=kTransferSize.
    const size_t kWinSize = kOneMB;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
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

    TEST_INFO("S2 rank %d: SignalCumulativeFence passed.", myRank);
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
TEST_F(HostApiTest, MultipleSendersOneReceiver)
{
    if(!validateTestPrerequisites(/*min=*/3))
    {
        GTEST_SKIP() << "Need at least 3 MPI processes";
    }

    const int    myRank   = rank();
    ncclComm_t   comm     = getActiveCommunicator();
    hipStream_t  stream   = getActiveStream();
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
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
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

    TEST_INFO("WS3 rank %d: MultipleSendersOneReceiver passed.", myRank);
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
TEST_F(HostApiTest, WaitSignalMultipleDescriptors)
{
    if(!validateTestPrerequisites(/*min=*/3))
    {
        GTEST_SKIP() << "Need at least 3 MPI processes";
    }

    const int    myRank   = rank();
    ncclComm_t   comm     = getActiveCommunicator();
    hipStream_t  stream   = getActiveStream();
    const size_t kSize    = 256;
    const size_t kSendSlot = 8192;
    const size_t kWinSize  = kSendSlot + kSize;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
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

    TEST_INFO("WS4 rank %d: WaitSignalMultipleDescriptors passed.", myRank);
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
TEST_F(HostApiTest, LargePut)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank  = rank();
    ncclComm_t   comm    = getActiveCommunicator();
    hipStream_t  stream  = getActiveStream();

    // 256 MiB — large but avoids OOM on most ROCm systems.
    const size_t kLargeSize = 256ULL * 1024 * 1024;
    const uint8_t kByte     = 0xAB;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kLargeSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kLargeSize, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
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

    TEST_INFO("P4 rank %d: LargePut passed.", myRank);
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
TEST_F(HostApiTest, AllToAllPut)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank   = rank();
    const int    nRanks_  = nRanks();
    ncclComm_t   comm     = getActiveCommunicator();
    hipStream_t  stream   = getActiveStream();
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
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
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

    TEST_INFO("P5 rank %d: AllToAllPut passed (recv from rank %d).", myRank, recvFrom);
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
TEST_F(HostApiTest, SignalImpliesPriorPutsDelivered)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank  = rank();
    ncclComm_t   comm    = getActiveCommunicator();
    hipStream_t  stream  = getActiveStream();
    const size_t kSize   = 256;
    const size_t kWinSize = kOneMB;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
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

    TEST_INFO("O2 rank %d: SignalImpliesPriorPutsDelivered passed.", myRank);
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
TEST_F(HostApiTest, PutSignalInvalidCtx)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    ncclResult_t res = ncclInvalidArgument;
    if(myRank == 0)
    {
        void* srcBuf = static_cast<uint8_t*>(winBuf) + kSendOffset;
        res = ncclPutSignal(
            srcBuf, kTransferSize, ncclUint8,
            /*peer=*/1, win, /*peerWinOffset=*/0,
            kSigIdx, /*ctx=*/1, kFlags, comm, stream);
    }
    ASSERT_MPI_NE(ncclSuccess, res);
    TEST_INFO("E5 rank %d: PutSignalInvalidCtx done.", myRank);
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
TEST_F(HostApiTest, WaitSignalNullDescs)
{
    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    ncclResult_t res = ncclWaitSignal(/*nDesc=*/1, /*signalDescs=*/nullptr, comm, stream);

    EXPECT_EQ(ncclInvalidArgument, res)
        << "E6: expected ncclInvalidArgument for null descs with nDesc=1";

    TEST_INFO("E6 rank %d: WaitSignalNullDescs done.", myRank);
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
TEST_F(HostApiTest, WaitSignalZeroDesc)
{
    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    ncclResult_t res = ncclWaitSignal(/*nDesc=*/0, /*signalDescs=*/nullptr, comm, stream);
    EXPECT_TRUE(res == ncclSuccess || res == ncclInvalidArgument)
        << "E7: expected ncclSuccess or ncclInvalidArgument for nDesc=0, got "
        << static_cast<int>(res);

    TEST_INFO("E7 rank %d: WaitSignalZeroDesc done (result=%d).", myRank, static_cast<int>(res));
}

// ============================================================================
// M1 — TwoCommunicatorsIndependentWindows
// ============================================================================

/**
 * @test HostApiTest.TwoCommunicatorsIndependentWindows
 * @brief Two independent communicators each own a window; PUTs on each must not interfere.
 *
 * The fixture's SetUp creates communicator 1.  This test creates communicator 2
 * manually.  Each communicator registers its own buffer/window, rank 0 PUTs to
 * rank 1 through each independently (different seeds), rank 1 waits and verifies
 * both receive regions.
 */
TEST_F(HostApiTest, TwoCommunicatorsIndependentWindows)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank  = rank();
    ncclComm_t   comm1   = getActiveCommunicator();
    hipStream_t  stream1 = getActiveStream();

    // ---- Create a second communicator manually ----
    ncclUniqueId id2{};
    if(myRank == 0)
    {
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&id2));
    }
    MPI_Bcast(&id2, sizeof(id2), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t comm2 = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommInitRank(&comm2, nRanks(), id2, myRank));
    auto comm2Guard = makeScopeGuard([&]() { if(comm2) ncclCommDestroy(comm2); });

    hipStream_t stream2 = nullptr;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream2));
    auto stream2Guard = makeScopeGuard([&]() { if(stream2) hipStreamDestroy(stream2); });

    // ---- Comm 1: buffer, window, PUT ----
    void* buf1 = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&buf1, kOneMB));
    auto buf1Guard = makeScopeGuard([&]() { freeFineGrainBuffer(buf1); });

    ncclWindow_t win1 = nullptr;
    NcclWindowGuard wg1(comm1, buf1, kOneMB, &win1, kWinMode);
    ASSERT_MPI_NE(win1, nullptr);
    ASSERT_MPI_EQ(ncclSuccess, wg1.initResult());

    // ---- Comm 2: buffer, window, PUT ----
    void* buf2 = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&buf2, kOneMB));
    auto buf2Guard = makeScopeGuard([&]() { freeFineGrainBuffer(buf2); });

    ncclWindow_t win2 = nullptr;
    NcclWindowGuard wg2(comm2, buf2, kOneMB, &win2, kWinMode);
    ASSERT_MPI_NE(win2, nullptr);
    ASSERT_MPI_EQ(ncclSuccess, wg2.initResult());

    const int kSeed1 = 42;
    const int kSeed2 = 99;

    // Rank 0: PUT through comm1 and comm2
    ncclResult_t put1Res = ncclSuccess;
    ncclResult_t put2Res = ncclSuccess;
    if(myRank == 0)
    {
        void* src1 = static_cast<uint8_t*>(buf1) + kSendOffset;
        FillBuf(src1, kTransferSize, kSeed1);
        put1Res = ncclPutSignal(src1, kTransferSize, ncclUint8,
                                /*peer=*/1, win1, kRecvOffset,
                                kSigIdx, kCtx, kFlags, comm1, stream1);

        void* src2 = static_cast<uint8_t*>(buf2) + kSendOffset;
        FillBuf(src2, kTransferSize, kSeed2);
        put2Res = ncclPutSignal(src2, kTransferSize, ncclUint8,
                                /*peer=*/1, win2, kRecvOffset,
                                kSigIdx, kCtx, kFlags, comm2, stream2);
    }
    ASSERT_MPI_EQ(ncclSuccess, put1Res);
    ASSERT_MPI_EQ(ncclSuccess, put2Res);

    // Rank 1: wait on each communicator independently
    ncclResult_t wait1Res = ncclSuccess;
    ncclResult_t wait2Res = ncclSuccess;
    if(myRank == 1)
    {
        ncclWaitSignalDesc_t d1{1, 0, kSigIdx, kCtx};
        wait1Res = ncclWaitSignal(1, &d1, comm1, stream1);

        ncclWaitSignalDesc_t d2{1, 0, kSigIdx, kCtx};
        wait2Res = ncclWaitSignal(1, &d2, comm2, stream2);
    }
    ASSERT_MPI_EQ(ncclSuccess, wait1Res);
    ASSERT_MPI_EQ(ncclSuccess, wait2Res);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream1));
    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream2));

    bool allOk = true;
    if(myRank == 1)
    {
        allOk = VerifyBuf(static_cast<uint8_t*>(buf1) + kRecvOffset, kTransferSize, kSeed1) &&
                VerifyBuf(static_cast<uint8_t*>(buf2) + kRecvOffset, kTransferSize, kSeed2);
    }
    ASSERT_MPI_TRUE(allOk);

    TEST_INFO("M1 rank %d: TwoCommunicatorsIndependentWindows passed.", myRank);
}

// ============================================================================
// M2 — StressManySmallPuts
// ============================================================================

/**
 * @test HostApiTest.StressManySmallPuts
 * @brief 100 small (64-byte) PUTs to distinct offsets; rank 1 waits with opCnt=100.
 *
 * Stress-tests the signal counter path and many-operation queueing.
 * Each PUT uses seed=i so every 64-byte slot carries a distinguishable pattern.
 */
TEST_F(HostApiTest, StressManySmallPuts)
{
    if(!validateTestPrerequisites(/*min=*/2, /*max=*/2))
    {
        GTEST_SKIP() << "Need exactly 2 MPI processes";
    }

    const int    myRank   = rank();
    ncclComm_t   comm     = getActiveCommunicator();
    hipStream_t  stream   = getActiveStream();
    const int    kNumPuts = 100;
    const size_t kPutSize = 64;
    const size_t kRecvArea = static_cast<size_t>(kNumPuts) * kPutSize;
    const size_t kSendSlot = kRecvArea;
    const size_t kWinSize  = kRecvArea + kPutSize;

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kWinSize));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kWinSize, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    ncclResult_t putRes = ncclSuccess;
    if(myRank == 0)
    {
        uint8_t* sendRegion = static_cast<uint8_t*>(winBuf) + kSendSlot;
        for(int i = 0; i < kNumPuts && putRes == ncclSuccess; ++i)
        {
            FillBuf(sendRegion, kPutSize, /*seed=*/i);
            size_t peerOff = static_cast<size_t>(i) * kPutSize;
            putRes = ncclPutSignal(sendRegion, kPutSize, ncclUint8,
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
            size_t off = static_cast<size_t>(i) * kPutSize;
            if(!VerifyBuf(base + off, kPutSize, /*seed=*/i))
            {
                allOk = false;
                break;
            }
        }
    }
    ASSERT_MPI_TRUE(allOk);

    TEST_INFO("M2 rank %d: StressManySmallPuts passed (%d x %zu bytes).",
              myRank, kNumPuts, kPutSize);
}

// ============================================================================
// P6 — PutToSelf
// ============================================================================

/**
 * @test HostApiTest.PutToSelf
 * @brief Rank 0 PUTs to its own window (peer = self), testing the loopback path.
 *
 * All ranks register a window (collective), but only rank 0 performs the
 * PUT + WaitSignal cycle targeting itself.  Other ranks idle but participate
 * in collective assertions (ASSERT_MPI_*) to avoid deadlock.
 */
TEST_F(HostApiTest, PutToSelf)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();
    hipStream_t  stream = getActiveStream();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    NcclWindowGuard wg(comm, winBuf, kOneMB, &win, kWinMode);
    ASSERT_MPI_NE(win, nullptr);
    ASSERT_MPI_EQ(ncclSuccess, wg.initResult());

    const int kSelfSeed = 77;

    ncclResult_t putRes = ncclSuccess;
    if(myRank == 0)
    {
        void* sendRegion = static_cast<uint8_t*>(winBuf) + kSendOffset;
        FillBuf(sendRegion, kTransferSize, kSelfSeed);
        putRes = ncclPutSignal(sendRegion, kTransferSize, ncclUint8,
                               /*peer=*/0, win, kRecvOffset,
                               kSigIdx, kCtx, kFlags, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, putRes);

    ncclResult_t waitRes = ncclSuccess;
    if(myRank == 0)
    {
        ncclWaitSignalDesc_t desc{1, 0, kSigIdx, kCtx};
        waitRes = ncclWaitSignal(1, &desc, comm, stream);
    }
    ASSERT_MPI_EQ(ncclSuccess, waitRes);

    ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream));

    bool ok = (myRank != 0) ||
              VerifyBuf(static_cast<uint8_t*>(winBuf) + kRecvOffset, kTransferSize, kSelfSeed);
    ASSERT_MPI_TRUE(ok);

    TEST_INFO("P6 rank %d: PutToSelf passed.", myRank);
}

// ============================================================================
// W3 — DoubleDeregister
// ============================================================================

/**
 * @test HostApiTest.DoubleDeregister
 * @brief Calling ncclCommWindowDeregister twice on the same handle must not crash.
 *
 * The first deregister should succeed.  The second should return an error
 * (or at least not ncclSuccess).  No NcclWindowGuard is used because we
 * need manual control over the deregister calls.
 */
TEST_F(HostApiTest, DoubleDeregister)
{
    if(!validateTestPrerequisites(/*min=*/2))
    {
        GTEST_SKIP() << "Need at least 2 MPI processes";
    }

    const int    myRank = rank();
    ncclComm_t   comm   = getActiveCommunicator();

    void* winBuf = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, allocFineGrainBuffer(&winBuf, kOneMB));
    auto winBufGuard = makeScopeGuard([&]() { freeFineGrainBuffer(winBuf); });

    ncclWindow_t win = nullptr;
    ASSERT_MPI_EQ(ncclSuccess,
                  ncclCommWindowRegister(comm, winBuf, kOneMB, &win, kWinMode));
    ASSERT_MPI_NE(win, nullptr);

    // First deregister — should succeed.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowDeregister(comm, win));

    // Second deregister on the stale handle — expect failure (not ncclSuccess).
    ncclResult_t res2 = ncclCommWindowDeregister(comm, win);
    ASSERT_MPI_NE(ncclSuccess, res2);

    TEST_INFO("W3 rank %d: DoubleDeregister passed (second deregister returned %d).",
              myRank, static_cast<int>(res2));
}

} // namespace RcclUnitTesting

#endif // MPI_TESTS_ENABLED
