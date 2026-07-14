/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file SymmetricWindowMPITests.cpp
 * @brief Tests for relaxed symmetric buffer registration (one-buffer path)
 *
 * Validates that symmetric kernels work correctly across different window
 * registration patterns:
 * - Both send and recv buffers registered (baseline)
 * - Only one buffer registered (relaxed / one-buffer path)
 * - No buffers registered (non-symmetric fallback)
 * - In-place operations with a single window
 *
 * REQUIRED Environment Variables:
 *   NCCL_CUMEM_ENABLE=1          Enables cuMem API for symmetric support
 *   NCCL_DEBUG=INFO              Enables debug logging to observe kernel path
 *   HSA_NO_SCRATCH_RECLAIM=1     Required for multi-GPU RCCL tests
 *
 * Run examples:
 *   mpirun -np 2 --bind-to none ./rccl-UnitTestsMPI --gtest_filter=SymWin_*
 *   mpirun -np 8 --bind-to none -x NCCL_DEBUG=INFO \
 *     ./rccl-UnitTestsMPI --gtest_filter=SymWin_AllReduce.*
 */

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include <cstdlib>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace {
    constexpr size_t DEFAULT_COUNT = 256 * 1024;
    constexpr int MIN_RANKS = 2;
}

// ============================================================================
// Base class for symmetric window tests
// ============================================================================

class SymmetricWindowTestBase : public MPITestBase
{
protected:
    struct NcclBufInfo {
        void* ptr = nullptr;
        size_t size = 0;
    };

    struct WinInfo {
        ncclWindow_t win = nullptr;
        ncclComm_t comm = nullptr;
    };

    std::vector<NcclBufInfo> allocatedBufs_;
    std::vector<WinInfo> registeredWins_;

    void SetUp() override
    {
        MPITestBase::SetUp();
    }

    void TearDown() override
    {
        for (auto it = registeredWins_.rbegin(); it != registeredWins_.rend(); ++it) {
            if (it->win && it->comm) {
                ncclCommWindowDeregister(it->comm, it->win);
            }
        }
        registeredWins_.clear();

        for (auto it = allocatedBufs_.rbegin(); it != allocatedBufs_.rend(); ++it) {
            if (it->ptr) {
                ncclMemFree(it->ptr);
            }
        }
        allocatedBufs_.clear();

        MPITestBase::TearDown();
    }

    void* allocNcclBuf(size_t size)
    {
        void* ptr = nullptr;
        ncclResult_t res = ncclMemAlloc(&ptr, size);
        if (res != ncclSuccess || ptr == nullptr) return nullptr;
        allocatedBufs_.push_back({ptr, size});
        return ptr;
    }

    ncclWindow_t registerWindow(ncclComm_t comm, void* buf, size_t size,
                                int flags = NCCL_WIN_COLL_SYMMETRIC)
    {
        ncclWindow_t win = nullptr;
        ncclResult_t res = ncclCommWindowRegister(comm, buf, size, &win, flags);
        if (res != ncclSuccess) return nullptr;
        registeredWins_.push_back({win, comm});
        return win;
    }

    bool setupForSymmetric(int minRanks = MIN_RANKS)
    {
        const char* cuMemEnv = std::getenv("NCCL_CUMEM_ENABLE");
        if (!cuMemEnv || std::string(cuMemEnv) != "1") return false;

        if (!validateTestPrerequisites(minRanks)) return false;
        if (createTestCommunicator() != ncclSuccess) return false;

        // Verify ncclMemAlloc works (proxy check for VMM/symmetric support)
        void* testBuf = nullptr;
        ncclResult_t res = ncclMemAlloc(&testBuf, 4096);
        if (res != ncclSuccess || testBuf == nullptr) return false;
        ncclMemFree(testBuf);

        return true;
    }

    template<typename T>
    void initSendBuffer(void* buffer, size_t count, int rank)
    {
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(buffer, count,
            [rank](size_t) { return static_cast<T>(static_cast<float>(rank + 1)); }));
    }

    template<typename T>
    bool checkAllReduceResult(void* buffer, size_t count, int nRanks)
    {
        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2));
        return verifyBufferData<T>(buffer, count, [expected](size_t) { return expected; });
    }

    template<typename T>
    bool checkReduceScatterResult(void* buffer, size_t count, int nRanks)
    {
        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2));
        return verifyBufferData<T>(buffer, count, [expected](size_t) { return expected; });
    }

    template<typename T>
    bool checkAllGatherResult(void* buffer, size_t countPerRank, int nRanks)
    {
        return verifyBufferData<T>(buffer, countPerRank * nRanks,
            [countPerRank](size_t i) {
                int srcRank = i / countPerRank;
                return static_cast<T>(static_cast<float>(srcRank + 1));
            });
    }
};

// ============================================================================
// AllReduce with symmetric windows
// ============================================================================

class SymWin_AllReduce : public SymmetricWindowTestBase {};

TEST_F(SymWin_AllReduce, BothWindows_OutOfPlace)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t count = DEFAULT_COUNT;
    const size_t bufSize = count * sizeof(T);

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    void* sendBuf = allocNcclBuf(bufSize);
    void* recvBuf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    ncclWindow_t sendWin = registerWindow(comm, sendBuf, bufSize);
    ncclWindow_t recvWin = registerWindow(comm, recvBuf, bufSize);
    ASSERT_MPI_NE(sendWin, nullptr);
    ASSERT_MPI_NE(recvWin, nullptr);

    initSendBuffer<T>(sendBuf, count, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllReduceResult<T>(recvBuf, count, nRanks));
    TEST_INFO("Rank %d: BothWindows_OutOfPlace passed", rank);
}

TEST_F(SymWin_AllReduce, OnlySendWindow_OutOfPlace)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t count = DEFAULT_COUNT;
    const size_t bufSize = count * sizeof(T);

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    void* sendBuf = allocNcclBuf(bufSize);
    void* recvBuf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    // Register ONLY send buffer
    ncclWindow_t sendWin = registerWindow(comm, sendBuf, bufSize);
    ASSERT_MPI_NE(sendWin, nullptr);
    // recvBuf intentionally NOT registered

    initSendBuffer<T>(sendBuf, count, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllReduceResult<T>(recvBuf, count, nRanks));
    TEST_INFO("Rank %d: OnlySendWindow_OutOfPlace passed (relaxed path)", rank);
}

TEST_F(SymWin_AllReduce, OnlyRecvWindow_OutOfPlace)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t count = DEFAULT_COUNT;
    const size_t bufSize = count * sizeof(T);

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    void* sendBuf = allocNcclBuf(bufSize);
    void* recvBuf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    // Register ONLY recv buffer
    // sendBuf intentionally NOT registered
    ncclWindow_t recvWin = registerWindow(comm, recvBuf, bufSize);
    ASSERT_MPI_NE(recvWin, nullptr);

    initSendBuffer<T>(sendBuf, count, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllReduceResult<T>(recvBuf, count, nRanks));
    TEST_INFO("Rank %d: OnlyRecvWindow_OutOfPlace passed (relaxed path)", rank);
}

TEST_F(SymWin_AllReduce, NoWindows_OutOfPlace)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t count = DEFAULT_COUNT;
    const size_t bufSize = count * sizeof(T);

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    void* sendBuf = allocNcclBuf(bufSize);
    void* recvBuf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    // No window registration — non-symmetric fallback path
    initSendBuffer<T>(sendBuf, count, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllReduceResult<T>(recvBuf, count, nRanks));
    TEST_INFO("Rank %d: NoWindows_OutOfPlace passed (non-symmetric fallback)", rank);
}

TEST_F(SymWin_AllReduce, SingleWindow_InPlace)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t count = DEFAULT_COUNT;
    const size_t bufSize = count * sizeof(T);

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    void* buf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(buf, nullptr);

    // Single window registration, in-place operation — primary one-buffer use case
    ncclWindow_t win = registerWindow(comm, buf, bufSize);
    ASSERT_MPI_NE(win, nullptr);

    initSendBuffer<T>(buf, count, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllReduce(buf, buf, count, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllReduceResult<T>(buf, count, nRanks));
    TEST_INFO("Rank %d: SingleWindow_InPlace passed (one-buffer path)", rank);
}

// ============================================================================
// ReduceScatter with symmetric windows
// ============================================================================

class SymWin_ReduceScatter : public SymmetricWindowTestBase {};

TEST_F(SymWin_ReduceScatter, BothWindows)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t countPerRank = DEFAULT_COUNT;

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t sendSize = countPerRank * nRanks * sizeof(T);
    const size_t recvSize = countPerRank * sizeof(T);

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    ncclWindow_t sendWin = registerWindow(comm, sendBuf, sendSize);
    ncclWindow_t recvWin = registerWindow(comm, recvBuf, recvSize);
    ASSERT_MPI_NE(sendWin, nullptr);
    ASSERT_MPI_NE(recvWin, nullptr);

    initSendBuffer<T>(sendBuf, countPerRank * nRanks, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclReduceScatter(sendBuf, recvBuf, countPerRank, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkReduceScatterResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: ReduceScatter BothWindows passed", rank);
}

TEST_F(SymWin_ReduceScatter, OnlySendWindow)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t countPerRank = DEFAULT_COUNT;

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t sendSize = countPerRank * nRanks * sizeof(T);
    const size_t recvSize = countPerRank * sizeof(T);

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    // Only send buffer registered
    ncclWindow_t sendWin = registerWindow(comm, sendBuf, sendSize);
    ASSERT_MPI_NE(sendWin, nullptr);

    initSendBuffer<T>(sendBuf, countPerRank * nRanks, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclReduceScatter(sendBuf, recvBuf, countPerRank, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkReduceScatterResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: ReduceScatter OnlySendWindow passed (relaxed path)", rank);
}

TEST_F(SymWin_ReduceScatter, NoWindows)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t countPerRank = DEFAULT_COUNT;

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t sendSize = countPerRank * nRanks * sizeof(T);
    const size_t recvSize = countPerRank * sizeof(T);

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    initSendBuffer<T>(sendBuf, countPerRank * nRanks, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclReduceScatter(sendBuf, recvBuf, countPerRank, ncclFloat, ncclSum, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkReduceScatterResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: ReduceScatter NoWindows passed (non-symmetric fallback)", rank);
}

// ============================================================================
// AllGather with symmetric windows
// ============================================================================

class SymWin_AllGather : public SymmetricWindowTestBase {};

TEST_F(SymWin_AllGather, BothWindows)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t countPerRank = DEFAULT_COUNT;

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t sendSize = countPerRank * sizeof(T);
    const size_t recvSize = countPerRank * nRanks * sizeof(T);

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    ncclWindow_t sendWin = registerWindow(comm, sendBuf, sendSize);
    ncclWindow_t recvWin = registerWindow(comm, recvBuf, recvSize);
    ASSERT_MPI_NE(sendWin, nullptr);
    ASSERT_MPI_NE(recvWin, nullptr);

    initSendBuffer<T>(sendBuf, countPerRank, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllGather(sendBuf, recvBuf, countPerRank, ncclFloat, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllGatherResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: AllGather BothWindows passed", rank);
}

TEST_F(SymWin_AllGather, OnlyRecvWindow)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t countPerRank = DEFAULT_COUNT;

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t sendSize = countPerRank * sizeof(T);
    const size_t recvSize = countPerRank * nRanks * sizeof(T);

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    // Only recv buffer registered
    ncclWindow_t recvWin = registerWindow(comm, recvBuf, recvSize);
    ASSERT_MPI_NE(recvWin, nullptr);

    initSendBuffer<T>(sendBuf, countPerRank, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllGather(sendBuf, recvBuf, countPerRank, ncclFloat, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllGatherResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: AllGather OnlyRecvWindow passed (relaxed path)", rank);
}

TEST_F(SymWin_AllGather, NoWindows)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    using T = float;
    const size_t countPerRank = DEFAULT_COUNT;

    ncclComm_t comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int rank, nRanks;
    ncclCommUserRank(comm, &rank);
    ncclCommCount(comm, &nRanks);

    const size_t sendSize = countPerRank * sizeof(T);
    const size_t recvSize = countPerRank * nRanks * sizeof(T);

    void* sendBuf = allocNcclBuf(sendSize);
    void* recvBuf = allocNcclBuf(recvSize);
    ASSERT_MPI_NE(sendBuf, nullptr);
    ASSERT_MPI_NE(recvBuf, nullptr);

    initSendBuffer<T>(sendBuf, countPerRank, rank);

    ASSERT_MPI_EQ(ncclSuccess,
        ncclAllGather(sendBuf, recvBuf, countPerRank, ncclFloat, comm, stream));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream));

    ASSERT_TRUE(checkAllGatherResult<T>(recvBuf, countPerRank, nRanks));
    TEST_INFO("Rank %d: AllGather NoWindows passed (non-symmetric fallback)", rank);
}

// ============================================================================
// Window lifecycle tests
// ============================================================================

class SymWin_WindowLifecycle : public SymmetricWindowTestBase {};

TEST_F(SymWin_WindowLifecycle, RegisterDeregister_Basic)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    int rank;
    ncclCommUserRank(comm, &rank);

    const size_t bufSize = 1024 * 1024;
    void* buf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(buf, nullptr);

    ncclWindow_t win = registerWindow(comm, buf, bufSize);
    ASSERT_MPI_NE(win, nullptr);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowDeregister(comm, win));

    // Null out tracked entry so TearDown skips double-deregister
    registeredWins_.back().win = nullptr;

    TEST_INFO("Rank %d: RegisterDeregister_Basic passed", rank);
}

TEST_F(SymWin_WindowLifecycle, MultipleWindows)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    int rank;
    ncclCommUserRank(comm, &rank);

    const int numWindows = 4;
    const size_t bufSize = 256 * 1024;
    std::vector<ncclWindow_t> wins(numWindows);

    for (int i = 0; i < numWindows; i++) {
        void* buf = allocNcclBuf(bufSize);
        ASSERT_MPI_NE(buf, nullptr);

        wins[i] = registerWindow(comm, buf, bufSize);
        ASSERT_MPI_NE(wins[i], nullptr);
    }

    // Verify all windows are unique
    for (int i = 0; i < numWindows; i++) {
        for (int j = i + 1; j < numWindows; j++) {
            ASSERT_MPI_NE(wins[i], wins[j]);
        }
    }

    TEST_INFO("Rank %d: MultipleWindows passed (%d windows)", rank, numWindows);
}

TEST_F(SymWin_WindowLifecycle, RepeatedRegisterDeregister)
{
    if (!setupForSymmetric()) {
        GTEST_SKIP() << "Requires symmetric support with 2+ ranks";
    }

    ncclComm_t comm = getActiveCommunicator();
    int rank;
    ncclCommUserRank(comm, &rank);

    const size_t bufSize = 512 * 1024;
    void* buf = allocNcclBuf(bufSize);
    ASSERT_MPI_NE(buf, nullptr);

    const int iterations = 3;
    for (int i = 0; i < iterations; i++) {
        ncclWindow_t win = nullptr;
        ASSERT_MPI_EQ(ncclSuccess,
            ncclCommWindowRegister(comm, buf, bufSize, &win, NCCL_WIN_COLL_SYMMETRIC));
        ASSERT_MPI_NE(win, nullptr);

        ASSERT_MPI_EQ(ncclSuccess, ncclCommWindowDeregister(comm, win));
    }

    TEST_INFO("Rank %d: RepeatedRegisterDeregister passed (%d iterations)",
              rank, iterations);
}

#endif // MPI_TESTS_ENABLED
