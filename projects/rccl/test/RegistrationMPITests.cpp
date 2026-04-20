/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file RegistrationMPITests.cpp
 * @brief Tests for buffer registration functionality in RCCL
 *
 * This file contains tests for:
 * 1. User Buffer Registration (UBR) - explicit buffer registration via ncclCommRegister
 * 2. Graph Capture Registration - automatic buffer registration during HIP graph capture
 *
 * REQUIRED Environment Variables:
 *   NCCL_DEBUG=INFO              Enable debug logging
 *   NCCL_DEBUG_SUBSYS=REG        Enable REG subsystem logging
 *   NCCL_LOCAL_REGISTER=1        Enable local buffer registration (UBR tests)
 *   NCCL_GRAPH_REGISTER=1        Enable graph buffer registration (Graph tests)
 *   RCCL_MPI_LOG_ALL_RANKS=1     Enable per-rank logging for log verification
 *
 * Run examples:
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=UBR_*
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=GraphCapture_*
 */

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include <cstdlib>
#include <sstream>
#include <fstream>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

// Test Configuration
namespace RegTestConfig {
    constexpr size_t SMALL_COUNT  = 1024;           // 4KB for float
    constexpr size_t MEDIUM_COUNT = 256 * 1024;     // 1MB for float
    constexpr size_t LARGE_COUNT  = 1024 * 1024;    // 4MB for float

    using DefaultType = hip_bfloat16;

    constexpr int MIN_RANKS_DEFAULT    = 2;
    constexpr int MIN_RANKS_ALLTOALL   = 4;
    constexpr int MIN_NODES_MULTINODE  = 2;
}

// REG Log Checker - Pattern checking for registration debug output
class REGLogChecker
{
public:
    explicit REGLogChecker(const std::string& logContent)
        : m_content(logContent) {}

    bool hasIPCRegistration() const
    {
        return hasPattern("IPC register buffer") ||
               hasPattern("IPC registering buffer") ||
               (hasPattern("Proxy rank") && hasPattern("register success"));
    }

    bool hasIPCReuse() const
    {
        return hasPattern("IPC reuse buffer");
    }

    bool hasNETRegistration() const
    {
        return hasPattern("NET register userbuff") ||
               hasPattern("NET reuse buffer");
    }

    bool hasAnyRegistrationSuccess() const
    {
        return hasIPCRegistration() || hasIPCReuse() || hasNETRegistration();
    }

    bool hasIPCFailure() const
    {
        return hasPattern("failed to IPC register") ||
               hasPattern("legacy IPC blocked");
    }

    std::string getSummary() const
    {
        std::ostringstream ss;
        ss << "REG Log: ";
        if (hasIPCReuse()) ss << "[IPC-REUSE] ";
        if (hasIPCRegistration()) ss << "[IPC-REG] ";
        if (hasNETRegistration()) ss << "[NET-REG] ";
        if (hasIPCFailure()) ss << "[IPC-FAIL] ";
        if (!hasAnyRegistrationSuccess() && !hasIPCFailure()) ss << "[NO-REG]";
        return ss.str();
    }

    size_t getContentLength() const { return m_content.size(); }

private:
    bool hasPattern(const std::string& pattern) const
    {
        return m_content.find(pattern) != std::string::npos;
    }

    std::string m_content;
};

// Registration Test Base Class
class RegistrationTestBase : public MPITestBase
{
protected:
    struct RegInfo {
        void* buffer = nullptr;
        void* handle = nullptr;
        size_t size = 0;
        bool registered = false;
    };

    // Buffer Management
    RegInfo allocateAndRegister(size_t size)
    {
        RegInfo info;
        info.size = size;

        if (hipMalloc(&info.buffer, size) != hipSuccess) {
            return info;
        }

        ncclResult_t result = ncclCommRegister(getActiveCommunicator(),
                                                info.buffer, size, &info.handle);
        info.registered = (result == ncclSuccess && info.handle != nullptr);

        return info;
    }

    void cleanupRegInfo(RegInfo& info)
    {
        if (info.handle) {
            ncclCommDeregister(getActiveCommunicator(), info.handle);
            info.handle = nullptr;
        }
        if (info.buffer) {
            (void)hipFree(info.buffer);
            info.buffer = nullptr;
        }
        info.registered = false;
    }

    // Test Setup
    bool setupMultiNode(int minRanks = 2, int minNodes = 2)
    {
        int nodeCount = MPITestConstants::detectNodeCount();
        if (nodeCount < minNodes) {
            return false;
        }
        if (!validateTestPrerequisites(minRanks, kNoProcessLimit,
                                        kNoPowerOfTwoRequired, minNodes, kNoNodeLimit)) {
            return false;
        }
        return (createTestCommunicator() == ncclSuccess);
    }

    // Environment Checks
    bool isUBREnabled()
    {
        const char* localReg = getenv("NCCL_LOCAL_REGISTER");
        return (localReg && std::string(localReg) == "1");
    }

    bool isGraphRegisterEnabled()
    {
        const char* graphReg = getenv("NCCL_GRAPH_REGISTER");
        return (graphReg && std::string(graphReg) == "1");
    }

    bool isPerRankLoggingEnabled()
    {
        const char* env = getenv("RCCL_MPI_LOG_ALL_RANKS");
        return (env && std::string(env) == "1");
    }

    // Log File Access
    std::string readRankLogFile()
    {
        std::string logPath = MPIHelpers::getRankLogFilePath(getTestMpiRank());
        std::ifstream file(logPath);
        if (!file.is_open()) {
            TEST_WARN("Could not open rank log file: %s", logPath.c_str());
            return "";
        }
        std::ostringstream ss;
        ss << file.rdbuf();
        return ss.str();
    }

    REGLogChecker getLogChecker()
    {
        return REGLogChecker(readRankLogFile());
    }

    // Data Initialization and Verification
    template<typename T>
    void initSendBuffer(void* buffer, size_t count, int rank)
    {
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(buffer, count,
            [rank](size_t) { return static_cast<T>(static_cast<float>(rank + 1)); }));
    }

    template<typename T>
    bool verifyAllReduceResult(void* buffer, size_t count, int nRanks)
    {
        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2));
        return verifyBufferData<T>(buffer, count, [expected](size_t) { return expected; });
    }

    template<typename T>
    bool verifyReduceScatterResult(void* buffer, size_t count, int nRanks)
    {
        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2));
        return verifyBufferData<T>(buffer, count, [expected](size_t) { return expected; });
    }

    template<typename T>
    bool verifyAllGatherResult(void* buffer, size_t countPerRank, int nRanks)
    {
        return verifyBufferData<T>(buffer, countPerRank * nRanks,
            [countPerRank](size_t i) {
                int srcRank = i / countPerRank;
                return static_cast<T>(static_cast<float>(srcRank + 1));
            });
    }

    template<typename T>
    bool verifyBroadcastResult(void* buffer, size_t count, T rootValue)
    {
        return verifyBufferData<T>(buffer, count,
            [rootValue](size_t) { return rootValue; });
    }
};

// ============================================================================
// User Buffer Registration (UBR) Tests
// ============================================================================

class UBR_AllReduce : public RegistrationTestBase {};

TEST_F(UBR_AllReduce, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_DEFAULT, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    using T = RegTestConfig::DefaultType;
    const size_t count = RegTestConfig::MEDIUM_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(count * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_MPI_NE(sendInfo.buffer, nullptr);
    ASSERT_MPI_NE(recvInfo.buffer, nullptr);
    ASSERT_MPI_NE(sendInfo.handle, nullptr);
    ASSERT_MPI_NE(recvInfo.handle, nullptr);

    initSendBuffer<T>(sendInfo.buffer, count, rank);

    ncclResult_t result = ncclAllReduce(sendInfo.buffer, recvInfo.buffer, count,
                                         getNcclDataType<T>(), ncclSum,
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, result);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    ASSERT_TRUE(verifyAllReduceResult<T>(recvInfo.buffer, count, nRanks));
}

class UBR_AllGather : public RegistrationTestBase {};

TEST_F(UBR_AllGather, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_DEFAULT, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    using T = RegTestConfig::DefaultType;
    const size_t countPerRank = RegTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(countPerRank * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(countPerRank * nRanks * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_MPI_NE(sendInfo.buffer, nullptr);
    ASSERT_MPI_NE(recvInfo.buffer, nullptr);
    ASSERT_MPI_NE(sendInfo.handle, nullptr);
    ASSERT_MPI_NE(recvInfo.handle, nullptr);

    initSendBuffer<T>(sendInfo.buffer, countPerRank, rank);

    ncclResult_t result = ncclAllGather(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                         getNcclDataType<T>(),
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, result);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    ASSERT_TRUE(verifyAllGatherResult<T>(recvInfo.buffer, countPerRank, nRanks));
}

class UBR_ReduceScatter : public RegistrationTestBase {};

TEST_F(UBR_ReduceScatter, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_DEFAULT, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    using T = RegTestConfig::DefaultType;
    const size_t countPerRank = RegTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(countPerRank * nRanks * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(countPerRank * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_MPI_NE(sendInfo.buffer, nullptr);
    ASSERT_MPI_NE(recvInfo.buffer, nullptr);
    ASSERT_MPI_NE(sendInfo.handle, nullptr);
    ASSERT_MPI_NE(recvInfo.handle, nullptr);

    initSendBuffer<T>(sendInfo.buffer, countPerRank * nRanks, rank);

    ncclResult_t result = ncclReduceScatter(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                             getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, result);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    ASSERT_TRUE(verifyReduceScatterResult<T>(recvInfo.buffer, countPerRank, nRanks));
}

class UBR_Broadcast : public RegistrationTestBase {};

TEST_F(UBR_Broadcast, NonZeroRoot_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes with 4+ ranks";
    }

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    using T = RegTestConfig::DefaultType;
    const size_t count = RegTestConfig::MEDIUM_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const int root = nRanks - 1;

    RegInfo bufInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(bufInfo);
    });

    ASSERT_MPI_NE(bufInfo.buffer, nullptr);
    ASSERT_MPI_NE(bufInfo.handle, nullptr);

    const T rootValue = static_cast<T>(99.0f);
    if (rank == root) {
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(bufInfo.buffer, count,
            [rootValue](size_t) { return rootValue; }));
    } else {
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(bufInfo.buffer, count,
            [](size_t) { return static_cast<T>(0.0f); }));
    }

    ncclResult_t result = ncclBroadcast(bufInfo.buffer, bufInfo.buffer, count,
                                         getNcclDataType<T>(), root,
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, result);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    ASSERT_TRUE(verifyBroadcastResult<T>(bufInfo.buffer, count, rootValue));
}

class UBR_AllToAll : public RegistrationTestBase {};

TEST_F(UBR_AllToAll, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes with 4+ ranks";
    }

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    using T = RegTestConfig::DefaultType;
    const size_t countPerRank = RegTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t totalCount = countPerRank * nRanks;

    RegInfo sendInfo = allocateAndRegister(totalCount * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(totalCount * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_MPI_NE(sendInfo.buffer, nullptr);
    ASSERT_MPI_NE(recvInfo.buffer, nullptr);
    ASSERT_MPI_NE(sendInfo.handle, nullptr);
    ASSERT_MPI_NE(recvInfo.handle, nullptr);

    ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(sendInfo.buffer, totalCount,
        [rank, countPerRank](size_t i) {
            int destRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(rank * 100 + destRank));
        }));

    ncclResult_t result = ncclAllToAll(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                        getNcclDataType<T>(),
                                        getActiveCommunicator(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, result);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    bool verified = verifyBufferData<T>(recvInfo.buffer, totalCount,
        [rank, countPerRank](size_t i) {
            int srcRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(srcRank * 100 + rank));
        });
    ASSERT_TRUE(verified);
}

class UBR_SendRecv : public RegistrationTestBase {};

TEST_F(UBR_SendRecv, RingPattern_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_DEFAULT, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes for SendRecv UBR";
    }

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    using T = RegTestConfig::DefaultType;
    const size_t count = RegTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(count * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_MPI_NE(sendInfo.buffer, nullptr);
    ASSERT_MPI_NE(recvInfo.buffer, nullptr);
    ASSERT_MPI_NE(sendInfo.handle, nullptr);
    ASSERT_MPI_NE(recvInfo.handle, nullptr);

    int sendPeer = (rank + 1) % nRanks;
    int recvPeer = (rank - 1 + nRanks) % nRanks;

    initSendBuffer<T>(sendInfo.buffer, count, rank);

    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    ncclResult_t sendResult = ncclSend(sendInfo.buffer, count, getNcclDataType<T>(),
                                        sendPeer, getActiveCommunicator(), getActiveStream());
    ncclResult_t recvResult = ncclRecv(recvInfo.buffer, count, getNcclDataType<T>(),
                                        recvPeer, getActiveCommunicator(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());

    ASSERT_EQ(ncclSuccess, sendResult);
    ASSERT_EQ(ncclSuccess, recvResult);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    T expected = static_cast<T>(static_cast<float>(recvPeer + 1));
    bool verified = verifyBufferData<T>(recvInfo.buffer, count,
        [expected](size_t) { return expected; });
    ASSERT_TRUE(verified);
}

// ============================================================================
// UBR Concurrent Registration Hang Reproduction Test
// ============================================================================

/**
 * @brief Test to reproduce UBR hang with multiple separate buffer registrations
 *
 * This test reproduces a hang that occurs when:
 * 1. Multiple SEPARATE buffers are allocated (not views of a contiguous buffer)
 * 2. Each buffer is registered with ncclCommRegister
 * 3. All buffers are used inside ncclGroupStart/ncclGroupEnd with send/recv
 *
 * The hang is caused by concurrent blocking IPC registration calls
 * (ncclProxyCallBlocking) inside the grouped operation creating circular
 * wait conditions.
 *
 * ROOT CAUSE:
 * - ipcRegisterBuffer() in p2p.cc calls ncclProxyCallBlocking() which is a
 *   synchronous RPC that blocks waiting for peer to import the IPC handle
 * - When all ranks simultaneously try to register multiple different buffers
 *   with each other inside a group, they create circular dependencies
 * - Rank 0 waits for Rank 1 to import buffer A, while Rank 1 waits for Rank 0
 *   to import buffer B -> DEADLOCK
 *
 * EXPECTED BEHAVIOR:
 * - SeparateBuffers_Grouped: HANGS (demonstrates the bug)
 * - ContiguousBuffer_Grouped: WORKS (demonstrates the fix)
 *
 * To run this test with a timeout to detect the hang:
 *   timeout 60 mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter=UBR_ConcurrentRegHang*
 */
class UBR_ConcurrentRegHang : public RegistrationTestBase
{
protected:
    using T = RegTestConfig::DefaultType;
    static constexpr size_t ELEMENTS_PER_PEER = 1024;  // Small size to trigger hang quickly

    // Allocate and register multiple SEPARATE buffers (one per peer)
    // This is the pattern that causes the hang
    struct MultiBufferInfo {
        std::vector<void*> buffers;
        std::vector<void*> handles;
        size_t countPerBuffer = 0;
        int nPeers = 0;
        bool allocated = false;
    };

    MultiBufferInfo allocateSeparateBuffers(size_t countPerBuffer, int nPeers)
    {
        MultiBufferInfo info;
        info.countPerBuffer = countPerBuffer;
        info.nPeers = nPeers;
        info.buffers.resize(nPeers, nullptr);
        info.handles.resize(nPeers, nullptr);

        size_t bufSize = countPerBuffer * sizeof(T);

        for (int i = 0; i < nPeers; i++) {
            // Allocate SEPARATE buffer for each peer
            if (hipMalloc(&info.buffers[i], bufSize) != hipSuccess) {
                cleanupMultiBuffers(info);
                return info;
            }

            // Register each buffer separately
            ncclResult_t result = ncclCommRegister(getActiveCommunicator(),
                                                    info.buffers[i], bufSize,
                                                    &info.handles[i]);
            if (result != ncclSuccess || info.handles[i] == nullptr) {
                TEST_WARN("Failed to register buffer %d", i);
            }
        }

        info.allocated = true;
        return info;
    }

    void cleanupMultiBuffers(MultiBufferInfo& info)
    {
        for (int i = 0; i < info.nPeers; i++) {
            if (info.handles[i]) {
                ncclCommDeregister(getActiveCommunicator(), info.handles[i]);
                info.handles[i] = nullptr;
            }
            if (info.buffers[i]) {
                (void)hipFree(info.buffers[i]);
                info.buffers[i] = nullptr;
            }
        }
        info.allocated = false;
    }

    // Allocate a SINGLE contiguous buffer and create views (the fix)
    struct ContiguousBufferInfo {
        void* contiguousBuffer = nullptr;
        void* handle = nullptr;
        std::vector<void*> views;  // Pointers into contiguous buffer
        size_t countPerView = 0;
        int nViews = 0;
        bool allocated = false;
    };

    ContiguousBufferInfo allocateContiguousWithViews(size_t countPerView, int nViews)
    {
        ContiguousBufferInfo info;
        info.countPerView = countPerView;
        info.nViews = nViews;
        info.views.resize(nViews, nullptr);

        size_t totalSize = countPerView * nViews * sizeof(T);

        // Allocate ONE contiguous buffer
        if (hipMalloc(&info.contiguousBuffer, totalSize) != hipSuccess) {
            return info;
        }

        // Register the contiguous buffer ONCE
        ncclResult_t result = ncclCommRegister(getActiveCommunicator(),
                                                info.contiguousBuffer, totalSize,
                                                &info.handle);
        if (result != ncclSuccess) {
            (void)hipFree(info.contiguousBuffer);
            info.contiguousBuffer = nullptr;
            return info;
        }

        // Create views (pointers) into the contiguous buffer
        char* base = static_cast<char*>(info.contiguousBuffer);
        for (int i = 0; i < nViews; i++) {
            info.views[i] = base + (i * countPerView * sizeof(T));
        }

        info.allocated = true;
        return info;
    }

    void cleanupContiguousBuffer(ContiguousBufferInfo& info)
    {
        if (info.handle) {
            ncclCommDeregister(getActiveCommunicator(), info.handle);
            info.handle = nullptr;
        }
        if (info.contiguousBuffer) {
            (void)hipFree(info.contiguousBuffer);
            info.contiguousBuffer = nullptr;
        }
        info.views.clear();
        info.allocated = false;
    }
};

/**
 * @brief THIS TEST SHOULD HANG - Demonstrates the UBR concurrent registration bug
 *
 * Pattern that causes hang:
 * - N separate send buffers, each registered separately
 * - N separate recv buffers, each registered separately
 * - ncclGroupStart()
 * - N x ncclSend() + N x ncclRecv() with different buffers
 * - ncclGroupEnd()
 *
 * The ncclProxyCallBlocking() calls inside the group create circular waits.
 *
 * IMPORTANT: Run with timeout to detect hang:
 *   timeout 30 mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter=*SeparateBuffers_Grouped*
 */
TEST_F(UBR_ConcurrentRegHang, SeparateBuffers_Grouped_EXPECTED_HANG)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 4+ ranks across 2+ nodes";
    }

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    TEST_INFO("Rank %d: Testing SEPARATE buffer pattern (expected to HANG)", rank);
    TEST_INFO("Rank %d: Allocating %d separate send buffers + %d separate recv buffers",
              rank, nRanks, nRanks);

    // Allocate SEPARATE buffers for each peer (this causes the hang)
    MultiBufferInfo sendInfo = allocateSeparateBuffers(ELEMENTS_PER_PEER, nRanks);
    MultiBufferInfo recvInfo = allocateSeparateBuffers(ELEMENTS_PER_PEER, nRanks);

    auto cleanup = makeScopeGuard([&]() {
        cleanupMultiBuffers(sendInfo);
        cleanupMultiBuffers(recvInfo);
    });

    ASSERT_MPI_TRUE(sendInfo.allocated);
    ASSERT_MPI_TRUE(recvInfo.allocated);

    // Initialize send buffers
    for (int peer = 0; peer < nRanks; peer++) {
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(sendInfo.buffers[peer], ELEMENTS_PER_PEER,
            [rank, peer](size_t) {
                return static_cast<T>(static_cast<float>(rank * 100 + peer));
            }));
    }

    TEST_INFO("Rank %d: Starting ncclGroupStart/ncclGroupEnd with %d send/recv pairs",
              rank, nRanks);
    TEST_INFO("Rank %d: >>> If this hangs, the test has reproduced the bug <<<", rank);

    // This grouped operation with separate buffers should HANG
    // because of concurrent blocking IPC registration
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());

    for (int peer = 0; peer < nRanks; peer++) {
        if (peer != rank) {
            ncclSend(sendInfo.buffers[peer], ELEMENTS_PER_PEER, getNcclDataType<T>(),
                     peer, getActiveCommunicator(), getActiveStream());
            ncclRecv(recvInfo.buffers[peer], ELEMENTS_PER_PEER, getNcclDataType<T>(),
                     peer, getActiveCommunicator(), getActiveStream());
        }
    }

    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());  // THIS IS WHERE IT HANGS

    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    TEST_INFO("Rank %d: Completed (if you see this, the hang did not occur)", rank);

    // Verify results
    for (int peer = 0; peer < nRanks; peer++) {
        if (peer != rank) {
            bool verified = verifyBufferData<T>(recvInfo.buffers[peer], ELEMENTS_PER_PEER,
                [peer, rank](size_t) {
                    return static_cast<T>(static_cast<float>(peer * 100 + rank));
                });
            ASSERT_TRUE(verified) << "Data verification failed for peer " << peer;
        }
    }
}

/**
 * @brief THIS TEST SHOULD WORK - Demonstrates the fix using contiguous buffer
 *
 * Pattern that works:
 * - 1 contiguous send buffer with views for each peer
 * - 1 contiguous recv buffer with views for each peer
 * - Only 2 buffer registrations total (not 2*N)
 * - ncclGroupStart()
 * - N x ncclSend() + N x ncclRecv() with views
 * - ncclGroupEnd()
 *
 * Works because UBR only needs to register 2 buffers, not 2*N.
 */
TEST_F(UBR_ConcurrentRegHang, ContiguousBuffer_Grouped_SHOULD_WORK)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 4+ ranks across 2+ nodes";
    }

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    TEST_INFO("Rank %d: Testing CONTIGUOUS buffer pattern (should work)", rank);
    TEST_INFO("Rank %d: Allocating 1 contiguous send buffer + 1 contiguous recv buffer with %d views each",
              rank, nRanks);

    // Allocate contiguous buffers with views (this is the fix)
    ContiguousBufferInfo sendInfo = allocateContiguousWithViews(ELEMENTS_PER_PEER, nRanks);
    ContiguousBufferInfo recvInfo = allocateContiguousWithViews(ELEMENTS_PER_PEER, nRanks);

    auto cleanup = makeScopeGuard([&]() {
        cleanupContiguousBuffer(sendInfo);
        cleanupContiguousBuffer(recvInfo);
    });

    ASSERT_MPI_TRUE(sendInfo.allocated);
    ASSERT_MPI_TRUE(recvInfo.allocated);

    // Initialize send buffer views
    for (int peer = 0; peer < nRanks; peer++) {
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(sendInfo.views[peer], ELEMENTS_PER_PEER,
            [rank, peer](size_t) {
                return static_cast<T>(static_cast<float>(rank * 100 + peer));
            }));
    }

    TEST_INFO("Rank %d: Starting ncclGroupStart/ncclGroupEnd with %d send/recv pairs using views",
              rank, nRanks);

    // This grouped operation with contiguous buffer views should WORK
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());

    for (int peer = 0; peer < nRanks; peer++) {
        if (peer != rank) {
            ncclSend(sendInfo.views[peer], ELEMENTS_PER_PEER, getNcclDataType<T>(),
                     peer, getActiveCommunicator(), getActiveStream());
            ncclRecv(recvInfo.views[peer], ELEMENTS_PER_PEER, getNcclDataType<T>(),
                     peer, getActiveCommunicator(), getActiveStream());
        }
    }

    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());

    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    TEST_INFO("Rank %d: Completed successfully", rank);

    // Verify results
    for (int peer = 0; peer < nRanks; peer++) {
        if (peer != rank) {
            bool verified = verifyBufferData<T>(recvInfo.views[peer], ELEMENTS_PER_PEER,
                [peer, rank](size_t) {
                    return static_cast<T>(static_cast<float>(peer * 100 + rank));
                });
            ASSERT_TRUE(verified) << "Data verification failed for peer " << peer;
        }
    }
}

/**
 * @brief ncclAllToAll with CONTIGUOUS registered buffers - SHOULD WORK
 *
 * This test uses ncclAllToAll directly (not grouped send/recv) with
 * a single contiguous registered buffer. This matches the correct
 * pattern used after fixing the PARAM benchmark.
 *
 * ncclAllToAll internally uses contiguous send/recv buffers, so
 * only 2 buffer registrations are needed.
 */
TEST_F(UBR_ConcurrentRegHang, AllToAll_ContiguousBuffer_SHOULD_WORK)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 4+ ranks across 2+ nodes";
    }

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t countPerRank = ELEMENTS_PER_PEER;
    const size_t totalCount = countPerRank * nRanks;
    const size_t totalSize = totalCount * sizeof(T);

    TEST_INFO("Rank %d: Testing ncclAllToAll with CONTIGUOUS registered buffers", rank);

    // Allocate single contiguous buffers
    RegInfo sendInfo = allocateAndRegister(totalSize);
    RegInfo recvInfo = allocateAndRegister(totalSize);

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_MPI_NE(sendInfo.buffer, nullptr);
    ASSERT_MPI_NE(recvInfo.buffer, nullptr);
    ASSERT_MPI_NE(sendInfo.handle, nullptr);
    ASSERT_MPI_NE(recvInfo.handle, nullptr);

    // Initialize send buffer: each chunk destined for peer i contains (rank*100 + i)
    ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(sendInfo.buffer, totalCount,
        [rank, countPerRank](size_t i) {
            int destRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(rank * 100 + destRank));
        }));

    TEST_INFO("Rank %d: Calling ncclAllToAll with registered contiguous buffers", rank);

    ncclResult_t result = ncclAllToAll(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                        getNcclDataType<T>(),
                                        getActiveCommunicator(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, result);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    TEST_INFO("Rank %d: ncclAllToAll completed successfully", rank);

    // Verify: chunk i should contain (srcRank*100 + myRank) from srcRank=i
    bool verified = verifyBufferData<T>(recvInfo.buffer, totalCount,
        [rank, countPerRank](size_t i) {
            int srcRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(srcRank * 100 + rank));
        });
    ASSERT_TRUE(verified);

    TEST_INFO("Rank %d: Data verification passed", rank);
}

/**
 * @brief Stress test: Multiple iterations of ncclAllToAll with contiguous buffers
 *
 * This test runs many iterations of ncclAllToAll with registered contiguous
 * buffers to verify stability under repeated use.
 */
TEST_F(UBR_ConcurrentRegHang, AllToAll_ContiguousBuffer_Stress_SHOULD_WORK)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 4+ ranks across 2+ nodes";
    }

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t countPerRank = ELEMENTS_PER_PEER;
    const size_t totalCount = countPerRank * nRanks;
    const size_t totalSize = totalCount * sizeof(T);
    const int NUM_ITERATIONS = 50;

    TEST_INFO("Rank %d: Stress test - %d iterations of ncclAllToAll with contiguous buffers",
              rank, NUM_ITERATIONS);

    RegInfo sendInfo = allocateAndRegister(totalSize);
    RegInfo recvInfo = allocateAndRegister(totalSize);

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_MPI_NE(sendInfo.buffer, nullptr);
    ASSERT_MPI_NE(recvInfo.buffer, nullptr);
    ASSERT_MPI_NE(sendInfo.handle, nullptr);
    ASSERT_MPI_NE(recvInfo.handle, nullptr);

    int errors = 0;

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // Initialize with iteration-dependent pattern
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(sendInfo.buffer, totalCount,
            [rank, countPerRank, iter](size_t i) {
                int destRank = i / countPerRank;
                return static_cast<T>(static_cast<float>(rank * 100 + destRank + iter));
            }));

        ASSERT_MPI_EQ(hipSuccess, hipMemset(recvInfo.buffer, 0, totalSize));

        ncclResult_t result = ncclAllToAll(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                            getNcclDataType<T>(),
                                            getActiveCommunicator(), getActiveStream());
        if (result != ncclSuccess) {
            errors++;
            continue;
        }

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        bool verified = verifyBufferData<T>(recvInfo.buffer, totalCount,
            [rank, countPerRank, iter](size_t i) {
                int srcRank = i / countPerRank;
                return static_cast<T>(static_cast<float>(srcRank * 100 + rank + iter));
            });

        if (!verified) {
            errors++;
            if (rank == 0) {
                TEST_WARN("Iteration %d: verification failed", iter);
            }
        }

        if ((iter + 1) % 10 == 0 && rank == 0) {
            TEST_INFO("Completed %d/%d iterations, errors=%d", iter + 1, NUM_ITERATIONS, errors);
        }
    }

    ASSERT_EQ(0, errors);
    TEST_INFO("Rank %d: Stress test passed - %d iterations with 0 errors", rank, NUM_ITERATIONS);
}

/**
 * @brief Side-by-side comparison: Contiguous vs Separate buffer patterns
 *
 * This test runs both patterns back-to-back to clearly demonstrate
 * that the contiguous buffer pattern works while separate buffers
 * would hang (the separate buffer test is commented out to avoid hang).
 */
TEST_F(UBR_ConcurrentRegHang, ContiguousVsSeparate_Comparison_SHOULD_WORK)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 4+ ranks across 2+ nodes";
    }

    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    TEST_INFO("Rank %d: Comparison test - demonstrating contiguous buffer pattern works", rank);

    // ========================================
    // PATTERN 1: Contiguous buffer (WORKS)
    // ========================================
    TEST_INFO("Rank %d: --- PATTERN 1: Contiguous buffer (should work) ---", rank);

    ContiguousBufferInfo sendContig = allocateContiguousWithViews(ELEMENTS_PER_PEER, nRanks);
    ContiguousBufferInfo recvContig = allocateContiguousWithViews(ELEMENTS_PER_PEER, nRanks);

    auto cleanupContig = makeScopeGuard([&]() {
        cleanupContiguousBuffer(sendContig);
        cleanupContiguousBuffer(recvContig);
    });

    ASSERT_MPI_TRUE(sendContig.allocated);
    ASSERT_MPI_TRUE(recvContig.allocated);

    // Log buffer info
    TEST_INFO("Rank %d: Contiguous send buffer: base=%p, handle=%p, %d views",
              rank, sendContig.contiguousBuffer, sendContig.handle, nRanks);
    TEST_INFO("Rank %d: Contiguous recv buffer: base=%p, handle=%p, %d views",
              rank, recvContig.contiguousBuffer, recvContig.handle, nRanks);

    // Initialize
    for (int peer = 0; peer < nRanks; peer++) {
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(sendContig.views[peer], ELEMENTS_PER_PEER,
            [rank, peer](size_t) {
                return static_cast<T>(static_cast<float>(rank * 100 + peer));
            }));
    }

    // Execute grouped send/recv with contiguous views
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    for (int peer = 0; peer < nRanks; peer++) {
        if (peer != rank) {
            ncclSend(sendContig.views[peer], ELEMENTS_PER_PEER, getNcclDataType<T>(),
                     peer, getActiveCommunicator(), getActiveStream());
            ncclRecv(recvContig.views[peer], ELEMENTS_PER_PEER, getNcclDataType<T>(),
                     peer, getActiveCommunicator(), getActiveStream());
        }
    }
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    // Verify
    bool pattern1_ok = true;
    for (int peer = 0; peer < nRanks; peer++) {
        if (peer != rank) {
            bool verified = verifyBufferData<T>(recvContig.views[peer], ELEMENTS_PER_PEER,
                [peer, rank](size_t) {
                    return static_cast<T>(static_cast<float>(peer * 100 + rank));
                });
            if (!verified) {
                pattern1_ok = false;
                TEST_WARN("Rank %d: Contiguous pattern verification failed for peer %d", rank, peer);
            }
        }
    }

    ASSERT_TRUE(pattern1_ok);
    TEST_INFO("Rank %d: PATTERN 1 (Contiguous) completed successfully!", rank);

    // ========================================
    // PATTERN 2: Separate buffers (WOULD HANG)
    // ========================================
    // NOTE: We skip actually running this pattern because it would hang.
    // The SeparateBuffers_Grouped_EXPECTED_HANG test demonstrates this.
    TEST_INFO("Rank %d: --- PATTERN 2: Separate buffers (skipped - would hang) ---", rank);
    TEST_INFO("Rank %d: To see the hang, run: UBR_ConcurrentRegHang.SeparateBuffers_Grouped_EXPECTED_HANG", rank);

    // Summary
    TEST_INFO("Rank %d: ===== COMPARISON SUMMARY =====", rank);
    TEST_INFO("Rank %d: Contiguous buffer pattern: PASSED", rank);
    TEST_INFO("Rank %d: Separate buffer pattern:   SKIPPED (known to hang)", rank);
}

/**
 * @brief Comparison test - Run same operation without UBR registration
 *
 * This test allocates separate buffers but does NOT register them.
 * It should work because without explicit registration, RCCL doesn't
 * attempt IPC registration during the collective.
 */
TEST_F(UBR_ConcurrentRegHang, SeparateBuffers_NoRegistration_SHOULD_WORK)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 4+ ranks across 2+ nodes";
    }

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    TEST_INFO("Rank %d: Testing SEPARATE buffers WITHOUT registration (baseline)", rank);

    // Allocate separate buffers but DO NOT register them
    std::vector<void*> sendBufs(nRanks, nullptr);
    std::vector<void*> recvBufs(nRanks, nullptr);
    size_t bufSize = ELEMENTS_PER_PEER * sizeof(T);

    auto cleanup = makeScopeGuard([&]() {
        for (int i = 0; i < nRanks; i++) {
            if (sendBufs[i]) (void)hipFree(sendBufs[i]);
            if (recvBufs[i]) (void)hipFree(recvBufs[i]);
        }
    });

    for (int i = 0; i < nRanks; i++) {
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBufs[i], bufSize));
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBufs[i], bufSize));
    }

    // Initialize
    for (int peer = 0; peer < nRanks; peer++) {
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(sendBufs[peer], ELEMENTS_PER_PEER,
            [rank, peer](size_t) {
                return static_cast<T>(static_cast<float>(rank * 100 + peer));
            }));
    }

    TEST_INFO("Rank %d: Starting grouped send/recv without UBR registration", rank);

    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());

    for (int peer = 0; peer < nRanks; peer++) {
        if (peer != rank) {
            ncclSend(sendBufs[peer], ELEMENTS_PER_PEER, getNcclDataType<T>(),
                     peer, getActiveCommunicator(), getActiveStream());
            ncclRecv(recvBufs[peer], ELEMENTS_PER_PEER, getNcclDataType<T>(),
                     peer, getActiveCommunicator(), getActiveStream());
        }
    }

    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    TEST_INFO("Rank %d: Completed successfully (no UBR = no hang)", rank);

    // Verify results
    for (int peer = 0; peer < nRanks; peer++) {
        if (peer != rank) {
            bool verified = verifyBufferData<T>(recvBufs[peer], ELEMENTS_PER_PEER,
                [peer, rank](size_t) {
                    return static_cast<T>(static_cast<float>(peer * 100 + rank));
                });
            ASSERT_TRUE(verified) << "Data verification failed for peer " << peer;
        }
    }
}

// ============================================================================
// Graph Capture Registration Tests
// ============================================================================

class GraphCapture_AllToAll : public RegistrationTestBase {};

TEST_F(GraphCapture_AllToAll, MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_DEFAULT, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    ASSERT_TRUE(isPerRankLoggingEnabled()) << "RCCL_MPI_LOG_ALL_RANKS must be set to 1";

    ASSERT_TRUE(isGraphRegisterEnabled()) << "NCCL_GRAPH_REGISTER must be set to 1";

    using T = RegTestConfig::DefaultType;
    const size_t countPerRank = RegTestConfig::MEDIUM_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t totalCount = countPerRank * nRanks;
    const size_t bufSize = totalCount * sizeof(T);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;

    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bufSize));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bufSize));

    auto bufCleanup = makeScopeGuard([&]() {
        if (sendBuf) (void)hipFree(sendBuf);
        if (recvBuf) (void)hipFree(recvBuf);
    });

    ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(sendBuf, totalCount,
        [rank, countPerRank](size_t i) {
            int destRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(rank * 100 + destRank));
        }));

    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;

    // Graph capture
    ASSERT_MPI_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal));

    ncclResult_t ncclErr = ncclAllToAll(sendBuf, recvBuf, countPerRank,
                                         getNcclDataType<T>(),
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, ncclErr);

    ASSERT_MPI_EQ(hipSuccess, hipStreamEndCapture(getActiveStream(), &graph));
    ASSERT_MPI_NE(nullptr, graph);

    size_t numNodes = 0;
    ASSERT_MPI_EQ(hipSuccess, hipGraphGetNodes(graph, nullptr, &numNodes));
    ASSERT_MPI_GT(numNodes, 0u);
    TEST_INFO("Graph captured with %zu nodes", numNodes);

    ASSERT_MPI_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    auto graphCleanup = makeScopeGuard([&]() {
        if (graphExec) (void)hipGraphExecDestroy(graphExec);
        if (graph) (void)hipGraphDestroy(graph);
    });

    // Graph execution
    ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, bufSize));
    ASSERT_MPI_EQ(hipSuccess, hipGraphLaunch(graphExec, getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    // Verify registration
    REGLogChecker checker = getLogChecker();
    bool registrationDetected = checker.hasAnyRegistrationSuccess();
    TEST_INFO("AllToAll_MultiNode: %s (log size: %zu bytes)",
              checker.getSummary().c_str(), checker.getContentLength());

    ASSERT_TRUE(registrationDetected);

    // Verify results
    bool resultValid = verifyBufferData<T>(recvBuf, totalCount,
        [rank, countPerRank](size_t i) {
            int srcRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(srcRank * 100 + rank));
        });
    ASSERT_TRUE(resultValid);
    TEST_INFO("AllToAll graph test completed successfully");
}

class GraphCapture_AllReduce : public RegistrationTestBase {};

TEST_F(GraphCapture_AllReduce, MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_DEFAULT, RegTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    ASSERT_TRUE(isPerRankLoggingEnabled()) << "RCCL_MPI_LOG_ALL_RANKS must be set to 1";

    ASSERT_TRUE(isGraphRegisterEnabled()) << "NCCL_GRAPH_REGISTER must be set to 1";

    using T = RegTestConfig::DefaultType;
    const size_t count = RegTestConfig::LARGE_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t bufSize = count * sizeof(T);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;

    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bufSize));
    ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bufSize));

    auto bufCleanup = makeScopeGuard([&]() {
        if (sendBuf) (void)hipFree(sendBuf);
        if (recvBuf) (void)hipFree(recvBuf);
    });

    initSendBuffer<T>(sendBuf, count, rank);

    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;

    // Graph capture
    ASSERT_MPI_EQ(hipSuccess, hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeThreadLocal));

    ncclResult_t ncclErr = ncclAllReduce(sendBuf, recvBuf, count,
                                          getNcclDataType<T>(), ncclSum,
                                          getActiveCommunicator(), getActiveStream());
    ASSERT_MPI_EQ(ncclSuccess, ncclErr);

    ASSERT_MPI_EQ(hipSuccess, hipStreamEndCapture(getActiveStream(), &graph));
    ASSERT_MPI_NE(nullptr, graph);

    size_t numNodes = 0;
    ASSERT_MPI_EQ(hipSuccess, hipGraphGetNodes(graph, nullptr, &numNodes));
    ASSERT_MPI_GT(numNodes, 0u);
    TEST_INFO("AllReduce graph captured with %zu nodes", numNodes);

    ASSERT_MPI_EQ(hipSuccess, hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    auto graphCleanup = makeScopeGuard([&]() {
        if (graphExec) (void)hipGraphExecDestroy(graphExec);
        if (graph) (void)hipGraphDestroy(graph);
    });

    // Graph execution
    ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, bufSize));
    ASSERT_MPI_EQ(hipSuccess, hipGraphLaunch(graphExec, getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    // Verify registration
    REGLogChecker checker = getLogChecker();
    bool registrationDetected = checker.hasAnyRegistrationSuccess();
    TEST_INFO("AllReduce_MultiNode: %s (log size: %zu bytes)",
              checker.getSummary().c_str(), checker.getContentLength());

    ASSERT_TRUE(registrationDetected);

    // Verify results
    bool resultValid = verifyAllReduceResult<T>(recvBuf, count, nRanks);
    ASSERT_TRUE(resultValid);
    TEST_INFO("AllReduce graph test completed successfully");
}

// ============================================================================
// AllToAll/AllToAllv Interleaved Stress Test with UBR
// ============================================================================

/**
 * @brief Stress test for User Buffer Registration with interleaved AllToAll/AllToAllv
 *
 * This test exercises buffer registration under stress conditions:
 * - Interleaves AllToAll and AllToAllv calls
 * - Runs hundreds of iterations
 * - Tests small, medium, and large message sizes
 * - Uses multiple communicators (primary + split)
 * - Detects data mismatches, buffer corruption, and hangs
 */
class UBR_AllToAllStress : public RegistrationTestBase
{
protected:
    // Use bfloat16 as the test type (same as RegTestConfig::DefaultType)
    using T = RegTestConfig::DefaultType;  // hip_bfloat16

    // Test parameters
    static constexpr int ITERATIONS_PER_SIZE = 100;
    static constexpr size_t STRESS_SMALL_COUNT  = 256;             // 512B for bfloat16
    static constexpr size_t STRESS_MEDIUM_COUNT = 64 * 1024;       // 128KiB for bfloat16
    static constexpr size_t STRESS_LARGE_COUNT  = 4 * 1024 * 1024; // 8MiB for bfloat16

    struct CommContext {
        ncclComm_t comm = nullptr;
        int rank = 0;
        int nRanks = 0;
        std::string name;
    };

    // Create CommContext from a communicator
    CommContext makeCommContext(ncclComm_t comm, const std::string& name)
    {
        CommContext ctx;
        ctx.comm = comm;
        ctx.name = name;
        ncclCommUserRank(comm, &ctx.rank);
        ncclCommCount(comm, &ctx.nRanks);
        return ctx;
    }

    // Allocate and register a pair of send/recv buffers
    bool allocateBufferPair(size_t size, ncclComm_t comm,
                            RegInfo& sendInfo, RegInfo& recvInfo)
    {
        sendInfo.size = recvInfo.size = size;

        if (hipMalloc(&sendInfo.buffer, size) != hipSuccess) return false;
        if (hipMalloc(&recvInfo.buffer, size) != hipSuccess) {
            (void)hipFree(sendInfo.buffer);
            sendInfo.buffer = nullptr;
            return false;
        }

        ncclResult_t r1 = ncclCommRegister(comm, sendInfo.buffer, size, &sendInfo.handle);
        ncclResult_t r2 = ncclCommRegister(comm, recvInfo.buffer, size, &recvInfo.handle);
        sendInfo.registered = (r1 == ncclSuccess && sendInfo.handle);
        recvInfo.registered = (r2 == ncclSuccess && recvInfo.handle);

        return true;
    }

    void cleanupBufferPair(RegInfo& sendInfo, RegInfo& recvInfo, ncclComm_t comm)
    {
        auto cleanup = [&](RegInfo& info) {
            if (info.handle) { ncclCommDeregister(comm, info.handle); info.handle = nullptr; }
            if (info.buffer) { (void)hipFree(info.buffer); info.buffer = nullptr; }
            info.registered = false;
        };
        cleanup(sendInfo);
        cleanup(recvInfo);
    }

    // Compute pattern value - use integer patterns for bfloat16 precision
    // Pattern: rank * 100 + dest * 10 + (iter % 10) + offset
    static float computePattern(int rank, int peerRank, int iter, float offset = 0.0f)
    {
        return static_cast<float>(rank * 100 + peerRank * 10 + (iter % 10)) + offset;
    }

    void initBuffer(void* buffer, size_t countPerRank, int nRanks, int rank, int iter, float offset = 0.0f)
    {
        (void)initializeBufferWithPattern<T>(buffer, countPerRank * nRanks,
            [rank, countPerRank, iter, offset](size_t i) {
                int dest = static_cast<int>(i / countPerRank);
                return static_cast<T>(computePattern(rank, dest, iter, offset));
            });
    }

    bool verifyBuffer(void* buffer, size_t countPerRank, int nRanks, int myRank, int iter, float offset = 0.0f)
    {
        std::vector<T> data(countPerRank * nRanks);
        if (hipMemcpy(data.data(), buffer, data.size() * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess)
            return false;

        for (int src = 0; src < nRanks; src++) {
            float expected = computePattern(src, myRank, iter, offset);
            for (size_t i = 0; i < countPerRank; i++) {
                float actual = static_cast<float>(data[src * countPerRank + i]);
                // bfloat16 has ~3 decimal digits precision, use tolerance of 1.0
                if (std::abs(actual - expected) > 1.0f) {
                    TEST_WARN("Mismatch src=%d idx=%zu exp=%f got=%f iter=%d",
                              src, i, expected, actual, iter);
                    return false;
                }
            }
        }
        return true;
    }

    // Variable-size init/verify for AllToAllv
    void initBufferV(void* buffer, const std::vector<size_t>& counts,
                     const std::vector<size_t>& displs, int nRanks, int rank, int iter)
    {
        size_t total = 0;
        for (auto c : counts) total += c;
        std::vector<T> data(total);

        for (int dest = 0; dest < nRanks; dest++) {
            T value = static_cast<T>(computePattern(rank, dest, iter, 0.5f));
            for (size_t i = 0; i < counts[dest]; i++)
                data[displs[dest] + i] = value;
        }
        (void)hipMemcpy(buffer, data.data(), data.size() * sizeof(T), hipMemcpyHostToDevice);
    }

    bool verifyBufferV(void* buffer, const std::vector<size_t>& counts,
                       const std::vector<size_t>& displs, int nRanks, int myRank, int iter)
    {
        size_t total = 0;
        for (auto c : counts) total += c;
        std::vector<T> data(total);
        if (hipMemcpy(data.data(), buffer, data.size() * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess)
            return false;

        for (int src = 0; src < nRanks; src++) {
            float expected = computePattern(src, myRank, iter, 0.5f);
            for (size_t i = 0; i < counts[src]; i++) {
                float actual = static_cast<float>(data[displs[src] + i]);
                if (std::abs(actual - expected) > 1.0f) return false;
            }
        }
        return true;
    }

    // Core stress test runner
    bool runInterleavedStress(CommContext& ctx, size_t countPerRank, int iterations, const std::string& label)
    {
        RegInfo sendInfo, recvInfo;
        size_t totalSize = countPerRank * ctx.nRanks * sizeof(T);

        if (!allocateBufferPair(totalSize, ctx.comm, sendInfo, recvInfo)) {
            TEST_WARN("[%s] Buffer allocation failed", label.c_str());
            return false;
        }
        auto cleanup = makeScopeGuard([&]() { cleanupBufferPair(sendInfo, recvInfo, ctx.comm); });

        // Setup AllToAllv counts
        std::vector<size_t> sendcounts(ctx.nRanks), sdispls(ctx.nRanks);
        std::vector<size_t> recvcounts(ctx.nRanks), rdispls(ctx.nRanks);
        size_t baseCount = countPerRank / 4, sendTotal = 0, recvTotal = 0;

        for (int r = 0; r < ctx.nRanks; r++) {
            sendcounts[r] = ((ctx.rank + r + 1) % ctx.nRanks + 1) * baseCount;
            sdispls[r] = sendTotal; sendTotal += sendcounts[r];
            recvcounts[r] = ((r + ctx.rank + 1) % ctx.nRanks + 1) * baseCount;
            rdispls[r] = recvTotal; recvTotal += recvcounts[r];
        }

        RegInfo sendInfoV, recvInfoV;
        if (!allocateBufferPair(std::max(sendTotal, recvTotal) * sizeof(T), ctx.comm, sendInfoV, recvInfoV)) {
            TEST_WARN("[%s] AllToAllv buffer allocation failed", label.c_str());
            return false;
        }
        auto cleanupV = makeScopeGuard([&]() { cleanupBufferPair(sendInfoV, recvInfoV, ctx.comm); });

        int errors = 0;
        hipStream_t stream = getActiveStream();

        // Debug: Log buffer addresses at start (only rank 0)
        if (ctx.rank == 0) {
            TEST_INFO("[%s] Buffer addresses - sendBuf=%p recvBuf=%p sendBufV=%p recvBufV=%p",
                      label.c_str(), sendInfo.buffer, recvInfo.buffer, sendInfoV.buffer, recvInfoV.buffer);
        }

        for (int iter = 0; iter < iterations; iter++) {
            bool ok;
            const char* opType;

            if (iter % 2 == 0) {
                opType = "AllToAll";
                initBuffer(sendInfo.buffer, countPerRank, ctx.nRanks, ctx.rank, iter);

                // Debug: Show sample data before collective (first 2 iterations, rank 0)
                if (iter < 2 && ctx.rank == 0) {
                    T sample;
                    (void)hipMemcpy(&sample, sendInfo.buffer, sizeof(T), hipMemcpyDeviceToHost);
                    TEST_INFO("[%s] iter=%d PRE-AllToAll: sendBuf[0]=%.1f (expected=%.1f)",
                              label.c_str(), iter, static_cast<float>(sample),
                              computePattern(ctx.rank, 0, iter, 0.0f));
                }

                (void)hipMemset(recvInfo.buffer, 0, totalSize);
                if (ncclAlltoAll(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                 getNcclDataType<T>(), ctx.comm, stream) != ncclSuccess) { errors++; continue; }
                (void)hipStreamSynchronize(stream);

                // Debug: Show sample data after collective (first 2 iterations, rank 0)
                if (iter < 2 && ctx.rank == 0) {
                    T recvSample;
                    (void)hipMemcpy(&recvSample, recvInfo.buffer, sizeof(T), hipMemcpyDeviceToHost);
                    TEST_INFO("[%s] iter=%d POST-AllToAll: recvBuf[0]=%.1f (expected from rank0=%.1f)",
                              label.c_str(), iter, static_cast<float>(recvSample),
                              computePattern(0, ctx.rank, iter, 0.0f));
                }

                ok = verifyBuffer(recvInfo.buffer, countPerRank, ctx.nRanks, ctx.rank, iter);
            } else {
                opType = "AllToAllv";
                initBufferV(sendInfoV.buffer, sendcounts, sdispls, ctx.nRanks, ctx.rank, iter);

                // Debug: Show sample data before collective (first 2 odd iterations, rank 0)
                if (iter < 3 && ctx.rank == 0) {
                    T sample;
                    (void)hipMemcpy(&sample, sendInfoV.buffer, sizeof(T), hipMemcpyDeviceToHost);
                    TEST_INFO("[%s] iter=%d PRE-AllToAllv: sendBufV[0]=%.1f (expected=%.1f)",
                              label.c_str(), iter, static_cast<float>(sample),
                              computePattern(ctx.rank, 0, iter, 0.5f));
                }

                (void)hipMemset(recvInfoV.buffer, 0, recvTotal * sizeof(T));
                if (ncclAlltoAllv(sendInfoV.buffer, sendcounts.data(), sdispls.data(),
                                  recvInfoV.buffer, recvcounts.data(), rdispls.data(),
                                  getNcclDataType<T>(), ctx.comm, stream) != ncclSuccess) { errors++; continue; }
                (void)hipStreamSynchronize(stream);

                // Debug: Show sample data after collective (first 2 odd iterations, rank 0)
                if (iter < 3 && ctx.rank == 0) {
                    T recvSample;
                    (void)hipMemcpy(&recvSample, recvInfoV.buffer, sizeof(T), hipMemcpyDeviceToHost);
                    TEST_INFO("[%s] iter=%d POST-AllToAllv: recvBufV[0]=%.1f (expected from rank0=%.1f)",
                              label.c_str(), iter, static_cast<float>(recvSample),
                              computePattern(0, ctx.rank, iter, 0.5f));
                }

                ok = verifyBufferV(recvInfoV.buffer, recvcounts, rdispls, ctx.nRanks, ctx.rank, iter);
            }

            if (!ok) {
                errors++;
                if (ctx.rank == 0)
                    TEST_WARN("[%s] iter=%d %s VERIFICATION FAILED", label.c_str(), iter, opType);
            }

            // Log progress every 10 iterations (rank 0 only)
            if ((iter + 1) % 10 == 0 && ctx.rank == 0)
                TEST_INFO("[%s] Completed %d/%d iterations, errors=%d", label.c_str(), iter + 1, iterations, errors);
        }

        if (ctx.rank == 0)
            TEST_INFO("[%s] %s: %d iterations", label.c_str(), errors ? "FAILED" : "PASSED", iterations);
        return errors == 0;
    }
};

TEST_F(UBR_AllToAllStress, InterleavedMultiSize_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE))
        GTEST_SKIP() << "Requires 4+ ranks across 2+ nodes";
    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    CommContext ctx = makeCommContext(getActiveCommunicator(), "Primary");
    TEST_INFO("Starting stress test: %d ranks, %d iterations/size", ctx.nRanks, ITERATIONS_PER_SIZE);

    bool ok = true;
    const size_t sizes[] = {STRESS_SMALL_COUNT, STRESS_MEDIUM_COUNT, STRESS_LARGE_COUNT};
    const char* names[] = {"SMALL", "MEDIUM", "LARGE"};

    for (int i = 0; i < 3; i++) {
        TEST_INFO("--- Testing %s (%zu elements/rank, %zu bytes/rank) ---",
                  names[i], sizes[i], sizes[i] * sizeof(T));
        ok &= runInterleavedStress(ctx, sizes[i], ITERATIONS_PER_SIZE, names[i]);
    }
    ASSERT_TRUE(ok);
}

TEST_F(UBR_AllToAllStress, InterleavedWithSplitComm_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE))
        GTEST_SKIP() << "Requires 4+ ranks across 2+ nodes";
    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    CommContext primaryCtx = makeCommContext(getActiveCommunicator(), "Primary");

    // Split into even/odd groups
    ncclComm_t splitComm = nullptr;
    int color = primaryCtx.rank % 2;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSplit(primaryCtx.comm, color, primaryCtx.rank, &splitComm, nullptr));
    auto splitGuard = makeCommAutoGuard(splitComm);

    CommContext splitCtx = makeCommContext(splitComm, color == 0 ? "Split-Even" : "Split-Odd");
    if (splitCtx.nRanks < 2) GTEST_SKIP() << "Split comm too small";

    TEST_INFO("Primary: %d ranks, Split: %d ranks", primaryCtx.nRanks, splitCtx.nRanks);

    bool ok = true;
    const size_t sizes[] = {STRESS_SMALL_COUNT, STRESS_MEDIUM_COUNT, STRESS_LARGE_COUNT};
    const char* names[] = {"SMALL", "MEDIUM", "LARGE"};

    for (int i = 0; i < 3; i++) {
        TEST_INFO("--- Phase %d: %s ---", i, names[i]);
        ok &= runInterleavedStress(primaryCtx, sizes[i], ITERATIONS_PER_SIZE / 2, std::string(names[i]) + "-Primary");
        ok &= runInterleavedStress(splitCtx, sizes[i], ITERATIONS_PER_SIZE / 2, std::string(names[i]) + "-Split");
        MPI_Barrier(MPI_COMM_WORLD);
    }
    ASSERT_TRUE(ok);
}

TEST_F(UBR_AllToAllStress, ConcurrentMultiComm_MultiNode)
{
    if (!setupMultiNode(RegTestConfig::MIN_RANKS_ALLTOALL, RegTestConfig::MIN_NODES_MULTINODE))
        GTEST_SKIP() << "Requires 4+ ranks across 2+ nodes";
    ASSERT_TRUE(isUBREnabled()) << "NCCL_LOCAL_REGISTER must be set to 1";

    ncclComm_t primaryComm = getActiveCommunicator();
    CommContext ctx1 = makeCommContext(primaryComm, "Comm1");

    ncclComm_t comm2 = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSplit(primaryComm, 0, ctx1.rank, &comm2, nullptr));
    auto comm2Guard = makeCommAutoGuard(comm2);

    hipStream_t stream1 = getActiveStream(), stream2;
    ASSERT_MPI_EQ(hipSuccess, hipStreamCreate(&stream2));
    auto stream2Guard = makeStreamAutoGuard(stream2);

    const size_t countPerRank = STRESS_MEDIUM_COUNT;
    const size_t totalSize = countPerRank * ctx1.nRanks * sizeof(T);

    RegInfo send1, recv1, send2, recv2;
    ASSERT_MPI_TRUE(allocateBufferPair(totalSize, primaryComm, send1, recv1));
    ASSERT_MPI_TRUE(allocateBufferPair(totalSize, comm2, send2, recv2));
    auto bufCleanup = makeScopeGuard([&]() {
        cleanupBufferPair(send1, recv1, primaryComm);
        cleanupBufferPair(send2, recv2, comm2);
    });

    TEST_INFO("Concurrent multi-comm stress: 2 comms, 2 streams, %d iterations", ITERATIONS_PER_SIZE);

    int errors = 0;
    for (int iter = 0; iter < ITERATIONS_PER_SIZE; iter++) {
        initBuffer(send1.buffer, countPerRank, ctx1.nRanks, ctx1.rank, iter, 0.0f);
        initBuffer(send2.buffer, countPerRank, ctx1.nRanks, ctx1.rank, iter, 0.5f);
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(recv1.buffer, 0, totalSize, stream1));
        ASSERT_MPI_EQ(hipSuccess, hipMemsetAsync(recv2.buffer, 0, totalSize, stream2));

        ncclGroupStart();
        ncclAlltoAll(send1.buffer, recv1.buffer, countPerRank, getNcclDataType<T>(), primaryComm, stream1);
        ncclAlltoAll(send2.buffer, recv2.buffer, countPerRank, getNcclDataType<T>(), comm2, stream2);
        ncclGroupEnd();

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream1));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(stream2));

        if (!verifyBuffer(recv1.buffer, countPerRank, ctx1.nRanks, ctx1.rank, iter, 0.0f)) errors++;
        if (!verifyBuffer(recv2.buffer, countPerRank, ctx1.nRanks, ctx1.rank, iter, 0.5f)) errors++;

        if ((iter + 1) % 50 == 0 && ctx1.rank == 0)
            TEST_INFO("Concurrent: %d/%d iterations, errors=%d", iter + 1, ITERATIONS_PER_SIZE, errors);
    }

    ASSERT_EQ(0, errors);
    TEST_INFO("Concurrent multi-comm stress test completed successfully");
}

#endif // MPI_TESTS_ENABLED
