/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaFabricSimpleMPITests.cpp
 * @brief MPI end-to-end tests for DDA fabric SIMPLE collectives
 *
 * The selected payloads naturally bypass the default LL/LL128 tiers and
 * exercise multi-block SIMPLE kernels. Each test checks both output data and
 * the COLL log so a correct fallback cannot hide a DDA regression.
 */

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>
#include <string>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
constexpr size_t kAllReduceCount     = 8 * 1024 * 1024; // 32 MiB per rank
constexpr size_t kAllGatherCount     = 32 * 1024;       // 1 MiB total at 8 ranks
constexpr size_t kReduceScatterCount = 32 * 1024;       // 1 MiB input at 8 ranks
constexpr size_t kAllToAllCount      = 32 * 1024;       // 1 MiB input at 8 ranks
constexpr int    kRepeatedIterations = 10;

// Barrier stress test parameters: small buffers complete quickly, exposing
// timing windows where missing/broken barriers would cause data corruption.
// Large buffers (32MB) hide races because DMA latency masks the barrier window.
constexpr size_t kBarrierStressCount      = 128 * 1024;  // 512 KB per rank
constexpr int    kBarrierStressIterations = 100;

constexpr char kAllReduceNeedle[] =
    "DDA fabric AllReduce: launching tree (two-shot) kernel";
constexpr char kAllGatherNeedle[] =
    "DDA fabric AllGather: launching kernel";
constexpr char kReduceScatterNeedle[] =
    "DDA fabric ReduceScatter: launching kernel";
constexpr char kAllToAllNeedle[] =
    "DDA fabric AllToAll: launching kernel";

bool isGfx1250Device()
{
    hipDeviceProp_t props{};
    return hipGetDeviceProperties(&props, 0) == hipSuccess
        && std::string(props.gcnArchName).find("gfx1250") != std::string::npos;
}
} // namespace

class DdaFabricSimpleMPITest : public MPITestBase
{
protected:
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugSubsysGuard_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;
    int                                                   rank_{};
    int                                                   nRanks_{};

    void SetUp() override
    {
        MPITestBase::SetUp();
        debugGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG", "INFO");
        debugSubsysGuard_ =
            std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG_SUBSYS", "INIT,COLL");
        logCtx_ = std::make_unique<MPIHelpers::TestLogAssertionContext>(
            MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));

        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        if(!isGfx1250Device())
            GTEST_SKIP() << "DDA fabric SIMPLE requires gfx1250";

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));
    }

    void TearDown() override
    {
        MPITestBase::TearDown();
        logCtx_.reset();
        debugSubsysGuard_.reset();
        debugGuard_.reset();
    }

    void expectPath(const char* needle)
    {
        const std::string merged =
            logCtx_->readNcclDebugLog() + logCtx_->readPerRankStderrLog();
        EXPECT_NE(merged.find(needle), std::string::npos)
            << "Rank " << rank_ << " did not execute expected DDA fabric SIMPLE path: "
            << needle;
    }

    void runAllReduce(bool inPlace, int iterations = 1)
    {
        const size_t count =
            ((kAllReduceCount + static_cast<size_t>(nRanks_) - 1)
             / static_cast<size_t>(nRanks_))
            * static_cast<size_t>(nRanks_);
        const size_t bytes = count * sizeof(float);
        void* sendBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);

        void* recvBuf = sendBuf;
        void* separateRecvBuf = nullptr;
        if(!inPlace)
        {
            ASSERT_MPI_EQ(hipSuccess, hipMalloc(&separateRecvBuf, bytes));
            recvBuf = separateRecvBuf;
        }
        DeviceBufferAutoGuard recvGuard(separateRecvBuf);

        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
            sendBuf,
            count,
            [this](size_t) { return static_cast<float>(rank_ + 1); }));
        if(!inPlace)
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

        for(int i = 0; i < iterations; ++i)
        {
            ASSERT_MPI_EQ(ncclSuccess,
                          ncclAllReduce(sendBuf,
                                        recvBuf,
                                        count,
                                        ncclFloat32,
                                        ncclSum,
                                        getActiveCommunicator(),
                                        getActiveStream()));
        }

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        const float expected = static_cast<float>(nRanks_ * (nRanks_ + 1) / 2);
        ASSERT_MPI_TRUE(verifyBufferData<float>(
            recvBuf, count, [expected](size_t) { return expected; }));
        expectPath(kAllReduceNeedle);
    }

    void runAllGather(bool inPlace)
    {
        const size_t sendBytes = kAllGatherCount * sizeof(float);
        const size_t totalCount = kAllGatherCount * static_cast<size_t>(nRanks_);
        const size_t totalBytes = totalCount * sizeof(float);

        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, totalBytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, totalBytes));

        void* sendBuf = static_cast<float*>(recvBuf)
                      + static_cast<size_t>(rank_) * kAllGatherCount;
        void* separateSendBuf = nullptr;
        if(!inPlace)
        {
            ASSERT_MPI_EQ(hipSuccess, hipMalloc(&separateSendBuf, sendBytes));
            sendBuf = separateSendBuf;
        }
        DeviceBufferAutoGuard sendGuard(separateSendBuf);

        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
            sendBuf,
            kAllGatherCount,
            [this](size_t i) { return static_cast<float>(rank_ * 100000 + i); }));

        ASSERT_MPI_EQ(ncclSuccess,
                      ncclAllGather(sendBuf,
                                    recvBuf,
                                    kAllGatherCount,
                                    ncclFloat32,
                                    getActiveCommunicator(),
                                    getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        ASSERT_MPI_TRUE(verifyBufferData<float>(
            recvBuf, totalCount, [](size_t i) {
                const size_t src = i / kAllGatherCount;
                const size_t idx = i % kAllGatherCount;
                return static_cast<float>(src * 100000 + idx);
            }));
        expectPath(kAllGatherNeedle);
    }

    void runReduceScatter(bool inPlace, int iterations = 1)
    {
        const size_t totalCount = kReduceScatterCount * static_cast<size_t>(nRanks_);
        const size_t sendBytes = totalCount * sizeof(float);
        const size_t recvBytes = kReduceScatterCount * sizeof(float);

        void* sendBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, sendBytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);

        void* recvBuf = static_cast<float*>(sendBuf)
                      + static_cast<size_t>(rank_) * kReduceScatterCount;
        void* separateRecvBuf = nullptr;
        if(!inPlace)
        {
            ASSERT_MPI_EQ(hipSuccess, hipMalloc(&separateRecvBuf, recvBytes));
            recvBuf = separateRecvBuf;
        }
        DeviceBufferAutoGuard recvGuard(separateRecvBuf);

        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
            sendBuf, totalCount, [this](size_t i) {
                return static_cast<float>(rank_ + 1 + static_cast<int>(i % 17));
            }));
        if(!inPlace)
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, recvBytes));

        for(int i = 0; i < iterations; ++i)
        {
            ASSERT_MPI_EQ(ncclSuccess,
                          ncclReduceScatter(sendBuf,
                                            recvBuf,
                                            kReduceScatterCount,
                                            ncclFloat32,
                                            ncclSum,
                                            getActiveCommunicator(),
                                            getActiveStream()));
        }

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        const float rankSum = static_cast<float>(nRanks_ * (nRanks_ + 1) / 2);
        ASSERT_MPI_TRUE(verifyBufferData<float>(
            recvBuf, kReduceScatterCount, [this, rankSum](size_t i) {
                const size_t globalIdx =
                    static_cast<size_t>(rank_) * kReduceScatterCount + i;
                return rankSum
                     + static_cast<float>(nRanks_ * static_cast<int>(globalIdx % 17));
            }));
        expectPath(kReduceScatterNeedle);
    }

    void runAllToAll(int iterations = 1)
    {
        const size_t totalCount = kAllToAllCount * static_cast<size_t>(nRanks_);
        const size_t bytes = totalCount * sizeof(float);

        void* sendBuf = nullptr;
        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
            sendBuf, totalCount, [this](size_t i) {
                const size_t dest = i / kAllToAllCount;
                const size_t idx = i % kAllToAllCount;
                return static_cast<float>(
                    rank_ * 100000 + static_cast<int>(dest) * 10000 + idx % 997);
            }));
        ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

        for(int i = 0; i < iterations; ++i)
        {
            ASSERT_MPI_EQ(ncclSuccess,
                          ncclAllToAll(sendBuf,
                                       recvBuf,
                                       kAllToAllCount,
                                       ncclFloat32,
                                       getActiveCommunicator(),
                                       getActiveStream()));
        }

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        ASSERT_MPI_TRUE(verifyBufferData<float>(
            recvBuf, totalCount, [this](size_t i) {
                const size_t src = i / kAllToAllCount;
                const size_t idx = i % kAllToAllCount;
                return static_cast<float>(
                    static_cast<int>(src) * 100000 + rank_ * 10000 + idx % 997);
            }));
        expectPath(kAllToAllNeedle);
    }
};

class DdaMPI_FabricSimple : public DdaFabricSimpleMPITest
{};

TEST_F(DdaMPI_FabricSimple, AllReduceOutOfPlace)
{
    runAllReduce(false);
}

TEST_F(DdaMPI_FabricSimple, AllReduceInPlace)
{
    runAllReduce(true);
}

TEST_F(DdaMPI_FabricSimple, AllGatherOutOfPlace)
{
    runAllGather(false);
}

TEST_F(DdaMPI_FabricSimple, AllGatherInPlace)
{
    runAllGather(true);
}

TEST_F(DdaMPI_FabricSimple, ReduceScatterOutOfPlace)
{
    runReduceScatter(false);
}

TEST_F(DdaMPI_FabricSimple, ReduceScatterInPlace)
{
    runReduceScatter(true);
}

TEST_F(DdaMPI_FabricSimple, AllToAllOutOfPlace)
{
    runAllToAll();
}

TEST_F(DdaMPI_FabricSimple, RepeatedAllReduce)
{
    runAllReduce(false, kRepeatedIterations);
}

TEST_F(DdaMPI_FabricSimple, RepeatedReduceScatter)
{
    runReduceScatter(false, kRepeatedIterations);
}

TEST_F(DdaMPI_FabricSimple, RepeatedAllToAll)
{
    runAllToAll(kRepeatedIterations);
}

TEST_F(DdaMPI_FabricSimple, AlternatingCollectivesOnSameCommunicator)
{
    runAllReduce(false);
    if(HasFatalFailure())
        return;
    runAllGather(false);
    if(HasFatalFailure())
        return;
    runReduceScatter(false);
    if(HasFatalFailure())
        return;
    runAllToAll();
}

// ---------------------------------------------------------------------------
// Barrier stress tests: Small buffers expose timing windows where missing
// or broken barriers cause data corruption. These tests are designed to FAIL
// if the barrier synchronization is removed or broken.
//
// Rationale: Large buffers (32MB) hide races because DMA latency masks the
// barrier window. Small buffers (512KB) complete quickly, exposing the race.
// See CRITICAL-2 test results: 512KB buffers showed 29M errors without
// proper release semantics, while 32MB buffers passed incorrectly.
// ---------------------------------------------------------------------------

class DdaMPI_FabricBarrierStress : public DdaFabricSimpleMPITest
{
protected:
    void runBarrierStressAllReduce()
    {
        const size_t count =
            ((kBarrierStressCount + static_cast<size_t>(nRanks_) - 1)
             / static_cast<size_t>(nRanks_))
            * static_cast<size_t>(nRanks_);
        const size_t bytes = count * sizeof(float);

        void* sendBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);

        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        int totalMismatches = 0;

        for(int iter = 0; iter < kBarrierStressIterations; ++iter)
        {
            // Fresh pattern each iteration to detect stale reads
            const float pattern = static_cast<float>(iter * 1000 + rank_ + 1);
            ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
                sendBuf, count, [pattern](size_t) { return pattern; }));
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

            ASSERT_MPI_EQ(ncclSuccess,
                          ncclAllReduce(sendBuf,
                                        recvBuf,
                                        count,
                                        ncclFloat32,
                                        ncclSum,
                                        getActiveCommunicator(),
                                        getActiveStream()));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            // Expected: sum of all ranks' patterns for this iteration
            float expected = 0.0f;
            for(int r = 0; r < nRanks_; ++r)
            {
                expected += static_cast<float>(iter * 1000 + r + 1);
            }

            // Count mismatches without failing immediately
            std::vector<float> hostBuf(count);
            ASSERT_MPI_EQ(hipSuccess,
                          hipMemcpy(hostBuf.data(), recvBuf, bytes, hipMemcpyDeviceToHost));

            int iterMismatches = 0;
            for(size_t i = 0; i < count; ++i)
            {
                if(hostBuf[i] != expected)
                {
                    ++iterMismatches;
                }
            }

            if(iterMismatches > 0)
            {
                totalMismatches += iterMismatches;
                if(totalMismatches <= 5 * static_cast<int>(count))
                {
                    // Log first few failures
                    fprintf(stderr,
                            "Rank %d: Iteration %d had %d mismatches (expected %.0f)\n",
                            rank_, iter, iterMismatches, expected);
                }
            }
        }

        // Aggregate across ranks
        int globalMismatches = 0;
        MPI_Reduce(&totalMismatches, &globalMismatches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank_ == 0)
        {
            EXPECT_EQ(globalMismatches, 0)
                << "Barrier stress test detected " << globalMismatches
                << " data corruption errors across " << kBarrierStressIterations
                << " iterations. This indicates a barrier synchronization bug.";
        }

        expectPath(kAllReduceNeedle);
    }

    void runBarrierStressReduceScatter()
    {
        const size_t recvCount = kBarrierStressCount;
        const size_t totalCount = recvCount * static_cast<size_t>(nRanks_);
        const size_t sendBytes = totalCount * sizeof(float);
        const size_t recvBytes = recvCount * sizeof(float);

        void* sendBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, sendBytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);

        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, recvBytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        int totalMismatches = 0;

        for(int iter = 0; iter < kBarrierStressIterations; ++iter)
        {
            ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
                sendBuf, totalCount, [this, iter](size_t i) {
                    return static_cast<float>(iter * 1000 + rank_ + 1 + static_cast<int>(i % 17));
                }));
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, recvBytes));

            ASSERT_MPI_EQ(ncclSuccess,
                          ncclReduceScatter(sendBuf,
                                            recvBuf,
                                            recvCount,
                                            ncclFloat32,
                                            ncclSum,
                                            getActiveCommunicator(),
                                            getActiveStream()));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            // Verify this rank's slice
            std::vector<float> hostBuf(recvCount);
            ASSERT_MPI_EQ(hipSuccess,
                          hipMemcpy(hostBuf.data(), recvBuf, recvBytes, hipMemcpyDeviceToHost));

            const float rankSum = static_cast<float>(nRanks_ * (nRanks_ + 1) / 2);
            int iterMismatches = 0;
            for(size_t i = 0; i < recvCount; ++i)
            {
                const size_t globalIdx = static_cast<size_t>(rank_) * recvCount + i;
                const float expected =
                    static_cast<float>(iter * 1000) * nRanks_ + rankSum
                    + static_cast<float>(nRanks_ * static_cast<int>(globalIdx % 17));
                if(hostBuf[i] != expected)
                {
                    ++iterMismatches;
                }
            }

            totalMismatches += iterMismatches;
        }

        int globalMismatches = 0;
        MPI_Reduce(&totalMismatches, &globalMismatches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank_ == 0)
        {
            EXPECT_EQ(globalMismatches, 0)
                << "ReduceScatter barrier stress test detected " << globalMismatches
                << " data corruption errors.";
        }

        expectPath(kReduceScatterNeedle);
    }

    void runBarrierStressAllToAll()
    {
        const size_t countPerPeer = kBarrierStressCount;
        const size_t totalCount = countPerPeer * static_cast<size_t>(nRanks_);
        const size_t bytes = totalCount * sizeof(float);

        void* sendBuf = nullptr;
        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        int totalMismatches = 0;

        for(int iter = 0; iter < kBarrierStressIterations; ++iter)
        {
            ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
                sendBuf, totalCount, [this, iter](size_t i) {
                    const size_t dest = i / countPerPeer;
                    const size_t idx = i % countPerPeer;
                    return static_cast<float>(
                        iter * 100000 + rank_ * 1000 + static_cast<int>(dest) * 100 + idx % 97);
                }));
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

            ASSERT_MPI_EQ(ncclSuccess,
                          ncclAllToAll(sendBuf,
                                       recvBuf,
                                       countPerPeer,
                                       ncclFloat32,
                                       getActiveCommunicator(),
                                       getActiveStream()));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            std::vector<float> hostBuf(totalCount);
            ASSERT_MPI_EQ(hipSuccess,
                          hipMemcpy(hostBuf.data(), recvBuf, bytes, hipMemcpyDeviceToHost));

            int iterMismatches = 0;
            for(size_t i = 0; i < totalCount; ++i)
            {
                const size_t src = i / countPerPeer;
                const size_t idx = i % countPerPeer;
                const float expected = static_cast<float>(
                    iter * 100000 + static_cast<int>(src) * 1000 + rank_ * 100 + idx % 97);
                if(hostBuf[i] != expected)
                {
                    ++iterMismatches;
                }
            }

            totalMismatches += iterMismatches;
        }

        int globalMismatches = 0;
        MPI_Reduce(&totalMismatches, &globalMismatches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank_ == 0)
        {
            EXPECT_EQ(globalMismatches, 0)
                << "AllToAll barrier stress test detected " << globalMismatches
                << " data corruption errors.";
        }

        expectPath(kAllToAllNeedle);
    }
};

TEST_F(DdaMPI_FabricBarrierStress, AllReduceSmallBuffer)
{
    runBarrierStressAllReduce();
}

TEST_F(DdaMPI_FabricBarrierStress, ReduceScatterSmallBuffer)
{
    runBarrierStressReduceScatter();
}

TEST_F(DdaMPI_FabricBarrierStress, AllToAllSmallBuffer)
{
    runBarrierStressAllToAll();
}

#endif // MPI_TESTS_ENABLED
