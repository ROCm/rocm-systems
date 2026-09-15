/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaFabricSimpleMPITests.cpp
 * @brief MPI end-to-end tests for DDA fabric SIMPLE collectives
 *
 * The selected payloads bypass LL, while the fixture explicitly disables
 * LL128 before communicator creation, so every case exercises a multi-block
 * SIMPLE kernel. Each test checks both output data and the COLL log so a
 * correct fallback cannot hide a DDA regression.
 */

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
constexpr size_t kAllReduceCount     = 8 * 1024 * 1024 + 1;
constexpr size_t kReduceScatterCount = 32 * 1024;       // 1 MiB input at 8 ranks
constexpr size_t kAllToAllCount      = 32 * 1024;       // 1 MiB input at 8 ranks
constexpr int    kRepeatedIterations = 10;

// AllReduce exceeds its 16 MiB LL two-shot threshold; the fixture pins LL128 off.
constexpr size_t kAllReduceBarrierStressCount = kAllReduceCount;

// A shard just above the 512 KiB LL128 hard cap remains small enough to expose
// publication races in ReduceScatter. AllToAll is additionally capped by its
// 4 MiB total-message DDA threshold.
constexpr size_t kBarrierStressCount      = 128 * 1024 + 4;
constexpr int    kBarrierStressIterations = 100;
constexpr size_t kAllToAllMaxTotalBytes   = 4 * 1024 * 1024;

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
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             ll128Guard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             maxBlocksGuard_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;
    int                                                   rank_{};
    int                                                   nRanks_{};

    virtual const char* maxBlocksOverride() const
    {
        return nullptr;
    }

    void SetUp() override
    {
        MPITestBase::SetUp();
        debugGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG", "INFO");
        debugSubsysGuard_ =
            std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG_SUBSYS", "INIT,COLL");
        ll128Guard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("RCCL_DDA_LL128", "0");
        if(const char* value = maxBlocksOverride())
            maxBlocksGuard_ =
                std::make_unique<MPIHelpers::MpiEnvGuard>("RCCL_DDA_FABRIC_MAXBLOCKS", value);
        logCtx_ = std::make_unique<MPIHelpers::TestLogAssertionContext>(
            MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));

        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        if(!isGfx1250Device())
            GTEST_SKIP() << "DDA fabric SIMPLE requires gfx1250";
        if(rcclParamDdaLL128() != 0)
            GTEST_SKIP() << "RCCL_DDA_LL128 was cached as enabled before this fixture; "
                            "launch with RCCL_DDA_LL128=0 to exercise SIMPLE";

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));
        if(getActiveCommunicator()->ddaFabricBarrierState == nullptr)
            GTEST_SKIP() << "DDA fabric path did not initialize";
    }

    void TearDown() override
    {
        MPITestBase::TearDown();
        logCtx_.reset();
        maxBlocksGuard_.reset();
        ll128Guard_.reset();
        debugSubsysGuard_.reset();
        debugGuard_.reset();
    }

    void expectLog(const char* needle)
    {
        const std::string merged =
            logCtx_->readNcclDebugLog() + logCtx_->readPerRankStderrLog();
        EXPECT_NE(merged.find(needle), std::string::npos)
            << "Rank " << rank_ << " did not log expected DDA fabric marker: " << needle;
    }

    size_t allToAllCount(size_t desired) const
    {
        const size_t maxCount =
            kAllToAllMaxTotalBytes / (static_cast<size_t>(nRanks_) * sizeof(float));
        const size_t alignedMaxCount = maxCount - maxCount % (16 / sizeof(float));
        return desired < alignedMaxCount ? desired : alignedMaxCount;
    }

    void runAllReduce(bool inPlace, int iterations = 1)
    {
        const size_t countAlignment =
            static_cast<size_t>(nRanks_) * (16 / sizeof(float));
        const size_t count =
            ((kAllReduceCount + countAlignment - 1) / countAlignment) * countAlignment;
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
        expectLog(kAllReduceNeedle);
    }

    void runAllGather(bool inPlace)
    {
        // AllGather's LL128 predicate uses total bytes and RCCL_DDA_LL128 (default off).
        // Exceed the configured threshold while retaining 16-byte alignment.
        constexpr size_t countAlignment = 16 / sizeof(float);
        const size_t ll128ThresholdBytes =
            static_cast<size_t>(rcclParamDdaLL128Threshold());
        const size_t count =
            (ll128ThresholdBytes
             / (static_cast<size_t>(nRanks_) * sizeof(float) * countAlignment)
             + 1)
            * countAlignment;
        const size_t sendBytes = count * sizeof(float);
        const size_t totalCount = count * static_cast<size_t>(nRanks_);
        const size_t totalBytes = totalCount * sizeof(float);

        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, totalBytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, totalBytes));

        void* sendBuf = static_cast<float*>(recvBuf)
                      + static_cast<size_t>(rank_) * count;
        void* separateSendBuf = nullptr;
        if(!inPlace)
        {
            ASSERT_MPI_EQ(hipSuccess, hipMalloc(&separateSendBuf, sendBytes));
            sendBuf = separateSendBuf;
        }
        DeviceBufferAutoGuard sendGuard(separateSendBuf);

        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
            sendBuf,
            count,
            [this](size_t i) { return static_cast<float>(rank_ * 100000 + i); }));

        ASSERT_MPI_EQ(ncclSuccess,
                      ncclAllGather(sendBuf,
                                    recvBuf,
                                    count,
                                    ncclFloat32,
                                    getActiveCommunicator(),
                                    getActiveStream()));
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        ASSERT_MPI_TRUE(verifyBufferData<float>(
            recvBuf, totalCount, [count](size_t i) {
                const size_t src = i / count;
                const size_t idx = i % count;
                return static_cast<float>(src * 100000 + idx);
            }));
        expectLog(kAllGatherNeedle);
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
        expectLog(kReduceScatterNeedle);
    }

    void runAllToAll(int iterations = 1)
    {
        const size_t countPerPeer = allToAllCount(kAllToAllCount);
        ASSERT_GT(countPerPeer, 0u);
        const size_t totalCount = countPerPeer * static_cast<size_t>(nRanks_);
        const size_t bytes = totalCount * sizeof(float);

        void* sendBuf = nullptr;
        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        // Encoding must stay within float32 exact integer range (2^24 = 16.7M).
        // With 144 ranks: rank*10000 + dest*100 + idx%97 gives max ~1.44M.
        ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
            sendBuf, totalCount, [this, countPerPeer](size_t i) {
                const size_t dest = i / countPerPeer;
                const size_t idx = i % countPerPeer;
                return static_cast<float>(
                    rank_ * 10000 + static_cast<int>(dest) * 100 + idx % 97);
            }));
        ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

        for(int i = 0; i < iterations; ++i)
        {
            ASSERT_MPI_EQ(ncclSuccess,
                          ncclAllToAll(sendBuf,
                                       recvBuf,
                                       countPerPeer,
                                       ncclFloat32,
                                       getActiveCommunicator(),
                                       getActiveStream()));
        }

        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
        ASSERT_MPI_TRUE(verifyBufferData<float>(
            recvBuf, totalCount, [this, countPerPeer](size_t i) {
                const size_t src = i / countPerPeer;
                const size_t idx = i % countPerPeer;
                return static_cast<float>(
                    static_cast<int>(src) * 10000 + rank_ * 100 + idx % 97);
            }));
        expectLog(kAllToAllNeedle);
    }
};

class DdaFabricSimple : public DdaFabricSimpleMPITest
{};

class DdaFabricBlockCap : public DdaFabricSimpleMPITest
{
protected:
    const char* maxBlocksOverride() const override
    {
        return getTestMpiRank() == 0 ? "32" : "64";
    }
};

TEST_F(DdaFabricBlockCap, UsesCliqueWideMinimum)
{
    // Rank 0 advertises 32 while every other rank advertises 64. Observing 32
    // in every rank's local init log proves the bootstrap exchange selected the
    // communicator-wide minimum rather than retaining each local value.
    expectLog("communicator max blocks=32");
}

TEST_F(DdaFabricSimple, AllReduceOutOfPlace)
{
    runAllReduce(false);
}

TEST_F(DdaFabricSimple, AllReduceInPlace)
{
    runAllReduce(true);
}

TEST_F(DdaFabricSimple, AllGatherOutOfPlace)
{
    runAllGather(false);
}

TEST_F(DdaFabricSimple, AllGatherInPlace)
{
    runAllGather(true);
}

TEST_F(DdaFabricSimple, ReduceScatterOutOfPlace)
{
    runReduceScatter(false);
}

TEST_F(DdaFabricSimple, ReduceScatterInPlace)
{
    runReduceScatter(true);
}

TEST_F(DdaFabricSimple, AllToAllOutOfPlace)
{
    runAllToAll();
}

TEST_F(DdaFabricSimple, RepeatedAllReduce)
{
    runAllReduce(false, kRepeatedIterations);
}

TEST_F(DdaFabricSimple, RepeatedReduceScatter)
{
    runReduceScatter(false, kRepeatedIterations);
}

TEST_F(DdaFabricSimple, RepeatedAllToAll)
{
    runAllToAll(kRepeatedIterations);
}

TEST_F(DdaFabricSimple, AlternatingCollectivesOnSameCommunicator)
{
    runAllReduce(false);
    if(HasFatalFailure() || IsSkipped())
        return;
    runAllGather(false);
    if(HasFatalFailure() || IsSkipped())
        return;
    runReduceScatter(false);
    if(HasFatalFailure() || IsSkipped())
        return;
    runAllToAll();
}

// ---------------------------------------------------------------------------
// Barrier stress tests repeatedly exercise the SIMPLE synchronization path.
// AllReduce uses a payload above its LL range while the fixture disables LL128;
// ReduceScatter and AllToAll can use smaller 512 KiB shards/chunks.
// ---------------------------------------------------------------------------

class DdaFabricBarrierStress : public DdaFabricSimpleMPITest
{
protected:
    void runBarrierStressAllReduce()
    {
        const size_t countAlignment =
            static_cast<size_t>(nRanks_) * (16 / sizeof(float));
        const size_t count =
            ((kAllReduceBarrierStressCount + countAlignment - 1) / countAlignment)
            * countAlignment;
        const size_t bytes = count * sizeof(float);

        void* sendBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);

        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        long long totalMismatches = 0;
        int reportsEmitted = 0;

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
                if(reportsEmitted < 5)
                {
                    // Log first few failures
                    fprintf(stderr,
                            "Rank %d: Iteration %d had %d mismatches (expected %.0f)\n",
                            rank_, iter, iterMismatches, expected);
                    ++reportsEmitted;
                }
            }
        }

        // Aggregate across ranks
        long long globalMismatches = 0;
        MPI_Reduce(
            &totalMismatches, &globalMismatches, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank_ == 0)
        {
            EXPECT_EQ(globalMismatches, 0)
                << "Barrier stress test detected " << globalMismatches
                << " data corruption errors across " << kBarrierStressIterations
                << " iterations. This indicates a barrier synchronization bug.";
        }

        expectLog(kAllReduceNeedle);
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

        long long totalMismatches = 0;

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

        long long globalMismatches = 0;
        MPI_Reduce(
            &totalMismatches, &globalMismatches, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank_ == 0)
        {
            EXPECT_EQ(globalMismatches, 0)
                << "ReduceScatter barrier stress test detected " << globalMismatches
                << " data corruption errors.";
        }

        expectLog(kReduceScatterNeedle);
    }

    void runBarrierStressAllToAll()
    {
        const size_t countPerPeer = allToAllCount(kBarrierStressCount);
        ASSERT_GT(countPerPeer, 0u);
        const size_t totalCount = countPerPeer * static_cast<size_t>(nRanks_);
        const size_t bytes = totalCount * sizeof(float);

        void* sendBuf = nullptr;
        void* recvBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        long long totalMismatches = 0;

        for(int iter = 0; iter < kBarrierStressIterations; ++iter)
        {
            // Encoding must stay within float32 exact integer range (2^24 = 16.7M).
            // With 100 iters and 144 ranks: iter*100000 + rank*1000 + dest*7 + idx%7 gives max ~10M.
            ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
                sendBuf, totalCount, [this, iter, countPerPeer](size_t i) {
                    const size_t dest = i / countPerPeer;
                    const size_t idx = i % countPerPeer;
                    return static_cast<float>(
                        iter * 100000 + rank_ * 1000 + static_cast<int>(dest) * 7 + idx % 7);
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
                    iter * 100000 + static_cast<int>(src) * 1000 + rank_ * 7 + idx % 7);
                if(hostBuf[i] != expected)
                {
                    ++iterMismatches;
                }
            }

            totalMismatches += iterMismatches;
        }

        long long globalMismatches = 0;
        MPI_Reduce(
            &totalMismatches, &globalMismatches, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank_ == 0)
        {
            EXPECT_EQ(globalMismatches, 0)
                << "AllToAll barrier stress test detected " << globalMismatches
                << " data corruption errors.";
        }

        expectLog(kAllToAllNeedle);
    }
};

TEST_F(DdaFabricBarrierStress, AllReduceSmallBuffer)
{
    runBarrierStressAllReduce();
}

TEST_F(DdaFabricBarrierStress, ReduceScatterSmallBuffer)
{
    runBarrierStressReduceScatter();
}

TEST_F(DdaFabricBarrierStress, AllToAllSmallBuffer)
{
    runBarrierStressAllToAll();
}

#endif // MPI_TESTS_ENABLED
