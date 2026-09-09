/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaAllReduceMPITests.cpp
 * @brief MPI end-to-end tests for DDA AllReduce: the fabric LL tiers and the launch contract
 *
 * Mirrors test/DdaAllReduceTests.cpp but uses MPI + ncclCommInitRank instead of
 * ncclCommInitAll. Correctness is checked against the analytic sum; the COLL log
 * line confirms which LL tier actually ran (fallback paths also sum correctly).
 * One rank per process is also what lets the DDA IPC tier initialise at all, which
 * is why the launch-contract cases for AICOMRCCL-2184 live here.
 *
 * Run (example):
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaMPI_AllReduce.*
 */

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
constexpr size_t kOneShotCount = 65536; // 256 KiB f32; below default 1 MiB one-shot threshold

// Default thresholds put two-shot above 1 MiB; 262176 floats is 1 MiB + 128 B and
// satisfies the two-shot shard alignment rules at typical MPI rank counts (see
// DdaFabricEligibilityTests.cpp AllReduceLL_TwoShotClaimsPastOneShotThreshold).
constexpr size_t kTwoShotBaseCount = 262176;

constexpr char kOneShotLogNeedle[] = "taking DDA fabric LL one-shot path";
constexpr char kTwoShotLogNeedle[] = "taking DDA fabric LL two-shot path";

constexpr size_t kContractCount  = 1024 * 1024;
constexpr int    kContractIters  = 4;
constexpr char   kIpcLogNeedle[] = "AllReduce impl selected: algo DDA-IPC";

bool isGfx1250Device()
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
        return false;
    return std::string(props.gcnArchName).find("gfx1250") != std::string::npos;
}

void fillRankScalar(void* buf, size_t nElem, int rank)
{
    std::vector<float> host(nElem, static_cast<float>(rank + 1));
    HIP_CHECK(hipMemcpy(buf, host.data(), nElem * sizeof(float), hipMemcpyHostToDevice));
}

bool ddaLLTwoShotShapeOk(size_t count, int nRanks)
{
    const size_t bytes = count * sizeof(float);
    if(bytes % static_cast<size_t>(nRanks) != 0)
        return false;
    return (bytes / static_cast<size_t>(nRanks)) % 16 == 0;
}

// Pick a count past the default one-shot threshold whose total bytes satisfy the
// two-shot per-rank shard alignment for the active communicator width.
size_t twoShotCountForRanks(int nRanks)
{
    size_t count = kTwoShotBaseCount;
    if(!ddaLLTwoShotShapeOk(count, nRanks))
    {
        count = (count + 3) & ~size_t(3);
        for(int i = 0; i < 1024; ++i, count += 4)
        {
            if(ddaLLTwoShotShapeOk(count, nRanks))
                return count;
        }
        return 0;
    }
    return count;
}

bool logContainsNeedle(const MPIHelpers::TestLogAssertionContext& logCtx, const char* needle)
{
    const std::string merged = logCtx.readNcclDebugLog() + logCtx.readPerRankStderrLog();
    return merged.find(needle) != std::string::npos;
}
} // namespace

/**
 * @class DdaAllReduceMPITest
 * @brief Shared fixture: COLL log capture and gfx1250 gate for DDA fabric LL tiers.
 */
class DdaAllReduceMPITest : public MPITestBase
{
protected:
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugSubsysGuard_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;

    void SetUp() override
    {
        MPITestBase::SetUp();
        debugGuard_       = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG", "INFO");
        debugSubsysGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG_SUBSYS", "NET,INIT,COLL,TUNING");
        logCtx_           = std::make_unique<MPIHelpers::TestLogAssertionContext>(
            MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));
    }

    void TearDown() override
    {
        MPITestBase::TearDown();
        logCtx_.reset();
        debugSubsysGuard_.reset();
        debugGuard_.reset();
    }

    void runAllReduce(size_t count, const char* logNeedle, const char* testId)
    {
        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";

        //if(!isGfx1250Device())
        //    GTEST_SKIP() << "DDA fabric LL requires gfx1250";

        ASSERT_EQ(ncclSuccess, createTestCommunicator());

        int rank{}, nRanks{};
        ncclCommUserRank(getActiveCommunicator(), &rank);
        ncclCommCount(getActiveCommunicator(), &nRanks);

        const size_t bytes = count * sizeof(float);

        void* sendBuf = nullptr;
        ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);

        void* recvBuf = nullptr;
        ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
        DeviceBufferAutoGuard recvGuard(recvBuf);

        fillRankScalar(sendBuf, count, rank);
        ASSERT_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(sendBuf, recvBuf, count, ncclFloat32, ncclSum,
                                getActiveCommunicator(), getActiveStream()));
        ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        const float expectedSum = static_cast<float>(nRanks * (nRanks + 1) / 2);
        ASSERT_TRUE(verifyBufferData<float>(recvBuf, count,
                                            [expectedSum](size_t) { return expectedSum; }))
            << "Rank " << rank << ": DDA LL AllReduce verification failed";

        const bool tookExpectedPath = logContainsNeedle(*logCtx_, logNeedle);
        EXPECT_TRUE(tookExpectedPath)
            << "Rank " << rank << ": " << testId
            << " did not log the expected DDA LL tier (needle: \"" << logNeedle << "\")";

        if(getTestMpiRank() == 0 && tookExpectedPath)
            TEST_INFO("%s: DDA LL path confirmed via COLL log", testId);
    }

    // Rank 0 alone logs the selected backend, and a backend other than DDA IPC still has to honour
    // the launch contract, so this reports which tier ran rather than asserting one.
    void reportSelectedTier(const char* testId)
    {
        if(getTestMpiRank() != 0)
            return;

        if(logContainsNeedle(*logCtx_, kIpcLogNeedle))
            TEST_INFO("%s: ran on the DDA IPC tier", testId);
        else
            TEST_WARN("%s: DDA IPC was not selected; the contract was checked on another backend", testId);
    }
};

class DdaMPI_AllReduce : public DdaAllReduceMPITest
{};

TEST_F(DdaMPI_AllReduce, LLOneShotMultiRank)
{
    runAllReduce(kOneShotCount, kOneShotLogNeedle, "DdaMPI_AllReduce/LLOneShotMultiRank");
}

TEST_F(DdaMPI_AllReduce, LLTwoShotMultiRank)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
        GTEST_SKIP() << "Need at least 2 MPI ranks";

    int nRanks = MPIEnvironment::world_size;
    const size_t count = twoShotCountForRanks(nRanks);
    if(count == 0)
        GTEST_SKIP() << "Could not find a two-shot-aligned element count for nRanks="
                     << nRanks;

    runAllReduce(count, kTwoShotLogNeedle, "DdaMPI_AllReduce/LLTwoShotMultiRank");
}

// AICOMRCCL-2184: RCCL selects the collective's device itself, and the device the caller left
// current is not part of the API contract.
TEST_F(DdaMPI_AllReduce, ForeignCurrentDeviceUngrouped)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
        GTEST_SKIP() << "Need at least 2 MPI ranks";

    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    if(deviceCount < 2)
        GTEST_SKIP() << "Need at least 2 GPUs visible to each rank; found " << deviceCount;

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    int ownDevice = -1;
    HIP_CHECK(hipGetDevice(&ownDevice));

    const size_t bytes = kContractCount * sizeof(float);

    void* sendBuf = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
    DeviceBufferAutoGuard sendGuard(sendBuf);

    void* recvBuf = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
    DeviceBufferAutoGuard recvGuard(recvBuf);

    fillRankScalar(sendBuf, kContractCount, rank);
    ASSERT_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

    const int foreignDevice = (ownDevice + 1) % deviceCount;
    HIP_CHECK(hipSetDevice(foreignDevice));

    const ncclResult_t res = ncclAllReduce(sendBuf, recvBuf, kContractCount, ncclFloat32, ncclSum,
                                           getActiveCommunicator(), getActiveStream());
    EXPECT_EQ(ncclSuccess, res) << "Rank " << rank << " launched with device " << foreignDevice
                                << " current: " << ncclGetErrorString(res);

    int deviceAfterCall = -1;
    HIP_CHECK(hipGetDevice(&deviceAfterCall));
    EXPECT_EQ(foreignDevice, deviceAfterCall)
        << "Rank " << rank << ": AllReduce left device " << deviceAfterCall << " current";

    HIP_CHECK(hipSetDevice(ownDevice));
    ASSERT_EQ(ncclSuccess, res);
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

    const float expectedSum = static_cast<float>(nRanks * (nRanks + 1) / 2);
    ASSERT_TRUE(verifyBufferData<float>(recvBuf, kContractCount,
                                        [expectedSum](size_t) { return expectedSum; }))
        << "Rank " << rank << ": AllReduce issued with a foreign current device wrote wrong data";

    reportSelectedTier("DdaMPI_AllReduce/ForeignCurrentDeviceUngrouped");
}

// AICOMRCCL-2184: consecutive collectives on one communicator are ordered against each other even
// when they run on different streams.
TEST_F(DdaMPI_AllReduce, StreamAlternationUngrouped)
{
    if(!validateTestPrerequisites(kMinProcessesForMPI))
        GTEST_SKIP() << "Need at least 2 MPI ranks";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank{}, nRanks{};
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    hipStream_t altStream = nullptr;
    ASSERT_EQ(hipSuccess, hipStreamCreate(&altStream));
    HipStreamAutoGuard altStreamGuard(altStream);

    const size_t bytes = kContractCount * sizeof(float);

    void* sendBuf = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf, bytes));
    DeviceBufferAutoGuard sendGuard(sendBuf);

    void* recvBuf = nullptr;
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf, bytes));
    DeviceBufferAutoGuard recvGuard(recvBuf);

    fillRankScalar(sendBuf, kContractCount, rank);
    ASSERT_EQ(hipSuccess, hipMemset(recvBuf, 0, bytes));

    for(int iter = 0; iter < kContractIters; ++iter)
    {
        hipStream_t stream = (iter % 2 == 0) ? getActiveStream() : altStream;
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(sendBuf, recvBuf, kContractCount, ncclFloat32, ncclSum,
                                getActiveCommunicator(), stream))
            << "Rank " << rank << ": iteration " << iter;
    }

    ASSERT_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));
    ASSERT_EQ(hipSuccess, hipStreamSynchronize(altStream));

    const float expectedSum = static_cast<float>(nRanks * (nRanks + 1) / 2);
    ASSERT_TRUE(verifyBufferData<float>(recvBuf, kContractCount,
                                        [expectedSum](size_t) { return expectedSum; }))
        << "Rank " << rank << ": AllReduce across alternating streams wrote wrong data";

    reportSelectedTier("DdaMPI_AllReduce/StreamAlternationUngrouped");
}

#endif // MPI_TESTS_ENABLED
