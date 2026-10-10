/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaReduceScatterMPITests.cpp
 * @brief MPI end-to-end tests for DDA fabric ReduceScatter LL128.
 *
 * The payload is above the LL cap and inside the LL128 cap, so the selector
 * takes LL128 without an env override. Correctness is checked out of place,
 * in place, and across repeated launches; the COLL log confirms the LL128
 * launcher actually ran.
 *
 * Run (example):
 *   mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter=DdaMPI_ReduceScatter.LL128*
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

#include <memory>
#include <string>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
constexpr int  kRepeatedIterations = 4;
constexpr char kLl128PathNeedle[]  = "taking DDA fabric LL128 path";
constexpr char kLl128LaunchNeedle[] = "DDA fabric ReduceScatter LL128:";

bool isGfx1250Device()
{
    hipDeviceProp_t props{};
    return hipGetDeviceProperties(&props, 0) == hipSuccess
        && std::string(props.gcnArchName).find("gfx1250") != std::string::npos;
}

bool logContains(const MPIHelpers::TestLogAssertionContext& logCtx, const char* needle)
{
    const std::string merged = logCtx.readNcclDebugLog() + logCtx.readPerRankStderrLog();
    return merged.find(needle) != std::string::npos;
}

// Smallest 16-byte-aligned shard whose total message is just past the LL cap
// and still inside the LL128 cap. That is the first size the selector can
// hand to LL128.
size_t ll128ShardBytes(ncclComm* comm)
{
    const int nRanks = comm->nRanks;
    if(nRanks < 2)
        return 0;
    const size_t llCap    = rcclDdaLLThreshold(comm, ncclFuncReduceScatter);
    const size_t ll128Cap = rcclDdaLL128Threshold(comm, ncclFuncReduceScatter);
    if(ll128Cap <= llCap)
        return 0;
    const size_t minTotal = llCap + 16;
    if(minTotal > ll128Cap)
        return 0;
    size_t shard = (minTotal + static_cast<size_t>(nRanks) - 1) / static_cast<size_t>(nRanks);
    shard        = (shard + 15) & ~size_t{15};
    if(shard * static_cast<size_t>(nRanks) <= llCap)
        shard += 16;
    if(shard == 0 || shard * static_cast<size_t>(nRanks) > ll128Cap)
        return 0;
    return shard;
}
} // namespace

class DdaReduceScatterMPITest : public MPITestBase
{
protected:
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugSubsysGuard_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;
    int                                                  rank_{};
    int                                                  nRanks_{};
    size_t                                               recvCount_{};

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
            GTEST_SKIP() << "DDA fabric ReduceScatter LL128 requires gfx1250";

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));

        const size_t shardBytes = ll128ShardBytes(getActiveCommunicator());
        const bool   localSkip  = getActiveCommunicator()->ddaFabricMemHandler == nullptr
                             || getActiveCommunicator()->ddaLLEpochDev == nullptr
                             || shardBytes == 0;
        const std::string skipReason = mpiCoordinatedSkipReason(
            localSkip, "DDA fabric ReduceScatter LL128 is not available for this communicator");
        if(!skipReason.empty())
            GTEST_SKIP() << skipReason;
        recvCount_ = shardBytes / sizeof(float);
    }

    void TearDown() override
    {
        MPITestBase::TearDown();
        logCtx_.reset();
        debugSubsysGuard_.reset();
        debugGuard_.reset();
    }

    void runReduceScatter(bool inPlace, int iterations)
    {
        const size_t totalCount = recvCount_ * static_cast<size_t>(nRanks_);
        const size_t sendBytes  = totalCount * sizeof(float);
        const size_t recvBytes  = recvCount_ * sizeof(float);

        void* sendBuf = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&sendBuf, sendBytes));
        DeviceBufferAutoGuard sendGuard(sendBuf);

        void* recvBuf = static_cast<char*>(sendBuf) + static_cast<size_t>(rank_) * recvBytes;
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
                          ncclReduceScatter(sendBuf, recvBuf, recvCount_, ncclFloat32, ncclSum,
                                            getActiveCommunicator(), getActiveStream()));
        }
        ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

        const float rankSum = static_cast<float>(nRanks_ * (nRanks_ + 1) / 2);
        ASSERT_MPI_TRUE(verifyBufferData<float>(
            recvBuf, recvCount_, [this, rankSum](size_t i) {
                const size_t globalIdx = static_cast<size_t>(rank_) * recvCount_ + i;
                return rankSum + static_cast<float>(nRanks_ * static_cast<int>(globalIdx % 17));
            }));

        EXPECT_TRUE(logContains(*logCtx_, kLl128PathNeedle))
            << "Rank " << rank_ << " did not select DDA fabric ReduceScatter LL128";
        EXPECT_TRUE(logContains(*logCtx_, kLl128LaunchNeedle))
            << "Rank " << rank_ << " did not launch the ReduceScatter LL128 kernel";
    }
};

class DdaMPI_ReduceScatter : public DdaReduceScatterMPITest
{};

TEST_F(DdaMPI_ReduceScatter, LL128OutOfPlace)
{
    runReduceScatter(false, 1);
}

TEST_F(DdaMPI_ReduceScatter, LL128InPlace)
{
    runReduceScatter(true, 1);
}

TEST_F(DdaMPI_ReduceScatter, LL128Repeated)
{
    runReduceScatter(false, kRepeatedIterations);
}

#endif // MPI_TESTS_ENABLED
