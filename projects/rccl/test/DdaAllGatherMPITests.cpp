/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaAllGatherMPITests.cpp
 * @brief Slice placement for the DDA IPC AllGather (gfx942/gfx950, 8 ranks).
 *
 * Run (example):
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaMPI_AllGather.*
 */

#ifdef MPI_TESTS_ENABLED

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "algorithms/dda/dda_init_detail.h"
#include "comm.h"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
constexpr int kDdaIpcRanks = nccl_dda_detail::kDdaNranks;

// Keeps f(rank, i) unique per slot and exact in fp32 (7 * 2^20 + 2^18 < 2^24).
constexpr size_t kRankStride = size_t{1} << 20;

// One block per rank, then 8 MiB total: the gfx942 cap and a full grid-stride.
constexpr size_t kSmallCount = 1024;
constexpr size_t kGridCount  = 256 * 1024;

// No rank sends a negative value, so an unwritten slot cannot verify.
constexpr float kUnwritten = -1.0f;

float slicePattern(int rank, size_t i)
{
    return static_cast<float>(static_cast<size_t>(rank) * kRankStride + i);
}

bool isDdaIpcArch()
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        return false;
    }
    const std::string arch(props.gcnArchName);
    return arch.find("gfx942") != std::string::npos
           || arch.find("gfx950") != std::string::npos;
}

// ASSERT_MPI_* skips healthy ranks, so rank 0 can hide a peer's failure.
int countFailingRanks(bool localOk)
{
    int failing = localOk ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &failing, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return failing;
}
} // namespace

class DdaMPI_AllGather : public MPITestBase
{
protected:
    int rank_{};
    int nRanks_{};

    void SetUp() override
    {
        MPITestBase::SetUp();
        if(!validateTestPrerequisites(kDdaIpcRanks, kDdaIpcRanks, false, 1, 1))
        {
            GTEST_SKIP() << "DDA IPC AllGather needs 8 ranks on one node";
        }
        const bool optedOut = rcclParamDdaEnable() == 0;
        std::string skip = mpiCoordinatedSkipReason(
            !isDdaIpcArch() || optedOut,
            optedOut ? "RCCL_DDA_ENABLE=0" : "DDA IPC needs gfx942 or gfx950");
        if(!skip.empty())
        {
            GTEST_SKIP() << skip;
        }

        const ncclResult_t res = createTestCommunicator();
        ASSERT_EQ(0, countFailingRanks(res == ncclSuccess))
            << "Rank " << getTestMpiRank() << ": " << ncclGetErrorString(res);
        ncclComm_t comm = getActiveCommunicator();
        ncclCommUserRank(comm, &rank_);
        ncclCommCount(comm, &nRanks_);

        // Implicit launch order disables DDA for the process lifetime.
        skip = mpiCoordinatedSkipReason(comm->config.launchOrderImplicit == 1,
                                        "NCCL_LAUNCH_ORDER_IMPLICIT=1");
        if(!skip.empty())
        {
            GTEST_SKIP() << skip;
        }
        ASSERT_EQ(0, countFailingRanks(comm->ddaIpcBarrierState != nullptr))
            << "Rank " << rank_ << ": DDA IPC did not initialize";
    }

    void runPlacement(size_t count, bool inPlace)
    {
        const size_t totalCount = count * static_cast<size_t>(nRanks_);

        void* recvBuf = nullptr;
        ASSERT_EQ(0, countFailingRanks(
                         hipMalloc(&recvBuf, totalCount * sizeof(float))
                         == hipSuccess));
        DeviceBufferAutoGuard recvGuard(recvBuf);
        ASSERT_EQ(0, countFailingRanks(
                         initializeBufferWithPattern<float>(
                             recvBuf, totalCount,
                             [](size_t) { return kUnwritten; })
                         == hipSuccess));

        void* sendBuf
            = static_cast<float*>(recvBuf) + static_cast<size_t>(rank_) * count;
        void* separateSendBuf = nullptr;
        if(!inPlace)
        {
            ASSERT_EQ(0, countFailingRanks(
                             hipMalloc(&separateSendBuf, count * sizeof(float))
                             == hipSuccess));
            sendBuf = separateSendBuf;
        }
        DeviceBufferAutoGuard sendGuard(separateSendBuf);
        ASSERT_EQ(0, countFailingRanks(
                         initializeBufferWithPattern<float>(
                             sendBuf, count,
                             [this](size_t i)
                             {
                                 return slicePattern(rank_, i);
                             })
                         == hipSuccess));

        // A kernel-path fallback also gathers correctly, so pin the backend.
        int algo = -1, protocol = -1, maxChannels = -1;
        const ncclResult_t queryRes = rcclGetCollImplInfo(
            getActiveCommunicator(), ncclFuncAllGather, count, ncclFloat32,
            ncclSum, sendBuf, recvBuf, /*graphCapturing=*/0, &algo, &protocol,
            &maxChannels);
        const char* algoName = nullptr;
        rcclGetAlgoName(algo, &algoName);
        ASSERT_EQ(0, countFailingRanks(queryRes == ncclSuccess
                                       && algo == RCCL_DDA_IPC))
            << "Rank " << rank_ << ": " << count << " floats selected "
            << (algoName ? algoName : "?") << ", not DDA-IPC";

        const ncclResult_t agRes = ncclAllGather(
            sendBuf, recvBuf, count, ncclFloat32, getActiveCommunicator(),
            getActiveStream());
        const hipError_t syncErr = hipStreamSynchronize(getActiveStream());
        ASSERT_EQ(0, countFailingRanks(agRes == ncclSuccess
                                       && syncErr == hipSuccess))
            << "Rank " << rank_ << ": " << ncclGetErrorString(agRes) << " / "
            << hipGetErrorString(syncErr);

        auto [copyErr, host] = downloadBuffer<float>(recvBuf, totalCount);
        size_t mismatches = copyErr == hipSuccess ? 0 : totalCount;
        size_t firstBad   = totalCount;
        for(size_t j = 0; copyErr == hipSuccess && j < totalCount; ++j)
        {
            if(host[j] != slicePattern(static_cast<int>(j / count), j % count))
            {
                firstBad = mismatches == 0 ? j : firstBad;
                ++mismatches;
            }
        }
        ASSERT_EQ(0, countFailingRanks(mismatches == 0))
            << "Rank " << rank_ << ": " << mismatches << " of " << totalCount
            << " wrong, first at " << firstBad << " (count " << count << ")";
    }
};

// recvbuff[r * count + i] must hold f(r, i) on every rank.
TEST_F(DdaMPI_AllGather, SlicePlacementOutOfPlace)
{
    for(size_t count : {kSmallCount, kGridCount})
    {
        runPlacement(count, /*inPlace=*/false);
        if(HasFailure())
        {
            return;
        }
    }
}

TEST_F(DdaMPI_AllGather, SlicePlacementInPlace)
{
    for(size_t count : {kSmallCount, kGridCount})
    {
        runPlacement(count, /*inPlace=*/true);
        if(HasFailure())
        {
            return;
        }
    }
}

#endif // MPI_TESTS_ENABLED
