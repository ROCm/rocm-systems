/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaAllGatherMPITests.cpp
 * @brief DDA IPC AllGather (gfx942/gfx950, 8 ranks): slice placement, and
 *        fallback when RCCL_DDA_THRESHOLD is above the IPC scratch.
 *
 * Run (example):
 *   mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=DdaMPI_AllGather.*
 *   mpirun -np 8 -x RCCL_DDA_THRESHOLD=1073741824 ./rccl-UnitTestsMPI
 *     --gtest_filter=DdaMPI_AllGatherThreshold.*
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

// Unique per rank and exact in fp32 at any count (max 4095 * 8 + 7 < 2^24).
float rankPattern(int rank, size_t i)
{
    return static_cast<float>((i % 4096) * 8 + static_cast<size_t>(rank));
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

class DdaIpcAllGatherMPITest : public MPITestBase
{
protected:
    // Empty when this job can reach DDA IPC; every rank gets the same answer. Call before comm init.
    std::string preInitSkipReason()
    {
        if(!validateTestPrerequisites(kDdaIpcRanks, kDdaIpcRanks, false, 1, 1))
        {
            return "DDA IPC AllGather needs 8 ranks on one node";
        }
        const bool optedOut = rcclParamDdaEnable() == 0;
        return mpiCoordinatedSkipReason(
            !isDdaIpcArch() || optedOut,
            optedOut ? "RCCL_DDA_ENABLE=0" : "DDA IPC needs gfx942 or gfx950");
    }

    // Implicit launch order disables DDA for the process lifetime.
    std::string launchOrderSkipReason()
    {
        return mpiCoordinatedSkipReason(
            getActiveCommunicator()->config.launchOrderImplicit == 1,
            "NCCL_LAUNCH_ORDER_IMPLICIT=1");
    }

    int selectedAlgo(const void* sendBuf, void* recvBuf, size_t count)
    {
        int algo        = -1;
        int protocol    = -1;
        int maxChannels = -1;
        EXPECT_EQ(ncclSuccess,
                  rcclGetCollImplInfo(getActiveCommunicator(),
                                      ncclFuncAllGather, count, ncclFloat32,
                                      ncclSum, sendBuf, recvBuf,
                                      /*graphCapturing=*/0, &algo, &protocol,
                                      &maxChannels));
        return algo;
    }
};

class DdaMPI_AllGather : public DdaIpcAllGatherMPITest
{
protected:
    int rank_{};
    int nRanks_{};

    void SetUp() override
    {
        MPITestBase::SetUp();
        std::string skip = preInitSkipReason();
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

        skip = launchOrderSkipReason();
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
        void*            separateSendBuf = nullptr;
        const hipError_t sendAllocErr
            = inPlace ? hipSuccess
                      : hipMalloc(&separateSendBuf, count * sizeof(float));
        DeviceBufferAutoGuard sendGuard(separateSendBuf);
        ASSERT_EQ(0, countFailingRanks(sendAllocErr == hipSuccess));
        if(!inPlace)
        {
            sendBuf = separateSendBuf;
        }
        ASSERT_EQ(0, countFailingRanks(
                         initializeBufferWithPattern<float>(
                             sendBuf, count,
                             [this](size_t i)
                             {
                                 return slicePattern(rank_, i);
                             })
                         == hipSuccess));

        // A kernel-path fallback also gathers correctly, so pin the backend.
        const int   algo     = selectedAlgo(sendBuf, recvBuf, count);
        const char* algoName = nullptr;
        rcclGetAlgoName(algo, &algoName);
        ASSERT_EQ(0, countFailingRanks(algo == RCCL_DDA_IPC))
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

class DdaMPI_AllGatherThreshold : public DdaIpcAllGatherMPITest
{
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

// A total above scratch but under the threshold must fall back, not fail.
TEST_F(DdaMPI_AllGatherThreshold, AboveScratchFallsBack)
{
    const size_t threshold = 4 * static_cast<size_t>(DDA_IPC_BUFFER_SIZE);
    // RCCL caches the param on first read, so set it before comm init.
    MPIHelpers::MpiEnvGuard thresholdGuard(
        "RCCL_DDA_THRESHOLD", std::to_string(threshold).c_str());
    std::string skip = preInitSkipReason();
    if(!skip.empty())
    {
        GTEST_SKIP() << skip;
    }
    skip = mpiCoordinatedSkipReason(
        static_cast<size_t>(rcclParamDdaThreshold()) != threshold,
        "RCCL_DDA_THRESHOLD was read earlier; run alone or set it at launch");
    if(!skip.empty())
    {
        GTEST_SKIP() << skip;
    }

    const ncclResult_t initRes = createTestCommunicator();
    ASSERT_EQ(0, countFailingRanks(initRes == ncclSuccess))
        << "Rank " << getTestMpiRank() << ": " << ncclGetErrorString(initRes);
    ncclComm* comm   = getActiveCommunicator();
    int       rank   = -1;
    int       nRanks = 0;
    ASSERT_EQ(0, countFailingRanks(ncclCommUserRank(comm, &rank) == ncclSuccess
                                   && ncclCommCount(comm, &nRanks) == ncclSuccess));
    skip = launchOrderSkipReason();
    if(!skip.empty())
    {
        GTEST_SKIP() << skip;
    }

    // 2x the per-rank share that fills scratch: 64 MiB each, 512 MiB total.
    const size_t ranks      = static_cast<size_t>(nRanks);
    const size_t fullCount  = comm->ddaScratchBytes / (ranks * sizeof(float));
    const size_t count      = 2 * fullCount;
    const size_t totalCount = count * ranks;
    const size_t totalBytes = totalCount * sizeof(float);
    ASSERT_EQ(0, countFailingRanks(count * sizeof(float) <= comm->ddaScratchBytes
                                   && totalBytes > comm->ddaScratchBytes
                                   && totalBytes <= threshold))
        << "Rank " << rank << ": scratch " << comm->ddaScratchBytes << " B, total "
        << totalBytes << " B, threshold " << threshold << " B";

    void*            sendBuf = nullptr;
    void*            recvBuf = nullptr;
    const hipError_t sendErr = hipMalloc(&sendBuf, count * sizeof(float));
    DeviceBufferAutoGuard sendGuard(sendBuf);
    const hipError_t recvErr = hipMalloc(&recvBuf, totalBytes);
    DeviceBufferAutoGuard recvGuard(recvBuf);
    ASSERT_EQ(0, countFailingRanks(sendErr == hipSuccess && recvErr == hipSuccess));

    // Control: a total that exactly fills scratch still takes DDA IPC.
    ASSERT_EQ(0, countFailingRanks(selectedAlgo(sendBuf, recvBuf, fullCount)
                                   == RCCL_DDA_IPC))
        << "Rank " << rank << ": DDA IPC not picked for a total that fills scratch";
    EXPECT_EQ(0, countFailingRanks(selectedAlgo(sendBuf, recvBuf, count)
                                   != RCCL_DDA_IPC))
        << "Rank " << rank << ": DDA IPC picked for a total above scratch";

    const hipError_t sendInitErr = initializeBufferWithPattern<float>(
        sendBuf, count, [rank](size_t i) { return rankPattern(rank, i); });
    const hipError_t recvInitErr = initializeBufferWithPattern<float>(
        recvBuf, totalCount, [](size_t) { return kUnwritten; });
    ASSERT_EQ(0, countFailingRanks(sendInitErr == hipSuccess
                                   && recvInitErr == hipSuccess));

    const ncclResult_t agRes
        = ncclAllGather(sendBuf, recvBuf, count, ncclFloat32, comm, getActiveStream());
    const hipError_t syncErr = hipStreamSynchronize(getActiveStream());
    ASSERT_EQ(0, countFailingRanks(agRes == ncclSuccess && syncErr == hipSuccess))
        << "Rank " << rank << ": " << ncclGetErrorString(agRes) << " / "
        << hipGetErrorString(syncErr);

    size_t     firstBad = totalCount;
    const bool dataOk   = verifyBufferData<float>(
        recvBuf, totalCount,
        [count](size_t j) { return rankPattern(static_cast<int>(j / count), j % count); },
        /*num_samples=*/0, /*tolerance=*/0.0, &firstBad);
    ASSERT_EQ(0, countFailingRanks(dataOk))
        << "Rank " << rank << ": first mismatch at " << firstBad << " of " << totalCount;
}

#endif // MPI_TESTS_ENABLED
