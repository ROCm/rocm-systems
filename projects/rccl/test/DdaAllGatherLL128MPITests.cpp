/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file DdaAllGatherLL128MPITests.cpp
 * @brief MPI end-to-end tests for DDA fabric AllGather LL128.
 *
 * The block size follows the bytes each rank pushes. Covers the smallest LL128 size (partial
 * last slice), whole slices only, one size per block-size band (checked in the launch log), a
 * low block cap so each warp walks many slices, and in-place calls. Each call uses fresh data.
 *
 * Run (example):
 *   mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter='DdaMPI_AllGatherLL128.*'
 */

#ifdef MPI_TESTS_ENABLED

#include "DdaFabricMPITestFixture.hpp"
#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "algorithms/dda/all_gather/all_gather_dda_fabric_ll128.h"
#include "comm.h"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <sstream>
#include <string>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
// Four calls use each scratch bank twice.
constexpr int  kIterations          = 4;
constexpr auto kDrainTimeout        = std::chrono::seconds(120);
// Few enough blocks that each warp handles many slices.
constexpr char kLowMaxBlocks[]      = "8";
constexpr char kLl128LaunchNeedle[] = "DDA fabric AllGather LL128: ";
// Block-size bands in dda_all_gather_fabric_ll128.cu, by bytes each rank pushes.
constexpr size_t kSmallMaxPushBytes = 4ULL << 20;
constexpr size_t kMidMaxPushBytes   = 16ULL << 20;

// Block size of the first LL128 launch with this per-rank size in the log, or 0 if none.
unsigned loggedBlock(const MPIHelpers::TestLogAssertionContext& logCtx, size_t perRankBytes)
{
    std::istringstream log(logCtx.readNcclDebugLog() + logCtx.readPerRankStderrLog());
    const std::string  key = "perRankBytes=" + std::to_string(perRankBytes) + " ";
    for(std::string line; std::getline(log, line);)
    {
        const size_t at = line.find(kLl128LaunchNeedle);
        if(at == std::string::npos || line.find(key, at) == std::string::npos)
            continue;
        const size_t b = line.find("block=", at);
        if(b != std::string::npos)
            return static_cast<unsigned>(std::stoul(line.substr(b + 6)));
    }
    return 0;
}

// Smallest 16-byte-aligned per-rank size past the AllGather LL cap.
size_t smallestBytes(const ncclComm* comm, int nRanks)
{
    const size_t n     = static_cast<size_t>(nRanks);
    const size_t llCap = rcclDdaLLThreshold(comm, ncclFuncAllGather);
    size_t       bytes = (llCap / n + 16) & ~size_t{15};
    while(bytes * n <= llCap)
        bytes += 16;
    return bytes;
}

// Per-rank size whose pushed bytes, (nRanks - 1) x per-rank, are at or just past `pushBytes`.
size_t bytesForPush(size_t pushBytes, int nRanks, bool past)
{
    const size_t perRank = (pushBytes / static_cast<size_t>(nRanks - 1)) & ~size_t{15};
    return past ? perRank + 16 : perRank;
}
} // namespace

class DdaMPI_AllGatherLL128 : public DdaFabricMPITest
{
protected:
    using SizeFn = std::function<size_t(const ncclComm*, int)>;

    // Gathers sizeFor(comm, nRanks) bytes per rank kIterations times with fresh data and checks
    // every call took LL128. A nonzero `threads` also checks the block size of the launch.
    // Must be the last statement of the test body: the coordinated skips and asserts return from here.
    void runAllGather(const SizeFn& sizeFor, bool inPlace, unsigned threads = 0, const char* maxBlocks = nullptr)
    {
        if(!validateTestPrerequisites(kMinProcessesForMPI))
        {
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        }

        DDA_FABRIC_SKIP_OR_FAIL(archSkipReason("DDA fabric AllGather LL128 requires gfx1250"));

        // Fabric init reads RCCL_DDA_FABRIC_MAXBLOCKS with getenv on every comm, so the override is per test.
        std::unique_ptr<MPIHelpers::MpiEnvGuard> maxBlocksGuard;
        if(maxBlocks != nullptr)
        {
            maxBlocksGuard = std::make_unique<MPIHelpers::MpiEnvGuard>("RCCL_DDA_FABRIC_MAXBLOCKS", maxBlocks);
        }

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

        DDA_FABRIC_SKIP_OR_FAIL(fabricInitSkipReason());

        ncclComm_t comm = getActiveCommunicator();
        // These switch the DDA LL128 tier off, or pin the block size, for the process.
        const bool off = rcclNcclAlgoEnvIsSet() || rcclParamDdaEnable() == 0 || comm->config.launchOrderImplicit == 1
                         || rcclDdaLL128Threshold(comm, ncclFuncAllGather) == 0
                         || (threads != 0 && std::getenv("RCCL_DDA_LL128_AG_THREADS") != nullptr);
        const std::string offSkip = mpiCoordinatedSkipReason(
            off, "DDA AllGather LL128 off or block size pinned (NCCL_ALGO, RCCL_DDA_ENABLE, thresholds, "
                 "implicit launch order, RCCL_DDA_LL128_AG_THREADS)");
        if(!offSkip.empty())
        {
            GTEST_SKIP() << offSkip;
        }

        if(maxBlocks != nullptr)
        {
            ASSERT_MPI_EQ(std::atoi(maxBlocks), comm->ddaFabricMaxBlocks);
        }

        int rank = -1, nRanks = 0;
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(comm, &rank));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(comm, &nRanks));

        // The size must take LL128: past the LL cap, inside the LL128 cap and the per-rank slot.
        const size_t bytes = sizeFor(comm, nRanks);
        const size_t total = bytes * static_cast<size_t>(nRanks);
        const size_t slotBytes =
            dda::common::ddaLL128AgSlotWords(dda::common::ddaBankSize(comm->ddaScratchBytes), nRanks)
            / static_cast<size_t>(dda::common::kDdaLL128WireWordsPerSlice)
            * static_cast<size_t>(dda::common::kDdaLL128DataBytesPerSlice);
        SCOPED_TRACE("per-rank bytes " + std::to_string(bytes));
        ASSERT_MPI_TRUE(total > rcclDdaLLThreshold(comm, ncclFuncAllGather)
                        && total <= rcclDdaLL128Threshold(comm, ncclFuncAllGather) && bytes <= slotBytes);

        const size_t count      = bytes / sizeof(float);
        const size_t totalCount = count * static_cast<size_t>(nRanks);
        void*        recvBuf    = nullptr;
        ASSERT_MPI_EQ(hipSuccess, hipMalloc(&recvBuf, totalCount * sizeof(float)));
        DeviceBufferAutoGuard recvGuard(recvBuf);
        void* sendBuf         = static_cast<char*>(recvBuf) + static_cast<size_t>(rank) * bytes;
        void* separateSendBuf = nullptr;
        if(!inPlace)
        {
            ASSERT_MPI_EQ(hipSuccess, hipMalloc(&separateSendBuf, bytes));
            sendBuf = separateSendBuf;
        }
        DeviceBufferAutoGuard sendGuard(separateSendBuf);

        // Exact in float: every value stays far below 2^24.
        auto value = [](int r, size_t i, int iter) {
            return static_cast<float>((r + 1) * 4096 + static_cast<int>(i % 1021) + iter * 37);
        };
        for(int iter = 0; iter < kIterations; ++iter)
        {
            ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0xff, totalCount * sizeof(float)));
            ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<float>(
                sendBuf, count, [&](size_t i) { return value(rank, i, iter); }));
            const ncclResult_t launchRes = ncclAllGather(sendBuf, recvBuf, count, ncclFloat32, comm, getActiveStream());
            // A kernel spinning on a peer that never pushed aborts the job instead of hanging it.
            DdaFabricMPI::abortJobUnlessAllRanksDrained(
                DdaFabricMPI::waitStreamBounded(getActiveStream(), kDrainTimeout), "DDA AllGather LL128");
            ASSERT_MPI_EQ(ncclSuccess, launchRes);
            ASSERT_MPI_TRUE(verifyBufferData<float>(recvBuf, totalCount, [&](size_t i) {
                return value(static_cast<int>(i / count), i % count, iter);
            }));
        }

        // Every call must take LL128; a fallback would still gather correctly.
        const std::string launch = std::string(kLl128LaunchNeedle) + "nRanks=" + std::to_string(nRanks)
                                   + " perRankBytes=" + std::to_string(bytes) + " ";
        EXPECT_EQ(countLogNeedle(launch.c_str()), static_cast<size_t>(kIterations)) << "Rank " << rank;
        if(threads != 0)
        {
            EXPECT_EQ(loggedBlock(*logCtx_, bytes), threads) << "Rank " << rank;
        }
    }
};

TEST_F(DdaMPI_AllGatherLL128, SmallestSize)
{
    runAllGather(smallestBytes, false);
}

TEST_F(DdaMPI_AllGatherLL128, SmallestSizeInPlace)
{
    runAllGather(smallestBytes, true);
}

// A per-rank size of whole slices, so no slice is partial.
TEST_F(DdaMPI_AllGatherLL128, WholeSlicesOnly)
{
    runAllGather(
        [](const ncclComm* comm, int nRanks) {
            const size_t slice = dda::common::kDdaLL128DataBytesPerSlice;
            return (smallestBytes(comm, nRanks) / slice + 1) * slice;
        },
        false);
}

TEST_F(DdaMPI_AllGatherLL128, SmallBlocksUpTo4MiBPushed)
{
    runAllGather([](const ncclComm*, int nRanks) { return bytesForPush(kSmallMaxPushBytes, nRanks, false); }, false,
                 128);
}

TEST_F(DdaMPI_AllGatherLL128, MidBlocksPast4MiBPushed)
{
    runAllGather([](const ncclComm*, int nRanks) { return bytesForPush(kSmallMaxPushBytes, nRanks, true); }, false,
                 256);
}

TEST_F(DdaMPI_AllGatherLL128, LargeBlocksPast16MiBPushed)
{
    runAllGather([](const ncclComm*, int nRanks) { return bytesForPush(kMidMaxPushBytes, nRanks, true); }, false,
                 512);
}

TEST_F(DdaMPI_AllGatherLL128, ManySlicesPerWarp)
{
    runAllGather([](const ncclComm*, int nRanks) { return bytesForPush(kSmallMaxPushBytes, nRanks, false); }, false,
                 0, kLowMaxBlocks);
}

TEST_F(DdaMPI_AllGatherLL128, ManySlicesPerWarpLargeBlocks)
{
    runAllGather([](const ncclComm*, int nRanks) { return bytesForPush(kMidMaxPushBytes, nRanks, true); }, false,
                 0, kLowMaxBlocks);
}

TEST_F(DdaMPI_AllGatherLL128, ManySlicesPerWarpInPlace)
{
    runAllGather([](const ncclComm*, int nRanks) { return bytesForPush(kSmallMaxPushBytes, nRanks, true); }, true,
                 0, kLowMaxBlocks);
}

#endif // MPI_TESTS_ENABLED
