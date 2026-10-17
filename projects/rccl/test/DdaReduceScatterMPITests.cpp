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
 * in place, for bf16, and across repeated launches. Each launch uses a new
 * input and is checked before the next one, so a stale bank or a missed epoch
 * bump cannot hide behind the previous sum. A near-cap launch lowers the block
 * cap so each warp walks the slice loop more than once. The COLL log confirms
 * the LL128 launcher actually ran.
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
#include "algorithms/dda/device/CollCommon_ll128.h"
#include "rccl_common.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <memory>
#include <string>
#include <type_traits>

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
constexpr int  kRepeatedIterations = 4;
// 32 blocks is under a gfx1250 CU cap, so the override only lowers the grid.
// 32 * (512/32) = 512 warps cover the smallest LL128 shard in one pass and
// leave a near-cap shard (at least 2 MiB) for a second pass.
constexpr char kNearCapMaxBlocks[] = "32";
constexpr int  kLl128Threads       = 512;
constexpr char kLl128PathNeedle[]   = "taking DDA fabric LL128 path";
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

// Largest 16-byte-aligned shard the LL128 cap and this rank's slot both admit.
size_t ll128NearCapShardBytes(ncclComm* comm)
{
    const int nRanks = comm->nRanks;
    if(nRanks < 2)
        return 0;
    const size_t llCap    = rcclDdaLLThreshold(comm, ncclFuncReduceScatter);
    const size_t ll128Cap = rcclDdaLL128Threshold(comm, ncclFuncReduceScatter);
    if(ll128Cap <= llCap)
        return 0;
    size_t shard = (ll128Cap / static_cast<size_t>(nRanks)) & ~size_t{15};
    const size_t slotWords = dda::common::ddaLL128SlotWords(
        dda::common::ddaBankSize(comm->ddaScratchBytes), nRanks, 1);
    const size_t maxSlices = slotWords / static_cast<size_t>(dda::common::kDdaLL128WireWordsPerSlice);
    const size_t maxBytes  = maxSlices * static_cast<size_t>(dda::common::kDdaLL128DataBytesPerSlice);
    if(shard > maxBytes)
        shard = maxBytes & ~size_t{15};
    if(shard == 0 || shard * static_cast<size_t>(nRanks) <= llCap
       || shard * static_cast<size_t>(nRanks) > ll128Cap)
        return 0;
    return shard;
}

// True when the launcher's grid cannot cover the shard in one slice stride.
bool shardTakesAnotherPass(const ncclComm* comm, size_t shardBytes)
{
    const size_t slices = dda::common::ddaLL128Slices(shardBytes);
    const int warpsPerBlock = kLl128Threads / dda::common::kDdaLL128Warp;
    int blocks = comm->ddaFabricMaxBlocks;
    if(blocks < 1)
        blocks = 1;
    const size_t wantBlocks =
        (slices + static_cast<size_t>(warpsPerBlock) - 1) / static_cast<size_t>(warpsPerBlock);
    if(wantBlocks < static_cast<size_t>(blocks))
        blocks = static_cast<int>(wantBlocks);
    if(comm->ddaLLEpochLen > 0 && blocks > comm->ddaLLEpochLen)
        blocks = comm->ddaLLEpochLen;
    if(blocks < 1)
        blocks = 1;
    const size_t warps = static_cast<size_t>(blocks) * static_cast<size_t>(warpsPerBlock);
    return slices > warps;
}
} // namespace

class DdaReduceScatterMPITest : public MPITestBase
{
protected:
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugSubsysGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             maxBlocksGuard_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;
    int                                                  rank_{};
    int                                                  nRanks_{};
    size_t                                               shardBytes_{};

    // Empty keeps the CU-derived grid. A derived fixture can lower it so a
    // near-cap shard takes more than one pass of the slice loop.
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
        if(const char* value = maxBlocksOverride())
            maxBlocksGuard_ =
                std::make_unique<MPIHelpers::MpiEnvGuard>("RCCL_DDA_FABRIC_MAXBLOCKS", value);
        logCtx_ = std::make_unique<MPIHelpers::TestLogAssertionContext>(
            MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));

        if(!validateTestPrerequisites(kMinProcessesForMPI))
            GTEST_SKIP() << "Need at least 2 MPI ranks";
        if(!isGfx1250Device())
            GTEST_SKIP() << "DDA fabric ReduceScatter LL128 requires gfx1250";

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
        ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(getActiveCommunicator(), &rank_));
        ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(getActiveCommunicator(), &nRanks_));

        // Fabric init reports success and leaves the handler null when VMM or
        // the clique is unavailable. On gfx1250 that is a failure, not a skip.
        // A zero shard means the RS thresholds no longer leave an LL128 window.
        shardBytes_ = ll128ShardBytes(getActiveCommunicator());
        ASSERT_MPI_TRUE(getActiveCommunicator()->ddaFabricMemHandler != nullptr);
        ASSERT_MPI_TRUE(getActiveCommunicator()->ddaLLEpochDev != nullptr);
        ASSERT_MPI_NE(shardBytes_, static_cast<size_t>(0));
    }

    void TearDown() override
    {
        MPITestBase::TearDown();
        logCtx_.reset();
        maxBlocksGuard_.reset();
        debugSubsysGuard_.reset();
        debugGuard_.reset();
    }

    template <typename T>
    void runReduceScatter(ncclDataType_t dtype, size_t recvCount, bool inPlace, int iterations)
    {
        const size_t totalCount = recvCount * static_cast<size_t>(nRanks_);
        const size_t sendBytes  = totalCount * sizeof(T);
        const size_t recvBytes  = recvCount * sizeof(T);

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

        for(int iter = 0; iter < iterations; ++iter)
        {
            // A fresh pattern each launch: replaying one input would still sum
            // correctly if this call read the previous call's lines.
            ASSERT_MPI_EQ(hipSuccess, initializeBufferWithPattern<T>(
                sendBuf, totalCount, [this, iter](size_t i) {
                    if constexpr(std::is_same<T, float>::value)
                    {
                        return static_cast<T>(rank_ + 1 + iter + static_cast<int>(i % 17));
                    }
                    else
                    {
                        // Powers of two stay exact in bf16, including the sum of one
                        // per rank, for every clique this path allows.
                        return static_cast<T>(static_cast<float>(1 << iter));
                    }
                }));
            if(!inPlace)
                ASSERT_MPI_EQ(hipSuccess, hipMemset(recvBuf, 0, recvBytes));

            ASSERT_MPI_EQ(ncclSuccess,
                          ncclReduceScatter(sendBuf, recvBuf, recvCount, dtype, ncclSum,
                                            getActiveCommunicator(), getActiveStream()));
            ASSERT_MPI_EQ(hipSuccess, hipStreamSynchronize(getActiveStream()));

            ASSERT_MPI_TRUE(verifyBufferData<T>(
                recvBuf, recvCount, [this, iter, recvCount](size_t i) {
                    if constexpr(std::is_same<T, float>::value)
                    {
                        const float rankSum = static_cast<float>(
                            nRanks_ * (nRanks_ + 1) / 2 + nRanks_ * iter);
                        const size_t globalIdx = static_cast<size_t>(rank_) * recvCount + i;
                        return static_cast<T>(rankSum
                                              + static_cast<float>(nRanks_ * static_cast<int>(globalIdx % 17)));
                    }
                    else
                    {
                        return static_cast<T>(static_cast<float>(nRanks_ * (1 << iter)));
                    }
                }));
        }

        EXPECT_TRUE(logContains(*logCtx_, kLl128PathNeedle))
            << "Rank " << rank_ << " did not select DDA fabric ReduceScatter LL128";
        EXPECT_TRUE(logContains(*logCtx_, kLl128LaunchNeedle))
            << "Rank " << rank_ << " did not launch the ReduceScatter LL128 kernel";
    }
};

class DdaMPI_ReduceScatter : public DdaReduceScatterMPITest
{};

// Lowers the grid before the communicator is created, so the near-cap shard
// cannot fit in one pass of the slice loop on any legal rank count.
class DdaReduceScatterNearCapMPITest : public DdaReduceScatterMPITest
{
protected:
    const char* maxBlocksOverride() const override
    {
        return kNearCapMaxBlocks;
    }
};

class DdaMPI_ReduceScatterNearCap : public DdaReduceScatterNearCapMPITest
{};

TEST_F(DdaMPI_ReduceScatter, LL128OutOfPlace)
{
    runReduceScatter<float>(ncclFloat32, shardBytes_ / sizeof(float), false, 1);
}

TEST_F(DdaMPI_ReduceScatter, LL128InPlace)
{
    runReduceScatter<float>(ncclFloat32, shardBytes_ / sizeof(float), true, 1);
}

TEST_F(DdaMPI_ReduceScatter, LL128Repeated)
{
    runReduceScatter<float>(ncclFloat32, shardBytes_ / sizeof(float), false, kRepeatedIterations);
}

TEST_F(DdaMPI_ReduceScatter, LL128Bfloat16)
{
    runReduceScatter<hip_bfloat16>(ncclBfloat16, shardBytes_ / sizeof(hip_bfloat16), false, kRepeatedIterations);
}

TEST_F(DdaMPI_ReduceScatterNearCap, LL128NearCap)
{
    const size_t nearCapBytes = ll128NearCapShardBytes(getActiveCommunicator());
    ASSERT_MPI_NE(nearCapBytes, static_cast<size_t>(0));
    ASSERT_MPI_TRUE(shardTakesAnotherPass(getActiveCommunicator(), nearCapBytes));
    runReduceScatter<float>(ncclFloat32, nearCapBytes / sizeof(float), false, kRepeatedIterations);
}

#endif // MPI_TESTS_ENABLED
