/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "DeviceBufferHelpers.hpp"
#include <cstring>
#include <thread>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace GrowTestConfig {
    constexpr size_t kBufferElements = 1024;
    constexpr int    kMinRanks       = 2;
    constexpr int    kMinNodes       = 2;
}

class GrowMPITest : public MPITestBase
{
protected:
    ncclComm_t  initialComm_ = nullptr;
    ncclComm_t  midComm_     = nullptr;
    ncclComm_t  grownComm_   = nullptr;
    hipStream_t grownStream_ = nullptr;

    void TearDown() override
    {
        if (grownComm_)   { (void)ncclCommDestroy(grownComm_);   grownComm_   = nullptr; }
        if (midComm_)     { (void)ncclCommDestroy(midComm_);     midComm_     = nullptr; }
        if (initialComm_) { (void)ncclCommDestroy(initialComm_); initialComm_ = nullptr; }
        if (grownStream_) { (void)hipStreamDestroy(grownStream_); grownStream_ = nullptr; }
        MPITestBase::TearDown();
    }

    ncclResult_t buildComm(int nRanks, ncclComm_t* outComm)
    {
        const int wr = MPIEnvironment::world_rank;
        ncclUniqueId initId{};
        if (wr == 0) {
            RCCL_TEST_CHECK(ncclGetUniqueId(&initId));
        }
        MPI_Bcast(&initId, sizeof(initId), MPI_BYTE, 0, MPI_COMM_WORLD);
        if (wr < nRanks) {
            RCCL_TEST_CHECK(ncclCommInitRank(outComm, nRanks, initId, wr));
        }
        return ncclSuccess;
    }

    ncclResult_t growByOne(ncclComm_t existingComm, int existingNRanks, ncclComm_t* outComm)
    {
        const int wr       = MPIEnvironment::world_rank;
        const int newRank  = existingNRanks;
        const int newTotal = existingNRanks + 1;

        ncclUniqueId growId{};
        if (wr == 0) {
            RCCL_TEST_CHECK(ncclCommGetUniqueId(existingComm, &growId));
        }
        MPI_Bcast(&growId, sizeof(growId), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (wr < existingNRanks) {
            RCCL_TEST_CHECK(ncclCommGrow(existingComm, newTotal, &growId, -1,
                                         outComm, nullptr));
        } else if (wr == newRank) {
            RCCL_TEST_CHECK(ncclCommGrow(nullptr, newTotal, &growId, newRank,
                                         outComm, nullptr));
        }
        return ncclSuccess;
    }

    bool runAllReduceAndVerify(ncclComm_t comm, int nRanksInComm)
    {
        if (!comm) return false;

        if (!grownStream_) {
            if (hipStreamCreate(&grownStream_) != hipSuccess) return false;
        }

        const size_t count   = GrowTestConfig::kBufferElements;
        const size_t bufSize = count * sizeof(float);

        void* sendBuf = nullptr;
        void* recvBuf = nullptr;
        if (hipMalloc(&sendBuf, bufSize) != hipSuccess) return false;
        if (hipMalloc(&recvBuf, bufSize) != hipSuccess) {
            (void)hipFree(sendBuf);
            return false;
        }
        SCOPE_EXIT(if (sendBuf) (void)hipFree(sendBuf));
        SCOPE_EXIT(if (recvBuf) (void)hipFree(recvBuf));

        if (initializeBufferWithPattern<float>(sendBuf, count,
                [](size_t) { return 1.0f; }) != hipSuccess) {
            return false;
        }
        if (hipMemset(recvBuf, 0, bufSize) != hipSuccess) return false;

        ncclResult_t r = ncclAllReduce(sendBuf, recvBuf, count,
                                       ncclFloat, ncclSum, comm, grownStream_);
        if (r != ncclSuccess) return false;

        if (hipStreamSynchronize(grownStream_) != hipSuccess) return false;

        const float expected = static_cast<float>(nRanksInComm);
        return verifyBufferData<float>(recvBuf, count,
            [expected](size_t) { return expected; });
    }
};

// --- Test 1: Single grow, AllReduce verification ---

TEST_F(GrowMPITest, Grow_ExistingRanksGrow_AllreduceCorrect)
{
    if (!validateTestPrerequisites(GrowTestConfig::kMinRanks)) {
        GTEST_SKIP() << "Requires at least " << GrowTestConfig::kMinRanks << " MPI ranks";
    }

    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 1;

    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    ASSERT_MPI_EQ(ncclSuccess, growByOne(initialComm_, existing, &grownComm_));
    ASSERT_MPI_NE(grownComm_, nullptr);

    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));
}

// --- Test 2: Multi-node grow ---

TEST_F(GrowMPITest, Grow_MultiNode_ExistingRanksGrow_AllreduceCorrect)
{
    if (!validateTestPrerequisites(GrowTestConfig::kMinRanks,
                                    kNoProcessLimit,
                                    /*power_of_two=*/false,
                                    GrowTestConfig::kMinNodes)) {
        GTEST_SKIP() << "Requires "
                     << GrowTestConfig::kMinRanks << "+ ranks across "
                     << GrowTestConfig::kMinNodes << "+ nodes";
    }

    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 1;

    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    ASSERT_MPI_EQ(ncclSuccess, growByOne(initialComm_, existing, &grownComm_));
    ASSERT_MPI_NE(grownComm_, nullptr);

    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));
}

// --- Test 3: Unique ID handle consistency ---

TEST_F(GrowMPITest, Grow_GetUniqueId_BroadcastedToAll_HandleConsistent)
{
    if (!validateTestPrerequisites(GrowTestConfig::kMinRanks)) {
        GTEST_SKIP() << "Requires at least " << GrowTestConfig::kMinRanks << " MPI ranks";
    }

    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 1;

    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    ncclUniqueId growId{};
    if (wr == 0) {
        ASSERT_EQ(ncclSuccess, ncclCommGetUniqueId(initialComm_, &growId));
    }
    MPI_Bcast(&growId, sizeof(growId), MPI_BYTE, 0, MPI_COMM_WORLD);

    uint64_t myMagic = 0;
    std::memcpy(&myMagic, growId.internal, sizeof(myMagic));

    uint64_t minMagic = 0, maxMagic = 0;
    MPI_Allreduce(&myMagic, &minMagic, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&myMagic, &maxMagic, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

    ASSERT_MPI_EQ(minMagic, maxMagic);
    ASSERT_MPI_NE(myMagic, static_cast<uint64_t>(0));

    // Complete the grow to drain the queued bootstrap message on the boundary rank.
    ASSERT_MPI_EQ(ncclSuccess, growByOne(initialComm_, existing, &grownComm_));
    ASSERT_MPI_NE(grownComm_, nullptr);
}

// --- Test 4: Double grow (N-2 -> N-1 -> N) ---

TEST_F(GrowMPITest, Grow_DoubleGrow_SecondGrowWorks)
{
    if (MPIEnvironment::world_size < 4) {
        GTEST_SKIP() << "DoubleGrow requires at least 4 MPI ranks";
    }
    if (!validateTestPrerequisites(4)) {
        GTEST_SKIP() << "Requires at least 4 MPI ranks";
    }

    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int phase0    = worldSize - 2;
    const int phase1    = worldSize - 1;
    const int phase2    = worldSize;

    ASSERT_MPI_EQ(ncclSuccess, buildComm(phase0, &initialComm_));
    ASSERT_MPI_TRUE(wr >= phase0 || initialComm_ != nullptr);

    ASSERT_MPI_EQ(ncclSuccess, growByOne(initialComm_, phase0, &midComm_));
    ASSERT_MPI_TRUE(wr >= phase1 || midComm_ != nullptr);

    ASSERT_MPI_EQ(ncclSuccess, growByOne(midComm_, phase1, &grownComm_));
    ASSERT_MPI_NE(grownComm_, nullptr);

    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, phase2));
}

// --- Test 5: Grow then abort ---

TEST_F(GrowMPITest, Grow_ThenRevoke_CleanLifecycle)
{
    if (!validateTestPrerequisites(GrowTestConfig::kMinRanks)) {
        GTEST_SKIP() << "Requires at least " << GrowTestConfig::kMinRanks << " MPI ranks";
    }

    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 1;

    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    ASSERT_MPI_EQ(ncclSuccess, growByOne(initialComm_, existing, &grownComm_));
    ASSERT_MPI_NE(grownComm_, nullptr);

    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));

    ASSERT_MPI_EQ(ncclSuccess, ncclCommAbort(grownComm_));

    (void)ncclCommDestroy(grownComm_);
    grownComm_ = nullptr;

    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));
}

#endif // MPI_TESTS_ENABLED
