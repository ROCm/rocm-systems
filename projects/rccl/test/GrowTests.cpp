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
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace GrowTestConfig {
    constexpr size_t kBufferElements = 1024;
    constexpr int    kMinRanks       = 2;
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

    bool ensureStream()
    {
        if (!grownStream_) {
            return hipStreamCreate(&grownStream_) == hipSuccess;
        }
        return true;
    }

    bool runAllReduceAndVerify(ncclComm_t comm, int nRanksInComm)
    {
        if (!comm) return false;
        if (!ensureStream()) return false;

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

    ncclResult_t growByOneNcclConvention(ncclComm_t existingComm, int existingNRanks, ncclComm_t* outComm)
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
            if (wr == 0) {
                RCCL_TEST_CHECK(ncclCommGrow(existingComm, newTotal, &growId, -1,
                                             outComm, nullptr));
            } else {
                RCCL_TEST_CHECK(ncclCommGrow(existingComm, newTotal, nullptr, -1,
                                             outComm, nullptr));
            }
        } else if (wr == newRank) {
            RCCL_TEST_CHECK(ncclCommGrow(nullptr, newTotal, &growId, newRank,
                                         outComm, nullptr));
        }
        return ncclSuccess;
    }

    bool runSendRecvRing(ncclComm_t comm, int nRanksInComm)
    {
        if (!comm) return false;
        if (!ensureStream()) return false;

        int myRank = -1;
        if (ncclCommUserRank(comm, &myRank) != ncclSuccess) return false;

        const int sendPeer = (myRank + 1) % nRanksInComm;
        const int recvPeer = (myRank - 1 + nRanksInComm) % nRanksInComm;

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

        const float sendVal = static_cast<float>(myRank + 1);
        if (initializeBufferWithPattern<float>(sendBuf, count,
                [sendVal](size_t) { return sendVal; }) != hipSuccess) {
            return false;
        }
        if (hipMemset(recvBuf, 0, bufSize) != hipSuccess) return false;

        ncclResult_t r;
        r = ncclGroupStart();
        if (r != ncclSuccess) return false;

        r = ncclSend(sendBuf, count, ncclFloat, sendPeer, comm, grownStream_);
        if (r != ncclSuccess) return false;

        r = ncclRecv(recvBuf, count, ncclFloat, recvPeer, comm, grownStream_);
        if (r != ncclSuccess) return false;

        r = ncclGroupEnd();
        if (r != ncclSuccess) return false;

        if (hipStreamSynchronize(grownStream_) != hipSuccess) return false;

        const float expected = static_cast<float>(recvPeer + 1);
        return verifyBufferData<float>(recvBuf, count,
            [expected](size_t) { return expected; });
    }
};

// --- Single grow, SendRecv ring verification ---

TEST_F(GrowMPITest, Grow_SendRecv)
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

    ASSERT_MPI_TRUE(runSendRecvRing(grownComm_, worldSize));
}

// --- Double grow (N-2 -> N-1 -> N), AllReduce verification ---

TEST_F(GrowMPITest, DoubleGrow_AllReduce)
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

// --- Grow then abort ---

TEST_F(GrowMPITest, Grow_ThenAbort)
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
    grownComm_ = nullptr;

    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));
}

// --- Coordinator-only uniqueId (NCCL convention) ---

TEST_F(GrowMPITest, Grow_CoordinatorOnlyUniqueId)
{
    if (!validateTestPrerequisites(3)) {
        GTEST_SKIP() << "Requires at least 3 MPI ranks";
    }

    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 1;

    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    ASSERT_MPI_EQ(ncclSuccess, growByOneNcclConvention(initialComm_, existing, &grownComm_));
    ASSERT_MPI_NE(grownComm_, nullptr);

    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));
}

// --- Rank preservation after grow ---

TEST_F(GrowMPITest, Grow_RankPreservation)
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

    int grownRank = -1, grownCount = -1;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommUserRank(grownComm_, &grownRank));
    ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(grownComm_, &grownCount));

    ASSERT_MPI_EQ(grownCount, worldSize);

    int expectedRank;
    if (wr < existing) {
        int originalRank = -1;
        ASSERT_EQ(ncclSuccess, ncclCommUserRank(initialComm_, &originalRank));
        expectedRank = originalRank;
    } else {
        expectedRank = existing;
    }
    ASSERT_MPI_EQ(grownRank, expectedRank);
}

// --- Config inheritance from parent ---

TEST_F(GrowMPITest, Grow_ConfigInheritance)
{
    if (!validateTestPrerequisites(GrowTestConfig::kMinRanks)) {
        GTEST_SKIP() << "Requires at least " << GrowTestConfig::kMinRanks << " MPI ranks";
    }

    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 1;

    ncclUniqueId initId{};
    if (wr == 0) {
        ASSERT_EQ(ncclSuccess, ncclGetUniqueId(&initId));
    }
    MPI_Bcast(&initId, sizeof(initId), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (wr < existing) {
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.splitShare = 1;
        ASSERT_EQ(ncclSuccess, ncclCommInitRankConfig(&initialComm_, existing, initId, wr, &config));
    }

    ASSERT_MPI_EQ(ncclSuccess, growByOne(initialComm_, existing, &grownComm_));
    ASSERT_MPI_NE(grownComm_, nullptr);

    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));
}

// --- Non-blocking grow ---

TEST_F(GrowMPITest, Grow_NonBlocking)
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

    ncclConfig_t nbConfig = NCCL_CONFIG_INITIALIZER;
    nbConfig.blocking = 0;

    if (wr < existing) {
        ASSERT_EQ(ncclSuccess, ncclCommGrow(initialComm_, worldSize, &growId, -1,
                                            &grownComm_, &nbConfig));
    } else if (wr == existing) {
        ASSERT_EQ(ncclSuccess, ncclCommGrow(nullptr, worldSize, &growId, wr,
                                            &grownComm_, &nbConfig));
    }

    ASSERT_MPI_NE(grownComm_, nullptr);

    ncclResult_t asyncErr = ncclInProgress;
    constexpr int kMaxPollIter = 1000000;
    for (int i = 0; i < kMaxPollIter && asyncErr == ncclInProgress; ++i) {
        ASSERT_EQ(ncclSuccess, ncclCommGetAsyncError(grownComm_, &asyncErr));
    }
    ASSERT_MPI_EQ(asyncErr, ncclSuccess);

    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));
}

// --- Destroy parent after grow, use grown comm ---

TEST_F(GrowMPITest, Grow_ParentDestroyAfterGrow)
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

    ncclResult_t destroyRes = ncclSuccess;
    if (initialComm_) {
        destroyRes = ncclCommDestroy(initialComm_);
        initialComm_ = nullptr;
    }
    ASSERT_MPI_EQ(ncclSuccess, destroyRes);

    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));
}

// --- Grow then shrink (elastic cycle) ---

TEST_F(GrowMPITest, Grow_ThenShrink)
{
    if (!validateTestPrerequisites(3)) {
        GTEST_SKIP() << "Requires at least 3 MPI ranks";
    }

    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 1;

    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    ASSERT_MPI_EQ(ncclSuccess, growByOne(initialComm_, existing, &grownComm_));
    ASSERT_MPI_NE(grownComm_, nullptr);

    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));

    int lastRank = worldSize - 1;
    ncclComm_t shrunkComm = nullptr;
    ncclResult_t shrinkRes = ncclSuccess;
    bool shrinkVerified = true;

    if (wr != lastRank) {
        shrinkRes = ncclCommShrink(grownComm_, &lastRank, 1, &shrunkComm, nullptr, NCCL_SHRINK_DEFAULT);
        if (shrinkRes == ncclSuccess && shrunkComm != nullptr) {
            shrinkVerified = runAllReduceAndVerify(shrunkComm, worldSize - 1);
            (void)ncclCommDestroy(shrunkComm);
        } else {
            shrinkVerified = false;
        }
    }

    ASSERT_MPI_EQ(ncclSuccess, shrinkRes);
    ASSERT_MPI_TRUE(shrinkVerified);
    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));
}

#endif // MPI_TESTS_ENABLED
