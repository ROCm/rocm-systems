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

// --- Grow then shrink with NCCL_SHRINK_ABORT (torchcomms convention) ---

TEST_F(GrowMPITest, Grow_ThenShrinkAbort)
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
        shrinkRes = ncclCommShrink(grownComm_, &lastRank, 1, &shrunkComm, nullptr, NCCL_SHRINK_ABORT);
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

// --- Error recovery: abort grown comm, recreate, grow again ---

TEST_F(GrowMPITest, Grow_ErrorRecoveryRegrow)
{
    if (!validateTestPrerequisites(GrowTestConfig::kMinRanks)) {
        GTEST_SKIP() << "Requires at least " << GrowTestConfig::kMinRanks << " MPI ranks";
    }

    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 1;

    // Phase 1: Create initial comm and grow
    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    ASSERT_MPI_EQ(ncclSuccess, growByOne(initialComm_, existing, &grownComm_));
    ASSERT_MPI_NE(grownComm_, nullptr);

    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));

    // Phase 2: Simulate error — abort both comms (torchcomms error recovery path)
    ASSERT_MPI_EQ(ncclSuccess, ncclCommAbort(grownComm_));
    grownComm_ = nullptr;

    if (initialComm_) {
        ASSERT_EQ(ncclSuccess, ncclCommAbort(initialComm_));
        initialComm_ = nullptr;
    }

    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));

    // Phase 3: Recover — create fresh comm and grow again
    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    ASSERT_MPI_EQ(ncclSuccess, growByOne(initialComm_, existing, &grownComm_));
    ASSERT_MPI_NE(grownComm_, nullptr);

    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));
}


// --- Grow by multiple ranks at once (N-4 -> N), AllReduce verification ---
// Exercises bootstrapInit multi-root path: offset=parent->nRanks-1, isFirstFromRoot
// with a larger new-rank segment that spans multiple root zones.

TEST_F(GrowMPITest, Grow_MultiRankJump)
{
    constexpr int kJump = 4;
    if (MPIEnvironment::world_size < kJump + 2) {
        GTEST_SKIP() << "Grow_MultiRankJump requires at least " << kJump + 2 << " MPI ranks";
    }

    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - kJump;   // e.g. 28 with 32 ranks
    const int newTotal  = worldSize;            // e.g. 32

    // Build initial communicator with (worldSize - kJump) ranks
    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    // Coordinator (rank 0 of existing comm) produces uniqueId; broadcast via MPI
    ncclUniqueId growId{};
    if (wr == 0) {
        ASSERT_EQ(ncclSuccess, ncclCommGetUniqueId(initialComm_, &growId));
    }
    MPI_Bcast(&growId, sizeof(growId), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Grow: existing ranks pass nullptr uniqueId (non-boundary) or &growId (boundary/NCCL convention)
    // New ranks (wr in [existing, newTotal-1]) pass their target rank = wr
    if (wr < existing) {
        ASSERT_EQ(ncclSuccess,
            ncclCommGrow(initialComm_, newTotal, &growId, -1, &grownComm_, nullptr));
    } else {
        ASSERT_EQ(ncclSuccess,
            ncclCommGrow(nullptr, newTotal, &growId, wr, &grownComm_, nullptr));
    }

    ASSERT_MPI_NE(grownComm_, nullptr);

    // Verify rank count and AllReduce correctness
    int grownCount = -1;
    ASSERT_MPI_EQ(ncclSuccess, ncclCommCount(grownComm_, &grownCount));
    ASSERT_MPI_EQ(grownCount, newTotal);

    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, newTotal));
}

// --- GetUniqueId called twice on same parent; verify handles have distinct magic ---
// Exercises the childCount-based magic chain: each call to ncclCommGetUniqueId must
// produce a unique handle so concurrent grows don't collide.

TEST_F(GrowMPITest, GetUniqueId_DistinctHandles)
{
    if (MPIEnvironment::world_size < 4) {
        GTEST_SKIP() << "GetUniqueId_DistinctHandles requires at least 4 MPI ranks";
    }

    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 2;   // leave 2 slots for two sequential grows

    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    // --- First GetUniqueId call ---
    ncclUniqueId growId1{};
    if (wr == 0) {
        ASSERT_EQ(ncclSuccess, ncclCommGetUniqueId(initialComm_, &growId1));
    }
    MPI_Bcast(&growId1, sizeof(growId1), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Use growId1 to perform first grow (N-2 -> N-1)
    const int phase1 = existing + 1;
    if (wr < existing) {
        ASSERT_EQ(ncclSuccess,
            ncclCommGrow(initialComm_, phase1, &growId1, -1, &midComm_, nullptr));
    } else if (wr == existing) {
        ASSERT_EQ(ncclSuccess,
            ncclCommGrow(nullptr, phase1, &growId1, existing, &midComm_, nullptr));
    }
    ASSERT_MPI_TRUE(wr > existing || midComm_ != nullptr);

    // --- Second GetUniqueId call on midComm_ ---
    ncclUniqueId growId2{};
    if (wr == 0) {
        ASSERT_EQ(ncclSuccess, ncclCommGetUniqueId(midComm_, &growId2));
    }
    MPI_Bcast(&growId2, sizeof(growId2), MPI_BYTE, 0, MPI_COMM_WORLD);

    // The two handles must have different magic (childCount was bumped between calls)
    // Compare the first 8 bytes which hold the magic field
    uint64_t magic1 = 0, magic2 = 0;
    static_assert(sizeof(ncclUniqueId) >= 8, "ncclUniqueId too small");
    memcpy(&magic1, &growId1, sizeof(magic1));
    memcpy(&magic2, &growId2, sizeof(magic2));
    ASSERT_MPI_NE(magic1, magic2);

    // Use growId2 for second grow (N-1 -> N) to confirm handle is functional
    const int phase2 = worldSize;
    if (wr < phase1) {
        ASSERT_EQ(ncclSuccess,
            ncclCommGrow(midComm_, phase2, &growId2, -1, &grownComm_, nullptr));
    } else if (wr == existing + 1) {
        ASSERT_EQ(ncclSuccess,
            ncclCommGrow(nullptr, phase2, &growId2, wr, &grownComm_, nullptr));
    }
    ASSERT_MPI_NE(grownComm_, nullptr);
    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, phase2));
}


// --- Error guards: pre-GroupStart validation in ncclCommGrow_impl ---
// newcomm==NULL, nRanks<=0, nRanks<=existing all return ncclInvalidArgument
// immediately before ncclGroupStartInternal — safe to test on rank 0 only.

TEST_F(GrowMPITest, Grow_InvalidArguments)
{
    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 1;

    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    // Only rank 0 runs the error checks; all checks return before GroupStart
    // so no collective coordination is needed.
    if (wr == 0 && initialComm_ != nullptr) {
        ncclComm_t dummy = nullptr;

        // newcomm == NULL
        ASSERT_EQ(ncclInvalidArgument,
            ncclCommGrow(initialComm_, worldSize, nullptr, -1, nullptr, nullptr));

        // nRanks == 0 (non-positive)
        ASSERT_EQ(ncclInvalidArgument,
            ncclCommGrow(initialComm_, 0, nullptr, -1, &dummy, nullptr));

        // nRanks == existing (not strictly greater than current size)
        ASSERT_EQ(ncclInvalidArgument,
            ncclCommGrow(initialComm_, existing, nullptr, -1, &dummy, nullptr));
    }

    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));
}

// --- bcastGrowHandle: coordinator is the last rank of the parent comm ---
// Exercises the self-send guard: when parent->rank == parent->nRanks-1,
// bcastGrowHandle skips the send-to-self and sends only to rank 0.
// Rank 0 (other boundary) receives the handle via bcastGrowHandle(isRoot=false).

TEST_F(GrowMPITest, Grow_CoordinatorAtLastRank)
{
    if (!validateTestPrerequisites(3)) {
        GTEST_SKIP() << "Grow_CoordinatorAtLastRank requires at least 3 MPI ranks";
    }

    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 1;   // parent comm has (worldSize-1) ranks
    const int coordRank = existing - 1;    // last rank of parent comm = rank nRanks-1

    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    // Coordinator is the last rank of the parent comm (rank nRanks-1).
    // Inside bcastGrowHandle(isRoot=true):
    //   parent->rank == parent->nRanks-1 => skip send-to-self
    //   parent->rank != 0               => send to rank 0 via bootstrap
    ncclUniqueId growId{};
    if (wr == coordRank) {
        ASSERT_EQ(ncclSuccess, ncclCommGetUniqueId(initialComm_, &growId));
    }

    // Propagate uniqueId to the new rank (wr == existing) via point-to-point MPI.
    // Existing ranks other than coordinator and new rank do NOT get growId:
    //   - coordinator (wr==coordRank): already has it, passes it directly to ncclCommGrow
    //   - rank 0 (other boundary):     passes nullptr => receives via bcastGrowHandle bootstrap
    //   - other existing ranks:        pass nullptr  => non-boundary, no bootstrap interaction
    if (wr == coordRank) {
        MPI_Send(&growId, sizeof(growId), MPI_BYTE, existing, 0, MPI_COMM_WORLD);
    } else if (wr == existing) {
        MPI_Recv(&growId, sizeof(growId), MPI_BYTE, coordRank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Each rank calls ncclCommGrow with the appropriate arguments:
    if (wr == coordRank) {
        // Coordinator (last rank): has uniqueId, copies it directly -- no bootstrap recv
        ASSERT_EQ(ncclSuccess,
            ncclCommGrow(initialComm_, worldSize, &growId, -1, &grownComm_, nullptr));
    } else if (wr > 0 && wr < existing) {
        // Non-boundary existing ranks: nullptr uniqueId, no bootstrap interaction
        ASSERT_EQ(ncclSuccess,
            ncclCommGrow(initialComm_, worldSize, nullptr, -1, &grownComm_, nullptr));
    } else if (wr == 0) {
        // Rank 0 (other boundary): nullptr uniqueId => receives via bcastGrowHandle(isRoot=false)
        ASSERT_EQ(ncclSuccess,
            ncclCommGrow(initialComm_, worldSize, nullptr, -1, &grownComm_, nullptr));
    } else if (wr == existing) {
        // New rank: has uniqueId from coordinator via MPI_Recv
        ASSERT_EQ(ncclSuccess,
            ncclCommGrow(nullptr, worldSize, &growId, existing, &grownComm_, nullptr));
    }

    ASSERT_MPI_NE(grownComm_, nullptr);
    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));
}

// --- bootstrapGetUniqueId: NCCL_COMM_ID env rejects grow uniqueId creation ---
// When NCCL_COMM_ID is set, ncclCommGetUniqueId must return ncclInvalidUsage.
// This exercises the env-var guard in bootstrapGetUniqueId(handle, comm).

TEST_F(GrowMPITest, GetUniqueId_NcclCommIdEnvRejection)
{
    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 1;

    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    // Single-rank check: no bootstrap or group operations are triggered
    // when NCCL_COMM_ID is set (function returns immediately).
    if (wr == 0 && initialComm_ != nullptr) {
        setenv("NCCL_COMM_ID", "127.0.0.1:12345", 1);
        ncclUniqueId id{};
        ncclResult_t res = ncclCommGetUniqueId(initialComm_, &id);
        unsetenv("NCCL_COMM_ID");
        ASSERT_EQ(ncclInvalidUsage, res);
    }

    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));
}


TEST_F(GrowMPITest, ShrinkThenGrow_Debug)
{
    if (MPIEnvironment::world_size < 4) GTEST_SKIP();
    const int wr = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;

    ASSERT_MPI_EQ(ncclSuccess, buildComm(worldSize, &initialComm_));

    int excludeRank = worldSize - 1;
    ncclComm_t shrunkComm = nullptr;
    bool shrinkOk = true;
    if (wr != excludeRank) {
        ncclResult_t sr = ncclCommShrink(initialComm_, &excludeRank, 1, &shrunkComm, nullptr, NCCL_SHRINK_DEFAULT);
        shrinkOk = (sr == ncclSuccess) && (shrunkComm != nullptr);
    }
    ASSERT_MPI_TRUE(shrinkOk);
    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));

    ncclUniqueId growId{};
    if (wr == 0 && shrunkComm) {
        ASSERT_EQ(ncclSuccess, ncclCommGetUniqueId(shrunkComm, &growId));
    }
    MPI_Bcast(&growId, sizeof(growId), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (wr != excludeRank && shrunkComm) {
        ASSERT_EQ(ncclSuccess, ncclCommGrow(shrunkComm, worldSize, &growId, -1, &grownComm_, nullptr));
    } else if (wr == excludeRank) {
        ASSERT_EQ(ncclSuccess, ncclCommGrow(nullptr, worldSize, &growId, excludeRank, &grownComm_, nullptr));
    }
    ASSERT_MPI_NE(grownComm_, nullptr);
    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));
    if (shrunkComm) (void)ncclCommDestroy(shrunkComm);
}


// --- Full elastic cycle: grow N-1->N, shrink N->N-1, grow N-1->N again ---
TEST_F(GrowMPITest, GrowShrinkGrowCycle)
{
    if (MPIEnvironment::world_size < 4) GTEST_SKIP();
    const int wr        = MPIEnvironment::world_rank;
    const int worldSize = MPIEnvironment::world_size;
    const int existing  = worldSize - 1;

    // Phase 1: build initial comm (N-1 ranks) and grow to N
    ASSERT_MPI_EQ(ncclSuccess, buildComm(existing, &initialComm_));
    ASSERT_MPI_TRUE(wr >= existing || initialComm_ != nullptr);

    ASSERT_MPI_EQ(ncclSuccess, growByOne(initialComm_, existing, &grownComm_));
    ASSERT_MPI_NE(grownComm_, nullptr);
    ASSERT_MPI_TRUE(runAllReduceAndVerify(grownComm_, worldSize));

    // Phase 2: shrink back to N-1 (exclude last rank)
    int excludeRank = worldSize - 1;
    ncclComm_t shrunkComm = nullptr;
    bool shrinkOk = true;
    if (wr != excludeRank) {
        ncclResult_t sr = ncclCommShrink(grownComm_, &excludeRank, 1, &shrunkComm, nullptr, NCCL_SHRINK_DEFAULT);
        shrinkOk = (sr == ncclSuccess) && (shrunkComm != nullptr);
        if (shrinkOk) {
            shrinkOk = runAllReduceAndVerify(shrunkComm, worldSize - 1);
        }
    }
    ASSERT_MPI_TRUE(shrinkOk);
    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));

    // Phase 3: grow again from N-1 back to N
    ncclUniqueId growId{};
    if (wr == 0 && shrunkComm) {
        ASSERT_EQ(ncclSuccess, ncclCommGetUniqueId(shrunkComm, &growId));
    }
    MPI_Bcast(&growId, sizeof(growId), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t regrown = nullptr;
    if (wr != excludeRank && shrunkComm) {
        ASSERT_EQ(ncclSuccess, ncclCommGrow(shrunkComm, worldSize, &growId, -1, &regrown, nullptr));
    } else if (wr == excludeRank) {
        ASSERT_EQ(ncclSuccess, ncclCommGrow(nullptr, worldSize, &growId, excludeRank, &regrown, nullptr));
    }
    ASSERT_MPI_NE(regrown, nullptr);
    ASSERT_MPI_TRUE(runAllReduceAndVerify(regrown, worldSize));

    if (shrunkComm) (void)ncclCommDestroy(shrunkComm);
    if (regrown)    (void)ncclCommDestroy(regrown);
}

#endif // MPI_TESTS_ENABLED
