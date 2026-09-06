/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "nccl.h"

#include <memory>
#include <sched.h>
#include <string>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

class RevokeMPITest : public MPITestBase {};

static void computeSymmetricExclude(int worldRank, int worldSize,
                                    std::vector<int>& excludeList, bool& isExcluded)
{
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int localSize, localRank;
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int numNodes = worldSize / localSize;
    isExcluded = (localRank == localSize - 1);
    excludeList.clear();
    for (int n = 0; n < numNodes; n++)
        excludeList.push_back((n + 1) * localSize - 1);
}

/**
 * After revoke, collective operations must return ncclInvalidUsage on all ranks.
 */
TEST_F(RevokeMPITest, Revoke_RejectsCollectives)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);

    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, sizeof(float)));
    auto buf_guard = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });

    ncclResult_t result = ncclAllReduce(buf, buf, 1, ncclFloat, ncclSum, comm, stream);
    ASSERT_MPI_EQ(ncclInvalidUsage, result);
}

/**
 * Revoke parent communicator, split into child comm, run AllReduce on the
 * child and assert success. Validates that revoke leaves the parent valid as
 * a source for ncclCommSplit.
 */
TEST_F(RevokeMPITest, Revoke_ThenSplit_ChildWorks)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);

    ncclComm_t child = nullptr;
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    ASSERT_MPI_EQ(ncclSuccess, ncclCommSplit(parent, 0, rank, &child, nullptr));
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());

    ASSERT_NE(child, nullptr);
    auto child_guard = makeCommAutoGuard(child);

    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, sizeof(float)));
    auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, 1));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, 1));

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, 1, ncclFloat, ncclSum, child, stream));

    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
}

/**
 * Revoke parent communicator, then shrink (excluding the last rank), and run
 * AllReduce on the resulting child. Validates that revoke leaves the parent
 * valid as a source for ncclCommShrink and that the child operates with its
 * own resources independent of the revoked parent.
 */
TEST_F(RevokeMPITest, RevokeThenShrink_ChildWorks)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool             isExcluded;
    computeSymmetricExclude(rank, MPIEnvironment::world_size, excludeList, isExcluded);
    ncclComm_t child = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_DEFAULT));

        ASSERT_NE(child, nullptr);

        void* send_buf = nullptr;
        void* recv_buf = nullptr;
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, sizeof(float)));
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, sizeof(float)));
        auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
        auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, 1));
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, 1));

        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, 1, ncclFloat, ncclSum, child, stream));

        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    // Barrier before child destruction ensures all participating ranks
    // complete their operations before any rank starts destroying
    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(child));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Explicit lifecycle: revoke the parent on every rank, then call
 * ncclCommDestroy and assert ncclSuccess on every rank. Validates the
 * documented "revoke leaves comm safe for destroy" contract (other tests
 * rely on TearDown to call destroy implicitly).
 */
TEST_F(RevokeMPITest, Revoke_ThenDestroy_CleanLifecycle)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t parent = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommDestroy(parent));
    test_comm_ = nullptr;

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * In-flight collective recovery cycle: enqueue a large AllReduce on the
 * parent, revoke without waiting for completion, drain the stream, then
 * shrink and run another AllReduce on the child. Mirrors the Meta
 * fault-tolerance recovery flow: collective -> revoke -> shrink -> collective.
 */
TEST_F(RevokeMPITest, Collective_Revoke_Shrink_Collective)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;

    constexpr size_t kCount = 1024 * 1024;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
    auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, parent, stream));

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool             isExcluded;
    computeSymmetricExclude(rank, MPIEnvironment::world_size, excludeList, isExcluded);
    ncclComm_t child = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    // Barrier before child destruction ensures all participating ranks
    // complete their operations before any rank starts destroying
    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(child));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * P2P recovery cycle with in-flight ops: launch ncclSend/ncclRecv between
 * pairs, revoke the parent before draining, then shrink and verify a fresh
 * P2P exchange on the child succeeds. Validates that in-flight P2P is
 * halted gracefully and the child's resources are clean.
 */
TEST_F(RevokeMPITest, P2P_Revoke_Shrink_P2P)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent     = getActiveCommunicator();
    hipStream_t stream     = getActiveStream();
    int         rank       = MPIEnvironment::world_rank;
    int         world_size = MPIEnvironment::world_size;

    constexpr size_t kCount = 64 * 1024;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
    auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    if(rank % 2 == 0 && rank + 1 < world_size)
    {
        ASSERT_MPI_EQ(ncclSuccess,
                      ncclSend(send_buf, kCount, ncclFloat, rank + 1, parent, stream));
    }
    else if(rank % 2 == 1)
    {
        ASSERT_MPI_EQ(ncclSuccess,
                      ncclRecv(recv_buf, kCount, ncclFloat, rank - 1, parent, stream));
    }
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool             isExcluded;
    computeSymmetricExclude(rank, world_size, excludeList, isExcluded);
    ncclComm_t child = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        int child_rank = -1;
        int child_size = 0;
        ASSERT_EQ(ncclSuccess, ncclCommUserRank(child, &child_rank));
        ASSERT_EQ(ncclSuccess, ncclCommCount(child, &child_size));

        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        if(child_rank % 2 == 0 && child_rank + 1 < child_size)
        {
            ASSERT_EQ(ncclSuccess,
                      ncclSend(send_buf, kCount, ncclFloat, child_rank + 1, child, stream));
        }
        else if(child_rank % 2 == 1)
        {
            ASSERT_EQ(ncclSuccess,
                      ncclRecv(recv_buf, kCount, ncclFloat, child_rank - 1, child, stream));
        }
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());

        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    // Barrier before child destruction ensures all participating ranks
    // complete their operations before any rank starts destroying
    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(child));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// Revoking the same communicator twice must be rejected with ncclInvalidArgument.
TEST_F(RevokeMPITest, Revoke_DoubleRevoke_Rejected)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);

    ncclResult_t result = ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
    ASSERT_MPI_EQ(ncclInvalidArgument, result);
}

/**
 * Calling ncclCommRevoke with a null handle must be rejected with ncclInvalidArgument.
 */
TEST_F(RevokeMPITest, Revoke_NullComm_Rejected)
{
    ASSERT_MPI_EQ(ncclInvalidArgument, ncclCommRevoke(nullptr, NCCL_REVOKE_DEFAULT));
}

/**
 * Calling ncclCommRevoke with unsupported revokeFlags must be rejected
 * with ncclInvalidArgument.
 */
TEST_F(RevokeMPITest, Revoke_BadFlags_Rejected)
{
    ASSERT_MPI_EQ(ncclInvalidArgument, ncclCommRevoke(nullptr, /*revokeFlags=*/0x1));
}

/**
 * ncclCommFinalize on a revoked communicator must be rejected with
 * ncclInvalidUsage. After revoke, the user should call ncclCommDestroy
 * instead — finalize would double-drain streams already handled by
 * commRevokeAsync.
 */
TEST_F(RevokeMPITest, Revoke_ThenFinalize_Rejected)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclInvalidUsage, ncclCommFinalize(comm));
}

/**
 * Incomplete collective variant: rank np-1 does NOT call AllReduce, so the
 * collective is guaranteed to be stuck in-flight when revoke fires. This
 * verifies that revoke can recover from a truly incomplete operation, not
 * just one that might have raced to completion.
 *
 * Relies on ncclCommRevoke setting abortFlag=1 synchronously before launching
 * commRevokeAsync, so that the device-side
 * kernel waiting for the missing peer exits and the host stream sync inside
 * commRevokeAsync can proceed.
 *
 */
TEST_F(RevokeMPITest, IncompleteCollective_Revoke_Shrink_Collective)
{
    ASSERT_TRUE(validateTestPrerequisites(4,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 4 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;
    int         world_size = MPIEnvironment::world_size;

    constexpr size_t kCount = 1024 * 1024;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
    auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

    if (rank != world_size - 1) {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, parent, stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    if (rank != world_size - 1) {
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool             isExcluded;
    computeSymmetricExclude(rank, world_size, excludeList, isExcluded);
    ncclComm_t child = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    // Barrier before child destruction ensures all participating ranks
    // complete their operations before any rank starts destroying
    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(child));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * DDA fabric LL incomplete AllReduce + revoke. IncompleteCollective_Revoke_Shrink_Collective
 * uses 1M floats and can take DDA two-shot on gfx1250 by accident, but it does not
 * assert the COLL path. These two cases require the one-shot / two-shot log needle
 * so a generic-kernel fallback cannot mask a DDA abortFlag wait regression.
 */
namespace
{
constexpr size_t kDdaOneShotCount     = 65536; // 256 KiB f32; below the 1 MiB one-shot threshold
constexpr size_t kDdaTwoShotBaseCount = 262176;
constexpr char   kDdaOneShotNeedle[]  = "taking DDA fabric LL one-shot path";
constexpr char   kDdaTwoShotNeedle[]  = "taking DDA fabric LL two-shot path";

bool ddaIsGfx1250Device()
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
        return false;
    return std::string(props.gcnArchName).find("gfx1250") != std::string::npos;
}

bool ddaAllRanksTrue(bool local)
{
    int vote    = local ? 1 : 0;
    int minVote = 1;
    MPI_Allreduce(&vote, &minVote, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return minVote != 0;
}

bool ddaLLTwoShotShapeOk(size_t count, int nRanks)
{
    const size_t bytes = count * sizeof(float);
    if(bytes % static_cast<size_t>(nRanks) != 0)
        return false;
    return (bytes / static_cast<size_t>(nRanks)) % 16 == 0;
}

size_t ddaTwoShotCountForRanks(int nRanks)
{
    size_t count = kDdaTwoShotBaseCount;
    if(ddaLLTwoShotShapeOk(count, nRanks))
        return count;
    count = (count + 3) & ~size_t(3);
    for(int i = 0; i < 1024; ++i, count += 4)
    {
        if(ddaLLTwoShotShapeOk(count, nRanks))
            return count;
    }
    return 0;
}

std::string ddaSnapshotLogs(const MPIHelpers::TestLogAssertionContext& logCtx)
{
    return logCtx.readNcclDebugLog() + logCtx.readPerRankStderrLog();
}

bool ddaLogsContainNeedleSince(const MPIHelpers::TestLogAssertionContext& logCtx,
                               const std::string& before, const char* needle)
{
    const std::string now = ddaSnapshotLogs(logCtx);
    const std::string delta =
        now.size() >= before.size() ? now.substr(before.size()) : now;
    return delta.find(needle) != std::string::npos;
}
} // namespace

class RevokeDdaMPITest : public MPITestBase
{
protected:
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugSubsysGuard_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;

    void SetUp() override
    {
        MPITestBase::SetUp();
        if(!ddaAllRanksTrue(ddaIsGfx1250Device()))
            GTEST_SKIP() << "DDA fabric LL AllReduce requires gfx1250 on every rank";
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

    void runIncompleteDdaRevokeShrink(size_t count, const char* needle);
};

void RevokeDdaMPITest::runIncompleteDdaRevokeShrink(size_t count, const char* needle)
{
    ASSERT_TRUE(validateTestPrerequisites(4,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 4 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent     = getActiveCommunicator();
    hipStream_t stream     = getActiveStream();
    const int   rank       = MPIEnvironment::world_rank;
    const int   worldSize  = MPIEnvironment::world_size;
    const int   skipRank   = worldSize - 1;
    const size_t bytes     = count * sizeof(float);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&sendBuf, bytes));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recvBuf, bytes));
    auto sendGuard = makeScopeGuard([&]() { if(sendBuf) (void)hipFree(sendBuf); });
    auto recvGuard = makeScopeGuard([&]() { if(recvBuf) (void)hipFree(recvBuf); });

    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(sendBuf, count));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recvBuf, count));

    const std::string beforeParent = ddaSnapshotLogs(*logCtx_);
    if(rank != skipRank)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(sendBuf, recvBuf, count, ncclFloat32, ncclSum, parent, stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_TRUE(rank == skipRank || ddaLogsContainNeedleSince(*logCtx_, beforeParent, needle))
        << "Rank " << rank << ": incomplete AllReduce did not take expected DDA LL path";

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    if(rank != skipRank)
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool             isExcluded = false;
    computeSymmetricExclude(rank, worldSize, excludeList, isExcluded);
    ncclComm_t child = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(), &child, nullptr,
                                 NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(sendBuf, recvBuf, count, ncclFloat32, ncclSum, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(child));

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaLlTwoShot)
{
    const size_t count = ddaTwoShotCountForRanks(MPIEnvironment::world_size);
    if(count == 0)
        GTEST_SKIP() << "Could not find a two-shot-aligned element count";

    runIncompleteDdaRevokeShrink(count, kDdaTwoShotNeedle);
}

TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaLlOneShot)
{
    runIncompleteDdaRevokeShrink(kDdaOneShotCount, kDdaOneShotNeedle);
}

static void computeAsymmetricExclude(int worldRank, int worldSize,
                                     std::vector<int>& excludeList, bool& isExcluded)
{
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &nodeComm);
    int localSize, localRank;
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_rank(nodeComm, &localRank);

    int nodeId = 0;
    if (localSize > 0) nodeId = worldRank / localSize;

    MPI_Comm_free(&nodeComm);

    int nNodes = (worldSize + localSize - 1) / localSize;
    int nodesToExclude = std::max(1, nNodes / 2);

    excludeList.clear();
    isExcluded = false;
    for (int n = 0; n < nodesToExclude; n++) {
        int excludedRank = n * localSize + (localSize - 1);
        if (excludedRank < worldSize) {
            excludeList.push_back(excludedRank);
            if (worldRank == excludedRank) isExcluded = true;
        }
    }
}

/**
 * Asymmetric shrink: exclude one rank from each of the first half of nodes,
 * creating unequal per-node GPU counts (e.g. 7+7+8+8 on 4 nodes).
 * Exercises the asymmetric topology path where tree connections are
 * skipped and only ring algorithm is used.
 */
TEST_F(RevokeMPITest, Collective_Revoke_AsymmetricShrink_Collective)
{
    if(!validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired, 2, kNoNodeLimit))
    {
        GTEST_SKIP() << "Test requires at least 4 MPI processes across 2 nodes";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;

    constexpr size_t kCount = 1024 * 1024;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
    auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, parent, stream));

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool             isExcluded;
    computeAsymmetricExclude(rank, MPIEnvironment::world_size, excludeList, isExcluded);
    ncclComm_t child = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    // Barrier before child destruction ensures all participating ranks
    // complete their operations before any rank starts destroying
    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(child));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * NCCL_SHRINK_ABORT: enqueue a large collective on parent, then shrink with
 * the ABORT flag (terminates in-flight ops before creating child).
 * Verifies child collective works and rank renumbering is correct.
 * Covers Meta TorchComms requirement 3: ncclCommShrink with NCCL_SHRINK_ABORT.
 */
TEST_F(RevokeMPITest, ShrinkAbort_InFlight_ChildWorks_RankRenumbering)
{
    ASSERT_TRUE(validateTestPrerequisites(4,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 4 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;

    constexpr size_t kCount = 1024 * 1024;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
    auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, parent, stream));

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool             isExcluded;
    computeSymmetricExclude(rank, MPIEnvironment::world_size, excludeList, isExcluded);
    ncclComm_t child = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_ABORT));
        ASSERT_NE(child, nullptr);

        int childRank = -1, childSize = 0;
        ASSERT_EQ(ncclSuccess, ncclCommUserRank(child, &childRank));
        ASSERT_EQ(ncclSuccess, ncclCommCount(child, &childSize));

        int expectedSize = MPIEnvironment::world_size - static_cast<int>(excludeList.size());
        ASSERT_EQ(childSize, expectedSize);
        ASSERT_GE(childRank, 0);
        ASSERT_LT(childRank, childSize);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    // Barrier before child destruction ensures all participating ranks
    // complete their operations before any rank starts destroying
    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(child));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * In-flight collective, revoke, then explicit destroy on all ranks.
 * Verifies ncclCommDestroy returns ncclSuccess after revoke — no SIGABRT.
 * Covers Meta TorchComms requirement 1: revoke ≠ abort.
 */
TEST_F(RevokeMPITest, InFlightCollective_Revoke_Destroy_Clean)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    constexpr size_t kCount = 1024 * 1024;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
    auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, parent, stream));

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommDestroy(parent));
    test_comm_ = nullptr;

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Repeated revoke+shrink cycle: revoke parent -> shrink -> child collective ->
 * revoke child -> shrink child -> grandchild collective -> destroy all.
 * Verifies no resource leaks or crashes across multiple generations.
 * Covers Meta TorchComms requirement 4: no leaked resources.
 */
TEST_F(RevokeMPITest, RepeatedRevokeShrinkCycles_ResourceCleanup)
{
    if(!validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired, 2, kNoNodeLimit))
    {
        GTEST_SKIP() << "Test requires at least 4 MPI processes across 2 nodes";
    }

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent = getActiveCommunicator();
    hipStream_t stream = getActiveStream();
    int         rank   = MPIEnvironment::world_rank;

    constexpr size_t kCount = 64 * 1024;
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kCount * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kCount * sizeof(float)));
    auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
    auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, kCount));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, kCount));

    ASSERT_MPI_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, parent, stream));

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));
    HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> excludeList;
    bool             isExcluded;
    computeSymmetricExclude(rank, MPIEnvironment::world_size, excludeList, isExcluded);
    ncclComm_t child = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList.data(), excludeList.size(),
                                 &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, kCount, ncclFloat, ncclSum, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

        ASSERT_EQ(ncclSuccess, ncclCommRevoke(child, NCCL_REVOKE_DEFAULT));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(child));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// Revoke fixture using a non-blocking (config.blocking = 0) parent comm.
// Exercises the worker-thread async-job path that RevokeMPITest skips.
class RevokeNonBlockingMPITest : public MPITestBase
{
protected:
    // ncclCommDestroy on a non-blocking comm can return ncclInProgress;
    // use Abort to guarantee synchronous teardown between chained tests.
    void TearDown() override
    {
        if(test_comm_)
        {
            (void)ncclCommAbort(test_comm_);
            test_comm_ = nullptr;
        }
        MPITestBase::TearDown();
    }

    // Poll ncclCommGetAsyncError until the comm leaves ncclInProgress,
    // yielding the CPU between checks. Returns the terminal state.
    static ncclResult_t waitForAsyncResult(ncclComm_t comm)
    {
        ncclResult_t state = ncclInProgress;
        while(state == ncclInProgress)
        {
            ncclResult_t r = ncclCommGetAsyncError(comm, &state);
            if(r != ncclSuccess) return r;
            if(state == ncclInProgress) sched_yield();
        }
        return state;
    }

    ncclResult_t createTestCommunicator() override
    {
        int world_rank = MPIEnvironment::world_rank;
        int world_size = MPIEnvironment::world_size;

        if(world_rank == 0)
        {
            RCCL_TEST_CHECK(ncclGetUniqueId(&nccl_id_));
        }
        MPI_Bcast(&nccl_id_, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        config.blocking     = 0;

        RCCL_TEST_CHECK(ncclGroupStart());
        auto group_guard = makeScopeGuard([]() { (void)ncclGroupEnd(); });

        ncclResult_t res = ncclCommInitRankConfig(
            &test_comm_, world_size, nccl_id_, world_rank, &config);
        if(res != ncclSuccess && res != ncclInProgress) return res;

        auto comm_guard = makeScopeGuard(
            [this]()
            {
                if(test_comm_)
                {
                    (void)ncclCommAbort(test_comm_);
                    test_comm_ = nullptr;
                }
            });

        res = ncclGroupEnd();
        group_guard.dismiss();
        if(res != ncclSuccess && res != ncclInProgress) return res;

        // Drain ncclInProgress from the non-blocking init.
        RCCL_TEST_CHECK(waitForAsyncResult(test_comm_));

        HIP_TEST_CHECK(hipStreamCreate(&test_stream_));
        auto stream_guard = makeScopeGuard(
            [this]()
            {
                if(test_stream_)
                {
                    (void)hipStreamDestroy(test_stream_);
                    test_stream_ = nullptr;
                }
            });

        MPI_Barrier(MPI_COMM_WORLD);

        comm_guard.dismiss();
        stream_guard.dismiss();
        return ncclSuccess;
    }
};

// Non-blocking revoke must return ncclSuccess/ncclInProgress, then drain
// to ncclSuccess via ncclCommGetAsyncError; later collectives must reject.
TEST_F(RevokeNonBlockingMPITest, Revoke_NonBlocking_ReturnsInProgress)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    ncclResult_t res = ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
    ASSERT_MPI_TRUE(res == ncclSuccess || res == ncclInProgress);

    ASSERT_MPI_EQ(ncclSuccess, waitForAsyncResult(comm));

    MPI_Barrier(MPI_COMM_WORLD);

    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, sizeof(float)));
    auto buf_guard = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });

    ncclResult_t collRes =
        ncclAllReduce(buf, buf, 1, ncclFloat, ncclSum, comm, stream);
    ASSERT_MPI_EQ(ncclInvalidUsage, collRes);
}

// Non-blocking revoke -> shrink on the same parent, per the non-blocking
// contract: drain each async op (waitForAsyncResult) before the next.
// Regression for AICOMRCCL-2232.
TEST_F(RevokeNonBlockingMPITest, Revoke_NonBlocking_ThenShrink_NoRace)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    constexpr int kIterations = 10;
    int           rank        = MPIEnvironment::world_rank;
    int           world_size  = MPIEnvironment::world_size;

    for(int iter = 0; iter < kIterations; ++iter)
    {
        SCOPED_TRACE("iteration " + std::to_string(iter));

        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

        ncclComm_t parent = getActiveCommunicator();

        // Drain the async revoke before reusing the comm; otherwise shrink
        // races the parent teardown and child bootstrap fails (conn-refused).
        ncclResult_t res = ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT);
        ASSERT_MPI_TRUE(res == ncclSuccess || res == ncclInProgress);
        ASSERT_MPI_EQ(ncclSuccess, waitForAsyncResult(parent));

        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> excludeList;
        bool             isExcluded = false;
        computeSymmetricExclude(rank, world_size, excludeList, isExcluded);

        // Excluded ranks do not shrink; they contribute a trivial pass so the
        // collective assert below stays balanced across all ranks.
        ncclComm_t child    = NCCL_COMM_NULL;
        bool       shrinkOk = true;
        if(!isExcluded)
        {
            res = ncclCommShrink(parent,
                                 excludeList.data(),
                                 excludeList.size(),
                                 &child,
                                 nullptr,
                                 NCCL_SHRINK_DEFAULT);
            // *newcomm stays NCCL_COMM_NULL until the child job completes, so
            // drain the parent first, then the now-populated child handle. Each
            // stage records a non-fatal ADD_FAILURE (which does not return), so
            // the collective assert below still reports the failing stage and
            // iteration without leaving excluded ranks stuck in the barriers.
            if(res != ncclSuccess && res != ncclInProgress)
            {
                shrinkOk = false;
                ADD_FAILURE() << "shrink returned " << res;
            }
            else if(waitForAsyncResult(parent) != ncclSuccess)
            {
                shrinkOk = false;
                ADD_FAILURE() << "parent drain after shrink did not reach ncclSuccess";
            }
            else if(child == nullptr)
            {
                shrinkOk = false;
                ADD_FAILURE() << "child handle still NULL after parent drain";
            }
            else if(waitForAsyncResult(child) != ncclSuccess)
            {
                shrinkOk = false;
                ADD_FAILURE() << "child drain did not reach ncclSuccess";
            }
        }
        // Collective: every rank participates, so a failure on any included rank
        // fails the run instead of hanging excluded ranks in the barriers below.
        ASSERT_MPI_TRUE(shrinkOk);

        MPI_Barrier(MPI_COMM_WORLD);

        if(!isExcluded && child)
        {
            ASSERT_EQ(ncclSuccess, ncclCommDestroy(child));
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Tear down parent so the next iteration starts from a clean state.
        (void)cleanupTestCommunicator();
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// ncclCommAbort issued mid-flight on a non-blocking revoke must wind down
// the async-job worker via its abortFlag and return without hanging.
TEST_F(RevokeNonBlockingMPITest, Revoke_NonBlocking_AbortedMidFlight)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();

    ncclResult_t res = ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
    ASSERT_MPI_TRUE(res == ncclSuccess || res == ncclInProgress);

    // Abort immediately; do not poll the revoke result first.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommAbort(comm));

    // Comm is now destroyed by Abort. Prevent the framework's TearDown
    // from calling ncclCommDestroy on a dangling handle.
    test_comm_ = nullptr;

    MPI_Barrier(MPI_COMM_WORLD);
}

#endif // MPI_TESTS_ENABLED
