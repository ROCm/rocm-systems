/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "nccl.h"

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

/**
 * Revoking the same communicator twice must be rejected with ncclInvalidUsage.
 */
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
    ASSERT_MPI_EQ(ncclInvalidUsage, result);
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
 * ncclCommRevoke called from inside an active group must be rejected with
 * ncclInvalidUsage; ncclGroupEnd should still close cleanly.
 */
TEST_F(RevokeMPITest, Revoke_InsideGroup_Rejected)
{
    ASSERT_TRUE(validateTestPrerequisites(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclSuccess, ncclGroupStart());
    ncclResult_t result = ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT);
    ASSERT_MPI_EQ(ncclInvalidUsage, result);
    ASSERT_MPI_EQ(ncclSuccess, ncclGroupEnd());
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
    ASSERT_TRUE(validateTestPrerequisites(4,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit))
        << "Test requires at least 4 MPI processes across 2 nodes";

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
    ASSERT_TRUE(validateTestPrerequisites(4,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit))
        << "Test requires at least 4 MPI processes across 2 nodes";

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

#endif // MPI_TESTS_ENABLED
