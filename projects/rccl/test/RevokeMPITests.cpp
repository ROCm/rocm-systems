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

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

class RevokeMPITest : public MPITestBase {};

/**
 * Happy path: all ranks revoke their communicator and assert ncclSuccess.
 * Requires >=2 ranks and >=2 nodes (multi-node scenario).
 */
TEST_F(RevokeMPITest, Revoke_AllRanks_Succeeds)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes and 2 nodes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t comm = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(comm, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * After revoke, collective operations must return ncclInvalidUsage on all ranks.
 * Requires >=2 ranks and >=2 nodes.
 */
TEST_F(RevokeMPITest, Revoke_RejectsCollectives)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes and 2 nodes";

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
 * Revoke parent communicator, split into child comm, run AllReduce on child
 * across nodes and assert success.
 * Requires >=2 ranks and >=2 nodes.
 */
TEST_F(RevokeMPITest, Revoke_ThenSplit_ChildWorks)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes and 2 nodes";

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
 * Revoke parent communicator, then shrink (excluding the last rank on each
 * node to keep topology symmetric), and run AllReduce on the resulting child
 * across nodes. Validates that revoke forces shareResources=false on the
 * shrunk child so the child operates with its own fresh resources independent
 * of the revoked parent.
 * Requires >=2 ranks and >=2 nodes.
 */
TEST_F(RevokeMPITest, RevokeThenShrink_ChildWorks)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes and 2 nodes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent     = getActiveCommunicator();
    hipStream_t stream     = getActiveStream();
    int         rank       = MPIEnvironment::world_rank;
    int         world_size = MPIEnvironment::world_size;

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);

    int ranksPerNode    = world_size / 2;
    int excludeList[2]  = { ranksPerNode - 1, world_size - 1 };
    int excludeCount    = 2;
    bool isExcluded     = (rank == excludeList[0] || rank == excludeList[1]);
    ncclComm_t child    = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList, excludeCount, &child, nullptr, NCCL_SHRINK_DEFAULT));

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

        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, 1, ncclFloat, ncclSum, child, stream));

        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Shrink first (excluding the last rank on each node to keep topology
 * symmetric), then revoke the parent. Validates that the previously-created
 * child communicator continues to function for collectives even after the
 * parent is revoked (child is independent).
 * Requires >=2 ranks and >=2 nodes.
 */
TEST_F(RevokeMPITest, ShrinkThenRevoke_ChildUnaffected)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes and 2 nodes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent     = getActiveCommunicator();
    hipStream_t stream     = getActiveStream();
    int         rank       = MPIEnvironment::world_rank;
    int         world_size = MPIEnvironment::world_size;

    int ranksPerNode    = world_size / 2;
    int excludeList[2]  = { ranksPerNode - 1, world_size - 1 };
    int excludeCount    = 2;
    bool isExcluded     = (rank == excludeList[0] || rank == excludeList[1]);
    ncclComm_t child    = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList, excludeCount, &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        auto child_guard = makeCommAutoGuard(child);

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

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Shrink first (excluding the last rank on each node to keep topology
 * symmetric), then revoke the parent. Validates that subsequent collectives
 * on the *parent* are rejected with ncclInvalidUsage on every rank that still
 * holds the parent handle (including the excluded ranks), while leaving the
 * previously-created child untouched.
 * Requires >=2 ranks and >=2 nodes.
 */
TEST_F(RevokeMPITest, ShrinkThenRevoke_ParentRejectsCollectives)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes and 2 nodes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent     = getActiveCommunicator();
    hipStream_t stream     = getActiveStream();
    int         rank       = MPIEnvironment::world_rank;
    int         world_size = MPIEnvironment::world_size;

    int ranksPerNode    = world_size / 2;
    int excludeList[2]  = { ranksPerNode - 1, world_size - 1 };
    int excludeCount    = 2;
    bool isExcluded     = (rank == excludeList[0] || rank == excludeList[1]);
    ncclComm_t child    = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList, excludeCount, &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);

    void* buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, sizeof(float)));
    auto buf_guard = makeScopeGuard([&]() { if(buf) (void)hipFree(buf); });

    ncclResult_t result = ncclAllReduce(buf, buf, 1, ncclFloat, ncclSum, parent, stream);
    ASSERT_MPI_EQ(ncclInvalidUsage, result);

    if(!isExcluded)
    {
        auto child_guard = makeCommAutoGuard(child);
        (void) child_guard;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Explicit lifecycle: revoke the parent on every rank, then call
 * ncclCommDestroy and assert ncclSuccess on every rank. Validates the
 * documented "revoke leaves comm safe for destroy" contract across nodes
 * (the existing tests rely on TearDown to call destroy implicitly).
 * Requires >=2 ranks and >=2 nodes.
 */
TEST_F(RevokeMPITest, Revoke_ThenDestroy_CleanLifecycle)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes and 2 nodes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t parent = getActiveCommunicator();

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommDestroy(parent));
    test_comm_ = nullptr;

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Stress the in-flight scenario: enqueue a large AllReduce on every rank,
 * then call ncclCommRevoke before the collective kernel completes. Verifies
 * that revoke returns cleanly without hanging, the parent rejects subsequent
 * collectives, and a shrunk child built from the revoked parent runs an
 * AllReduce successfully.
 * Requires >=2 ranks and >=2 nodes.
 */
TEST_F(RevokeMPITest, Revoke_DuringInflightCollective_Succeeds)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes and 2 nodes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent     = getActiveCommunicator();
    hipStream_t stream     = getActiveStream();
    int         rank       = MPIEnvironment::world_rank;
    int         world_size = MPIEnvironment::world_size;

    constexpr size_t kCount = 64 * 1024 * 1024;
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

    ncclResult_t rejected = ncclAllReduce(send_buf, recv_buf, 1, ncclFloat, ncclSum, parent, stream);
    ASSERT_MPI_EQ(ncclInvalidUsage, rejected);

    MPI_Barrier(MPI_COMM_WORLD);

    int ranksPerNode   = world_size / 2;
    int excludeList[2] = { ranksPerNode - 1, world_size - 1 };
    int excludeCount   = 2;
    bool isExcluded    = (rank == excludeList[0] || rank == excludeList[1]);
    ncclComm_t child   = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList, excludeCount, &child, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(child, nullptr);
        auto child_guard = makeCommAutoGuard(child);

        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, 1, ncclFloat, ncclSum, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * Shrink first (excluding the last rank on each node to keep topology
 * symmetric), then revoke the parent, then split the shrunk child further.
 * Validates that a child created from a shrunk parent remains fully usable
 * for further sub-comm creation and collectives even after the parent has
 * been revoked.
 * Requires >=2 ranks and >=2 nodes.
 */
TEST_F(RevokeMPITest, ShrinkThenRevokeThenSplit_ChildWorks)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit))
        << "Test requires at least 2 MPI processes and 2 nodes";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  parent     = getActiveCommunicator();
    hipStream_t stream     = getActiveStream();
    int         rank       = MPIEnvironment::world_rank;
    int         world_size = MPIEnvironment::world_size;

    int ranksPerNode      = world_size / 2;
    int excludeList[2]    = { ranksPerNode - 1, world_size - 1 };
    int excludeCount      = 2;
    bool isExcluded       = (rank == excludeList[0] || rank == excludeList[1]);
    ncclComm_t shrinkChild = NCCL_COMM_NULL;

    if(!isExcluded)
    {
        ASSERT_EQ(ncclSuccess,
                  ncclCommShrink(parent, excludeList, excludeCount, &shrinkChild, nullptr, NCCL_SHRINK_DEFAULT));
        ASSERT_NE(shrinkChild, nullptr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_MPI_EQ(ncclSuccess, ncclCommRevoke(parent, NCCL_REVOKE_DEFAULT));

    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
    {
        auto shrink_guard = makeCommAutoGuard(shrinkChild);

        ncclComm_t splitChild = nullptr;
        ASSERT_EQ(ncclSuccess, ncclGroupStart());
        ASSERT_EQ(ncclSuccess, ncclCommSplit(shrinkChild, 0, rank, &splitChild, nullptr));
        ASSERT_EQ(ncclSuccess, ncclGroupEnd());

        ASSERT_NE(splitChild, nullptr);
        auto split_guard = makeCommAutoGuard(splitChild);

        void* send_buf = nullptr;
        void* recv_buf = nullptr;
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, sizeof(float)));
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, sizeof(float)));
        auto send_guard = makeScopeGuard([&]() { if(send_buf) (void)hipFree(send_buf); });
        auto recv_guard = makeScopeGuard([&]() { if(recv_buf) (void)hipFree(recv_buf); });

        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(send_buf, 1));
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recv_buf, 1));

        ASSERT_EQ(ncclSuccess,
                  ncclAllReduce(send_buf, recv_buf, 1, ncclFloat, ncclSum, splitChild, stream));

        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

#endif // MPI_TESTS_ENABLED
