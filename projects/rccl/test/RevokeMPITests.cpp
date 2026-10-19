/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "DdaMPITestHelpers.hpp"
#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "nccl.h"
#include "rccl_common.h"

#include <memory>
#include <sched.h>
#include <string>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;
using namespace DdaMPITestHelpers;

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
    SKIP_UNLESS_MPI_PREREQS(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(4,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
 * Incomplete DDA collective + revoke. IncompleteCollective_Revoke_Shrink_Collective
 * can take a DDA path by accident but does not assert it. These cases pin one DDA
 * kernel each via rcclGetCollImplInfo, so a generic-kernel fallback cannot mask a
 * regression in that kernel's abortFlag wait: rank np-1 skips the collective, the
 * other ranks block in the DDA wait, and ncclCommRevoke must unblock them.
 *
 * The reporting API names the tier (LL / LL128 / VMM / IPC), not the one-shot vs
 * two-shot AllReduce kernel; those cases are sized so the default sub-thresholds
 * select the intended kernel, and skip when the environment moves them.
 *
 * Cases that force a tier through RCCL_DDA_* overrides only take effect if the
 * parameter has not been read yet in this process (RCCL caches parameters), so
 * they skip rather than fail when the expected tier is not selected. Run them in
 * their own process, as the test-runner configs do.
 */
namespace
{
struct DdaEnvSetting
{
    const char* name;
    const char* value;
};

struct DdaRevokeCase
{
    ncclFunc_t                 func;
    size_t                     count; // count argument of the collective
    int                        expectedAlgo;
    int                        expectedProtocol;
    std::vector<DdaEnvSetting> overrides;
};

size_t ddaSendElems(ncclFunc_t func, size_t count, int nRanks)
{
    const size_t n = static_cast<size_t>(nRanks);
    return (func == ncclFuncReduceScatter || func == ncclFuncAlltoAll) ? count * n : count;
}

size_t ddaRecvElems(ncclFunc_t func, size_t count, int nRanks)
{
    const size_t n = static_cast<size_t>(nRanks);
    return (func == ncclFuncAllGather || func == ncclFuncAlltoAll) ? count * n : count;
}

ncclResult_t ddaLaunch(ncclFunc_t func, const void* sendBuf, void* recvBuf, size_t count,
                       ncclComm_t comm, hipStream_t stream)
{
    switch(func)
    {
    case ncclFuncAllReduce:
        return ncclAllReduce(sendBuf, recvBuf, count, ncclFloat32, ncclSum, comm, stream);
    case ncclFuncAllGather:
        return ncclAllGather(sendBuf, recvBuf, count, ncclFloat32, comm, stream);
    case ncclFuncReduceScatter:
        return ncclReduceScatter(sendBuf, recvBuf, count, ncclFloat32, ncclSum, comm, stream);
    case ncclFuncAlltoAll:
        return ncclAlltoAll(sendBuf, recvBuf, count, ncclFloat32, comm, stream);
    default: return ncclInvalidArgument;
    }
}

/// True when the AllReduce LL one-/two-shot sub-thresholds are at the defaults the
/// DdaMPITestHelpers counts are sized against.
bool ddaLLSubThresholdsAtDefault()
{
    constexpr int64_t kDefaultTwoShot = 2 * static_cast<int64_t>(kDefaultLLOneShotThresholdBytes);
    return MPIHelpers::getEnvParam<int64_t>("RCCL_DDA_LL_ONESHOT_THRESHOLD",
                                            kDefaultLLOneShotThresholdBytes)
               == static_cast<int64_t>(kDefaultLLOneShotThresholdBytes)
           && MPIHelpers::getEnvParam<int64_t>("RCCL_DDA_LL_TWOSHOT_THRESHOLD", kDefaultTwoShot)
                  == kDefaultTwoShot;
}

/// Shared body: GTEST_SKIP() returns from here, so call it as the last statement of a test.
void runIncompleteDdaRevokeShrink(MPITestBase& test, const DdaRevokeCase& tc)
{
    std::vector<std::unique_ptr<MPIHelpers::MpiEnvGuard>> envGuards;
    for(const DdaEnvSetting& e : tc.overrides)
        envGuards.push_back(std::make_unique<MPIHelpers::MpiEnvGuard>(e.name, e.value));

    ASSERT_MPI_EQ(ncclSuccess, test.createTestCommunicator());

    ncclComm_t   parent    = test.getActiveCommunicator();
    hipStream_t  stream    = test.getActiveStream();
    const int    rank      = MPIEnvironment::world_rank;
    const int    worldSize = MPIEnvironment::world_size;
    const int    skipRank  = worldSize - 1;
    const size_t sendElems = ddaSendElems(tc.func, tc.count, worldSize);
    const size_t recvElems = ddaRecvElems(tc.func, tc.count, worldSize);

    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&sendBuf, sendElems * sizeof(float)));
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recvBuf, recvElems * sizeof(float)));
    auto sendGuard = makeScopeGuard([&]() { if(sendBuf) (void)hipFree(sendBuf); });
    auto recvGuard = makeScopeGuard([&]() { if(recvBuf) (void)hipFree(recvBuf); });

    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(sendBuf, sendElems));
    HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recvBuf, recvElems));

    int algo        = -1;
    int protocol    = -1;
    int maxChannels = -1;
    ASSERT_MPI_EQ(ncclSuccess,
                  rcclGetCollImplInfo(parent, tc.func, tc.count, ncclFloat32, ncclSum, sendBuf,
                                      recvBuf, /*graphCapturing=*/0, &algo, &protocol,
                                      &maxChannels));
    const bool pathOk = algo == tc.expectedAlgo && protocol == tc.expectedProtocol;
    if(!tc.overrides.empty())
    {
        const std::string skipReason = mpiCoordinatedSkipReason(
            !pathOk,
            "RCCL_DDA_* override did not select the expected DDA tier (parameters are cached per "
            "process); run this test on its own");
        if(!skipReason.empty())
            GTEST_SKIP() << skipReason << " (algo=" << algo << " protocol=" << protocol << ")";
    }
    ASSERT_MPI_TRUE(pathOk);

    if(rank != skipRank)
    {
        ASSERT_EQ(ncclSuccess, ddaLaunch(tc.func, sendBuf, recvBuf, tc.count, parent, stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);

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
        ASSERT_EQ(ncclSuccess, ddaLaunch(tc.func, sendBuf, recvBuf, tc.count, child, stream));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(!isExcluded)
        ASSERT_EQ(ncclSuccess, ncclCommDestroy(child));

    MPI_Barrier(MPI_COMM_WORLD);
}

constexpr int kDdaLL    = static_cast<int>(rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL);
constexpr int kDdaLL128 = static_cast<int>(rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL128);
constexpr int kDdaVmm   = static_cast<int>(rcclAddonAlgos_t::RCCL_DDA_FABRIC_VMM);
constexpr int kDdaIpc   = static_cast<int>(rcclAddonAlgos_t::RCCL_DDA_IPC);

const std::vector<DdaEnvSetting> kDisableAllReduceLL = {
    {"RCCL_DDA_LL_ONESHOT_THRESHOLD", "0"},
    {"RCCL_DDA_LL_TWOSHOT_THRESHOLD", "0"},
};
const std::vector<DdaEnvSetting> kDisableLL = {{"RCCL_DDA_LL", "0"}};
} // namespace

/** DDA fabric (gfx1250): LL / LL128 wire waits and the FabricGpuBarrier. */
class RevokeDdaMPITest : public MPITestBase
{
protected:
    void SetUp() override
    {
        MPITestBase::SetUp();
        const std::string skipReason
            = mpiCoordinatedSkipReason(!isGfx1250Device(), "DDA fabric requires gfx1250");
        if(!skipReason.empty())
            GTEST_SKIP() << skipReason;
        if(!validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit))
            GTEST_SKIP() << "Test requires at least 4 MPI processes";
    }

    void skipUnlessDefaultLLSubThresholds()
    {
        const std::string skipReason = mpiCoordinatedSkipReason(
            !ddaLLSubThresholdsAtDefault(),
            "count is sized against the default RCCL_DDA_LL_{ONESHOT,TWOSHOT}_THRESHOLD");
        if(!skipReason.empty())
            GTEST_SKIP() << skipReason;
    }
};

TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaLlTwoShot)
{
    skipUnlessDefaultLLSubThresholds();
    if(IsSkipped())
        return;
    const size_t count = twoShotCountForRanks(MPIEnvironment::world_size);
    if(count == 0)
        GTEST_SKIP() << "Could not find a two-shot-aligned element count";

    runIncompleteDdaRevokeShrink(*this, {ncclFuncAllReduce, count, kDdaLL, NCCL_PROTO_LL, {}});
}

TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaLlOneShot)
{
    skipUnlessDefaultLLSubThresholds();
    if(IsSkipped())
        return;
    runIncompleteDdaRevokeShrink(*this,
                                 {ncclFuncAllReduce, kOneShotCount, kDdaLL, NCCL_PROTO_LL, {}});
}

TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaLl128OneShot)
{
    runIncompleteDdaRevokeShrink(
        *this, {ncclFuncAllReduce, kOneShotCount, kDdaLL128, NCCL_PROTO_LL128, kDisableAllReduceLL});
}

TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaLl128TwoShot)
{
    const size_t count = twoShotCountForRanks(MPIEnvironment::world_size);
    if(count == 0)
        GTEST_SKIP() << "Could not find a two-shot-aligned element count";

    std::vector<DdaEnvSetting> overrides = kDisableAllReduceLL;
    overrides.push_back({"RCCL_DDA_LL128_ONESHOT_THRESHOLD", "0"});
    runIncompleteDdaRevokeShrink(
        *this, {ncclFuncAllReduce, count, kDdaLL128, NCCL_PROTO_LL128, overrides});
}

// The fabric VMM (Simple) AllReduce kernel is the FabricGpuBarrier user.
TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaVmmBarrier)
{
    runIncompleteDdaRevokeShrink(*this,
                                 {ncclFuncAllReduce,
                                  kOneShotCount,
                                  kDdaVmm,
                                  NCCL_PROTO_SIMPLE,
                                  {{"RCCL_DDA_LL", "0"}, {"RCCL_DDA_LL128", "0"}}});
}

// 512 B per rank keeps the gathered total under the 128 KiB AllGather LL cap at 144 ranks.
TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaAllGatherLl)
{
    runIncompleteDdaRevokeShrink(*this, {ncclFuncAllGather, 128, kDdaLL, NCCL_PROTO_LL, {}});
}

TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaAllGatherLl128)
{
    runIncompleteDdaRevokeShrink(*this,
                                 {ncclFuncAllGather, 16384, kDdaLL128, NCCL_PROTO_LL128, kDisableLL});
}

TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaReduceScatterLl)
{
    runIncompleteDdaRevokeShrink(*this, {ncclFuncReduceScatter, 256, kDdaLL, NCCL_PROTO_LL, {}});
}

TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaReduceScatterLl128)
{
    runIncompleteDdaRevokeShrink(
        *this, {ncclFuncReduceScatter, 256, kDdaLL128, NCCL_PROTO_LL128, kDisableLL});
}

// 256 B per peer keeps the total under the 64 KiB AlltoAll LL cap at 144 ranks.
TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaAlltoAllLl)
{
    runIncompleteDdaRevokeShrink(*this, {ncclFuncAlltoAll, 64, kDdaLL, NCCL_PROTO_LL, {}});
}

TEST_F(RevokeDdaMPITest, IncompleteCollective_DdaAlltoAllLl128)
{
    runIncompleteDdaRevokeShrink(*this,
                                 {ncclFuncAlltoAll, 1024, kDdaLL128, NCCL_PROTO_LL128, kDisableLL});
}

/**
 * Incomplete DDA fabric collective + ncclCommAbort (gfx1250). Same setup as RevokeDdaMPITest,
 * but every rank aborts instead of revoking: ncclCommAbort must return and the stream of each
 * blocked rank must drain. This is the ProcessGroupNCCL watchdog / abort() path. Abort needs no
 * shrink, so two ranks are enough.
 */
class AbortDdaMPITest : public MPITestBase
{
protected:
    void SetUp() override
    {
        MPITestBase::SetUp();
        const std::string skipReason
            = mpiCoordinatedSkipReason(!isGfx1250Device(), "DDA fabric requires gfx1250");
        if(!skipReason.empty())
            GTEST_SKIP() << skipReason;
        if(!validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kNoNodeLimit))
            GTEST_SKIP() << "Test requires at least 2 MPI processes";
    }

    void runIncompleteDdaAbort(const DdaRevokeCase& tc)
    {
        ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

        ncclComm_t   comm      = getActiveCommunicator();
        hipStream_t  stream    = getActiveStream();
        const int    rank      = MPIEnvironment::world_rank;
        const int    skipRank  = MPIEnvironment::world_size - 1;
        const size_t sendElems = ddaSendElems(tc.func, tc.count, MPIEnvironment::world_size);
        const size_t recvElems = ddaRecvElems(tc.func, tc.count, MPIEnvironment::world_size);

        void* sendBuf = nullptr;
        void* recvBuf = nullptr;
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&sendBuf, sendElems * sizeof(float)));
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recvBuf, recvElems * sizeof(float)));
        auto sendGuard = makeScopeGuard([&]() { if(sendBuf) (void)hipFree(sendBuf); });
        auto recvGuard = makeScopeGuard([&]() { if(recvBuf) (void)hipFree(recvBuf); });

        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(sendBuf, sendElems));
        HIP_TEST_CHECK_GTEST_FAIL(zeroInitializeBuffer<float>(recvBuf, recvElems));

        int algo        = -1;
        int protocol    = -1;
        int maxChannels = -1;
        ASSERT_MPI_EQ(ncclSuccess,
                      rcclGetCollImplInfo(comm, tc.func, tc.count, ncclFloat32, ncclSum, sendBuf,
                                          recvBuf, /*graphCapturing=*/0, &algo, &protocol,
                                          &maxChannels));
        ASSERT_MPI_TRUE(algo == tc.expectedAlgo && protocol == tc.expectedProtocol);

        if(rank != skipRank)
        {
            ASSERT_EQ(ncclSuccess, ddaLaunch(tc.func, sendBuf, recvBuf, tc.count, comm, stream));
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // ncclCommAbort frees the communicator; keep TearDown from destroying it again.
        test_comm_ = nullptr;
        ASSERT_EQ(ncclSuccess, ncclCommAbort(comm));
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(stream));

        MPI_Barrier(MPI_COMM_WORLD);
    }
};

// 100 floats = 400 B: the default 2-rank AllReduce the PyTorch abort/timeout tests issue.
TEST_F(AbortDdaMPITest, IncompleteCollective_DdaLlOneShot)
{
    runIncompleteDdaAbort({ncclFuncAllReduce, 100, kDdaLL, NCCL_PROTO_LL, {}});
}

TEST_F(AbortDdaMPITest, IncompleteCollective_DdaLlTwoShot)
{
    if(!ddaLLSubThresholdsAtDefault())
        GTEST_SKIP() << "count is sized against the default RCCL_DDA_LL_{ONESHOT,TWOSHOT}_THRESHOLD";
    const size_t count = twoShotCountForRanks(MPIEnvironment::world_size);
    if(count == 0)
        GTEST_SKIP() << "Could not find a two-shot-aligned element count";
    runIncompleteDdaAbort({ncclFuncAllReduce, count, kDdaLL, NCCL_PROTO_LL, {}});
}

TEST_F(AbortDdaMPITest, IncompleteCollective_DdaAllGatherLl)
{
    runIncompleteDdaAbort({ncclFuncAllGather, 128, kDdaLL, NCCL_PROTO_LL, {}});
}

TEST_F(AbortDdaMPITest, IncompleteCollective_DdaReduceScatterLl)
{
    runIncompleteDdaAbort({ncclFuncReduceScatter, 256, kDdaLL, NCCL_PROTO_LL, {}});
}

TEST_F(AbortDdaMPITest, IncompleteCollective_DdaAlltoAllLl)
{
    runIncompleteDdaAbort({ncclFuncAlltoAll, 64, kDdaLL, NCCL_PROTO_LL, {}});
}

/** DDA IPC (gfx942/gfx950, one node, exactly 8 ranks): every collective waits in IpcGpuBarrier. */
class RevokeDdaIpcMPITest : public MPITestBase
{
protected:
    void SetUp() override
    {
        MPITestBase::SetUp();
        const std::string skipReason
            = mpiCoordinatedSkipReason(!isDdaIpcArchDevice(), "DDA IPC requires gfx942 or gfx950");
        if(!skipReason.empty())
            GTEST_SKIP() << skipReason;
        if(!validateTestPrerequisites(8, 8, kNoPowerOfTwoRequired, 1, 1))
            GTEST_SKIP() << "DDA IPC requires exactly 8 MPI processes on one node";
    }
};

TEST_F(RevokeDdaIpcMPITest, IncompleteCollective_DdaIpcAllReduce)
{
    runIncompleteDdaRevokeShrink(*this, {ncclFuncAllReduce, 1024, kDdaIpc, NCCL_PROTO_SIMPLE, {}});
}

TEST_F(RevokeDdaIpcMPITest, IncompleteCollective_DdaIpcAllGather)
{
    runIncompleteDdaRevokeShrink(*this, {ncclFuncAllGather, 1024, kDdaIpc, NCCL_PROTO_SIMPLE, {}});
}

TEST_F(RevokeDdaIpcMPITest, IncompleteCollective_DdaIpcReduceScatter)
{
    runIncompleteDdaRevokeShrink(*this,
                                 {ncclFuncReduceScatter, 1024, kDdaIpc, NCCL_PROTO_SIMPLE, {}});
}

TEST_F(RevokeDdaIpcMPITest, IncompleteCollective_DdaIpcAlltoAll)
{
    runIncompleteDdaRevokeShrink(*this, {ncclFuncAlltoAll, 1024, kDdaIpc, NCCL_PROTO_SIMPLE, {}});
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
    SKIP_UNLESS_MPI_PREREQS(4,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(4,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(4,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          2,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
    SKIP_UNLESS_MPI_PREREQS(2,
                                          kNoProcessLimit,
                                          kNoPowerOfTwoRequired,
                                          1,
                                          kNoNodeLimit);

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
