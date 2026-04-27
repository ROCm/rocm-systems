/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Regression tests for PR ROCm/rccl#1631: ncclCommDestroy and ncclCommAbort
// must not crash or hang when a collective is still in-flight on the network.
// These scenarios only occur with network transport, so both tests require
// at least 2 nodes and auto-skip on single-node runs.

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#include <cstdlib>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

namespace
{
constexpr size_t kBufferElements = 256 * 1024; // 1 MB of floats
constexpr size_t kBufferSize     = kBufferElements * sizeof(float);
} // namespace

class DestroyPendingCommTest : public MPITestBase
{
};

// Launch an AllReduce, then immediately destroy the communicator without
// waiting for the operation to complete.  Expects ncclSuccess and no hang.
TEST_F(DestroyPendingCommTest, Destroy_WithPendingAllReduce)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                         kNoProcessLimit,
                                         kNoPowerOfTwoRequired,
                                         2,
                                         kNoNodeLimit))
        << "Test requires at least 2 nodes with network transport";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kBufferSize));
    auto sendGuard = makeDeviceBufferAutoGuard(send_buf);
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kBufferSize));
    auto recvGuard = makeDeviceBufferAutoGuard(recv_buf);
    HIP_TEST_CHECK_GTEST_FAIL(hipMemset(send_buf, 1, kBufferSize));
    HIP_TEST_CHECK_GTEST_FAIL(hipMemset(recv_buf, 0, kBufferSize));

    RCCL_TEST_CHECK_GTEST_FAIL(ncclAllReduce(
        send_buf, recv_buf, kBufferElements, ncclFloat, ncclSum, comm, stream));

    // Destroy without syncing — this is the regression scenario from PR #1631.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommDestroy(comm));

    // Null out base-class handle so TearDown does not attempt a second destroy.
    test_comm_ = nullptr;

    // Stream may still have work; synchronize to let the GPU finish.
    (void)hipStreamSynchronize(stream);
}

// Launch an AllReduce, then immediately abort the communicator without
// waiting for the operation to complete.  Expects ncclSuccess and no hang.
TEST_F(DestroyPendingCommTest, Abort_WithPendingAllReduce)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                         kNoProcessLimit,
                                         kNoPowerOfTwoRequired,
                                         2,
                                         kNoNodeLimit))
        << "Test requires at least 2 nodes with network transport";

    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());

    ncclComm_t  comm   = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&send_buf, kBufferSize));
    auto sendGuard = makeDeviceBufferAutoGuard(send_buf);
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&recv_buf, kBufferSize));
    auto recvGuard = makeDeviceBufferAutoGuard(recv_buf);
    HIP_TEST_CHECK_GTEST_FAIL(hipMemset(send_buf, 1, kBufferSize));
    HIP_TEST_CHECK_GTEST_FAIL(hipMemset(recv_buf, 0, kBufferSize));

    RCCL_TEST_CHECK_GTEST_FAIL(ncclAllReduce(
        send_buf, recv_buf, kBufferElements, ncclFloat, ncclSum, comm, stream));

    // Abort without syncing — exercises the abort-flag path from PR #1631.
    ASSERT_MPI_EQ(ncclSuccess, ncclCommAbort(comm));

    // Null out base-class handle so TearDown does not attempt ncclCommDestroy
    // on an already-aborted communicator.
    test_comm_ = nullptr;

    (void)hipStreamSynchronize(stream);
}

#endif // MPI_TESTS_ENABLED
