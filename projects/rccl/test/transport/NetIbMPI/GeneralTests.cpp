/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "NetIbMPITestBase.hpp"
#include <sstream>
#include <array>

#ifdef MPI_TESTS_ENABLED

// Initialization Tests

TEST_F(NetIbMPITest, InitializePlugin) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    ncclResult_t result = InitNetIb();
    ASSERT_EQ(result, ncclSuccess) << "Failed to initialize NET IB plugin";
    EXPECT_NE(initCtx_, nullptr) << "Init context should be set on success";
}

TEST_F(NetIbMPITest, GetDeviceCount) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    EXPECT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    EXPECT_GT(ndev, 0) << "No IB devices found";

    if (MPIEnvironment::world_rank == 0) {
        TEST_INFO("Found %d IB device(s)", ndev);
    }
}

// Device Properties Tests

TEST_F(NetIbMPITest, GetDeviceProperties) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    for (int i = 0; i < ndev; i++) {
        ncclNetProperties_t props;
        memset(&props, 0, sizeof(props));

        EXPECT_EQ(GetDeviceProperties(i, &props), ncclSuccess)
            << "Failed to get properties for device " << i;

        EXPECT_NE(props.name, nullptr) << "Device " << i << " has NULL name";
        EXPECT_GT(props.speed, 0) << "Device " << i << " has invalid speed";
        EXPECT_NE(props.pciPath, nullptr) << "Device " << i << " has NULL pciPath";

        if (MPIEnvironment::world_rank == 0) {
            TEST_INFO("Device %d: name=%s speed=%d pciPath=%s",
                   i, props.name, props.speed, props.pciPath);
        }
    }
}

TEST_F(NetIbMPITest, GetDevicePropertiesInvalidDevice) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    ncclNetProperties_t props;

    // Invalid device ID (too large)
    ncclResult_t result = GetDeviceProperties(ndev + kInvalidDeviceOffset, &props);
    EXPECT_NE(result, ncclSuccess) << "Should fail for invalid device ID";
}

// Connection Setup Tests

TEST_F(NetIbMPITest, ListenAndConnect) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    if (rank == 0) {
        EXPECT_NE(pair.recvComm, nullptr) << "Recv comm should be established";
        EXPECT_NE(pair.listenComm, nullptr) << "Listen comm should exist";
    } else {
        EXPECT_NE(pair.sendComm, nullptr) << "Send comm should be established";
    }
}

TEST_F(NetIbMPITest, ConnectWithInvalidHandle) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    ncclNetHandle_t invalidHandle;
    memset(&invalidHandle, 0xFF, sizeof(invalidHandle));
    void* sendComm = nullptr;

    // Negative test: Connect with garbage handle
    ncclResult_t result = ConnectToRemote(0, &invalidHandle, &sendComm);
    EXPECT_EQ(result, ncclInternalError) << "Should fail with invalid handle";
}

// Memory Registration Tests

TEST_F(NetIbMPITest, RegisterHostMemory) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    const size_t bufferSize = kSmallBufferSize;
    void* buffer = malloc(bufferSize);
    ASSERT_NE(buffer, nullptr);
    auto bufferGuard = makeHostBufferAutoGuard(buffer);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    EXPECT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    EXPECT_NE(mhandle, nullptr);

    // Use NetMHandleGuard for automatic deregistration before connection closes
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));
}

TEST_F(NetIbMPITest, RegisterGpuMemory) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    const size_t bufferSize = kSmallBufferSize;
    void* buffer = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buffer, bufferSize));
    auto bufferGuard = makeDeviceBufferAutoGuard(buffer);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    EXPECT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_CUDA, &mhandle), ncclSuccess);
    EXPECT_NE(mhandle, nullptr);

    // Use NetMHandleGuard for automatic deregistration before connection closes
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));
}

TEST_F(NetIbMPITest, RegisterMemoryNullPointer) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    // Negative test: NULL buffer pointer
    ncclResult_t result = RegisterMemory(comm, nullptr, 4096, NCCL_PTR_HOST, &mhandle);
    EXPECT_NE(result, ncclSuccess) << "Should fail with NULL buffer";
}

TEST_F(NetIbMPITest, DeregisterNullHandle) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    // Edge case: Deregister NULL handle (should be no-op)
    EXPECT_EQ(DeregisterMemory(comm, nullptr), ncclSuccess);
}

// Send/Recv Tests

TEST_F(NetIbMPITest, SimpleSendRecv) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    const size_t bufferSize = kSmallBufferSize;
    const int tag = 42;

    void* buffer = malloc(bufferSize);
    ASSERT_NE(buffer, nullptr);
    auto bufferGuard = makeHostBufferAutoGuard(buffer);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);

    // Use NetMHandleGuard for automatic cleanup on failure (exception safety)
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;

    if (rank == 0) {
        // Receiver
        PostSingleRecv(pair.recvComm, buffer, bufferSize, tag, mhandle, &request);
    } else {
        // Sender
        fillHostBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(rank));
        PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for completion
    int sizes[1] = {0};
    ASSERT_NE(request, nullptr) << "Request must be non-NULL before waiting";
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Received size mismatch";

        // Verify received data
        int senderRank = 1;  // Data was sent by rank 1
        EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(senderRank))) << "Data validation failed";
    }

    // NetMHandleGuard will automatically deregister memory when test scope ends
    // Destructor order ensures MR is deregistered before connection closes:
    //   1. mhandleGuard destructor (deregisters MR)
    //   2. bufferGuard destructor (frees buffer)
    //   3. connGuard destructor (closes connection)
}

TEST_F(NetIbMPITest, SendRecvMultipleSizes) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    // Test various sizes
    std::vector<size_t> testSizes = {1, 64, 256, 1024, 4096, 16384, 65536};

    for (size_t size : testSizes) {
        const int tag = 100;
        const int seed = 2000 + static_cast<int>(size);  // Unique seed per size

        void* buffer = malloc(size);
        ASSERT_NE(buffer, nullptr);
        auto bufferGuard = makeHostBufferAutoGuard(buffer);  // Local guard for loop iteration

        void* mhandle = nullptr;
        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        ASSERT_EQ(RegisterMemory(comm, buffer, size, NCCL_PTR_HOST, &mhandle), ncclSuccess);
        NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

        void* request = nullptr;

        if (rank == 0) {
            memset(buffer, 0, size);
            PostSingleRecv(pair.recvComm, buffer, size, tag, mhandle, &request);
            ASSERT_NE(request, nullptr) << "Recv request should never be NULL";
        } else {
            fillHostBufferWithPattern<uint8_t>(buffer, size, makeBytePattern(seed));
            PostSendWithRetry(pair.sendComm, buffer, size, tag, mhandle, &request);
        }

        // Barrier 1: Ensure both ranks have posted their operations before waiting
        MPI_Barrier(MPI_COMM_WORLD);

        // Wait for completion
        int sizes[1] = {0};
        ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

        // Barrier 2: CRITICAL - Ensure BOTH ranks have completed before EITHER continues
        // This prevents rank A from starting next transfer while rank B is still
        // completing current transfer, which would cause request object reuse race conditions
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            EXPECT_EQ(sizes[0], size) << "Size mismatch for transfer of " << size << " bytes";
            EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, size, makeBytePattern(seed))) << "Data validation failed for size " << size;
        }

        // NetMHandleGuard will automatically deregister at end of loop iteration
    }
}

TEST_F(NetIbMPITest, SendRecvZeroSize) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    const size_t bufferSize = kSmallBufferSize;
    const int tag = 50;

    void* buffer = malloc(bufferSize);
    ASSERT_NE(buffer, nullptr);
    auto bufferGuard = makeHostBufferAutoGuard(buffer);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;

    if (rank == 0) {
        // Receiver - expect zero bytes
        PostSingleRecv(pair.recvComm, buffer, bufferSize, tag, mhandle, &request);
    } else {
        // Sender - send zero bytes
        PostSendWithRetry(pair.sendComm, buffer, 0, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for completion
    int sizes[1] = {-1};
    ASSERT_NE(request, nullptr) << "Request must be non-NULL before waiting";
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    if (rank == 0) {
        EXPECT_EQ(sizes[0], 0) << "Should receive zero bytes";
    }
}

TEST_F(NetIbMPITest, FlushAfterRecv) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    // Check if GDR is available
    ncclNetProperties_t props;
    ASSERT_EQ(GetDeviceProperties(0, &props), ncclSuccess);

    if (!(props.ptrSupport & NCCL_PTR_CUDA)) {
        GTEST_SKIP() << "GDR not supported, skipping flush test";
    }

    ConnectionPair pair;
    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    const size_t bufferSize = kSmallBufferSize;
    const int tag = 200;

    void* buffer = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buffer, bufferSize));
    auto bufferGuard = makeDeviceBufferAutoGuard(buffer);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_CUDA, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;

    if (rank == 0) {
        // Receiver
        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
    } else {
        // Sender
        ASSERT_EQ(initializeBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(rank)), hipSuccess);

        PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for completion
    int sizes[1] = {0};
    ASSERT_NE(request, nullptr) << "Request must be non-NULL before waiting";
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    if (rank == 0) {
        // Issue flush
        void* flushBuffers[1] = {buffer};
        int flushSizes[1] = {static_cast<int>(bufferSize)};
        void* flushHandles[1] = {mhandle};
        void* flushRequest = nullptr;

        ncclResult_t result = FlushRecv(pair.recvComm, 1, flushBuffers, flushSizes,
                                       flushHandles, &flushRequest);

        if (result == ncclSuccess && flushRequest != nullptr) {
            ASSERT_EQ(WaitForCompletion(flushRequest, nullptr), ncclSuccess);
        }
    }

    // NetMHandleGuard will automatically deregister at scope end
}

// Stress and Edge Case Tests

// Tests multiple sequential transfers on the same connection using host memory.
// This test validates that request objects can be properly reused across multiple
// send/recv operations without resource exhaustion or state corruption.
//
// SYNCHRONIZATION STRATEGY:
//   Two barriers per iteration ensure proper ordering:
//   1. After Post: Ensures both ranks have posted before either waits
//   2. After Completion: CRITICAL - Ensures BOTH ranks complete before EITHER
//      starts next iteration. Without this, rapid request reuse causes races.
//
// NULL REQUEST HANDLING:
//   NET IB isend() can return ncclSuccess with NULL request when the FIFO
//   isn't ready yet (receiver's irecv RDMA write hasn't reached sender yet).
//   This is NOT an error - it means "try again". The sender must retry until
//   it gets a valid request pointer. This is normal NET IB protocol behavior.
//
// NOTE: Flush (iflush) is intentionally NOT called because:
//   1. Flush is only needed for GPU Direct RDMA to ensure data visibility
//   2. For NCCL_PTR_HOST transfers, flush is unnecessary
TEST_F(NetIbMPITest, MultipleSequentialTransfers) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    const size_t bufferSize = kSmallBufferSize;
    const int numTransfers = kNumSequentialTransfers;

    void* sendBuffer = nullptr;
    void* recvBuffer = nullptr;
    HostBufferAutoGuard sendBufferGuard(nullptr);
    HostBufferAutoGuard recvBufferGuard(nullptr);

    if (rank == 0) {
        recvBuffer = malloc(bufferSize);
        ASSERT_NE(recvBuffer, nullptr);
        recvBufferGuard = makeHostBufferAutoGuard(recvBuffer);
    } else {
        sendBuffer = malloc(bufferSize);
        ASSERT_NE(sendBuffer, nullptr);
        sendBufferGuard = makeHostBufferAutoGuard(sendBuffer);
    }

    void* mhandle = nullptr;
    void* buffer = (rank == 0) ? recvBuffer : sendBuffer;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    for (int i = 0; i < numTransfers; i++) {
        const int tag = kTransferTagBase + i;
        const int seed = kBaseSeedOffset + i;  // Unique seed for each transfer
        void* request = nullptr;

        if (rank == 0) {
            memset(recvBuffer, 0, bufferSize);
            PostSingleRecv(pair.recvComm, recvBuffer, bufferSize, tag, mhandle, &request);
            ASSERT_NE(request, nullptr) << "Recv request should never be NULL";
        } else {
            fillHostBufferWithPattern<uint8_t>(sendBuffer, bufferSize, makeBytePattern(seed));
            PostSendWithRetry(pair.sendComm, sendBuffer, bufferSize, tag, mhandle, &request);
        }

        // Barrier 1: Ensure both ranks have posted their operations before waiting
        MPI_Barrier(MPI_COMM_WORLD);

        // Wait for completion
        int sizes[1] = {0};
        ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

        // Barrier 2: CRITICAL - Ensure BOTH ranks have completed before EITHER continues
        // This prevents rank A from starting transfer N+1 while rank B is still
        // completing transfer N, which would cause request object reuse race conditions
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            EXPECT_EQ(sizes[0], bufferSize) << "Transfer " << i << " size mismatch";

            EXPECT_TRUE(verifyHostBufferData<uint8_t>(recvBuffer, bufferSize, makeBytePattern(seed))) << "Transfer " << i << " data validation failed (seed=" << seed << ")";

            // NOTE: Flush is NOT called for host memory transfers
            // Flush (iflush) is only needed for GPU Direct RDMA to ensure data visibility on GPU.
            // For NCCL_PTR_HOST transfers, flush is unnecessary and calling it can cause
            // race conditions when request objects are rapidly reused.
            // The NET IB implementation will no-op the flush call for host memory anyway.
        }
    }

    // NetMHandleGuard will automatically deregister at scope end
}

TEST_F(NetIbMPITest, LargeTransfer) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);

    const size_t bufferSize = kLargeBufferSize; // 16 MB
    const int tag = 400;

    void* buffer = malloc(bufferSize);
    ASSERT_NE(buffer, nullptr);
    auto bufferGuard = makeHostBufferAutoGuard(buffer);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;

    if (rank == 0) {
        // Receiver
        PostSingleRecv(pair.recvComm, buffer, bufferSize, tag, mhandle, &request);
    } else {
        // Sender
        fillHostBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(rank));
        PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for completion with longer timeout for large transfer
    int sizes[1] = {0};
    ASSERT_NE(request, nullptr) << "Request must be non-NULL before waiting";
    ASSERT_EQ(WaitForCompletion(request, sizes, kLargeTransferTimeout), ncclSuccess);

    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Large transfer size mismatch";

        // Verify received data
        int senderRank = 1;  // Data was sent by rank 1
        EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(senderRank))) << "Large transfer data validation failed";
    }

    // NetMHandleGuard will automatically deregister at scope end
}

TEST_F(NetIbMPITest, CloseWithoutWaitingForCompletion) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(0, pair, connGuard);
}

TEST_F(NetIbMPITest, ListenCloseListen) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int mergedDev = CreateMergedDevice(4, rank);
    if (mergedDev == -1) {
        GTEST_SKIP() << "Failed to create merged device";
    }

    for (int iter = 0; iter < 3; iter++) {
        // Iteration 3: abandoned listen-close
        if (iter == 2) {
            if (rank == 0) {
                void* abandonedListen = nullptr;
                ncclNetHandle_t abandonedHandle;
                ASSERT_EQ(CreateListenComm(mergedDev, &abandonedHandle, &abandonedListen),
                          ncclSuccess)
                    << "Iter 3: abandoned listen failed";
                ASSERT_NE(abandonedListen, nullptr);
                ASSERT_EQ(CloseListenComm(abandonedListen), ncclSuccess)
                    << "Iter 3: close abandoned listen failed";
            }
        }

        ConnectionPair pair;
        ncclNetHandle_t handle;

        if (rank == 0) {
            ASSERT_EQ(CreateListenComm(mergedDev, &handle, &pair.listenComm), ncclSuccess)
                << "Listen failed iter " << iter;
            ASSERT_NE(pair.listenComm, nullptr);
        }

        if (rank == 0) {
            MPI_Send(&handle, sizeof(handle), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&handle, sizeof(handle), MPI_BYTE, peerRank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        int connectFailed = 0;
        if (rank == 0) {
            ncclResult_t r = ncclSuccess;
            while (!pair.recvComm && r == ncclSuccess)
                r = AcceptConnection(pair.listenComm, &pair.recvComm);
            if (r != ncclSuccess || !pair.recvComm) connectFailed = 1;
        } else {
            ncclResult_t r = ncclSuccess;
            while (!pair.sendComm && r == ncclSuccess)
                r = ConnectToRemote(mergedDev, &handle, &pair.sendComm);
            if (r != ncclSuccess || !pair.sendComm) connectFailed = 1;
        }
        MPI_Allreduce(MPI_IN_PLACE, &connectFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (connectFailed) {
            if (rank == 0 && pair.listenComm) CloseListenComm(pair.listenComm);
            FAIL() << "IB QP connect/accept failed (cross-subnet node pair)";
        }

        // Close: data comms first, then listen
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(pair.recvComm), ncclSuccess)
                << "CloseRecvComm failed iter " << iter;
        } else {
            ASSERT_EQ(CloseSendComm(pair.sendComm), ncclSuccess)
                << "CloseSendComm failed iter " << iter;
        }

        if (rank == 0) {
            ASSERT_EQ(CloseListenComm(pair.listenComm), ncclSuccess)
                << "CloseListenComm failed iter " << iter;
        }
    }
}

TEST_F(NetIbMPITest, MultipleSimultaneousListens) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(InitNetIb(), ncclSuccess);
    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    std::vector<ncclNetProperties_t> props(ndev);
    std::vector<int> physDevs;
    for (int i = 0; i < ndev; i++) {
        memset(&props[i], 0, sizeof(ncclNetProperties_t));
        ASSERT_EQ(GetDeviceProperties(i, &props[i]), ncclSuccess);
        if (!props[i].name || !strchr(props[i].name, '+'))
            physDevs.push_back(i);
    }

    if (physDevs.empty()) {
        GTEST_SKIP() << "Failed to find network devices";
    }

    int targetSpeed = props[physDevs[0]].speed;
    std::vector<int> compat;
    for (int d : physDevs)
        if (props[d].speed == targetSpeed) compat.push_back(d);

    int mergedDevA = CreateMergedDevice(2, rank);
    int mergedDevB = CreateMergedDevice(3, rank, 2);
    if (mergedDevA == -1 || mergedDevB == -1) {
        GTEST_SKIP() << "Failed to create merged device";
    }

    // Physical device = first NIC from merged A's group
    int physDev = compat[0];

    static constexpr size_t kTransferSize = 4096;
    static constexpr int kNumListens = 4;

    // 4 listens: 2 on merged A, 1 on merged B, 1 on physical NIC (member of merged A)
    struct ListenInfo {
        int dev;
        const char* label;
    };
    ListenInfo listens[kNumListens] = {
        {mergedDevA, "MergedA-1"},
        {mergedDevA, "MergedA-2"},
        {mergedDevB, "MergedB"},
        {physDev,    "PhysDev"},
    };

    void* listenComms[kNumListens] = {};
    void* recvComms[kNumListens] = {};
    void* sendComms[kNumListens] = {};
    ncclNetHandle_t handles[kNumListens] = {};

    // === Phase 1: Rank 0 creates ALL 4 listens simultaneously ===
    if (rank == 0) {
        for (int i = 0; i < kNumListens; i++) {
            ASSERT_EQ(CreateListenComm(listens[i].dev, &handles[i], &listenComms[i]),
                      ncclSuccess)
                << "CreateListenComm failed for " << listens[i].label;
            ASSERT_NE(listenComms[i], nullptr)
                << "listenComm NULL for " << listens[i].label;
        }

        // Verify all handles are pairwise unique
        for (int i = 0; i < kNumListens; i++) {
            for (int j = i + 1; j < kNumListens; j++) {
                bool eq = (memcmp(&handles[i], &handles[j], sizeof(ncclNetHandle_t)) == 0);
                EXPECT_FALSE(eq)
                    << "Handles identical for " << listens[i].label
                    << " and " << listens[j].label;
            }
        }

        // Send all handles to rank 1
        for (int i = 0; i < kNumListens; i++) {
            MPI_Send(&handles[i], sizeof(ncclNetHandle_t), MPI_BYTE,
                     peerRank, 60 + i, MPI_COMM_WORLD);
        }
    } else {
        for (int i = 0; i < kNumListens; i++) {
            MPI_Recv(&handles[i], sizeof(ncclNetHandle_t), MPI_BYTE,
                     peerRank, 60 + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // === Phase 2: ALL connect + accept ===
    int phase2Failed = 0;
    if (rank == 0) {
        for (int i = 0; i < kNumListens && !phase2Failed; i++) {
            ncclResult_t r = ncclSuccess;
            while (!recvComms[i] && r == ncclSuccess)
                r = AcceptConnection(listenComms[i], &recvComms[i]);
            if (r != ncclSuccess || !recvComms[i])
                phase2Failed = 1;
        }
    } else {
        for (int i = 0; i < kNumListens && !phase2Failed; i++) {
            ncclResult_t r = ncclSuccess;
            while (!sendComms[i] && r == ncclSuccess)
                r = ConnectToRemote(listens[i].dev, &handles[i], &sendComms[i]);
            if (r != ncclSuccess || !sendComms[i])
                phase2Failed = 1;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &phase2Failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (phase2Failed) {
        GTEST_SKIP() << "IB QP connect/accept failed (cross-subnet node pair)";
    }

    // === Phase 3: Transfer on ALL 4 connections ===
    for (int c = 0; c < kNumListens; c++) {
        void* comm = (rank == 0) ? recvComms[c] : sendComms[c];
        int seed = 2000 + c * 1000;
        int tag = 200 + c;

        void* buf = malloc(kTransferSize);
        ASSERT_NE(buf, nullptr);
        auto bufGuard = RCCLTestGuards::makeHostBufferAutoGuard(buf);
        void* mhandle = nullptr;
        ASSERT_EQ(RegisterMemory(comm, buf, kTransferSize, NCCL_PTR_HOST, &mhandle),
                  ncclSuccess);

        if (rank == 1) {
            fillHostBufferWithPattern<uint8_t>(buf, kTransferSize, makeBytePattern(seed));
        } else {
            memset(buf, 0xDE, kTransferSize);
        }

        void* req = nullptr;
        if (rank == 0) {
            void* rb[1] = {buf}; size_t rs[1] = {kTransferSize};
            int rt[1] = {tag}; void* rh[1] = {mhandle};
            ASSERT_EQ(PostRecv(recvComms[c], 1, rb, rs, rt, rh, &req), ncclSuccess);
            ASSERT_NE(req, nullptr);
        }

        if (rank == 1) {
            void* sendReq = nullptr;
            PostSendWithRetry(sendComms[c], buf, kTransferSize, tag, mhandle, &sendReq);
        }

        if (rank == 0) {
            int csz[1] = {0};
            ASSERT_EQ(WaitForCompletion(req, csz, kLargeTransferTimeoutMs), ncclSuccess)
                << listens[c].label << " recv timed out";
            EXPECT_EQ(csz[0], (int)kTransferSize)
                << listens[c].label << " size mismatch";

            size_t errIdx = 0;
            uint8_t errExp = 0, errGot = 0;
            bool ok = verifyHostBufferData<uint8_t>(buf, kTransferSize,
                makeBytePattern(seed), 0, 0.0, &errIdx, &errExp, &errGot);
            EXPECT_TRUE(ok) << listens[c].label << " data mismatch at byte " << errIdx
                            << ": expected " << (int)errExp << " got " << (int)errGot;
        }

        ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
    }

    // === Phase 4: Close ALL ===
    if (rank == 0) {
        for (int i = 0; i < kNumListens; i++)
            ASSERT_EQ(CloseRecvComm(recvComms[i]), ncclSuccess)
                << "CloseRecvComm failed for " << listens[i].label;
    } else {
        for (int i = 0; i < kNumListens; i++)
            ASSERT_EQ(CloseSendComm(sendComms[i]), ncclSuccess)
                << "CloseSendComm failed for " << listens[i].label;
    }

    if (rank == 0) {
        for (int i = 0; i < kNumListens; i++)
            ASSERT_EQ(CloseListenComm(listenComms[i]), ncclSuccess)
                << "CloseListenComm failed for " << listens[i].label;
    }
}

TEST_F(NetIbMPITest, MultipleSequentialConnections) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    // Open and close multiple connections sequentially to test resource cleanup
    const int kNumIterations = 5;
    for (int iter = 0; iter < kNumIterations; iter++) {
        ConnectionPair pair;
        ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess)
            << "Failed to setup connection on iteration " << iter;

        NetConnectionGuard connGuard(net_);
        if (rank == 0) {
            connGuard.setRecvComm(pair.recvComm);
            connGuard.setListenComm(pair.listenComm);
            EXPECT_NE(pair.recvComm, nullptr);
        } else {
            connGuard.setSendComm(pair.sendComm);
            EXPECT_NE(pair.sendComm, nullptr);
        }
        // connGuard destructor closes connection
    }

    if (MPIEnvironment::world_rank == 0) {
        TEST_INFO("Successfully completed %d sequential connection cycles", kNumIterations);
    }
}

TEST_F(NetIbMPITest, RapidConnectDisconnect) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    std::vector<ncclNetProperties_t> props(ndev);
    std::vector<int> physDevs;
    for (int i = 0; i < ndev; i++) {
        memset(&props[i], 0, sizeof(ncclNetProperties_t));
        ASSERT_EQ(GetDeviceProperties(i, &props[i]), ncclSuccess);
        if (!props[i].name || !strchr(props[i].name, '+'))
            physDevs.push_back(i);
    }
    ASSERT_FALSE(physDevs.empty()) << "No physical devices found";

    int mergedDev = CreateMergedDevice(3, rank);
    if (mergedDev == -1) {
        GTEST_SKIP() << "Failed to create merged device";
    }


    static constexpr int kIterations = 100;

    struct RdmaResourceCounts {
        int qp = -1;
        int cq = -1;
        int mr = -1;
        int pd = -1;
    };

    auto execCommand = [](const char* cmd) -> std::string {
        std::array<char, 256> buf{};
        std::string out;

        FILE* pipe = popen(cmd, "r");
        if (!pipe) return out;

        while (fgets(buf.data(), buf.size(), pipe) != nullptr)
            out += buf.data();

        pclose(pipe);
        return out;
    };

    auto countNonEmptyLines = [](const std::string& text) -> int {
        std::istringstream iss(text);
        std::string line;
        int count = 0;
        while (std::getline(iss, line)) {
            if (!line.empty()) count++;
        }
        return count;
    };

    auto readRdmaResourceCounts = [&]() -> RdmaResourceCounts {
        RdmaResourceCounts counts;

        std::string probe =
            execCommand("sh -c 'rdma resource show qp >/dev/null 2>&1 && "
                        "rdma resource show cq >/dev/null 2>&1 && "
                        "rdma resource show mr >/dev/null 2>&1 && "
                        "rdma resource show pd >/dev/null 2>&1 && echo OK'");
        if (probe.find("OK") == std::string::npos) return counts;

        std::string qpOut = execCommand("rdma resource show qp 2>/dev/null");
        std::string cqOut = execCommand("rdma resource show cq 2>/dev/null");
        std::string mrOut = execCommand("rdma resource show mr 2>/dev/null");
        std::string pdOut = execCommand("rdma resource show pd 2>/dev/null");

        if (qpOut.empty() || cqOut.empty() || mrOut.empty() || pdOut.empty()) return counts;

        counts.qp = countNonEmptyLines(qpOut);
        counts.cq = countNonEmptyLines(cqOut);
        counts.mr = countNonEmptyLines(mrOut);
        counts.pd = countNonEmptyLines(pdOut);

        return counts;
    };

    auto countsAvailable = [](const RdmaResourceCounts& c) -> bool {
        return c.qp >= 0 && c.cq >= 0 && c.mr >= 0 && c.pd >= 0;
    };

    RdmaResourceCounts before = readRdmaResourceCounts();

    MPI_Barrier(MPI_COMM_WORLD);

    for (int iter = 0; iter < kIterations; iter++) {
        void* listenComm = nullptr;
        void* recvComm = nullptr;
        void* sendComm = nullptr;
        ncclNetHandle_t handle;
        memset(&handle, 0, sizeof(handle));

        if (rank == 0) {
            ASSERT_EQ(CreateListenComm(mergedDev, &handle, &listenComm), ncclSuccess)
                << "CreateListenComm failed at iteration " << iter;
            ASSERT_NE(listenComm, nullptr)
                << "listenComm is NULL at iteration " << iter;

            MPI_Send(&handle, sizeof(ncclNetHandle_t), MPI_BYTE,
                     peerRank, 7000 + iter, MPI_COMM_WORLD);
        } else {
            MPI_Recv(&handle, sizeof(ncclNetHandle_t), MPI_BYTE,
                     peerRank, 7000 + iter, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            int attempts = 0;
            while (!recvComm && attempts < kMaxRetryAttempts) {
                ASSERT_EQ(AcceptConnection(listenComm, &recvComm), ncclSuccess)
                    << "AcceptConnection failed at iteration " << iter;
                if (!recvComm) usleep(kPollIntervalUs);
                attempts++;
            }
            ASSERT_NE(recvComm, nullptr)
                << "AcceptConnection did not complete at iteration " << iter;
        } else {
            int attempts = 0;
            while (!sendComm && attempts < kMaxRetryAttempts) {
                ASSERT_EQ(ConnectToRemote(mergedDev, &handle, &sendComm), ncclSuccess)
                    << "ConnectToRemote failed at iteration " << iter;
                if (!sendComm) usleep(kPollIntervalUs);
                attempts++;
            }
            ASSERT_NE(sendComm, nullptr)
                << "ConnectToRemote did not complete at iteration " << iter;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(recvComm), ncclSuccess)
                << "CloseRecvComm failed at iteration " << iter;
            ASSERT_EQ(CloseListenComm(listenComm), ncclSuccess)
                << "CloseListenComm failed at iteration " << iter;
        } else {
            ASSERT_EQ(CloseSendComm(sendComm), ncclSuccess)
                << "CloseSendComm failed at iteration " << iter;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    RdmaResourceCounts after = readRdmaResourceCounts();

    if (countsAvailable(before) && countsAvailable(after)) {
        EXPECT_EQ(after.qp, before.qp)
            << "QP leak detected on rank " << rank
            << ": before=" << before.qp << " after=" << after.qp;
        EXPECT_EQ(after.cq, before.cq)
            << "CQ leak detected on rank " << rank
            << ": before=" << before.cq << " after=" << after.cq;
        EXPECT_EQ(after.mr, before.mr)
            << "MR leak detected on rank " << rank
            << ": before=" << before.mr << " after=" << after.mr;
        EXPECT_EQ(after.pd, before.pd)
            << "PD leak detected on rank " << rank
            << ": before=" << before.pd << " after=" << after.pd;
    } else {
        GTEST_SKIP() << "RDMA resource counters unavailable; rapid connect/disconnect "
                     << "behavior was tested, but QP/CQ/MR/PD leak check was skipped";
    }
}

TEST_F(NetIbMPITest, MultiRecv) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    static constexpr int kWidth = 8;
    static constexpr int kNumBatches = 255;
    static constexpr int kBaseTag = 8000;
    static constexpr size_t kSizes[kWidth] = {
        1, 64, 256, 1024, 4096, 16384, 65536, 131072
    };

    auto FillPattern = [&](void* buf, size_t size, int batch, int slot) {
        fillHostBufferWithPattern<uint8_t>(buf, size, [batch, slot](size_t j) {
            return static_cast<uint8_t>((batch * 17 + slot * 31 + j) % 256);
        });
    };

    auto CheckPattern = [&](void* buf, size_t size, int batch, int slot) -> bool {
        return verifyHostBufferData<uint8_t>(buf, size, [batch, slot](size_t j) {
            return static_cast<uint8_t>((batch * 17 + slot * 31 + j) % 256);
        });
    };

    ConnectionPair pair;
    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);
    void* recvComm = pair.recvComm;
    void* sendComm = pair.sendComm;

    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    for (int batch = 0; batch < kNumBatches; batch++) {
        size_t sizes[kWidth];
        int tags[kWidth];
        for (int i = 0; i < kWidth; i++) {
            sizes[i] = kSizes[i];
            tags[i] = kBaseTag + batch * kWidth + i;
        }

        if (rank == 0) {
            void* bufs[kWidth] = {};
            void* mhs[kWidth] = {};
            int recvSizes[kWidth] = {};

            auto cleanup = makeScopeGuard([&]() {
                for (int i = 0; i < kWidth; i++) {
                    if (mhs[i]) { DeregisterMemory(recvComm, mhs[i]); mhs[i] = nullptr; }
                    if (bufs[i]) { free(bufs[i]); bufs[i] = nullptr; }
                }
            });

            for (int i = 0; i < kWidth; i++) {
                bufs[i] = malloc(sizes[i]);
                ASSERT_NE(bufs[i], nullptr)
                    << "malloc failed for recv batch=" << batch << " slot=" << i;
                memset(bufs[i], 0xCC, sizes[i]);

                ASSERT_EQ(RegisterMemory(recvComm, bufs[i], sizes[i], NCCL_PTR_HOST, &mhs[i]),
                          ncclSuccess)
                    << "RegisterMemory failed for recv batch=" << batch << " slot=" << i;
                ASSERT_NE(mhs[i], nullptr);
            }

            void* req = nullptr;
            ASSERT_EQ(PostRecv(recvComm, kWidth, bufs, sizes, tags, mhs, &req), ncclSuccess)
                << "PostRecv failed for batch=" << batch;
            ASSERT_NE(req, nullptr) << "PostRecv returned null request for batch=" << batch;

            MPI_Barrier(MPI_COMM_WORLD);

            ASSERT_EQ(WaitForCompletion(req, recvSizes, kLargeTransferTimeoutMs), ncclSuccess)
                << "WaitForCompletion failed for recv batch=" << batch;

            for (int i = 0; i < kWidth; i++) {
                EXPECT_EQ(recvSizes[i], static_cast<int>(sizes[i]))
                    << "recv size mismatch at batch=" << batch << " slot=" << i;

                EXPECT_TRUE(CheckPattern(bufs[i], sizes[i], batch, i))
                    << "data mismatch at batch=" << batch
                    << " slot=" << i
                    << " size=" << sizes[i];
            }

            // cleanup guard fires here, deregistering MRs and freeing buffers
        } else {
            void* bufs[kWidth] = {};
            void* mhs[kWidth] = {};
            void* reqs[kWidth] = {};

            auto cleanup = makeScopeGuard([&]() {
                for (int i = 0; i < kWidth; i++) {
                    if (mhs[i]) { DeregisterMemory(sendComm, mhs[i]); mhs[i] = nullptr; }
                    if (bufs[i]) { free(bufs[i]); bufs[i] = nullptr; }
                }
            });

            for (int i = 0; i < kWidth; i++) {
                bufs[i] = malloc(sizes[i]);
                ASSERT_NE(bufs[i], nullptr)
                    << "malloc failed for send batch=" << batch << " slot=" << i;

                FillPattern(bufs[i], sizes[i], batch, i);

                ASSERT_EQ(RegisterMemory(sendComm, bufs[i], sizes[i], NCCL_PTR_HOST, &mhs[i]),
                          ncclSuccess)
                    << "RegisterMemory failed for send batch=" << batch << " slot=" << i;
                ASSERT_NE(mhs[i], nullptr);
            }

            MPI_Barrier(MPI_COMM_WORLD);

            for (int i = 0; i < kWidth; i++) {
                reqs[i] = nullptr;
                PostSendWithRetry(sendComm, bufs[i], sizes[i], tags[i], mhs[i], &reqs[i]);
                ASSERT_NE(reqs[i], nullptr)
                    << "PostSend returned null request for batch=" << batch
                    << " slot=" << i;
            }

            for (int i = 0; i < kWidth; i++) {
                int sentSize[1] = {0}; // send request yields one size entry
                ASSERT_EQ(WaitForCompletion(reqs[i], sentSize, kLargeTransferTimeoutMs), ncclSuccess)
                    << "send completion failed for batch=" << batch << " slot=" << i;
            }

            // cleanup guard fires here, deregistering MRs and freeing buffers
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // connGuard closes comms on scope exit
}

TEST_F(NetIbMPITest, MultiRecvShuffled) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    static constexpr int kWidth = 8;
    static constexpr int kNumBatches = 255;
    static constexpr int kBaseTag = 12000;
    // stride 100 between batches keeps tags visually distinct in debug logs
    static constexpr int kTagStride = 100;
    static constexpr size_t kBaseSizes[kWidth] = {
        1, 64, 256, 1024, 4096, 16384, 65536, 131072
    };

    // kRecvOrder[slot] = msgId posted at that recv slot (recv posts in slot order)
    static constexpr int kRecvOrder[kWidth] = {3, 0, 6, 1, 7, 2, 5, 4};
    // kSendOrder[issueIndex] = msgId to send at that issue position
    static constexpr int kSendOrder[kWidth] = {5, 2, 7, 0, 6, 3, 1, 4};

    auto FillPattern = [&](void* buf, size_t size, int batch, int msgId) {
        fillHostBufferWithPattern<uint8_t>(buf, size, [batch, msgId](size_t j) {
            return static_cast<uint8_t>((batch * 19 + msgId * 37 + j) % 256);
        });
    };

    auto CheckPattern = [&](void* buf, size_t size, int batch, int msgId) -> bool {
        return verifyHostBufferData<uint8_t>(buf, size, [batch, msgId](size_t j) {
            return static_cast<uint8_t>((batch * 19 + msgId * 37 + j) % 256);
        });
    };

    ConnectionPair pair;
    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);
    void* recvComm = pair.recvComm;
    void* sendComm = pair.sendComm;

    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    for (int batch = 0; batch < kNumBatches; batch++) {
        int msgTags[kWidth];
        size_t msgSizes[kWidth];
        for (int msgId = 0; msgId < kWidth; msgId++) {
            msgTags[msgId] = kBaseTag + batch * kTagStride + msgId;
            msgSizes[msgId] = kBaseSizes[msgId];
        }

        if (rank == 0) {
            void* bufs[kWidth] = {};
            void* mhs[kWidth] = {};
            size_t recvSizesArg[kWidth];
            int recvTags[kWidth];
            int recvSizesOut[kWidth] = {};

            auto cleanup = makeScopeGuard([&]() {
                for (int i = 0; i < kWidth; i++) {
                    if (mhs[i]) { DeregisterMemory(recvComm, mhs[i]); mhs[i] = nullptr; }
                    if (bufs[i]) { free(bufs[i]); bufs[i] = nullptr; }
                }
            });

            for (int slot = 0; slot < kWidth; slot++) {
                int msgId = kRecvOrder[slot];
                recvTags[slot] = msgTags[msgId];
                recvSizesArg[slot] = msgSizes[msgId];

                bufs[slot] = malloc(recvSizesArg[slot]);
                ASSERT_NE(bufs[slot], nullptr)
                    << "malloc failed for recv batch=" << batch << " slot=" << slot;
                memset(bufs[slot], 0xCC, recvSizesArg[slot]);

                ASSERT_EQ(RegisterMemory(recvComm, bufs[slot], recvSizesArg[slot],
                                         NCCL_PTR_HOST, &mhs[slot]),
                          ncclSuccess)
                    << "RegisterMemory failed for recv batch=" << batch << " slot=" << slot;
                ASSERT_NE(mhs[slot], nullptr);
            }

            void* req = nullptr;
            ASSERT_EQ(PostRecv(recvComm, kWidth, bufs, recvSizesArg, recvTags, mhs, &req),
                      ncclSuccess)
                << "PostRecv failed for batch=" << batch;
            ASSERT_NE(req, nullptr) << "PostRecv returned null request for batch=" << batch;

            MPI_Barrier(MPI_COMM_WORLD);

            ASSERT_EQ(WaitForCompletion(req, recvSizesOut, kLargeTransferTimeoutMs), ncclSuccess)
                << "WaitForCompletion failed for recv batch=" << batch;

            for (int slot = 0; slot < kWidth; slot++) {
                int msgId = kRecvOrder[slot];
                EXPECT_EQ(recvSizesOut[slot], static_cast<int>(msgSizes[msgId]))
                    << "recv size mismatch at batch=" << batch
                    << " slot=" << slot
                    << " msgId=" << msgId;

                EXPECT_TRUE(CheckPattern(bufs[slot], msgSizes[msgId], batch, msgId))
                    << "data mismatch at batch=" << batch
                    << " slot=" << slot
                    << " expected msgId=" << msgId
                    << " tag=" << msgTags[msgId]
                    << " size=" << msgSizes[msgId];
            }

            // cleanup guard fires here, deregistering MRs and freeing buffers
        } else {
            void* bufs[kWidth] = {};
            void* mhs[kWidth] = {};
            void* reqs[kWidth] = {};

            auto cleanup = makeScopeGuard([&]() {
                for (int i = 0; i < kWidth; i++) {
                    if (mhs[i]) { DeregisterMemory(sendComm, mhs[i]); mhs[i] = nullptr; }
                    if (bufs[i]) { free(bufs[i]); bufs[i] = nullptr; }
                }
            });

            for (int msgId = 0; msgId < kWidth; msgId++) {
                bufs[msgId] = malloc(msgSizes[msgId]);
                ASSERT_NE(bufs[msgId], nullptr)
                    << "malloc failed for send batch=" << batch << " msgId=" << msgId;

                FillPattern(bufs[msgId], msgSizes[msgId], batch, msgId);

                ASSERT_EQ(RegisterMemory(sendComm, bufs[msgId], msgSizes[msgId],
                                         NCCL_PTR_HOST, &mhs[msgId]),
                          ncclSuccess)
                    << "RegisterMemory failed for send batch=" << batch
                    << " msgId=" << msgId;
                ASSERT_NE(mhs[msgId], nullptr);
            }

            MPI_Barrier(MPI_COMM_WORLD);

            for (int oi = 0; oi < kWidth; oi++) {
                int msgId = kSendOrder[oi];
                reqs[msgId] = nullptr;
                PostSendWithRetry(sendComm, bufs[msgId], msgSizes[msgId],
                                  msgTags[msgId], mhs[msgId], &reqs[msgId]);
                ASSERT_NE(reqs[msgId], nullptr)
                    << "PostSend returned null request for batch=" << batch
                    << " msgId=" << msgId;
            }

            for (int msgId = 0; msgId < kWidth; msgId++) {
                int sentSize[1] = {0}; // send request yields one size entry
                ASSERT_EQ(WaitForCompletion(reqs[msgId], sentSize, kLargeTransferTimeoutMs),
                          ncclSuccess)
                    << "send completion failed for batch=" << batch
                    << " msgId=" << msgId;
            }

            // cleanup guard fires here, deregistering MRs and freeing buffers
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // connGuard closes comms on scope exit
}

// CTS cache overflow stress test. N P2P connections x D outstanding recvs.
// Env: CTS_NUM_CONNS (32), CTS_QP_DEPTH (256), CTS_SNAP_EVERY (1).
// With RCCL_IB_P2P_DISABLE_CTS=0 the AINIC CTS table overflows ~256 entries
// and the drain times out; with =1 it passes. Requires exactly 2 ranks.
TEST_F(NetIbMPITest, CtsDepthStress) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " MPI processes";

    net_ = &netIbCast;
    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    const int rank        = MPIEnvironment::world_rank;
    const int kMaxRetries = 10000000;

    auto envInt = [](const char* name, int dflt) {
        const char* e = getenv(name);
        return e ? atoi(e) : dflt;
    };
    const int numConns  = envInt("CTS_NUM_CONNS",  32);
    const int qpDepth   = envInt("CTS_QP_DEPTH",   256);
    const int snapEvery = envInt("CTS_SNAP_EVERY", 1);
    ASSERT_GT(numConns,  0) << "CTS_NUM_CONNS must be > 0";
    ASSERT_GT(qpDepth,   0) << "CTS_QP_DEPTH must be > 0";
    ASSERT_GT(snapEvery, 0) << "CTS_SNAP_EVERY must be > 0";

    const int totalEntries = numConns * qpDepth;
    const size_t bufSize   = 4096;
    const int    baseTag   = 100;

    std::cout << "[Rank " << rank << "] CtsDepthStress:"
              << "  numConns=" << numConns
              << "  qpDepth=" << qpDepth
              << "  totalCtsEntries=" << totalEntries
              << "  snapEvery=" << snapEvery
              << "  ndev=" << ndev
              << "\n" << std::flush;

    // Self-cleaning so ASSERT_* mid-fill doesn't leak comms / MRs / bufs.
    struct Conn {
        ncclNet_t* net = nullptr;
        void* sendComm   = nullptr;
        void* recvComm   = nullptr;
        void* listenComm = nullptr;
        void* sendMhandle = nullptr;
        std::vector<void*> recvMhandles;
        std::vector<void*> recvBufs;
        std::vector<void*> recvRequests;

        Conn() = default;
        Conn(const Conn&) = delete;
        Conn& operator=(const Conn&) = delete;

        ~Conn() {
            if (!net) return;
            for (size_t d = 0; d < recvMhandles.size(); d++) {
                if (recvMhandles[d] && recvComm) net->deregMr(recvComm, recvMhandles[d]);
            }
            for (auto* b : recvBufs) if (b) free(b);
            if (sendMhandle && sendComm) net->deregMr(sendComm, sendMhandle);
            if (recvComm)   net->closeRecv(recvComm);
            if (sendComm)   net->closeSend(sendComm);
            if (listenComm) net->closeListen(listenComm);
        }
    };

    using namespace NetIbCts;
    CounterMap snapInit = takeSnapshot();
    std::cout << "[Rank " << rank << "] snapshot: Init\n" << std::flush;

    std::vector<std::unique_ptr<Conn>> conns(numConns);
    for (int i = 0; i < numConns; i++) {
        conns[i] = std::make_unique<Conn>();
        conns[i]->net = net_;
    }

    if (rank == 0) {
        std::vector<ncclNetHandle_t> handles(numConns);
        std::cout << "[Rank 0] listen() x " << numConns << "...\n" << std::flush;
        for (int i = 0; i < numConns; i++) {
            ASSERT_EQ(net_->listen(initCtx_, i % ndev, &handles[i],
                                   &conns[i]->listenComm), ncclSuccess)
                << "listen failed conn=" << i;
            ASSERT_EQ(rcclCastNetP2pPolicy(&handles[i], 1), ncclSuccess);
            MPI_Send(&handles[i], sizeof(ncclNetHandle_t), MPI_BYTE,
                     1, i, MPI_COMM_WORLD);
        }

        std::cout << "[Rank 0] accept() x " << numConns << "...\n" << std::flush;
        for (int i = 0; i < numConns; i++) {
            int retries = 0;
            while (!conns[i]->recvComm) {
                ASSERT_EQ(net_->accept(conns[i]->listenComm,
                                       &conns[i]->recvComm, nullptr),
                          ncclSuccess) << "accept failed conn=" << i;
                ++retries;
                if (retries % 1000 == 0) usleep(100);
                ASSERT_LT(retries, kMaxRetries) << "accept timeout conn=" << i;
            }
            conns[i]->recvBufs.resize(qpDepth);
            conns[i]->recvMhandles.resize(qpDepth, nullptr);
            conns[i]->recvRequests.resize(qpDepth, nullptr);
            for (int d = 0; d < qpDepth; d++) {
                conns[i]->recvBufs[d] = malloc(bufSize);
                ASSERT_NE(conns[i]->recvBufs[d], nullptr);
                ASSERT_EQ(RegisterMemory(conns[i]->recvComm,
                                         conns[i]->recvBufs[d], bufSize,
                                         NCCL_PTR_HOST,
                                         &conns[i]->recvMhandles[d]), ncclSuccess)
                    << "regMr failed conn=" << i << " depth=" << d;
            }
        }
        std::cout << "[Rank 0] all " << numConns << " connections up\n" << std::flush;

        MPI_Barrier(MPI_COMM_WORLD);

        // snapBeforePostRecv: fixed baseline for the final delta.
        // snapStep: advances each sample so prints are per-window.
        CounterMap snapBeforePostRecv = takeSnapshot();
        CounterMap snapStep           = snapBeforePostRecv;
        std::cout << "[Rank 0] snapshot: BeforePostRecv\n"
                  << "[Rank 0] filling " << numConns << " x " << qpDepth
                  << " = " << totalEntries << " CTS entries...\n" << std::flush;

        bool overflowDetected = false;
        int  overflowAtConn   = -1;

        for (int i = 0; i < numConns; i++) {
            for (int d = 0; d < qpDepth; d++) {
                void*  rb[1]  = {conns[i]->recvBufs[d]};
                size_t rs[1]  = {bufSize};
                int    rt[1]  = {baseTag + i * qpDepth + d};
                void*  rh[1]  = {conns[i]->recvMhandles[d]};
                ASSERT_EQ(PostRecv(conns[i]->recvComm, 1, rb, rs, rt, rh,
                                   &conns[i]->recvRequests[d]), ncclSuccess)
                    << "PostRecv failed conn=" << i << " depth=" << d;
                ASSERT_NE(conns[i]->recvRequests[d], nullptr);
            }

            if ((i + 1) % snapEvery == 0) {
                CounterMap snapNow = takeSnapshot();
                printSummary(rank, "filling step", i + 1, qpDepth, snapStep, snapNow);
                auto s = calcSummary(snapStep, snapNow);
                if ((s.retx_pkts > 0 || s.ack_timeout > 0) && !overflowDetected) {
                    overflowDetected = true;
                    overflowAtConn   = i + 1;
                    printDelta(rank, "BeforePostRecv", "OverflowPoint",
                               snapBeforePostRecv, snapNow);
                }
                snapStep = std::move(snapNow);
            }
        }

        CounterMap snapAfterRecv = takeSnapshot();
        std::cout << "\n[Rank 0] snapshot: AfterPostRecv\n" << std::flush;
        printDelta(rank, "BeforePostRecv", "AfterPostRecv",
                   snapBeforePostRecv, snapAfterRecv);
        printSummary(rank, "FINAL after all PostRecv", numConns, qpDepth,
                     snapBeforePostRecv, snapAfterRecv);

        if (!overflowDetected)
            std::cout << "[Rank 0] No overflow at " << totalEntries
                      << " entries. Try CTS_NUM_CONNS=" << numConns * 2 << "\n"
                      << std::flush;
        else
            std::cout << "[Rank 0] Overflow confirmed at conn=" << overflowAtConn
                      << " (~" << overflowAtConn * qpDepth << " entries)\n"
                      << std::flush;

        // On overflow, WaitForCompletion times out and we report the partial
        // completion count -- that's the observable CTS-overflow signature.
        MPI_Barrier(MPI_COMM_WORLD);

        std::cout << "[Rank 0] draining " << numConns << " connections"
                  << (overflowDetected ? " (OVERFLOW - expect drain timeout)" : "")
                  << "...\n" << std::flush;

        int totalCompleted = 0;
        for (int i = 0; i < numConns; i++) {
            for (int d = 0; d < qpDepth; d++) {
                int sizes[1] = {0};
                if (WaitForCompletion(conns[i]->recvRequests[d], sizes,
                                      kLargeTransferTimeoutMs) != ncclSuccess) {
                    std::cout << "[Rank 0] TIMEOUT conn=" << i
                              << " depth=" << d
                              << " total_completed=" << totalCompleted << "\n"
                              << std::flush;
                    goto drain_done;
                }
                totalCompleted++;
            }
            if ((i + 1) % 8 == 0)
                std::cout << "[Rank 0] drained " << (i+1) << "/" << numConns
                          << " (" << totalCompleted << " completions)\n"
                          << std::flush;
        }
        drain_done:
        std::cout << "[Rank 0] " << totalCompleted << "/" << totalEntries
                  << " completions done\n" << std::flush;

        CounterMap snapAfterBurst = takeSnapshot();
        std::cout << "[Rank 0] snapshot: AfterBurst\n" << std::flush;
        printDelta(rank, "AfterPostRecv", "AfterBurst", snapAfterRecv, snapAfterBurst);
        printSummary(rank, "AfterBurst delta", numConns, qpDepth,
                     snapAfterRecv, snapAfterBurst);
        printDelta(rank, "Init", "Full run", snapInit, snapAfterBurst);
        printSummary(rank, "Full run", numConns, qpDepth, snapInit, snapAfterBurst);

        EXPECT_EQ(totalCompleted, totalEntries);

        MPI_Barrier(MPI_COMM_WORLD);

    } else {
        HostBufferAutoGuard sendBufGuard(malloc(bufSize));
        void* const sendBuf = sendBufGuard.get();
        ASSERT_NE(sendBuf, nullptr);
        memset(sendBuf, 0xab, bufSize);

        std::cout << "[Rank 1] connect() x " << numConns << "...\n" << std::flush;
        for (int i = 0; i < numConns; i++) {
            ncclNetHandle_t handle;
            MPI_Recv(&handle, sizeof(ncclNetHandle_t), MPI_BYTE,
                     0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int retries = 0;
            while (!conns[i]->sendComm) {
                ASSERT_EQ(net_->connect(initCtx_, i % ndev, &handle,
                                        &conns[i]->sendComm, nullptr),
                          ncclSuccess) << "connect failed conn=" << i;
                ++retries;
                if (retries % 1000 == 0) usleep(100);
                ASSERT_LT(retries, kMaxRetries) << "connect timeout conn=" << i;
            }
            ASSERT_EQ(RegisterMemory(conns[i]->sendComm, sendBuf, bufSize,
                                     NCCL_PTR_HOST, &conns[i]->sendMhandle), ncclSuccess)
                << "regMr failed conn=" << i;
        }
        std::cout << "[Rank 1] all " << numConns << " connections up\n" << std::flush;

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        int totalCompleted = 0;
        for (int i = 0; i < numConns; i++) {
            std::vector<void*> sendRequests(qpDepth, nullptr);
            for (int d = 0; d < qpDepth; d++) {
                int sendTag = baseTag + i * qpDepth + d;
                PostSendWithRetry(conns[i]->sendComm, sendBuf, bufSize,
                                  sendTag, conns[i]->sendMhandle,
                                  &sendRequests[d]);
                ASSERT_NE(sendRequests[d], nullptr)
                    << "null send request conn=" << i << " depth=" << d;
            }
            for (int d = 0; d < qpDepth; d++) {
                int sizes[1] = {0};
                if (WaitForCompletion(sendRequests[d], sizes,
                                      kLargeTransferTimeoutMs) != ncclSuccess) {
                    std::cout << "[Rank 1] TIMEOUT conn=" << i
                              << " depth=" << d
                              << " total_completed=" << totalCompleted << "\n"
                              << std::flush;
                    goto sender_drain_done;
                }
                totalCompleted++;
            }
        }
        sender_drain_done:
        std::cout << "[Rank 1] " << totalCompleted << "/" << totalEntries
                  << " sends completed\n" << std::flush;
        EXPECT_EQ(totalCompleted, totalEntries);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

#endif // MPI_TESTS_ENABLED
