/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "NetIbMPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED

// Virtual Device Tests

TEST_F(NetIbMPITest, MakeVirtualDevice) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 devices for virtual device test";
    }

    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ncclResult_t result = MakeVirtualDevice(&vdev, &vProps);

    // Virtual device creation may or may not be supported
    if (result == ncclSuccess) {
        EXPECT_GE(vdev, 0) << "Virtual device ID should be non-negative";

        if (MPIEnvironment::world_rank == 0) {
            TEST_INFO("Created virtual device %d from physical devices 0 and 1", vdev);
        }
    }
}

TEST_F(NetIbMPITest, MakeVirtualDeviceInvalidProps) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    // Negative test: Zero devices
    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 0;

    int vdev = -1;
    ncclResult_t result = MakeVirtualDevice(&vdev, &vProps);
    EXPECT_EQ(result, ncclInvalidUsage) << "Should fail with zero devices";
}

// NIC Fusion (vNIC) Tests

TEST_F(NetIbMPITest, ConnectAndTransfer_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    // Create a fused vNIC from physical devices 0 and 1.
    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ASSERT_EQ(MakeVirtualDevice(&vdev, &vProps), ncclSuccess)
        << "Failed to create fused vNIC from devices 0 and 1";
    ASSERT_GE(vdev, 0);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(vdev, pair, connGuard);

    const size_t bufferSize = kSmallBufferSize;
    const int tag = 500;
    const int seed = 5000;

    void* buffer = malloc(bufferSize);
    ASSERT_NE(buffer, nullptr);
    auto bufferGuard = makeHostBufferAutoGuard(buffer);

    // Register the buffer on each sub-device's PD.
    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;

    if (rank == 0) {
        memset(buffer, 0, bufferSize);
        PostSingleRecv(pair.recvComm, buffer, bufferSize, tag, mhandle, &request);
    } else {
        fillHostBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(seed));
        PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    // Verify data integrity.
    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Received size mismatch";
        EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(seed))) << "Data validation failed on vNIC transfer";
    }
}

TEST_F(NetIbMPITest, AsymmetricMerge_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    // Skip when resiliency is enabled: asymmetric makeVDevice leaves stale
    // entries in the global IbCastMergedDevs table (IbCastNMergedDevs is not
    // reset by finalize), corrupting subsequent failover tests that also
    // call makeVDevice. See docs/asymmetric-merge-test-isolation-bug.md.
    const char* failoverEnv = getenv("NCCL_IB_RESILIENCY_PORT_FAILOVER");
    if (failoverEnv && strcmp(failoverEnv, "1") == 0) {
        GTEST_SKIP() << "Skipped: AsymmetricMerge_VNic corrupts global device table, "
                     << "breaking subsequent failover tests (known test isolation bug)";
    }

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    // Rank 0: create a fused vNIC (ndevs=2). Rank 1: use physical device 0 (ndevs=1).
    int vdev = -1;
    if (rank == 0) {
        ncclNetVDeviceProps_t vProps;
        vProps.ndevs = 2;
        vProps.devs[0] = 0;
        vProps.devs[1] = 1;
        ASSERT_EQ(MakeVirtualDevice(&vdev, &vProps), ncclSuccess)
            << "Failed to create fused vNIC from devices 0 and 1";
        ASSERT_GE(vdev, 0);
    }

    // Inline connection setup: each rank uses a different device ID.
    // Rank 0 (receiver): listen on vdev (ndevs=2, 2 PDs, doubled QPs)
    // Rank 1 (sender):   connect on physical dev 0 (ndevs=1, 1 PD)
    // ncclIbCalculateNqps uses max(local, remote) so both sides get the same QP count.
    ConnectionPair pair;

    if (rank == 0) {
        ASSERT_EQ(CreateListenComm(vdev, &pair.handle, &pair.listenComm), ncclSuccess);
        MPI_Send(&pair.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD);

        int done = 0;
        while (!done) {
            ncclResult_t result = AcceptConnection(pair.listenComm, &pair.recvComm);
            if (result == ncclSuccess && pair.recvComm != nullptr) {
                done = 1;
            }
        }
    } else {
        MPI_Recv(&pair.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Connect using physical device 0 (ndevs=1)
        int done = 0;
        while (!done) {
            ncclResult_t result = ConnectToRemote(0, &pair.handle, &pair.sendComm);
            if (result == ncclSuccess && pair.sendComm != nullptr) {
                done = 1;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    const size_t bufferSize = kSmallBufferSize;
    const int tag = 510;
    const int seed = 5100;

    void* buffer = malloc(bufferSize);
    ASSERT_NE(buffer, nullptr);
    auto bufferGuard = makeHostBufferAutoGuard(buffer);

    // Rank 0 registers on 2 PDs (vNIC), rank 1 registers on 1 PD (physical dev).
    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;

    if (rank == 0) {
        memset(buffer, 0, bufferSize);
        PostSingleRecv(pair.recvComm, buffer, bufferSize, tag, mhandle, &request);
    } else {
        fillHostBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(seed));
        PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    // Verify data integrity across the asymmetric connection.
    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Received size mismatch";
        EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(seed))) << "Data validation failed on asymmetric vNIC transfer";
    }
}

TEST_F(NetIbMPITest, CloseWithoutTransfer_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    const int rank = MPIEnvironment::world_rank;

    // Create a fused vNIC from physical devices 0 and 1.
    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ASSERT_EQ(MakeVirtualDevice(&vdev, &vProps), ncclSuccess)
        << "Failed to create fused vNIC from devices 0 and 1";
    ASSERT_GE(vdev, 0);

    // Phase 1: connect through the vNIC, then close immediately without any transfer.
    // Scoped block ensures RAII teardown happens before phase 2.
    {
        ConnectionPair pair;
        NetConnectionGuard connGuard(net_);
        SetupConnectionWithGuard(vdev, pair, connGuard);
        // Guard triggers closeSend/closeRecv/closeListen on a connection that
        // never had regMr, isend, or irecv called.
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 2: reconnect on the same vdev and do a transfer to verify no corruption.
    ConnectionPair pair2;
    NetConnectionGuard connGuard2(net_);
    SetupConnectionWithGuard(vdev, pair2, connGuard2);

    const size_t bufferSize = kSmallBufferSize;
    const int tag = 520;
    const int seed = 5200;

    void* buffer = malloc(bufferSize);
    ASSERT_NE(buffer, nullptr);
    auto bufferGuard = makeHostBufferAutoGuard(buffer);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair2.recvComm : pair2.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;

    if (rank == 0) {
        memset(buffer, 0, bufferSize);
        PostSingleRecv(pair2.recvComm, buffer, bufferSize, tag, mhandle, &request);
    } else {
        fillHostBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(seed));
        PostSendWithRetry(pair2.sendComm, buffer, bufferSize, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    // Verify the vNIC is still functional after the no-transfer teardown.
    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Received size mismatch";
        EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(seed))) << "Data validation failed after no-transfer teardown reconnect";
    }
}

TEST_F(NetIbMPITest, RegDeregCycling_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    const int rank = MPIEnvironment::world_rank;

    // Create a fused vNIC from physical devices 0 and 1.
    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ASSERT_EQ(MakeVirtualDevice(&vdev, &vProps), ncclSuccess)
        << "Failed to create fused vNIC from devices 0 and 1";
    ASSERT_GE(vdev, 0);

    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(vdev, pair, connGuard);

    const size_t bufferSize = kSmallBufferSize;
    void* buffer = malloc(bufferSize);
    ASSERT_NE(buffer, nullptr);
    auto bufferGuard = makeHostBufferAutoGuard(buffer);

    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    // Cycle 50× regMr→deregMr on the same buffer to stress multi-PD MR cache.
    const int kCycles = 50;
    for (int i = 0; i < kCycles; i++) {
        void* mhandle = nullptr;
        ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess)
            << "regMr failed on cycle " << i;
        ASSERT_NE(mhandle, nullptr);
        ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess)
            << "deregMr failed on cycle " << i;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Verify the connection is still functional after 50 reg/dereg cycles.
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    const int tag = 530;
    const int seed = 5300;
    void* request = nullptr;

    // Send/recv to verify the connection works after MR cache stress.
    if (rank == 0) {
        memset(buffer, 0, bufferSize);
        PostSingleRecv(pair.recvComm, buffer, bufferSize, tag, mhandle, &request);
    } else {
        fillHostBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(seed));
        PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    // Data integrity check after MR cache cycling.
    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Received size mismatch";
        EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(seed))) << "Data validation failed after reg/dereg cycling";
    }
}

TEST_F(NetIbMPITest, LargeTransfer_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    const int rank = MPIEnvironment::world_rank;

    // Create a fused vNIC from physical devices 0 and 1.
    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ASSERT_EQ(MakeVirtualDevice(&vdev, &vProps), ncclSuccess)
        << "Failed to create fused vNIC from devices 0 and 1";
    ASSERT_GE(vdev, 0);

    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(vdev, pair, connGuard);

    // 64MB buffer — ncclIbMultiSend stripes this across the doubled QPs.
    const size_t bufferSize = 64 * 1024 * 1024;
    const int tag = 540;
    const int seed = 5400;

    void* buffer = malloc(bufferSize);
    ASSERT_NE(buffer, nullptr);
    auto bufferGuard = makeHostBufferAutoGuard(buffer);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;

    if (rank == 0) {
        memset(buffer, 0, bufferSize);
        PostSingleRecv(pair.recvComm, buffer, bufferSize, tag, mhandle, &request);
    } else {
        fillHostBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(seed));
        PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Extended timeout for 16MB transfer across doubled QPs.
    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(request, sizes, kLargeTransferTimeout), ncclSuccess);

    // Full 64MB byte-by-byte verification
    // ncclIbMultiSend would corrupt data at QP split points.
    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Large transfer size mismatch";
        EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(seed))) << "Large vNIC transfer data validation failed";
    }
}

TEST_F(NetIbMPITest, MixedSizes_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    const int rank = MPIEnvironment::world_rank;

    // Create a fused vNIC from physical devices 0 and 1.
    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ASSERT_EQ(MakeVirtualDevice(&vdev, &vProps), ncclSuccess)
        << "Failed to create fused vNIC from devices 0 and 1";
    ASSERT_GE(vdev, 0);

    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(vdev, pair, connGuard);

    // Sizes: 1B, 3MB, 3B, 5MB, 7B, 7MB, 64B, 16MB, 1B, 11MB, 4MB, 1B.
    // Tiny sizes may use only one QP, large ones stripe across both.
    // Odd MB sizes (3, 5, 7, 11) produce uneven QP splits.
    std::vector<size_t> testSizes = {
        1, 3*1024*1024, 3, 5*1024*1024, 7, 7*1024*1024,
        64, 16*1024*1024, 1, 11*1024*1024, 4*1024*1024, 1
    };

    for (size_t idx = 0; idx < testSizes.size(); idx++) {
        size_t size = testSizes[idx];
        const int tag = 550;
        const int seed = 5500 + static_cast<int>(idx);

        void* buffer = malloc(size);
        ASSERT_NE(buffer, nullptr) << "malloc failed for size " << size;
        auto bufferGuard = makeHostBufferAutoGuard(buffer);

        void* mhandle = nullptr;
        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        ASSERT_EQ(RegisterMemory(comm, buffer, size, NCCL_PTR_HOST, &mhandle), ncclSuccess)
            << "regMr failed for size " << size;
        NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

        void* request = nullptr;

        if (rank == 0) {
            memset(buffer, 0, size);
            PostSingleRecv(pair.recvComm, buffer, size, tag, mhandle, &request);
        } else {
            fillHostBufferWithPattern<uint8_t>(buffer, size, makeBytePattern(seed));
            PostSendWithRetry(pair.sendComm, buffer, size, tag, mhandle, &request);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        int sizes[1] = {0};
        int timeout = (size > 1024 * 1024) ? kLargeTransferTimeout : kDefaultTimeoutMs;
        ASSERT_EQ(WaitForCompletion(request, sizes, timeout), ncclSuccess);

        // Prevent request reuse race between iterations.
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            EXPECT_EQ(sizes[0], size) << "Size mismatch for transfer of " << size << " bytes";
            EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, size, makeBytePattern(seed))) << "Data validation failed for size " << size;
        }
    }
}

TEST_F(NetIbMPITest, UnalignedSizeTransfer_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    const int rank = MPIEnvironment::world_rank;

    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ASSERT_EQ(MakeVirtualDevice(&vdev, &vProps), ncclSuccess)
        << "Failed to create fused vNIC from devices 0 and 1";
    ASSERT_GE(vdev, 0);

    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(vdev, pair, connGuard);

    // Sizes around 128-byte QP striping alignment boundaries.
    // ncclIbMultiSend computes chunkSize = DIVUP(DIVUP(size, nqps), 128) * 128.
    // These sizes produce uneven QP splits where one QP gets more data than the other.
    // 127: all on QP 0, QP 1 posts zero-sge. 129: 128B on QP 0, 1B on QP 1.
    // 255: 128B each, QP 1 gets 127B. 257: 256B on QP 0, 1B remainder on QP 1.
    std::vector<size_t> testSizes = {127, 129, 255, 257, 511, 513};

    for (size_t idx = 0; idx < testSizes.size(); idx++) {
        size_t size = testSizes[idx];
        const int tag = 560;
        const int seed = 5600 + static_cast<int>(idx);

        // Per-iteration buffer and MR registration on 2 PDs.
        void* buffer = malloc(size);
        ASSERT_NE(buffer, nullptr) << "malloc failed for size " << size;
        auto bufferGuard = makeHostBufferAutoGuard(buffer);

        void* mhandle = nullptr;
        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        ASSERT_EQ(RegisterMemory(comm, buffer, size, NCCL_PTR_HOST, &mhandle), ncclSuccess)
            << "regMr failed for size " << size;
        NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

        void* request = nullptr;

        if (rank == 0) {
            memset(buffer, 0, size);
            PostSingleRecv(pair.recvComm, buffer, size, tag, mhandle, &request);
        } else {
            fillHostBufferWithPattern<uint8_t>(buffer, size, makeBytePattern(seed));
            PostSendWithRetry(pair.sendComm, buffer, size, tag, mhandle, &request);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        int sizes[1] = {0};
        ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

        // Prevent request reuse race between iterations.
        MPI_Barrier(MPI_COMM_WORLD);

        // Byte level verification at the striping boundary.
        if (rank == 0) {
            EXPECT_EQ(sizes[0], size) << "Size mismatch for transfer of " << size << " bytes";
            EXPECT_TRUE(verifyHostBufferData<uint8_t>(buffer, size, makeBytePattern(seed))) << "Data validation failed for size " << size;
        }
    }
}

TEST_F(NetIbMPITest, Bidirectional_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    // Both ranks create a fused vNIC.
    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ASSERT_EQ(MakeVirtualDevice(&vdev, &vProps), ncclSuccess)
        << "Failed to create fused vNIC from devices 0 and 1";
    ASSERT_GE(vdev, 0);

    // Two connections through the same vNIC with reversed roles.
    // ConnA: rank 0 receives, rank 1 sends (tag 0 for MPI handle exchange).
    // ConnB: rank 1 receives, rank 0 sends (tag 1 for MPI handle exchange).
    ConnectionPair connA, connB;

    // Phase 1: Both ranks create their listener, then exchange handles via MPI.
    // Rank 0 receives on ConnA, rank 1 receives on ConnB.
    if (rank == 0) {
        ASSERT_EQ(CreateListenComm(vdev, &connA.handle, &connA.listenComm), ncclSuccess);
    } else {
        ASSERT_EQ(CreateListenComm(vdev, &connB.handle, &connB.listenComm), ncclSuccess);
    }

    // Exchange: rank 0 sends connA handle, rank 1 sends connB handle.
    if (rank == 0) {
        MPI_Send(&connA.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD);
        MPI_Recv(&connB.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(&connA.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&connB.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, 1, MPI_COMM_WORLD);
    }

    // Phase 2: Connect and accept interleaved to avoid deadlock.
    // Rank 0: connect on ConnB (sender), accept on ConnA (receiver).
    // Rank 1: connect on ConnA (sender), accept on ConnB (receiver).
    ncclNetHandle_t* connectHandle = (rank == 0) ? &connB.handle : &connA.handle;
    void** connectSendComm = (rank == 0) ? &connB.sendComm : &connA.sendComm;
    void*  myListenComm    = (rank == 0) ? connA.listenComm : connB.listenComm;
    void** acceptRecvComm  = (rank == 0) ? &connA.recvComm : &connB.recvComm;

    while (*connectSendComm == nullptr || *acceptRecvComm == nullptr) {
        if (*connectSendComm == nullptr) {
            ConnectToRemote(vdev, connectHandle, connectSendComm);
        }
        if (*acceptRecvComm == nullptr) {
            AcceptConnection(myListenComm, acceptRecvComm);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // RAII guards for both connections.
    NetConnectionGuard guardA(net_);
    NetConnectionGuard guardB(net_);
    if (rank == 0) {
        guardA.setRecvComm(connA.recvComm);
        guardA.setListenComm(connA.listenComm);
        guardB.setSendComm(connB.sendComm);
    } else {
        guardA.setSendComm(connA.sendComm);
        guardB.setRecvComm(connB.recvComm);
        guardB.setListenComm(connB.listenComm);
    }

    const size_t bufferSize = kSmallBufferSize;

    // Each rank has a send buffer and a recv buffer.
    void* sendBuf = malloc(bufferSize);
    ASSERT_NE(sendBuf, nullptr);
    auto sendGuard = makeHostBufferAutoGuard(sendBuf);

    void* recvBuf = malloc(bufferSize);
    ASSERT_NE(recvBuf, nullptr);
    auto recvGuard = makeHostBufferAutoGuard(recvBuf);
    memset(recvBuf, 0, bufferSize);

    // Register send buffer on the send comm, recv buffer on the recv comm.
    void* sendComm = (rank == 0) ? connB.sendComm : connA.sendComm;
    void* recvComm = (rank == 0) ? connA.recvComm : connB.recvComm;

    void* sendMhandle = nullptr;
    ASSERT_EQ(RegisterMemory(sendComm, sendBuf, bufferSize, NCCL_PTR_HOST, &sendMhandle), ncclSuccess);
    NetMHandleGuard sendMhGuard(sendMhandle, NetMHandleDeleter(net_, sendComm));

    void* recvMhandle = nullptr;
    ASSERT_EQ(RegisterMemory(recvComm, recvBuf, bufferSize, NCCL_PTR_HOST, &recvMhandle), ncclSuccess);
    NetMHandleGuard recvMhGuard(recvMhandle, NetMHandleDeleter(net_, recvComm));

    const int sendTag = 570;
    const int recvTag = 570;
    const int sendSeed = 5700 + rank;

    // Fill send buffer with rank-specific pattern.
    fillHostBufferWithPattern<uint8_t>(sendBuf, bufferSize, makeBytePattern(sendSeed));

    // Post recv and send simultaneously on both connections.
    void* recvRequest = nullptr;
    void* sendRequest = nullptr;

    PostSingleRecv(recvComm, recvBuf, bufferSize, recvTag, recvMhandle, &recvRequest);

    PostSendWithRetry(sendComm, sendBuf, bufferSize, sendTag, sendMhandle, &sendRequest);

    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for both completions.
    int recvSizesOut[1] = {0};
    ASSERT_EQ(WaitForCompletion(recvRequest, recvSizesOut), ncclSuccess);

    int sendSizesOut[1] = {0};
    ASSERT_EQ(WaitForCompletion(sendRequest, sendSizesOut), ncclSuccess);

    MPI_Barrier(MPI_COMM_WORLD);

    // Verify received data matches the peer's send pattern.
    int peerSeed = 5700 + peerRank;
    EXPECT_EQ(recvSizesOut[0], bufferSize) << "Received size mismatch";
    EXPECT_TRUE(verifyHostBufferData<uint8_t>(recvBuf, bufferSize, makeBytePattern(peerSeed))) << "Bidirectional vNIC transfer data validation failed";
}

TEST_F(NetIbMPITest, FlushRepeated_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    // Check GDR support on device 0 before creating the vNIC.
    ncclNetProperties_t props;
    ASSERT_EQ(GetDeviceProperties(0, &props), ncclSuccess);
    if (!(props.ptrSupport & NCCL_PTR_CUDA)) {
        GTEST_SKIP() << "GDR not supported, skipping flush test";
    }

    // Create a fused vNIC from physical devices 0 and 1.
    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ASSERT_EQ(MakeVirtualDevice(&vdev, &vProps), ncclSuccess);
    ASSERT_GE(vdev, 0);

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(vdev, pair, connGuard);

    const size_t bufferSize = kSmallBufferSize;
    const int tag = 600;
    const int numIterations = 50;

    // Allocate GPU buffer and register with NCCL_PTR_CUDA on the vNIC connection.
    void* gpuBuffer = nullptr;
    HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&gpuBuffer, bufferSize));
    auto gpuGuard = makeDeviceBufferAutoGuard(gpuBuffer);

    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, gpuBuffer, bufferSize, NCCL_PTR_CUDA, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    for (int iter = 0; iter < numIterations; iter++) {
        const int seed = 6000 + iter;
        void* request = nullptr;

        if (rank == 1) {
            // Fill GPU buffer via DeviceBufferHelpers (host vector → hipMemcpy in one call).
            ASSERT_EQ(initializeBufferWithPattern<uint8_t>(gpuBuffer, bufferSize, makeBytePattern(seed)), hipSuccess);
            PostSendWithRetry(pair.sendComm, gpuBuffer, bufferSize, tag, mhandle, &request);
        } else {
            // Rank 0: post recv on GPU buffer.
            PostSingleRecv(pair.recvComm, gpuBuffer, bufferSize, tag, mhandle, &request);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        int sizes[1] = {0};
        ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

        if (rank == 0) {
            ASSERT_EQ(sizes[0], bufferSize) << "Iter " << iter << ": received size mismatch";

            // Flush GPU memory to ensure RDMA write is visible on GPU.
            void* flushBuffers[1] = {gpuBuffer};
            int flushSizes[1] = {static_cast<int>(bufferSize)};
            void* flushHandles[1] = {mhandle};
            void* flushRequest = nullptr;

            ncclResult_t flushResult = FlushRecv(pair.recvComm, 1, flushBuffers, flushSizes,
                                                 flushHandles, &flushRequest);
            if (flushResult == ncclSuccess && flushRequest != nullptr) {
                ASSERT_EQ(WaitForCompletion(flushRequest, nullptr), ncclSuccess);
            }

            // Verify GPU buffer via DeviceBufferHelpers (hipMemcpy + compare in one call).
            ASSERT_TRUE(verifyBufferData<uint8_t>(gpuBuffer, bufferSize, makeBytePattern(seed)))
                << "Iter " << iter << ": data verification failed after flush";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

TEST_F(NetIbMPITest, SequentialTransfers_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    // Create a fused vNIC from physical devices 0 and 1.
    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ASSERT_EQ(MakeVirtualDevice(&vdev, &vProps), ncclSuccess);
    ASSERT_GE(vdev, 0);

    const int rank = MPIEnvironment::world_rank;

    // Single connection through the vNIC, reused across all 100 iterations.
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    SetupConnectionWithGuard(vdev, pair, connGuard);

    const size_t bufferSize = kSmallBufferSize;
    const int tag = 700;
    const int numIterations = 100;

    // Single buffer registered once on both sub-device PDs, reused every iteration.
    void* buffer = malloc(bufferSize);
    ASSERT_NE(buffer, nullptr);
    auto bufferGuard = makeHostBufferAutoGuard(buffer);

    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    for (int iter = 0; iter < numIterations; iter++) {
        // Unique seed per iteration to detect stale data from prior rounds.
        const int seed = 7000 + iter;
        void* request = nullptr;

        if (rank == 1) {
            // Fill buffer with per-iteration pattern.
            fillHostBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(seed));
            PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
        } else {
            // Zero buffer before recv to ensure verification catches stale data.
            memset(buffer, 0, bufferSize);
            PostSingleRecv(pair.recvComm, buffer, bufferSize, tag, mhandle, &request);
        }

        // Sync both ranks before polling for completion.
        MPI_Barrier(MPI_COMM_WORLD);

        int sizes[1] = {0};
        ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

        // Verify received data matches the iteration-specific pattern.
        if (rank == 0) {
            ASSERT_EQ(sizes[0], bufferSize) << "Iter " << iter << ": received size mismatch";
            ASSERT_TRUE(verifyHostBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(seed))) << "Iter " << iter << ": data verification failed";
        }

        // Sync before next iteration to prevent request reuse races.
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// =============================================================================
// Test: SendRecvDifferentMemoryTypes
//
// Creates a 2-NIC merged virtual device per rank (CreateMergedDevice(2, rank)).
// Rank 0 (receiver) and Rank 1 (sender) each use their own merged device.
// Tests all four memory type combinations: host→host, host→GPU, GPU→host, GPU→GPU.
//
// This exercises:
//   - Asymmetric merged device usage (different vNIC per rank)
//   - Multi-QP data path with 2 QPs per rank
//   - GDR path selection based on pointer type
//   - Memory registration across multiple Protection Domains
//   - Flush semantics for GPU memory receives
// =============================================================================
TEST_F(NetIbMPITest, SendRecvDifferentMemoryTypes) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int mergedDev = CreateMergedDevice(2, rank);
    if (mergedDev == -1) {
        GTEST_SKIP() << "Failed to create merged device";
    }

    ncclNetProperties_t mProps;
    memset(&mProps, 0, sizeof(mProps));
    ASSERT_EQ(GetDeviceProperties(mergedDev, &mProps), ncclSuccess);

    // Check GDR support across both ranks
    int localGdr = (mProps.ptrSupport & NCCL_PTR_CUDA) ? 1 : 0;
    int globalGdr = 0;
    MPI_Allreduce(&localGdr, &globalGdr, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    bool gdr = (globalGdr == 1);

    struct Combo {
        int send;
        int recv;
        const char* desc;
    };
    Combo combos[] = {
        {NCCL_PTR_HOST, NCCL_PTR_HOST, "Host->Host"},
        {NCCL_PTR_HOST, NCCL_PTR_CUDA, "Host->GPU"},
        {NCCL_PTR_CUDA, NCCL_PTR_HOST, "GPU->Host"},
        {NCCL_PTR_CUDA, NCCL_PTR_CUDA, "GPU->GPU"},
    };

    for (auto& c : combos) {
        bool needGpu = (c.send == NCCL_PTR_CUDA || c.recv == NCCL_PTR_CUDA);
        if (needGpu && !gdr) {
            if (rank == 0) fprintf(stderr, "  [SKIP] %s (no GDR)\n", c.desc);
            continue;
        }

        int memType = (rank == 0) ? c.recv : c.send;

        ConnectionPair pair;
        ASSERT_EQ(SetupConnection(mergedDev, pair, rank, peerRank), ncclSuccess);

        NetConnectionGuard conn(net_);
        if (rank == 0) {
            conn.setRecvComm(pair.recvComm);
            conn.setListenComm(pair.listenComm);
        } else {
            conn.setSendComm(pair.sendComm);
        }

        const size_t sz = kSmallBufferSize;
        void* buf = nullptr;
        RCCLTestGuards::DeviceBufferAutoGuard deviceBufGuard;
        RCCLTestGuards::HostBufferAutoGuard hostBufGuard;
        if (memType == NCCL_PTR_CUDA) {
            HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buf, sz));
            deviceBufGuard.set(buf);
        } else {
            buf = malloc(sz);
            ASSERT_NE(buf, nullptr);
            hostBufGuard.set(buf);
        }

        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        void* mh = nullptr;
        ASSERT_EQ(RegisterMemory(comm, buf, sz, memType, &mh), ncclSuccess);
        NetMHandleGuard mhG(mh, NetMHandleDeleter(net_, comm));

        uint8_t seed = static_cast<uint8_t>(c.send * 10 + c.recv * 3 + 42);
        int tag = 700 + c.send * 2 + c.recv;

        if (rank == 1) {
            if (memType == NCCL_PTR_CUDA) {
                ASSERT_EQ(initializeBufferWithPattern<uint8_t>(buf, sz, makeBytePattern(seed)), hipSuccess);
            } else {
                fillHostBufferWithPattern<uint8_t>(buf, sz, makeBytePattern(seed));
            }
        } else {
            if (memType == NCCL_PTR_CUDA) {
                HIP_TEST_CHECK_GTEST_FAIL(hipMemset(buf, 0, sz));
            } else {
                memset(buf, 0, sz);
            }
        }

        void* req = nullptr;
        if (rank == 0) {
            void* b[1] = {buf};
            size_t s[1] = {sz};
            int t[1] = {tag};
            void* h[1] = {mh};
            ASSERT_EQ(PostRecv(pair.recvComm, 1, b, s, t, h, &req), ncclSuccess);
            ASSERT_NE(req, nullptr);
        } else {
            PostSendWithRetry(pair.sendComm, buf, sz, tag, mh, &req);
        }

        int csz[1] = {0};
        ASSERT_EQ(WaitForCompletion(req, csz), ncclSuccess);

        if (rank == 0) {
            EXPECT_EQ(csz[0], static_cast<int>(sz));

            if (memType == NCCL_PTR_CUDA) {
                void* fb[1] = {buf};
                int fs[1] = {static_cast<int>(sz)};
                void* fh[1] = {mh};
                void* fr = nullptr;
                if (FlushRecv(pair.recvComm, 1, fb, fs, fh, &fr) == ncclSuccess && fr) {
                    ASSERT_EQ(WaitForCompletion(fr, nullptr), ncclSuccess);
                }
            }

            if (memType == NCCL_PTR_CUDA) {
                EXPECT_TRUE(verifyBufferData<uint8_t>(buf, sz, makeBytePattern(seed))) << "Data mismatch for " << c.desc;
            } else {
                EXPECT_TRUE(verifyHostBufferData<uint8_t>(buf, sz, makeBytePattern(seed))) << "Data mismatch for " << c.desc;
            }
        }
    }
}

TEST_F(NetIbMPITest, SendRecvMultipleSizesFusion) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly kExactTwoProcesses processes";

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int mergedDev = CreateMergedDevice(3, rank);
    if (mergedDev == -1) {
        GTEST_SKIP() << "Failed to create merged device";
    }

    // Build test size list
    long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) pageSize = 4096;

    std::vector<size_t> testSizes = {
        // Tiny — size < nqps (3 QPs → some get 0-byte chunks)
        1,
        // Non-power-of-two — uneven 3-way splits
        3, 7, 1023, 4097, 65537,
        // Page boundary
        (size_t)pageSize,
        // Powers of two: 2 bytes to 64 MB
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
        2048, 4096, 8192, 16384, 32768, 65536,
        128 * 1024, 256 * 1024, 512 * 1024,
        1024 * 1024,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        8 * 1024 * 1024,
        16 * 1024 * 1024,
        32 * 1024 * 1024,
        64 * 1024 * 1024,
    };

    // Sort and deduplicate (pageSize might duplicate 4096)
    std::sort(testSizes.begin(), testSizes.end());
    testSizes.erase(std::unique(testSizes.begin(), testSizes.end()), testSizes.end());

    size_t maxSize = testSizes.back();

    // Allocate max-size buffer once, register once
    void* buffer = malloc(maxSize);
    ASSERT_NE(buffer, nullptr);
    auto bufferGuard = makeHostBufferAutoGuard(buffer);

    // Setup connection through merged device
    ConnectionPair pair;
    ncclNetHandle_t handle;
    if (rank == 0) {
        ASSERT_EQ(CreateListenComm(mergedDev, &handle, &pair.listenComm), ncclSuccess);
        MPI_Send(&handle, sizeof(handle), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD);
        while (!pair.recvComm) AcceptConnection(pair.listenComm, &pair.recvComm);
    } else {
        MPI_Recv(&handle, sizeof(handle), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        while (!pair.sendComm) ConnectToRemote(mergedDev, &handle, &pair.sendComm);
    }

    NetConnectionGuard conn(net_);
    if (rank == 0) { conn.setRecvComm(pair.recvComm); conn.setListenComm(pair.listenComm); }
    else conn.setSendComm(pair.sendComm);

    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mhandle = nullptr;
    ASSERT_EQ(RegisterMemory(comm, buffer, maxSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhGuard(mhandle, NetMHandleDeleter(net_, comm));

    for (size_t idx = 0; idx < testSizes.size(); idx++) {
        size_t sz = testSizes[idx];
        int tag = 800 + (int)idx;
        int seed = 9000 + (int)idx;

        if (rank == 1) {
            fillHostBufferWithPattern<uint8_t>(buffer, sz, makeBytePattern(seed));
        } else {
            memset(buffer, 0xDE, sz);
        }

        // Transfer
        void* req = nullptr;
        if (rank == 0) {
            void* b[1] = {buffer}; size_t s[1] = {sz}; int t[1] = {tag}; void* h[1] = {mhandle};
            ASSERT_EQ(PostRecv(pair.recvComm, 1, b, s, t, h, &req), ncclSuccess);
            ASSERT_NE(req, nullptr) << "Recv request NULL for size " << sz;
        } else {
            PostSendWithRetry(pair.sendComm, buffer, sz, tag, mhandle, &req);
        }

        int csz[1] = {0};
        int timeout = (sz > 1024 * 1024) ? kLargeTransferTimeoutMs : kDefaultTimeoutMs;
        ASSERT_EQ(WaitForCompletion(req, csz, timeout), ncclSuccess)
            << "Transfer timeout for size " << sz;

        // Verify on receiver
        if (rank == 0) {
            bool sizeOk = (csz[0] == (int)sz);
            bool dataOk = false;
            size_t errIdx = 0;
            uint8_t errExp = 0, errGot = 0;

            if (sizeOk) {
                dataOk = verifyHostBufferData<uint8_t>(buffer, sz,
                    makeBytePattern(seed), 0, 0.0, &errIdx, &errExp, &errGot);
            }

            bool ok = sizeOk && dataOk;
            if (!ok) {
                if (!sizeOk) {
                    fprintf(stderr, "  [FAIL] size=%10zu  size mismatch: got %d\n", sz, csz[0]);
                } else {
                    fprintf(stderr, "  [FAIL] size=%10zu  data error at byte %zu: "
                            "expected %u, got %u\n", sz, errIdx, errExp, errGot);
                }
            }
            fflush(stderr);

            EXPECT_TRUE(ok) << "Failed for size " << sz;
        }
    }
}

TEST_F(NetIbMPITest, MultidirectionalTransfer) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int mergedSendDev = CreateMergedDevice(4, rank);
    if (mergedSendDev == -1) {
        GTEST_SKIP() << "Failed to create merged device";
    }
    int mergedRecvDev = mergedSendDev;

    // Setup bidirectional connections
    void* sendCommFwd = nullptr, *recvCommFwd = nullptr, *listenCommFwd = nullptr;
    void* sendCommBwd = nullptr, *recvCommBwd = nullptr, *listenCommBwd = nullptr;

    // Forward: Rank 1 listens, Rank 0 connects
    {
        ncclNetHandle_t h;
        if (rank == 1) {
            ASSERT_EQ(CreateListenComm(mergedRecvDev, &h, &listenCommFwd), ncclSuccess);
            MPI_Send(&h, sizeof(h), MPI_BYTE, peerRank, 80, MPI_COMM_WORLD);
            while (!recvCommFwd) AcceptConnection(listenCommFwd, &recvCommFwd);
        } else {
            MPI_Recv(&h, sizeof(h), MPI_BYTE, peerRank, 80, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            while (!sendCommFwd) ConnectToRemote(mergedSendDev, &h, &sendCommFwd);
        }
    }

    // Backward: Rank 0 listens, Rank 1 connects
    {
        ncclNetHandle_t h;
        if (rank == 0) {
            ASSERT_EQ(CreateListenComm(mergedRecvDev, &h, &listenCommBwd), ncclSuccess);
            MPI_Send(&h, sizeof(h), MPI_BYTE, peerRank, 81, MPI_COMM_WORLD);
            while (!recvCommBwd) AcceptConnection(listenCommBwd, &recvCommBwd);
        } else {
            MPI_Recv(&h, sizeof(h), MPI_BYTE, peerRank, 81, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            while (!sendCommBwd) ConnectToRemote(mergedSendDev, &h, &sendCommBwd);
        }
    }

    void* mySendComm = (rank == 0) ? sendCommFwd : sendCommBwd;
    void* myRecvComm = (rank == 0) ? recvCommBwd : recvCommFwd;

    // Patterns
    struct Pattern {
        const char* name;
        size_t sendSize, recvSize;
        int sendSeed, recvSeed;
        int tag;
    };

    // to be extended for running multirank
    std::vector<Pattern> patterns = {
        {"Allgather (4KB)", 4096, 4096,
         (rank == 0) ? 1000 : 2000, (rank == 0) ? 2000 : 1000, 300},
        {"Alltoall (8KB)", 8192, 8192,
         (rank == 0) ? 3000 : 4000, (rank == 0) ? 4000 : 3000, 301},
        {"Hypercube (4KB/16KB)", (size_t)((rank == 0) ? 4096 : 16384),
         (size_t)((rank == 0) ? 16384 : 4096),
         (rank == 0) ? 5000 : 6000, (rank == 0) ? 6000 : 5000, 302},
    };

    for (size_t pi = 0; pi < patterns.size(); pi++) {
        const Pattern& pat = patterns[pi];

        void* sendBuf = malloc(pat.sendSize);
        ASSERT_NE(sendBuf, nullptr);
        auto sendBufGuard = RCCLTestGuards::makeHostBufferAutoGuard(sendBuf);
        void* sendMr = nullptr;
        ASSERT_EQ(RegisterMemory(mySendComm, sendBuf, pat.sendSize,
                                 NCCL_PTR_HOST, &sendMr), ncclSuccess);

        void* recvBuf = malloc(pat.recvSize);
        ASSERT_NE(recvBuf, nullptr);
        auto recvBufGuard = RCCLTestGuards::makeHostBufferAutoGuard(recvBuf);
        void* recvMr = nullptr;
        ASSERT_EQ(RegisterMemory(myRecvComm, recvBuf, pat.recvSize,
                                 NCCL_PTR_HOST, &recvMr), ncclSuccess);

        fillHostBufferWithPattern<uint8_t>(sendBuf, pat.sendSize, makeBytePattern(pat.sendSeed));
        memset(recvBuf, 0xDE, pat.recvSize);

        // BOTH ranks post recv
        void* recvReq = nullptr;
        {
            void* rb[1] = {recvBuf}; size_t rs[1] = {pat.recvSize};
            int rt[1] = {pat.tag}; void* rh[1] = {recvMr};
            ASSERT_EQ(PostRecv(myRecvComm, 1, rb, rs, rt, rh, &recvReq), ncclSuccess);
            ASSERT_NE(recvReq, nullptr);
        }

        // BOTH ranks post send with retry
        {
            void* sendReq = nullptr;
            PostSendWithRetry(mySendComm, sendBuf, pat.sendSize, pat.tag, sendMr, &sendReq);
        }

        // Wait for recv
        {
            int csz[1] = {0};
            ASSERT_EQ(WaitForCompletion(recvReq, csz, kLargeTransferTimeoutMs), ncclSuccess)
                << pat.name << " recv timed out on rank " << rank;
            EXPECT_EQ(csz[0], (int)pat.recvSize)
                << pat.name << " size mismatch on rank " << rank;
        }

        // Verify
        {
            size_t errIdx = 0;
            uint8_t errExp = 0, errGot = 0;
            bool ok = verifyHostBufferData<uint8_t>(recvBuf, pat.recvSize,
                makeBytePattern(pat.recvSeed), 0, 0.0, &errIdx, &errExp, &errGot);
            EXPECT_TRUE(ok) << pat.name << " rank " << rank
                            << " data mismatch at byte " << errIdx
                            << ": expected " << (int)errExp << " got " << (int)errGot;
        }

        ASSERT_EQ(DeregisterMemory(mySendComm, sendMr), ncclSuccess);
        ASSERT_EQ(DeregisterMemory(myRecvComm, recvMr), ncclSuccess);
    }

    // Close
    if (rank == 0) {
        ASSERT_EQ(CloseSendComm(sendCommFwd), ncclSuccess);
        ASSERT_EQ(CloseRecvComm(recvCommBwd), ncclSuccess);
        ASSERT_EQ(CloseListenComm(listenCommBwd), ncclSuccess);
    } else {
        ASSERT_EQ(CloseRecvComm(recvCommFwd), ncclSuccess);
        ASSERT_EQ(CloseSendComm(sendCommBwd), ncclSuccess);
        ASSERT_EQ(CloseListenComm(listenCommFwd), ncclSuccess);
    }
}

TEST_F(NetIbMPITest, MultipleOutstandingSendRecv) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    // --- Init and discover devices ---
    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int mergedDev = CreateMergedDevice(2, rank);
    if (mergedDev == -1) {
        GTEST_SKIP() << "Failed to create merged device";
    }

    // --- Parameters ---
    static constexpr int kNumOutstanding = 8;
    static constexpr int kTagBase = 900;
    static constexpr int kSendDelayUs = 50000;  // 50ms delay before sender posts

    // --- Setup connection ---
    ConnectionPair pair;
    ncclNetHandle_t handle;
    if (rank == 0) {
        ASSERT_EQ(CreateListenComm(mergedDev, &handle, &pair.listenComm), ncclSuccess);
        MPI_Send(&handle, sizeof(handle), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD);
        while (!pair.recvComm) AcceptConnection(pair.listenComm, &pair.recvComm);
    } else {
        MPI_Recv(&handle, sizeof(handle), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        while (!pair.sendComm) ConnectToRemote(mergedDev, &handle, &pair.sendComm);
    }

    NetConnectionGuard conn(net_);
    if (rank == 0) { conn.setRecvComm(pair.recvComm); conn.setListenComm(pair.listenComm); }
    else conn.setSendComm(pair.sendComm);

    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    // --- Allocate per-operation buffers and register MRs ---
    std::vector<RCCLTestGuards::HostBufferAutoGuard> bufGuards;
    std::vector<void*> bufs(kNumOutstanding, nullptr);
    std::vector<void*> mrs(kNumOutstanding, nullptr);
    for (int i = 0; i < kNumOutstanding; i++) {
        bufs[i] = malloc(kLargeBufferSize);
        ASSERT_NE(bufs[i], nullptr);
        bufGuards.push_back(RCCLTestGuards::makeHostBufferAutoGuard(bufs[i]));
        ASSERT_EQ(RegisterMemory(comm, bufs[i], kLargeBufferSize, NCCL_PTR_HOST, &mrs[i]),
                  ncclSuccess);
    }

    // --- GPU resources for async work on sender ---
    RCCLTestGuards::DeviceBufferAutoGuard gpuAGuard;
    RCCLTestGuards::DeviceBufferAutoGuard gpuBGuard;
    RCCLTestGuards::HipStreamAutoGuard gpuStreamGuard;
    void* gpuA = nullptr;
    void* gpuB = nullptr;
    hipStream_t gpuStream = nullptr;
    if (rank == 1) {
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&gpuA, kLargeBufferSize));
        gpuAGuard.set(gpuA);
        HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&gpuB, kLargeBufferSize));
        gpuBGuard.set(gpuB);
        HIP_TEST_CHECK_GTEST_FAIL(hipStreamCreate(&gpuStream));
        gpuStreamGuard.set(gpuStream);
        HIP_TEST_CHECK_GTEST_FAIL(hipMemset(gpuA, 0xAA, kLargeBufferSize));
        HIP_TEST_CHECK_GTEST_FAIL(hipMemset(gpuB, 0x00, kLargeBufferSize));
    }

    // =========================================================================
    // Timeline:
    //
    //   Receiver (rank 0):
    //     Phase 1: PostRecv 0..7 (all recvs posted immediately)
    //     (idle — waiting for data)
    //     Phase 3: WaitForCompletion 0..7, verify data
    //
    //   Sender (rank 1):
    //     (idle)
    //     Phase 2: sleep 50ms, then fill bufs + PostSend 0..7,
    //              then IMMEDIATELY overwrite all bufs with 0xFF + launch GPU work
    //     Phase 3: verify GPU work
    //
    // What we're testing:
    //   After PostSend, sender overwrites the send buffers with garbage (0xFF).
    //   If RDMA has already read the original data before the overwrite,
    //   receiver sees correct data. If RDMA reads after overwrite → corruption.
    // =========================================================================

    std::vector<void*> recvReqs(kNumOutstanding, nullptr);

    // ===================== Phase 1: Receiver posts ALL recvs =====================
    if (rank == 0) {
        for (int i = 0; i < kNumOutstanding; i++) {
            memset(bufs[i], 0xDE, kLargeBufferSize);  // Poison
            void* rb[1] = {bufs[i]};
            size_t rs[1] = {kLargeBufferSize};
            int rt[1] = {kTagBase + i};
            void* rh[1] = {mrs[i]};
            ASSERT_EQ(PostRecv(pair.recvComm, 1, rb, rs, rt, rh, &recvReqs[i]),
                      ncclSuccess);
            ASSERT_NE(recvReqs[i], nullptr) << "Recv request NULL for op " << i;
        }
    }

    // ===================== Phase 2: Sender sleeps, sends, then corrupts =====================
    if (rank == 1) {
        // Deliberate delay
        usleep(kSendDelayUs);

        // Fill each buffer with a unique pattern and post send
        for (int i = 0; i < kNumOutstanding; i++) {
            int seed = kTagBase + i;
            fillHostBufferWithPattern<uint8_t>(bufs[i], kLargeBufferSize, makeBytePattern(seed));

            void* sendReq = nullptr;
            ASSERT_EQ(PostSend(pair.sendComm, bufs[i], kLargeBufferSize,
                               kTagBase + i, mrs[i], &sendReq), ncclSuccess)
                << "PostSend failed for op " << i;
            // Don't care about sendReq — we verify via recv completions
        }

        // IMMEDIATELY overwrite all send buffers with garbage.
        // This is the critical test: if RDMA hasn't finished reading the
        // buffer yet, receiver will see 0xFF instead of the pattern.
        for (int i = 0; i < kNumOutstanding; i++) {
            memset(bufs[i], 0xFF, kLargeBufferSize);
        }

        // Also launch async GPU work — overlaps with RDMA completion
        for (int i = 0; i < 10; i++) {
            HIP_TEST_CHECK_GTEST_FAIL(
                hipMemcpyAsync(gpuB, gpuA, kLargeBufferSize,
                               hipMemcpyDeviceToDevice, gpuStream));
            HIP_TEST_CHECK_GTEST_FAIL(
                hipMemsetAsync(gpuA, 0xBB + i, kLargeBufferSize, gpuStream));
        }
    }

    // ===================== Phase 3: Verify =====================

    // Receiver: wait for all recv completions and verify data
    if (rank == 0) {
        for (int i = 0; i < kNumOutstanding; i++) {
            int csz[1] = {0};
            ASSERT_EQ(WaitForCompletion(recvReqs[i], csz, kLargeTransferTimeoutMs),
                      ncclSuccess)
                << "Recv " << i << " timed out";
            EXPECT_EQ(csz[0], (int)kLargeBufferSize) << "Size mismatch recv op " << i;
        }

        // Verify data integrity — should be ORIGINAL pattern, not 0xFF
        int ok = 0;
        for (int i = 0; i < kNumOutstanding; i++) {
            int seed = kTagBase + i;
            size_t errIdx = 0;
            uint8_t errExp = 0, errGot = 0;
            bool good = verifyHostBufferData<uint8_t>(bufs[i], kLargeBufferSize,
                makeBytePattern(seed), 0, 0.0, &errIdx, &errExp, &errGot);

            if (good) {
                ok++;
            } else {
                bool sawFF = (errGot == 0xFF);
                fprintf(stderr, "  [FAIL] op %d byte %zu: expected %u got %u%s\n",
                        i, errIdx, errExp, errGot,
                        sawFF ? " (0xFF = sender overwrote before RDMA read!)" : "");
            }
        }
        EXPECT_EQ(ok, kNumOutstanding);
    }

    // Sender: verify GPU work
    if (rank == 1) {
        hipError_t syncErr = hipStreamSynchronize(gpuStream);
        EXPECT_EQ(syncErr, hipSuccess) << "hipStreamSynchronize failed: "
                                       << hipGetErrorString(syncErr);

        std::vector<uint8_t> h(kLargeBufferSize);

        // After 10 iterations: gpuA = 0xC4, gpuB = 0xC3
        {
            hipError_t err = hipMemcpy(h.data(), gpuA, kLargeBufferSize, hipMemcpyDeviceToHost);
            EXPECT_EQ(err, hipSuccess) << "hipMemcpy gpuA failed: " << hipGetErrorString(err);
            bool aOk = std::all_of(h.begin(), h.end(),
                                   [](uint8_t v){ return v == 0xC4; });
            EXPECT_TRUE(aOk);
        }
        {
            hipError_t err = hipMemcpy(h.data(), gpuB, kLargeBufferSize, hipMemcpyDeviceToHost);
            EXPECT_EQ(err, hipSuccess) << "hipMemcpy gpuB failed: " << hipGetErrorString(err);
            bool bOk = std::all_of(h.begin(), h.end(),
                                   [](uint8_t v){ return v == 0xC3; });
            EXPECT_TRUE(bOk);
        }
    }

    // ===================== Cleanup =====================
    // MR deregistration must happen before bufGuards free the underlying memory.
    for (int i = 0; i < kNumOutstanding; i++) {
        if (mrs[i]) DeregisterMemory(comm, mrs[i]);
    }
    // bufGuards, gpuAGuard, gpuBGuard, gpuStreamGuard clean up automatically.
}

// =============================================================================
// Test: MergeMultipleDevices
//
// Tests makeVDevice() with increasing numbers of physical devices:
//   - 3 devices: should succeed (within NCCL_NET_MAX_DEVS_PER_NIC = 4 limit)
//   - 4 devices: should succeed (at the limit)
//   - 5 devices: should fail (exceeds the limit)
//
// makeVDevice() is additive — each successful call appends a new merged device
// to the device list. Physical devices remain visible (hiding is done by the
// topology layer's XML manipulation, not by makeVDevice).
//
// =============================================================================
TEST_F(NetIbMPITest, MergeMultipleDevices) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    // --- Collect all device properties and identify physical (non-merged) devices ---
    std::vector<ncclNetProperties_t> allProps(ndev);
    std::vector<int> physicalDevices;

    for (int i = 0; i < ndev; i++) {
        memset(&allProps[i], 0, sizeof(ncclNetProperties_t));
        ASSERT_EQ(GetDeviceProperties(i, &allProps[i]), ncclSuccess);

        bool isMerged = (allProps[i].name && strchr(allProps[i].name, '+') != nullptr);
        if (!isMerged) {
            physicalDevices.push_back(i);
        }
    }

    if (physicalDevices.size() < 5) {
        GTEST_SKIP() << "Need at least 5 physical (non-merged) IB devices, found "
                     << physicalDevices.size();
    }

    // --- Find 5 physical devices with matching speed ---
    int targetSpeed = allProps[physicalDevices[0]].speed;
    std::vector<int> selected;

    for (size_t i = 0; i < physicalDevices.size() && selected.size() < 5; i++) {
        int devIdx = physicalDevices[i];
        if (allProps[devIdx].speed == targetSpeed) {
            selected.push_back(devIdx);
        }
    }

    if (selected.size() < 5) {
        GTEST_SKIP() << "Could not find 5 physical devices with matching speed. "
                     << "Found " << selected.size() << " devices at speed " << targetSpeed;
    }

    // =========================================================================
    // Sub-test 1: Merge 3 devices (should succeed, under limit of 4)
    // =========================================================================
    {
        int ndevBefore = 0;
        ASSERT_EQ(GetDeviceCount(&ndevBefore), ncclSuccess);

        ncclNetVDeviceProps_t vProps;
        memset(&vProps, 0, sizeof(vProps));
        vProps.ndevs = 3;
        vProps.devs[0] = selected[0];
        vProps.devs[1] = selected[1];
        vProps.devs[2] = selected[2];

        int vdev = -1;
        ncclResult_t result = MakeVirtualDevice(&vdev, &vProps);

        EXPECT_EQ(result, ncclSuccess)
            << "Merging 3 devices should succeed (within limit of 4)";

        if (result == ncclSuccess) {
            EXPECT_GE(vdev, 0) << "Virtual device index should be non-negative";

            // Device count should increase by 1
            int ndevAfter = 0;
            ASSERT_EQ(GetDeviceCount(&ndevAfter), ncclSuccess);
            EXPECT_EQ(ndevAfter, ndevBefore + 1)
                << "makeVDevice should add exactly one device";

            // Verify the new merged device properties
            ncclNetProperties_t vdevProps;
            memset(&vdevProps, 0, sizeof(vdevProps));
            ASSERT_EQ(GetDeviceProperties(vdev, &vdevProps), ncclSuccess);

            // Name should contain two '+' separators (3 devices)
            ASSERT_NE(vdevProps.name, nullptr);
            std::string mergedName(vdevProps.name);
            int plusCount = 0;
            for (char c : mergedName) {
                if (c == '+') plusCount++;
            }
            EXPECT_EQ(plusCount, 2)
                << "3-device merge should have 2 '+' separators, got " << plusCount
                << " in name '" << mergedName << "'";

            // Speed should be sum of 3 constituents
            int expectedSpeed = allProps[selected[0]].speed +
                                allProps[selected[1]].speed +
                                allProps[selected[2]].speed;
            EXPECT_EQ(vdevProps.speed, expectedSpeed)
                << "Merged speed should be " << expectedSpeed << ", got " << vdevProps.speed;
        }
    }

    // =========================================================================
    // Sub-test 2: Merge 4 devices (should succeed, at the limit)
    // =========================================================================
    {
        int ndevBefore = 0;
        ASSERT_EQ(GetDeviceCount(&ndevBefore), ncclSuccess);

        ncclNetVDeviceProps_t vProps;
        memset(&vProps, 0, sizeof(vProps));
        vProps.ndevs = 4;
        vProps.devs[0] = selected[0];
        vProps.devs[1] = selected[1];
        vProps.devs[2] = selected[2];
        vProps.devs[3] = selected[3];

        int vdev = -1;
        ncclResult_t result = MakeVirtualDevice(&vdev, &vProps);

        EXPECT_EQ(result, ncclSuccess)
            << "Merging 4 devices should succeed (at the limit of NCCL_NET_MAX_DEVS_PER_NIC)";

        if (result == ncclSuccess) {
            EXPECT_GE(vdev, 0);

            int ndevAfter = 0;
            ASSERT_EQ(GetDeviceCount(&ndevAfter), ncclSuccess);
            EXPECT_EQ(ndevAfter, ndevBefore + 1);

            ncclNetProperties_t vdevProps;
            memset(&vdevProps, 0, sizeof(vdevProps));
            ASSERT_EQ(GetDeviceProperties(vdev, &vdevProps), ncclSuccess);

            // Name should contain three '+' separators (4 devices)
            ASSERT_NE(vdevProps.name, nullptr);
            std::string mergedName(vdevProps.name);
            int plusCount = 0;
            for (char c : mergedName) {
                if (c == '+') plusCount++;
            }
            EXPECT_EQ(plusCount, 3)
                << "4-device merge should have 3 '+' separators, got " << plusCount
                << " in name '" << mergedName << "'";

            // Speed should be sum of 4 constituents
            int expectedSpeed = allProps[selected[0]].speed +
                                allProps[selected[1]].speed +
                                allProps[selected[2]].speed +
                                allProps[selected[3]].speed;
            EXPECT_EQ(vdevProps.speed, expectedSpeed)
                << "Merged speed should be " << expectedSpeed << ", got " << vdevProps.speed;
        }
    }

    // =========================================================================
    // Sub-test 3: Merge 5 devices (should FAIL, exceeds limit of 4)
    // =========================================================================
    {
        int ndevBefore = 0;
        ASSERT_EQ(GetDeviceCount(&ndevBefore), ncclSuccess);

        ncclNetVDeviceProps_t vProps;
        memset(&vProps, 0, sizeof(vProps));
        vProps.ndevs = 5;
        vProps.devs[0] = selected[0];
        vProps.devs[1] = selected[1];
        vProps.devs[2] = selected[2];
        vProps.devs[3] = selected[3];

        int vdev = -1;
        ncclResult_t result = MakeVirtualDevice(&vdev, &vProps);

        EXPECT_NE(result, ncclSuccess)
            << "Merging 5 devices should fail (exceeds NCCL_NET_MAX_DEVS_PER_NIC = 4)";

        // Device count should NOT have changed
        int ndevAfter = 0;
        ASSERT_EQ(GetDeviceCount(&ndevAfter), ncclSuccess);
        EXPECT_EQ(ndevAfter, ndevBefore)
            << "Failed merge should not add any devices";
    }
}

TEST_F(NetIbMPITest, Reconnect_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    int ndev = 0;
    AssertInitAndGetDevices(&ndev);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    // Create a fused vNIC once, reuse across all reconnect cycles.
    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ASSERT_EQ(MakeVirtualDevice(&vdev, &vProps), ncclSuccess);
    ASSERT_GE(vdev, 0);

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    const size_t bufferSize = kSmallBufferSize;
    const int tag = 800;
    const int numCycles = 10;

    for (int cycle = 0; cycle < numCycles; cycle++) {
        const int seed = 8000 + cycle;

        // Each cycle creates a fresh connection through the same vNIC.
        ConnectionPair pair;
        ASSERT_EQ(SetupConnection(vdev, pair, rank, peerRank), ncclSuccess)
            << "Cycle " << cycle << ": SetupConnection failed";

        // Allocate and register a fresh buffer per cycle.
        void* buffer = malloc(bufferSize);
        ASSERT_NE(buffer, nullptr);

        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        void* mhandle = nullptr;
        ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess)
            << "Cycle " << cycle << ": regMr failed";

        void* request = nullptr;

        if (rank == 1) {
            // Fill with cycle-specific pattern.
            fillHostBufferWithPattern<uint8_t>(buffer, bufferSize, makeBytePattern(seed));
            PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
        } else {
            memset(buffer, 0, bufferSize);
            PostSingleRecv(pair.recvComm, buffer, bufferSize, tag, mhandle, &request);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        int sizes[1] = {0};
        ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

        // Verify received data matches the cycle-specific pattern.
        if (rank == 0) {
            ASSERT_EQ(sizes[0], bufferSize) << "Cycle " << cycle << ": received size mismatch";
            ASSERT_TRUE(verifyHostBufferData<uint8_t>(buffer, bufferSize, makeBytePattern(seed))) << "Cycle " << cycle << ": data verification failed";
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Manually deregister and close all resources for this cycle.
        // deregMr iterates ndevs=2, removing MR cache entries for each sub-device.
        ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess)
            << "Cycle " << cycle << ": deregMr failed";

        // closeSend/closeRecv destroy per-sub-device QPs, PDs, FIFO MRs, and sockets.
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(pair.recvComm), ncclSuccess);
            ASSERT_EQ(CloseListenComm(pair.listenComm), ncclSuccess);
        } else {
            ASSERT_EQ(CloseSendComm(pair.sendComm), ncclSuccess);
        }

        free(buffer);

        // Sync before next cycle to ensure both ranks have fully torn down.
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

TEST_F(NetIbMPITest, MultiRecvGPUShuffled) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int dev = CreateMergedDevice(3, rank);
    if (dev == -1) {
        GTEST_SKIP() << "Failed to create 3-NIC merged device";
    }

    {
        ncclNetProperties_t props = {};
        ASSERT_EQ(GetDeviceProperties(dev, &props), ncclSuccess);
        if (!(props.ptrSupport & NCCL_PTR_CUDA)) {
            GTEST_SKIP() << "GDR not supported on this device, skipping GPU MultiRecv test";
        }
    }

    static constexpr int kWidth = 8;
    static constexpr int kNumBatches = 255;
    static constexpr int kBaseTag = 16000;
    static constexpr int kTagStride = 100;
    // Keep sizes modest to limit GPU memory allocation per batch
    static constexpr size_t kBaseSizes[kWidth] = {
        64, 256, 1024, 4096, 16384, 65536, 131072, 262144
    };

    // kRecvOrder[slot] = msgId posted at that recv slot
    static constexpr int kRecvOrder[kWidth] = {3, 0, 6, 1, 7, 2, 5, 4};
    // kSendOrder[issueIndex] = msgId to send at that issue position
    static constexpr int kSendOrder[kWidth] = {5, 2, 7, 0, 6, 3, 1, 4};

    // Patterns keyed by msgId (same formula as MultiRecvShuffled for consistency)
    auto FillPattern = [&](uint8_t* hostBuf, size_t size, int batch, int msgId) {
        fillHostBufferWithPattern<uint8_t>(hostBuf, size, [batch, msgId](size_t j) {
            return static_cast<uint8_t>((batch * 19 + msgId * 37 + j) % 256);
        });
    };

    auto CheckPattern = [&](const uint8_t* hostBuf, size_t size, int batch, int msgId) -> bool {
        return verifyHostBufferData<uint8_t>(hostBuf, size, [batch, msgId](size_t j) {
            return static_cast<uint8_t>((batch * 19 + msgId * 37 + j) % 256);
        });
    };

    ConnectionPair pair;
    ASSERT_EQ(SetupConnection(dev, pair, rank, peerRank), ncclSuccess);
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
            void* gpuBufs[kWidth] = {};
            void* mhs[kWidth] = {};
            size_t recvSizesArg[kWidth];
            int recvTags[kWidth];
            int recvSizesOut[kWidth] = {};

            auto cleanup = makeScopeGuard([&]() {
                for (int i = 0; i < kWidth; i++) {
                    if (mhs[i]) { DeregisterMemory(recvComm, mhs[i]); mhs[i] = nullptr; }
                    if (gpuBufs[i]) { hipFreeWrapper(gpuBufs[i]); gpuBufs[i] = nullptr; }
                }
            });

            for (int slot = 0; slot < kWidth; slot++) {
                int msgId = kRecvOrder[slot];
                recvTags[slot] = msgTags[msgId];
                recvSizesArg[slot] = msgSizes[msgId];

                HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&gpuBufs[slot], recvSizesArg[slot]));
                HIP_TEST_CHECK_GTEST_FAIL(hipMemset(gpuBufs[slot], 0xCC, recvSizesArg[slot]));

                ASSERT_EQ(RegisterMemory(recvComm, gpuBufs[slot], recvSizesArg[slot],
                                         NCCL_PTR_CUDA, &mhs[slot]),
                          ncclSuccess)
                    << "GPU RegisterMemory failed for recv batch=" << batch << " slot=" << slot;
                ASSERT_NE(mhs[slot], nullptr);
            }

            void* req = nullptr;
            ASSERT_EQ(PostRecv(recvComm, kWidth, gpuBufs, recvSizesArg, recvTags, mhs, &req),
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
                    << " slot=" << slot << " msgId=" << msgId;

                std::vector<uint8_t> hostBuf(msgSizes[msgId]);
                HIP_TEST_CHECK_GTEST_FAIL(
                    hipMemcpy(hostBuf.data(), gpuBufs[slot], msgSizes[msgId],
                              hipMemcpyDeviceToHost));

                EXPECT_TRUE(CheckPattern(hostBuf.data(), msgSizes[msgId], batch, msgId))
                    << "data mismatch at batch=" << batch
                    << " slot=" << slot << " expected msgId=" << msgId
                    << " tag=" << msgTags[msgId] << " size=" << msgSizes[msgId];
            }

            // cleanup guard fires here, deregistering MRs and freeing GPU buffers
        } else {
            void* gpuBufs[kWidth] = {};
            void* mhs[kWidth] = {};
            void* reqs[kWidth] = {};

            auto cleanup = makeScopeGuard([&]() {
                for (int i = 0; i < kWidth; i++) {
                    if (mhs[i]) { DeregisterMemory(sendComm, mhs[i]); mhs[i] = nullptr; }
                    if (gpuBufs[i]) { hipFreeWrapper(gpuBufs[i]); gpuBufs[i] = nullptr; }
                }
            });

            for (int msgId = 0; msgId < kWidth; msgId++) {
                HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&gpuBufs[msgId], msgSizes[msgId]));

                std::vector<uint8_t> hostBuf(msgSizes[msgId]);
                FillPattern(hostBuf.data(), msgSizes[msgId], batch, msgId);
                HIP_TEST_CHECK_GTEST_FAIL(
                    hipMemcpy(gpuBufs[msgId], hostBuf.data(), msgSizes[msgId],
                              hipMemcpyHostToDevice));

                ASSERT_EQ(RegisterMemory(sendComm, gpuBufs[msgId], msgSizes[msgId],
                                         NCCL_PTR_CUDA, &mhs[msgId]),
                          ncclSuccess)
                    << "GPU RegisterMemory failed for send batch=" << batch
                    << " msgId=" << msgId;
                ASSERT_NE(mhs[msgId], nullptr);
            }

            MPI_Barrier(MPI_COMM_WORLD);

            for (int oi = 0; oi < kWidth; oi++) {
                int msgId = kSendOrder[oi];
                reqs[msgId] = nullptr;
                PostSendWithRetry(sendComm, gpuBufs[msgId], msgSizes[msgId],
                                  msgTags[msgId], mhs[msgId], &reqs[msgId]);

                ASSERT_NE(reqs[msgId], nullptr)
                    << "PostSend returned null request for batch=" << batch
                    << " msgId=" << msgId;
            }

            for (int msgId = 0; msgId < kWidth; msgId++) {
                int sentSize[1] = {0}; // send request yields one size entry
                ASSERT_EQ(WaitForCompletion(reqs[msgId], sentSize, kLargeTransferTimeoutMs),
                          ncclSuccess)
                    << "send completion failed for batch=" << batch << " msgId=" << msgId;
            }

            // cleanup guard fires here, deregistering MRs and freeing GPU buffers
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // connGuard closes comms on scope exit
}

#endif // MPI_TESTS_ENABLED
