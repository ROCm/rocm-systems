/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "DeviceBufferHelpers.hpp"
#include "HostBufferHelpers.hpp"
#include "nccl.h"
#include "net.h"
#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>

#ifdef MPI_TESTS_ENABLED

// Import helper namespaces
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

// External NET IB plugin
extern ncclNet_t ncclNetIb;

// NET IB-specific resource deleters
struct NetMHandleDeleter {
    ncclNet_t* net;
    void* comm;

    NetMHandleDeleter(ncclNet_t* n = nullptr, void* c = nullptr) : net(n), comm(c) {}

    void operator()(void* mhandle) const {
        if (mhandle && net && comm) {
            int rank = MPIEnvironment::world_rank;
            TEST_INFO("Rank %d: NetMHandleDeleter - Deregistering memory handle (mhandle=%p, comm=%p)",
                      rank, mhandle, comm);
            ncclResult_t result = net->deregMr(comm, mhandle);
            TEST_INFO("Rank %d: NetMHandleDeleter - deregMr result: %d", rank, result);
        }
    }
};

// NET IB connection guard
class NetConnectionGuard {
private:
    ncclNet_t* net_;
    void* sendComm_;
    void* recvComm_;
    void* listenComm_;

public:
    explicit NetConnectionGuard(ncclNet_t* net)
        : net_(net), sendComm_(nullptr), recvComm_(nullptr), listenComm_(nullptr) {}

    ~NetConnectionGuard() {
        if (sendComm_ && net_) {
            net_->closeSend(sendComm_);
        }
        if (recvComm_ && net_) {
            net_->closeRecv(recvComm_);
        }
        if (listenComm_ && net_) {
            net_->closeListen(listenComm_);
        }
    }

    void setSendComm(void* comm) { sendComm_ = comm; }
    void setRecvComm(void* comm) { recvComm_ = comm; }
    void setListenComm(void* comm) { listenComm_ = comm; }

    NetConnectionGuard(const NetConnectionGuard&) = delete;
    NetConnectionGuard& operator=(const NetConnectionGuard&) = delete;
};

// Type alias for NetMHandleGuard using ResourceGuard
using NetMHandleGuard = RCCLTestGuards::ResourceGuard<void*, NetMHandleDeleter>;

// Test fixture for NET IB tests
class NetIbMPITest : public MPITestBase {
protected:
    static constexpr int kMinProcessesForMPI = 2;
    static constexpr bool kRequirePowerOfTwo = true;
    static constexpr int kNoNodeLimit = MPITestConstants::kNoNodeLimit;

    // Buffer pattern constants
    static constexpr int kBytePatternModulo = 256;

    // Timing constants
    static constexpr int kDefaultTimeoutMs = 5000;
    static constexpr int kLargeTransferTimeoutMs = 30000;
    static constexpr int kPollIntervalUs = 10000;  // 10ms
    static constexpr int kPollIntervalMs = 10;
    static constexpr int kMaxRetryAttempts = 1000;  // For NULL request handling

    // Buffer size constants
    static constexpr size_t kSmallBufferSize = 4096;
    static constexpr size_t kLargeBufferSize = 16 * 1024 * 1024;  // 16 MB

    // Test seed constants
    static constexpr int kBaseSeedOffset = 1000;
    static constexpr int kMultiSizeSeedOffset = 2000;

    // Debug output constants
    static constexpr int kNumDebugSamples = 4;

    // Invalid device ID offset for negative tests
    static constexpr int kInvalidDeviceOffset = 100;

    // Process count constants
    static constexpr int kExactTwoProcesses = 2;
    static constexpr int kMinGpusPerNode = 1;

    // Transfer test constants
    static constexpr int kNumSequentialTransfers = 100;
    static constexpr int kTransferTagBase = 300;

    // Timeout constants
    static constexpr int kLargeTransferTimeout = 30000;

    ncclNet_t* net_;
    int numDevices_;
    std::vector<int> deviceIds_;
    void* initCtx_;

    void SetUp() override {
        MPITestBase::SetUp();
        net_ = &ncclNetIb;
        numDevices_ = 0;
        initCtx_ = nullptr;
    }

    void TearDown() override {
        if (initCtx_) {
            net_->finalize(initCtx_);
            initCtx_ = nullptr;
        }
        MPITestBase::TearDown();
    }

    // Helper: Initialize NET IB plugin
    ncclResult_t InitNetIb() {
        ncclNetCommConfig_t commConfig = {};
        commConfig.trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
        return net_->init(&initCtx_, 0, &commConfig, nullptr, nullptr);
    }

    // Helper: Get number of devices
    ncclResult_t GetDeviceCount(int* ndev) {
        return net_->devices(ndev);
    }

    // Helper: Get device properties
    ncclResult_t GetDeviceProperties(int dev, ncclNetProperties_t* props) {
        return net_->getProperties(dev, props);
    }

    // Helper: Create listen comm
    ncclResult_t CreateListenComm(int dev, ncclNetHandle_t* handle, void** listenComm) {
        return net_->listen(initCtx_, dev, handle, listenComm);
    }

    // Helper: Connect to remote
    ncclResult_t ConnectToRemote(int dev, ncclNetHandle_t* handle, void** sendComm) {
        return net_->connect(initCtx_, dev, handle, sendComm, nullptr);
    }

    // Helper: Accept connection
    ncclResult_t AcceptConnection(void* listenComm, void** recvComm) {
        return net_->accept(listenComm, recvComm, nullptr);
    }

    // Helper: Register memory
    ncclResult_t RegisterMemory(void* comm, void* data, size_t size, int type, void** mhandle) {
        return net_->regMr(comm, data, size, type, mhandle);
    }

    // Helper: Register DMA-BUF memory
    ncclResult_t RegisterDmaBufMemory(void* comm, void* data, size_t size, int type,
                                      uint64_t offset, int fd, void** mhandle) {
        return net_->regMrDmaBuf(comm, data, size, type, offset, fd, mhandle);
    }

    // Helper: Deregister memory
    ncclResult_t DeregisterMemory(void* comm, void* mhandle) {
        return net_->deregMr(comm, mhandle);
    }

    // Helper: Post send operation
    ncclResult_t PostSend(void* sendComm, void* data, size_t size, int tag,
                         void* mhandle, void** request) {
        return net_->isend(sendComm, data, size, tag, mhandle, nullptr, request);
    }

    // Helper: Post recv operation
    ncclResult_t PostRecv(void* recvComm, int n, void** data, size_t* sizes,
                         int* tags, void** mhandles, void** request) {
        return net_->irecv(recvComm, n, data, sizes, tags, mhandles, nullptr, request);
    }

    // Helper: Flush operation
    ncclResult_t FlushRecv(void* recvComm, int n, void** data, int* sizes,
                          void** mhandles, void** request) {
        return net_->iflush(recvComm, n, data, sizes, mhandles, request);
    }

    // Helper: Test request completion
    // No implementation for this method in the NET IB plugin
    ncclResult_t TestRequest(void* request, int* done, int* sizes) {
        return net_->test(request, done, sizes);
    }

    // Helper: Close send comm
    ncclResult_t CloseSendComm(void* sendComm) {
        return net_->closeSend(sendComm);
    }

    // Helper: Close recv comm
    ncclResult_t CloseRecvComm(void* recvComm) {
        return net_->closeRecv(recvComm);
    }

    // Helper: Close listen comm
    ncclResult_t CloseListenComm(void* listenComm) {
        return net_->closeListen(listenComm);
    }

    // Helper: Make virtual device
    ncclResult_t MakeVirtualDevice(int* dev, ncclNetVDeviceProps_t* props) {
        return net_->makeVDevice(dev, props);
    }

    // Helper: Setup connection between two ranks
    struct ConnectionPair {
        void* sendComm = nullptr;
        void* recvComm = nullptr;
        void* listenComm = nullptr;
        ncclNetHandle_t handle;
    };

    ncclResult_t SetupConnection(int dev, ConnectionPair& pair, int rank, int peerRank) {
        if (rank == 0) {
            // Rank 0: Listen
            RCCL_TEST_CHECK(CreateListenComm(dev, &pair.handle, &pair.listenComm));

            // Send handle to peer
            MPI_Send(&pair.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD);

            // Accept connection
            int done = 0;
            while (!done) {
                ncclResult_t result = AcceptConnection(pair.listenComm, &pair.recvComm);
                if (result == ncclSuccess && pair.recvComm != nullptr) {
                    done = 1;
                }
            }
        } else {
            // Rank 1: Connect
            MPI_Recv(&pair.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Connect to peer
            int done = 0;
            while (!done) {
                ncclResult_t result = ConnectToRemote(dev, &pair.handle, &pair.sendComm);
                if (result == ncclSuccess && pair.sendComm != nullptr) {
                    done = 1;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        return ncclSuccess;
    }

    // Helper: Initialize device buffer with pattern using DeviceBufferHelpers
    hipError_t InitializeBuffer(void* buffer, size_t size, int pattern) {
        // Use template-based helper with custom pattern: (pattern + i) % kBytePatternModulo
        return initializeBufferWithPattern<uint8_t>(
            buffer, size,
            [pattern](size_t i) { return static_cast<uint8_t>((pattern + i) % kBytePatternModulo); }
        );
    }

    // Helper: Verify device buffer pattern using DeviceBufferHelpers
    bool VerifyBuffer(void* buffer, size_t size, int pattern) {
        // Use template-based helper with pattern verification
        return verifyBufferData<uint8_t>(
            buffer, size,
            [pattern](size_t i) {
                return static_cast<uint8_t>((pattern + i) % kBytePatternModulo);
            }
        );
    }

    // Helper: Fill host buffer with pattern using HostBufferHelpers
    void FillHostBuffer(void* buffer, size_t size, int seed) {
        fillHostBufferWithPattern<uint8_t>(
            buffer, size,
            [seed](size_t i) { return static_cast<uint8_t>((seed + i) % kBytePatternModulo); }
        );
    }

    // Helper: Verify host buffer pattern using HostBufferHelpers
    bool VerifyHostBuffer(const void* buffer, size_t size, int seed) {
        return verifyHostBufferData<uint8_t>(
            buffer, size,
            [seed](size_t i) { return static_cast<uint8_t>((seed + i) % kBytePatternModulo); }
        );
    }

    // Helper: Retry until the receiver's FIFO slot is ready.
    void PostSendWithRetry(void* sendComm, void* data, size_t size, int tag,
                           void* mhandle, void** request) {
        int attempts = 0;
        do {
            ncclResult_t result = PostSend(sendComm, data, size, tag, mhandle, request);
            ASSERT_EQ(result, ncclSuccess);
            if (*request != nullptr) break;
            if (++attempts >= kMaxRetryAttempts) {
                FAIL() << "PostSend returned NULL request after " << kMaxRetryAttempts << " attempts";
            }
            usleep(kPollIntervalUs);
        } while (*request == nullptr);
    }

    // Helper: Wait for request completion with timeout
    ncclResult_t WaitForCompletion(void* request, int* sizes, int timeoutMs = kDefaultTimeoutMs) {
        if (!request) return ncclInternalError;

        int done = 0;
        int attempts = 0;
        const int maxAttempts = timeoutMs / kPollIntervalMs;

        while (!done && attempts < maxAttempts) {
            ncclResult_t result = TestRequest(request, &done, sizes);

            if (result != ncclSuccess) {
                return result;
            }

            if (done) {
                break;
            } else {
                usleep(kPollIntervalUs); // 10ms
                attempts++;
            }
        }

        return done ? ncclSuccess : ncclInternalError;
    }
};

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

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    for (int i = 0; i < ndev; i++) {
        ncclNetProperties_t props;
        memset(&props, 0, sizeof(props));

        EXPECT_EQ(GetDeviceProperties(i, &props), ncclSuccess)
            << "Failed to get properties for device " << i;

        // Verify properties are valid
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

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

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

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

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

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

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

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

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

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

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

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

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

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

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

    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    // Edge case: Deregister NULL handle (should be no-op)
    EXPECT_EQ(DeregisterMemory(comm, nullptr), ncclSuccess);
}

// Send/Recv Tests
TEST_F(NetIbMPITest, SimpleSendRecv) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

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
        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
    } else {
        // Sender
        FillHostBuffer(buffer, bufferSize, rank);

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
        EXPECT_TRUE(VerifyHostBuffer(buffer, bufferSize, senderRank)) << "Data validation failed";
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

    int rank = MPIEnvironment::world_rank;

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
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

            void* recvBuffers[1] = {buffer};
            size_t recvSizes[1] = {size};
            int recvTags[1] = {tag};
            void* recvHandles[1] = {mhandle};

            ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                              recvHandles, &request), ncclSuccess);
            ASSERT_NE(request, nullptr) << "Recv request should never be NULL";
        } else {
            FillHostBuffer(buffer, size, seed);

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

            EXPECT_TRUE(VerifyHostBuffer(buffer, size, seed)) << "Data validation failed for size " << size;
        }

        // NetMHandleGuard will automatically deregister at end of loop iteration
    }
}

TEST_F(NetIbMPITest, SendRecvZeroSize) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

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
        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
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

// Flush Tests

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
    auto bufferGuard = makeDeviceBufferAutoGuard(buffer);  // false = device memory

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
        ASSERT_EQ(InitializeBuffer(buffer, bufferSize, rank), hipSuccess);

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
            int flushDone = 0;
            ASSERT_EQ(WaitForCompletion(flushRequest, nullptr), ncclSuccess);
        }
    }

    // NetMHandleGuard will automatically deregister at scope end
}

// Virtual Device Tests

TEST_F(NetIbMPITest, MakeVirtualDevice) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

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

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    // Negative test: Zero devices
    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 0;

    int vdev = -1;
    ncclResult_t result = MakeVirtualDevice(&vdev, &vProps);
    EXPECT_EQ(result, ncclInvalidUsage) << "Should fail with zero devices";

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

    int rank = MPIEnvironment::world_rank;

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
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

            void* recvBuffers[1] = {recvBuffer};
            size_t recvSizes[1] = {bufferSize};
            int recvTags[1] = {tag};
            void* recvHandles[1] = {mhandle};

            ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                              recvHandles, &request), ncclSuccess);
            ASSERT_NE(request, nullptr) << "Recv request should never be NULL";
        } else {
            FillHostBuffer(sendBuffer, bufferSize, seed);

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

            EXPECT_TRUE(VerifyHostBuffer(recvBuffer, bufferSize, seed)) << "Transfer " << i << " data validation failed (seed=" << seed << ")";

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

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

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
        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
    } else {
        // Sender
        FillHostBuffer(buffer, bufferSize, rank);

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
        EXPECT_TRUE(VerifyHostBuffer(buffer, bufferSize, senderRank)) << "Large transfer data validation failed";
    }

    // NetMHandleGuard will automatically deregister at scope end
}

TEST_F(NetIbMPITest, CloseWithoutWaitingForCompletion) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

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
}

TEST_F(NetIbMPITest, ConnectAndTransfer_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

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

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    // Connect through the vNIC
    ConnectionPair pair;
    ASSERT_EQ(SetupConnection(vdev, pair, rank, peerRank), ncclSuccess);

    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

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

        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
    } else {
        FillHostBuffer(buffer, bufferSize, seed);

        PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    // Verify data integrity.
    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Received size mismatch";

        EXPECT_TRUE(VerifyHostBuffer(buffer, bufferSize, seed)) << "Data validation failed on vNIC transfer";
    }

}

TEST_F(NetIbMPITest, AsymmetricMerge_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

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

        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
    } else {
        FillHostBuffer(buffer, bufferSize, seed);

        PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    // Verify data integrity across the asymmetric connection.
    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Received size mismatch";

        EXPECT_TRUE(VerifyHostBuffer(buffer, bufferSize, seed)) << "Data validation failed on asymmetric vNIC transfer";
    }
}

TEST_F(NetIbMPITest, CloseWithoutTransfer_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

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
        ASSERT_EQ(SetupConnection(vdev, pair, rank, peerRank), ncclSuccess);

        // Guard triggers closeSend/closeRecv/closeListen on a connection that
        // never had regMr, isend, or irecv called.
        NetConnectionGuard connGuard(net_);
        if (rank == 0) {
            connGuard.setRecvComm(pair.recvComm);
            connGuard.setListenComm(pair.listenComm);
        } else {
            connGuard.setSendComm(pair.sendComm);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Phase 2: reconnect on the same vdev and do a transfer to verify no corruption.
    ConnectionPair pair2;
    ASSERT_EQ(SetupConnection(vdev, pair2, rank, peerRank), ncclSuccess);

    NetConnectionGuard connGuard2(net_);
    if (rank == 0) {
        connGuard2.setRecvComm(pair2.recvComm);
        connGuard2.setListenComm(pair2.listenComm);
    } else {
        connGuard2.setSendComm(pair2.sendComm);
    }

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

        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair2.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
    } else {
        FillHostBuffer(buffer, bufferSize, seed);

        PostSendWithRetry(pair2.sendComm, buffer, bufferSize, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    // Verify the vNIC is still functional after the no-transfer teardown.
    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Received size mismatch";

        EXPECT_TRUE(VerifyHostBuffer(buffer, bufferSize, seed)) << "Data validation failed after no-transfer teardown reconnect";
    }
}

TEST_F(NetIbMPITest, RegDeregCycling_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

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
    ASSERT_EQ(SetupConnection(vdev, pair, rank, peerRank), ncclSuccess);

    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

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

        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
    } else {
        FillHostBuffer(buffer, bufferSize, seed);

        PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    // Data integrity check after MR cache cycling.
    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Received size mismatch";

        EXPECT_TRUE(VerifyHostBuffer(buffer, bufferSize, seed)) << "Data validation failed after reg/dereg cycling";
    }
}

TEST_F(NetIbMPITest, LargeTransfer_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

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
    ASSERT_EQ(SetupConnection(vdev, pair, rank, peerRank), ncclSuccess);

    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

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

        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
    } else {
        FillHostBuffer(buffer, bufferSize, seed);

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

        EXPECT_TRUE(VerifyHostBuffer(buffer, bufferSize, seed)) << "Large vNIC transfer data validation failed";
    }
}

TEST_F(NetIbMPITest, MixedSizes_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

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
    ASSERT_EQ(SetupConnection(vdev, pair, rank, peerRank), ncclSuccess);

    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

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

            void* recvBuffers[1] = {buffer};
            size_t recvSizes[1] = {size};
            int recvTags[1] = {tag};
            void* recvHandles[1] = {mhandle};

            ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                              recvHandles, &request), ncclSuccess);
        } else {
            FillHostBuffer(buffer, size, seed);

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

            EXPECT_TRUE(VerifyHostBuffer(buffer, size, seed)) << "Data validation failed for size " << size;
        }
    }
}

TEST_F(NetIbMPITest, UnalignedSizeTransfer_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 IB devices for NIC fusion tests";
    }

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ASSERT_EQ(MakeVirtualDevice(&vdev, &vProps), ncclSuccess)
        << "Failed to create fused vNIC from devices 0 and 1";
    ASSERT_GE(vdev, 0);

    ConnectionPair pair;
    ASSERT_EQ(SetupConnection(vdev, pair, rank, peerRank), ncclSuccess);

    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

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

            void* recvBuffers[1] = {buffer};
            size_t recvSizes[1] = {size};
            int recvTags[1] = {tag};
            void* recvHandles[1] = {mhandle};

            ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                              recvHandles, &request), ncclSuccess);
        } else {
            FillHostBuffer(buffer, size, seed);

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

            EXPECT_TRUE(VerifyHostBuffer(buffer, size, seed)) << "Data validation failed for size " << size;
        }
    }
}

TEST_F(NetIbMPITest, Bidirectional_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

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
    FillHostBuffer(sendBuf, bufferSize, sendSeed);

    // Post recv and send simultaneously on both connections.
    void* recvRequest = nullptr;
    void* sendRequest = nullptr;

    void* recvBuffers[1] = {recvBuf};
    size_t recvSizes[1] = {bufferSize};
    int recvTags[1] = {recvTag};
    void* recvHandles[1] = {recvMhandle};

    ASSERT_EQ(PostRecv(recvComm, 1, recvBuffers, recvSizes, recvTags,
                      recvHandles, &recvRequest), ncclSuccess);

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

    EXPECT_TRUE(VerifyHostBuffer(recvBuf, bufferSize, peerSeed)) << "Bidirectional vNIC transfer data validation failed";
}

TEST_F(NetIbMPITest, FlushRepeated_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

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

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ConnectionPair pair;
    ASSERT_EQ(SetupConnection(vdev, pair, rank, peerRank), ncclSuccess);

    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

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
            ASSERT_EQ(InitializeBuffer(gpuBuffer, bufferSize, seed), hipSuccess);

            PostSendWithRetry(pair.sendComm, gpuBuffer, bufferSize, tag, mhandle, &request);
        } else {
            // Rank 0: post recv on GPU buffer.
            void* recvBuffers[1] = {gpuBuffer};
            size_t recvSizes[1] = {bufferSize};
            int recvTags[1] = {tag};
            void* recvHandles[1] = {mhandle};

            ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                              recvHandles, &request), ncclSuccess);
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
            ASSERT_TRUE(VerifyBuffer(gpuBuffer, bufferSize, seed))
                << "Iter " << iter << ": data verification failed after flush";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

TEST_F(NetIbMPITest, SequentialTransfers_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

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

    int rank = MPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    // Single connection through the vNIC, reused across all 100 iterations.
    ConnectionPair pair;
    ASSERT_EQ(SetupConnection(vdev, pair, rank, peerRank), ncclSuccess);

    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

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
            FillHostBuffer(buffer, bufferSize, seed);

            PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
        } else {
            // Zero buffer before recv to ensure verification catches stale data.
            memset(buffer, 0, bufferSize);

            void* recvBuffers[1] = {buffer};
            size_t recvSizes[1] = {bufferSize};
            int recvTags[1] = {tag};
            void* recvHandles[1] = {mhandle};

            ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                              recvHandles, &request), ncclSuccess);
        }

        // Sync both ranks before polling for completion.
        MPI_Barrier(MPI_COMM_WORLD);

        int sizes[1] = {0};
        ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

        // Verify received data matches the iteration-specific pattern.
        if (rank == 0) {
            ASSERT_EQ(sizes[0], bufferSize) << "Iter " << iter << ": received size mismatch";

            ASSERT_TRUE(VerifyHostBuffer(buffer, bufferSize, seed)) << "Iter " << iter << ": data verification failed";
        }

        // Sync before next iteration to prevent request reuse races.
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

TEST_F(NetIbMPITest, Reconnect_VNic) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                         false, kMinGpusPerNode, kNoNodeLimit))
        << "Test requires exactly " << kExactTwoProcesses << " processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

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
            FillHostBuffer(buffer, bufferSize, seed);

            PostSendWithRetry(pair.sendComm, buffer, bufferSize, tag, mhandle, &request);
        } else {
            memset(buffer, 0, bufferSize);

            void* recvBuffers[1] = {buffer};
            size_t recvSizes[1] = {bufferSize};
            int recvTags[1] = {tag};
            void* recvHandles[1] = {mhandle};

            ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                              recvHandles, &request), ncclSuccess);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        int sizes[1] = {0};
        ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

        // Verify received data matches the cycle-specific pattern.
        if (rank == 0) {
            ASSERT_EQ(sizes[0], bufferSize) << "Cycle " << cycle << ": received size mismatch";

            ASSERT_TRUE(VerifyHostBuffer(buffer, bufferSize, seed)) << "Cycle " << cycle << ": data verification failed";
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

#endif // MPI_TESTS_ENABLED
