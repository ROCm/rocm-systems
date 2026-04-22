/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_NET_IB_MPI_TEST_BASE_HPP_
#define RCCL_TEST_NET_IB_MPI_TEST_BASE_HPP_

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include "MPITestBase.hpp"
#include "NetIbCastInspect.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "DeviceBufferHelpers.hpp"
#include "HostBufferHelpers.hpp"
#include "nccl.h"
#include "net.h"
#include <vector>
#include <memory>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#ifdef MPI_TESTS_ENABLED

// Import helper namespaces
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

// Skip a Cast test when any required WRR scheduler env var is absent or wrong.
// Must be called from the test body (not a helper), because GTEST_SKIP() only
// interrupts execution when expanded inline in the test scope.
// All vars below are set by the cast_base section in net_ib_transport.json.
#define CAST_ENV_CHECK_OR_SKIP()                                                         \
    do {                                                                                 \
        struct { const char* name; const char* required; } _vars[] = {                  \
            { "RCCL_IB_QP_SCHED_ENABLE",          "1"      },                           \
            { "RCCL_IB_QP_SCHED_WRR_ENABLE",      "1"      },                           \
            { "RCCL_IB_QP_SCHED_WEIGHT",          nullptr  },                           \
            { "RCCL_IB_QP_SCHED_UPDATE_INTERVAL", nullptr  },                           \
            { "RCCL_IB_QP_SCHED_RESET_INTERVAL",  nullptr  },                           \
            { "RCCL_IB_QP_SCHED_SPLIT_DATA_MIN",  nullptr  },                           \
            { "NCCL_IB_QPS_PER_CONNECTION",        nullptr  },                           \
            { "NCCL_IB_SPLIT_DATA_ON_QPS",         nullptr  },                           \
        };                                                                               \
        for (auto& _v : _vars) {                                                         \
            const char* _val = getenv(_v.name);                                          \
            bool _missing = !_val || _val[0] == '\0';                                    \
            bool _wrong   = _v.required && (!_val || strcmp(_val, _v.required) != 0);   \
            if (_missing || _wrong) {                                                    \
                GTEST_SKIP() << "Cast tests require all WRR scheduler env vars. "       \
                                "Missing or wrong: " << _v.name                         \
                             << " (expected: " << (_v.required ? _v.required : "<any>") \
                             << "). Use cast_* configs in net_ib_transport.json.";       \
            }                                                                            \
        }                                                                                \
    } while (0)

// External NET IB plugin
extern ncclNet_t ncclNetIb;
// External NET IB-CAST plugin (WRR scheduler, multi-QP)
extern ncclNet_t netIbCast;

// Select plugin by NCCL_NET env var name; falls back to ncclNetIb.
inline ncclNet_t* GetPlugin() {
    static ncclNet_t* plugins[] = {&ncclNetIb, &netIbCast};
    const char* env = getenv("NCCL_NET");
    if (env) {
        for (auto* p : plugins) {
            if (strcmp(env, p->name) == 0) {
                TEST_INFO("Rank %d: Using plugin %s", MPIEnvironment::world_rank, p->name);
                return p;
            }
        }
    }
    TEST_INFO("Rank %d: Using default plugin %s", MPIEnvironment::world_rank, ncclNetIb.name);
    return &ncclNetIb;
}

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
        net_ = GetPlugin();
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

    // Composite block: Init plugin + assert device count > 0.
    // Pass a non-null pointer to receive the count; pass nullptr to discard it.
    void AssertInitAndGetDevices(int* ndev) {
        int local = 0;
        int* p = ndev ? ndev : &local;
        ASSERT_EQ(InitNetIb(), ncclSuccess);
        ASSERT_EQ(GetDeviceCount(p), ncclSuccess);
        ASSERT_GT(*p, 0);
    }

    // Composite block: SetupConnection + wire up NetConnectionGuard for RAII cleanup.
    // dev: device index. Uses world_rank to determine listener vs connector.
    void SetupConnectionWithGuard(int dev, ConnectionPair& pair,
                                  NetConnectionGuard& guard) {
        const int rank     = MPIEnvironment::world_rank;
        const int peerRank = (rank + 1) % 2;
        ASSERT_EQ(SetupConnection(dev, pair, rank, peerRank), ncclSuccess);
        if (rank == 0) {
            guard.setRecvComm(pair.recvComm);
            guard.setListenComm(pair.listenComm);
        } else {
            guard.setSendComm(pair.sendComm);
        }
    }

    // Composite block: Post a single irecv. Wraps the 4-array boilerplate.
    void PostSingleRecv(void* recvComm, void* buf, size_t size, int tag,
                        void* mhandle, void** request) {
        void*  bufs[1]    = {buf};
        size_t sizes[1]   = {size};
        int    tags[1]    = {tag};
        void*  handles[1] = {mhandle};
        ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, request), ncclSuccess);
    }

    // Helper: create a merged device from N physical NICs.
    // Returns merged device index, or -1 if not enough devices / merge failed.
    // physDevs: indices of physical (non-merged) devices
    // props: device properties (indexed by device index)
    // nNicsToMerge: how many NICs to merge (e.g., 2, 3, 4)
    // rank: MPI rank of this process
    int CreateMergedDevice(int nNicsToMerge, int rank, int offset = 0)
    {
        if (nNicsToMerge <= 0 || nNicsToMerge > NCCL_NET_MAX_DEVS_PER_NIC) {
            fprintf(stderr,
                    "Rank %d requested invalid merge size %d (valid range: 1..%d)\n",
                    rank, nNicsToMerge, NCCL_NET_MAX_DEVS_PER_NIC);
            return -1;
        }

        int outMergedDev = -1;

        int ndev = 0;
        RCCL_TEST_CHECK(GetDeviceCount(&ndev));
        if (ndev <= 0) return -1;

        std::vector<ncclNetProperties_t> props(ndev);
        std::vector<int> physDevs;
        for (int i = 0; i < ndev; i++) {
            memset(&props[i], 0, sizeof(ncclNetProperties_t));
            RCCL_TEST_CHECK(GetDeviceProperties(i, &props[i]));
            if (!props[i].name || !strchr(props[i].name, '+'))
                physDevs.push_back(i);
        }

        if (physDevs.size() < offset + 1) {
            return -1;
        }

        int targetSpeed = props[physDevs[offset]].speed;
        std::vector<int> compat;
        for (int d : physDevs)
            if (props[d].speed == targetSpeed) compat.push_back(d);

        if ((int)compat.size() < nNicsToMerge * 2) {
            return CreateMergedDevice(nNicsToMerge, rank, offset + 1);
        }

        int baseOffset = (rank % 2) ? nNicsToMerge : 0;
        int finalOffset = baseOffset + offset;

        if (finalOffset + nNicsToMerge > (int)compat.size()) return -1;

        ncclNetVDeviceProps_t vProps;
        memset(&vProps, 0, sizeof(vProps));
        vProps.ndevs = nNicsToMerge;
        for (int i = 0; i < nNicsToMerge; i++)
            vProps.devs[i] = compat[finalOffset + i];

        if (MakeVirtualDevice(&outMergedDev, &vProps) != ncclSuccess || outMergedDev < 0) {
            fprintf(stderr,
                    "Rank %d failed to create %d-NIC merged device at offset %d, retrying with offset %d\n",
                    rank, nNicsToMerge, offset, offset + 1);
            return CreateMergedDevice(nNicsToMerge, rank, offset + 1);
        }

        return outMergedDev;
    }


    // On return: rank 0 owns listenComm+recvComm, rank 1 owns sendComm.
    // Caller is responsible for closing all comms.
    void SetupCastConnection(int dev,
                             void** listenComm, void** sendComm, void** recvComm) {
        const int rank = MPIEnvironment::world_rank;
        const int peer = 1 - rank;
        ncclNetHandle_t handle;
        memset(&handle, 0, sizeof(handle));

        if (rank == 0) {
            ASSERT_EQ(CreateListenComm(dev, &handle, listenComm), ncclSuccess);
            ASSERT_NE(*listenComm, nullptr);

            MPI_Send(&handle, sizeof(handle), MPI_BYTE, peer, 0, MPI_COMM_WORLD);

            for (int i = 0; i < kMaxRetryAttempts && *recvComm == nullptr; i++) {
                ASSERT_EQ(AcceptConnection(*listenComm, recvComm), ncclSuccess);
                if (*recvComm == nullptr) usleep(kPollIntervalUs);
            }
            ASSERT_NE(*recvComm, nullptr);
        } else {
            MPI_Recv(&handle, sizeof(handle), MPI_BYTE, peer, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

            for (int i = 0; i < kMaxRetryAttempts && *sendComm == nullptr; i++) {
                ncclResult_t r = ConnectToRemote(dev, &handle, sendComm);
                ASSERT_EQ(r, ncclSuccess);
                if (*sendComm == nullptr) usleep(kPollIntervalUs);
            }
            ASSERT_NE(*sendComm, nullptr);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Composite block: Warmup send + read real nqps from sendComm on rank 1.
    // Both ranks call this together. actualNqps is broadcast so rank 0 can coordinate.
    // buf/mhandle must already be registered against the caller's comm.
    int GetActualNqps(void* sendComm, void* recvComm,
                      void* buf, size_t size, int tag, void* mhandle) {
        const int rank = MPIEnvironment::world_rank;
        CastDoSendRecv(rank, sendComm, recvComm, buf, size, tag, mhandle);
        int nqps = 0;
        if (rank == 1) {
            struct ncclIbCastSchedState probe = {};
            EXPECT_EQ(ncclIbCastGetSchedState(sendComm, &probe), ncclSuccess);
            nqps = probe.nqps;
        }
        MPI_Bcast(&nqps, 1, MPI_INT, 1, MPI_COMM_WORLD);
        EXPECT_GT(nqps, 0);
        return nqps;
    }

    // Read RCCL_IB_QP_SCHED_SPLIT_DATA_MIN from the environment.
    // Falls back to 65536 if unset (matches the RCCL default).
    static uint32_t GetSplitDataMin() {
        const char* v = getenv("RCCL_IB_QP_SCHED_SPLIT_DATA_MIN");
        return (v && v[0]) ? static_cast<uint32_t>(std::stoul(v)) : 65536u;
    }

    // Build an equal-weight token vector summing to totTokens for nqps QPs.
    // Remainder distributed to the first slots.
    static std::vector<int> EqualTokens(int nqps, int totTokens = 100) {
        std::vector<int> t(nqps, totTokens / nqps);
        for (int i = 0; i < totTokens % nqps; i++) t[i]++;
        return t;
    }

    // Composite block: Single-message send/recv pair for CAST tests.
    // rank 0 posts irecv and waits; rank 1 posts isend (with retry) and waits.
    // Both sides must have already registered buf/mhandle against their comm.
    void CastDoSendRecv(int rank, void* sendComm, void* recvComm,
                        void* buf, size_t size, int tag, void* mhandle) {
        void* req = nullptr;
        if (rank == 0) {
            void*  bufs[1]    = {buf};
            size_t sizes[1]   = {size};
            int    tags[1]    = {tag};
            void*  handles[1] = {mhandle};
            ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &req), ncclSuccess);
            ASSERT_NE(req, nullptr);
            int sz = 0;
            ASSERT_EQ(WaitForCompletion(req, &sz, 10000), ncclSuccess);
        } else {
            PostSendWithRetry(sendComm, buf, size, tag, mhandle, &req);
            int sz = 0;
            ASSERT_EQ(WaitForCompletion(req, &sz, 10000), ncclSuccess);
        }
    }

    // Composite block: Concurrent N-message send/recv for CAST tests.
    // All N sends/recvs are posted before any completion is waited on, allowing
    // the transport to pipeline multiple WRs in flight simultaneously.
    //
    // bufs[i] / baseTag+i must be pre-registered via mhandle (a single MR
    // covering the whole multi-message buffer is fine).
    //
    // rank 0: posts N irecvs, then waits for all N completions.
    // rank 1: posts N isends (with per-message retry), then waits for all N.
    void CastDoBatchSendRecv(int rank, void* sendComm, void* recvComm,
                             char* sendBuf, char* recvBuf,
                             size_t msgSz, int nMsgs, int baseTag, void* mhandle) {
        std::vector<void*> reqs(nMsgs, nullptr);
        if (rank == 0) {
            for (int i = 0; i < nMsgs; i++) {
                void*  bufs[1]    = {recvBuf + i * msgSz};
                size_t sizes[1]   = {msgSz};
                int    tags[1]    = {baseTag + i};
                void*  handles[1] = {mhandle};
                ASSERT_EQ(PostRecv(recvComm, 1, bufs, sizes, tags, handles, &reqs[i]), ncclSuccess);
                ASSERT_NE(reqs[i], nullptr);
            }
            for (int i = 0; i < nMsgs; i++) {
                int sz = 0;
                ASSERT_EQ(WaitForCompletion(reqs[i], &sz, 10000), ncclSuccess);
            }
        } else {
            for (int i = 0; i < nMsgs; i++) {
                PostSendWithRetry(sendComm, sendBuf + i * msgSz, msgSz, baseTag + i, mhandle, &reqs[i]);
            }
            for (int i = 0; i < nMsgs; i++) {
                int sz = 0;
                ASSERT_EQ(WaitForCompletion(reqs[i], &sz, 10000), ncclSuccess);
            }
        }
    }
};

#endif /* MPI_TESTS_ENABLED */

#endif /* RCCL_TEST_NET_IB_MPI_TEST_BASE_HPP_ */
