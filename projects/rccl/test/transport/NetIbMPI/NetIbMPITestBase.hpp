/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_NET_IB_MPI_TEST_BASE_HPP_
#define RCCL_TEST_NET_IB_MPI_TEST_BASE_HPP_

#include <gtest/gtest.h>
// Pre-seed the bf16 guard macros and pull in the new hip_bf16.h before
// hip_runtime.h transitively includes the old hip_bfloat16.h. Otherwise
// device.h's #error guard fires during the device-pass compile (the test
// include path exposes the hipified librccl device.h via
// ${PROJECT_BINARY_DIR}/hipify/src/include).
#if defined(ROCM_VERSION) && ROCM_VERSION >= 60000
  #if !defined(_HIP_INCLUDE_HIP_AMD_DETAIL_HIP_BFLOAT16_H_) && !defined(_HIP_BFLOAT16_H_)
    #define _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_BFLOAT16_H_
    #define _HIP_BFLOAT16_H_
    #include <hip/hip_bf16.h>
  #endif
#endif
#include <hip/hip_runtime.h>
#include "MPITestBase.hpp"
#include "NetIbCastInspect.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "DeviceBufferHelpers.hpp"
#include "HostBufferHelpers.hpp"
#include "nccl.h"
#include "net.h"
#include "plugin/nccl_net.h"
#include <vector>
#include <memory>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <dirent.h>
#include <unistd.h>

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

// Skip when RCCL_IB_QP_SCHED_UPDATE_INTERVAL is below min_us: tests that
// assert exact per-send token state need the RTT-driven update suspended.
#define CAST_REQUIRE_UPDATE_INTERVAL_OR_SKIP(min_us)                                     \
    do {                                                                                 \
        const char* _ui = getenv("RCCL_IB_QP_SCHED_UPDATE_INTERVAL");                    \
        long long _v = (_ui && _ui[0]) ? std::atoll(_ui) : 0;                            \
        if (_v < (long long)(min_us)) {                                                  \
            GTEST_SKIP() << "Requires RCCL_IB_QP_SCHED_UPDATE_INTERVAL >= "             \
                         << (long long)(min_us) << " us (current: " << _v << ")";        \
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

    // Returns true if the IB device has at least one routable GID (non-zero IPv4-mapped
    // or global-scope IPv6). NICs with only link-local GIDs cannot do cross-node RDMA.
    static bool HasRoutableGid(const char* devName) {
        char path[PATH_MAX];
        if (snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/1/gids", devName) >= PATH_MAX)
            return false;
        DIR* d = opendir(path);
        if (!d) return false;
        struct dirent* ent;
        bool found = false;
        while ((ent = readdir(d)) != nullptr) {
            if (ent->d_name[0] == '.') continue;
            char gidPath[PATH_MAX];
            snprintf(gidPath, sizeof(gidPath), "%s/%s", path, ent->d_name);
            FILE* f = fopen(gidPath, "r");
            if (!f) continue;
            char gid[64] = {};
            fscanf(f, "%63s", gid);
            fclose(f);
            // Skip all-zero GIDs and link-local (fe80::) GIDs
            bool allZero = (strcmp(gid, "0000:0000:0000:0000:0000:0000:0000:0000") == 0);
            bool linkLocal = (strncmp(gid, "fe80:", 5) == 0);
            if (!allZero && !linkLocal) { found = true; break; }
        }
        closedir(d);
        return found;
    }

    // Helper: create a merged device from N physical NICs.
    // Returns merged device index, or -1 if no suitable group found.
    // Iterates speed groups (fastest-first by enumeration order). Within each
    // group, slides a window of nNicsToMerge; skips windows containing a NIC
    // without a routable GID (those can set up QPs but drop RDMA traffic cross-node).
    // speedGroupStart: index into physDevs indicating which speed group to try first.
    // rank: MPI rank of this process
    int CreateMergedDevice(int nNicsToMerge, int rank, int speedGroupStart = 0)
    {
        if (nNicsToMerge <= 0 || nNicsToMerge > NCCL_NET_MAX_DEVS_PER_NIC) {
            fprintf(stderr,
                    "Rank %d requested invalid merge size %d (valid range: 1..%d)\n",
                    rank, nNicsToMerge, NCCL_NET_MAX_DEVS_PER_NIC);
            return -1;
        }

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

        if (speedGroupStart >= (int)physDevs.size()) return -1;

        // Build the speed group starting at speedGroupStart
        int targetSpeed = props[physDevs[speedGroupStart]].speed;
        std::vector<int> compat;
        for (int d : physDevs)
            if (props[d].speed == targetSpeed) compat.push_back(d);

        // Try each consecutive window of nNicsToMerge within this speed group
        for (int w = 0; w + nNicsToMerge <= (int)compat.size(); w++) {
            bool routable = true;
            for (int i = 0; i < nNicsToMerge; i++) {
                const char* name = props[compat[w + i]].name;
                if (name && !HasRoutableGid(name)) { routable = false; break; }
            }
            if (!routable) continue;

            ncclNetVDeviceProps_t vProps;
            memset(&vProps, 0, sizeof(vProps));
            vProps.ndevs = nNicsToMerge;
            for (int i = 0; i < nNicsToMerge; i++)
                vProps.devs[i] = compat[w + i];

            int outMergedDev = -1;
            if (MakeVirtualDevice(&outMergedDev, &vProps) == ncclSuccess && outMergedDev >= 0)
                return outMergedDev;
        }

        // This speed group exhausted — advance to the next one
        int nextStart = speedGroupStart;
        while (nextStart < (int)physDevs.size() &&
               props[physDevs[nextStart]].speed == targetSpeed)
            nextStart++;

        return CreateMergedDevice(nNicsToMerge, rank, nextStart);
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

    // ===============================================================
    // Stress test infrastructure
    // ===============================================================

    // Process count for multi-rank tests
    static constexpr int kMinFourProcesses = 4;
    // Timeout for stress tests
    static constexpr int kStressTimeoutMs  = 60000;   // 60s

    // ── RDMA resource leak detection ─────────────────────────────────
    struct RdmaResourceCounts {
        int qp = -1;
        int cq = -1;
        int mr = -1;
        int pd = -1;
        bool valid() const { return qp >= 0 && cq >= 0 && mr >= 0 && pd >= 0; }
    };

    static std::string ExecShellCommand(const char* cmd) {
        std::array<char, 256> buf{};
        std::string out;
        FILE* pipe = popen(cmd, "r");
        if (!pipe) return out;
        while (fgets(buf.data(), buf.size(), pipe) != nullptr)
            out += buf.data();
        pclose(pipe);
        return out;
    }

    static int CountNonEmptyLines(const std::string& text) {
        std::istringstream iss(text);
        std::string line;
        int count = 0;
        while (std::getline(iss, line))
            if (!line.empty()) count++;
        return count;
    }

    RdmaResourceCounts CaptureRdmaResources() {
        RdmaResourceCounts counts;
        std::string probe =
            ExecShellCommand("sh -c 'rdma resource show qp >/dev/null 2>&1 && "
                             "rdma resource show cq >/dev/null 2>&1 && "
                             "rdma resource show mr >/dev/null 2>&1 && "
                             "rdma resource show pd >/dev/null 2>&1 && echo OK'");
        if (probe.find("OK") == std::string::npos) return counts;
        // Filter to objects owned by this PID so concurrent processes on shared
        // nodes do not cause spurious leak reports.
        // `rdma resource show` lines contain "pid <N>"; grep for our PID.
        // If the output format doesn't include "pid", fall back to system-wide count.
        const std::string pid = std::to_string(getpid());
        const std::string pidFilter = " pid " + pid + " ";
        auto countOwned = [&](const char* resource) -> int {
            std::string raw = ExecShellCommand(
                (std::string("rdma resource show ") + resource + " 2>/dev/null").c_str());
            // If any line contains our pid, count only those lines.
            if (raw.find(pidFilter) != std::string::npos) {
                std::istringstream iss(raw);
                std::string line;
                int n = 0;
                while (std::getline(iss, line))
                    if (!line.empty() && line.find(pidFilter) != std::string::npos) n++;
                return n;
            }
            // PID not in output — fall back to system-wide count.
            return CountNonEmptyLines(raw);
        };
        counts.qp = countOwned("qp");
        counts.cq = countOwned("cq");
        counts.mr = countOwned("mr");
        counts.pd = countOwned("pd");
        return counts;
    }

    void AssertNoRdmaLeaks(const RdmaResourceCounts& before,
                           const RdmaResourceCounts& after,
                           const char* label = "") {
        int rank = MPIEnvironment::world_rank;
        if (!before.valid() || !after.valid()) {
            GTEST_LOG_(WARNING) << "RDMA resource counting unavailable on this node; "
                                << "leak check skipped for: " << label;
            return;
        }
        EXPECT_EQ(after.qp, before.qp)
            << label << " QP leak on rank " << rank
            << ": before=" << before.qp << " after=" << after.qp;
        EXPECT_EQ(after.cq, before.cq)
            << label << " CQ leak on rank " << rank
            << ": before=" << before.cq << " after=" << after.cq;
        EXPECT_EQ(after.mr, before.mr)
            << label << " MR leak on rank " << rank
            << ": before=" << before.mr << " after=" << after.mr;
        EXPECT_EQ(after.pd, before.pd)
            << label << " PD leak on rank " << rank
            << ": before=" << before.pd << " after=" << after.pd;
    }

    // ── DoSendRecv: single-iteration pattern-verified transfer ──────
    // Both ranks call together. Rank 0 recvs, rank 1 sends.
    // patternSeed is used for both fill and verify.
    void DoSendRecv(void* sendComm, void* recvComm,
                    void* sendBuf, void* recvBuf,
                    size_t size, int tag,
                    void* sendMh, void* recvMh,
                    int patternSeed, int timeoutMs = kDefaultTimeoutMs) {
        const int rank = MPIEnvironment::world_rank;
        void* req = nullptr;

        if (rank == 0) {
            PostSingleRecv(recvComm, recvBuf, size, tag, recvMh, &req);
        } else {
            if (size > 0)
                fillHostBufferWithPattern<uint8_t>(sendBuf, size, makeBytePattern(patternSeed));
            PostSendWithRetry(sendComm, sendBuf, size, tag, sendMh, &req);
        }

        int sz = 0;
        // Use EXPECT_ (non-fatal) so both ranks always reach MPI_Barrier.
        // ASSERT_ here would exit the failing rank before the barrier,
        // leaving the other rank hung indefinitely.
        EXPECT_EQ(WaitForCompletion(req, &sz, timeoutMs), ncclSuccess)
            << "WaitForCompletion failed on rank " << rank << " tag=" << tag;

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0 && size > 0) {
            size_t errIdx; uint8_t errExp, errGot;
            bool ok = verifyHostBufferData<uint8_t>(
                recvBuf, size, makeBytePattern(patternSeed),
                0, 0.0, &errIdx, &errExp, &errGot);
            EXPECT_TRUE(ok) << "Data mismatch at byte " << errIdx
                            << " (tag=" << tag << " seed=" << patternSeed << ")";
        }
    }

    // ── Multi-rank connection helpers ────────────────────────────────
    struct DirectedConnection {
        int senderRank   = -1;
        int receiverRank = -1;
        void* sendComm   = nullptr;  // non-null on senderRank
        void* recvComm   = nullptr;  // non-null on receiverRank
        void* listenComm = nullptr;  // non-null on receiverRank
    };

    // Setup a point-to-point connection between two specific ranks.
    // All ranks must call this together; non-participating ranks only hit the barrier.
    void SetupDirectedConnection(int dev, DirectedConnection& conn,
                                 int senderRank, int receiverRank,
                                 int mpiTag = 0) {
        const int rank = MPIEnvironment::world_rank;
        conn.senderRank   = senderRank;
        conn.receiverRank = receiverRank;
        ncclNetHandle_t handle;
        memset(&handle, 0, sizeof(handle));

        // Use EXPECT_/ADD_FAILURE instead of ASSERT_ so that all ranks always
        // reach MPI_Barrier even when a connection step fails.  ASSERT_ returns
        // immediately on the failing rank, which leaves the other ranks stuck
        // at the barrier indefinitely.
        bool ok = true;
        if (rank == receiverRank) {
            ncclResult_t r = CreateListenComm(dev, &handle, &conn.listenComm);
            EXPECT_EQ(r, ncclSuccess) << "CreateListenComm failed, rank=" << rank;
            EXPECT_NE(conn.listenComm, nullptr);
            ok = (r == ncclSuccess && conn.listenComm != nullptr);
            if (ok) {
                MPI_Send(&handle, sizeof(handle), MPI_BYTE, senderRank, mpiTag, MPI_COMM_WORLD);
                for (int i = 0; i < kMaxRetryAttempts && conn.recvComm == nullptr; i++) {
                    r = AcceptConnection(conn.listenComm, &conn.recvComm);
                    EXPECT_EQ(r, ncclSuccess) << "AcceptConnection failed, rank=" << rank;
                    if (!conn.recvComm) usleep(kPollIntervalUs);
                }
                EXPECT_NE(conn.recvComm, nullptr);
            } else {
                // Send a zeroed handle so the sender doesn't block on MPI_Recv.
                MPI_Send(&handle, sizeof(handle), MPI_BYTE, senderRank, mpiTag, MPI_COMM_WORLD);
            }
        } else if (rank == senderRank) {
            MPI_Recv(&handle, sizeof(handle), MPI_BYTE, receiverRank, mpiTag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < kMaxRetryAttempts && conn.sendComm == nullptr; i++) {
                ncclResult_t r = ConnectToRemote(dev, &handle, &conn.sendComm);
                EXPECT_EQ(r, ncclSuccess) << "ConnectToRemote failed, rank=" << rank;
                if (!conn.sendComm) usleep(kPollIntervalUs);
            }
            EXPECT_NE(conn.sendComm, nullptr);
        }
        // All ranks synchronize — must be reached unconditionally.
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void CloseDirectedConnection(DirectedConnection& conn) {
        const int rank = MPIEnvironment::world_rank;
        if (rank == conn.senderRank && conn.sendComm) {
            CloseSendComm(conn.sendComm);
            conn.sendComm = nullptr;
        }
        if (rank == conn.receiverRank) {
            if (conn.recvComm) {
                CloseRecvComm(conn.recvComm);
                conn.recvComm = nullptr;
            }
            if (conn.listenComm) {
                CloseListenComm(conn.listenComm);
                conn.listenComm = nullptr;
            }
        }
    }

    // Fan-in: multiple senders → one receiver
    void SetupFanIn(int dev, int receiverRank,
                    const std::vector<int>& senderRanks,
                    std::vector<DirectedConnection>& conns) {
        conns.resize(senderRanks.size());
        for (size_t i = 0; i < senderRanks.size(); i++) {
            SetupDirectedConnection(dev, conns[i], senderRanks[i], receiverRank,
                                    /*mpiTag=*/100 + static_cast<int>(i));
        }
    }

    // Fan-out: one sender → multiple receivers
    void SetupFanOut(int dev, int senderRank,
                     const std::vector<int>& receiverRanks,
                     std::vector<DirectedConnection>& conns) {
        conns.resize(receiverRanks.size());
        for (size_t i = 0; i < receiverRanks.size(); i++) {
            SetupDirectedConnection(dev, conns[i], senderRank, receiverRanks[i],
                                    /*mpiTag=*/200 + static_cast<int>(i));
        }
    }

    // All-to-all: N*(N-1) directed connections among numRanks
    void SetupAllToAll(int dev, int numRanks,
                       std::vector<DirectedConnection>& conns) {
        conns.clear();
        for (int src = 0; src < numRanks; src++) {
            for (int dst = 0; dst < numRanks; dst++) {
                if (src == dst) continue;
                DirectedConnection c;
                SetupDirectedConnection(dev, c, src, dst,
                                        /*mpiTag=*/300 + src * numRanks + dst);
                conns.push_back(std::move(c));
            }
        }
    }

    // Do a send/recv on a DirectedConnection. Both ranks call together.
    // senderBuf is used on senderRank; receiverBuf on receiverRank.
    void DoDirectedSendRecv(DirectedConnection& conn,
                            void* senderBuf, void* receiverBuf,
                            size_t size, int tag,
                            void* senderMh, void* receiverMh,
                            int patternSeed, int timeoutMs = kStressTimeoutMs) {
        const int rank = MPIEnvironment::world_rank;
        void* req = nullptr;
        bool postOk = true;

        // Post recv/send with non-fatal checks so we always reach MPI_Barrier.
        // Using ASSERT_ here would skip the barrier on failure, deadlocking
        // all other ranks that are not sender/receiver for this connection.
        if (rank == conn.receiverRank) {
            void*  bufs[1]    = {receiverBuf};
            size_t sizes[1]   = {size};
            int    tags[1]    = {tag};
            void*  handles[1] = {receiverMh};
            ncclResult_t r = PostRecv(conn.recvComm, 1, bufs, sizes, tags, handles, &req);
            EXPECT_EQ(r, ncclSuccess) << "PostRecv failed, rank=" << rank;
            postOk = (r == ncclSuccess && req != nullptr);
        }
        if (rank == conn.senderRank) {
            if (size > 0)
                fillHostBufferWithPattern<uint8_t>(senderBuf, size, makeBytePattern(patternSeed));
            // Retry until FIFO slot is available (receiver hasn't posted yet).
            int attempts = 0;
            ncclResult_t r = ncclSuccess;
            do {
                r = PostSend(conn.sendComm, senderBuf, size, tag, senderMh, &req);
                if (r != ncclSuccess || req != nullptr) break;
                if (++attempts >= kMaxRetryAttempts) {
                    ADD_FAILURE() << "PostSend NULL after " << attempts
                                  << " retries, rank=" << rank << " tag=" << tag;
                    postOk = false;
                    break;
                }
                usleep(kPollIntervalUs);
            } while (req == nullptr);
            if (r != ncclSuccess) {
                ADD_FAILURE() << "PostSend error " << r << ", rank=" << rank;
                postOk = false;
            }
        }

        // Wait for completion — non-fatal so MPI_Barrier is always reached.
        if (req && postOk) {
            int sz = 0;
            ncclResult_t r = WaitForCompletion(req, &sz, timeoutMs);
            EXPECT_EQ(r, ncclSuccess)
                << "DoDirectedSendRecv timeout, rank=" << rank << " tag=" << tag;
        }

        // Unconditional barrier — every rank must reach this even on failure.
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == conn.receiverRank && size > 0 && postOk) {
            size_t errIdx; uint8_t errExp, errGot;
            bool ok = verifyHostBufferData<uint8_t>(
                receiverBuf, size, makeBytePattern(patternSeed),
                0, 0.0, &errIdx, &errExp, &errGot);
            EXPECT_TRUE(ok) << "Data mismatch at byte " << errIdx
                            << " (tag=" << tag << " seed=" << patternSeed << ")";
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

    // Poll req until done or maxPolls exhausted. Safe to call with req==nullptr.
    // Does not assert completion — the recv may legitimately time out when
    // the sender faulted.
    void DrainRecvRequest(void* req, int maxPolls = 500) {
        if (req == nullptr) return;
        for (int poll = 0; poll < maxPolls; ++poll) {
            int done = 0, sz = 0;
            if (TestRequest(req, &done, &sz) != ncclSuccess) break;
            if (done) break;
            usleep(kPollIntervalUs);
        }
    }

    // Deregister mhandle, close comms rank-conditionally, barrier.
    // rank 0 closes recvComm + listenComm; rank 1 closes sendComm.
    // Call after any pre-teardown MPI_Barrier the test needs.
    void TeardownConnection(void* recvComm, void* listenComm,
                            void* sendComm, void* mhandle) {
        const int rank = MPIEnvironment::world_rank;
        void* comm = (rank == 0) ? recvComm : sendComm;
        ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
        if (rank == 0) {
            ASSERT_EQ(CloseRecvComm(recvComm), ncclSuccess);
            ASSERT_EQ(CloseListenComm(listenComm), ncclSuccess);
        } else {
            ASSERT_EQ(CloseSendComm(sendComm), ncclSuccess);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Assert WRR scheduler has been initialised with equal-weight tokens.
    // Checks: schedInit==true, nqps==expectedNqps, initTotTokens==100,
    //         each QP >= floor(100/nqps), sum(initQpTokens)==initTotTokens.
    // Call from rank 1 (sender) — state comes from sendComm.
    void ExpectEqualWeightInitTokens(const ncclIbCastSchedState& state, int expectedNqps) {
        ASSERT_TRUE(state.schedInit);
        EXPECT_EQ(state.nqps, expectedNqps);
        EXPECT_EQ(state.initTotTokens, 100);
        const int base = 100 / expectedNqps;
        for (int i = 0; i < state.nqps; i++)
            EXPECT_GE(state.initQpTokens[i], base)
                << "QP " << i << " initToken below equal-weight floor";
        int sum = 0;
        for (int i = 0; i < state.nqps; i++) sum += state.initQpTokens[i];
        EXPECT_EQ(sum, state.initTotTokens);
    }

    // Assert sum(activeQpTokens) == activeTotTokens.
    void ExpectActiveTokenSumInvariant(const ncclIbCastSchedState& state) {
        int activeSum = 0;
        for (int i = 0; i < state.nqps; i++) activeSum += state.activeQpTokens[i];
        EXPECT_EQ(activeSum, state.activeTotTokens);
    }
};

// ============================================================================
// CTS hw_counters helpers (for CtsDepthStress and friends).
// Snapshots /sys/class/infiniband/<dev>/ports/<N>/hw_counters/ and the
// device-level /sys/class/infiniband/<dev>/hw_counters/ for CTS deltas.
// ============================================================================
namespace NetIbCts {

using CounterMap = std::map<std::string, long long>;

inline const std::vector<std::string>& kCtsKeywords() {
    static const std::vector<std::string> v = {
        "cts_pkts", "cts_bytes",
        "cts_retx",          // retransmit = overflow signal
        "cts_ack_timeout",   // ACK timeout = overflow signal
        "cts_miss", "cts_cache", "cts_match",
        "nak", "rdma_ccl",
    };
    return v;
}

inline bool isCtsRelevant(const std::string& name) {
    std::string lower(name.size(), '\0');
    std::transform(name.begin(), name.end(), lower.begin(),
                   [](char c) { return static_cast<char>(
                       std::tolower(static_cast<unsigned char>(c))); });
    for (const auto& kw : kCtsKeywords())
        if (lower.find(kw) != std::string::npos) return true;
    return false;
}

inline void readCountersDir(const std::string& dir,
                            const std::string& keyPrefix,
                            CounterMap& out) {
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        if (ent->d_name[0] == '.') continue;
        std::ifstream f(dir + "/" + ent->d_name);
        long long val = 0;
        if (f >> val)
            out[keyPrefix + "/" + ent->d_name] = val;
    }
    closedir(d);
}

inline std::vector<std::string> listDirEntries(const std::string& dir) {
    std::vector<std::string> entries;
    DIR* d = opendir(dir.c_str());
    if (!d) return entries;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr)
        if (ent->d_name[0] != '.') entries.push_back(ent->d_name);
    closedir(d);
    std::sort(entries.begin(), entries.end());
    return entries;
}

inline CounterMap readHwCounters(const std::string& ibdev) {
    CounterMap result;
    const std::string devBase = "/sys/class/infiniband/" + ibdev;
    for (const auto& portName : listDirEntries(devBase + "/ports")) {
        readCountersDir(devBase + "/ports/" + portName + "/hw_counters",
                        ibdev + "/port" + portName, result);
    }
    readCountersDir(devBase + "/hw_counters", ibdev + "/dev", result);
    return result;
}

inline std::vector<std::string> listIbDevices() {
    return listDirEntries("/sys/class/infiniband");
}

inline CounterMap takeSnapshot() {
    CounterMap snap;
    for (const auto& dev : listIbDevices()) {
        auto m = readHwCounters(dev);
        snap.insert(m.begin(), m.end());
    }
    return snap;
}

inline std::string formatSignedDelta(long long delta) {
    return (delta >= 0 ? "+" : "") + std::to_string(delta);
}

inline void printDelta(int rank,
                       const std::string& fromLabel,
                       const std::string& toLabel,
                       const CounterMap& before,
                       const CounterMap& after) {
    struct Row { std::string name; long long vb, va, delta; };
    std::vector<Row> rows;
    for (const auto& kv : after) {
        if (!isCtsRelevant(kv.first)) continue;
        long long vb = 0;
        auto it = before.find(kv.first);
        if (it != before.end()) vb = it->second;
        long long delta = kv.second - vb;
        if (delta != 0)
            rows.push_back({kv.first, vb, kv.second, delta});
    }

    const int wName = 55, wVal = 12, wDelta = 12;
    std::cout << "\n[Rank " << rank << "] "
              << fromLabel << " -> " << toLabel << "\n";
    if (rows.empty()) {
        std::cout << "  (no CTS-relevant changes)\n" << std::flush;
        return;
    }
    std::cout << "  " << std::left  << std::setw(wName)  << "counter"
              <<         std::right << std::setw(wVal)   << "before"
              <<         std::right << std::setw(wVal)   << "after"
              <<         std::right << std::setw(wDelta) << "delta"
              << "\n  " << std::string(wName + wVal + wVal + wDelta, '-') << "\n";
    for (const auto& r : rows)
        std::cout << "  " << std::left  << std::setw(wName)  << r.name
                  <<         std::right << std::setw(wVal)   << r.vb
                  <<         std::right << std::setw(wVal)   << r.va
                  <<         std::right << std::setw(wDelta) << formatSignedDelta(r.delta)
                  << "\n";
    std::cout << std::flush;
}

struct SnapSummary {
    long long pkts        = 0;
    long long bytes       = 0;
    long long retx_pkts   = 0;  // cts_retx_pkts   - overflow signal
    long long ack_timeout = 0;  // cts_ack_timeout - overflow signal
};

inline SnapSummary calcSummary(const CounterMap& before, const CounterMap& after) {
    SnapSummary s;
    for (const auto& kv : after) {
        auto it = before.find(kv.first);
        long long d = kv.second - (it != before.end() ? it->second : 0LL);
        if (d == 0) continue;
        const std::string& n = kv.first;
        if (n.find("cts_retx_pkts")    != std::string::npos) s.retx_pkts   += d;
        if (n.find("cts_ack_timeout")  != std::string::npos) s.ack_timeout += d;
        if (n.find("cts_pkts")         != std::string::npos &&
            n.find("retx")             == std::string::npos) s.pkts        += d;
        if (n.find("cts_bytes")        != std::string::npos &&
            n.find("retx")             == std::string::npos) s.bytes       += d;
    }
    return s;
}

inline void printSummary(int rank,
                         const std::string& label,
                         int connsDone,
                         int qpDepth,
                         const CounterMap& before,
                         const CounterMap& after) {
    auto s = calcSummary(before, after);
    long long bpp = s.pkts > 0 ? s.bytes / s.pkts : 0;

    std::cout << "[Rank " << rank << "]"
              << "  conns=" << std::setw(4) << connsDone
              << "  entries=" << std::setw(6) << (connsDone * qpDepth)
              << "  cts_pkts=" << std::setw(6) << s.pkts
              << "  bytes/pkt=" << bpp
              << "  retx=" << s.retx_pkts
              << "  ack_timeout=" << s.ack_timeout;
    if (s.retx_pkts > 0 || s.ack_timeout > 0) {
        std::cout << "  <<< OVERFLOW at ~"
                  << connsDone * qpDepth << " entries >>>";
    }
    std::cout << "  [" << label << "]\n" << std::flush;
}

}  // namespace NetIbCts

#endif /* MPI_TESTS_ENABLED */

#endif /* RCCL_TEST_NET_IB_MPI_TEST_BASE_HPP_ */
