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
#include "NetIbFaultInject.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "DeviceBufferHelpers.hpp"
#include "HostBufferHelpers.hpp"
#include "nccl.h"
#include "net.h"
#include "plugin/nccl_net.h"
#include <atomic>
#include <chrono>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <condition_variable>
#include <dirent.h>
#include <unistd.h>
#include <thread>
#include <functional>

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
// External NET IB-CAST plugin (WRR scheduler, multi-QP, AINIC features)
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

// Deregistrations that failed inside a worker. A destructor cannot fail a test,
// and swallowing the result would turn a refused deregistration into a phantom
// MR leak later, so the count is recorded here and the harness checks it on the
// main thread once the workers have joined. The serial teardown asserts on the
// same return value.
//
// Process-global, and deliberately so: it is written by every worker of every
// threaded body, and a per-fixture member would need threading through helpers
// that have no fixture. That makes it correct only under two properties, both of
// which hold and neither of which is free. GTest runs the tests in a translation
// unit one after another on the calling thread, so no two bodies are counting at
// once; and every threaded body joins its workers before returning, so no worker
// from a finished test can still be writing. Each run resets the counter first,
// which limits a stray increment to the test that produced it rather than
// blaming the next one -- but if either property is ever given up (a
// GTest-parallel runner, or a body that abandons a worker on timeout), this
// counter becomes wrong, not merely imprecise.
inline std::atomic<int> g_workerDeregFailures{0};

// Worker threads must not invoke TEST_INFO or any helper that can call MPI.
// This deleter is used only by the threaded test bodies.
struct NetMHandleWorkerDeleter {
    ncclNet_t* net;
    void* comm;

    NetMHandleWorkerDeleter(ncclNet_t* n = nullptr, void* c = nullptr) : net(n), comm(c) {}

    void operator()(void* mhandle) const {
        if (mhandle && net && comm) {
            if (net->deregMr(comm, mhandle) != ncclSuccess) {
                g_workerDeregFailures.fetch_add(1, std::memory_order_relaxed);
            }
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
using NetMHandleWorkerGuard = RCCLTestGuards::ResourceGuard<void*, NetMHandleWorkerDeleter>;

// Test fixture for NET IB tests
class NetIbMPITest : public MPITestBase {
protected:
    static constexpr int kMinProcessesForMPI = 2;
    static constexpr bool kRequirePowerOfTwo = true;
    static constexpr int kNoNodeLimit = MPITestConstants::kNoNodeLimit;

    // Timing constants
    static constexpr int kDefaultTimeoutMs = 5000;
    static constexpr int kLargeTransferTimeoutMs = 30000;
    static constexpr int kConnectTimeoutMs = 30000;  // Handshake watchdog (see SetupConnection)
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
        // Zero-initialized: SetupConnectionForThread sends the handle
        // unconditionally (so a listen() failure can't strand the peer's
        // MPI_Recv), which would otherwise put uninitialized stack bytes on
        // the wire and trip sanitizers.
        ncclNetHandle_t handle{};
    };

    ncclResult_t SetupConnection(int dev, ConnectionPair& pair, int rank, int peerRank) {
        // Cap the accept/connect handshake so a dead fabric fails fast instead
        // of spinning forever (AICOMRCCL-1577).
        const int maxAttempts = kConnectTimeoutMs / kPollIntervalMs;
        if (rank == 0) {
            // Rank 0: Listen
            RCCL_TEST_CHECK(CreateListenComm(dev, &pair.handle, &pair.listenComm));

            // Send handle to peer
            MPI_Send(&pair.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD);

            // Accept connection
            int done = 0;
            int attempts = 0;
            while (!done) {
                ncclResult_t result = AcceptConnection(pair.listenComm, &pair.recvComm);
                if (result != ncclSuccess) {
                    return result;
                }
                if (pair.recvComm != nullptr) {
                    done = 1;
                    break;
                }
                if (++attempts >= maxAttempts) {
                    return ncclInternalError;
                }
                usleep(kPollIntervalUs);
            }
        } else {
            // Rank 1: Connect
            MPI_Recv(&pair.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Connect to peer
            int done = 0;
            int attempts = 0;
            while (!done) {
                ncclResult_t result = ConnectToRemote(dev, &pair.handle, &pair.sendComm);
                if (result != ncclSuccess) {
                    return result;
                }
                if (pair.sendComm != nullptr) {
                    done = 1;
                    break;
                }
                if (++attempts >= maxAttempts) {
                    return ncclInternalError;
                }
                usleep(kPollIntervalUs);
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

    // Count physical IB devices only. The plugin's devices()/GetDeviceCount
    // returns ncclNMergedIbDevs, which GROWS as makeVDevice creates virtual
    // NICs — so it is order-dependent across tests in the same process. Note a
    // single-device vNIC (e.g. a deduped merge) also reports vProps.ndevs == 1,
    // so ndevs alone cannot tell it apart from a physical NIC. Each physical NIC
    // is registered with a unique underlying index in vProps.devs[0], while a
    // 1-device vNIC reuses an existing physical index — so the count of DISTINCT
    // single-device vProps.devs[0] values is the true physical device count.
    int GetPhysicalDeviceCount() {
        int n = 0;
        if (GetDeviceCount(&n) != ncclSuccess) return 0;
        std::set<int> physIdx;
        for (int i = 0; i < n; i++) {
            ncclNetProperties_t props;
            memset(&props, 0, sizeof(props));
            if (GetDeviceProperties(i, &props) != ncclSuccess) continue;
            if (props.vProps.ndevs <= 1) physIdx.insert(props.vProps.devs[0]);
        }
        return (int)physIdx.size();
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

    static bool PortIsEthernet(const char* portsPath, const char* port) {
        char path[PATH_MAX];
        if (snprintf(path, sizeof(path), "%s/%s/link_layer", portsPath, port) >= (int)sizeof(path))
            return false;

        FILE* linkLayerFile = fopen(path, "r");
        if (!linkLayerFile) return false;

        char linkLayer[32] = {};
        bool linkLayerRead = (fscanf(linkLayerFile, "%31s", linkLayer) == 1);
        fclose(linkLayerFile);

        return linkLayerRead && strcmp(linkLayer, "Ethernet") == 0;
    }

    // familiesOut, when given, accumulates kGidFamily* over every routable GID on
    // the port rather than the first one found. Which single GID the plugin will
    // pick is not knowable from here -- it depends on NCCL_IB_GID_INDEX,
    // NCCL_IB_ADDR_FAMILY, the RoCE version files and prefix matching
    // (connect.cc) -- so sampling one and calling it "the" family would be an
    // arbitrary choice that can disagree with the plugin's on a dual-stack port.
    // The whole set carries no such assumption.
    static bool PortHasRoutableGid(const char* portsPath, const char* port,
                                   unsigned* familiesOut = nullptr) {
        char path[PATH_MAX];
        if (snprintf(path, sizeof(path), "%s/%s/gids", portsPath, port) >= (int)sizeof(path))
            return false;

        DIR* gidDir = opendir(path);
        if (!gidDir) return false;

        struct dirent* ent;
        bool found = false;
        // Scanning stops at the first routable GID unless the caller wants the
        // family set, which needs all of them.
        while ((!found || familiesOut) && (ent = readdir(gidDir)) != nullptr) {
            if (ent->d_name[0] == '.') continue;
            char gidPath[PATH_MAX];
            if (snprintf(gidPath, sizeof(gidPath), "%s/%s", path, ent->d_name) >= (int)sizeof(gidPath))
                continue;
            FILE* f = fopen(gidPath, "r");
            if (!f) continue;
            char gid[64] = {};
            bool gidRead = (fscanf(f, "%63s", gid) == 1);
            fclose(f);
            if (!gidRead) continue;
            // Skip all-zero GIDs and link-local (fe80::) GIDs
            bool allZero = (strcmp(gid, "0000:0000:0000:0000:0000:0000:0000:0000") == 0);
            bool linkLocal = (strncmp(gid, "fe80:", 5) == 0);
            if (allZero || linkLocal) continue;
            found = true;
            // A GID that cannot be parsed still counts as routable -- that is the
            // decision this function has always made -- it just adds no family, so
            // a port whose GIDs are all unreadable stays uncomparable and the
            // caller falls back to index symmetry alone.
            uint8_t raw[16] = {};
            if (familiesOut && ParseGidText(gid, raw)) *familiesOut |= GidAddrFamilyBit(raw);
        }
        closedir(gidDir);
        return found;
    }

    // Returns true unless the device is RoCE with no routable GID on any port: such a port
    // completes QP setup and then silently drops cross-node RDMA traffic. On InfiniBand every
    // GID is link-local, so the GID table says nothing about routability there. When the device
    // cannot be inspected, assume it is usable rather than dropping the NIC.
    //
    // The ports come from the device directory rather than ncclNetProperties_t::port, which the
    // plugin sets to portNum + realPort. realPort counts VF siblings on one PCI path, so for
    // every VF past the first it names a port that does not exist in sysfs.
    // familiesOut, when given, collects the address families of every routable GID
    // on the device, so the ranks can check they have one in common rather than
    // only agreeing that each end has something routable. It is left empty when
    // the answer is "routable" for any other reason -- no ports directory, or no
    // Ethernet port at all, i.e. a real IB link -- and the caller reads an empty
    // set as "cannot tell".
    static bool CanRouteCrossNode(const char* devName, unsigned* familiesOut = nullptr) {
        char portsPath[PATH_MAX];
        if (snprintf(portsPath, sizeof(portsPath), "/sys/class/infiniband/%s/ports", devName)
            >= (int)sizeof(portsPath))
            return true;

        DIR* portsDir = opendir(portsPath);
        if (!portsDir) return true;

        int ethernetPorts = 0;
        bool routable = false;
        struct dirent* ent;
        // As in PortHasRoutableGid: the family set needs every port looked at, the
        // routable verdict alone does not.
        while ((!routable || familiesOut) && (ent = readdir(portsDir)) != nullptr) {
            if (ent->d_name[0] == '.') continue;
            if (!PortIsEthernet(portsPath, ent->d_name)) continue;
            ethernetPorts++;
            if (PortHasRoutableGid(portsPath, ent->d_name, familiesOut)) routable = true;
        }
        closedir(portsDir);

        return routable || ethernetPorts == 0;
    }

    // sysfs writes a GID as eight colon-separated 16-bit groups. ibv_gid is those
    // same sixteen bytes in network order, which is what the plugin's subnet
    // comparison expects.
    static bool ParseGidText(const char* text, uint8_t out[16]) {
        unsigned g[8];
        if (sscanf(text, "%4x:%4x:%4x:%4x:%4x:%4x:%4x:%4x", &g[0], &g[1], &g[2], &g[3], &g[4],
                   &g[5], &g[6], &g[7]) != 8)
            return false;
        for (int i = 0; i < 8; i++) {
            out[2 * i]     = static_cast<uint8_t>((g[i] >> 8) & 0xFF);
            out[2 * i + 1] = static_cast<uint8_t>(g[i] & 0xFF);
        }
        return true;
    }

    // What CreateMergedDevice() tried before giving up. Empty after a success.
    // Callers put it in their GTEST_SKIP() message, so a skip states which speed
    // groups existed and why each one was rejected.
    std::string mergeSkipReason_;

    void AppendMergeSkipReason(const std::string& clause) {
        if (!mergeSkipReason_.empty()) mergeSkipReason_ += "; ";
        mergeSkipReason_ += clause;
    }

    // Helper: create a merged device from N physical NICs.
    // Returns merged device index, or -1 if no suitable group found, in which case
    // mergeSkipReason_ describes the attempt.
    // Iterates speed groups in order of first appearance. Within each group, slides
    // a window of nNicsToMerge; skips windows containing a NIC that cannot carry
    // cross-node RDMA traffic.
    // speedGroupStart: index into physDevs indicating which speed group to try first.
    int CreateMergedDevice(int nNicsToMerge, int speedGroupStart = 0)
    {
        mergeSkipReason_.clear();

        if (nNicsToMerge <= 0 || nNicsToMerge > NCCL_NET_MAX_DEVS_PER_NIC) {
            mergeSkipReason_ = "invalid merge size " + std::to_string(nNicsToMerge) +
                               " (valid range: 1.." + std::to_string(NCCL_NET_MAX_DEVS_PER_NIC) + ")";
            return -1;
        }

        int mergedDev = CreateMergedDeviceFromSpeedGroup(nNicsToMerge, speedGroupStart);
        if (mergedDev < 0)
            mergeSkipReason_ = "no group of " + std::to_string(nNicsToMerge) +
                               " mergeable NICs: " + mergeSkipReason_;
        else
            mergeSkipReason_.clear();
        return mergedDev;
    }

    int CreateMergedDeviceFromSpeedGroup(int nNicsToMerge, int speedGroupStart)
    {
        int ndev = 0;
        RCCL_TEST_CHECK(GetDeviceCount(&ndev));
        if (ndev <= 0) {
            AppendMergeSkipReason("the plugin reports no devices");
            return -1;
        }

        std::vector<ncclNetProperties_t> props(ndev);
        std::vector<int> physDevs;
        for (int i = 0; i < ndev; i++) {
            memset(&props[i], 0, sizeof(ncclNetProperties_t));
            RCCL_TEST_CHECK(GetDeviceProperties(i, &props[i]));
            if (!props[i].name || !strchr(props[i].name, '+'))
                physDevs.push_back(i);
        }

        if (speedGroupStart >= (int)physDevs.size()) {
            AppendMergeSkipReason(std::to_string(physDevs.size()) +
                                  " physical NICs, none from index " +
                                  std::to_string(speedGroupStart) + " on");
            return -1;
        }

        // A speed group is every NIC at that speed, wherever it sits in physDevs, so walking the
        // start index would revisit a group whose members are not contiguous.
        std::set<int> triedSpeeds;

        for (int start = speedGroupStart; start < (int)physDevs.size(); start++) {
            int targetSpeed = props[physDevs[start]].speed;
            if (!triedSpeeds.insert(targetSpeed).second) continue;

            std::vector<int> compat;
            for (int d : physDevs)
                if (props[d].speed == targetSpeed) compat.push_back(d);

            int windowsTried = 0;
            int windowsUnroutable = 0;
            int windowsRefused = 0;
            std::string firstUnroutableNic;

            // Try each consecutive window of nNicsToMerge within this speed group
            for (int w = 0; w + nNicsToMerge <= (int)compat.size(); w++) {
                windowsTried++;
                bool routable = true;
                for (int i = 0; i < nNicsToMerge; i++) {
                    const ncclNetProperties_t& dev = props[compat[w + i]];
                    if (dev.name && !CanRouteCrossNode(dev.name)) {
                        routable = false;
                        if (firstUnroutableNic.empty()) firstUnroutableNic = dev.name;
                        break;
                    }
                }
                if (!routable) { windowsUnroutable++; continue; }

                ncclNetVDeviceProps_t vProps;
                memset(&vProps, 0, sizeof(vProps));
                vProps.ndevs = nNicsToMerge;
                for (int i = 0; i < nNicsToMerge; i++)
                    vProps.devs[i] = compat[w + i];

                int outMergedDev = -1;
                if (MakeVirtualDevice(&outMergedDev, &vProps) == ncclSuccess && outMergedDev >= 0)
                    return outMergedDev;
                windowsRefused++;
            }

            std::ostringstream clause;
            if (windowsTried == 0) {
                clause << compat.size() << " of the " << nNicsToMerge
                       << " NICs required at speed " << targetSpeed;
            } else {
                clause << compat.size() << " NICs at speed " << targetSpeed << ": "
                       << windowsTried << " windows tried, " << windowsUnroutable << " unroutable";
                if (!firstUnroutableNic.empty()) clause << " (" << firstUnroutableNic << ")";
                clause << ", " << windowsRefused << " refused by makeVDevice";
            }
            AppendMergeSkipReason(clause.str());
        }

        return -1;
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
    // Multithreading helpers
    //
    // `rccl-tests -t` gives every host worker a distinct communicator. NetIB
    // validation follows that contract: workers operate on independent
    // send/recv comm pairs, not concurrently on one comm object. The latter
    // is outside the NCCL communicator thread-safety contract and would turn
    // a test into a C++ data race instead of useful validation.
    //
    // All MPI calls remain on the GTest/main thread. This avoids requiring
    // MPI_THREAD_MULTIPLE and gives every phase an explicit rank-wide failure
    // handshake before a peer can be stranded in a blocking MPI operation.
    // Worker bodies must not call fatal GTest macros (ASSERT_*/FAIL()); they
    // return ThreadResult and are reported after all workers join.
    // ===============================================================

    struct ThreadResult {
        bool ok = true;
        std::string msg;
    };

    struct ThreadConnection {
        void* ctx = nullptr;
        ConnectionPair pair;
    };

    class ThreadStartGate {
    public:
        explicit ThreadStartGate(int expected) : expected_(expected) {}

        bool ArriveAndWait() {
            std::unique_lock<std::mutex> lock(mutex_);
            if (cancelled_) return false;
            if (++arrived_ == expected_) {
                released_ = true;
                condition_.notify_all();
                return true;
            }
            condition_.wait(lock, [&] { return released_ || cancelled_; });
            return !cancelled_;
        }

        void Cancel() {
            std::lock_guard<std::mutex> lock(mutex_);
            cancelled_ = true;
            condition_.notify_all();
        }

    private:
        const int expected_;
        int arrived_ = 0;
        bool released_ = false;
        bool cancelled_ = false;
        std::mutex mutex_;
        std::condition_variable condition_;
    };

    struct ThreadWorkerRun {
        std::vector<ThreadResult> results;
        std::vector<std::thread::id> threadIds;
        int maxConcurrentWorkers = 0;
    };

    // MPI tags used by the main-thread handle exchanges for independently
    // created connections. MPI only guarantees MPI_TAG_UB >= 32767.
    static constexpr int kThreadTagStride = 1000;
    static constexpr int kMaxThreadTagOffset = 1; // listener-ready flag + handle
    static constexpr int kMpiGuaranteedTagUb = 32767;
    // Upper bound on connections a single worker may own; each one consumes its
    // own tag stride during the main-thread handle exchange.
    static constexpr int kMaxConnsPerWorker = 2;

    static_assert((MPIEnvironment::kMaxThreads * kMaxConnsPerWorker - 1) * kThreadTagStride
                          + kMaxThreadTagOffset
                      <= kMpiGuaranteedTagUb,
                  "worst-case per-thread MPI tag must fit in the tag range every "
                  "MPI implementation is required to provide");

    static void UpdateMaximum(std::atomic<int>& maximum, int value) {
        int observed = maximum.load(std::memory_order_relaxed);
        while (observed < value
               && !maximum.compare_exchange_weak(observed, value, std::memory_order_relaxed)) {
        }
    }

    ThreadWorkerRun RunThreadWorkers(int nThreads, std::function<ThreadResult(int)> body) {
        ThreadWorkerRun run;
        run.results.resize(nThreads);
        run.threadIds.resize(nThreads);

        // The HIP current device is thread-local, and MPIEnvironment binds it
        // once on the main thread. Without propagating it, a worker touching GPU
        // memory would land on device 0 regardless of this rank's assignment.
        int hipDevice = -1;
        // Kept, not swallowed: treating a failed query as "no device to propagate"
        // would let every worker run on the default device, which is the exact
        // situation this propagation exists to prevent.
        const hipError_t deviceQueryError = hipGetDevice(&hipDevice);

        ThreadStartGate startGate(nThreads);
        ThreadStartGate bodyGate(nThreads);
        std::atomic<int> inFlight{0};
        std::atomic<int> maxInFlight{0};
        auto worker = [&](int threadIdx) {
            run.threadIds[threadIdx] = std::this_thread::get_id();
            // Reported only after both gates below: leaving early would strand
            // the sibling workers waiting on them.
            const hipError_t deviceError = (deviceQueryError != hipSuccess)
                                               ? deviceQueryError
                                               : hipSetDevice(hipDevice);
            if (!startGate.ArriveAndWait()) {
                run.results[threadIdx].ok = false;
                run.results[threadIdx].msg = "worker launch was cancelled";
                return;
            }
            const int active = inFlight.fetch_add(1, std::memory_order_relaxed) + 1;
            UpdateMaximum(maxInFlight, active);
            // Do not let an eagerly scheduled worker complete its body before
            // the other workers have actually entered it. This makes the
            // concurrency assertion deterministic rather than scheduler-timing
            // dependent.
            if (!bodyGate.ArriveAndWait()) {
                inFlight.fetch_sub(1, std::memory_order_relaxed);
                run.results[threadIdx].ok = false;
                run.results[threadIdx].msg = "worker body start was cancelled";
                return;
            }
            if (deviceError != hipSuccess) {
                inFlight.fetch_sub(1, std::memory_order_relaxed);
                run.results[threadIdx].ok = false;
                run.results[threadIdx].msg =
                    (deviceQueryError != hipSuccess)
                        ? std::string("hipGetDevice failed on the main thread, so the rank's "
                                      "device could not be propagated into this worker: ")
                              + hipGetErrorString(deviceQueryError)
                        : "hipSetDevice(" + std::to_string(hipDevice)
                              + ") failed in the worker: " + hipGetErrorString(deviceError);
                return;
            }
            try {
                run.results[threadIdx] = body(threadIdx);
            } catch (const std::exception& error) {
                run.results[threadIdx].ok = false;
                run.results[threadIdx].msg = std::string("worker threw: ") + error.what();
            } catch (...) {
                run.results[threadIdx].ok = false;
                run.results[threadIdx].msg = "worker threw a non-standard exception";
            }
            inFlight.fetch_sub(1, std::memory_order_relaxed);
        };

        std::vector<std::thread> workers;
        try {
            for (int t = nThreads - 1; t >= 1; --t) workers.emplace_back(worker, t);
            worker(0);
        } catch (const std::exception& error) {
            startGate.Cancel();
            bodyGate.Cancel();
            run.results[0].ok = false;
            run.results[0].msg = std::string("failed to launch worker: ") + error.what();
        }
        for (auto& workerThread : workers) workerThread.join();

        run.maxConcurrentWorkers = maxInFlight.load(std::memory_order_relaxed);
        return run;
    }

    void VerifyThreadFanOut(const ThreadWorkerRun& run, int nThreads) {
        const std::set<std::thread::id> distinct(run.threadIds.begin(), run.threadIds.end());
        if (static_cast<int>(distinct.size()) != nThreads) {
            ADD_FAILURE() << "expected " << nThreads
                          << " distinct worker threads, observed " << distinct.size()
                          << " — thread fan-out did not happen";
        }
        // bodyGate holds every worker at the top of the body until all N have
        // arrived, so the peak is deterministically N rather than merely ">= 2".
        // Asserting the exact value also catches a regression in the gate itself
        // (a gate that stopped blocking would still let two workers overlap by
        // chance and satisfy a ">= 2" check).
        if (nThreads > 1 && run.maxConcurrentWorkers != nThreads) {
            ADD_FAILURE() << "expected " << nThreads
                          << " workers concurrently inside the body, observed at most "
                          << run.maxConcurrentWorkers
                          << " — the start/body gate did not hold all workers";
        }
    }

    // Reduces worker outcomes across ranks and reports them on rank 0, which is
    // the only rank with GTest listeners. The failing worker's message travels
    // with the flag: without that, a failure on a non-zero rank shows up as
    // "failed on a peer rank" and the actual reason is lost, since non-zero ranks
    // have their output muted unless RCCL_MPI_LOG_ALL_RANKS is set.
    bool SynchronizeThreadResults(const std::vector<ThreadResult>& results, const char* phase) {
        static constexpr int kResultTextBytes = 512;

        int localFailed = 0;
        int localFailures = 0;
        int omitted = 0;
        std::string localText;
        for (size_t threadIdx = 0; threadIdx < results.size(); ++threadIdx) {
            if (results[threadIdx].ok) continue;
            localFailed = 1;
            localFailures++;
            // The text crosses in a fixed-size MPI_Allgather, so it has to be
            // bounded. A mass failure is usually the same message N times, and the
            // first few carry it -- but a reader must not mistake a truncated list
            // for the whole one, so the number left out travels with it.
            if (localText.size() < kResultTextBytes / 2) {
                localText += "thread " + std::to_string(threadIdx) + ": "
                             + results[threadIdx].msg + "; ";
            } else {
                omitted++;
            }
        }
        if (omitted > 0) {
            // Prepended, not appended: the snprintf below cuts the tail, which is
            // where a note about truncation would be the first thing lost.
            localText = "[" + std::to_string(localFailures) + " worker(s) failed on this rank, "
                        + std::to_string(omitted) + " message(s) not shown] " + localText;
        }

        int globalFailed = 0;
        if (MPI_Allreduce(&localFailed, &globalFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD)
            != MPI_SUCCESS) {
            ADD_FAILURE() << phase << ": MPI_Allreduce failed";
            return false;
        }
        if (!globalFailed) return true;

        std::vector<char> sendText(kResultTextBytes, '\0');
        snprintf(sendText.data(), kResultTextBytes, "%s", localText.c_str());
        std::vector<char> allText(kResultTextBytes * MPIEnvironment::world_size, '\0');
        if (MPI_Allgather(sendText.data(), kResultTextBytes, MPI_CHAR, allText.data(),
                          kResultTextBytes, MPI_CHAR, MPI_COMM_WORLD) != MPI_SUCCESS) {
            ADD_FAILURE() << phase << ": MPI_Allgather of worker messages failed";
            return false;
        }

        for (int rank = 0; rank < MPIEnvironment::world_size; ++rank) {
            const char* text = allText.data() + rank * kResultTextBytes;
            if (text[0] == '\0') continue;
            ADD_FAILURE() << phase << ", rank " << rank << ": " << text;
        }
        return false;
    }

    // ── Worker-safe data path ────────────────────────────────────────
    //
    // The ordinary transfer helpers (PostSendWithRetry, DoSendRecv, CastDoSendRecv)
    // barrier and assert fatally, so a worker cannot use them: a fatal assertion
    // only returns from the worker, and a worker-side collective strands the peer
    // rank when a sibling fails. These return ThreadResult instead, and the main
    // thread reports it after joining.
    // ────────────────────────────────────────────────────────────────

    ThreadResult WorkerRegister(void* comm, void* data, size_t size, int type, void** mhandle) {
        ThreadResult result;
        if (RegisterMemory(comm, data, size, type, mhandle) != ncclSuccess || *mhandle == nullptr) {
            result.ok = false;
            result.msg = "regMr failed";
        }
        return result;
    }

    // isend returns success with a null request until the receiver's FIFO slot is
    // available, so the caller has to retry. busyPoll retries without sleeping,
    // for timing tests: with the backoff, waiting for the slot costs one poll
    // interval and that interval, not the plugin, is what a measurement then
    // reports.
    //
    // Both paths stop at timeoutMs. Bounding the sleeping one by a retry count
    // instead made the wait depend on which path a caller took -- a caller asking
    // for a longer timeout got the retry cap anyway, and one asking for a shorter
    // timeout waited past it. The default is that former cap, so the behaviour a
    // caller gets without saying anything is unchanged.
    static constexpr int kSlotWaitDefaultMs = kMaxRetryAttempts * kPollIntervalMs;

    // nullRetries, when given, counts how many times isend came back without a
    // request. FifoPressureSenderFast reports that count in its failure message
    // rather than asserting on it: nothing orders the sender's attempt against the
    // receiver publishing a slot, so a descheduled sender can legitimately see
    // none. Without the count a test whose subject is backpressure would have to
    // reimplement this loop just to observe it.
    ThreadResult WorkerPostSend(void* sendComm, void* data, size_t size, int tag,
                                void* mhandle, void** request, bool busyPoll = false,
                                int timeoutMs = kSlotWaitDefaultMs,
                                int* nullRetries = nullptr) {
        ThreadResult result;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        do {
            if (PostSend(sendComm, data, size, tag, mhandle, request) != ncclSuccess) {
                result.ok = false;
                result.msg = "isend failed";
                return result;
            }
            if (*request != nullptr) break;
            if (nullRetries) (*nullRetries)++;
            if (std::chrono::steady_clock::now() >= deadline) {
                result.ok = false;
                result.msg = "isend kept returning a NULL request for "
                             + std::to_string(timeoutMs) + " ms: the receiver never "
                             "published a FIFO slot";
                return result;
            }
            if (!busyPoll) usleep(kPollIntervalUs);
        } while (*request == nullptr);
        return result;
    }

    ThreadResult WorkerPostRecv(void* recvComm, void* data, size_t size, int tag,
                                void* mhandle, void** request) {
        ThreadResult result;
        void*  bufs[1]    = {data};
        size_t sizes[1]   = {size};
        int    tags[1]    = {tag};
        void*  handles[1] = {mhandle};
        if (PostRecv(recvComm, 1, bufs, sizes, tags, handles, request) != ncclSuccess
            || *request == nullptr) {
            result.ok = false;
            result.msg = "irecv failed or returned a NULL request";
        }
        return result;
    }

    ThreadResult WorkerWait(void* request, int* sizes, int timeoutMs = kDefaultTimeoutMs) {
        ThreadResult result;
        if (WaitForCompletion(request, sizes, timeoutMs) != ncclSuccess) {
            result.ok = false;
            result.msg = "request did not complete within the timeout";
        }
        return result;
    }

    // Same wait without the 10 ms backoff between polls. A small transfer
    // completes in tens of microseconds, so the sleeping wait spends one full
    // interval on every one of them: measurements built on it report the poll
    // interval rather than anything the plugin did, and contention inside a
    // plugin call disappears under it. Only timing tests should use this -- it
    // burns a core per waiting worker, which is why the functional tests keep
    // the sleeping version.
    ThreadResult WorkerWaitBusy(void* request, int* sizes, int timeoutMs = kDefaultTimeoutMs) {
        ThreadResult result;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        int done = 0;
        while (!done) {
            if (TestRequest(request, &done, sizes) != ncclSuccess) {
                result.ok = false;
                result.msg = "test failed while polling for completion";
                return result;
            }
            if (done) break;
            if (std::chrono::steady_clock::now() >= deadline) {
                result.ok = false;
                result.msg = "request did not complete within the timeout";
                return result;
            }
        }
        return result;
    }

    // One transfer on the worker's own connection, without touching buffer
    // contents. For GPU buffers, or when the caller verifies the payload itself.
    ThreadResult WorkerSendRecvRaw(int rank, ConnectionPair& pair, void* buffer, size_t size,
                                   int tag, void* mhandle, int timeoutMs = kDefaultTimeoutMs,
                                   int* receivedSize = nullptr, bool busyPoll = false) {
        void* request = nullptr;
        ThreadResult result =
            (rank == 0)
                ? WorkerPostRecv(pair.recvComm, buffer, size, tag, mhandle, &request)
                : WorkerPostSend(pair.sendComm, buffer, size, tag, mhandle, &request, busyPoll,
                                 std::max(timeoutMs, kSlotWaitDefaultMs));
        if (!result.ok) return result;

        int sizes[1] = {0};
        result = busyPoll ? WorkerWaitBusy(request, sizes, timeoutMs)
                          : WorkerWait(request, sizes, timeoutMs);
        if (!result.ok) return result;
        if (receivedSize) *receivedSize = sizes[0];
        return result;
    }

    // One transfer plus host-side payload check: the sender fills a seeded
    // pattern, the receiver clears its buffer first and verifies afterwards. A
    // distinct seed per worker turns any cross-connection delivery into a data
    // failure rather than a silent pass.
    ThreadResult WorkerSendRecvPattern(int rank, ConnectionPair& pair, void* buffer, size_t size,
                                       int tag, void* mhandle, int seed,
                                       int timeoutMs = kDefaultTimeoutMs) {
        if (rank == 0) {
            memset(buffer, 0, size);
        } else {
            fillHostBufferWithPattern<uint8_t>(buffer, size, makeBytePattern(seed));
        }

        int received = 0;
        ThreadResult result = WorkerSendRecvRaw(rank, pair, buffer, size, tag, mhandle,
                                               timeoutMs, &received);
        if (!result.ok) return result;

        if (rank == 0) {
            if (received != (int)size) {
                result.ok = false;
                result.msg = "received size mismatch";
                return result;
            }
            if (size > 0
                && !verifyHostBufferData<uint8_t>(buffer, size, makeBytePattern(seed))) {
                result.ok = false;
                result.msg = "data validation failed";
            }
        }
        return result;
    }

    // Post a whole batch of equal-sized slots carved out of one registered buffer,
    // then drain and verify every one. Both slot-pressure bodies did exactly this,
    // differing only in how they labelled a failure, so the label is the parameter.
    //
    // Slot identity rests on the tag, not the payload: with the whole batch in
    // flight at once a per-slot seed would alias into another worker's patterns
    // modulo 256, so every slot carries this worker's single pattern and a payload
    // arriving on the wrong connection is a mismatch rather than a clean pass.
    ThreadResult WorkerBatchPostDrain(int rank, ConnectionPair& pair, void* buffer,
                                      size_t slotSize, int slots, void* mhandle, int pattern,
                                      const std::string& where) {
        ThreadResult result;
        std::vector<void*> requests(slots, nullptr);
        for (int i = 0; i < slots; i++) {
            char* slot = static_cast<char*>(buffer) + i * slotSize;
            if (rank == 0) {
                memset(slot, 0, slotSize);
                result = WorkerPostRecv(pair.recvComm, slot, slotSize, i, mhandle, &requests[i]);
            } else {
                fillHostBufferWithPattern<uint8_t>(slot, slotSize, makeBytePattern(pattern));
                // kStressTimeoutMs rather than the default slot wait: this post has
                // to outlast the peer rank publishing all of its own slots under
                // N-way contention on one NIC, which the drain below also budgets for.
                result = WorkerPostSend(pair.sendComm, slot, slotSize, i, mhandle, &requests[i],
                                        /*busyPoll=*/false, kStressTimeoutMs);
            }
            if (!result.ok) return result;
        }
        for (int i = 0; i < slots; i++) {
            int sizes[1] = {0};
            result = WorkerWait(requests[i], sizes, kStressTimeoutMs);
            if (!result.ok) return result;
            if (rank != 0) continue;
            const char* slot = static_cast<const char*>(buffer) + i * slotSize;
            if (sizes[0] != static_cast<int>(slotSize)) {
                result.ok = false;
                result.msg = where + "slot " + std::to_string(i) + ": received "
                             + std::to_string(sizes[0]) + " of " + std::to_string(slotSize)
                             + " bytes";
                return result;
            }
            if (!verifyHostBufferData<uint8_t>(slot, slotSize, makeBytePattern(pattern))) {
                result.ok = false;
                result.msg = where + "slot " + std::to_string(i)
                             + ": payload is not this worker's pattern";
                return result;
            }
        }
        return result;
    }

    // A worker's registered host buffer, kept alive as one movable value: the
    // allocation and its registration, released registration-first when this
    // dies because mhandleGuard is declared after bufferGuard. Returning it lets
    // a body that needs the handle for a loop hold both without re-pasting the
    // preamble, and keeps the guard order one reviewed fact rather than one per
    // call site. On failure `result.ok` is false and the guards are empty no-ops.
    struct WorkerHostBuffer {
        ThreadResult result;
        void* buffer = nullptr;
        void* mhandle = nullptr;
        HostBufferAutoGuard bufferGuard;
        NetMHandleWorkerGuard mhandleGuard;
    };

    // The allocate-and-register preamble on its own: malloc(size, floored to 1),
    // register it on this rank's comm, and hand back both guards live. A zero
    // size still yields a one-byte registered buffer, as the transfer helpers
    // expect.
    WorkerHostBuffer WorkerSetupHostBuffer(int rank, ConnectionPair& pair, size_t size) {
        WorkerHostBuffer h;
        const size_t allocSize = size ? size : 1;
        h.buffer = malloc(allocSize);
        if (!h.buffer) {
            h.result.ok = false;
            h.result.msg = "malloc failed";
            return h;
        }
        h.bufferGuard = makeHostBufferAutoGuard(h.buffer);

        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        h.result = WorkerRegister(comm, h.buffer, allocSize, NCCL_PTR_HOST, &h.mhandle);
        if (!h.result.ok) return h;
        h.mhandleGuard = NetMHandleWorkerGuard(h.mhandle, NetMHandleWorkerDeleter(net_, comm));
        return h;
    }

    // Composite: allocate a host buffer, register it, run one seeded transfer,
    // and release both. Covers the common single-transfer worker body. The
    // holder outlives the transfer, so the buffer is deregistered and freed only
    // after WorkerSendRecvPattern has returned.
    ThreadResult WorkerHostTransfer(int rank, ConnectionPair& pair, size_t size, int tag,
                                    int seed, int timeoutMs = kDefaultTimeoutMs) {
        WorkerHostBuffer h = WorkerSetupHostBuffer(rank, pair, size);
        if (!h.result.ok) return h.result;

        return WorkerSendRecvPattern(rank, pair, h.buffer, size, tag, h.mhandle, seed, timeoutMs);
    }

    // ── Worker-safe IB-CAST scheduler inspection ─────────────────────
    // Token and cursor state is per send communicator, so a worker can arm and
    // read its own connection without disturbing the others.

    ThreadResult WorkerCastSetTokens(void* sendComm, const std::vector<int>& tokens) {
        ThreadResult result;
        if (ncclIbCastSetTokens(sendComm, tokens.data(), (int)tokens.size()) != ncclSuccess) {
            result.ok = false;
            result.msg = "ncclIbCastSetTokens failed";
        }
        return result;
    }

    ThreadResult WorkerCastGetSchedState(void* sendComm, struct ncclIbCastSchedState* out) {
        ThreadResult result;
        memset(out, 0, sizeof(*out));
        if (ncclIbCastGetSchedState(sendComm, out) != ncclSuccess) {
            result.ok = false;
            result.msg = "ncclIbCastGetSchedState failed";
        }
        return result;
    }

    // QPs the connection actually uses. NCCL_IB_QPS_PER_CONNECTION states only a
    // request: on a merged device the plugin creates that many per member, so
    // arming faults from the environment would leave the remaining QPs healthy
    // and "the send must fail" would depend on which QP the scheduler picked.
    // Valid once the scheduler is warm, which the first successful send does.
    ThreadResult WorkerCastLiveNqps(void* sendComm, int* nqps) {
        struct ncclIbCastSchedState state;
        ThreadResult result = WorkerCastGetSchedState(sendComm, &state);
        if (!result.ok) return result;
        if (state.nqps <= 0) {
            result.ok = false;
            result.msg = "scheduler reports nqps=" + std::to_string(state.nqps);
            return result;
        }
        *nqps = state.nqps;
        return result;
    }

    // The connection's live QP count, agreed across ranks: on a merged device the plugin
    // creates the requested QPs per member, so an environment-derived split threshold is
    // wrong, and a worker cannot broadcast the real one.
    // 0 usable, 1 single queue pair (caller skips), -1 failure (reported here).
    int ThreadedCastAgreedNqps(int dev, int* nqps) {
        void* listenComm = nullptr;
        void* sendComm = nullptr;
        void* recvComm = nullptr;

        // The probe brings up its own connection rather than calling
        // SetupCastConnection, which asserts fatally and sends the handle only after
        // its assertions: a listen that fails on rank 0 leaves rank 1 waiting in
        // MPI_Recv forever, which is a hang instead of the failure this helper
        // promises. Here the handle carries a status word, so both ranks agree before
        // either one waits on the other.
        const int rank = MPIEnvironment::world_rank;
        const int peer = 1 - rank;
        ncclNetHandle_t handle;
        memset(&handle, 0, sizeof(handle));
        int localOk = 1;

        // ncclNetHandle_t is a char array, so the handshake copies rather than assigns.
        struct ProbeHandshake {
            int             ok;
            ncclNetHandle_t handle;
        };

        if (rank == 0) {
            if (CreateListenComm(dev, &handle, &listenComm) != ncclSuccess || !listenComm)
                localOk = 0;
            ProbeHandshake msg{};
            msg.ok = localOk;
            memcpy(msg.handle, handle, sizeof(handle));
            MPI_Send(&msg, sizeof(msg), MPI_BYTE, peer, 0, MPI_COMM_WORLD);
            if (localOk) {
                for (int i = 0; i < kMaxRetryAttempts && recvComm == nullptr; i++) {
                    if (AcceptConnection(listenComm, &recvComm) != ncclSuccess) break;
                    if (!recvComm) usleep(kPollIntervalUs);
                }
                if (!recvComm) localOk = 0;
            }
        } else {
            ProbeHandshake msg{};
            MPI_Recv(&msg, sizeof(msg), MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            localOk = msg.ok;
            if (localOk) {
                memcpy(handle, msg.handle, sizeof(handle));
                for (int i = 0; i < kMaxRetryAttempts && sendComm == nullptr; i++) {
                    if (ConnectToRemote(dev, &handle, &sendComm) != ncclSuccess) break;
                    if (!sendComm) usleep(kPollIntervalUs);
                }
                if (!sendComm) localOk = 0;
            }
        }

        int bothUp = 0;
        if (MPI_Allreduce(&localOk, &bothUp, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS)
            bothUp = 0;
        if (!bothUp) {
            ADD_FAILURE() << "could not establish the probe connection used to agree the "
                             "connection's live QP count";
            TeardownConnection(recvComm, listenComm, sendComm, nullptr);
            return -1;
        }

        void* comm = (rank == 0) ? recvComm : sendComm;
        std::vector<char> probe(128, 0);
        void* mhandle = nullptr;
        const int registered =
            RegisterMemory(comm, probe.data(), probe.size(), NCCL_PTR_HOST, &mhandle)
                    == ncclSuccess
                ? 1
                : 0;
        // Agreed before either side moves: a one-sided failure would otherwise send
        // the failing rank into the teardown barrier while its peer waits inside
        // GetActualNqps for traffic that is never coming, and the test would hang
        // instead of reporting anything. TeardownConnection accepts a null handle.
        int bothOk = 0;
        if (MPI_Allreduce(&registered, &bothOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD)
            != MPI_SUCCESS) {
            bothOk = 0;
        }
        if (!bothOk) {
            ADD_FAILURE() << "registering the probe buffer failed on at least one rank, so the "
                             "connection's live QP count could not be agreed";
            TeardownConnection(recvComm, listenComm, sendComm, mhandle);
            return -1;
        }
        *nqps = GetActualNqps(sendComm, recvComm, probe.data(), probe.size(), 1, mhandle);
        TeardownConnection(recvComm, listenComm, sendComm, mhandle);
        // One queue pair is a legitimate configuration -- CastSingleQPBypassesWrr
        // covers it -- but not one these branches can make claims about: the scheduler
        // returns before split selection and token accounting when nqps is 1, so a
        // token delta of 1 never appears and the split path is never taken. Reported
        // separately from a failure so the caller can skip rather than fail.
        if (*nqps == 1) return 1;
        if (*nqps > 0) return 0;
        ADD_FAILURE() << "the scheduler reported " << *nqps
                      << " queue pairs after a successful warm-up transfer";
        return -1;
    }

    // Warm the scheduler up (it initializes on the first send) and arm equal weights.
    // nqps is the live count ThreadedCastAgreedNqps agreed on the main thread, so both
    // ranks size their transfers the same way and merged devices work; the sender
    // confirms its own connection reports the same count.
    ThreadResult WorkerCastPrepareTokens(int rank, ConnectionPair& pair, void* buffer,
                                         void* mhandle, int nqps, int tag, int seed) {
        ThreadResult result = WorkerSendRecvPattern(rank, pair, buffer, 64, tag, mhandle, seed);
        if (!result.ok || rank != 1) return result;

        // The expectations are built from the agreed live count, so this checks that
        // this worker's own connection reports the same thing; a mismatch means the
        // expectations belong to a different connection than the one carrying data.
        int liveNqps = 0;
        result = WorkerCastLiveNqps(pair.sendComm, &liveNqps);
        if (!result.ok) return result;
        if (liveNqps != nqps) {
            // Both ranks sized their buffers from the value the main thread agreed,
            // so a worker's connection reporting something else means the
            // expectations below are about a different connection than the one
            // carrying the data.
            result.ok = false;
            result.msg = "this worker's connection reports nqps=" + std::to_string(liveNqps)
                         + " but the run agreed on " + std::to_string(nqps);
            return result;
        }
        return WorkerCastSetTokens(pair.sendComm, EqualTokens(liveNqps));
    }

    // Worker-safe CAST transfer with a token-consumption expectation. Only the
    // sender owns scheduler state, so it checks the delta while the receiver
    // verifies the payload. Pass a negative delta to skip the token check.
    ThreadResult WorkerCastTransferExpectTokens(int rank, ConnectionPair& pair, void* buffer,
                                                size_t size, int tag, void* mhandle, int seed,
                                                int expectedTokenDelta,
                                                int timeoutMs = kLargeTransferTimeoutMs) {
        struct ncclIbCastSchedState before = {};
        ThreadResult result;
        if (rank == 1) {
            result = WorkerCastGetSchedState(pair.sendComm, &before);
            if (!result.ok) return result;
        }

        result = WorkerSendRecvPattern(rank, pair, buffer, size, tag, mhandle, seed, timeoutMs);
        if (!result.ok) return result;

        if (rank == 1 && expectedTokenDelta >= 0) {
            struct ncclIbCastSchedState after = {};
            result = WorkerCastGetSchedState(pair.sendComm, &after);
            if (!result.ok) return result;
            const int delta = before.activeTotTokens - after.activeTotTokens;
            if (delta != expectedTokenDelta) {
                result.ok = false;
                result.msg = "expected a WRR token delta of "
                             + std::to_string(expectedTokenDelta) + " at size "
                             + std::to_string(size) + ", observed " + std::to_string(delta);
            }
        }
        return result;
    }


#if defined(ENABLE_FAULT_INJECTION)
    // ── Worker-safe IB-CAST fault injection ──────────────────────────
    // The pre-call intercept hooks store their state in the communicator
    // (faultQpDelayUs / faultQpError in ncclIbNetCommBase), so a worker can
    // break its own connection without touching anybody else's. The
    // libibverbs-level ops registry is process-global and deliberately not
    // used here.

    ThreadResult WorkerCastFaultArmError(void* sendComm, int nqps) {
        ThreadResult result;
        for (int qp = 0; qp < nqps; qp++) {
            if (ncclIbCastFaultSetQpError(sendComm, qp, /*inject=*/true) != ncclSuccess) {
                result.ok = false;
                result.msg = "ncclIbCastFaultSetQpError failed on QP " + std::to_string(qp);
                return result;
            }
        }
        return result;
    }

    ThreadResult WorkerCastFaultSetDelay(void* sendComm, int qpIdx, uint32_t delayUs) {
        ThreadResult result;
        if (ncclIbCastFaultSetQpDelay(sendComm, qpIdx, delayUs) != ncclSuccess) {
            result.ok = false;
            result.msg = "ncclIbCastFaultSetQpDelay failed";
        }
        return result;
    }

    ThreadResult WorkerCastFaultClear(void* sendComm) {
        ThreadResult result;
        if (ncclIbCastFaultClear(sendComm) != ncclSuccess) {
            result.ok = false;
            result.msg = "ncclIbCastFaultClear failed";
        }
        return result;
    }

    int WorkerCastFatalCount(void* sendComm) {
        int count = 0;
        if (ncclIbCastFaultGetFatalCount(sendComm, &count) != ncclSuccess) return -1;
        return count;
    }

    ThreadResult WorkerCastDriveQpToError(void* sendComm, int qpIdx) {
        ThreadResult result;
        if (ncclIbCastFaultDriveQpToError(sendComm, qpIdx) != ncclSuccess) {
            result.ok = false;
            result.msg = "ncclIbCastFaultDriveQpToError failed";
        }
        return result;
    }

    // Post one send and poll it, treating both a failed isend and a fatal error
    // count as an outcome rather than a test failure. Used by fault tests that
    // accept either signal.
    struct WorkerFaultSendOutcome {
        ncclResult_t sendRet = ncclSuccess;
        bool         completed = false;
        int          fatalCount = 0;
    };

    WorkerFaultSendOutcome WorkerCastFaultSend(void* sendComm, void* buffer, size_t size, int tag,
                                               void* mhandle, int pollIterations) {
        WorkerFaultSendOutcome outcome;
        void* request = nullptr;
        for (int attempt = 0; attempt < kMaxRetryAttempts; attempt++) {
            outcome.sendRet = PostSend(sendComm, buffer, size, tag, mhandle, &request);
            if (outcome.sendRet != ncclSuccess || request != nullptr) break;
            usleep(kPollIntervalUs);
        }
        outcome.fatalCount = WorkerCastFatalCount(sendComm);

        if (outcome.sendRet == ncclSuccess && request != nullptr) {
            for (int poll = 0; poll < pollIterations; poll++) {
                int done = 0;
                int sizes[1] = {0};
                const ncclResult_t testRet = TestRequest(request, &done, sizes);
                if (testRet != ncclSuccess) {
                    outcome.sendRet = testRet;
                    break;
                }
                outcome.fatalCount = WorkerCastFatalCount(sendComm);
                if (done) {
                    outcome.completed = true;
                    break;
                }
                if (outcome.fatalCount > 0) break;
                usleep(kPollIntervalUs);
            }
        }
        return outcome;
    }

    // Shared worker body for the failover tests: drive the sender's QP 0 into
    // the error state, then require the payload to still arrive over the
    // surviving device of a fused NIC, with no fatal error and a device state
    // that is no longer Ok.
    //
    // What the concurrency exercises here is overlapping failovers, not recovery:
    // the suites that run this leave NCCL_IB_RESILIENCY_PORT_RECOVERY at its
    // default of off, because these assertions require the device to stay
    // not-Ok and recovery would undo that underneath them. Resiliency state is per
    // communicator, so N workers drive their own links into error at the same
    // moment on one fused device, and every one of them has to keep routing over
    // the survivor. The process-global recovery thread is a different claim, and
    // fault_recovery_multithread_2 is where it is made.
    ThreadResult WorkerCastFailoverTransfer(int rank, ConnectionPair& pair, void* buffer,
                                            size_t size, int tag, void* mhandle, int seed,
                                            int messages = 1, std::atomic<int>* arrived = nullptr,
                                            int expected = 0,
                                            NetMHandleWorkerGuard* registration = nullptr,
                                            HostBufferAutoGuard* allocation = nullptr) {
        ThreadResult result;
        // Without a gate here the failures are merely started from several workers,
        // not overlapping: each worker allocates, registers and warms up first, so a
        // fast one can be through failover before a slow one reaches this line, and
        // simultaneous errors on the one fused device -- the thing this body claims
        // to cover -- go untested. Bounded, and a timeout is reported rather than
        // asserted, since a worker cannot fail the test alone.
        if (arrived && expected > 1
            && !WorkerRendezvous(*arrived, expected, kFailureGatePolls)) {
            result.ok = false;
            result.msg = "workers did not all reach the link failure together";
            return result;
        }
        if (rank == 1) {
            result = WorkerCastDriveQpToError(pair.sendComm, 0);
            if (!result.ok) return result;
        }

        for (int i = 0; i < messages; i++) {
            result = WorkerSendRecvPattern(rank, pair, buffer, size, tag + i, mhandle, seed + i,
                                           kLargeTransferTimeoutMs);
            if (!result.ok) {
                result.msg = "after driving QP 0 to error: " + result.msg;
                // A timed-out wait leaves the request posted: WorkerSendRecvPattern
                // drops the pointer without cancelling it, so returning straight to
                // the caller would let its guards free memory the NIC may still be
                // writing into. Same treatment as WorkerTransferAcrossQpFailure.
                if (registration || allocation)
                    return WorkerRetainAfterAbandonedRequest(result, pair, rank, registration,
                                                             allocation);
                return result;
            }
        }

        if (rank != 1) return result;

        const int fatalCount = WorkerCastFatalCount(pair.sendComm);
        if (fatalCount != 0) {
            result.ok = false;
            result.msg = "failover should have handled the QP error, but the fatal count is "
                         + std::to_string(fatalCount);
            return result;
        }

        struct ncclIbCastResiliencyState state = {};
        if (ncclIbCastGetResiliencyState(pair.sendComm, &state) != ncclSuccess) {
            result.ok = false;
            result.msg = "ncclIbCastGetResiliencyState failed";
            return result;
        }
        // 0 is ncclIbResiliencyDevStateOk; anything else means the failure was
        // registered (Error, RecoveryInProgress or Recovered).
        if (state.devState[0] == 0) {
            result.ok = false;
            result.msg = "device 0 still reports Ok after its QP was driven to error";
        }
        return result;
    }

    // The receiver cannot be asked how many queue pairs its connection uses, so a flush
    // walks this many; extra indices are refused harmlessly.
    static constexpr int kMaxQpsToFlush = 8;

    // For a failure where a request may still be live and the test cannot reach it:
    // WorkerSendRecvPattern owns its request and a timed-out wait does not cancel the
    // work, so the buffer and its registration must not be released as the worker
    // unwinds. This side's queue pairs are driven to error first, which retires
    // outstanding work with a flush status, and both guards are then retained because
    // nothing here can prove the request is gone. Leaking one buffer on a failing test
    // is the cheaper mistake.
    ThreadResult WorkerRetainAfterAbandonedRequest(ThreadResult failure, ConnectionPair& pair,
                                                   int rank, NetMHandleWorkerGuard* registration,
                                                   HostBufferAutoGuard* allocation) {
        if (rank == 1) {
            int nqps = 0;
            if (!WorkerCastLiveNqps(pair.sendComm, &nqps).ok || nqps <= 0) nqps = 1;
            for (int qp = 0; qp < nqps; qp++) WorkerCastDriveQpToError(pair.sendComm, qp);
        } else {
            for (int qp = 0; qp < kMaxQpsToFlush; qp++)
                ncclIbCastFaultDriveRecvQpToError(pair.recvComm, qp);
        }
        failure.msg += "; the buffer and its registration are retained, since a request may "
                       "still reference them";
        if (registration) registration->release();
        if (allocation) allocation->release();
        return failure;
    }

    // Bounded gate for the failure point: 30 s is well past the setup a sibling
    // still has to finish, and a worker that already failed must not hang the rest.
    static constexpr int kFailureGatePolls = 3000;  // 3000 * 10ms

    // Failover with requests already in flight. isend only gets a request once the
    // receiver has published a FIFO slot, so "all sends posted" implies "all receives
    // posted" -- the ordering the serial test buys with a barrier a worker cannot use.
    // buffer must hold messages * size bytes, one slice each.
    ThreadResult WorkerCastFailoverInFlight(int rank, ConnectionPair& pair, void* buffer,
                                            size_t size, int tag, void* mhandle, int seed,
                                            int messages, std::atomic<int>* arrived = nullptr,
                                            int expected = 0,
                                            NetMHandleWorkerGuard* registration = nullptr,
                                            HostBufferAutoGuard* allocation = nullptr,
                                            int* repostsOut = nullptr) {
        ThreadResult result;
        std::vector<void*> requests(messages, nullptr);

        // Every exit from the first successful post onwards goes through this: a request
        // still posted keeps the NIC writing into the buffer the caller's guards are
        // about to release. Waiting is tried first; anything that will not complete is
        // flushed by driving this side's queue pairs to error, which retires outstanding
        // work with a flush status. If even that leaves a request live, both the
        // registration and the allocation are retained -- keeping the MR while the memory
        // is freed would not be safer -- and the message says so.
        auto finish = [&](ThreadResult failure) -> ThreadResult {
            auto waitAll = [&]() {
                int live = 0;
                for (int i = 0; i < messages; i++) {
                    if (!requests[i]) continue;
                    int sizes[1] = {0};
                    if (WorkerWait(requests[i], sizes, kLargeTransferTimeoutMs).ok) {
                        requests[i] = nullptr;
                    } else {
                        live++;
                    }
                }
                return live;
            };

            if (waitAll() > 0) {
                // Each rank flushes its own side. Only the sender can be asked for the
                // live count, so the receiver walks the same span it was told to expect;
                // driving a queue pair that does not exist is harmless here, and using
                // the send-side call on a receive communicator would flush nothing.
                int nqps = 0;
                if (rank == 1) {
                    if (!WorkerCastLiveNqps(pair.sendComm, &nqps).ok || nqps <= 0) nqps = 1;
                    for (int qp = 0; qp < nqps; qp++) WorkerCastDriveQpToError(pair.sendComm, qp);
                } else {
                    for (int qp = 0; qp < kMaxQpsToFlush; qp++)
                        ncclIbCastFaultDriveRecvQpToError(pair.recvComm, qp);
                }
                const int live = waitAll();
                if (live > 0) {
                    failure.msg += "; " + std::to_string(live)
                                   + " request(s) never completed even after flushing, so the "
                                     "buffer and its registration are retained rather than "
                                     "released";
                    if (registration) registration->release();
                    if (allocation) allocation->release();
                }
            }
            return failure;
        };

        // Gate before posting, not after. A gate between the posts and the transition
        // would have an early worker wait for its siblings with its work already
        // posted, and hardware needs no polling to retire it -- so the wait itself
        // would empty the queue pair this test wants broken while it is busy. Here it
        // only lines the workers up, and each one then posts and breaks its own link
        // without pausing in between.
        if (arrived && expected > 1
            && !WorkerRendezvous(*arrived, expected, kFailureGatePolls)) {
            result.ok = false;
            result.msg = "workers did not all reach the link failure together";
            return finish(result);
        }

        for (int i = 0; i < messages; i++) {
            char* slice = static_cast<char*>(buffer) + static_cast<size_t>(i) * size;
            if (rank == 0) {
                memset(slice, 0, size);
                result = WorkerPostRecv(pair.recvComm, slice, size, tag + i, mhandle,
                                        &requests[i]);
            } else {
                fillHostBufferWithPattern<uint8_t>(slice, size, makeBytePattern(seed + i));
                result = WorkerPostSend(pair.sendComm, slice, size, tag + i, mhandle,
                                        &requests[i]);
            }
            if (!result.ok) return finish(result);
        }

        // Nothing is polled before the transition: polling would drain the
        // communicator's shared completion queues and hurry the very requests this
        // is about. The ordering is all that is arranged here -- posted, then broken,
        // then waited on -- and the note above the count at the end of this function
        // says why the stronger claim is not made.
        if (rank == 1) {
            result = WorkerCastDriveQpToError(pair.sendComm, 0);
            if (!result.ok) return finish(result);
        }

        for (int i = 0; i < messages; i++) {
            int sizes[1] = {0};
            result = WorkerWait(requests[i], sizes, kLargeTransferTimeoutMs);
            if (!result.ok) {
                result.msg = "request " + std::to_string(i)
                             + " never completed after QP 0 was driven to error";
                return finish(result);
            }
            requests[i] = nullptr;
            if (rank != 0) continue;

            char* slice = static_cast<char*>(buffer) + static_cast<size_t>(i) * size;
            if (sizes[0] != (int)size
                || !verifyHostBufferData<uint8_t>(slice, size, makeBytePattern(seed + i))) {
                result.ok = false;
                result.msg = "message " + std::to_string(i)
                             + " arrived corrupted or short after failover";
                return finish(result);
            }
        }

        if (rank != 1) return result;

        // Reported, not asserted, and that is the end of a long thread. Four ways of
        // establishing "the failure caught several requests in flight" were tried
        // here and none of them holds with recovery off: the QP delay hook sleeps
        // before post_send, so it only slows posting; outstandingRequests counts
        // requests already registered as failed, which is zero before the failure;
        // polling each request in turn is not a snapshot, because TestRequest drains
        // the communicator's shared completion queues; and the repost counter stays
        // at zero even in the serial body, which orders the break against a transfer
        // with an MPI handshake and passes -- so it does not record this path at all
        // without port recovery.
        //
        // What the threaded body does establish is the ordering: every message is
        // posted, then this worker's QP 0 is driven to error, then every message must
        // arrive intact with no fatal error, while the sibling workers do the same to
        // their own connections. Whether a data work request was on the wire at the
        // instant of the transition is not observable from here, and the serial body
        // does not observe it either -- it arranges it with a handshake a worker
        // cannot use. The count is handed back for the main thread to log: TEST_INFO
        // reaches MPI_Comm_rank, which a worker must not call.
        if (repostsOut) {
            int reposts = -1;
            if (ncclIbCastGetRepostCount(pair.sendComm, &reposts) != ncclSuccess) reposts = -1;
            *repostsOut = reposts;
        }

        const int fatalCount = WorkerCastFatalCount(pair.sendComm);
        if (fatalCount != 0) {
            result.ok = false;
            result.msg = "failover should have absorbed the QP error, but the fatal count is "
                         + std::to_string(fatalCount);
        }
        return result;
    }

    // Poll a receive the peer may never satisfy, and report whether it finished.
    bool WorkerDrainRecv(void* request, int pollIterations) {
        if (!request) return true;
        for (int poll = 0; poll < pollIterations; poll++) {
            int done = 0;
            int sizes[1] = {0};
            if (TestRequest(request, &done, sizes) != ncclSuccess) return false;
            if (done) return true;
            usleep(kPollIntervalUs);
        }
        return false;
    }

    // Releases a receive the peer's injected send will never satisfy. It cannot be
    // waited out -- the failed send consumed its FIFO slot -- and the work request holds
    // the memory region until the queue pair dies, which is after the worker must
    // deregister. Driving this side to error flushes it instead.
    ThreadResult WorkerCastFlushAbandonedRecv(void* recvComm, void* request,
                                              NetMHandleWorkerGuard* registration = nullptr,
                                              HostBufferAutoGuard* allocation = nullptr) {
        ThreadResult result;
        if (!request) return result;

        // The API rejects an index past the connection's QP count, which is how
        // the loop learns where to stop.
        static constexpr int kQpProbeLimit = 64;
        for (int qp = 0; qp < kQpProbeLimit; qp++) {
            if (ncclIbCastFaultDriveRecvQpToError(recvComm, qp) != ncclSuccess) break;
        }

        // A flush completion surfaces as an error, and that is the expected
        // outcome here: all that matters is that the request stops being
        // outstanding before the memory region goes away.
        static constexpr int kFlushPolls = 500;  // 500 * 10ms = 5s
        for (int poll = 0; poll < kFlushPolls; poll++) {
            int done = 0;
            int sizes[1] = {0};
            if (TestRequest(request, &done, sizes) != ncclSuccess) return result;
            if (done) return result;
            usleep(kPollIntervalUs);
        }
        // This failure says precisely that the receive is still live, so releasing the
        // memory now is the one thing that must not happen: the caller's guards would
        // free a buffer the device can still write into. Keeping the registration
        // without the allocation would not be safer, so both are retained.
        result.ok = false;
        result.msg = "the abandoned receive was still outstanding after its queue pairs were "
                     "driven to error";
        if (registration || allocation) {
            result.msg += "; the buffer and its registration are retained, since the request "
                          "may still reference them";
            if (registration) registration->release();
            if (allocation) allocation->release();
        }
        return result;
    }

    // Both sides lose QP 0 and the payload rides the surviving device. The serial test
    // breaks the queue pairs mid-transfer, which needs an MPI handshake a worker cannot
    // make; both in-flight orders left the two sides a message apart, so the break here
    // happens while the connection is idle.
    //
    // The guards are taken rather than left to the caller because this helper is the one
    // that knows a transfer was attempted. WorkerSendRecvPattern owns its request and a
    // timed-out wait does not cancel it, so returning that failure straight up would let
    // the caller's guards deregister and free memory the NIC may still be writing into --
    // and a transfer that runs immediately after both queue pairs were driven to error is
    // the likeliest one in the file to time out. Callers that pass no guards get the old
    // behaviour, which is correct only when they hold no buffer of their own.
    ThreadResult WorkerTransferAcrossQpFailure(int rank, ConnectionPair& pair, void* buffer,
                                               size_t size, int tag, void* mhandle, int seed,
                                               int timeoutMs,
                                               NetMHandleWorkerGuard* registration = nullptr,
                                               HostBufferAutoGuard* allocation = nullptr) {
        ThreadResult result;
        if (rank == 0) {
            if (ncclIbCastFaultDriveRecvQpToError(pair.recvComm, 0) != ncclSuccess) {
                result.ok = false;
                result.msg = "ncclIbCastFaultDriveRecvQpToError failed";
                return result;
            }
        } else {
            result = WorkerCastDriveQpToError(pair.sendComm, 0);
            if (!result.ok) return result;
        }
        // Nothing has been posted yet on either exit above, so those return unchanged.
        result = WorkerSendRecvPattern(rank, pair, buffer, size, tag, mhandle, seed, timeoutMs);
        if (!result.ok && (registration || allocation))
            return WorkerRetainAfterAbandonedRequest(result, pair, rank, registration, allocation);
        return result;
    }
#endif /* ENABLE_FAULT_INJECTION */

    ncclResult_t InitNetIbCtx(void** ctxOut) {
        ncclNetCommConfig_t commConfig = {};
        commConfig.trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
        return net_->init(ctxOut, 0, &commConfig, nullptr, nullptr);
    }

    ncclResult_t CreateListenCommCtx(void* ctx, int dev, ncclNetHandle_t* handle, void** listenComm) {
        return net_->listen(ctx, dev, handle, listenComm);
    }

    ncclResult_t ConnectToRemoteCtx(void* ctx, int dev, ncclNetHandle_t* handle, void** sendComm) {
        return net_->connect(ctx, dev, handle, sendComm, nullptr);
    }

    // Runs on the main thread only. The listener-ready flag lets the connector
    // stop cleanly when listen() failed rather than interpreting a zeroed
    // handle and blocking indefinitely in connect().
    ThreadResult SetupConnectionForThread(void* ctx, int dev, ConnectionPair& pair,
                                          int rank, int peerRank, int mpiTag) {
        ThreadResult result;
        int listenerReady = 0;
        if (rank == 0) {
            if (CreateListenCommCtx(ctx, dev, &pair.handle, &pair.listenComm) == ncclSuccess) {
                listenerReady = 1;
            } else {
                result.ok = false;
                result.msg = "CreateListenComm failed";
            }

            MPI_Send(&listenerReady, 1, MPI_INT, peerRank, mpiTag, MPI_COMM_WORLD);
            MPI_Send(&pair.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, mpiTag + 1, MPI_COMM_WORLD);
            if (listenerReady) {
                for (int attempts = 0; pair.recvComm == nullptr; ++attempts) {
                    if (AcceptConnection(pair.listenComm, &pair.recvComm) != ncclSuccess) {
                        result.ok = false;
                        result.msg = "AcceptConnection error";
                        break;
                    }
                    if (pair.recvComm == nullptr) {
                        if (attempts >= kMaxRetryAttempts) {
                            result.ok = false;
                            result.msg = "AcceptConnection timed out";
                            break;
                        }
                        usleep(kPollIntervalUs);
                    }
                }
            }
        } else {
            MPI_Recv(&listenerReady, 1, MPI_INT, peerRank, mpiTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&pair.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, mpiTag + 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (!listenerReady) {
                result.ok = false;
                result.msg = "peer CreateListenComm failed";
            } else {
                for (int attempts = 0; pair.sendComm == nullptr; ++attempts) {
                    if (ConnectToRemoteCtx(ctx, dev, &pair.handle, &pair.sendComm) != ncclSuccess) {
                        result.ok = false;
                        result.msg = "ConnectToRemote error";
                        break;
                    }
                    if (pair.sendComm == nullptr) {
                        if (attempts >= kMaxRetryAttempts) {
                            result.ok = false;
                            result.msg = "ConnectToRemote timed out";
                            break;
                        }
                        usleep(kPollIntervalUs);
                    }
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return result;
    }

    void TeardownConnectionForThread(ThreadConnection& connection, int rank) {
        if (rank == 0) {
            if (connection.pair.recvComm) net_->closeRecv(connection.pair.recvComm);
            if (connection.pair.listenComm) net_->closeListen(connection.pair.listenComm);
        } else {
            if (connection.pair.sendComm) net_->closeSend(connection.pair.sendComm);
        }
        if (connection.ctx) net_->finalize(connection.ctx);
        connection = ThreadConnection{};
    }

    void TeardownThreadConnections(std::vector<ThreadConnection>& connections, int rank) {
        for (auto& connection : connections) TeardownConnectionForThread(connection, rank);
    }

    // Which device each worker connects over. Spreading puts workers on
    // different NICs, so concurrency reaches several devices' protection
    // domains and MR caches instead of hammering one.
    struct ThreadDevPolicy {
        int  dev    = 0;
        bool spread = false;

        static ThreadDevPolicy Fixed(int dev) { return ThreadDevPolicy{dev, false}; }
        static ThreadDevPolicy Spread() { return ThreadDevPolicy{0, true}; }
    };

    // Payload seed for worker threadIdx at sequence position seq. Workers must
    // differ here, otherwise a payload delivered on the wrong connection still
    // verifies and the cross-worker check is vacuous.
    //
    // makeBytePattern() observes the seed only modulo 256, so the stride has to
    // be odd: an even stride collides for workers whose indices differ by
    // 256/gcd(stride, 256) -- with a stride of 100000 that is every 8th worker.
    // The base keeps the seed off zero, whose pattern begins with the 0x00 that
    // the receive buffer is cleared to.
    static constexpr int kWorkerSeedBase   = 55;
    static constexpr int kWorkerSeedStride = 100001;
    // With an odd stride the 256 indices a cap of 256 permits still land on 256
    // distinct values modulo 256, one each. Worker 256 would be the first to share
    // a pattern with worker 0, and a payload delivered on the wrong connection would
    // then verify clean, so the cap is what the build refuses to let past 256 -- not
    // the count of usable indices below it.
    static_assert(MPIEnvironment::kMaxThreads <= 256,
                  "WorkerSeed patterns alias once the worker cap exceeds 256; widen the "
                  "pattern before raising it");
    static_assert(kWorkerSeedStride % 2 == 1,
                  "WorkerSeed stride must be odd, or workers a power of two apart collide");
    static int WorkerSeed(int threadIdx, int seq) {
        return kWorkerSeedBase + threadIdx * kWorkerSeedStride + seq;
    }

    // Plugin-visible indices of physical (non-merged) devices. devices() also
    // reports virtual NICs created by earlier tests, and a deduped 1-device vNIC
    // reuses an existing physical index, so dedupe on vProps.devs[0].
    // Only devices that can actually deliver traffic are returned: a RoCE NIC
    // with link-local GIDs only sets up QPs without complaint and then drops
    // packets silently, which would surface as an unexplained connect or accept
    // timeout on whichever worker happened to be handed that device.
    std::vector<int> PhysicalDeviceIndices() {
        std::vector<int> indices;
        int n = 0;
        if (GetDeviceCount(&n) != ncclSuccess) return indices;
        std::set<int> seen;
        for (int i = 0; i < n; i++) {
            ncclNetProperties_t props;
            memset(&props, 0, sizeof(props));
            if (GetDeviceProperties(i, &props) != ncclSuccess) continue;
            if (props.vProps.ndevs > 1) continue;
            if (props.name && !CanRouteCrossNode(props.name)) continue;
            if (seen.insert(props.vProps.devs[0]).second) indices.push_back(i);
        }
        return indices;
    }

    // The cross-rank device agreement travels as one 64-bit mask, so it covers
    // plugin device indices 0..63.
    static constexpr int kAgreedDevMaskBits = 64;
    static constexpr unsigned kGidFamilyV4 = 1u << 0;
    static constexpr unsigned kGidFamilyV6 = 1u << 1;

    // The plugin's getGidAddrFamily(), byte for byte: AF_INET covers both
    // ::ffff:a.b.c.d and the IPv4-mapped multicast form ff0e::ffff:a.b.c.d, and
    // everything else is AF_INET6. Reading only the first of those would classify
    // a multicast-form GID as IPv6 here while the plugin calls it IPv4, which
    // would invent a cross-rank mismatch and drop a usable NIC.
    static unsigned GidAddrFamilyBit(const uint8_t gid[16]) {
        static const uint8_t kV4Suffix[4]         = {0x00, 0x00, 0xff, 0xff};
        static const uint8_t kZeros[8]            = {0, 0, 0, 0, 0, 0, 0, 0};
        static const uint8_t kMulticastPrefix[4]  = {0xff, 0x0e, 0x00, 0x00};
        const bool v4Suffix = memcmp(gid + 8, kV4Suffix, 4) == 0;
        const bool v4Mapped = v4Suffix && memcmp(gid, kZeros, 8) == 0;
        const bool v4MappedMulticast =
            v4Suffix && memcmp(gid, kMulticastPrefix, 4) == 0 && memcmp(gid + 4, kZeros, 4) == 0;
        return (v4Mapped || v4MappedMulticast) ? kGidFamilyV4 : kGidFamilyV6;
    }

    // Device indices both ranks agree on. PhysicalDeviceIndices() reads local
    // sysfs only, so where the two nodes have different NICs alive, slot k would
    // sit on different hardware per rank: connect and accept still succeed and
    // the traffic is then dropped, which surfaces as an unexplained completion
    // timeout -- the very failure the routability filter exists to prevent.
    // Intersecting the two lists keeps every slot symmetric. Index symmetry is not
    // reachability, though, so the surviving slots are then checked for a shared
    // GID address family: see the second half of this function.
    bool AgreedPhysicalDeviceIndices(std::vector<int>* out, std::string* reason) {
        out->clear();
        unsigned long long localMask = 0;
        int aboveMask = 0;
        for (int dev : PhysicalDeviceIndices()) {
            // The agreement mask is one 64-bit word, so an index at or above 64
            // cannot take part. No machine has that many plugin-visible devices,
            // but dropping one with nothing in the log would quietly narrow the
            // spread, so say how many did not fit.
            if (dev < 0 || dev >= kAgreedDevMaskBits) { aboveMask++; continue; }
            localMask |= (1ull << dev);
        }
        if (aboveMask > 0) {
            TEST_INFO("Rank %d: %d device index(es) outside [0,%d) are left out of the "
                      "cross-rank device agreement and will not be spread onto",
                      MPIEnvironment::world_rank, aboveMask, kAgreedDevMaskBits);
        }
        unsigned long long agreed = 0;
        if (MPI_Allreduce(&localMask, &agreed, 1, MPI_UNSIGNED_LONG_LONG, MPI_BAND,
                          MPI_COMM_WORLD) != MPI_SUCCESS) {
            *reason = "MPI_Allreduce of the routable device mask failed";
            return false;
        }
        for (int dev = 0; dev < kAgreedDevMaskBits; ++dev)
            if (agreed & (1ull << dev)) out->push_back(dev);
        if (out->empty()) {
            *reason = "no device index is routable on both ranks, so workers cannot be spread";
            return false;
        }

        // CanRouteCrossNode() reads local sysfs, so an agreed index only means each
        // rank has something routable in slot k -- not that the two ends can reach
        // each other. So the ranks exchange, per slot, the set of address families
        // its routable GIDs offer, and keep the slots that have one in common.
        //
        // A set and not one sampled GID. Which GID the plugin will actually use is
        // not knowable from here: it depends on NCCL_IB_GID_INDEX,
        // NCCL_IB_ADDR_FAMILY, the RoCE version files and prefix matching, all
        // resolved in connect.cc at connect time. Picking whichever GID sysfs
        // happens to list first would therefore be an arbitrary choice that a
        // dual-stack port can make disagree with the plugin's, dropping a usable
        // NIC or keeping an unusable one. Asking only whether the two ends share a
        // family assumes nothing about the selection, and is still enough to catch
        // the case worth catching, since the plugin cannot pair AF_INET with
        // AF_INET6 whatever else it decides.
        //
        // Family and not subnet, which is what one would reach for first. On this
        // fabric a working link's two ends are deliberately on different subnets:
        // every NIC on a node shares one /24 whose third octet is the node
        // (10.101.43.x on mia1-p01-g43, all eight of rdma0..7), so any two nodes
        // differ and the traffic is routed. Requiring a shared /24 -- the plugin's
        // NCCL_IB_SUBNET_PREFIX_LEN default -- would therefore reject every
        // cross-node pair that in fact works. That rule belongs to plane matching
        // in connect.cc, not to reachability. Family still holds: gidSameSubnet()
        // rejects AF_INET against AF_INET6 before it looks at any prefix.
        std::vector<uint8_t> localFamilies(kAgreedDevMaskBits, 0);
        for (int dev : *out) {
            ncclNetProperties_t props;
            memset(&props, 0, sizeof(props));
            if (GetDeviceProperties(dev, &props) != ncclSuccess || !props.name) continue;
            unsigned families = 0;
            if (!CanRouteCrossNode(props.name, &families)) continue;
            localFamilies[dev] = static_cast<uint8_t>(families);
        }

        const int worldSize = MPIEnvironment::world_size;
        std::vector<uint8_t> allFamilies(localFamilies.size() * worldSize, 0);
        if (MPI_Allgather(localFamilies.data(), (int)localFamilies.size(), MPI_BYTE,
                          allFamilies.data(), (int)localFamilies.size(), MPI_BYTE,
                          MPI_COMM_WORLD) != MPI_SUCCESS) {
            *reason = "MPI_Allgather of the per-device GID address families failed";
            return false;
        }

        std::vector<int> sharedFamily;
        int unknown = 0;
        for (int dev : *out) {
            bool anyComparable = false;
            bool anyDisjoint   = false;
            for (int r = 0; r < worldSize; ++r) {
                if (r == MPIEnvironment::world_rank) continue;
                const uint8_t peer = allFamilies[r * kAgreedDevMaskBits + dev];
                // An empty set at either end -- an IB link, whose GIDs say nothing
                // about routability, or a port whose GIDs sysfs wrote in a form this
                // parser does not know -- leaves that pair uncomparable. A slot no
                // peer could be compared against is kept, which is the behaviour
                // before this check existed.
                if (localFamilies[dev] == 0 || peer == 0) continue;
                anyComparable = true;
                if ((localFamilies[dev] & peer) == 0) anyDisjoint = true;
            }
            if (!anyComparable) unknown++;
            if (!anyDisjoint) sharedFamily.push_back(dev);
        }

        if (sharedFamily.empty()) {
            // An uncomparable slot (localFamilies[dev] == 0 or peer == 0 on every
            // peer) never sets anyDisjoint, so it is never excluded here -- it
            // would still be sitting in sharedFamily if any such slot existed.
            // Empty therefore means every agreed slot was comparable *and*
            // disjoint: not a parser guess, but AF_INET-vs-AF_INET6 on every one
            // of them, which the plugin cannot connect regardless of
            // NCCL_IB_GID_INDEX or anything else it picks. Restoring the
            // index-symmetric list here would send the threaded suites into
            // connect/accept retries against devices already proven not to pair,
            // instead of the synchronized skip this check exists to produce.
            *reason = "every agreed device index has disjoint GID address families across "
                      "ranks (AF_INET on one side, AF_INET6 on the other), so none of them "
                      "can connect";
            out->clear();
            return false;
        }
        if (sharedFamily.size() != out->size()) {
            TEST_INFO("Rank %d: %zu of %zu agreed device index(es) dropped for having no GID "
                      "address family in common across ranks (%d uncomparable and kept)",
                      MPIEnvironment::world_rank, out->size() - sharedFamily.size(), out->size(),
                      unknown);
        }
        *out = sharedFamily;
        return true;
    }

    int ResolveWorkerDev(const ThreadDevPolicy& policy, const std::vector<int>& physical,
                         int slotIdx) {
        if (!policy.spread) return policy.dev;
        return physical[slotIdx % physical.size()];
    }

    // Bounded rendezvous for workers that must all reach a point before any of
    // them moves on, which the start gates cannot express: they only synchronize
    // entry into the body. Bounded so a worker that failed earlier cannot hang
    // its siblings; the caller reports the timeout as its own failure.
    // aborted, when given, releases the wait early: a worker that failed before
    // arriving never increments the counter, so without it the siblings spin out the
    // whole budget and report "only K of N", which hides the failure that caused it.
    static bool WorkerRendezvous(std::atomic<int>& arrived, int expected, int pollIterations,
                                 const std::atomic<bool>* aborted = nullptr) {
        arrived.fetch_add(1, std::memory_order_release);
        for (int poll = 0; poll < pollIterations; poll++) {
            if (arrived.load(std::memory_order_acquire) >= expected) return true;
            if (aborted && aborted->load(std::memory_order_acquire)) return false;
            usleep(kPollIntervalUs);
        }
        return false;
    }

    // Mirrors the normal rccl-tests -t execution model: contexts and
    // communicators are established by the main thread, then workers drive
    // independent comms concurrently. MPI is used only between phases.
    //
    // connsPerWorker > 1 gives every worker several independent connections,
    // which a body needs when it has to prove that state from one connection
    // does not leak into a fresh one.
    void RunMultiThreadedIndependentGroups(
        ThreadDevPolicy policy, int nThreads, int connsPerWorker,
        std::function<ThreadResult(int, std::vector<ConnectionPair*>&)> body) {
        const int rank     = MPIEnvironment::world_rank;
        const int peerRank = (rank + 1) % 2;
        const int nSlots   = nThreads * connsPerWorker;
        std::vector<int> physical;
        if (policy.spread) {
            std::string reason;
            // Both ranks reach the same verdict, so the skip cannot strand a peer.
            if (!AgreedPhysicalDeviceIndices(&physical, &reason)) GTEST_SKIP() << reason;
        }
        std::vector<ThreadConnection> connections(nSlots);
        std::vector<ThreadResult> initResults(nSlots);

        g_workerDeregFailures.store(0, std::memory_order_relaxed);

        for (int slot = 0; slot < nSlots; ++slot) {
            if (InitNetIbCtx(&connections[slot].ctx) != ncclSuccess
                || connections[slot].ctx == nullptr) {
                initResults[slot].ok = false;
                initResults[slot].msg = "InitNetIb failed or returned a null context";
            }
        }
        if (!SynchronizeThreadResults(initResults, "NetIB context initialization")) {
            MPI_Barrier(MPI_COMM_WORLD);
            TeardownThreadConnections(connections, rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return;
        }

        std::vector<ThreadResult> setupResults(nSlots);
        for (int slot = 0; slot < nSlots; ++slot) {
            setupResults[slot] = SetupConnectionForThread(
                connections[slot].ctx, ResolveWorkerDev(policy, physical, slot),
                connections[slot].pair, rank, peerRank, slot * kThreadTagStride);
        }
        if (!SynchronizeThreadResults(setupResults, "NetIB connection setup")) {
            MPI_Barrier(MPI_COMM_WORLD);
            TeardownThreadConnections(connections, rank);
            MPI_Barrier(MPI_COMM_WORLD);
            return;
        }

        ThreadWorkerRun run = RunThreadWorkers(nThreads, [&](int threadIdx) {
            std::vector<ConnectionPair*> pairs;
            pairs.reserve(connsPerWorker);
            for (int c = 0; c < connsPerWorker; ++c)
                pairs.push_back(&connections[threadIdx * connsPerWorker + c].pair);
            return body(threadIdx, pairs);
        });
        VerifyThreadFanOut(run, nThreads);
        SynchronizeThreadResults(run.results, "NetIB threaded data path");

        MPI_Barrier(MPI_COMM_WORLD);
        TeardownThreadConnections(connections, rank);

        // Reduced, so both ranks fail together and the message names the rank that
        // saw it. A local-only check would fail on rank 1 alone, whose output the
        // runner mutes, leaving a result mismatch with no visible reason.
        const int localDeregFailures = g_workerDeregFailures.load(std::memory_order_relaxed);
        int totalDeregFailures = 0;
        if (MPI_Allreduce(&localDeregFailures, &totalDeregFailures, 1, MPI_INT, MPI_SUM,
                          MPI_COMM_WORLD)
            != MPI_SUCCESS) {
            // Unchecked, a failed reduction leaves the total unset and the check
            // below passes or fails on whatever was on the stack.
            ADD_FAILURE() << "MPI_Allreduce of the worker deregMr failures failed, so the "
                             "count cannot be trusted (" << localDeregFailures
                          << " on this rank)";
            MPI_Barrier(MPI_COMM_WORLD);
            return;
        }
        EXPECT_EQ(totalDeregFailures, 0)
            << "a worker's deregMr was refused (" << localDeregFailures << " on this rank, "
            << totalDeregFailures
            << " across both), which would otherwise surface as an MR leak";
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void RunMultiThreadedIndependent(ThreadDevPolicy policy, int nThreads,
                                     std::function<ThreadResult(int, ConnectionPair&)> body) {
        RunMultiThreadedIndependentGroups(
            policy, nThreads, 1,
            [&](int threadIdx, std::vector<ConnectionPair*>& pairs) {
                return body(threadIdx, *pairs[0]);
            });
    }

    void RunMultiThreadedIndependent(int dev, int nThreads,
                                     std::function<ThreadResult(int, ConnectionPair&)> body) {
        RunMultiThreadedIndependent(ThreadDevPolicy::Fixed(dev), nThreads, body);
    }

    // The envelope every threaded test body shares: snapshot the RDMA resource
    // counts, run the workers, barrier so no rank captures its "after" while a
    // peer is still releasing, and fail if anything leaked. Held here so the
    // load-bearing barrier is not retyped by hand at each site.
    void RunThreadedBody(ThreadDevPolicy policy, int nThreads, const char* label,
                         std::function<ThreadResult(int, ConnectionPair&)> body) {
        const RdmaResourceCounts before = CaptureRdmaResources();
        RunMultiThreadedIndependent(policy, nThreads, std::move(body));
        MPI_Barrier(MPI_COMM_WORLD);
        AssertNoRdmaLeaks(before, CaptureRdmaResources(), label);
    }

    // Threaded size sweep: every worker walks the list on its own connection with
    // a per-worker payload seed, so a transfer delivered on the wrong connection
    // fails verification. Registers the whole buffer once and reuses that handle
    // for every size. Wraps the run in an RDMA resource leak check.
    // How a threaded size sweep gets its memory. Some serial bodies allocate and
    // register once and reuse that for every step; others allocate and register
    // exactly the current size on each step. On a fused device the second shape is
    // the point of the test, since every registration fans out across both members'
    // protection domains and caches. The allocation churns with it, as the serial
    // body does: registering sub-ranges of one block would keep handing back the
    // same cache entry once a covering registration were live. The threaded branch
    // has to match whichever its serial body does, or it quietly covers less.
    enum class SweepRegistration { Once, PerSize };

    void RunThreadedSizeSweep(ThreadDevPolicy policy, int nThreads,
                              const std::vector<size_t>& sizes, int repeats,
                              const char* label,
                              SweepRegistration registration = SweepRegistration::Once) {
        const int rank = MPIEnvironment::world_rank;
        size_t maxSize = 1;
        for (size_t size : sizes) maxSize = std::max(maxSize, size);

        RunThreadedBody(
            policy, nThreads, label, [&](int threadIdx, ConnectionPair& pair) -> ThreadResult {
                const bool perSize = (registration == SweepRegistration::PerSize);

                // Once: one allocation and one registration covering the largest
                // step, reused by all of them.
                WorkerHostBuffer whole;
                if (!perSize) {
                    whole = WorkerSetupHostBuffer(rank, pair, maxSize);
                    if (!whole.result.ok) return whole.result;
                }

                // One pattern for this worker across the whole sweep. Varying it per
                // step aliased modulo 256 -- WorkerSeed(0, 8) and WorkerSeed(8, 0) are
                // the same byte -- and workers walk the ladder independently, so in a
                // sweep that mixes sizes a 256-byte send misrouted into another
                // worker's 1-byte receive would be clamped and then pass both the size
                // and the payload check. The tag and the expected size identify the
                // step; the payload identifies the worker.
                const int workerPattern = WorkerSeed(threadIdx, 0);
                ThreadResult result;
                int tag = 0;
                for (size_t size : sizes) {
                    // PerSize: allocated and registered fresh for this step and
                    // released at the end of it, so the next step starts over.
                    WorkerHostBuffer step;
                    if (perSize) {
                        step = WorkerSetupHostBuffer(rank, pair, size);
                        if (!step.result.ok) return step.result;
                    }
                    void* buffer  = perSize ? step.buffer  : whole.buffer;
                    void* mhandle = perSize ? step.mhandle : whole.mhandle;

                    for (int repeat = 0; repeat < repeats; repeat++) {
                        const int timeout = (size > 1024 * 1024) ? kLargeTransferTimeoutMs
                                                                 : kDefaultTimeoutMs;
                        result = WorkerSendRecvPattern(rank, pair, buffer, size, tag, mhandle,
                                                       workerPattern, timeout);
                        if (!result.ok) return result;
                        tag++;
                    }
                }
                return result;
            });
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

    RdmaResourceCounts CaptureRdmaResources() {
        RdmaResourceCounts counts;
        std::string probe =
            ExecShellCommand("sh -c 'rdma resource show qp >/dev/null 2>&1 && "
                             "rdma resource show cq >/dev/null 2>&1 && "
                             "rdma resource show mr >/dev/null 2>&1 && "
                             "rdma resource show pd >/dev/null 2>&1 && echo OK'");
        if (probe.find("OK") == std::string::npos) return counts;
        // Count only objects owned by this PID, so concurrent processes on a
        // shared node cannot perturb the before/after comparison.
        //
        // `rdma resource show` emits one line per object. Kernel-owned objects
        // carry "comm [ib_core]" and no pid field at all; objects owned by a
        // userspace process carry " pid <N> comm <name> ". A process holding no
        // RDMA objects therefore matches zero lines, which is the correct answer
        // (zero), not a signal that the filter failed.
        //
        // Deliberately no fall back to a system-wide count when nothing matches:
        // that would make the two snapshots use different counting modes
        // whenever the process acquires or releases its last object between
        // them, turning a real leak into a nonsensical negative delta and an
        // unrelated neighbour process into a spurious leak failure.
        const std::string pid = std::to_string(getpid());
        const std::string pidFilter = " pid " + pid + " ";
        auto countOwned = [&](const char* resource) -> int {
            std::string raw = ExecShellCommand(
                (std::string("rdma resource show ") + resource + " 2>/dev/null").c_str());
            std::istringstream iss(raw);
            std::string line;
            int n = 0;
            while (std::getline(iss, line))
                if (!line.empty() && line.find(pidFilter) != std::string::npos) n++;
            return n;
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
