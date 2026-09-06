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

    // Both ranks leave this together, or neither does.
    //
    // Every failure here used to return early, and each early return stranded the
    // peer: a listen failure never sent the handle, so the connector sat in
    // MPI_Recv until the suite timeout, and a failure after that skipped the
    // closing barrier, so the other side sat there instead. A caller cannot fix
    // that from outside, however carefully it checks the return value, because by
    // then its peer is already blocked. So the handle message carries a status
    // word, and the closing barrier became a reduction of it.
    ncclResult_t SetupConnection(int dev, ConnectionPair& pair, int rank, int peerRank) {
        // Cap the accept/connect handshake so a dead fabric fails fast instead
        // of spinning forever (AICOMRCCL-1577).
        const int maxAttempts = kConnectTimeoutMs / kPollIntervalMs;
        struct SetupHandshake {
            int status;  // 1 when the listener is ready and the handle is valid
            ncclNetHandle_t handle;
        } handshake = {};
        ncclResult_t local = ncclSuccess;

        if (rank == 0) {
            local = CreateListenComm(dev, &pair.handle, &pair.listenComm);
            handshake.status = (local == ncclSuccess) ? 1 : 0;
            if (local == ncclSuccess) memcpy(handshake.handle, pair.handle, sizeof(pair.handle));
            // Sent even on failure: the peer is waiting for this message.
            MPI_Send(&handshake, sizeof(handshake), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD);

            int done = 0;
            int attempts = 0;
            while (local == ncclSuccess && !done) {
                local = AcceptConnection(pair.listenComm, &pair.recvComm);
                if (local != ncclSuccess) break;
                if (pair.recvComm != nullptr) {
                    done = 1;
                    break;
                }
                if (++attempts >= maxAttempts) {
                    local = ncclInternalError;
                    break;
                }
                usleep(kPollIntervalUs);
            }
        } else {
            MPI_Recv(&handshake, sizeof(handshake), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            if (!handshake.status) {
                local = ncclRemoteError;  // the peer could not listen
            } else {
                memcpy(pair.handle, handshake.handle, sizeof(pair.handle));
                int done = 0;
                int attempts = 0;
                while (!done) {
                    local = ConnectToRemote(dev, &pair.handle, &pair.sendComm);
                    if (local != ncclSuccess) break;
                    if (pair.sendComm != nullptr) {
                        done = 1;
                        break;
                    }
                    if (++attempts >= maxAttempts) {
                        local = ncclInternalError;
                        break;
                    }
                    usleep(kPollIntervalUs);
                }
            }
        }

        // Replaces the closing barrier, so a one-sided failure ends the setup on
        // both ranks instead of leaving one of them in the barrier.
        int ok = (local == ncclSuccess) ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (local == ncclSuccess && ok) return ncclSuccess;

        // Now that the failure is recoverable rather than fatal, whatever this
        // rank did create has to be closed here: every caller asserts on this
        // result before it constructs its NetConnectionGuard, so nothing else
        // will. A leaked listener or QP would outlive the test and contaminate
        // the rest of the process. Data comms first, then the listener.
        if (pair.sendComm) { CloseSendComm(pair.sendComm); pair.sendComm = nullptr; }
        if (pair.recvComm) { CloseRecvComm(pair.recvComm); pair.recvComm = nullptr; }
        if (pair.listenComm) { CloseListenComm(pair.listenComm); pair.listenComm = nullptr; }
        return (local != ncclSuccess) ? local : ncclRemoteError;
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
    // Same contract as SetupConnection: the ranks agree before anyone asserts, so
    // a one-sided failure cannot leave the peer waiting for a handle or a barrier.
    // The assertion at the end is reached by both ranks with the same verdict,
    // which is what makes a fatal failure safe here.
    void SetupCastConnection(int dev,
                             void** listenComm, void** sendComm, void** recvComm) {
        const int rank = MPIEnvironment::world_rank;
        const int peer = 1 - rank;
        struct SetupHandshake {
            int status;
            ncclNetHandle_t handle;
        } handshake = {};
        // ncclIbListen writes a uint64_t magic through the handle pointer it's
        // given, so listen()/connect() need an 8-byte-aligned buffer rather
        // than &handshake.handle (which sits at a 4-byte offset after status).
        // This local is aligned; the handshake only ever carries a byte-copy
        // of it, same as SetupConnection's pair.handle.
        alignas(8) ncclNetHandle_t handle{};
        // Tracks this rank's own outcome, kept separate from the peer's status
        // in the handshake so the assertion message below can tell "my own
        // accept/connect failed" apart from "the peer's listen failed", instead
        // of both ranks reporting the same generic verdict.
        bool localOk = true;
        const char* localReason = "ok";

        if (rank == 0) {
            localOk = (CreateListenComm(dev, &handle, listenComm) == ncclSuccess)
                      && *listenComm != nullptr;
            handshake.status = localOk ? 1 : 0;
            if (!localOk) localReason = "listen failed";
            if (localOk) memcpy(handshake.handle, handle, sizeof(handle));
            // Sent even on failure: the peer is waiting for this message.
            MPI_Send(&handshake, sizeof(handshake), MPI_BYTE, peer, 0, MPI_COMM_WORLD);

            for (int i = 0; localOk && i < kMaxRetryAttempts && *recvComm == nullptr; i++) {
                if (AcceptConnection(*listenComm, recvComm) != ncclSuccess) {
                    localOk = false;
                    localReason = "accept failed";
                }
                if (localOk && *recvComm == nullptr) usleep(kPollIntervalUs);
            }
            if (localOk && *recvComm == nullptr) { localOk = false; localReason = "accept timed out"; }
        } else {
            MPI_Recv(&handshake, sizeof(handshake), MPI_BYTE, peer, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            if (handshake.status != 1) {
                localOk = false;
                localReason = "peer's listen failed";
            } else {
                memcpy(handle, handshake.handle, sizeof(handle));
                for (int i = 0; localOk && i < kMaxRetryAttempts && *sendComm == nullptr; i++) {
                    if (ConnectToRemote(dev, &handle, sendComm) != ncclSuccess) {
                        localOk = false;
                        localReason = "connect failed";
                    }
                    if (localOk && *sendComm == nullptr) usleep(kPollIntervalUs);
                }
                if (localOk && *sendComm == nullptr) { localOk = false; localReason = "connect timed out"; }
            }
        }

        int ok = localOk ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!ok) {
            // The assertion below is fatal and unwinds through the caller's
            // ASSERT_NO_FATAL_FAILURE before its teardown runs, so anything this
            // rank created must be released here or it stays open for the rest
            // of the process. Data comms first, then the listener.
            if (*sendComm) { CloseSendComm(*sendComm); *sendComm = nullptr; }
            if (*recvComm) { CloseRecvComm(*recvComm); *recvComm = nullptr; }
            if (*listenComm) { CloseListenComm(*listenComm); *listenComm = nullptr; }
        }
        ASSERT_EQ(ok, 1) << "IB-CAST connection setup failed on at least one rank (this rank: "
                         << localReason << ")";
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

    ThreadResult WorkerPostSend(void* sendComm, void* data, size_t size, int tag,
                                void* mhandle, void** request, bool busyPoll = false,
                                int timeoutMs = kSlotWaitDefaultMs) {
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

    // How a threaded size sweep registers memory. Some serial bodies register the
    // whole buffer once, others register exactly the current size on every step;
    // on a fused device that second shape is the point of the test, since each
    // registration fans out across both members' protection domains and caches.
    // The threaded branch has to match whichever its serial body does, or it
    // quietly covers less.
    enum class SweepRegistration { Once, PerSize };

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
