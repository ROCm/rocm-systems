/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Whitebox tests for the NET/IB remote GDR flush flag (RCCL_IB_GDR_FLUSH_FLAG=1).
//
// Communicator init enables the flag on the net context only when every rank
// can use it; the tests make that decision over MPI the same way. With it on,
// each receiver hands its sender a flag page, and the sender of a flagged
// request appends an RO=0
// RDMA write of (id + 1) to the receiver's per-QP flag after the RO=1 payload,
// so the GPU can wait on the flag instead of an iflush. Whether a request is
// flagged is chosen per irecv by ncclIbSetGpuFlushFlagHint and travels to the
// sender in the CTS FIFO entry. On completion of a flagged receive the recv
// comm publishes (nQps << 56) | (id + 1), which net.cc copies into the
// gdrFlushSeq slot the GPU reads.
//
// The tests drive the plugin directly over a 2-rank connection (rank 0
// receives, rank 1 sends) and cover the ordering rules, setup, per-request gating, CTS hint
// isolation with many outstanding receives, flag-before-data ordering as seen
// by a GPU observer, connection lifecycle and concurrent connections.
//
// Expected per-request QP count comes from NCCL_IB_QPS_PER_CONNECTION and
// NCCL_IB_SPLIT_DATA_ON_QPS, so run the binary under each configuration.

#include "NetIbMPITestBase.hpp"

#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <random>
#include <thread>

#ifdef MPI_TESTS_ENABLED

namespace {

constexpr int      kDescFirstShift = 48;
constexpr int      kDescCountShift = 56;
constexpr uint64_t kDescSeqMask    = (1ULL << kDescFirstShift) - 1;
constexpr int      kFlagStrideU64  = 128 / sizeof(uint64_t); // NCCL_GDR_FLUSH_FLAG_STRIDE
constexpr int      kMaxWaitQps     = 8;                      // NCCL_GDR_FLUSH_MAX_WAIT_QPS

struct ObserverResult {
    unsigned long long mismatches;
    unsigned long long firstBad;
    int                timedOut;
};

// Mirrors prims_simple waitGdrFlushFlag: one thread per block spins on the
// per-QP flags, the block synchronizes, then every thread reads payload.
__global__ void observeFlagsThenData(const uint64_t* flags, int first, int count, uint64_t seq,
                                     const uint8_t* data, size_t size, int seed,
                                     uint64_t timeoutTicks, ObserverResult* res) {
    __shared__ int timedOut;
    if (threadIdx.x == 0) {
        timedOut = 0;
        const uint64_t start = wall_clock64();
        for (int q = first; q < first + count && !timedOut; q++) {
            while (*(volatile const uint64_t*)(flags + q * kFlagStrideU64) < seq) {
                if (wall_clock64() - start > timeoutTicks) {
                    timedOut = 1;
                    break;
                }
            }
        }
    }
    __syncthreads();
    if (timedOut) {
        if (threadIdx.x == 0) atomicOr(&res->timedOut, 1);
        return;
    }
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
        if (data[i] != (uint8_t)((seed + i) % 256)) {
            atomicAdd(&res->mismatches, 1ULL);
            atomicMin(&res->firstBad, (unsigned long long)i);
        }
    }
}

class GdrFlushFlagTest : public NetIbMPITest {
protected:
    // Flag entries checked for stray writes beyond the connection's QPs.
    static constexpr int kFlagCheckEntries = 512;
    static constexpr int kTightTimeoutMs   = 10000;
    static constexpr int kMaxRecvs         = 8;

    static int envInt(const char* name, int def) {
        const char* v = getenv(name);
        return (v && *v) ? atoi(v) : def;
    }
    static bool flagEnvEnabled() { return envInt("RCCL_IB_GDR_FLUSH_FLAG", 0) != 0; }
    static int  configuredQps() { return std::max(1, envInt("NCCL_IB_QPS_PER_CONNECTION", 1)); }
    static int  expectedFlagCount() {
        return envInt("NCCL_IB_SPLIT_DATA_ON_QPS", 0) ? configuredQps() : 1;
    }
    // A request's flags are the nQps consecutive QPs net_ib rotates it onto.
    static int expectedFirstQp(uint64_t id) {
        return (int)((id * expectedFlagCount()) % configuredQps());
    }
    static uint64_t makeDesc(int count, int first, uint64_t seq) {
        return ((uint64_t)count << kDescCountShift) | ((uint64_t)first << kDescFirstShift) | (seq & kDescSeqMask);
    }
    static uint64_t expectedDesc(uint64_t id) {
        return makeDesc(expectedFlagCount(), expectedFirstQp(id), id + 1);
    }
    // Records request `id` as the last flagged one on each QP it used.
    static void noteFlagged(std::vector<uint64_t>& lastSeq, uint64_t id) {
        for (int q = expectedFirstQp(id); q < expectedFirstQp(id) + expectedFlagCount(); q++)
            lastSeq[q] = std::max(lastSeq[q], id + 1);
    }
    // First QP flag that differs from `lastSeq`, or -1.
    static int firstUnexpectedQpFlag(const uint64_t* flags, const std::vector<uint64_t>& lastSeq) {
        for (int q = 0; q < (int)lastSeq.size(); q++)
            if (loadFlag(flags, q) != lastSeq[q]) return q;
        return -1;
    }
    static uint64_t loadFlag(const uint64_t* flags, int q) {
        return __atomic_load_n(flags + q * kFlagStrideU64, __ATOMIC_ACQUIRE);
    }

    int gdrPtrSupport() {
        ncclNetProperties_t props;
        if (GetDeviceProperties(0, &props) != ncclSuccess) return 0;
        return props.ptrSupport & (NCCL_PTR_CUDA | NCCL_PTR_DMABUF);
    }

    // Peermem regMr() where available, otherwise a dma-buf export of the
    // page-aligned range (ncclIbRegMrDmaBuf aligns the address down itself).
    ncclResult_t RegisterGdrMemory(void* comm, void* buffer, size_t size, void** mhandle) {
        if (gdrPtrSupport() & NCCL_PTR_CUDA)
            return RegisterMemory(comm, buffer, size, NCCL_PTR_CUDA, mhandle);

        const uintptr_t pageSize = sysconf(_SC_PAGESIZE);
        const uintptr_t alignedAddr = reinterpret_cast<uintptr_t>(buffer) & ~(pageSize - 1);
        const size_t alignedSize =
            (reinterpret_cast<uintptr_t>(buffer) + size - alignedAddr + pageSize - 1) & ~(pageSize - 1);

        int fd = -1;
        uint64_t exportOffset = 0;
        hsa_status_t exportStatus = hsa_amd_portable_export_dmabuf(reinterpret_cast<const void*>(alignedAddr),
                                                                   alignedSize, &fd, &exportOffset);
        if (exportStatus != HSA_STATUS_SUCCESS || fd < 0) return ncclSystemError;

        ncclResult_t res = RegisterDmaBufMemory(comm, buffer, size, NCCL_PTR_CUDA, exportOffset, fd, mhandle);
        (void)close(fd);
        return res;
    }

    // Collective, like communicator init: both ranks must be able to use the
    // flag, and then it is enabled on this test's net contexts. Returns the
    // reason to skip, or an empty string to run.
    std::string EnableFlagForComm() {
        std::string why;
        if (!flagEnvEnabled()) why = "requires RCCL_IB_GDR_FLUSH_FLAG=1";
        else if (net_ != &ncclNetIb) why = "the flag is ncclNetIb only";
        else if (!ncclIbGdrFlushFlagLocalSupport())
            why = "adaptive routing, out-of-order RQ, ID-based matching or resiliency rules the flag out";
        else if (!gdrPtrSupport()) why = "no GDR backend (neither peermem nor dma-buf) on this device";
        else if (expectedFlagCount() > kMaxWaitQps) why = "the flag needs at most 8 QPs per request";
        int skip = why.empty() ? 0 : 1;
        MPI_Allreduce(MPI_IN_PLACE, &skip, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (skip && why.empty()) why = "peer rank cannot run the flag path";
        if (skip) return why;
        ncclIbSetGdrFlushFlag(initCtx_, 1);
        threadCtxInit_ = [](void* ctx) { ncclIbSetGdrFlushFlag(ctx, 1); };
        return std::string();
    }

    // The flag stays off on a net context the communicator did not enable it
    // on, whatever RCCL_IB_GDR_FLUSH_FLAG says.
    std::string DisabledSkipReason() {
        std::string why;
        if (net_ != &ncclNetIb) why = "the flag is ncclNetIb only";
        else if (!gdrPtrSupport()) why = "no GDR backend (neither peermem nor dma-buf) on this device";
        int skip = why.empty() ? 0 : 1;
        MPI_Allreduce(MPI_IN_PLACE, &skip, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (skip && why.empty()) why = "peer rank cannot run the disabled flag path";
        return skip ? why : std::string();
    }

    static bool AnyRankFailed() {
        int failed = HasFailure() ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        return failed != 0;
    }

    // Collective: true only when `ok` holds on every rank, so an ASSERT on the
    // result returns from both ranks together instead of leaving one blocked.
    static bool AllRanksOk(bool ok) {
        int bad = ok ? 0 : 1;
        MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        return bad == 0;
    }

    // Rank 0 reports whether the recv comm set up the flag; both ranks
    // receive the answer so they leave the test together.
    bool FlagSetUp(const ConnectionPair& pair, uint64_t** flags) {
        uint64_t* f = (MPIEnvironment::world_rank == 0) ? ncclIbGetGpuFlushFlag(pair.recvComm) : nullptr;
        int ok = (MPIEnvironment::world_rank == 0) ? (f != nullptr) : 0;
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (flags) *flags = f;
        return ok != 0;
    }

    // Polls a request without the base class's 10 ms sleep so back-to-back
    // transfers keep their natural timing.
    bool TightWait(void* req, int* sizes, std::string* err, int timeoutMs = kTightTimeoutMs) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        int done = 0;
        while (!done) {
            ncclResult_t r = TestRequest(req, &done, sizes);
            if (r != ncclSuccess) {
                if (err) *err = "test() returned " + std::to_string(r);
                return false;
            }
            if (!done && std::chrono::steady_clock::now() > deadline) {
                if (err) *err = "request did not complete within " + std::to_string(timeoutMs) + " ms";
                return false;
            }
        }
        return true;
    }

    bool TightSend(void* sendComm, void* data, size_t size, int tag, void* mh, void** req,
                   std::string* err) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kTightTimeoutMs);
        *req = nullptr;
        while (*req == nullptr) {
            ncclResult_t r = PostSend(sendComm, data, size, tag, mh, req);
            if (r != ncclSuccess) {
                if (err) *err = "isend() returned " + std::to_string(r);
                return false;
            }
            if (*req == nullptr && std::chrono::steady_clock::now() > deadline) {
                if (err) *err = "isend() found no CTS within timeout";
                return false;
            }
        }
        return true;
    }

    bool PostFlaggedRecv(void* recvComm, bool flag, int n, void** bufs, size_t* sizes, int* tags,
                         void** mhs, void** req, std::string* err) {
        ncclIbSetGpuFlushFlagHint(recvComm, flag ? 1 : 0);
        ncclResult_t r = PostRecv(recvComm, n, bufs, sizes, tags, mhs, req);
        if (r != ncclSuccess || *req == nullptr) {
            if (err) *err = "irecv() failed: rc=" + std::to_string(r);
            return false;
        }
        return true;
    }

    bool WaitFlags(const uint64_t* flags, int first, int count, uint64_t seq, int timeoutMs = kTightTimeoutMs) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        for (int q = first; q < first + count; q++) {
            while (loadFlag(flags, q) < seq) {
                if (std::chrono::steady_clock::now() > deadline) return false;
            }
        }
        return true;
    }

    // Only the nqps entries are zeroed at setup; the rest of the page is
    // whatever the allocator returned, so stray writes are detected against a
    // snapshot taken right after connect.
    static std::vector<uint64_t> SnapshotFlags(const uint64_t* flags) {
        std::vector<uint64_t> snap(kFlagCheckEntries);
        for (int q = 0; q < kFlagCheckEntries; q++) snap[q] = loadFlag(flags, q);
        return snap;
    }

    // First flag index in [from, kFlagCheckEntries) that differs from `snap`, or -1.
    static int firstStrayFlag(const uint64_t* flags, int from, const std::vector<uint64_t>& snap) {
        for (int q = from; q < kFlagCheckEntries; q++)
            if (loadFlag(flags, q) != snap[q]) return q;
        return -1;
    }

    // First of the nqps zeroed-at-setup entries that is not zero, or -1.
    static int firstNonZeroQpFlag(const uint64_t* flags) {
        for (int q = 0; q < configuredQps(); q++)
            if (loadFlag(flags, q) != 0) return q;
        return -1;
    }

    bool FlushAndWait(void* recvComm, int n, void** bufs, size_t* sizes, void** mhs, std::string* err) {
        int fs[kMaxRecvs];
        for (int i = 0; i < n; i++) fs[i] = (int)sizes[i];
        void* freq = nullptr;
        ncclResult_t r = FlushRecv(recvComm, n, bufs, fs, mhs, &freq);
        if (r != ncclSuccess) {
            if (err) *err = "iflush() returned " + std::to_string(r);
            return false;
        }
        return freq == nullptr || TightWait(freq, nullptr, err);
    }

    static bool VerifyPattern(void* buf, size_t size, int seed, std::string* err) {
        size_t errIdx = 0;
        uint8_t exp = 0, act = 0;
        if (verifyBufferData<uint8_t>(buf, size, makeBytePattern(seed), 0, 0, &errIdx, &exp, &act)) return true;
        if (err) {
            *err = "data mismatch at " + std::to_string(errIdx) + "/" + std::to_string(size) +
                   " exp=" + std::to_string(exp) + " act=" + std::to_string(act);
        }
        return false;
    }

    // Receiver-side checks after request `id` completed. The descriptor must
    // name this request and its QPs when it is flagged and stay put otherwise.
    void CheckCompletion(void* recvComm, const uint64_t* flags, const std::vector<uint64_t>& snap,
                         bool flagged, uint64_t id, uint64_t descBefore, const char* what) {
        const int count = expectedFlagCount();
        const int first = expectedFirstQp(id);
        const uint64_t desc = ncclIbGetGpuFlushFlagSeq(recvComm);
        if (flagged) {
            EXPECT_EQ(desc, expectedDesc(id)) << what << ": descriptor of flagged request id " << id;
            EXPECT_TRUE(WaitFlags(flags, first, count, id + 1))
                << what << ": flags of QPs " << first << ".." << first + count - 1 << " never reached " << id + 1;
        } else {
            EXPECT_EQ(desc, descBefore) << what << ": unflagged request id " << id << " changed the descriptor";
        }
        const int stray = firstStrayFlag(flags, configuredQps(), snap);
        EXPECT_EQ(stray, -1) << what << ": flag " << stray << " written beyond the connection's " << configuredQps()
                             << " QP(s)";
    }
};

// Each setting that breaks RO=0 ordering behind the payload rules the flag out
// on its own: out-of-order RQ, port-failover resiliency, ID-based matching,
// adaptive routing on any device, or no device. NIC fusion is per connection
// (Enabled_FusedNicKeepsRdmaReadFlush).
TEST_F(GdrFlushFlagTest, Ordering_EachDisqualifierRulesTheFlagOut) {
    constexpr int64_t kAutoMatching = -2, kByIndex = 0, kById = 1;
    const int noAr[2] = {0, 0}, oneAr[2] = {0, 1};

    EXPECT_TRUE(ncclIbGdrFlushFlagOrdered(0, 0, kAutoMatching, 2, noAr));
    EXPECT_TRUE(ncclIbGdrFlushFlagOrdered(0, 0, kByIndex, 2, noAr));

    EXPECT_FALSE(ncclIbGdrFlushFlagOrdered(1, 0, kAutoMatching, 2, noAr)) << "NCCL_IB_OOO_RQ";
    EXPECT_FALSE(ncclIbGdrFlushFlagOrdered(0, 1, kAutoMatching, 2, noAr)) << "NCCL_IB_RESILIENCY_PORT_FAILOVER";
    EXPECT_FALSE(ncclIbGdrFlushFlagOrdered(0, 0, kById, 2, noAr)) << "NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME=1";
    EXPECT_FALSE(ncclIbGdrFlushFlagOrdered(0, 0, kAutoMatching, 2, oneAr)) << "adaptive routing";
    EXPECT_FALSE(ncclIbGdrFlushFlagOrdered(0, 0, kAutoMatching, 0, noAr)) << "no IB device";
}

// Enabled: the flag page starts zeroed with a clear descriptor, and a
// receive posted before any hint call is unflagged.
TEST_F(GdrFlushFlagTest, Enabled_FlagsZeroedAndHintDefaultsOff) {
    if (!validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                   false, kMinGpusPerNode, kNoNodeLimit))
        GTEST_SKIP() << "requires exactly " << kExactTwoProcesses << " ranks";
    AssertInitAndGetDevices(nullptr);
    const std::string why = EnableFlagForComm();
    if (!why.empty()) GTEST_SKIP() << why;

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    ASSERT_SETUP_CONNECTION(0, pair, connGuard);
    uint64_t* flags = nullptr;
    ASSERT_TRUE(FlagSetUp(pair, &flags)) << "the flag is enabled but the recv comm has none";

    std::vector<uint64_t> snap;
    if (rank == 0) {
        EXPECT_EQ(ncclIbGetGpuFlushFlagSeq(pair.recvComm), 0u);
        EXPECT_EQ(firstNonZeroQpFlag(flags), -1);
        snap = SnapshotFlags(flags);
    }

    const size_t size = 65536 + 3;
    void* buffer = nullptr;
    ASSERT_TRUE(AllRanksOk(hipMalloc(&buffer, size) == hipSuccess));
    auto bufGuard = makeDeviceBufferAutoGuard(buffer);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mh = nullptr;
    ASSERT_TRUE(AllRanksOk(RegisterGdrMemory(comm, buffer, size, &mh) == ncclSuccess));
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    std::string err;
    void* req = nullptr;
    if (rank == 0) {
        void* bufs[1] = {buffer};
        size_t sizes[1] = {size};
        int tags[1] = {11};
        void* mhs[1] = {mh};
        EXPECT_EQ(PostRecv(pair.recvComm, 1, bufs, sizes, tags, mhs, &req), ncclSuccess);
        int got[1] = {0};
        EXPECT_TRUE(req && TightWait(req, got, &err)) << err;
        EXPECT_EQ(got[0], (int)size);
        CheckCompletion(pair.recvComm, flags, snap, /*flagged=*/false, 0, 0, "no-hint receive");
        EXPECT_EQ(firstStrayFlag(flags, 0, snap), -1) << "an unhinted receive wrote a flag";
        EXPECT_TRUE(FlushAndWait(pair.recvComm, 1, bufs, sizes, mhs, &err)) << err;
        EXPECT_TRUE(VerifyPattern(buffer, size, 5, &err)) << err;
    } else {
        EXPECT_EQ(initializeBufferWithPattern<uint8_t>(buffer, size, makeBytePattern(5)), hipSuccess);
        EXPECT_TRUE(TightSend(pair.sendComm, buffer, size, 11, mh, &req, &err)) << err;
        EXPECT_TRUE(TightWait(req, nullptr, &err)) << err;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// Not enabled for the communicator: no flag page, and a hint set by the caller
// has no effect.
TEST_F(GdrFlushFlagTest, Disabled_NoFlagAndHintIgnored) {
    if (!validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                   false, kMinGpusPerNode, kNoNodeLimit))
        GTEST_SKIP() << "requires exactly " << kExactTwoProcesses << " ranks";
    AssertInitAndGetDevices(nullptr);
    const std::string why = DisabledSkipReason();
    if (!why.empty()) GTEST_SKIP() << why;

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    ASSERT_SETUP_CONNECTION(0, pair, connGuard);
    EXPECT_FALSE(FlagSetUp(pair, nullptr));

    const size_t size = 4097;
    void* buffer = nullptr;
    ASSERT_TRUE(AllRanksOk(hipMalloc(&buffer, size) == hipSuccess));
    auto bufGuard = makeDeviceBufferAutoGuard(buffer);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mh = nullptr;
    ASSERT_TRUE(AllRanksOk(RegisterGdrMemory(comm, buffer, size, &mh) == ncclSuccess));
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    std::string err;
    void* req = nullptr;
    for (int it = 0; it < 4; it++) {
        const int seed = 40 + it;
        if (rank == 0) {
            void* bufs[1] = {buffer};
            size_t sizes[1] = {size};
            int tags[1] = {20 + it};
            void* mhs[1] = {mh};
            EXPECT_TRUE(PostFlaggedRecv(pair.recvComm, /*flag=*/true, 1, bufs, sizes, tags, mhs, &req, &err))
                << err;
            EXPECT_TRUE(req && TightWait(req, nullptr, &err)) << err;
            EXPECT_EQ(ncclIbGetGpuFlushFlagSeq(pair.recvComm), 0u) << "descriptor published without the flag";
            EXPECT_TRUE(FlushAndWait(pair.recvComm, 1, bufs, sizes, mhs, &err)) << err;
            EXPECT_TRUE(VerifyPattern(buffer, size, seed, &err)) << err;
        } else {
            EXPECT_EQ(initializeBufferWithPattern<uint8_t>(buffer, size, makeBytePattern(seed)), hipSuccess);
            EXPECT_TRUE(TightSend(pair.sendComm, buffer, size, 20 + it, mh, &req, &err)) << err;
            EXPECT_TRUE(TightWait(req, nullptr, &err)) << err;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// Enabled on the net context, but not for this listen comm, as for a
// communicator that shares its parent's context and decided against the flag
// (ncclCommGrow with a rank that cannot use it): the receiver offers no flag
// page, so the sender writes none and the RDMA read flush stays.
TEST_F(GdrFlushFlagTest, Enabled_ListenCommOffKeepsRdmaReadFlush) {
    if (!validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                   false, kMinGpusPerNode, kNoNodeLimit))
        GTEST_SKIP() << "requires exactly " << kExactTwoProcesses << " ranks";
    AssertInitAndGetDevices(nullptr);
    const std::string why = EnableFlagForComm();
    if (!why.empty()) GTEST_SKIP() << why;
    listenCommInit_ = [](void* listenComm) { ncclIbSetListenGdrFlushFlag(listenComm, 0); };

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    ASSERT_SETUP_CONNECTION(0, pair, connGuard);
    EXPECT_FALSE(FlagSetUp(pair, nullptr)) << "the recv comm took the context's flag over the listen comm's";

    const size_t size = 8191;
    void* buffer = nullptr;
    ASSERT_TRUE(AllRanksOk(hipMalloc(&buffer, size) == hipSuccess));
    auto bufGuard = makeDeviceBufferAutoGuard(buffer);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mh = nullptr;
    ASSERT_TRUE(AllRanksOk(RegisterGdrMemory(comm, buffer, size, &mh) == ncclSuccess));
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    std::string err;
    void* req = nullptr;
    for (int it = 0; it < 2; it++) {
        const int seed = 60 + it;
        if (rank == 0) {
            void* bufs[1] = {buffer};
            size_t sizes[1] = {size};
            int tags[1] = {30 + it};
            void* mhs[1] = {mh};
            EXPECT_TRUE(PostFlaggedRecv(pair.recvComm, /*flag=*/true, 1, bufs, sizes, tags, mhs, &req, &err))
                << err;
            EXPECT_TRUE(req && TightWait(req, nullptr, &err)) << err;
            EXPECT_EQ(ncclIbGetGpuFlushFlagSeq(pair.recvComm), 0u) << "descriptor published without the flag";
            EXPECT_TRUE(FlushAndWait(pair.recvComm, 1, bufs, sizes, mhs, &err)) << err;
            EXPECT_TRUE(VerifyPattern(buffer, size, seed, &err)) << err;
        } else {
            EXPECT_EQ(initializeBufferWithPattern<uint8_t>(buffer, size, makeBytePattern(seed)), hipSuccess);
            EXPECT_TRUE(TightSend(pair.sendComm, buffer, size, 30 + it, mh, &req, &err)) << err;
            EXPECT_TRUE(TightWait(req, nullptr, &err)) << err;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// Enabled for the communicator, but over a fused (two-NIC) device: the flag WR
// would carry one NIC's keys on the other NIC's QPs, so the connection keeps
// the RDMA read flush and a flagged hint has no effect. Both sides fuse with
// makeVDevice here.
TEST_F(GdrFlushFlagTest, Enabled_FusedNicKeepsRdmaReadFlush) {
    if (!validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                   false, kMinGpusPerNode, kNoNodeLimit))
        GTEST_SKIP() << "requires exactly " << kExactTwoProcesses << " ranks";
    int ndev = 0;
    AssertInitAndGetDevices(&ndev);
    if (!AllRanksOk(ndev >= 2)) GTEST_SKIP() << "requires 2 IB devices on every rank to fuse";
    const std::string why = EnableFlagForComm();
    if (!why.empty()) GTEST_SKIP() << why;

    ncclNetVDeviceProps_t vProps = {};
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;
    int vdev = -1;
    ASSERT_TRUE(AllRanksOk(MakeVirtualDevice(&vdev, &vProps) == ncclSuccess && vdev >= 0))
        << "failed to fuse devices 0 and 1";

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    ASSERT_SETUP_CONNECTION(vdev, pair, connGuard);
    EXPECT_FALSE(FlagSetUp(pair, nullptr)) << "the flag was set up over a fused NIC";

    const size_t size = 65536 + 3;
    void* buffer = nullptr;
    ASSERT_TRUE(AllRanksOk(hipMalloc(&buffer, size) == hipSuccess));
    auto bufGuard = makeDeviceBufferAutoGuard(buffer);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mh = nullptr;
    ASSERT_TRUE(AllRanksOk(RegisterGdrMemory(comm, buffer, size, &mh) == ncclSuccess));
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    std::string err;
    void* req = nullptr;
    const int seed = 60;
    if (rank == 0) {
        void* bufs[1] = {buffer};
        size_t sizes[1] = {size};
        int tags[1] = {60};
        void* mhs[1] = {mh};
        EXPECT_TRUE(PostFlaggedRecv(pair.recvComm, /*flag=*/true, 1, bufs, sizes, tags, mhs, &req, &err)) << err;
        EXPECT_TRUE(req && TightWait(req, nullptr, &err)) << err;
        EXPECT_EQ(ncclIbGetGpuFlushFlagSeq(pair.recvComm), 0u) << "descriptor published without the flag";
        EXPECT_TRUE(FlushAndWait(pair.recvComm, 1, bufs, sizes, mhs, &err)) << err;
        EXPECT_TRUE(VerifyPattern(buffer, size, seed, &err)) << err;
    } else {
        EXPECT_EQ(initializeBufferWithPattern<uint8_t>(buffer, size, makeBytePattern(seed)), hipSuccess);
        EXPECT_TRUE(TightSend(pair.sendComm, buffer, size, 60, mh, &req, &err)) << err;
        EXPECT_TRUE(TightWait(req, nullptr, &err)) << err;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// Sequential receives with a random hint and size each time. Flagged requests
// must publish their own descriptor and flags and need no iflush; unflagged
// ones must leave both untouched. Sizes straddle the split-data threshold and
// include sub-chunk sizes that leave some QPs with zero-length writes.
TEST_F(GdrFlushFlagTest, PerRequestGating_RandomHintsAndSizes) {
    if (!validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                   false, kMinGpusPerNode, kNoNodeLimit))
        GTEST_SKIP() << "requires exactly " << kExactTwoProcesses << " ranks";
    AssertInitAndGetDevices(nullptr);
    const std::string why = EnableFlagForComm();
    if (!why.empty()) GTEST_SKIP() << why;

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    ASSERT_SETUP_CONNECTION(0, pair, connGuard);
    uint64_t* flags = nullptr;
    ASSERT_TRUE(FlagSetUp(pair, &flags));
    const std::vector<uint64_t> snap = (rank == 0) ? SnapshotFlags(flags) : std::vector<uint64_t>();

    const size_t kSizes[] = {1, 7, 4097, 65536 + 3, 1 << 20, (4 << 20) + 129};
    const size_t maxSize = *std::max_element(std::begin(kSizes), std::end(kSizes));
    void* buffer = nullptr;
    ASSERT_TRUE(AllRanksOk(hipMalloc(&buffer, maxSize) == hipSuccess));
    auto bufGuard = makeDeviceBufferAutoGuard(buffer);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mh = nullptr;
    ASSERT_TRUE(AllRanksOk(RegisterGdrMemory(comm, buffer, maxSize, &mh) == ncclSuccess));
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    std::mt19937 rng(20261019);
    const int kIters = 300;
    std::vector<uint64_t> lastSeq(configuredQps(), 0);
    std::string err;
    for (int it = 0; it < kIters; it++) {
        if (AnyRankFailed()) break;
        const bool flagged = rng() & 1;
        const size_t size = kSizes[rng() % (sizeof(kSizes) / sizeof(kSizes[0]))];
        const int seed = (it * 13 + 1) % 256;
        const int tag = 1000 + it;
        void* req = nullptr;
        if (rank == 0) {
            void* bufs[1] = {buffer};
            size_t sizes[1] = {size};
            int tags[1] = {tag};
            void* mhs[1] = {mh};
            const uint64_t descBefore = ncclIbGetGpuFlushFlagSeq(pair.recvComm);
            const bool posted = PostFlaggedRecv(pair.recvComm, flagged, 1, bufs, sizes, tags, mhs, &req, &err);
            EXPECT_TRUE(posted) << err;
            int got[1] = {0};
            const bool completed = posted && TightWait(req, got, &err);
            EXPECT_TRUE(completed) << "iter " << it << ": " << err;
            if (!completed) continue;
            EXPECT_EQ(got[0], (int)size) << "iter " << it;
            CheckCompletion(pair.recvComm, flags, snap, flagged, it, descBefore, "gating");
            if (flagged) {
                noteFlagged(lastSeq, it);
            } else {
                EXPECT_TRUE(FlushAndWait(pair.recvComm, 1, bufs, sizes, mhs, &err)) << err;
            }
            const int bad = firstUnexpectedQpFlag(flags, lastSeq);
            EXPECT_EQ(bad, -1) << "iter " << it << ": flag " << bad << " does not hold its QP's last flagged id + 1";
            EXPECT_TRUE(VerifyPattern(buffer, size, seed, &err))
                << "iter " << it << " flagged=" << flagged << " size=" << size << ": " << err;
        } else {
            const bool initialized =
                initializeBufferWithPattern<uint8_t>(buffer, size, makeBytePattern(seed)) == hipSuccess;
            EXPECT_TRUE(initialized);
            const bool posted = initialized && TightSend(pair.sendComm, buffer, size, tag, mh, &req, &err);
            EXPECT_TRUE(posted) << err;
            EXPECT_TRUE(posted && TightWait(req, nullptr, &err)) << err;
        }
    }
    (void)AnyRankFailed();
}

// Many outstanding multi-receives with mixed hints are posted before the
// sender looks at the CTS FIFO, and completions are tested out of order. Each
// completion must report its own descriptor; at the end of a round every used
// flag must hold exactly the highest flagged id + 1, which catches a sender
// that read a stale or neighbouring flagFlush field. Rounds wrap the CTS FIFO
// several times, and all-flagged rounds pile every request onto QP 0 when
// data is not split.
TEST_F(GdrFlushFlagTest, OutstandingMixedHints_CtsIsolation) {
    if (!validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                   false, kMinGpusPerNode, kNoNodeLimit))
        GTEST_SKIP() << "requires exactly " << kExactTwoProcesses << " ranks";
    AssertInitAndGetDevices(nullptr);
    const std::string why = EnableFlagForComm();
    if (!why.empty()) GTEST_SKIP() << why;

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    ASSERT_SETUP_CONNECTION(0, pair, connGuard);
    uint64_t* flags = nullptr;
    ASSERT_TRUE(FlagSetUp(pair, &flags));
    const std::vector<uint64_t> snap = (rank == 0) ? SnapshotFlags(flags) : std::vector<uint64_t>();

    constexpr int kWindow = 16;
    constexpr int kMaxN = 4;
    constexpr int kRounds = 48;
    const size_t kSizes[] = {1, 513, 4096, 65536 + 1, 262144 + 3};
    const size_t slotBytes = *std::max_element(std::begin(kSizes), std::end(kSizes));
    const size_t total = slotBytes * kWindow * kMaxN;
    char* buffer = nullptr;
    ASSERT_TRUE(AllRanksOk(hipMalloc((void**)&buffer, total) == hipSuccess));
    auto bufGuard = makeDeviceBufferAutoGuard(buffer);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mh = nullptr;
    ASSERT_TRUE(AllRanksOk(RegisterGdrMemory(comm, buffer, total, &mh) == ncclSuccess));
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    struct Op {
        bool flagged;
        int n;
        size_t sizes[kMaxN];
        int tags[kMaxN];
        int seeds[kMaxN];
        void* bufs[kMaxN];
        void* mhs[kMaxN];
        void* req;
        uint64_t id;
    };

    // `rng` must advance identically on both ranks; rank 0's completion order
    // draws from its own generator.
    std::mt19937 rng(777);
    std::mt19937 orderRng(4242);
    uint64_t nextId = 0;
    std::vector<uint64_t> lastSeq(configuredQps(), 0);
    std::string err;
    for (int round = 0; round < kRounds; round++) {
        int anyFailed = HasFailure() ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &anyFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (anyFailed) break;
        const int mode = round % 4;  // 0,1: random hints; 2: all flagged; 3: none flagged
        std::vector<Op> ops(kWindow);
        for (int w = 0; w < kWindow; w++) {
            Op& op = ops[w];
            op.flagged = (mode == 2) || (mode < 2 && (rng() & 1));
            op.n = 1 + rng() % kMaxN;
            for (int r = 0; r < op.n; r++) {
                op.sizes[r] = kSizes[rng() % (sizeof(kSizes) / sizeof(kSizes[0]))];
                op.tags[r] = 100 + r;
                op.seeds[r] = (round * 131 + w * 7 + r * 3 + 1) % 256;
                op.bufs[r] = buffer + ((size_t)w * kMaxN + r) * slotBytes;
                op.mhs[r] = mh;
            }
            op.req = nullptr;
            op.id = nextId++;
        }

        // A failure here must reach both ranks: returning on one would leave the
        // other blocked in the next collective.
        int setupFailed = 0;
        if (rank == 0) {
            for (Op& op : ops) {
                if (!PostFlaggedRecv(pair.recvComm, op.flagged, op.n, op.bufs, op.sizes, op.tags, op.mhs, &op.req,
                                     &err)) {
                    ADD_FAILURE() << "round " << round << ": " << err;
                    setupFailed = 1;
                    break;
                }
            }
        } else {
            for (Op& op : ops)
                for (int r = 0; r < op.n && !setupFailed; r++)
                    if (initializeBufferWithPattern<uint8_t>(op.bufs[r], op.sizes[r], makeBytePattern(op.seeds[r])) !=
                        hipSuccess) {
                        ADD_FAILURE() << "round " << round << ": buffer init failed";
                        setupFailed = 1;
                    }
        }
        MPI_Allreduce(MPI_IN_PLACE, &setupFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (setupFailed) break;

        if (rank == 0) {
            std::vector<int> pending(kWindow);
            for (int w = 0; w < kWindow; w++) pending[w] = w;
            std::shuffle(pending.begin(), pending.end(), orderRng);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kTightTimeoutMs);
            while (!pending.empty() && !HasFailure()) {
                for (size_t i = 0; i < pending.size();) {
                    Op& op = ops[pending[i]];
                    const uint64_t descBefore = ncclIbGetGpuFlushFlagSeq(pair.recvComm);
                    int done = 0;
                    int got[kMaxN] = {0};
                    if (TestRequest(op.req, &done, got) != ncclSuccess) {
                        ADD_FAILURE() << "round " << round << " id " << op.id << ": test() failed";
                        break;
                    }
                    if (!done) {
                        i++;
                        continue;
                    }
                    for (int r = 0; r < op.n; r++)
                        EXPECT_EQ(got[r], (int)op.sizes[r]) << "round " << round << " id " << op.id << " sub " << r;
                    CheckCompletion(pair.recvComm, flags, snap, op.flagged, op.id, descBefore, "outstanding");
                    if (op.flagged) {
                        noteFlagged(lastSeq, op.id);
                    } else {
                        EXPECT_TRUE(FlushAndWait(pair.recvComm, op.n, op.bufs, op.sizes, op.mhs, &err)) << err;
                    }
                    for (int r = 0; r < op.n; r++)
                        EXPECT_TRUE(VerifyPattern(op.bufs[r], op.sizes[r], op.seeds[r], &err))
                            << "round " << round << " id " << op.id << " flagged=" << op.flagged << " sub " << r
                            << ": " << err;
                    pending.erase(pending.begin() + i);
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    ADD_FAILURE() << "round " << round << ": " << pending.size() << " receives never completed";
                    break;
                }
            }
            const int bad = firstUnexpectedQpFlag(flags, lastSeq);
            EXPECT_EQ(bad, -1) << "round " << round << ": flag " << bad << " does not hold its QP's last flagged id + 1";
        } else {
            std::vector<void*> sreqs;
            for (size_t w = 0; w < ops.size() && !HasFailure(); w++) {
                Op& op = ops[w];
                for (int r = op.n - 1; r >= 0; r--) {
                    void* sreq = nullptr;
                    if (!TightSend(pair.sendComm, op.bufs[r], op.sizes[r], op.tags[r], mh, &sreq, &err)) {
                        ADD_FAILURE() << "round " << round << " id " << op.id << ": " << err;
                        break;
                    }
                    sreqs.push_back(sreq);
                }
            }
            for (void* sreq : sreqs) {
                if (!TightWait(sreq, nullptr, &err)) {
                    ADD_FAILURE() << "round " << round << ": " << err;
                    break;
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// A GPU kernel that spins on the flags and then reads the payload, as
// prims_simple does, is launched before the sender starts. Every byte must
// already hold the new pattern: the flag write must not overtake the RO=1
// payload, on any QP. Coarse and uncached receive buffers alternate.
TEST_F(GdrFlushFlagTest, GpuObserver_PayloadVisibleOnceFlagSet) {
    if (!validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                   false, kMinGpusPerNode, kNoNodeLimit))
        GTEST_SKIP() << "requires exactly " << kExactTwoProcesses << " ranks";
    AssertInitAndGetDevices(nullptr);
    const std::string why = EnableFlagForComm();
    if (!why.empty()) GTEST_SKIP() << why;

    const int rank = MPIEnvironment::world_rank;
    ConnectionPair pair;
    NetConnectionGuard connGuard(net_);
    ASSERT_SETUP_CONNECTION(0, pair, connGuard);
    uint64_t* flags = nullptr;
    ASSERT_TRUE(FlagSetUp(pair, &flags));

    const size_t kSizes[] = {1, 4097, (1 << 20) + 3, 16 << 20, (16 << 20) + 4099};
    const size_t maxSize = *std::max_element(std::begin(kSizes), std::end(kSizes));
    void* bufs[2] = {nullptr, nullptr};
    ASSERT_TRUE(AllRanksOk(hipMalloc(&bufs[0], maxSize) == hipSuccess));
    auto bufGuard0 = makeDeviceBufferAutoGuard(bufs[0]);
    ASSERT_TRUE(AllRanksOk(hipExtMallocWithFlags(&bufs[1], maxSize, hipDeviceMallocUncached) == hipSuccess));
    auto bufGuard1 = makeDeviceBufferAutoGuard(bufs[1]);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    void* mhs[2] = {nullptr, nullptr};
    ASSERT_TRUE(AllRanksOk(RegisterGdrMemory(comm, bufs[0], maxSize, &mhs[0]) == ncclSuccess));
    NetMHandleGuard mhGuard0(mhs[0], NetMHandleDeleter(net_, comm));
    ASSERT_TRUE(AllRanksOk(RegisterGdrMemory(comm, bufs[1], maxSize, &mhs[1]) == ncclSuccess));
    NetMHandleGuard mhGuard1(mhs[1], NetMHandleDeleter(net_, comm));

    ObserverResult* res = nullptr;
    hipStream_t stream = nullptr;
    uint64_t timeoutTicks = 0;
    bool observerReady = true;
    if (rank == 0) {
        int dev = 0, rateKHz = 0;
        observerReady = hipHostMalloc((void**)&res, sizeof(*res), hipHostMallocCoherent) == hipSuccess &&
                        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess &&
                        hipGetDevice(&dev) == hipSuccess &&
                        hipDeviceGetAttribute(&rateKHz, hipDeviceAttributeWallClockRate, dev) == hipSuccess;
        timeoutTicks = (uint64_t)rateKHz * 1000 * 10;
    }
    if (!AllRanksOk(observerReady)) {
        if (stream) (void)hipStreamDestroy(stream);
        if (res) (void)hipHostFree(res);
        FAIL() << "observer setup failed on rank 0";
    }

    const int count = expectedFlagCount();
    const int kIters = 120;
    std::string err;
    for (int it = 0; it < kIters; it++) {
        if (AnyRankFailed()) break;
        const int b = it & 1;
        const size_t size = kSizes[it % (sizeof(kSizes) / sizeof(kSizes[0]))];
        const int seed = (it * 37 + 11) % 256;
        void* req = nullptr;
        int setupFailed = 0;
        if (rank == 0) {
            *res = ObserverResult{0, ULLONG_MAX, 0};
            void* rb[1] = {bufs[b]};
            size_t rs[1] = {size};
            int rt[1] = {it};
            void* rh[1] = {mhs[b]};
            if (!PostFlaggedRecv(pair.recvComm, true, 1, rb, rs, rt, rh, &req, &err)) {
                ADD_FAILURE() << err;
                setupFailed = 1;
            } else {
                hipLaunchKernelGGL(observeFlagsThenData, dim3(64), dim3(256), 0, stream, flags, expectedFirstQp(it),
                                   count, (uint64_t)it + 1, (const uint8_t*)bufs[b], size, seed, timeoutTicks, res);
                if (hipGetLastError() != hipSuccess) {
                    ADD_FAILURE() << "observer kernel launch failed";
                    setupFailed = 1;
                }
            }
        } else {
            if (initializeBufferWithPattern<uint8_t>(bufs[b], size, makeBytePattern(seed)) != hipSuccess) {
                ADD_FAILURE() << "send buffer initialization failed";
                setupFailed = 1;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &setupFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (setupFailed) break;
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            int got[1] = {0};
            EXPECT_TRUE(TightWait(req, got, &err)) << "iter " << it << ": " << err;
            EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
            EXPECT_EQ(res->timedOut, 0) << "iter " << it << ": GPU never saw flag value " << it + 1;
            EXPECT_EQ(res->mismatches, 0ull)
                << "iter " << it << " size=" << size << (b ? " uncached" : " coarse")
                << ": GPU read stale payload after the flag, first bad byte " << res->firstBad;
            EXPECT_EQ(ncclIbGetGpuFlushFlagSeq(pair.recvComm), expectedDesc(it));
        } else {
            const bool posted = TightSend(pair.sendComm, bufs[b], size, it, mhs[b], &req, &err);
            EXPECT_TRUE(posted) << err;
            EXPECT_TRUE(posted && TightWait(req, nullptr, &err)) << err;
        }
    }
    if (stream) (void)hipStreamDestroy(stream);
    if (res) (void)hipHostFree(res);
    MPI_Barrier(MPI_COMM_WORLD);
}

// Connect, move flagged and unflagged data, close; repeated. Each new recv comm
// must start with a zeroed flag page and clear descriptor, and the cycles must
// not leak RDMA objects or the flag page's device memory.
TEST_F(GdrFlushFlagTest, Lifecycle_ReconnectFreshStateNoLeaks) {
    if (!validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                   false, kMinGpusPerNode, kNoNodeLimit))
        GTEST_SKIP() << "requires exactly " << kExactTwoProcesses << " ranks";
    AssertInitAndGetDevices(nullptr);
    const std::string why = EnableFlagForComm();
    if (!why.empty()) GTEST_SKIP() << why;

    const int rank = MPIEnvironment::world_rank;
    const size_t size = 65536 + 5;
    void* buffer = nullptr;
    ASSERT_TRUE(AllRanksOk(hipMalloc(&buffer, size) == hipSuccess));
    auto bufGuard = makeDeviceBufferAutoGuard(buffer);

    RdmaResourceCounts before;
    size_t freeBefore = 0, totalMem = 0;
    // A leaked 64 KiB flag page per cycle must exceed `slack` below, even when
    // the allocator hands pages out of larger cached fragments.
    constexpr int kCycles = 64;
    std::string err;
    for (int cycle = 0; cycle <= kCycles; cycle++) {
        if (AnyRankFailed()) break;
        if (cycle == 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            before = CaptureRdmaResources();
            ASSERT_TRUE(AllRanksOk(hipMemGetInfo(&freeBefore, &totalMem) == hipSuccess));
        }
        ConnectionPair pair;
        NetConnectionGuard connGuard(net_);
        ASSERT_SETUP_CONNECTION(0, pair, connGuard);
        uint64_t* flags = nullptr;
        ASSERT_TRUE(FlagSetUp(pair, &flags)) << "cycle " << cycle;
        std::vector<uint64_t> snap;
        if (rank == 0) {
            EXPECT_EQ(ncclIbGetGpuFlushFlagSeq(pair.recvComm), 0u) << "cycle " << cycle;
            EXPECT_EQ(firstNonZeroQpFlag(flags), -1) << "cycle " << cycle << ": QP flags not zeroed";
            snap = SnapshotFlags(flags);
        }
        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        void* mh = nullptr;
        ASSERT_TRUE(AllRanksOk(RegisterGdrMemory(comm, buffer, size, &mh) == ncclSuccess));
        NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

        const bool hints[] = {true, false, true};
        uint64_t descBefore = 0;
        for (int i = 0; i < 3; i++) {
            const int seed = cycle * 3 + i + 1;
            void* req = nullptr;
            if (rank == 0) {
                void* rb[1] = {buffer};
                size_t rs[1] = {size};
                int rt[1] = {i};
                void* rh[1] = {mh};
                descBefore = ncclIbGetGpuFlushFlagSeq(pair.recvComm);
                if (!PostFlaggedRecv(pair.recvComm, hints[i], 1, rb, rs, rt, rh, &req, &err)) {
                    ADD_FAILURE() << "cycle " << cycle << ": " << err;
                    break;
                }
                if (!TightWait(req, nullptr, &err)) {
                    ADD_FAILURE() << "cycle " << cycle << ": " << err;
                    break;
                }
                CheckCompletion(pair.recvComm, flags, snap, hints[i], i, descBefore, "lifecycle");
                if (!hints[i]) EXPECT_TRUE(FlushAndWait(pair.recvComm, 1, rb, rs, rh, &err)) << err;
                EXPECT_TRUE(VerifyPattern(buffer, size, seed, &err)) << "cycle " << cycle << ": " << err;
            } else {
                if (initializeBufferWithPattern<uint8_t>(buffer, size, makeBytePattern(seed)) != hipSuccess) {
                    ADD_FAILURE() << "cycle " << cycle << ": send buffer initialization failed";
                    break;
                }
                if (!TightSend(pair.sendComm, buffer, size, i, mh, &req, &err) || !TightWait(req, nullptr, &err)) {
                    ADD_FAILURE() << "cycle " << cycle << ": " << err;
                    break;
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    AssertNoRdmaLeaks(before, CaptureRdmaResources(), "flag lifecycle");
    size_t freeAfter = 0;
    ASSERT_EQ(hipMemGetInfo(&freeAfter, &totalMem), hipSuccess);
    // kCycles leaked flag pages are 4 MiB.
    const size_t slack = 1u << 20;
    EXPECT_GE(freeAfter + slack, freeBefore)
        << "device memory dropped by " << (freeBefore - freeAfter) << " bytes over " << kCycles << " cycles";
}

// Independent connections driven from concurrent threads, as with rccl-tests
// -t. Each recv comm owns its hint, flag page and descriptor, so random
// per-thread hints must never bleed across connections.
TEST_F(GdrFlushFlagTest, MultiThread_IndependentCommsKeepOwnState) {
    if (!validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                   false, kMinGpusPerNode, kNoNodeLimit))
        GTEST_SKIP() << "requires exactly " << kExactTwoProcesses << " ranks";
    AssertInitAndGetDevices(nullptr);
    const std::string why = EnableFlagForComm();
    if (!why.empty()) GTEST_SKIP() << why;

    const int rank = MPIEnvironment::world_rank;
    int hipDev = 0;
    ASSERT_TRUE(AllRanksOk(hipGetDevice(&hipDev) == hipSuccess));
    const int count = expectedFlagCount();
    constexpr int kThreads = 4;
    constexpr int kIters = 96;
    const size_t kSizes[] = {1, 4097, 65536 + 3, (1 << 20) + 7};
    const size_t maxSize = *std::max_element(std::begin(kSizes), std::end(kSizes));

    RunMultiThreadedIndependent(0, kThreads, [&](int threadIdx, ConnectionPair& pair) -> ThreadResult {
        ThreadResult result;
        auto fail = [&](const std::string& msg) {
            result.ok = false;
            result.msg = "thread " + std::to_string(threadIdx) + ": " + msg;
            return result;
        };
        if (hipSetDevice(hipDev) != hipSuccess) return fail("hipSetDevice failed");
        uint64_t* flags = (rank == 0) ? ncclIbGetGpuFlushFlag(pair.recvComm) : nullptr;
        if (rank == 0 && !flags) return fail("flag not set up");
        if (rank == 0 && firstNonZeroQpFlag(flags) != -1) return fail("QP flags not zeroed");
        const std::vector<uint64_t> snap = (rank == 0) ? SnapshotFlags(flags) : std::vector<uint64_t>();

        void* buffer = nullptr;
        if (hipMalloc(&buffer, maxSize) != hipSuccess) return fail("hipMalloc failed");
        auto bufGuard = makeDeviceBufferAutoGuard(buffer);
        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        void* mh = nullptr;
        if (RegisterGdrMemory(comm, buffer, maxSize, &mh) != ncclSuccess) return fail("regMr failed");
        NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

        std::mt19937 rng(1000 + threadIdx);
        std::vector<uint64_t> lastSeq(configuredQps(), 0);
        std::string err;
        for (int it = 0; it < kIters; it++) {
            const bool flagged = rng() & 1;
            const size_t size = kSizes[rng() % (sizeof(kSizes) / sizeof(kSizes[0]))];
            const int seed = (threadIdx * 61 + it * 5 + 1) % 256;
            const int tag = it;
            void* req = nullptr;
            if (rank == 0) {
                void* rb[1] = {buffer};
                size_t rs[1] = {size};
                int rt[1] = {tag};
                void* rh[1] = {mh};
                const uint64_t descBefore = ncclIbGetGpuFlushFlagSeq(pair.recvComm);
                if (!PostFlaggedRecv(pair.recvComm, flagged, 1, rb, rs, rt, rh, &req, &err)) return fail(err);
                if (!TightWait(req, nullptr, &err)) return fail("iter " + std::to_string(it) + ": " + err);
                const uint64_t desc = ncclIbGetGpuFlushFlagSeq(pair.recvComm);
                if (flagged) {
                    if (desc != expectedDesc(it))
                        return fail("iter " + std::to_string(it) + ": descriptor " + std::to_string(desc));
                    if (!WaitFlags(flags, expectedFirstQp(it), count, it + 1))
                        return fail("iter " + std::to_string(it) + ": flag late");
                    noteFlagged(lastSeq, it);
                } else {
                    if (desc != descBefore) return fail("iter " + std::to_string(it) + ": unflagged moved desc");
                    if (!FlushAndWait(pair.recvComm, 1, rb, rs, rh, &err)) return fail(err);
                }
                if (firstUnexpectedQpFlag(flags, lastSeq) != -1)
                    return fail("iter " + std::to_string(it) + ": a QP flag does not hold its last flagged id + 1");
                if (firstStrayFlag(flags, configuredQps(), snap) != -1) return fail("stray flag write");
                if (!VerifyPattern(buffer, size, seed, &err)) return fail("iter " + std::to_string(it) + ": " + err);
            } else {
                if (initializeBufferWithPattern<uint8_t>(buffer, size, makeBytePattern(seed)) != hipSuccess)
                    return fail("pattern init failed");
                if (!TightSend(pair.sendComm, buffer, size, tag, mh, &req, &err)) return fail(err);
                if (!TightWait(req, nullptr, &err)) return fail(err);
            }
        }
        return result;
    });
}

}  // namespace

#endif  // MPI_TESTS_ENABLED
