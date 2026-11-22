/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-segment DMA-BUF tests for the IB-MSEG plugin on the NET proxy path.
// They require two processes, IB/RoCE with GDR, and cuMem/HIP DMA-BUF export.
// Coverage includes per-segment registration, offsets, boundary-split and
// whole-buffer transfers, flush MR selection, segment limits and layout checks,
// and regressions. Registration goes through the plugin's regMrDmaBuf with fd -1.
// Rank 0 receives, rank 1 sends, and unsupported environments are skipped.

#include "NetIbMPITestBase.hpp"
#include "NetIbMultiSegmentHelpers.hpp"
#include "HostApiHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MultiSegmentTestSupport.hpp"

#include "transport/rccl_mseg.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace RCCLNetIbTests;

extern int64_t ncclParamMultiSegmentRegister();

namespace {
constexpr int    kNumSegments = 4;
constexpr size_t kSegBytes    = 2u * 1024 * 1024; // rounded up to VMM granularity

// The library's effective switch (env or build default), so the tests follow a default change.
bool MultiSegmentRegisterOn() { return ncclParamMultiSegmentRegister() != 0; }

// Opt-in CI strictness: when set, a net other than IB-MSEG or a missing DMA-BUF, VMM or dma-buf export API fails the
// test instead of skipping.
const RCCLTestHelpers::MsegPrereq kDmaBufPrereq{"UT_NET_IB_MSEG_REQUIRE_DMABUF"};
void FailIfDmaBufRequired(const char* what) { kDmaBufPrereq.FailIfRequired(what); }

using RCCLHostApiHelpers::AllSentinel;
using RCCLHostApiHelpers::FillBuf;
using RCCLHostApiHelpers::FillSentinel;
using RCCLHostApiHelpers::VerifyBuf;
} // namespace

// Register an nSeg window and honor skip/fail from the bool helper. Also sets
// sendMh_/recvMh_ so flush tests share the same preamble as transfer tests.
#define SETUP_REGISTERED_OR_SKIP(nSeg, pair, guard, mh, comm) \
    SETUP_REGISTERED_OR_SKIP_MIN_NODES(nSeg, pair, guard, mh, comm, kMinGpusPerNode)

#define SETUP_REGISTERED_OR_SKIP_MIN_NODES(nSeg, pair, guard, mh, comm, minNodes) \
    do {                                                                          \
        if (!SetupRegistered((nSeg), (pair), (guard), &(mh), &(comm), (minNodes))) \
            GTEST_SKIP_OR_RETURN(skipReason_);                                    \
        sendMh_ = recvMh_ = (mh);                                                 \
    } while (0)

class NetIbMultiSegmentMPITest : public NetIbMPITest {
protected:
    RCCLTestHelpers::OwnedVmmPool pool_;

    void SetUp() override {
        NetIbMPITest::SetUp();
        if (HasFatalFailure() || IsSkipped()) return;
        // Every rank resolves the same net from the same environment, so no rank sync is needed.
        if (net_ != &ncclNetIbMseg) {
            FailIfDmaBufRequired("the net is not IB-MSEG");
            GTEST_SKIP() << "multi-segment registration is on IB-MSEG only "
                            "(NCCL_MULTI_SEGMENT_REGISTER=1 or NCCL_NET=IB-MSEG)";
        }
    }

    void TearDown() override {
        pool_.Release();
        NetIbMPITest::TearDown();
    }

    MultiSegmentVmmBuffer* AllocSym(int nSeg, size_t segBytes = kSegBytes) { return pool_.AllocUniform(nSeg, segBytes); }
    MultiSegmentVmmBuffer* AllocLayout(const std::vector<size_t>& segBytes) { return pool_.AllocLayout(segBytes); }

    bool SyncSkip(bool want) {
        return MPIHelpers::anyRankTrue(want);
    }

    // SyncSkip for a DMA-BUF or VMM prerequisite: a failure instead of a skip under the require knob.
    bool MissingPrereq(bool missing) {
        return kDmaBufPrereq.Skip(missing, "a DMA-BUF or VMM prerequisite is missing on at least one rank");
    }

    bool PtrSupported(int mask) {
        ncclNetProperties_t props; memset(&props, 0, sizeof(props));
        if (GetDeviceProperties(0, &props) != ncclSuccess) return false;
        return (props.ptrSupport & mask) != 0;
    }

    // The direct GDR read is the RCCL fallback (param default is 1 = scratchpad).
    // RCCL_PARAM caches on first read, so this process must be launched with
    // RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 (test_runner one-mpirun-per-test).
    static bool directGdrFlushEnabled() {
        const char* v = getenv("RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING");
        return v && atoi(v) == 0;
    }

    // One-directional transfer (rank1 -> rank0) with independent source and
    // destination registration-relative offsets. This is the general form
    // required by DeepEP-style per-peer window layouts.
    void SendRecvChunkAtOffsets(ConnectionPair& pair, void* sBuf, void* rBuf,
                                size_t srcOff, size_t dstOff, size_t size,
                                int tag, uint8_t seed) {
        const int rank = MPIEnvironment::world_rank;
        void* req = nullptr;
        if (rank == 0) {
            void* buf = static_cast<uint8_t*>(rBuf) + dstOff;
            PostSingleRecv(pair.recvComm, buf, size, tag, recvMh_, &req);
            int sz = 0;
            EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
            EXPECT_EQ(sz, static_cast<int>(size))
                << "reported recv size srcOff=" << srcOff << " dstOff=" << dstOff;
        } else {
            FillBuf(static_cast<uint8_t*>(sBuf) + srcOff, size, seed);
            void* buf = static_cast<uint8_t*>(sBuf) + srcOff;
            PostSendWithRetry(pair.sendComm, buf, size, tag, sendMh_, &req);
            int sz = 0;
            EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(rBuf) + dstOff, size, seed))
                << "data mismatch srcOff=" << srcOff << " dstOff=" << dstOff
                << " size=" << size;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Common same-offset form used by the original test matrix.
    void SendRecvChunk(ConnectionPair& pair, void* sBuf, void* rBuf,
                       size_t off, size_t size, int tag, uint8_t seed) {
        SendRecvChunkAtOffsets(pair, sBuf, rBuf, off, off, size, tag, seed);
    }

    void SendRecvChunkAtOffsetsChecked(ConnectionPair& pair, void* sBuf, void* rBuf,
                                       size_t totalSize, size_t srcOff, size_t dstOff,
                                       size_t size, int tag, uint8_t seed, uint8_t sentinel) {
        const int rank = MPIEnvironment::world_rank;
        if (rank == 0) FillSentinel(rBuf, totalSize, sentinel);
        MPI_Barrier(MPI_COMM_WORLD);

        SendRecvChunkAtOffsets(pair, sBuf, rBuf, srcOff, dstOff, size, tag, seed);

        if (rank == 0) {
            const size_t tail = totalSize - dstOff - size;
            EXPECT_TRUE(dstOff == 0 || AllSentinel(rBuf, dstOff, sentinel))
                << "bytes before destination were overwritten";
            EXPECT_TRUE(tail == 0 || AllSentinel(static_cast<uint8_t*>(rBuf) + dstOff + size, tail, sentinel))
                << "bytes after destination were overwritten";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Allocate, connect, and register an nSeg window. Returns false on skip
    // (caller GTEST_SKIP_OR_RETURN) or failed setup. ncclInvalidUsage after
    // DMA-BUF is advertised is a failure, not a skip.
    bool SetupRegistered(int nSeg, ConnectionPair& pair, NetConnectionGuard& guard,
                         void** mh, void** comm, int minNodes = kMinGpusPerNode) {
        skipReason_.clear();
        if (!validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                       false, minNodes, kNoNodeLimit)) {
            skipReason_ = (minNodes > kMinGpusPerNode)
                              ? "test requires ranks on at least two nodes"
                              : "MPI process prerequisites not met; see rank-0 output";
            return false;
        }
        if (SyncSkip(!MultiSegmentRegisterOn())) {
            skipReason_ = "requires NCCL_MULTI_SEGMENT_REGISTER=1";
            return false;
        }
        int ndev = 0; AssertInitAndGetDevices(&ndev);
        if (MissingPrereq(!PtrSupported(NCCL_PTR_DMABUF))) {
            skipReason_ = "DMA-BUF registration not supported";
            return false;
        }
        if (gdrFlushFlag_ && !EnableGdrFlushFlag()) return false;
        MultiSegmentVmmBuffer* buf = AllocSym(nSeg);
        if (MissingPrereq(buf == nullptr)) { skipReason_ = "multi-segment VMM allocation unavailable"; return false; }
        lastBuf_ = buf;
        std::string why;
        if (SetupConnectionWithGuard(0, pair, guard, &why) != ncclSuccess) {
            ADD_FAILURE() << "NET/IB connection setup failed on at least one rank (this rank: " << why << ")";
            return false;
        }
        const int rank = MPIEnvironment::world_rank;
        *comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        *mh = nullptr;
        ncclResult_t r = RegisterMultiSegmentMr(net_, *comm, *buf, mh);
#if NCCL_CUMEM_DMABUF_EXPORT_GATE
        EXPECT_EQ(r, ncclSuccess) << "multi-segment registration failed (the AIRUNTIME-2351 bug)";
        EXPECT_NE(*mh, nullptr);
#else
        (void)r;
        FailIfDmaBufRequired("dma-buf export API unavailable at build time");
        skipReason_ = "dma-buf export API unavailable at build time";
        return false;
#endif
        const bool ok = (r == ncclSuccess && *mh != nullptr);
        if (SyncSkip(!ok)) {
            if (ok) ADD_FAILURE() << "peer failed multi-segment registration";
            if (*mh != nullptr) {
                (void)net_->deregMr(*comm, *mh);
                *mh = nullptr;
            }
            return false;
        }
        return true;
    }

    // Collective, like communicator init: turns the GDR flush flag on for this test's net
    // contexts only when every rank can use it. Sets skipReason_ and returns false otherwise.
    bool EnableGdrFlushFlag() {
        const char* env = getenv("RCCL_IB_GDR_FLUSH_FLAG");
        if (!env || atoi(env) == 0) skipReason_ = "requires RCCL_IB_GDR_FLUSH_FLAG=1";
        else if (net_ != &ncclNetIbMseg) skipReason_ = "the flag is ncclNetIbMseg only";
        else if (!IbMsegGdrFlushFlagLocalSupport())
            skipReason_ = "adaptive routing, out-of-order RQ, ID-based matching or resiliency rules the flag out";
        if (SyncSkip(!skipReason_.empty())) {
            if (skipReason_.empty()) skipReason_ = "peer rank cannot run the flag path";
            return false;
        }
        IbMsegSetGdrFlushFlag(initCtx_, 1);
        threadCtxInit_ = [](void* ctx) { IbMsegSetGdrFlushFlag(ctx, 1); };
        return true;
    }

    // SetupConnection pairs ranks 0 and 1 only; this connects any listener/peer pair.
    ncclResult_t ConnectAsPair(int dev, ConnectionPair& pair, bool listener, int peer) {
        const int maxAttempts = kConnectTimeoutMs / kPollIntervalMs;
        if (listener) {
            const ncclResult_t r = CreateListenComm(dev, &pair.handle, &pair.listenComm);
            MPI_Send(&pair.handle, sizeof(pair.handle), MPI_BYTE, peer, 0, MPI_COMM_WORLD);
            if (r != ncclSuccess) return r;
            for (int i = 0; pair.recvComm == nullptr; i++) {
                if (i >= maxAttempts) return ncclInternalError;
                const ncclResult_t a = AcceptConnection(pair.listenComm, &pair.recvComm);
                if (a != ncclSuccess) return a;
                if (pair.recvComm == nullptr) usleep(kPollIntervalUs);
            }
            return ncclSuccess;
        }
        MPI_Recv(&pair.handle, sizeof(pair.handle), MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; pair.sendComm == nullptr; i++) {
            if (i >= maxAttempts) return ncclInternalError;
            const ncclResult_t c = ConnectToRemote(dev, &pair.handle, &pair.sendComm);
            if (c != ncclSuccess) return c;
            if (pair.sendComm == nullptr) usleep(kPollIntervalUs);
        }
        return ncclSuccess;
    }

    std::string            skipReason_;
    bool                   gdrFlushFlag_ = false;
    MultiSegmentVmmBuffer* lastBuf_ = nullptr;
    void* sendMh_ = nullptr;
    void* recvMh_ = nullptr;
};

// POSITIVE: register a 4-segment window and move each segment end to end.
TEST_F(NetIbMultiSegmentMPITest, PerSegmentRegistrationAndTransfer) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(kNumSegments, pair, guard, mh, comm);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    for (int s = 0; s < kNumSegments; s++)
        SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, (size_t)s * lastBuf_->segSize, lastBuf_->segSize,
                      /*tag=*/100 + s, /*seed=*/static_cast<uint8_t>(0xA0 + s));
}

// SELECTION: transfers anchored at different offsets inside each segment.
TEST_F(NetIbMultiSegmentMPITest, IntraSegmentOffsetSelection) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(kNumSegments, pair, guard, mh, comm);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    const size_t chunk = 65536;
    int tag = 200;
    for (int s = 0; s < kNumSegments; s++) {
        size_t segBase = (size_t)s * lastBuf_->segSize;
        for (size_t sub : {size_t{0}, lastBuf_->segSize / 2, lastBuf_->segSize - chunk})
            SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, segBase + sub, chunk, tag++,
                          static_cast<uint8_t>(0x10 + s));
    }
}

// Sender starts in segment 1, receiver in segment 0. A shared cursor would
// pick the wrong lkey/rkey (DeepEP-style independent offsets).
TEST_F(NetIbMultiSegmentMPITest, DeepEP_AsymmetricOffsetTransfer) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(kNumSegments, pair, guard, mh, comm);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    const size_t srcOff = lastBuf_->segSize + 4096; // sender segment 1
    const size_t dstOff = 64 * 1024;                // receiver segment 0
    const size_t size   = 128 * 1024;
    SendRecvChunkAtOffsets(pair, lastBuf_->ptr, lastBuf_->ptr,
                           srcOff, dstOff, size, /*tag=*/350, /*seed=*/0xD3);
}

// A single transfer that straddles a segment boundary succeeds because the
// sender splits the RDMA write at the receiver's boundary using
// the per-segment rkeys published in the CTS FIFO. Data must arrive intact.
TEST_F(NetIbMultiSegmentMPITest, CrossBoundaryTransferSucceeds) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(kNumSegments, pair, guard, mh, comm);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    // 64 KiB before + 64 KiB after the seg0/seg1 boundary.
    const size_t off  = lastBuf_->segSize - 65536;
    const size_t size = 131072;
    SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, off, size, /*tag=*/400, /*seed=*/0xC3);
}

// GDR flush flag over a segmented send: once each boundary-crossing receive completes,
// its QPs' flags must hold id + 1, or a GPU waiting on them would pass on a stale flag.
TEST_F(NetIbMultiSegmentMPITest, GdrFlushFlag_SegmentedSendWritesFlag) {
    gdrFlushFlag_ = true;
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(kNumSegments, pair, guard, mh, comm);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    const int rank = MPIEnvironment::world_rank;
    const uint64_t* flags = (rank == 0) ? IbMsegGetGpuFlushFlag(pair.recvComm) : nullptr;
    ASSERT_TRUE(MPIHelpers::allRanksTrue(rank != 0 || flags != nullptr))
        << "receiver did not set up the flag; both sides need one NIC";

    constexpr int    kFlagStrideU64 = 128 / sizeof(uint64_t); // NCCL_GDR_FLUSH_FLAG_STRIDE
    const char*      qpsEnv = getenv("NCCL_IB_QPS_PER_CONNECTION");
    const char*      splitEnv = getenv("NCCL_IB_SPLIT_DATA_ON_QPS");
    const int        nQps = std::max(1, qpsEnv ? atoi(qpsEnv) : 1);
    const int        count = (splitEnv && atoi(splitEnv)) ? nQps : 1;
    const size_t     size = 131072;
    for (int s = 1; s < kNumSegments; s++) {
        const uint64_t id = s - 1;
        if (rank == 0) IbMsegSetGpuFlushFlagHint(pair.recvComm, 1);
        SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, (size_t)s * lastBuf_->segSize - size / 2, size,
                      /*tag=*/450 + s, /*seed=*/static_cast<uint8_t>(0x50 + s));
        if (rank != 0) continue;
        const int first = (int)((id * count) % nQps);
        EXPECT_EQ(IbMsegGetGpuFlushFlagSeq(pair.recvComm),
                  ((uint64_t)count << 56) | ((uint64_t)first << 48) | (id + 1))
            << "descriptor after the receive across boundary " << s;
        for (int q = first; q < first + count; q++) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            uint64_t v = 0;
            while ((v = __atomic_load_n(flags + q * kFlagStrideU64, __ATOMIC_ACQUIRE)) < id + 1 &&
                   std::chrono::steady_clock::now() < deadline) {
            }
            EXPECT_EQ(v, id + 1) << "QP " << q << " flag after the receive across boundary " << s;
        }
    }
}

// Multi-node: cross independent src/dst boundaries in an 8-segment window.
// Sentinel detects a wrong lkey/rkey or length.
TEST_F(NetIbMultiSegmentMPITest, DeepEP_MultiNodeAsymmetricCrossBoundaryStress) {
    constexpr int kWideSegments = 8;
    constexpr int kIterations   = 32;

    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP_MIN_NODES(kWideSegments, pair, guard, mh, comm, /*minNodes=*/2);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    const size_t seg = lastBuf_->segSize;
    const size_t total = lastBuf_->totalSize;
    for (int i = 0; i < kIterations; ++i) {
        const auto c = RCCLTestHelpers::BoundaryStress(i, kWideSegments, seg, /*dstMul=*/5, /*dstAdd=*/1, /*rightShift=*/2);
        SendRecvChunkAtOffsetsChecked(pair, lastBuf_->ptr, lastBuf_->ptr, total,
                                      c.srcOff, c.dstOff, c.size, /*tag=*/800 + i,
                                      static_cast<uint8_t>(0x20 + i), /*sentinel=*/0xE5);
    }
}

// A transfer spanning the entire multi-segment buffer crosses every boundary
// and exercises the full splitting builder.
TEST_F(NetIbMultiSegmentMPITest, WholeBufferSingleTransfer) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(kNumSegments, pair, guard, mh, comm);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, 0, lastBuf_->totalSize, /*tag=*/500, /*seed=*/0x5E);
}

// NEGATIVE: a buffer with more than RCCL_IB_REG_SEG_DEFAULT physical segments is
// rejected at registration with ncclInvalidUsage and produces no handle.
TEST_F(NetIbMultiSegmentMPITest, ExceedsMaxSegmentsRejected) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses,
                            false, kMinGpusPerNode, kNoNodeLimit);
    int ndev = 0; AssertInitAndGetDevices(&ndev);
    if (MissingPrereq(!PtrSupported(NCCL_PTR_DMABUF))) GTEST_SKIP() << "DMA-BUF registration not supported";

    const int rank = MPIEnvironment::world_rank;
    MultiSegmentVmmBuffer* big = AllocSym(RCCL_IB_REG_SEG_DEFAULT + 1);
    if (MissingPrereq(big == nullptr)) GTEST_SKIP() << "could not allocate over-cap VMM window";

    ConnectionPair pair; NetConnectionGuard guard(net_);
    ASSERT_SETUP_CONNECTION(0, pair, guard);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    void* mh = nullptr;
    ncclResult_t r = RegisterMultiSegmentMr(net_, comm, *big, &mh);
#if NCCL_CUMEM_DMABUF_EXPORT_GATE
    EXPECT_EQ(r, ncclInvalidUsage) << "over-cap segment buffer must be rejected";
    EXPECT_EQ(mh, nullptr) << "no handle should be produced for an over-cap buffer";
#else
    (void)r;
    FailIfDmaBufRequired("dma-buf export API unavailable at build time");
    GTEST_SKIP() << "dma-buf export API unavailable at build time";
#endif
    MPI_Barrier(MPI_COMM_WORLD);
}

// An interior segment shorter than the stride (2S, S, 2S) still registers. The splitter
// accepts the layout, so both sides keep the handle.
TEST_F(NetIbMultiSegmentMPITest, NonUniformLayoutRegisters) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses,
                            false, kMinGpusPerNode, kNoNodeLimit);
    if (SyncSkip(!MultiSegmentRegisterOn())) GTEST_SKIP() << "requires NCCL_MULTI_SEGMENT_REGISTER=1";
#if !NCCL_CUMEM_DMABUF_EXPORT_GATE
    FailIfDmaBufRequired("dma-buf export API unavailable at build time");
    GTEST_SKIP() << "dma-buf export API unavailable at build time";
#endif
    int ndev = 0; AssertInitAndGetDevices(&ndev);
    if (MissingPrereq(!PtrSupported(NCCL_PTR_DMABUF))) GTEST_SKIP() << "DMA-BUF registration not supported";

    const int rank = MPIEnvironment::world_rank;
    MultiSegmentVmmBuffer* buf = AllocLayout({2 * kSegBytes, kSegBytes, 2 * kSegBytes});
    if (MissingPrereq(buf == nullptr)) GTEST_SKIP() << "non-uniform VMM allocation unavailable";
    ASSERT_NE(buf->segSizes[0], buf->segSizes[1]) << "granularity rounding made the layout uniform";

    ConnectionPair pair; NetConnectionGuard guard(net_);
    ASSERT_SETUP_CONNECTION(0, pair, guard);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    void* mh = nullptr;
    EXPECT_EQ(RegisterMultiSegmentMr(net_, comm, *buf, &mh), ncclSuccess)
        << "a non-uniform segment layout must register";
    EXPECT_NE(mh, nullptr);
    if (mh != nullptr) EXPECT_EQ(ncclSuccess, net_->deregMr(comm, mh));
    MPI_Barrier(MPI_COMM_WORLD);
}

// OFF SWITCH: NCCL_NET=IB-MSEG with NCCL_MULTI_SEGMENT_REGISTER=0 keeps the legacy CTS on both ends,
// so the receiver declines a multi-segment registration and the caller falls back to staging.
// The sender only splits locally onto the receiver's single rkey, so it still registers.
TEST_F(NetIbMultiSegmentMPITest, MultiSegmentRegisterOff_DeclinesRegistration) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses,
                            false, kMinGpusPerNode, kNoNodeLimit);
    if (SyncSkip(MultiSegmentRegisterOn()))
        GTEST_SKIP() << "Requires NCCL_MULTI_SEGMENT_REGISTER=0";
    int ndev = 0; AssertInitAndGetDevices(&ndev);
    if (MissingPrereq(!PtrSupported(NCCL_PTR_DMABUF))) GTEST_SKIP() << "DMA-BUF registration not supported";

    const int rank = MPIEnvironment::world_rank;
    MultiSegmentVmmBuffer* buf = AllocSym(kNumSegments);
    if (MissingPrereq(buf == nullptr)) GTEST_SKIP() << "multi-segment VMM allocation unavailable";

    ConnectionPair pair; NetConnectionGuard guard(net_);
    ASSERT_SETUP_CONNECTION(0, pair, guard);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    void* mh = nullptr;
    ncclResult_t r = RegisterMultiSegmentMr(net_, comm, *buf, &mh);
#if NCCL_CUMEM_DMABUF_EXPORT_GATE
    if (rank == 0) {
        EXPECT_EQ(r, ncclInvalidUsage)
            << "NCCL_MULTI_SEGMENT_REGISTER=0 receiver still registered a multi-segment buffer";
        EXPECT_EQ(mh, nullptr);
        if (mh != nullptr) (void)net_->deregMr(comm, mh);
    } else {
        EXPECT_EQ(r, ncclSuccess) << "sender multi-segment registration must not depend on the receiver's CTS";
        EXPECT_NE(mh, nullptr);
        if (mh != nullptr) EXPECT_EQ(ncclSuccess, net_->deregMr(comm, mh));
    }
#else
    (void)r;
    FailIfDmaBufRequired("dma-buf export API unavailable at build time");
    GTEST_SKIP() << "dma-buf export API unavailable at build time";
#endif
    MPI_Barrier(MPI_COMM_WORLD);
}

// REGRESSION: a single-segment window registered through the multi-segment
// entry point still registers and transfers. IbMsegMultiSend builds that send
// as one work request per chunk.
TEST_F(NetIbMultiSegmentMPITest, SingleSegmentThroughMultiSegPath) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(1, pair, guard, mh, comm);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, 0, 65536, /*tag=*/300, /*seed=*/0x77);
}

// iflush after a recv into a non-zero segment reads that segment's first byte.
// With GDR flush off, iflush still succeeds (no request) without a boundary error.
TEST_F(NetIbMultiSegmentMPITest, MultiSegmentFlushSelectsSegmentMr) {
    if (SyncSkip(!directGdrFlushEnabled()))
        GTEST_SKIP() << "Requires RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 "
                        "(one-read iflush; RCCL_PARAM caches per process)";
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(kNumSegments, pair, guard, mh, comm);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    const int    seg   = 2;                              // a non-zero segment
    const size_t chunk = 65536;
    const size_t off   = (size_t)seg * lastBuf_->segSize;
    const int    tag   = 600;
    const int    rank  = MPIEnvironment::world_rank;

    if (rank == 0) {
        void* rbuf = static_cast<uint8_t*>(lastBuf_->ptr) + off;
        void* req  = nullptr;
        PostSingleRecv(pair.recvComm, rbuf, chunk, tag, mh, &req);
        int sz = 0;
        EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);

        void* fbufs[1]  = {rbuf};
        int   fsizes[1] = {static_cast<int>(chunk)};
        void* fhs[1]    = {mh};
        void* freq      = nullptr;
        EXPECT_EQ(FlushRecv(pair.recvComm, 1, fbufs, fsizes, fhs, &freq), ncclSuccess)
            << "iflush must handle a multi-segment handle (segment 2) without a boundary error";
        if (freq != nullptr) {
            int fsz = 0;
            EXPECT_EQ(WaitForCompletion(freq, &fsz, kDefaultTimeoutMs), ncclSuccess)
                << "flush RDMA read did not complete";
        }
        EXPECT_TRUE(VerifyBuf(rbuf, chunk, 0xC0)) << "data mismatch after flush";
    } else {
        void* sbuf = static_cast<uint8_t*>(lastBuf_->ptr) + off;
        FillBuf(sbuf, chunk, 0xC0);
        void* req = nullptr;
        PostSendWithRetry(pair.sendComm, sbuf, chunk, tag, mh, &req);
        int sz = 0;
        EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// FLUSH: one read of a whole-window receive fences every physical segment.
// iflush must accept the full-range size and still complete.
TEST_F(NetIbMultiSegmentMPITest, MultiSegmentFlushTouchesEverySegment) {
    if (SyncSkip(!directGdrFlushEnabled()))
        GTEST_SKIP() << "Requires RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 "
                        "(one-read iflush; RCCL_PARAM caches per process)";
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(kNumSegments, pair, guard, mh, comm);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    const size_t total = lastBuf_->totalSize;
    const int    tag   = 601;
    const int    rank  = MPIEnvironment::world_rank;

    if (rank == 0) {
        void* rbuf = lastBuf_->ptr;
        void* req  = nullptr;
        PostSingleRecv(pair.recvComm, rbuf, total, tag, mh, &req);
        int sz = 0;
        EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);

        void* fbufs[1]  = {rbuf};
        int   fsizes[1] = {static_cast<int>(total)};
        void* fhs[1]    = {mh};
        void* freq      = nullptr;
        EXPECT_EQ(FlushRecv(pair.recvComm, 1, fbufs, fsizes, fhs, &freq), ncclSuccess)
            << "iflush must fence a whole-buffer receive with one read";
        if (freq != nullptr) {
            int fsz = 0;
            EXPECT_EQ(WaitForCompletion(freq, &fsz, kDefaultTimeoutMs), ncclSuccess)
                << "whole-buffer flush RDMA read did not complete";
        }
        EXPECT_TRUE(VerifyBuf(rbuf, total, 0xA5))
            << "whole-buffer payload mismatch after flush";
    } else {
        void* sbuf = lastBuf_->ptr;
        FillBuf(sbuf, total, 0xA5);
        void* req = nullptr;
        PostSendWithRetry(pair.sendComm, sbuf, total, tag, mh, &req);
        int sz = 0;
        EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// A multi-recv can combine buffers backed by different composite handles.
// One read of the last receive fences the earlier receives on the same device.
TEST_F(NetIbMultiSegmentMPITest, MultiRecvFlushTouchesEveryHandle) {
    if (SyncSkip(!directGdrFlushEnabled()))
        GTEST_SKIP() << "Requires RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 "
                        "(one-read iflush; RCCL_PARAM caches per process)";
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh0 = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(kNumSegments, pair, guard, mh0, comm);
    NetMHandleGuard mhGuard0(mh0, NetMHandleDeleter(net_, comm));

    MultiSegmentVmmBuffer* buf0 = lastBuf_;
    MultiSegmentVmmBuffer* buf1 = AllocSym(kNumSegments);
    if (MissingPrereq(buf1 == nullptr)) GTEST_SKIP() << "second multi-segment VMM allocation unavailable";
    void* mh1 = nullptr;
    ASSERT_EQ(RegisterMultiSegmentMr(net_, comm, *buf1, &mh1), ncclSuccess);
    ASSERT_NE(mh1, nullptr);
    NetMHandleGuard mhGuard1(mh1, NetMHandleDeleter(net_, comm));

    const size_t chunk = 64 * 1024;
    void* bufs[2] = {
        static_cast<uint8_t*>(buf0->ptr) + chunk,
        static_cast<uint8_t*>(buf1->ptr) + 2 * buf1->segSize + chunk,
    };
    size_t recvSizes[2] = {chunk, chunk};
    int tags[2] = {602, 603};
    void* handles[2] = {mh0, mh1};
    const int rank = MPIEnvironment::world_rank;

    if (rank == 0) {
        void* req = nullptr;
        ASSERT_EQ(PostRecv(pair.recvComm, 2, bufs, recvSizes, tags, handles, &req), ncclSuccess);
        int completedSizes[2] = {};
        EXPECT_EQ(WaitForCompletion(req, completedSizes, kLargeTransferTimeoutMs), ncclSuccess);

        int flushSizes[2] = {static_cast<int>(chunk), static_cast<int>(chunk)};
        void* flushReq = nullptr;
        EXPECT_EQ(FlushRecv(pair.recvComm, 2, bufs, flushSizes, handles, &flushReq), ncclSuccess);
        if (flushReq != nullptr) {
            int flushSize = 0;
            EXPECT_EQ(WaitForCompletion(flushReq, &flushSize, kDefaultTimeoutMs), ncclSuccess);
        }
        EXPECT_TRUE(VerifyBuf(bufs[0], chunk, 0x62));
        EXPECT_TRUE(VerifyBuf(bufs[1], chunk, 0x63));
    } else {
        void* reqs[2] = {};
        FillBuf(bufs[0], chunk, 0x62);
        FillBuf(bufs[1], chunk, 0x63);
        PostSendWithRetry(pair.sendComm, bufs[0], chunk, tags[0], handles[0], &reqs[0]);
        PostSendWithRetry(pair.sendComm, bufs[1], chunk, tags[1], handles[1], &reqs[1]);
        for (int i = 0; i < 2; i++) {
            int sentSize = 0;
            EXPECT_EQ(WaitForCompletion(reqs[i], &sentSize, kLargeTransferTimeoutMs), ncclSuccess);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// One multi-recv may mix a single-segment handle (nSegments 0 in the side
// table) with a multi-segment one whose entry crosses a boundary. Both orders
// run, so the flush also sees a single-segment final entry behind a
// multi-segment one.
TEST_F(NetIbMultiSegmentMPITest, MultiRecvMixesSingleAndMultiSegmentHandles) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mhMulti = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(kNumSegments, pair, guard, mhMulti, comm);
    NetMHandleGuard mhGuardMulti(mhMulti, NetMHandleDeleter(net_, comm));
    MultiSegmentVmmBuffer* multi = lastBuf_;

    MultiSegmentVmmBuffer* single = AllocSym(1);
    if (MissingPrereq(single == nullptr)) GTEST_SKIP() << "single-segment VMM allocation unavailable";
    void* mhSingle = nullptr;
    ASSERT_EQ(RegisterMultiSegmentMr(net_, comm, *single, &mhSingle), ncclSuccess);
    ASSERT_NE(mhSingle, nullptr);
    NetMHandleGuard mhGuardSingle(mhSingle, NetMHandleDeleter(net_, comm));

    const size_t chunk = 64 * 1024;
    void* singleBuf = static_cast<uint8_t*>(single->ptr) + chunk;
    // Starts in seg1 so a flush through segment 0's MR faults.
    void* multiBuf  = static_cast<uint8_t*>(multi->ptr) + 2 * multi->segSize - chunk / 2; // crosses seg1/seg2
    const int rank = MPIEnvironment::world_rank;

    for (int order = 0; order < 2; order++) {
        void* bufs[2]    = {singleBuf, multiBuf};
        void* handles[2] = {mhSingle, mhMulti};
        if (order == 1) { std::swap(bufs[0], bufs[1]); std::swap(handles[0], handles[1]); }
        size_t sizes[2]  = {chunk, chunk};
        int tags[2]      = {700 + 2 * order, 701 + 2 * order};
        const uint8_t seeds[2] = {static_cast<uint8_t>(0x70 + 2 * order), static_cast<uint8_t>(0x71 + 2 * order)};

        if (rank == 0) {
            void* req = nullptr;
            ASSERT_EQ(PostRecv(pair.recvComm, 2, bufs, sizes, tags, handles, &req), ncclSuccess);
            int completedSizes[2] = {};
            EXPECT_EQ(WaitForCompletion(req, completedSizes, kLargeTransferTimeoutMs), ncclSuccess);

            int flushSizes[2] = {static_cast<int>(chunk), static_cast<int>(chunk)};
            void* flushReq = nullptr;
            EXPECT_EQ(FlushRecv(pair.recvComm, 2, bufs, flushSizes, handles, &flushReq), ncclSuccess);
            if (directGdrFlushEnabled())
                EXPECT_NE(flushReq, nullptr) << "GDR flush is enabled; iflush must return a request";
            if (flushReq != nullptr) {
                int flushSize = 0;
                EXPECT_EQ(WaitForCompletion(flushReq, &flushSize, kDefaultTimeoutMs), ncclSuccess) << "order " << order;
            }
            for (int i = 0; i < 2; i++)
                EXPECT_TRUE(VerifyBuf(bufs[i], chunk, seeds[i])) << "order " << order << " entry " << i;
        } else {
            void* reqs[2] = {};
            for (int i = 0; i < 2; i++) FillBuf(bufs[i], chunk, seeds[i]);
            for (int i = 0; i < 2; i++)
                PostSendWithRetry(pair.sendComm, bufs[i], chunk, tags[i], handles[i], &reqs[i]);
            for (int i = 0; i < 2; i++) {
                int sentSize = 0;
                EXPECT_EQ(WaitForCompletion(reqs[i], &sentSize, kLargeTransferTimeoutMs), ncclSuccess);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// nSegments==1 host (NCCL_PTR_HOST) and device (NCCL_PTR_CUDA) buffers still
// register and transfer on the unsegmented fast path.
TEST_F(NetIbMultiSegmentMPITest, HostAndDeviceSingleSegmentRegression) {
    SKIP_UNLESS_MPI_PREREQS(kExactTwoProcesses, kExactTwoProcesses,
                            false, kMinGpusPerNode, kNoNodeLimit);
    int ndev = 0; AssertInitAndGetDevices(&ndev);

    const int    rank = MPIEnvironment::world_rank;
    const size_t size = 65536;

    ConnectionPair pair; NetConnectionGuard guard(net_);
    ASSERT_SETUP_CONNECTION(0, pair, guard);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    // --- Host memory (NCCL_PTR_HOST): always available. ---
    {
        std::vector<uint8_t> host(size, 0);
        void* mh = nullptr;
        ASSERT_EQ(RegisterMemory(comm, host.data(), size, NCCL_PTR_HOST, &mh), ncclSuccess);
        ASSERT_NE(mh, nullptr);
        NetMHandleGuard g(mh, NetMHandleDeleter(net_, comm));

        const int tag = 700; void* req = nullptr;
        if (rank == 0) {
            PostSingleRecv(pair.recvComm, host.data(), size, tag, mh, &req);
            int sz = 0;
            EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
            for (size_t i = 0; i < size; i++)
                EXPECT_EQ(host[i], static_cast<uint8_t>(0x5A + (i & 0xFF)));
        } else {
            for (size_t i = 0; i < size; i++) host[i] = static_cast<uint8_t>(0x5A + (i & 0xFF));
            PostSendWithRetry(pair.sendComm, host.data(), size, tag, mh, &req);
            int sz = 0;
            EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // --- Device memory (NCCL_PTR_CUDA), single contiguous MR via regMr. ---
    if (!SyncSkip(!PtrSupported(NCCL_PTR_CUDA))) {
        void* dptr = nullptr;
        bool allocOk = (hipMalloc(&dptr, size) == hipSuccess);
        void* mh = nullptr;
        ncclResult_t rr = allocOk ? RegisterMemory(comm, dptr, size, NCCL_PTR_CUDA, &mh)
                                  : ncclInvalidUsage;
        bool regOk = allocOk && rr == ncclSuccess && mh != nullptr;
        if (!SyncSkip(!regOk)) {
            NetMHandleGuard g(mh, NetMHandleDeleter(net_, comm));
            const int tag = 701; void* req = nullptr;
            if (rank == 0) {
                PostSingleRecv(pair.recvComm, dptr, size, tag, mh, &req);
                int sz = 0;
                EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
                EXPECT_TRUE(VerifyBuf(dptr, size, 0x3C));
            } else {
                FillBuf(dptr, size, 0x3C);
                PostSendWithRetry(pair.sendComm, dptr, size, tag, mh, &req);
                int sz = 0;
                EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        } else if (mh != nullptr) {
            (void)net_->deregMr(comm, mh);
        }
        if (allocOk) (void)hipFree(dptr);
    }
}

// Queue 3 groups of 8 receives over 17-segment buffers on one send QP before any Test(); every group must complete.
// Opt-in: RCCL_MSEG_SQ_STRESS=1 with NCCL_IB_QPS_PER_CONNECTION=1.
TEST_F(NetIbMultiSegmentMPITest, SendQueueFatChainOversubscribe) {
    const char* stress = std::getenv("RCCL_MSEG_SQ_STRESS");
    const bool want = stress && std::atoi(stress) != 0;
    if (SyncSkip(!want))
        GTEST_SKIP() << "set RCCL_MSEG_SQ_STRESS=1 to run the send-queue stress";

    constexpr int kSegs = 17;
    constexpr int kRecvs = 8; // NCCL_NET_IB_MAX_RECVS
    constexpr int kGroups = 3;
    constexpr int kWaitMs = 15000;

    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(kSegs, pair, guard, mh, comm);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));

    const size_t dstOff = lastBuf_->segSize / 2;
    const size_t xfer = lastBuf_->totalSize - dstOff;
    void* dst = static_cast<uint8_t*>(lastBuf_->ptr) + dstOff;
    void* src = lastBuf_->ptr;

    void* recvPtrs[kRecvs];
    size_t recvSizes[kRecvs];
    int recvTags[kRecvs];
    void* recvMhs[kRecvs];
    for (int r = 0; r < kRecvs; r++) {
        recvPtrs[r] = dst;
        recvSizes[r] = xfer;
        recvMhs[r] = mh;
    }

    const int rank = MPIEnvironment::world_rank;
    void* recvReqs[kGroups] = {};
    void* sendReqs[kGroups * kRecvs] = {};

    if (rank == 0) {
        for (int g = 0; g < kGroups; g++) {
            for (int r = 0; r < kRecvs; r++) recvTags[r] = 9000 + g * kRecvs + r;
            ASSERT_EQ(PostRecv(pair.recvComm, kRecvs, recvPtrs, recvSizes, recvTags, recvMhs,
                               &recvReqs[g]),
                      ncclSuccess);
            ASSERT_NE(recvReqs[g], nullptr);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank != 0) {
        FillBuf(src, xfer, 0x51);
        bool keepPosting = true;
        for (int g = 0; g < kGroups && keepPosting; g++) {
            for (int r = 0; r < kRecvs && keepPosting; r++) {
                const int tag = 9000 + g * kRecvs + r;
                void** req = &sendReqs[g * kRecvs + r];
                int attempts = 0;
                ncclResult_t st = ncclSuccess;
                do {
                    *req = nullptr;
                    st = PostSend(pair.sendComm, src, xfer, tag, sendMh_, req);
                    if (st != ncclSuccess) break;
                    if (*req != nullptr) break;
                    usleep(kPollIntervalUs);
                } while (++attempts < 500);
                EXPECT_EQ(st, ncclSuccess)
                    << "isend failed posting group " << g << " recv " << r
                    << " (SQ oversubscribe / fatal post)";
                EXPECT_NE(*req, nullptr)
                    << "isend never matched CTS for group " << g << " recv " << r;
                keepPosting = (st == ncclSuccess && *req != nullptr);
            }
        }
        for (int i = 0; i < kGroups * kRecvs; i++) {
            if (sendReqs[i] == nullptr) continue;
            int sz = 0;
            EXPECT_EQ(WaitForCompletion(sendReqs[i], &sz, kWaitMs), ncclSuccess)
                << "send " << i << " did not complete (SQ oversubscribe)";
        }
    } else {
        for (int g = 0; g < kGroups; g++) {
            int completed[kRecvs] = {};
            EXPECT_EQ(WaitForCompletion(recvReqs[g], completed, kWaitMs), ncclSuccess)
                << "recv group " << g << " did not complete (SQ oversubscribe)";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// SCALE: rank r < N/2 receives from r + N/2, a rank on the other node under the
// runner's ppr layout. All pairs move chunks across every boundary at once.
TEST_F(NetIbMultiSegmentMPITest, ConcurrentPairsMoveEverySegmentBoundary) {
    if (!validateTestPrerequisites(kExactTwoProcesses, MPITestConstants::kNoProcessLimit,
                                   false, kMinGpusPerNode, kNoNodeLimit))
        GTEST_SKIP() << "MPI process prerequisites not met; see rank-0 output";
    const int rank = MPIEnvironment::world_rank;
    const int half = MPIEnvironment::world_size / 2;
    if (SyncSkip(MPIEnvironment::world_size % 2 != 0)) GTEST_SKIP() << "needs an even number of ranks";
    if (SyncSkip(!MultiSegmentRegisterOn())) GTEST_SKIP() << "requires NCCL_MULTI_SEGMENT_REGISTER=1";
    int ndev = 0; AssertInitAndGetDevices(&ndev);
    if (MissingPrereq(!PtrSupported(NCCL_PTR_DMABUF))) GTEST_SKIP() << "DMA-BUF registration not supported";
#if !NCCL_CUMEM_DMABUF_EXPORT_GATE
    FailIfDmaBufRequired("dma-buf export API unavailable at build time");
    GTEST_SKIP() << "dma-buf export API unavailable at build time";
#endif
    MultiSegmentVmmBuffer* buf = AllocSym(kNumSegments);
    if (MissingPrereq(buf == nullptr)) GTEST_SKIP() << "multi-segment VMM allocation unavailable";

    const bool receiver = rank < half;
    const int peer = receiver ? rank + half : rank - half;
    const int dev = (rank % half) % std::max(1, GetPhysicalDeviceCount());
    ConnectionPair pair; NetConnectionGuard guard(net_);
    const bool connected = ConnectAsPair(dev, pair, receiver, peer) == ncclSuccess;
    if (receiver) {
        guard.setRecvComm(pair.recvComm);
        guard.setListenComm(pair.listenComm);
    } else {
        guard.setSendComm(pair.sendComm);
    }
    EXPECT_TRUE(connected) << "pair " << rank << "<->" << peer;
    if (!MPIHelpers::allRanksTrue(connected)) {
        if (connected) ADD_FAILURE() << "connection failed on a peer rank";
        return;
    }
    void* comm = receiver ? pair.recvComm : pair.sendComm;
    void* mh = nullptr;
    EXPECT_EQ(RegisterMultiSegmentMr(net_, comm, *buf, &mh), ncclSuccess);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    if (!MPIHelpers::allRanksTrue(mh != nullptr)) {
        if (mh != nullptr) ADD_FAILURE() << "multi-segment registration failed on a peer rank";
        return;
    }

    const size_t chunk = buf->segSize / 4;
    const uint8_t pairSeed = static_cast<uint8_t>(0x40 + 7 * (receiver ? rank : peer));
    for (int s = 1; s < kNumSegments; s++) {
        const size_t off = (size_t)s * buf->segSize - chunk / 2;
        uint8_t* p = static_cast<uint8_t*>(buf->ptr) + off;
        const uint8_t seed = static_cast<uint8_t>(pairSeed + s);
        void* req = nullptr;
        int sz = 0;
        // No ASSERTs here: a rank returning early would hang the final allRanksTrue.
        if (receiver) {
            FillSentinel(buf->ptr, buf->totalSize, 0xEE);
            PostSingleRecv(pair.recvComm, p, chunk, 700 + s, mh, &req);
            EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
            EXPECT_EQ(sz, static_cast<int>(chunk)) << "reported size at boundary " << s;
            EXPECT_TRUE(VerifyBuf(p, chunk, seed)) << "boundary " << s << " from rank " << peer;
            EXPECT_TRUE(AllSentinel(buf->ptr, off, 0xEE)) << "bytes before boundary " << s;
        } else {
            FillBuf(p, chunk, seed);
            PostSendWithRetry(pair.sendComm, p, chunk, 700 + s, mh, &req);
            EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
        }
    }
    EXPECT_TRUE(MPIHelpers::allRanksTrue(!HasFailure())) << "a pair failed; see that rank's output";
}

#endif // MPI_TESTS_ENABLED
