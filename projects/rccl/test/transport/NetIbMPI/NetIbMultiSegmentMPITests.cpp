/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-segment DMA-BUF tests for the classic NET/IB proxy path.
// They require two processes, IB/RoCE with GDR, and cuMem/HIP DMA-BUF export.
// Coverage includes per-segment registration, offsets, boundary-split and
// whole-buffer transfers, flush MR selection, segment limits, and regressions.
// Rank 0 receives, rank 1 sends, and unsupported environments are skipped.

#include "NetIbMPITestBase.hpp"
#include "NetIbMultiSegmentHelpers.hpp"
#include "MPIHelpers.hpp"

#include "../../../src/transport/net_ib/multiseg.h"

#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace RCCLNetIbTests;

namespace {
constexpr int    kNumSegments = 4;
constexpr size_t kSegBytes    = 2u * 1024 * 1024; // rounded up to VMM granularity
} // namespace

// SetupConnectionWithGuard uses ASSERT_EQ, which only returns from that helper.
#define ASSERT_SETUP_CONNECTION(dev, pair, guard) \
    ASSERT_NO_FATAL_FAILURE(SetupConnectionWithGuard((dev), (pair), (guard)))

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
    std::vector<MultiSegmentVmmBuffer*> owned_;

    void TearDown() override {
        for (auto* b : owned_) { FreeMultiSegmentVmm(*b); delete b; }
        owned_.clear();
        NetIbMPITest::TearDown();
    }

    MultiSegmentVmmBuffer* AllocSym(int nSeg, size_t segBytes = kSegBytes) {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return nullptr;
        auto* b = new MultiSegmentVmmBuffer();
        if (!AllocMultiSegmentVmm(dev, nSeg, segBytes, b)) { delete b; return nullptr; }
        owned_.push_back(b);
        return b;
    }

    bool SyncSkip(bool want) {
        return MPIHelpers::anyRankTrue(want);
    }

    bool PtrSupported(int mask) {
        ncclNetProperties_t props; memset(&props, 0, sizeof(props));
        if (GetDeviceProperties(0, &props) != ncclSuccess) return false;
        return (props.ptrSupport & mask) != 0;
    }

    // Per-segment iflush is the RCCL fallback (param default is 1 = scratchpad).
    // RCCL_PARAM caches on first read, so this process must be launched with
    // RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 (test_runner one-mpirun-per-test).
    static bool directGdrFlushEnabled() {
        const char* v = getenv("RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING");
        return v && atoi(v) == 0;
    }

    static void FillDevice(void* dptr, size_t size, uint8_t seed) {
        if (size == 0) return;
        std::vector<uint8_t> h(size);
        for (size_t i = 0; i < size; i++) h[i] = static_cast<uint8_t>(seed + (i & 0xFF));
        ASSERT_EQ(hipMemcpy(dptr, h.data(), size, hipMemcpyHostToDevice), hipSuccess);
    }

    static bool VerifyDevice(void* dptr, size_t size, uint8_t seed) {
        if (size == 0) return true;
        std::vector<uint8_t> h(size, 0);
        if (hipMemcpy(h.data(), dptr, size, hipMemcpyDeviceToHost) != hipSuccess) return false;
        for (size_t i = 0; i < size; i++)
            if (h[i] != static_cast<uint8_t>(seed + (i & 0xFF))) return false;
        return true;
    }

    static void FillDeviceConstant(void* dptr, size_t size, uint8_t value) {
        if (size == 0) return;
        ASSERT_EQ(hipMemset(dptr, value, size), hipSuccess);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    static bool VerifyDeviceConstant(void* dptr, size_t size, uint8_t value) {
        if (size == 0) return true;
        std::vector<uint8_t> h(size, 0);
        if (hipMemcpy(h.data(), dptr, size, hipMemcpyDeviceToHost) != hipSuccess) return false;
        for (uint8_t byte : h)
            if (byte != value) return false;
        return true;
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
        } else {
            FillDevice(static_cast<uint8_t*>(sBuf) + srcOff, size, seed);
            void* buf = static_cast<uint8_t*>(sBuf) + srcOff;
            PostSendWithRetry(pair.sendComm, buf, size, tag, sendMh_, &req);
            int sz = 0;
            EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            EXPECT_TRUE(VerifyDevice(static_cast<uint8_t*>(rBuf) + dstOff, size, seed))
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
        if (rank == 0) FillDeviceConstant(rBuf, totalSize, sentinel);
        MPI_Barrier(MPI_COMM_WORLD);

        SendRecvChunkAtOffsets(pair, sBuf, rBuf, srcOff, dstOff, size, tag, seed);

        if (rank == 0) {
            EXPECT_TRUE(VerifyDeviceConstant(rBuf, dstOff, sentinel))
                << "bytes before destination were overwritten";
            EXPECT_TRUE(VerifyDeviceConstant(static_cast<uint8_t*>(rBuf) + dstOff + size,
                                             totalSize - dstOff - size, sentinel))
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
            if (minNodes > kMinGpusPerNode)
                skipReason_ = "test requires ranks on at least two nodes";
            return false;
        }
        int ndev = 0; AssertInitAndGetDevices(&ndev);
        if (SyncSkip(!PtrSupported(NCCL_PTR_DMABUF))) {
            skipReason_ = "DMA-BUF registration not supported";
            return false;
        }
        MultiSegmentVmmBuffer* buf = AllocSym(nSeg);
        if (SyncSkip(buf == nullptr)) { skipReason_ = "multi-segment VMM allocation unavailable"; return false; }
        lastBuf_ = buf;
        SetupConnectionWithGuard(0, pair, guard);
        RETURN_FALSE_IF_GTEST_STOPPED();
        const int rank = MPIEnvironment::world_rank;
        *comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        *mh = nullptr;
        ncclResult_t r = RegisterMultiSegmentMr(*comm, *buf, mh);
        EXPECT_EQ(r, ncclSuccess) << "multi-segment registration failed (the AIRUNTIME-2351 bug)";
        EXPECT_NE(*mh, nullptr);
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

    std::string            skipReason_;
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
    const std::vector<size_t> edgeWidths = {
        size_t{1}, size_t{63}, size_t{4095}, size_t{65535}, size_t{131071}
    };

    for (int i = 0; i < kIterations; ++i) {
        const int srcBoundary = 1 + (i % (kWideSegments - 1));
        const int dstBoundary = 1 + ((i * 5 + 1) % (kWideSegments - 1));
        const size_t left = edgeWidths[static_cast<size_t>(i) % edgeWidths.size()];
        const size_t right = edgeWidths[static_cast<size_t>(i + 2) % edgeWidths.size()];
        const size_t srcOff = static_cast<size_t>(srcBoundary) * seg - left;
        const size_t dstOff = static_cast<size_t>(dstBoundary) * seg - left;
        const size_t size = left + right;

        SendRecvChunkAtOffsetsChecked(pair, lastBuf_->ptr, lastBuf_->ptr, total,
                                      srcOff, dstOff, size, /*tag=*/800 + i,
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

// NEGATIVE: a buffer with more than NCCL_IB_MAX_SEGMENTS physical segments is
// rejected at registration with ncclInvalidUsage and produces no handle. The
// wire protocol carries at most NCCL_IB_MAX_SEGMENTS segments.
TEST_F(NetIbMultiSegmentMPITest, ExceedsMaxSegmentsRejected) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit));
    int ndev = 0; AssertInitAndGetDevices(&ndev);
    if (SyncSkip(!PtrSupported(NCCL_PTR_DMABUF))) GTEST_SKIP() << "DMA-BUF registration not supported";

    const int rank = MPIEnvironment::world_rank;
    MultiSegmentVmmBuffer* big = AllocSym(NCCL_IB_MAX_SEGMENTS + 1);
    if (SyncSkip(big == nullptr)) GTEST_SKIP() << "could not allocate over-cap VMM window";

    ConnectionPair pair; NetConnectionGuard guard(net_);
    ASSERT_SETUP_CONNECTION(0, pair, guard);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    void* mh = nullptr;
    ncclResult_t r = RegisterMultiSegmentMr(comm, *big, &mh);
#if NCCL_CUMEM_DMABUF_EXPORT_GATE
    EXPECT_EQ(r, ncclInvalidUsage) << "over-cap segment buffer must be rejected";
    EXPECT_EQ(mh, nullptr) << "no handle should be produced for an over-cap buffer";
#else
    (void)r;
    GTEST_SKIP() << "dma-buf export API unavailable at build time";
#endif
    MPI_Barrier(MPI_COMM_WORLD);
}

// REGRESSION: a single-segment window registered through the multi-segment
// entry point still registers and transfers via the nSeg==1 fast path (which
// leaves ncclIbMultiSend on its original, unmodified code path).
TEST_F(NetIbMultiSegmentMPITest, SingleSegmentThroughMultiSegPath) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(1, pair, guard, mh, comm);
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, 0, 65536, /*tag=*/300, /*seed=*/0x77);
}

// iflush after a recv into a non-zero segment must use that segment's MR, not
// segment 0. With GDR flush off, iflush still succeeds (no request) without a
// boundary error.
TEST_F(NetIbMultiSegmentMPITest, MultiSegmentFlushSelectsSegmentMr) {
    if (SyncSkip(!directGdrFlushEnabled()))
        GTEST_SKIP() << "Requires RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 "
                        "(per-segment iflush chain; RCCL_PARAM caches per process)";
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
        EXPECT_TRUE(VerifyDevice(rbuf, chunk, 0xC0)) << "data mismatch after flush";
    } else {
        void* sbuf = static_cast<uint8_t*>(lastBuf_->ptr) + off;
        FillDevice(sbuf, chunk, 0xC0);
        void* req = nullptr;
        PostSendWithRetry(pair.sendComm, sbuf, chunk, tag, mh, &req);
        int sz = 0;
        EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// FLUSH: a receive that covers the whole multi-segment window must fence every
// physical segment (not only data[0]). iflush must accept the full-range size
// without a boundary error and still complete.
TEST_F(NetIbMultiSegmentMPITest, MultiSegmentFlushTouchesEverySegment) {
    if (SyncSkip(!directGdrFlushEnabled()))
        GTEST_SKIP() << "Requires RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 "
                        "(per-segment iflush chain; RCCL_PARAM caches per process)";
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
            << "iflush must fence every segment of a whole-buffer receive";
        if (freq != nullptr) {
            int fsz = 0;
            EXPECT_EQ(WaitForCompletion(freq, &fsz, kDefaultTimeoutMs), ncclSuccess)
                << "whole-buffer flush RDMA read did not complete";
        }
        EXPECT_TRUE(VerifyDevice(rbuf, total, 0xA5))
            << "whole-buffer payload mismatch after flush";
    } else {
        void* sbuf = lastBuf_->ptr;
        FillDevice(sbuf, total, 0xA5);
        void* req = nullptr;
        PostSendWithRetry(pair.sendComm, sbuf, total, tag, mh, &req);
        int sz = 0;
        EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// A multi-recv can combine buffers backed by different composite handles.
// iflush must fence every non-zero entry, not only the final receive.
TEST_F(NetIbMultiSegmentMPITest, MultiRecvFlushTouchesEveryHandle) {
    if (SyncSkip(!directGdrFlushEnabled()))
        GTEST_SKIP() << "Requires RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 "
                        "(per-segment iflush chain; RCCL_PARAM caches per process)";
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh0 = nullptr; void* comm = nullptr;
    SETUP_REGISTERED_OR_SKIP(kNumSegments, pair, guard, mh0, comm);
    NetMHandleGuard mhGuard0(mh0, NetMHandleDeleter(net_, comm));

    MultiSegmentVmmBuffer* buf0 = lastBuf_;
    MultiSegmentVmmBuffer* buf1 = AllocSym(kNumSegments);
    if (SyncSkip(buf1 == nullptr)) GTEST_SKIP() << "second multi-segment VMM allocation unavailable";
    void* mh1 = nullptr;
    ASSERT_EQ(RegisterMultiSegmentMr(comm, *buf1, &mh1), ncclSuccess);
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
        EXPECT_TRUE(VerifyDevice(bufs[0], chunk, 0x62));
        EXPECT_TRUE(VerifyDevice(bufs[1], chunk, 0x63));
    } else {
        void* reqs[2] = {};
        FillDevice(bufs[0], chunk, 0x62);
        FillDevice(bufs[1], chunk, 0x63);
        PostSendWithRetry(pair.sendComm, bufs[0], chunk, tags[0], handles[0], &reqs[0]);
        PostSendWithRetry(pair.sendComm, bufs[1], chunk, tags[1], handles[1], &reqs[1]);
        for (int i = 0; i < 2; i++) {
            int sentSize = 0;
            EXPECT_EQ(WaitForCompletion(reqs[i], &sentSize, kLargeTransferTimeoutMs), ncclSuccess);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// nSegments==1 host (NCCL_PTR_HOST) and device (NCCL_PTR_CUDA) buffers still
// register and transfer on the unsegmented fast path.
TEST_F(NetIbMultiSegmentMPITest, HostAndDeviceSingleSegmentRegression) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit));
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
                EXPECT_TRUE(VerifyDevice(dptr, size, 0x3C));
            } else {
                FillDevice(dptr, size, 0x3C);
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

// Flood 3×8-recv 16-seg chains (~249 WQEs each) onto one send QP without Test().
// Opt-in: RCCL_MSEG_SQ_STRESS=1. Use NCCL_NET=IB and NCCL_IB_QPS_PER_CONNECTION=1.
TEST_F(NetIbMultiSegmentMPITest, SendQueueFatChainOversubscribe) {
    const char* stress = std::getenv("RCCL_MSEG_SQ_STRESS");
    const bool want = stress && std::atoi(stress) != 0;
    if (SyncSkip(!want))
        GTEST_SKIP() << "set RCCL_MSEG_SQ_STRESS=1 to run the Point 1 SQ repro";
    const char* net = std::getenv("NCCL_NET");
    const bool isCast = net && (strcasecmp(net, "IB-CAST") == 0 || strcasecmp(net, "ib-cast") == 0);
    if (SyncSkip(isCast))
        GTEST_SKIP() << "CAST P2P send admits one segmented group at a time; use NCCL_NET=IB";

    constexpr int kSegs = NCCL_IB_MAX_SEGMENTS;
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
        FillDevice(src, xfer, 0x51);
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

#endif // MPI_TESTS_ENABLED
