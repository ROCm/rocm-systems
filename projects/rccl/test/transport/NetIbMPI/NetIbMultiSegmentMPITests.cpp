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

#include <cstring>
#include <vector>

#ifdef MPI_TESTS_ENABLED

using namespace RCCLNetIbTests;

namespace {
constexpr int    kNumSegments = 4;
constexpr size_t kSegBytes    = 2u * 1024 * 1024; // rounded up to VMM granularity
constexpr int    kMaxSegments = 16;               // mirrors NCCL_IB_MAX_SEGMENTS
} // namespace

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

    // Common setup: prerequisites, GDR, allocate an nSeg window, connect, and
    // register it through the multi-segment entry point. Returns false if any
    // precondition is unmet; when the failure is a graceful skip it records
    // skipReason_ so the caller can GTEST_SKIP() from the test body (GTEST_SKIP
    // expands to a void return and cannot be used inside this bool helper).
    // Use SETUP_OR_SKIP() at the call site to honor a recorded skip.
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
        const int rank = MPIEnvironment::world_rank;
        *comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        ncclResult_t r = RegisterMultiSegmentMr(*comm, *buf, mh);
        if (SyncSkip(r == ncclInvalidUsage && *mh == nullptr)) {
            skipReason_ = "dma-buf multi-segment registration unavailable on this build/host";
            return false;
        }
        EXPECT_EQ(r, ncclSuccess) << "multi-segment registration failed (the AIRUNTIME-2351 bug)";
        EXPECT_NE(*mh, nullptr);
        return (r == ncclSuccess && *mh != nullptr);
    }

    std::string            skipReason_;
    MultiSegmentVmmBuffer* lastBuf_ = nullptr;
    void* sendMh_ = nullptr;
    void* recvMh_ = nullptr;
};

// Honor a graceful skip recorded by SetupRegistered from the test body (where a
// void return from GTEST_SKIP is valid). Used as: if (!SetupRegistered(...)) SETUP_OR_SKIP();
#define SETUP_OR_SKIP()                                                    \
    do {                                                                   \
        if (!skipReason_.empty()) GTEST_SKIP() << skipReason_;             \
        return;                                                            \
    } while (0)

// POSITIVE: register a 4-segment window and move each segment end to end.
TEST_F(NetIbMultiSegmentMPITest, PerSegmentRegistrationAndTransfer) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    if (!SetupRegistered(kNumSegments, pair, guard, &mh, &comm)) SETUP_OR_SKIP();
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    sendMh_ = recvMh_ = mh;
    for (int s = 0; s < kNumSegments; s++)
        SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, (size_t)s * lastBuf_->segSize, lastBuf_->segSize,
                      /*tag=*/100 + s, /*seed=*/static_cast<uint8_t>(0xA0 + s));
}

// SELECTION: transfers anchored at different offsets inside each segment.
TEST_F(NetIbMultiSegmentMPITest, IntraSegmentOffsetSelection) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    if (!SetupRegistered(kNumSegments, pair, guard, &mh, &comm)) SETUP_OR_SKIP();
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    sendMh_ = recvMh_ = mh;
    const size_t chunk = 65536;
    int tag = 200;
    for (int s = 0; s < kNumSegments; s++) {
        size_t segBase = (size_t)s * lastBuf_->segSize;
        for (size_t sub : {size_t{0}, lastBuf_->segSize / 2, lastBuf_->segSize - chunk})
            SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, segBase + sub, chunk, tag++,
                          static_cast<uint8_t>(0x10 + s));
    }
}

// DeepEP-style per-peer windows use different source and destination offsets
// for one logical transfer. Exercise that pattern on the classic CTS-FIFO wire:
// the sender starts in segment 1 while the receiver starts in segment 0, so a
// shared registration-relative cursor would select the wrong lkey or rkey.
TEST_F(NetIbMultiSegmentMPITest, DeepEP_AsymmetricOffsetTransfer) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    if (!SetupRegistered(kNumSegments, pair, guard, &mh, &comm)) SETUP_OR_SKIP();
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    sendMh_ = recvMh_ = mh;

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
    if (!SetupRegistered(kNumSegments, pair, guard, &mh, &comm)) SETUP_OR_SKIP();
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    sendMh_ = recvMh_ = mh;
    // 64 KiB before + 64 KiB after the seg0/seg1 boundary.
    const size_t off  = lastBuf_->segSize - 65536;
    const size_t size = 131072;
    SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, off, size, /*tag=*/400, /*seed=*/0xC3);
}

// MULTI-NODE STRESS: repeatedly cross independently selected source and
// destination boundaries in an 8-segment window. The destination is reset to a
// sentinel on every iteration so incorrect WR splitting, lkey/rkey selection,
// or length accounting is detected as payload corruption or an adjacent write.
TEST_F(NetIbMultiSegmentMPITest, DeepEP_MultiNodeAsymmetricCrossBoundaryStress) {
    constexpr int kWideSegments = 8;
    constexpr int kIterations   = 32;

    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    if (!SetupRegistered(kWideSegments, pair, guard, &mh, &comm, /*minNodes=*/2))
        SETUP_OR_SKIP();
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    sendMh_ = recvMh_ = mh;

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
    if (!SetupRegistered(kNumSegments, pair, guard, &mh, &comm)) SETUP_OR_SKIP();
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    sendMh_ = recvMh_ = mh;
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
    MultiSegmentVmmBuffer* big = AllocSym(kMaxSegments + 1);
    if (SyncSkip(big == nullptr)) GTEST_SKIP() << "could not allocate over-cap VMM window";

    ConnectionPair pair; NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, pair, guard);
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    void* mh = nullptr;
    ncclResult_t r = RegisterMultiSegmentMr(comm, *big, &mh);
    if (SyncSkip(r == ncclInvalidUsage && mh == nullptr && (kMaxSegments + 1) > kMaxSegments)) {
        EXPECT_EQ(mh, nullptr) << "no handle should be produced for an over-cap buffer";
        SUCCEED();
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }
    EXPECT_EQ(r, ncclInvalidUsage) << "over-cap segment buffer must be rejected";
    EXPECT_EQ(mh, nullptr);
    MPI_Barrier(MPI_COMM_WORLD);
}

// REGRESSION: a single-segment window registered through the multi-segment
// entry point still registers and transfers via the nSeg==1 fast path (which
// leaves ncclIbMultiSend on its original, unmodified code path).
TEST_F(NetIbMultiSegmentMPITest, SingleSegmentThroughMultiSegPath) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    if (!SetupRegistered(1, pair, guard, &mh, &comm)) SETUP_OR_SKIP();
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    sendMh_ = recvMh_ = mh;
    SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, 0, 65536, /*tag=*/300, /*seed=*/0x77);
}

// FLUSH: after receiving into a NON-zero segment, ncclIbIflush must fence the
// buffer using that segment's MR (rkey selected by ncclIbMrForRange), not
// segment 0's. The flush 4-byte read never straddles a boundary, so this
// validates per-segment key selection on the flush path.
// When GDR flush is disabled, iflush returns success with no request; the test
// still asserts iflush accepts a multi-segment handle without a boundary error.
TEST_F(NetIbMultiSegmentMPITest, MultiSegmentFlushSelectsSegmentMr) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    if (!SetupRegistered(kNumSegments, pair, guard, &mh, &comm)) SETUP_OR_SKIP();
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

// REGRESSION (pointer types): the segment-aware changes only affect nSegments>1
// handles; single-region host (NCCL_PTR_HOST) and device (NCCL_PTR_CUDA) buffers
// must still register and transfer through the unchanged nSegments==1 fast path
// (which also leaves ncclIbMultiSend on its original, non-segmented path).
// Covers the reviewer's "host and device allocations" request (multi-segment
// VMM itself is device-only).
TEST_F(NetIbMultiSegmentMPITest, HostAndDeviceSingleSegmentRegression) {
    ASSERT_TRUE(validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                          false, kMinGpusPerNode, kNoNodeLimit));
    int ndev = 0; AssertInitAndGetDevices(&ndev);

    const int    rank = MPIEnvironment::world_rank;
    const size_t size = 65536;

    ConnectionPair pair; NetConnectionGuard guard(net_);
    SetupConnectionWithGuard(0, pair, guard);
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

#endif // MPI_TESTS_ENABLED
