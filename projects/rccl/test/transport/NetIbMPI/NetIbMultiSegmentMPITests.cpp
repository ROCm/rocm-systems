/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-segment DMA-BUF tests for the CLASSIC NET/IB proxy path with the
// Option B (wire-protocol) segment-aware data path (AIRUNTIME-2351 classic-path
// follow-up). Unlike Option A -- which requires every transfer to stay within a
// single physical segment -- Option B publishes the receiver's full segment
// layout in the CTS FIFO and splits RDMA writes at segment boundaries, so a
// single transfer may span multiple segments.
//
// These require 2 processes, an IB/RoCE device with GDR, and the cuMem/HIP
// dma-buf export API; otherwise they GTEST_SKIP. Rank 0 is the receiver, rank 1
// the sender (matches SetupConnectionWithGuard).

#include "NetIbMPITestBase.hpp"
#include "NetIbMultiSegmentHelpers.hpp"

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
        int f = want ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &f, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        return f != 0;
    }

    bool GdrSupported() {
        ncclNetProperties_t props; memset(&props, 0, sizeof(props));
        if (GetDeviceProperties(0, &props) != ncclSuccess) return false;
        return (props.ptrSupport & NCCL_PTR_CUDA) != 0;
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

    // One-directional transfer of [off, off+size) (rank1 -> rank0) on an
    // established connection, with data verification on the receiver. With
    // Option B, size/off may span multiple physical segments.
    void SendRecvChunk(ConnectionPair& pair, void* sBuf, void* rBuf,
                       size_t off, size_t size, int tag, uint8_t seed) {
        const int rank = MPIEnvironment::world_rank;
        void* req = nullptr;
        if (rank == 0) {
            void* buf = static_cast<uint8_t*>(rBuf) + off;
            PostSingleRecv(pair.recvComm, buf, size, tag, recvMh_, &req);
            int sz = 0;
            EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
        } else {
            FillDevice(static_cast<uint8_t*>(sBuf) + off, size, seed);
            void* buf = static_cast<uint8_t*>(sBuf) + off;
            PostSendWithRetry(pair.sendComm, buf, size, tag, sendMh_, &req);
            int sz = 0;
            EXPECT_EQ(WaitForCompletion(req, &sz, kLargeTransferTimeoutMs), ncclSuccess);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
            EXPECT_TRUE(VerifyDevice(static_cast<uint8_t*>(rBuf) + off, size, seed))
                << "data mismatch off=" << off << " size=" << size;
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Common setup: prerequisites, GDR, allocate an nSeg window, connect, and
    // register it through the multi-segment entry point. Returns false (with the
    // test already SKIP/So) if any precondition is unmet; on success fills mh.
    bool SetupRegistered(int nSeg, ConnectionPair& pair, NetConnectionGuard& guard, void** mh, void** comm) {
        if (!validateTestPrerequisites(kExactTwoProcesses, kExactTwoProcesses,
                                       false, kMinGpusPerNode, kNoNodeLimit))
            return false;
        int ndev = 0; AssertInitAndGetDevices(&ndev);
        if (SyncSkip(!GdrSupported())) { GTEST_SKIP() << "GDR (NCCL_PTR_CUDA) not supported"; return false; }
        MultiSegmentVmmBuffer* buf = AllocSym(nSeg);
        if (SyncSkip(buf == nullptr)) { GTEST_SKIP() << "multi-segment VMM allocation unavailable"; return false; }
        lastBuf_ = buf;
        SetupConnectionWithGuard(0, pair, guard);
        const int rank = MPIEnvironment::world_rank;
        *comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        ncclResult_t r = RegisterMultiSegmentMr(*comm, *buf, mh);
        if (SyncSkip(r == ncclInvalidUsage && *mh == nullptr)) {
            GTEST_SKIP() << "dma-buf multi-segment registration unavailable on this build/host";
            return false;
        }
        EXPECT_EQ(r, ncclSuccess) << "multi-segment registration failed (the AIRUNTIME-2351 bug)";
        EXPECT_NE(*mh, nullptr);
        return (r == ncclSuccess && *mh != nullptr);
    }

    MultiSegmentVmmBuffer* lastBuf_ = nullptr;
    void* sendMh_ = nullptr;
    void* recvMh_ = nullptr;
};

// POSITIVE: register a 4-segment window and move each segment end to end.
TEST_F(NetIbMultiSegmentMPITest, PerSegmentRegistrationAndTransfer) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    if (!SetupRegistered(kNumSegments, pair, guard, &mh, &comm)) return;
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    sendMh_ = recvMh_ = mh;
    for (int s = 0; s < kNumSegments; s++)
        SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, (size_t)s * lastBuf_->segSize, lastBuf_->segSize,
                      /*tag=*/100 + s, /*seed=*/static_cast<uint8_t>(0xA0 + s));
}

// SELECTION: transfers anchored at different offsets inside each segment.
TEST_F(NetIbMultiSegmentMPITest, IntraSegmentOffsetSelection) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    if (!SetupRegistered(kNumSegments, pair, guard, &mh, &comm)) return;
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

// OPTION B FLAGSHIP: a single transfer that straddles a segment boundary now
// SUCCEEDS -- the sender splits the RDMA write at the receiver's boundary using
// the per-segment rkeys published in the CTS FIFO. Data must arrive intact.
TEST_F(NetIbMultiSegmentMPITest, CrossBoundaryTransferSucceeds) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    if (!SetupRegistered(kNumSegments, pair, guard, &mh, &comm)) return;
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    sendMh_ = recvMh_ = mh;
    // 64 KiB before + 64 KiB after the seg0/seg1 boundary.
    const size_t off  = lastBuf_->segSize - 65536;
    const size_t size = 131072;
    SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, off, size, /*tag=*/400, /*seed=*/0xC3);
}

// OPTION B: a single transfer spanning the ENTIRE multi-segment buffer (crosses
// every boundary) completes and verifies -- exercises the full splitting builder.
TEST_F(NetIbMultiSegmentMPITest, WholeBufferSingleTransfer) {
    ConnectionPair pair; NetConnectionGuard guard(net_); void* mh = nullptr; void* comm = nullptr;
    if (!SetupRegistered(kNumSegments, pair, guard, &mh, &comm)) return;
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
    if (SyncSkip(!GdrSupported())) GTEST_SKIP() << "GDR not supported";

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
    if (!SetupRegistered(1, pair, guard, &mh, &comm)) return;
    NetMHandleGuard mhGuard(mh, NetMHandleDeleter(net_, comm));
    sendMh_ = recvMh_ = mh;
    SendRecvChunk(pair, lastBuf_->ptr, lastBuf_->ptr, 0, 65536, /*tag=*/300, /*seed=*/0x77);
}

#endif // MPI_TESTS_ENABLED
