/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-segment DMA-BUF registration tests for ncclRmaIbProxy (AIRUNTIME-2351).
// Run with NCCL_NET=IB NCCL_CUMEM_ENABLE=1; see docs/dev/gin-multi-segment-dmabuf.md.

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_RMA_IB_PROXY

#include "RmaMPITestBase.hpp"
#include "GinMultiSegmentHelpers.hpp"
#include "MPIHelpers.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace RCCLRmaTests
{

namespace
{

// Mirrors NCCL_RMA_MAX_SEGMENTS in src/transport/net_ib/gin.cc (not exposed to
// the test target); update if the backend cap changes.
constexpr int    kGinMaxSegments = 16;

constexpr size_t kSegRequestBytes = 2u * 1024 * 1024;
constexpr int    kNumSegments     = 4;
constexpr size_t kSignalSize      = 64;

// INFO marker emitted by the backend when the per-segment path fires.
constexpr const char* kMultiSegMarker = "multi-segment buffer";

// Edge-case payload sizes from 0 up to `maxBytes`, anchored around byte/word,
// page (4K), 64K, and the per-segment boundary `seg`. Deduplicated + sorted.
inline std::vector<size_t> EdgeCaseSizes(size_t seg, size_t maxBytes)
{
    std::vector<size_t> v;
    auto add = [&](size_t s) { if (s <= maxBytes) v.push_back(s); };
    for (size_t s : {size_t{0}, size_t{1}, size_t{2}, size_t{3}, size_t{7},
                     size_t{63}, size_t{64}, size_t{65}, size_t{255}, size_t{256},
                     size_t{4095}, size_t{4096}, size_t{4097},
                     size_t{65535}, size_t{65536}, size_t{65537}})
        add(s);
    // Per-segment boundary neighbourhood (the split points under test).
    if (seg >= 1)     { add(seg - 1); add(seg); add(seg + 1); add(seg + 4096); }
    if (2 * seg >= 1) { add(2 * seg - 1); add(2 * seg); add(2 * seg + 1); }
    add(3 * seg);
    add(maxBytes ? maxBytes - 1 : 0);
    add(maxBytes);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

} // namespace

// GIN proxy fixture + NCCL INFO log capture to confirm the per-segment path
// fired (vs single-MR fallback when cuMem enumeration is unavailable).
class RmaMultiSegmentMPITest : public RmaMPITestBase
{
protected:
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             cuMemGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugGuard_;
    std::unique_ptr<MPIHelpers::MpiEnvGuard>             debugSubsysGuard_;
    std::unique_ptr<MPIHelpers::TestLogAssertionContext> logCtx_;

    int GetNumContexts() const override { return 1; }

    void SetUp() override
    {
        // Per-segment enumeration needs the cuMem path; the marker gate below
        // covers cases where the param was already cached process-wide.
        cuMemGuard_       = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_CUMEM_ENABLE",  "1");
        debugGuard_       = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG",         "INFO");
        debugSubsysGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_DEBUG_SUBSYS",  "ALL");

        RmaMPITestBase::SetUp();

        logCtx_ = std::make_unique<MPIHelpers::TestLogAssertionContext>(
            MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));
    }

    void TearDown() override
    {
        // Deregister IB MRs (base TearDown) BEFORE releasing their backing VMM;
        // freeing VMM under a live DMA-BUF MR aborts/stalls cleanup (AIRUNTIME-2351).
        RmaMPITestBase::TearDown();
        for (auto& b : vmmBuffers_)
            FreeMultiSegmentVmm(*b);
        vmmBuffers_.clear();
        logCtx_.reset();
        debugSubsysGuard_.reset();
        debugGuard_.reset();
        cuMemGuard_.reset();
    }

    std::string readAllLogs() const
    {
        if (!logCtx_) return {};
        return logCtx_->readNcclDebugLog() + logCtx_->readPerRankStderrLog();
    }

    // Collective skip: if ANY rank wants to skip, all ranks return true so they
    // GTEST_SKIP together (a unilateral skip would hang peers).
    bool SyncSkip(bool wantSkip)
    {
        int flag = wantSkip ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        return flag != 0;
    }

    // True only if EVERY rank observed the per-segment registration marker.
    bool AllTookMultiSegPath()
    {
        int local = (readAllLogs().find(kMultiSegMarker) != std::string::npos) ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &local, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return local != 0;
    }

    // Allocate a fixture-owned N-segment VMM window (freed in TearDown after MR
    // dereg). Returns nullptr on failure so the caller can SyncSkip. Uses the
    // rank's CURRENT GPU (round-robin assigned by the harness), not the IB-device
    // index defaultDevice_, or rank>0 would fault touching dev-0 memory.
    MultiSegmentVmmBuffer* AllocSym(int nSegments, size_t segBytes)
    {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess)
            return nullptr;
        auto buf = std::make_unique<MultiSegmentVmmBuffer>();
        if (!AllocMultiSegmentVmm(dev, nSegments, segBytes, buf.get()))
            return nullptr;
        vmmBuffers_.push_back(std::move(buf));
        return vmmBuffers_.back().get();
    }

    MultiSegmentVmmBuffer* AllocDeepEpElastic(size_t gpuBytes, size_t cpuBytes)
    {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess)
            return nullptr;
        auto buf = std::make_unique<MultiSegmentVmmBuffer>();
        if (!AllocDeepEpElasticVmm(dev, gpuBytes, cpuBytes, buf.get()))
            return nullptr;
        vmmBuffers_.push_back(std::move(buf));
        return vmmBuffers_.back().get();
    }

    std::vector<std::unique_ptr<MultiSegmentVmmBuffer>> vmmBuffers_;
};

// Reproducer (AIRUNTIME-2351): a multi-segment window must register per-segment
// and move data correctly end to end. Flagship positive case.
TEST_F(RmaMultiSegmentMPITest, Reproducer_MultiSegmentRegistrationAndTransfer)
{
    if (!SetUpFixture(/*minProcs=*/2, /*maxProcs=*/2)) return;

    MultiSegmentVmmBuffer* sb = AllocSym(kNumSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* rb = AllocSym(kNumSegments, kSegRequestBytes);

    if (SyncSkip(sb == nullptr || rb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t kSize = sb->totalSize;

    if (worldRank_ == 0)
        FillBuf(sb->ptr, kSize, /*seed=*/0xA0);

    void *sendMh = nullptr, *sendGh = nullptr, *recvMh = nullptr, *recvGh = nullptr;
    EXPECT_EQ(ncclSuccess, RegMr(sb->ptr, kSize, &sendMh, &sendGh))
        << "multi-segment send buffer registration failed (the AIRUNTIME-2351 bug)";
    EXPECT_EQ(ncclSuccess, RegMr(rb->ptr, kSize, &recvMh, &recvGh))
        << "multi-segment recv buffer registration failed (the AIRUNTIME-2351 bug)";

    // Confirm the per-segment path fired; otherwise the feature isn't exercised.
    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "Buffer registered as a single MR (cuMem disabled or "
                        "range not segmented) - multi-segment path not exercised";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, /*context=*/0,
                             /*srcOff=*/0, sendMh, kSize,
                             /*dstOff=*/0, recvMh, /*peerRank=*/1, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
        EXPECT_TRUE(VerifyBuf(rb->ptr, kSize, /*seed=*/0xA0))
            << "data corrupted across segment boundaries";
}

// IPut starting/ending mid-segment so the WR builder splits on a non-zero
// per-segment offset on both sides (most prone to addr/lkey/rkey errors).
TEST_F(RmaMultiSegmentMPITest, IPutCrossSegmentBoundaryAtOffset)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* sb = AllocSym(kNumSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* rb = AllocSym(kNumSegments, kSegRequestBytes);

    if (SyncSkip(sb == nullptr || rb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t      segSize   = sb->segSize;
    const size_t      off       = segSize / 2;              // start mid-first-segment
    const size_t      kSize     = sb->totalSize - segSize;  // end mid-last-segment
    constexpr uint8_t kSentinel = 0xCC;

    if (worldRank_ == 0)
        FillBuf(static_cast<uint8_t*>(sb->ptr) + off, kSize, /*seed=*/0x5A);
    if (worldRank_ == 1)
        FillSentinel(rb->ptr, rb->totalSize, kSentinel);

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, sb->totalSize, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, rb->totalSize, &recvMh, &recvGh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, 0, /*srcOff=*/off, sendMh, kSize,
                             /*dstOff=*/off, recvMh, 1, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
    {
        EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(rb->ptr) + off, kSize, /*seed=*/0x5A))
            << "payload wrong after multi-segment split at offset " << off;
        EXPECT_TRUE(AllSentinel(rb->ptr, off, kSentinel))
            << "bytes before dstOff were overwritten";
        EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(rb->ptr) + off + kSize,
                                rb->totalSize - (off + kSize), kSentinel))
            << "bytes after the transfer were overwritten";
    }
}

// IGet of a whole multi-segment remote buffer — read opcode through the split.
TEST_F(RmaMultiSegmentMPITest, IGetMultiSegment)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* bb = AllocSym(kNumSegments, kSegRequestBytes);

    if (SyncSkip(bb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t kSize = bb->totalSize;
    if (worldRank_ == 1)
        FillBuf(bb->ptr, kSize, /*seed=*/0xC3);

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(bb->ptr, kSize, &mh, &gh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iget(rmaCtx_, 0, /*remoteOff=*/0, mh, kSize,
                             /*localOff=*/0, mh, /*peerRank=*/1, &req));
        ASSERT_TRUE(PollUntilDone(req));
        EXPECT_TRUE(VerifyBuf(bb->ptr, kSize, /*seed=*/0xC3))
            << "iget data corrupted across segment boundaries";
    }
    Barrier();
}

// IGet starting/ending mid-segment so the read WR builder splits on a non-zero
// per-segment offset on both the remote (source) and local (dest) sides. Mirrors
// IPutCrossSegmentBoundaryAtOffset for the read opcode (only whole-buffer IGet
// was covered before).
TEST_F(RmaMultiSegmentMPITest, IGetCrossSegmentBoundaryAtOffset)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* sb = AllocSym(kNumSegments, kSegRequestBytes); // remote source (rank 1)
    MultiSegmentVmmBuffer* rb = AllocSym(kNumSegments, kSegRequestBytes); // local dest   (rank 0)

    if (SyncSkip(sb == nullptr || rb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t      segSize   = sb->segSize;
    const size_t      off       = segSize / 2;              // start mid-first-segment
    const size_t      kSize     = sb->totalSize - segSize;  // end mid-last-segment
    constexpr uint8_t kSentinel = 0xD4;

    if (worldRank_ == 1)
        FillBuf(static_cast<uint8_t*>(sb->ptr) + off, kSize, /*seed=*/0x6E);
    if (worldRank_ == 0)
        FillSentinel(rb->ptr, rb->totalSize, kSentinel);

    void *srcMh, *srcGh, *dstMh, *dstGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, sb->totalSize, &srcMh, &srcGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, rb->totalSize, &dstMh, &dstGh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iget(rmaCtx_, 0, /*remoteOff=*/off, srcMh, kSize,
                             /*localOff=*/off, dstMh, /*peerRank=*/1, &req));
        ASSERT_TRUE(PollUntilDone(req));

        EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(rb->ptr) + off, kSize, /*seed=*/0x6E))
            << "iget payload wrong after multi-segment split at offset " << off;
        EXPECT_TRUE(AllSentinel(rb->ptr, off, kSentinel))
            << "bytes before localOff were overwritten by iget";
        EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(rb->ptr) + off + kSize,
                                rb->totalSize - (off + kSize), kSentinel))
            << "bytes after the iget were overwritten";
    }
    Barrier();
}

// DeepEP Engram pattern (DeepEP/csrc/kernels/backend/symmetric.hpp and
// DeepEP/csrc/kernels/elastic/engram.hpp): one symmetric VMM window contains a
// large GPU receive segment followed by an independently-sized CPU storage
// segment. An IGet reads from a non-zero offset in the remote CPU segment into a
// different non-zero offset in the local GPU segment. This specifically guards
// independent local and remote registration-relative offset tracking.
TEST_F(RmaMultiSegmentMPITest, DeepEP_EngramMixedWindowIGet)
{
    if (!SetUpFixture(2, 2)) return;

    constexpr size_t kMiB        = 1024 * 1024;
    constexpr size_t kGpuBytes   = 4 * kMiB;
    constexpr size_t kCpuBytes   = 2 * kMiB;
    constexpr size_t kRemoteOff  = kGpuBytes + 4096;
    constexpr size_t kLocalOff   = 64 * 1024;
    constexpr size_t kPayload    = 128 * 1024;
    constexpr uint8_t kSentinel  = 0xD7;

    MultiSegmentVmmBuffer* window = AllocDeepEpElastic(kGpuBytes, kCpuBytes);
    if (SyncSkip(window == nullptr))
        GTEST_SKIP() << "DeepEP-style GPU+CPU VMM allocation unavailable on this runtime";

    if (worldRank_ == 1)
        FillBuf(static_cast<uint8_t*>(window->ptr) + kRemoteOff, kPayload, /*seed=*/0x4D);
    if (worldRank_ == 0)
        FillSentinel(window->ptr, kGpuBytes, kSentinel);

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(window->ptr, window->totalSize, &mh, &gh));
    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "DeepEP window did not take the multi-segment GIN registration path";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iget(rmaCtx_, 0,
                             /*remoteOff=*/kRemoteOff, mh, kPayload,
                             /*localOff=*/kLocalOff, mh, /*peerRank=*/1, &req));
        ASSERT_TRUE(PollUntilDone(req));
        EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(window->ptr) + kLocalOff,
                              kPayload, /*seed=*/0x4D))
            << "DeepEP-style CPU-to-GPU IGet corrupted data";
        EXPECT_TRUE(AllSentinel(window->ptr, kLocalOff, kSentinel));
        EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(window->ptr) + kLocalOff + kPayload,
                                kGpuBytes - kLocalOff - kPayload, kSentinel));
    }
    Barrier();
}

// Multi-node stress form of the DeepEP Engram fetch pattern. The registered
// window is [GPU receive area][CPU Engram storage] at DeepEP's 2 MiB alignment.
// Each IGet reads a changing non-zero remote CPU offset into an unrelated local
// GPU offset, while sentinels ensure the GPU receive area is not over-written.
TEST_F(RmaMultiSegmentMPITest, DeepEP_MultiNodeEngramMixedWindowIGetStress)
{
    if (!SetUpFixture(2, 2)) return;
    if (MPIEnvironment::cached_multi_node_result != 1)
        GTEST_SKIP() << "requires exactly one rank on each of two nodes";

    constexpr size_t kMiB       = 1024 * 1024;
    constexpr size_t kGpuBytes  = 8 * kMiB;
    constexpr size_t kCpuBytes  = 4 * kMiB;
    constexpr int    kIterations = 32;
    constexpr uint8_t kSentinel = 0xD9;
    const std::vector<size_t> payloadSizes = {
        size_t{1}, size_t{63}, size_t{4095}, size_t{4096},
        size_t{65535}, size_t{65536}, size_t{131072}, size_t{262144}
    };

    MultiSegmentVmmBuffer* window = AllocDeepEpElastic(kGpuBytes, kCpuBytes);
    if (SyncSkip(window == nullptr))
        GTEST_SKIP() << "DeepEP-style GPU+CPU VMM allocation unavailable on this runtime";

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(window->ptr, window->totalSize, &mh, &gh));
    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "DeepEP window did not take the multi-segment GIN registration path";

    for (int i = 0; i < kIterations; ++i)
    {
        const size_t len = payloadSizes[static_cast<size_t>(i) % payloadSizes.size()];
        const size_t remoteSpan = kCpuBytes - len - 4096;
        const size_t localSpan = kGpuBytes - len - 65536;
        const size_t remoteOff = kGpuBytes + 4096 +
                                 (static_cast<size_t>(i) * 131071) % remoteSpan;
        const size_t localOff = 65536 +
                                (static_cast<size_t>(i) * 65537) % localSpan;
        const uint8_t seed = static_cast<uint8_t>(0x40 + i);

        if (worldRank_ == 1)
            FillBuf(static_cast<uint8_t*>(window->ptr) + remoteOff, len, seed);
        if (worldRank_ == 0)
            FillSentinel(window->ptr, kGpuBytes, kSentinel);

        Barrier();
        if (worldRank_ == 0)
        {
            void* req = nullptr;
            ASSERT_EQ(ncclSuccess,
                      rma_->iget(rmaCtx_, 0, remoteOff, mh, len,
                                 localOff, mh, /*peerRank=*/1, &req))
                << "DeepEP Engram IGet post failed at iteration " << i;
            ASSERT_TRUE(PollUntilDone(req))
                << "DeepEP Engram IGet stalled at iteration " << i;
            EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(window->ptr) + localOff,
                                  len, seed))
                << "DeepEP Engram payload mismatch at iteration " << i;
            EXPECT_TRUE(AllSentinel(window->ptr, localOff, kSentinel))
                << "bytes before DeepEP fetch destination were overwritten";
            EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(window->ptr) + localOff + len,
                                    kGpuBytes - localOff - len, kSentinel))
                << "bytes after DeepEP fetch destination were overwritten";
        }
        Barrier();
    }
}

// Receiver-side flush over a multi-segment buffer: after a plain iput, rank 1
// must fence EVERY physical segment via iflush (one loopback read per segment)
// before reading. Exercises the multi-segment flush path; a segment-0-only
// flush faults here on a multi-segment handle.
TEST_F(RmaMultiSegmentMPITest, IFlushMultiSegment)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* sb = AllocSym(kNumSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* rb = AllocSym(kNumSegments, kSegRequestBytes);

    if (SyncSkip(sb == nullptr || rb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t kSize = sb->totalSize;
    if (worldRank_ == 0)
        FillBuf(sb->ptr, kSize, /*seed=*/0x3C);

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, kSize, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, kSize, &recvMh, &recvGh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, 0, 0, sendMh, kSize, 0, recvMh, /*peerRank=*/1, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
    {
        void* freq = nullptr;
        EXPECT_EQ(ncclSuccess, rma_->iflush(rmaCtx_, 0, recvMh, /*peerRank=*/0, &freq))
            << "multi-segment iflush post failed";
        EXPECT_TRUE(PollUntilDone(freq)) << "multi-segment flush did not complete";
        EXPECT_TRUE(VerifyBuf(rb->ptr, kSize, /*seed=*/0x3C))
            << "data corrupted across segment boundaries after flush";
    }
    Barrier();
}

// Flush after a PARTIAL, offset multi-segment iput: only a sub-range straddling
// an interior boundary is written, then rank 1 flushes the whole handle. Since
// iflush fences EVERY physical segment (no offset/size args), it must fence the
// touched segments and leave the untouched sentinel bytes intact. Complements
// IFlushMultiSegment (which flushes after a full-window iput).
TEST_F(RmaMultiSegmentMPITest, IFlushAfterPartialMultiSegmentPut)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* sb = AllocSym(kNumSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* rb = AllocSym(kNumSegments, kSegRequestBytes);

    if (SyncSkip(sb == nullptr || rb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t      segSize   = sb->segSize;
    const size_t      off       = segSize / 2;              // start mid-first-segment
    const size_t      kSize     = sb->totalSize - segSize;  // end mid-last-segment
    constexpr uint8_t kSentinel = 0x71;

    if (worldRank_ == 0)
        FillBuf(static_cast<uint8_t*>(sb->ptr) + off, kSize, /*seed=*/0x4F);
    if (worldRank_ == 1)
        FillSentinel(rb->ptr, rb->totalSize, kSentinel);

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, sb->totalSize, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, rb->totalSize, &recvMh, &recvGh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, 0, /*srcOff=*/off, sendMh, kSize,
                             /*dstOff=*/off, recvMh, /*peerRank=*/1, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
    {
        void* freq = nullptr;
        EXPECT_EQ(ncclSuccess, rma_->iflush(rmaCtx_, 0, recvMh, /*peerRank=*/0, &freq))
            << "multi-segment iflush post failed after partial put";
        EXPECT_TRUE(PollUntilDone(freq)) << "multi-segment flush did not complete";

        EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(rb->ptr) + off, kSize, /*seed=*/0x4F))
            << "partial payload wrong after flush across segment boundaries";
        EXPECT_TRUE(AllSentinel(rb->ptr, off, kSentinel))
            << "bytes before dstOff were overwritten";
        EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(rb->ptr) + off + kSize,
                                rb->totalSize - (off + kSize), kSentinel))
            << "bytes after the transfer were overwritten";
    }
    Barrier();
}

// REGRESSION (flush fast path): iflush on an ordinary single-allocation buffer
// must still fence and verify. IFlushMultiSegment only covers the multi-segment
// handle and SingleSegmentRegression never flushes, so the nSeg==1 flush path --
// which the multi-segment change must not regress -- is otherwise untested. No
// AllTookMultiSegPath gate: a single-segment buffer intentionally does NOT take
// the per-segment path.
TEST_F(RmaMultiSegmentMPITest, IFlushSingleSegmentRegression)
{
    if (!SetUpFixture(2, 2)) return;

    const size_t kSize = 1u << 20; // 1 MiB, plain hipMalloc => single segment
    void* sendBuf = AllocBuf(kSize);
    void* recvBuf = AllocBuf(kSize);
    ASSERT_NE(sendBuf, nullptr);
    ASSERT_NE(recvBuf, nullptr);

    if (worldRank_ == 0)
        FillBuf(sendBuf, kSize, /*seed=*/0x2D);

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sendBuf, kSize, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(recvBuf, kSize, &recvMh, &recvGh));

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, 0, 0, sendMh, kSize, 0, recvMh, /*peerRank=*/1, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
    {
        void* freq = nullptr;
        EXPECT_EQ(ncclSuccess, rma_->iflush(rmaCtx_, 0, recvMh, /*peerRank=*/0, &freq))
            << "single-segment iflush post failed";
        EXPECT_TRUE(PollUntilDone(freq)) << "single-segment flush did not complete";
        EXPECT_TRUE(VerifyBuf(recvBuf, kSize, /*seed=*/0x2D))
            << "data corrupted after single-segment flush";
    }
    Barrier();
}

// IPutSignal over a multi-segment payload: data WRs split per segment, then a
// chained signal WR. Verifies both payload and the atomic.
TEST_F(RmaMultiSegmentMPITest, IPutSignalMultiSegment)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* sb = AllocSym(kNumSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* rb = AllocSym(kNumSegments, kSegRequestBytes);

    if (SyncSkip(sb == nullptr || rb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t kSize = sb->totalSize;

    void* sigBuf = AllocBuf(kSignalSize);
    ASSERT_NE(sigBuf, nullptr);

    if (worldRank_ == 0)
        FillBuf(sb->ptr, kSize, /*seed=*/0x55);

    void *sendMh, *sendGh, *recvMh, *recvGh, *sigMh, *sigGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, kSize,       &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, kSize,       &recvMh, &recvGh));
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf,  kSignalSize, &sigMh,  &sigGh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iputSignal(rmaCtx_, 0,
                                   /*srcOff=*/0, sendMh, kSize,
                                   /*dstOff=*/0, recvMh, /*peerRank=*/1,
                                   /*signalOff=*/0, sigMh, /*signalValue=*/0,
                                   NCCL_NET_SIGNAL_OP_INC, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
    {
        EXPECT_TRUE(VerifyBuf(rb->ptr, kSize, /*seed=*/0x55))
            << "multi-segment iputSignal payload mismatch";
        EXPECT_EQ(ReadSignal(sigBuf), 1u)
            << "signal not delivered after multi-segment payload";
    }
}

// Sweep IPut payloads from 0 to the full window across edge sizes (byte, word,
// page, 64K, and segment boundaries). Verifies the payload landed and that no
// byte past `size` was touched (catches over-write / boundary-split errors).
TEST_F(RmaMultiSegmentMPITest, IPutSizeSweepFromZero)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* sb = AllocSym(kNumSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* rb = AllocSym(kNumSegments, kSegRequestBytes);

    if (SyncSkip(sb == nullptr || rb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t      total     = sb->totalSize;
    const size_t      seg       = sb->segSize;
    constexpr uint8_t kSentinel = 0xBD;

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, total, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, total, &recvMh, &recvGh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    const std::vector<size_t> sizes = EdgeCaseSizes(seg, total);
    for (size_t idx = 0; idx < sizes.size(); ++idx)
    {
        const size_t  sz   = sizes[idx];
        const uint8_t seed = static_cast<uint8_t>(0x40 + (idx & 0x3F));

        if (worldRank_ == 0 && sz > 0)
            FillBuf(sb->ptr, sz, seed);
        if (worldRank_ == 1)
            FillSentinel(rb->ptr, total, kSentinel); // reset every iteration

        Barrier();
        if (worldRank_ == 0)
        {
            void* req = nullptr;
            ASSERT_EQ(ncclSuccess,
                      rma_->iput(rmaCtx_, 0, 0, sendMh, sz, 0, recvMh, 1, &req))
                << "iput post failed at size=" << sz;
            ASSERT_TRUE(PollUntilDone(req)) << "iput did not complete at size=" << sz;
        }
        Barrier();

        if (worldRank_ == 1)
        {
            EXPECT_TRUE(VerifyBuf(rb->ptr, sz, seed))
                << "payload mismatch at size=" << sz << " (seg=" << seg << ")";
            EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(rb->ptr) + sz,
                                    total - sz, kSentinel))
                << "bytes past size=" << sz << " were overwritten";
        }
        Barrier();
    }
}

// Same sweep but starting at an offset that sits just inside the first segment,
// so every transfer begins mid-segment and most cross >=1 boundary. Stresses
// the non-zero per-segment offset arithmetic on both local and remote sides.
TEST_F(RmaMultiSegmentMPITest, IPutSizeSweepAtBoundaryOffset)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* sb = AllocSym(kNumSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* rb = AllocSym(kNumSegments, kSegRequestBytes);

    if (SyncSkip(sb == nullptr || rb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t      total     = sb->totalSize;
    const size_t      seg       = sb->segSize;
    const size_t      off       = (seg >= 64) ? seg - 64 : 0; // straddle first boundary
    constexpr uint8_t kSentinel = 0x9C;

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, total, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, total, &recvMh, &recvGh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    const std::vector<size_t> sizes = EdgeCaseSizes(seg, total - off);
    for (size_t idx = 0; idx < sizes.size(); ++idx)
    {
        const size_t  sz   = sizes[idx];
        const uint8_t seed = static_cast<uint8_t>(0x80 + (idx & 0x3F));

        if (worldRank_ == 0 && sz > 0)
            FillBuf(static_cast<uint8_t*>(sb->ptr) + off, sz, seed);
        if (worldRank_ == 1)
            FillSentinel(rb->ptr, total, kSentinel);

        Barrier();
        if (worldRank_ == 0)
        {
            void* req = nullptr;
            ASSERT_EQ(ncclSuccess,
                      rma_->iput(rmaCtx_, 0, off, sendMh, sz, off, recvMh, 1, &req))
                << "iput post failed at size=" << sz << " off=" << off;
            ASSERT_TRUE(PollUntilDone(req))
                << "iput did not complete at size=" << sz << " off=" << off;
        }
        Barrier();

        if (worldRank_ == 1)
        {
            EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(rb->ptr) + off, sz, seed))
                << "payload mismatch at size=" << sz << " off=" << off;
            EXPECT_TRUE(AllSentinel(rb->ptr, off, kSentinel))
                << "bytes before off=" << off << " were overwritten (size=" << sz << ")";
            EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(rb->ptr) + off + sz,
                                    total - off - sz, kSentinel))
                << "bytes past off+size were overwritten (size=" << sz << ")";
        }
        Barrier();
    }
}

// NEGATIVE: a buffer spanning more than NCCL_RMA_MAX_SEGMENTS must be rejected
// with ncclInvalidUsage (no truncation, no crash). Gated on the path being live.
TEST_F(RmaMultiSegmentMPITest, RegisterExceedsMaxSegmentsRejected)
{
    if (!SetUpFixture(2, 2)) return;

    // Confirm the per-segment path is live first, else the assertion is moot.
    {
        MultiSegmentVmmBuffer* probe = AllocSym(2, kSegRequestBytes);
        if (SyncSkip(probe == nullptr))
            GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

        void *pmh = nullptr, *pgh = nullptr;
        EXPECT_EQ(ncclSuccess, RegMr(probe->ptr, probe->totalSize, &pmh, &pgh));
        if (SyncSkip(!AllTookMultiSegPath()))
            GTEST_SKIP() << "multi-segment path not exercised on this host";
    }

    const int kOverCap = kGinMaxSegments + 1;
    MultiSegmentVmmBuffer* big = AllocSym(kOverCap, kSegRequestBytes);
    if (SyncSkip(big == nullptr))
        GTEST_SKIP() << "Could not allocate " << kOverCap << " VMM segments";

    void *mh = nullptr, *gh = nullptr;
    ncclResult_t r = RegMr(big->ptr, big->totalSize, &mh, &gh);
    EXPECT_EQ(r, ncclInvalidUsage)
        << "registration of a " << kOverCap << "-segment buffer should be rejected "
        << "with ncclInvalidUsage (cap=" << kGinMaxSegments << "), got " << r;
    EXPECT_EQ(mh, nullptr) << "no MR handle should be produced on rejection";
}

// REGRESSION: an ordinary single-allocation buffer must still register and
// transfer via the nSeg==1 fast path.
TEST_F(RmaMultiSegmentMPITest, SingleSegmentRegression)
{
    if (!SetUpFixture(2, 2)) return;

    const size_t kSize = 1u << 20; // 1 MiB, plain hipMalloc => single segment

    void* sendBuf = AllocBuf(kSize);
    void* recvBuf = AllocBuf(kSize);
    ASSERT_NE(sendBuf, nullptr);
    ASSERT_NE(recvBuf, nullptr);

    if (worldRank_ == 0)
        FillBuf(sendBuf, kSize, /*seed=*/0x77);

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sendBuf, kSize, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(recvBuf, kSize, &recvMh, &recvGh));

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, 0, 0, sendMh, kSize, 0, recvMh, 1, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
        EXPECT_TRUE(VerifyBuf(recvBuf, kSize, /*seed=*/0x77));
}

// NEGATIVE (cross-rank symmetry guard): ranks register windows with different
// physical segment counts; the backend must reject collectively with
// ncclInternalError instead of running the mismatched-stride all-gather.
TEST_F(RmaMultiSegmentMPITest, RegisterAsymmetricSegmentCountRejected)
{
    if (!SetUpFixture(2, 2)) return;

    // Distinct counts per rank (2 vs 8) so enumeration is very unlikely to agree.
    const int myNSeg = (worldRank_ == 0) ? 2 : 8;
    MultiSegmentVmmBuffer* bb = AllocSym(myNSeg, kSegRequestBytes);
    if (SyncSkip(bb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    void *mh = nullptr, *gh = nullptr;
    ncclResult_t r = RegMr(bb->ptr, bb->totalSize, &mh, &gh);

    // Both ranks must have taken the per-segment path, else there is nothing to test.
    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    // If enumeration happened to return identical counts there is no asymmetry
    // to reject (registration succeeds on both ranks); skip rather than misfire.
    if (SyncSkip(r == ncclSuccess))
        GTEST_SKIP() << "ranks enumerated identical segment counts; no asymmetry";

    EXPECT_EQ(r, ncclInternalError)
        << "asymmetric per-rank segment count must be rejected (rank " << worldRank_
        << " requested " << myNSeg << " segments)";
    EXPECT_EQ(mh, nullptr) << "no MR handle should be produced on rejection";
}

// NEGATIVE (range guard): out-of-range IPut offsets/sizes must be rejected with
// ncclInvalidArgument and must NOT post anything (recv stays untouched).
TEST_F(RmaMultiSegmentMPITest, IPutOutOfRangeRejectedNoCorruption)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* sb = AllocSym(kNumSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* rb = AllocSym(kNumSegments, kSegRequestBytes);
    if (SyncSkip(sb == nullptr || rb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t      total     = sb->totalSize;
    constexpr uint8_t kSentinel = 0xE7;

    if (worldRank_ == 1)
        FillSentinel(rb->ptr, total, kSentinel);

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, total, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, total, &recvMh, &recvGh));

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        // size exactly one byte past the window.
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iput(rmaCtx_, 0, 0, sendMh, total + 1, 0, recvMh, 1, &req))
            << "oversized transfer must be rejected";
        // valid size but srcOff pushes the source past the end.
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iput(rmaCtx_, 0, /*srcOff=*/1, sendMh, total, 0, recvMh, 1, &req))
            << "src offset overrun must be rejected";
        // valid size but dstOff pushes the destination past the end.
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iput(rmaCtx_, 0, 0, sendMh, total, /*dstOff=*/1, recvMh, 1, &req))
            << "dst offset overrun must be rejected";
        EXPECT_EQ(req, nullptr) << "rejected iput must not produce a request";
    }
    Barrier();

    // Nothing was posted, so the receiver's window must be byte-for-byte intact.
    if (worldRank_ == 1)
        EXPECT_TRUE(AllSentinel(rb->ptr, total, kSentinel))
            << "rejected iput corrupted the destination window";
}

// NEGATIVE (range guard): out-of-range IGet offsets/sizes must be rejected.
TEST_F(RmaMultiSegmentMPITest, IGetOutOfRangeRejected)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* bb = AllocSym(kNumSegments, kSegRequestBytes);
    if (SyncSkip(bb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t total = bb->totalSize;

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(bb->ptr, total, &mh, &gh));

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iget(rmaCtx_, 0, /*remoteOff=*/0, mh, total + 1, 0, mh, 1, &req))
            << "oversized iget must be rejected";
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iget(rmaCtx_, 0, /*remoteOff=*/1, mh, total, 0, mh, 1, &req))
            << "remote offset overrun must be rejected";
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iget(rmaCtx_, 0, 0, mh, total, /*localOff=*/1, mh, 1, &req))
            << "local offset overrun must be rejected";
        EXPECT_EQ(req, nullptr) << "rejected iget must not produce a request";
    }
    Barrier();
}

// NEGATIVE (range guard): IPutSignal must reject both an out-of-range payload
// and an out-of-range signal offset (the 8-byte atomic) without posting.
TEST_F(RmaMultiSegmentMPITest, IPutSignalOutOfRangeRejectedNoCorruption)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* sb = AllocSym(kNumSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* rb = AllocSym(kNumSegments, kSegRequestBytes);
    if (SyncSkip(sb == nullptr || rb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t      total     = sb->totalSize;
    constexpr uint8_t kSentinel = 0x3B;

    void* sigBuf = AllocBuf(kSignalSize);
    ASSERT_NE(sigBuf, nullptr);
    if (worldRank_ == 1)
        FillSentinel(rb->ptr, total, kSentinel);

    void *sendMh, *sendGh, *recvMh, *recvGh, *sigMh, *sigGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, total,        &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, total,        &recvMh, &recvGh));
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf,  kSignalSize,  &sigMh,  &sigGh));

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        // Payload past the window (valid signal offset).
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iputSignal(rmaCtx_, 0, 0, sendMh, total + 1, 0, recvMh, 1,
                                   /*signalOff=*/0, sigMh, 0, NCCL_NET_SIGNAL_OP_INC, &req))
            << "oversized iputSignal payload must be rejected";
        // Signal atomic straddles the end of the signal window (signalOff+8 > size).
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iputSignal(rmaCtx_, 0, 0, sendMh, total, 0, recvMh, 1,
                                   /*signalOff=*/kSignalSize - 4, sigMh, 0,
                                   NCCL_NET_SIGNAL_OP_INC, &req))
            << "out-of-range signal offset must be rejected";
        EXPECT_EQ(req, nullptr) << "rejected iputSignal must not produce a request";
    }
    Barrier();

    if (worldRank_ == 1)
    {
        EXPECT_TRUE(AllSentinel(rb->ptr, total, kSentinel))
            << "rejected iputSignal corrupted the destination window";
        EXPECT_EQ(ReadSignal(sigBuf), 0u)
            << "rejected iputSignal must not deliver the signal";
    }
}

// SCALE/CORRUPTION variation: exercise EVERY internal segment boundary of a
// wide (many-segment) window. For each boundary, transfer a chunk that straddles
// it and verify the payload landed exactly and no neighbouring byte was touched.
TEST_F(RmaMultiSegmentMPITest, BoundaryStressNoCorruption)
{
    if (!SetUpFixture(2, 2)) return;

    constexpr int     kWideSegments = 8; // more boundaries == closer to "at scale"
    MultiSegmentVmmBuffer* sb = AllocSym(kWideSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* rb = AllocSym(kWideSegments, kSegRequestBytes);
    if (SyncSkip(sb == nullptr || rb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t      total     = sb->totalSize;
    const size_t      seg       = sb->segSize;
    const int         nSeg      = sb->nSegments;
    constexpr uint8_t kSentinel = 0x6F;

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, total, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, total, &recvMh, &recvGh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    // Straddle each interior boundary k*seg with a chunk that lands in both the
    // preceding and following segment, so a WR split happens at every boundary.
    const size_t kHalf = (seg >= 256) ? 128 : seg / 2;
    for (int k = 1; k < nSeg; ++k)
    {
        const size_t off  = k * seg - kHalf;
        const size_t len  = 2 * kHalf;
        const uint8_t seed = static_cast<uint8_t>(0x10 + k);

        if (worldRank_ == 0)
            FillBuf(static_cast<uint8_t*>(sb->ptr) + off, len, seed);
        if (worldRank_ == 1)
            FillSentinel(rb->ptr, total, kSentinel);

        Barrier();
        if (worldRank_ == 0)
        {
            void* req = nullptr;
            ASSERT_EQ(ncclSuccess,
                      rma_->iput(rmaCtx_, 0, off, sendMh, len, off, recvMh, 1, &req))
                << "iput failed straddling boundary " << k;
            ASSERT_TRUE(PollUntilDone(req)) << "iput stalled at boundary " << k;
        }
        Barrier();

        if (worldRank_ == 1)
        {
            EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(rb->ptr) + off, len, seed))
                << "payload wrong across boundary " << k;
            EXPECT_TRUE(AllSentinel(rb->ptr, off, kSentinel))
                << "bytes before boundary " << k << " transfer were overwritten";
            EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(rb->ptr) + off + len,
                                    total - off - len, kSentinel))
                << "bytes after boundary " << k << " transfer were overwritten";
        }
        Barrier();
    }

    // Full-window transfer across all boundaries at once.
    if (worldRank_ == 0)
        FillBuf(sb->ptr, total, /*seed=*/0xAB);
    if (worldRank_ == 1)
        FillSentinel(rb->ptr, total, kSentinel);

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, 0, 0, sendMh, total, 0, recvMh, 1, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
        EXPECT_TRUE(VerifyBuf(rb->ptr, total, /*seed=*/0xAB))
            << "full multi-segment window transfer corrupted data";
}

// MULTI-NODE IGET STRESS: DeepEP-style operations use independent remote and
// local offsets. Repeatedly cross a different physical boundary on each side,
// exercising remote rkey and local lkey selection while sentinels detect writes
// outside the requested destination range.
TEST_F(RmaMultiSegmentMPITest, MultiNodeAsymmetricIGetBoundaryStress)
{
    if (!SetUpFixture(2, 2)) return;
    if (MPIEnvironment::cached_multi_node_result != 1)
        GTEST_SKIP() << "requires exactly one rank on each of two nodes";

    constexpr int kWideSegments = 8;
    constexpr int kIterations   = 32;
    MultiSegmentVmmBuffer* sb = AllocSym(kWideSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* rb = AllocSym(kWideSegments, kSegRequestBytes);
    if (SyncSkip(sb == nullptr || rb == nullptr))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t total = sb->totalSize;
    const size_t seg = sb->segSize;
    constexpr uint8_t kSentinel = 0xA7;
    const std::vector<size_t> edgeWidths = {
        size_t{1}, size_t{63}, size_t{4095}, size_t{65535}, size_t{131071}
    };

    void *srcMh, *srcGh, *dstMh, *dstGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, total, &srcMh, &srcGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, total, &dstMh, &dstGh));
    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    for (int i = 0; i < kIterations; ++i)
    {
        const int remoteBoundary = 1 + (i % (kWideSegments - 1));
        const int localBoundary = 1 + ((i * 3 + 2) % (kWideSegments - 1));
        const size_t left = edgeWidths[static_cast<size_t>(i) % edgeWidths.size()];
        const size_t right = edgeWidths[static_cast<size_t>(i + 3) % edgeWidths.size()];
        const size_t remoteOff = static_cast<size_t>(remoteBoundary) * seg - left;
        const size_t localOff = static_cast<size_t>(localBoundary) * seg - left;
        const size_t len = left + right;
        const uint8_t seed = static_cast<uint8_t>(0x30 + i);

        if (worldRank_ == 1)
            FillBuf(static_cast<uint8_t*>(sb->ptr) + remoteOff, len, seed);
        if (worldRank_ == 0)
            FillSentinel(rb->ptr, total, kSentinel);

        Barrier();
        if (worldRank_ == 0)
        {
            void* req = nullptr;
            ASSERT_EQ(ncclSuccess,
                      rma_->iget(rmaCtx_, 0, remoteOff, srcMh, len,
                                 localOff, dstMh, /*peerRank=*/1, &req))
                << "iget post failed at iteration " << i;
            ASSERT_TRUE(PollUntilDone(req)) << "iget stalled at iteration " << i;
            EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(rb->ptr) + localOff, len, seed))
                << "payload mismatch at iteration " << i;
            EXPECT_TRUE(AllSentinel(rb->ptr, localOff, kSentinel))
                << "bytes before local destination were overwritten at iteration " << i;
            EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(rb->ptr) + localOff + len,
                                    total - localOff - len, kSentinel))
                << "bytes after local destination were overwritten at iteration " << i;
        }
        Barrier();
    }
}

} // namespace RCCLRmaTests

#else // !RCCL_HAS_RMA_IB_PROXY

#include <gtest/gtest.h>

TEST(RmaMultiSegmentMPITest, BuildSkipped)
{
    GTEST_SKIP() << "IB Proxy GIN backend not built into this binary. Skipping multi-segment GIN tests...";
}

#endif // RCCL_HAS_RMA_IB_PROXY

#endif // MPI_TESTS_ENABLED
