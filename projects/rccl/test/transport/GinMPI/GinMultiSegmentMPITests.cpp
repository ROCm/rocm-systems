/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Multi-segment DMA-BUF registration tests for ncclGinIbProxy (AIRUNTIME-2351).
// Run with NCCL_NET=IB NCCL_CUMEM_ENABLE=1; see docs/dev/gin-multi-segment-dmabuf.md.

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_GIN_IB_PROXY

#include "GinMPITestBase.hpp"
#include "GinMultiSegmentHelpers.hpp"
#include "MPIHelpers.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace RCCLGinTests
{

namespace
{

// Mirrors NCCL_GIN_MAX_SEGMENTS in src/transport/net_ib/gin.cc (not exposed to
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
class GinMultiSegmentMPITest : public GinMPITestBase
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

        GinMPITestBase::SetUp(); // resolves gin_ (or GTEST_SKIPs when NCCL_NET has no GIN backend)

        logCtx_ = std::make_unique<MPIHelpers::TestLogAssertionContext>(
            MPIHelpers::makeCombinedAssertionLogOptions(getTestMpiRank()));
    }

    void TearDown() override
    {
        GinMPITestBase::TearDown();
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

    // Allocate an identical N-segment VMM window on this rank (symmetric across ranks).
    bool AllocSym(int nSegments, size_t segBytes, MultiSegmentVmmBuffer* out)
    {
        return AllocMultiSegmentVmm(defaultDevice_, nSegments, segBytes, out);
    }
};

// Reproducer (AIRUNTIME-2351): a multi-segment window must register per-segment
// and move data correctly end to end. Flagship positive case.
TEST_F(GinMultiSegmentMPITest, Reproducer_MultiSegmentRegistrationAndTransfer)
{
    if (!SetUpFixture(/*minProcs=*/2, /*maxProcs=*/2)) return;

    MultiSegmentVmmBuffer sendBuf, recvBuf;
    bool allocOk = AllocSym(kNumSegments, kSegRequestBytes, &sendBuf);
    allocOk     &= AllocSym(kNumSegments, kSegRequestBytes, &recvBuf);
    MultiSegmentVmmGuard sg(std::move(sendBuf));
    MultiSegmentVmmGuard rg(std::move(recvBuf));

    if (SyncSkip(!allocOk))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t kSize = sg.get().totalSize;

    if (worldRank_ == 0)
        FillBuf(sg.get().ptr, kSize, /*seed=*/0xA0);

    void *sendMh = nullptr, *sendGh = nullptr, *recvMh = nullptr, *recvGh = nullptr;
    EXPECT_EQ(ncclSuccess, RegMr(sg.get().ptr, kSize, &sendMh, &sendGh))
        << "multi-segment send buffer registration failed (the AIRUNTIME-2351 bug)";
    EXPECT_EQ(ncclSuccess, RegMr(rg.get().ptr, kSize, &recvMh, &recvGh))
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
                  gin_->iput(ginCtx_, /*context=*/0,
                             /*srcOff=*/0, sendMh, kSize,
                             /*dstOff=*/0, recvMh, /*peerRank=*/1, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
        EXPECT_TRUE(VerifyBuf(rg.get().ptr, kSize, /*seed=*/0xA0))
            << "data corrupted across segment boundaries";
}

// IPut starting/ending mid-segment so the WR builder splits on a non-zero
// per-segment offset on both sides (most prone to addr/lkey/rkey errors).
TEST_F(GinMultiSegmentMPITest, IPutCrossSegmentBoundaryAtOffset)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer s, r;
    bool allocOk = AllocSym(kNumSegments, kSegRequestBytes, &s);
    allocOk     &= AllocSym(kNumSegments, kSegRequestBytes, &r);
    MultiSegmentVmmGuard sg(std::move(s));
    MultiSegmentVmmGuard rg(std::move(r));

    if (SyncSkip(!allocOk))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t      segSize   = sg.get().segSize;
    const size_t      off       = segSize / 2;                 // start mid-first-segment
    const size_t      kSize     = sg.get().totalSize - segSize; // end mid-last-segment
    constexpr uint8_t kSentinel = 0xCC;

    if (worldRank_ == 0)
        FillBuf(static_cast<uint8_t*>(sg.get().ptr) + off, kSize, /*seed=*/0x5A);
    if (worldRank_ == 1)
        FillSentinel(rg.get().ptr, rg.get().totalSize, kSentinel);

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sg.get().ptr, sg.get().totalSize, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rg.get().ptr, rg.get().totalSize, &recvMh, &recvGh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  gin_->iput(ginCtx_, 0, /*srcOff=*/off, sendMh, kSize,
                             /*dstOff=*/off, recvMh, 1, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
    {
        EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(rg.get().ptr) + off, kSize, /*seed=*/0x5A))
            << "payload wrong after multi-segment split at offset " << off;
        EXPECT_TRUE(AllSentinel(rg.get().ptr, off, kSentinel))
            << "bytes before dstOff were overwritten";
        EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(rg.get().ptr) + off + kSize,
                                rg.get().totalSize - (off + kSize), kSentinel))
            << "bytes after the transfer were overwritten";
    }
}

// IGet of a whole multi-segment remote buffer — read opcode through the split.
TEST_F(GinMultiSegmentMPITest, IGetMultiSegment)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer b;
    bool allocOk = AllocSym(kNumSegments, kSegRequestBytes, &b);
    MultiSegmentVmmGuard bg(std::move(b));

    if (SyncSkip(!allocOk))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t kSize = bg.get().totalSize;
    if (worldRank_ == 1)
        FillBuf(bg.get().ptr, kSize, /*seed=*/0xC3);

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(bg.get().ptr, kSize, &mh, &gh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  gin_->iget(ginCtx_, 0, /*remoteOff=*/0, mh, kSize,
                             /*localOff=*/0, mh, /*peerRank=*/1, &req));
        ASSERT_TRUE(PollUntilDone(req));
        EXPECT_TRUE(VerifyBuf(bg.get().ptr, kSize, /*seed=*/0xC3))
            << "iget data corrupted across segment boundaries";
    }
    Barrier();
}

// IPutSignal over a multi-segment payload: data WRs split per segment, then a
// chained signal WR. Verifies both payload and the atomic.
TEST_F(GinMultiSegmentMPITest, IPutSignalMultiSegment)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer s, r;
    bool allocOk = AllocSym(kNumSegments, kSegRequestBytes, &s);
    allocOk     &= AllocSym(kNumSegments, kSegRequestBytes, &r);
    MultiSegmentVmmGuard sg(std::move(s));
    MultiSegmentVmmGuard rg(std::move(r));

    if (SyncSkip(!allocOk))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t kSize = sg.get().totalSize;

    void* sigBuf = AllocBuf(kSignalSize);
    ASSERT_NE(sigBuf, nullptr);

    if (worldRank_ == 0)
        FillBuf(sg.get().ptr, kSize, /*seed=*/0x55);

    void *sendMh, *sendGh, *recvMh, *recvGh, *sigMh, *sigGh;
    ASSERT_EQ(ncclSuccess, RegMr(sg.get().ptr, kSize,       &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rg.get().ptr, kSize,       &recvMh, &recvGh));
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf,       kSignalSize, &sigMh,  &sigGh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  gin_->iputSignal(ginCtx_, 0,
                                   /*srcOff=*/0, sendMh, kSize,
                                   /*dstOff=*/0, recvMh, /*peerRank=*/1,
                                   /*signalOff=*/0, sigMh, /*signalValue=*/0,
                                   NCCL_NET_SIGNAL_OP_INC, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
    {
        EXPECT_TRUE(VerifyBuf(rg.get().ptr, kSize, /*seed=*/0x55))
            << "multi-segment iputSignal payload mismatch";
        EXPECT_EQ(ReadSignal(sigBuf), 1u)
            << "signal not delivered after multi-segment payload";
    }
}

// Sweep IPut payloads from 0 to the full window across edge sizes (byte, word,
// page, 64K, and segment boundaries). Verifies the payload landed and that no
// byte past `size` was touched (catches over-write / boundary-split errors).
TEST_F(GinMultiSegmentMPITest, IPutSizeSweepFromZero)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer s, r;
    bool allocOk = AllocSym(kNumSegments, kSegRequestBytes, &s);
    allocOk     &= AllocSym(kNumSegments, kSegRequestBytes, &r);
    MultiSegmentVmmGuard sg(std::move(s));
    MultiSegmentVmmGuard rg(std::move(r));

    if (SyncSkip(!allocOk))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t      total     = sg.get().totalSize;
    const size_t      seg       = sg.get().segSize;
    constexpr uint8_t kSentinel = 0xBD;

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sg.get().ptr, total, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rg.get().ptr, total, &recvMh, &recvGh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    const std::vector<size_t> sizes = EdgeCaseSizes(seg, total);
    for (size_t idx = 0; idx < sizes.size(); ++idx)
    {
        const size_t  sz   = sizes[idx];
        const uint8_t seed = static_cast<uint8_t>(0x40 + (idx & 0x3F));

        if (worldRank_ == 0 && sz > 0)
            FillBuf(sg.get().ptr, sz, seed);
        if (worldRank_ == 1)
            FillSentinel(rg.get().ptr, total, kSentinel); // reset every iteration

        Barrier();
        if (worldRank_ == 0)
        {
            void* req = nullptr;
            ASSERT_EQ(ncclSuccess,
                      gin_->iput(ginCtx_, 0, 0, sendMh, sz, 0, recvMh, 1, &req))
                << "iput post failed at size=" << sz;
            ASSERT_TRUE(PollUntilDone(req)) << "iput did not complete at size=" << sz;
        }
        Barrier();

        if (worldRank_ == 1)
        {
            EXPECT_TRUE(VerifyBuf(rg.get().ptr, sz, seed))
                << "payload mismatch at size=" << sz << " (seg=" << seg << ")";
            EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(rg.get().ptr) + sz,
                                    total - sz, kSentinel))
                << "bytes past size=" << sz << " were overwritten";
        }
        Barrier();
    }
}

// Same sweep but starting at an offset that sits just inside the first segment,
// so every transfer begins mid-segment and most cross >=1 boundary. Stresses
// the non-zero per-segment offset arithmetic on both local and remote sides.
TEST_F(GinMultiSegmentMPITest, IPutSizeSweepAtBoundaryOffset)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer s, r;
    bool allocOk = AllocSym(kNumSegments, kSegRequestBytes, &s);
    allocOk     &= AllocSym(kNumSegments, kSegRequestBytes, &r);
    MultiSegmentVmmGuard sg(std::move(s));
    MultiSegmentVmmGuard rg(std::move(r));

    if (SyncSkip(!allocOk))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t      total     = sg.get().totalSize;
    const size_t      seg       = sg.get().segSize;
    const size_t      off       = (seg >= 64) ? seg - 64 : 0; // straddle first boundary
    constexpr uint8_t kSentinel = 0x9C;

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sg.get().ptr, total, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rg.get().ptr, total, &recvMh, &recvGh));

    if (SyncSkip(!AllTookMultiSegPath()))
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    const std::vector<size_t> sizes = EdgeCaseSizes(seg, total - off);
    for (size_t idx = 0; idx < sizes.size(); ++idx)
    {
        const size_t  sz   = sizes[idx];
        const uint8_t seed = static_cast<uint8_t>(0x80 + (idx & 0x3F));

        if (worldRank_ == 0 && sz > 0)
            FillBuf(static_cast<uint8_t*>(sg.get().ptr) + off, sz, seed);
        if (worldRank_ == 1)
            FillSentinel(rg.get().ptr, total, kSentinel);

        Barrier();
        if (worldRank_ == 0)
        {
            void* req = nullptr;
            ASSERT_EQ(ncclSuccess,
                      gin_->iput(ginCtx_, 0, off, sendMh, sz, off, recvMh, 1, &req))
                << "iput post failed at size=" << sz << " off=" << off;
            ASSERT_TRUE(PollUntilDone(req))
                << "iput did not complete at size=" << sz << " off=" << off;
        }
        Barrier();

        if (worldRank_ == 1)
        {
            EXPECT_TRUE(VerifyBuf(static_cast<uint8_t*>(rg.get().ptr) + off, sz, seed))
                << "payload mismatch at size=" << sz << " off=" << off;
            EXPECT_TRUE(AllSentinel(rg.get().ptr, off, kSentinel))
                << "bytes before off=" << off << " were overwritten (size=" << sz << ")";
            EXPECT_TRUE(AllSentinel(static_cast<uint8_t*>(rg.get().ptr) + off + sz,
                                    total - off - sz, kSentinel))
                << "bytes past off+size were overwritten (size=" << sz << ")";
        }
        Barrier();
    }
}

// NEGATIVE: a buffer spanning more than NCCL_GIN_MAX_SEGMENTS must be rejected
// with ncclInvalidUsage (no truncation, no crash). Gated on the path being live.
TEST_F(GinMultiSegmentMPITest, RegisterExceedsMaxSegmentsRejected)
{
    if (!SetUpFixture(2, 2)) return;

    // Confirm the per-segment path is live first, else the assertion is moot.
    {
        MultiSegmentVmmBuffer probe;
        bool probeOk = AllocSym(2, kSegRequestBytes, &probe);
        MultiSegmentVmmGuard pg(std::move(probe));
        if (SyncSkip(!probeOk))
            GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

        void *pmh = nullptr, *pgh = nullptr;
        EXPECT_EQ(ncclSuccess, RegMr(pg.get().ptr, pg.get().totalSize, &pmh, &pgh));
        if (SyncSkip(!AllTookMultiSegPath()))
            GTEST_SKIP() << "multi-segment path not exercised on this host";
    }

    const int kOverCap = kGinMaxSegments + 1;
    MultiSegmentVmmBuffer big;
    bool allocOk = AllocSym(kOverCap, kSegRequestBytes, &big);
    MultiSegmentVmmGuard bg(std::move(big));
    if (SyncSkip(!allocOk))
        GTEST_SKIP() << "Could not allocate " << kOverCap << " VMM segments";

    void *mh = nullptr, *gh = nullptr;
    ncclResult_t r = RegMr(bg.get().ptr, bg.get().totalSize, &mh, &gh);
    EXPECT_EQ(r, ncclInvalidUsage)
        << "registration of a " << kOverCap << "-segment buffer should be rejected "
        << "with ncclInvalidUsage (cap=" << kGinMaxSegments << "), got " << r;
    EXPECT_EQ(mh, nullptr) << "no MR handle should be produced on rejection";
}

// REGRESSION: an ordinary single-allocation buffer must still register and
// transfer via the nSeg==1 fast path.
TEST_F(GinMultiSegmentMPITest, SingleSegmentRegression)
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
                  gin_->iput(ginCtx_, 0, 0, sendMh, kSize, 0, recvMh, 1, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
        EXPECT_TRUE(VerifyBuf(recvBuf, kSize, /*seed=*/0x77));
}

} // namespace RCCLGinTests

#else // !RCCL_HAS_GIN_IB_PROXY

#include <gtest/gtest.h>

TEST(GinMultiSegmentMPITest, BuildSkipped)
{
    GTEST_SKIP() << "IB Proxy GIN backend not built into this binary. Skipping multi-segment GIN tests...";
}

#endif // RCCL_HAS_GIN_IB_PROXY

#endif // MPI_TESTS_ENABLED
