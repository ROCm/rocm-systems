/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
// Multi-segment DMA-BUF registration tests for the IB-MSEG RMA backend, IbMsegRmaIbProxy (AIRUNTIME-2351).
// Run with NCCL_NET=IB-MSEG or NCCL_MULTI_SEGMENT_REGISTER=1, and NCCL_CUMEM_ENABLE=1.

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_RMA_IB_PROXY

#include "RmaMultiSegmentFixture.hpp"

#include <numeric>
#include <random>

namespace RCCLRmaTests
{

// Reproducer (AIRUNTIME-2351): a multi-segment window must register per-segment
// and move data correctly end to end. Flagship positive case.
TEST_F(RmaMultiSegmentMPITest, Reproducer_MultiSegmentRegistrationAndTransfer)
{
    if (!SetUpFixture(/*minProcs=*/2, /*maxProcs=*/2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    void *sendMh = nullptr, *recvMh = nullptr;
    // A registration failure here is the AIRUNTIME-2351 bug.
    RMA_MSEG_PREPARE_OR_SKIP(PreparePair(&sb, &rb, &sendMh, &recvMh));

    Rank0Round(false, sb, sendMh, 0, sb->totalSize, rb, recvMh, 0, /*seed=*/0xA0, /*sentinel=*/0x00,
               "data corrupted across segment boundaries");
}

// Two mappings plus half a third: the final MR is clipped, and a put ending at the last registered byte works.
TEST_F(RmaMultiSegmentMPITest, PartialFinalSegmentRegistrationAndTransfer)
{
    if (!SetUpFixture(/*minProcs=*/2, /*maxProcs=*/2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb, /*nSegments=*/3))
    {
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";
    }

    const size_t registeredBytes = 2 * sb->segSize + sb->segSize / 2;
    const size_t transferOffset  = 2 * sb->segSize - 4096;
    const size_t transferBytes   = registeredBytes - transferOffset;

    void *sendMh = nullptr, *recvMh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, registeredBytes, &sendMh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, registeredBytes, &recvMh));

    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Rank0Round(false, sb, sendMh, transferOffset, transferBytes, rb, recvMh, transferOffset, /*seed=*/0x71,
               /*sentinel=*/0xB7, "partial final physical segment");
}


// Sub-page start and end: each DMA-BUF export must cover every page its MR spans, or ibv_reg_dmabuf_mr fails.
TEST_F(RmaMultiSegmentMPITest, SubPageBoundsRegistrationAndTransfer)
{
    if (!SetUpFixture(/*minProcs=*/2, /*maxProcs=*/2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb, /*nSegments=*/3))
    {
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";
    }

    constexpr size_t kHeadSkip   = 512;
    constexpr uint8_t kSentinel  = 0xC3;
    const size_t registeredBytes = 2 * sb->segSize + sb->segSize / 2 + 100 - kHeadSkip;
    uint8_t* sendBase = static_cast<uint8_t*>(sb->ptr) + kHeadSkip;
    uint8_t* recvBase = static_cast<uint8_t*>(rb->ptr) + kHeadSkip;

    if (worldRank_ == 0)
        FillBuf(sendBase, registeredBytes, /*seed=*/0x5D);
    if (worldRank_ == 1)
        FillSentinel(rb->ptr, rb->totalSize, kSentinel);

    void *sendMh = nullptr, *recvMh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(sendBase, registeredBytes, &sendMh));
    ASSERT_EQ(ncclSuccess, RegMr(recvBase, registeredBytes, &recvMh));

    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    ASSERT_NO_FATAL_FAILURE(Rank0Put(0, sendMh, registeredBytes, 0, recvMh));

    if (worldRank_ == 1)
    {
        ExpectPayloadIsolated(rb->ptr, rb->totalSize, kHeadSkip,
                              registeredBytes, /*seed=*/0x5D, kSentinel,
                              "sub-page registration bounds");
    }
}

// IPut starting/ending mid-segment so the WR builder splits on a non-zero
// per-segment offset on both sides (most prone to addr/lkey/rkey errors).
TEST_F(RmaMultiSegmentMPITest, IPutCrossSegmentBoundaryAtOffset)
{
    if (!SetUpFixture(2, 2)) return;
    RunMidSegmentTransfer(MidSegmentOp::Put, /*seed=*/0x5A, /*sentinel=*/0xCC);
}

// IGet of a whole multi-segment remote buffer — read opcode through the split.
TEST_F(RmaMultiSegmentMPITest, IGetMultiSegment)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* bb = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(AllocWindow(kNumSegments, &bb, "Multi-segment VMM allocation unavailable on this host"));
    void* mh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(bb->ptr, bb->totalSize, &mh));

    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Rank0Round(true, bb, mh, 0, bb->totalSize, bb, mh, 0, /*seed=*/0xC3, /*sentinel=*/0x00,
               "iget data corrupted across segment boundaries");
}

// IGet from and to mid-segment offsets, so the read builder splits on non-zero offsets on both sides.
TEST_F(RmaMultiSegmentMPITest, IGetCrossSegmentBoundaryAtOffset)
{
    if (!SetUpFixture(2, 2)) return;
    RunMidSegmentTransfer(MidSegmentOp::Get, /*seed=*/0x6E, /*sentinel=*/0xD4);
}


// findRemotePeerForLocalRank returns a rank on another node with that local rank, or -1. MPI topology only.
TEST_F(RmaMultiSegmentMPITest, FindRemotePeerForLocalRankTopology)
{
    if (!SetUpFixture(2, 2)) return;

    // Two ranks share a node (local ranks 0 and 1, no remote match) or sit one per node (both local rank 0).
    const bool multiNode = MPIEnvironment::cached_multi_node_result == 1;
    // Both lookups are collective; every rank makes them before checking.
    const int peer = MPIHelpers::findRemotePeerForLocalRank(multiNode ? 0 : worldRank_);
    const int missing = MPIHelpers::findRemotePeerForLocalRank(worldSize_);
    EXPECT_EQ(missing, -1) << "no node has local rank " << worldSize_;
    EXPECT_EQ(peer, multiNode ? 1 - worldRank_ : -1) << (multiNode ? "two nodes" : "one node");
}

// After a plain iput, rank 1's iflush on a multi-segment window completes, and every segment's bytes are visible.
TEST_F(RmaMultiSegmentMPITest, IFlushMultiSegment)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    void *sendMh = nullptr, *recvMh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PreparePair(&sb, &rb, &sendMh, &recvMh));

    bool flushOff = false;
    ASSERT_NO_FATAL_FAILURE(Rank0Round(false, sb, sendMh, 0, sb->totalSize, rb, recvMh, 0, /*seed=*/0x3C,
                                       /*sentinel=*/0x00, "data corrupted across segment boundaries after flush",
                                       0, &flushOff));
    if (MPIHelpers::anyRankTrue(flushOff)) GTEST_SKIP() << "GDR flush is disabled; iflush built no flush WR";
}

// A partial put across an interior boundary, then a whole-handle iflush: written bytes land and the
// untouched sentinels stay intact.
TEST_F(RmaMultiSegmentMPITest, IFlushAfterPartialMultiSegmentPut)
{
    if (!SetUpFixture(2, 2)) return;
    RunMidSegmentTransfer(MidSegmentOp::PutThenFlush, /*seed=*/0x4F, /*sentinel=*/0x71);
}

// iflush on an ordinary single-allocation buffer still fences; no other test flushes an nSeg == 1 handle.
TEST_F(RmaMultiSegmentMPITest, IFlushSingleSegmentRegression)
{
    if (!SetUpFixture(2, 2)) return;

    void *recvBuf = nullptr, *recvMh = nullptr;
    ASSERT_NO_FATAL_FAILURE(PreparePlainPair(/*seed=*/0x2D, &recvBuf, &recvMh));

    const bool flushOff = Rank1Flush(recvMh);
    if (worldRank_ == 1)
    {
        EXPECT_TRUE(VerifyBuf(recvBuf, kMiB, /*seed=*/0x2D))
            << "data corrupted after single-segment flush";
    }
    Barrier();
    if (MPIHelpers::anyRankTrue(flushOff)) GTEST_SKIP() << "GDR flush is disabled; iflush built no flush WR";
}

// IPutSignal over a multi-segment payload: data WRs split per segment, then a
// chained signal WR. Verifies both payload and the atomic.
TEST_F(RmaMultiSegmentMPITest, IPutSignalMultiSegment)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    void *sendMh = nullptr, *recvMh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PreparePair(&sb, &rb, &sendMh, &recvMh));

    const size_t kSize = sb->totalSize;

    void* sigBuf = AllocBuf(kSignalSize);
    ASSERT_NE(sigBuf, nullptr);

    if (worldRank_ == 0)
        FillBuf(sb->ptr, kSize, /*seed=*/0x55);

    void* sigMh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf, kSignalSize, &sigMh));

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iputSignal(rmaCtx_, 0,
                                   /*srcOff=*/0, sendMh, kSize,
                                   /*dstOff=*/0, recvMh, /*peerRank=*/1,
                                   /*signalOff=*/0, sigMh, /*signalValue=*/0,
                                   NCCL_NET_SIGNAL_OP_INC, /*isStrongSignal=*/false, ncclRmaOptFlagsDefault, &req));
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


// Signal in a multi-segment window: straddling, misaligned or out-of-window atomics are rejected unposted,
// and atomics at either side of a boundary and inside segment 2 land only there.
TEST_F(RmaMultiSegmentMPITest, IPutSignalInLaterSegment)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr, *sw = nullptr;
    void *sendMh = nullptr, *recvMh = nullptr, *sigMh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PreparePair(&sb, &rb, &sendMh, &recvMh, kNumSegments, /*requirePath=*/false));
    RMA_MSEG_PREPARE_OR_SKIP(PrepareSignalWindow(3, &sw, &sigMh));
    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    const size_t      kSize     = sb->totalSize;
    const size_t      seg       = sw->segSize;
    const size_t      sigTotal  = sw->totalSize;
    constexpr uint8_t kSentinel = 0x6E;

    if (worldRank_ == 0)
        FillBuf(sb->ptr, kSize, /*seed=*/0x4C);
    if (worldRank_ == 1)
        FillSentinel(rb->ptr, kSize, kSentinel);

    Barrier();
    if (worldRank_ == 0)
    {
        const size_t badOffs[] = {seg - 4, seg + 4, sigTotal - 4, sigTotal};
        for (size_t off : badOffs)
        {
            void* req = nullptr;
            EXPECT_EQ(ncclInvalidArgument,
                      rma_->iputSignal(rmaCtx_, 0, 0, sendMh, kSize, 0, recvMh, 1,
                                       off, sigMh, 0, NCCL_NET_SIGNAL_OP_INC,
                                       /*isStrongSignal=*/false, ncclRmaOptFlagsDefault, &req))
                << "signal at offset " << off << " must be rejected";
            EXPECT_EQ(req, nullptr) << "rejected iputSignal at offset " << off << " posted a request";
        }
    }
    Barrier();
    if (worldRank_ == 1)
    {
        EXPECT_TRUE(AllSentinel(rb->ptr, kSize, kSentinel))
            << "rejected iputSignal wrote the payload";
        EXPECT_TRUE(AllSentinel(sw->ptr, sigTotal, 0))
            << "rejected iputSignal wrote the signal window";
    }
    Barrier();

    struct { size_t off; uint32_t op; uint64_t value; uint64_t expect; } cases[] = {
        {2 * seg + 64, NCCL_NET_SIGNAL_OP_INC, 0, 1},
        {seg - 8,      NCCL_NET_SIGNAL_OP_ADD, 5, 5},
        {seg,          NCCL_NET_SIGNAL_OP_ADD, 7, 7},
    };
    if (worldRank_ == 0)
    {
        for (const auto& c : cases)
        {
            void* req = nullptr;
            ASSERT_EQ(ncclSuccess,
                      rma_->iputSignal(rmaCtx_, 0, 0, sendMh, kSize, 0, recvMh, 1,
                                       c.off, sigMh, c.value, c.op,
                                       /*isStrongSignal=*/false, ncclRmaOptFlagsDefault, &req))
                << "signal at offset " << c.off;
            ASSERT_TRUE(PollUntilDone(req));
        }
    }
    Barrier();

    if (worldRank_ == 1)
    {
        EXPECT_TRUE(VerifyBuf(rb->ptr, kSize, /*seed=*/0x4C))
            << "iputSignal payload mismatch";
        std::vector<SignalWord> want;
        for (const auto& c : cases) want.push_back({c.off, c.expect});
        ExpectOnlySignals(sw, want);
    }
}

// IPut sizes from 0 to the full window at edge sizes; the payload lands and no byte past `size` changes.
TEST_F(RmaMultiSegmentMPITest, IPutSizeSweepFromZero)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    void *sendMh = nullptr, *recvMh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PreparePair(&sb, &rb, &sendMh, &recvMh));

    constexpr uint8_t kSentinel = 0xBD;

    RunIPutSizeSweep(sb, rb, sendMh, recvMh, /*offset=*/0,
                     /*seedBase=*/0x40, kSentinel);
}

// The same sweep from an offset just inside segment 0, so most transfers start mid-segment and cross one.
TEST_F(RmaMultiSegmentMPITest, IPutSizeSweepAtBoundaryOffset)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    void *sendMh = nullptr, *recvMh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PreparePair(&sb, &rb, &sendMh, &recvMh));

    const size_t off = (sb->segSize >= 64) ? sb->segSize - 64 : 0;
    constexpr uint8_t kSentinel = 0x9C;

    RunIPutSizeSweep(sb, rb, sendMh, recvMh, off,
                     /*seedBase=*/0x80, kSentinel);
}

// NEGATIVE: a buffer spanning more than the fixed per-registration cap (RCCL_IB_REG_SEG_DEFAULT) must be
// rejected with ncclInvalidUsage (no truncation, no crash). Gated on the path being live.
TEST_F(RmaMultiSegmentMPITest, RegisterExceedsMaxSegmentsRejected)
{
    if (!SetUpFixture(2, 2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(ProbeMultiSegmentPath());

    const int kOverCap = RCCL_IB_REG_SEG_DEFAULT + 1;
    MultiSegmentVmmBuffer* big = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(
        AllocWindow(kOverCap, &big, "Could not allocate " + std::to_string(kOverCap) + " VMM segments"));

    void* mh = nullptr;
    ncclResult_t r = RegMr(big->ptr, big->totalSize, &mh);
    EXPECT_EQ(r, ncclInvalidUsage)
        << "registration of a " << kOverCap << "-segment buffer should be rejected "
        << "with ncclInvalidUsage (cap=" << RCCL_IB_REG_SEG_DEFAULT << "), got " << r;
    EXPECT_EQ(mh, nullptr) << "no MR handle should be produced on rejection";
}

// NCCL_MULTI_SEGMENT_REGISTER=0 keeps a multi-segment window on the single-MR path, whether or not that MR registers.
TEST_F(RmaMultiSegmentMPITest, MultiSegmentRegisterOff_KeepsSingleRegistration)
{
    if (!SetUpFixture(2, 2)) return;
    if (MPIHelpers::anyRankTrue(rcclRmaMultiSegEnabled()))
        GTEST_SKIP() << "Requires NCCL_MULTI_SEGMENT_REGISTER=0";

    MultiSegmentVmmBuffer* buf = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(AllocWindow(kNumSegments, &buf, "Multi-segment VMM allocation not supported"));

    void* mh = nullptr;
    const ncclResult_t r = RegMr(buf->ptr, buf->totalSize, &mh);
    if (r == ncclSuccess)
        EXPECT_NE(mh, nullptr) << "a successful single registration must return an MR handle";
    else
        EXPECT_EQ(mh, nullptr) << "a failed single registration must not return an MR handle, got " << r;
    const bool local = rcclRmaIbProxyMrSegmentCount(mh) > 1;
    EXPECT_FALSE(MPIHelpers::anyRankTrue(local))
        << "NCCL_MULTI_SEGMENT_REGISTER=0 still took the per-segment RMA path";
}

// REGRESSION: an ordinary single-allocation buffer must still register and
// transfer via the nSeg==1 fast path.
TEST_F(RmaMultiSegmentMPITest, SingleSegmentRegression)
{
    if (!SetUpFixture(2, 2)) return;

    void *recvBuf = nullptr, *recvMh = nullptr;
    ASSERT_NO_FATAL_FAILURE(PreparePlainPair(/*seed=*/0x77, &recvBuf, &recvMh));

    if (worldRank_ == 1)
        EXPECT_TRUE(VerifyBuf(recvBuf, kMiB, /*seed=*/0x77));
}

// Ranks with different segment counts are rejected collectively with ncclInvalidUsage before the gathers.
TEST_F(RmaMultiSegmentMPITest, RegisterAsymmetricSegmentCountRejected)
{
    if (!SetUpFixture(2, 2)) return;

    // Distinct counts per rank (2 vs 8) so enumeration is very unlikely to agree.
    const int myNSeg = (worldRank_ == 0) ? 2 : 8;
    MultiSegmentVmmBuffer* bb = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(AllocWindow(myNSeg, &bb, "Multi-segment VMM allocation unavailable on this host"));

    void* mh = nullptr;
    ncclResult_t r = RegMr(bb->ptr, bb->totalSize, &mh);

    // Only the per-segment count consensus rejects collectively, and a rejection leaves no handle to read.
    if (!MPIHelpers::allRanksTrue(r == ncclInvalidUsage))
    {
        if (!MultiSegmentPathAvailable())
            GTEST_SKIP() << "multi-segment path not exercised on this host";
        // Skip only if every rank succeeded (identical enumeration). any-rank skip hid unilateral success.
        if (MPIHelpers::allRanksTrue(r == ncclSuccess))
        {
            GTEST_SKIP() << "ranks enumerated identical segment counts; no asymmetry";
        }
    }

    EXPECT_EQ(r, ncclInvalidUsage)
        << "asymmetric per-rank segment count must be rejected (rank " << worldRank_
        << " requested " << myNSeg << " segments)";
    EXPECT_EQ(mh, nullptr) << "no MR handle should be produced on rejection";
}

// Equal segment counts with different per-rank boundaries register, and each side splits the put on its own.
TEST_F(RmaMultiSegmentMPITest, RegisterEqualCountDifferentBoundariesTransfer)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* bb = nullptr;
    void* mh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(worldRank_ == 0 ? PrepareElastic(4 * kMiB, 2 * kMiB, &bb, &mh)
                                             : PrepareElastic(2 * kMiB, 4 * kMiB, &bb, &mh));

    constexpr size_t kTotal = 6 * kMiB;
    Rank0Round(false, bb, mh, 0, kTotal, bb, mh, 0, /*seed=*/0x5A, /*sentinel=*/0xC3,
               "equal counts, different boundaries", /*dstChecked=*/kTotal);
}

// Only rank 0 exceeds the cap; both ranks must meet in the status collective and reject, or this hangs.
TEST_F(RmaMultiSegmentMPITest, RankLocalRegistrationFailureRejectedCollectively)
{
    if (!SetUpFixture(2, 2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(ProbeMultiSegmentPath());

    const int myNSeg =
        worldRank_ == 0 ? RCCL_IB_REG_SEG_DEFAULT + 1 : 2;
    MultiSegmentVmmBuffer* bb = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(AllocWindow(myNSeg, &bb, "asymmetric failure layout allocation unavailable"));

    void* mh = nullptr;
    const ncclResult_t r = RegMr(bb->ptr, bb->totalSize, &mh);
    EXPECT_EQ(r, ncclInvalidUsage);
    EXPECT_EQ(mh, nullptr);
}


// The per-comm registration records are reused: after a rejection, shrinking and growing registrations
// each see only their own layout.
TEST_F(RmaMultiSegmentMPITest, RegistrationAfterCollectiveRejectionTransfers)
{
    if (!SetUpFixture(2, 2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(ProbeMultiSegmentPath());

    const int badNSeg = worldRank_ == 0 ? RCCL_IB_REG_SEG_DEFAULT + 1 : 2;
    MultiSegmentVmmBuffer* bad = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(AllocWindow(badNSeg, &bad, "asymmetric failure layout allocation unavailable"));
    void* badMh = nullptr;
    ASSERT_EQ(ncclInvalidUsage, RegMr(bad->ptr, bad->totalSize, &badMh));
    ASSERT_EQ(badMh, nullptr);

    constexpr uint8_t kSentinel = 0xA5;
    const int nSegs[] = {4, 2, 3};
    for (int i = 0; i < (int)(sizeof(nSegs) / sizeof(nSegs[0])); i++)
    {
        SCOPED_TRACE(::testing::Message() << "round " << i << " nSeg " << nSegs[i]);
        MultiSegmentVmmBuffer* bb = pool_.AllocUniform(nSegs[i], kSegRequestBytes);
        ASSERT_FALSE(MPIHelpers::anyRankTrue(bb == nullptr)) << "symmetric allocation failed after rejection";

        void* mh = nullptr;
        ASSERT_EQ(ncclSuccess, RegMr(bb->ptr, bb->totalSize, &mh));
        ASSERT_NE(mh, nullptr);

        ASSERT_NO_FATAL_FAILURE(Rank0Round(false, bb, mh, 0, bb->totalSize, bb, mh, 0, (uint8_t)(0x30 + i),
                                           kSentinel, "put after rejection"));
    }
}

// NEGATIVE (range guard): out-of-range IPut offsets/sizes must be rejected with
// ncclInvalidArgument and must NOT post anything (recv stays untouched).
TEST_F(RmaMultiSegmentMPITest, IPutOutOfRangeRejectedNoCorruption)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    void *sendMh = nullptr, *recvMh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PreparePair(&sb, &rb, &sendMh, &recvMh, kNumSegments, /*requirePath=*/false));

    const size_t      total     = sb->totalSize;
    constexpr uint8_t kSentinel = 0xE7;

    if (worldRank_ == 1)
        FillSentinel(rb->ptr, total, kSentinel);
    ExpectOutOfRangeRejected(/*get=*/false, sendMh, recvMh, total);

    // Nothing was posted, so the receiver's window must be byte-for-byte intact.
    if (worldRank_ == 1)
    {
        EXPECT_TRUE(AllSentinel(rb->ptr, total, kSentinel))
            << "rejected iput corrupted the destination window";
    }
}

// NEGATIVE (range guard): out-of-range IGet offsets/sizes must be rejected.
TEST_F(RmaMultiSegmentMPITest, IGetOutOfRangeRejected)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* bb = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(AllocWindow(kNumSegments, &bb, "Multi-segment VMM allocation unavailable on this host"));
    void* mh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(bb->ptr, bb->totalSize, &mh));
    // The remote offset is the source side and the local offset the destination side.
    ExpectOutOfRangeRejected(/*get=*/true, mh, mh, bb->totalSize);
}

// NEGATIVE (range guard): IPutSignal must reject both an out-of-range payload
// and an out-of-range signal offset (the 8-byte atomic) without posting.
TEST_F(RmaMultiSegmentMPITest, IPutSignalOutOfRangeRejectedNoCorruption)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    void *sendMh = nullptr, *recvMh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PreparePair(&sb, &rb, &sendMh, &recvMh, kNumSegments, /*requirePath=*/false));

    const size_t      total     = sb->totalSize;
    constexpr uint8_t kSentinel = 0x3B;

    void* sigBuf = AllocBuf(kSignalSize);
    ASSERT_NE(sigBuf, nullptr);
    if (worldRank_ == 1)
        FillSentinel(rb->ptr, total, kSentinel);

    void* sigMh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf, kSignalSize, &sigMh));

    Barrier();
    if (worldRank_ == 0)
    {
        // A payload past the window, a signal atomic straddling the signal window's end, and an unaligned one.
        const struct { size_t size; uint64_t signalOff; const char* why; } cases[] = {
            {total + 1, 0, "oversized payload"},
            {total, kSignalSize - 4, "out-of-range signal offset"},
            {total, 4, "unaligned signal atomic"},
        };
        for (const auto& c : cases)
        {
            void* req = nullptr;
            EXPECT_EQ(ncclInvalidArgument,
                      rma_->iputSignal(rmaCtx_, 0, 0, sendMh, c.size, 0, recvMh, 1,
                                       c.signalOff, sigMh, 0, NCCL_NET_SIGNAL_OP_INC,
                                       /*isStrongSignal=*/false, ncclRmaOptFlagsDefault, &req))
                << "iputSignal " << c.why << " must be rejected";
            EXPECT_EQ(req, nullptr) << "rejected iputSignal (" << c.why << ") must not produce a request";
        }
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

// Straddle every internal boundary of a wide window; each chunk lands exactly and no neighbour changes.
TEST_F(RmaMultiSegmentMPITest, BoundaryStressNoCorruption)
{
    if (!SetUpFixture(2, 2)) return;

    constexpr int     kWideSegments = 8; // more boundaries == closer to "at scale"
    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    void *sendMh = nullptr, *recvMh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PreparePair(&sb, &rb, &sendMh, &recvMh, kWideSegments));

    const size_t      total     = sb->totalSize;
    const size_t      seg       = sb->segSize;
    const int         nSeg      = sb->nSegments;
    constexpr uint8_t kSentinel = 0x6F;

    // Straddle each interior boundary k*seg with a chunk that lands in both the
    // preceding and following segment, so a WR split happens at every boundary.
    const size_t kHalf = (seg >= 256) ? 128 : seg / 2;
    for (int k = 1; k < nSeg; ++k)
    {
        ASSERT_NO_FATAL_FAILURE(Rank0Round(false, sb, sendMh, k * seg - kHalf, 2 * kHalf, rb, recvMh,
                                           k * seg - kHalf, static_cast<uint8_t>(0x10 + k), kSentinel,
                                           "boundary " + std::to_string(k)));
    }

    // Full-window transfer across all boundaries at once.
    Rank0Round(false, sb, sendMh, 0, total, rb, recvMh, 0, /*seed=*/0xAB, kSentinel,
               "full multi-segment window transfer corrupted data");
}

// Multi-node IGet crossing a different boundary on each side, with sentinels around the destination.
TEST_F(RmaMultiSegmentMPITest, MultiNodeAsymmetricIGetBoundaryStress)
{
    if (!SetUpFixture(2, 2)) return;
    if (MPIEnvironment::cached_multi_node_result != 1)
        GTEST_SKIP() << "requires exactly one rank on each of two nodes";

    constexpr int kWideSegments = 8;
    constexpr int kIterations   = 32;
    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    void *srcMh = nullptr, *dstMh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PreparePair(&sb, &rb, &srcMh, &dstMh, kWideSegments));

    for (int i = 0; i < kIterations; ++i)
    {
        // The remote window is the stress source and the local one its destination.
        const auto c = BoundaryStress(i, kWideSegments, sb->segSize, /*dstMul=*/3, /*dstAdd=*/2, /*rightShift=*/3);
        ASSERT_NO_FATAL_FAILURE(Rank0Round(true, sb, srcMh, c.srcOff, c.size, rb, dstMh, c.dstOff,
                                           static_cast<uint8_t>(0x30 + i), /*sentinel=*/0xA7,
                                           "asymmetric IGet iteration " + std::to_string(i)));
    }
}

// Post-path fixture (WR batching, aggregation, contexts, wait order). Each rank puts to the next rank, so
// it runs on any rank and node count.
class RmaMultiSegmentPostMPITest : public RmaMultiSegmentMPITest
{
protected:
    enum class OpKind { Put, Get, PutSignal };
    enum class WaitOrder { InOrder, Reversed, Shuffled };

    // One segment-split transfer. For a put local is the source; for a get it
    // is the destination.
    struct Chain
    {
        uint64_t localOff  = 0;
        uint64_t remoteOff = 0;
        size_t   size      = 0;
    };

    struct Op
    {
        OpKind   kind      = OpKind::Put;
        void*    localMh   = nullptr;
        void*    remoteMh  = nullptr;
        Chain    chain;
        int      ctx       = 0;
        bool     aggregate = false;
        void*    sigMh     = nullptr;  // nullptr selects sig_
        uint64_t sigOff    = 0;
        uint64_t sigAdd    = 0;        // 0 selects NCCL_NET_SIGNAL_OP_INC
        int      peer      = -1;       // -1 selects SendPeer()
    };

    struct Region
    {
        uint64_t dstOff = 0;
        size_t   size   = 0;
        uint64_t srcOff = 0;
        int      seed   = -1;  // -1 selects Expect::seed
    };

    // Window bytes expected after a round: `regions` copied from a source filled
    // with `seed`, the sentinel everywhere else.
    struct Expect
    {
        void*               window = nullptr;
        size_t              total  = 0;
        std::vector<Region> regions;
        int                 seed   = 0;
    };

    static constexpr uint8_t kSentinel       = 0xD3;
    static constexpr int     kIbRequestSlots = RCCL_RMA_MAX_REQUESTS;
    // Send-queue depth of an RMA queue pair.
    static constexpr int     kSendQueueWrs   = RCCL_RMA_SEND_QP_WRS;

    int numContexts_ = 1;
    int GetNumContexts() const override { return numContexts_; }

    MultiSegmentVmmBuffer* src_   = nullptr;
    MultiSegmentVmmBuffer* dst_   = nullptr;
    void*                  sig_   = nullptr;
    void*                  srcMh_ = nullptr;
    void*                  dstMh_ = nullptr;
    void*                  sigMh_ = nullptr;

    int SendPeer() const { return (worldRank_ + 1) % worldSize_; }
    int RecvPeer() const { return (worldRank_ + worldSize_ - 1) % worldSize_; }
    static int SeedOf(int rank) { return 0x21 + 37 * rank; }

    // Mirrors the backend clamp of NCCL_RMA_IB_WR_BATCHSIZE to [2, 64].
    static int WrBatchSize()
    {
        return std::clamp(MPIHelpers::getEnvParam<int>("NCCL_RMA_IB_WR_BATCHSIZE", 64), 2, 64);
    }

    // Registers wide src_/dst_ windows and a signal word, and fills src_
    // with this rank's pattern. Returns a skip reason, or "" when ready.
    std::string PrepareWindows()
    {
        if (!AllocSymPair(&src_, &dst_, kWideSegments, kSegRequestBytes))
        {
            return "multi-segment VMM allocation unavailable on this host";
        }
        sig_ = AllocBuf(kSignalSize);
        if (prereq_.Skip(sig_ == nullptr, "signal buffer allocation failed")) return "signal buffer allocation failed";
        FillBuf(src_->ptr, src_->totalSize, SeedOf(worldRank_));
        EXPECT_EQ(ncclSuccess, RegMr(src_->ptr, src_->totalSize, &srcMh_));
        EXPECT_EQ(ncclSuccess, RegMr(dst_->ptr, dst_->totalSize, &dstMh_));
        EXPECT_EQ(ncclSuccess, RegMr(sig_, kSignalSize, &sigMh_));
        if (HasFailure()) return "registration failed";
        if (!MultiSegmentPathAvailable()) return "multi-segment path not exercised on this host";
        return "";
    }

    static std::vector<size_t> SegOffTable(const MultiSegmentVmmBuffer& b)
    {
        std::vector<size_t> off(1, 0);
        for (int s = 0; s < b.nSegments; s++)
        {
            off.push_back(off.back() + (b.segSizes.empty() ? b.segSize : b.segSizes[s]));
        }
        return off;
    }

    // Size from (localOff, remoteOff) whose chain is exactly `target` data WRs on
    // two windows laid out like `b`, or 0 when no in-range size gives that count.
    static size_t SizeForDataWrs(const MultiSegmentVmmBuffer& b, uint64_t localOff,
                                 uint64_t remoteOff, int target)
    {
        const std::vector<size_t> off = SegOffTable(b);
        std::vector<size_t> ends;
        for (size_t o : off)
        {
            if (o > localOff) ends.push_back(o - localOff);
            if (o > remoteOff) ends.push_back(o - remoteOff);
        }
        std::sort(ends.begin(), ends.end());
        ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
        const rcclIbSegTable table = {0, static_cast<uint32_t>(b.nSegments),
                                      reinterpret_cast<const uint64_t*>(off.data())};
        for (size_t size : ends)
        {
            if (localOff + size > b.totalSize || remoteOff + size > b.totalSize) break;
            rcclIbSegCursor cur;
            rcclIbSegCursorInit(&cur, &table, &table, localOff, remoteOff, size);
            rcclIbSegSlice slice;
            int wrs = 0;
            while (wrs <= target && rcclIbSegCursorNext(&cur, &table, &table, &slice)) wrs++;
            if (wrs == target && cur.remaining == 0) return size;
            if (wrs > target) break;
        }
        return 0;
    }

    // Chains with the given data-WR counts, back to back in `b`. The remote side
    // is shifted half a segment so each spanned segment costs two WRs.
    static std::vector<Chain> PlanChains(const MultiSegmentVmmBuffer& b, const std::vector<int>& dataWrs)
    {
        std::vector<Chain> chains;
        const size_t half = b.segSize / 2;
        uint64_t cursor = 0;
        for (int target : dataWrs)
        {
            const size_t size = SizeForDataWrs(b, cursor, cursor + half, target);
            if (size == 0) return {};
            chains.push_back({cursor, cursor + half, size});
            cursor = (cursor + half + size + b.segSize - 1) / b.segSize * b.segSize;
        }
        return chains;
    }

    Op MakeOp(OpKind kind, void* localMh, void* remoteMh, const Chain& chain,
              int ctx, bool aggregate) const
    {
        Op op;
        op.kind      = kind;
        op.localMh   = localMh;
        op.remoteMh  = remoteMh;
        op.chain     = chain;
        op.ctx       = ctx;
        op.aggregate = aggregate;
        return op;
    }

    ncclResult_t Post(const Op& op, void** req)
    {
        const uint32_t flags = op.aggregate ? ncclRmaOptFlagsAggregateRequests : ncclRmaOptFlagsDefault;
        const Chain& c = op.chain;
        const uint32_t peer = op.peer < 0 ? SendPeer() : op.peer;
        *req = nullptr;
        switch (op.kind)
        {
        case OpKind::Put:
            return rma_->iput(rmaCtx_, op.ctx, c.localOff, op.localMh, c.size,
                              c.remoteOff, op.remoteMh, peer, flags, req);
        case OpKind::Get:
            return rma_->iget(rmaCtx_, op.ctx, c.remoteOff, op.remoteMh, c.size,
                              c.localOff, op.localMh, peer, flags, req);
        case OpKind::PutSignal:
            return rma_->iputSignal(rmaCtx_, op.ctx, c.localOff, op.localMh, c.size,
                                    c.remoteOff, op.remoteMh, peer, op.sigOff,
                                    op.sigMh ? op.sigMh : sigMh_, op.sigAdd,
                                    op.sigAdd ? NCCL_NET_SIGNAL_OP_ADD : NCCL_NET_SIGNAL_OP_INC,
                                    /*isStrongSignal=*/false, flags, req);
        }
        return ncclInternalError;
    }

    static Region PutRegion(const Chain& c) { return {c.remoteOff, c.size, c.localOff}; }
    static Region GetRegion(const Chain& c) { return {c.localOff, c.size, c.remoteOff}; }

    // dst_ contents after RecvPeer() ran the same put plan into this rank.
    Expect IntoDst(const std::vector<Chain>& puts) const
    {
        Expect e{dst_->ptr, dst_->totalSize, {}, SeedOf(RecvPeer())};
        for (const Chain& c : puts) e.regions.push_back(PutRegion(c));
        return e;
    }

    std::vector<size_t> WaitOrderFor(size_t n, WaitOrder order) const
    {
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), size_t{0});
        if (order == WaitOrder::Reversed) std::reverse(idx.begin(), idx.end());
        if (order == WaitOrder::Shuffled)
        {
            std::shuffle(idx.begin(), idx.end(), std::mt19937(0x5EEDu + static_cast<unsigned>(worldRank_)));
        }
        return idx;
    }

    // One test() call. A request still pending is drained so it is not leaked.
    bool DoneOnFirstTest(void* req)
    {
        if (req == nullptr) return false;
        if (TestOnce(req)) return true;
        ADD_FAILURE() << "request was not complete on its first test()";
        (void)PollUntilDone(req);
        return false;
    }

    // Like PollUntilDone, but returns test()'s error, or ncclInProgress on timeout.
    ncclResult_t PollStatus(void* req, int timeoutMs)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        for (;;)
        {
            int done = 0;
            const ncclResult_t r = rma_->test(collComm_, req, &done);
            if (r != ncclSuccess || done) return r;
            if (std::chrono::steady_clock::now() >= deadline) return ncclInProgress;
            std::this_thread::sleep_for(std::chrono::microseconds(kPollSleepUs));
        }
    }

    // Deregisters now rather than in TearDown.
    ncclResult_t DeregMr(void* mh)
    {
        auto it = std::find(registeredMhandles_.begin(), registeredMhandles_.end(), mh);
        if (it != registeredMhandles_.end()) registeredMhandles_.erase(it);
        return rma_->deregMrSym(collComm_, mh);
    }

    // One D2H copy; reports the first byte that differs from the expectation.
    void ExpectRegions(const Expect& e)
    {
        std::vector<uint8_t> want(e.total, kSentinel), got(e.total);
        for (const Region& r : e.regions)
        {
            const int seed = r.seed < 0 ? e.seed : r.seed;
            for (size_t i = 0; i < r.size; i++)
                want[r.dstOff + i] = static_cast<uint8_t>((seed + r.srcOff + i) % 256);
        }
        ASSERT_EQ(hipSuccess, hipMemcpy(got.data(), e.window, e.total, hipMemcpyDeviceToHost));
        const auto diff = std::mismatch(want.begin(), want.end(), got.begin());
        EXPECT_TRUE(diff.first == want.end())
            << "first mismatch at offset " << (diff.first - want.begin()) << ": want 0x" << std::hex
            << int(*diff.first) << " got 0x" << int(*diff.second);
    }

    // Reset the expected windows, post `ops` (an iflush of dst_ after op
    // `flushAfter`), wait in `waitIdx` order, then check payloads and signals.
    void RunAndVerify(const std::vector<Op>& ops, const std::vector<size_t>& waitIdx,
                      const std::vector<Expect>& expects, uint64_t signals, int flushAfter = -1)
    {
        for (const Expect& e : expects) FillSentinel(e.window, e.total, kSentinel);
        FillSentinel(sig_, kSignalSize, 0);
        Barrier();

        bool ok = true;
        std::vector<void*> reqs(ops.size(), nullptr);
        void* flushReq = nullptr;
        for (size_t i = 0; i < ops.size(); i++)
        {
            const ncclResult_t r = Post(ops[i], &reqs[i]);
            if (r != ncclSuccess || reqs[i] == nullptr)
            {
                ADD_FAILURE() << "op " << i << " post failed: " << r;
                ok = false;
            }
            if (static_cast<int>(i) == flushAfter &&
                rma_->iflush(rmaCtx_, ops[i].ctx, dstMh_, RecvPeer(), &flushReq) != ncclSuccess)
            {
                ADD_FAILURE() << "iflush after op " << i << " failed";
                ok = false;
            }
        }
        if (flushReq) ok = PollUntilDone(flushReq) && ok;
        for (size_t i : waitIdx)
        {
            if (reqs[i]) ok = PollUntilDone(reqs[i]) && ok;
        }
        if (!MPIHelpers::allRanksTrue(ok))
        {
            ADD_FAILURE() << "post or wait failed on at least one rank";
            return;
        }
        Barrier();
        for (const Expect& e : expects) ExpectRegions(e);
        EXPECT_EQ(ReadSignal(sig_), signals) << "signals from rank " << RecvPeer();
        Barrier();
    }
};

// Chains up to NCCL_RMA_IB_WR_BATCHSIZE WRs go through the batch; longer ones
// drain it and post directly. Both sides of that limit must land intact.
TEST_F(RmaMultiSegmentPostMPITest, ChainsAroundWrBatchSizeLandIntact)
{
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());

    const int batch  = WrBatchSize();
    const int maxWrs = 2 * src_->nSegments - 1;
    for (bool aggregate : {false, true})
    {
        for (OpKind kind : {OpKind::Put, OpKind::PutSignal})
        {
            const int signalWrs = kind == OpKind::PutSignal ? 1 : 0;
            for (int total : {1, 2, batch - 1, batch, batch + 1, maxWrs, maxWrs + 1})
            {
                const int dataWrs = total - signalWrs;
                if (dataWrs < 1 || dataWrs > maxWrs) continue;
                SCOPED_TRACE(::testing::Message() << "aggregate=" << aggregate << " putSignal="
                                                  << signalWrs << " wrs=" << total << " batch=" << batch);
                const std::vector<Chain> chains = PlanChains(*src_, {dataWrs});
                ASSERT_EQ(chains.size(), 1u);
                RunAndVerify({MakeOp(kind, srcMh_, dstMh_, chains[0], 0, aggregate)}, {0},
                             {IntoDst(chains)}, static_cast<uint64_t>(signalWrs));
                if (HasFailure()) return;
            }
        }
    }
}

// Aggregated requests share one doorbell that test() on any of them rings.
// They must complete in any wait order, also with an iflush posted mid-batch.
TEST_F(RmaMultiSegmentPostMPITest, QueuedRequestsCompleteInAnyWaitOrder)
{
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());

    const std::vector<Chain> chains = PlanChains(*src_, {1, 3, std::min(WrBatchSize() + 1, 7), 2, 5});
    ASSERT_EQ(chains.size(), 5u);
    for (bool aggregate : {false, true})
    {
        std::vector<Op> ops;
        uint64_t signals = 0;
        for (size_t i = 0; i < chains.size(); i++)
        {
            const OpKind kind = i % 2 ? OpKind::PutSignal : OpKind::Put;
            signals += kind == OpKind::PutSignal;
            ops.push_back(MakeOp(kind, srcMh_, dstMh_, chains[i], 0, aggregate));
        }
        for (WaitOrder order : {WaitOrder::InOrder, WaitOrder::Reversed, WaitOrder::Shuffled})
        {
            SCOPED_TRACE(::testing::Message() << "aggregate=" << aggregate << " order=" << int(order));
            RunAndVerify(ops, WaitOrderFor(ops.size(), order), {IntoDst(chains)}, signals,
                         /*flushAfter=*/1);
            if (HasFailure()) return;
        }
    }
}

// Single-segment and multi-segment requests, puts and gets, share the per-peer
// QP and batch. Interleaved, each payload must land only in its own range.
TEST_F(RmaMultiSegmentPostMPITest, InterleavedSingleAndMultiSegmentOps)
{
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());

    constexpr size_t kPlain = kMiB;
    void* plainSrc = AllocBuf(kPlain);
    void* plainDst = AllocBuf(kPlain);
    MultiSegmentVmmBuffer* getBuf = pool_.AllocUniform(kWideSegments, kSegRequestBytes);
    if (prereq_.Skip(plainSrc == nullptr || plainDst == nullptr || getBuf == nullptr, kVmmAllocFailed))
    {
        GTEST_SKIP() << "buffer allocation failed";
    }
    FillBuf(plainSrc, kPlain, SeedOf(worldRank_));
    void *plainSrcMh = nullptr, *plainDstMh = nullptr, *getMh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(plainSrc, kPlain, &plainSrcMh));
    ASSERT_EQ(ncclSuccess, RegMr(plainDst, kPlain, &plainDstMh));
    ASSERT_EQ(ncclSuccess, RegMr(getBuf->ptr, getBuf->totalSize, &getMh));

    const std::vector<Chain> puts = PlanChains(*src_, {3, 5, 2});
    const std::vector<Chain> gets = PlanChains(*getBuf, {4, 1});
    ASSERT_EQ(puts.size(), 3u);
    ASSERT_EQ(gets.size(), 2u);
    const Chain plainA{0, 0, kPlain / 4};
    const Chain plainB{kPlain / 4, kPlain / 2, kPlain / 4};

    for (bool aggregate : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "aggregate=" << aggregate);
        const std::vector<Op> ops = {
            MakeOp(OpKind::Put,       srcMh_,     dstMh_,     puts[0], 0, aggregate),
            MakeOp(OpKind::Put,       plainSrcMh, plainDstMh, plainA,  0, aggregate),
            MakeOp(OpKind::Get,       getMh,      srcMh_,     gets[0], 0, aggregate),
            MakeOp(OpKind::PutSignal, plainSrcMh, plainDstMh, plainB,  0, aggregate),
            MakeOp(OpKind::PutSignal, srcMh_,     dstMh_,     puts[1], 0, aggregate),
            MakeOp(OpKind::Get,       getMh,      srcMh_,     gets[1], 0, aggregate),
            MakeOp(OpKind::Put,       srcMh_,     dstMh_,     puts[2], 0, aggregate),
        };
        const std::vector<Expect> expects = {
            IntoDst(puts),
            {plainDst, kPlain, {PutRegion(plainA), PutRegion(plainB)}, SeedOf(RecvPeer())},
            {getBuf->ptr, getBuf->totalSize, {GetRegion(gets[0]), GetRegion(gets[1])}, SeedOf(SendPeer())},
        };
        RunAndVerify(ops, WaitOrderFor(ops.size(), WaitOrder::Shuffled), expects, /*signals=*/2);
        if (HasFailure()) return;
    }
}

// Ops wider than one round post round by round from the peer's queue, and the ops behind them wait their turn.
// A wide put or put-signal, a short put and a wide get on one peer must each land intact.
TEST_F(RmaMultiSegmentPostMPITest, OpsWiderThanOneRoundLandIntact)
{
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());

    // PlanChains costs two WRs per spanned segment, so 48 segments hold an 80-WR chain plus a short one.
    constexpr int kRoundSegments = 48;
    constexpr int kWideWrs       = 80;
    static_assert(kWideWrs > RCCL_RMA_MAX_DATA_WRS, "the wide chains must take a second round");
    MultiSegmentVmmBuffer* src = pool_.AllocUniform(kRoundSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* dst = pool_.AllocUniform(kRoundSegments, kSegRequestBytes);
    MultiSegmentVmmBuffer* getWin = pool_.AllocUniform(kRoundSegments, kSegRequestBytes);
    if (prereq_.Skip(src == nullptr || dst == nullptr || getWin == nullptr, kVmmAllocFailed))
    {
        GTEST_SKIP() << "wide window allocation failed";
    }
    FillBuf(src->ptr, src->totalSize, SeedOf(worldRank_));
    void *srcMh = nullptr, *dstMh = nullptr, *getMh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(src->ptr, src->totalSize, &srcMh));
    ASSERT_EQ(ncclSuccess, RegMr(dst->ptr, dst->totalSize, &dstMh));
    ASSERT_EQ(ncclSuccess, RegMr(getWin->ptr, getWin->totalSize, &getMh));

    const std::vector<Chain> puts = PlanChains(*src, {kWideWrs, 3});
    const std::vector<Chain> gets = PlanChains(*getWin, {kWideWrs});
    ASSERT_EQ(puts.size(), 2u);
    ASSERT_EQ(gets.size(), 1u);
    for (OpKind wide : {OpKind::Put, OpKind::PutSignal})
    {
        SCOPED_TRACE(::testing::Message() << "putSignal=" << (wide == OpKind::PutSignal));
        const std::vector<Op> ops = {
            MakeOp(wide,        srcMh, dstMh, puts[0], 0, /*aggregate=*/false),
            MakeOp(OpKind::Put, srcMh, dstMh, puts[1], 0, /*aggregate=*/false),
            MakeOp(OpKind::Get, getMh, srcMh, gets[0], 0, /*aggregate=*/false),
        };
        const std::vector<Expect> expects = {
            {dst->ptr, dst->totalSize, {PutRegion(puts[0]), PutRegion(puts[1])}, SeedOf(RecvPeer())},
            {getWin->ptr, getWin->totalSize, {GetRegion(gets[0])}, SeedOf(SendPeer())},
        };
        RunAndVerify(ops, WaitOrderFor(ops.size(), WaitOrder::Reversed), expects,
                     /*signals=*/wide == OpKind::PutSignal ? 1u : 0u);
        if (HasFailure()) return;
    }
}

// A zero-size op posts no WR and takes the previous request's id, so it
// completes only after that request has, which must then test done at once.
TEST_F(RmaMultiSegmentPostMPITest, ZeroSizeOpsCompleteWithQueuedPredecessor)
{
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());

    const std::vector<Chain> chains = PlanChains(*src_, {3, 4});
    ASSERT_EQ(chains.size(), 2u);
    const Chain none{};
    for (bool aggregate : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "aggregate=" << aggregate);
        FillSentinel(dst_->ptr, dst_->totalSize, kSentinel);
        FillSentinel(sig_, kSignalSize, 0);
        Barrier();

        bool ok = true;
        auto post = [&](OpKind kind, void* localMh, void* remoteMh, const Chain& c) {
            void* req = nullptr;
            ok = Post(MakeOp(kind, localMh, remoteMh, c, 0, aggregate), &req) == ncclSuccess && req && ok;
            return req;
        };
        void* first   = post(OpKind::Put, srcMh_, dstMh_, chains[0]);
        void* zeroPut = post(OpKind::Put, srcMh_, dstMh_, none);
        void* zeroGet = post(OpKind::Get, dstMh_, srcMh_, none);
        ok = PollUntilDone(zeroPut) && ok;
        ok = DoneOnFirstTest(first) && ok;
        ok = DoneOnFirstTest(zeroGet) && ok;

        void* signalOnly = post(OpKind::PutSignal, srcMh_, dstMh_, none);
        void* second     = post(OpKind::Put, srcMh_, dstMh_, chains[1]);
        ok = PollUntilDone(second) && ok;
        ok = DoneOnFirstTest(signalOnly) && ok;
        ok = DoneOnFirstTest(post(OpKind::Put, srcMh_, dstMh_, none)) && ok;

        ASSERT_TRUE(MPIHelpers::allRanksTrue(ok)) << "post or wait failed on at least one rank";
        Barrier();
        ExpectRegions(IntoDst(chains));
        EXPECT_EQ(ReadSignal(sig_), 1u) << "signal-only put from rank " << RecvPeer();
        Barrier();
        if (HasFailure()) return;
    }
}

// Each context batches on its own QP. Waiting on one context must not need the
// others' doorbells, and their still-queued requests must land afterwards.
TEST_F(RmaMultiSegmentPostMPITest, ContextBatchesDrainIndependently)
{
    numContexts_ = std::max(2, MPIHelpers::getEnvParam<int>("RCCL_TEST_RMA_CONTEXTS", 2));
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());

    const std::vector<Chain> chains = PlanChains(*src_, {2, 3, 1, 4, 2, 3});
    ASSERT_EQ(chains.size(), 6u);
    std::vector<Op> ops;
    uint64_t signals = 0;
    for (size_t i = 0; i < chains.size(); i++)
    {
        const OpKind kind = i % 3 == 2 ? OpKind::PutSignal : OpKind::Put;
        signals += kind == OpKind::PutSignal;
        ops.push_back(MakeOp(kind, srcMh_, dstMh_, chains[i],
                             static_cast<int>(i) % numContexts_, /*aggregate=*/true));
    }
    std::vector<size_t> waitIdx = WaitOrderFor(ops.size(), WaitOrder::InOrder);
    std::stable_sort(waitIdx.begin(), waitIdx.end(),
                     [&](size_t a, size_t b) { return ops[a].ctx > ops[b].ctx; });
    RunAndVerify(ops, waitIdx, {IntoDst(chains)}, signals);
}

// A zero-size op on a fresh context completes at once; a signal-only put routes to its offset's segment.
TEST_F(RmaMultiSegmentPostMPITest, ZeroSizeOnFreshContextAndSignalOnlyIntoLaterSegment)
{
    numContexts_ = 2;
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());
    MultiSegmentVmmBuffer* sw = nullptr;
    void* swMh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareSignalWindow(3, &sw, &swMh));

    const std::vector<Chain> chains = PlanChains(*src_, {3});
    ASSERT_EQ(chains.size(), 1u);
    const Chain none{};
    const size_t seg = sw->segSize;
    const uint64_t sigOffs[] = {seg - 8, seg, 2 * seg + 64};
    FillSentinel(dst_->ptr, dst_->totalSize, kSentinel);
    Barrier();

    bool ok = true;
    void* queued = nullptr;
    void* fresh  = nullptr;
    ok = Post(MakeOp(OpKind::Put, srcMh_, dstMh_, chains[0], 0, /*aggregate=*/true), &queued) == ncclSuccess;
    ok = Post(MakeOp(OpKind::Get, dstMh_, srcMh_, none, 1, /*aggregate=*/true), &fresh) == ncclSuccess && ok;
    ok = DoneOnFirstTest(fresh) && ok;
    ok = PollUntilDone(queued) && ok;
    std::vector<SignalWord> want;
    for (uint64_t off : sigOffs)
    {
        Op op = MakeOp(OpKind::PutSignal, srcMh_, dstMh_, none, 1, /*aggregate=*/true);
        op.sigMh  = swMh;
        op.sigOff = off;
        void* req = nullptr;
        ok = Post(op, &req) == ncclSuccess && PollUntilDone(req) && ok;
        want.push_back({off, 1});
    }
    ASSERT_TRUE(MPIHelpers::allRanksTrue(ok)) << "post or wait failed on at least one rank";
    Barrier();

    ExpectRegions(IntoDst(chains));
    ExpectOnlySignals(sw, want);
    Barrier();
}

// Rejected ops return before taking a request slot or a sequence id. Only even
// ranks reject, 2x the slot pool in total, between queued ops of their own.
TEST_F(RmaMultiSegmentPostMPITest, RejectedOpsLeaveNoSlotOrSequenceGap)
{
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());

    constexpr int kBurst          = 64;
    const bool    rejecter        = worldRank_ % 2 == 0;
    const size_t  total           = src_->totalSize;
    const std::vector<Chain> chains = PlanChains(*src_, {3, 2, 4});
    ASSERT_EQ(chains.size(), 3u);

    for (bool aggregate : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "aggregate=" << aggregate);
        FillSentinel(dst_->ptr, dst_->totalSize, kSentinel);
        Barrier();

        bool ok = true;
        auto post = [&](const Op& op) {
            void* req = nullptr;
            ok = Post(op, &req) == ncclSuccess && req && ok;
            return req;
        };
        auto rejected = [&](int i) {
            Op op = MakeOp(OpKind::Put, srcMh_, dstMh_, {0, 0, total + 1}, 0, aggregate);
            if (i % 4 == 1) op = MakeOp(OpKind::Get, dstMh_, srcMh_, {0, 1, total}, 0, aggregate);
            if (i % 4 == 2)
            {
                op = MakeOp(OpKind::PutSignal, srcMh_, dstMh_, chains[0], 0, aggregate);
                op.sigOff = 4;
            }
            void* req = nullptr;
            const ncclResult_t r = i % 4 == 3
                ? rma_->iputSignal(rmaCtx_, 0, 0, srcMh_, 0, 0, dstMh_, SendPeer(), 0, sigMh_, 0,
                                   /*signalOp=*/0xFF, false, ncclRmaOptFlagsDefault, &req)
                : Post(op, &req);
            return r != ncclSuccess && req == nullptr;
        };

        void* first = post(MakeOp(OpKind::Put, srcMh_, dstMh_, chains[0], 0, aggregate));
        int accepted = 0;
        for (int i = 0; rejecter && i < kIbRequestSlots; i++) accepted += !rejected(i);
        EXPECT_EQ(accepted, 0) << "invalid ops were accepted";
        void* zero = post(MakeOp(OpKind::Put, srcMh_, dstMh_, Chain{}, 0, aggregate));
        ok = PollUntilDone(zero) && ok;
        ok = DoneOnFirstTest(first) && ok;

        std::vector<void*> reqs;
        for (int i = 0; i < kBurst; i++)
        {
            reqs.push_back(post(MakeOp(OpKind::Put, srcMh_, dstMh_, chains[1 + i % 2], 0, aggregate)));
        }
        for (auto it = reqs.rbegin(); it != reqs.rend(); ++it)
        {
            if (*it) ok = PollUntilDone(*it) && ok;
        }

        ASSERT_TRUE(MPIHelpers::allRanksTrue(ok)) << "post or wait failed on at least one rank";
        Barrier();
        ExpectRegions(IntoDst(chains));
        Barrier();
        if (HasFailure()) return;
    }
}

// Every registration reuses the comm's consensus buffer and builds fresh
// per-rank tables. Many cycles over changing layouts must each move intact data.
TEST_F(RmaMultiSegmentPostMPITest, RegistrationChurnKeepsTransfersIntact)
{
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());

    const int cycles = MPIHelpers::getEnvParam<int>("RCCL_TEST_RMA_REG_CYCLES", 100);
    struct Layout { MultiSegmentVmmBuffer* src; MultiSegmentVmmBuffer* dst; };
    std::vector<Layout> layouts;
    for (int nSeg : {2, 7, kWideSegments})
    {
        Layout l{};
        if (!AllocSymPair(&l.src, &l.dst, nSeg, kSegRequestBytes))
        {
            GTEST_SKIP() << nSeg << "-segment VMM allocation unavailable";
        }
        FillBuf(l.src->ptr, l.src->totalSize, SeedOf(worldRank_));
        layouts.push_back(l);
    }

    for (int i = 0; i < cycles; i++)
    {
        const Layout& l = layouts[i % layouts.size()];
        SCOPED_TRACE(::testing::Message() << "cycle " << i << " nSeg " << l.src->nSegments);
        void *srcMh = nullptr, *dstMh = nullptr;
        ASSERT_EQ(ncclSuccess, RegMr(l.src->ptr, l.src->totalSize, &srcMh));
        ASSERT_EQ(ncclSuccess, RegMr(l.dst->ptr, l.dst->totalSize, &dstMh));
        const Chain c{0, 0, l.src->totalSize};
        RunAndVerify({MakeOp(OpKind::Put, srcMh, dstMh, c, 0, /*aggregate=*/i % 2 == 1)}, {0},
                     {{l.dst->ptr, l.dst->totalSize, {PutRegion(c)}, SeedOf(RecvPeer())}}, 0);
        EXPECT_EQ(ncclSuccess, DeregMr(srcMh));
        EXPECT_EQ(ncclSuccess, DeregMr(dstMh));
        if (HasFailure()) return;
    }
}

// Only a chain's last WR is signaled, so its WRs hold send-queue slots until a
// test() polls the CQ. Max-length putSignals filling the queue must all land.
TEST_F(RmaMultiSegmentPostMPITest, IPutSignalFillsSendQueueWithoutTest)
{
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());

    const int maxWrs = 2 * src_->nSegments - 1;
    const std::vector<Chain> chains = PlanChains(*src_, {maxWrs});
    ASSERT_EQ(chains.size(), 1u);
    // Without test() each putSignal also holds one of the comm's request slots.
    const int inflight = std::min(kSendQueueWrs / (maxWrs + 1), kIbRequestSlots);
    const std::vector<Op> ops(inflight, MakeOp(OpKind::PutSignal, srcMh_, dstMh_, chains[0], 0, false));
    RunAndVerify(ops, WaitOrderFor(ops.size(), WaitOrder::Reversed), {IntoDst(chains)},
                 static_cast<uint64_t>(inflight));
}

// The proxies keep RCCL_RMA_MAX_INFLIGHT_REQUESTS per peer: that many worst-case putSignals without test() all land.
TEST_F(RmaMultiSegmentPostMPITest, IPutSignalAtProxyInflightCapFitsSendQueue)
{
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());

    const int maxWrs = 2 * src_->nSegments - 1;
    const std::vector<Chain> chains = PlanChains(*src_, {maxWrs});
    ASSERT_EQ(chains.size(), 1u);
    const int inflight = RCCL_RMA_MAX_INFLIGHT_REQUESTS;
    ASSERT_LE(inflight * RCCL_RMA_MAX_SIGNAL_WRS, kSendQueueWrs) << "the proxy cap must fit the send queue";
    for (bool aggregate : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "aggregate=" << aggregate);
        const std::vector<Op> ops(inflight, MakeOp(OpKind::PutSignal, srcMh_, dstMh_, chains[0], 0, aggregate));
        RunAndVerify(ops, WaitOrderFor(ops.size(), WaitOrder::Reversed), {IntoDst(chains)},
                     static_cast<uint64_t>(inflight));
        if (HasFailure()) return;
    }
}

// Opt-in (RCCL_MSEG_SQ_STRESS=1): twice the send queue of putSignals, past the proxy cap. Overflow stays on this
// context: rejected posts return no request, accepted ones finish, and the other context works.
TEST_F(RmaMultiSegmentPostMPITest, IPutSignalSendQueueOversubscribe)
{
    if (MPIHelpers::anyRankTrue(MPIHelpers::getEnvParam<int>("RCCL_MSEG_SQ_STRESS", 0) == 0))
    {
        GTEST_SKIP() << "set RCCL_MSEG_SQ_STRESS=1 to flood the RMA send queue";
    }
    numContexts_ = 2;
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());

    constexpr int kWaitMs = 15000;
    const int maxWrs = 2 * src_->nSegments - 1;
    const std::vector<Chain> chains = PlanChains(*src_, {maxWrs});
    ASSERT_EQ(chains.size(), 1u);
    const Op op = MakeOp(OpKind::PutSignal, srcMh_, dstMh_, chains[0], 0, false);
    Barrier();

    int rejected = 0, failed = 0;
    std::vector<void*> reqs;
    for (int i = 0; i < 2 * kSendQueueWrs / (maxWrs + 1); i++)
    {
        void* req = nullptr;
        if (Post(op, &req) == ncclSuccess)
        {
            if (req) reqs.push_back(req);
            else ADD_FAILURE() << "accepted post " << i << " returned no request";
            continue;
        }
        rejected++;
        if (req) ADD_FAILURE() << "rejected post " << i << " returned a request";
    }
    for (size_t i = 0; i < reqs.size(); i++)
    {
        const ncclResult_t r = PollStatus(reqs[i], kWaitMs);
        if (r == ncclInProgress) ADD_FAILURE() << "request " << i << " hung after the flood";
        failed += r != ncclSuccess && r != ncclInProgress;
    }
    TEST_INFO("send-queue flood: %zu accepted, %d rejected, %d ended in an error",
              reqs.size(), rejected, failed);
    EXPECT_FALSE(reqs.empty()) << "the flood posted nothing";

    EXPECT_TRUE(MPIHelpers::allRanksTrue(!HasFailure())) << "flood hung or misreported on some rank";
    if (HasFailure()) return;

    // Context 1 uses src and dst bytes the flood never touches, checked alone.
    const Chain& flood = chains[0];
    const Chain spare{src_->totalSize - flood.remoteOff, 0, flood.remoteOff};
    ASSERT_GE(spare.localOff, flood.localOff + flood.size);
    FillSentinel(dst_->ptr, dst_->totalSize, kSentinel);
    Barrier();
    void* req = nullptr;
    const ncclResult_t r = Post(MakeOp(OpKind::Put, srcMh_, dstMh_, spare, /*ctx=*/1, false), &req);
    const bool ok = r == ncclSuccess && req != nullptr && PollUntilDone(req);
    EXPECT_TRUE(MPIHelpers::allRanksTrue(ok)) << "context 1 put failed after the flood: " << r;
    if (HasFailure()) return;
    Barrier();
    ExpectRegions({dst_->ptr, spare.size, {PutRegion(spare)}, SeedOf(RecvPeer())});
}

// Every rank puts its slot into every peer and gets every peer's slot, driving
// N-1 queue pairs at once. Slots of total/N bytes cut segments at N-dependent offsets.
TEST_F(RmaMultiSegmentPostMPITest, AllPairsPutAndGetLandInOwnSlots)
{
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());
    MultiSegmentVmmBuffer* getBuf = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(AllocWindow(kWideSegments, &getBuf, "get window allocation unavailable"));
    void* getMh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(getBuf->ptr, getBuf->totalSize, &getMh));

    const size_t slot = src_->totalSize / worldSize_ / 8 * 8;
    auto slotOf = [&](int r) { return Chain{r * slot, r * slot, slot}; };
    for (bool aggregate : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "aggregate=" << aggregate);
        std::vector<Op> ops;
        Expect intoDst{dst_->ptr, dst_->totalSize, {}, 0};
        Expect intoGet{getBuf->ptr, getBuf->totalSize, {}, 0};
        for (int k = 1; k < worldSize_; k++)
        {
            const int to   = (worldRank_ + k) % worldSize_;
            const int from = (worldRank_ + worldSize_ - k) % worldSize_;
            Op put = MakeOp(OpKind::Put, srcMh_, dstMh_, slotOf(worldRank_), 0, aggregate);
            Op get = MakeOp(OpKind::Get, getMh, srcMh_, slotOf(to), 0, aggregate);
            put.peer = get.peer = to;
            ops.push_back(put);
            ops.push_back(get);
            Region in = PutRegion(slotOf(from));
            in.seed = SeedOf(from);
            intoDst.regions.push_back(in);
            Region got = GetRegion(slotOf(to));
            got.seed = SeedOf(to);
            intoGet.regions.push_back(got);
        }
        RunAndVerify(ops, WaitOrderFor(ops.size(), WaitOrder::Shuffled), {intoDst, intoGet}, 0);
        if (HasFailure()) return;
    }
}

// Every other rank signals one root through a multi-segment signal window: its
// own word in segment (rank % nSeg) and one word all of them add to.
TEST_F(RmaMultiSegmentPostMPITest, ManyToOneSignalsIntoMultiSegmentWindow)
{
    if (!SetUpFixture(/*minProcesses=*/2)) return;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareWindows());
    MultiSegmentVmmBuffer* sw = nullptr;
    void* swMh = nullptr;
    RMA_MSEG_PREPARE_OR_SKIP(PrepareSignalWindow(4, &sw, &swMh));

    constexpr int    kRounds = 8;
    constexpr int    kRoot   = 0;
    const size_t     seg     = sw->segSize;
    const uint64_t   shared  = seg + seg / 2;
    const size_t     slot    = src_->totalSize / worldSize_ / 8 * 8;
    auto ownOff = [&](int r) { return static_cast<uint64_t>(r % sw->nSegments) * seg + 8 * r; };
    auto slotOf = [&](int r) { return Chain{r * slot, r * slot, slot}; };
    FillSentinel(dst_->ptr, dst_->totalSize, kSentinel);
    Barrier();

    bool ok = true;
    if (worldRank_ != kRoot)
    {
        std::vector<void*> reqs;
        for (int i = 0; i < kRounds; i++)
        {
            Op own = MakeOp(OpKind::PutSignal, srcMh_, dstMh_, slotOf(worldRank_), 0, i % 2 == 1);
            own.peer   = kRoot;
            own.sigMh  = swMh;
            own.sigOff = ownOff(worldRank_);
            Op add = MakeOp(OpKind::PutSignal, srcMh_, dstMh_, Chain{}, 0, i % 2 == 1);
            add.peer   = kRoot;
            add.sigMh  = swMh;
            add.sigOff = shared;
            add.sigAdd = static_cast<uint64_t>(worldRank_ + 1);
            for (const Op& op : {own, add})
            {
                void* req = nullptr;
                ok = Post(op, &req) == ncclSuccess && req && ok;
                if (req) reqs.push_back(req);
            }
        }
        for (size_t i : WaitOrderFor(reqs.size(), WaitOrder::Shuffled))
            ok = PollUntilDone(reqs[i]) && ok;
    }
    ASSERT_TRUE(MPIHelpers::allRanksTrue(ok)) << "post or wait failed on at least one rank";
    Barrier();

    if (worldRank_ == kRoot)
    {
        Expect payload{dst_->ptr, dst_->totalSize, {}, 0};
        uint64_t sum = 0;
        std::vector<SignalWord> want;
        for (int r = 0; r < worldSize_; r++)
        {
            if (r == kRoot) continue;
            Region in = PutRegion(slotOf(r));
            in.seed = SeedOf(r);
            payload.regions.push_back(in);
            sum += static_cast<uint64_t>(r + 1);
            want.push_back({ownOff(r), uint64_t{kRounds}});
        }
        want.push_back({shared, kRounds * sum});
        ExpectOnlySignals(sw, want);
        ExpectRegions(payload);
    }
    Barrier();
}


} // namespace RCCLRmaTests

#else // !RCCL_HAS_RMA_IB_PROXY

#include <gtest/gtest.h>

TEST(RmaMultiSegmentMPITest, BuildSkipped)
{
    GTEST_SKIP() << "IB RMA proxy backend not built into this binary. Skipping multi-segment RMA tests...";
}

#endif // RCCL_HAS_RMA_IB_PROXY

#endif // MPI_TESTS_ENABLED
