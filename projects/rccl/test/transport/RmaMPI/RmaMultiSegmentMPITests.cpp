/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
// Multi-segment DMA-BUF registration tests for ncclRmaIbProxy / IbCastRmaIbProxy
// (AIRUNTIME-2351). Run with NCCL_NET=IB or NCCL_NET=IB-CAST and NCCL_CUMEM_ENABLE=1.

#ifdef MPI_TESTS_ENABLED
#ifdef RCCL_HAS_RMA_IB_PROXY

#include "RmaMultiSegmentFixture.hpp"

namespace RCCLRmaTests
{

// Reproducer (AIRUNTIME-2351): a multi-segment window must register per-segment
// and move data correctly end to end. Flagship positive case.
TEST_F(RmaMultiSegmentMPITest, Reproducer_MultiSegmentRegistrationAndTransfer)
{
    if (!SetUpFixture(/*minProcs=*/2, /*maxProcs=*/2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t kSize = sb->totalSize;

    if (worldRank_ == 0)
        FillBuf(sb->ptr, kSize, /*seed=*/0xA0);

    void *sendMh = nullptr, *sendGh = nullptr, *recvMh = nullptr, *recvGh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, kSize, &sendMh, &sendGh))
        << "multi-segment send buffer registration failed (the AIRUNTIME-2351 bug)";
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, kSize, &recvMh, &recvGh))
        << "multi-segment recv buffer registration failed (the AIRUNTIME-2351 bug)";

    // Confirm the per-segment path fired; otherwise the feature isn't exercised.
    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "Buffer registered as a single MR (cuMem disabled or "
                        "range not segmented) - multi-segment path not exercised";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, /*context=*/0,
                             /*srcOff=*/0, sendMh, kSize,
                             /*dstOff=*/0, recvMh, /*peerRank=*/1, ncclRmaOptFlagsDefault, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
        EXPECT_TRUE(VerifyBuf(rb->ptr, kSize, /*seed=*/0xA0))
            << "data corrupted across segment boundaries";
}

// Register two complete physical mappings plus half of a third. The final MR
// must be clipped to the requested range, while a transfer ending at the last
// registered byte succeeds without touching the mapped-but-unregistered tail.
TEST_F(RmaMultiSegmentMPITest, PartialFinalSegmentRegistrationAndTransfer)
{
    if (!SetUpFixture(/*minProcs=*/2, /*maxProcs=*/2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb, /*nSegments=*/3))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t registeredBytes = 2 * sb->segSize + sb->segSize / 2;
    const size_t transferOffset  = 2 * sb->segSize - 4096;
    const size_t transferBytes   = registeredBytes - transferOffset;
    constexpr uint8_t kSentinel  = 0xB7;

    if (worldRank_ == 0)
        FillBuf(static_cast<uint8_t*>(sb->ptr) + transferOffset,
                transferBytes, /*seed=*/0x71);
    if (worldRank_ == 1)
        FillSentinel(rb->ptr, rb->totalSize, kSentinel);

    void *sendMh = nullptr, *sendGh = nullptr;
    void *recvMh = nullptr, *recvGh = nullptr;
    ASSERT_EQ(ncclSuccess,
              RegMr(sb->ptr, registeredBytes, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess,
              RegMr(rb->ptr, registeredBytes, &recvMh, &recvGh));

    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, 0, transferOffset, sendMh, transferBytes,
                             transferOffset, recvMh, 1, ncclRmaOptFlagsDefault, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
        ExpectPayloadIsolated(rb->ptr, rb->totalSize, transferOffset,
                              transferBytes, /*seed=*/0x71, kSentinel,
                              "partial final physical segment");
}

// IPut starting/ending mid-segment so the WR builder splits on a non-zero
// per-segment offset on both sides (most prone to addr/lkey/rkey errors).
TEST_F(RmaMultiSegmentMPITest, IPutCrossSegmentBoundaryAtOffset)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb))
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

    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, 0, /*srcOff=*/off, sendMh, kSize,
                             /*dstOff=*/off, recvMh, 1, ncclRmaOptFlagsDefault, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
    {
        ExpectPayloadIsolated(rb->ptr, rb->totalSize, off, kSize,
                              /*seed=*/0x5A, kSentinel,
                              "IPut at offset " + std::to_string(off));
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

    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iget(rmaCtx_, 0, /*remoteOff=*/0, mh, kSize,
                             /*localOff=*/0, mh, /*peerRank=*/1, ncclRmaOptFlagsDefault, &req));
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

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb))
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

    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iget(rmaCtx_, 0, /*remoteOff=*/off, srcMh, kSize,
                             /*localOff=*/off, dstMh, /*peerRank=*/1, ncclRmaOptFlagsDefault, &req));
        ASSERT_TRUE(PollUntilDone(req));

        ExpectPayloadIsolated(rb->ptr, rb->totalSize, off, kSize,
                              /*seed=*/0x6E, kSentinel,
                              "IGet at offset " + std::to_string(off));
    }
    Barrier();
}

// Receiver-side flush over a multi-segment buffer: after a plain iput, rank 1
// must fence EVERY physical segment via iflush (one loopback read per segment)
// before reading. Exercises the multi-segment flush path; a segment-0-only
// flush faults here on a multi-segment handle.
TEST_F(RmaMultiSegmentMPITest, IFlushMultiSegment)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t kSize = sb->totalSize;
    if (worldRank_ == 0)
        FillBuf(sb->ptr, kSize, /*seed=*/0x3C);

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, kSize, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, kSize, &recvMh, &recvGh));

    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, 0, 0, sendMh, kSize, 0, recvMh, /*peerRank=*/1, ncclRmaOptFlagsDefault, &req));
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

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb))
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

    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, 0, /*srcOff=*/off, sendMh, kSize,
                             /*dstOff=*/off, recvMh, /*peerRank=*/1, ncclRmaOptFlagsDefault, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
    {
        void* freq = nullptr;
        EXPECT_EQ(ncclSuccess, rma_->iflush(rmaCtx_, 0, recvMh, /*peerRank=*/0, &freq))
            << "multi-segment iflush post failed after partial put";
        EXPECT_TRUE(PollUntilDone(freq)) << "multi-segment flush did not complete";

        ExpectPayloadIsolated(rb->ptr, rb->totalSize, off, kSize,
                              /*seed=*/0x4F, kSentinel,
                              "partial IPut followed by flush");
    }
    Barrier();
}

// REGRESSION (flush fast path): iflush on an ordinary single-allocation buffer
// must still fence and verify. IFlushMultiSegment only covers the multi-segment
// handle and SingleSegmentRegression never flushes, so the nSeg==1 flush path --
// which the multi-segment change must not regress -- is otherwise untested. No
// nSegments gate: a single-segment buffer intentionally does NOT take
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
                  rma_->iput(rmaCtx_, 0, 0, sendMh, kSize, 0, recvMh, /*peerRank=*/1, ncclRmaOptFlagsDefault, &req));
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

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb))
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

    if (!MultiSegmentPathAvailable())
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

// Sweep IPut payloads from 0 to the full window across edge sizes (byte, word,
// page, 64K, and segment boundaries). Verifies the payload landed and that no
// byte past `size` was touched (catches over-write / boundary-split errors).
TEST_F(RmaMultiSegmentMPITest, IPutSizeSweepFromZero)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    constexpr uint8_t kSentinel = 0xBD;

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, sb->totalSize, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, rb->totalSize, &recvMh, &recvGh));

    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    RunIPutSizeSweep(sb, rb, sendMh, recvMh, /*offset=*/0,
                     /*seedBase=*/0x40, kSentinel);
}

// Same sweep but starting at an offset that sits just inside the first segment,
// so every transfer begins mid-segment and most cross >=1 boundary. Stresses
// the non-zero per-segment offset arithmetic on both local and remote sides.
TEST_F(RmaMultiSegmentMPITest, IPutSizeSweepAtBoundaryOffset)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t off = (sb->segSize >= 64) ? sb->segSize - 64 : 0;
    constexpr uint8_t kSentinel = 0x9C;

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, sb->totalSize, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, rb->totalSize, &recvMh, &recvGh));

    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    RunIPutSizeSweep(sb, rb, sendMh, recvMh, off,
                     /*seedBase=*/0x80, kSentinel);
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
        if (!MultiSegmentPathAvailable())
            GTEST_SKIP() << "multi-segment path not exercised on this host";
    }

    const int kOverCap = NCCL_RMA_MAX_SEGMENTS + 1;
    MultiSegmentVmmBuffer* big = AllocSym(kOverCap, kSegRequestBytes);
    if (SyncSkip(big == nullptr))
        GTEST_SKIP() << "Could not allocate " << kOverCap << " VMM segments";

    void *mh = nullptr, *gh = nullptr;
    ncclResult_t r = RegMr(big->ptr, big->totalSize, &mh, &gh);
    EXPECT_EQ(r, ncclInvalidUsage)
        << "registration of a " << kOverCap << "-segment buffer should be rejected "
        << "with ncclInvalidUsage (cap=" << NCCL_RMA_MAX_SEGMENTS << "), got " << r;
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
                  rma_->iput(rmaCtx_, 0, 0, sendMh, kSize, 0, recvMh, 1, ncclRmaOptFlagsDefault, &req));
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
    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    // If enumeration happened to return identical counts there is no asymmetry
    // to reject (registration succeeds on both ranks); skip rather than misfire.
    // Skip only if every rank succeeded (identical enumeration). any-rank skip hid unilateral success.
    if (MPIHelpers::allRanksTrue(r == ncclSuccess))
        GTEST_SKIP() << "ranks enumerated identical segment counts; no asymmetry";

    EXPECT_EQ(r, ncclInvalidUsage)
        << "asymmetric per-rank segment count must be rejected (rank " << worldRank_
        << " requested " << myNSeg << " segments)";
    EXPECT_EQ(mh, nullptr) << "no MR handle should be produced on rejection";
}

// POSITIVE: equal segment counts with different per-rank boundaries (asymmetric
// windows / mixed GPU+CPU splits). Registration keeps a per-rank segOff table
// and splits the full-window put on each side independently.
TEST_F(RmaMultiSegmentMPITest, RegisterEqualCountDifferentBoundariesTransfer)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* bb =
        worldRank_ == 0 ? AllocDeepEpElastic(4 * kMiB, 2 * kMiB)
                        : AllocDeepEpElastic(2 * kMiB, 4 * kMiB);
    if (SyncSkip(bb == nullptr))
        GTEST_SKIP() << "mixed GPU/CPU VMM allocation unavailable on this host";

    void *mh = nullptr, *gh = nullptr;
    ASSERT_EQ(ncclSuccess, RegMr(bb->ptr, bb->totalSize, &mh, &gh));
    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    constexpr size_t kTotal = 6 * kMiB;
    constexpr uint8_t kSentinel = 0xC3;
    if (worldRank_ == 0)
        FillBuf(bb->ptr, kTotal, /*seed=*/0x5A);
    if (worldRank_ == 1)
        FillSentinel(bb->ptr, kTotal, kSentinel);

    Barrier();
    if (worldRank_ == 0)
    {
        void* req = nullptr;
        ASSERT_EQ(ncclSuccess,
                  rma_->iput(rmaCtx_, 0, 0, mh, kTotal, 0, mh, 1, ncclRmaOptFlagsDefault, &req));
        ASSERT_TRUE(PollUntilDone(req));
    }
    Barrier();

    if (worldRank_ == 1)
        EXPECT_TRUE(VerifyBuf(bb->ptr, kTotal, /*seed=*/0x5A));
}

// NEGATIVE: only rank 0 exceeds the segment cap. Rank 1 successfully registers
// locally, then both ranks must meet in the status collective, reject, and
// clean up. If rank 0 returns early this test hangs in registration.
TEST_F(RmaMultiSegmentMPITest, RankLocalRegistrationFailureRejectedCollectively)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer* probe = AllocSym(2, kSegRequestBytes);
    if (SyncSkip(probe == nullptr))
        GTEST_SKIP() << "multi-segment VMM allocation unavailable on this host";
    void *probeMh = nullptr, *probeGh = nullptr;
    ASSERT_EQ(ncclSuccess,
              RegMr(probe->ptr, probe->totalSize, &probeMh, &probeGh));
    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    const int myNSeg =
        worldRank_ == 0 ? NCCL_RMA_MAX_SEGMENTS + 1 : 2;
    MultiSegmentVmmBuffer* bb = AllocSym(myNSeg, kSegRequestBytes);
    if (SyncSkip(bb == nullptr))
        GTEST_SKIP() << "asymmetric failure layout allocation unavailable";

    void *mh = nullptr, *gh = nullptr;
    const ncclResult_t r = RegMr(bb->ptr, bb->totalSize, &mh, &gh);
    EXPECT_EQ(r, ncclInvalidUsage);
    EXPECT_EQ(mh, nullptr);
}

// NEGATIVE (range guard): out-of-range IPut offsets/sizes must be rejected with
// ncclInvalidArgument and must NOT post anything (recv stays untouched).
TEST_F(RmaMultiSegmentMPITest, IPutOutOfRangeRejectedNoCorruption)
{
    if (!SetUpFixture(2, 2)) return;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb))
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
                  rma_->iput(rmaCtx_, 0, 0, sendMh, total + 1, 0, recvMh, 1, ncclRmaOptFlagsDefault, &req))
            << "oversized transfer must be rejected";
        // valid size but srcOff pushes the source past the end.
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iput(rmaCtx_, 0, /*srcOff=*/1, sendMh, total, 0, recvMh, 1, ncclRmaOptFlagsDefault, &req))
            << "src offset overrun must be rejected";
        // valid size but dstOff pushes the destination past the end.
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iput(rmaCtx_, 0, 0, sendMh, total, /*dstOff=*/1, recvMh, 1, ncclRmaOptFlagsDefault, &req))
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
                  rma_->iget(rmaCtx_, 0, /*remoteOff=*/0, mh, total + 1, 0, mh, 1, ncclRmaOptFlagsDefault, &req))
            << "oversized iget must be rejected";
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iget(rmaCtx_, 0, /*remoteOff=*/1, mh, total, 0, mh, 1, ncclRmaOptFlagsDefault, &req))
            << "remote offset overrun must be rejected";
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iget(rmaCtx_, 0, 0, mh, total, /*localOff=*/1, mh, 1, ncclRmaOptFlagsDefault, &req))
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

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb))
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
                                   /*signalOff=*/0, sigMh, 0, NCCL_NET_SIGNAL_OP_INC,
                                   /*isStrongSignal=*/false, ncclRmaOptFlagsDefault, &req))
            << "oversized iputSignal payload must be rejected";
        // Signal atomic straddles the end of the signal window (signalOff+8 > size).
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iputSignal(rmaCtx_, 0, 0, sendMh, total, 0, recvMh, 1,
                                   /*signalOff=*/kSignalSize - 4, sigMh, 0,
                                   NCCL_NET_SIGNAL_OP_INC, /*isStrongSignal=*/false, ncclRmaOptFlagsDefault, &req))
            << "out-of-range signal offset must be rejected";
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iputSignal(rmaCtx_, 0, 0, sendMh, total, 0, recvMh, 1,
                                   /*signalOff=*/4, sigMh, 0,
                                   NCCL_NET_SIGNAL_OP_INC, /*isStrongSignal=*/false, ncclRmaOptFlagsDefault, &req))
            << "unaligned signal atomic must be rejected";
        // 8-byte aligned offset past the end of the signal window. A
        // cross-segment 8-byte atomic cannot exist on an 8-aligned boundary.
        EXPECT_EQ(ncclInvalidArgument,
                  rma_->iputSignal(rmaCtx_, 0, 0, sendMh, /*size=*/0, 0, recvMh, 1,
                                   /*signalOff=*/kSignalSize, sigMh, 0,
                                   NCCL_NET_SIGNAL_OP_INC, /*isStrongSignal=*/false, ncclRmaOptFlagsDefault, &req))
            << "signal offset at the window end must be rejected";
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
    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb, kWideSegments))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t      total     = sb->totalSize;
    const size_t      seg       = sb->segSize;
    const int         nSeg      = sb->nSegments;
    constexpr uint8_t kSentinel = 0x6F;

    void *sendMh, *sendGh, *recvMh, *recvGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, total, &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, total, &recvMh, &recvGh));

    if (!MultiSegmentPathAvailable())
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
                      rma_->iput(rmaCtx_, 0, off, sendMh, len, off, recvMh, 1, ncclRmaOptFlagsDefault, &req))
                << "iput failed straddling boundary " << k;
            ASSERT_TRUE(PollUntilDone(req)) << "iput stalled at boundary " << k;
        }
        Barrier();

        if (worldRank_ == 1)
        {
            ExpectPayloadIsolated(rb->ptr, total, off, len, seed, kSentinel,
                                  "boundary " + std::to_string(k));
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
                  rma_->iput(rmaCtx_, 0, 0, sendMh, total, 0, recvMh, 1, ncclRmaOptFlagsDefault, &req));
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
    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb, kWideSegments))
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
    if (!MultiSegmentPathAvailable())
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
                                 localOff, dstMh, /*peerRank=*/1, ncclRmaOptFlagsDefault, &req))
                << "iget post failed at iteration " << i;
            ASSERT_TRUE(PollUntilDone(req)) << "iget stalled at iteration " << i;
            ExpectPayloadIsolated(rb->ptr, total, localOff, len, seed,
                                  kSentinel,
                                  "asymmetric IGet iteration " + std::to_string(i));
        }
        Barrier();
    }
}

// Flood 40 in-flight 16-seg iputSignals (~17 WQEs each) without Test().
// Opt-in: RCCL_MSEG_SQ_STRESS=1. Use NCCL_NET=IB.
TEST_F(RmaMultiSegmentMPITest, IPutSignalSendQueueOversubscribe)
{
    const char* stress = std::getenv("RCCL_MSEG_SQ_STRESS");
    const bool want = stress && std::atoi(stress) != 0;
    if (SyncSkip(!want))
        GTEST_SKIP() << "set RCCL_MSEG_SQ_STRESS=1 to run the send-queue stress";
    const char* net = std::getenv("NCCL_NET");
    const bool isCast = net && (strcasecmp(net, "IB-CAST") == 0 || strcasecmp(net, "ib-cast") == 0);
    if (SyncSkip(isCast))
        GTEST_SKIP() << "CAST GIN waits on WR credits; use NCCL_NET=IB";

    if (!SetUpFixture(2, 2)) return;

    constexpr int kSegs = NCCL_RMA_MAX_SEGMENTS;
    constexpr int kInflight = 40;
    constexpr int kWaitMs = 15000;

    MultiSegmentVmmBuffer *sb = nullptr, *rb = nullptr;
    if (!AllocSymPair(&sb, &rb, kSegs))
        GTEST_SKIP() << "Multi-segment VMM allocation unavailable on this host";

    const size_t kSize = sb->totalSize;
    void* sigBuf = AllocBuf(kSignalSize);
    ASSERT_NE(sigBuf, nullptr);

    if (worldRank_ == 0)
        FillBuf(sb->ptr, kSize, /*seed=*/0x51);

    void *sendMh, *sendGh, *recvMh, *recvGh, *sigMh, *sigGh;
    ASSERT_EQ(ncclSuccess, RegMr(sb->ptr, kSize,       &sendMh, &sendGh));
    ASSERT_EQ(ncclSuccess, RegMr(rb->ptr, kSize,       &recvMh, &recvGh));
    ASSERT_EQ(ncclSuccess, RegMr(sigBuf,  kSignalSize, &sigMh,  &sigGh));

    if (!MultiSegmentPathAvailable())
        GTEST_SKIP() << "multi-segment path not exercised on this host";

    Barrier();
    if (worldRank_ == 0)
    {
        void* reqs[kInflight] = {};
        int posted = 0;
        for (int i = 0; i < kInflight; i++)
        {
            ncclResult_t st = rma_->iputSignal(rmaCtx_, 0,
                                               /*srcOff=*/0, sendMh, kSize,
                                               /*dstOff=*/0, recvMh, /*peerRank=*/1,
                                               /*signalOff=*/0, sigMh, /*signalValue=*/0,
                                               NCCL_NET_SIGNAL_OP_INC,
                                               /*isStrongSignal=*/false,
                                               ncclRmaOptFlagsDefault, &reqs[i]);
            EXPECT_EQ(st, ncclSuccess)
                << "iputSignal " << i << " of " << kInflight
                << " failed (SQ oversubscribe / fatal post)";
            if (st != ncclSuccess) break;
            posted = i + 1;
        }
        for (int i = 0; i < posted; i++)
        {
            if (reqs[i] == nullptr) continue;
            EXPECT_TRUE(PollUntilDone(reqs[i], kWaitMs))
                << "iputSignal " << i << " did not complete (SQ oversubscribe)";
        }
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
