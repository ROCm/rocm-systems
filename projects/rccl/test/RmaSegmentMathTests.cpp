/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <climits>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "transport/net_ib/rma_multiseg.h"

namespace {

void FillUniformOffsets(size_t* off, int nSeg, size_t segBytes)
{
    for (int i = 0; i <= nSeg; ++i) off[i] = static_cast<size_t>(i) * segBytes;
}

int CountLayout(const size_t* localOff, int nLocal, const size_t* remoteOff, int nRemote, int maxWr)
{
    const size_t size = localOff[nLocal] < remoteOff[nRemote] ? localOff[nLocal] : remoteOff[nRemote];
    return ncclRmaCountLayoutDataWrs(localOff, nLocal, remoteOff, nRemote, 0, 0, size, maxWr);
}

} // namespace

TEST(RmaSegmentMathTest, SplitsAtVerbsLengthLimitWithoutLargeAllocation)
{
    ASSERT_GT(SIZE_MAX, static_cast<size_t>(UINT32_MAX));
    size_t remaining = static_cast<size_t>(UINT32_MAX) + 1;

    const size_t first =
        ncclRmaSegmentSliceBytes(remaining, remaining, remaining);
    EXPECT_EQ(first, static_cast<size_t>(UINT32_MAX));
    remaining -= first;
    EXPECT_EQ(ncclRmaSegmentSliceBytes(remaining, remaining, remaining),
              size_t{1});
}

TEST(RmaSegmentMathTest, FixedDataWrBudgetRejectsUnrepresentableChain)
{
    ASSERT_GT(SIZE_MAX, static_cast<size_t>(UINT32_MAX));
    const size_t oversize =
        static_cast<size_t>(UINT32_MAX) * (NCCL_RMA_MAX_DATA_WRS + 1ULL);
    const size_t exact =
        static_cast<size_t>(UINT32_MAX) * NCCL_RMA_MAX_DATA_WRS;

    EXPECT_FALSE(ncclRmaDataWrBudgetFull(0, NCCL_RMA_MAX_DATA_WRS));
    EXPECT_TRUE(ncclRmaDataWrBudgetFull(NCCL_RMA_MAX_DATA_WRS, NCCL_RMA_MAX_DATA_WRS));
    EXPECT_EQ(ncclRmaCountPairedDataWrs(0, NCCL_RMA_MAX_DATA_WRS), 0);
    EXPECT_EQ(ncclRmaCountPairedDataWrs(1, NCCL_RMA_MAX_DATA_WRS), 1);
    EXPECT_EQ(ncclRmaCountPairedDataWrs(exact, NCCL_RMA_MAX_DATA_WRS), NCCL_RMA_MAX_DATA_WRS);
    EXPECT_EQ(ncclRmaCountPairedDataWrs(oversize, NCCL_RMA_MAX_DATA_WRS),
              NCCL_RMA_MAX_DATA_WRS + 1);
}

// Aligned 4/8/16-segment 8 GiB windows on matching peers. 16x8 GiB needs 48 WRs.
TEST(RmaSegmentMathTest, HomogeneousSameKindNodesFitDataWrBudget)
{
    ASSERT_GT(SIZE_MAX, static_cast<size_t>(UINT32_MAX));
    const size_t eightGiB = size_t{8} << 30;
    const size_t twoMiB = size_t{2} << 20;
    const int oldBudget = 2 * NCCL_RMA_MAX_SEGMENTS;
    size_t segOff[NCCL_RMA_MAX_SEGMENTS + 1];

    for (int nSeg : {4, 8, NCCL_RMA_MAX_SEGMENTS}) {
        FillUniformOffsets(segOff, nSeg, eightGiB);
        const int wrs = CountLayout(segOff, nSeg, segOff, nSeg, NCCL_RMA_MAX_DATA_WRS);
        EXPECT_EQ(wrs, nSeg * 3) << "aligned 8 GiB same-kind, nSeg=" << nSeg;
        EXPECT_LE(wrs, NCCL_RMA_MAX_DATA_WRS);
        if (nSeg <= 8) {
            EXPECT_LE(wrs, oldBudget) << "4/8 GPU same-kind 8 GiB windows fit 32 WRs";
        } else {
            EXPECT_EQ(CountLayout(segOff, nSeg, segOff, nSeg, oldBudget), oldBudget + 1);
            EXPECT_EQ(wrs, 48);
        }

        FillUniformOffsets(segOff, nSeg, twoMiB);
        EXPECT_EQ(CountLayout(segOff, nSeg, segOff, nSeg, NCCL_RMA_MAX_DATA_WRS), nSeg);
    }
}

TEST(RmaSegmentMathTest, OnlyFinalWrIsSignaled)
{
    constexpr int kWrs = 7;
    for (int i = 0; i < kWrs; ++i)
        EXPECT_EQ(ncclRmaWrIsSignaled(i, kWrs), i == kWrs - 1);
    EXPECT_FALSE(ncclRmaWrIsSignaled(0, 0));
}

TEST(RmaSegmentMathTest, SignalAtomicMustBeAlignedWithinSegment)
{
    EXPECT_TRUE(ncclRmaSignalOffsetValid(/*signalOff=*/8, /*segmentEnd=*/16));
    EXPECT_FALSE(ncclRmaSignalOffsetValid(/*signalOff=*/4, /*segmentEnd=*/16));
    EXPECT_FALSE(ncclRmaSignalOffsetValid(/*signalOff=*/8, /*segmentEnd=*/12));
}

namespace {

struct FakeWr
{
    int id;
    FakeWr* next;
};

int Posted(FakeWr* wr, int nWr, FakeWr* badWr)
{
    return ncclRmaPostedWrCount(wr, nWr, badWr, offsetof(FakeWr, next));
}

} // namespace

// wrap_ibv_post_send fails even after a prefix of the chain is accepted. Only
// the last WR is signaled, so posted==0 frees the request and posted>0 keeps it.
TEST(RmaSegmentMathTest, PrefixPostCountsWrsBeforeBadWr)
{
    FakeWr wr[3];
    wr[0] = {0, &wr[1]};
    wr[1] = {1, &wr[2]};
    wr[2] = {2, nullptr};

    EXPECT_EQ(Posted(&wr[0], 3, nullptr), 3);
    EXPECT_EQ(Posted(&wr[0], 3, &wr[0]), 0);
    EXPECT_EQ(Posted(&wr[0], 3, &wr[1]), 1);
    EXPECT_EQ(Posted(&wr[0], 3, &wr[2]), 2);
    EXPECT_EQ(Posted(&wr[0], 2, nullptr), 2);
    EXPECT_EQ(Posted(&wr[0], 0, &wr[0]), 0);
    EXPECT_EQ(Posted(nullptr, 3, &wr[0]), 0);
}

TEST(RmaSegmentMathTest, PrefixPostKeepsRequestOnlyWhenSomethingPosted)
{
    int keep = 0, failed = 0;
    EXPECT_EQ(ncclRmaCompletePostedRequest(ncclSystemError, /*posted=*/1, /*nWr=*/3, &keep, &failed), ncclSuccess);
    EXPECT_TRUE(keep);
    EXPECT_TRUE(failed);
    EXPECT_EQ(ncclRmaCompletePostedRequest(ncclSystemError, /*posted=*/0, /*nWr=*/3, &keep, &failed), ncclSystemError);
    EXPECT_FALSE(keep);
    EXPECT_FALSE(failed);
    EXPECT_EQ(ncclRmaCompletePostedRequest(ncclSuccess, /*posted=*/0, /*nWr=*/0, &keep, &failed), ncclSuccess);
    EXPECT_TRUE(keep);
    EXPECT_FALSE(failed);
    EXPECT_EQ(ncclRmaCompletePostedRequest(ncclSuccess, /*posted=*/2, /*nWr=*/2, &keep, &failed), ncclSuccess);
    EXPECT_TRUE(keep);
    EXPECT_FALSE(failed);
    EXPECT_TRUE(ncclRmaPrefixPostLostSignaledTail(/*posted=*/1, /*nWr=*/3));
    EXPECT_FALSE(ncclRmaPrefixPostLostSignaledTail(/*posted=*/3, /*nWr=*/3));
    EXPECT_FALSE(ncclRmaPrefixPostLostSignaledTail(/*posted=*/0, /*nWr=*/3));
    EXPECT_FALSE(ncclRmaPrefixPostLostSignaledTail(/*posted=*/0, /*nWr=*/0));
}

// A freed id must not leave a gap the next request would wait on, nor be
// reused while a later request holds a higher id.
TEST(RmaSegmentMathTest, ReleaseSeqIdOnlyReturnsNewestId)
{
    uint64_t posted = 5;
    ncclRmaReleaseSeqId(/*id=*/5, &posted);
    EXPECT_EQ(posted, uint64_t{4});

    posted = 7;
    ncclRmaReleaseSeqId(/*id=*/6, &posted);
    EXPECT_EQ(posted, uint64_t{7}) << "a later request already holds id 7";
}

// A failed handle calloc must still reach the status AllGather. memcpy of
// segOff is skipped until the handle exists.
TEST(RmaSegmentMathTest, FailedHandleCallocDoesNotCopySegmentOffsets)
{
    EXPECT_FALSE(ncclRmaRegistrationHandleReady(nullptr, 1));
    EXPECT_FALSE(ncclRmaRegistrationHandleReady(nullptr, NCCL_RMA_MAX_SEGMENTS));
    char handle{};
    EXPECT_FALSE(ncclRmaRegistrationHandleReady(&handle, 0));
    EXPECT_FALSE(ncclRmaRegistrationHandleReady(&handle, NCCL_RMA_MAX_SEGMENTS + 1));
    EXPECT_TRUE(ncclRmaRegistrationHandleReady(&handle, 1));
    EXPECT_TRUE(ncclRmaRegistrationHandleReady(&handle, NCCL_RMA_MAX_SEGMENTS));
}

TEST(RmaSegmentMathTest, SegmentCountsMatchIgnoresBoundaries)
{
    EXPECT_TRUE(ncclRmaSegmentCountsMatch(2, 2));
    EXPECT_TRUE(ncclRmaSegmentCountsMatch(NCCL_RMA_MAX_SEGMENTS, NCCL_RMA_MAX_SEGMENTS));
    EXPECT_FALSE(ncclRmaSegmentCountsMatch(2, 1));
    EXPECT_FALSE(ncclRmaSegmentCountsMatch(0, 0));
    EXPECT_FALSE(ncclRmaSegmentCountsMatch(NCCL_RMA_MAX_SEGMENTS + 1, NCCL_RMA_MAX_SEGMENTS + 1));
}

TEST(RmaSegmentMathTest, SegOffTableValidRequiresZeroStartAndOrder)
{
    const size_t single[] = {0, 4096};
    const size_t multi[]  = {0, 2097152, 4194304, 4194404};
    const size_t empty[]  = {0, 0};
    const size_t repeat[] = {0, 4096, 4096, 8192};
    EXPECT_TRUE(ncclRmaSegOffTableValid(single, 1));
    EXPECT_TRUE(ncclRmaSegOffTableValid(multi, 3));
    EXPECT_TRUE(ncclRmaSegOffTableValid(empty, 1));
    EXPECT_TRUE(ncclRmaSegOffTableValid(repeat, 3));

    const size_t nonZeroStart[] = {64, 4096};
    const size_t decreasing[]   = {0, 8192, 4096, 12288};
    const size_t lastWraps[]    = {0, 4096, 8192, 100};
    EXPECT_FALSE(ncclRmaSegOffTableValid(nonZeroStart, 1));
    EXPECT_FALSE(ncclRmaSegOffTableValid(decreasing, 3));
    EXPECT_FALSE(ncclRmaSegOffTableValid(lastWraps, 3));
    EXPECT_TRUE(ncclRmaSegOffTableValid(lastWraps, 2)) << "entries past nSegments are ignored";

    EXPECT_FALSE(ncclRmaSegOffTableValid(nullptr, 1));
    EXPECT_FALSE(ncclRmaSegOffTableValid(single, 0));
    EXPECT_FALSE(ncclRmaSegOffTableValid(multi, NCCL_RMA_MAX_SEGMENTS + 1));
}

TEST(RmaSegmentMathTest, PeerSegOffIndexesPerRankTable)
{
    size_t local[] = {0, 100};
    size_t table[(NCCL_RMA_MAX_SEGMENTS + 1) * 2] = {};
    table[1] = 4096;
    table[NCCL_RMA_MAX_SEGMENTS + 2] = 8192;
    EXPECT_EQ(ncclRmaPeerSegOff(nullptr, local, 0), local);
    EXPECT_EQ(ncclRmaPeerSegOff(table, local, -1), local);
    EXPECT_EQ(ncclRmaPeerSegOff(table, local, 0)[1], size_t{4096});
    EXPECT_EQ(ncclRmaPeerSegOff(table, local, 1)[1], size_t{8192});
}

// ncclRmaNextSlice is the step gin.cc's WR builder takes; walk it like the builder does.
TEST(RmaSegmentMathTest, NextSliceSplitsAtBothSidesBoundaries)
{
    const size_t local[] = {0, 100, 250};
    const size_t remote[] = {0, 60, 250};
    const size_t expectBytes[] = {60, 40, 150};
    const int expectLocal[] = {0, 0, 1};
    const int expectRemote[] = {0, 1, 1};
    uint64_t lOff = 0, rOff = 0;
    size_t rem = 250;
    for (int i = 0; i < 3; ++i) {
        int ls = -1, rs = -1;
        const size_t chunk = ncclRmaNextSlice(local, 2, remote, 2, lOff, rOff, rem, &ls, &rs);
        EXPECT_EQ(chunk, expectBytes[i]) << "slice " << i;
        EXPECT_EQ(ls, expectLocal[i]) << "slice " << i;
        EXPECT_EQ(rs, expectRemote[i]) << "slice " << i;
        lOff += chunk;
        rOff += chunk;
        rem -= chunk;
    }
    EXPECT_EQ(rem, size_t{0});
    EXPECT_EQ(ncclRmaCountLayoutDataWrs(local, 2, remote, 2, 0, 0, 250, NCCL_RMA_MAX_DATA_WRS), 3);
}

TEST(RmaSegmentMathTest, NextSliceFlushModeStepsOneRemoteSegment)
{
    const size_t remote[] = {0, 4096, 4096 + 10, 3 * 4096};
    const size_t expectBytes[] = {4096, 10, 2 * 4096 - 10};
    uint64_t rOff = 0;
    size_t rem = remote[3];
    for (int i = 0; i < 3; ++i) {
        int ls = -1, rs = -1;
        const size_t chunk = ncclRmaNextSlice(nullptr, 0, remote, 3, 0, rOff, rem, &ls, &rs);
        EXPECT_EQ(chunk, expectBytes[i]) << "segment " << i;
        EXPECT_EQ(ls, 0);
        EXPECT_EQ(rs, i);
        rOff += chunk;
        rem -= chunk;
    }
    EXPECT_EQ(ncclRmaCountLayoutDataWrs(nullptr, 0, remote, 3, 0, 0, remote[3], NCCL_RMA_MAX_FLUSH_WRS), 3);
}

TEST(RmaSegmentMathTest, NextSliceStartsMidSegmentAndCapsAtSgeLength)
{
    ASSERT_GT(SIZE_MAX, static_cast<size_t>(UINT32_MAX));
    const size_t big = static_cast<size_t>(UINT32_MAX) + 4096;
    const size_t local[] = {0, big};
    const size_t remote[] = {0, 512, big};
    int ls = -1, rs = -1;
    EXPECT_EQ(ncclRmaNextSlice(local, 1, remote, 2, 100, 100, 1000, &ls, &rs), size_t{412});
    EXPECT_EQ(rs, 0);
    EXPECT_EQ(ncclRmaNextSlice(local, 1, remote, 2, 512, 512, big - 512, &ls, &rs), static_cast<size_t>(UINT32_MAX));
    EXPECT_EQ(rs, 1);
}

// A zero slice would spin the builder forever; it must report past-the-end instead.
TEST(RmaSegmentMathTest, NextSliceReturnsZeroPastEitherTable)
{
    const size_t local[] = {0, 100};
    const size_t remote[] = {0, 200};
    int ls = -1, rs = -1;
    EXPECT_EQ(ncclRmaNextSlice(local, 1, remote, 1, 100, 0, 50, &ls, &rs), size_t{0});
    EXPECT_EQ(ncclRmaNextSlice(local, 1, remote, 1, 0, 200, 50, &ls, &rs), size_t{0});
    EXPECT_EQ(ncclRmaNextSlice(nullptr, 0, remote, 1, 0, 250, 50, &ls, &rs), size_t{0});
    EXPECT_EQ(ncclRmaCountLayoutDataWrs(local, 1, remote, 1, 0, 0, 150, NCCL_RMA_MAX_DATA_WRS),
              NCCL_RMA_MAX_DATA_WRS + 1);
}

// Asymmetric windows: three local segments against one remote segment is three slices.
TEST(RmaSegmentMathTest, NextSliceLocalHasMoreSegmentsThanRemote)
{
    const size_t local[] = {0, 50, 120, 300};
    const size_t remote[] = {0, 300};
    const size_t expectBytes[] = {50, 70, 180};
    uint64_t lOff = 0, rOff = 0;
    size_t rem = 300;
    for (int i = 0; i < 3; ++i) {
        int ls = -1, rs = -1;
        const size_t chunk = ncclRmaNextSlice(local, 3, remote, 1, lOff, rOff, rem, &ls, &rs);
        EXPECT_EQ(chunk, expectBytes[i]) << "slice " << i;
        EXPECT_EQ(ls, i) << "slice " << i;
        EXPECT_EQ(rs, 0) << "slice " << i;
        lOff += chunk;
        rOff += chunk;
        rem -= chunk;
    }
    EXPECT_EQ(rem, size_t{0});
    EXPECT_EQ(ncclRmaCountLayoutDataWrs(local, 3, remote, 1, 0, 0, 300, NCCL_RMA_MAX_DATA_WRS), 3);
}

TEST(RmaSegmentMathTest, WrCreditsAvailableRespectsCapacity)
{
    EXPECT_TRUE(ncclRmaWrCreditsAvailable(0, NCCL_RMA_MAX_SIGNAL_WRS, NCCL_RMA_SEND_QP_WRS));
    EXPECT_TRUE(ncclRmaWrCreditsAvailable(NCCL_RMA_SEND_QP_WRS - 2, 2, NCCL_RMA_SEND_QP_WRS));
    EXPECT_FALSE(ncclRmaWrCreditsAvailable(NCCL_RMA_SEND_QP_WRS - 1, 2, NCCL_RMA_SEND_QP_WRS));
    EXPECT_FALSE(ncclRmaWrCreditsAvailable(0, NCCL_RMA_SEND_QP_WRS + 1, NCCL_RMA_SEND_QP_WRS));
    EXPECT_FALSE(ncclRmaWrCreditsAvailable(-1, 1, NCCL_RMA_SEND_QP_WRS));
}

// Every full signal chain that fits the sender QP is posted; the next one waits.
TEST(RmaSegmentMathTest, WrCreditsBlockChainPastQpDepth)
{
    ncclRmaWrCredits c = {};
    uint64_t posted = 0;
    const int chain = NCCL_RMA_MAX_SIGNAL_WRS;
    const int depth = NCCL_RMA_SEND_QP_WRS;
    const int nFit = depth / chain;
    ASSERT_GT(nFit, 0);
    for (int i = 0; i < nFit; ++i) {
        ASSERT_TRUE(ncclRmaWrCreditsAvailable(ncclRmaWrCreditsInUse(&c, posted, 2), chain, depth))
            << "chain " << i;
        ncclRmaWrCreditsRecord(&c, ++posted, chain);
    }
    EXPECT_EQ(ncclRmaWrCreditsInUse(&c, posted, 2), nFit * chain);
    EXPECT_FALSE(ncclRmaWrCreditsAvailable(ncclRmaWrCreditsInUse(&c, posted, 2), chain, depth));
    ncclRmaWrCreditsRelease(&c, 1);
    EXPECT_EQ(ncclRmaWrCreditsInUse(&c, posted, 2), (nFit - 1) * chain);
    EXPECT_TRUE(ncclRmaWrCreditsAvailable(ncclRmaWrCreditsInUse(&c, posted, 2), chain, depth));
}

// Ids without an entry are single-MR ops; they cost untrackedCost until released.
TEST(RmaSegmentMathTest, WrCreditsMixTrackedAndUntrackedIds)
{
    ncclRmaWrCredits c = {};
    uint64_t posted = 0;
    ++posted;                                  // id 1: untracked
    ncclRmaWrCreditsRecord(&c, ++posted, 40);  // id 2: 40-WR chain
    ++posted;                                  // id 3: untracked
    EXPECT_EQ(ncclRmaWrCreditsInUse(&c, posted, 2), 40 + 2 * 2);
    ncclRmaWrCreditsRelease(&c, 2);
    EXPECT_EQ(c.trackedIds, 0);
    EXPECT_EQ(ncclRmaWrCreditsInUse(&c, posted, 2), 2);
    ncclRmaWrCreditsRelease(&c, 3);
    EXPECT_EQ(ncclRmaWrCreditsInUse(&c, posted, 2), 0);
    ncclRmaWrCreditsRelease(&c, 2);
    EXPECT_EQ(c.releasedSeq, uint64_t{3}) << "release never moves backwards";
}

TEST(RmaSegmentMathTest, WrCreditsRingReusesSlotsAcrossWrap)
{
    ncclRmaWrCredits c = {};
    uint64_t posted = 0;
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < NCCL_RMA_CREDIT_IDS; ++i) ncclRmaWrCreditsRecord(&c, ++posted, 3);
        EXPECT_EQ(c.trackedIds, NCCL_RMA_CREDIT_IDS);
        EXPECT_EQ(ncclRmaWrCreditsInUse(&c, posted, 2), 3 * NCCL_RMA_CREDIT_IDS);
        ncclRmaWrCreditsRelease(&c, posted);
        EXPECT_EQ(c.trackedIds, 0);
        EXPECT_EQ(c.trackedWrs, 0);
    }
    ncclRmaWrCreditsRecord(&c, c.releasedSeq, 5);
    EXPECT_EQ(c.trackedIds, 0) << "an already-released id is not recorded";
    ncclRmaWrCreditsRecord(&c, ++posted, 0);
    EXPECT_EQ(c.trackedIds, 0) << "a zero-WR request is not recorded";
}

// A second record of the same id replaces its length. It must not add another tracked id.
TEST(RmaSegmentMathTest, WrCreditsRecordReplacesExistingId)
{
    ncclRmaWrCredits c = {};
    uint64_t posted = 0;
    ncclRmaWrCreditsRecord(&c, ++posted, 10);
    ncclRmaWrCreditsRecord(&c, posted, 25);
    EXPECT_EQ(c.trackedIds, 1);
    EXPECT_EQ(c.trackedWrs, 25);
    EXPECT_EQ(ncclRmaWrCreditsInUse(&c, posted, 2), 25);
    ncclRmaWrCreditsRelease(&c, posted);
    EXPECT_EQ(c.trackedIds, 0);
    EXPECT_EQ(c.trackedWrs, 0);
    EXPECT_EQ(ncclRmaWrCreditsInUse(&c, posted, 2), 0);
}

// One completion can advance completedSeq across several ids, tracked and empty.
TEST(RmaSegmentMathTest, WrCreditsReleaseJumpsSeveralIds)
{
    ncclRmaWrCredits c = {};
    uint64_t posted = 0;
    ncclRmaWrCreditsRecord(&c, ++posted, 4);  // id 1
    ++posted;                                 // id 2: untracked hole
    ncclRmaWrCreditsRecord(&c, ++posted, 7);  // id 3
    ++posted;                                 // id 4: untracked hole
    ncclRmaWrCreditsRecord(&c, ++posted, 9);  // id 5
    EXPECT_EQ(ncclRmaWrCreditsInUse(&c, posted, 1), 4 + 7 + 9 + 2);
    ncclRmaWrCreditsRelease(&c, 4);
    EXPECT_EQ(c.releasedSeq, uint64_t{4});
    EXPECT_EQ(c.trackedIds, 1);
    EXPECT_EQ(c.trackedWrs, 9);
    // The one live id is tracked, so the untracked cost does not apply.
    EXPECT_EQ(ncclRmaWrCreditsInUse(&c, posted, 1), 9);
}

TEST(RmaSegmentMathTest, LayoutsMatchRequiresEqualBoundaries)
{
    const size_t lhs[] = {0, 4096, 8192};
    const size_t rhs[] = {0, 4096, 8192};
    const size_t shorter[] = {0, 4096};
    const size_t shifted[] = {0, 2048, 8192};
    EXPECT_TRUE(ncclRmaLayoutsMatch(2, lhs, 2, rhs));
    EXPECT_FALSE(ncclRmaLayoutsMatch(2, lhs, 1, shorter));
    EXPECT_FALSE(ncclRmaLayoutsMatch(2, lhs, 2, shifted));
    EXPECT_FALSE(ncclRmaLayoutsMatch(0, lhs, 0, lhs));
    EXPECT_FALSE(ncclRmaLayoutsMatch(NCCL_RMA_MAX_SEGMENTS + 1, lhs, NCCL_RMA_MAX_SEGMENTS + 1, lhs));
}

TEST(RmaSegmentMathTest, SegIndexOfUsesHalfOpenOffsets)
{
    const size_t off[] = {0, 4096, 8192};
    EXPECT_EQ(ncclRmaSegIndexOf(2, off, 0), 0);
    EXPECT_EQ(ncclRmaSegIndexOf(2, off, 4095), 0);
    EXPECT_EQ(ncclRmaSegIndexOf(2, off, 4096), 1);
    EXPECT_EQ(ncclRmaSegIndexOf(2, off, 8192), 1);
    EXPECT_EQ(ncclRmaSegIndexOf(0, off, 0), 0);
}

TEST(RmaSegmentMathTest, OffsetRangeOkRejectsPastEnd)
{
    const size_t off[] = {0, 4096, 8192};
    EXPECT_TRUE(ncclRmaOffsetRangeOk(2, off, 0, 8192));
    EXPECT_TRUE(ncclRmaOffsetRangeOk(2, off, 4096, 4096));
    EXPECT_FALSE(ncclRmaOffsetRangeOk(2, off, 4096, 4097));
    EXPECT_FALSE(ncclRmaOffsetRangeOk(2, off, 8192, 1));
    EXPECT_FALSE(ncclRmaOffsetRangeOk(0, off, 0, 0));
}

TEST(RmaSegmentMathTest, TranscriptHeaderRejectsWrongMagicOrVersion)
{
    EXPECT_TRUE(ncclRmaTranscriptHeaderOk(0x524d5347u, 1u, 0x524d5347u, 1u));
    EXPECT_FALSE(ncclRmaTranscriptHeaderOk(0u, 1u, 0x524d5347u, 1u));
    EXPECT_FALSE(ncclRmaTranscriptHeaderOk(0x524d5347u, 2u, 0x524d5347u, 1u));
}

TEST(RmaSegmentMathTest, SegmentCapMatchesClassicIb)
{
    EXPECT_EQ(NCCL_RMA_MAX_SEGMENTS, NCCL_IB_MAX_SEGMENTS);
}

namespace {

// Matches NCCL_IB_MAX_DEVS_PER_NIC / NCCL_NET_MAX_DEVS_PER_NIC_V12.
constexpr int kMaxDevsPerNic = 8;

void FillUnusedSlots(int* slots, int filled)
{
    for (int d = filled; d < kMaxDevsPerNic; ++d)
        slots[d] = -1;
}

} // namespace

// Registration is recvComm slot order {physical 7, physical 3}. The QP used
// for the post has remDevIdx=1, but remDevs[1] is physical 7. Indexing rkeys
// by remDevIdx would select the MR for physical 3.
TEST(RmaSegmentMathTest, DevSlotFollowsPhysicalIbDevNNotQpRemDevIdx)
{
    int ibDevNs[kMaxDevsPerNic];
    ibDevNs[0] = 7;
    ibDevNs[1] = 3;
    FillUnusedSlots(ibDevNs, 2);

    constexpr int remDevIdx = 1;
    constexpr int remoteIbDevN = 7;
    EXPECT_EQ(ncclRmaDevSlotOf(ibDevNs, /*rank=*/0, remoteIbDevN, kMaxDevsPerNic), 0);
    EXPECT_NE(ncclRmaDevSlotOf(ibDevNs, /*rank=*/0, remoteIbDevN, kMaxDevsPerNic), remDevIdx);
}

// Data WRs look up the remote QP's physical device. Flush RDMA_READs look up
// the local QP's physical device in the same remote handle.
TEST(RmaSegmentMathTest, FlushLooksUpLocalIbDevNInRemoteRegistration)
{
    int remoteIbDevNs[kMaxDevsPerNic];
    remoteIbDevNs[0] = 7;
    remoteIbDevNs[1] = 3;
    FillUnusedSlots(remoteIbDevNs, 2);

    constexpr int remoteRank = 0;
    constexpr int remoteIbDevN = 3;
    constexpr int localIbDevN = 7;
    EXPECT_EQ(ncclRmaRkeyIbDevN(localIbDevN, remoteIbDevN, /*flushMode=*/0), remoteIbDevN);
    EXPECT_EQ(ncclRmaRkeyIbDevN(localIbDevN, remoteIbDevN, /*flushMode=*/1), localIbDevN);
    EXPECT_EQ(ncclRmaDevSlotOf(remoteIbDevNs, remoteRank, remoteIbDevN, kMaxDevsPerNic), 1);
    EXPECT_EQ(ncclRmaDevSlotOf(remoteIbDevNs, remoteRank, localIbDevN, kMaxDevsPerNic), 0);
}

// Signal atomics use remDevs[qp->remDevIdx].ibv_dev_index, not remDevIdx.
TEST(RmaSegmentMathTest, SignalLooksUpRemotePhysicalDeviceOnHandle)
{
    int signalIbDevNs[kMaxDevsPerNic];
    signalIbDevNs[0] = 4;
    FillUnusedSlots(signalIbDevNs, 1);

    constexpr int remDevIdx = 1;
    constexpr int remDevPhysical = 4;
    EXPECT_EQ(ncclRmaDevSlotOf(signalIbDevNs, /*rank=*/0, remDevPhysical, kMaxDevsPerNic), 0);
    EXPECT_EQ(ncclRmaDevSlotOf(signalIbDevNs, /*rank=*/0, remDevIdx, kMaxDevsPerNic), -1);
}

TEST(RmaSegmentMathTest, DevSlotIsRankStrided)
{
    int ibDevNs[2 * kMaxDevsPerNic];
    for (int i = 0; i < 2 * kMaxDevsPerNic; ++i)
        ibDevNs[i] = -1;
    ibDevNs[0] = 2;
    ibDevNs[kMaxDevsPerNic] = 9;
    ibDevNs[kMaxDevsPerNic + 1] = 2;

    EXPECT_EQ(ncclRmaDevSlotOf(ibDevNs, /*rank=*/0, /*ibDevN=*/2, kMaxDevsPerNic), 0);
    EXPECT_EQ(ncclRmaDevSlotOf(ibDevNs, /*rank=*/1, /*ibDevN=*/2, kMaxDevsPerNic), 1);
    EXPECT_EQ(ncclRmaDevSlotOf(ibDevNs, /*rank=*/1, /*ibDevN=*/9, kMaxDevsPerNic), 0);
    EXPECT_EQ(ncclRmaDevSlotOf(ibDevNs, /*rank=*/0, /*ibDevN=*/9, kMaxDevsPerNic), -1);
}

TEST(RmaSegmentMathTest, DevSlotRejectsMissingAndInvalidDevices)
{
    int ibDevNs[kMaxDevsPerNic];
    ibDevNs[0] = 0;
    FillUnusedSlots(ibDevNs, 1);

    EXPECT_EQ(ncclRmaDevSlotOf(ibDevNs, /*rank=*/0, /*ibDevN=*/0, kMaxDevsPerNic), 0);
    EXPECT_EQ(ncclRmaDevSlotOf(ibDevNs, /*rank=*/0, /*ibDevN=*/1, kMaxDevsPerNic), -1);
    EXPECT_EQ(ncclRmaDevSlotOf(ibDevNs, /*rank=*/0, /*ibDevN=*/-1, kMaxDevsPerNic), -1);
    EXPECT_EQ(ncclRmaDevSlotOf(nullptr, /*rank=*/0, /*ibDevN=*/0, kMaxDevsPerNic), -1);
    EXPECT_EQ(ncclRmaDevSlotOf(ibDevNs, /*rank=*/-1, /*ibDevN=*/0, kMaxDevsPerNic), -1);
    EXPECT_EQ(ncclRmaDevSlotOf(ibDevNs, /*rank=*/0, /*ibDevN=*/0, /*maxDevsPerNic=*/0), -1);
}
