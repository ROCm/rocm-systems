/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <climits>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include "../src/transport/net_ib/gin.h"

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

// nranks>64 overlays compact consensus on the unused registration stack.
TEST(RmaSegmentMathTest, CompactConsensusOverlaysRegistrationStackWhenHeapFails)
{
    char heap{};
    char stack[64 * 128];
    EXPECT_EQ(ncclRmaCompactConsensusRecv(&heap, stack, sizeof(stack), 65, sizeof(int)), &heap);
    EXPECT_EQ(ncclRmaCompactConsensusRecv(nullptr, stack, sizeof(stack), 65, sizeof(int)), stack);
    EXPECT_EQ(ncclRmaCompactConsensusRecv(nullptr, stack, sizeof(stack), 65, sizeof(ncclResult_t)), stack);
    EXPECT_EQ(ncclRmaCompactConsensusRecv(nullptr, stack, 64 * sizeof(int), 65, sizeof(int)), nullptr);
    EXPECT_EQ(ncclRmaCompactConsensusRecv(nullptr, stack, sizeof(stack), 0, sizeof(int)), nullptr);
    EXPECT_EQ(ncclRmaCompactConsensusRecv(nullptr, nullptr, sizeof(stack), 65, sizeof(int)), nullptr);
}

TEST(RmaSegmentMathTest, SegmentCountsMatchIgnoresBoundaries)
{
    EXPECT_TRUE(ncclRmaSegmentCountsMatch(2, 2));
    EXPECT_TRUE(ncclRmaSegmentCountsMatch(NCCL_RMA_MAX_SEGMENTS, NCCL_RMA_MAX_SEGMENTS));
    EXPECT_FALSE(ncclRmaSegmentCountsMatch(2, 1));
    EXPECT_FALSE(ncclRmaSegmentCountsMatch(0, 0));
    EXPECT_FALSE(ncclRmaSegmentCountsMatch(NCCL_RMA_MAX_SEGMENTS + 1, NCCL_RMA_MAX_SEGMENTS + 1));
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
