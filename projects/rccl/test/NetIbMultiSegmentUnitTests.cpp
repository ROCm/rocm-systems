/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only unit tests for IB-MSEG's multi-segment CTS protocol math in src/transport/net_ib_multiseg/seg_p2p_mseg.h.
// They run in rccl-UnitTestsFixtures without IB hardware, a GPU, or MPI.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "transport/net_ib_multiseg/seg_p2p_mseg.h"

namespace {

struct Layout {
    std::vector<uintptr_t> start;
    std::vector<size_t>    len;
    int n() const { return static_cast<int>(start.size()); }
};

Layout MakeUniform(uintptr_t base, size_t seg, int nSeg) {
    Layout L;
    for (int s = 0; s < nSeg; s++) { L.start.push_back(base + (uintptr_t)s * seg); L.len.push_back(seg); }
    return L;
}

int SegOf(const Layout& L, uintptr_t addr, size_t len) {
    return IbMsegSegmentIndexForRange(L.n(), L.start.data(), L.len.data(), addr, len);
}

// The segOff[nSeg+1] table (off[nSeg] == total bytes) the zero-length lookup takes, plus segment VAs.
struct SplitTables {
    std::vector<uint64_t> va;
    std::vector<uint64_t> off;
    int n() const { return static_cast<int>(va.size()); }
};

SplitTables MakeTables(const Layout& L) {
    SplitTables T;
    uint64_t cum = 0;
    for (int s = 0; s < L.n(); s++) { T.va.push_back(L.start[s]); T.off.push_back(cum); cum += L.len[s]; }
    T.off.push_back(cum);
    return T;
}

constexpr uintptr_t kBase = 0x100000000ULL;
constexpr size_t    kSeg  = 2u * 1024 * 1024;

} // namespace

// === Segment selection =====================================================

TEST(NetIbMultiSeg, StartOfEachSegmentMapsToThatSegment) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    for (int s = 0; s < 4; s++)
        EXPECT_EQ(SegOf(L, kBase + (uintptr_t)s * kSeg, 4096), s) << "segment " << s;
}

TEST(NetIbMultiSeg, RangeCrossingBoundaryRejected) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    EXPECT_EQ(SegOf(L, kBase + kSeg - 1, 2), -1);
    EXPECT_EQ(SegOf(L, kBase + kSeg / 2, 2 * kSeg), -1);
}

TEST(NetIbMultiSeg, AddressLengthOverflowRejected) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    EXPECT_EQ(SegOf(L, kBase + 16, SIZE_MAX), -1);
}

TEST(NetIbMultiSeg, ZeroLengthWrUsesContainingSegment) {
    SplitTables t = MakeTables(MakeUniform(kBase, kSeg, 4));
    EXPECT_EQ(IbMsegSegmentForZeroLengthOffset(t.n(), t.off.data(), 0), 0);
    EXPECT_EQ(IbMsegSegmentForZeroLengthOffset(t.n(), t.off.data(), kSeg), 1);
    EXPECT_EQ(IbMsegSegmentForZeroLengthOffset(t.n(), t.off.data(), 2 * kSeg + 64), 2);
}

TEST(NetIbMultiSeg, ZeroLengthWrAtLastExclusiveEndUsesLastMr) {
    SplitTables t = MakeTables(MakeUniform(kBase, kSeg, 4));
    EXPECT_EQ(IbMsegSegmentForZeroLengthOffset(t.n(), t.off.data(), 4 * kSeg), 3);
    EXPECT_EQ(IbMsegSegmentForZeroLengthOffset(t.n(), t.off.data(), 4 * kSeg + 1), -1);
}

TEST(NetIbMultiSeg, ZeroLengthWrUnequalSegments) {
    constexpr uint64_t gpuBytes = 6u * 1024 * 1024;
    Layout L{{kBase, kBase + gpuBytes}, {gpuBytes, kSeg}};
    SplitTables t = MakeTables(L);
    EXPECT_EQ(IbMsegSegmentForZeroLengthOffset(t.n(), t.off.data(), gpuBytes - 1), 0);
    EXPECT_EQ(IbMsegSegmentForZeroLengthOffset(t.n(), t.off.data(), gpuBytes), 1);
    EXPECT_EQ(IbMsegSegmentForZeroLengthOffset(t.n(), t.off.data(), gpuBytes + kSeg), 1);
}

TEST(NetIbMultiSeg, ZeroLengthWrEmptyLayout) {
    const uint64_t off[] = {0};
    EXPECT_EQ(IbMsegSegmentForZeroLengthOffset(0, off, 0), -1);
}

TEST(NetIbCtsLayout, AcceptsMonotonicNonzeroRkeys) {
    const uint64_t start[] = {kBase, kBase + kSeg, kBase + 2 * kSeg};
    const uint32_t rkeys[] = {1u, 2u, 3u};
    EXPECT_TRUE(IbMsegCtsRemoteLayoutValid(3, /*remDevIdx=*/0, /*maxDevs=*/2, RCCL_IB_REG_SEG_DEFAULT, start, rkeys));
}

TEST(NetIbCtsLayout, RejectsZeroSegmentCount) {
    const uint64_t start[] = {kBase};
    const uint32_t rkeys[] = {1u};
    EXPECT_FALSE(IbMsegCtsRemoteLayoutValid(0, 0, 2, RCCL_IB_REG_SEG_DEFAULT, start, rkeys));
}

TEST(NetIbCtsLayout, RejectsSegmentCountAboveCap) {
    const uint64_t start[] = {kBase};
    const uint32_t rkeys[] = {1u};
    EXPECT_FALSE(IbMsegCtsRemoteLayoutValid(RCCL_IB_REG_SEG_DEFAULT + 1, 0, 2, RCCL_IB_REG_SEG_DEFAULT, start, rkeys));
}

TEST(NetIbCtsLayout, RejectsOutOfRangeDeviceIndex) {
    const uint64_t start[] = {kBase, kBase + kSeg};
    const uint32_t rkeys[] = {1u, 2u};
    EXPECT_FALSE(IbMsegCtsRemoteLayoutValid(2, /*remDevIdx=*/-1, 2, RCCL_IB_REG_SEG_DEFAULT, start, rkeys));
    EXPECT_FALSE(IbMsegCtsRemoteLayoutValid(2, /*remDevIdx=*/2, 2, RCCL_IB_REG_SEG_DEFAULT, start, rkeys));
}

TEST(NetIbCtsLayout, RejectsNonMonotonicSegStart) {
    const uint64_t start[] = {kBase, kBase, kBase + 2 * kSeg};
    const uint32_t rkeys[] = {1u, 2u, 3u};
    EXPECT_FALSE(IbMsegCtsRemoteLayoutValid(3, 0, 2, RCCL_IB_REG_SEG_DEFAULT, start, rkeys));
}

TEST(NetIbCtsLayout, RejectsZeroRkey) {
    const uint64_t start[] = {kBase, kBase + kSeg};
    const uint32_t rkeys[] = {1u, 0u};
    EXPECT_FALSE(IbMsegCtsRemoteLayoutValid(2, 0, 2, RCCL_IB_REG_SEG_DEFAULT, start, rkeys));
}

TEST(NetIbCtsLayout, RejectsNullTables) {
    const uint64_t start[] = {kBase, kBase + kSeg};
    const uint32_t rkeys[] = {1u, 2u};
    EXPECT_FALSE(IbMsegCtsRemoteLayoutValid(2, 0, 2, RCCL_IB_REG_SEG_DEFAULT, nullptr, rkeys));
    EXPECT_FALSE(IbMsegCtsRemoteLayoutValid(2, 0, 2, RCCL_IB_REG_SEG_DEFAULT, start, nullptr));
}

TEST(NetIbCtsLayout, AcceptsSegmentCountAtCap) {
    std::vector<uint64_t> start(RCCL_IB_REG_SEG_DEFAULT);
    std::vector<uint32_t> rkeys(RCCL_IB_REG_SEG_DEFAULT, 1u);
    for (int s = 0; s < RCCL_IB_REG_SEG_DEFAULT; s++) start[s] = kBase + (uint64_t)s * kSeg;
    EXPECT_TRUE(
        IbMsegCtsRemoteLayoutValid(RCCL_IB_REG_SEG_DEFAULT, 0, 2, RCCL_IB_REG_SEG_DEFAULT, start.data(), rkeys.data()));
}

TEST(NetIbCtsLayout, AcceptsLastDeviceIndex) {
    const uint64_t start[] = {kBase, kBase + kSeg};
    const uint32_t rkeys[] = {1u, 2u};
    EXPECT_TRUE(IbMsegCtsRemoteLayoutValid(2, /*remDevIdx=*/1, /*maxDevs=*/2, RCCL_IB_REG_SEG_DEFAULT, start, rkeys));
}

TEST(NetIbCtsLayout, InlineLayoutHoldsAtMostTheInlineLimit) {
    std::vector<uint64_t> start(RCCL_IB_CTS_INLINE_SEGS + 1);
    std::vector<uint32_t> rkeys(RCCL_IB_CTS_INLINE_SEGS + 1, 1u);
    for (size_t s = 0; s < start.size(); s++) start[s] = kBase + (uint64_t)s * kSeg;
    EXPECT_TRUE(IbMsegCtsRemoteLayoutValid(RCCL_IB_CTS_INLINE_SEGS, 0, 2, RCCL_IB_CTS_INLINE_SEGS, start.data(),
                                           rkeys.data()));
    EXPECT_FALSE(IbMsegCtsRemoteLayoutValid(RCCL_IB_CTS_INLINE_SEGS + 1, 0, 2, RCCL_IB_CTS_INLINE_SEGS,
                                            start.data(), rkeys.data()));
}

// The published layout is {va0, off[n + 1], rkeys[ndevs][n]}.
TEST(NetIbLayoutBlob, PacksOffsetsThenPerDeviceRkeys) {
    EXPECT_EQ(rcclIbLayoutBlobBytes(6, 2), sizeof(uint64_t) * 8 + sizeof(uint32_t) * 12);
    std::vector<uint64_t> blob(16, 0);
    EXPECT_EQ(rcclIbLayoutBlobOff(blob.data()), blob.data() + 1);
    EXPECT_EQ(reinterpret_cast<uint64_t*>(rcclIbLayoutBlobRkeys(blob.data(), 6, 0)), blob.data() + 8);
    EXPECT_EQ(rcclIbLayoutBlobRkeys(blob.data(), 6, 1), rcclIbLayoutBlobRkeys(blob.data(), 6, 0) + 6);
}

TEST(NetIbLayoutBlob, ValidNeedsIncreasingOffsetsFromZeroAndNonzeroRkeys) {
    std::vector<uint64_t> blob(16, 0);
    for (uint32_t s = 0; s <= 3; s++) rcclIbLayoutBlobOff(blob.data())[s] = s * kSeg;
    for (uint32_t d = 0; d < 2; d++)
        for (uint32_t s = 0; s < 3; s++) rcclIbLayoutBlobRkeys(blob.data(), 3, d)[s] = 1 + s + 3 * d;
    EXPECT_TRUE(rcclIbLayoutBlobValid(blob.data(), 3, 2));
    EXPECT_FALSE(rcclIbLayoutBlobValid(nullptr, 3, 2));
    EXPECT_FALSE(rcclIbLayoutBlobValid(blob.data(), 0, 2));
    EXPECT_FALSE(rcclIbLayoutBlobValid(blob.data(), 3, 0));
    rcclIbLayoutBlobRkeys(blob.data(), 3, 1)[2] = 0;
    EXPECT_FALSE(rcclIbLayoutBlobValid(blob.data(), 3, 2));
    rcclIbLayoutBlobRkeys(blob.data(), 3, 1)[2] = 6;
    rcclIbLayoutBlobOff(blob.data())[2] = rcclIbLayoutBlobOff(blob.data())[1];
    EXPECT_FALSE(rcclIbLayoutBlobValid(blob.data(), 3, 2));
    rcclIbLayoutBlobOff(blob.data())[2] = 2 * kSeg;
    rcclIbLayoutBlobOff(blob.data())[0] = 1;
    EXPECT_FALSE(rcclIbLayoutBlobValid(blob.data(), 3, 2));
}

TEST(NetIbMultiSeg, HostVmmBaseZeroDoesNotWrapRemaining)
{
    const size_t segSize = kSeg;
    EXPECT_EQ(IbMsegBytesRemainingInSegment(kBase, 0, segSize), segSize);
    EXPECT_EQ(IbMsegBytesRemainingInSegment(kBase + 4096, kBase, segSize), segSize - 4096);
}
