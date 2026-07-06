/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only unit tests for the classic NET/IB multi-segment math (AIRUNTIME-2351
// classic-path follow-up). These exercise the pure, dependency-free helpers in
// src/transport/net_ib/multiseg.h and run in rccl-UnitTestsFixtures without IB
// hardware, a GPU, or MPI.
//
// This is the Option B (wire-protocol) variant of the unit tests: in addition
// to the segment-selection / uniformity helpers shared with Option A, it covers
// ncclIbSplitTransfer, the segment-boundary-splitting builder used by the
// segment-aware ncclIbMultiSend.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

// Pure helpers under test (no ibverbs / RCCL deps).
#include "../src/transport/net_ib/multiseg.h"

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
    return ncclIbSegmentIndexForRange(L.n(), L.start.data(), L.len.data(), addr, len);
}

// Build the (segVA[], segOff[nSeg+1]) tables ncclIbSplitTransfer expects.
struct SplitTables {
    std::vector<uint64_t> va;
    std::vector<uint64_t> off; // size nSeg+1, off[nSeg] == total bytes
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

// === Shared selection / uniformity helpers (also used by Option A) ==========

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

TEST(NetIbMultiSeg, UniformLayoutAccepted) {
    std::vector<size_t> len(4, kSeg);
    EXPECT_TRUE(ncclIbSegmentsUniform(4, len.data()));
}

TEST(NetIbMultiSeg, NonUniformInteriorRejected) {
    std::vector<size_t> len = {kSeg, kSeg / 2, kSeg, kSeg};
    EXPECT_FALSE(ncclIbSegmentsUniform(4, len.data()));
}

// === Option B: ncclIbSplitTransfer ==========================================

// A single-segment transfer on both sides yields exactly one slice with linear
// addresses (the single-segment fast path in ncclIbMultiSend).
TEST(NetIbSplit, SingleSegmentBothSidesOneSlice) {
    Layout local  = MakeUniform(kBase, kSeg, 1);
    Layout remote = MakeUniform(0x900000000ULL, kSeg, 1);
    SplitTables lt = MakeTables(local), rt = MakeTables(remote);
    ncclIbSegSlice out[8];
    int ns = ncclIbSplitTransfer(lt.n(), lt.va.data(), lt.off.data(),
                                 rt.n(), rt.va.data(), rt.off.data(),
                                 /*off*/ 4096, /*len*/ 65536, out, 8);
    ASSERT_EQ(ns, 1);
    EXPECT_EQ(out[0].localAddr,  kBase + 4096);
    EXPECT_EQ(out[0].remoteAddr, 0x900000000ULL + 4096);
    EXPECT_EQ(out[0].len, 65536u);
    EXPECT_EQ(out[0].localSeg, 0);
    EXPECT_EQ(out[0].remoteSeg, 0);
}

// A transfer contained in one segment on both sides stays a single slice even
// when the buffer itself is multi-segment.
TEST(NetIbSplit, WithinSegmentNoSplit) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    SplitTables t = MakeTables(L);
    ncclIbSegSlice out[8];
    int ns = ncclIbSplitTransfer(t.n(), t.va.data(), t.off.data(),
                                 t.n(), t.va.data(), t.off.data(),
                                 /*off*/ kSeg + 1024, /*len*/ 4096, out, 8);
    ASSERT_EQ(ns, 1);
    EXPECT_EQ(out[0].localSeg, 1);
    EXPECT_EQ(out[0].remoteSeg, 1);
    EXPECT_EQ(out[0].len, 4096u);
}

// A transfer straddling one boundary (symmetric layout) splits into two slices
// with the correct per-segment addresses and lengths.
TEST(NetIbSplit, CrossingOneBoundarySplitsIntoTwo) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    SplitTables t = MakeTables(L);
    uint64_t off = kSeg - 1024;   // 1 KiB before the seg0/seg1 boundary
    uint64_t len = 4096;          // ends 3 KiB into seg1
    ncclIbSegSlice out[8];
    int ns = ncclIbSplitTransfer(t.n(), t.va.data(), t.off.data(),
                                 t.n(), t.va.data(), t.off.data(),
                                 off, len, out, 8);
    ASSERT_EQ(ns, 2);
    EXPECT_EQ(out[0].localSeg, 0);
    EXPECT_EQ(out[0].localAddr, kBase + kSeg - 1024);
    EXPECT_EQ(out[0].len, 1024u);
    EXPECT_EQ(out[1].localSeg, 1);
    EXPECT_EQ(out[1].localAddr, kBase + kSeg);
    EXPECT_EQ(out[1].len, 3072u);
    // Slices reassemble to the original range with no gaps/overlaps.
    EXPECT_EQ(out[0].len + out[1].len, len);
}

// Asymmetric layouts: the sender and receiver segment at different sizes; the
// transfer must split at the union of both sides' boundaries.
TEST(NetIbSplit, AsymmetricLayoutsSplitAtBothBoundaries) {
    Layout local  = MakeUniform(kBase, kSeg, 4);          // 2 MiB segments
    Layout remote = MakeUniform(0x900000000ULL, kSeg / 2, 8); // 1 MiB segments
    SplitTables lt = MakeTables(local), rt = MakeTables(remote);
    // Transfer the whole buffer.
    uint64_t total = 4 * kSeg;
    ncclIbSegSlice out[64];
    int ns = ncclIbSplitTransfer(lt.n(), lt.va.data(), lt.off.data(),
                                 rt.n(), rt.va.data(), rt.off.data(),
                                 0, total, out, 64);
    // Remote has the finer granularity (1 MiB): 8 slices, each 1 MiB.
    ASSERT_EQ(ns, 8);
    uint64_t sum = 0, cursor = 0;
    for (int k = 0; k < ns; k++) {
        EXPECT_EQ(out[k].localAddr,  kBase + cursor);
        EXPECT_EQ(out[k].remoteAddr, 0x900000000ULL + cursor);
        EXPECT_EQ(out[k].len, kSeg / 2);
        EXPECT_EQ(out[k].localSeg, static_cast<int>(cursor / kSeg));
        EXPECT_EQ(out[k].remoteSeg, static_cast<int>(cursor / (kSeg / 2)));
        sum += out[k].len; cursor += out[k].len;
    }
    EXPECT_EQ(sum, total);
}

// Full-buffer transfer over a symmetric N-segment layout yields N slices.
TEST(NetIbSplit, FullBufferProducesOneSlicePerSegment) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    SplitTables t = MakeTables(L);
    ncclIbSegSlice out[8];
    int ns = ncclIbSplitTransfer(t.n(), t.va.data(), t.off.data(),
                                 t.n(), t.va.data(), t.off.data(),
                                 0, 4 * kSeg, out, 8);
    ASSERT_EQ(ns, 4);
    for (int k = 0; k < ns; k++) { EXPECT_EQ(out[k].len, kSeg); EXPECT_EQ(out[k].localSeg, k); }
}

TEST(NetIbSplit, ZeroLengthProducesNoSlices) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    SplitTables t = MakeTables(L);
    ncclIbSegSlice out[8];
    int ns = ncclIbSplitTransfer(t.n(), t.va.data(), t.off.data(),
                                 t.n(), t.va.data(), t.off.data(),
                                 kSeg, 0, out, 8);
    EXPECT_EQ(ns, 0);
}

TEST(NetIbSplit, OutOfRangeRejected) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    SplitTables t = MakeTables(L);
    ncclIbSegSlice out[8];
    // Starts one byte past the end of the registered range.
    int ns = ncclIbSplitTransfer(t.n(), t.va.data(), t.off.data(),
                                 t.n(), t.va.data(), t.off.data(),
                                 4 * kSeg, 16, out, 8);
    EXPECT_EQ(ns, -1);
}

TEST(NetIbSplit, MaxSlicesOverflowRejected) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    SplitTables t = MakeTables(L);
    ncclIbSegSlice out[2];
    // Whole buffer needs 4 slices but only 2 are allowed.
    int ns = ncclIbSplitTransfer(t.n(), t.va.data(), t.off.data(),
                                 t.n(), t.va.data(), t.off.data(),
                                 0, 4 * kSeg, out, 2);
    EXPECT_EQ(ns, -1);
}

// Sweep: for a range that starts and ends anywhere, the produced slices always
// tile the range contiguously with no gaps or overlaps and never exceed a
// single segment on either side.
TEST(NetIbSplit, SlicesTileRangeContiguously) {
    Layout local  = MakeUniform(kBase, kSeg, 4);
    Layout remote = MakeUniform(0x900000000ULL, kSeg / 4, 16); // 512 KiB segments
    SplitTables lt = MakeTables(local), rt = MakeTables(remote);
    for (uint64_t off : {uint64_t{0}, uint64_t{1024}, kSeg - 4096, kSeg + kSeg / 2}) {
        for (uint64_t len : {uint64_t{4096}, kSeg / 4, kSeg, kSeg + 12345}) {
            if (off + len > 4 * kSeg) continue;
            ncclIbSegSlice out[64];
            int ns = ncclIbSplitTransfer(lt.n(), lt.va.data(), lt.off.data(),
                                         rt.n(), rt.va.data(), rt.off.data(),
                                         off, len, out, 64);
            ASSERT_GT(ns, 0) << "off=" << off << " len=" << len;
            uint64_t cursor = off, sum = 0;
            for (int k = 0; k < ns; k++) {
                EXPECT_EQ(out[k].localAddr,  kBase + cursor);
                EXPECT_EQ(out[k].remoteAddr, 0x900000000ULL + cursor);
                cursor += out[k].len; sum += out[k].len;
            }
            EXPECT_EQ(sum, len) << "off=" << off << " len=" << len;
        }
    }
}
