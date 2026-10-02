/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only unit tests for classic NET/IB multi-segment math.
// They cover segment selection, layout uniformity, and transfer splitting in
// src/transport/net_ib/multiseg.h and run in rccl-UnitTestsFixtures without
// IB hardware, a GPU, or MPI.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

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

// Build the (segVA[], segOff[nSeg+1]) tables ncclIbSplitTransferAtOffsets expects.
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

TEST(NetIbMultiSeg, LegacyConnectMetadataHasNoCapabilities) {
    const char devName[32] = "bnxt_re0";
    EXPECT_EQ(ncclIbGetConnectCaps(devName, sizeof(devName)), 0u);
}

TEST(NetIbMultiSeg, ConnectCapabilitiesPreserveDeviceName) {
    char devName[32] = "bnxt_re0";
    ncclIbSetConnectCaps(devName, sizeof(devName), NCCL_IB_CAP_MULTISEG);
    EXPECT_STREQ(devName, "bnxt_re0");
    EXPECT_EQ(ncclIbGetConnectCaps(devName, sizeof(devName)),
              NCCL_IB_CAP_MULTISEG);
}

// Merged vNIC names can fill MAX_MERGED_DEV_NAME (648). The helper must still
// terminate the C string before the 8-byte trailer and keep the trailer valid.
TEST(NetIbMultiSeg, ConnectCapabilitiesTruncatesNameThatFillsTrailer) {
    char devName[648];
    std::memset(devName, 'x', sizeof(devName));
    ncclIbSetConnectCaps(devName, sizeof(devName), NCCL_IB_CAP_MULTISEG);
    const size_t trailer = sizeof(ncclIbConnectCapsTrailer);
    EXPECT_EQ(devName[sizeof(devName) - trailer - 1], '\0');
    EXPECT_EQ(ncclIbGetConnectCaps(devName, sizeof(devName)),
              NCCL_IB_CAP_MULTISEG);
}

// A peer that never advertised NCCL_IB_CAP_MULTISEG cannot parse the
// per-segment CTS side table, so multi-segment registration must be declined.
// Every MPI test connects two same-build peers, so only a unit test reaches it.
TEST(NetIbMultiSeg, MultiSegRegistrationDeclinedWithoutPeerCapability) {
    EXPECT_TRUE(ncclIbDeclineMultiSegRegistration(0u, 4));
    EXPECT_TRUE(ncclIbDeclineMultiSegRegistration(~NCCL_IB_CAP_MULTISEG, 4));
}

TEST(NetIbMultiSeg, MultiSegRegistrationAcceptedWithPeerCapability) {
    EXPECT_FALSE(ncclIbDeclineMultiSegRegistration(NCCL_IB_CAP_MULTISEG, 4));
}

TEST(NetIbMultiSeg, SingleSegmentRegistrationNeverDeclined) {
    EXPECT_FALSE(ncclIbDeclineMultiSegRegistration(0u, 1));
    EXPECT_FALSE(ncclIbDeclineMultiSegRegistration(NCCL_IB_CAP_MULTISEG, 1));
}

// === Segment selection and uniformity helpers ===============================

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

TEST(NetIbMultiSeg, OverlappingRangeWholeBufferTouchesEverySegment) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    int out[8];
    int n = ncclIbSegmentsOverlappingRange(L.n(), L.start.data(), L.len.data(),
                                           kBase, 4 * kSeg, out, 8);
    ASSERT_EQ(n, 4);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 1);
    EXPECT_EQ(out[2], 2);
    EXPECT_EQ(out[3], 3);
}

TEST(NetIbMultiSeg, OverlappingRangeSingleSegment) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    int out[8];
    int n = ncclIbSegmentsOverlappingRange(L.n(), L.start.data(), L.len.data(),
                                           kBase + 2 * kSeg + 64, 128, out, 8);
    ASSERT_EQ(n, 1);
    EXPECT_EQ(out[0], 2);
}

TEST(NetIbMultiSeg, OverlappingRangeCrossesOneBoundary) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    int out[8];
    int n = ncclIbSegmentsOverlappingRange(L.n(), L.start.data(), L.len.data(),
                                           kBase + kSeg - 64, 128, out, 8);
    ASSERT_EQ(n, 2);
    EXPECT_EQ(out[0], 0);
    EXPECT_EQ(out[1], 1);
}

TEST(NetIbMultiSeg, OverlappingRangeZeroLengthIsEmpty) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    int out[8];
    EXPECT_EQ(ncclIbSegmentsOverlappingRange(L.n(), L.start.data(), L.len.data(),
                                             kBase, 0, out, 8),
              0);
}

// Zero-length WRs pick the remote rkey by registration-relative offset.
TEST(NetIbMultiSeg, ZeroLengthWrUsesContainingSegment) {
    SplitTables t = MakeTables(MakeUniform(kBase, kSeg, 4));
    EXPECT_EQ(ncclIbSegmentForZeroLengthOffset(t.n(), t.off.data(), 0), 0);
    EXPECT_EQ(ncclIbSegmentForZeroLengthOffset(t.n(), t.off.data(), kSeg), 1);
    EXPECT_EQ(ncclIbSegmentForZeroLengthOffset(t.n(), t.off.data(), 2 * kSeg + 64), 2);
}

TEST(NetIbMultiSeg, ZeroLengthWrAtLastExclusiveEndUsesLastMr) {
    SplitTables t = MakeTables(MakeUniform(kBase, kSeg, 4));
    EXPECT_EQ(ncclIbSegmentForZeroLengthOffset(t.n(), t.off.data(), 4 * kSeg), 3);
    EXPECT_EQ(ncclIbSegmentForZeroLengthOffset(t.n(), t.off.data(), 4 * kSeg + 1), -1);
}

TEST(NetIbMultiSeg, ZeroLengthWrUnequalSegments) {
    constexpr uint64_t gpuBytes = 6u * 1024 * 1024;
    Layout L{{kBase, kBase + gpuBytes}, {gpuBytes, kSeg}};
    SplitTables t = MakeTables(L);
    EXPECT_EQ(ncclIbSegmentForZeroLengthOffset(t.n(), t.off.data(), gpuBytes - 1), 0);
    EXPECT_EQ(ncclIbSegmentForZeroLengthOffset(t.n(), t.off.data(), gpuBytes), 1);
    EXPECT_EQ(ncclIbSegmentForZeroLengthOffset(t.n(), t.off.data(), gpuBytes + kSeg), 1);
}

TEST(NetIbMultiSeg, ZeroLengthWrEmptyLayout) {
    const uint64_t off[] = {0};
    EXPECT_EQ(ncclIbSegmentForZeroLengthOffset(0, off, 0), -1);
}

TEST(NetIbMultiSeg, UniformLayoutAccepted) {
    std::vector<size_t> len(4, kSeg);
    EXPECT_TRUE(ncclIbSegmentsUniform(4, len.data()));
}

TEST(NetIbMultiSeg, TrailingSegmentMayBeSmaller) {
    std::vector<size_t> len = {kSeg, kSeg, kSeg, kSeg / 2};
    EXPECT_TRUE(ncclIbSegmentsUniform(4, len.data()));
}

TEST(NetIbMultiSeg, ShortLeadingSegmentAccepted) {
    std::vector<size_t> len = {kSeg / 2, kSeg, kSeg, kSeg};
    EXPECT_TRUE(ncclIbSegmentsUniform(4, len.data()));
}

TEST(NetIbMultiSeg, TrailingSegmentMayNotBeLarger) {
    std::vector<size_t> len = {kSeg, kSeg, kSeg, kSeg * 2};
    EXPECT_FALSE(ncclIbSegmentsUniform(4, len.data()));
}

// With two segments there is no interior, so any pair passes: a larger
// trailing segment and the DeepEP [GPU][CPU] shape both register.
TEST(NetIbMultiSeg, AnyTwoSegmentLayoutIsUniform) {
    std::vector<size_t> growing = {kSeg, 2 * kSeg};
    std::vector<size_t> deepEp  = {3 * kSeg, kSeg};
    EXPECT_TRUE(ncclIbSegmentsUniform(2, growing.data()));
    EXPECT_TRUE(ncclIbSegmentsUniform(2, deepEp.data()));
}

TEST(NetIbMultiSeg, SingleOrEmptyLayoutIsUniform) {
    std::vector<size_t> len = {kSeg};
    EXPECT_TRUE(ncclIbSegmentsUniform(1, len.data()));
    EXPECT_TRUE(ncclIbSegmentsUniform(0, nullptr));
}

TEST(NetIbMultiSeg, NonUniformInteriorRejected) {
    std::vector<size_t> len = {kSeg, kSeg / 2, kSeg, kSeg};
    EXPECT_FALSE(ncclIbSegmentsUniform(4, len.data()));
}

// === Transfer splitting helpers =============================================

// A single-segment transfer on both sides yields exactly one slice with linear
// addresses (the single-segment fast path in ncclIbMultiSend).
TEST(NetIbSplit, SingleSegmentBothSidesOneSlice) {
    Layout local  = MakeUniform(kBase, kSeg, 1);
    Layout remote = MakeUniform(0x900000000ULL, kSeg, 1);
    SplitTables lt = MakeTables(local), rt = MakeTables(remote);
    ncclIbSegSlice out[8];
    int ns = ncclIbSplitTransferAtOffsets(lt.n(), lt.va.data(), lt.off.data(),
                                          rt.n(), rt.va.data(), rt.off.data(),
                                          /*localOff*/ 4096, /*remoteOff*/ 8192, /*len*/ 65536, out, 8);
    ASSERT_EQ(ns, 1);
    EXPECT_EQ(out[0].localAddr,  kBase + 4096);
    EXPECT_EQ(out[0].remoteAddr, 0x900000000ULL + 8192);
    EXPECT_EQ(out[0].len, 65536u);
    EXPECT_EQ(out[0].localSeg, 0);
    EXPECT_EQ(out[0].remoteSeg, 0);
}

// A transfer contained in one segment on both sides stays a single slice even
// when the buffer itself is multi-segment.
TEST(NetIbSplit, WithinSegmentNoSplit) {
    Layout local  = MakeUniform(kBase, kSeg, 4);
    Layout remote = MakeUniform(0x900000000ULL, kSeg, 4);
    SplitTables lt = MakeTables(local), rt = MakeTables(remote);
    ncclIbSegSlice out[8];
    int ns = ncclIbSplitTransferAtOffsets(lt.n(), lt.va.data(), lt.off.data(),
                                          rt.n(), rt.va.data(), rt.off.data(),
                                          /*localOff*/ kSeg + 1024, /*remoteOff*/ 2 * kSeg + 512,
                                          /*len*/ 4096, out, 8);
    ASSERT_EQ(ns, 1);
    EXPECT_EQ(out[0].localSeg, 1);
    EXPECT_EQ(out[0].remoteSeg, 2);
    EXPECT_EQ(out[0].localAddr, kBase + kSeg + 1024);
    EXPECT_EQ(out[0].remoteAddr, 0x900000000ULL + 2 * kSeg + 512);
    EXPECT_EQ(out[0].len, 4096u);
}

// A transfer straddling one local boundary splits into two slices with the
// correct per-segment addresses, while the remote side stays in one segment.
TEST(NetIbSplit, CrossingOneBoundarySplitsIntoTwo) {
    Layout local  = MakeUniform(kBase, kSeg, 4);
    Layout remote = MakeUniform(0x900000000ULL, kSeg, 4);
    SplitTables lt = MakeTables(local), rt = MakeTables(remote);
    uint64_t localOff  = kSeg - 1024;     // 1 KiB before the local seg0/seg1 boundary
    uint64_t remoteOff = 2 * kSeg + 4096; // well inside remote seg2
    uint64_t len = 4096;                  // ends 3 KiB into local seg1
    ncclIbSegSlice out[8];
    int ns = ncclIbSplitTransferAtOffsets(lt.n(), lt.va.data(), lt.off.data(),
                                          rt.n(), rt.va.data(), rt.off.data(),
                                          localOff, remoteOff, len, out, 8);
    ASSERT_EQ(ns, 2);
    EXPECT_EQ(out[0].localSeg, 0);
    EXPECT_EQ(out[0].localAddr, kBase + kSeg - 1024);
    EXPECT_EQ(out[0].remoteSeg, 2);
    EXPECT_EQ(out[0].remoteAddr, 0x900000000ULL + remoteOff);
    EXPECT_EQ(out[0].len, 1024u);
    EXPECT_EQ(out[1].localSeg, 1);
    EXPECT_EQ(out[1].localAddr, kBase + kSeg);
    EXPECT_EQ(out[1].remoteSeg, 2);
    EXPECT_EQ(out[1].remoteAddr, 0x900000000ULL + remoteOff + 1024);
    EXPECT_EQ(out[1].len, 3072u);
    // Slices reassemble to the original range with no gaps/overlaps.
    EXPECT_EQ(out[0].len + out[1].len, len);
}

// Asymmetric layouts: the sender and receiver segment at different sizes and
// start at different offsets; the transfer must split at the union of both
// sides' boundaries.
TEST(NetIbSplit, AsymmetricLayoutsSplitAtBothBoundaries) {
    Layout local  = MakeUniform(kBase, kSeg, 4);              // 2 MiB segments
    Layout remote = MakeUniform(0x900000000ULL, kSeg / 2, 8); // 1 MiB segments
    SplitTables lt = MakeTables(local), rt = MakeTables(remote);
    const uint64_t localOff  = kSeg / 4; // 512 KiB
    const uint64_t remoteOff = 0;
    const uint64_t len = 3 * kSeg;
    ncclIbSegSlice out[64];
    int ns = ncclIbSplitTransferAtOffsets(lt.n(), lt.va.data(), lt.off.data(),
                                          rt.n(), rt.va.data(), rt.off.data(),
                                          localOff, remoteOff, len, out, 64);
    // Remote boundaries every 1 MiB plus local ones at 1.5, 3.5 and 5.5 MiB.
    ASSERT_EQ(ns, 9);
    uint64_t sum = 0, cursor = 0;
    for (int k = 0; k < ns; k++) {
        const uint64_t l = localOff + cursor, r = remoteOff + cursor;
        EXPECT_EQ(out[k].localAddr,  kBase + l);
        EXPECT_EQ(out[k].remoteAddr, 0x900000000ULL + r);
        EXPECT_EQ(out[k].localSeg, static_cast<int>(l / kSeg));
        EXPECT_EQ(out[k].remoteSeg, static_cast<int>(r / (kSeg / 2)));
        EXPECT_EQ((l + out[k].len - 1) / kSeg, l / kSeg) << "slice " << k << " crosses a local boundary";
        EXPECT_EQ((r + out[k].len - 1) / (kSeg / 2), r / (kSeg / 2)) << "slice " << k << " crosses a remote boundary";
        sum += out[k].len; cursor += out[k].len;
    }
    EXPECT_EQ(sum, len);
}

// Local and remote API buffers may begin at different offsets within their
// respective registrations. Splitting must preserve both offsets rather than
// treating the request-relative offset as registration-relative.
TEST(NetIbSplit, IndependentLocalAndRemoteStartOffsets) {
    Layout local  = MakeUniform(kBase, kSeg, 4);
    Layout remote = MakeUniform(0x900000000ULL, kSeg / 2, 8);
    SplitTables lt = MakeTables(local), rt = MakeTables(remote);
    ncclIbSegSlice out[8];
    const uint64_t localOff  = kSeg + 512;
    const uint64_t remoteOff = kSeg / 2 - 512;
    int ns = ncclIbSplitTransferAtOffsets(
        lt.n(), lt.va.data(), lt.off.data(),
        rt.n(), rt.va.data(), rt.off.data(),
        localOff, remoteOff, 2048, out, 8);
    ASSERT_EQ(ns, 2);
    EXPECT_EQ(out[0].localAddr, kBase + localOff);
    EXPECT_EQ(out[0].remoteAddr, 0x900000000ULL + remoteOff);
    EXPECT_EQ(out[0].len, 512u);
    EXPECT_EQ(out[0].localSeg, 1);
    EXPECT_EQ(out[0].remoteSeg, 0);
    EXPECT_EQ(out[1].localAddr, kBase + localOff + 512);
    EXPECT_EQ(out[1].remoteAddr, 0x900000000ULL + kSeg / 2);
    EXPECT_EQ(out[1].len, 1536u);
    EXPECT_EQ(out[1].localSeg, 1);
    EXPECT_EQ(out[1].remoteSeg, 1);
}

// DeepEP Engram uses one unequal [GPU][CPU] window and reads remote CPU storage
// into a different offset in the local GPU segment. Preserve that consumer
// geometry as a host-only regression for independent offset/key selection.
TEST(NetIbSplit, DeepEP_EngramCpuToGpuOffsets) {
    constexpr uint64_t gpuBytes = 6u * 1024 * 1024;
    constexpr uint64_t cpuBytes = 2u * 1024 * 1024;
    constexpr uint64_t remoteBase = 0x900000000ULL;
    Layout local{{kBase, kBase + gpuBytes}, {gpuBytes, cpuBytes}};
    Layout remote{{remoteBase, remoteBase + gpuBytes}, {gpuBytes, cpuBytes}};
    SplitTables lt = MakeTables(local), rt = MakeTables(remote);
    ncclIbSegSlice out[4];
    const uint64_t localOff = 64 * 1024;
    const uint64_t remoteOff = gpuBytes + 4096;
    const uint64_t len = 128 * 1024;
    int ns = ncclIbSplitTransferAtOffsets(
        lt.n(), lt.va.data(), lt.off.data(),
        rt.n(), rt.va.data(), rt.off.data(),
        localOff, remoteOff, len, out, 4);
    ASSERT_EQ(ns, 1);
    EXPECT_EQ(out[0].localSeg, 0);
    EXPECT_EQ(out[0].remoteSeg, 1);
    EXPECT_EQ(out[0].localAddr, kBase + localOff);
    EXPECT_EQ(out[0].remoteAddr, remoteBase + remoteOff);
    EXPECT_EQ(out[0].len, len);
}

// Full-buffer transfer over an N-segment layout yields N slices, even when the
// remote copy starts one segment into a larger registration.
TEST(NetIbSplit, FullBufferProducesOneSlicePerSegment) {
    Layout local  = MakeUniform(kBase, kSeg, 4);
    Layout remote = MakeUniform(0x900000000ULL, kSeg, 5);
    SplitTables lt = MakeTables(local), rt = MakeTables(remote);
    ncclIbSegSlice out[8];
    int ns = ncclIbSplitTransferAtOffsets(lt.n(), lt.va.data(), lt.off.data(),
                                          rt.n(), rt.va.data(), rt.off.data(),
                                          /*localOff*/ 0, /*remoteOff*/ kSeg, 4 * kSeg, out, 8);
    ASSERT_EQ(ns, 4);
    for (int k = 0; k < ns; k++) {
        EXPECT_EQ(out[k].len, kSeg);
        EXPECT_EQ(out[k].localSeg, k);
        EXPECT_EQ(out[k].remoteSeg, k + 1);
    }
}

TEST(NetIbSplit, ZeroLengthProducesNoSlices) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    SplitTables t = MakeTables(L);
    ncclIbSegSlice out[8];
    int ns = ncclIbSplitTransferAtOffsets(t.n(), t.va.data(), t.off.data(),
                                          t.n(), t.va.data(), t.off.data(),
                                          kSeg, 2 * kSeg, 0, out, 8);
    EXPECT_EQ(ns, 0);
}

// Either side starting one byte past the end of its registered range fails.
TEST(NetIbSplit, OutOfRangeRejected) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    SplitTables t = MakeTables(L);
    ncclIbSegSlice out[8];
    EXPECT_EQ(ncclIbSplitTransferAtOffsets(t.n(), t.va.data(), t.off.data(),
                                           t.n(), t.va.data(), t.off.data(),
                                           /*localOff*/ 4 * kSeg, /*remoteOff*/ 0, 16, out, 8),
              -1);
    EXPECT_EQ(ncclIbSplitTransferAtOffsets(t.n(), t.va.data(), t.off.data(),
                                           t.n(), t.va.data(), t.off.data(),
                                           /*localOff*/ 0, /*remoteOff*/ 4 * kSeg, 16, out, 8),
              -1);
}

TEST(NetIbSplit, MaxSlicesOverflowRejected) {
    Layout L = MakeUniform(kBase, kSeg, 4);
    SplitTables t = MakeTables(L);
    ncclIbSegSlice out[2];
    // Offsets half a segment apart put a boundary every 1 MiB: 6 slices, 2 allowed.
    int ns = ncclIbSplitTransferAtOffsets(t.n(), t.va.data(), t.off.data(),
                                          t.n(), t.va.data(), t.off.data(),
                                          0, kSeg / 2, 3 * kSeg, out, 2);
    EXPECT_EQ(ns, -1);
}

// Sweep: for ranges that start anywhere on each side, the produced slices
// always tile the range contiguously with no gaps or overlaps and never exceed
// a single segment on either side.
TEST(NetIbSplit, SlicesTileRangeContiguously) {
    constexpr uint64_t remoteSeg = kSeg / 4; // 512 KiB segments
    Layout local  = MakeUniform(kBase, kSeg, 4);
    Layout remote = MakeUniform(0x900000000ULL, remoteSeg, 16);
    SplitTables lt = MakeTables(local), rt = MakeTables(remote);
    for (uint64_t off : {uint64_t{0}, uint64_t{1024}, kSeg - 4096, kSeg + kSeg / 2}) {
        for (uint64_t shift : {uint64_t{4096}, remoteSeg / 2 + 123}) {
            for (uint64_t len : {uint64_t{4096}, kSeg / 4, kSeg, kSeg + 12345}) {
                const uint64_t remoteOff = off + shift;
                if (remoteOff + len > 4 * kSeg) continue;
                ncclIbSegSlice out[64];
                int ns = ncclIbSplitTransferAtOffsets(lt.n(), lt.va.data(), lt.off.data(),
                                                      rt.n(), rt.va.data(), rt.off.data(),
                                                      off, remoteOff, len, out, 64);
                ASSERT_GT(ns, 0) << "off=" << off << " remoteOff=" << remoteOff << " len=" << len;
                uint64_t cursor = 0;
                for (int k = 0; k < ns; k++) {
                    const uint64_t l = off + cursor, r = remoteOff + cursor;
                    EXPECT_EQ(out[k].localAddr,  kBase + l);
                    EXPECT_EQ(out[k].remoteAddr, 0x900000000ULL + r);
                    EXPECT_EQ((l + out[k].len - 1) / kSeg, l / kSeg);
                    EXPECT_EQ((r + out[k].len - 1) / remoteSeg, r / remoteSeg);
                    cursor += out[k].len;
                }
                EXPECT_EQ(cursor, len) << "off=" << off << " remoteOff=" << remoteOff << " len=" << len;
            }
        }
    }
}

TEST(NetIbCtsLayout, AcceptsMonotonicNonzeroRkeys) {
    const uint64_t start[] = {kBase, kBase + kSeg, kBase + 2 * kSeg};
    const uint32_t rkeys[] = {1u, 2u, 3u};
    EXPECT_TRUE(ncclIbCtsRemoteLayoutValid(3, /*remDevIdx=*/0, /*maxDevs=*/2, NCCL_IB_MAX_SEGMENTS, start, rkeys));
}

TEST(NetIbCtsLayout, RejectsZeroSegmentCount) {
    const uint64_t start[] = {kBase};
    const uint32_t rkeys[] = {1u};
    EXPECT_FALSE(ncclIbCtsRemoteLayoutValid(0, 0, 2, NCCL_IB_MAX_SEGMENTS, start, rkeys));
}

TEST(NetIbCtsLayout, RejectsSegmentCountAboveCap) {
    const uint64_t start[] = {kBase};
    const uint32_t rkeys[] = {1u};
    EXPECT_FALSE(ncclIbCtsRemoteLayoutValid(NCCL_IB_MAX_SEGMENTS + 1, 0, 2, NCCL_IB_MAX_SEGMENTS, start, rkeys));
}

TEST(NetIbCtsLayout, RejectsOutOfRangeDeviceIndex) {
    const uint64_t start[] = {kBase, kBase + kSeg};
    const uint32_t rkeys[] = {1u, 2u};
    EXPECT_FALSE(ncclIbCtsRemoteLayoutValid(2, /*remDevIdx=*/-1, 2, NCCL_IB_MAX_SEGMENTS, start, rkeys));
    EXPECT_FALSE(ncclIbCtsRemoteLayoutValid(2, /*remDevIdx=*/2, 2, NCCL_IB_MAX_SEGMENTS, start, rkeys));
}

TEST(NetIbCtsLayout, RejectsNonMonotonicSegStart) {
    const uint64_t start[] = {kBase, kBase, kBase + 2 * kSeg};
    const uint32_t rkeys[] = {1u, 2u, 3u};
    EXPECT_FALSE(ncclIbCtsRemoteLayoutValid(3, 0, 2, NCCL_IB_MAX_SEGMENTS, start, rkeys));
}

TEST(NetIbCtsLayout, RejectsZeroRkey) {
    const uint64_t start[] = {kBase, kBase + kSeg};
    const uint32_t rkeys[] = {1u, 0u};
    EXPECT_FALSE(ncclIbCtsRemoteLayoutValid(2, 0, 2, NCCL_IB_MAX_SEGMENTS, start, rkeys));
}

TEST(NetIbCtsLayout, RejectsNullTables) {
    const uint64_t start[] = {kBase, kBase + kSeg};
    const uint32_t rkeys[] = {1u, 2u};
    EXPECT_FALSE(ncclIbCtsRemoteLayoutValid(2, 0, 2, NCCL_IB_MAX_SEGMENTS, nullptr, rkeys));
    EXPECT_FALSE(ncclIbCtsRemoteLayoutValid(2, 0, 2, NCCL_IB_MAX_SEGMENTS, start, nullptr));
}

TEST(NetIbCtsLayout, AcceptsSegmentCountAtCap) {
    std::vector<uint64_t> start(NCCL_IB_MAX_SEGMENTS);
    std::vector<uint32_t> rkeys(NCCL_IB_MAX_SEGMENTS, 1u);
    for (int s = 0; s < NCCL_IB_MAX_SEGMENTS; s++) start[s] = kBase + (uint64_t)s * kSeg;
    EXPECT_TRUE(
        ncclIbCtsRemoteLayoutValid(NCCL_IB_MAX_SEGMENTS, 0, 2, NCCL_IB_MAX_SEGMENTS, start.data(), rkeys.data()));
}

TEST(NetIbCtsLayout, AcceptsLastDeviceIndex) {
    const uint64_t start[] = {kBase, kBase + kSeg};
    const uint32_t rkeys[] = {1u, 2u};
    EXPECT_TRUE(ncclIbCtsRemoteLayoutValid(2, /*remDevIdx=*/1, /*maxDevs=*/2, NCCL_IB_MAX_SEGMENTS, start, rkeys));
}

TEST(NetIbMultiSeg, HostVmmBaseZeroDoesNotWrapRemaining)
{
    const size_t segSize = kSeg;
    EXPECT_EQ(ncclIbBytesRemainingInSegment(kBase, 0, segSize), segSize);
    EXPECT_EQ(ncclIbBytesRemainingInSegment(kBase + 4096, kBase, segSize), segSize - 4096);
}
