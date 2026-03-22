/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ncclBuildRings: given prev/next arrays from rank's perspective, produce the
// ring traversal order starting from `rank`.
//
// rings[r*nranks + i] = the i-th rank visited in ring r, starting from `rank`.

#include "nccl.h"
#include "graph/rings.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>

namespace RcclUnitTesting {

// ---------------------------------------------------------------------------
// Helper: build identity ring  0→1→2→…→N-1→0
// ---------------------------------------------------------------------------
static void makeLinearRing(int nranks, std::vector<int>& next, std::vector<int>& prev) {
    next.resize(nranks);
    prev.resize(nranks);
    for (int i = 0; i < nranks; i++) {
        next[i] = (i + 1) % nranks;
        prev[i] = (i - 1 + nranks) % nranks;
    }
}

// Verify that the ring is a proper permutation starting from `rank`.
static void verifyRing(const std::vector<int>& rings, int ring, int rank, int nranks) {
    ASSERT_GE((int)rings.size(), (ring + 1) * nranks) << "ring vector too small";
    // First element is `rank`
    EXPECT_EQ(rings[ring * nranks + 0], rank) << "ring does not start at rank " << rank;
    // All ranks appear exactly once
    std::vector<int> seen(nranks, 0);
    for (int i = 0; i < nranks; i++) {
        int r = rings[ring * nranks + i];
        ASSERT_GE(r, 0) << "negative rank at position " << i;
        ASSERT_LT(r, nranks) << "rank " << r << " out of bounds at position " << i;
        seen[r]++;
    }
    for (int i = 0; i < nranks; i++) {
        EXPECT_EQ(seen[i], 1) << "rank " << i << " appears " << seen[i] << " times";
    }
}

// ---------------------------------------------------------------------------
// 2-rank ring
// ---------------------------------------------------------------------------
TEST(RingsTest, TwoRanksFromRank0) {
    const int nranks = 2;
    const int nrings = 1;
    std::vector<int> next, prev;
    makeLinearRing(nranks, next, prev);

    std::vector<int> rings(nrings * nranks, -1);
    ncclResult_t ret = ncclBuildRings(nrings, rings.data(), 0, nranks, prev.data(), next.data());
    EXPECT_EQ(ret, ncclSuccess);

    // From rank 0: 0 → 1 → (back to 0)
    EXPECT_EQ(rings[0], 0);
    EXPECT_EQ(rings[1], 1);
    verifyRing(rings, 0, 0, nranks);
}

TEST(RingsTest, TwoRanksFromRank1) {
    const int nranks = 2;
    const int nrings = 1;
    std::vector<int> next, prev;
    makeLinearRing(nranks, next, prev);

    std::vector<int> rings(nrings * nranks, -1);
    ncclResult_t ret = ncclBuildRings(nrings, rings.data(), 1, nranks, prev.data(), next.data());
    EXPECT_EQ(ret, ncclSuccess);

    // From rank 1: 1 → 0 → (back to 1)
    EXPECT_EQ(rings[0], 1);
    EXPECT_EQ(rings[1], 0);
    verifyRing(rings, 0, 1, nranks);
}

// ---------------------------------------------------------------------------
// 4-rank ring
// ---------------------------------------------------------------------------
TEST(RingsTest, FourRanksLinearFromRank0) {
    const int nranks = 4;
    const int nrings = 1;
    std::vector<int> next, prev;
    makeLinearRing(nranks, next, prev);

    std::vector<int> rings(nrings * nranks, -1);
    ncclResult_t ret = ncclBuildRings(nrings, rings.data(), 0, nranks, prev.data(), next.data());
    EXPECT_EQ(ret, ncclSuccess);

    EXPECT_EQ(rings[0], 0);
    EXPECT_EQ(rings[1], 1);
    EXPECT_EQ(rings[2], 2);
    EXPECT_EQ(rings[3], 3);
}

TEST(RingsTest, FourRanksLinearFromRank2) {
    const int nranks = 4;
    const int nrings = 1;
    std::vector<int> next, prev;
    makeLinearRing(nranks, next, prev);

    std::vector<int> rings(nrings * nranks, -1);
    ncclResult_t ret = ncclBuildRings(nrings, rings.data(), 2, nranks, prev.data(), next.data());
    EXPECT_EQ(ret, ncclSuccess);

    // From rank 2: 2 → 3 → 0 → 1
    EXPECT_EQ(rings[0], 2);
    EXPECT_EQ(rings[1], 3);
    EXPECT_EQ(rings[2], 0);
    EXPECT_EQ(rings[3], 1);
    verifyRing(rings, 0, 2, nranks);
}

// ---------------------------------------------------------------------------
// Reversed ring (N-1 → N-2 → … → 0 → N-1)
// ---------------------------------------------------------------------------
TEST(RingsTest, FourRanksReversedFromRank0) {
    const int nranks = 4;
    const int nrings = 1;

    // next[i] = (i - 1 + nranks) % nranks  (reverse direction)
    std::vector<int> next(nranks), prev(nranks);
    for (int i = 0; i < nranks; i++) {
        next[i] = (i - 1 + nranks) % nranks;
        prev[i] = (i + 1) % nranks;
    }

    std::vector<int> rings(nrings * nranks, -1);
    ncclResult_t ret = ncclBuildRings(nrings, rings.data(), 0, nranks, prev.data(), next.data());
    EXPECT_EQ(ret, ncclSuccess);

    // From rank 0: 0 → 3 → 2 → 1
    EXPECT_EQ(rings[0], 0);
    EXPECT_EQ(rings[1], 3);
    EXPECT_EQ(rings[2], 2);
    EXPECT_EQ(rings[3], 1);
    verifyRing(rings, 0, 0, nranks);
}

// ---------------------------------------------------------------------------
// 8-rank ring
// ---------------------------------------------------------------------------
TEST(RingsTest, EightRanksFromRank0) {
    const int nranks = 8;
    const int nrings = 1;
    std::vector<int> next, prev;
    makeLinearRing(nranks, next, prev);

    std::vector<int> rings(nrings * nranks, -1);
    ncclResult_t ret = ncclBuildRings(nrings, rings.data(), 0, nranks, prev.data(), next.data());
    EXPECT_EQ(ret, ncclSuccess);

    for (int i = 0; i < nranks; i++) {
        EXPECT_EQ(rings[i], i) << "position " << i;
    }
    verifyRing(rings, 0, 0, nranks);
}

TEST(RingsTest, EightRanksFromRank5) {
    const int nranks = 8;
    const int nrings = 1;
    std::vector<int> next, prev;
    makeLinearRing(nranks, next, prev);

    std::vector<int> rings(nrings * nranks, -1);
    ncclResult_t ret = ncclBuildRings(nrings, rings.data(), 5, nranks, prev.data(), next.data());
    EXPECT_EQ(ret, ncclSuccess);

    // From rank 5: 5, 6, 7, 0, 1, 2, 3, 4
    int expected[] = {5, 6, 7, 0, 1, 2, 3, 4};
    for (int i = 0; i < nranks; i++) {
        EXPECT_EQ(rings[i], expected[i]) << "position " << i;
    }
    verifyRing(rings, 0, 5, nranks);
}

// ---------------------------------------------------------------------------
// Multiple rings
// ---------------------------------------------------------------------------
TEST(RingsTest, TwoRingsFromRank0) {
    const int nranks = 4;
    const int nrings = 2;
    // Ring 0: 0→1→2→3→0 (linear)
    // Ring 1: 0→2→1→3→0 (different order)
    std::vector<int> next(nrings * nranks), prev(nrings * nranks);

    // Ring 0 (forward)
    for (int i = 0; i < nranks; i++) {
        next[0 * nranks + i] = (i + 1) % nranks;
        prev[0 * nranks + i] = (i - 1 + nranks) % nranks;
    }
    // Ring 1 (stride-2, wrapping): 0→2→1→3→0
    // next[1*4+0]=2, next[1*4+2]=1, next[1*4+1]=3, next[1*4+3]=0
    next[1 * nranks + 0] = 2;
    next[1 * nranks + 2] = 1;
    next[1 * nranks + 1] = 3;
    next[1 * nranks + 3] = 0;
    prev[1 * nranks + 2] = 0;
    prev[1 * nranks + 1] = 2;
    prev[1 * nranks + 3] = 1;
    prev[1 * nranks + 0] = 3;

    std::vector<int> rings(nrings * nranks, -1);
    ncclResult_t ret = ncclBuildRings(nrings, rings.data(), 0, nranks, prev.data(), next.data());
    EXPECT_EQ(ret, ncclSuccess);

    // Ring 0: 0,1,2,3
    EXPECT_EQ(rings[0 * nranks + 0], 0);
    EXPECT_EQ(rings[0 * nranks + 1], 1);
    EXPECT_EQ(rings[0 * nranks + 2], 2);
    EXPECT_EQ(rings[0 * nranks + 3], 3);

    // Ring 1: 0,2,1,3
    EXPECT_EQ(rings[1 * nranks + 0], 0);
    EXPECT_EQ(rings[1 * nranks + 1], 2);
    EXPECT_EQ(rings[1 * nranks + 2], 1);
    EXPECT_EQ(rings[1 * nranks + 3], 3);
}

// ---------------------------------------------------------------------------
// Error: broken ring (doesn't loop back to start)
// ---------------------------------------------------------------------------
TEST(RingsTest, BrokenRingReturnsError) {
    const int nranks = 4;
    const int nrings = 1;
    // Build a broken next array: 0→1→2→1 (cycle that doesn't include all ranks)
    std::vector<int> next = {1, 2, 1, 0};  // rank 2 points back to 1, not 3
    std::vector<int> prev = {3, 0, 1, 2};

    std::vector<int> rings(nrings * nranks, -1);
    // ncclBuildRings should detect the broken ring and return an error
    ncclResult_t ret = ncclBuildRings(nrings, rings.data(), 0, nranks, prev.data(), next.data());
    EXPECT_NE(ret, ncclSuccess);
}

} // namespace RcclUnitTesting
