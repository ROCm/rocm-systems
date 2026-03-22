/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for ncclGetBtree and ncclGetDtree from src/graph/trees.cc
//
// The binary tree structure for N ranks (root = 0):
//
//   For N=4:   0---2
//                 / \
//                1   3
//
//   For N=8:   0-------4
//                     / \
//                    2   6
//                   / \ / \
//                  1  3 5  7

#include "nccl.h"
#include "trees.h"
#include <gtest/gtest.h>

namespace RcclUnitTesting {

// ---------------------------------------------------------------------------
// ncclGetBtree
// ---------------------------------------------------------------------------

// Header signature: ncclGetBtree(nranks, rank, up, down0, down1, parentChildType)
// Note: "down0" is the left child (rank - lowbit), "down1" is the right child

TEST(BtreeTest, SingleRank) {
    int u = 0, d0 = 0, d1 = 0, pct = 0;
    ncclResult_t ret = ncclGetBtree(1, 0, &u, &d0, &d1, &pct);
    EXPECT_EQ(ret, ncclSuccess);
    // Single rank: root with no children, no parent
    EXPECT_EQ(u, -1);
    EXPECT_EQ(d0, -1);
    EXPECT_EQ(d1, -1);
}

TEST(BtreeTest, TwoRanks_Rank0) {
    int u = 0, d0 = 0, d1 = 0, pct = 0;
    ncclResult_t ret = ncclGetBtree(2, 0, &u, &d0, &d1, &pct);
    EXPECT_EQ(ret, ncclSuccess);
    // Rank 0 is root: no parent, no left child, right child is rank 1
    EXPECT_EQ(u, -1);
    EXPECT_EQ(d0, -1);
    EXPECT_EQ(d1, 1);
}

TEST(BtreeTest, TwoRanks_Rank1) {
    int u = 0, d0 = 0, d1 = 0, pct = 0;
    ncclResult_t ret = ncclGetBtree(2, 1, &u, &d0, &d1, &pct);
    EXPECT_EQ(ret, ncclSuccess);
    // Rank 1 is a leaf: parent is 0, no children
    EXPECT_EQ(u, 0);
    EXPECT_EQ(d0, -1);
    EXPECT_EQ(d1, -1);
}

// 4-rank tree:
//   0---2
//      / \
//     1   3
TEST(BtreeTest, FourRanks_Root) {
    int u = 0, d0 = 0, d1 = 0, pct = 0;
    ncclResult_t ret = ncclGetBtree(4, 0, &u, &d0, &d1, &pct);
    EXPECT_EQ(ret, ncclSuccess);
    EXPECT_EQ(u, -1);
    EXPECT_EQ(d0, -1);
    EXPECT_EQ(d1, 2);  // right child
}

TEST(BtreeTest, FourRanks_Rank1) {
    int u = 0, d0 = 0, d1 = 0, pct = 0;
    ncclResult_t ret = ncclGetBtree(4, 1, &u, &d0, &d1, &pct);
    EXPECT_EQ(ret, ncclSuccess);
    EXPECT_EQ(u, 2);   // parent
    EXPECT_EQ(d0, -1); // no children (leaf)
    EXPECT_EQ(d1, -1);
}

TEST(BtreeTest, FourRanks_Rank2) {
    int u = 0, d0 = 0, d1 = 0, pct = 0;
    ncclResult_t ret = ncclGetBtree(4, 2, &u, &d0, &d1, &pct);
    EXPECT_EQ(ret, ncclSuccess);
    EXPECT_EQ(u, 0);   // parent is root
    EXPECT_EQ(d0, 1);  // left child
    EXPECT_EQ(d1, 3);  // right child
}

TEST(BtreeTest, FourRanks_Rank3) {
    int u = 0, d0 = 0, d1 = 0, pct = 0;
    ncclResult_t ret = ncclGetBtree(4, 3, &u, &d0, &d1, &pct);
    EXPECT_EQ(ret, ncclSuccess);
    EXPECT_EQ(u, 2);   // parent
    EXPECT_EQ(d0, -1); // no children (leaf)
    EXPECT_EQ(d1, -1);
}

// 8-rank tree:
//   0---4
//      / \
//     2   6
//    / \ / \
//   1  3 5  7
TEST(BtreeTest, EightRanks_Root) {
    int u = 0, d0 = 0, d1 = 0, pct = 0;
    ncclResult_t ret = ncclGetBtree(8, 0, &u, &d0, &d1, &pct);
    EXPECT_EQ(ret, ncclSuccess);
    EXPECT_EQ(u, -1);
    EXPECT_EQ(d0, -1);
    EXPECT_EQ(d1, 4);
}

TEST(BtreeTest, EightRanks_Rank4) {
    int u = 0, d0 = 0, d1 = 0, pct = 0;
    ncclResult_t ret = ncclGetBtree(8, 4, &u, &d0, &d1, &pct);
    EXPECT_EQ(ret, ncclSuccess);
    EXPECT_EQ(u, 0);   // parent
    EXPECT_EQ(d0, 2);  // left child
    EXPECT_EQ(d1, 6);  // right child
}

TEST(BtreeTest, EightRanks_Rank2) {
    int u = 0, d0 = 0, d1 = 0, pct = 0;
    ncclResult_t ret = ncclGetBtree(8, 2, &u, &d0, &d1, &pct);
    EXPECT_EQ(ret, ncclSuccess);
    EXPECT_EQ(u, 4);   // parent
    EXPECT_EQ(d0, 1);  // left child
    EXPECT_EQ(d1, 3);  // right child
}

TEST(BtreeTest, EightRanks_Rank6) {
    int u = 0, d0 = 0, d1 = 0, pct = 0;
    ncclResult_t ret = ncclGetBtree(8, 6, &u, &d0, &d1, &pct);
    EXPECT_EQ(ret, ncclSuccess);
    EXPECT_EQ(u, 4);   // parent
    EXPECT_EQ(d0, 5);  // left child
    EXPECT_EQ(d1, 7);  // right child
}

TEST(BtreeTest, EightRanks_LeafNodes) {
    // Ranks 1, 3, 5, 7 are all leaves
    for (int rank : {1, 3, 5, 7}) {
        int u = 0, d0 = 0, d1 = 0, pct = 0;
        ncclResult_t ret = ncclGetBtree(8, rank, &u, &d0, &d1, &pct);
        EXPECT_EQ(ret, ncclSuccess) << "rank=" << rank;
        EXPECT_NE(u, -1) << "leaf rank " << rank << " should have a parent";
        EXPECT_EQ(d0, -1) << "leaf rank " << rank << " should have no left child";
        EXPECT_EQ(d1, -1) << "leaf rank " << rank << " should have no right child";
    }
}

// Verify tree connectivity: every non-root rank's parent should list it as a child
TEST(BtreeTest, FourRanks_ParentChildConsistency) {
    const int nranks = 4;
    struct NodeInfo { int u, d0, d1, pct; };
    NodeInfo info[nranks];

    for (int r = 0; r < nranks; r++) {
        ncclResult_t ret = ncclGetBtree(nranks, r, &info[r].u, &info[r].d0, &info[r].d1, &info[r].pct);
        EXPECT_EQ(ret, ncclSuccess) << "rank=" << r;
    }

    // For each non-root rank, verify it appears as a child of its parent
    for (int r = 1; r < nranks; r++) {
        int parent = info[r].u;
        ASSERT_GE(parent, 0) << "rank " << r << " has no parent";
        ASSERT_LT(parent, nranks);
        bool found = (info[parent].d0 == r || info[parent].d1 == r);
        EXPECT_TRUE(found) << "rank " << r << " not found as child of rank " << parent;
    }
}

TEST(BtreeTest, EightRanks_ParentChildConsistency) {
    const int nranks = 8;
    struct NodeInfo { int u, d0, d1, pct; };
    NodeInfo info[nranks];

    for (int r = 0; r < nranks; r++) {
        ncclResult_t ret = ncclGetBtree(nranks, r, &info[r].u, &info[r].d0, &info[r].d1, &info[r].pct);
        EXPECT_EQ(ret, ncclSuccess) << "rank=" << r;
    }

    // Root has no parent
    EXPECT_EQ(info[0].u, -1);

    for (int r = 1; r < nranks; r++) {
        int parent = info[r].u;
        ASSERT_GE(parent, 0) << "rank " << r << " has no parent";
        ASSERT_LT(parent, nranks);
        bool found = (info[parent].d0 == r || info[parent].d1 == r);
        EXPECT_TRUE(found) << "rank " << r << " not found as child of rank " << parent;
    }
}

// ---------------------------------------------------------------------------
// ncclGetDtree — double binary tree
// ---------------------------------------------------------------------------

// For an even number of ranks, the second tree is the mirror of the first.
TEST(DtreeTest, TwoRanks_Rank0) {
    int s0 = 0, d0_0 = 0, d0_1 = 0, pct0 = 0;
    int s1 = 0, d1_0 = 0, d1_1 = 0, pct1 = 0;
    ncclResult_t ret = ncclGetDtree(2, 0, &s0, &d0_0, &d0_1, &pct0,
                                             &s1, &d1_0, &d1_1, &pct1);
    EXPECT_EQ(ret, ncclSuccess);
    // Tree 0: rank 0 is root (nranks=2, even: mirror)
    EXPECT_EQ(s0, -1);  // no parent
    EXPECT_EQ(d0_1, 1); // right child
    // Tree 1: mirror → rank nranks-1-0 = 1 is root in mirrored btree
    // mirrored: s1 = nranks-1-u, d1_0 = nranks-1-d0, d1_1 = nranks-1-d1
    // btree for nranks=2, rank=1: u=0, d0=-1, d1=-1
    // mirror of rank 1: s1=nranks-1-0=1 (parent in mirror), d1_0=-1, d1_1=-1
    // Actually for rank 0 in mirror: btree(2, 1, ...) → u=0, d0=-1, d1=-1
    // s1=nranks-1-0=1, d1_0=-1, d1_1=-1
    EXPECT_EQ(s1, 1);   // parent in second tree
    EXPECT_EQ(d1_0, -1);
    EXPECT_EQ(d1_1, -1);
}

TEST(DtreeTest, FourRanks_Rank0) {
    int s0 = 0, d0_0 = 0, d0_1 = 0, pct0 = 0;
    int s1 = 0, d1_0 = 0, d1_1 = 0, pct1 = 0;
    ncclResult_t ret = ncclGetDtree(4, 0, &s0, &d0_0, &d0_1, &pct0,
                                             &s1, &d1_0, &d1_1, &pct1);
    EXPECT_EQ(ret, ncclSuccess);
    // Tree 0: rank 0 is root
    EXPECT_EQ(s0, -1);
    EXPECT_EQ(d0_1, 2);  // right child is 2 (from btree)

    // Tree 1 is mirror: btree(4, nranks-1-0=3, ...) gives u=2, d0=-1, d1=-1
    // mirror: s1=nranks-1-2=1, d1_0=-1, d1_1=-1
    EXPECT_EQ(s1, 1);    // parent in second tree is nranks-1-2=1
    EXPECT_EQ(d1_0, -1);
    EXPECT_EQ(d1_1, -1);
}

TEST(DtreeTest, ThreeRanks_ShiftTree) {
    // Odd nranks: second tree is a shift by 1
    // nranks=3 is odd
    int s0 = 0, d0_0 = 0, d0_1 = 0, pct0 = 0;
    int s1 = 0, d1_0 = 0, d1_1 = 0, pct1 = 0;
    ncclResult_t ret = ncclGetDtree(3, 0, &s0, &d0_0, &d0_1, &pct0,
                                             &s1, &d1_0, &d1_1, &pct1);
    EXPECT_EQ(ret, ncclSuccess);
    // Just verify it succeeds and produces valid rank indices
    auto validRank = [](int r, int nranks) {
        return r == -1 || (r >= 0 && r < nranks);
    };
    EXPECT_TRUE(validRank(s0, 3));
    EXPECT_TRUE(validRank(d0_0, 3));
    EXPECT_TRUE(validRank(d0_1, 3));
    EXPECT_TRUE(validRank(s1, 3));
    EXPECT_TRUE(validRank(d1_0, 3));
    EXPECT_TRUE(validRank(d1_1, 3));
}

TEST(DtreeTest, FourRanks_AllRanksValid) {
    const int nranks = 4;
    for (int rank = 0; rank < nranks; rank++) {
        int s0 = 0, d0_0 = 0, d0_1 = 0, pct0 = 0;
        int s1 = 0, d1_0 = 0, d1_1 = 0, pct1 = 0;
        ncclResult_t ret = ncclGetDtree(nranks, rank, &s0, &d0_0, &d0_1, &pct0,
                                                       &s1, &d1_0, &d1_1, &pct1);
        EXPECT_EQ(ret, ncclSuccess) << "rank=" << rank;

        auto validRank = [nranks](int r) {
            return r == -1 || (r >= 0 && r < nranks);
        };
        EXPECT_TRUE(validRank(s0))   << "rank=" << rank << " s0=" << s0;
        EXPECT_TRUE(validRank(d0_0)) << "rank=" << rank << " d0_0=" << d0_0;
        EXPECT_TRUE(validRank(d0_1)) << "rank=" << rank << " d0_1=" << d0_1;
        EXPECT_TRUE(validRank(s1))   << "rank=" << rank << " s1=" << s1;
        EXPECT_TRUE(validRank(d1_0)) << "rank=" << rank << " d1_0=" << d1_0;
        EXPECT_TRUE(validRank(d1_1)) << "rank=" << rank << " d1_1=" << d1_1;
    }
}

} // namespace RcclUnitTesting
