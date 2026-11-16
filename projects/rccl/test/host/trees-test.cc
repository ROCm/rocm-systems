/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/graph/trees.cc: the binary tree and the double binary tree over node indexes.

#include <gtest/gtest.h>

#include <vector>

#include TREES_CC_PATH

namespace {

struct TreeNode {
  int up = -9, down0 = -9, down1 = -9, childType = -9;
};

TreeNode Btree(int nranks, int rank) {
  TreeNode n;
  EXPECT_EQ(ncclSuccess, ncclGetBtree(nranks, rank, &n.up, &n.down0, &n.down1, &n.childType));
  return n;
}

struct DoubleTree {
  TreeNode t0, t1;
};

DoubleTree Dtree(int nranks, int rank) {
  DoubleTree d;
  EXPECT_EQ(ncclSuccess, ncclGetDtree(nranks, rank, &d.t0.up, &d.t0.down0, &d.t0.down1, &d.t0.childType, &d.t1.up,
                                      &d.t1.down0, &d.t1.down1, &d.t1.childType));
  return d;
}

TEST(TreesMicrotest, GetBtree_MatchesTheFourteenRankPicture) {
  const int expected[14][3] = {{-1, -1, 8}, {2, -1, -1}, {4, 1, 3},  {2, -1, -1},  {8, 2, 6},
                               {6, -1, -1}, {4, 5, 7},   {6, -1, -1}, {0, 4, 12},  {10, -1, -1},
                               {12, 9, 11}, {10, -1, -1}, {8, 10, 13}, {12, -1, -1}};

  for (int r = 0; r < 14; r++) {
    TreeNode n = Btree(14, r);
    EXPECT_EQ(expected[r][0], n.up) << r;
    EXPECT_EQ(expected[r][1], n.down0) << r;
    EXPECT_EQ(expected[r][2], n.down1) << r;
  }
}

TEST(TreesMicrotest, GetBtree_ChildTypeSaysWhichSlotOfTheParent) {
  EXPECT_EQ(0, Btree(14, 4).childType);  // 8's first child
  EXPECT_EQ(1, Btree(14, 12).childType);
  EXPECT_EQ(1, Btree(14, 8).childType);  // the root only has a second child
  EXPECT_EQ(1, Btree(14, 13).childType);
  EXPECT_EQ(-9, Btree(14, 0).childType);  // the root has no parent to describe
}

TEST(TreesMicrotest, GetBtree_SingleRankIsAlone) {
  TreeNode n = Btree(1, 0);

  EXPECT_EQ(-1, n.up);
  EXPECT_EQ(-1, n.down0);
  EXPECT_EQ(-1, n.down1);
}

TEST(TreesMicrotest, GetBtree_EveryRankHangsOffTheRootOnce) {
  for (int nranks = 2; nranks <= 70; nranks++) {
    std::vector<int> seen(nranks, 0);
    for (int r = 0; r < nranks; r++) {
      TreeNode n = Btree(nranks, r);
      for (int child : {n.down0, n.down1}) {
        if (child == -1) continue;
        ASSERT_LT(child, nranks) << nranks << "/" << r;
        seen[child]++;
        TreeNode c = Btree(nranks, child);
        EXPECT_EQ(r, c.up) << nranks << "/" << child;
        EXPECT_EQ(child == n.down1 ? 1 : 0, c.childType) << nranks << "/" << child;
      }
    }
    EXPECT_EQ(0, seen[0]) << nranks;
    for (int r = 1; r < nranks; r++) EXPECT_EQ(1, seen[r]) << nranks << "/" << r;
  }
}

TEST(TreesMicrotest, GetDtree_EvenCountMirrorsTheSecondTree) {
  for (int r = 0; r < 12; r++) {
    DoubleTree d = Dtree(12, r);
    TreeNode m = Btree(12, 11 - r);
    auto mirror = [](int x) { return x == -1 ? -1 : 11 - x; };

    EXPECT_EQ(mirror(m.up), d.t1.up) << r;
    EXPECT_EQ(mirror(m.down0), d.t1.down0) << r;
    EXPECT_EQ(mirror(m.down1), d.t1.down1) << r;
    EXPECT_EQ(Btree(12, r).up, d.t0.up) << r;
  }
  DoubleTree root = Dtree(12, 11);
  EXPECT_EQ(-1, root.t1.up);
  EXPECT_EQ(3, root.t1.down1);
}

TEST(TreesMicrotest, GetDtree_OddCountShiftsTheSecondTreeByOne) {
  DoubleTree one = Dtree(13, 1);
  EXPECT_EQ(-1, one.t1.up);  // rank 1 is the second root
  EXPECT_EQ(9, one.t1.down1);

  DoubleTree zero = Dtree(13, 0);
  EXPECT_EQ(9, zero.t1.up);
  EXPECT_EQ(11, zero.t1.down0);
  EXPECT_EQ(-1, zero.t1.down1);
  EXPECT_EQ(Btree(13, 12).childType, zero.t1.childType);
}

DoubleTree DtreeCompact(int nranks, int rank) {
  DoubleTree d;
  EXPECT_EQ(ncclSuccess, ncclGetDtreeCompact(nranks, rank, &d.t0.up, &d.t0.down0, &d.t0.down1, &d.t0.childType,
                                             &d.t1.up, &d.t1.down0, &d.t1.down1, &d.t1.childType));
  return d;
}

TEST(TreesMicrotest, GetDtreeCompact_MatchesTheEightNodePicture) {
  const int expected[8][3] = {{-1, -1, 2}, {2, 4, 5}, {0, 1, 3}, {2, 6, 7},
                              {1, -1, -1}, {1, -1, -1}, {3, -1, -1}, {3, -1, -1}};

  for (int r = 0; r < 8; r++) {
    TreeNode n = DtreeCompact(8, r).t0;
    EXPECT_EQ(expected[r][0], n.up) << r;
    EXPECT_EQ(expected[r][1], n.down0) << r;
    EXPECT_EQ(expected[r][2], n.down1) << r;
  }
}

TEST(TreesMicrotest, GetDtreeCompact_FirstTreeSpineStaysInTheFirstHalf) {
  for (int nranks = 1; nranks <= 33; nranks++) {
    int half = (nranks + 1) / 2;
    std::vector<int> ups(nranks, 0);
    for (int r = 0; r < nranks; r++) {
      DoubleTree d = DtreeCompact(nranks, r);
      for (int child : {d.t0.down0, d.t0.down1}) {
        if (child == -1) continue;
        EXPECT_LT(r, half) << nranks << "/" << r;  // only the first half has children
        ASSERT_LT(child, nranks);
        EXPECT_EQ(r, DtreeCompact(nranks, child).t0.up) << nranks << "/" << child;
      }
      for (int up : {d.t0.up, d.t1.up}) {
        if (up != -1) ups[r]++;
      }
    }
    for (int r = 0; r < nranks; r++) EXPECT_GE(ups[r], nranks > 1 ? 1 : 0) << nranks << "/" << r;
  }
  EXPECT_EQ(-1, DtreeCompact(1, 0).t0.up);
}

}  // namespace
