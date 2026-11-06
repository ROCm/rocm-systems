/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for the hierarchical alltoall peer split in
// alltoall_hier_policy.h. No GPU or MPI required.

#include <gtest/gtest.h>

#include "alltoall_hier_policy.h"

using alltoall_hier::netCtaCount;
using alltoall_hier::peerCount;
using alltoall_hier::peerUsesNet;

namespace {

// Ranks are laid out as consecutive LSA teams of lsaSize, one per node.
int lsaBaseOf(int rank, int lsaSize) { return rank / lsaSize * lsaSize; }

bool usesNet(int peer, int rank, int lsaSize, int groupSize) {
  return peerUsesNet(peer, rank, lsaBaseOf(rank, lsaSize), lsaSize, groupSize);
}

// Puts rank receives on one DevComm, from each sender's own view of the split.
int sentTo(bool net, int rank, int nRanks, int lsaSize, int groupSize) {
  int n = 0;
  for (int s = 0; s < nRanks; s++) n += usesNet(rank, s, lsaSize, groupSize) == net;
  return n;
}

}  // namespace

TEST(AllToAllHierPolicy, NetCtaCount) {
  EXPECT_EQ(netCtaCount(2), 1);
  EXPECT_EQ(netCtaCount(3), 1);
  EXPECT_EQ(netCtaCount(16), 8);
}

TEST(AllToAllHierPolicy, SplitIsSymmetric) {
  for (int nodes = 1; nodes <= 3; nodes++) {
    for (int lsaSize = 1; lsaSize <= 8; lsaSize++) {
      const int nRanks = nodes * lsaSize;
      for (int groupSize = 0; groupSize <= lsaSize + 1; groupSize++) {
        for (int a = 0; a < nRanks; a++) {
          for (int b = 0; b < nRanks; b++) {
            EXPECT_EQ(usesNet(b, a, lsaSize, groupSize), usesNet(a, b, lsaSize, groupSize))
                << "nRanks " << nRanks << " lsaSize " << lsaSize << " groupSize " << groupSize
                << " ranks " << a << " and " << b;
          }
        }
      }
    }
  }
}

TEST(AllToAllHierPolicy, WaitCountMatchesPutsReceived) {
  for (int nodes = 1; nodes <= 3; nodes++) {
    for (int lsaSize = 1; lsaSize <= 8; lsaSize++) {
      const int nRanks = nodes * lsaSize;
      for (int groupSize = 0; groupSize <= lsaSize + 1; groupSize++) {
        for (int rank = 0; rank < nRanks; rank++) {
          const int base = lsaBaseOf(rank, lsaSize);
          for (bool net : {false, true}) {
            EXPECT_EQ(peerCount(net, rank, nRanks, base, lsaSize, groupSize),
                      sentTo(net, rank, nRanks, lsaSize, groupSize))
                << "nRanks " << nRanks << " lsaSize " << lsaSize << " groupSize " << groupSize
                << " rank " << rank << (net ? " net" : " sdma");
          }
          EXPECT_EQ(peerCount(false, rank, nRanks, base, lsaSize, groupSize) +
                        peerCount(true, rank, nRanks, base, lsaSize, groupSize),
                    nRanks);
        }
      }
    }
  }
}

// 8 ranks on one node with groups of 3: groups {0,1,2}, {3,4,5}, {6,7}.
TEST(AllToAllHierPolicy, PartialGroup) {
  EXPECT_EQ(peerCount(false, 0, 8, 0, 8, 3), 3);
  EXPECT_EQ(peerCount(false, 4, 8, 0, 8, 3), 3);
  EXPECT_EQ(peerCount(false, 7, 8, 0, 8, 3), 2);
  EXPECT_EQ(peerCount(true, 7, 8, 0, 8, 3), 6);
  EXPECT_FALSE(peerUsesNet(6, 7, 0, 8, 3));
  EXPECT_TRUE(peerUsesNet(5, 6, 0, 8, 3));
}

// Group size 0 keeps every LSA peer on SDMA; peers on other nodes use the network.
TEST(AllToAllHierPolicy, NoGroupsTwoNodes) {
  EXPECT_EQ(peerCount(false, 3, 16, 0, 8, 0), 8);
  EXPECT_EQ(peerCount(true, 3, 16, 0, 8, 0), 8);
  EXPECT_FALSE(peerUsesNet(7, 0, 0, 8, 0));
  EXPECT_TRUE(peerUsesNet(8, 0, 0, 8, 0));
  EXPECT_TRUE(peerUsesNet(0, 8, 8, 8, 0));
}
