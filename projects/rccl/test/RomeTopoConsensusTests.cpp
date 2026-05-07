/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "graph/rome_topo_consensus.h"
#include "gtest/gtest.h"
#include <array>

namespace RcclUnitTesting {

TEST(RomeTopoConsensus, emptyRanks) {
  EXPECT_EQ(rcclCheckRomeTopoModelIdxConsensus(
      0,
      [](int) { return 0; },
      [](int) { return ""; },
      [](int) { return 0ULL; }),
    ncclSuccess);
}

TEST(RomeTopoConsensus, allSameModelIdx) {
  constexpr int n = 4;
  std::array<int, n> idx{{7, 7, 7, 7}};
  std::array<const char*, n> names{{"a", "a", "b", "b"}};
  std::array<uint64_t, n> hosts{{1, 1, 2, 2}};
  EXPECT_EQ(rcclCheckRomeTopoModelIdxConsensus(
      n,
      [&](int r) { return idx[r]; },
      [&](int r) { return names[r]; },
      [&](int r) { return hosts[r]; }),
    ncclSuccess);
}

TEST(RomeTopoConsensus, mismatchFails) {
  constexpr int n = 2;
  std::array<int, n> idx{{1, 2}};
  std::array<const char*, n> names{{"h0", "h1"}};
  std::array<uint64_t, n> hosts{{10, 11}};
  EXPECT_EQ(rcclCheckRomeTopoModelIdxConsensus(
      n,
      [&](int r) { return idx[r]; },
      [&](int r) { return names[r]; },
      [&](int r) { return hosts[r]; }),
    ncclInvalidUsage);
}

TEST(RomeTopoConsensus, tieBreakLowerFirstRankWinsThenMinorityFails) {
  // idx 2 at ranks 0,1; idx 3 at ranks 2,3 -> plurality tie; first occurrence rank 0 -> refIdx 2
  constexpr int n = 4;
  std::array<int, n> idx{{2, 2, 3, 3}};
  std::array<const char*, n> names{{"a", "a", "a", "a"}};
  std::array<uint64_t, n> hosts{{1, 1, 1, 1}};
  EXPECT_EQ(rcclCheckRomeTopoModelIdxConsensus(
      n,
      [&](int r) { return idx[r]; },
      [&](int r) { return names[r]; },
      [&](int r) { return hosts[r]; }),
    ncclInvalidUsage);
}

} // namespace RcclUnitTesting
