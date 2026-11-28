/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for ncclCheckDdaNranksRelaxConsensus(), which guards against the
// hang RCCL_DDA_NRANKS_RELAX can otherwise cause: it is read per process, so a
// launcher that sets it on only some ranks of a communicator would previously
// have some ranks enter ncclDdaIpcCommInit()'s bootstrap allgather while the
// rest return early -- the entering ranks then block forever. This is called
// once every rank has unconditionally populated its slot in the bootstrap-
// exchanged allGather3Data array in init.cc, mirroring
// rcclCheckRomeTopoModelIdxConsensus (see RomeTopoConsensusTests.cpp).

#include "algorithms/dda/ipc/dda_nranks_relax_consensus.h"
#include "gtest/gtest.h"
#include <array>

namespace RcclUnitTesting {

TEST(DdaNranksRelaxConsensus, EmptyRanks) {
  EXPECT_EQ(ncclCheckDdaNranksRelaxConsensus(
      0,
      [](int) { return false; },
      [](int) { return ""; }),
    ncclSuccess);
}

TEST(DdaNranksRelaxConsensus, AllAgreeEnabled) {
  constexpr int n = 8;
  std::array<bool, n> relax{{true, true, true, true, true, true, true, true}};
  std::array<const char*, n> hosts{{"h", "h", "h", "h", "h", "h", "h", "h"}};
  EXPECT_EQ(ncclCheckDdaNranksRelaxConsensus(
      n,
      [&](int r) { return relax[r]; },
      [&](int r) { return hosts[r]; }),
    ncclSuccess);
}

TEST(DdaNranksRelaxConsensus, AllAgreeDisabled) {
  constexpr int n = 4;
  std::array<bool, n> relax{{false, false, false, false}};
  std::array<const char*, n> hosts{{"h", "h", "h", "h"}};
  EXPECT_EQ(ncclCheckDdaNranksRelaxConsensus(
      n,
      [&](int r) { return relax[r]; },
      [&](int r) { return hosts[r]; }),
    ncclSuccess);
}

// The case this function exists to catch: one rank set the variable, the rest
// did not. Must fail closed (not silently pick a value) since letting init
// proceed here is exactly the divergence that produces the hang.
TEST(DdaNranksRelaxConsensus, OneRankDisagreesFails) {
  constexpr int n = 4;
  std::array<bool, n> relax{{true, false, false, false}};
  std::array<const char*, n> hosts{{"h0", "h1", "h2", "h3"}};
  EXPECT_EQ(ncclCheckDdaNranksRelaxConsensus(
      n,
      [&](int r) { return relax[r]; },
      [&](int r) { return hosts[r]; }),
    ncclInvalidUsage);
}

// Same as above with the reference (rank 0) on the minority side, to confirm
// disagreement is detected relative to rank 0 specifically, not by majority
// vote -- unlike the Rome topology check, there is no "winning" value to elect
// here, since every non-zero-vs-zero split is equally a misconfiguration.
TEST(DdaNranksRelaxConsensus, MinorityAtRankZeroStillFails) {
  constexpr int n = 4;
  std::array<bool, n> relax{{false, true, true, true}};
  std::array<const char*, n> hosts{{"h0", "h1", "h2", "h3"}};
  EXPECT_EQ(ncclCheckDdaNranksRelaxConsensus(
      n,
      [&](int r) { return relax[r]; },
      [&](int r) { return hosts[r]; }),
    ncclInvalidUsage);
}

} // namespace RcclUnitTesting
