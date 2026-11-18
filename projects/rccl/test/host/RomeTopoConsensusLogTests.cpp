/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Vote-outcome tests for rcclCheckRomeTopoModelIdxConsensus.
//
// These live in the host-only binary rather than next to the rest of the
// RomeTopoConsensus tests because they assert on what the function *logged*,
// and that is only observable here. rccl-HostUnitTests links the stub
// ncclDebugLog from test/host/wrapper_link_stubs.cpp, which writes WARN
// straight to stderr; rccl-UnitTests, which also compiles
// test/RomeTopoConsensusTests.cpp, links the real logger, where output goes to
// ncclDebugFile (stdout by default) and the default ERROR level suppresses WARN
// entirely. Keeping these cases out of the shared file avoids assertions that
// would depend on which logger happens to be linked.

#include "graph/rome_topo_consensus.h"
#include "common/LogCapture.hpp"
#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace RcclUnitTesting {
namespace {

// Runs one consensus check that is expected to fail, and pins *which* index won
// the vote.
//
// Asserting the return code alone would not do that:
// rcclCheckRomeTopoModelIdxConsensus reports ncclInvalidUsage for any
// disagreement, so with several groups the result is identical no matter which
// index wins -- a broken tie-break would look exactly like a correct one. The
// emitted "voted refIdx N from K of M" is the only channel through which the
// function reports its decision, so each case pins that string.
//
// All ranks share one hostname and host hash: those feed only the per-host
// listing of mismatching ranks inside the warning, so a single host keeps the
// message deterministic and each case focused on the vote itself.
void expectMismatchWithVote(const std::vector<int>& idx, const std::string& expectedVote) {
  const int nranks = static_cast<int>(idx.size());
  ncclResult_t result = ncclSuccess;
  const std::string log = CaptureLog([&]() {
    result = rcclCheckRomeTopoModelIdxConsensus(
        nranks,
        [&](int r) { return idx[r]; },
        [](int) { return "h"; },
        [](int) { return 1ULL; });
  });
  EXPECT_EQ(result, ncclInvalidUsage);
  EXPECT_NE(log.find(expectedVote), std::string::npos)
      << "expected the warning to contain: " << expectedVote
      << "\nactual log:\n" << log;
}

}  // namespace

TEST(RomeTopoConsensusLog, threeWayVoteStrictLoserAndTieBreakLoser) {
  // Which arms of the plurality comparison on rome_topo_consensus.cc:36 each
  // case reaches depends on the order the loop walks `tallies`, a
  // std::unordered_map. That order is deterministic but not obvious: it follows
  // both `key % bucket_count` (reserve(nranks) sizes the table -- 7 buckets for
  // nranks 6 and 7, 5 for nranks 5) and the order the keys were first inserted.
  // It was established empirically per case rather than derived, so each case
  // states the order it relies on.

  // Vote counts 4 (key 1), 2 (key 2), 1 (key 8), visited as 2, 8, 1. Key 2
  // takes `cnt > refVotes` True (refVotes -1 -> 2), then key 8 has cnt 1, so
  // `cnt > refVotes` is False *and* `cnt == refVotes` is False too -- closing
  // the `cnt == refVotes` False arm.
  expectMismatchWithVote({1, 1, 1, 1, 2, 2, 8}, "voted refIdx 1 from 4 of 7");

  // Three groups tied at 2 votes, first ranks 0 (key 1), 2 (key 2), 4 (key 8).
  // Same 2, 8, 1 order: key 2 leads, key 8 ties on votes but has the higher
  // firstRank -- closing the `firstRank < refFirstRank` False arm -- and key 1
  // then satisfies both, exercising the True arm as well.
  expectMismatchWithVote({1, 1, 2, 2, 8, 8}, "voted refIdx 1 from 2 of 6");

  // Pins the strictness of the comparison itself, which the two cases above
  // cannot: in both, the group that should win is visited last, so weakening
  // `cnt > refVotes` to `cnt >= refVotes` elects the same index and goes
  // unnoticed. Here keys 2 and 1 tie at 2 votes with first ranks 1 and 3 and
  // the visit order is 2, 1, 6, so the tie must be broken by the lower first
  // rank: `>` keeps key 2, `>=` would hand it to key 1.
  expectMismatchWithVote({6, 2, 2, 1, 1}, "voted refIdx 2 from 2 of 5");
}

}  // namespace RcclUnitTesting
