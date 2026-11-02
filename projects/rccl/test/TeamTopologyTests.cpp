/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
// Cft/CftMultimem/RankToLsa/RankToWorld in core.cc have no error channel, so a degenerate team reaches device kernels.
#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

// Host prototypes for the team API (host.h since NCCL 2.32); core__funcs.h carries only the __CUDACC__ device twins.
#include "nccl_device/core.h"
#include "nccl_device/host.h"

// ncclTeamOuterFactor is NCCL_HOST_DEVICE_INLINE and sits outside every __CUDACC__ guard, so it is callable here.
#include "nccl_device/impl/core__funcs.h"

#include "common/EnvVars.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {

namespace {

// On a 1-rank comm every team size collapses to 1, so the single-rank tests are shape pins, not behavior pins.
constexpr int kSingleRank = 1;
constexpr int kDualRank = 2;

// Enumerators are 0..2, so 3 is the only non-enumerator still inside the type's [0,3] range; 99 would be UB.
constexpr ncclCftTeamMode_t kInvalidCftMode = static_cast<ncclCftTeamMode_t>(3);

// Non-zero rank and non-unit stride exercise the (rank - team.rank) * team.stride term; the base term is 0 at 1 rank.
constexpr ncclTeam_t kOffsetTeam{/*nRanks=*/4, /*rank=*/2, /*stride=*/3};

// Opt-in CI strictness: when set, a host with too few GPUs must fail the dual-rank test instead of silently skipping.
constexpr const char* kRequireMultiGpuEnvVar = "UT_TEAM_TOPOLOGY_REQUIRE_MULTI_GPU";

// Every slot is either nullptr or a comm that must be destroyed, so the destructor is safe on any exit path.
struct CommResources {
  explicit CommResources(int rankCount) : comms(static_cast<size_t>(rankCount), nullptr) {}

  ~CommResources() {
    for (ncclComm_t comm : comms) {
      if (comm != nullptr) {
        (void)ncclCommDestroy(comm);
      }
    }
  }

  std::vector<ncclComm_t> comms;
};

// One row of a rank-map table: the caller-built team and the team rank fed in, and the mapped rank expected out.
struct TeamCase {
  const char* name;
  ncclTeam_t team;
  int rank;
  int expected;
};

// One row of the ncclTeamOuterFactor table: the parent team and inner size fed in, and the factored team expected out.
struct FactorCase {
  const char* name;
  ncclTeam_t parent;
  int innerSize;
  ncclTeam_t expected;
};

// Never call hipGetDeviceCount() here: HIP state does not survive TestBed's fork(), killing every later suite.
int getDetectedGpuCount() {
  static const int detectedGpus = EnvVars().GetNumDetectedGpus();
  return detectedGpus;
}

// True only in a process re-exec'd by ProcessIsolatedTestRunner, which sets this marker before execv().
bool isIsolatedChild() {
  return std::getenv(ProcessIsolatedTestRunner::kReexecMarkerEnvVar) != nullptr;
}

void expectTeamEquals(const ncclTeam_t& team, int nRanks, int rank, int stride) {
  EXPECT_EQ(team.nRanks, nRanks);
  EXPECT_EQ(team.rank, rank);
  EXPECT_EQ(team.stride, stride);
}

// Loopback pins the single-node bootstrap; IB is off because none of these calls move data.
ProcessIsolatedTestRunner::TestConfig makeTeamTopologyConfig(const std::string& name, std::function<void()> testFn,
                                                             int numGpus) {
  // numGpus is inert here: RUN_ISOLATED_TESTS runs sequentially, so the child just inherits the parent's devices.
  return ProcessIsolatedTestRunner::TestConfig(name, testFn)
    .withEnvironment({
      {"NCCL_SOCKET_IFNAME", "lo"},
      {"NCCL_IB_DISABLE", "1"}
    })
    .withTimeout(std::chrono::seconds(60))
    .withNumGpus(static_cast<size_t>(numGpus));
}

// A 1-rank comm plus the golden LSA team, which proves ncclDevrInitOnce ran before the pins below.
void initSingleRankCommAndCheckLsa(CommResources& resources) {
  ASSERT_EQ(ncclCommInitAll(resources.comms.data(), kSingleRank, nullptr), ncclSuccess);
  const ncclTeam_t lsaTeam = ncclTeamLsa(resources.comms[0]);
  expectTeamEquals(lsaTeam, 1, 0, 1);
}

// Observed, not endorsed: CUDA_VERSION is never defined in RCCL, so gpuCftSupport is 0 and cftSize/cftMcSize stay 1.
void runCftMultimemTest() {
  CommResources resources(kSingleRank);
  ASSERT_NO_FATAL_FAILURE(initSingleRankCommAndCheckLsa(resources));
  expectTeamEquals(ncclTeamCftMultimem(resources.comms[0]), 1, 0, 1);
}

// Observed, not endorsed: HIER_MULTIMEM factors the flat team by cftMcSize, which is 1, so the factor is the identity.
void runCftHierMultimemTest() {
  CommResources resources(kSingleRank);
  ASSERT_NO_FATAL_FAILURE(initSingleRankCommAndCheckLsa(resources));
  expectTeamEquals(ncclTeamCft(resources.comms[0], NCCL_CFT_TEAM_FLAT), 1, 0, 1);
  expectTeamEquals(ncclTeamCft(resources.comms[0], NCCL_CFT_TEAM_HIER_MULTIMEM), 1, 0, 1);
}

// stride 0 collapses the rank map onto the caller's own world rank; kOffsetTeam then pins the non-degenerate map.
void runCftZeroTeamAliasesSelfTest() {
  CommResources resources(kSingleRank);
  ASSERT_NO_FATAL_FAILURE(initSingleRankCommAndCheckLsa(resources));

  int worldRank = -1;
  ASSERT_EQ(ncclCommUserRank(resources.comms[0], &worldRank), ncclSuccess);

  const ncclTeam_t zeroTeam = ncclTeamCft(resources.comms[0], kInvalidCftMode);
  expectTeamEquals(zeroTeam, 0, 0, 0);

  // Map is comm->rank + (rank - team.rank) * team.stride: zeroTeam aliases every input, kOffsetTeam only at team.rank.
  const std::vector<TeamCase> cases = {
    {"first_member", zeroTeam, 0, worldRank},
    {"second_member", zeroTeam, 1, worldRank},
    {"far_member", zeroTeam, 5, worldRank},
    {"negative_member", zeroTeam, -3, worldRank},
    {"offset_self", kOffsetTeam, 2, worldRank},
    {"offset_next", kOffsetTeam, 3, worldRank + 3},
    {"offset_far", kOffsetTeam, 5, worldRank + 9},
    {"offset_below", kOffsetTeam, 0, worldRank - 6},
  };
  for (const TeamCase& c : cases) {
    SCOPED_TRACE(c.name);
    EXPECT_EQ(ncclTeamRankToWorld(resources.comms[0], c.team, c.rank), c.expected);
  }
}

// ncclTeamRankToLsa is unclamped: lsaSelf + (rank - team.rank) * team.stride, and lsaSelf is 0 on a 1-rank comm.
void runRankToLsaNoBoundsCheckTest() {
  CommResources resources(kSingleRank);
  ASSERT_NO_FATAL_FAILURE(initSingleRankCommAndCheckLsa(resources));

  const ncclTeam_t stridedTeam{/*nRanks=*/2, /*rank=*/0, /*stride=*/2};
  const std::vector<TeamCase> cases = {
    {"in_range", stridedTeam, 0, 0},
    {"one_past_end", stridedTeam, 1, 2},
    {"far_past_end", stridedTeam, 3, 6},
    {"below_zero", stridedTeam, -2, -4},
    {"far_out_of_range", stridedTeam, 100, 200},
    {"offset_self", kOffsetTeam, 2, 0},
    {"offset_next", kOffsetTeam, 3, 3},
    {"offset_far", kOffsetTeam, 5, 9},
    {"offset_below", kOffsetTeam, 0, -6},
    {"offset_negative", kOffsetTeam, -1, -9},
  };
  for (const TeamCase& c : cases) {
    SCOPED_TRACE(c.name);
    EXPECT_EQ(ncclTeamRankToLsa(resources.comms[0], c.team, c.rank), c.expected);
  }
}

// Known defect, pinned red-on-fix: cftSize is stuck at 1, so HIER_LSA computes 1 / lsaSize and truncates to 0.
void runCftHierLsaMultiRankDefectTest() {
  // Never add a GTEST_SKIP() to this body: the harness scores a skipped isolated child as a passed parent test.
  CommResources resources(kDualRank);
  ASSERT_EQ(ncclCommInitAll(resources.comms.data(), kDualRank, nullptr), ncclSuccess);

  for (int rank = 0; rank < kDualRank; ++rank) {
    SCOPED_TRACE("rank " + std::to_string(rank));
    const ncclTeam_t lsaTeam = ncclTeamLsa(resources.comms[rank]);
    EXPECT_EQ(lsaTeam.nRanks, kDualRank) << "two GPUs on one node share one LSA domain";

    // cftMcSize stays 1 while lsaSize is 2, so CftMultimem must stay a singleton and not track the LSA team.
    expectTeamEquals(ncclTeamCftMultimem(resources.comms[rank]), 1, 0, 1);

    // FLAT reads cftSize and HIER_MULTIMEM factors by cftMcSize, both 1 here, so neither may track lsaSize, which is 2.
    expectTeamEquals(ncclTeamCft(resources.comms[rank], NCCL_CFT_TEAM_FLAT), 1, 0, 1);
    expectTeamEquals(ncclTeamCft(resources.comms[rank], NCCL_CFT_TEAM_HIER_MULTIMEM), 1, 0, 1);

    // Only a rank-0 team leaves the base of base + (r - team.rank) * 1 exposed, so these pin lsaSelf and comm->rank.
    EXPECT_EQ(ncclTeamRankToLsa(resources.comms[rank], ncclTeam_t{/*nRanks=*/2, /*rank=*/0, /*stride=*/1}, 0), rank);
    EXPECT_EQ(ncclTeamRankToWorld(resources.comms[rank], ncclTeam_t{/*nRanks=*/2, /*rank=*/0, /*stride=*/1}, 0), rank);

    const ncclTeam_t hierLsa = ncclTeamCft(resources.comms[rank], NCCL_CFT_TEAM_HIER_LSA);
    EXPECT_EQ(hierLsa.nRanks, 0) << "A non-zero nRanks means the cftSize / lsaSize truncation in ncclTeamCft was "
                                 << "fixed. Invert this pin into a real assertion on the outer-factor team, drop "
                                 << "the defect note above, and re-check the FLAT and HIER_MULTIMEM tests, which "
                                 << "also assume cftSize is stuck at 1.";
    // stride proves HIER_LSA ran rather than the invalid-mode branch, which yields nRanks 0 with stride 0.
    EXPECT_EQ(hierLsa.stride, lsaTeam.nRanks);
  }
}

} // namespace

// Pure arithmetic on an inline header helper: no comm, no GPU and no isolated child, so it pins the map on any box.
TEST(TeamTopologyTests, OuterFactor_DividesRanksAndRankAndMultipliesStride) {
  const std::vector<FactorCase> cases = {
    {"rank_not_multiple_of_inner", {/*nRanks=*/8, /*rank=*/5, /*stride=*/1}, 4, {2, 1, 4}},
    {"strided_parent", {/*nRanks=*/12, /*rank=*/7, /*stride=*/2}, 3, {4, 2, 6}},
    {"inner_size_one_is_identity", {/*nRanks=*/4, /*rank=*/3, /*stride=*/2}, 1, {4, 3, 2}},
    {"singleton_parent_truncates_to_empty", {/*nRanks=*/1, /*rank=*/0, /*stride=*/1}, 8, {0, 0, 8}},
  };
  for (const FactorCase& c : cases) {
    SCOPED_TRACE(c.name);
    const ncclTeam_t ans = ncclTeamOuterFactor(c.parent, c.innerSize);
    expectTeamEquals(ans, c.expected.nRanks, c.expected.rank, c.expected.stride);
  }
}

TEST(TeamTopologyTests, CftMultimem_ReturnsDegenerateSingletonTeam) {
  RUN_ISOLATED_TESTS(makeTeamTopologyConfig("TeamTopologyTests.CftMultimem_ReturnsDegenerateSingletonTeam",
                                            []() { runCftMultimemTest(); }, kSingleRank));
}

TEST(TeamTopologyTests, Cft_HierMultimem_MatchesFlatWhenCftUnsupported) {
  RUN_ISOLATED_TESTS(makeTeamTopologyConfig("TeamTopologyTests.Cft_HierMultimem_MatchesFlatWhenCftUnsupported",
                                            []() { runCftHierMultimemTest(); }, kSingleRank));
}

TEST(TeamTopologyTests, RankToWorld_ZeroTeamAliasesSelf_OffsetTeamMapsLinearly) {
  RUN_ISOLATED_TESTS(makeTeamTopologyConfig("TeamTopologyTests.RankToWorld_ZeroTeamAliasesSelf_OffsetTeamMapsLinearly",
                                            []() { runCftZeroTeamAliasesSelfTest(); }, kSingleRank));
}

TEST(TeamTopologyTests, RankToLsa_NoBoundsCheck_ExtrapolatesOutOfRange) {
  RUN_ISOLATED_TESTS(makeTeamTopologyConfig("TeamTopologyTests.RankToLsa_NoBoundsCheck_ExtrapolatesOutOfRange",
                                            []() { runRankToLsaNoBoundsCheckTest(); }, kSingleRank));
}

// Gate in the parent only: EnvVars reports 0 GPUs in a re-exec'd child, and a skipped child is scored as a pass.
TEST(TeamTopologyTests, Cft_HierLsa_MultiRank_ReturnsEmptyTeam_PinsKnownDefect) {
  if (!isIsolatedChild() && getDetectedGpuCount() < kDualRank) {
    // CI sets the strictness variable so this coverage cannot vanish into a green run on a 1-GPU node.
    if (std::getenv(kRequireMultiGpuEnvVar) != nullptr) {
      FAIL() << kRequireMultiGpuEnvVar << " is set, but this host reports " << getDetectedGpuCount()
             << " visible GPU(s) and this test needs at least " << kDualRank << " to make lsaSize exceed 1. "
             << "Run it on a multi-GPU node, or unset the variable to allow the skip.";
    }
    GTEST_SKIP() << "This test requires at least 2 visible GPUs to make lsaSize exceed 1.";
  }
  // Only bites on the symmetric path: the ROCm non-symmetric path overwrites lsaSize with comm->localRanks anyway.
  RUN_ISOLATED_TESTS(makeTeamTopologyConfig("TeamTopologyTests.Cft_HierLsa_MultiRank_ReturnsEmptyTeam_PinsKnownDefect",
                                            []() { runCftHierLsaMultiRankDefectTest(); }, kDualRank)
                       .clearVariable("NCCL_LSA_TEAM_SIZE"));
}

} // namespace RcclUnitTesting
