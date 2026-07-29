// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"

#include "rocjitsu/code/patch/instruction_builder.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <ranges>
#include <unordered_set>
#include <vector>

namespace rocjitsu {
namespace {

TEST(ConSanBranchOnlyRelayRouter, PlansCommitsEmitsAndRecordsTypedRelayOwnership) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint64_t kEntrySource = 0u;
  constexpr uint64_t kEntryTarget = 300'000u;
  constexpr uint64_t kReturnSource = kEntryTarget + sizeof(uint32_t);
  constexpr uint64_t kReturnTarget = sizeof(uint32_t);
  constexpr uint64_t kTextSize = kReturnSource + sizeof(uint32_t);
  constexpr uint64_t kPristineRelay = 100'000u;
  constexpr uint64_t kOwnedAnchorRelay = 200'000u;
  constexpr uint64_t kGeneratedRelay = 200'004u;
  constexpr uint64_t kOwnedReservoirRelay = 100'004u;

  BranchOnlyRelayRouter router;
  // Discovery order must not affect deterministic coordinate-based routing.
  EXPECT_TRUE(router.offer(kOwnedReservoirRelay, BranchOnlyRelayProvenance::OwnedReservoir));
  EXPECT_TRUE(router.offer(kGeneratedRelay, BranchOnlyRelayProvenance::GeneratedBank));
  EXPECT_TRUE(router.offer(kPristineRelay, BranchOnlyRelayProvenance::PristineNop));
  EXPECT_TRUE(router.offer(kOwnedAnchorRelay, BranchOnlyRelayProvenance::OwnedAnchor));
  EXPECT_FALSE(router.offer(kPristineRelay, BranchOnlyRelayProvenance::GeneratedBank));

  DbiPatchPlacementPlanner planner(kArch, kTextSize);
  BranchOnlyRelayPlanFailure failure = BranchOnlyRelayPlanFailure::Reservation;
  std::string error;
  const auto route = router.plan_pair(planner, kEntrySource, kEntryTarget, kReturnSource,
                                      kReturnTarget, &error, &failure);
  ASSERT_TRUE(route) << error;
  EXPECT_EQ(failure, BranchOnlyRelayPlanFailure::None);
  EXPECT_EQ(route->entry_relay_offsets, (std::vector<uint64_t>{kPristineRelay, kGeneratedRelay}));
  EXPECT_EQ(route->return_relay_offsets,
            (std::vector<uint64_t>{kOwnedAnchorRelay, kOwnedReservoirRelay}));
  ASSERT_EQ(route->claims.size(), 4u);

  // Only pristine capacity needs a new placement reservation. Generated
  // banks are offered after their appended storage is already owned.
  ASSERT_EQ(planner.occupied_ranges().size(), 1u);
  EXPECT_EQ(planner.occupied_ranges()[0],
            (std::pair<uint64_t, uint64_t>{kPristineRelay, kPristineRelay + sizeof(uint32_t)}));

  BranchOnlyDirectRelayReservoirSet reservoirs;
  reservoirs.reservoirs.resize(1u);
  reservoirs.reservoir_by_relay.emplace(kOwnedReservoirRelay, 0u);
  const std::array invalid_reservoir_claims = {
      BranchOnlyRelayClaim{kOwnedReservoirRelay, BranchOnlyRelayProvenance::OwnedReservoir},
      BranchOnlyRelayClaim{kOwnedReservoirRelay + sizeof(uint32_t),
                           BranchOnlyRelayProvenance::OwnedReservoir},
  };
  EXPECT_FALSE(reservoirs.mark_claims_used(invalid_reservoir_claims, &error));
  EXPECT_FALSE(reservoirs.reservoirs.front().used);
  ASSERT_TRUE(reservoirs.mark_claims_used(route->claims, &error)) << error;
  EXPECT_TRUE(reservoirs.reservoirs.front().used);

  ASSERT_TRUE(router.commit(*route, &error)) << error;
  EXPECT_EQ(router.available_count(), 0u);
  EXPECT_FALSE(router.commit(*route, &error));

  std::vector<uint8_t> text(kTextSize);
  std::vector<ConSanPatchInfo> patches;
  ASSERT_TRUE(BranchOnlyRelayRouter::emit_and_record(text, *route, kEntryTarget, kReturnTarget,
                                                     kArch, patches, &error))
      << error;

  const auto expect_route = [&](std::span<const uint64_t> relays, uint64_t target) {
    for (size_t index = 0u; index < relays.size(); ++index) {
      const uint64_t source = relays[index];
      const uint64_t hop_target = index + 1u < relays.size() ? relays[index + 1u] : target;
      const auto delta = compute_sopp_branch_simm16(source, hop_target);
      ASSERT_TRUE(delta);
      uint32_t emitted = 0u;
      std::memcpy(&emitted, text.data() + source, sizeof(emitted));
      EXPECT_EQ(emitted, build_s_branch(*delta, kArch));
    }
  };
  expect_route(route->entry_relay_offsets, kEntryTarget);
  expect_route(route->return_relay_offsets, kReturnTarget);

  ASSERT_EQ(patches.size(), 2u);
  const auto pristine = std::ranges::find(patches, kPristineRelay, &ConSanPatchInfo::anchor_offset);
  ASSERT_NE(pristine, patches.end());
  EXPECT_EQ(pristine->kind, ConSanPatchKind::TrampolineNopBranchRelay);
  EXPECT_EQ(pristine->original_size, sizeof(uint32_t));
  const auto generated =
      std::ranges::find(patches, kGeneratedRelay, &ConSanPatchInfo::anchor_offset);
  ASSERT_NE(generated, patches.end());
  EXPECT_EQ(generated->kind, ConSanPatchKind::TrampolineBranchRelayReservoir);
  EXPECT_EQ(generated->original_size, 0u);
}

TEST(ConSanBranchOnlyRelayRouter, ReportsWhichHalfOfPairedRoutingIsUnreachable) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::PristineNop));
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::GeneratedBank));
  DbiPatchPlacementPlanner planner(kArch, 300'008u);

  BranchOnlyRelayPlanFailure failure = BranchOnlyRelayPlanFailure::None;
  std::string error;
  EXPECT_FALSE(router.plan_pair(planner, /*entry_source=*/0u, /*entry_target=*/300'000u,
                                /*return_source=*/300'004u, /*return_target=*/4u, &error,
                                &failure));
  EXPECT_EQ(failure, BranchOnlyRelayPlanFailure::ReturnRoute);
  EXPECT_NE(error.find("original continuation"), std::string::npos);
  EXPECT_TRUE(planner.occupied_ranges().empty());
}

TEST(ConSanBranchOnlyRelayRouter, PlansMultipleDisjointEntryAndReturnPairsAtomically) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay :
       {100'000u, 100'004u, 100'008u, 100'012u, 200'000u, 200'004u, 200'008u, 200'012u}) {
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir));
  }
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 0u,
          .entry_target = 300'000u,
          .return_source = 300'004u,
          .return_target = 4u,
      },
      BranchOnlyRelayPairRequest{
          .entry_source = 8u,
          .entry_target = 300'008u,
          .return_source = 300'012u,
          .return_target = 12u,
      },
  };
  DbiPatchPlacementPlanner planner(kArch, 300'016u);
  std::string error;

  BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  ASSERT_TRUE(plan.complete()) << error;
  ASSERT_EQ(plan.routes.size(), requests.size());
  std::unordered_set<uint64_t> claims;
  for (const BranchOnlyRelayRoute &route : plan.routes) {
    EXPECT_FALSE(route.entry_relay_offsets.empty());
    EXPECT_FALSE(route.return_relay_offsets.empty());
    for (const BranchOnlyRelayClaim &claim : route.claims)
      EXPECT_TRUE(claims.insert(claim.offset).second);
  }
  EXPECT_EQ(claims.size(), 8u);
  EXPECT_TRUE(planner.occupied_ranges().empty());
  EXPECT_TRUE(router.commit(plan.routes, &error)) << error;
  EXPECT_EQ(router.available_count(), 0u);
}

} // namespace
} // namespace rocjitsu
