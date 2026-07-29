// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <set>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace rocjitsu {
namespace {

class RelayTestInstruction : public Instruction {
public:
  RelayTestInstruction(std::string_view mnemonic, int size, uint64_t flags,
                       std::optional<int64_t> branch_delta, const uint32_t *raw)
      : Instruction(mnemonic, nullptr), branch_delta_(branch_delta) {
    size_ = size;
    flags_ = flags;
    raw_encoding_ = raw;
  }

  std::optional<int64_t> branch_offset_bytes() const override { return branch_delta_; }

private:
  std::optional<int64_t> branch_delta_;
};

class RelayTestTextSection : public Section {
public:
  RelayTestTextSection(std::unique_ptr<char[]> data, size_t size)
      : Section(".text"), data_(std::move(data)), size_(size) {}

  size_t size() const override { return size_; }
  const char *data() const override { return data_.get(); }
  uint32_t sectionHeaderNameIdx() const override { return 0u; }
  uint64_t sectionOffset() const override { return 0u; }

private:
  std::unique_ptr<char[]> data_;
  size_t size_ = 0u;
};

class RelayTestCodeObject : public CodeObject {
public:
  explicit RelayTestCodeObject(std::vector<uint32_t> words) {
    const size_t byte_size = words.size() * sizeof(uint32_t);
    image_.resize(byte_size);
    std::memcpy(image_.data(), words.data(), byte_size);
    auto text = std::make_unique<char[]>(byte_size);
    std::memcpy(text.get(), words.data(), byte_size);
    sections_.push_back(std::make_unique<RelayTestTextSection>(std::move(text), byte_size));
    text_sections_.push_back(sections_.back().get());
  }
};

constexpr uint32_t kRelayTestDonor = 0x1000u;
constexpr uint32_t kRelayTestClauseTwo = 0x1041u;
constexpr uint32_t kRelayTestEnd = 0x2000u;

constexpr BranchOnlyRelayOwnerIdentity lds_relay_owner(uint64_t value) {
  return {BranchOnlyRelayOwnerKind::LdsReservoir, value};
}

class RelayTestDecoder : public Decoder {
public:
  Instruction *decode(const rj_code_binary_inst_t *word) override {
    if (*word == kRelayTestEnd)
      return new RelayTestInstruction("s_endpgm", 4, PROGRAM_TERMINATOR, std::nullopt, word);
    if (*word == kRelayTestClauseTwo)
      return new RelayTestInstruction("s_clause", 4, 0u, std::nullopt, word);
    return new RelayTestInstruction("s_mov_b32", 4, 0u, std::nullopt, word);
  }
};

std::vector<BasicBlock *> relay_block_ptrs(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  std::vector<BasicBlock *> pointers;
  pointers.reserve(blocks.size());
  for (const std::unique_ptr<BasicBlock> &block : blocks)
    pointers.push_back(block.get());
  return pointers;
}

std::span<const uint8_t> relay_test_text(const RelayTestCodeObject &object) {
  const Section *text = object.text_sections().front();
  return {reinterpret_cast<const uint8_t *>(text->data()), text->size()};
}

bool brute_force_fixed_relay_batch(std::span<const BranchOnlyRelayPairRequest> requests,
                                   std::span<const uint64_t> relays) {
  struct Demand {
    uint64_t source = 0u;
    uint64_t target = 0u;
  };
  std::vector<Demand> demands;
  demands.reserve(2u * requests.size());
  for (const BranchOnlyRelayPairRequest &request : requests)
    demands.push_back({request.entry_source, request.entry_target});
  for (const BranchOnlyRelayPairRequest &request : requests)
    demands.push_back({request.return_source, request.return_target});

  std::vector<bool> used(relays.size(), false);
  std::function<bool(size_t)> solve_demand;
  std::function<bool(size_t, uint64_t)> route_demand;
  solve_demand = [&](size_t demand_index) {
    return demand_index == demands.size() ||
           route_demand(demand_index, demands[demand_index].source);
  };
  route_demand = [&](size_t demand_index, uint64_t cursor) {
    const Demand &demand = demands[demand_index];
    if (compute_sopp_branch_simm16(cursor, demand.target))
      return solve_demand(demand_index + 1u);
    const bool forward = demand.target > demand.source;
    for (size_t relay_index = 0u; relay_index < relays.size(); ++relay_index) {
      const uint64_t relay = relays[relay_index];
      const bool between = forward ? cursor < relay && relay < demand.target
                                   : demand.target < relay && relay < cursor;
      if (used[relay_index] || !between || !compute_sopp_branch_simm16(cursor, relay))
        continue;
      used[relay_index] = true;
      if (route_demand(demand_index, relay))
        return true;
      used[relay_index] = false;
    }
    return false;
  };
  return solve_demand(0u);
}

void expect_valid_complete_batch(std::span<const BranchOnlyRelayPairRequest> requests,
                                 std::span<const uint64_t> offered_relays,
                                 const BranchOnlyRelayBatchPlan &plan) {
  ASSERT_TRUE(plan.complete());
  ASSERT_EQ(plan.routes.size(), requests.size());
  const std::set<uint64_t> offered(offered_relays.begin(), offered_relays.end());
  std::set<uint64_t> used;
  for (size_t pair = 0u; pair < requests.size(); ++pair) {
    SCOPED_TRACE(::testing::Message() << "pair=" << pair);
    const BranchOnlyRelayPairRequest &request = requests[pair];
    const BranchOnlyRelayRoute &route = plan.routes[pair];
    const auto expect_route = [&](std::span<const uint64_t> relays, uint64_t source,
                                  uint64_t target) {
      uint64_t cursor = source;
      const bool forward = target > source;
      for (uint64_t relay : relays) {
        EXPECT_TRUE(offered.contains(relay));
        EXPECT_TRUE(used.insert(relay).second) << "relay reused across demands";
        EXPECT_TRUE(forward ? cursor < relay && relay < target : target < relay && relay < cursor);
        EXPECT_TRUE(compute_sopp_branch_simm16(cursor, relay).has_value());
        cursor = relay;
      }
      EXPECT_TRUE(compute_sopp_branch_simm16(cursor, target).has_value());
    };
    expect_route(route.entry_relay_offsets, request.entry_source, request.entry_target);
    expect_route(route.return_relay_offsets, request.return_source, request.return_target);
    EXPECT_EQ(route.claims.size(),
              route.entry_relay_offsets.size() + route.return_relay_offsets.size());
  }
}

void expect_same_batch_plan(const BranchOnlyRelayBatchPlan &lhs,
                            const BranchOnlyRelayBatchPlan &rhs) {
  EXPECT_EQ(lhs.failure, rhs.failure);
  EXPECT_EQ(lhs.strategy, rhs.strategy);
  EXPECT_EQ(lhs.work_budget_exhausted, rhs.work_budget_exhausted);
  EXPECT_EQ(lhs.routing_invariant_failed, rhs.routing_invariant_failed);
  EXPECT_EQ(lhs.route_optimization_exhausted, rhs.route_optimization_exhausted);
  EXPECT_EQ(lhs.route_optimization_invariant_failed, rhs.route_optimization_invariant_failed);
  EXPECT_EQ(lhs.search_work_consumed, rhs.search_work_consumed);
  EXPECT_EQ(lhs.scan_work_consumed, rhs.scan_work_consumed);
  EXPECT_EQ(lhs.route_optimization_search_work_consumed,
            rhs.route_optimization_search_work_consumed);
  EXPECT_EQ(lhs.route_optimization_scan_work_consumed, rhs.route_optimization_scan_work_consumed);
  EXPECT_EQ(lhs.rejected_pair_indices, rhs.rejected_pair_indices);
  EXPECT_EQ(lhs.rejection_reasons, rhs.rejection_reasons);
  EXPECT_EQ(lhs.pair_strategies, rhs.pair_strategies);
  ASSERT_EQ(lhs.routes.size(), rhs.routes.size());
  for (size_t pair = 0u; pair < lhs.routes.size(); ++pair) {
    EXPECT_EQ(lhs.routes[pair].entry_relay_offsets, rhs.routes[pair].entry_relay_offsets);
    EXPECT_EQ(lhs.routes[pair].return_relay_offsets, rhs.routes[pair].return_relay_offsets);
    ASSERT_EQ(lhs.routes[pair].retired_relay_claims.size(),
              rhs.routes[pair].retired_relay_claims.size());
    for (size_t retired = 0u; retired < lhs.routes[pair].retired_relay_claims.size(); ++retired) {
      EXPECT_EQ(lhs.routes[pair].retired_relay_claims[retired].offset,
                rhs.routes[pair].retired_relay_claims[retired].offset);
      EXPECT_EQ(lhs.routes[pair].retired_relay_claims[retired].provenance,
                rhs.routes[pair].retired_relay_claims[retired].provenance);
      EXPECT_EQ(lhs.routes[pair].retired_relay_claims[retired].owner_affinity,
                rhs.routes[pair].retired_relay_claims[retired].owner_affinity);
    }
    ASSERT_EQ(lhs.routes[pair].claims.size(), rhs.routes[pair].claims.size());
    for (size_t claim = 0u; claim < lhs.routes[pair].claims.size(); ++claim) {
      EXPECT_EQ(lhs.routes[pair].claims[claim].offset, rhs.routes[pair].claims[claim].offset);
      EXPECT_EQ(lhs.routes[pair].claims[claim].provenance,
                rhs.routes[pair].claims[claim].provenance);
      EXPECT_EQ(lhs.routes[pair].claims[claim].owner_affinity,
                rhs.routes[pair].claims[claim].owner_affinity);
    }
  }
}

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
  EXPECT_FALSE(router.offer(2u, BranchOnlyRelayProvenance::PristineNop));

  DbiPatchPlacementPlanner planner(kArch, kTextSize);
  BranchOnlyRelayPlanOutcome outcome;
  std::string error;
  const auto route = router.plan_pair(planner, kEntrySource, kEntryTarget, kReturnSource,
                                      kReturnTarget, &error, &outcome);
  ASSERT_TRUE(route) << error;
  EXPECT_EQ(outcome.failure, BranchOnlyRelayPlanFailure::None);
  EXPECT_EQ(route->entry_relay_offsets,
            (std::vector<uint64_t>{kOwnedReservoirRelay, kGeneratedRelay}));
  EXPECT_EQ(route->return_relay_offsets,
            (std::vector<uint64_t>{kOwnedAnchorRelay, kPristineRelay}));
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
      BranchOnlyRelayClaim{
          .offset = kOwnedReservoirRelay,
          .provenance = BranchOnlyRelayProvenance::OwnedReservoir,
          .owner_affinity = std::nullopt,
      },
      BranchOnlyRelayClaim{
          .offset = kOwnedReservoirRelay + sizeof(uint32_t),
          .provenance = BranchOnlyRelayProvenance::OwnedReservoir,
          .owner_affinity = std::nullopt,
      },
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

TEST(ConSanBranchOnlyRelayRouter, RetiresHalfOpenRangesAndIgnoresEmptyOnes) {
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {100u, 104u, 108u})
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::PristineNop));

  router.retire_range(100u, 0u);
  EXPECT_EQ(router.available_count(), 3u);
  router.retire_range(100u, 2u * sizeof(uint32_t));
  EXPECT_EQ(router.available_count(), 1u);
  router.retire_range(108u, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(router.available_count(), 0u);
}

TEST(ConSanBranchOnlyRelayRouter, ReportsWhichHalfOfPairedRoutingIsUnreachable) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  DbiPatchPlacementPlanner planner(kArch, 300'008u);

  BranchOnlyRelayPlanOutcome outcome;
  std::string error;
  EXPECT_FALSE(router.plan_pair(planner, /*entry_source=*/0u, /*entry_target=*/100'000u,
                                /*return_source=*/300'004u, /*return_target=*/4u, &error,
                                &outcome));
  EXPECT_EQ(outcome.failure, BranchOnlyRelayPlanFailure::ReturnRoute);
  EXPECT_NE(error.find("original continuation"), std::string::npos);
  EXPECT_TRUE(planner.occupied_ranges().empty());
}

TEST(ConSanBranchOnlyRelayRouter, ReportsUnavailablePristineRelayAsUnreachableEntry) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  {
    BranchOnlyRelayRouter router;
    DbiPatchPlacementPlanner planner(kArch, 200'008u);
    BranchOnlyRelayPlanOutcome outcome;
    std::string error;

    EXPECT_FALSE(router.plan_pair(planner, /*entry_source=*/0u, /*entry_target=*/200'000u,
                                  /*return_source=*/200'004u, /*return_target=*/100'004u, &error,
                                  &outcome));
    EXPECT_EQ(outcome.failure, BranchOnlyRelayPlanFailure::EntryRoute);
    EXPECT_NE(error.find("appended entry"), std::string::npos);
    EXPECT_TRUE(planner.occupied_ranges().empty());
  }

  {
    BranchOnlyRelayRouter router;
    ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::PristineNop));
    DbiPatchPlacementPlanner planner(kArch, 200'008u);
    ASSERT_TRUE(planner.reserve_existing_range(100'000u, sizeof(uint32_t)));
    const std::vector occupied_before(planner.occupied_ranges().begin(),
                                      planner.occupied_ranges().end());
    BranchOnlyRelayPlanOutcome outcome;
    std::string error;

    EXPECT_FALSE(router.plan_pair(planner, /*entry_source=*/0u, /*entry_target=*/200'000u,
                                  /*return_source=*/200'004u, /*return_target=*/100'004u, &error,
                                  &outcome));
    EXPECT_EQ(outcome.failure, BranchOnlyRelayPlanFailure::EntryRoute);
    EXPECT_TRUE(std::ranges::equal(planner.occupied_ranges(), occupied_before));
  }
}

TEST(ConSanBranchOnlyRelayRouter, RejectsCrossedDestinationAssignment) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 0u,
          .entry_target = 200'000u,
          .return_source = 200'004u,
          .return_target = 100'004u,
      },
      BranchOnlyRelayPairRequest{
          .entry_source = 130'000u,
          .entry_target = 120'000u,
          .return_source = 120'004u,
          .return_target = 100'008u,
      },
  };
  DbiPatchPlacementPlanner planner(kArch, 200'008u);
  std::string error = "stale caller diagnostic";

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::EntryRoute);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{1u}));
  EXPECT_EQ(plan.rejection_reasons[1], BranchOnlyRelayPairRejection::InvalidEntryCoordinates);
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
  EXPECT_NE(error.find("invalid entry-coordinate"), std::string::npos);
  EXPECT_TRUE(planner.occupied_ranges().empty());
}

TEST(ConSanBranchOnlyRelayRouter, PreservesFeasibleExactPairsWhenInterchangeableFlowWouldCross) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint64_t kFirstRelay = 100'000u;
  constexpr uint64_t kSecondRelay = 200'000u;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(kFirstRelay, BranchOnlyRelayProvenance::OwnedReservoir));
  ASSERT_TRUE(router.offer(kSecondRelay, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 0u,
          .entry_target = 200'008u,
          .return_source = 200'012u,
          .return_target = 100'004u,
      },
      BranchOnlyRelayPairRequest{
          .entry_source = 100'004u,
          .entry_target = 300'012u,
          .return_source = 300'016u,
          .return_target = 200'016u,
      },
  };
  DbiPatchPlacementPlanner planner(kArch, 300'020u);
  std::string error;

  BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  ASSERT_TRUE(plan.complete()) << error;
  ASSERT_EQ(plan.routes.size(), requests.size());
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{kFirstRelay}));
  EXPECT_EQ(plan.routes[1].entry_relay_offsets, (std::vector<uint64_t>{kSecondRelay}));
  EXPECT_TRUE(plan.routes[0].return_relay_offsets.empty());
  EXPECT_TRUE(plan.routes[1].return_relay_offsets.empty());
  EXPECT_TRUE(router.commit(plan.routes, &error)) << error;
  EXPECT_EQ(router.available_count(), 0u);
}

TEST(ConSanBranchOnlyRelayRouter, BacktracksAcrossEntryPairsToPreserveACompleteBatch) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {100'000u, 120'000u})
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 0u,
          .entry_target = 200'008u,
          .return_source = 200'012u,
          .return_target = 100'004u,
      },
      BranchOnlyRelayPairRequest{
          .entry_source = 110'000u,
          .entry_target = 250'000u,
          .return_source = 250'004u,
          .return_target = 130'000u,
      },
  };
  DbiPatchPlacementPlanner planner(kArch, 250'008u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
  EXPECT_EQ(plan.routes[1].entry_relay_offsets, (std::vector<uint64_t>{120'000u}));
}

TEST(ConSanBranchOnlyRelayRouter, BacktracksAcrossEntryAndReturnHalves) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {20'004u, 40'004u, 100'004u, 160'004u, 180'004u, 200'004u, 220'004u}) {
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir));
  }
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 4u,
          .entry_target = 240'004u,
          .return_source = 240'008u,
          .return_target = 12u,
      },
      BranchOnlyRelayPairRequest{
          .entry_source = 80'004u,
          .entry_target = 260'004u,
          .return_source = 260'008u,
          .return_target = 80'012u,
      },
  };
  DbiPatchPlacementPlanner planner(kArch, 260'012u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  ASSERT_TRUE(plan.complete()) << error;
  std::unordered_set<uint64_t> used;
  for (const BranchOnlyRelayRoute &route : plan.routes) {
    for (uint64_t relay : route.entry_relay_offsets)
      EXPECT_TRUE(used.insert(relay).second);
    for (uint64_t relay : route.return_relay_offsets)
      EXPECT_TRUE(used.insert(relay).second);
  }
  EXPECT_EQ(used.size(), 6u);
}

TEST(ConSanBranchOnlyRelayRouter, BacktracksAfterAShortestRouteStrandsAnotherPair) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {80'000u, 120'000u, 200'000u})
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 0u,
          .entry_target = 240'000u,
          .return_source = 240'004u,
          .return_target = 120'004u,
      },
      BranchOnlyRelayPairRequest{
          .entry_source = 40'000u,
          .entry_target = 280'000u,
          .return_source = 280'004u,
          .return_target = 160'004u,
      },
  };
  DbiPatchPlacementPlanner planner(kArch, 280'008u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{120'000u}));
  EXPECT_EQ(plan.routes[1].entry_relay_offsets, (std::vector<uint64_t>{80'000u, 200'000u}));
}

TEST(ConSanBranchOnlyRelayRouter, MinimizesMaterializedRelayOwners) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr BranchOnlyRelayOwnerIdentity kFirstOwner = lds_relay_owner(11u);
  constexpr BranchOnlyRelayOwnerIdentity kSecondOwner = lds_relay_owner(22u);
  const std::array offers = {
      std::tuple{100'000u, kSecondOwner},
      std::tuple{120'000u, kFirstOwner},
      std::tuple{200'000u, kFirstOwner},
      std::tuple{220'000u, kSecondOwner},
  };
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 0u,
          .entry_target = 300'000u,
          .return_source = 300'004u,
          .return_target = 200'004u,
      },
  };
  BranchOnlyRelayRouter router;
  for (const auto &[offset, owner] : offers)
    ASSERT_TRUE(router.offer(offset, BranchOnlyRelayProvenance::OwnedReservoir, owner));
  DbiPatchPlacementPlanner planner(kArch, 300'008u);
  std::string error;
  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  ASSERT_TRUE(plan.complete()) << error;
  ASSERT_EQ(plan.routes.size(), 1u);
  EXPECT_EQ(plan.routes.front().entry_relay_offsets, (std::vector<uint64_t>{120'000u, 200'000u}));
  EXPECT_TRUE(plan.routes.front().return_relay_offsets.empty());
  ASSERT_EQ(plan.routes.front().claims.size(), 2u);
  EXPECT_TRUE(std::ranges::all_of(plan.routes.front().claims, [&](const auto &claim) {
    return claim.owner_affinity == kFirstOwner;
  }));
}

TEST(ConSanBranchOnlyRelayRouter, SharesOneOwnerAcrossBatchAndExactPairFallback) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr BranchOnlyRelayOwnerIdentity kFirstOwner = lds_relay_owner(11u);
  constexpr BranchOnlyRelayOwnerIdentity kSecondOwner = lds_relay_owner(22u);
  const std::array offers = {
      std::tuple{100'000u, kFirstOwner},  std::tuple{100'004u, kFirstOwner},
      std::tuple{220'000u, kFirstOwner},  std::tuple{220'004u, kFirstOwner},
      std::tuple{100'008u, kSecondOwner}, std::tuple{100'012u, kSecondOwner},
      std::tuple{220'008u, kSecondOwner}, std::tuple{220'012u, kSecondOwner},
  };
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 0u,
          .entry_target = 320'000u,
          .return_source = 320'008u,
          .return_target = 200'008u,
      },
      BranchOnlyRelayPairRequest{
          .entry_source = 4u,
          .entry_target = 320'004u,
          .return_source = 320'012u,
          .return_target = 200'012u,
      },
  };

  for (bool force_pair_fallback : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "force_pair_fallback=" << force_pair_fallback);
    BranchOnlyRelayRouter router;
    for (const auto &[offset, owner] : offers)
      ASSERT_TRUE(router.offer(offset, BranchOnlyRelayProvenance::OwnedReservoir, owner));
    DbiPatchPlacementPlanner planner(kArch, 320'016u);
    BranchOnlyRelaySearchLimits limits;
    if (force_pair_fallback) {
      limits.batch_base_search_work = 1u;
      limits.batch_search_work_per_demand = 0u;
    }
    std::string error;

    const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error, limits);

    ASSERT_TRUE(plan.complete()) << error;
    EXPECT_EQ(plan.strategy, force_pair_fallback ? BranchOnlyRelayPlanStrategy::ExactPairFallback
                                                 : BranchOnlyRelayPlanStrategy::ExactBatch);
    std::set<BranchOnlyRelayOwnerIdentity> owners;
    size_t claim_count = 0u;
    for (const BranchOnlyRelayRoute &route : plan.routes) {
      for (const BranchOnlyRelayClaim &claim : route.claims) {
        ASSERT_TRUE(claim.owner_affinity);
        owners.insert(*claim.owner_affinity);
        ++claim_count;
      }
    }
    EXPECT_EQ(claim_count, 4u);
    EXPECT_EQ(owners.size(), 1u);
  }
}

TEST(ConSanBranchOnlyRelayRouter, GreedyFallbackOwnerIsFreeForLaterExactPair) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {120'000u, 240'000u, 420'008u, 540'008u, 650'004u, 770'004u}) {
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir,
                             /*owner_affinity=*/lds_relay_owner(11u)));
  }
  for (uint64_t relay : {100'000u, 220'000u, 430'008u, 550'008u, 660'004u, 780'004u}) {
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir,
                             /*owner_affinity=*/lds_relay_owner(22u)));
  }
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 0u,
          .entry_target = 300'000u,
          .return_source = 900'004u,
          .return_target = 600'004u,
      },
      BranchOnlyRelayPairRequest{
          .entry_source = 300'008u,
          .entry_target = 600'008u,
          .return_source = 600'012u,
          .return_target = 500'012u,
      },
  };
  BranchOnlyRelaySearchLimits limits;
  limits.batch_base_search_work = 1u;
  limits.batch_search_work_per_demand = 0u;
  limits.pair_search_work = 20u;
  DbiPatchPlacementPlanner planner(kArch, 900'008u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error, limits);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::GreedyPairFallback);
  ASSERT_EQ(plan.pair_strategies.size(), 2u);
  EXPECT_EQ(plan.pair_strategies[0], BranchOnlyRelayPlanStrategy::GreedyPairFallback);
  EXPECT_EQ(plan.pair_strategies[1], BranchOnlyRelayPlanStrategy::ExactPairFallback);
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{120'000u, 240'000u}));
  EXPECT_EQ(plan.routes[0].return_relay_offsets, (std::vector<uint64_t>{770'004u, 650'004u}));
  EXPECT_EQ(plan.routes[1].entry_relay_offsets, (std::vector<uint64_t>{420'008u, 540'008u}));
  EXPECT_TRUE(std::ranges::all_of(plan.routes[1].claims, [](const BranchOnlyRelayClaim &claim) {
    return claim.owner_affinity == lds_relay_owner(11u);
  }));
}

TEST(ConSanBranchOnlyRelayRouter, FindsCompleteRouteWhenSeveralMaterializedOwnersAreRequired) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(120'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(11u)));
  ASSERT_TRUE(router.offer(220'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(22u)));

  DbiPatchPlacementPlanner planner(kArch, 300'008u);
  std::string error;
  const auto route = router.plan_pair(planner, 0u, 300'000u, 300'004u, 200'004u, &error);
  ASSERT_TRUE(route) << error;
  EXPECT_EQ(route->entry_relay_offsets, (std::vector<uint64_t>{120'000u, 220'000u}));
  EXPECT_TRUE(route->return_relay_offsets.empty());
  ASSERT_EQ(route->claims.size(), 2u);
  EXPECT_EQ(route->claims[0].owner_affinity, lds_relay_owner(11u));
  EXPECT_EQ(route->claims[1].owner_affinity, lds_relay_owner(22u));
}

TEST(ConSanBranchOnlyRelayRouter, ManyOwnerGroupsPreserveExactTierAndBoundedOptimization) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t owner = 0u; owner < 48u; ++owner) {
    ASSERT_TRUE(router.offer(100'000u + owner * sizeof(uint32_t),
                             BranchOnlyRelayProvenance::OwnedReservoir,
                             lds_relay_owner(owner + 1u)));
    ASSERT_TRUE(router.offer(200'000u + owner * sizeof(uint32_t),
                             BranchOnlyRelayProvenance::OwnedReservoir,
                             lds_relay_owner(owner + 1u)));
  }

  DbiPatchPlacementPlanner planner(kArch, 300'008u);
  BranchOnlyRelayPlanOutcome outcome;
  std::string error;
  const auto route = router.plan_pair(planner, 0u, 300'000u, 300'004u, 200'004u, &error, &outcome);
  ASSERT_TRUE(route) << error;
  EXPECT_EQ(outcome.strategy, BranchOnlyRelayPlanStrategy::ExactBatch);
  EXPECT_FALSE(outcome.work_budget_exhausted);
  EXPECT_FALSE(outcome.route_optimization_exhausted);
  ASSERT_EQ(route->claims.size(), 2u);
  EXPECT_EQ(route->claims[0].owner_affinity, route->claims[1].owner_affinity);
  EXPECT_LT(outcome.scan_work_consumed, 20'000u);
  // With no zero-cost relay, one owner is a proved lower bound. The first
  // exact route already meets it, so no branch-and-bound search is needed.
  EXPECT_EQ(outcome.route_optimization_search_work_consumed, 0u);
  EXPECT_GT(outcome.route_optimization_scan_work_consumed, 0u);
}

TEST(ConSanBranchOnlyRelayRouter, OwnerTierClassificationPreservesRoutingSearchWindow) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr std::array kRelays = {100'000u, 120'000u, 200'000u, 220'000u};
  const auto plan = [&](bool tagged) {
    BranchOnlyRelayRouter router;
    for (uint64_t relay : kRelays) {
      EXPECT_TRUE(
          router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir,
                       tagged ? std::optional<BranchOnlyRelayOwnerIdentity>(lds_relay_owner(11u))
                              : std::nullopt));
    }
    DbiPatchPlacementPlanner planner(kArch, 300'008u);
    BranchOnlyRelayPlanOutcome outcome;
    std::string error;
    EXPECT_TRUE(router.plan_pair(planner, 0u, 300'000u, 300'004u, 200'004u, &error, &outcome))
        << error;
    return outcome;
  };

  const BranchOnlyRelayPlanOutcome untagged = plan(false);
  const BranchOnlyRelayPlanOutcome tagged = plan(true);

  EXPECT_EQ(tagged.strategy, BranchOnlyRelayPlanStrategy::ExactBatch);
  EXPECT_EQ(tagged.search_work_consumed, untagged.search_work_consumed);
  // Owner grouping and minimization use their independent meters, so tagging
  // the same routing inventory cannot shrink the feasibility window.
  EXPECT_EQ(tagged.scan_work_consumed, untagged.scan_work_consumed);
  // An untagged router knows at offer time that it has no owner groups to
  // optimize, so it does not spend the independent minimization allowance.
  EXPECT_EQ(untagged.route_optimization_scan_work_consumed, 0u);
  EXPECT_GT(tagged.route_optimization_scan_work_consumed,
            untagged.route_optimization_scan_work_consumed);
}

TEST(ConSanBranchOnlyRelayRouter, ExactPairOwnerOptimizationPreservesRelayCapacity) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(120'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(11u)));
  ASSERT_TRUE(router.offer(250'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(22u)));
  for (uint64_t relay : {80'000u, 160'000u, 240'000u}) {
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir,
                             /*owner_affinity=*/lds_relay_owner(33u)));
  }
  for (uint64_t relay : {100'000u, 220'000u}) {
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir,
                             /*owner_affinity=*/lds_relay_owner(44u)));
  }
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 0u,
          .entry_target = 300'000u,
          .return_source = 300'004u,
          .return_target = 260'004u,
      },
  };
  BranchOnlyRelaySearchLimits limits;
  limits.batch_base_search_work = 1u;
  limits.batch_search_work_per_demand = 0u;
  DbiPatchPlacementPlanner planner(kArch, 300'012u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error, limits);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactPairFallback);
  ASSERT_EQ(plan.routes.size(), 1u);
  // The first one-owner improvement uses the three owner-33 relays. The
  // relay-cap-aware search continues through equal-owner alternatives and
  // retains the two-relay owner-44 route.
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{100'000u, 220'000u}));
  EXPECT_TRUE(plan.routes[0].return_relay_offsets.empty());
  EXPECT_GT(plan.route_optimization_search_work_consumed, 0u);
  EXPECT_FALSE(plan.route_optimization_exhausted);
  EXPECT_LT(plan.route_optimization_scan_work_consumed, limits.pair_route_optimization_scan_work);
}

TEST(ConSanBranchOnlyRelayRouter, ExactPairZeroOwnerOptimizationPreservesRelayCapacity) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {120'000u, 200'000u, 240'000u})
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{80'000u, 280'000u, 320'000u, 160'000u},
  };
  BranchOnlyRelaySearchLimits limits;
  limits.batch_base_search_work = 1u;
  limits.batch_search_work_per_demand = 0u;
  DbiPatchPlacementPlanner planner(kArch, 400'000u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error, limits);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactPairFallback);
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{200'000u}));
  EXPECT_EQ(plan.routes[0].return_relay_offsets, (std::vector<uint64_t>{240'000u}));
  EXPECT_EQ(plan.routes[0].claims.size(), 2u);
  EXPECT_GT(plan.route_optimization_search_work_consumed, 0u);
}

TEST(ConSanBranchOnlyRelayRouter, ExactPairEqualOwnerOptimizationPreservesRelayCapacity) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(
      router.offer(120'000u, BranchOnlyRelayProvenance::OwnedReservoir, lds_relay_owner(7u)));
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  ASSERT_TRUE(
      router.offer(240'000u, BranchOnlyRelayProvenance::OwnedReservoir, lds_relay_owner(7u)));
  ASSERT_TRUE(router.offer(400'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{80'000u, 280'000u, 320'000u, 160'000u},
  };
  BranchOnlyRelaySearchLimits limits;
  limits.batch_base_search_work = 1u;
  limits.batch_search_work_per_demand = 0u;
  DbiPatchPlacementPlanner planner(kArch, 500'000u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error, limits);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactPairFallback);
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{200'000u}));
  EXPECT_EQ(plan.routes[0].return_relay_offsets, (std::vector<uint64_t>{240'000u}));
  EXPECT_EQ(plan.routes[0].claims.size(), 2u);
  EXPECT_GT(plan.route_optimization_search_work_consumed, 0u);
}

TEST(ConSanBranchOnlyRelayRouter, ExactPairTightOwnerBoundPreservesRelayCapacity) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {120'000u, 200'000u, 240'000u})
    ASSERT_TRUE(
        router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir, lds_relay_owner(7u)));
  const std::array requests = {
      BranchOnlyRelayPairRequest{80'000u, 280'000u, 320'000u, 160'000u},
  };
  BranchOnlyRelaySearchLimits limits;
  limits.batch_base_search_work = 1u;
  limits.batch_search_work_per_demand = 0u;
  DbiPatchPlacementPlanner planner(kArch, 400'000u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error, limits);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactPairFallback);
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{200'000u}));
  EXPECT_EQ(plan.routes[0].return_relay_offsets, (std::vector<uint64_t>{240'000u}));
  EXPECT_EQ(plan.routes[0].claims.size(), 2u);
  EXPECT_GT(plan.route_optimization_search_work_consumed, 0u);
}

TEST(ConSanBranchOnlyRelayRouter, ExactPairOptimizationCannotConsumeLaterFeasibilityRelay) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  const std::array requests = {
      BranchOnlyRelayPairRequest{87'348u, 173'404u, 217'460u, 43'148u},
      BranchOnlyRelayPairRequest{215'472u, 380'544u, 389'844u, 52'164u},
  };
  using OptionalOwner = std::optional<BranchOnlyRelayOwnerIdentity>;
  const std::array<std::pair<uint64_t, OptionalOwner>, 7> relays = {
      std::pair{34'724u, OptionalOwner{lds_relay_owner(4u)}},
      std::pair{114'620u, OptionalOwner{lds_relay_owner(3u)}},
      std::pair{149'188u, OptionalOwner{}},
      std::pair{249'900u, OptionalOwner{}},
      std::pair{323'244u, OptionalOwner{}},
      std::pair{352'320u, OptionalOwner{lds_relay_owner(1u)}},
      std::pair{394'428u, OptionalOwner{}},
  };
  BranchOnlyRelaySearchLimits limits;
  limits.batch_base_search_work = 1u;
  limits.batch_search_work_per_demand = 0u;
  const auto plan_with_limits = [&](const BranchOnlyRelaySearchLimits &selected_limits) {
    BranchOnlyRelayRouter router;
    for (const auto &[offset, owner] : relays) {
      EXPECT_TRUE(router.offer(offset, BranchOnlyRelayProvenance::OwnedReservoir, owner));
    }
    DbiPatchPlacementPlanner planner(kArch, 400'000u);
    return router.plan_pairs(planner, requests, nullptr, selected_limits);
  };
  BranchOnlyRelaySearchLimits feasibility_only_limits = limits;
  feasibility_only_limits.pair_route_optimization_search_work = 1u;
  feasibility_only_limits.pair_route_optimization_search_work_per_demand = 0u;

  const BranchOnlyRelayBatchPlan feasibility_baseline = plan_with_limits(feasibility_only_limits);
  const BranchOnlyRelayBatchPlan optimized = plan_with_limits(limits);

  ASSERT_TRUE(feasibility_baseline.complete());
  ASSERT_TRUE(optimized.complete());
  EXPECT_TRUE(optimized.rejected_pair_indices.empty());
  EXPECT_TRUE(std::ranges::none_of(optimized.routes[0].claims,
                                   [](const auto &claim) { return claim.offset == 149'188u; }));
  EXPECT_TRUE(std::ranges::any_of(optimized.routes[1].claims,
                                  [](const auto &claim) { return claim.offset == 149'188u; }));
}

TEST(ConSanBranchOnlyRelayRouter, ExactBatchOwnerOptimizationDoesNotUseRelayCountTieBreak) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(120'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(11u)));
  for (uint64_t relay : {110'000u, 140'000u, 260'000u}) {
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir,
                             /*owner_affinity=*/lds_relay_owner(33u)));
  }
  for (uint64_t relay : {100'000u, 220'000u}) {
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir,
                             /*owner_affinity=*/lds_relay_owner(44u)));
  }
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 300'000u, 300'004u, 260'004u},
  };
  DbiPatchPlacementPlanner planner(kArch, 300'012u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactBatch);
  // The first strict owner improvement wins for a complete batch; the later
  // equal-owner, shorter route is a pair-capacity policy, not a batch policy.
  EXPECT_EQ(plan.routes[0].entry_relay_offsets,
            (std::vector<uint64_t>{110'000u, 140'000u, 260'000u}));
  EXPECT_TRUE(plan.routes[0].return_relay_offsets.empty());
}

TEST(ConSanBranchOnlyRelayRouter, OneOwnerBaselineStillFindsUnownedBatch) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {41'020u, 86'852u, 98'216u, 140'780u, 196'020u, 285'788u})
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::PristineNop));
  for (uint64_t relay : {122'140u, 223'712u, 296'212u, 337'104u})
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir,
                             /*owner_affinity=*/lds_relay_owner(7u)));
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 0u,
          .entry_target = 240'884u,
          .return_source = 900'004u,
          .return_target = 890'004u,
      },
      BranchOnlyRelayPairRequest{
          .entry_source = 4u,
          .entry_target = 385'192u,
          .return_source = 800'004u,
          .return_target = 790'004u,
      },
  };
  DbiPatchPlacementPlanner planner(kArch, 1'000'008u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_GT(plan.route_optimization_search_work_consumed, 0u);
  EXPECT_FALSE(plan.route_optimization_exhausted);
  for (const BranchOnlyRelayRoute &route : plan.routes) {
    EXPECT_TRUE(std::ranges::all_of(
        route.claims, [](const BranchOnlyRelayClaim &claim) { return !claim.owner_affinity; }));
  }
}

TEST(ConSanBranchOnlyRelayRouter, LargeInventoryReservesScanWorkForOwnerMinimizer) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {41'020u, 86'852u, 98'216u, 140'780u, 196'020u, 285'788u})
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::PristineNop));
  for (uint64_t relay : {122'140u, 223'712u, 296'212u, 337'104u})
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir,
                             /*owner_affinity=*/lds_relay_owner(7u)));
  for (uint64_t relay = 0u; relay < 4'000u; ++relay) {
    ASSERT_TRUE(router.offer(500'000u + relay * sizeof(uint32_t),
                             BranchOnlyRelayProvenance::OwnedReservoir));
  }
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 0u,
          .entry_target = 240'884u,
          .return_source = 900'004u,
          .return_target = 890'004u,
      },
      BranchOnlyRelayPairRequest{
          .entry_source = 4u,
          .entry_target = 385'192u,
          .return_source = 800'004u,
          .return_target = 790'004u,
      },
  };
  DbiPatchPlacementPlanner planner(kArch, 1'000'008u);
  BranchOnlyRelaySearchLimits limits;
  limits.batch_route_optimization_scan_work = 12'000u;
  limits.batch_route_optimization_scan_work_per_demand_relay = 0u;
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error, limits);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactBatch);
  // The optional lower-bound accelerator must leave enough of the shared scan
  // allowance for the minimizer to enter its search, even when neither pass
  // can traverse the full inventory.
  EXPECT_GT(plan.route_optimization_search_work_consumed, 0u);
  EXPECT_TRUE(plan.route_optimization_exhausted);
  EXPECT_LE(plan.route_optimization_scan_work_consumed, limits.batch_route_optimization_scan_work);
}

TEST(ConSanBranchOnlyRelayRouter, BoundedOwnerOptimizationRetainsExactFeasibleRoute) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(33u)));
  ASSERT_TRUE(router.offer(120'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(11u)));
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(33u)));
  ASSERT_TRUE(router.offer(220'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(22u)));
  DbiPatchPlacementPlanner planner(kArch, 300'008u);
  BranchOnlyRelaySearchLimits limits;
  limits.batch_route_optimization_search_work = 1u;
  limits.batch_route_optimization_search_work_per_demand = 0u;
  limits.batch_route_optimization_scan_work = 4u;
  limits.batch_route_optimization_scan_work_per_demand_relay = 0u;
  BranchOnlyRelayPlanOutcome outcome;
  std::string error;

  const auto route =
      router.plan_pair(planner, 0u, 300'000u, 300'004u, 200'004u, &error, &outcome, limits);

  ASSERT_TRUE(route) << error;
  EXPECT_EQ(outcome.strategy, BranchOnlyRelayPlanStrategy::ExactBatch);
  EXPECT_FALSE(outcome.work_budget_exhausted);
  EXPECT_TRUE(outcome.route_optimization_exhausted);
  EXPECT_EQ(route->entry_relay_offsets, (std::vector<uint64_t>{120'000u, 220'000u}));
}

TEST(ConSanBranchOnlyRelayRouter, PartialBatchRetainsOwnerOptimizationExhaustion) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(33u)));
  ASSERT_TRUE(router.offer(120'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(11u)));
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(33u)));
  ASSERT_TRUE(router.offer(220'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(22u)));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 300'000u, 300'004u, 200'004u},
      BranchOnlyRelayPairRequest{400'000u, 900'000u, 900'004u, 900'008u},
  };
  DbiPatchPlacementPlanner planner(kArch, 900'012u);
  BranchOnlyRelaySearchLimits limits;
  limits.batch_route_optimization_search_work = 1u;
  limits.batch_route_optimization_search_work_per_demand = 0u;
  limits.pair_route_optimization_search_work = 1u;
  limits.pair_route_optimization_search_work_per_demand = 0u;
  limits.pair_route_optimization_scan_work = 4u;
  limits.pair_route_optimization_scan_work_per_demand_relay = 0u;
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error, limits);

  EXPECT_FALSE(plan.complete());
  ASSERT_FALSE(plan.routes[0].entry_relay_offsets.empty());
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{1u}));
  EXPECT_TRUE(plan.route_optimization_exhausted);
}

TEST(ConSanBranchOnlyRelayRouter, UnavailablePristineRelayIsFilteredBeforePlanning) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::PristineNop,
                           /*owner_affinity=*/lds_relay_owner(33u)));
  ASSERT_TRUE(router.offer(120'000u, BranchOnlyRelayProvenance::PristineNop,
                           /*owner_affinity=*/lds_relay_owner(11u)));
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::PristineNop,
                           /*owner_affinity=*/lds_relay_owner(33u)));
  ASSERT_TRUE(router.offer(220'000u, BranchOnlyRelayProvenance::PristineNop,
                           /*owner_affinity=*/lds_relay_owner(22u)));
  DbiPatchPlacementPlanner planner(kArch, 300'008u);
  ASSERT_TRUE(planner.reserve_existing_range(120'000u, sizeof(uint32_t)));
  BranchOnlyRelaySearchLimits limits;
  limits.batch_route_optimization_search_work = 1u;
  limits.batch_route_optimization_search_work_per_demand = 0u;
  limits.batch_route_optimization_scan_work = 4u;
  limits.batch_route_optimization_scan_work_per_demand_relay = 0u;
  BranchOnlyRelayPlanOutcome outcome;
  std::string error;

  const auto route =
      router.plan_pair(planner, 0u, 300'000u, 300'004u, 200'004u, &error, &outcome, limits);

  ASSERT_TRUE(route) << error;
  EXPECT_EQ(outcome.failure, BranchOnlyRelayPlanFailure::None);
  EXPECT_EQ(std::ranges::find(route->entry_relay_offsets, 120'000u),
            route->entry_relay_offsets.end());
  EXPECT_GT(outcome.route_optimization_search_work_consumed +
                outcome.route_optimization_scan_work_consumed,
            0u);
}

TEST(ConSanBranchOnlyRelayRouter, FreeUnownedRelaysRemainZeroMarginalAcrossPlans) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {120'000u, 200'000u, 400'000u, 500'000u})
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir,
                             /*owner_affinity=*/lds_relay_owner(11u)));
  ASSERT_TRUE(router.offer(420'000u, BranchOnlyRelayProvenance::PristineNop));
  ASSERT_TRUE(router.offer(520'000u, BranchOnlyRelayProvenance::PristineNop));

  DbiPatchPlacementPlanner first_planner(kArch, 600'008u);
  std::string error;
  const auto first = router.plan_pair(first_planner, 0u, 300'000u, 300'004u, 200'004u, &error);
  ASSERT_TRUE(first) << error;
  ASSERT_TRUE(router.commit(*first, &error)) << error;

  DbiPatchPlacementPlanner second_planner(kArch, 600'008u);
  const auto second =
      router.plan_pair(second_planner, 300'000u, 600'000u, 600'004u, 500'004u, &error);
  ASSERT_TRUE(second) << error;
  EXPECT_EQ(second->entry_relay_offsets, (std::vector<uint64_t>{420'000u, 520'000u}));
  EXPECT_TRUE(std::ranges::all_of(
      second->claims, [](const BranchOnlyRelayClaim &claim) { return !claim.owner_affinity; }));
}

TEST(ConSanBranchOnlyRelayRouter, IgnoresRelayAliasingAnyFixedPairCoordinate) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  ASSERT_TRUE(router.offer(130'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'000u, 200'004u, 100'004u},
      BranchOnlyRelayPairRequest{130'000u, 250'000u, 250'004u, 130'004u},
  };
  DbiPatchPlacementPlanner planner(kArch, 250'008u);
  std::string error;

  BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
  EXPECT_TRUE(plan.routes[1].entry_relay_offsets.empty());
  ASSERT_EQ(plan.routes[1].retired_relay_claims.size(), 1u);
  EXPECT_EQ(plan.routes[1].retired_relay_claims.front().offset, 130'000u);
  ASSERT_TRUE(router.commit(plan.routes, &error)) << error;
  EXPECT_EQ(router.available_count(), 0u);
}

TEST(ConSanBranchOnlyRelayRouter, CommittedEndpointCannotBecomeALaterRelay) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {100'000u, 120'000u, 200'000u, 210'000u})
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir));
  DbiPatchPlacementPlanner planner(kArch, 400'008u);
  std::string error;

  auto first = router.plan_pair(planner, 0u, 200'000u, 200'004u, 4u, &error);
  ASSERT_TRUE(first) << error;
  ASSERT_TRUE(router.commit(*first, &error)) << error;
  EXPECT_EQ(router.available_count(), 1u);

  // The first commit retired its 200'000 entry target. The only remaining
  // relay cannot serve both halves of this second pair.
  BranchOnlyRelayPlanOutcome outcome;
  auto second = router.plan_pair(planner, 90'000u, 320'000u, 320'004u, 90'004u, &error, &outcome);
  EXPECT_FALSE(second);
  EXPECT_EQ(outcome.failure, BranchOnlyRelayPlanFailure::RelayContention);
}

TEST(ConSanBranchOnlyRelayRouter, CommitRejectsChangedEndpointRetirementProvenance) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  DbiPatchPlacementPlanner planner(kArch, 200'008u);
  std::string error;

  auto route = router.plan_pair(planner, 0u, 200'000u, 200'004u, 100'004u, &error);
  ASSERT_TRUE(route) << error;
  ASSERT_EQ(route->retired_relay_claims.size(), 1u);
  router.retire_range(200'000u, sizeof(uint32_t));
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::GeneratedBank));

  EXPECT_FALSE(router.commit(*route, &error));
  EXPECT_EQ(error, "branch-only router endpoint retirement changed before commit");
  EXPECT_EQ(router.available_count(), 2u);
}

TEST(ConSanBranchOnlyRelayRouter, CommitRejectsChangedRelayOwnerAffinity) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(11u)));
  DbiPatchPlacementPlanner planner(kArch, 200'008u);
  std::string error;
  const auto route = router.plan_pair(planner, 0u, 200'000u, 200'004u, 100'004u, &error);
  ASSERT_TRUE(route) << error;
  ASSERT_EQ(route->claims.size(), 1u);
  EXPECT_EQ(route->claims.front().owner_affinity, lds_relay_owner(11u));

  router.retire_range(100'000u, sizeof(uint32_t));
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(22u)));
  EXPECT_FALSE(router.commit(*route, &error));
  EXPECT_EQ(error, "branch-only router claim changed before commit");
  EXPECT_EQ(router.available_count(), 1u);
}

TEST(ConSanBranchOnlyRelayRouter, RejectsInvalidCoordinatesBeforeSearching) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  DbiPatchPlacementPlanner planner(kArch, 300'008u);
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 3u, 300'004u, 200'000u},
  };
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::EntryRoute);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u}));
  EXPECT_NE(error.find("dword-aligned"), std::string::npos);
}

TEST(ConSanBranchOnlyRelayRouter, InvalidPairDoesNotSuppressValidPartialClaims) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  DbiPatchPlacementPlanner planner(kArch, 700'004u);
  const std::array requests = {
      BranchOnlyRelayPairRequest{2u, 100'000u, 600'004u, 500'004u},
      BranchOnlyRelayPairRequest{0u, 200'000u, 200'004u, 100'004u},
  };
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::EntryRoute);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u}));
  EXPECT_EQ(plan.rejection_reasons[0], BranchOnlyRelayPairRejection::InvalidEntryCoordinates);
  EXPECT_TRUE(plan.routes[0].claims.empty());
  EXPECT_TRUE(plan.routes[0].retired_relay_claims.empty());
  EXPECT_EQ(plan.routes[1].entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
  ASSERT_EQ(plan.routes[1].claims.size(), 1u);
  EXPECT_EQ(plan.routes[1].claims.front().offset, 100'000u);
}

TEST(ConSanBranchOnlyRelayRouter, DuplicateEntryCoordinateRejectsBothOwningPairs) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 100'004u, 500'004u, 400'004u},
      BranchOnlyRelayPairRequest{0u, 120'004u, 520'004u, 420'004u},
      BranchOnlyRelayPairRequest{4u, 200'000u, 300'004u, 200'004u},
  };
  DbiPatchPlacementPlanner planner(kArch, 520'008u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u, 1u}));
  EXPECT_EQ(plan.rejection_reasons[0], BranchOnlyRelayPairRejection::InvalidEntryCoordinates);
  EXPECT_EQ(plan.rejection_reasons[1], BranchOnlyRelayPairRejection::InvalidEntryCoordinates);
  EXPECT_EQ(plan.routes[2].entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
}

TEST(ConSanBranchOnlyRelayRouter, DuplicateReturnCoordinateRejectsBothOwningPairs) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 100'000u, 300'004u, 200'004u},
      BranchOnlyRelayPairRequest{4u, 120'004u, 300'004u, 220'004u},
  };
  DbiPatchPlacementPlanner planner(kArch, 300'008u);

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests);

  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::ReturnRoute);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u, 1u}));
  EXPECT_EQ(plan.rejection_reasons[0], BranchOnlyRelayPairRejection::InvalidReturnCoordinates);
  EXPECT_EQ(plan.rejection_reasons[1], BranchOnlyRelayPairRejection::InvalidReturnCoordinates);
}

TEST(ConSanBranchOnlyRelayRouter, RemovedDuplicateOwnerCannotRejectALaterPair) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 100'004u, 500'004u, 400'004u},
      BranchOnlyRelayPairRequest{0u, 120'004u, 520'004u, 420'004u},
      BranchOnlyRelayPairRequest{4u, 200'000u, 500'004u, 400'008u},
  };
  DbiPatchPlacementPlanner planner(kArch, 520'008u);

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests);

  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u, 1u}));
  EXPECT_EQ(plan.rejection_reasons[2], BranchOnlyRelayPairRejection::None);
  EXPECT_EQ(plan.routes[2].entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
}

TEST(ConSanBranchOnlyRelayRouter, ReturnCoordinateCollisionReleasesEntryCoordinates) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  // Pairs 0 and 1 collide on return_source, so both release their entry
  // coordinates. Pair 2 can then reuse pair 0's entry pair.
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 100'004u, 500'004u, 400'004u},
      BranchOnlyRelayPairRequest{8u, 108'004u, 500'004u, 400'008u},
      BranchOnlyRelayPairRequest{0u, 100'004u, 300'004u, 200'004u},
  };
  DbiPatchPlacementPlanner planner(kArch, 500'008u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u, 1u})) << error;
  EXPECT_EQ(plan.rejection_reasons[2], BranchOnlyRelayPairRejection::None);
}

TEST(ConSanBranchOnlyRelayRouter, NonMonotonicReturnAloneReportsReturnRoute) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 100'000u, 200'004u, 300'004u},
  };
  DbiPatchPlacementPlanner planner(kArch, 300'008u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::ReturnRoute);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u}));
  EXPECT_EQ(plan.rejection_reasons[0], BranchOnlyRelayPairRejection::InvalidReturnCoordinates);
  EXPECT_NE(error.find("invalid return-coordinate"), std::string::npos);
}

TEST(ConSanBranchOnlyRelayRouter, MixedRejectionsReportEachPairCause) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{100'000u, 80'000u, 300'004u, 200'004u},
      BranchOnlyRelayPairRequest{500'000u, 700'000u, 900'004u, 800'004u},
      BranchOnlyRelayPairRequest{0u, 200'000u, 200'004u, 4u},
  };
  DbiPatchPlacementPlanner planner(kArch, 900'008u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::EntryRoute);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u, 1u, 2u}));
  EXPECT_EQ(plan.rejection_reasons[0], BranchOnlyRelayPairRejection::InvalidEntryCoordinates);
  EXPECT_EQ(plan.rejection_reasons[1], BranchOnlyRelayPairRejection::EntryUnreachable);
  EXPECT_EQ(plan.rejection_reasons[2], BranchOnlyRelayPairRejection::RelayContention);
  EXPECT_NE(error.find("invalid entry-coordinate"), std::string::npos);
  EXPECT_NE(error.find("unreachable appended entry"), std::string::npos);
  EXPECT_NE(error.find("relay-contended"), std::string::npos);
}

TEST(ConSanBranchOnlyRelayRouter, IndividuallyReachableHalvesReportRelayContention) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'000u, 200'004u, 4u},
  };
  DbiPatchPlacementPlanner planner(kArch, 200'008u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::RelayContention);
  EXPECT_EQ(plan.rejection_reasons[0], BranchOnlyRelayPairRejection::RelayContention);
  EXPECT_NE(error.find("relay-contended"), std::string::npos);
}

TEST(ConSanBranchOnlyRelayRouter, ReachableEntryAndUnreachableReturnReportReturnRoute) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(129'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 250'000u, 900'004u, 500'004u},
  };
  DbiPatchPlacementPlanner planner(kArch, 900'008u);

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests);

  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::ReturnRoute);
  EXPECT_EQ(plan.rejection_reasons[0], BranchOnlyRelayPairRejection::ReturnUnreachable);
}

TEST(ConSanBranchOnlyRelayRouter, RejectsAPairOnlyOnceWhenBothHalvesAreUnreachable) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  DbiPatchPlacementPlanner planner(kArch, 300'008u);
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'000u, 300'004u, 100'000u},
  };
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::EntryRoute);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u}));
  EXPECT_TRUE(plan.routes[0].claims.empty());
}

TEST(ConSanBranchOnlyRelayRouter, RejectedPairCannotConsumeARelayFromAFeasiblePair) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{500'000u, 700'000u, 300'004u, 100'000u},
      BranchOnlyRelayPairRequest{0u, 120'000u, 300'008u, 100'004u},
  };
  DbiPatchPlacementPlanner planner(kArch, 700'004u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::EntryRoute);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u}));
  EXPECT_TRUE(plan.routes[0].claims.empty());
  EXPECT_TRUE(plan.routes[0].retired_relay_claims.empty());
  EXPECT_EQ(plan.routes[1].return_relay_offsets, (std::vector<uint64_t>{200'000u}));
  ASSERT_EQ(plan.routes[1].claims.size(), 1u);
  EXPECT_EQ(plan.routes[1].claims.front().offset, 200'000u);
}

TEST(ConSanBranchOnlyRelayRouter, PairExactFallbackPreservesEntryReturnFeasibility) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  const std::array relays = {128'840u, 157'124u, 203'900u};
  for (uint64_t relay : relays)
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 400'000u, 700'004u, 600'004u},
      BranchOnlyRelayPairRequest{80'056u, 288'192u, 288'196u, 80'060u},
  };
  DbiPatchPlacementPlanner planner(kArch, 700'008u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactPairFallback);
  EXPECT_FALSE(plan.work_budget_exhausted);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u}));
  EXPECT_EQ(plan.routes[1].entry_relay_offsets, (std::vector<uint64_t>{157'124u}));
  EXPECT_EQ(plan.routes[1].return_relay_offsets, (std::vector<uint64_t>{203'900u}));
}

TEST(ConSanBranchOnlyRelayRouter, InfeasibleCorridorBatchStopsAtDeterministicWorkBudget) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (size_t relay = 0u; relay < 12u; ++relay)
    ASSERT_TRUE(router.offer(129'000u + relay * 8u, BranchOnlyRelayProvenance::OwnedReservoir));
  for (size_t relay = 0u; relay < 12u; ++relay)
    ASSERT_TRUE(router.offer(258'000u + relay * 8u, BranchOnlyRelayProvenance::OwnedReservoir));
  for (size_t relay = 0u; relay < 5u; ++relay)
    ASSERT_TRUE(router.offer(385'000u + relay * 8u, BranchOnlyRelayProvenance::OwnedReservoir));
  std::vector<BranchOnlyRelayPairRequest> requests;
  for (size_t pair = 0u; pair < 6u; ++pair) {
    requests.push_back({
        .entry_source = 8u + pair * 16u,
        .entry_target = 400'000u + pair * 16u,
        .return_source = 200'000u + pair * 16u,
        .return_target = 199'000u + pair * 16u,
    });
  }
  DbiPatchPlacementPlanner planner(kArch, 400'200u);
  DbiPatchPlacementPlanner repeated_planner(kArch, 400'200u);
  std::string error;
  std::string repeated_error;
  const BranchOnlyRelaySearchLimits limits;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error, limits);
  const BranchOnlyRelayBatchPlan repeated =
      router.plan_pairs(repeated_planner, requests, &repeated_error, limits);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactPairFallback);
  EXPECT_TRUE(plan.work_budget_exhausted);
  EXPECT_TRUE(std::ranges::none_of(plan.rejection_reasons, [](auto reason) {
    return reason == BranchOnlyRelayPairRejection::WorkBudget;
  }));
  EXPECT_NE(error.find("batch work was bounded"), std::string::npos);
  EXPECT_GT(plan.search_work_consumed, 0u);
  EXPECT_LE(plan.search_work_consumed,
            limits.batch_base_search_work +
                2u * requests.size() * limits.batch_search_work_per_demand +
                requests.size() * limits.pair_search_work);
  constexpr size_t kRelayCount = 29u;
  EXPECT_LE(plan.scan_work_consumed,
            limits.batch_base_scan_work +
                2u * requests.size() * kRelayCount * limits.batch_scan_work_per_demand_relay +
                limits.batch_relay_qualification_work +
                kRelayCount * limits.batch_relay_qualification_work_per_relay +
                kRelayCount * limits.batch_relay_qualification_work_per_relay_range_level +
                limits.batch_fallback_setup_work +
                requests.size() *
                    (limits.pair_base_scan_work + kRelayCount * limits.pair_scan_work_per_relay +
                     2u * limits.pair_greedy_work));
  EXPECT_TRUE(std::ranges::any_of(
      plan.routes, [](const BranchOnlyRelayRoute &route) { return !route.claims.empty(); }));
  EXPECT_EQ(error, repeated_error);
  ASSERT_NO_FATAL_FAILURE(expect_same_batch_plan(plan, repeated));
}

TEST(ConSanBranchOnlyRelayRouter, TinyLimitsExerciseObservableGreedyFallback) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  DbiPatchPlacementPlanner planner(kArch, 200'008u);
  BranchOnlyRelayPlanOutcome outcome;
  const BranchOnlyRelaySearchLimits limits{
      .batch_base_search_work = 1u,
      .batch_search_work_per_demand = 0u,
      .batch_base_scan_work = 100u,
      .batch_scan_work_per_demand_relay = 0u,
      .batch_fallback_setup_work = 100u,
      .pair_search_work = 1u,
      .pair_base_scan_work = 100u,
      .pair_scan_work_per_relay = 0u,
      .pair_greedy_work = 100u,
  };
  std::string error;

  auto route =
      router.plan_pair(planner, 0u, 200'000u, 200'004u, 100'004u, &error, &outcome, limits);

  ASSERT_TRUE(route) << error;
  EXPECT_EQ(outcome.failure, BranchOnlyRelayPlanFailure::None);
  EXPECT_EQ(outcome.strategy, BranchOnlyRelayPlanStrategy::GreedyPairFallback);
  EXPECT_TRUE(outcome.work_budget_exhausted);
  EXPECT_EQ(route->entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
  EXPECT_EQ(outcome.search_work_consumed, 2u);
  EXPECT_GT(outcome.scan_work_consumed, 0u);
  EXPECT_LE(outcome.scan_work_consumed, 300u);

  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'000u, 200'004u, 100'004u},
  };
  DbiPatchPlacementPlanner batch_planner(kArch, 200'008u);
  const BranchOnlyRelayBatchPlan batch =
      router.plan_pairs(batch_planner, requests, nullptr, limits);
  EXPECT_EQ(outcome.failure, batch.failure);
  EXPECT_EQ(outcome.strategy, batch.strategy);
  EXPECT_EQ(outcome.work_budget_exhausted, batch.work_budget_exhausted);
  EXPECT_EQ(outcome.routing_invariant_failed, batch.routing_invariant_failed);
  EXPECT_EQ(outcome.route_optimization_invariant_failed, batch.route_optimization_invariant_failed);
  EXPECT_EQ(outcome.search_work_consumed, batch.search_work_consumed);
  EXPECT_EQ(outcome.scan_work_consumed, batch.scan_work_consumed);
  EXPECT_EQ(outcome.route_optimization_search_work_consumed,
            batch.route_optimization_search_work_consumed);
  EXPECT_EQ(outcome.route_optimization_scan_work_consumed,
            batch.route_optimization_scan_work_consumed);
  EXPECT_EQ(batch.pair_strategies, (std::vector{BranchOnlyRelayPlanStrategy::GreedyPairFallback}));
}

TEST(ConSanBranchOnlyRelayRouter, RecordsBatchedPlanAndFailureTelemetryWithSharedUnits) {
  ConSanBranchOnlyRoutingTelemetry telemetry;
  BranchOnlyRelayPlanOutcome outcome;
  outcome.failure = BranchOnlyRelayPlanFailure::None;
  outcome.strategy = BranchOnlyRelayPlanStrategy::GreedyPairFallback;
  outcome.work_budget_exhausted = true;
  outcome.routing_invariant_failed = true;
  outcome.route_optimization_exhausted = true;
  outcome.route_optimization_invariant_failed = true;
  outcome.search_work_consumed = 17u;
  outcome.scan_work_consumed = 23u;
  outcome.route_optimization_search_work_consumed = 29u;
  outcome.route_optimization_scan_work_consumed = 31u;
  const std::array strategies = {
      BranchOnlyRelayPlanStrategy::ExactBatch,
      BranchOnlyRelayPlanStrategy::ExactPairFallback,
      BranchOnlyRelayPlanStrategy::GreedyPairFallback,
  };

  record_branch_only_relay_plan(telemetry, outcome, strategies);
  record_branch_only_relay_failure(telemetry, BranchOnlyRelayPlanFailure::Reservation);
  record_branch_only_relay_rejection(telemetry, BranchOnlyRelayPairRejection::EntryUnreachable);

  EXPECT_EQ(telemetry.pair_attempt_count, 3u);
  EXPECT_EQ(telemetry.plan_call_count, 1u);
  EXPECT_EQ(telemetry.work_budget_exhaustion_count, 1u);
  EXPECT_EQ(telemetry.routing_invariant_failure_count, 1u);
  EXPECT_EQ(telemetry.route_optimization_exhaustion_count, 1u);
  EXPECT_EQ(telemetry.route_optimization_invariant_failure_count, 1u);
  EXPECT_EQ(telemetry.exact_pair_fallback_attempt_count, 2u);
  EXPECT_EQ(telemetry.greedy_pair_fallback_attempt_count, 1u);
  EXPECT_EQ(telemetry.search_work_count, 17u);
  EXPECT_EQ(telemetry.scan_work_count, 23u);
  EXPECT_EQ(telemetry.route_optimization_search_work_count, 29u);
  EXPECT_EQ(telemetry.route_optimization_scan_work_count, 31u);
  EXPECT_EQ(telemetry.reservation_failure_count, 1u);
  EXPECT_EQ(telemetry.entry_route_failure_count, 1u);
  EXPECT_EQ(telemetry.return_route_failure_count, 0u);
}

TEST(ConSanBranchOnlyRelayRouter, ComputesTelemetryDeltaAcrossEveryCounter) {
  const ConSanBranchOnlyRoutingTelemetry before{
      .pair_attempt_count = 1u,
      .plan_call_count = 2u,
      .entry_route_failure_count = 3u,
      .return_route_failure_count = 4u,
      .relay_contention_failure_count = 5u,
      .work_budget_failure_count = 6u,
      .work_budget_exhaustion_count = 7u,
      .routing_invariant_failure_count = 16u,
      .route_optimization_exhaustion_count = 8u,
      .route_optimization_invariant_failure_count = 9u,
      .reservation_failure_count = 10u,
      .exact_pair_fallback_attempt_count = 11u,
      .greedy_pair_fallback_attempt_count = 4u,
      .search_work_count = 12u,
      .scan_work_count = 13u,
      .route_optimization_search_work_count = 14u,
      .route_optimization_scan_work_count = 15u,
  };
  const ConSanBranchOnlyRoutingTelemetry after{
      .pair_attempt_count = 13u,
      .plan_call_count = 15u,
      .entry_route_failure_count = 17u,
      .return_route_failure_count = 19u,
      .relay_contention_failure_count = 21u,
      .work_budget_failure_count = 23u,
      .work_budget_exhaustion_count = 25u,
      .routing_invariant_failure_count = 42u,
      .route_optimization_exhaustion_count = 27u,
      .route_optimization_invariant_failure_count = 29u,
      .reservation_failure_count = 31u,
      .exact_pair_fallback_attempt_count = 33u,
      .greedy_pair_fallback_attempt_count = 12u,
      .search_work_count = 35u,
      .scan_work_count = 37u,
      .route_optimization_search_work_count = 39u,
      .route_optimization_scan_work_count = 41u,
  };

  const ConSanBranchOnlyRoutingTelemetry delta = branch_only_relay_telemetry_delta(after, before);

  EXPECT_EQ(delta.pair_attempt_count, 12u);
  EXPECT_EQ(delta.plan_call_count, 13u);
  EXPECT_EQ(delta.entry_route_failure_count, 14u);
  EXPECT_EQ(delta.return_route_failure_count, 15u);
  EXPECT_EQ(delta.relay_contention_failure_count, 16u);
  EXPECT_EQ(delta.work_budget_failure_count, 17u);
  EXPECT_EQ(delta.work_budget_exhaustion_count, 18u);
  EXPECT_EQ(delta.routing_invariant_failure_count, 26u);
  EXPECT_EQ(delta.route_optimization_exhaustion_count, 19u);
  EXPECT_EQ(delta.route_optimization_invariant_failure_count, 20u);
  EXPECT_EQ(delta.reservation_failure_count, 21u);
  EXPECT_EQ(delta.exact_pair_fallback_attempt_count, 22u);
  EXPECT_EQ(delta.greedy_pair_fallback_attempt_count, 8u);
  EXPECT_EQ(delta.search_work_count, 23u);
  EXPECT_EQ(delta.scan_work_count, 24u);
  EXPECT_EQ(delta.route_optimization_search_work_count, 25u);
  EXPECT_EQ(delta.route_optimization_scan_work_count, 26u);
  EXPECT_FALSE(branch_only_relay_telemetry_is_empty(delta));
  EXPECT_TRUE(branch_only_relay_telemetry_is_empty({}));
}

#ifdef NDEBUG
TEST(ConSanBranchOnlyRelayRouter, ClampsRegressiveTelemetryDeltaFieldsInReleaseBuilds) {
  const ConSanBranchOnlyRoutingTelemetry before{
      .plan_call_count = 7u,
      .scan_work_count = 100u,
  };
  const ConSanBranchOnlyRoutingTelemetry after{
      .pair_attempt_count = 3u,
      .plan_call_count = 3u,
  };

  const ConSanBranchOnlyRoutingTelemetry delta = branch_only_relay_telemetry_delta(after, before);

  EXPECT_EQ(delta.pair_attempt_count, 3u);
  EXPECT_EQ(delta.plan_call_count, 0u);
  EXPECT_EQ(delta.scan_work_count, 0u);
}
#endif

TEST(ConSanBranchOnlyRelayRouter, ZeroTierLimitsAreNormalizedInsteadOfDisablingRouting) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  DbiPatchPlacementPlanner planner(kArch, 16u);
  const BranchOnlyRelaySearchLimits limits{
      .batch_base_search_work = 0u,
      .batch_search_work_per_demand = 0u,
      .batch_base_scan_work = 0u,
      .batch_scan_work_per_demand_relay = 0u,
      .batch_route_optimization_search_work = 0u,
      .batch_route_optimization_search_work_per_demand = 0u,
      .batch_route_optimization_scan_work = 0u,
      .batch_route_optimization_scan_work_per_demand_relay = 0u,
      .batch_relay_qualification_work = 0u,
      .batch_relay_qualification_work_per_relay = 0u,
      .batch_relay_qualification_work_per_relay_range_level = 0u,
      .batch_fallback_setup_work = 0u,
      .pair_search_work = 0u,
      .pair_base_scan_work = 0u,
      .pair_scan_work_per_relay = 0u,
      .pair_route_optimization_search_work = 0u,
      .pair_route_optimization_search_work_per_demand = 0u,
      .pair_route_optimization_scan_work = 0u,
      .pair_route_optimization_scan_work_per_demand_relay = 0u,
      .pair_greedy_work = 0u,
  };
  BranchOnlyRelayPlanOutcome outcome;

  const auto route = router.plan_pair(planner, 0u, 4u, 12u, 8u, nullptr, &outcome, limits);

  ASSERT_TRUE(route);
  EXPECT_EQ(outcome.failure, BranchOnlyRelayPlanFailure::None);
  EXPECT_EQ(outcome.strategy, BranchOnlyRelayPlanStrategy::GreedyPairFallback);
  EXPECT_TRUE(outcome.work_budget_exhausted);
  EXPECT_EQ(outcome.search_work_consumed, 2u);
}

TEST(ConSanBranchOnlyRelayRouter,
     OverflowingPerInputHeadroomProductSaturatesWithoutDisablingExactRouting) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100u, BranchOnlyRelayProvenance::OwnedReservoir));
  DbiPatchPlacementPlanner planner(kArch, 1'000u);
  constexpr size_t kMaximum = std::numeric_limits<size_t>::max();
  constexpr size_t kWrappingMultiplier = kMaximum / 2u + 1u;
  const BranchOnlyRelaySearchLimits limits{
      .batch_base_search_work = 0u,
      .batch_search_work_per_demand = kWrappingMultiplier,
      .batch_base_scan_work = 0u,
      .batch_scan_work_per_demand_relay = kWrappingMultiplier,
  };
  BranchOnlyRelayPlanOutcome outcome;

  const auto route = router.plan_pair(planner, 0u, 4u, 12u, 8u, nullptr, &outcome, limits);

  ASSERT_TRUE(route);
  EXPECT_EQ(outcome.strategy, BranchOnlyRelayPlanStrategy::ExactBatch);
  EXPECT_FALSE(outcome.work_budget_exhausted);
}

TEST(ConSanBranchOnlyRelayRouter, OverflowingBaseHeadroomSumSaturatesWithoutDisablingExactRouting) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100u, BranchOnlyRelayProvenance::OwnedReservoir));
  DbiPatchPlacementPlanner planner(kArch, 1'000u);
  constexpr size_t kMaximum = std::numeric_limits<size_t>::max();
  const BranchOnlyRelaySearchLimits limits{
      .batch_base_search_work = kMaximum - 1u,
      .batch_search_work_per_demand = 1u,
      .batch_base_scan_work = kMaximum - 1u,
      .batch_scan_work_per_demand_relay = 1u,
  };
  BranchOnlyRelayPlanOutcome outcome;

  const auto route = router.plan_pair(planner, 0u, 4u, 12u, 8u, nullptr, &outcome, limits);

  ASSERT_TRUE(route);
  EXPECT_EQ(outcome.strategy, BranchOnlyRelayPlanStrategy::ExactBatch);
  EXPECT_FALSE(outcome.work_budget_exhausted);
}

TEST(ConSanBranchOnlyRelayRouter, ScanBudgetBoundsLargeRelayInventory) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  constexpr size_t kRelayCount = 1'024u;
  for (size_t relay = 1u; relay <= kRelayCount; ++relay)
    ASSERT_TRUE(router.offer(relay * sizeof(uint32_t), BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 400'000u, 500'004u, 400'004u},
  };
  const BranchOnlyRelaySearchLimits limits{
      .batch_base_search_work = 100'000u,
      .batch_search_work_per_demand = 0u,
      .batch_base_scan_work = 512u,
      .batch_scan_work_per_demand_relay = 0u,
      .batch_fallback_setup_work = 20'000u,
      .pair_search_work = 8'192u,
      .pair_base_scan_work = 512u,
      .pair_scan_work_per_relay = 0u,
      .pair_greedy_work = 1u,
  };
  DbiPatchPlacementPlanner planner(kArch, 500'008u);

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, nullptr, limits);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::WorkBudget);
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::GreedyPairFallback);
  EXPECT_TRUE(plan.work_budget_exhausted);
  // Qualification and fallback inventory construction are precharged. The
  // exact, pair, and greedy tiers refuse their first precharge and bill none.
  const size_t qualification_work = kRelayCount * (1u + std::bit_width(size_t{4u}));
  EXPECT_EQ(plan.scan_work_consumed,
            qualification_work + kRelayCount * std::bit_width(kRelayCount));
}

TEST(ConSanBranchOnlyRelayRouter, IneligibleRelayTraversalConsumesSearchBudget) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  for (uint64_t relay = 0u; relay < 16u; ++relay) {
    ASSERT_TRUE(router.offer(300'000u + relay * sizeof(uint32_t),
                             BranchOnlyRelayProvenance::OwnedReservoir));
  }
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'000u, 200'004u, 100'004u},
  };
  BranchOnlyRelaySearchLimits limits;
  limits.batch_base_search_work = 8u;
  limits.batch_search_work_per_demand = 0u;
  DbiPatchPlacementPlanner planner(kArch, 400'000u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error, limits);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactPairFallback);
  EXPECT_TRUE(plan.work_budget_exhausted);
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
}

TEST(ConSanBranchOnlyRelayRouter, PristineQualificationHasAnIndependentWorkBound) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay = 1u; relay <= 1'024u; ++relay) {
    ASSERT_TRUE(router.offer(relay * sizeof(uint32_t), BranchOnlyRelayProvenance::PristineNop));
  }
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'000u, 200'004u, 100'004u},
  };
  BranchOnlyRelaySearchLimits limits;
  limits.batch_relay_qualification_work = 1u;
  limits.batch_relay_qualification_work_per_relay = 0u;
  limits.batch_relay_qualification_work_per_relay_range_level = 0u;
  DbiPatchPlacementPlanner planner(kArch, 300'000u);
  const std::vector<std::pair<uint64_t, uint64_t>> occupied_before(
      planner.occupied_ranges().begin(), planner.occupied_ranges().end());

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, nullptr, limits);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::WorkBudget);
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactPairFallback);
  EXPECT_TRUE(plan.work_budget_exhausted);
  EXPECT_EQ(plan.rejection_reasons, (std::vector{BranchOnlyRelayPairRejection::WorkBudget}));
  EXPECT_TRUE(std::ranges::equal(planner.occupied_ranges(), occupied_before));
}

TEST(ConSanBranchOnlyRelayRouter, OwnedQualificationDoesNotChargePlacementOccupancy) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'004u, 300'000u, 100'004u},
  };
  BranchOnlyRelaySearchLimits limits;
  limits.batch_relay_qualification_work = 0u;
  limits.batch_relay_qualification_work_per_relay = 4u;
  limits.batch_relay_qualification_work_per_relay_range_level = 0u;
  DbiPatchPlacementPlanner empty_planner(kArch, 400'000u);
  DbiPatchPlacementPlanner occupied_planner(kArch, 400'000u);
  for (uint64_t range = 0u; range < 16u; ++range) {
    ASSERT_TRUE(occupied_planner.reserve_existing_range(350'000u + 2u * range * sizeof(uint32_t),
                                                        sizeof(uint32_t)));
  }

  const BranchOnlyRelayBatchPlan empty_plan =
      router.plan_pairs(empty_planner, requests, nullptr, limits);
  const BranchOnlyRelayBatchPlan occupied_plan =
      router.plan_pairs(occupied_planner, requests, nullptr, limits);

  ASSERT_TRUE(empty_plan.complete());
  ASSERT_TRUE(occupied_plan.complete());
  EXPECT_EQ(occupied_plan.scan_work_consumed, empty_plan.scan_work_consumed);
  EXPECT_FALSE(occupied_plan.work_budget_exhausted);
}

TEST(ConSanBranchOnlyRelayRouter, PristineQualificationUsesLogarithmicRangeHeadroom) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::PristineNop));
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::PristineNop));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'004u, 300'000u, 100'004u},
  };
  const auto make_planner = [=] {
    DbiPatchPlacementPlanner planner(kArch, 400'000u);
    for (uint64_t range = 0u; range < 16u; ++range) {
      EXPECT_TRUE(planner.reserve_existing_range(350'000u + 2u * range * sizeof(uint32_t),
                                                 sizeof(uint32_t)));
    }
    return planner;
  };
  BranchOnlyRelaySearchLimits scalable_limits;
  scalable_limits.batch_relay_qualification_work = 0u;
  scalable_limits.batch_relay_qualification_work_per_relay = 4u;
  scalable_limits.batch_relay_qualification_work_per_relay_range_level = 1u;
  BranchOnlyRelaySearchLimits fixed_limits = scalable_limits;
  fixed_limits.batch_relay_qualification_work_per_relay_range_level = 0u;
  DbiPatchPlacementPlanner scalable_planner = make_planner();
  DbiPatchPlacementPlanner fixed_planner = make_planner();

  const BranchOnlyRelayBatchPlan scalable_plan =
      router.plan_pairs(scalable_planner, requests, nullptr, scalable_limits);
  const BranchOnlyRelayBatchPlan fixed_plan =
      router.plan_pairs(fixed_planner, requests, nullptr, fixed_limits);

  ASSERT_TRUE(scalable_plan.complete());
  EXPECT_FALSE(scalable_plan.work_budget_exhausted);
  EXPECT_FALSE(fixed_plan.complete());
  EXPECT_EQ(fixed_plan.failure, BranchOnlyRelayPlanFailure::WorkBudget);
  EXPECT_TRUE(fixed_plan.work_budget_exhausted);
}

TEST(ConSanBranchOnlyRelayRouter, PristineQualificationWorkScalesWithRangeTreeDepth) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr size_t kSmallRangeCount = 16u;
  constexpr size_t kLargeRangeCount = 4'096u;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::PristineNop));
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::PristineNop));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'004u, 300'000u, 100'004u},
  };
  const auto make_planner = [=](size_t range_count) {
    DbiPatchPlacementPlanner planner(kArch, 500'000u);
    for (uint64_t range = 0u; range < range_count; ++range) {
      EXPECT_TRUE(planner.reserve_existing_range(400'000u + 2u * range * sizeof(uint32_t),
                                                 sizeof(uint32_t)));
    }
    return planner;
  };
  DbiPatchPlacementPlanner small_planner = make_planner(kSmallRangeCount);
  DbiPatchPlacementPlanner large_planner = make_planner(kLargeRangeCount);

  const BranchOnlyRelayBatchPlan small_plan = router.plan_pairs(small_planner, requests);
  const BranchOnlyRelayBatchPlan large_plan = router.plan_pairs(large_planner, requests);

  ASSERT_TRUE(small_plan.complete());
  ASSERT_TRUE(large_plan.complete());
  constexpr size_t kPristineRelayCount = 2u;
  EXPECT_EQ(large_plan.scan_work_consumed - small_plan.scan_work_consumed,
            kPristineRelayCount *
                (std::bit_width(kLargeRangeCount) - std::bit_width(kSmallRangeCount)));
}

TEST(ConSanBranchOnlyRelayRouter, BoundedQualificationRetainsSoundPrefixForRouting) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {100'000u, 200'000u, 350'000u})
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'004u, 300'000u, 100'004u},
  };
  BranchOnlyRelaySearchLimits limits;
  limits.batch_relay_qualification_work = 8u;
  limits.batch_relay_qualification_work_per_relay = 0u;
  limits.batch_relay_qualification_work_per_relay_range_level = 0u;
  DbiPatchPlacementPlanner planner(kArch, 400'000u);
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error, limits);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactBatch);
  EXPECT_TRUE(plan.work_budget_exhausted);
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
  EXPECT_EQ(plan.routes[0].return_relay_offsets, (std::vector<uint64_t>{200'000u}));
}

TEST(ConSanBranchOnlyRelayRouter, QualifiedPristineClaimsReserveAndCommitTheSameRelays) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::PristineNop));
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::PristineNop));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'004u, 300'000u, 100'004u},
  };
  DbiPatchPlacementPlanner planner(kArch, 400'000u);
  ASSERT_TRUE(planner.reserve_existing_range(350'000u, sizeof(uint32_t)));
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  ASSERT_TRUE(plan.complete()) << error;
  ASSERT_EQ(plan.routes.size(), 1u);
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
  EXPECT_EQ(plan.routes[0].return_relay_offsets, (std::vector<uint64_t>{200'000u}));
  const std::array expected_ranges = {
      std::pair<uint64_t, uint64_t>{100'000u, 100'004u},
      std::pair<uint64_t, uint64_t>{200'000u, 200'004u},
      std::pair<uint64_t, uint64_t>{350'000u, 350'004u},
  };
  EXPECT_TRUE(std::ranges::equal(planner.occupied_ranges(), expected_ranges));
  EXPECT_TRUE(router.commit(plan.routes, &error)) << error;
  EXPECT_EQ(router.available_count(), 0u);
}

TEST(ConSanBranchOnlyRelayRouter, ExactPairRemovalPrechargeFallsBackWithoutPartialClaims) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  DbiPatchPlacementPlanner planner(kArch, 200'008u);
  const std::vector<std::pair<uint64_t, uint64_t>> occupied_before(
      planner.occupied_ranges().begin(), planner.occupied_ranges().end());
  const BranchOnlyRelaySearchLimits limits{
      .batch_base_search_work = 1u,
      .batch_search_work_per_demand = 0u,
      .batch_base_scan_work = 100u,
      .batch_scan_work_per_demand_relay = 0u,
      .batch_fallback_setup_work = 100u,
      .pair_search_work = 100u,
      .pair_base_scan_work = 7u,
      .pair_scan_work_per_relay = 0u,
      .pair_greedy_work = 100u,
  };
  BranchOnlyRelayPlanOutcome outcome;

  const auto route =
      router.plan_pair(planner, 0u, 200'000u, 200'004u, 100'004u, nullptr, &outcome, limits);

  ASSERT_TRUE(route);
  EXPECT_EQ(outcome.strategy, BranchOnlyRelayPlanStrategy::GreedyPairFallback);
  EXPECT_TRUE(outcome.work_budget_exhausted);
  EXPECT_EQ(route->entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
  EXPECT_TRUE(std::ranges::equal(planner.occupied_ranges(), occupied_before));
}

TEST(ConSanBranchOnlyRelayRouter, DiscardedOptimizedPairDoesNotReportRetainedExhaustion) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(33u)));
  ASSERT_TRUE(router.offer(120'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(11u)));
  ASSERT_TRUE(router.offer(200'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(33u)));
  ASSERT_TRUE(router.offer(220'000u, BranchOnlyRelayProvenance::OwnedReservoir,
                           /*owner_affinity=*/lds_relay_owner(22u)));
  DbiPatchPlacementPlanner planner(kArch, 300'008u);
  BranchOnlyRelaySearchLimits limits;
  limits.batch_base_search_work = 1u;
  limits.batch_search_work_per_demand = 0u;
  limits.batch_base_scan_work = 1'000u;
  limits.batch_scan_work_per_demand_relay = 0u;
  limits.batch_fallback_setup_work = 1'000u;
  limits.pair_search_work = 1'000u;
  limits.pair_base_scan_work = 35u;
  limits.pair_scan_work_per_relay = 0u;
  limits.pair_route_optimization_search_work = 1u;
  limits.pair_route_optimization_search_work_per_demand = 0u;
  limits.pair_route_optimization_scan_work = 5u;
  limits.pair_route_optimization_scan_work_per_demand_relay = 0u;
  limits.pair_greedy_work = 1'000u;
  BranchOnlyRelayPlanOutcome outcome;

  const auto route =
      router.plan_pair(planner, 0u, 300'000u, 300'004u, 200'004u, nullptr, &outcome, limits);

  ASSERT_TRUE(route);
  EXPECT_EQ(outcome.strategy, BranchOnlyRelayPlanStrategy::GreedyPairFallback);
  EXPECT_TRUE(outcome.work_budget_exhausted);
  EXPECT_FALSE(outcome.route_optimization_exhausted);
  EXPECT_GT(outcome.route_optimization_scan_work_consumed, 0u);
}

TEST(ConSanBranchOnlyRelayRouter, DefaultFallbackSetupPreservesLargeInventoryWindow) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  constexpr size_t kRelayCount = 6'000u;
  for (size_t relay = 1u; relay <= kRelayCount; ++relay)
    ASSERT_TRUE(router.offer(relay * sizeof(uint32_t), BranchOnlyRelayProvenance::OwnedReservoir));
  DbiPatchPlacementPlanner planner(kArch, 500'008u);
  BranchOnlyRelaySearchLimits limits;
  limits.batch_base_search_work = 1u;
  limits.batch_search_work_per_demand = 0u;
  limits.pair_search_work = 1u;
  limits.pair_base_scan_work = 1u;
  limits.pair_scan_work_per_relay = 0u;
  limits.pair_greedy_work = 1u;
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 400'000u, 500'004u, 400'004u},
  };

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, nullptr, limits);

  // Reaching the greedy tier proves that fallback inventory setup remained
  // available; setup refusal stops at ExactPairFallback.
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::GreedyPairFallback);
  EXPECT_TRUE(plan.work_budget_exhausted);
}

TEST(ConSanBranchOnlyRelayRouter, GreedyRemovalPrechargeFailureIsPairAtomic) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'000u, 200'004u, 100'004u},
  };
  const BranchOnlyRelaySearchLimits limits{
      .batch_base_search_work = 1u,
      .batch_search_work_per_demand = 0u,
      .batch_base_scan_work = 100u,
      .batch_scan_work_per_demand_relay = 0u,
      .batch_fallback_setup_work = 100u,
      .pair_search_work = 1u,
      .pair_base_scan_work = 100u,
      .pair_scan_work_per_relay = 0u,
      .pair_greedy_work = 1u,
  };
  DbiPatchPlacementPlanner planner(kArch, 200'008u);
  const std::vector<std::pair<uint64_t, uint64_t>> occupied_before(
      planner.occupied_ranges().begin(), planner.occupied_ranges().end());

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, nullptr, limits);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::WorkBudget);
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::GreedyPairFallback);
  EXPECT_EQ(plan.rejection_reasons, (std::vector{BranchOnlyRelayPairRejection::WorkBudget}));
  EXPECT_TRUE(plan.routes.front().claims.empty());
  EXPECT_TRUE(std::ranges::equal(planner.occupied_ranges(), occupied_before));
}

TEST(ConSanBranchOnlyRelayRouter, ExhaustedFallbackSetupRejectsBatchTransactionally) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  for (uint64_t relay : {100'000u, 200'000u, 300'000u, 400'000u})
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 400'004u, 500'004u, 400'008u},
  };
  const BranchOnlyRelaySearchLimits limits{
      .batch_base_search_work = 1u,
      .batch_search_work_per_demand = 0u,
      .batch_base_scan_work = 100u,
      .batch_scan_work_per_demand_relay = 0u,
      .batch_fallback_setup_work = 1u,
  };
  DbiPatchPlacementPlanner planner(kArch, 500'008u);
  const std::vector<std::pair<uint64_t, uint64_t>> occupied_before(
      planner.occupied_ranges().begin(), planner.occupied_ranges().end());

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, nullptr, limits);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::WorkBudget);
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactPairFallback);
  EXPECT_TRUE(plan.work_budget_exhausted);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u}));
  EXPECT_EQ(plan.rejection_reasons, (std::vector{BranchOnlyRelayPairRejection::WorkBudget}));
  EXPECT_EQ(plan.pair_strategies, (std::vector{BranchOnlyRelayPlanStrategy::ExactPairFallback}));
  EXPECT_TRUE(std::ranges::equal(planner.occupied_ranges(), occupied_before));
}

TEST(ConSanBranchOnlyRelayRouter, GreedyFallbackFailureIsPairAtomic) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  ASSERT_TRUE(router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  const std::array requests = {
      BranchOnlyRelayPairRequest{0u, 200'000u, 200'004u, 4u},
      BranchOnlyRelayPairRequest{8u, 200'008u, 200'012u, 100'004u},
  };
  const BranchOnlyRelaySearchLimits limits{
      .batch_base_search_work = 1u,
      .batch_search_work_per_demand = 0u,
      .batch_base_scan_work = 100u,
      .batch_scan_work_per_demand_relay = 0u,
      .batch_fallback_setup_work = 100u,
      .pair_search_work = 1u,
      .pair_base_scan_work = 100u,
      .pair_scan_work_per_relay = 0u,
      .pair_greedy_work = 100u,
  };
  DbiPatchPlacementPlanner planner(kArch, 200'016u);

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, nullptr, limits);

  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::GreedyPairFallback);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u}));
  EXPECT_TRUE(plan.routes[0].claims.empty());
  EXPECT_TRUE(plan.routes[0].retired_relay_claims.empty());
  EXPECT_EQ(plan.routes[1].entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
}

TEST(ConSanBranchOnlyRelayRouter, IndependentPairsDoNotStarveExactBatchSearch) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  std::vector<BranchOnlyRelayPairRequest> requests = {
      BranchOnlyRelayPairRequest{0u, 200'008u, 200'012u, 100'004u},
      BranchOnlyRelayPairRequest{110'000u, 250'000u, 250'004u, 130'000u},
  };
  std::vector<uint64_t> relays = {100'000u, 120'000u};
  constexpr size_t kIndependentPairCount = 32u;
  for (size_t pair = 0u; pair < kIndependentPairCount; ++pair) {
    const uint64_t base = (pair + 1u) * 1'000'000u;
    requests.push_back(
        BranchOnlyRelayPairRequest{base, base + 200'000u, base + 400'004u, base + 200'004u});
    relays.push_back(base + 100'000u);
    relays.push_back(base + 300'004u);
  }

  BranchOnlyRelayRouter core_router;
  ASSERT_TRUE(core_router.offer(100'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  ASSERT_TRUE(core_router.offer(120'000u, BranchOnlyRelayProvenance::OwnedReservoir));
  DbiPatchPlacementPlanner core_planner(kArch, 250'008u);
  const BranchOnlyRelayBatchPlan core_plan =
      core_router.plan_pairs(core_planner, std::span(requests).first(2u));
  ASSERT_TRUE(core_plan.complete());

  BranchOnlyRelayRouter router;
  for (uint64_t relay : relays)
    ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir));
  DbiPatchPlacementPlanner planner(kArch, 33'000'000u);
  DbiPatchPlacementPlanner repeated_planner(kArch, 33'000'000u);
  std::string error;
  std::string repeated_error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);
  const BranchOnlyRelayBatchPlan repeated =
      router.plan_pairs(repeated_planner, requests, &repeated_error);

  ASSERT_TRUE(plan.complete()) << error;
  EXPECT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactBatch);
  EXPECT_FALSE(plan.work_budget_exhausted);
  EXPECT_EQ(plan.routes[0].entry_relay_offsets, core_plan.routes[0].entry_relay_offsets);
  EXPECT_EQ(plan.routes[1].entry_relay_offsets, core_plan.routes[1].entry_relay_offsets);
  ASSERT_NO_FATAL_FAILURE(expect_valid_complete_batch(requests, relays, plan));
  EXPECT_EQ(error, repeated_error);
  ASSERT_NO_FATAL_FAILURE(expect_same_batch_plan(plan, repeated));
}

TEST(ConSanBranchOnlyRelayRouter, BoundedSolverMatchesBruteForceOnSmallRandomBatches) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr size_t kTrialCount = 96u;
  std::mt19937_64 rng(0x5a17c0deu);
  std::uniform_int_distribution<uint64_t> offset_words(0u, 4'999u);
  std::uniform_int_distribution<size_t> extra_relay_count(0u, 6u);
  std::uniform_int_distribution<size_t> demand_choice(0u, 3u);
  std::uniform_int_distribution<size_t> owner_choice(0u, 4u);
  size_t feasible_trials = 0u;

  for (size_t trial = 0u; trial < kTrialCount; ++trial) {
    SCOPED_TRACE(::testing::Message() << "trial=" << trial);
    const std::array requests = {
        BranchOnlyRelayPairRequest{
            .entry_source = offset_words(rng) * sizeof(uint32_t),
            .entry_target = 180'000u + offset_words(rng) * sizeof(uint32_t),
            .return_source = 320'000u + offset_words(rng) * sizeof(uint32_t),
            .return_target = 140'000u + offset_words(rng) * sizeof(uint32_t),
        },
        BranchOnlyRelayPairRequest{
            .entry_source = 20'000u + offset_words(rng) * sizeof(uint32_t),
            .entry_target = 220'000u + offset_words(rng) * sizeof(uint32_t),
            .return_source = 340'000u + offset_words(rng) * sizeof(uint32_t),
            .return_target = 100'000u + offset_words(rng) * sizeof(uint32_t),
        },
    };
    std::set<uint64_t> endpoints;
    for (const BranchOnlyRelayPairRequest &request : requests) {
      endpoints.insert(request.entry_source);
      endpoints.insert(request.entry_target);
      endpoints.insert(request.return_source);
      endpoints.insert(request.return_target);
    }
    std::set<uint64_t> relay_set;
    const std::array demands = {
        std::pair{requests[0].entry_source, requests[0].entry_target},
        std::pair{requests[0].return_source, requests[0].return_target},
        std::pair{requests[1].entry_source, requests[1].entry_target},
        std::pair{requests[1].return_source, requests[1].return_target},
    };
    if (trial % 2u == 0u) {
      // Seed half the corpus with a legal independent route for every demand,
      // then add noise. The other half remains unconstrained so the oracle
      // exercises both feasible and infeasible results.
      for (const auto &[source, target] : demands) {
        const uint64_t lower = std::min(source, target);
        const uint64_t upper = std::max(source, target);
        uint64_t relay = ((lower + (upper - lower) / 2u) / sizeof(uint32_t)) * sizeof(uint32_t);
        while ((endpoints.contains(relay) || relay_set.contains(relay)) &&
               relay + sizeof(uint32_t) < upper) {
          relay += sizeof(uint32_t);
        }
        ASSERT_GT(relay, lower);
        ASSERT_LT(relay, upper);
        relay_set.insert(relay);
      }
    }
    const size_t desired_size = relay_set.size() + (trial % 2u == 0u ? extra_relay_count(rng) % 4u
                                                                     : extra_relay_count(rng));
    while (relay_set.size() < desired_size) {
      const auto [source, target] = demands[demand_choice(rng)];
      const uint64_t lower_word = std::min(source, target) / sizeof(uint32_t) + 1u;
      const uint64_t upper_word = std::max(source, target) / sizeof(uint32_t) - 1u;
      std::uniform_int_distribution<uint64_t> relay_word(lower_word, upper_word);
      const uint64_t relay = relay_word(rng) * sizeof(uint32_t);
      if (!endpoints.contains(relay))
        relay_set.insert(relay);
    }
    const std::vector<uint64_t> relays(relay_set.begin(), relay_set.end());
    std::vector<std::optional<BranchOnlyRelayOwnerIdentity>> relay_owners;
    relay_owners.reserve(relays.size());
    for (size_t relay = 0u; relay < relays.size(); ++relay) {
      const size_t owner = owner_choice(rng);
      relay_owners.push_back(
          owner == 0u ? std::nullopt
                      : std::optional<BranchOnlyRelayOwnerIdentity>(lds_relay_owner(owner)));
    }
    const bool expected = brute_force_fixed_relay_batch(requests, relays);
    feasible_trials += expected ? 1u : 0u;

    size_t minimum_owner_count = std::numeric_limits<size_t>::max();
    for (uint32_t owner_mask = 0u; owner_mask < 1u << 4u; ++owner_mask) {
      std::vector<uint64_t> selected_relays;
      for (size_t relay = 0u; relay < relays.size(); ++relay) {
        if (!relay_owners[relay]) {
          selected_relays.push_back(relays[relay]);
          continue;
        }
        const uint32_t owner_bit = 1u
                                   << static_cast<uint32_t>(relay_owners[relay]->producer_key - 1u);
        if ((owner_mask & owner_bit) != 0u)
          selected_relays.push_back(relays[relay]);
      }
      if (brute_force_fixed_relay_batch(requests, selected_relays))
        minimum_owner_count = std::min<size_t>(minimum_owner_count, std::popcount(owner_mask));
    }
    EXPECT_EQ(minimum_owner_count != std::numeric_limits<size_t>::max(), expected);

    BranchOnlyRelayRouter router;
    for (size_t relay = 0u; relay < relays.size(); ++relay) {
      ASSERT_TRUE(router.offer(relays[relay], BranchOnlyRelayProvenance::OwnedReservoir,
                               relay_owners[relay]));
    }
    DbiPatchPlacementPlanner planner(kArch, 400'000u);
    std::string error;
    const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

    EXPECT_EQ(plan.complete(), expected) << "trial=" << trial << " diagnostic=" << error;
    if (plan.complete()) {
      ASSERT_NO_FATAL_FAILURE(expect_valid_complete_batch(requests, relays, plan));
      ASSERT_EQ(plan.strategy, BranchOnlyRelayPlanStrategy::ExactBatch);
      ASSERT_FALSE(plan.route_optimization_exhausted);
      std::set<BranchOnlyRelayOwnerIdentity> selected_owners;
      for (const BranchOnlyRelayRoute &route : plan.routes) {
        for (const BranchOnlyRelayClaim &claim : route.claims) {
          if (claim.owner_affinity)
            selected_owners.insert(*claim.owner_affinity);
        }
      }
      EXPECT_EQ(selected_owners.size(), minimum_owner_count);
    }

    BranchOnlyRelaySearchLimits pair_limits;
    pair_limits.batch_base_search_work = 1u;
    pair_limits.batch_search_work_per_demand = 0u;
    BranchOnlyRelaySearchLimits feasibility_only_limits = pair_limits;
    feasibility_only_limits.pair_route_optimization_search_work = 1u;
    feasibility_only_limits.pair_route_optimization_search_work_per_demand = 0u;
    const auto plan_with_limits = [&](const BranchOnlyRelaySearchLimits &limits) {
      BranchOnlyRelayRouter pair_router;
      for (size_t relay = 0u; relay < relays.size(); ++relay) {
        EXPECT_TRUE(pair_router.offer(relays[relay], BranchOnlyRelayProvenance::OwnedReservoir,
                                      relay_owners[relay]));
      }
      DbiPatchPlacementPlanner pair_planner(kArch, 400'000u);
      return pair_router.plan_pairs(pair_planner, requests, nullptr, limits);
    };
    const BranchOnlyRelayBatchPlan feasibility_baseline = plan_with_limits(feasibility_only_limits);
    const BranchOnlyRelayBatchPlan pair_optimized = plan_with_limits(pair_limits);
    // Every nonfinal optimized route is a subset of its feasibility baseline,
    // whose complete relay set stays unavailable to later pairs. Therefore
    // optimization cannot make a feasible baseline incomplete or increase the
    // total number of selected relays.
    EXPECT_FALSE(feasibility_baseline.complete() && !pair_optimized.complete());
    if (pair_optimized.complete() && feasibility_baseline.complete()) {
      ASSERT_EQ(pair_optimized.strategy, BranchOnlyRelayPlanStrategy::ExactPairFallback);
      ASSERT_EQ(feasibility_baseline.strategy, BranchOnlyRelayPlanStrategy::ExactPairFallback);
      const auto selected_owner_count = [](const BranchOnlyRelayBatchPlan &candidate) {
        std::set<BranchOnlyRelayOwnerIdentity> owners;
        for (const BranchOnlyRelayRoute &route : candidate.routes) {
          for (const BranchOnlyRelayClaim &claim : route.claims) {
            if (claim.owner_affinity)
              owners.insert(*claim.owner_affinity);
          }
        }
        return owners.size();
      };
      const auto selected_relay_count = [](const BranchOnlyRelayBatchPlan &candidate) {
        return std::accumulate(candidate.routes.begin(), candidate.routes.end(), size_t{0u},
                               [](size_t count, const BranchOnlyRelayRoute &route) {
                                 return count + route.claims.size();
                               });
      };
      EXPECT_LE(selected_owner_count(pair_optimized), selected_owner_count(feasibility_baseline));
      EXPECT_LE(selected_relay_count(pair_optimized), selected_relay_count(feasibility_baseline));
    }
  }
  // Keep both oracle outcomes meaningfully represented without depending on
  // a standard library's uniform-distribution mapping.
  EXPECT_GT(feasible_trials, kTrialCount * 3u / 10u);
  EXPECT_LT(feasible_trials, kTrialCount * 7u / 10u);
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

TEST(ConSanBranchOnlyRelayRouter, EmptyBatchIsACompleteNoOp) {
  BranchOnlyRelayRouter router;
  DbiPatchPlacementPlanner planner(ROCJITSU_CODE_ARCH_RDNA4, 16u);
  const BranchOnlyRelayBatchPlan plan =
      router.plan_pairs(planner, std::span<const BranchOnlyRelayPairRequest>{});

  EXPECT_TRUE(plan.complete());
  EXPECT_TRUE(plan.routes.empty());
  EXPECT_TRUE(planner.occupied_ranges().empty());
}

TEST(ConSanBranchOnlyRelayRouter, ClassifiesReservoirInstructionBoundaries) {
  static constexpr std::array<uint32_t, 4> kRaw = {
      0x11111111u,
      0x22222222u,
      0x33333333u,
      0x44444444u,
  };
  const std::vector<uint8_t> text(4u * sizeof(uint32_t));
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  const auto accepted = [&](std::string_view mnemonic, int size, uint64_t flags = 0u,
                            std::optional<int64_t> branch_delta = std::nullopt,
                            uint64_t offset = 0u) {
    RelayTestInstruction instruction(mnemonic, size, flags, branch_delta, kRaw.data());
    return is_consan_branch_relay_reservoir_instruction(instruction, offset, text, kArch);
  };

  EXPECT_TRUE(accepted("s_mov_b32", 4));
  EXPECT_FALSE(accepted("s_mov_b32", 4, 0u, std::nullopt, 2u));
  EXPECT_FALSE(accepted("s_mov_b32", 4, 0u, std::nullopt, text.size() + sizeof(uint32_t)));
  EXPECT_FALSE(accepted("s_mov_b32", 8, 0u, std::nullopt, text.size() - sizeof(uint32_t)));
  EXPECT_FALSE(accepted("ds_read_b32", 4));
  EXPECT_FALSE(accepted("s_clause", 4));
  EXPECT_FALSE(accepted("s_delay_alu", 4));
  EXPECT_TRUE(accepted("flat_load_dword", 3 * sizeof(uint32_t)));
  EXPECT_TRUE(accepted("flat_store_dword", 3 * sizeof(uint32_t)));
  EXPECT_FALSE(accepted("global_load_dword", 3 * sizeof(uint32_t)));
  EXPECT_FALSE(accepted("flat_load_dword", 3 * sizeof(uint32_t), BRANCH));
  EXPECT_FALSE(accepted("flat_load_dword", 3 * sizeof(uint32_t), 0u, 4));
  EXPECT_FALSE(accepted("flat_load_dword", 4 * sizeof(uint32_t)));
}

TEST(ConSanBranchOnlyRelayRouter, PlansOnlyMinimumWidthRunsAndChunksLongRuns) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  const auto plan = [&](size_t donor_word_count, size_t target_relay_count) {
    std::vector<uint32_t> words(donor_word_count, kRelayTestDonor);
    words.push_back(kRelayTestEnd);
    RelayTestCodeObject object(std::move(words));
    RelayTestDecoder decoder;
    auto blocks = BasicBlock::build(object, decoder, kArch);
    const std::vector<BasicBlock *> block_ptrs = relay_block_ptrs(blocks);
    DbiPatchPlacementPlanner planner(kArch, relay_test_text(object).size());
    BranchOnlyRelayRouter router;
    BranchOnlyDirectRelayReservoirSet reservoirs;
    std::string error;
    EXPECT_TRUE(router.plan_direct_reservoirs(block_ptrs, relay_test_text(object), {}, kArch,
                                              relay_test_text(object).size() / 2u,
                                              target_relay_count, planner, reservoirs, &error))
        << error;
    return reservoirs;
  };

  EXPECT_TRUE(plan(15u, 1u).reservoirs.empty());
  const BranchOnlyDirectRelayReservoirSet exact = plan(16u, 1u);
  ASSERT_EQ(exact.reservoirs.size(), 1u);
  EXPECT_EQ(exact.reservoirs.front().original_words.size(), 16u);
  EXPECT_EQ(exact.reservoir_by_relay.size(), 15u);

  BranchOnlyDirectRelayReservoirSet long_run = plan(100u, 200u);
  ASSERT_EQ(long_run.reservoirs.size(), 2u);
  std::vector<size_t> widths;
  for (const BranchOnlyDirectRelayReservoir &reservoir : long_run.reservoirs)
    widths.push_back(reservoir.original_words.size());
  std::ranges::sort(widths);
  EXPECT_EQ(widths, (std::vector<size_t>{36u, 64u}));
  const auto first = std::ranges::min(
      long_run.reservoirs, {}, [](const auto &reservoir) { return reservoir.anchor_offset; });
  const auto last = std::ranges::max(long_run.reservoirs, {},
                                     [](const auto &reservoir) { return reservoir.anchor_offset; });
  EXPECT_LE(first.anchor_offset + first.original_words.size() * sizeof(uint32_t),
            last.anchor_offset);
}

TEST(ConSanBranchOnlyRelayRouter, SplitsReservoirRunsAtProtectedAndClauseRanges) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  {
    std::vector<uint32_t> words(40u, kRelayTestDonor);
    words.push_back(kRelayTestEnd);
    RelayTestCodeObject object(std::move(words));
    RelayTestDecoder decoder;
    auto blocks = BasicBlock::build(object, decoder, kArch);
    const std::vector<BasicBlock *> block_ptrs = relay_block_ptrs(blocks);
    const std::array protected_ranges = {
        std::pair<uint64_t, uint64_t>{16u * sizeof(uint32_t), 20u * sizeof(uint32_t)},
        std::pair<uint64_t, uint64_t>{20u * sizeof(uint32_t), 24u * sizeof(uint32_t)},
    };
    DbiPatchPlacementPlanner planner(kArch, relay_test_text(object).size());
    BranchOnlyRelayRouter router;
    BranchOnlyDirectRelayReservoirSet reservoirs;
    std::string error;

    ASSERT_TRUE(router.plan_direct_reservoirs(block_ptrs, relay_test_text(object), protected_ranges,
                                              kArch, relay_test_text(object).size() / 2u, 100u,
                                              planner, reservoirs, &error))
        << error;
    ASSERT_EQ(reservoirs.reservoirs.size(), 2u);
    EXPECT_TRUE(std::ranges::all_of(reservoirs.reservoirs, [](const auto &reservoir) {
      return reservoir.original_words.size() == 16u;
    }));
  }

  {
    std::vector<uint32_t> words(16u, kRelayTestDonor);
    words.push_back(kRelayTestClauseTwo);
    words.insert(words.end(), 18u, kRelayTestDonor);
    words.push_back(kRelayTestEnd);
    RelayTestCodeObject object(std::move(words));
    RelayTestDecoder decoder;
    auto blocks = BasicBlock::build(object, decoder, kArch);
    const std::vector<BasicBlock *> block_ptrs = relay_block_ptrs(blocks);
    DbiPatchPlacementPlanner planner(kArch, relay_test_text(object).size());
    BranchOnlyRelayRouter router;
    BranchOnlyDirectRelayReservoirSet reservoirs;
    std::string error;

    ASSERT_TRUE(router.plan_direct_reservoirs(block_ptrs, relay_test_text(object), {}, kArch,
                                              relay_test_text(object).size() / 2u, 100u, planner,
                                              reservoirs, &error))
        << error;
    ASSERT_EQ(reservoirs.reservoirs.size(), 2u);
    EXPECT_TRUE(std::ranges::all_of(reservoirs.reservoirs, [](const auto &reservoir) {
      return reservoir.original_words.size() == 16u;
    }));
  }
}

TEST(ConSanBranchOnlyRelayRouter, DirectReservoirPlanningShortCircuitsAndAdoptsAtomically) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  {
    DbiPatchPlacementPlanner planner(kArch, 0u);
    BranchOnlyRelayRouter router;
    BranchOnlyDirectRelayReservoirSet reservoirs;
    std::string error;
    EXPECT_TRUE(
        router.plan_direct_reservoirs({}, {}, {}, kArch, 0u, 0u, planner, reservoirs, &error));
    EXPECT_TRUE(reservoirs.reservoirs.empty());
    EXPECT_FALSE(
        router.plan_direct_reservoirs({}, {}, {}, kArch, 0u, 1u, planner, reservoirs, &error));
    EXPECT_NE(error.find("pristine text"), std::string::npos);
  }

  std::vector<uint32_t> words(16u, kRelayTestDonor);
  words.push_back(kRelayTestEnd);
  RelayTestCodeObject object(std::move(words));
  RelayTestDecoder decoder;
  auto blocks = BasicBlock::build(object, decoder, kArch);
  const std::vector<BasicBlock *> block_ptrs = relay_block_ptrs(blocks);
  for (uint64_t occupied_word : std::array<uint64_t, 2>{0u, sizeof(uint32_t)}) {
    DbiPatchPlacementPlanner planner(kArch, relay_test_text(object).size());
    BranchOnlyRelayRouter router;
    ASSERT_TRUE(router.offer(occupied_word, BranchOnlyRelayProvenance::PristineNop));
    BranchOnlyDirectRelayReservoirSet reservoirs;
    std::string error;

    ASSERT_TRUE(router.plan_direct_reservoirs(block_ptrs, relay_test_text(object), {}, kArch, 0u,
                                              1u, planner, reservoirs, &error))
        << error;
    EXPECT_TRUE(reservoirs.reservoirs.empty());
    EXPECT_EQ(router.available_count(), 1u);
  }
}

TEST(ConSanBranchOnlyRelayRouter, PreplannedDirectReservoirIsZeroCostRoutingCapacity) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr size_t kDonorWord = 31'250u;
  constexpr size_t kTextWords = 62'500u;
  constexpr uint32_t kRelayTestExcluded = 0x3000u;
  std::vector<uint32_t> words(kTextWords, kRelayTestExcluded);
  std::fill_n(words.begin() + kDonorWord, 16u, kRelayTestDonor);
  words.push_back(kRelayTestEnd);
  RelayTestCodeObject object(std::move(words));
  class ExcludedRelayTestDecoder : public RelayTestDecoder {
  public:
    Instruction *decode(const rj_code_binary_inst_t *word) override {
      if (*word == kRelayTestExcluded)
        return new RelayTestInstruction("ds_read_b32", 4, 0u, std::nullopt, word);
      return RelayTestDecoder::decode(word);
    }
  } decoder;
  auto blocks = BasicBlock::build(object, decoder, kArch);
  const std::vector<BasicBlock *> block_ptrs = relay_block_ptrs(blocks);
  DbiPatchPlacementPlanner planner(kArch, relay_test_text(object).size());
  BranchOnlyRelayRouter router;
  BranchOnlyDirectRelayReservoirSet reservoirs;
  std::string error;

  ASSERT_TRUE(router.plan_direct_reservoirs(block_ptrs, relay_test_text(object), {}, kArch, 0u, 1u,
                                            planner, reservoirs, &error))
      << error;
  ASSERT_EQ(reservoirs.reservoirs.size(), 1u);
  const size_t occupied_before = planner.occupied_ranges().size();
  const auto route = router.plan_pair(planner, 0u, 250'000u, 250'004u, 240'004u, &error);

  ASSERT_TRUE(route) << error;
  ASSERT_EQ(route->claims.size(), 1u);
  EXPECT_EQ(route->claims.front().provenance, BranchOnlyRelayProvenance::OwnedReservoir);
  EXPECT_FALSE(route->claims.front().owner_affinity);
  EXPECT_EQ(planner.occupied_ranges().size(), occupied_before);
}

TEST(ConSanBranchOnlyRelayRouter, DirectReservoirFailureRollsBackTheWholeCall) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  std::vector<uint32_t> words(100u, kRelayTestDonor);
  words.push_back(kRelayTestEnd);
  RelayTestCodeObject object(std::move(words));
  RelayTestDecoder decoder;
  auto blocks = BasicBlock::build(object, decoder, kArch);
  const std::vector<BasicBlock *> block_ptrs = relay_block_ptrs(blocks);
  DbiPatchPlacementPlanner planner(kArch, relay_test_text(object).size());
  BranchOnlyRelayRouter router;
  BranchOnlyDirectRelayReservoirSet reservoirs;
  reservoirs.reservoirs.resize(1u);
  // The later candidate begins at byte 144, so its first relay conflicts with
  // this pre-existing index only after the earlier candidate was adopted.
  ASSERT_TRUE(reservoirs.reservoir_by_relay.emplace(148u, 0u).second);
  std::string error;

  EXPECT_FALSE(router.plan_direct_reservoirs(block_ptrs, relay_test_text(object), {}, kArch, 0u,
                                             std::numeric_limits<size_t>::max(), planner,
                                             reservoirs, &error));

  EXPECT_NE(error.find("atomically adopt"), std::string::npos);
  EXPECT_EQ(router.available_count(), 0u);
  EXPECT_EQ(reservoirs.reservoirs.size(), 1u);
  EXPECT_EQ(reservoirs.reservoir_by_relay.size(), 1u);
  EXPECT_EQ(reservoirs.reservoir_by_relay.at(148u), 0u);
  EXPECT_TRUE(planner.occupied_ranges().empty());
}

TEST(ConSanBranchOnlyRelayRouter, InventoriesUsedAndUnusedDirectReservoirFootprint) {
  BranchOnlyDirectRelayReservoirSet reservoirs;
  reservoirs.reservoirs = {
      BranchOnlyDirectRelayReservoir{
          .anchor_offset = 64u,
          .original_words = std::vector<uint32_t>(16u),
          .placement = {},
          .used = true,
      },
      BranchOnlyDirectRelayReservoir{
          .anchor_offset = 256u,
          .original_words = std::vector<uint32_t>(32u),
          .placement = {},
          .used = false,
      },
  };

  const ConSanBranchOnlyReservoirTelemetry telemetry = reservoirs.telemetry();
  EXPECT_EQ(telemetry.planned_reservoir_count, 2u);
  EXPECT_EQ(telemetry.used_reservoir_count, 1u);
  EXPECT_EQ(telemetry.unused_reservoir_count, 1u);
  EXPECT_EQ(telemetry.planned_appended_bytes, (16u + 32u) * sizeof(uint32_t) + 2u * 4u);
  EXPECT_EQ(telemetry.used_appended_bytes, 16u * sizeof(uint32_t) + 4u);
  EXPECT_EQ(telemetry.unused_appended_bytes, 32u * sizeof(uint32_t) + 4u);
}

TEST(ConSanBranchOnlyRelayRouter, EmitsDirectReservoirAtItsOwnedAppendedOffset) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint64_t kAnchor = 64u;
  constexpr uint64_t kBody = 512u;
  constexpr uint64_t kOriginalSize = 16u * sizeof(uint32_t);
  BranchOnlyDirectRelayReservoir reservoir{
      .anchor_offset = kAnchor,
      .original_words = std::vector<uint32_t>(16u, build_s_mov_b32(8u, 8u, kArch)),
      .placement =
          {
              .kind = DbiPatchPlacementKind::AppendedCave,
              .anchor_offset = kAnchor,
              .original_size = kOriginalSize,
              .body_offset = kBody,
              .body_size = kOriginalSize,
              .return_branch_offset = kBody + kOriginalSize,
              .return_target = kAnchor + kOriginalSize,
          },
      .used = true,
  };
  // Model an unrelated appended allocation before this reservoir. Emission is
  // offset-driven and must not require the reservoir to start at text.end().
  std::vector<uint8_t> text(256u, 0u);
  std::memcpy(text.data() + kAnchor, reservoir.original_words.data(), kOriginalSize);
  std::vector<ConSanPatchInfo> patches;
  std::string error;

  ASSERT_TRUE(BranchOnlyRelayRouter::emit_direct_reservoir(text, reservoir, kArch, patches, &error))
      << error;

  ASSERT_EQ(text.size(), kBody + kOriginalSize + sizeof(uint32_t));
  EXPECT_EQ(0, std::memcmp(text.data() + kBody, reservoir.original_words.data(), kOriginalSize));
  uint32_t entry = 0u;
  uint32_t ret = 0u;
  std::memcpy(&entry, text.data() + kAnchor, sizeof(entry));
  std::memcpy(&ret, text.data() + kBody + kOriginalSize, sizeof(ret));
  ASSERT_TRUE(compute_sopp_branch_simm16(kAnchor, kBody));
  EXPECT_EQ(entry, build_s_branch(*compute_sopp_branch_simm16(kAnchor, kBody), kArch));
  EXPECT_EQ(
      ret, build_s_branch(
               *compute_sopp_branch_simm16(kBody + kOriginalSize, kAnchor + kOriginalSize), kArch));
  ASSERT_EQ(patches.size(), 1u);
  EXPECT_EQ(patches.front().kind, ConSanPatchKind::TrampolineBranchRelayReservoir);
  EXPECT_EQ(patches.front().original_size, kOriginalSize);
  BranchOnlyDirectRelayReservoirSet reservoirs;
  reservoirs.reservoirs.push_back(reservoir);
  EXPECT_EQ(reservoirs.telemetry().used_appended_bytes, patches.front().trampoline_size);
}

} // namespace
} // namespace rocjitsu
