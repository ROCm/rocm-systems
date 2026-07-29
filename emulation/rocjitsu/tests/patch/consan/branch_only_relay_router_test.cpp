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
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <string_view>
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
  BranchOnlyRelayPlanFailure failure = BranchOnlyRelayPlanFailure::Reservation;
  std::string error;
  const auto route = router.plan_pair(planner, kEntrySource, kEntryTarget, kReturnSource,
                                      kReturnTarget, &error, &failure);
  ASSERT_TRUE(route) << error;
  EXPECT_EQ(failure, BranchOnlyRelayPlanFailure::None);
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

TEST(ConSanBranchOnlyRelayRouter, ReportsUnreachableEntryAndReservationWithoutPartialPlanning) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  {
    BranchOnlyRelayRouter router;
    DbiPatchPlacementPlanner planner(kArch, 200'008u);
    BranchOnlyRelayPlanFailure failure = BranchOnlyRelayPlanFailure::None;
    std::string error;

    EXPECT_FALSE(router.plan_pair(planner, /*entry_source=*/0u, /*entry_target=*/200'000u,
                                  /*return_source=*/200'004u, /*return_target=*/100'004u, &error,
                                  &failure));
    EXPECT_EQ(failure, BranchOnlyRelayPlanFailure::EntryRoute);
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
    BranchOnlyRelayPlanFailure failure = BranchOnlyRelayPlanFailure::None;
    std::string error;

    EXPECT_FALSE(router.plan_pair(planner, /*entry_source=*/0u, /*entry_target=*/200'000u,
                                  /*return_source=*/200'004u, /*return_target=*/100'004u, &error,
                                  &failure));
    EXPECT_EQ(failure, BranchOnlyRelayPlanFailure::Reservation);
    EXPECT_TRUE(std::ranges::equal(planner.occupied_ranges(), occupied_before));
  }
}

TEST(ConSanBranchOnlyRelayRouter, RejectsCrossedDestinationAssignment) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  BranchOnlyRelayRouter router;
  const std::array requests = {
      BranchOnlyRelayPairRequest{
          .entry_source = 0u,
          .entry_target = 200'000u,
          .return_source = 200'004u,
          .return_target = 100'000u,
      },
      BranchOnlyRelayPairRequest{
          .entry_source = 130'000u,
          .entry_target = 120'000u,
          .return_source = 120'004u,
          .return_target = 100'004u,
      },
  };
  DbiPatchPlacementPlanner planner(kArch, 200'008u);
  std::string error = "stale caller diagnostic";

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::EntryRoute);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u, 1u}));
  EXPECT_EQ(error, "branch-only router requires distinct dword-aligned monotonic pair coordinates");
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
  ASSERT_TRUE(router.commit(plan.routes, &error)) << error;
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
      BranchOnlyRelayPairRequest{500'000u, 400'000u, 400'004u, 300'004u},
      BranchOnlyRelayPairRequest{0u, 200'000u, 200'004u, 100'004u},
  };
  std::string error;

  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

  EXPECT_FALSE(plan.complete());
  EXPECT_EQ(plan.failure, BranchOnlyRelayPlanFailure::EntryRoute);
  EXPECT_EQ(plan.rejected_pair_indices, (std::vector<size_t>{0u}));
  EXPECT_TRUE(plan.routes[0].claims.empty());
  EXPECT_EQ(plan.routes[1].entry_relay_offsets, (std::vector<uint64_t>{100'000u}));
  ASSERT_EQ(plan.routes[1].claims.size(), 1u);
  EXPECT_EQ(plan.routes[1].claims.front().offset, 100'000u);
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
  EXPECT_EQ(plan.routes[1].return_relay_offsets, (std::vector<uint64_t>{200'000u}));
  ASSERT_EQ(plan.routes[1].claims.size(), 1u);
  EXPECT_EQ(plan.routes[1].claims.front().offset, 200'000u);
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
  std::string error;

  const auto start = std::chrono::steady_clock::now();
  const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);
  const auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_FALSE(plan.complete());
  EXPECT_NE(error.find("search budget exhausted"), std::string::npos);
  EXPECT_LT(elapsed, std::chrono::seconds(1));
}

TEST(ConSanBranchOnlyRelayRouter, BoundedSolverMatchesBruteForceOnSmallRandomBatches) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  std::mt19937_64 random(0x5a17c0deu);
  std::uniform_int_distribution<uint64_t> offset_words(0u, 4'999u);
  std::uniform_int_distribution<size_t> relay_count(0u, 6u);
  std::uniform_int_distribution<uint64_t> relay_words(10'000u, 89'999u);

  for (size_t trial = 0u; trial < 64u; ++trial) {
    const std::array requests = {
        BranchOnlyRelayPairRequest{
            .entry_source = offset_words(random) * sizeof(uint32_t),
            .entry_target = 180'000u + offset_words(random) * sizeof(uint32_t),
            .return_source = 320'000u + offset_words(random) * sizeof(uint32_t),
            .return_target = 140'000u + offset_words(random) * sizeof(uint32_t),
        },
        BranchOnlyRelayPairRequest{
            .entry_source = 20'000u + offset_words(random) * sizeof(uint32_t),
            .entry_target = 220'000u + offset_words(random) * sizeof(uint32_t),
            .return_source = 360'000u + offset_words(random) * sizeof(uint32_t),
            .return_target = 100'000u + offset_words(random) * sizeof(uint32_t),
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
    const size_t desired_relay_count = relay_count(random);
    while (relay_set.size() < desired_relay_count) {
      const uint64_t relay = relay_words(random) * sizeof(uint32_t);
      if (!endpoints.contains(relay))
        relay_set.insert(relay);
    }
    const std::vector<uint64_t> relays(relay_set.begin(), relay_set.end());
    const bool expected = brute_force_fixed_relay_batch(requests, relays);

    BranchOnlyRelayRouter router;
    for (uint64_t relay : relays)
      ASSERT_TRUE(router.offer(relay, BranchOnlyRelayProvenance::OwnedReservoir));
    DbiPatchPlacementPlanner planner(kArch, 400'000u);
    std::string error;
    const BranchOnlyRelayBatchPlan plan = router.plan_pairs(planner, requests, &error);

    EXPECT_EQ(plan.complete(), expected) << "trial=" << trial << " diagnostic=" << error;
  }
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
}

} // namespace
} // namespace rocjitsu
